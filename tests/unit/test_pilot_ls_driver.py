"""Check the Sprint 3.1 integration driver against doubles of the native tools.

The control is a Sprint 2.4 run that the driver of that sprint grades on the small pair of its own
driver tests: two TDD periods, six opportunities per member, six slot grants, four receive ports.
The double of the tapped tool (tests/unit/pilot_ls_double.py) decodes every plan of that run again
with the model receiver and computes each observation with the reference, so every check of the
driver has a known answer, and each injected fault must fail the check that names it. NumPy is
needed, as for the driver.
"""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    np = None

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot_ls_double as double  # noqa: E402
import test_replay_pusch_driver as control_tests  # noqa: E402

CHECKS = 22
CASES = 9
EXECUTABLE = '''#!{python}
import sys
sys.path.insert(0, {tests!r})
import pilot_ls_double
sys.exit(pilot_ls_double.main())
'''
ENVIRONMENT = ("PILOT_LS_FAULT", "PILOT_LS_BUILD", "REPLAY_PUSCH_FAULT", "REPLAY_PUSCH_BUILD")


def load_driver():
    spec = importlib.util.spec_from_file_location("pilot_ls_driver", ROOT / "tests/integration/pilot_ls.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(np is None, "the integration driver needs NumPy")
class DriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.driver = load_driver()
        # The accepted Sprint 2.4 run: the driver of that sprint on the doubles of its tools.
        cls.stage = control_tests.DriverTests
        cls.stage.setUpClass()
        cls.addClassCleanup(cls.stage.tearDownClass)
        cls.root = cls.stage.root / "s31"
        cls.root.mkdir()
        cls.control = cls.stage.reference["out"]
        assert cls.stage.reference["status"] == 0, cls.stage.reference["failed"]
        config = json.loads((ROOT / "configs/integration/pilot_ls.json").read_text())
        config["pair"]["expect"]["opportunities"] = 12
        cls.config = cls.root / "config.json"
        cls.config.write_text(json.dumps(config))
        cls.tool = cls.root / "pilot_ls"
        cls.tool.write_text(EXECUTABLE.format(python=sys.executable, tests=str(Path(__file__).resolve().parent)))
        cls.tool.chmod(0o755)
        # One run without a fault, read by several tests.
        cls.reference = cls.run_once(cls.root / "reference")

    @classmethod
    def receipt_of(cls):
        tools = {"pilot_ls": cls.tool, "replay_pusch": cls.stage.tools["replay_pusch"]}
        return {"executables": {name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                                for name, path in tools.items()},
                "ocudu_build": "double", "pin": {"cuda": True}}

    @staticmethod
    def native(tool, plan_path, output, log, extra=()):
        """The tool double without a new interpreter: the same files, the same status, the same last line."""
        output.mkdir(parents=True)
        said = io.StringIO()
        with contextlib.redirect_stdout(said), contextlib.redirect_stderr(said):
            try:
                status = double.main(["--plan", str(plan_path), "--output-dir", str(output), *extra])
            except Exception as error:  # a process that crashes exits with status 1
                said.write(f"{type(error).__name__}: {error}\n")
                status = 1
        log.write_text(f"{tool} --plan {plan_path} --output-dir {output}\n" + said.getvalue())
        try:
            return status, json.loads((output / "native.json").read_text())
        except (OSError, ValueError):
            return status, None

    @classmethod
    def run_once(cls, out, fault=None, config=None, control=None, extra=(), environment=None, processes=False):
        driver = cls.driver
        saved = (driver.native_adapter.verified_receipt, driver.run_native)
        driver.native_adapter.verified_receipt = lambda build_dir: cls.receipt_of()
        if not processes:
            driver.run_native = cls.native
        changed = dict(environment or {})
        if fault:
            changed["PILOT_LS_FAULT"] = fault
        kept = {key: os.environ.get(key) for key in ENVIRONMENT}
        for key in kept:
            os.environ.pop(key, None)
        os.environ.update(changed)
        arguments = ["--config", str(config or cls.config), "--s24-run", str(control or cls.control),
                     "--output-dir", str(out), *extra]
        try:
            with contextlib.redirect_stdout(io.StringIO()) as said:
                status = driver.main(arguments)
        finally:
            driver.native_adapter.verified_receipt, driver.run_native = saved
            for key, value in kept.items():
                os.environ.pop(key, None)
                if value is not None:
                    os.environ[key] = value
        grade = json.loads((out / "grade.json").read_text())
        failed = sorted(f"{item['case']} {item['check']}" for item in grade["results"] if not item["passed"])
        return {"status": status, "grade": grade, "failed": failed, "out": out, "said": said.getvalue(),
                "lines": (out / "grade.txt").read_text()}

    def setUp(self):
        scratch = tempfile.TemporaryDirectory(dir=self.root)
        self.addCleanup(scratch.cleanup)
        self.out = Path(scratch.name)

    def faulty(self, fault, **options):
        return self.run_once(self.out / fault, fault=fault, **options)

    # ---------------------------------------------------------------- the reference
    def test_the_reference_passes_every_check(self):
        run = self.reference
        grade, lines, out = run["grade"], run["lines"], run["out"]
        self.assertEqual((run["status"], run["failed"]), (0, []))
        self.assertEqual((grade["judgment"], grade["hardware_gate"], grade["cases"], grade["cases_passed"], grade["checks"]),
                         ("pass", "closed", CASES, CASES, CHECKS))
        self.assertIn(f"pilot_ls: {CASES} of {CASES} cases passed, {CHECKS} of {CHECKS} checks\n", lines)
        self.assertNotIn("rehearsal", run["said"])
        self.assertEqual(grade["estimator_precision"],
                         "complex64, computed in double precision from single-precision operands")
        # The control: the Sprint 2.4 run and the five runs of it that are tapped.
        self.assertIn("PASS  S3.1 control accepted run: the Sprint 2.4 driver graded reference as pass: 8 of 8 cases, "
                      "0 failed checks, hardware gate closed", lines)
        self.assertIn("5 runs of the Sprint 2.4 gate verify with the reader of that sprint", lines)
        self.assertIn("(unity/shipped-1rx, unity/slot-grants, direct, reference-pair, payload)", lines)
        # Unity: 1.00119 on every pilot, at the coordinates of each grant.
        self.assertIn("unity/shipped-1rx: 2 observations on 2 grids, cpu 2 of 2 decoded, cuda 2 of 2 decoded", lines)
        self.assertIn("demo-51prb-4rx: 6 grids, 14688 estimates of shape [4, 2, 306] from subcarrier 0 in steps of two, "
                      "each between 1.0011865 and 1.0011865 with a zero imaginary part and a set validity bit", lines)
        self.assertIn("two-prb-offset: 1 grids, 24 estimates of shape [1, 2, 12] from subcarrier 180 in steps of two", lines)
        self.assertIn("expected 1.0011865, the stored pilot of magnitude one per component against the reference of "
                      "0.998815", lines)
        # The direct path: the known channel on four receive planes.
        self.assertIn("PASS  S3.1 direct known channel: 14688 of 14688 estimates on 6 grids and 4 receive planes equal "
                      "g H times the pilot ratio", lines)
        self.assertIn("PASS  S3.1 direct receive planes: the phase from each receive plane to the next agrees", lines)
        self.assertIn("only 0 of 14688 estimates stay inside the tolerance", lines)
        # The pair.
        self.assertIn("PASS  S3.1 reference-pair ingress: 12 of 12 observations name the readback digest of the grid "
                      "that the accepted replay_pusch decoded", lines)
        self.assertIn("PASS  S3.1 reference-pair reference: 12 of 12 payloads and validity masks equal the reference", lines)
        self.assertIn("name the source pusch_dmrs, port 0, the reference amplitude 1.412538, the stage O2, the producer "
                      "observer", lines)
        self.assertIn("29376 of 29376 estimates at the declared pilots are nonzero and 29376 validity bits are set", lines)
        self.assertIn("PASS  S3.1 reference-pair decodes: with the tap on, 24 radio records equal those of the control "
                      "reference-pair", lines)
        self.assertIn("24 of 24 files of decoded bytes are those of the control, byte for byte", lines)
        self.assertIn("NOTE  S3.1 reference-pair the estimates of the two members differ by", lines)
        # Other data, another grid, the same estimate.
        self.assertIn("PASS  S3.1 payload data invariance: 12 of 12 grids differ from the grid of the same opportunity "
                      "with the fixed block, and 12 of 12 estimates are the estimate of that grid bit for bit", lines)
        self.assertIn("PASS  S3.1 tap-off native run: the pair through the same executable without the tap: "
                      "0 observations on 12 grids", lines)
        self.assertIn("PASS  S3.1 rerun one opportunity:", lines)
        self.assertIn("PASS  S3.1 backends executed path: cpu: 50 on the host path; cuda: 50 on the resident path", lines)
        self.assertIn("100 decodes planned in 6 runs: 100 records", lines)
        self.assertIn("38 observations for 38 tapped grids", lines)
        self.assertIn("with the observer attached every decode requested from the CUDA processor ran its resident path",
                      lines)
        # The evidence.
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual((manifest["selector"], manifest["axes"]["receiver_stage"], manifest["host_rehearsal"]),
                         ("pilot_ls", "o2_pilot_ls", False))
        self.assertEqual([item["name"] for item in manifest["runs"]],
                         ["unity/shipped-1rx", "unity/slot-grants", "direct", "reference-pair", "payload", "tap-off"])
        self.assertEqual([item["tap"] for item in manifest["runs"]], ["pilot_ls"] * 5 + ["none"])
        self.assertEqual(manifest["control"]["judgment"], "pass")
        for name in ("grade.json", "grade.txt", "comparison.json", "manifest.json", "adapter-receipt.json"):
            self.assertTrue((out / name).is_file(), name)
        self.assertTrue((out / "reference_pair/run/native/observer/o2.jsonl").is_file())
        self.assertFalse((out / "tap_off/run/native/observer").exists())
        comparison = json.loads((out / "comparison.json").read_text())
        self.assertEqual(comparison["direct"]["inside"], comparison["direct"]["estimates"])
        self.assertEqual(comparison["backends"]["observations"], comparison["backends"]["tapped_grids"])
        self.assertEqual(len(json.loads((out / "comparison.json").read_text())["refusals"]), 14)

    def test_the_driver_never_overwrites_a_run_and_refuses_a_bad_setup(self):
        with self.assertRaises(SystemExit) as stopped, contextlib.redirect_stderr(io.StringIO()):
            self.driver.main(["--config", str(self.config), "--s24-run", str(self.control), "--output-dir",
                              str(self.reference["out"])])
        self.assertEqual(stopped.exception.code, 2)
        changes = (lambda c: c.update(backends=["cpu"]), lambda c: c["observer"].update(tap="none"),
                   lambda c: c["tolerance"].pop("phase_rad"), lambda c: c["control"]["runs"].pop("payload"),
                   lambda c: c.update(selector="replay_pusch"), lambda c: c["observer"].update(epoch=0),
                   lambda c: c["unity"]["expect"].pop("estimate"))
        for number, change in enumerate(changes):
            config = json.loads(self.config.read_text())
            change(config)
            path = self.out / f"config-{number}.json"
            path.write_text(json.dumps(config))
            with self.assertRaises(SystemExit) as stopped, contextlib.redirect_stderr(io.StringIO()):
                self.run_once(self.out / f"run-{number}", config=path)
            self.assertEqual(stopped.exception.code, 2, number)
            self.assertFalse((self.out / f"run-{number}").exists())

    # ---------------------------------------------------------------- the control
    def test_a_run_that_sprint_2_4_did_not_accept_is_no_control(self):
        copied = self.out / "control"
        shutil.copytree(self.control, copied)
        grade = json.loads((copied / "grade.json").read_text())
        grade["hardware_gate"] = "open"
        (copied / "grade.json").write_text(json.dumps(grade))
        run = self.run_once(self.out / "open-gate", control=copied)
        self.assertEqual(run["status"], 1)
        self.assertIn("control accepted run", run["failed"])
        # A plan that was edited after the run is refused, and nothing is compared with it.
        copied = self.out / "edited"
        shutil.copytree(self.control, copied)
        plan = copied / "reference_pair/run/plan.json"
        plan.write_text(plan.read_text() + "\n")
        run = self.run_once(self.out / "edited-plan", control=copied)
        self.assertEqual(run["status"], 1)
        self.assertIn("control records", run["failed"])
        self.assertIn("reference-pair control", run["failed"])
        self.assertIn("plan.json does not have the hash the manifest of the Sprint 2.4 run names", run["lines"])
        # A directory that is no run of the Sprint 2.4 driver.
        run = self.run_once(self.out / "no-control", control=self.out)
        self.assertEqual((run["status"], run["grade"]["cases_passed"]), (1, 0))

    # ---------------------------------------------------------------- faults of the estimate
    def test_an_estimate_against_the_stored_pilot_fails_unity_and_the_reference(self):
        run = self.faulty("stored-pilot")
        self.assertEqual(run["status"], 1)
        for check in ("unity slot-grants estimate", "unity shipped-1rx estimate", "reference-pair reference"):
            self.assertIn(check, run["failed"])
        self.assertIn("values 1.0000000 to 1.0000000", run["lines"])
        # The decodes are untouched by a wrong estimate: the join to the control still holds.
        self.assertNotIn("reference-pair decodes", run["failed"])

    def test_a_missing_conjugate_fails_the_known_channel_and_both_oracles(self):
        run = self.faulty("conjugate")
        for check in ("direct known channel", "reference-pair reference", "reference-pair sparse oracle"):
            self.assertIn(check, run["failed"])

    def test_reversed_receive_planes_fail_the_known_channel(self):
        run = self.faulty("swap-planes")
        for check in ("direct known channel", "direct receive planes", "reference-pair reference"):
            self.assertIn(check, run["failed"])

    def test_shifted_coordinates_fail_unity(self):
        run = self.faulty("shifted")
        for check in ("unity slot-grants estimate", "direct known channel", "reference-pair sparse oracle"):
            self.assertIn(check, run["failed"])

    def test_an_estimate_that_depends_on_data_fails_the_payload_case(self):
        run = self.faulty("data-dependent")
        for check in ("payload data invariance", "reference-pair reference"):
            self.assertIn(check, run["failed"])

    def test_a_wrong_validity_bit_and_a_wrong_slot_fail_the_reference(self):
        run = self.faulty("mask-clear")
        for check in ("reference-pair reference", "reference-pair sparse oracle"):
            self.assertIn(check, run["failed"])
        run = self.faulty("wrong-slot")
        self.assertIn("reference-pair reference", run["failed"])
        self.assertIn("0 of 12 carry the slot count", run["lines"])

    # ---------------------------------------------------------------- faults of the tap
    def test_a_tap_that_changes_a_decode_fails_the_join_to_the_control(self):
        run = self.faulty("changed-decode")
        for check in ("reference-pair decodes", "tap-off records", "unity native run", "payload native run"):
            self.assertIn(check, run["failed"])
        self.assertIn("outcome is 'payload_mismatch' with the tap and 'decoded' without", run["lines"])

    def test_a_tap_that_writes_the_grid_fails_the_ingress_join(self):
        run = self.faulty("touch-grid")
        for check in ("reference-pair ingress", "reference-pair decodes"):
            self.assertIn(check, run["failed"])
        self.assertIn("11 of 12 observations name the readback digest", run["lines"])

    def test_an_observation_of_another_grid_a_missing_one_and_a_leak_fail_the_reader(self):
        for fault, words in (("other-grid", "does not name the grid of its opportunity"),
                             ("drop-observation", "observation records for"),
                             ("leak", "host does not hold exactly its fields")):
            run = self.faulty(fault)
            self.assertIn("reference-pair native run", run["failed"], fault)
            self.assertIn(words, run["lines"], fault)

    def test_observations_without_a_tap_fail_the_tap_off_case(self):
        run = self.faulty("tap-when-off")
        self.assertIn("tap-off native run", run["failed"])
        self.assertIn("a run without a tap holds an observation", run["lines"])

    def test_a_fallback_to_the_host_path_cannot_close_the_hardware_gate(self):
        run = self.faulty("host-fallback")
        self.assertEqual((run["status"], run["grade"]["judgment"], run["grade"]["hardware_gate"]), (1, "fail", "open"))
        self.assertIn("backends executed path", run["failed"])
        self.assertIn("reference-pair decodes", run["failed"])  # the executed path differs from the control's

    def test_a_tool_that_refuses_nothing_and_one_that_exits_badly_fail(self):
        run = self.faulty("accept-all")
        self.assertIn("refusals plan rules", run["failed"])
        self.assertIn("NOT REFUSED AS REQUIRED", run["lines"])
        run = self.faulty("exit")
        for check in ("unity native run", "direct native run", "reference-pair native run", "backends accounting"):
            self.assertIn(check, run["failed"])
        self.assertEqual(run["grade"]["cases_passed"], 1)

    # ---------------------------------------------------------------- rehearsal
    def test_a_host_rehearsal_is_graded_and_never_closes_the_hardware_gate(self):
        rehearsed = self.stage.run_once(self.out / "s24", environment={"REPLAY_PUSCH_BUILD": "host"},
                                        extra=("--host-rehearsal",))
        self.assertEqual((rehearsed["status"], rehearsed["grade"]["judgment"]), (0, "rehearsal"))
        run = self.run_once(self.out / "rehearsal", control=rehearsed["out"], extra=("--host-rehearsal",),
                            environment={"PILOT_LS_BUILD": "host"})
        grade = run["grade"]
        self.assertEqual((run["status"], run["failed"], grade["judgment"]), (0, [], "rehearsal"))
        self.assertTrue(grade["hardware_gate"].startswith("open"))
        self.assertIn("host rehearsal: the hardware gate stays open", run["lines"])
        self.assertIn("REHEARSAL on a build without CUDA, not a hardware result", run["lines"])
        # A rehearsed control cannot be the control of an acceptance, and the reverse.
        run = self.run_once(self.out / "mixed", control=rehearsed["out"])
        self.assertIn("control accepted run", run["failed"])
        run = self.run_once(self.out / "claimed", extra=("--host-rehearsal",))
        self.assertIn("control accepted run", run["failed"])

    def test_the_driver_runs_the_tool_as_a_process(self):
        run = self.run_once(self.out / "processes", processes=True)
        self.assertEqual((run["status"], run["failed"], run["grade"]["checks"]), (0, [], CHECKS))
        log = (run["out"] / "reference_pair/run/native.log").read_text()
        self.assertIn(f"{self.tool} --plan ", log)
        self.assertIn("pilot_ls double: s31-pair, tap pilot_ls, 12 observations, 12 opportunities", log)


if __name__ == "__main__":
    unittest.main()
