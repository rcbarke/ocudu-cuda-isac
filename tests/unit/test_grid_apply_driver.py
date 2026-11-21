"""Check the Sprint 2.3 integration driver against a double of the native tool.

The stimulus is a small pair on the demonstration grant: two TDD periods, six opportunities, six
slot grants, four receive ports. The double (tests/unit/grid_apply_double.py) applies the reference
arithmetic, so every check of the driver has a known answer, and each injected fault must fail
the check that names it. NumPy is needed, as for the driver.
"""

import contextlib
import hashlib
import importlib.util
import io
import json
import math
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
from ocudu_cuda_isac.episodes import episode as episodes  # noqa: E402
from ocudu_cuda_isac.episodes import grant as grants  # noqa: E402
from ocudu_cuda_isac.episodes import schedule as schedules  # noqa: E402
from ocudu_cuda_isac.episodes.episode import TC_PER_SECOND  # noqa: E402
import grid_apply_double as double  # noqa: E402

DATA = complex(0.31640625, 0.31640625)  # a stored 16QAM point
SCALE = 3.4e-4
EXECUTABLE = '''#!{python}
import sys
sys.path.insert(0, {tests!r})
import grid_apply_double
sys.exit(grid_apply_double.main())
'''


def load_driver():
    spec = importlib.util.spec_from_file_location("grid_apply_driver", ROOT / "tests/integration/grid_apply.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def store_of(grant, directory):
    """A store entry with a transmit grid the grant accepts: its pilots and one data point."""
    width = grant.grid_subcarriers
    x = [0j] * (14 * width)
    for (symbol, k), value in zip(grant.pilot_index, grant.pilot_reference[0]):
        x[symbol * width + k] = complex(grants.bf16(value.real), grants.bf16(value.imag))
    for symbol, k in grant.data_index:
        x[symbol * width + k] = DATA
    grants.export(grant, directory, {"x": {"values": x, "provenance": {"double": True}}})
    return json.loads((directory / "manifest.json").read_text())["files"]["x.npy"]["sha256"]


def make_run(root):
    """What the accepted Sprint 2.1 and Sprint 2.2 runs leave: the grant stores and a matched pair."""
    base = grants.load(ROOT / "configs/grants/demo_51prb_4rx.json")
    schedule = schedules.build({"schema": schedules.SCHEMA, "schedule_id": "unit",
                                "tdd": {"pattern": "DDDDDDSUUU", "ul_slots": [7, 8, 9]}, "start_sfn": 0, "periods": 2,
                                "payload": "fixed"}, base)
    s21, s22 = root / "s21/grant_fixture", root / "s22/episode_pair"
    waveforms = {}
    for grant_id, grant in schedule.grants.items():
        directory = (s21 if grant is base else s22 / "grant_fixture") / grant_id / "store"
        waveforms[grant_id] = {"x_sha256": store_of(grant, directory), "store": str(directory)}
    store_of(grants.load(ROOT / "configs/grants/two_prb_offset.json"), s21 / "two-prb-offset" / "store")
    h_shape, w_shape = episodes.shapes(schedule)
    opportunity, rx, symbol, k = np.meshgrid(*(np.arange(size) for size in (h_shape[0], h_shape[1], 14, h_shape[4])),
                                             indexing="ij")
    direct = SCALE * np.exp(1j * (0.4 * rx - 0.011 * k))
    echo = 0.0078 * SCALE * np.exp(1j * (0.9 * rx - 0.03 * k - 0.3 * symbol - 1.1 * opportunity))
    h_a = direct.astype("<c8")[:, :, None]
    h_b = (direct + echo).astype("<c8")[:, :, None]
    symbols, subcarriers = zip(*sorted(schedule.opportunities[0].grant.occupied()))
    power = float(np.mean(np.abs(h_a[0, :, 0][:, list(symbols), list(subcarriers)]) ** 2))
    sigma = math.sqrt(power / 1000.0)
    generator = np.random.Generator(np.random.PCG64(20261120))
    w = ((generator.standard_normal(w_shape) + 1j * generator.standard_normal(w_shape)) * sigma / math.sqrt(2)).astype("<c8")
    motion = {"kind": "constant_velocity", "position_m": [10.0, 25.0, 1.5], "velocity_m_s": [0.0, 2.0, 0.0]}
    geometry = {"ue_tx_m": [0.0, 0.0, 1.5], "gnb_rx_m": [20.0, 0.0, 3.0],
                "rx_element_offsets_m": [[0.0, 0.02 * n, 0.0] for n in range(4)]}
    description = {"geometry": geometry, "array": {}, "scene": {"kind": "free_space"},
                   "target": {"model": "constant_rcs_cuboid", "sigma_m2": 1.0, "motion": motion}, "solver": {},
                   "time_evolution": {}, "noise": {"snr_db": 30.0, "sigma": sigma, "seed": 20261120},
                   "calibration": {"reference_power": power, "grid_gain": 1.0 / math.sqrt(power)},
                   "software": {}, "provenance": {}}
    truth = []
    for item in schedule.opportunities:
        position, velocity = episodes.target_state(motion, item.slot_start_tc / TC_PER_SECOND)
        truth.append({"opportunity": item.index, "position_m": position, "velocity_m_s": velocity})
    episodes.export(s22 / "reference_pair/store", "unit-pair", schedule, description,
                    {"a": h_a.tobytes(), "b": h_b.tobytes(), "w": w.tobytes()}, truth, None, waveforms)
    return {"gain": 1.0 / math.sqrt(power), "g_sigma": sigma / math.sqrt(power), "run": root / "s22"}


@unittest.skipIf(np is None, "the integration driver needs NumPy")
class DriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.driver = load_driver()
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.made = make_run(cls.root)
        config = json.loads((ROOT / "configs/integration/grid_apply.json").read_text())
        config["pair"]["expect"].update(opportunities=6, grid_gain=cls.made["gain"], g_sigma=cls.made["g_sigma"])
        cls.config = cls.root / "config.json"
        cls.config.write_text(json.dumps(config))
        cls.tool = cls.root / "grid_apply"
        cls.tool.write_text(EXECUTABLE.format(python=sys.executable, tests=str(Path(__file__).resolve().parent)))
        cls.tool.chmod(0o755)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        driver = self.driver
        receipt = {"executables": {"grid_apply": {"path": str(self.tool),
                                                  "sha256": hashlib.sha256(self.tool.read_bytes()).hexdigest()},
                                   "grant_fixture": {"path": "/nonexistent", "sha256": "0" * 64}},
                   "ocudu_build": "double"}
        self.receipt = receipt
        self.saved = (driver.native_adapter.verified_receipt, driver.run_native, os.environ.get("GRID_APPLY_FAULT"))
        driver.native_adapter.verified_receipt = lambda build_dir: self.receipt
        driver.run_native = self.in_process
        self.addCleanup(self.restore)
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.out = Path(scratch.name)

    def restore(self):
        self.driver.native_adapter.verified_receipt, self.driver.run_native, fault = self.saved
        os.environ.pop("GRID_APPLY_FAULT", None)
        if fault is not None:
            os.environ["GRID_APPLY_FAULT"] = fault

    @staticmethod
    def in_process(tool, plan_path, output, log):
        """The double without a new interpreter: the same files, the same status, the same last line."""
        output.mkdir(parents=True)
        said = io.StringIO()
        with contextlib.redirect_stdout(said), contextlib.redirect_stderr(said):
            try:
                status = double.main(["--plan", str(plan_path), "--output-dir", str(output)])
            except Exception as error:  # a process that crashes exits with status 1
                said.write(f"{type(error).__name__}: {error}\n")
                status = 1
        log.write_text(f"{tool} --plan {plan_path} --output-dir {output}\n" + said.getvalue())
        try:
            return status, json.loads((output / "native.json").read_text())
        except (OSError, ValueError):
            return status, None

    def run_driver(self, name, fault=None, config=None, run=None, extra=()):
        os.environ.pop("GRID_APPLY_FAULT", None)
        if fault:
            os.environ["GRID_APPLY_FAULT"] = fault
        out = self.out / name
        with contextlib.redirect_stdout(io.StringIO()):
            status = self.driver.main(["--config", str(config or self.config), "--s22-run", str(run or self.made["run"]),
                                       "--output-dir", str(out), *extra])
        grade = json.loads((out / "grade.json").read_text())
        failed = sorted(f"{item['case']} {item['check']}" for item in grade["results"] if not item["passed"])
        return status, grade, failed, out

    def test_the_reference_passes_every_check(self):
        status, grade, failed, out = self.run_driver("pass")
        self.assertEqual((status, failed), (0, []))
        self.assertEqual((grade["judgment"], grade["cases"], grade["cases_passed"], grade["checks"]), ("pass", 5, 5, 25))
        lines = (out / "grade.txt").read_text()
        self.assertIn("grid_apply: 5 of 5 cases passed, 25 of 25 checks", lines)
        self.assertIn("g = 1: 1+3j, 2-4j, 6.5+0j, 14+0j", lines)
        self.assertIn("g = 2: 2+6j, 4-8j, 13+0j, 28+0j", lines)
        self.assertIn("PASS  S2.3 unity slot 17 grid: demo-51prb-4rx.s17: 4 of 4 receive planes return x.npy", lines)
        self.assertIn("411264 of 411264 equal the reference converted to single precision bit for bit", lines)
        self.assertIn("12 of 12 readback digests equal the reference's", lines)
        self.assertIn("two-prb-offset occupies 312 elements on subcarriers 180 to 203", lines)
        self.assertIn("a missing gain, a zero gain, a negative gain, a gain that is not finite, two gains in one run, "
                      "a gain 2e-6 off 1 / sqrt(reference_power), an override on an entry", lines)
        self.assertIn("NOTE  S2.3 reference-pair fill time of one grid, not graded", lines)
        comparison = json.loads((out / "comparison.json").read_text())
        pair = comparison["reference-pair"]
        self.assertEqual(pair["stored"]["mismatched"], 0)
        self.assertAlmostEqual(pair["unit_power"]["python"], 1.0, delta=1e-6)
        self.assertAlmostEqual(pair["noise"]["held_db"], -30.0, delta=0.1)
        self.assertLessEqual(pair["difference"]["largest_error_over_bound"], 1.0)
        self.assertEqual(pair["access"]["violations"], [])
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual((manifest["judgment"], manifest["selector"]), ("pass", "grid_apply"))
        # The receiver side of the archived run: descriptors and stored grids only.
        ingress = sorted(path.name for path in (out / "reference_pair/native/ingress").iterdir())
        self.assertEqual(ingress, [f"{n:04d}.c64" for n in range(12)] + ["ingress.jsonl"])
        with self.assertRaises(SystemExit) as refused, contextlib.redirect_stderr(io.StringIO()):
            self.driver.main(["--config", str(self.config), "--s22-run", str(self.made["run"]),  # never overwritten
                              "--output-dir", str(out)])
        self.assertEqual(refused.exception.code, 2)

    def test_the_tool_is_started_as_a_separate_process(self):
        self.driver.run_native = self.saved[1]
        status, grade, failed, out = self.run_driver("process")
        self.assertEqual((status, failed, grade["checks"]), (0, [], 25))
        self.assertIn(str(self.tool), (out / "reference_pair/native.log").read_text())
        self.assertIn("grid_apply: input error:", (out / "refusals/gain-0/native.log").read_text())

    def test_arithmetic_faults_fail_the_comparisons(self):
        expected = {
            "noise-unscaled": ["reference-pair noise after storage", "reference-pair stored grids",
                               "reference-pair values before storage", "unit-vectors zero channel"],
            "single-gain": ["reference-pair stored grids", "reference-pair values before storage",
                            "unit-vectors zero channel"],
            "permuted": ["allocation changed grant", "reference-pair receive planes", "reference-pair second reader",
                         "reference-pair stored grids", "reference-pair values before storage",
                         "unit-vectors Sprint 0.3 vector"],
            "unrounded": ["reference-pair second reader", "reference-pair stored grids", "unit-vectors zero channel"],
        }
        for fault, wanted in expected.items():
            status, _, failed, out = self.run_driver(fault, fault=fault)
            with self.subTest(fault=fault):
                self.assertEqual(status, 1)
                for name in wanted:
                    self.assertIn(name, failed)
        # The gain in single precision stays inside the unit-scale tolerance: only the bit comparison sees it.
        lines = (self.out / "single-gain/grade.txt").read_text()
        self.assertRegex(lines, r"FAIL  S2\.3 reference-pair values before storage: .*every one within 1e-6")
        self.assertNotIn("411264 of 411264 equal the reference converted", lines)

    def test_binding_and_noise_faults_fail_their_checks(self):
        status, _, failed, out = self.run_driver("base", fault="base-grid")
        self.assertEqual(status, 1)
        for name in ("reference-pair grant of each opportunity", "reference-pair stored grids", "allocation changed grant"):
            self.assertIn(name, failed)
        self.assertIn("unity slot 8 grid", failed)
        self.assertNotIn("unity slot 7 grid", failed)
        status, _, failed, out = self.run_driver("noise", fault="other-noise")
        self.assertEqual(status, 1)
        self.assertIn("reference-pair difference of the two cases", failed)
        self.assertIn("reference-pair values before storage", failed)

    def test_access_and_state_faults_fail_their_checks(self):
        for fault, wanted, text in (
                ("leak-file", ["reference-pair access"], "the receiver side could read: file ingress/plan.json"),
                ("leak-field", ["reference-pair native run"], "is not an o1-grid/v1 record"),
                ("stale-digest", ["reference-pair native run"], "differs from its readback digest"),
                ("not-retained", ["reference-pair native run", "unity native run"], "FAIL  S2.3 unity native run"),
                ("exit", ["allocation changed grant", "reference-pair native run", "unit-vectors Sprint 0.3 vector",
                          "unity native run"], "exited with status 3")):
            status, _, failed, out = self.run_driver(fault, fault=fault)
            with self.subTest(fault=fault):
                self.assertEqual(status, 1)
                for name in wanted:
                    self.assertIn(name, failed)
                self.assertIn(text, (out / "grade.txt").read_text())

    def test_a_tool_that_refuses_nothing_fails_the_refusals(self):
        status, _, failed, out = self.run_driver("accept", fault="accept-all")
        self.assertEqual(status, 1)
        self.assertEqual([name for name in failed if name.startswith("refusals")],
                         ["refusals binding rules", "refusals gain rules"])
        self.assertIn("allocation another carrier", failed)
        self.assertIn("NOT REFUSED AS REQUIRED: ", (out / "grade.txt").read_text())

    def test_a_missing_stimulus_fails_the_cases_that_need_it(self):
        status, grade, failed, out = self.run_driver("empty", run=self.out / "no-such-run")
        self.assertEqual(status, 1)
        self.assertEqual(failed, ["allocation stimulus", "reference-pair stimulus", "refusals stimulus", "unity stimulus"])
        self.assertEqual((grade["cases_passed"], grade["judgment"]), (1, "fail"))
        self.assertIn("the stimulus of the Sprint 2.2 run is unusable", (out / "grade.txt").read_text())
        # The runs moved: the paths the entry recorded are gone. The slot grants are found beside the
        # pair, the base grant only when its directory is named.
        moved = self.out / "moved"
        shutil.copytree(self.made["run"], moved / "s22")
        manifest_path = moved / "s22/episode_pair/reference_pair/store/manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for item in manifest["grants"].values():
            item["x"]["store"] = "/nonexistent/" + item["x"]["store"]
        manifest_path.write_text(json.dumps(manifest))
        status, _, failed, out = self.run_driver("no-store", run=moved / "s22")
        self.assertEqual(status, 1)
        self.assertIn("no store entry of grant demo-51prb-4rx was found. Name the directory that holds "
                      "demo-51prb-4rx/store with --grant-store.", (out / "grade.txt").read_text())
        status, _, failed, out = self.run_driver("named-store", run=moved / "s22",
                                                 extra=("--grant-store", str(self.root / "s21/grant_fixture")))
        self.assertEqual((status, failed), (0, []))
        # A run that lacks the grant of the allocation case.
        saved = self.root / "s21/grant_fixture/two-prb-offset"
        saved.rename(self.root / "s21/held")
        try:
            status, _, failed, out = self.run_driver("no-changed-grant")
        finally:
            (self.root / "s21/held").rename(saved)
        self.assertEqual((status, failed), (1, ["allocation changed grant"]))
        self.assertIn("no verified store entry of the grant two-prb-offset", (out / "grade.txt").read_text())

    def test_an_expectation_of_another_run_fails_the_plan_and_gain_checks(self):
        config = json.loads(self.config.read_text())
        config["pair"]["expect"].update(grid_gain=2942.4243526773034, opportunities=24)
        path = self.out / "other.json"
        path.write_text(json.dumps(config))
        status, _, failed, _ = self.run_driver("other", config=path)
        self.assertEqual((status, failed), (1, ["reference-pair plan"]))

    def test_an_unusable_setup_grades_nothing(self):
        config = json.loads(self.config.read_text())
        for change in (lambda c: c.update(selector="episode_pair"), lambda c: c["tolerance"].pop("noise_db"),
                       lambda c: c["tolerance"].update(unit_scale_relative=1e-3),
                       lambda c: c["tolerance"].update(gain_rule_relative=1e-3),
                       lambda c: c["tolerance"].update(storage_relative=0.01), lambda c: c["pair"].pop("expect"),
                       lambda c: c.update(allocation={})):
            edited = json.loads(json.dumps(config))
            change(edited)
            path = self.out / "config.json"
            path.write_text(json.dumps(edited))
            with self.assertRaises(SystemExit) as refused, contextlib.redirect_stderr(io.StringIO()):
                self.driver.main(["--config", str(path), "--s22-run", str(self.made["run"]),
                                  "--output-dir", str(self.out / "never")])
            self.assertEqual(refused.exception.code, 2)
        # An adapter build from before this sprint holds no grid_apply.
        del self.receipt["executables"]["grid_apply"]
        with self.assertRaises(SystemExit) as refused, contextlib.redirect_stderr(io.StringIO()) as said:
            self.driver.main(["--config", str(self.config), "--s22-run", str(self.made["run"]),
                              "--output-dir", str(self.out / "never")])
        self.assertEqual(refused.exception.code, 2)
        self.assertIn("holds no grid_apply", said.getvalue())
        self.assertFalse((self.out / "never").exists())

    def test_double_gives_the_same_bytes_with_and_without_numpy(self):
        """The plain Python path of the double is the scalar reference; the NumPy path must equal it."""
        plan = self.out / "plan.json"
        entry = self.made["run"] / "episode_pair/reference_pair/store"
        self.driver.replay.write_plan(self.driver.replay.entry_plan("unit-both", [entry]), plan)
        digests = []
        for use_numpy in (True, False):
            saved, out = double.np, self.out / f"numpy-{use_numpy}"
            out.mkdir()
            double.np = np if use_numpy else None
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(double.main(["--plan", str(plan), "--output-dir", str(out), "--no-prestore"]), 0)
            finally:
                double.np = saved
            report = json.loads((out / "native.json").read_text())
            self.assertFalse((out / "prestore").exists())
            digests.append([item["readback_sha256"] for item in report["opportunities"]])
            self.assertAlmostEqual(report["realizations"][0]["first"]["unit_power"], 1.0, delta=1e-6)
        self.assertEqual(digests[0], digests[1])


if __name__ == "__main__":
    unittest.main()
