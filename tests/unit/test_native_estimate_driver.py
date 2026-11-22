"""Check the Sprint 5.2 integration driver against a double of the patched tool and of the consumers of its streams.

The control is a Sprint 3.1 run that the driver of that sprint grades on the doubles of its own
driver tests, on top of a Sprint 2.4 run that is graded the same way: three flat grants, the
direct channel and the small pair of two TDD periods. The double of this sprint
(tests/unit/native_estimate_double.py) decodes those plans again with the model receiver and
makes, for every decode on the resident path, the estimate that the matched estimator makes of
the stage O2, so every check of the driver has a known answer, and each injected fault must fail
the check that names it. NumPy is needed, as for the drivers this one is stacked on.
"""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    np = None

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import native_estimate_double as double  # noqa: E402
import test_pilot_ls_driver as control_tests  # noqa: E402

CASES = 10
ENVIRONMENT = ("NATIVE_ESTIMATE_FAULT", "SNAPSHOT_RING_FAULT", "PILOT_LS_FAULT", "PILOT_LS_BUILD", "REPLAY_PUSCH_FAULT",
               "REPLAY_PUSCH_BUILD")


def load_driver():
    spec = importlib.util.spec_from_file_location("native_estimate_driver", ROOT / "tests/integration/native_estimate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(np is None, "the integration driver needs NumPy")
class DriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.driver = load_driver()
        # The accepted Sprint 3.1 run: the driver of that sprint on the doubles of its tools.
        cls.stage = control_tests.DriverTests
        cls.stage.setUpClass()
        cls.addClassCleanup(cls.stage.tearDownClass)
        assert cls.stage.reference["status"] == 0, cls.stage.reference["failed"]
        cls.control = cls.stage.reference["out"]
        cls.root = cls.stage.root / "s52"
        cls.root.mkdir()
        config = json.loads((ROOT / "configs/integration/native_estimate.json").read_text())
        config["expect"]["pair_opportunities"] = 12  # the small pair
        cls.config = cls.root / "config.json"
        cls.config.write_text(json.dumps(config))
        cls.tool = cls.root / "tool"
        cls.tool.write_text("a double\n")
        cls.reference = cls.run_once(cls.root / "reference")

    @classmethod
    def receipts(cls):
        series = cls.driver.receiver_patch.identity(cls.driver.receiver_patch.load_series())
        tools = {name: {"path": str(cls.tool), "sha256": hashlib.sha256(cls.tool.read_bytes()).hexdigest()}
                 for name in ("pilot_ls", "obs_tail", "obs_feed", "replay_pusch")}
        return {"control": {"executables": tools, "ocudu_build": "double", "pin": {"cuda": True}},
                "o3": {"executables": tools, "ocudu_build": "double-o3", "pin": {"cuda": True},
                       "patch": dict(series, tree_kind="patched-worktree"),
                       "o3_route": {"compiled": True, "tap_interface_version": 1}}}

    @classmethod
    def run_once(cls, out, fault=None, config=None, receipts=None):
        driver = cls.driver
        saved = (driver.native_adapter.verified_receipt, driver.run_scenario)
        chosen = receipts or cls.receipts()
        driver.native_adapter.verified_receipt = lambda build: chosen["o3" if "o3" in str(build) else "control"]
        driver.run_scenario = double.scenario
        kept = {key: os.environ.get(key) for key in ENVIRONMENT}
        for key in kept:
            os.environ.pop(key, None)
        if fault:
            os.environ["NATIVE_ESTIMATE_FAULT"] = fault
        arguments = ["--config", str(config or cls.config), "--s31-run", str(cls.control), "--output-dir", str(out)]
        try:
            with contextlib.redirect_stdout(io.StringIO()) as said:
                status = driver.main(arguments)
        finally:
            driver.native_adapter.verified_receipt, driver.run_scenario = saved
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
        self.assertEqual((run["status"], run["failed"]), (0, []), lines[-3000:])
        self.assertEqual((grade["judgment"], grade["hardware_gate"], grade["cases"], grade["cases_passed"]),
                         ("pass", "closed", CASES, CASES))
        self.assertIn(f"native_estimate: {CASES} of {CASES} cases passed, {grade['checks']} of {grade['checks']} checks\n",
                      lines)
        for line in (
                "PASS  S5.2 control accepted run: the Sprint 3.1 driver graded reference as pass",
                "PASS  S5.2 patch series: patches/ocudu/series.json",
                "0001-pusch-resident-estimate-tap.patch with the SHA-256",
                "PASS  S5.2 flat value slot-grants: every element of",
                "the unity channel reads 1.0011865 in the stage O2 and 1.0009765625 in the FP16 of the receiver",
                "PASS  S5.2 flat element: element [1, 3, 2, 5] of the payload is position 2621: DM-RS symbol 1 (symbol 11 "
                "of the slot), receive port 2 (array element 2) and subcarrier 41 of the allocation",
                "PASS  S5.2 varying parity: ",
                "PASS  S5.2 noisy parity: 12 of 12 estimates meet the declared tolerance against the matched estimator",
                "PASS  S5.2 noisy stage O2: 12 of 12 payloads and validity masks of the stage O2 have the SHA-256 that "
                "the accepted Sprint 3.1 run recorded",
                "PASS  S5.2 hook-off decodes: 24 radio records equal those the accepted replay_pusch wrote without an "
                "observer",
                "PASS  S5.2 side-stream capture: 12 decodes of the CUDA processor: 0 published, 12 captured",
                "copied on the stream side and published on its completion event",
                "PASS  S5.2 timed no checksum: between the first fill and the drain of both rings the hash of the core "
                "computed 0 digests over 0 bytes",
                "PASS  S5.2 timed account: 12 decodes of the CUDA processor: 12 published",
                "PASS  S5.2 timed records: the consumer of the O3 stream kept 12 records, sequences 0 to 11 in order",
                "PASS  S5.2 refusals rules: 6 of 6 broken plans are refused by the plan check and by the patched tool",
                "PASS  S5.2 refusals no route: the pilot_ls of the build against the unpatched receiver refuses a plan "
                "that names an estimate with exit status 2",
                "PASS  S5.2 backends provenance: ",
                "cuda on the resident path, estimate captured:",
                "cpu on the host path, estimate none:",
                "NOTE  S5.2 hook-off CUDA PUSCH service time"):
            self.assertIn(line, lines)
        parity = json.loads((out / "o3_parity.json").read_text())
        self.assertEqual((parity["schema"], parity["passed"], parity["stage"]), ("o3-parity/v1", True, "O3"))
        self.assertEqual(sorted({item["case"] for item in parity["runs"]}), ["flat", "noisy", "side-stream", "varying"])
        total = parity["totals"]
        self.assertGreater(total["samples"], 0)
        self.assertAlmostEqual(total["nmse"], total["numerator"] / total["denominator"])
        self.assertEqual(total["samples"], sum(row["pilot_support"]["samples"] for item in parity["runs"]
                                               for row in item["rows"]))
        self.assertEqual(parity["patch"]["patches"][0]["file"], "0001-pusch-resident-estimate-tap.patch")
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual(manifest["axes"]["receiver_stage"], "o3_native_estimate")
        traced = [run for run in manifest["runs"] if run["trace"]]
        self.assertTrue(traced and all((out / run["trace"]).is_file() for run in traced))
        for name in ("grade.txt", "o3_parity.json", "manifest.json", "comparison.json"):
            self.assertTrue((out / name).read_bytes().isascii(), name)

    # ---------------------------------------------------------------- faults
    def test_a_zero_filled_buffer_fails_the_nonzero_reference(self):
        run = self.faulty("zero-buffer")
        self.assertEqual(run["status"], 1)
        for check in ("flat parity slot-grants", "flat value slot-grants", "noisy parity", "varying parity",
                      "timed records"):
            self.assertIn(check, run["failed"])
        # The decodes and the stage O2 are untouched: only the exported estimate is wrong.
        self.assertNotIn("noisy decodes", run["failed"])
        self.assertNotIn("noisy stage O2", run["failed"])

    def test_an_estimate_that_was_never_smoothed_fails_where_smoothing_matters(self):
        run = self.faulty("no-smoothing")
        self.assertEqual(run["status"], 1)
        self.assertIn("noisy parity", run["failed"])
        self.assertIn("side-stream parity", run["failed"])
        # A flat channel is filtered to itself: that case cannot tell, and says so by passing.
        self.assertNotIn("flat parity slot-grants", run["failed"])

    def test_a_scale_a_stale_buffer_and_a_port_order_each_fail_their_check(self):
        # The ratio of the two reference amplitudes is 1.00119, about one FP16 spacing at unity: the
        # tolerance of one spacing cannot see it, and the exact value of the flat estimate does.
        scaled = self.faulty("wrong-scale")
        self.assertEqual(scaled["failed"], ["flat value shipped-1rx", "flat value slot-grants"])
        stale = self.faulty("stale")
        self.assertIn("noisy parity", stale["failed"])
        self.assertNotIn("flat parity slot-grants", stale["failed"])
        swapped = self.faulty("port-swap")
        self.assertIn("noisy parity", swapped["failed"])

    def test_a_record_before_its_completion_and_a_lost_record_fail(self):
        early = self.faulty("early-publish")
        self.assertIn("timed native run", early["failed"])
        self.assertIn("the times of a published estimate are not in order", early["lines"])
        lost = self.faulty("lost-record")
        self.assertIn("timed records", lost["failed"])

    def test_a_host_fallback_and_a_build_that_accepts_without_the_route_fail(self):
        fallback = self.faulty("host-fallback")
        self.assertEqual(fallback["status"], 1)
        self.assertIn("backends provenance", fallback["failed"])
        self.assertIn("noisy capture", fallback["failed"])
        accepting = self.faulty("unpatched-accepts")
        self.assertEqual(accepting["failed"], ["refusals no route"])

    def test_a_receipt_without_the_series_fails_the_patch_case(self):
        receipts = self.receipts()
        receipts["o3"] = dict(receipts["o3"], patch=dict(receipts["o3"]["patch"], series_sha256="0" * 64))
        run = self.run_once(self.out / "series", receipts=receipts)
        self.assertEqual(run["failed"], ["patch patched receiver"])
        receipts = self.receipts()
        receipts["control"] = dict(receipts["control"], ocudu_build="double-o3")
        run = self.run_once(self.out / "control", receipts=receipts)
        self.assertEqual(run["failed"], ["patch control build"])

    def test_a_declared_tolerance_is_applied(self):
        config = json.loads(self.config.read_text())
        config["tolerance"]["flat"]["abs_ulps"] = 0.0
        strict = self.out / "config.json"
        strict.write_text(json.dumps(config))
        self.assertEqual(self.run_once(self.out / "strict", config=strict)["status"], 0)  # the double is exact
        config["tolerance"]["race_free_ulps"] = -1
        strict.write_text(json.dumps(config))
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                self.run_once(self.out / "refused", config=strict)


if __name__ == "__main__":
    unittest.main()
