#!/usr/bin/env python3
"""The Sprint 3.6 integration driver on a stand-in control: python tests/unit/test_sense_core_driver.py -v

The driver, tests/integration/sense_core.py, is a hardware command. Here it runs as a rehearsal:
on the host twin of the core with the engine host_reference, against a stand-in for the accepted
Sprint 3.5 run that tests/unit/sense_core_double.py makes from synthetic records and a model
with random weights. The rehearsal grades every case but the checks of a device, writes the
three records of the card, and leaves its hardware gate open. The module also checks that the
driver refuses a configuration, a build and a control it cannot grade, before it grades anything.

It needs cmake, a C++17 compiler, NumPy, PyTorch, ONNX and pybind11, and no CUDA.
"""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

try:
    import numpy as np
    import onnx  # noqa: F401
    import torch  # noqa: F401
except ImportError:
    np = None

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import native_core  # noqa: E402
import sense_core_build  # noqa: E402

DRIVER = ROOT / "tests/integration/sense_core.py"
BUILD = SCRATCH = PROBLEM = CONTROL = None


def setUpModule():
    global BUILD, SCRATCH, PROBLEM, CONTROL
    if np is None:
        PROBLEM = "the rehearsal needs NumPy, PyTorch and ONNX in this interpreter"
        return
    BUILD, PROBLEM = sense_core_build.host_twin()
    if PROBLEM is not None:
        return
    import sense_core_double as double
    SCRATCH = tempfile.TemporaryDirectory()
    CONTROL = double.make_control(Path(SCRATCH.name), BUILD / "sense_feed")


def tearDownModule():
    if SCRATCH is not None:
        SCRATCH.cleanup()


class DriverTests(unittest.TestCase):
    def setUp(self):
        if PROBLEM is not None:
            self.skipTest(PROBLEM)
        import sense_core_double as double
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.scratch = Path(scratch.name)
        self.config = double.rehearsal_config(BUILD)

    def drive(self, config=None, control=None, rehearsal=True, out=None):
        path = self.scratch / "config.json"
        path.write_text(json.dumps(self.config if config is None else config))
        command = [sys.executable, str(DRIVER), "--config", str(path), "--s35-run", str(control or CONTROL),
                   "--output-dir", str(out or self.scratch / "out")] + (["--rehearsal"] if rehearsal else [])
        return subprocess.run(command, capture_output=True, text=True, timeout=900)

    def test_a_rehearsal_grades_every_case_and_leaves_the_hardware_gate_open(self):
        done = self.drive()
        self.assertEqual(done.returncode, 0, done.stdout[-3000:] + done.stderr[-2000:])
        out = self.scratch / "out"
        grade = json.loads((out / "grade.json").read_text())
        self.assertEqual((grade["judgment"], grade["cases"], grade["cases_passed"], grade["checks_failed"],
                          grade["rehearsal"], grade["runtime"], grade["engine"]),
                         ("pass", 8, 8, 0, True, "native", "host_reference"))
        self.assertTrue(grade["hardware_gate"].startswith("open"))
        self.assertEqual(sorted({item["case"] for item in grade["results"]}),
                         sorted(("core", "control", "engine", "units", "parity", "live", "overload", "reference")))
        lines = (out / "grade.txt").read_text()
        self.assertNotIn("FAIL  ", lines)
        self.assertIn("NOTE  S3.6 engine rehearsal: no TensorRT engine is built", lines)
        # The three records of the card, and the receipt of the build.
        receipt = json.loads((out / "isac-core.json").read_text())
        self.assertEqual(receipt["sources_sha256"], native_core.sources_sha256())
        parity = json.loads((out / "runtime_parity.json").read_text())
        totals = parity["totals"]
        self.assertEqual((parity["schema"], totals["problems"], totals["flips"], totals["decisions_agree"]),
                         ("sense-runtime-parity/v1", 0, [], totals["windows"]))
        self.assertGreaterEqual(totals["windows"], 20)
        self.assertLessEqual(totals["largest_input_difference_of_its_bound"], 1.0)
        self.assertEqual(parity["tolerance"]["fp32"]["logit_absolute"], 1e-3)
        latency = json.loads((out / "runtime_latency.json").read_text())
        self.assertEqual((latency["schema"], latency["graded"], latency["native"]["timing_valid"]),
                         ("sense-runtime-latency/v1", False, False))
        for name in ("end_to_end", "post_window", "cusense_boundary"):
            part = latency["native"][name]
            self.assertEqual(part["count"], latency["native"]["scored"])
            self.assertTrue(0 < part["p50_us"] <= part["p99_us"] <= part["max_us"], name)
            self.assertIn(name, latency["definitions"])
        self.assertEqual(latency["native"]["feed"]["dropped"], 0)
        self.assertEqual(latency["reference"]["runtime"], "reference")
        self.assertIn("loadavg", latency["host_state"]["before"])
        reference = json.loads((out / "cusense_reference.json").read_text())
        measured = reference["measured"]["latency"]
        self.assertEqual((reference["schema"], reference["graded"], measured["published_us"], measured["comparable"]),
                         ("cusense-reference/v1", False, 241, False))
        self.assertIn("blocked", measured["why_not_comparable"])
        text = " ".join(reference["differences"])
        for word in ("20 MHz", "GB10", "GH200", "simulated scene", "not open source"):
            self.assertIn(word, text)
        # The overload case left an output that verifies and counts what it lost.
        counts = json.loads((out / "comparison.json").read_text())["overload"]
        self.assertGreater(counts["feed"]["dropped"], 0)
        self.assertEqual(counts["feed"]["pending_at_cutoff"], 0)
        # Evidence is never overwritten.
        again = self.drive()
        self.assertEqual(again.returncode, 2)
        self.assertIn("already exists", again.stderr)

    def test_the_driver_refuses_what_it_cannot_grade(self):
        done = self.drive(config=dict(self.config, schema="integration-config/other/v1"))
        self.assertEqual(done.returncode, 2)
        self.assertIn("integration-config/sense_core/v1", done.stderr)
        self.assertFalse((self.scratch / "out").exists())
        broken = json.loads(json.dumps(self.config))
        broken["live"]["budget_us"] = [0, 0, 0]
        self.assertEqual(self.drive(config=broken).returncode, 2)
        done = self.drive(control=self.scratch)
        self.assertEqual(done.returncode, 2)
        self.assertIn("recover.json", done.stderr)
        # Without --rehearsal the twin is no build for the gate.
        done = self.drive(rehearsal=False)
        self.assertEqual(done.returncode, 2)
        self.assertIn("host twin", done.stderr)
        done = self.drive(config=dict(self.config, core_build=str(self.scratch / "missing")))
        self.assertEqual(done.returncode, 2)
        self.assertIn("holds no receipt", done.stderr)

    def test_an_artifact_that_is_not_the_frozen_one_fails_the_control_and_blocks_the_rest(self):
        control = self.scratch / "control"
        shutil.copytree(CONTROL, control)
        weights = control / "cusense_frontend/train/o3/artifact/weights.f32"
        data = bytearray(weights.read_bytes())
        data[40] ^= 1
        weights.write_bytes(bytes(data))
        done = self.drive(control=control)
        self.assertEqual(done.returncode, 1)
        grade = json.loads((self.scratch / "out/grade.json").read_text())
        self.assertEqual((grade["judgment"], grade["cases_passed"]), ("fail", 1))
        failed = [item for item in grade["results"] if not item["passed"]]
        self.assertIn("the checkpoint was altered", failed[0]["detail"])
        self.assertEqual({item["case"] for item in failed}, {"control", "engine", "units", "parity", "live", "overload",
                                                              "reference"})
        self.assertFalse((self.scratch / "out/runtime_parity.json").exists())


if __name__ == "__main__":
    unittest.main()
