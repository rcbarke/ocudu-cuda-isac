"""Check the Sprint 3.6 gate runner where it decides before anything is built. No compiler, NumPy or GPU is needed."""

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/accept_s3_6.sh"


class AcceptS36Tests(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name) / "checkout"
        (self.root / "scripts").mkdir(parents=True)
        shutil.copy(SCRIPT, self.root / "scripts/accept_s3_6.sh")

    def run_gate(self, *arguments):
        return subprocess.run(["bash", str(self.root / "scripts/accept_s3_6.sh"), *arguments], capture_output=True,
                              text=True, timeout=120)

    def test_usage(self):
        done = self.run_gate("--help")
        self.assertEqual(done.returncode, 0)
        for word in ("--run-dir", "--s35-run", "--build-dir", "make accept-s3.5", "TENSORRT_ROOT"):
            self.assertIn(word, done.stdout)
        for arguments in (("--unknown",), ("--run-dir",), ("--s35-run",), ("--build-dir",)):
            done = self.run_gate(*arguments)
            self.assertEqual(done.returncode, 2, arguments)
            self.assertIn("Usage:", done.stderr)

    def test_without_the_environment_nothing_runs_and_evidence_is_never_overwritten(self):
        done = self.run_gate("--run-dir", str(self.root / "run"))
        self.assertEqual(done.returncode, 2)
        self.assertIn("Run 'make env' first", done.stderr)
        self.assertFalse((self.root / "run").exists())
        (self.root / "isac/bin").mkdir(parents=True)
        (self.root / "isac/bin/python").write_text("#!/bin/sh\nexit 1\n")
        (self.root / "isac/bin/python").chmod(0o755)
        (self.root / "run").mkdir()
        done = self.run_gate("--run-dir", str(self.root / "run"))
        self.assertEqual(done.returncode, 2)
        self.assertIn("Evidence is never overwritten", done.stderr)

    def test_a_host_without_tensorrt_and_without_a_control_is_not_accepted(self):
        # An interpreter that fails every command: the probe fails, no core is built, no control is found.
        (self.root / "isac/bin").mkdir(parents=True)
        (self.root / "isac/bin/python").write_text("#!/bin/sh\necho 'native_core: no TensorRT was found' >&2\nexit 1\n")
        (self.root / "isac/bin/python").chmod(0o755)
        done = self.run_gate("--run-dir", str(self.root / "run"))
        self.assertEqual(done.returncode, 1)
        summary = (self.root / "run/summary.txt").read_text()
        self.assertIn("FAIL  S3.6 D0 probe: this host has no usable TensorRT backend", summary)
        self.assertIn("The declared fallback is the LibTorch backend", summary)
        self.assertIn("SKIP  S3.6 core build with the TensorRT backend: the probe failed", summary)
        self.assertIn("FAIL  S3.6 control: no Sprint 3.5 verdict ACCEPTED on this DGX Spark", summary)
        self.assertIn("SKIP  S3.6 integration driver, records and second reader", summary)
        self.assertTrue(summary.rstrip().endswith("Sprint 3.6: NOT ACCEPTED on this host"))
        self.assertNotIn("Sprint 3.6: ACCEPTED", summary)

    def test_the_gate_runs_what_the_plan_names(self):
        text = SCRIPT.read_text(encoding="ascii")
        for needed in ("scripts/native_core.py probe", "scripts/native_core.py build --tensorrt on",
                       "tests/unit/test_sense_core.py -v", "tests/unit/test_core_link.py -v",
                       "tests/unit/test_core_imports.py -v", "tests/integration/sense_core.py --config "
                       "configs/integration/sense_core.json", "runtime_parity.json", "runtime_latency.json",
                       "cusense_reference.json", "isac-core.json", "scripts/check_pins.py --worktrees"):
            self.assertIn(needed, text)
        config = json.loads((ROOT / "configs/integration/sense_core.json").read_text(encoding="ascii"))
        self.assertEqual((config["schema"], config["selector"], config["expect"]["published_live_inference_us"]),
                         ("integration-config/sense_core/v1", "sense_core", 241))
        # A rehearsal cannot close the gate: the script requires a closed hardware gate of a run that was none.
        self.assertIn('grade["hardware_gate"] == "closed" and grade["rehearsal"] is False', text)


if __name__ == "__main__":
    unittest.main()
