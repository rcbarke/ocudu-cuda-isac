"""Check the Sprint 4.1 gate runner where it decides before anything is built. No compiler, NumPy, GPU or MPS is needed."""

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/accept_s4_1.sh"


class AcceptS41Tests(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name) / "checkout"
        (self.root / "scripts").mkdir(parents=True)
        shutil.copy(SCRIPT, self.root / "scripts/accept_s4_1.sh")

    def run_gate(self, *arguments):
        return subprocess.run(["bash", str(self.root / "scripts/accept_s4_1.sh"), *arguments], capture_output=True,
                              text=True, timeout=120)

    def test_usage(self):
        done = self.run_gate("--help")
        self.assertEqual(done.returncode, 0)
        for word in ("--run-dir", "--s36-run", "--s35-run", "--core-build", "make accept-s3.6", "make native-build-o3",
                     "nvidia-cuda-mps-control"):
            self.assertIn(word, done.stdout)
        for arguments in (("--unknown",), ("--run-dir",), ("--s36-run",), ("--s35-run",), ("--core-build",)):
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

    def test_a_host_without_a_receiver_a_core_and_a_control_is_not_accepted(self):
        # An interpreter that fails every command: no test runs, no core verifies, no control is found.
        (self.root / "isac/bin").mkdir(parents=True)
        (self.root / "isac/bin/python").write_text("#!/bin/sh\nexit 1\n")
        (self.root / "isac/bin/python").chmod(0o755)
        done = self.run_gate("--run-dir", str(self.root / "run"))
        self.assertEqual(done.returncode, 1)
        summary = (self.root / "run/summary.txt").read_text()
        self.assertIn("FAIL  S4.1 unit gate, run directly: python tests/unit/test_mps_pacing.py", summary)
        self.assertIn("FAIL  S4.1 adapter: build/ocudu-spark-o3 holds no patched receiver. Build it once: make "
                      "native-build-o3", summary)
        self.assertIn("FAIL  S4.1 sensing core:", summary)
        self.assertIn("FAIL  S4.1 control: no Sprint 3.6 verdict ACCEPTED on this DGX Spark", summary)
        self.assertIn("SKIP  S4.1 integration driver, record and second reader", summary)
        self.assertTrue(summary.rstrip().endswith("Sprint 4.1: NOT ACCEPTED on this host"))
        self.assertNotIn("Sprint 4.1: ACCEPTED", summary)
        self.assertFalse((self.root / "run/mps_pacing").exists())

    def test_the_gate_runs_what_the_plan_names(self):
        text = SCRIPT.read_text(encoding="ascii")
        for needed in ("tests/unit/test_mps_pacing.py -v", "tests/unit/test_mps_pacing_driver.py -v",
                       "scripts/native_adapter.py build --o3", "scripts/native_adapter.py verify --o3",
                       "pilot_ls --pacing", "scripts/native_core.py verify --build-dir",
                       "scripts/run_m1.py mps status", "scripts/run_m1.py mps stop",
                       "tests/integration/mps_pacing.py --config configs/integration/mps_pacing.json",
                       "mps_pacing.json", "pacing/releases.jsonl", "scripts/check_pins.py --worktrees"):
            self.assertIn(needed, text)
        # The gate builds no receiver and traces no episode.
        for refused in ("native_receiver.py build", "make_episodes", "native-build-o3 >"):
            self.assertNotIn(refused, text)
        config = json.loads((ROOT / "configs/integration/mps_pacing.json").read_text(encoding="ascii"))
        self.assertEqual((config["schema"], config["selector"], config["episodes"]["group"]),
                         ("integration-config/mps_pacing/v1", "mps_pacing", "validation"))
        self.assertEqual(config["contention"]["order"], ["mps_default", "mps_priority", "mps_thread_pct",
                                                         "release_after_radio"])
        # A rehearsal cannot close the gate: the script requires a closed hardware gate of a run that was none.
        self.assertIn('grade["hardware_gate"] == "closed" and grade["rehearsal"] is False', text)
        self.assertIn('record["graded"] is not False', text)
        # The second reader counts lines that the native tool writes without a space after the colon, and
        # lines that a Python writer puts one in.
        for pattern in ("""grep -cE '"schema": ?"pace-release/v1"'""", """grep -cE '"outcome": ?"decoded"'""",
                        """grep -cE '"outcome": ?"skipped"'"""):
            self.assertIn(pattern, text)
        line = self.root / "line.jsonl"
        line.write_text('{"outcome":"on_time","schema":"pace-release/v1"}\n{"outcome": "skipped", "schema": '
                        '"pace-release/v1"}\n')
        counted = subprocess.run(["grep", "-cE", '"schema": ?"pace-release/v1"', str(line)], capture_output=True, text=True)
        self.assertEqual(counted.stdout.strip(), "2")


if __name__ == "__main__":
    unittest.main()
