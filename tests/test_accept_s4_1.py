"""Check the Sprint 4.1 gate runner where it decides before anything is built. No compiler, NumPy, GPU or MPS is needed."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
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
        # The gate requires a tool that reads a lead-in, a record in which no release is the first use of its
        # receiver, and a second reader that finds the lead-in apart from the releases and the radio records.
        self.assertIn('"$said" == "pilot_ls pacing: pilot-ls-plan/v4"*"lead-in ahead of the schedule"*', text)
        self.assertIn('if cold["first_use_in_schedule"] != 0 or cold["skipped_first_releases"] != 0 or cold["runs"] < 1:',
                      text)
        self.assertIn("""grep -cE '"schema": ?"pace-lead-in/v1"' "$arm/native/pacing/lead_in.jsonl" """.strip(), text)
        self.assertIn("""grep -cE '"first_use": ?true' "$arm/native/radio/radio.jsonl" """.strip(), text)
        self.assertIn("(( releases > 0 && radio + skipped == releases && (ahead == 0 || cold == 0) ))", text)
        config_lead = config["pacing"]["lead_in"]
        self.assertEqual((config_lead, config["pacing"]["cold_start_without_lead_in"]["run"]),
                         ({"releases": 6, "lead_us": 10000}, "s41-spark-ecf8-20261123T140213Z"))
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

    def test_the_record_is_read_with_its_scores_and_a_record_that_was_not_written_is_named(self):
        # What the gate runs on the record of the driver, taken from the script and run as the gate runs it.
        text = SCRIPT.read_text(encoding="ascii")
        opening = "record=$(\"$python\" -c '"
        first = text.index(opening) + len(opening)
        code = text[first:text.index("' \"$out\"", first)]
        runs = [{"warmup": False, "control": "mps_default", "fault": None,
                 "sharing": {"requested": "mps", "label": "mps", "verified": True}}]
        tails = {"p50_us": 187.2, "p99_us": 219.5}
        record = {"graded": False, "rehearsal": False, "load": {"scripted": True},
                  "arms": {name: runs for name in ("dsp_only", "capture_only", "ai_only", "concurrent")},
                  "contention": {"service": {name: tails for name in ("dsp_only", "capture_only", "concurrent")},
                                 "shifts": {"mps_default": {"quantile": "p99", "shift_us": 2.7, "allowed_us": 25.7}},
                                 "rule": {"status": "proposed"}, "control_in_effect": "mps_default"},
                  "cold_start": {"lead_in": {"releases": 6, "lead_us": 10000}, "first_use_in_schedule": 0,
                                 "skipped_first_releases": 0, "runs": 9, "by_decode_us": [{"p50_us": 1320.7}],
                                 "first_release_us": {"p50_us": 189.6}},
                  # The scores of the run s41-spark-ecf8-20261123T161419Z: one flip in 960 windows, at the threshold.
                  "decisions": {"joined": 960, "agree": 959, "same_input": 772, "same_input_same_logit": 772,
                                "other_input": 188, "largest_logit_difference_other_input": 3.489e-3, "problems": [],
                                "flips": [{"at_threshold": True}]}}

        def read(made):
            where = Path(tempfile.mkdtemp(dir=self.root))
            if made is not None:
                (where / "mps_pacing.json").write_text(json.dumps(made), encoding="ascii")
            return subprocess.run([sys.executable, "-c", code, str(where)], capture_output=True, text=True, timeout=60)

        done = read(record)
        self.assertEqual((done.returncode, done.stderr), (0, ""))
        self.assertIn("960 windows beside the receiver against the sensor alone: 772 on the same input bytes, 772 of them "
                      "with the same logit bit for bit; 188 on two decodes that differ, logits within 3.49e-03; 959 decide "
                      "alike and 1 flip(s) lie at the threshold", done.stdout)
        self.assertIn("the first decode of a receiver 1320.7 us in the median of 9 runs", done.stdout)
        # A window that breaks the rule of the scores, and a flip that is not at the threshold, fail the record.
        broken = json.loads(json.dumps(record))
        broken["decisions"]["problems"] = ["r2: window 387b28319ca93c0d/2: the same input bytes gave two logits"]
        done = read(broken)
        self.assertEqual((done.returncode, done.stdout), (1, ""))
        self.assertIn("a window of the concurrent arm breaks the rule of the scores: r2: window 387b28319ca93c0d/2",
                      done.stderr)
        away = json.loads(json.dumps(record))
        away["decisions"]["flips"] = [{"at_threshold": False}]
        self.assertEqual(read(away).returncode, 1)
        # A driver that failed a case before its last one wrote no record: the gate says so, in one sentence.
        done = read(None)
        self.assertEqual((done.returncode, done.stdout), (1, ""))
        self.assertIn("mps_pacing.json was not written. The driver writes it in its last case, records", done.stderr)
        self.assertNotIn("Traceback", done.stderr)
        self.assertIn("or holds a window that breaks the rule of the scores", text)


if __name__ == "__main__":
    unittest.main()
