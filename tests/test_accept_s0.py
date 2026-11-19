"""Check the Sprint 0 gate runner against tool doubles. No environment, source or GPU is needed."""

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/accept_s0.sh"
COUNTS = {"on_time": 1, "late": 1, "dropped": 1, "missing": 1}
# The interpreter double answers each gate from FAKE_* variables and logs the call.
PYTHON_DOUBLE = """#!/bin/sh
[ "$1" = "-c" ] && exec {real} "$@"
echo "$*" >> "$FAKE_LOG"
case "$*" in
    "-m pip freeze --all") echo "example==1.0" ;;
    *"--profile core"*) exit "${{FAKE_CORE:-0}}" ;;
    *"--backend cpu"*) exit "${{FAKE_CPU:-0}}" ;;
    *"--backend cuda"*) exit "${{FAKE_CUDA:-0}}" ;;
    *"check_pins.py --worktrees"*) echo "ext/source 0123 (Git worktree)"; exit "${{FAKE_PINS:-0}}" ;;
    *"-m unittest discover"*) printf '%s\\n' "$FAKE_UNIT" >&2; exit "${{FAKE_UNIT_STATUS:-0}}" ;;
    *) echo "interpreter double: unexpected $*" >&2; exit 64 ;;
esac
"""
# The replay double writes three files. FAKE_SMOKE_DRIFT makes every run differ.
SMOKE_DOUBLE = """#!/bin/sh
[ "$1" = "--output-dir" ] || exit 64
mkdir -p "$2"
printf '%s\\n' "$FAKE_SUMMARY" > "$2/timing_summary.json"
echo ingress > "$2/receiver_ingress.jsonl"
echo "oracle ${FAKE_SMOKE_DRIFT:+$$}" > "$2/offline_oracle.jsonl"
"""


@unittest.skipUnless(shutil.which("bash"), "the gate runner is a Bash script")
class SprintZeroGateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / "host checkout"
        (self.root / "scripts").mkdir(parents=True)
        (self.root / "isac/bin").mkdir(parents=True)
        shutil.copy(SCRIPT, self.root / "scripts/accept_s0.sh")
        self.write(self.root / "isac/bin/python", PYTHON_DOUBLE.format(real=shlex.quote(sys.executable)))
        self.write(self.root / "scripts/run_smoke.sh", SMOKE_DOUBLE)
        self.calls = self.root.parent / "calls.log"
        self.run_dir = self.root.parent / "evidence"

    @staticmethod
    def write(path, text):
        path.write_text(text)
        path.chmod(0o755)

    def gate(self, *arguments, run_dir=None, **settings):
        summary = {"scheduled_updates": 4, "counts": COUNTS}
        env = {key: value for key, value in os.environ.items() if not key.startswith("FAKE_")}
        env.update(FAKE_LOG=str(self.calls), FAKE_UNIT="Ran 59 tests in 1.000s\n\nOK",
                   FAKE_SUMMARY=json.dumps(summary))
        env.update({f"FAKE_{key.upper()}": str(value) for key, value in settings.items()})
        command = ["bash", str(self.root / "scripts/accept_s0.sh"),
                   "--run-dir", str(run_dir or self.run_dir), *arguments]
        return subprocess.run(command, env=env, capture_output=True, text=True, timeout=120, check=False)

    def test_all_gates_pass_on_a_simulation_host_and_leave_evidence(self):
        result = self.gate("cpu", "cuda")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("FAIL", result.stdout)
        self.assertIn("Sprint 0: ACCEPTED on this host, including simulation on cpu cuda", result.stdout)
        self.assertIn("unit tests (59 ran, 0 skipped)", result.stdout)
        self.assertEqual((self.run_dir / "summary.txt").read_text(), result.stdout)
        for name in ("python-packages.txt", "sources.txt", "smoke-sha256.txt", "unit-tests.log",
                     "smoke-a/timing_summary.json", "smoke-b/offline_oracle.jsonl"):
            self.assertTrue((self.run_dir / name).is_file(), name)
        self.assertEqual(len((self.run_dir / "smoke-sha256.txt").read_text().splitlines()), 3)
        calls = self.calls.read_text()
        for expected in ("--profile core --report", "check_pins.py --worktrees",
                         "--profile simulation --backend cpu", "--profile simulation --backend cuda",
                         "-m unittest discover -s tests -v"):
            self.assertIn(expected, calls)

    def test_development_host_runs_the_core_gates_only(self):
        result = self.gate(unit="Ran 59 tests in 1.000s\n\nOK (skipped=4)")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("ACCEPTED as a development host", result.stdout)
        self.assertIn("still needs a provisioned scene host", result.stdout)
        self.assertIn("unit tests (59 ran, 4 skipped)", result.stdout)
        self.assertNotIn("--profile simulation", self.calls.read_text())

    def test_any_failed_gate_rejects_the_host_and_the_rest_still_run(self):
        cases = (({"cuda": 1}, "FAIL  S0.2 simulation on cuda"), ({"core": 1}, "FAIL  S0.1 environment report"),
                 ({"pins": 1}, "FAIL  S0.2 source identities"),
                 ({"unit_status": 1, "unit": "Ran 59 tests in 1.000s\n\nFAILED (failures=1)"}, "FAIL  unit tests"),
                 ({"unit": "Ran 0 tests in 0.000s\n\nOK"}, "FAIL  unit tests (none ran)"),
                 ({"unit": "no tests were collected"}, "FAIL  unit tests"),
                 ({"unit": "Ran 59 tests in 1.000s\n\nOK (skipped=4)"}, "FAIL  unit tests (4 skipped on a simulation host"),
                 ({"smoke_drift": 1}, "FAIL  S0.3 three output files byte-identical on repeat"),
                 ({"summary": json.dumps({"scheduled_updates": 4, "counts": {**COUNTS, "missing": 0}})},
                  "FAIL  S0.3 four scheduled"),
                 ({"summary": json.dumps({"scheduled_updates": 3, "counts": COUNTS})}, "FAIL  S0.3 four scheduled"))
        for index, (settings, message) in enumerate(cases):
            self.calls.write_text("")
            result = self.gate("cpu", "cuda", run_dir=self.root.parent / f"evidence {index}", **settings)
            with self.subTest(settings=settings):
                self.assertEqual(result.returncode, 1)
                self.assertIn(message, result.stdout)
                self.assertIn("Sprint 0: NOT ACCEPTED on this host", result.stdout)
                self.assertNotIn("Sprint 0: ACCEPTED", result.stdout)
                self.assertIn("-m unittest discover", self.calls.read_text())

    def test_arguments_and_existing_evidence_are_refused_before_any_gate_runs(self):
        self.run_dir.mkdir()
        keep = self.run_dir / "summary.txt"
        keep.write_text("earlier run\n")
        refusals = ((("cpu",), {}, "already exists"), (("gpu",), {"run_dir": self.root.parent / "new"}, "Usage"),
                    (("--run-dir",), {"run_dir": self.root.parent / "new"}, "Usage"))
        for arguments, options, message in refusals:
            result = self.gate(*arguments, **options)
            with self.subTest(arguments=arguments):
                self.assertEqual(result.returncode, 2)
                self.assertIn(message, result.stderr)
        self.assertEqual(keep.read_text(), "earlier run\n")
        self.assertFalse((self.root.parent / "new").exists())
        self.assertFalse(self.calls.exists())
        (self.root / "isac/bin/python").unlink()
        result = self.gate("cpu", run_dir=self.root.parent / "new")
        self.assertEqual(result.returncode, 2)
        self.assertIn("make env", result.stderr)

    @unittest.skipUnless(shutil.which("git"), "the committed-tree check reads Git metadata")
    def test_evidence_must_come_from_a_committed_tree(self):
        identity = {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
                    "GIT_AUTHOR_NAME": "fixture", "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
                    "GIT_COMMITTER_NAME": "fixture", "GIT_COMMITTER_EMAIL": "fixture@example.invalid"}

        def git(*arguments):
            subprocess.run(["git", *arguments], cwd=self.root, env={**os.environ, **identity},
                           capture_output=True, text=True, check=True)

        git("init", "-q")
        git("add", "-A")
        git("commit", "-q", "-m", "fixture checkout")
        result = self.gate(run_dir=self.root.parent / "clean")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS  committed tree", result.stdout)
        self.assertRegex(result.stdout, r"commit [0-9a-f]{40}")
        (self.root / "scripts/run_smoke.sh").write_text(SMOKE_DOUBLE + "# local edit\n")
        result = self.gate(run_dir=self.root.parent / "dirty")
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  committed tree", result.stdout)
        self.assertIn("run_smoke.sh", (self.root.parent / "dirty/git-status.txt").read_text())


if __name__ == "__main__":
    unittest.main()
