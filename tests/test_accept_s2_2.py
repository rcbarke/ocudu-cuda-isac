"""Check the Sprint 2.2 gate runner against tool doubles. No compiler, source, build, Sionna or GPU is needed."""

import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/accept_s2_2.sh"
TEST = "tests/integrationtests/phy/upper/channel_processors/pusch_e2e_pipeline_test"
CHANNEL = b"target-present channel"
CHANNEL_SHA256 = hashlib.sha256(CHANNEL).hexdigest()
GRADE = """PASS  S2.2 diagnostic exporter: scripts/export_rt_episode.py wrote rt-cfr-pair/v1
PASS  S2.2 grant-fixture schedule: 24 opportunities on the uplink slots [7, 8, 9] of DDDDDDSUUU
NOTE  S2.2 grant-fixture x.npy of demo-51prb-4rx against the Sprint 2.1 accepted run (not graded): identical
PASS  S2.2 reference-pair scattered path: target present minus absent equals the bistatic radar return
PASS  S2.2 episode-set groups: whole trajectories assigned before tracing
episode_pair: 4 of 4 cases passed, 4 of 4 checks"""
GRADE_JSON = {"judgment": "pass", "cases": 4, "cases_passed": 4, "checks": 4, "checks_passed": 4, "checks_failed": 0}
RECEIPT = {"executables": {"grant_fixture": {"sha256": "0123456789abcdef" * 4}}, "tests_run": 2, "compiler_warnings": 0,
           "pin": {"cuda": True, "archives": 50}}
RECOVERED = {"sha256": {"b_h.npy": CHANNEL_SHA256},
             "coefficient": {"symbol": 2, "subcarrier": 0, "offset_hz": -9180000, "value": [-3.33e-4, 7.56e-5],
                             "analytic_relative_error": 1.7e-5},
             "phase_advance": [{"measured_rad": -0.08741, "interval_s": 3.2109375e-4},
                               {"measured_rad": -0.04820, "interval_s": 1.7890625e-4}]}
S21_SUMMARY = """Sprint 2.1 acceptance on spark-ecf8, 2026-11-20T22:32:32Z
PASS  committed tree
PASS  S2.1 integration grade: 3 of 3 cases, 38 of 38 checks passed (grant_fixture/grade.json)
Sprint 2.1: ACCEPTED on this DGX Spark. One legal uplink grant and its transmit reference are archived.
"""
UNIT = "Ran 210 tests in 4.000s\n\nOK"
# The interpreter double answers each gate from FAKE_* variables and logs the call. The driver
# double writes the files the real driver leaves: its grade and the stored pair.
PYTHON_DOUBLE = """#!/bin/sh
[ "$1" = "-c" ] && exec {real} "$@"
echo "$*" >> "$FAKE_LOG"
case "$*" in
    "-m pip freeze --all") echo "example==1.0" ;;
    *"-m unittest discover"*) printf '%s\\n' "$FAKE_UNIT" >&2; exit "${{FAKE_UNIT_STATUS:-0}}" ;;
    "tests/unit/test_episode_pair.py -v") printf '%s\\n' "$FAKE_CARD" >&2; exit "${{FAKE_CARD_STATUS:-0}}" ;;
    *"--profile simulation --backend cpu --with-native"*)
        [ "${{FAKE_ENVIRONMENT:-0}}" = 0 ] || echo "ValueError: installed Sionna RT file differs from the pinned tree" >&2
        exit "${{FAKE_ENVIRONMENT:-0}}" ;;
    "scripts/native_adapter.py build --ocudu-build-dir "*)
        echo "Adapter build: build/ocudu-adapter"
        [ "${{FAKE_BUILD:-0}}" = 0 ] || echo "native_adapter: the adapter build step exited with status 2" >&2
        exit "${{FAKE_BUILD:-0}}" ;;
    "scripts/native_adapter.py verify")
        [ "${{FAKE_VERIFY:-0}}" = 0 ] || {{ echo "native_adapter: the adapter sources changed after the build" >&2; exit 1; }}
        printf '%s\\n' "$FAKE_RECEIPT" ;;
    "tests/integration/episode_pair.py --config configs/integration/episode_pair.json --grant-store "*)
        out=$7
        echo "grant store: $5" >> "$FAKE_LOG"
        echo "Run artifacts: $out"
        [ -z "${{FAKE_NO_EVIDENCE:-}}" ] || exit "${{FAKE_DRIVER:-0}}"
        mkdir -p "$out/reference_pair/store"
        printf '%s' "${{FAKE_CHANNEL:-target-present channel}}" > "$out/reference_pair/store/b_h.npy"
        printf '%s\\n' "$FAKE_GRADE" > "$out/grade.txt"
        printf '%s\\n' "$FAKE_GRADE_JSON" > "$out/grade.json"
        exit "${{FAKE_DRIVER:-0}}" ;;
    "-m ocudu_cuda_isac.episodes.episode show "*)
        [ "${{FAKE_SHOW:-0}}" = 0 ] || {{ echo "episode: b_h.npy differs from the size or hash in the manifest" >&2; exit 2; }}
        printf '%s\\n' "$FAKE_RECOVERED" ;;
    *"check_pins.py --worktrees") echo "ext/ocudu 27e84cb3 (Git worktree)"; exit "${{FAKE_PINS:-0}}" ;;
    *) echo "interpreter double: unexpected $*" >&2; exit 64 ;;
esac
"""


@unittest.skipUnless(shutil.which("bash") and shutil.which("sha256sum"), "the gate runner is a Bash script")
class SprintTwoEpisodeGateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / "host checkout"
        self.bin = self.root / "isac/bin"
        (self.root / "scripts").mkdir(parents=True)
        self.bin.mkdir(parents=True)
        shutil.copy(SCRIPT, self.root / "scripts/accept_s2_2.sh")
        python = self.bin / "python"
        python.write_text(PYTHON_DOUBLE.format(real=shlex.quote(sys.executable)))
        python.chmod(0o755)
        # The build Sprint 1.1 published and the run in which Sprint 2.1 archived the grant record.
        self.build = self.root / "build/ocudu-spark"
        (self.build / TEST).parent.mkdir(parents=True)
        (self.build / TEST).write_text("#!/bin/sh\n")
        (self.bin / "pusch_e2e_pipeline_test").symlink_to(self.build / TEST)
        (self.build / "isac-native.json").write_text(json.dumps({"target": "all"}))
        self.s21 = self.accepted_run("s21-spark-ecf8-20261120T223232Z")
        self.calls = self.root.parent / "calls.log"
        self.run_dir = self.root.parent / "evidence"

    def accepted_run(self, name, summary=S21_SUMMARY, fixture=True):
        run = self.root / "results/runs" / name
        (run / "grant_fixture" if fixture else run).mkdir(parents=True)
        (run / "summary.txt").write_text(summary)
        return run

    def gate(self, *arguments, run_dir=None, **settings):
        env = {key: value for key, value in os.environ.items() if not key.startswith("FAKE_")}
        env.update(FAKE_LOG=str(self.calls), FAKE_UNIT=UNIT, FAKE_CARD="Ran 23 tests in 3.200s\n\nOK",
                   FAKE_GRADE=GRADE, FAKE_GRADE_JSON=json.dumps(GRADE_JSON), FAKE_RECEIPT=json.dumps(RECEIPT),
                   FAKE_RECOVERED=json.dumps(RECOVERED))
        env.update({f"FAKE_{key.upper()}": value if isinstance(value, str) else json.dumps(value)
                    for key, value in settings.items()})
        command = [shutil.which("bash"), str(self.root / "scripts/accept_s2_2.sh"),
                   "--run-dir", str(run_dir or self.run_dir), *arguments]
        return subprocess.run(command, env=env, capture_output=True, text=True, timeout=120, check=False)

    def summary(self):
        return (self.run_dir / "summary.txt").read_text()

    def test_all_gates_pass_on_the_accepted_record(self):
        result = self.gate()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("FAIL", result.stdout)
        self.assertNotIn("SKIP", result.stdout)
        summary = self.summary()
        self.assertEqual(summary, result.stdout)
        for line in ("PASS  unit tests (210 ran, 0 skipped)",
                     "PASS  S2.2 unit gate, run directly: python tests/unit/test_episode_pair.py (23 ran, 0 skipped)",
                     "PASS  S2.2 grant record: the accepted Sprint 2.1 run s21-spark-ecf8-20261120T223232Z holds it",
                     "PASS  S2.2 fixture adapter: built against the Sprint 1.1 build and verified, grant_fixture "
                     "0123456789ab, with ENABLE_CUDA, 2 adapter tests, 0 compiler warnings",
                     "PASS  S2.2 reference-pair scattered path: target present minus absent",
                     "NOTE  S2.2 grant-fixture x.npy of demo-51prb-4rx",
                     "PASS  S2.2 integration grade: 4 of 4 cases, 4 of 4 checks passed",
                     "H at symbol 2, subcarrier 0 (-9.18 MHz) = -3.330000e-04+7.560000e-05j, 1.7e-05 from the analytic "
                     "value; phase steps of the return -0.08741 rad in 321.09 us and -0.04820 rad in 178.91 us; b_h.npy "
                     f"SHA-256 {CHANNEL_SHA256[:16]} by the reader and by sha256sum",
                     "PASS  S2.2 sources at the locked revisions after the runs",
                     "Sprint 2.2: ACCEPTED on this DGX Spark. The uplink grant fixture, the matched pair on it and the "
                     "episode set are archived (S2.1 run s21-spark-ecf8-20261120T223232Z). Next: S2.3 grid_apply."):
            self.assertIn(line, summary)
        calls = self.calls.read_text()
        self.assertIn(f"grant store: {self.s21 / 'grant_fixture'}", calls)
        self.assertIn(f"scripts/native_adapter.py build --ocudu-build-dir {self.build}", calls)
        self.assertEqual(summary.count("PASS  "), 12)

    def test_the_newest_accepted_sprint_2_1_run_is_used_and_a_failed_one_is_not(self):
        self.accepted_run("s21-spark-ecf8-20261121T090000Z", S21_SUMMARY.replace("PASS  committed", "FAIL  committed"))
        self.accepted_run("s21-spark-ecf8-20261121T100000Z", "Sprint 2.1: NOT ACCEPTED on this host\n")
        newest = self.accepted_run("s21-spark-ecf8-20261120T235900Z")
        self.assertEqual(self.gate().returncode, 0)
        self.assertIn(f"grant store: {newest / 'grant_fixture'}", self.calls.read_text())
        self.assertIn("S2.1 run s21-spark-ecf8-20261120T235900Z", self.summary())

    def test_a_named_sprint_2_1_run_and_a_development_host(self):
        other = self.accepted_run("s21-devhost-20261119T000000Z", S21_SUMMARY.replace("DGX Spark", "host"))
        result = self.gate("--s21-run", str(other))
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn(f"grant store: {other / 'grant_fixture'}", self.calls.read_text())
        self.assertIn("Sprint 2.2: ACCEPTED on this host.", result.stdout)

    def test_without_an_accepted_grant_record_nothing_is_traced(self):
        shutil.rmtree(self.s21)
        self.accepted_run("s21-spark-ecf8-20261120T223232Z", fixture=False)  # a verdict without the record
        result = self.gate()
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S2.2 grant record: no Sprint 2.1 ACCEPTED verdict with a grant_fixture directory", result.stdout)
        self.assertIn("Run 'make accept-s2.1' on this host first", result.stdout)
        self.assertIn("SKIP  S2.2 reference check, adapter build, integration driver and pair recovery", result.stdout)
        self.assertIn("Sprint 2.2: NOT ACCEPTED on this host", result.stdout)
        self.assertNotIn("tests/integration/episode_pair.py", self.calls.read_text())

    def test_a_missing_sprint_1_1_build_is_named(self):
        (self.bin / "pusch_e2e_pipeline_test").unlink()
        result = self.gate()
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S2.2 needs the Sprint 1.1 build", result.stdout)
        self.assertNotIn("native_adapter.py", self.calls.read_text())

    def test_failures_of_single_gates(self):
        cases = (
            (dict(unit="Ran 210 tests in 4.0s\n\nFAILED (failures=1)", unit_status="1"), "FAIL  unit tests (210 ran"),
            (dict(card="Ran 0 tests in 0.0s\n\nOK"), "FAIL  S2.2 unit gate, run directly: python "
                                                    "tests/unit/test_episode_pair.py (none ran)"),
            (dict(environment="1"), "FAIL  S2.2 references: the pinned Sionna and Sionna RT"),
            (dict(build="1"), "FAIL  S2.2 fixture adapter: build or verification failed"),
            (dict(verify="1"), "FAIL  S2.2 fixture adapter: build or verification failed"),
            (dict(grade=GRADE.replace("PASS  S2.2 reference-pair scattered", "FAIL  S2.2 reference-pair scattered"),
                  grade_json=dict(GRADE_JSON, judgment="fail", cases_passed=3, checks_passed=3, checks_failed=1),
                  driver="1"), "FAIL  S2.2 integration grade: 3 of 4 cases, 3 of 4 checks passed, driver exit status 1"),
            (dict(no_evidence="1", driver="2"), "FAIL  S2.2 integration driver produced no graded checks"),
            (dict(grade_json=dict(GRADE_JSON, cases=0, cases_passed=0)), "FAIL  S2.2 integration grade: 0 of 0 cases"),
            (dict(show="1"), "FAIL  S2.2 a second reader recovers the pair (see recovered.log)"),
            (dict(channel="another channel"), "FAIL  S2.2 a second reader recovers the pair: b_h.npy is"),
            (dict(pins="1"), "FAIL  S2.2 sources at the locked revisions after the runs"),
        )
        for number, (settings, line) in enumerate(cases):
            run_dir = self.root.parent / f"evidence-{number}"
            result = self.gate(run_dir=run_dir, **settings)
            with self.subTest(line=line):
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn(line, result.stdout)
                self.assertIn("Sprint 2.2: NOT ACCEPTED on this host", result.stdout)
                self.assertNotIn("Sprint 2.2: ACCEPTED", result.stdout)
        skipped = self.gate(run_dir=self.root.parent / "evidence-build", build="1")
        self.assertIn("SKIP  S2.2 integration driver and pair recovery: no adapter", skipped.stdout)

    def test_a_failed_check_of_the_driver_fails_the_gate_even_with_exit_status_zero(self):
        result = self.gate(grade=GRADE.replace("PASS  S2.2 episode-set groups", "FAIL  S2.2 episode-set groups"))
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S2.2 episode-set groups", result.stdout)

    def test_uncommitted_changes_fail_and_evidence_is_never_overwritten(self):
        if not shutil.which("git"):
            self.skipTest("git is not installed")
        commands = (["init", "-q"], ["add", "scripts"], ["-c", "user.name=t", "-c", "user.email=t@example.com",
                                                         "commit", "-q", "-m", "gate"])
        for command in commands:
            subprocess.run(["git", "-C", str(self.root), *command], check=True, capture_output=True)
        result = self.gate()
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  committed tree (uncommitted changes are listed in git-status.txt)", result.stdout)
        self.assertIn("isac/", (self.run_dir / "git-status.txt").read_text())
        again = self.gate()
        self.assertEqual(again.returncode, 2)
        self.assertIn("already exists. Evidence is never overwritten.", again.stderr)

    def test_usage(self):
        self.assertEqual(self.gate("--help").returncode, 0)
        for arguments in (("--unknown",), ("--s21-run",)):
            result = self.gate(*arguments, run_dir=self.root.parent / "evidence-usage")
            self.assertEqual(result.returncode, 2)
            self.assertIn("Usage: bash scripts/accept_s2_2.sh", result.stderr)
        (self.bin / "python").unlink()
        missing = self.gate(run_dir=self.root.parent / "evidence-python")
        self.assertEqual(missing.returncode, 2)
        self.assertIn("Run 'make env' first", missing.stderr)


if __name__ == "__main__":
    unittest.main()
