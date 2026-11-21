"""Check the Sprint 2.3 gate runner against tool doubles. No compiler, source, build, NumPy or GPU is needed."""

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

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/accept_s2_3.sh"
TEST = "tests/integrationtests/phy/upper/channel_processors/pusch_e2e_pipeline_test"
GRID = b"stored grid of opportunity 0"
GRID_SHA256 = hashlib.sha256(GRID).hexdigest()
GRADE = """PASS  S2.3 unit-vectors Sprint 0.3 vector: through the pinned resource grid, g = 1: 1+3j, 2-4j, 6.5+0j, 14+0j
PASS  S2.3 unity slot 7 grid: demo-51prb-4rx: 4 of 4 receive planes return x.npy bit for bit
PASS  S2.3 reference-pair stored grids: read back from the pinned grid: 1645056 of 1645056 elements
NOTE  S2.3 reference-pair fill time of one grid, not graded: mean 1158 us
PASS  S2.3 allocation changed grant: the grant changes on one channel that holds its own coordinates
PASS  S2.3 refusals gain rules: refused by the plan check and by the native tool with status 2
grid_apply: 5 of 5 cases passed, 5 of 5 checks"""
GRADE_JSON = {"judgment": "pass", "cases": 5, "cases_passed": 5, "checks": 5, "checks_passed": 5, "checks_failed": 0}
RECEIPT = {"executables": {"grant_fixture": {"sha256": "0123456789abcdef" * 4},
                           "grid_apply": {"sha256": "fedcba9876543210" * 4}},
           "tests_run": 19, "compiler_warnings": 0, "pin": {"cuda": True, "archives": 50}}
CTEST = "\n".join([f" {n + 3:2d}/19 Test  #{n + 3}: replay_core_case_{n} ..............   Passed    0.00 sec"
                   for n in range(14)]
                  + ["19/19 Test #19: grid_apply_pinned_grid ....................   Passed    0.02 sec",
                     "100% tests passed, 0 tests failed out of 19"])
RECOVERED = {"readback_sha256": GRID_SHA256, "grid_file": "0000.c64", "verified_grids": 48,
             "opportunity_id": "rz-22963ee9d985.0000", "slot": 7, "grant_id": "demo-51prb-4rx",
             "gain": {"value": 2942.4243526773034, "source": "entry"},
             "element": {"rx_port": 0, "symbol": 2, "subcarrier": 0, "value": [0.75, -1.1796875]}}
S22_SUMMARY = """Sprint 2.2 acceptance on spark-ecf8, 2026-11-21T00:44:33Z
PASS  committed tree
PASS  S2.2 integration grade: 4 of 4 cases, 26 of 26 checks passed (episode_pair/grade.json)
Sprint 2.2: ACCEPTED on this DGX Spark. The uplink grant fixture, the matched pair on it and the episode set are archived.
"""
UNIT = "Ran 265 tests in 50.000s\n\nOK"
CARD = "Ran 29 tests in 6.500s\n\nOK"
# The interpreter double answers each gate from FAKE_* variables and logs the call. The builder
# double leaves the CTest log of the adapter build, the driver double its grade and one stored grid.
PYTHON_DOUBLE = """#!/bin/sh
[ "$1" = "-c" ] && exec {real} "$@"
echo "$*" >> "$FAKE_LOG"
case "$*" in
    "-m pip freeze --all") echo "example==1.0" ;;
    *"-m unittest discover"*) printf '%s\\n' "$FAKE_UNIT" >&2; exit "${{FAKE_UNIT_STATUS:-0}}" ;;
    "tests/unit/test_grid_apply.py -v") printf '%s\\n' "$FAKE_CARD" >&2; exit "${{FAKE_CARD_STATUS:-0}}" ;;
    *"--profile core --with-native"*)
        [ "${{FAKE_ENVIRONMENT:-0}}" = 0 ] || echo "ValueError: Native binaries differ from the locked full-build receipt" >&2
        exit "${{FAKE_ENVIRONMENT:-0}}" ;;
    "scripts/native_adapter.py build --ocudu-build-dir "*)
        echo "Adapter build: build/ocudu-adapter"
        mkdir -p "$6"
        [ -n "${{FAKE_NO_CTEST:-}}" ] || printf '%s\\n' "$FAKE_CTEST" > "$6/adapter-test.log"
        [ "${{FAKE_BUILD:-0}}" = 0 ] || echo "native_adapter: the adapter build step exited with status 2" >&2
        exit "${{FAKE_BUILD:-0}}" ;;
    "scripts/native_adapter.py verify")
        [ "${{FAKE_VERIFY:-0}}" = 0 ] || {{ echo "native_adapter: the adapter sources changed after the build" >&2; exit 1; }}
        printf '%s\\n' "$FAKE_RECEIPT" ;;
    "tests/integration/grid_apply.py --config configs/integration/grid_apply.json --s22-run "*)
        for out; do :; done
        echo "s22 run: $5" >> "$FAKE_LOG"
        echo "Run artifacts: $out"
        [ -z "${{FAKE_NO_EVIDENCE:-}}" ] || exit "${{FAKE_DRIVER:-0}}"
        mkdir -p "$out/reference_pair/native/ingress"
        printf '%s' "${{FAKE_GRID:-stored grid of opportunity 0}}" > "$out/reference_pair/native/ingress/0000.c64"
        printf '%s\\n' "$FAKE_GRADE" > "$out/grade.txt"
        printf '%s\\n' "$FAKE_GRADE_JSON" > "$out/grade.json"
        exit "${{FAKE_DRIVER:-0}}" ;;
    "-m ocudu_cuda_isac.episodes.grid_apply show "*)
        [ "${{FAKE_SHOW:-0}}" = 0 ] || {{ echo "grid_apply: the stored grid of opportunity 3 differs from its readback digest" >&2; exit 2; }}
        printf '%s\\n' "$FAKE_RECOVERED" ;;
    *"check_pins.py --worktrees") echo "ext/ocudu 27e84cb3 (Git worktree)"; exit "${{FAKE_PINS:-0}}" ;;
    *) echo "interpreter double: unexpected $*" >&2; exit 64 ;;
esac
"""


@unittest.skipUnless(shutil.which("bash") and shutil.which("sha256sum"), "the gate runner is a Bash script")
class SprintTwoGridGateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / "host checkout"
        self.bin = self.root / "isac/bin"
        (self.root / "scripts").mkdir(parents=True)
        self.bin.mkdir(parents=True)
        shutil.copy(SCRIPT, self.root / "scripts/accept_s2_3.sh")
        python = self.bin / "python"
        python.write_text(PYTHON_DOUBLE.format(real=shlex.quote(sys.executable)))
        python.chmod(0o755)
        # The build Sprint 1.1 published and the run in which Sprint 2.2 archived the stimulus.
        self.build = self.root / "build/ocudu-spark"
        (self.build / TEST).parent.mkdir(parents=True)
        (self.build / TEST).write_text("#!/bin/sh\n")
        (self.bin / "pusch_e2e_pipeline_test").symlink_to(self.build / TEST)
        (self.build / "isac-native.json").write_text(json.dumps({"target": "all"}))
        self.s22 = self.accepted_run("s22-spark-ecf8-20261121T004433Z")
        self.calls = self.root.parent / "calls.log"
        self.run_dir = self.root.parent / "evidence"

    def accepted_run(self, name, summary=S22_SUMMARY, pair=True):
        run = self.root / "results/runs" / name
        run.mkdir(parents=True)
        if pair:
            (run / "episode_pair/reference_pair/store").mkdir(parents=True)
            (run / "episode_pair/reference_pair/store/manifest.json").write_text("{}")
        (run / "summary.txt").write_text(summary)
        return run

    def gate(self, *arguments, run_dir=None, **settings):
        env = {key: value for key, value in os.environ.items() if not key.startswith("FAKE_")}
        env.update(FAKE_LOG=str(self.calls), FAKE_UNIT=UNIT, FAKE_CARD=CARD, FAKE_GRADE=GRADE,
                   FAKE_GRADE_JSON=json.dumps(GRADE_JSON), FAKE_RECEIPT=json.dumps(RECEIPT), FAKE_CTEST=CTEST,
                   FAKE_RECOVERED=json.dumps(RECOVERED))
        env.update({f"FAKE_{key.upper()}": value if isinstance(value, str) else json.dumps(value)
                    for key, value in settings.items()})
        command = [shutil.which("bash"), str(self.root / "scripts/accept_s2_3.sh"),
                   "--run-dir", str(run_dir or self.run_dir), *arguments]
        return subprocess.run(command, env=env, capture_output=True, text=True, timeout=120, check=False)

    def summary(self):
        return (self.run_dir / "summary.txt").read_text()

    def test_all_gates_pass_on_the_accepted_stimulus(self):
        result = self.gate()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("FAIL", result.stdout)
        self.assertNotIn("SKIP", result.stdout)
        summary = self.summary()
        self.assertEqual(summary, result.stdout)
        for line in ("PASS  unit tests (265 ran, 0 skipped)",
                     "PASS  S2.3 unit gate, run directly: python tests/unit/test_grid_apply.py (29 ran, 0 skipped)",
                     "PASS  S2.3 stimulus: the accepted Sprint 2.2 run s22-spark-ecf8-20261121T004433Z holds the pair",
                     "PASS  S2.3 references: the Sprint 1.1 build, from the project interpreter (environment.json)",
                     "PASS  S2.3 adapter: built against the Sprint 1.1 build and verified, grid_apply fedcba987654 and "
                     "grant_fixture 0123456789ab, with ENABLE_CUDA, 19 adapter tests, 0 compiler warnings",
                     "PASS  S2.3 native unit cases: 14 cases of the host-free core and the unit vectors through the "
                     "pinned resource grid passed in the adapter build",
                     "Applying the stimulus of s22-spark-ecf8-20261121T004433Z with configs/integration/grid_apply.json.",
                     "PASS  S2.3 reference-pair stored grids: read back from the pinned grid",
                     "NOTE  S2.3 reference-pair fill time of one grid, not graded",
                     "PASS  S2.3 integration grade: 5 of 5 cases, 5 of 5 checks passed",
                     "PASS  S2.3 a second reader recovers a stored grid: 48 stored grids verified; Y of opportunity "
                     "rz-22963ee9d985.0000 (slot 7, grant demo-51prb-4rx) at port 0, symbol 2, subcarrier 0 = "
                     f"+0.750000-1.179688j with g = 2942.4244; readback SHA-256 {GRID_SHA256[:16]} by the reader and by "
                     "sha256sum",
                     "PASS  S2.3 sources at the locked revisions after the runs",
                     "Sprint 2.3: ACCEPTED on this DGX Spark. The native channel application with the grid gain fills "
                     "the pinned resource grid from the archived stimulus (S2.2 run s22-spark-ecf8-20261121T004433Z). "
                     "Next: S2.4 replay_pusch."):
            self.assertIn(line, summary)
        calls = self.calls.read_text()
        self.assertIn(f"s22 run: {self.s22}", calls)
        self.assertIn(f"scripts/native_adapter.py build --ocudu-build-dir {self.build}", calls)
        self.assertIn("scripts/verify_environment.py --profile core --with-native", calls)
        self.assertNotIn("--grant-store", calls)
        self.assertEqual(summary.count("PASS  "), 14)

    def test_the_newest_accepted_sprint_2_2_run_is_used_and_a_failed_one_is_not(self):
        self.accepted_run("s22-spark-ecf8-20261121T090000Z", S22_SUMMARY.replace("PASS  committed", "FAIL  committed"))
        self.accepted_run("s22-spark-ecf8-20261121T100000Z", "Sprint 2.2: NOT ACCEPTED on this host\n")
        self.accepted_run("s22-spark-ecf8-20261121T110000Z", pair=False)  # a verdict without the stored pair
        newest = self.accepted_run("s22-spark-ecf8-20261121T020000Z")
        self.assertEqual(self.gate().returncode, 0)
        self.assertIn(f"s22 run: {newest}", self.calls.read_text())
        self.assertIn("S2.2 run s22-spark-ecf8-20261121T020000Z", self.summary())

    def test_a_named_run_a_development_host_and_named_grant_stores(self):
        other = self.accepted_run("s22-devhost-20261119T000000Z", S22_SUMMARY.replace("DGX Spark", "host"))
        result = self.gate("--s22-run", str(other), "--grant-store", "/moved/s21/grant_fixture",
                           "--grant-store", "/moved/s22 run/grant_fixture")
        self.assertEqual(result.returncode, 0, result.stdout)
        calls = self.calls.read_text()
        self.assertIn(f"s22 run: {other}", calls)
        self.assertIn("--grant-store /moved/s21/grant_fixture --grant-store /moved/s22 run/grant_fixture --output-dir", calls)
        self.assertIn("Sprint 2.3: ACCEPTED on this host.", result.stdout)

    def test_without_an_accepted_stimulus_nothing_is_applied(self):
        shutil.rmtree(self.s22)
        self.accepted_run("s22-spark-ecf8-20261121T004433Z", pair=False)
        result = self.gate()
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S2.3 stimulus: no Sprint 2.2 ACCEPTED verdict with a stored pair", result.stdout)
        self.assertIn("Run 'make accept-s2.2' on this host first", result.stdout)
        self.assertIn("SKIP  S2.3 reference check, adapter build, integration driver and grid recovery", result.stdout)
        self.assertIn("Sprint 2.3: NOT ACCEPTED on this host", result.stdout)
        self.assertNotIn("tests/integration/grid_apply.py", self.calls.read_text())

    def test_a_missing_sprint_1_1_build_is_named(self):
        (self.bin / "pusch_e2e_pipeline_test").unlink()
        result = self.gate()
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S2.3 needs the Sprint 1.1 build", result.stdout)
        self.assertNotIn("native_adapter.py", self.calls.read_text())

    def test_failures_of_single_gates(self):
        cases = (
            (dict(unit="Ran 265 tests in 4.0s\n\nFAILED (failures=1)", unit_status="1"), "FAIL  unit tests (265 ran"),
            (dict(card="Ran 0 tests in 0.0s\n\nOK"), "FAIL  S2.3 unit gate, run directly: python "
                                                    "tests/unit/test_grid_apply.py (none ran)"),
            # The native unit cases are part of the unit gate: a run that skipped them cannot close it.
            (dict(card="Ran 29 tests in 0.5s\n\nOK (skipped=3)"), "FAIL  S2.3 unit gate, run directly: python "
             "tests/unit/test_grid_apply.py (29 ran, 3 skipped: the native unit cases need cmake"),
            (dict(environment="1"), "FAIL  S2.3 references: the Sprint 1.1 build"),
            (dict(build="1"), "FAIL  S2.3 adapter: build or verification failed"),
            (dict(verify="1"), "FAIL  S2.3 adapter: build or verification failed"),
            (dict(receipt=dict(RECEIPT, executables={"grant_fixture": {"sha256": "0" * 64}})),
             "FAIL  S2.3 adapter: the receipt names no grid_apply"),
            (dict(ctest=CTEST.replace("replay_core_case_13 ..............   Passed", "replay_core_case_13 ..............***Failed")),
             "FAIL  S2.3 native unit cases: 13 cases of the core passed"),
            (dict(ctest=CTEST.replace("grid_apply_pinned_grid ....................   Passed", "grid_apply_pinned_grid ....................***Failed")),
             "FAIL  S2.3 native unit cases: 14 cases of the core passed and the pinned-grid self-test did not"),
            (dict(no_ctest="1"), "FAIL  S2.3 native unit cases: 0 cases of the core passed"),
            (dict(grade=GRADE.replace("PASS  S2.3 reference-pair stored", "FAIL  S2.3 reference-pair stored"),
                  grade_json=dict(GRADE_JSON, judgment="fail", cases_passed=4, checks_passed=4, checks_failed=1),
                  driver="1"), "FAIL  S2.3 integration grade: 4 of 5 cases, 4 of 5 checks passed, driver exit status 1"),
            (dict(no_evidence="1", driver="2"), "FAIL  S2.3 integration driver produced no graded checks"),
            (dict(grade_json=dict(GRADE_JSON, cases=0, cases_passed=0)), "FAIL  S2.3 integration grade: 0 of 0 cases"),
            (dict(show="1"), "FAIL  S2.3 a second reader recovers a stored grid (see recovered.log)"),
            (dict(grid="another grid"), "FAIL  S2.3 a second reader recovers a stored grid: its digest is"),
            (dict(pins="1"), "FAIL  S2.3 sources at the locked revisions after the runs"),
        )
        for number, (settings, line) in enumerate(cases):
            run_dir = self.root.parent / f"evidence-{number}"
            result = self.gate(run_dir=run_dir, **settings)
            with self.subTest(line=line):
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn(line, result.stdout)
                self.assertIn("Sprint 2.3: NOT ACCEPTED on this host", result.stdout)
                self.assertNotIn("Sprint 2.3: ACCEPTED", result.stdout)
        skipped = self.gate(run_dir=self.root.parent / "evidence-build", build="1")
        self.assertIn("SKIP  S2.3 native unit cases, integration driver and grid recovery: no adapter", skipped.stdout)
        # A skip in the portable suite is reported and does not fail it: only the unit gate must be complete.
        portable = self.gate(run_dir=self.root.parent / "evidence-skip", unit="Ran 265 tests in 9.0s\n\nOK (skipped=12)")
        self.assertEqual(portable.returncode, 0, portable.stdout)
        self.assertIn("PASS  unit tests (265 ran, 12 skipped)", portable.stdout)

    def test_a_failed_check_of_the_driver_fails_the_gate_even_with_exit_status_zero(self):
        result = self.gate(grade=GRADE.replace("PASS  S2.3 refusals gain rules", "FAIL  S2.3 refusals gain rules"))
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S2.3 refusals gain rules", result.stdout)

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
        for arguments in (("--unknown",), ("--s22-run",), ("--grant-store",)):
            result = self.gate(*arguments, run_dir=self.root.parent / "evidence-usage")
            self.assertEqual(result.returncode, 2)
            self.assertIn("Usage: bash scripts/accept_s2_3.sh", result.stderr)
        (self.bin / "python").unlink()
        missing = self.gate(run_dir=self.root.parent / "evidence-python")
        self.assertEqual(missing.returncode, 2)
        self.assertIn("Run 'make env' first", missing.stderr)


if __name__ == "__main__":
    unittest.main()
