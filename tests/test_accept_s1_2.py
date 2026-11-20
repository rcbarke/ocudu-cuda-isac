"""Check the Sprint 1.2 gate runner against tool doubles. No compiler, source, build or GPU is needed."""

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/accept_s1_2.sh"
TEST = "tests/integrationtests/phy/upper/channel_processors/pusch_e2e_pipeline_test"
PAPER_GRADE = """PASS  S1.2 paper run completed: the native test and the transcript validator exited zero
PASS  S1.2 paper CPU decodes: 200/200 CRC passes
PASS  S1.2 paper GPU decodes: 200/200 CRC passes
NOTE  S1.2 paper performance, GPU mean: 264.9 us here; 183.9 us in the paper (+44.0 percent)"""
SMOKE_GRADE = """PASS  S1.2 smoke run completed: the native test and the transcript validator exited zero
PASS  S1.2 smoke GPU decodes: 20/20 CRC passes"""
S11_SUMMARY = """Sprint 1.1 acceptance on spark-ecf8, 2026-11-20T01:10:24Z, spark profile
PASS  committed tree
PASS  S1.1 native build command (native-build.log, evidence in native-build/)
Sprint 1.1: ACCEPTED on this DGX Spark. Next gate: make paper-baseline (S1.2).
"""
# The interpreter double answers each gate from FAKE_* variables and logs the call. A measured run
# writes the files the real runner leaves: a status, and a transcript unless the run stopped early.
PYTHON_DOUBLE = """#!/bin/sh
[ "$1" = "-c" ] && exec {real} "$@"
echo "$*" >> "$FAKE_LOG"
measure() {{
    profile=$1 status=$2 evidence=$3
    echo "Run artifacts: $evidence"
    [ -z "${{FAKE_NO_EVIDENCE:-}}" ] || exit "$status"
    mkdir -p "$evidence"
    if [ "$status" = 0 ]; then
        printf 'stage=completed\\nexit_code=0\\n' > "$evidence/status.txt"
        echo "transcript of $profile" > "$evidence/pusch.log"
    elif [ -n "${{FAKE_EARLY:-}}" ]; then
        printf 'stage=build-identity\\nexit_code=1\\n' > "$evidence/status.txt"
        printf '%s\\n' "Build identity changed. Rebuild using scripts/build_ocudu.sh." "  second line" > "$evidence/error.txt"
    else
        printf 'stage=run\\nexit_code=1\\n' > "$evidence/status.txt"
        printf '%s\\n' "timeout exited with status 2 (see pusch.log)" "  *** FAIL: resident GPU pipeline ***" > "$evidence/error.txt"
        echo "transcript of a failed $profile" > "$evidence/pusch.log"
    fi
    exit "$status"
}}
case "$*" in
    "-m pip freeze --all") echo "example==1.0" ;;
    *"-m unittest discover"*) printf '%s\\n' "$FAKE_UNIT" >&2; exit "${{FAKE_UNIT_STATUS:-0}}" ;;
    *"--profile core --with-native"*)
        [ "${{FAKE_NATIVE:-0}}" = 0 ] || echo "ValueError: Native binaries differ from the locked full-build receipt" >&2
        exit "${{FAKE_NATIVE:-0}}" ;;
    *"native_ocudu.py run --profile paper --skip-build --build-dir "*) measure paper "${{FAKE_PAPER:-0}}" "$9" ;;
    *"native_ocudu.py run --profile smoke --skip-build --build-dir "*) measure smoke "${{FAKE_SMOKE:-0}}" "$9" ;;
    *"native_ocudu.py grade-run "*/paper) printf '%s\\n' "$FAKE_PAPER_GRADE"; exit "${{FAKE_PAPER_GRADE_STATUS:-0}}" ;;
    *"native_ocudu.py grade-run "*/smoke) printf '%s\\n' "$FAKE_SMOKE_GRADE"; exit "${{FAKE_SMOKE_GRADE_STATUS:-0}}" ;;
    *"check_pins.py --worktrees --only ocudu") echo "ext/ocudu 27e84cb3 (Git worktree)"; exit "${{FAKE_PINS:-0}}" ;;
    *) echo "interpreter double: unexpected $*" >&2; exit 64 ;;
esac
"""


@unittest.skipUnless(shutil.which("bash"), "the gate runner is a Bash script")
class SprintOneReceiverGateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / "host checkout"
        self.bin = self.root / "isac/bin"
        (self.root / "scripts").mkdir(parents=True)
        self.bin.mkdir(parents=True)
        shutil.copy(SCRIPT, self.root / "scripts/accept_s1_2.sh")
        python = self.bin / "python"
        python.write_text(PYTHON_DOUBLE.format(real=shlex.quote(sys.executable)))
        python.chmod(0o755)
        # The build Sprint 1.1 published: the test executable, its link and the receipt, which names
        # the evidence directory inside the accepted Sprint 1.1 run.
        self.build = self.root / "build/ocudu-spark"
        self.s11 = self.root / "results/runs/s11-spark-ecf8-20261120T011024Z"
        (self.build / TEST).parent.mkdir(parents=True)
        (self.build / TEST).write_text("#!/bin/sh\n")
        (self.bin / "pusch_e2e_pipeline_test").symlink_to(self.build / TEST)
        (self.s11 / "native-build").mkdir(parents=True)
        (self.s11 / "summary.txt").write_text(S11_SUMMARY)
        self.receipt = self.build / "isac-native.json"
        self.receipt.write_text(json.dumps({"build_evidence": str(self.s11 / "native-build"), "target": "all"}))
        self.calls = self.root.parent / "calls.log"
        self.run_dir = self.root.parent / "evidence"

    def gate(self, *arguments, run_dir=None, **settings):
        env = {key: value for key, value in os.environ.items() if not key.startswith("FAKE_")}
        env.update(FAKE_LOG=str(self.calls), FAKE_UNIT="Ran 130 tests in 1.000s\n\nOK",
                   FAKE_PAPER_GRADE=PAPER_GRADE, FAKE_SMOKE_GRADE=SMOKE_GRADE)
        env.update({f"FAKE_{key.upper()}": str(value) for key, value in settings.items()})
        command = [shutil.which("bash"), str(self.root / "scripts/accept_s1_2.sh"),
                   "--run-dir", str(run_dir or self.run_dir), *arguments]
        return subprocess.run(command, env=env, capture_output=True, text=True, timeout=120, check=False)

    def test_all_gates_pass_on_the_accepted_build_and_close_sprint_one(self):
        result = self.gate()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("FAIL", result.stdout)
        self.assertNotIn("SKIP", result.stdout)
        summary = (self.run_dir / "summary.txt").read_text()
        # Everything on the screen is in the summary, in the same order.
        self.assertEqual(result.stdout, summary)
        lines = result.stdout.splitlines()
        for line in ("PASS  unit tests (130 ran, 0 skipped)",
                     "PASS  S1.2 the Sprint 1.1 build is intact: gnb and pusch_e2e_pipeline_test in isac/bin match "
                     "its receipt (native-env.json)",
                     "PASS  S1.2 the measured build is the one Sprint 1.1 accepted: run s11-spark-ecf8-20261120T011024Z",
                     "PASS  S1.2 paper run command (paper-run.log, evidence in paper/)",
                     *PAPER_GRADE.splitlines(),
                     "PASS  S1.2 smoke run command (smoke-run.log, evidence in smoke/)",
                     *SMOKE_GRADE.splitlines(),
                     "PASS  S1.2 OCUDU source at the locked revision after the runs (sources-after.txt)",
                     f"Evidence: {self.run_dir}"):
            self.assertIn(line, lines)
        self.assertEqual(lines[-2:], [
            "Sprint 1.2: ACCEPTED on this DGX Spark. Table II correctness is reproduced. The timing comparison is "
            "recorded in paper/table-ii.json and is not a gate.",
            "Sprint 1: ACCEPTED on this DGX Spark (S1.1 run s11-spark-ecf8-20261120T011024Z, S1.2 run evidence). "
            "Next: S2.1 grant_fixture."])
        for name in ("python-packages.txt", "unit-tests.log", "native-env.log", "paper-run.log", "paper-grade.txt",
                     "smoke-run.log", "smoke-grade.txt", "sources-after.txt", "paper/pusch.log", "smoke/pusch.log"):
            self.assertTrue((self.run_dir / name).is_file(), name)
        self.assertEqual((self.run_dir / "paper-run.log").read_text(), f"Run artifacts: {self.run_dir / 'paper'}\n")
        calls = self.calls.read_text().splitlines()
        # Each profile measures the build that isac/bin links to, into its own new evidence directory.
        for profile in ("paper", "smoke"):
            self.assertIn(f"scripts/native_ocudu.py run --profile {profile} --skip-build --build-dir {self.build} "
                          f"--run-dir {self.run_dir / profile}", calls)
            self.assertIn(f"scripts/native_ocudu.py grade-run {self.run_dir / profile}", calls)
        self.assertIn(f"scripts/verify_environment.py --profile core --with-native --report "
                      f"{self.run_dir / 'native-env.json'}", calls)
        # The build is verified before it is measured. The paper row runs before the smoke, and
        # the pin is checked only after both ran against the source.
        position = [next(index for index, call in enumerate(calls) if step in call)
                    for step in ("-m unittest discover", "--with-native", "run --profile paper", "grade-run",
                                 "run --profile smoke", "check_pins.py --worktrees --only ocudu")]
        self.assertEqual(position, sorted(position))
        self.assertFalse([call for call in calls if "native_ocudu.py build" in call])

    def test_any_failed_gate_rejects_the_host_and_the_rest_still_run(self):
        cases = (
            ({"paper_grade": PAPER_GRADE.replace("PASS  S1.2 paper GPU", "FAIL  S1.2 paper GPU"),
              "paper_grade_status": 1}, "FAIL  S1.2 paper GPU decodes: 200/200 CRC passes"),
            ({"smoke_grade": SMOKE_GRADE.replace("PASS  S1.2 smoke GPU", "FAIL  S1.2 smoke GPU"),
              "smoke_grade_status": 1}, "FAIL  S1.2 smoke GPU decodes: 20/20 CRC passes"),
            ({"paper_grade": "Traceback (most recent call last):\nKeyError: 'gpu'", "paper_grade_status": 1},
             "FAIL  S1.2 paper evidence could not be graded (see paper-grade.txt)\n"
             "      Traceback (most recent call last):\n      KeyError: 'gpu'\n"),
            ({"paper_grade": ""}, "FAIL  S1.2 paper evidence produced no graded checks (see paper-grade.txt)"),
            ({"paper_grade": "NOTE  S1.2 paper performance only"},
             "FAIL  S1.2 paper evidence produced no graded checks (see paper-grade.txt)"),
            ({"unit_status": 1, "unit": "FAIL: test_grade (tests.GradeTest.test_grade)\nRan 130 tests in 1s\n\n"
                                        "FAILED (failures=1)"},
             "FAIL  unit tests (130 ran; see unit-tests.log)\n      FAIL: test_grade (tests.GradeTest.test_grade)\n"),
            ({"unit": "Ran 0 tests in 0.000s\n\nOK"}, "FAIL  unit tests (none ran)"),
            ({"unit": "no tests were collected"}, "FAIL  unit tests"),
            ({"native": 1}, "FAIL  S1.2 the Sprint 1.1 build is intact: gnb and pusch_e2e_pipeline_test in isac/bin "
                            "match its receipt (native-env.json) (see native-env.log)\n"
                            "      ValueError: Native binaries differ from the locked full-build receipt\n"),
            ({"pins": 1}, "FAIL  S1.2 OCUDU source at the locked revision after the runs"),
        )
        for index, (settings, message) in enumerate(cases):
            self.calls.write_text("")
            result = self.gate(run_dir=self.root.parent / f"evidence {index}", **settings)
            with self.subTest(settings=settings):
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn(message, result.stdout)
                self.assertIn("Sprint 1.2: NOT ACCEPTED on this host", result.stdout)
                self.assertNotIn("ACCEPTED on this DGX Spark", result.stdout)
                calls = self.calls.read_text()
                for later in ("run --profile paper", "run --profile smoke", "grade-run", "check_pins.py --worktrees"):
                    self.assertIn(later, calls)

    def test_failed_run_is_graded_when_the_test_ran_and_skipped_when_it_never_started(self):
        failing = PAPER_GRADE.replace("PASS  S1.2 paper run completed", "FAIL  S1.2 paper run completed")
        result = self.gate(paper=1, paper_grade=failing, paper_grade_status=1)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        shown = result.stdout.splitlines()
        stopped = ("FAIL  S1.2 paper run command stopped in stage run: timeout exited with status 2 (see pusch.log) "
                   "(see paper-run.log)")
        # The grade of the archived transcript follows the failed command and names the gates.
        self.assertLess(shown.index(stopped), shown.index(failing.splitlines()[0]))
        self.assertEqual([line for line in shown if line.startswith("FAIL")], [stopped, failing.splitlines()[0]])
        self.assertNotIn("*** FAIL: resident GPU pipeline ***", result.stdout)
        self.assertIn("PASS  S1.2 smoke run command (smoke-run.log, evidence in smoke/)", shown)
        # A run that stopped before the native test started has nothing to grade.
        self.calls.write_text("")
        result = self.gate(run_dir=self.root.parent / "early", paper=1, early=1)
        self.assertEqual(result.returncode, 1)
        shown = result.stdout.splitlines()
        position = shown.index("FAIL  S1.2 paper run command stopped in stage build-identity: Build identity changed. "
                               "Rebuild using scripts/build_ocudu.sh. (see paper-run.log)")
        self.assertEqual(shown[position + 1:position + 3], [
            "        second line",
            "SKIP  S1.2 paper evidence grade: the run stopped before the native test wrote a transcript"])
        self.assertNotIn(f"grade-run {self.root.parent / 'early/paper'}", self.calls.read_text())
        self.assertFalse((self.root.parent / "early/paper-grade.txt").exists())
        self.assertIn("Sprint 1.2: NOT ACCEPTED on this host", shown)
        # A runner that dies before it writes anything is still one clear failure.
        result = self.gate(run_dir=self.root.parent / "silent", smoke=3, no_evidence=1, paper=3)
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S1.2 smoke run command stopped in stage unknown: no evidence written (see smoke-run.log)",
                      result.stdout)

    def test_gate_measures_only_a_build_that_sprint_1_1_accepted(self):
        accepted = "S1.2 the measured build is the one Sprint 1.1 accepted"
        summary = self.s11 / "summary.txt"
        cases = (
            (S11_SUMMARY.replace("Sprint 1.1: ACCEPTED on this DGX Spark", "Sprint 1.1: NOT ACCEPTED on this host"),
             "an unaccepted run"),
            (S11_SUMMARY.replace("PASS  S1.1 native build command", "FAIL  S1.1 native build command"),
             "a summary that holds a failed gate"),
            (S11_SUMMARY.replace("on this DGX Spark", "on this host for the local profile"), "a local-profile build"),
            (None, "a build made outside the gate"),
        )
        for index, (text, case) in enumerate(cases):
            summary.unlink(missing_ok=True)
            if text is not None:
                summary.write_text(text)
            self.calls.write_text("")
            result = self.gate(run_dir=self.root.parent / f"record {index}")
            with self.subTest(case=case):
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn(f"FAIL  {accepted}: no ACCEPTED verdict in {summary}", result.stdout)
                self.assertIn("Run 'make accept-s1.1', then this gate again.", result.stdout)
                self.assertNotIn("Sprint 1: ACCEPTED", result.stdout)
                # The measurement is still taken and graded, so one run shows every open item.
                self.assertIn("run --profile paper", self.calls.read_text())
        summary.write_text(S11_SUMMARY)
        self.receipt.write_text("not json")
        result = self.gate(run_dir=self.root.parent / "unreadable receipt")
        self.assertEqual(result.returncode, 1)
        self.assertIn(f"FAIL  {accepted}", result.stdout)
        # The gate that the current Sprint 1.1 runner prints is accepted as well.
        self.receipt.write_text(json.dumps({"build_evidence": str(self.s11 / "native-build")}))
        summary.write_text(S11_SUMMARY.replace("make paper-baseline (S1.2)", "make accept-s1.2 (S1.2)"))
        self.assertEqual(self.gate(run_dir=self.root.parent / "current wording").returncode, 0)

    def test_missing_build_is_one_failure_and_nothing_is_measured(self):
        needs = ("FAIL  S1.2 needs the Sprint 1.1 build: isac/bin/pusch_e2e_pipeline_test is not a link into a "
                 "build with a receipt")
        link = self.bin / "pusch_e2e_pipeline_test"

        def unpublished():
            link.unlink()

        def copied():
            link.unlink()
            shutil.copy(self.build / TEST, link)

        def elsewhere():
            link.unlink()
            link.symlink_to(self.root / "older/pusch_e2e_pipeline_test")

        for index, damage in enumerate((unpublished, copied, elsewhere, self.receipt.unlink)):
            damage()
            self.calls.write_text("")
            result = self.gate(run_dir=self.root.parent / f"no build {index}")
            with self.subTest(damage=damage.__name__):
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                shown = result.stdout.splitlines()
                self.assertEqual([line for line in shown if line.startswith("FAIL")], [needs])
                self.assertIn("      Run 'make accept-s1.1' on this host first.", shown)
                self.assertIn("SKIP  S1.2 build checks, paper run and smoke run: no Sprint 1.1 build", shown)
                calls = self.calls.read_text()
                for skipped in ("native_ocudu.py", "--with-native"):
                    self.assertNotIn(skipped, calls)
                self.assertIn("check_pins.py --worktrees --only ocudu", calls)
            link.unlink(missing_ok=True)
            link.symlink_to(self.build / TEST)
            self.receipt.write_text(json.dumps({"build_evidence": str(self.s11 / "native-build")}))
        self.assertEqual(self.gate(run_dir=self.root.parent / "restored").returncode, 0)

    def test_arguments_and_existing_evidence_are_refused_before_any_gate_runs(self):
        self.run_dir.mkdir()
        keep = self.run_dir / "summary.txt"
        keep.write_text("earlier run\n")
        new = {"run_dir": self.root.parent / "new"}
        refusals = (((), {}, "already exists"), (("cuda",), new, "Usage"), (("--run-dir",), new, "Usage"),
                    (("--machine", "local"), new, "Usage"), (("--build-dir", "build/other"), new, "Usage"),
                    (("--skip-build",), new, "Usage"))
        for arguments, options, message in refusals:
            result = self.gate(*arguments, **options)
            with self.subTest(arguments=arguments):
                self.assertEqual(result.returncode, 2)
                self.assertIn(message, result.stderr)
        self.assertEqual(keep.read_text(), "earlier run\n")
        self.assertFalse((self.root.parent / "new").exists())
        self.assertFalse(self.calls.exists())
        self.assertEqual(self.gate("--help").returncode, 0)
        (self.bin / "python").unlink()
        result = self.gate(run_dir=self.root.parent / "new")
        self.assertEqual(result.returncode, 2)
        self.assertIn("make env", result.stderr)

    @unittest.skipUnless(shutil.which("git"), "the committed-tree check reads Git metadata")
    def test_evidence_needs_a_committed_tree(self):
        identity = {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
                    "GIT_AUTHOR_NAME": "fixture", "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
                    "GIT_COMMITTER_NAME": "fixture", "GIT_COMMITTER_EMAIL": "fixture@example.invalid"}

        def git(*arguments):
            subprocess.run(["git", *arguments], cwd=self.root, env={**os.environ, **identity},
                           capture_output=True, text=True, check=True)

        (self.root / ".gitignore").write_text("/isac/\n/build/\n/results/runs/\n")
        git("init", "-q")
        git("add", "-A")
        git("commit", "-q", "-m", "fixture checkout")
        result = self.gate(run_dir=self.root / "results/runs/s12-clean")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS  committed tree", result.stdout)
        self.assertRegex(result.stdout, r"commit [0-9a-f]{40}")
        # The evidence the gate wrote is ignored, so a second run starts from the same clean tree.
        self.assertEqual(self.gate(run_dir=self.root / "results/runs/s12-again").returncode, 0)
        (self.root / "scripts/accept_s1_2.sh").write_text(SCRIPT.read_text() + "# local edit\n")
        result = self.gate(run_dir=self.root / "results/runs/s12-dirty")
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  committed tree", result.stdout)
        self.assertIn("accept_s1_2.sh", (self.root / "results/runs/s12-dirty/git-status.txt").read_text())


if __name__ == "__main__":
    unittest.main()
