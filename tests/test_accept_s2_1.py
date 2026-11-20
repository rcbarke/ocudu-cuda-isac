"""Check the Sprint 2.1 gate runner against tool doubles. No compiler, source, build, Sionna or GPU is needed."""

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

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/accept_s2_1.sh"
TEST = "tests/integrationtests/phy/upper/channel_processors/pusch_e2e_pipeline_test"
BLOCK = b"original transport block"
BLOCK_SHA256 = hashlib.sha256(BLOCK).hexdigest()
GRADE = """PASS  S2.1 two-prb-offset grant record: valid pusch-grant/v1, 384 bits
PASS  S2.1 two-prb-offset data symbols: 288 symbols of both references compared, 0 outside the storage tolerance
NOTE  S2.1 two-prb-offset grid hashes: x.npy ba9258f0630ad918
PASS  S2.1 demo-51prb-4rx receiver check: host PUSCH processor on 4 receive port(s): CRC pass
grant_fixture: 2 of 2 cases passed, 3 of 3 checks"""
GRADE_JSON = {"judgment": "pass", "cases": 2, "cases_passed": 2, "checks": 3, "checks_passed": 3, "checks_failed": 0}
RECEIPT = {"executables": {"grant_fixture": {"sha256": "0123456789abcdef" * 4}}, "tests_run": 2, "compiler_warnings": 0,
           "pin": {"cuda": True, "archives": 52}}
RECOVERED = {"tb_sha256": BLOCK_SHA256, "pilot": {"symbol": 2, "subcarrier": 180},
             "data": {"symbol": 0, "subcarrier": 180, "value": [0.31640625, 0.31640625]}}
S11_SUMMARY = """Sprint 1.1 acceptance on spark-ecf8, 2026-11-20T01:10:24Z, spark profile
PASS  committed tree
PASS  S1.1 native build command (native-build.log, evidence in native-build/)
Sprint 1.1: ACCEPTED on this DGX Spark. Next gate: make accept-s1.2 (S1.2).
"""
UNIT = "Ran 170 tests in 1.000s\n\nOK"
# The interpreter double answers each gate from FAKE_* variables and logs the call. The driver
# double writes the files the real driver leaves: its grade and the store entry of each case.
PYTHON_DOUBLE = """#!/bin/sh
[ "$1" = "-c" ] && exec {real} "$@"
echo "$*" >> "$FAKE_LOG"
case "$*" in
    "-m pip freeze --all") echo "example==1.0" ;;
    *"-m unittest discover"*) printf '%s\\n' "$FAKE_UNIT" >&2; exit "${{FAKE_UNIT_STATUS:-0}}" ;;
    "tests/unit/test_grant_fixture.py -v") printf '%s\\n' "$FAKE_CARD" >&2; exit "${{FAKE_CARD_STATUS:-0}}" ;;
    *"--profile simulation --backend cpu --with-native"*)
        [ "${{FAKE_ENVIRONMENT:-0}}" = 0 ] || echo "ValueError: installed Sionna file differs from the pinned tree" >&2
        exit "${{FAKE_ENVIRONMENT:-0}}" ;;
    "scripts/native_adapter.py build --ocudu-build-dir "*)
        echo "Adapter build: build/ocudu-adapter"
        [ "${{FAKE_BUILD:-0}}" = 0 ] || echo "native_adapter: the adapter build step exited with status 2" >&2
        exit "${{FAKE_BUILD:-0}}" ;;
    "scripts/native_adapter.py verify")
        [ "${{FAKE_VERIFY:-0}}" = 0 ] || {{ echo "native_adapter: the adapter sources changed after the build" >&2; exit 1; }}
        printf '%s\\n' "$FAKE_RECEIPT" ;;
    "tests/integration/grant_fixture.py --config configs/integration/grant_fixture.json --output-dir "*)
        out=$5
        echo "Run artifacts: $out"
        [ -z "${{FAKE_NO_EVIDENCE:-}}" ] || exit "${{FAKE_DRIVER:-0}}"
        mkdir -p "$out/two-prb-offset/store"
        printf '%s' "${{FAKE_BLOCK:-original transport block}}" > "$out/two-prb-offset/store/tb.bin"
        printf '%s\\n' "$FAKE_GRADE" > "$out/grade.txt"
        printf '%s\\n' "$FAKE_GRADE_JSON" > "$out/grade.json"
        exit "${{FAKE_DRIVER:-0}}" ;;
    "-m ocudu_cuda_isac.episodes.grant show "*)
        [ "${{FAKE_SHOW:-0}}" = 0 ] || {{ echo "grant: tb.bin differs from the size or hash in the manifest" >&2; exit 2; }}
        printf '%s\\n' "$FAKE_RECOVERED" ;;
    *"check_pins.py --worktrees --only ocudu") echo "ext/ocudu 27e84cb3 (Git worktree)"; exit "${{FAKE_PINS:-0}}" ;;
    *) echo "interpreter double: unexpected $*" >&2; exit 64 ;;
esac
"""


@unittest.skipUnless(shutil.which("bash") and shutil.which("sha256sum"), "the gate runner is a Bash script")
class SprintTwoGrantGateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / "host checkout"
        self.bin = self.root / "isac/bin"
        (self.root / "scripts").mkdir(parents=True)
        self.bin.mkdir(parents=True)
        shutil.copy(SCRIPT, self.root / "scripts/accept_s2_1.sh")
        python = self.bin / "python"
        python.write_text(PYTHON_DOUBLE.format(real=shlex.quote(sys.executable)))
        python.chmod(0o755)
        # What the gate reads itself: the configuration's first case and that grant's identifier.
        (self.root / "configs/integration").mkdir(parents=True)
        (self.root / "configs/grants").mkdir()
        (self.root / "configs/grants/two_prb_offset.json").write_text(json.dumps({"grant_id": "two-prb-offset"}))
        (self.root / "configs/integration/grant_fixture.json").write_text(json.dumps({"cases": [
            {"grant": "configs/grants/two_prb_offset.json", "expect": {"tb_sha256": BLOCK_SHA256}}]}))
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
        env.update(FAKE_LOG=str(self.calls), FAKE_UNIT=UNIT, FAKE_CARD="Ran 36 tests in 0.300s\n\nOK",
                   FAKE_GRADE=GRADE, FAKE_GRADE_JSON=json.dumps(GRADE_JSON), FAKE_RECEIPT=json.dumps(RECEIPT),
                   FAKE_RECOVERED=json.dumps(RECOVERED))
        env.update({f"FAKE_{key.upper()}": value if isinstance(value, str) else json.dumps(value)
                    for key, value in settings.items()})
        command = [shutil.which("bash"), str(self.root / "scripts/accept_s2_1.sh"),
                   "--run-dir", str(run_dir or self.run_dir), *arguments]
        return subprocess.run(command, env=env, capture_output=True, text=True, timeout=120, check=False)

    def test_all_gates_pass_on_the_accepted_build(self):
        result = self.gate()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("FAIL", result.stdout)
        self.assertNotIn("SKIP", result.stdout)
        summary = (self.run_dir / "summary.txt").read_text()
        self.assertEqual(result.stdout, summary)  # everything on the screen is in the summary, in the same order
        lines = result.stdout.splitlines()
        for line in ("PASS  unit tests (170 ran, 0 skipped)",
                     "PASS  S2.1 unit gate, run directly: python tests/unit/test_grant_fixture.py (36 ran, 0 skipped)",
                     "PASS  S2.1 references: the pinned Sionna on the CPU and the Sprint 1.1 build, from one "
                     "interpreter (environment.json)",
                     "PASS  S2.1 native boundary: the OCUDU build is the one Sprint 1.1 accepted, run "
                     "s11-spark-ecf8-20261120T011024Z",
                     "PASS  S2.1 adapter build command (adapter-build.log, evidence in adapter-build/)",
                     "PASS  S2.1 adapter receipt: grant_fixture 0123456789ab, compiled and linked like "
                     "pusch_e2e_pipeline_test (with ENABLE_CUDA, 52 OCUDU archives), 2 adapter tests, 0 compiler "
                     "warnings (adapter-receipt.json)",
                     *GRADE.splitlines()[:-1],
                     "PASS  S2.1 integration grade: 2 of 2 cases, 3 of 3 checks passed (grant_fixture/grade.json)",
                     "PASS  S2.1 a second reader recovers the record: pilot 0 at symbol 2, subcarrier 180; data symbol 0 "
                     f"at symbol 0, subcarrier 180 = +0.316406+0.316406j; transport block SHA-256 {BLOCK_SHA256[:16]} "
                     "by the reader and by sha256sum (recovered.json)",
                     "PASS  S2.1 OCUDU source at the locked revision after the runs (sources-after.txt)",
                     f"Evidence: {self.run_dir}"):
            self.assertIn(line, lines)
        self.assertNotIn(GRADE.splitlines()[-1], lines)  # the driver's own tally is not a gate line
        self.assertEqual(lines[-1], "Sprint 2.1: ACCEPTED on this DGX Spark. One legal uplink grant and its transmit "
                                    "reference are archived, and both transmit references agree (S1.1 run "
                                    "s11-spark-ecf8-20261120T011024Z). Next: S2.2 episode_pair and S2.3 grid_apply.")
        for name in ("python-packages.txt", "unit-tests.log", "grant-fixture-unit.log", "environment.log",
                     "adapter-build.log", "adapter-receipt.json", "grant-fixture.log", "grant_fixture/grade.txt",
                     "recovered.json", "sources-after.txt"):
            self.assertTrue((self.run_dir / name).is_file(), name)
        calls = self.calls.read_text().splitlines()
        self.assertIn(f"scripts/verify_environment.py --profile simulation --backend cpu --with-native --report "
                      f"{self.run_dir / 'environment.json'}", calls)
        # The adapter is built against the build that isac/bin links to, with its logs in this run.
        self.assertIn(f"scripts/native_adapter.py build --ocudu-build-dir {self.build} --run-dir "
                      f"{self.run_dir / 'adapter-build'}", calls)
        self.assertIn(f"-m ocudu_cuda_isac.episodes.grant show {self.run_dir / 'grant_fixture/two-prb-offset/store'} "
                      "--pilot 0 --data 0", calls)
        # References are verified before the adapter is built, the adapter before it runs, and the
        # pin is checked only after everything ran against the source.
        position = [next(index for index, call in enumerate(calls) if step in call)
                    for step in ("-m unittest discover", "tests/unit/test_grant_fixture.py -v", "--with-native",
                                 "native_adapter.py build", "native_adapter.py verify",
                                 "tests/integration/grant_fixture.py", "episodes.grant show",
                                 "check_pins.py --worktrees --only ocudu")]
        self.assertEqual(position, sorted(position))
        self.assertFalse([call for call in calls if "native_ocudu.py" in call])  # OCUDU is never rebuilt

    def test_a_local_profile_build_is_accepted_and_named_as_such(self):
        (self.s11 / "summary.txt").write_text(
            S11_SUMMARY.replace("ACCEPTED on this DGX Spark. Next gate: make accept-s1.2 (S1.2).",
                                "ACCEPTED on this host for the local profile. The DeepSig Table II row stays "
                                "assigned to a GB10."))
        result = self.gate()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Sprint 2.1: ACCEPTED on this host. One legal uplink grant", result.stdout)
        self.assertNotIn("DGX Spark", result.stdout.splitlines()[-1])

    def test_any_failed_gate_rejects_the_host_and_the_rest_still_run(self):
        failing = GRADE.replace("PASS  S2.1 two-prb-offset data symbols: 288 symbols of both references compared, 0",
                                "FAIL  S2.1 two-prb-offset data symbols: 288 symbols of both references compared, 7")
        failed_grade = dict(GRADE_JSON, judgment="fail", cases_passed=1, checks_passed=2, checks_failed=1)
        cases = (
            ({"grade": failing, "grade_json": failed_grade, "driver": 1},
             "FAIL  S2.1 two-prb-offset data symbols: 288 symbols of both references compared, 7 outside the storage "
             "tolerance\n"),
            ({"grade": failing, "grade_json": failed_grade, "driver": 1},
             "FAIL  S2.1 integration grade: 1 of 2 cases, 2 of 3 checks passed, driver exit status 1"),
            # A grade file that claims success cannot hide a failing driver, and the reverse.
            ({"driver": 1}, "FAIL  S2.1 integration grade: 2 of 2 cases, 3 of 3 checks passed, driver exit status 1"),
            ({"grade_json": failed_grade}, "FAIL  S2.1 integration grade: 1 of 2 cases, 2 of 3 checks passed, driver "
                                           "exit status 0"),
            ({"grade_json": dict(GRADE_JSON, cases=0, cases_passed=0)}, "FAIL  S2.1 integration grade: 0 of 0 cases"),
            ({"grade": "NOTE  S2.1 two-prb-offset grid hashes only"},
             "FAIL  S2.1 integration driver produced no graded checks (see grant-fixture.log)"),
            ({"no_evidence": "1", "driver": 2},
             "FAIL  S2.1 integration driver produced no graded checks (see grant-fixture.log)"),
            ({"block": "another block"}, "FAIL  S2.1 a second reader recovers the record: the transport block hash is "
                                         f"{BLOCK_SHA256} by the reader, {hashlib.sha256(b'another block').hexdigest()} "
                                         f"by sha256sum, {BLOCK_SHA256} expected"),
            ({"recovered": dict(RECOVERED, tb_sha256="0" * 64)},
             f"FAIL  S2.1 a second reader recovers the record: the transport block hash is {'0' * 64} by the reader"),
            ({"show": 1}, "FAIL  S2.1 a second reader recovers the record (see recovered.log)\n"
                          "      grant: tb.bin differs from the size or hash in the manifest\n"),
            ({"verify": 1}, "FAIL  S2.1 adapter receipt (see adapter-verify.log)\n"
                            "      native_adapter: the adapter sources changed after the build\n"),
            ({"environment": 1}, "FAIL  S2.1 references: the pinned Sionna on the CPU and the Sprint 1.1 build, from one "
                                 "interpreter (environment.json) (see environment.log)\n"
                                 "      ValueError: installed Sionna file differs from the pinned tree\n"),
            ({"unit_status": 1, "unit": "FAIL: test_x (unit.test_grant_fixture.T.test_x)\nRan 170 tests in 1s\n\n"
                                        "FAILED (failures=1)"},
             "FAIL  unit tests (170 ran; see unit-tests.log)\n      FAIL: test_x (unit.test_grant_fixture.T.test_x)\n"),
            ({"unit": "Ran 0 tests in 0.000s\n\nOK"}, "FAIL  unit tests (none ran)"),
            ({"card": "Ran 0 tests in 0.000s\n\nOK"},
             "FAIL  S2.1 unit gate, run directly: python tests/unit/test_grant_fixture.py (none ran)"),
            ({"card_status": 1, "card": "ModuleNotFoundError: No module named 'ocudu_cuda_isac'"},
             "FAIL  S2.1 unit gate, run directly: python tests/unit/test_grant_fixture.py (0 ran; see "
             "grant-fixture-unit.log)"),
            ({"pins": 1}, "FAIL  S2.1 OCUDU source at the locked revision after the runs"),
        )
        for index, (settings, message) in enumerate(cases):
            self.calls.write_text("")
            result = self.gate(run_dir=self.root.parent / f"evidence {index}", **settings)
            with self.subTest(settings=settings):
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn(message, result.stdout)
                self.assertIn("Sprint 2.1: NOT ACCEPTED on this host", result.stdout)
                self.assertNotIn("Sprint 2.1: ACCEPTED", result.stdout)
                calls = self.calls.read_text()
                for later in ("native_adapter.py build", "tests/integration/grant_fixture.py", "episodes.grant show",
                              "check_pins.py --worktrees"):
                    self.assertIn(later, calls)

    def test_a_failed_adapter_build_is_one_failure_and_nothing_is_compared(self):
        result = self.gate(build=1)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        shown = result.stdout.splitlines()
        self.assertEqual([line for line in shown if line.startswith("FAIL")],
                         ["FAIL  S2.1 adapter build command (see adapter-build.log)"])
        self.assertIn("      native_adapter: the adapter build step exited with status 2", shown)
        self.assertIn("SKIP  S2.1 adapter receipt, integration driver and record recovery: no adapter", shown)
        calls = self.calls.read_text()
        for skipped in ("native_adapter.py verify", "tests/integration/grant_fixture.py", "episodes.grant show"):
            self.assertNotIn(skipped, calls)
        self.assertIn("check_pins.py --worktrees --only ocudu", calls)

    def test_gate_uses_only_a_build_that_sprint_1_1_accepted(self):
        boundary = "S2.1 native boundary"
        summary = self.s11 / "summary.txt"
        cases = (
            (S11_SUMMARY.replace("Sprint 1.1: ACCEPTED on this DGX Spark", "Sprint 1.1: NOT ACCEPTED on this host"),
             "an unaccepted run"),
            (S11_SUMMARY.replace("PASS  S1.1 native build command", "FAIL  S1.1 native build command"),
             "a summary that holds a failed gate"),
            (None, "a build made outside the gate"),
        )
        for index, (text, case) in enumerate(cases):
            summary.unlink(missing_ok=True)
            if text is not None:
                summary.write_text(text)
            result = self.gate(run_dir=self.root.parent / f"record {index}")
            with self.subTest(case=case):
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn(f"FAIL  {boundary}: no Sprint 1.1 ACCEPTED verdict in {summary}", result.stdout)
                self.assertIn("Run 'make accept-s1.1', then this gate again.", result.stdout)
                self.assertNotIn("Sprint 2.1: ACCEPTED", result.stdout)
        summary.write_text(S11_SUMMARY)
        self.receipt.write_text("not json")
        result = self.gate(run_dir=self.root.parent / "unreadable receipt")
        self.assertEqual(result.returncode, 1)
        self.assertIn(f"FAIL  {boundary}", result.stdout)

    def test_missing_build_is_one_failure_and_nothing_is_built(self):
        needs = ("FAIL  S2.1 needs the Sprint 1.1 build: isac/bin/pusch_e2e_pipeline_test is not a link into a "
                 "build with a receipt")
        link = self.bin / "pusch_e2e_pipeline_test"

        def unpublished():
            link.unlink()

        def copied():
            link.unlink()
            shutil.copy(self.build / TEST, link)

        for index, damage in enumerate((unpublished, copied, self.receipt.unlink)):
            damage()
            self.calls.write_text("")
            result = self.gate(run_dir=self.root.parent / f"no build {index}")
            with self.subTest(damage=damage.__name__):
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                shown = result.stdout.splitlines()
                self.assertEqual([line for line in shown if line.startswith("FAIL")], [needs])
                self.assertIn("      Run 'make accept-s1.1' on this host first.", shown)
                self.assertIn("SKIP  S2.1 reference check, adapter build, integration driver and record recovery: "
                              "no Sprint 1.1 build", shown)
                calls = self.calls.read_text()
                for skipped in ("native_adapter.py", "--with-native", "tests/integration/grant_fixture.py"):
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
                    (("--machine", "local"), new, "Usage"), (("--jobs", "8"), new, "Usage"))
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
        result = self.gate(run_dir=self.root / "results/runs/s21-clean")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS  committed tree", result.stdout)
        self.assertRegex(result.stdout, r"commit [0-9a-f]{40}")
        # The evidence the gate wrote is ignored, so a second run starts from the same clean tree.
        self.assertEqual(self.gate(run_dir=self.root / "results/runs/s21-again").returncode, 0)
        (self.root / "scripts/accept_s2_1.sh").write_text(SCRIPT.read_text() + "# local edit\n")
        result = self.gate(run_dir=self.root / "results/runs/s21-dirty")
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  committed tree", result.stdout)
        self.assertIn("accept_s2_1.sh", (self.root / "results/runs/s21-dirty/git-status.txt").read_text())


if __name__ == "__main__":
    unittest.main()
