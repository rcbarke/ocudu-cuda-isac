"""Check the Sprint 2.4 gate runner against tool doubles. No compiler, source, build, NumPy or GPU is needed."""

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

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/accept_s2_4.sh"
TEST = "tests/integrationtests/phy/upper/channel_processors/pusch_e2e_pipeline_test"
BLOCK = b"original transport block of the demonstration grant"
BLOCK_SHA256 = hashlib.sha256(BLOCK).hexdigest()
GRADE = """PASS  S2.4 unity native run: unity/slot-grants: 6 grids on 4 receive plane(s), cpu 6 of 6 decoded, cuda 6 of 6 decoded
PASS  S2.4 direct decodes: high-SNR direct path: cpu 24 of 24 decoded, cuda 24 of 24 decoded
PASS  S2.4 reference-pair decodes: the physical pair at 30 dB: cpu 48 of 48 decoded, cuda 48 of 48 decoded
PASS  S2.4 payload decodes: cpu 48 of 48 decoded, cuda 48 of 48 decoded
PASS  S2.4 bad-input never acknowledged: 0 of 24 decodes of bad input report a CRC pass
PASS  S2.4 backends executed path: cpu: 140 on the host path; cuda: 138 on the resident path, 2 on the host path
NOTE  S2.4 backends PUSCH service time of cuda, not graded: mean 240 us
replay_pusch: 8 of 8 cases passed, 6 of 6 checks"""
GRADE_JSON = {"judgment": "pass", "hardware_gate": "closed", "cases": 8, "cases_passed": 8, "checks": 6,
              "checks_passed": 6, "checks_failed": 0}
REHEARSAL_JSON = dict(GRADE_JSON, judgment="rehearsal",
                      hardware_gate="open: the CUDA processor did not execute, this is a host rehearsal")
RECEIPT = {"executables": {"grant_fixture": {"sha256": "0123456789abcdef" * 4},
                           "grid_apply": {"sha256": "fedcba9876543210" * 4},
                           "replay_pusch": {"sha256": "0f1e2d3c4b5a6978" * 4}},
           "tests_run": 34, "compiler_warnings": 0, "pin": {"cuda": True, "archives": 50}}
HOST_RECEIPT = dict(RECEIPT, pin={"cuda": False, "archives": 50})
CTEST = "\n".join([f" {n + 20:2d}/34 Test #{n + 20}: dispatch_core_case_{n} ..............   Passed    0.00 sec"
                   for n in range(12)]
                  + ["34/34 Test #34: replay_pusch_host_decode ..................   Passed    0.21 sec",
                     "100% tests passed, 0 tests failed out of 34"])
RECOVERED = {"verified_records": 96, "key": "rz-22963ee9d985.0000/cuda", "slot": 7, "grant_id": "demo-51prb-4rx",
             "outcome": "decoded", "executed": "resident", "tb_bytes": 1217, "decoded_file": "0000.cuda.bin",
             "decoded_sha256": BLOCK_SHA256, "ingress_sha256": "d7b97254a5990ca7" * 4}
S22_SUMMARY = """Sprint 2.2 acceptance on spark-ecf8, 2026-11-21T00:44:33Z
PASS  committed tree
Sprint 2.2: ACCEPTED on this DGX Spark. The uplink grant fixture, the matched pair on it and the episode set are archived.
"""
S23_SUMMARY = """Sprint 2.3 acceptance on spark-ecf8, 2026-11-21T03:24:01Z
PASS  committed tree
Sprint 2.3: ACCEPTED on this DGX Spark. The native channel application with the grid gain fills the pinned resource grid.
"""
UNIT = "Ran 320 tests in 120.000s\n\nOK"
CARD = "Ran 21 tests in 6.500s\n\nOK"
# The interpreter double answers each gate from FAKE_* variables and logs the call. The builder
# double leaves the CTest log of the adapter build. The driver double leaves its grade, the plan of
# the pair with the original block beside it, and the decoded bytes of one decode.
PYTHON_DOUBLE = """#!/bin/sh
[ "$1" = "-c" ] && exec {real} "$@"
echo "$*" >> "$FAKE_LOG"
case "$*" in
    "-m pip freeze --all") echo "example==1.0" ;;
    *"-m unittest discover"*) printf '%s\\n' "$FAKE_UNIT" >&2; exit "${{FAKE_UNIT_STATUS:-0}}" ;;
    "tests/unit/test_replay_pusch.py -v") printf '%s\\n' "$FAKE_CARD" >&2; exit "${{FAKE_CARD_STATUS:-0}}" ;;
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
    "tests/integration/replay_pusch.py --config configs/integration/replay_pusch.json --s22-run "*)
        for out; do :; done
        echo "s22 run: $5" >> "$FAKE_LOG"
        echo "s23 run: $7" >> "$FAKE_LOG"
        echo "Run artifacts: $out"
        [ -z "${{FAKE_NO_EVIDENCE:-}}" ] || exit "${{FAKE_DRIVER:-0}}"
        pair="$out/reference_pair/run"
        mkdir -p "$pair/native/radio"
        printf '%s' "$FAKE_BLOCK" > "$pair/tb.bin"
        printf '%s' "${{FAKE_DECODED:-$FAKE_BLOCK}}" > "$pair/native/radio/0000.cuda.bin"
        printf '{{"grants": [{{"grant_id": "demo-51prb-4rx", "tb": {{"path": "%s"}}}}]}}\\n' "$pair/tb.bin" > "$pair/plan.json"
        printf '%s\\n' "$FAKE_GRADE" > "$out/grade.txt"
        printf '%s\\n' "$FAKE_GRADE_JSON" > "$out/grade.json"
        exit "${{FAKE_DRIVER:-0}}" ;;
    "-m ocudu_cuda_isac.episodes.replay_pusch show "*)
        [ "${{FAKE_SHOW:-0}}" = 0 ] || {{ echo "replay_pusch: the decoded bytes of radio record 3 differ from their hash or size" >&2; exit 2; }}
        printf '%s\\n' "$FAKE_RECOVERED" ;;
    *"check_pins.py --worktrees") echo "ext/ocudu 27e84cb3 (Git worktree)"; exit "${{FAKE_PINS:-0}}" ;;
    *) echo "interpreter double: unexpected $*" >&2; exit 64 ;;
esac
"""


@unittest.skipUnless(shutil.which("bash") and shutil.which("sha256sum") and shutil.which("cmp"),
                     "the gate runner is a Bash script")
class SprintTwoReceiverGateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / "host checkout"
        self.bin = self.root / "isac/bin"
        (self.root / "scripts").mkdir(parents=True)
        self.bin.mkdir(parents=True)
        shutil.copy(SCRIPT, self.root / "scripts/accept_s2_4.sh")
        python = self.bin / "python"
        python.write_text(PYTHON_DOUBLE.format(real=shlex.quote(sys.executable)))
        python.chmod(0o755)
        # The build Sprint 1.1 published, the run in which Sprint 2.2 archived the stimulus and the
        # run in which Sprint 2.3 applied it.
        self.build = self.root / "build/ocudu-spark"
        (self.build / TEST).parent.mkdir(parents=True)
        (self.build / TEST).write_text("#!/bin/sh\n")
        (self.bin / "pusch_e2e_pipeline_test").symlink_to(self.build / TEST)
        (self.build / "isac-native.json").write_text(json.dumps({"target": "all"}))
        self.s22 = self.stimulus_run("s22-spark-ecf8-20261121T004433Z")
        self.s23 = self.grid_run("s23-spark-ecf8-20261121T032401Z", self.s22)
        self.calls = self.root.parent / "calls.log"
        self.run_dir = self.root.parent / "evidence"

    def stimulus_run(self, name, summary=S22_SUMMARY, pair=True):
        run = self.root / "results/runs" / name
        run.mkdir(parents=True)
        if pair:
            (run / "episode_pair/reference_pair/store").mkdir(parents=True)
            (run / "episode_pair/reference_pair/store/manifest.json").write_text("{}")
        (run / "summary.txt").write_text(summary)
        return run

    def grid_run(self, name, s22, summary=S23_SUMMARY, grids=True):
        run = self.root / "results/runs" / name
        (run / "grid_apply/reference_pair/native").mkdir(parents=True)
        if grids:
            (run / "grid_apply/reference_pair/native/native.json").write_text("{}")
        (run / "grid_apply/manifest.json").write_text(json.dumps({"s22_run": str(s22)}))
        (run / "summary.txt").write_text(summary)
        return run

    def gate(self, *arguments, run_dir=None, **settings):
        env = {key: value for key, value in os.environ.items() if not key.startswith("FAKE_")}
        env.update(FAKE_LOG=str(self.calls), FAKE_UNIT=UNIT, FAKE_CARD=CARD, FAKE_GRADE=GRADE,
                   FAKE_GRADE_JSON=json.dumps(GRADE_JSON), FAKE_RECEIPT=json.dumps(RECEIPT), FAKE_CTEST=CTEST,
                   FAKE_RECOVERED=json.dumps(RECOVERED), FAKE_BLOCK=BLOCK.decode("ascii"))
        env.update({f"FAKE_{key.upper()}": value if isinstance(value, str) else json.dumps(value)
                    for key, value in settings.items()})
        command = [shutil.which("bash"), str(self.root / "scripts/accept_s2_4.sh"),
                   "--run-dir", str(run_dir or self.run_dir), *arguments]
        return subprocess.run(command, env=env, capture_output=True, text=True, timeout=120, check=False)

    def summary(self):
        return (self.run_dir / "summary.txt").read_text()

    def test_all_gates_pass_on_the_accepted_grids_and_stimulus(self):
        result = self.gate()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("FAIL", result.stdout)
        self.assertNotIn("SKIP", result.stdout)
        self.assertNotIn("REHEARS", result.stdout)
        summary = self.summary()
        self.assertEqual(summary, result.stdout)
        for line in ("PASS  unit tests (320 ran, 0 skipped)",
                     "PASS  S2.4 unit gate, run directly: python tests/unit/test_replay_pusch.py (21 ran, 0 skipped)",
                     "PASS  S2.4 grids: the accepted Sprint 2.3 run s23-spark-ecf8-20261121T032401Z holds the stored grids",
                     "PASS  S2.4 stimulus: the accepted Sprint 2.2 run s22-spark-ecf8-20261121T004433Z holds the pair and "
                     "the slot grants, and it is the stimulus that s23-spark-ecf8-20261121T032401Z applied",
                     "PASS  S2.4 references: the Sprint 1.1 build, from the project interpreter (environment.json)",
                     "PASS  S2.4 adapter: built against the Sprint 1.1 build and verified, replay_pusch 0f1e2d3c4b5a beside "
                     "grid_apply fedcba987654 and grant_fixture 0123456789ab, with ENABLE_CUDA, 34 adapter tests, 0 "
                     "compiler warnings",
                     "PASS  S2.4 native unit cases: 12 cases of the host-free dispatch core and one decode through the "
                     "pinned host PUSCH processor passed in the adapter build",
                     "PASS  S2.4 hardware: the adapter is linked against the CUDA build, so the CUDA processor can execute",
                     "Decoding the stimulus of s22-spark-ecf8-20261121T004433Z on the grids of "
                     "s23-spark-ecf8-20261121T032401Z with configs/integration/replay_pusch.json.",
                     "PASS  S2.4 reference-pair decodes: the physical pair at 30 dB",
                     "NOTE  S2.4 backends PUSCH service time of cuda, not graded",
                     "PASS  S2.4 integration grade: 8 of 8 cases, 6 of 6 checks, judgment pass, hardware gate closed",
                     "PASS  S2.4 a second reader recovers a decode: 96 radio records verified; the decode "
                     "rz-22963ee9d985.0000/cuda (slot 7, grant demo-51prb-4rx) is decoded on the resident path with 1217 "
                     "bytes from the grid d7b97254a5990ca7; the decoded bytes are the original transport block byte for "
                     f"byte by cmp, SHA-256 {BLOCK_SHA256[:16]} by the reader and by sha256sum",
                     "PASS  S2.4 sources at the locked revisions after the runs",
                     "Sprint 2.4: ACCEPTED on this DGX Spark. The host and the CUDA PUSCH processor of the pinned source "
                     "decode the replayed grids and return the original transport blocks (S2.2 run "
                     "s22-spark-ecf8-20261121T004433Z, S2.3 run s23-spark-ecf8-20261121T032401Z). Next: 'make accept-s2' "
                     "closes Sprint 2, then S3.1 pilot_ls."):
            self.assertIn(line, summary)
        calls = self.calls.read_text()
        self.assertIn(f"s22 run: {self.s22}", calls)
        self.assertIn(f"s23 run: {self.s23}", calls)
        self.assertIn(f"scripts/native_adapter.py build --ocudu-build-dir {self.build}", calls)
        self.assertIn("scripts/verify_environment.py --profile core --with-native", calls)
        self.assertIn("-m ocudu_cuda_isac.episodes.replay_pusch show", calls)
        self.assertNotIn("--grant-store", calls)
        self.assertNotIn("--host-rehearsal", calls)
        self.assertEqual(summary.count("PASS  "), 17)

    def test_the_newest_accepted_sprint_2_3_run_is_used_with_the_stimulus_it_applied(self):
        self.grid_run("s23-spark-ecf8-20261121T090000Z", self.s22, S23_SUMMARY.replace("PASS  committed", "FAIL  committed"))
        self.grid_run("s23-spark-ecf8-20261121T100000Z", self.s22, "Sprint 2.3: NOT ACCEPTED on this host\n")
        self.grid_run("s23-spark-ecf8-20261121T110000Z", self.s22, grids=False)  # a verdict without stored grids
        # A newer Sprint 2.2 run exists, and the newest accepted Sprint 2.3 run applied an older one.
        self.stimulus_run("s22-spark-ecf8-20261121T050000Z")
        older = self.stimulus_run("s22-spark-ecf8-20261121T020000Z")
        newest = self.grid_run("s23-spark-ecf8-20261121T060000Z", older)
        result = self.gate()
        self.assertEqual(result.returncode, 0, result.stdout)
        calls = self.calls.read_text()
        self.assertIn(f"s22 run: {older}", calls)
        self.assertIn(f"s23 run: {newest}", calls)
        self.assertIn("(S2.2 run s22-spark-ecf8-20261121T020000Z, S2.3 run s23-spark-ecf8-20261121T060000Z)", self.summary())

    def test_named_runs_a_development_host_and_named_grant_stores(self):
        stimulus = self.stimulus_run("s22-devhost-20261119T000000Z", S22_SUMMARY.replace("DGX Spark", "host"))
        grids = self.grid_run("s23-devhost-20261119T010000Z", "/moved/away/s22-devhost-20261119T000000Z",
                              S23_SUMMARY.replace("DGX Spark", "host"))
        # The runs were moved: the path the Sprint 2.3 run recorded is gone, and the stimulus is named.
        result = self.gate("--s23-run", str(grids), "--s22-run", str(stimulus),
                           "--grant-store", "/moved/s21/grant_fixture", "--grant-store", "/moved/s22 run/grant_fixture")
        self.assertEqual(result.returncode, 0, result.stdout)
        calls = self.calls.read_text()
        self.assertIn(f"s22 run: {stimulus}", calls)
        self.assertIn(f"s23 run: {grids}", calls)
        self.assertIn("--grant-store /moved/s21/grant_fixture --grant-store /moved/s22 run/grant_fixture --output-dir", calls)
        self.assertIn("Sprint 2.4: ACCEPTED on this host.", result.stdout)
        # Without the name, the recorded path decides, and it is gone.
        lost = self.gate("--s23-run", str(grids), run_dir=self.root.parent / "evidence-lost")
        self.assertEqual(lost.returncode, 1)
        self.assertIn("FAIL  S2.4 stimulus: no Sprint 2.2 ACCEPTED verdict with a stored pair in "
                      "/moved/away/s22-devhost-20261119T000000Z", lost.stdout)
        self.assertIn("If the runs were moved, name the directory with --s22-run.", lost.stdout)

    def test_another_stimulus_than_the_one_the_grids_came_from_is_refused(self):
        other = self.stimulus_run("s22-spark-ecf8-20261121T050000Z")
        result = self.gate("--s22-run", str(other))
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S2.4 stimulus: the Sprint 2.3 run s23-spark-ecf8-20261121T032401Z applied the stimulus of "
                      "s22-spark-ecf8-20261121T004433Z, and --s22-run names s22-spark-ecf8-20261121T050000Z", result.stdout)
        self.assertIn("SKIP  S2.4 reference check, adapter build, integration driver and decode recovery: no stimulus",
                      result.stdout)
        self.assertNotIn("tests/integration/replay_pusch.py", self.calls.read_text())
        # A Sprint 2.3 run that names no stimulus, and a stimulus without an accepted verdict.
        (self.s23 / "grid_apply/manifest.json").write_text("{}")
        result = self.gate(run_dir=self.root.parent / "evidence-unnamed")
        self.assertIn("does not name the Sprint 2.2 run it applied", result.stdout)
        (self.s23 / "grid_apply/manifest.json").write_text(json.dumps({"s22_run": str(self.s22)}))
        (self.s22 / "summary.txt").write_text("Sprint 2.2: NOT ACCEPTED on this host\n")
        result = self.gate(run_dir=self.root.parent / "evidence-unaccepted")
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S2.4 stimulus: no Sprint 2.2 ACCEPTED verdict with a stored pair in", result.stdout)

    def test_without_accepted_grids_nothing_is_decoded(self):
        shutil.rmtree(self.s23)
        result = self.gate()
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S2.4 grids: no Sprint 2.3 ACCEPTED verdict with stored grids in results/runs/s23-*", result.stdout)
        self.assertIn("Run 'make accept-s2.3' on this host first", result.stdout)
        self.assertIn("SKIP  S2.4 reference check, adapter build, integration driver and decode recovery", result.stdout)
        self.assertIn("Sprint 2.4: NOT ACCEPTED on this host", result.stdout)
        self.assertNotIn("tests/integration/replay_pusch.py", self.calls.read_text())

    def test_a_missing_sprint_1_1_build_is_named(self):
        (self.bin / "pusch_e2e_pipeline_test").unlink()
        result = self.gate()
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S2.4 needs the Sprint 1.1 build", result.stdout)
        self.assertNotIn("native_adapter.py", self.calls.read_text())

    def test_a_build_without_cuda_cannot_be_accepted(self):
        # The driver would fail its own check too. The gate says why before it runs.
        failed_grade = dict(GRADE_JSON, judgment="fail", hardware_gate="open", cases_passed=7, checks_passed=5, checks_failed=1)
        result = self.gate(receipt=HOST_RECEIPT, grade_json=failed_grade, driver="1",
                           grade=GRADE.replace("PASS  S2.4 backends executed path", "FAIL  S2.4 backends executed path"))
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S2.4 hardware: the accepted build has no CUDA, so the CUDA processor cannot execute here.",
                      result.stdout)
        self.assertIn("Rehearse with --host-rehearsal, accept on the DGX Spark.", result.stdout)
        self.assertIn("FAIL  S2.4 backends executed path", result.stdout)
        self.assertIn("Sprint 2.4: NOT ACCEPTED on this host", result.stdout)
        # A driver that reports a pass on such a build still cannot close the gate.
        result = self.gate(receipt=HOST_RECEIPT, run_dir=self.root.parent / "evidence-claimed")
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("Sprint 2.4: ACCEPTED", result.stdout)

    def test_a_host_rehearsal_is_reported_and_never_accepted(self):
        result = self.gate("--host-rehearsal", receipt=HOST_RECEIPT, grade_json=REHEARSAL_JSON,
                           recovered=dict(RECOVERED, executed="host"))
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertNotIn("Sprint 2.4: ACCEPTED", result.stdout)
        self.assertNotIn("FAIL", result.stdout)
        for line in ("HOST REHEARSAL: the host path is graded on a build without CUDA. This run cannot accept the sprint.",
                     "NOTE  S2.4 hardware: host rehearsal on an adapter built for the host path. The CUDA decodes of this "
                     "gate stay open.",
                     "PASS  S2.4 integration grade: 8 of 8 cases, 6 of 6 checks, judgment rehearsal, hardware gate open",
                     "is decoded on the host path with 1217 bytes",
                     "Sprint 2.4: REHEARSED on this host, NOT ACCEPTED. The host path returned the original transport "
                     "blocks (S2.2 run s22-spark-ecf8-20261121T004433Z, S2.3 run s23-spark-ecf8-20261121T032401Z). The "
                     "CUDA decodes are open: run 'make accept-s2.4' on the DGX Spark."):
            self.assertIn(line, result.stdout)
        self.assertIn("--host-rehearsal --output-dir", self.calls.read_text())
        # A rehearsal whose driver reports a pass was not a rehearsal, and a failed one is not accepted either.
        result = self.gate("--host-rehearsal", receipt=HOST_RECEIPT, run_dir=self.root.parent / "evidence-pass")
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S2.4 integration grade: 8 of 8 cases, 6 of 6 checks, judgment pass, hardware gate closed, "
                      "driver exit status 0", result.stdout)
        self.assertIn("Sprint 2.4: NOT ACCEPTED on this host", result.stdout)
        # A grade that says rehearsal cannot close the gate that was run without the option.
        result = self.gate(grade_json=REHEARSAL_JSON, run_dir=self.root.parent / "evidence-claim")
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S2.4 integration grade: 8 of 8 cases, 6 of 6 checks, judgment rehearsal, hardware gate open",
                      result.stdout)

    def test_failures_of_single_gates(self):
        cases = (
            (dict(unit="Ran 320 tests in 4.0s\n\nFAILED (failures=1)", unit_status="1"), "FAIL  unit tests (320 ran"),
            (dict(card="Ran 0 tests in 0.0s\n\nOK"), "FAIL  S2.4 unit gate, run directly: python "
                                                    "tests/unit/test_replay_pusch.py (none ran)"),
            # The native unit cases are part of the unit gate: a run that skipped them cannot close it.
            (dict(card="Ran 21 tests in 0.5s\n\nOK (skipped=3)"), "FAIL  S2.4 unit gate, run directly: python "
             "tests/unit/test_replay_pusch.py (21 ran, 3 skipped: the native unit cases need cmake"),
            (dict(environment="1"), "FAIL  S2.4 references: the Sprint 1.1 build"),
            (dict(build="1"), "FAIL  S2.4 adapter: build or verification failed"),
            (dict(verify="1"), "FAIL  S2.4 adapter: build or verification failed"),
            (dict(receipt=dict(RECEIPT, executables={key: value for key, value in RECEIPT["executables"].items()
                                                     if key != "replay_pusch"})),
             "FAIL  S2.4 adapter: the receipt names no replay_pusch"),
            (dict(ctest=CTEST.replace("dispatch_core_case_11 ..............   Passed", "dispatch_core_case_11 ..............***Failed")),
             "FAIL  S2.4 native unit cases: 11 cases of the dispatch core passed"),
            (dict(ctest=CTEST.replace("replay_pusch_host_decode ..................   Passed",
                                      "replay_pusch_host_decode ..................***Failed")),
             "FAIL  S2.4 native unit cases: 12 cases of the dispatch core passed and the host decode self-test did not"),
            (dict(no_ctest="1"), "FAIL  S2.4 native unit cases: 0 cases of the dispatch core passed"),
            (dict(grade=GRADE.replace("PASS  S2.4 reference-pair decodes", "FAIL  S2.4 reference-pair decodes"),
                  grade_json=dict(GRADE_JSON, judgment="fail", hardware_gate="open", cases_passed=7, checks_passed=5,
                                  checks_failed=1), driver="1"),
             "FAIL  S2.4 integration grade: 7 of 8 cases, 5 of 6 checks, judgment fail, hardware gate open, driver exit "
             "status 1"),
            (dict(no_evidence="1", driver="2"), "FAIL  S2.4 integration driver produced no graded checks"),
            (dict(grade_json=dict(GRADE_JSON, cases=0, cases_passed=0)), "FAIL  S2.4 integration grade: 0 of 0 cases"),
            (dict(grade_json=dict(GRADE_JSON, hardware_gate="open")),
             "FAIL  S2.4 integration grade: 8 of 8 cases, 6 of 6 checks, judgment pass, hardware gate open"),
            (dict(show="1"), "FAIL  S2.4 a second reader recovers a decode (see recovered.log)"),
            # The decoded bytes differ from the original block: by one byte, and with the reader's digest of the original.
            (dict(decoded=BLOCK.decode("ascii")[:-1] + "X"), "FAIL  S2.4 a second reader recovers a decode: the decoded bytes"),
            (dict(decoded="other bytes", recovered=dict(RECOVERED, decoded_sha256=hashlib.sha256(b"other bytes").hexdigest())),
             "or cmp finds them different from the original transport block"),
            (dict(pins="1"), "FAIL  S2.4 sources at the locked revisions after the runs"),
        )
        for number, (settings, line) in enumerate(cases):
            run_dir = self.root.parent / f"evidence-{number}"
            result = self.gate(run_dir=run_dir, **settings)
            with self.subTest(line=line):
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn(line, result.stdout)
                self.assertIn("Sprint 2.4: NOT ACCEPTED on this host", result.stdout)
                self.assertNotIn("Sprint 2.4: ACCEPTED", result.stdout)
                self.assertNotIn("REHEARSED", result.stdout)
        skipped = self.gate(run_dir=self.root.parent / "evidence-build", build="1")
        self.assertIn("SKIP  S2.4 native unit cases, integration driver and decode recovery: no adapter", skipped.stdout)
        # A skip in the portable suite is reported and does not fail it: only the unit gate must be complete.
        portable = self.gate(run_dir=self.root.parent / "evidence-skip", unit="Ran 320 tests in 9.0s\n\nOK (skipped=12)")
        self.assertEqual(portable.returncode, 0, portable.stdout)
        self.assertIn("PASS  unit tests (320 ran, 12 skipped)", portable.stdout)

    def test_a_failed_check_of_the_driver_fails_the_gate_even_with_exit_status_zero(self):
        result = self.gate(grade=GRADE.replace("PASS  S2.4 bad-input never", "FAIL  S2.4 bad-input never"))
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S2.4 bad-input never acknowledged", result.stdout)
        self.assertNotIn("Sprint 2.4: ACCEPTED", result.stdout)

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
        for arguments in (("--unknown",), ("--s22-run",), ("--s23-run",), ("--grant-store",)):
            result = self.gate(*arguments, run_dir=self.root.parent / "evidence-usage")
            self.assertEqual(result.returncode, 2)
            self.assertIn("Usage: bash scripts/accept_s2_4.sh", result.stderr)
        (self.bin / "python").unlink()
        missing = self.gate(run_dir=self.root.parent / "evidence-python")
        self.assertEqual(missing.returncode, 2)
        self.assertIn("Run 'make env' first", missing.stderr)


if __name__ == "__main__":
    unittest.main()
