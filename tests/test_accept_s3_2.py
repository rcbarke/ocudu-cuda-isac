"""Check the Sprint 3.2 gate runner against tool doubles. No compiler, source, build, NumPy or GPU is needed."""

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

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/accept_s3_2.sh"
TEST = "tests/integrationtests/phy/upper/channel_processors/pusch_e2e_pipeline_test"
PAYLOAD = b"the estimate of the first opportunity, complex64"
HEADER = b"a record header and its host block and its mask: "
PAYLOAD_SHA256 = hashlib.sha256(PAYLOAD).hexdigest()
GRADE = """PASS  S3.2 control accepted run: the Sprint 3.1 driver graded pilot_ls as pass
PASS  S3.2 audit accepted estimates: 48 of 48 payloads and validity masks have the SHA-256 that the accepted Sprint 3.1 run recorded
PASS  S3.2 timed no checksum: between the first fill and the drain of the ring the hash of the core computed 0 digests
PASS  S3.2 timed no system call: the producer made 0 system calls between an admission and its publication store
PASS  S3.2 timed tap cost: service time of the tap from the read of the grid to the publication store
PASS  S3.2 slow-consumer overload: 21 observations found no credit and were dropped at once and counted
PASS  S3.2 terminated-consumer new epoch: 2 consumers attached
PASS  S3.2 backends executed path: 336 of 336 decodes requested from the CUDA processor ran its resident path
NOTE  S3.2 timed delay of the spinning consumer from the publication store to its read, not graded: median 0.4 us
snapshot_ring: 10 of 10 cases passed, 8 of 8 checks"""
GRADE_JSON = {"judgment": "pass", "hardware_gate": "closed", "cases": 10, "cases_passed": 10, "checks": 8,
              "checks_passed": 8, "checks_failed": 0}
REHEARSAL_JSON = dict(GRADE_JSON, judgment="rehearsal",
                      hardware_gate="open: the CUDA processor did not execute, this is a host rehearsal")
ACCEPTED = {"grant_fixture": {"sha256": "0123456789abcdef" * 4}, "grid_apply": {"sha256": "fedcba9876543210" * 4},
            "replay_pusch": {"sha256": "0f1e2d3c4b5a6978" * 4}, "pilot_ls": {"sha256": "a1b2c3d4e5f60718" * 4}}
RECEIPT = {"executables": {"grant_fixture": ACCEPTED["grant_fixture"], "grid_apply": {"sha256": "1111222233334444" * 4},
                           "replay_pusch": {"sha256": "5555666677778888" * 4}, "pilot_ls": {"sha256": "9999aaaabbbbcccc" * 4},
                           "obs_tail": {"sha256": "ddddeeeeffff0000" * 4}, "obs_feed": {"sha256": "1234123412341234" * 4}},
           "tests_run": 75, "compiler_warnings": 0, "pin": {"cuda": True, "archives": 50}}
HOST_RECEIPT = dict(RECEIPT, pin={"cuda": False, "archives": 50})
CTEST = "\n".join([f" {n + 1:2d}/75 Test #{n + 1}: obs_ring_case_{n} ..............   Passed    0.00 sec" for n in range(16)]
                  + [f" {n + 20:2d}/75 Test #{n + 20}: obs_publisher_case_{n} ..............   Passed    0.00 sec"
                     for n in range(6)]
                  + [f"{n + 30}/75 Test #{n + 30}: {name} .........................   Passed    0.25 sec"
                     for n, name in enumerate(("replay_core_evidence", "dispatch_core_timed", "observer_core_timed",
                                               "pilot_ls_host_tap"))]
                  + ["100% tests passed, 0 tests failed out of 75"])
RECOVERED = {"records": 48, "reader": "native", "wait": "spin", "file": "0001-00000000.obs",
             "keys": {"epoch": 1, "sequence": 0}, "what": {"stage": "O2", "source": "pusch_dmrs",
                                                           "layout": "ocudu_o2_comb_c64", "evidence": "timed"},
             "shape": {"shape": [4, 2, 306]}, "validity": {"valid": 2448},
             "payload": {"bytes": len(PAYLOAD), "digest_kind": None},
             "staged_host": {"opportunity_sequence": 0, "grid_index": 0, "grid_generation": 1}}
ACCOUNT = {"mode": "timed", "hash_in_window": {"digests": 0, "updates": 0, "bytes": 0}}
S31_SUMMARY = """Sprint 3.1 acceptance on spark-ecf8, 2026-11-21T17:45:11Z
PASS  committed tree
Sprint 3.1: ACCEPTED on this DGX Spark. The observer exports the pilot least squares of the PUSCH DM-RS.
"""
S31_REHEARSED = """Sprint 3.1 acceptance on devhost, 2026-11-21T17:45:11Z
HOST REHEARSAL: the host path is graded on a build without CUDA. This run cannot accept the sprint.
Sprint 3.1: REHEARSED on this host, NOT ACCEPTED. The observer exported the pilot least squares.
"""
UNIT = "Ran 470 tests in 300.000s\n\nOK"
CARD = "Ran 20 tests in 12.500s\n\nOK"
# The interpreter double answers each gate from FAKE_* variables and logs the call. The builder
# double leaves the CTest log of the adapter build. The driver double leaves its grade, the record
# that the consumer of the timed run kept and the payload file of the audited run.
PYTHON_DOUBLE = """#!/bin/sh
[ "$1" = "-c" ] && exec {real} "$@"
echo "$*" >> "$FAKE_LOG"
case "$*" in
    "-m pip freeze --all") echo "example==1.0" ;;
    *"-m unittest discover"*) printf '%s\\n' "$FAKE_UNIT" >&2; exit "${{FAKE_UNIT_STATUS:-0}}" ;;
    "tests/unit/test_snapshot_ring.py -v") printf '%s\\n' "$FAKE_CARD" >&2; exit "${{FAKE_CARD_STATUS:-0}}" ;;
    *"--profile core --with-native"*) exit "${{FAKE_ENVIRONMENT:-0}}" ;;
    "scripts/native_adapter.py build --ocudu-build-dir "*)
        echo "Adapter build: build/ocudu-adapter"
        mkdir -p "$6"
        [ -n "${{FAKE_NO_CTEST:-}}" ] || printf '%s\\n' "$FAKE_CTEST" > "$6/adapter-test.log"
        [ "${{FAKE_BUILD:-0}}" = 0 ] || echo "native_adapter: the adapter build step exited with status 2" >&2
        exit "${{FAKE_BUILD:-0}}" ;;
    "scripts/native_adapter.py verify")
        [ "${{FAKE_VERIFY:-0}}" = 0 ] || {{ echo "native_adapter: the adapter sources changed after the build" >&2; exit 1; }}
        printf '%s\\n' "$FAKE_RECEIPT" ;;
    "tests/integration/snapshot_ring.py --config configs/integration/snapshot_ring.json --s31-run "*)
        for out; do :; done
        echo "s31 run: $5" >> "$FAKE_LOG"
        echo "Run artifacts: $out"
        [ -z "${{FAKE_NO_EVIDENCE:-}}" ] || exit "${{FAKE_DRIVER:-0}}"
        mkdir -p "$out/timed/run/consumer-0" "$out/timed/run/native" "$out/audit/run/native/observer"
        printf '%s%s' "$FAKE_HEADER" "${{FAKE_KEPT:-$FAKE_ESTIMATE}}" > "$out/timed/run/consumer-0/0001-00000000.obs"
        printf '%s' "${{FAKE_AUDITED:-$FAKE_ESTIMATE}}" > "$out/audit/run/native/observer/0000.o2.c64"
        printf '%s\\n' "$FAKE_GRADE" > "$out/grade.txt"
        printf '%s\\n' "$FAKE_GRADE_JSON" > "$out/grade.json"
        exit "${{FAKE_DRIVER:-0}}" ;;
    "-m ocudu_cuda_isac.sensing.obs show "*)
        [ "${{FAKE_SHOW:-0}}" = 0 ] || {{ echo "obs: the record 0001-00000000.obs differs from its line of the report" >&2; exit 2; }}
        printf '%s\\n' "$FAKE_RECOVERED" ;;
    "-m ocudu_cuda_isac.episodes.snapshot_ring show "*) printf '%s\\n' "$FAKE_ACCOUNT" ;;
    *"check_pins.py --worktrees") echo "ext/ocudu 27e84cb3 (Git worktree)"; exit "${{FAKE_PINS:-0}}" ;;
    *) echo "interpreter double: unexpected $*" >&2; exit 64 ;;
esac
"""


@unittest.skipUnless(shutil.which("bash") and shutil.which("sha256sum"), "the gate runner is a Bash script")
class SprintThreeRingGateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / "host checkout"
        self.bin = self.root / "isac/bin"
        (self.root / "scripts").mkdir(parents=True)
        self.bin.mkdir(parents=True)
        shutil.copy(SCRIPT, self.root / "scripts/accept_s3_2.sh")
        python = self.bin / "python"
        python.write_text(PYTHON_DOUBLE.format(real=shlex.quote(sys.executable)))
        python.chmod(0o755)
        # The build Sprint 1.1 published and the run that Sprint 3.1 accepted.
        self.build = self.root / "build/ocudu-spark"
        (self.build / TEST).parent.mkdir(parents=True)
        (self.build / TEST).write_text("#!/bin/sh\n")
        (self.bin / "pusch_e2e_pipeline_test").symlink_to(self.build / TEST)
        (self.build / "isac-native.json").write_text(json.dumps({"target": "all"}))
        self.s31 = self.control_run("s31-spark-ecf8-20261121T174511Z")
        self.calls = self.root.parent / "calls.log"
        self.run_dir = self.root.parent / "evidence"

    def control_run(self, name, summary=S31_SUMMARY, plans=True, executables=None, payload=PAYLOAD):
        run = self.root / "results/runs" / name
        pair = run / "pilot_ls/reference_pair/run"
        (pair / "native/observer").mkdir(parents=True)
        (run / "pilot_ls/grade.json").write_text("{}")
        if plans:
            (pair / "plan.json").write_text("{}")
        (pair / "native/observer/0000.o2.c64").write_bytes(payload)
        (run / "adapter-receipt.json").write_text(json.dumps({"executables": executables or ACCEPTED}))
        (run / "summary.txt").write_text(summary)
        return run

    def gate(self, *arguments, run_dir=None, **settings):
        env = {key: value for key, value in os.environ.items() if not key.startswith("FAKE_")}
        env.update(FAKE_LOG=str(self.calls), FAKE_UNIT=UNIT, FAKE_CARD=CARD, FAKE_GRADE=GRADE,
                   FAKE_GRADE_JSON=json.dumps(GRADE_JSON), FAKE_RECEIPT=json.dumps(RECEIPT), FAKE_CTEST=CTEST,
                   FAKE_RECOVERED=json.dumps(RECOVERED), FAKE_ACCOUNT=json.dumps(ACCOUNT),
                   FAKE_ESTIMATE=PAYLOAD.decode("ascii"), FAKE_HEADER=HEADER.decode("ascii"))
        env.update({f"FAKE_{key.upper()}": value if isinstance(value, str) else json.dumps(value)
                    for key, value in settings.items()})
        command = [shutil.which("bash"), str(self.root / "scripts/accept_s3_2.sh"),
                   "--run-dir", str(run_dir or self.run_dir), *arguments]
        return subprocess.run(command, env=env, capture_output=True, text=True, timeout=120, check=False)

    def summary(self):
        return (self.run_dir / "summary.txt").read_text()

    def test_all_gates_pass_on_the_accepted_control(self):
        result = self.gate()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for word in ("FAIL", "SKIP", "REHEARS"):
            self.assertNotIn(word, result.stdout)
        summary = self.summary()
        self.assertEqual(summary, result.stdout)
        for line in ("PASS  unit tests (470 ran, 0 skipped)",
                     "PASS  S3.2 unit gate, run directly: python tests/unit/test_snapshot_ring.py (20 ran, 0 skipped)",
                     "PASS  S3.2 control: the Sprint 3.1 run s31-spark-ecf8-20261121T174511Z is ACCEPTED and holds the "
                     "plan of the reference pair and the accepted estimates",
                     "PASS  S3.2 references: the Sprint 1.1 build, from the project interpreter (environment.json)",
                     "PASS  S3.2 adapter: built against the Sprint 1.1 build and verified, pilot_ls 9999aaaabbbb, obs_tail "
                     "ddddeeeeffff and obs_feed 123412341234, with ENABLE_CUDA, 75 adapter tests, 0 compiler warnings",
                     "PASS  S3.2 accepted executables: the fixture adapter, which links no amended core, has the bytes of "
                     "the Sprint 3.1 run: grant_fixture 0123456789ab; rebuilt from the amended cores: replay_pusch "
                     "555566667777 (accepted: 0f1e2d3c4b5a), grid_apply 111122223333 (accepted: fedcba987654), pilot_ls "
                     "9999aaaabbbb (accepted: a1b2c3d4e5f6)",
                     "PASS  S3.2 native unit cases: 16 cases of the ring and 6 of the producer glue passed, the timed "
                     "cases of the three cores passed, and one grant through the pinned grid and the pinned host PUSCH "
                     "processor decodes in the mode timed without one SHA-256 from its fill to its release",
                     "PASS  S3.2 hardware: the adapter is linked against the CUDA build, so the CUDA processor can execute",
                     "Publishing the reference pair of s31-spark-ecf8-20261121T174511Z with "
                     "configs/integration/snapshot_ring.json.",
                     "PASS  S3.2 timed no checksum: between the first fill and the drain",
                     "NOTE  S3.2 timed delay of the spinning consumer",
                     "PASS  S3.2 integration grade: 10 of 10 cases, 8 of 8 checks, judgment pass, hardware gate closed",
                     "PASS  S3.2 a second reader recovers a timed record: 48 records kept by the native consumer in the "
                     "mode spin; record 0 of epoch 1 is the O2 estimate of pusch_dmrs in the layout ocudu_o2_comb_c64, "
                     "shape [4, 2, 306], 2448 valid elements, published in the mode timed with no digest, from grid 0 of "
                     "generation 1; the run computed 0 digests over 0 bytes between its first fill and its drain; SHA-256 "
                     f"of its payload {PAYLOAD_SHA256[:16]} by sha256sum, the digest of 0000.o2.c64 of the audited run and "
                     "of the accepted run s31-spark-ecf8-20261121T174511Z",
                     "PASS  S3.2 sources at the locked revisions after the runs",
                     "Sprint 3.2: ACCEPTED on this DGX Spark. Every observation leaves the receiver as one obs/v1 record "
                     "on the ring obs-ring/v1, first in, first out, or is dropped and counted.",
                     "accepted Sprint 3.1 run s31-spark-ecf8-20261121T174511Z. Next: S3.3 radar cube, features and known "
                     "return."):
            self.assertIn(line, summary)
        calls = self.calls.read_text()
        self.assertIn(f"s31 run: {self.s31}", calls)
        self.assertIn(f"scripts/native_adapter.py build --ocudu-build-dir {self.build}", calls)
        self.assertIn("-m ocudu_cuda_isac.sensing.obs show", calls)
        self.assertIn("-m ocudu_cuda_isac.episodes.snapshot_ring show", calls)
        self.assertNotIn("--host-rehearsal", calls)
        self.assertEqual(summary.count("PASS  "), 19)

    def test_the_newest_accepted_sprint_3_1_run_is_the_control(self):
        self.control_run("s31-spark-ecf8-20261121T180000Z", S31_SUMMARY.replace("PASS  committed", "FAIL  committed"))
        self.control_run("s31-spark-ecf8-20261121T190000Z", "Sprint 3.1: NOT ACCEPTED on this host\n")
        self.control_run("s31-spark-ecf8-20261121T200000Z", plans=False)
        self.control_run("s31-spark-ecf8-20261121T210000Z", S31_REHEARSED)
        newest = self.control_run("s31-spark-ecf8-20261121T175000Z")
        result = self.gate()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn(f"s31 run: {newest}", self.calls.read_text())
        other = self.control_run("s31-devhost-20261119T000000Z", S31_SUMMARY.replace("DGX Spark", "host"))
        result = self.gate("--s31-run", str(other), run_dir=self.root.parent / "evidence-named")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("Sprint 3.2: ACCEPTED on this host.", result.stdout)

    def test_without_an_accepted_control_nothing_is_published(self):
        shutil.rmtree(self.s31)
        result = self.gate()
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S3.2 control: no Sprint 3.1 ACCEPTED verdict with the plan of the reference pair in "
                      "results/runs/s31-*", result.stdout)
        self.assertIn("Run 'make accept-s3.1' on this host first", result.stdout)
        self.assertIn("SKIP  S3.2 reference check, adapter build, integration driver and record recovery: no control",
                      result.stdout)
        self.assertIn("Sprint 3.2: NOT ACCEPTED on this host", result.stdout)
        self.assertNotIn("tests/integration/snapshot_ring.py", self.calls.read_text())

    def test_a_fixture_adapter_with_other_bytes_fails(self):
        self.control_run("s31-spark-ecf8-20261121T175500Z",
                         executables=dict(ACCEPTED, grant_fixture={"sha256": "9" * 64}))
        result = self.gate()
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S3.2 accepted executables: the fixture adapter links no core that this sprint amended and "
                      "must keep its bytes: grant_fixture is 0123456789ab and was 999999999999", result.stdout)

    def test_a_build_without_cuda_cannot_be_accepted_and_a_rehearsal_never_is(self):
        result = self.gate(receipt=HOST_RECEIPT)
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S3.2 hardware: the accepted build has no CUDA", result.stdout)
        self.assertNotIn("Sprint 3.2: ACCEPTED", result.stdout)
        # A rehearsal needs a rehearsed control: the accepted run is not one.
        result = self.gate("--host-rehearsal", receipt=HOST_RECEIPT, grade_json=REHEARSAL_JSON,
                           run_dir=self.root.parent / "evidence-no-control")
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S3.2 control: no Sprint 3.1 REHEARSED verdict", result.stdout)
        rehearsed = self.control_run("s31-devhost-20261121T220000Z", S31_REHEARSED)
        result = self.gate("--host-rehearsal", receipt=HOST_RECEIPT, grade_json=REHEARSAL_JSON,
                           run_dir=self.root.parent / "evidence-rehearsal")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertNotIn("FAIL", result.stdout)
        for line in ("HOST REHEARSAL: the host path is graded on a build without CUDA.",
                     "PASS  S3.2 control: the Sprint 3.1 run s31-devhost-20261121T220000Z is REHEARSED",
                     "NOTE  S3.2 hardware: host rehearsal on an adapter built for the host path.",
                     "PASS  S3.2 integration grade: 10 of 10 cases, 8 of 8 checks, judgment rehearsal, hardware gate open",
                     "Sprint 3.2: REHEARSED on this host, NOT ACCEPTED."):
            self.assertIn(line, result.stdout)
        self.assertIn(f"s31 run: {rehearsed}", self.calls.read_text())
        self.assertIn("--host-rehearsal --output-dir", self.calls.read_text())
        # A rehearsal whose driver reports a pass was not a rehearsal, and the reverse.
        result = self.gate("--host-rehearsal", receipt=HOST_RECEIPT, run_dir=self.root.parent / "evidence-pass")
        self.assertEqual(result.returncode, 1)
        result = self.gate(grade_json=REHEARSAL_JSON, run_dir=self.root.parent / "evidence-claim")
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S3.2 integration grade: 10 of 10 cases, 8 of 8 checks, judgment rehearsal", result.stdout)

    def test_failures_of_single_gates(self):
        failed = dict(GRADE_JSON, judgment="fail", hardware_gate="open", cases_passed=9, checks_passed=7, checks_failed=1)
        cases = (
            (dict(unit="Ran 470 tests in 4.0s\n\nFAILED (failures=1)", unit_status="1"), "FAIL  unit tests (470 ran"),
            (dict(card="Ran 0 tests in 0.0s\n\nOK"), "tests/unit/test_snapshot_ring.py (none ran)"),
            # The native unit cases are part of the unit gate: a run that skipped them cannot close it.
            (dict(card="Ran 20 tests in 0.5s\n\nOK (skipped=9)"), "(20 ran, 9 skipped: the native unit cases need cmake"),
            (dict(environment="1"), "FAIL  S3.2 references: the Sprint 1.1 build"),
            (dict(build="1"), "FAIL  S3.2 adapter: build or verification failed"),
            (dict(verify="1"), "FAIL  S3.2 adapter: build or verification failed"),
            (dict(receipt=dict(RECEIPT, executables=ACCEPTED)), "FAIL  S3.2 adapter: the receipt names no pilot_ls, obs_tail"),
            (dict(ctest=CTEST.replace("obs_ring_case_15 ..............   Passed", "obs_ring_case_15 ..............***Failed")),
             "FAIL  S3.2 native unit cases: 15 of 16 cases of the ring"),
            (dict(ctest=CTEST.replace("obs_publisher_case_2 ..............   Passed", "obs_publisher_case_2 ...***Failed")),
             "and 5 of 6 of the producer glue"),
            (dict(ctest=CTEST.replace("pilot_ls_host_tap .........................   Passed", "pilot_ls_host_tap ...***Failed")),
             "and 3 of the 4 timed cases of the cores and of the tool"),
            (dict(no_ctest="1"), "FAIL  S3.2 native unit cases: 0 of 16 cases of the ring"),
            (dict(grade=GRADE.replace("PASS  S3.2 timed no checksum", "FAIL  S3.2 timed no checksum"), grade_json=failed,
                  driver="1"), "FAIL  S3.2 integration grade: 9 of 10 cases, 7 of 8 checks, judgment fail"),
            (dict(grade=GRADE.replace("PASS  S3.2 timed tap cost", "FAIL  S3.2 timed tap cost")),
             "FAIL  S3.2 timed tap cost"),
            (dict(no_evidence="1", driver="2"), "FAIL  S3.2 integration driver produced no graded checks"),
            (dict(grade_json=dict(GRADE_JSON, hardware_gate="open")), "judgment pass, hardware gate open"),
            (dict(show="1"), "FAIL  S3.2 a second reader recovers a timed record (see recovered.log)"),
            # The payload that the consumer kept is not the one the audited run wrote.
            (dict(kept="other estimate of the first opportunity, complex"),
             f"FAIL  S3.2 a second reader recovers a timed record: its payload has the digest"),
            (dict(audited="another estimate"), f"its payload has the digest {PAYLOAD_SHA256} by sha256sum, 0000.o2.c64 of "
                                               "the audited run"),
            (dict(pins="1"), "FAIL  S3.2 sources at the locked revisions after the runs"),
        )
        for number, (settings, line) in enumerate(cases):
            result = self.gate(run_dir=self.root.parent / f"evidence-{number}", **settings)
            with self.subTest(line=line):
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn(line, result.stdout)
                self.assertIn("Sprint 3.2: NOT ACCEPTED on this host", result.stdout)
                self.assertNotIn("Sprint 3.2: ACCEPTED", result.stdout)
        skipped = self.gate(run_dir=self.root.parent / "evidence-build", build="1")
        self.assertIn("SKIP  S3.2 native unit cases, integration driver and record recovery: no adapter", skipped.stdout)

    def test_the_estimate_of_the_accepted_run_must_be_the_timed_one(self):
        self.control_run("s31-spark-ecf8-20261121T175500Z", payload=b"an estimate of another tree")
        result = self.gate()
        self.assertEqual(result.returncode, 1)
        self.assertIn(f"its payload has the digest {PAYLOAD_SHA256} by sha256sum, 0000.o2.c64 of the audited run "
                      f"{PAYLOAD_SHA256} and of the accepted run s31-spark-ecf8-20261121T175500Z", result.stdout)

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
        again = self.gate()
        self.assertEqual(again.returncode, 2)
        self.assertIn("already exists. Evidence is never overwritten.", again.stderr)

    def test_usage(self):
        self.assertEqual(self.gate("--help").returncode, 0)
        for arguments in (("--unknown",), ("--s31-run",)):
            result = self.gate(*arguments, run_dir=self.root.parent / "evidence-usage")
            self.assertEqual(result.returncode, 2)
            self.assertIn("Usage: bash scripts/accept_s3_2.sh", result.stderr)
        (self.bin / "python").unlink()
        missing = self.gate(run_dir=self.root.parent / "evidence-python")
        self.assertEqual(missing.returncode, 2)
        self.assertIn("Run 'make env' first", missing.stderr)


if __name__ == "__main__":
    unittest.main()
