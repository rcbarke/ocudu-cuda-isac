"""Check the Sprint 3.1 gate runner against tool doubles. No compiler, source, build, NumPy or GPU is needed."""

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

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/accept_s3_1.sh"
TEST = "tests/integrationtests/phy/upper/channel_processors/pusch_e2e_pipeline_test"
PAYLOAD = b"the estimate of the first opportunity, complex64"
GRID = b"the stored grid of the first opportunity"
PAYLOAD_SHA256 = hashlib.sha256(PAYLOAD).hexdigest()
GRID_SHA256 = hashlib.sha256(GRID).hexdigest()
GRADE = """PASS  S3.1 control accepted run: the Sprint 2.4 driver graded replay_pusch as pass
PASS  S3.1 unity slot-grants estimate: demo-51prb-4rx: 6 grids, 14688 estimates of shape [4, 2, 306]
PASS  S3.1 direct known channel: 58752 of 58752 estimates on 24 grids and 4 receive planes
PASS  S3.1 reference-pair ingress: 48 of 48 observations name the readback digest of the grid
PASS  S3.1 reference-pair decodes: with the tap on, 96 radio records equal those of the control reference-pair
PASS  S3.1 payload data invariance: 48 of 48 grids differ
PASS  S3.1 backends executed path: cpu: 152 on the host path; cuda: 150 on the resident path, 2 on the host path
NOTE  S3.1 reference-pair service time of the tap, not graded: mean 40 us
pilot_ls: 9 of 9 cases passed, 7 of 7 checks"""
GRADE_JSON = {"judgment": "pass", "hardware_gate": "closed", "cases": 9, "cases_passed": 9, "checks": 7,
              "checks_passed": 7, "checks_failed": 0}
REHEARSAL_JSON = dict(GRADE_JSON, judgment="rehearsal",
                      hardware_gate="open: the CUDA processor did not execute, this is a host rehearsal")
ACCEPTED = {"grant_fixture": {"sha256": "0123456789abcdef" * 4}, "grid_apply": {"sha256": "fedcba9876543210" * 4},
            "replay_pusch": {"sha256": "0f1e2d3c4b5a6978" * 4}}
RECEIPT = {"executables": dict(ACCEPTED, pilot_ls={"sha256": "a1b2c3d4e5f60718" * 4}),
           "tests_run": 48, "compiler_warnings": 0, "pin": {"cuda": True, "archives": 50}}
HOST_RECEIPT = dict(RECEIPT, pin={"cuda": False, "archives": 50})
CTEST = "\n".join([f" {n + 35:2d}/48 Test #{n + 35}: observer_core_case_{n} ..............   Passed    0.00 sec"
                   for n in range(11)]
                  + ["48/48 Test #48: pilot_ls_host_tap .........................   Passed    0.25 sec",
                     "100% tests passed, 0 tests failed out of 48"])
RECOVERED = {"verified_observations": 48, "stage": "O2", "source": "pusch_dmrs", "producer": "observer",
             "opportunity_id": "rz-22963ee9d985.0000", "slot": 7, "grant_id": "demo-51prb-4rx", "shape": [4, 2, 306],
             "valid": 2448, "payload_file": "0000.o2.c64", "payload_sha256": PAYLOAD_SHA256,
             "ingress_sha256": GRID_SHA256,
             "element": {"rx_plane": 0, "symbol": 2, "subcarrier": 0, "value": [0.8125, -0.59375]}}
S24_SUMMARY = """Sprint 2.4 acceptance on spark-ecf8, 2026-11-21T08:34:24Z
PASS  committed tree
Sprint 2.4: ACCEPTED on this DGX Spark. The host and the CUDA PUSCH processor of the pinned source decode the replayed grids.
"""
S24_REHEARSED = """Sprint 2.4 acceptance on devhost, 2026-11-21T08:34:24Z
HOST REHEARSAL: the host path is graded on a build without CUDA. This run cannot accept the sprint.
Sprint 2.4: REHEARSED on this host, NOT ACCEPTED. The host path returned the original transport blocks.
"""
UNIT = "Ran 420 tests in 240.000s\n\nOK"
CARD = "Ran 31 tests in 9.500s\n\nOK"
# The interpreter double answers each gate from FAKE_* variables and logs the call. The builder
# double leaves the CTest log of the adapter build. The driver double leaves its grade, the payload
# of the first observation and the grid it was computed from.
PYTHON_DOUBLE = """#!/bin/sh
[ "$1" = "-c" ] && exec {real} "$@"
echo "$*" >> "$FAKE_LOG"
case "$*" in
    "-m pip freeze --all") echo "example==1.0" ;;
    *"-m unittest discover"*) printf '%s\\n' "$FAKE_UNIT" >&2; exit "${{FAKE_UNIT_STATUS:-0}}" ;;
    "tests/unit/test_pilot_ls.py -v") printf '%s\\n' "$FAKE_CARD" >&2; exit "${{FAKE_CARD_STATUS:-0}}" ;;
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
    "tests/integration/pilot_ls.py --config configs/integration/pilot_ls.json --s24-run "*)
        for out; do :; done
        echo "s24 run: $5" >> "$FAKE_LOG"
        echo "Run artifacts: $out"
        [ -z "${{FAKE_NO_EVIDENCE:-}}" ] || exit "${{FAKE_DRIVER:-0}}"
        pair="$out/reference_pair/run/native"
        mkdir -p "$pair/observer" "$pair/ingress"
        printf '%s' "${{FAKE_PAYLOAD:-$FAKE_ESTIMATE}}" > "$pair/observer/0000.o2.c64"
        printf '%s' "${{FAKE_STORED:-$FAKE_GRID}}" > "$pair/ingress/0000.c64"
        printf '%s\\n' "$FAKE_GRADE" > "$out/grade.txt"
        printf '%s\\n' "$FAKE_GRADE_JSON" > "$out/grade.json"
        exit "${{FAKE_DRIVER:-0}}" ;;
    "-m ocudu_cuda_isac.episodes.pilot_ls show "*)
        [ "${{FAKE_SHOW:-0}}" = 0 ] || {{ echo "pilot_ls: the payload of observation record 3 differs from its digest or size" >&2; exit 2; }}
        printf '%s\\n' "$FAKE_RECOVERED" ;;
    *"check_pins.py --worktrees") echo "ext/ocudu 27e84cb3 (Git worktree)"; exit "${{FAKE_PINS:-0}}" ;;
    *) echo "interpreter double: unexpected $*" >&2; exit 64 ;;
esac
"""


@unittest.skipUnless(shutil.which("bash") and shutil.which("sha256sum"), "the gate runner is a Bash script")
class SprintThreeObserverGateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / "host checkout"
        self.bin = self.root / "isac/bin"
        (self.root / "scripts").mkdir(parents=True)
        self.bin.mkdir(parents=True)
        shutil.copy(SCRIPT, self.root / "scripts/accept_s3_1.sh")
        python = self.bin / "python"
        python.write_text(PYTHON_DOUBLE.format(real=shlex.quote(sys.executable)))
        python.chmod(0o755)
        # The build Sprint 1.1 published and the run that Sprint 2.4 accepted.
        self.build = self.root / "build/ocudu-spark"
        (self.build / TEST).parent.mkdir(parents=True)
        (self.build / TEST).write_text("#!/bin/sh\n")
        (self.bin / "pusch_e2e_pipeline_test").symlink_to(self.build / TEST)
        (self.build / "isac-native.json").write_text(json.dumps({"target": "all"}))
        self.s24 = self.control_run("s24-spark-ecf8-20261121T083424Z")
        self.calls = self.root.parent / "calls.log"
        self.run_dir = self.root.parent / "evidence"

    def control_run(self, name, summary=S24_SUMMARY, plans=True, executables=None, opportunity="rz-22963ee9d985.0000",
                    digest=GRID_SHA256):
        run = self.root / "results/runs" / name
        pair = run / "replay_pusch/reference_pair/run"
        (pair / "native/radio").mkdir(parents=True)
        (run / "replay_pusch/grade.json").write_text("{}")
        if plans:
            (pair / "plan.json").write_text("{}")
        (pair / "native/radio/radio.jsonl").write_text(json.dumps(
            {"opportunity_id": opportunity, "ingress": {"readback_sha256": digest}}) + "\n")
        (run / "adapter-receipt.json").write_text(json.dumps({"executables": executables or ACCEPTED}))
        (run / "summary.txt").write_text(summary)
        return run

    def gate(self, *arguments, run_dir=None, **settings):
        env = {key: value for key, value in os.environ.items() if not key.startswith("FAKE_")}
        env.update(FAKE_LOG=str(self.calls), FAKE_UNIT=UNIT, FAKE_CARD=CARD, FAKE_GRADE=GRADE,
                   FAKE_GRADE_JSON=json.dumps(GRADE_JSON), FAKE_RECEIPT=json.dumps(RECEIPT), FAKE_CTEST=CTEST,
                   FAKE_RECOVERED=json.dumps(RECOVERED), FAKE_ESTIMATE=PAYLOAD.decode("ascii"),
                   FAKE_GRID=GRID.decode("ascii"))
        env.update({f"FAKE_{key.upper()}": value if isinstance(value, str) else json.dumps(value)
                    for key, value in settings.items()})
        command = [shutil.which("bash"), str(self.root / "scripts/accept_s3_1.sh"),
                   "--run-dir", str(run_dir or self.run_dir), *arguments]
        return subprocess.run(command, env=env, capture_output=True, text=True, timeout=120, check=False)

    def summary(self):
        return (self.run_dir / "summary.txt").read_text()

    def test_all_gates_pass_on_the_accepted_control(self):
        result = self.gate()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("FAIL", result.stdout)
        self.assertNotIn("SKIP", result.stdout)
        self.assertNotIn("REHEARS", result.stdout)
        summary = self.summary()
        self.assertEqual(summary, result.stdout)
        for line in ("PASS  unit tests (420 ran, 0 skipped)",
                     "PASS  S3.1 unit gate, run directly: python tests/unit/test_pilot_ls.py (31 ran, 0 skipped)",
                     "PASS  S3.1 control: the Sprint 2.4 run s24-spark-ecf8-20261121T083424Z is ACCEPTED and holds the "
                     "plans and the radio records of the untapped decodes",
                     "PASS  S3.1 references: the Sprint 1.1 build, from the project interpreter (environment.json)",
                     "PASS  S3.1 adapter: built against the Sprint 1.1 build and verified, pilot_ls a1b2c3d4e5f6 beside "
                     "replay_pusch 0f1e2d3c4b5a, grid_apply fedcba987654 and grant_fixture 0123456789ab, with ENABLE_CUDA, "
                     "48 adapter tests, 0 compiler warnings",
                     "PASS  S3.1 control executables: this tree builds the executables that the Sprint 2.4 run used, byte "
                     "for byte: replay_pusch 0f1e2d3c4b5a, grid_apply fedcba987654, grant_fixture 0123456789ab",
                     "PASS  S3.1 native unit cases: 11 cases of the host-free observer core passed, and one grant through "
                     "the pinned grid and the pinned host PUSCH processor reads 1.00119 on every pilot",
                     "PASS  S3.1 hardware: the adapter is linked against the CUDA build, so the CUDA processor can execute",
                     "Tapping the plans of s24-spark-ecf8-20261121T083424Z with configs/integration/pilot_ls.json.",
                     "PASS  S3.1 reference-pair decodes: with the tap on, 96 radio records equal",
                     "NOTE  S3.1 reference-pair service time of the tap, not graded",
                     "PASS  S3.1 integration grade: 9 of 9 cases, 7 of 7 checks, judgment pass, hardware gate closed",
                     "PASS  S3.1 a second reader recovers an observation: 48 observations verified; the O2 estimate of "
                     "pusch_dmrs by the observer on rz-22963ee9d985.0000 (slot 7, grant demo-51prb-4rx) has the shape "
                     "[4, 2, 306] with 2448 valid elements, and its first element, symbol 2, subcarrier 0 of receive "
                     f"plane 0, is 0.812500-0.593750j; SHA-256 of the payload {PAYLOAD_SHA256[:16]} by the reader and by "
                     f"sha256sum; the grid it was computed from is {GRID_SHA256[:16]} by sha256sum, in the observation "
                     "and in the radio record of s24-spark-ecf8-20261121T083424Z",
                     "PASS  S3.1 sources at the locked revisions after the runs",
                     "Sprint 3.1: ACCEPTED on this DGX Spark. The observer exports the pilot least squares of the PUSCH "
                     "DM-RS, the stage O2, from the grid that the host and the CUDA PUSCH processor of the pinned source "
                     "decode, and every decode is the one of the accepted Sprint 2.4 run "
                     "s24-spark-ecf8-20261121T083424Z. Next: S3.2 snapshot_ring."):
            self.assertIn(line, summary)
        calls = self.calls.read_text()
        self.assertIn(f"s24 run: {self.s24}", calls)
        self.assertIn(f"scripts/native_adapter.py build --ocudu-build-dir {self.build}", calls)
        self.assertIn("scripts/verify_environment.py --profile core --with-native", calls)
        self.assertIn("-m ocudu_cuda_isac.episodes.pilot_ls show", calls)
        self.assertNotIn("--host-rehearsal", calls)
        self.assertEqual(summary.count("PASS  "), 18)

    def test_the_newest_accepted_sprint_2_4_run_is_the_control(self):
        self.control_run("s24-spark-ecf8-20261121T090000Z", S24_SUMMARY.replace("PASS  committed", "FAIL  committed"))
        self.control_run("s24-spark-ecf8-20261121T100000Z", "Sprint 2.4: NOT ACCEPTED on this host\n")
        self.control_run("s24-spark-ecf8-20261121T110000Z", plans=False)  # a verdict without archived plans
        self.control_run("s24-spark-ecf8-20261121T120000Z", S24_REHEARSED)  # a rehearsal is no control of an acceptance
        newest = self.control_run("s24-spark-ecf8-20261121T085000Z")
        result = self.gate()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn(f"s24 run: {newest}", self.calls.read_text())
        self.assertIn("accepted Sprint 2.4 run s24-spark-ecf8-20261121T085000Z. Next: S3.2 snapshot_ring.", self.summary())

    def test_a_named_run_and_a_development_host(self):
        other = self.control_run("s24-devhost-20261119T000000Z", S24_SUMMARY.replace("DGX Spark", "host"))
        result = self.gate("--s24-run", str(other))
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn(f"s24 run: {other}", self.calls.read_text())
        self.assertIn("Sprint 3.1: ACCEPTED on this host.", result.stdout)

    def test_without_an_accepted_control_nothing_is_tapped(self):
        shutil.rmtree(self.s24)
        result = self.gate()
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S3.1 control: no Sprint 2.4 ACCEPTED verdict with archived plans in results/runs/s24-*",
                      result.stdout)
        self.assertIn("Run 'make accept-s2.4' on this host first", result.stdout)
        self.assertIn("SKIP  S3.1 reference check, adapter build, integration driver and observation recovery: no control",
                      result.stdout)
        self.assertIn("Sprint 3.1: NOT ACCEPTED on this host", result.stdout)
        self.assertNotIn("tests/integration/pilot_ls.py", self.calls.read_text())

    def test_a_missing_sprint_1_1_build_is_named(self):
        (self.bin / "pusch_e2e_pipeline_test").unlink()
        result = self.gate()
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S3.1 needs the Sprint 1.1 build", result.stdout)
        self.assertNotIn("native_adapter.py", self.calls.read_text())

    def test_an_accepted_executable_with_other_bytes_fails(self):
        self.control_run("s24-spark-ecf8-20261121T095000Z",
                         executables=dict(ACCEPTED, replay_pusch={"sha256": "9" * 64}))
        result = self.gate()
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S3.1 control executables: an executable that Sprint 2 accepted has other bytes in this "
                      "build: replay_pusch is 0f1e2d3c4b5a and was 999999999999", result.stdout)
        self.assertIn("Sprint 3.1: NOT ACCEPTED on this host", result.stdout)

    def test_a_build_without_cuda_cannot_be_accepted(self):
        failed_grade = dict(GRADE_JSON, judgment="fail", hardware_gate="open", cases_passed=8, checks_passed=6, checks_failed=1)
        result = self.gate(receipt=HOST_RECEIPT, grade_json=failed_grade, driver="1",
                           grade=GRADE.replace("PASS  S3.1 backends executed path", "FAIL  S3.1 backends executed path"))
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S3.1 hardware: the accepted build has no CUDA, so the CUDA processor cannot execute here.",
                      result.stdout)
        self.assertIn("Sprint 3.1: NOT ACCEPTED on this host", result.stdout)
        # A driver that reports a pass on such a build still cannot close the gate.
        result = self.gate(receipt=HOST_RECEIPT, run_dir=self.root.parent / "evidence-claimed")
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("Sprint 3.1: ACCEPTED", result.stdout)

    def test_a_host_rehearsal_is_reported_and_never_accepted(self):
        # A rehearsal needs a rehearsed control: the accepted run is not one.
        result = self.gate("--host-rehearsal", receipt=HOST_RECEIPT, grade_json=REHEARSAL_JSON)
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S3.1 control: no Sprint 2.4 REHEARSED verdict with archived plans", result.stdout)
        self.assertIn("Run 'make accept-s2.4 REHEARSAL=1' on this host first", result.stdout)
        rehearsed = self.control_run("s24-devhost-20261121T120000Z", S24_REHEARSED)
        result = self.gate("--host-rehearsal", receipt=HOST_RECEIPT, grade_json=REHEARSAL_JSON,
                           run_dir=self.root.parent / "evidence-rehearsal")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertNotIn("Sprint 3.1: ACCEPTED", result.stdout)
        self.assertNotIn("FAIL", result.stdout)
        for line in ("HOST REHEARSAL: the host path is graded on a build without CUDA. This run cannot accept the sprint.",
                     "PASS  S3.1 control: the Sprint 2.4 run s24-devhost-20261121T120000Z is REHEARSED",
                     "NOTE  S3.1 hardware: host rehearsal on an adapter built for the host path. The CUDA decodes of this "
                     "gate stay open.",
                     "PASS  S3.1 integration grade: 9 of 9 cases, 7 of 7 checks, judgment rehearsal, hardware gate open",
                     "Sprint 3.1: REHEARSED on this host, NOT ACCEPTED. The observer exported the pilot least squares "
                     "from the grids of the rehearsed Sprint 2.4 run s24-devhost-20261121T120000Z and the host path "
                     "decodes were unchanged. The CUDA decodes are open: run 'make accept-s3.1' on the DGX Spark."):
            self.assertIn(line, result.stdout)
        calls = self.calls.read_text()
        self.assertIn(f"s24 run: {rehearsed}", calls)
        self.assertIn("--host-rehearsal --output-dir", calls)
        # A rehearsal whose driver reports a pass was not a rehearsal.
        result = self.gate("--host-rehearsal", receipt=HOST_RECEIPT, run_dir=self.root.parent / "evidence-pass")
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S3.1 integration grade: 9 of 9 cases, 7 of 7 checks, judgment pass, hardware gate closed, "
                      "driver exit status 0", result.stdout)
        # A grade that says rehearsal cannot close the gate that was run without the option.
        result = self.gate(grade_json=REHEARSAL_JSON, run_dir=self.root.parent / "evidence-claim")
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S3.1 integration grade: 9 of 9 cases, 7 of 7 checks, judgment rehearsal, hardware gate open",
                      result.stdout)

    def test_failures_of_single_gates(self):
        cases = (
            (dict(unit="Ran 420 tests in 4.0s\n\nFAILED (failures=1)", unit_status="1"), "FAIL  unit tests (420 ran"),
            (dict(card="Ran 0 tests in 0.0s\n\nOK"), "FAIL  S3.1 unit gate, run directly: python "
                                                    "tests/unit/test_pilot_ls.py (none ran)"),
            # The native unit cases are part of the unit gate: a run that skipped them cannot close it.
            (dict(card="Ran 31 tests in 0.5s\n\nOK (skipped=4)"), "FAIL  S3.1 unit gate, run directly: python "
             "tests/unit/test_pilot_ls.py (31 ran, 4 skipped: the native unit cases need cmake"),
            (dict(environment="1"), "FAIL  S3.1 references: the Sprint 1.1 build"),
            (dict(build="1"), "FAIL  S3.1 adapter: build or verification failed"),
            (dict(verify="1"), "FAIL  S3.1 adapter: build or verification failed"),
            (dict(receipt=dict(RECEIPT, executables=ACCEPTED)), "FAIL  S3.1 adapter: the receipt names no pilot_ls"),
            (dict(ctest=CTEST.replace("observer_core_case_10 ..............   Passed", "observer_core_case_10 ..............***Failed")),
             "FAIL  S3.1 native unit cases: 10 cases of the observer core passed"),
            (dict(ctest=CTEST.replace("pilot_ls_host_tap .........................   Passed",
                                      "pilot_ls_host_tap .........................***Failed")),
             "FAIL  S3.1 native unit cases: 11 cases of the observer core passed and the tap self-test on the pinned "
             "grid did not"),
            (dict(no_ctest="1"), "FAIL  S3.1 native unit cases: 0 cases of the observer core passed"),
            (dict(grade=GRADE.replace("PASS  S3.1 reference-pair decodes", "FAIL  S3.1 reference-pair decodes"),
                  grade_json=dict(GRADE_JSON, judgment="fail", hardware_gate="open", cases_passed=8, checks_passed=6,
                                  checks_failed=1), driver="1"),
             "FAIL  S3.1 integration grade: 8 of 9 cases, 6 of 7 checks, judgment fail, hardware gate open, driver exit "
             "status 1"),
            (dict(no_evidence="1", driver="2"), "FAIL  S3.1 integration driver produced no graded checks"),
            (dict(grade_json=dict(GRADE_JSON, cases=0, cases_passed=0)), "FAIL  S3.1 integration grade: 0 of 0 cases"),
            (dict(grade_json=dict(GRADE_JSON, hardware_gate="open")),
             "FAIL  S3.1 integration grade: 9 of 9 cases, 7 of 7 checks, judgment pass, hardware gate open"),
            (dict(show="1"), "FAIL  S3.1 a second reader recovers an observation (see recovered.log)"),
            # The payload or the grid on the disk is not the one the observation names.
            (dict(payload="other estimate"), "FAIL  S3.1 a second reader recovers an observation: the payload has the "
                                             f"digest {PAYLOAD_SHA256}"),
            (dict(stored="another grid"), f"{GRID_SHA256} in the observation and {GRID_SHA256} in the radio record"),
            (dict(recovered=dict(RECOVERED, ingress_sha256="0" * 64)), "or its grid is"),
            (dict(pins="1"), "FAIL  S3.1 sources at the locked revisions after the runs"),
        )
        for number, (settings, line) in enumerate(cases):
            run_dir = self.root.parent / f"evidence-{number}"
            result = self.gate(run_dir=run_dir, **settings)
            with self.subTest(line=line):
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn(line, result.stdout)
                self.assertIn("Sprint 3.1: NOT ACCEPTED on this host", result.stdout)
                self.assertNotIn("Sprint 3.1: ACCEPTED", result.stdout)
                self.assertNotIn("REHEARSED", result.stdout)
        skipped = self.gate(run_dir=self.root.parent / "evidence-build", build="1")
        self.assertIn("SKIP  S3.1 native unit cases, integration driver and observation recovery: no adapter",
                      skipped.stdout)
        # A skip in the portable suite is reported and does not fail it: only the unit gate must be complete.
        portable = self.gate(run_dir=self.root.parent / "evidence-skip", unit="Ran 420 tests in 9.0s\n\nOK (skipped=12)")
        self.assertEqual(portable.returncode, 0, portable.stdout)
        self.assertIn("PASS  unit tests (420 ran, 12 skipped)", portable.stdout)

    def test_the_grid_of_the_control_must_be_the_grid_of_the_observation(self):
        self.control_run("s24-spark-ecf8-20261121T095000Z", digest="1" * 64)
        result = self.gate()
        self.assertEqual(result.returncode, 1)
        self.assertIn(f"{GRID_SHA256} in the observation and {'1' * 64} in the radio record", result.stdout)
        self.control_run("s24-spark-ecf8-20261121T096000Z", opportunity="rz-other.0000")
        result = self.gate(run_dir=self.root.parent / "evidence-other")
        self.assertEqual(result.returncode, 1)
        self.assertIn("and another opportunity in the radio record", result.stdout)

    def test_a_failed_check_of_the_driver_fails_the_gate_even_with_exit_status_zero(self):
        result = self.gate(grade=GRADE.replace("PASS  S3.1 payload data", "FAIL  S3.1 payload data"))
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S3.1 payload data invariance", result.stdout)
        self.assertNotIn("Sprint 3.1: ACCEPTED", result.stdout)

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
        for arguments in (("--unknown",), ("--s24-run",)):
            result = self.gate(*arguments, run_dir=self.root.parent / "evidence-usage")
            self.assertEqual(result.returncode, 2)
            self.assertIn("Usage: bash scripts/accept_s3_1.sh", result.stderr)
        (self.bin / "python").unlink()
        missing = self.gate(run_dir=self.root.parent / "evidence-python")
        self.assertEqual(missing.returncode, 2)
        self.assertIn("Run 'make env' first", missing.stderr)


if __name__ == "__main__":
    unittest.main()
