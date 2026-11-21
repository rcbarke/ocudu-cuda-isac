"""Check the Sprint 3.3 gate runner against tool doubles. No compiler, source, build, NumPy or GPU is needed."""

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

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/accept_s3_3.sh"
TEST = "tests/integrationtests/phy/upper/channel_processors/pusch_e2e_pipeline_test"
VIEW = b"the delay-Doppler power of the target-present window, float64"
VIEW_SHA256 = hashlib.sha256(VIEW).hexdigest()
GRADE = """PASS  S3.3 control accepted run: the Sprint 3.2 driver graded snapshot_ring as pass
PASS  S3.3 stream output: 48 records of the ring in sequence order: 2 complete windows
PASS  S3.3 offline feature values: the view and the vector of each window against explicit Fourier sums
PASS  S3.3 known-return one resolvable bin: the geometry of the entry gives a return at 179.8 ns and -43.3 Hz
PASS  S3.3 causality future samples: with the payload of the 2 records after the completing one replaced by 1e+30
PASS  S3.3 mask sentinel: 4 elements of observation 36 of the present window lose their validity bit
PASS  S3.3 live same features: the 16 answers of the live sensor are those of the recorded stream
NOTE  S3.3 stream axes of a window, signed: delay 32 bins of 54.47 ns
pilot_features: 8 of 8 cases passed, 7 of 7 checks"""
GRADE_JSON = {"judgment": "pass", "hardware_gate": "closed", "cases": 8, "cases_passed": 8, "checks": 7,
              "checks_passed": 7, "checks_failed": 0}
REHEARSAL_JSON = dict(GRADE_JSON, judgment="rehearsal",
                      hardware_gate="open: the CUDA processor did not execute, this is a host rehearsal")
ACCEPTED = {"grant_fixture": {"sha256": "0123456789abcdef" * 4}, "grid_apply": {"sha256": "1111222233334444" * 4},
            "replay_pusch": {"sha256": "5555666677778888" * 4}, "pilot_ls": {"sha256": "9999aaaabbbbcccc" * 4},
            "obs_tail": {"sha256": "ddddeeeeffff0000" * 4}, "obs_feed": {"sha256": "1234123412341234" * 4}}
RECEIPT = {"executables": ACCEPTED, "tests_run": 75, "compiler_warnings": 0, "pin": {"cuda": True, "archives": 50}}
HOST_RECEIPT = dict(RECEIPT, pin={"cuda": False, "archives": 50})
RECOVER = {"sensor": "live/run/sensor", "window": 1, "stream_view": "stream/sensor/views/000015.dd.f64"}
RECOVERED = {"spec_id": "dd-uniform-8", "runtime": "reference", "source": "ring", "complete_windows": 2,
             "steps": {"cube": "comb_cube (numpy on the cpu)", "static": "window_mean (numpy on the cpu)",
                       "delay": "ifft (numpy on the cpu)", "doppler": "fft (numpy on the cpu)",
                       "features": "dd_compact (numpy on the cpu)"},
             "window": {"id": "1/89abcdef01234567/7", "contributing": 8, "acquisition_us": 35033.33,
                        "delay_resolution_ns": 54.466, "doppler_resolution_hz": 25.0,
                        "peak": {"delay_bin": 3, "doppler_bin": -2, "delay_ns": 163.4, "doppler_hz": -50.0},
                        "view_file": "views/000015.dd.f64", "view_sha256": VIEW_SHA256,
                        "features": {"peak_to_median_db": 22.2}}}
S32_SUMMARY = """Sprint 3.2 acceptance on spark-ecf8, 2026-11-21T20:44:25Z
PASS  committed tree
Sprint 3.2: ACCEPTED on this DGX Spark. Every observation leaves the receiver as one obs/v1 record on the ring.
"""
S32_REHEARSED = """Sprint 3.2 acceptance on devhost, 2026-11-21T20:44:25Z
HOST REHEARSAL: the host path is graded on a build without CUDA. This run cannot accept the sprint.
Sprint 3.2: REHEARSED on this host, NOT ACCEPTED. The observations were published on the ring.
"""
UNIT = "Ran 480 tests in 300.000s\n\nOK"
CARD = "Ran 27 tests in 3.500s\n\nOK"
RULE = "Ran 3 tests in 0.200s\n\nOK"
# The interpreter double answers each gate from FAKE_* variables and logs the call. The driver
# double leaves its grade, the place of the window to recover and the two view files.
PYTHON_DOUBLE = """#!/bin/sh
[ "$1" = "-c" ] && exec {real} "$@"
echo "$*" >> "$FAKE_LOG"
case "$*" in
    "-m pip freeze --all") echo "example==1.0" ;;
    *"-m unittest discover"*) printf '%s\\n' "$FAKE_UNIT" >&2; exit "${{FAKE_UNIT_STATUS:-0}}" ;;
    "tests/unit/test_pilot_features.py -v") printf '%s\\n' "$FAKE_CARD" >&2; exit "${{FAKE_CARD_STATUS:-0}}" ;;
    "tests/unit/test_core_imports.py -v") printf '%s\\n' "$FAKE_RULE" >&2; exit "${{FAKE_RULE_STATUS:-0}}" ;;
    *"--profile core --with-native"*) exit "${{FAKE_ENVIRONMENT:-0}}" ;;
    "scripts/native_adapter.py build --ocudu-build-dir "*)
        echo "Adapter build: build/ocudu-adapter"
        mkdir -p "$6"
        [ "${{FAKE_BUILD:-0}}" = 0 ] || echo "native_adapter: the adapter build step exited with status 2" >&2
        exit "${{FAKE_BUILD:-0}}" ;;
    "scripts/native_adapter.py verify")
        [ "${{FAKE_VERIFY:-0}}" = 0 ] || {{ echo "native_adapter: the adapter sources changed after the build" >&2; exit 1; }}
        printf '%s\\n' "$FAKE_RECEIPT" ;;
    "tests/integration/pilot_features.py --config configs/integration/pilot_features.json --s32-run "*)
        for out; do :; done
        echo "s32 run: $5" >> "$FAKE_LOG"
        echo "Run artifacts: $out"
        [ -z "${{FAKE_NO_EVIDENCE:-}}" ] || exit "${{FAKE_DRIVER:-0}}"
        mkdir -p "$out/live/run/sensor/views" "$out/stream/sensor/views"
        printf '%s' "${{FAKE_LIVE:-$FAKE_VIEW}}" > "$out/live/run/sensor/views/000015.dd.f64"
        printf '%s' "${{FAKE_RECORDED:-$FAKE_VIEW}}" > "$out/stream/sensor/views/000015.dd.f64"
        [ -n "${{FAKE_NO_RECOVER:-}}" ] || printf '%s\\n' "$FAKE_RECOVER" > "$out/recover.json"
        printf '%s\\n' "$FAKE_GRADE" > "$out/grade.txt"
        printf '%s\\n' "$FAKE_GRADE_JSON" > "$out/grade.json"
        exit "${{FAKE_DRIVER:-0}}" ;;
    "-m ocudu_cuda_isac.sensing.features show "*)
        [ "${{FAKE_SHOW:-0}}" = 0 ] || {{ echo "features: line 15: the view, its hash, its peak or the vector do not agree" >&2; exit 2; }}
        printf '%s\\n' "$FAKE_RECOVERED" ;;
    *"check_pins.py --worktrees") echo "ext/ocudu 27e84cb3 (Git worktree)"; exit "${{FAKE_PINS:-0}}" ;;
    *) echo "interpreter double: unexpected $*" >&2; exit 64 ;;
esac
"""


@unittest.skipUnless(shutil.which("bash") and shutil.which("sha256sum"), "the gate runner is a Bash script")
class SprintThreeFeaturesGateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / "host checkout"
        self.bin = self.root / "isac/bin"
        (self.root / "scripts").mkdir(parents=True)
        self.bin.mkdir(parents=True)
        shutil.copy(SCRIPT, self.root / "scripts/accept_s3_3.sh")
        python = self.bin / "python"
        python.write_text(PYTHON_DOUBLE.format(real=shlex.quote(sys.executable)))
        python.chmod(0o755)
        # The build Sprint 1.1 published and the run that Sprint 3.2 accepted.
        self.build = self.root / "build/ocudu-spark"
        (self.build / TEST).parent.mkdir(parents=True)
        (self.build / TEST).write_text("#!/bin/sh\n")
        (self.bin / "pusch_e2e_pipeline_test").symlink_to(self.build / TEST)
        (self.build / "isac-native.json").write_text(json.dumps({"target": "all"}))
        self.s32 = self.control_run("s32-spark-ecf8-20261121T204425Z")
        self.calls = self.root.parent / "calls.log"
        self.run_dir = self.root.parent / "evidence"

    def control_run(self, name, summary=S32_SUMMARY, archive=True, executables=None):
        run = self.root / "results/runs" / name
        ring = run / "snapshot_ring"
        (ring / "timed/run/consumer-0").mkdir(parents=True)
        (ring / "audit/run/native/observer").mkdir(parents=True)
        (ring / "grade.json").write_text("{}")
        (ring / "audit/run/native/observer/o2.jsonl").write_text("")
        if archive:
            (ring / "timed/run/consumer-0/consumer.json").write_text("{}")
        (run / "adapter-receipt.json").write_text(json.dumps({"executables": executables or ACCEPTED}))
        (run / "summary.txt").write_text(summary)
        return run

    def gate(self, *arguments, run_dir=None, **settings):
        env = {key: value for key, value in os.environ.items() if not key.startswith("FAKE_")}
        env.update(FAKE_LOG=str(self.calls), FAKE_UNIT=UNIT, FAKE_CARD=CARD, FAKE_RULE=RULE, FAKE_GRADE=GRADE,
                   FAKE_GRADE_JSON=json.dumps(GRADE_JSON), FAKE_RECEIPT=json.dumps(RECEIPT),
                   FAKE_RECOVER=json.dumps(RECOVER), FAKE_RECOVERED=json.dumps(RECOVERED), FAKE_VIEW=VIEW.decode("ascii"))
        env.update({f"FAKE_{key.upper()}": value if isinstance(value, str) else json.dumps(value)
                    for key, value in settings.items()})
        command = [shutil.which("bash"), str(self.root / "scripts/accept_s3_3.sh"),
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
        for line in ("PASS  unit tests (480 ran, 0 skipped)",
                     "PASS  S3.3 unit gate, run directly: python tests/unit/test_pilot_features.py (27 ran, 0 skipped)",
                     "PASS  S3.3 import rule, run directly: python tests/unit/test_core_imports.py (3 ran, 0 skipped)",
                     "PASS  S3.3 control: the Sprint 3.2 run s32-spark-ecf8-20261121T204425Z is ACCEPTED and holds the "
                     "records its consumer kept from the ring and the files of its audited observer",
                     "PASS  S3.3 references: the Sprint 1.1 build, from the project interpreter (environment.json)",
                     "PASS  S3.3 adapter: built against the Sprint 1.1 build and verified, pilot_ls 9999aaaabbbb, with "
                     "ENABLE_CUDA, 75 adapter tests, 0 compiler warnings",
                     "PASS  S3.3 accepted executables: this sprint changes no native source, and every tool has the bytes "
                     "of the Sprint 3.2 run: grant_fixture 0123456789ab, grid_apply 111122223333, obs_feed 123412341234, "
                     "obs_tail ddddeeeeffff, pilot_ls 9999aaaabbbb, replay_pusch 555566667777",
                     "PASS  S3.3 hardware: the adapter is linked against the CUDA build, so the CUDA processor can execute",
                     "Features of the reference pair of s32-spark-ecf8-20261121T204425Z with "
                     "configs/integration/pilot_features.json.",
                     "PASS  S3.3 known-return one resolvable bin: the geometry of the entry gives a return at 179.8 ns",
                     "NOTE  S3.3 stream axes of a window, signed",
                     "PASS  S3.3 integration grade: 8 of 8 cases, 7 of 7 checks, judgment pass, hardware gate closed",
                     "PASS  S3.3 a second reader recovers the target-present window of the live sensor: 2 complete windows "
                     "from a ring in the runtime reference; window 1/89abcdef01234567/7 holds 8 observations over 35.03 "
                     "ms, with cube comb_cube (numpy on the cpu), delay ifft (numpy on the cpu), doppler fft (numpy on "
                     "the cpu), features dd_compact (numpy on the cpu), static window_mean (numpy on the cpu); its "
                     "strongest moving cell lies at delay bin 3 and Doppler bin -2, 163.4 ns and -50.0 Hz, 22.2 dB above "
                     "the median, with bins of 54.47 ns and 25.00 Hz; SHA-256 of its view "
                     f"{VIEW_SHA256[:16]} by sha256sum, the digest its line names and that of the same window from the "
                     "records of the accepted run s32-spark-ecf8-20261121T204425Z",
                     "PASS  S3.3 sources at the locked revisions after the runs",
                     "Sprint 3.3: ACCEPTED on this DGX Spark. The reference of the sensing core forms the radar cube, the "
                     "delay-Doppler view and the causal features from the records of the ring, one record at a time.",
                     "from the files of its observer and from a live ring. Next: S3.4 presence head."):
            self.assertIn(line, summary)
        calls = self.calls.read_text()
        self.assertIn(f"s32 run: {self.s32}", calls)
        self.assertIn(f"scripts/native_adapter.py build --ocudu-build-dir {self.build}", calls)
        self.assertIn(f"-m ocudu_cuda_isac.sensing.features show {self.run_dir}/pilot_features/live/run/sensor "
                      "--window 1", calls)
        self.assertNotIn("--host-rehearsal", calls)
        self.assertEqual(summary.count("PASS  "), 18)

    def test_the_newest_accepted_sprint_3_2_run_is_the_control(self):
        self.control_run("s32-spark-ecf8-20261121T210000Z", S32_SUMMARY.replace("PASS  committed", "FAIL  committed"))
        self.control_run("s32-spark-ecf8-20261121T220000Z", "Sprint 3.2: NOT ACCEPTED on this host\n")
        self.control_run("s32-spark-ecf8-20261121T230000Z", archive=False)
        self.control_run("s32-spark-ecf8-20261121T235000Z", S32_REHEARSED)
        newest = self.control_run("s32-spark-ecf8-20261121T205000Z")
        result = self.gate()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn(f"s32 run: {newest}", self.calls.read_text())
        other = self.control_run("s32-devhost-20261119T000000Z", S32_SUMMARY.replace("DGX Spark", "host"))
        result = self.gate("--s32-run", str(other), run_dir=self.root.parent / "evidence-named")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("Sprint 3.3: ACCEPTED on this host.", result.stdout)

    def test_without_an_accepted_control_nothing_is_computed(self):
        shutil.rmtree(self.s32)
        result = self.gate()
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S3.3 control: no Sprint 3.2 ACCEPTED verdict with the records of the ring and the files of "
                      "the audited observer in results/runs/s32-*", result.stdout)
        self.assertIn("Run 'make accept-s3.2' on this host first", result.stdout)
        self.assertIn("SKIP  S3.3 reference check, adapter build, integration driver and window recovery: no control",
                      result.stdout)
        self.assertIn("Sprint 3.3: NOT ACCEPTED on this host", result.stdout)
        self.assertNotIn("tests/integration/pilot_features.py", self.calls.read_text())

    def test_a_tool_with_other_bytes_fails(self):
        self.control_run("s32-spark-ecf8-20261121T205500Z", executables=dict(ACCEPTED, pilot_ls={"sha256": "9" * 64}))
        result = self.gate()
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S3.3 accepted executables: this sprint changes no native source, so every tool must keep "
                      "the bytes of the Sprint 3.2 run: pilot_ls is 9999aaaabbbb and was 999999999999", result.stdout)
        # A tool of the accepted run that this build no longer holds is a change too.
        self.control_run("s32-spark-ecf8-20261121T205600Z", executables=dict(ACCEPTED, obs_probe={"sha256": "7" * 64}))
        result = self.gate(run_dir=self.root.parent / "evidence-missing")
        self.assertEqual(result.returncode, 1)
        self.assertIn("obs_probe is missing and was 777777777777", result.stdout)

    def test_a_build_without_cuda_cannot_be_accepted_and_a_rehearsal_never_is(self):
        result = self.gate(receipt=HOST_RECEIPT)
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S3.3 hardware: the accepted build has no CUDA", result.stdout)
        self.assertNotIn("Sprint 3.3: ACCEPTED", result.stdout)
        # A rehearsal needs a rehearsed control: the accepted run is not one.
        result = self.gate("--host-rehearsal", receipt=HOST_RECEIPT, grade_json=REHEARSAL_JSON,
                           run_dir=self.root.parent / "evidence-no-control")
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S3.3 control: no Sprint 3.2 REHEARSED verdict", result.stdout)
        rehearsed = self.control_run("s32-devhost-20261121T220000Z", S32_REHEARSED)
        result = self.gate("--host-rehearsal", receipt=HOST_RECEIPT, grade_json=REHEARSAL_JSON,
                           run_dir=self.root.parent / "evidence-rehearsal")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertNotIn("FAIL", result.stdout)
        for line in ("HOST REHEARSAL: the host path is graded on a build without CUDA.",
                     "PASS  S3.3 control: the Sprint 3.2 run s32-devhost-20261121T220000Z is REHEARSED",
                     "NOTE  S3.3 hardware: host rehearsal on an adapter built for the host path.",
                     "PASS  S3.3 integration grade: 8 of 8 cases, 7 of 7 checks, judgment rehearsal, hardware gate open",
                     "Sprint 3.3: REHEARSED on this host, NOT ACCEPTED."):
            self.assertIn(line, result.stdout)
        self.assertIn(f"s32 run: {rehearsed}", self.calls.read_text())
        self.assertIn("--host-rehearsal --output-dir", self.calls.read_text())
        # A rehearsal whose driver reports a pass was not a rehearsal, and the reverse.
        result = self.gate("--host-rehearsal", receipt=HOST_RECEIPT, run_dir=self.root.parent / "evidence-pass")
        self.assertEqual(result.returncode, 1)
        result = self.gate(grade_json=REHEARSAL_JSON, run_dir=self.root.parent / "evidence-claim")
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S3.3 integration grade: 8 of 8 cases, 7 of 7 checks, judgment rehearsal", result.stdout)

    def test_failures_of_single_gates(self):
        failed = dict(GRADE_JSON, judgment="fail", hardware_gate="open", cases_passed=7, checks_passed=6, checks_failed=1)
        cases = (
            (dict(unit="Ran 480 tests in 4.0s\n\nFAILED (failures=1)", unit_status="1"), "FAIL  unit tests (480 ran"),
            (dict(card="Ran 0 tests in 0.0s\n\nOK"), "tests/unit/test_pilot_features.py (none ran)"),
            # Without NumPy every vector of the unit gate is skipped: such a run cannot close it.
            (dict(card="Ran 24 tests in 0.1s\n\nOK (skipped=23)"), "(24 ran, 23 skipped: the reference needs NumPy in isac"),
            (dict(card="Ran 24 tests in 3.0s\n\nOK (skipped=1)"), "(24 ran, 1 skipped: the reference needs NumPy in isac, "
                                                                 "and its live case cmake and a C++17 compiler"),
            (dict(rule="Ran 3 tests in 0.2s\n\nFAILED (failures=1)", rule_status="1"),
             "FAIL  S3.3 import rule, run directly: python tests/unit/test_core_imports.py (3 ran"),
            (dict(environment="1"), "FAIL  S3.3 references: the Sprint 1.1 build"),
            (dict(build="1"), "FAIL  S3.3 adapter: build or verification failed"),
            (dict(verify="1"), "FAIL  S3.3 adapter: build or verification failed"),
            (dict(receipt=dict(RECEIPT, executables={"grant_fixture": ACCEPTED["grant_fixture"]})),
             "FAIL  S3.3 adapter: the receipt names no pilot_ls"),
            (dict(grade=GRADE.replace("PASS  S3.3 known-return", "FAIL  S3.3 known-return"), grade_json=failed,
                  driver="1"), "FAIL  S3.3 integration grade: 7 of 8 cases, 6 of 7 checks, judgment fail"),
            (dict(grade=GRADE.replace("PASS  S3.3 live same features", "FAIL  S3.3 live same features")),
             "FAIL  S3.3 live same features"),
            (dict(no_evidence="1", driver="2"), "FAIL  S3.3 integration driver produced no graded checks"),
            (dict(grade_json=dict(GRADE_JSON, hardware_gate="open")), "judgment pass, hardware gate open"),
            (dict(show="1"), "FAIL  S3.3 a second reader recovers the target-present window of the live sensor (see "
                             "recovered.log)"),
            (dict(no_recover="1"), "FAIL  S3.3 a second reader recovers the target-present window of the live sensor (see "
                                   "recovered.log)"),
            # The view that the live sensor wrote is not the one its line names, or not that of the recorded stream.
            (dict(live="another view of the target-present window, float64, x"),
             "FAIL  S3.3 a second reader recovers the target-present window of the live sensor: its view has the digest"),
            (dict(recorded="another view"), f"its line names {VIEW_SHA256}, and the view of the recorded stream has"),
            (dict(pins="1"), "FAIL  S3.3 sources at the locked revisions after the runs"),
        )
        for number, (settings, line) in enumerate(cases):
            result = self.gate(run_dir=self.root.parent / f"evidence-{number}", **settings)
            with self.subTest(line=line):
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn(line, result.stdout)
                self.assertIn("Sprint 3.3: NOT ACCEPTED on this host", result.stdout)
                self.assertNotIn("Sprint 3.3: ACCEPTED", result.stdout)
        skipped = self.gate(run_dir=self.root.parent / "evidence-build", build="1")
        self.assertIn("SKIP  S3.3 integration driver and window recovery: no adapter", skipped.stdout)

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
        for arguments in (("--unknown",), ("--s32-run",)):
            result = self.gate(*arguments, run_dir=self.root.parent / "evidence-usage")
            self.assertEqual(result.returncode, 2)
            self.assertIn("Usage: bash scripts/accept_s3_3.sh", result.stderr)
        (self.bin / "python").unlink()
        missing = self.gate(run_dir=self.root.parent / "evidence-python")
        self.assertEqual(missing.returncode, 2)
        self.assertIn("Run 'make env' first", missing.stderr)


if __name__ == "__main__":
    unittest.main()
