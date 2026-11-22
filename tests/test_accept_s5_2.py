"""Check the Sprint 5.2 gate runner against tool doubles. No compiler, source, build, NumPy or GPU is needed."""

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

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/accept_s5_2.sh"
TEST = "tests/integrationtests/phy/upper/channel_processors/pusch_e2e_pipeline_test"
PAYLOAD = b"the estimate of the first decode, complex FP16"
PAYLOAD_SHA256 = hashlib.sha256(PAYLOAD).hexdigest()
SERIES = {"series_sha256": "23cf49ee6a4b6c77" * 4, "base_commit": "27e84cb3da65205b55dbd56ace6fac93a035331e",
          "patches": [{"file": "0001-pusch-resident-estimate-tap.patch", "sha256": "466d37cc4aeb4f36" * 4}],
          "files": {"include/tap.h": "b3" * 32, "lib/demod.cpp": "48" * 32}}
PATCHED_GRADE = """PASS  S5.2 patched receiver patch series 23cf49ee6a4b on 27e84cb3da65: 0001-pusch-resident-estimate-tap.patch 466d37cc4aeb, 2 files touched, built from a patched-worktree
PASS  S5.2 patched receiver full Release build completed (target all)
PASS  S5.2 patched receiver CUDA compilation: 14 kernel sources for sm_121
NOTE  host compiler GCC 13.3 matches the reference"""
GRADE = """PASS  S5.2 control accepted run: the Sprint 3.1 driver graded pilot_ls as pass
PASS  S5.2 flat value slot-grants: every element of 12 estimates is (1.0009765625+0j)
PASS  S5.2 noisy parity: 48 of 48 estimates meet the declared tolerance against the matched estimator
PASS  S5.2 timed account: 48 decodes of the CUDA processor: 48 published
PASS  S5.2 refusals no route: the pilot_ls of the build against the unpatched receiver refuses a plan that names an estimate
NOTE  S5.2 hook-off CUDA PUSCH service time, mean over the decodes after the first, not graded: unpatched 331.0 us
native_estimate: 10 of 10 cases passed, 5 of 5 checks"""
GRADE_JSON = {"judgment": "pass", "hardware_gate": "closed", "cases": 10, "cases_passed": 10, "checks": 5,
              "checks_passed": 5, "checks_failed": 0}
PARITY = {"schema": "o3-parity/v1", "passed": True, "patch": SERIES,
          "totals": {"numerator": 1.25e-3, "denominator": 4.0e5, "samples": 235008, "nmse": 3.125e-9},
          "runs": [{"rows": [{}] * 48}, {"rows": [{}] * 12}]}
TOOLS = {name: {"sha256": digest * 4} for name, digest in (
    ("grant_fixture", "0123456789abcdef"), ("grid_apply", "1111222233334444"), ("replay_pusch", "5555666677778888"),
    ("pilot_ls", "9999aaaabbbbcccc"), ("obs_tail", "ddddeeeeffff0000"), ("obs_feed", "1234123412341234"))}
RECEIPT = {"executables": TOOLS, "tests_run": 88, "compiler_warnings": 0, "pin": {"cuda": True}}
O3_RECEIPT = dict(RECEIPT, executables=dict(TOOLS, pilot_ls={"sha256": "0f0f1e1e2d2d3c3c" * 4}),
                  o3_route={"compiled": True, "tap_interface_version": 1}, patch=dict(SERIES, tree_kind="patched-worktree"))
CTEST = "\n".join([f" {n + 1:2d}/88 Test #{n + 1}: estimate_core_case_{n} ..............   Passed    0.00 sec"
                   for n in range(11)]
                  + ["80/88 Test #80: pilot_ls_o3_route ................   Passed    0.01 sec",
                     "100% tests passed, 0 tests failed out of 88"])
RECOVERED = {"account": {"captured": 48, "published": 0, "eligible": 48},
             "first_record": {"opportunity_id": "rz-a.0000", "shape": [2, 51, 4, 12], "payload_sha256": PAYLOAD_SHA256,
                              "element_0": [0.7421875, -0.1903076171875],
                              "producer": {"launch": 1, "start_prb": 0, "dmrs_symbols": [2, 11], "scale": 1.4125375747680664,
                                           "steps": ["pilot_ls", "frequency_interpolation", "smoothing"]}}}
S31_SUMMARY = """Sprint 3.1 acceptance on spark-ecf8, 2026-11-21T17:45:11Z
PASS  committed tree
Sprint 3.1: ACCEPTED on this DGX Spark. The observer exports the pilot least squares of the PUSCH DM-RS.
"""
UNIT = "Ran 590 tests in 420.000s\n\nOK"
CARD = "Ran 17 tests in 9.500s\n\nOK"
# The interpreter double answers each gate from FAKE_* variables and logs the call. The builders
# leave what the real ones leave: an evidence directory, the patched tree, the CTest log.
PYTHON_DOUBLE = """#!/bin/sh
[ "$1" = "-c" ] && exec {real} "$@"
echo "$*" >> "$FAKE_LOG"
case "$*" in
    "-m pip freeze --all") echo "example==1.0" ;;
    *"-m unittest discover"*) printf '%s\\n' "$FAKE_UNIT" >&2; exit "${{FAKE_UNIT_STATUS:-0}}" ;;
    "tests/unit/test_native_estimate.py -v") printf '%s\\n' "$FAKE_CARD" >&2; exit "${{FAKE_CARD_STATUS:-0}}" ;;
    "scripts/receiver_patch.py check")
        [ "${{FAKE_SERIES_STATUS:-0}}" = 0 ] || {{ echo "receiver_patch: the series names another hash" >&2; exit 1; }}
        printf '%s\\n' "$FAKE_SERIES" ;;
    *"--profile core --with-native"*) exit "${{FAKE_ENVIRONMENT:-0}}" ;;
    "scripts/native_ocudu.py build --patched --run-dir "*)
        mkdir -p "$5" build/ocudu-src-o3
        echo "Run artifacts: $5"
        [ -z "${{FAKE_RELINK:-}}" ] || ln -sfn "$PWD/build/ocudu-spark-o3/{test}" isac/bin/pusch_e2e_pipeline_test
        [ "${{FAKE_PATCHED_BUILD:-0}}" = 0 ] || echo "build: c++ exited with status 1" >&2
        exit "${{FAKE_PATCHED_BUILD:-0}}" ;;
    "scripts/native_ocudu.py grade "*) printf '%s\\n' "$FAKE_PATCHED_GRADE"; exit "${{FAKE_PATCHED_GRADE_STATUS:-0}}" ;;
    "scripts/native_adapter.py build --ocudu-build-dir "*)
        mkdir -p "$6"; printf '%s\\n' "$FAKE_CTEST" > "$6/adapter-test.log"; exit "${{FAKE_BUILD:-0}}" ;;
    "scripts/native_adapter.py verify") printf '%s\\n' "$FAKE_RECEIPT" ;;
    "scripts/native_adapter.py build --o3 --run-dir "*)
        mkdir -p "$5"; printf '%s\\n' "$FAKE_CTEST" > "$5/adapter-test.log"; exit "${{FAKE_O3_BUILD:-0}}" ;;
    "scripts/native_adapter.py verify --o3")
        [ "${{FAKE_O3_VERIFY:-0}}" = 0 ] || {{ echo "native_adapter: this adapter build has no O3 route" >&2; exit 1; }}
        printf '%s\\n' "$FAKE_O3_RECEIPT" ;;
    "tests/integration/native_estimate.py --config configs/integration/native_estimate.json --s31-run "*)
        for out; do :; done
        echo "s31 run: $5" >> "$FAKE_LOG"
        mkdir -p "$out/noisy/run/native/estimate"
        printf '%s' "${{FAKE_KEPT:-$FAKE_ESTIMATE}}" > "$out/noisy/run/native/estimate/0000.o3.fp16"
        printf '%s\\n' "$FAKE_GRADE" > "$out/grade.txt"
        printf '%s\\n' "$FAKE_GRADE_JSON" > "$out/grade.json"
        [ -n "${{FAKE_NO_PARITY:-}}" ] || printf '%s\\n' "$FAKE_PARITY" > "$out/o3_parity.json"
        exit "${{FAKE_DRIVER:-0}}" ;;
    "-m ocudu_cuda_isac.episodes.native_estimate show "*) printf '%s\\n' "$FAKE_RECOVERED" ;;
    *"check_pins.py --worktrees") echo "ext/ocudu 27e84cb3 (Git worktree)"; exit "${{FAKE_PINS:-0}}" ;;
    "scripts/receiver_patch.py verify build/ocudu-src-o3") echo "{{}}"; exit "${{FAKE_TREE:-0}}" ;;
    *) echo "interpreter double: unexpected $*" >&2; exit 64 ;;
esac
"""


@unittest.skipUnless(shutil.which("bash") and shutil.which("sha256sum"), "the gate runner is a Bash script")
class SprintFiveEstimateGateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / "host checkout"
        self.bin = self.root / "isac/bin"
        (self.root / "scripts").mkdir(parents=True)
        self.bin.mkdir(parents=True)
        shutil.copy(SCRIPT, self.root / "scripts/accept_s5_2.sh")
        python = self.bin / "python"
        python.write_text(PYTHON_DOUBLE.format(real=shlex.quote(sys.executable), test=TEST))
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

    def control_run(self, name, summary=S31_SUMMARY, manifest=True):
        run = self.root / "results/runs" / name
        (run / "pilot_ls").mkdir(parents=True)
        (run / "pilot_ls/grade.json").write_text("{}")
        if manifest:
            (run / "pilot_ls/manifest.json").write_text("{}")
        (run / "summary.txt").write_text(summary)
        return run

    def gate(self, *arguments, run_dir=None, **settings):
        env = {key: value for key, value in os.environ.items() if not key.startswith("FAKE_")}
        env.update(FAKE_LOG=str(self.calls), FAKE_UNIT=UNIT, FAKE_CARD=CARD, FAKE_GRADE=GRADE,
                   FAKE_GRADE_JSON=json.dumps(GRADE_JSON), FAKE_RECEIPT=json.dumps(RECEIPT),
                   FAKE_O3_RECEIPT=json.dumps(O3_RECEIPT), FAKE_CTEST=CTEST, FAKE_SERIES=json.dumps(SERIES),
                   FAKE_PATCHED_GRADE=PATCHED_GRADE, FAKE_PARITY=json.dumps(PARITY), FAKE_RECOVERED=json.dumps(RECOVERED),
                   FAKE_ESTIMATE=PAYLOAD.decode("ascii"))
        env.update({f"FAKE_{key.upper()}": value if isinstance(value, str) else json.dumps(value)
                    for key, value in settings.items()})
        command = [shutil.which("bash"), str(self.root / "scripts/accept_s5_2.sh"),
                   "--run-dir", str(run_dir or self.run_dir), *arguments]
        return subprocess.run(command, env=env, capture_output=True, text=True, timeout=120, check=False)

    def summary(self):
        return (self.run_dir / "summary.txt").read_text()

    def test_all_gates_pass_on_the_accepted_control(self):
        result = self.gate("--jobs", "6")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for word in ("FAIL", "SKIP"):
            self.assertNotIn(word, result.stdout)
        summary = self.summary()
        for line in ("PASS  unit tests (590 ran, 0 skipped)",
                     "PASS  S5.2 unit gate, run directly: python tests/unit/test_native_estimate.py (17 ran, 0 skipped)",
                     "PASS  S5.2 patch series: patches/ocudu/series.json 23cf49ee6a4b6c77 on 27e84cb3da65: "
                     "0001-pusch-resident-estimate-tap.patch 466d37cc4aeb4f36; 2 files touched",
                     "PASS  S5.2 control: the Sprint 3.1 run s31-spark-ecf8-20261121T174511Z is ACCEPTED on this DGX Spark",
                     "PASS  S5.2 references: the Sprint 1.1 build, from the project interpreter (environment.json)",
                     "PASS  S5.2 patched receiver: build command (patched-build.log, evidence in patched-build/)",
                     "PASS  S5.2 patched receiver patch series 23cf49ee6a4b on 27e84cb3da65",
                     "PASS  S5.2 hook-free control: the accepted Sprint 1.1 build",
                     "PASS  S5.2 control adapter: built against the Sprint 1.1 build and verified, pilot_ls 9999aaaabbbb "
                     "without the route, with ENABLE_CUDA, 88 adapter tests, 0 compiler warnings; 11 cases of the capture "
                     "core passed, and pilot_ls reports no route",
                     "PASS  S5.2 O3 adapter: built against the patched receiver and verified, pilot_ls 0f0f1e1e2d2d with "
                     "the route, tap interface version 1, series 23cf49ee6a4b6c77 built from a patched-worktree",
                     "PASS  S5.2 hardware: the O3 adapter is linked against the CUDA build of the patched receiver",
                     "PASS  S5.2 noisy parity: 48 of 48 estimates meet the declared tolerance",
                     "NOTE  S5.2 hook-off CUDA PUSCH service time",
                     "PASS  S5.2 integration grade: 10 of 10 cases, 5 of 5 checks, judgment pass, hardware gate closed",
                     "PASS  S5.2 parity record: 60 estimates of 2 runs on the pilot support: numerator 1.250000e-03, "
                     "denominator 4.000000e+05, 235008 samples, NMSE 3.125e-09, against the series 23cf49ee6a4b6c77",
                     "PASS  S5.2 a second reader recovers an estimate: 48 estimates of 48 decodes of the CUDA processor "
                     "captured; the first is the estimate of rz-a.0000 in the shape [2, 51, 4, 12], launch 1 of the "
                     "producer, from resource block 0 on DM-RS symbols [2, 11], after pilot_ls, frequency_interpolation, "
                     "smoothing, scale 1.4125376, element 0 = +0.7421875-0.1903076j; SHA-256 of 0000.o3.fp16 "
                     f"{PAYLOAD_SHA256[:16]} by sha256sum",
                     "PASS  S5.2 sources at the locked revisions after the runs",
                     "PASS  S5.2 patched tree: build/ocudu-src-o3 is the pinned commit with exactly the series applied",
                     "Sprint 5.2: ACCEPTED on this DGX Spark. The receiver's own CUDA channel estimate leaves the patched "
                     "receiver as the stage O3",
                     "Next: S3.5 cuSense-like pipeline on O3 records."):
            self.assertIn(line, summary)
        calls = self.calls.read_text()
        self.assertIn(f"s31 run: {self.s31}", calls)
        self.assertIn(f"scripts/native_ocudu.py build --patched --run-dir {self.run_dir}/patched-build --jobs 6", calls)
        self.assertIn(f"scripts/native_adapter.py build --ocudu-build-dir {self.build}", calls)
        self.assertIn("scripts/native_adapter.py build --o3", calls)
        self.assertEqual(summary.count("PASS  "), 23)

    def test_a_failed_patched_build_closes_nothing_and_skips_the_route(self):
        result = self.gate(patched_build="1")
        self.assertEqual(result.returncode, 1)
        summary = self.summary()
        self.assertIn("FAIL  S5.2 patched receiver: build command exited with status 1", summary)
        self.assertIn("SKIP  S5.2 O3 adapter: no patched receiver", summary)
        self.assertIn("SKIP  S5.2 integration driver, parity record and record recovery: no pair of adapters", summary)
        self.assertIn("Sprint 5.2: NOT ACCEPTED on this host", summary)
        self.assertNotIn("tests/integration/native_estimate.py", self.calls.read_text())
        # A build whose grade does not name its series is not a patched receiver either.
        graded = self.gate(run_dir=self.root.parent / "ungraded",
                           patched_grade=PATCHED_GRADE.replace("PASS  S5.2 patched receiver patch series",
                                                               "FAIL  S5.2 patched receiver patch series"))
        self.assertEqual(graded.returncode, 1)
        self.assertIn("FAIL  S5.2 patched receiver patch series", graded.stdout)

    def test_the_accepted_build_and_its_link_must_survive_the_patched_build(self):
        result = self.gate(relink="1")
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S5.2 hook-free control: the receipt of the accepted build or the link in isac/bin changed",
                      self.summary())

    def test_an_adapter_without_its_role_fails(self):
        result = self.gate(receipt=O3_RECEIPT)
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S5.2 control adapter: the receipt is unusable", self.summary())
        plain = self.gate(run_dir=self.root.parent / "no-route", o3_verify="1")
        self.assertEqual(plain.returncode, 1)
        self.assertIn("FAIL  S5.2 O3 adapter: build or verification failed", plain.stdout)
        host = self.gate(run_dir=self.root.parent / "host", o3_receipt=dict(O3_RECEIPT, pin={"cuda": False}))
        self.assertEqual(host.returncode, 1)
        self.assertIn("FAIL  S5.2 O3 adapter: the receipt names no route, no patch or no CUDA", host.stdout)

    def test_a_failed_check_a_missing_parity_and_a_changed_payload_fail(self):
        failing = self.gate(grade=GRADE.replace("PASS  S5.2 noisy parity", "FAIL  S5.2 noisy parity"),
                            grade_json=dict(GRADE_JSON, judgment="fail", checks_failed=1, cases_passed=9), driver="1")
        self.assertEqual(failing.returncode, 1)
        self.assertIn("FAIL  S5.2 noisy parity", failing.stdout)
        self.assertIn("FAIL  S5.2 integration grade: 9 of 10 cases", failing.stdout)
        missing = self.gate(run_dir=self.root.parent / "no-parity", no_parity="1")
        self.assertEqual(missing.returncode, 1)
        self.assertIn("FAIL  S5.2 parity record: o3_parity.json is missing or unreadable", missing.stdout)
        unmet = self.gate(run_dir=self.root.parent / "unmet", parity=dict(PARITY, passed=False))
        self.assertEqual(unmet.returncode, 1)
        self.assertIn("FAIL  S5.2 parity record: 60 estimates of 2 runs", unmet.stdout)
        changed = self.gate(run_dir=self.root.parent / "changed", kept="another payload")
        self.assertEqual(changed.returncode, 1)
        self.assertIn("FAIL  S5.2 a second reader recovers an estimate: 0000.o3.fp16 has the digest", changed.stdout)

    def test_the_control_must_be_accepted_on_a_spark_and_evidence_is_never_overwritten(self):
        shutil.rmtree(self.s31)
        self.control_run("s31-devhost-20261119T000000Z", S31_SUMMARY.replace("DGX Spark", "host"))
        self.control_run("s31-spark-ecf8-20261121T180000Z", manifest=False)
        result = self.gate()
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S5.2 control: no Sprint 3.1 verdict ACCEPTED on this DGX Spark", result.stdout)
        self.assertIn("SKIP  S5.2 patched receiver, adapters, integration driver and record recovery: no control",
                      result.stdout)
        again = self.gate()
        self.assertEqual(again.returncode, 2)
        self.assertIn("already exists. Evidence is never overwritten.", again.stderr)
        self.assertEqual(self.gate("--jobs", "0", run_dir=self.root.parent / "usage").returncode, 2)
        self.assertEqual(self.gate("--host-rehearsal", run_dir=self.root.parent / "usage").returncode, 2)


if __name__ == "__main__":
    unittest.main()
