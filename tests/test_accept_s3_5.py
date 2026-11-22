"""Check the Sprint 3.5 gate runner against tool doubles. No compiler, source, build, NumPy, PyTorch or GPU is needed."""

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

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/accept_s3_5.sh"
TEST = "tests/integrationtests/phy/upper/channel_processors/pusch_e2e_pipeline_test"
WEIGHTS = "forty-three tensors as float32"
TEMPLATE = "four antennas, 612 subcarriers and two symbols as complex64"
WEIGHTS_SHA256 = hashlib.sha256(WEIGHTS.encode("ascii")).hexdigest()
TEMPLATE_SHA256 = hashlib.sha256(TEMPLATE.encode("ascii")).hexdigest()
ARTIFACT_SHA256 = "a1b2c3d4e5f60718" * 4
GRADE = """PASS  S3.5 control accepted run: the Sprint 3.4 driver graded presence_head as pass
PASS  S3.5 split episode set: the set demo-set-v1 (status proposed) verifies with every episode in it
PASS  S3.5 records account: 14 runs, 16320 opportunities: stage O2: 16320 eligible, 16320 published, 0 dropped
PASS  S3.5 windows oracle: 28 of 28 windows of the stage O3 agree with the coherent average computed again
PASS  S3.5 train statistics: the template H_B, the complex mean of the records behind the 2400 target-absent
PASS  S3.5 second-input fit: the joint loss falls from 1.635268
PASS  S3.5 freeze no test access: the manifests of both arms list 600 training and 80 validation episode keys
PASS  S3.5 live sensor: python -m ocudu_cuda_isac.sensing.cusense follow, a second process
PASS  S3.5 parity scores: 160 held-out windows joined by their identifier
PASS  S3.5 evaluation reference: cusense_reference.json records the measured position error
NOTE  S3.5 evaluation held-out episodes at the frozen thresholds, read with the truth after the decisions were written
cusense_frontend: 12 of 12 cases passed, 10 of 10 checks"""
GRADE_JSON = {"judgment": "pass", "hardware_gate": "closed", "cases": 12, "cases_passed": 12, "checks": 10,
              "checks_passed": 10, "checks_failed": 0}
ACCEPTED = {"grant_fixture": {"sha256": "0123456789abcdef" * 4}, "grid_apply": {"sha256": "1111222233334444" * 4},
            "replay_pusch": {"sha256": "5555666677778888" * 4}, "pilot_ls": {"sha256": "9999aaaabbbbcccc" * 4},
            "obs_tail": {"sha256": "ddddeeeeffff0000" * 4}, "obs_feed": {"sha256": "1234123412341234" * 4}}
SERIES = {"series_sha256": "23cf49ee6a4b6c77" * 4, "base_commit": "27e84cb3d2c1" + "0" * 28,
          "patches": [{"file": "0001-resident-estimate-tap.patch", "sha256": "ab" * 32}], "files": ["a.h", "b.cu", "c.cpp"]}
RECEIPT = {"executables": ACCEPTED, "tests_run": 77, "compiler_warnings": 0, "pin": {"cuda": True, "archives": 50},
           "patch": {"series_sha256": SERIES["series_sha256"], "tree_kind": "patched-worktree"},
           "o3_route": {"compiled": True, "tap_interface_version": 1}}
DEVICE = {"device": "cuda:0", "type": "cuda", "torch": "2.15.0", "cuda_runtime": "13.2", "cuda_available": True,
          "name": "NVIDIA GB10", "capability": "12.1"}
CPU_DEVICE = {"device": "cpu", "type": "cpu", "torch": "2.15.0", "cuda_runtime": "13.2", "cuda_available": False}
RECOVER = {"artifact": "train/o3/artifact", "freeze": "train/o3/freeze.json", "sensor": "live/held-out-00/sensor",
           "window": 0, "offline": "parity/held-out-00", "second_input": "train/o2/artifact",
           "evaluation": "cusense_eval.json", "reference": "cusense_reference.json"}
FREEZE = {"artifact": {"sha256": ARTIFACT_SHA256, "weights_sha256": WEIGHTS_SHA256, "template_sha256": TEMPLATE_SHA256},
          "frozen_utc": "2026-11-22T09:10:00Z"}
INSPECTED = {"model_id": "cusense-o3-v1", "artifact_sha256": ARTIFACT_SHA256, "entries": {"input": "o3_fp16"},
             "input_shape": [1, 4, 612], "weights_sha256": WEIGHTS_SHA256,
             "grid": {"rows": 50, "cols": 34}, "files": ["model.json", "template.c64", "template.count.u4", "weights.f32"],
             "threshold": {"logit": -13.141029, "negatives": 320, "allowed": 16, "exceeding": 16}}
SHOWN = {"model_id": "cusense-o3-v1", "artifact_sha256": ARTIFACT_SHA256, "runtime": "reference", "device": "cuda:0",
         "loads": 1, "scored_windows": 160,
         "window": {"id": "1/89abcdef01234567/0", "contributing": 3, "logit": 9.25, "probability": 0.9999, "present": True,
                    "map_sum": 1.0, "map_peak": 0.0421, "cell": [21, 12], "position_m": [3.1, 5.8]}}
STATISTICS = {"eligible": 80, "mean_m": 1.895, "median_m": 0.911, "p90_m": 5.241, "rmse_m": 2.642, "max_m": 6.082}
EVAL = {"schema": "cusense-eval/v1", "graded": False, "role": "held_out", "arms": {"o3": {
    "input": "o3_fp16",
    "presence": {"episodes": {"target_present": {"total": 10, "detected": 10, "missed": 0, "missing": 0},
                              "target_absent": {"total": 10, "false_alarm": 1, "quiet": 9, "missing": 0}}},
    "position": {"unit": "m", "raw": STATISTICS, "smoothed": dict(STATISTICS, mean_m=1.801),
                 "kalman": dict(STATISTICS, mean_m=1.742)}}}}
REFERENCE = {"schema": "cusense-reference/v1", "graded": False,
             "published": {"position_error_cm": {"mean": 73.3}, "live_inference_us": {"value": 241}},
             "measured": {"latency": {"runtime": "reference"}},
             "differences": ["a simulated scene", "20 MHz against 100 MHz", "a GB10 against a GH200"]}
S34_SUMMARY = """Sprint 3.4 acceptance on spark-ecf8, 2026-11-22T02:44:32Z
PASS  committed tree
Sprint 3.4: ACCEPTED on this DGX Spark. One linear presence head on the delay-Doppler features was trained.
"""
S52_SUMMARY = """Sprint 5.2 acceptance on spark-ecf8, 2026-11-22T04:53:18Z
PASS  committed tree
Sprint 5.2: ACCEPTED on this DGX Spark. The receiver's own CUDA channel estimate leaves the patched receiver.
"""
UNIT = "Ran 660 tests in 980.000s\n\nOK"
CARD = "Ran 58 tests in 95.500s\n\nOK"
RULE = "Ran 3 tests in 1.200s\n\nOK"
# The interpreter double answers each gate from FAKE_* variables and logs the call. The driver
# double leaves its grade, the place of what a second reader recovers, the freeze record, the
# weights and the template of the artifact and the two records of the held-out episodes.
PYTHON_DOUBLE = """#!/bin/sh
[ "$1" = "-c" ] && exec {real} "$@"
echo "$*" >> "$FAKE_LOG"
case "$*" in
    "-m pip freeze --all") echo "example==1.0" ;;
    *"-m unittest discover"*) printf '%s\\n' "$FAKE_UNIT" >&2; exit "${{FAKE_UNIT_STATUS:-0}}" ;;
    "tests/unit/test_cusense_frontend.py -v") printf '%s\\n' "$FAKE_CARD" >&2; exit "${{FAKE_CARD_STATUS:-0}}" ;;
    "tests/unit/test_core_imports.py -v") printf '%s\\n' "$FAKE_RULE" >&2; exit "${{FAKE_RULE_STATUS:-0}}" ;;
    "scripts/receiver_patch.py check")
        [ "${{FAKE_PATCH:-0}}" = 0 ] || {{ echo "receiver_patch: a patch has another hash than its manifest names" >&2; exit 1; }}
        printf '%s\\n' "$FAKE_SERIES" ;;
    "scripts/receiver_patch.py verify build/ocudu-src-o3") echo "{{}}"; exit "${{FAKE_TREE:-0}}" ;;
    *"--profile core --with-native"*) exit "${{FAKE_ENVIRONMENT:-0}}" ;;
    "scripts/native_adapter.py build --o3 --run-dir "*)
        echo "Adapter build: build/ocudu-adapter-o3"
        mkdir -p "$5"
        [ "${{FAKE_BUILD:-0}}" = 0 ] || echo "native_adapter: the adapter build step exited with status 2" >&2
        exit "${{FAKE_BUILD:-0}}" ;;
    "scripts/native_adapter.py verify --o3")
        [ "${{FAKE_VERIFY:-0}}" = 0 ] || {{ echo "native_adapter: the adapter sources changed after the build" >&2; exit 1; }}
        printf '%s\\n' "$FAKE_RECEIPT" ;;
    "-m ocudu_cuda_isac.sensing.cusense device")
        [ -z "${{FAKE_NO_TORCH:-}}" ] || {{ echo "No module named 'torch'" >&2; exit 1; }}
        printf '%s\\n' "$FAKE_DEVICE" ;;
    "tests/integration/cusense_frontend.py --config configs/integration/cusense_frontend.json --s34-run "*)
        for out; do :; done
        echo "s34 run: $5" >> "$FAKE_LOG"
        echo "episode set: $7" >> "$FAKE_LOG"
        echo "Run artifacts: $out"
        [ -z "${{FAKE_NO_EVIDENCE:-}}" ] || exit "${{FAKE_DRIVER:-0}}"
        mkdir -p "$out/train/o3/artifact" "$out/live/held-out-00/sensor"
        printf '%s' "$FAKE_WEIGHTS" > "$out/train/o3/artifact/weights.f32"
        printf '%s' "$FAKE_TEMPLATE" > "$out/train/o3/artifact/template.c64"
        printf '%s\\n' "$FAKE_FREEZE" > "$out/train/o3/freeze.json"
        [ -n "${{FAKE_NO_RECOVER:-}}" ] || printf '%s\\n' "$FAKE_RECOVER" > "$out/recover.json"
        [ -n "${{FAKE_NO_EVAL:-}}" ] || printf '%s\\n' "$FAKE_EVAL" > "$out/cusense_eval.json"
        [ -n "${{FAKE_NO_REFERENCE:-}}" ] || printf '%s\\n' "$FAKE_REFERENCE" > "$out/cusense_reference.json"
        printf '%s\\n' "$FAKE_GRADE" > "$out/grade.txt"
        printf '%s\\n' "$FAKE_GRADE_JSON" > "$out/grade.json"
        exit "${{FAKE_DRIVER:-0}}" ;;
    "-m ocudu_cuda_isac.sensing.cusense inspect "*)
        [ "${{FAKE_INSPECT:-0}}" = 0 ] || {{ echo "cusense: the artifact has another hash than the one asked for" >&2; exit 2; }}
        printf '%s\\n' "$FAKE_INSPECTED" ;;
    "-m ocudu_cuda_isac.sensing.cusense show "*)
        [ "${{FAKE_SHOW:-0}}" = 0 ] || {{ echo "cusense: decision 7: the digest of its input does not agree" >&2; exit 2; }}
        printf '%s\\n' "$FAKE_SHOWN" ;;
    *"check_pins.py --worktrees") echo "ext/ocudu 27e84cb3 (Git worktree)"; exit "${{FAKE_PINS:-0}}" ;;
    *) echo "interpreter double: unexpected $*" >&2; exit 64 ;;
esac
"""


@unittest.skipUnless(shutil.which("bash") and shutil.which("sha256sum"), "the gate runner is a Bash script")
class SprintThreeCuSenseGateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / "host checkout"
        self.bin = self.root / "isac/bin"
        (self.root / "scripts").mkdir(parents=True)
        self.bin.mkdir(parents=True)
        shutil.copy(SCRIPT, self.root / "scripts/accept_s3_5.sh")
        python = self.bin / "python"
        python.write_text(PYTHON_DOUBLE.format(real=shlex.quote(sys.executable)))
        python.chmod(0o755)
        # The patched receiver that Sprint 5.2 built, the runs that Sprint 3.4 and Sprint 5.2 accepted, and
        # the episode set that the Sprint 3.4 run was graded on.
        self.patched = self.root / "build/ocudu-spark-o3"
        self.patched.mkdir(parents=True)
        self.set = self.episode_set("demo-set-20261122T022210Z")
        self.s34 = self.control_run("s34-spark-ecf8-20261122T024432Z")
        self.s52 = self.estimate_run("s52-spark-ecf8-20261122T045318Z")
        self.calls = self.root.parent / "calls.log"
        self.run_dir = self.root.parent / "evidence"

    def control_run(self, name, summary=S34_SUMMARY, head=True, episode_set=None):
        run = self.root / "results/runs" / name
        out = run / "presence_head"
        (out / "train/artifact").mkdir(parents=True)
        (out / "grade.json").write_text("{}")
        (out / "spec.json").write_text("{}")
        (out / "manifest.json").write_text(json.dumps({"episode_set": {"set_id": "demo-set-v1",
                                                                       "directory": str(episode_set or self.set)}}))
        if head:
            (out / "train/artifact/head.json").write_text("{}")
        (run / "summary.txt").write_text(summary)
        return run

    def estimate_run(self, name, summary=S52_SUMMARY, receipt=True, executables=None):
        run = self.root / "results/runs" / name
        run.mkdir(parents=True)
        if receipt:
            (run / "o3-adapter-receipt.json").write_text(json.dumps({"executables": executables or ACCEPTED}))
        (run / "summary.txt").write_text(summary)
        return run

    def episode_set(self, name, manifest=True):
        directory = self.root / "data/episodes" / name
        (directory / "episodes").mkdir(parents=True)
        if manifest:
            (directory / "manifest.json").write_text("{}")
        return directory

    def gate(self, *arguments, run_dir=None, **settings):
        env = {key: value for key, value in os.environ.items() if not key.startswith("FAKE_")}
        env.update(FAKE_LOG=str(self.calls), FAKE_UNIT=UNIT, FAKE_CARD=CARD, FAKE_RULE=RULE, FAKE_GRADE=GRADE,
                   FAKE_GRADE_JSON=json.dumps(GRADE_JSON), FAKE_RECEIPT=json.dumps(RECEIPT), FAKE_DEVICE=json.dumps(DEVICE),
                   FAKE_RECOVER=json.dumps(RECOVER), FAKE_FREEZE=json.dumps(FREEZE), FAKE_INSPECTED=json.dumps(INSPECTED),
                   FAKE_SHOWN=json.dumps(SHOWN), FAKE_WEIGHTS=WEIGHTS, FAKE_TEMPLATE=TEMPLATE, FAKE_SERIES=json.dumps(SERIES),
                   FAKE_EVAL=json.dumps(EVAL), FAKE_REFERENCE=json.dumps(REFERENCE))
        env.update({f"FAKE_{key.upper()}": value if isinstance(value, str) else json.dumps(value)
                    for key, value in settings.items()})
        command = [shutil.which("bash"), str(self.root / "scripts/accept_s3_5.sh"),
                   "--run-dir", str(run_dir or self.run_dir), *arguments]
        return subprocess.run(command, env=env, capture_output=True, text=True, timeout=120, check=False)

    def summary(self):
        return (self.run_dir / "summary.txt").read_text()

    def test_all_gates_pass_on_the_accepted_controls(self):
        result = self.gate()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for word in ("FAIL", "SKIP", "REHEARS"):
            self.assertNotIn(word, result.stdout)
        summary = self.summary()
        self.assertEqual(summary, result.stdout)
        for line in ("PASS  unit tests (660 ran, 0 skipped)",
                     "PASS  S3.5 unit gate, run directly: python tests/unit/test_cusense_frontend.py (58 ran, 0 skipped)",
                     "PASS  S3.5 import rule, run directly: python tests/unit/test_core_imports.py (3 ran, 0 skipped)",
                     "PASS  S3.5 patch series: patches/ocudu/series.json 23cf49ee6a4b6c77 on 27e84cb3d2c1, 1 patch "
                     "file(s), 3 files touched",
                     "PASS  S3.5 control: the Sprint 3.4 run s34-spark-ecf8-20261122T024432Z is ACCEPTED on this DGX Spark "
                     "and holds its grade, its manifest and its frozen head",
                     "PASS  S3.5 estimate: the Sprint 5.2 run s52-spark-ecf8-20261122T045318Z is ACCEPTED on this DGX "
                     "Spark and holds the receipt of the adapter with the O3 route",
                     "PASS  S3.5 stimulus: the episode set demo-set-20261122T022210Z holds a manifest and its episodes",
                     "PASS  S3.5 references: the Sprint 1.1 build, from the project interpreter (environment.json)",
                     "PASS  S3.5 adapter: built against the patched receiver of Sprint 5.2 and verified, pilot_ls "
                     "9999aaaabbbb with the route, tap interface version 1, obs_tail ddddeeeeffff, series "
                     "23cf49ee6a4b6c77, with ENABLE_CUDA, 77 adapter tests, 0 compiler warnings",
                     "PASS  S3.5 hardware: the adapter is linked against the CUDA build of the patched receiver",
                     "PASS  S3.5 accepted executables: this sprint changes no native source, and every tool has the bytes "
                     "of the Sprint 5.2 run: grant_fixture 0123456789ab, grid_apply 111122223333, obs_feed 123412341234, "
                     "obs_tail ddddeeeeffff, pilot_ls 9999aaaabbbb, replay_pusch 555566667777",
                     "PASS  S3.5 device: PyTorch 2.15.0 reaches cuda:0 (NVIDIA GB10, capability 12.1, CUDA runtime 13.2): "
                     "the model is trained there and scored in a second CUDA process (device.json)",
                     "cuSense-like branch on the episode set demo-set-20261122T022210Z, on the control "
                     "s34-spark-ecf8-20261122T024432Z, with configs/integration/cusense_frontend.json.",
                     "PASS  S3.5 parity scores: 160 held-out windows joined by their identifier",
                     "NOTE  S3.5 evaluation held-out episodes at the frozen thresholds",
                     "PASS  S3.5 integration grade: 12 of 12 cases, 10 of 10 checks, judgment pass, hardware gate closed",
                     "PASS  S3.5 a second reader recovers the frozen model and a held-out decision: the model "
                     "cusense-o3-v1 reads the input entry o3_fp16 of shape [1, 4, 612] and answers on a grid of 50 by 34 "
                     "cells, from the files model.json, template.c64, template.count.u4, weights.f32; its threshold is "
                     "the logit -13.1410, from 320 target-absent validation windows with 16 of 16 allowed above it; the "
                     "sensor loaded it 1 time(s) on cuda:0 in the runtime reference and scored 160 windows: window "
                     "1/89abcdef01234567/0 from 3 observations has the logit 9.2500 and the probability 0.9999, above the "
                     "threshold, and a map that sums to 1.0000 with its largest value 0.0421 in the cell [21, 12] at "
                     f"3.10, 5.80 m; the artifact {ARTIFACT_SHA256[:16]} is the one the freeze record of "
                     f"2026-11-22T09:10:00Z names, and its weights and its template have the SHA-256 "
                     f"{WEIGHTS_SHA256[:16]} and {TEMPLATE_SHA256[:16]} by sha256sum",
                     "PASS  S3.5 records, written and not graded: cusense_eval.json holds, for the input entry o3_fp16, "
                     "10 of 10 target-present held-out episodes detected and 1 of 10 target-absent alarmed, and a "
                     "position error with a mean of 1.895 m, a median of 0.911 m and a 90th percentile of 5.241 m over 80 "
                     "windows, 1.742 m in the mean after the Kalman filter; cusense_reference.json puts that beside the "
                     "published mean of 73.3 cm with \"graded\": false and lists 3 differences; the latency carries the "
                     "runtime reference and is not compared with the published 241 us",
                     "PASS  S3.5 sources at the locked revisions after the runs",
                     "Sprint 3.5: ACCEPTED on this DGX Spark. The cuSense-like branch runs on the receiver's own estimate",
                     "frozen before a held-out entry was replayed.",
                     "the linear head of s34-spark-ecf8-20261122T024432Z were scored on the records of the same run.",
                     "are not graded. Next: S3.6 native sensing core."):
            self.assertIn(line, summary)
        calls = self.calls.read_text()
        self.assertIn(f"s34 run: {self.s34}", calls)
        self.assertIn(f"episode set: {self.set}", calls)
        self.assertIn(f"scripts/native_adapter.py build --o3 --run-dir {self.run_dir}/adapter-build", calls)
        self.assertIn("scripts/native_adapter.py verify --o3", calls)
        self.assertIn(f"-m ocudu_cuda_isac.sensing.cusense inspect {self.run_dir}/cusense_frontend/train/o3/artifact "
                      f"--sha256 {ARTIFACT_SHA256}", calls)
        self.assertIn(f"-m ocudu_cuda_isac.sensing.cusense show {self.run_dir}/cusense_frontend/live/held-out-00/sensor "
                      "--window 0", calls)
        # The receiver is never rebuilt, no episode is traced and there is no rehearsal.
        for word in ("native_ocudu.py", "export_rt_episode", "--host-rehearsal", "--patched"):
            self.assertNotIn(word, calls)
        self.assertEqual(summary.count("PASS  "), 26)

    def test_the_newest_accepted_runs_and_the_set_of_the_control_are_used(self):
        self.control_run("s34-spark-ecf8-20261122T030000Z", S34_SUMMARY.replace("PASS  committed", "FAIL  committed"))
        self.control_run("s34-spark-ecf8-20261122T040000Z", "Sprint 3.4: NOT ACCEPTED on this host\n")
        self.control_run("s34-spark-ecf8-20261122T050000Z", head=False)
        # A run that was accepted on another host, and a rehearsal, are no control of this gate.
        self.control_run("s34-devhost-20261122T060000Z", S34_SUMMARY.replace("this DGX Spark", "this host"))
        self.control_run("s34-devhost-20261122T070000Z", "Sprint 3.4: REHEARSED on this host, NOT ACCEPTED.\n")
        other_set = self.episode_set("demo-set-20261122T080000Z")
        newest = self.control_run("s34-spark-ecf8-20261122T025500Z", episode_set=other_set)
        self.estimate_run("s52-spark-ecf8-20261122T050000Z", receipt=False)
        self.estimate_run("s52-spark-ecf8-20261122T060000Z", "Sprint 5.2: NOT ACCEPTED on this host\n")
        result = self.gate()
        self.assertEqual(result.returncode, 0, result.stdout)
        # The stimulus is the set that the chosen Sprint 3.4 run names, not the newest one under data/episodes.
        self.episode_set("demo-set-20261122T090000Z")
        self.assertIn(f"s34 run: {newest}", self.calls.read_text())
        self.assertIn(f"episode set: {other_set}", self.calls.read_text())
        self.assertIn("the Sprint 5.2 run s52-spark-ecf8-20261122T045318Z is ACCEPTED", result.stdout)
        named = self.episode_set("demo-set-named")
        result = self.gate("--s34-run", str(self.s34), "--s52-run", str(self.s52), "--episode-set", str(named),
                           run_dir=self.root.parent / "evidence-named")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn(f"s34 run: {self.s34}", self.calls.read_text())
        self.assertIn(f"episode set: {named}", self.calls.read_text())

    def test_without_a_control_a_set_or_the_patched_receiver_nothing_is_computed(self):
        shutil.rmtree(self.s34)
        result = self.gate()
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S3.5 control: no Sprint 3.4 verdict ACCEPTED on this DGX Spark with its grade, its manifest "
                      "and its frozen head in results/runs/s34-*", result.stdout)
        self.assertIn("Run 'make accept-s3.4' on this host first", result.stdout)
        self.assertIn("SKIP  S3.5 reference check, adapter build, integration driver, artifact recovery and records: no "
                      "control or no episode set", result.stdout)
        self.assertIn("Sprint 3.5: NOT ACCEPTED on this host", result.stdout)
        self.assertNotIn("tests/integration/cusense_frontend.py", self.calls.read_text())
        # Without the accepted Sprint 5.2 run there is no adapter to compare with.
        self.control_run("s34-spark-ecf8-20261122T024432Z")
        shutil.rmtree(self.s52)
        result = self.gate(run_dir=self.root.parent / "evidence-no-estimate")
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S3.5 estimate: no Sprint 5.2 verdict ACCEPTED on this DGX Spark with the receipt of its O3 "
                      "adapter in results/runs/s52-*", result.stdout)
        self.assertIn("Run 'make accept-s5.2' on this host first", result.stdout)
        # With both controls and without the set the Sprint 3.4 run names, the gate says so and traces none.
        self.estimate_run("s52-spark-ecf8-20261122T045318Z")
        shutil.rmtree(self.set)
        result = self.gate(run_dir=self.root.parent / "evidence-no-set")
        self.assertEqual(result.returncode, 1)
        self.assertIn(f"FAIL  S3.5 stimulus: no episode set with a manifest in {self.set}", result.stdout)
        self.assertIn("must still be where it was, or be named with --episode-set. This gate traces none.", result.stdout)
        self.assertNotIn("export_rt_episode", self.calls.read_text())
        # With everything but the patched receiver, the gate does not build one.
        self.episode_set("demo-set-20261122T022210Z")
        shutil.rmtree(self.patched)
        result = self.gate(run_dir=self.root.parent / "evidence-no-build")
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S3.5 needs the patched receiver of Sprint 5.2: build/ocudu-spark-o3 is missing", result.stdout)
        self.assertIn("SKIP  S3.5 reference check, adapter build, integration driver, artifact recovery and records: no "
                      "patched receiver", result.stdout)
        self.assertNotIn("native_adapter.py", self.calls.read_text())
        self.assertNotIn("tests/integration/cusense_frontend.py", self.calls.read_text())

    def test_a_tool_with_other_bytes_fails(self):
        self.estimate_run("s52-spark-ecf8-20261122T050000Z", executables=dict(ACCEPTED, pilot_ls={"sha256": "9" * 64}))
        result = self.gate()
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S3.5 accepted executables: this sprint changes no native source, so every tool must keep "
                      "the bytes of the Sprint 5.2 run: pilot_ls is 9999aaaabbbb and was 999999999999", result.stdout)

    def test_without_the_route_or_a_cuda_device_the_gate_cannot_be_accepted(self):
        cases = ((dict(receipt={key: value for key, value in RECEIPT.items() if key != "o3_route"}),
                  "FAIL  S3.5 adapter: the receipt names no route to the receiver's estimate, no patch, no CUDA"),
                 (dict(receipt={key: value for key, value in RECEIPT.items() if key != "patch"}),
                  "FAIL  S3.5 adapter: the receipt names no route to the receiver's estimate, no patch, no CUDA"),
                 (dict(receipt=dict(RECEIPT, pin={"cuda": False, "archives": 50})),
                  "FAIL  S3.5 adapter: the receipt names no route to the receiver's estimate, no patch, no CUDA"),
                 (dict(device=CPU_DEVICE), "FAIL  S3.5 device: PyTorch 2.15.0 reaches no CUDA device in isac, so the model "
                                           "cannot be scored on the selected device"),
                 (dict(no_torch="1"), "FAIL  S3.5 device: PyTorch does not load in isac"))
        for number, (settings, line) in enumerate(cases):
            result = self.gate(run_dir=self.root.parent / f"evidence-{number}", **settings)
            with self.subTest(line=line):
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn(line, result.stdout)
                self.assertNotIn("Sprint 3.5: ACCEPTED", result.stdout)
        self.assertIn("No module named 'torch'", result.stdout)
        # There is no rehearsal of this gate: the option is not one of its options.
        result = self.gate("--host-rehearsal", run_dir=self.root.parent / "evidence-rehearsal")
        self.assertEqual(result.returncode, 2)
        self.assertIn("There is no host rehearsal: the receiver's estimate exists only on a CUDA device.", result.stderr)

    def test_failures_of_single_gates(self):
        failed = dict(GRADE_JSON, judgment="fail", hardware_gate="open", cases_passed=11, checks_passed=9, checks_failed=1)
        again = "FAIL  S3.5 a second reader recovers the frozen model and a held-out decision (see recovered.log)"
        cases = (
            (dict(unit="Ran 660 tests in 4.0s\n\nFAILED (failures=1)", unit_status="1"), "FAIL  unit tests (660 ran"),
            (dict(card="Ran 0 tests in 0.0s\n\nOK"), "tests/unit/test_cusense_frontend.py (none ran)"),
            # Without PyTorch every vector of the unit gate is skipped: such a run cannot close it.
            (dict(card="Ran 58 tests in 0.1s\n\nOK (skipped=55)"),
             "(58 ran, 55 skipped: the branch needs NumPy and PyTorch in isac, 'make env-full'"),
            (dict(card="Ran 58 tests in 9.0s\n\nOK (skipped=1)"), "and its live case cmake and a C++17 compiler"),
            (dict(rule="Ran 3 tests in 0.2s\n\nFAILED (failures=1)", rule_status="1"),
             "FAIL  S3.5 import rule, run directly: python tests/unit/test_core_imports.py (3 ran"),
            (dict(patch="1"), "FAIL  S3.5 patch series: the manifest disagrees with its patch files or with the lock"),
            (dict(environment="1"), "FAIL  S3.5 references: the Sprint 1.1 build"),
            (dict(build="1"), "FAIL  S3.5 adapter: build or verification failed"),
            (dict(verify="1"), "FAIL  S3.5 adapter: build or verification failed"),
            (dict(grade=GRADE.replace("PASS  S3.5 parity scores", "FAIL  S3.5 parity scores"), grade_json=failed,
                  driver="1"), "FAIL  S3.5 integration grade: 11 of 12 cases, 9 of 10 checks, judgment fail"),
            (dict(grade=GRADE.replace("PASS  S3.5 freeze no test access", "FAIL  S3.5 freeze no test access")),
             "FAIL  S3.5 freeze no test access"),
            (dict(no_evidence="1", driver="2"), "FAIL  S3.5 integration driver produced no graded checks"),
            (dict(grade_json=dict(GRADE_JSON, hardware_gate="open")), "judgment pass, hardware gate open"),
            (dict(grade_json=dict(GRADE_JSON, judgment="rehearsal")), "judgment rehearsal, hardware gate closed"),
            (dict(show="1"), again), (dict(inspect="1"), again), (dict(no_recover="1"), again),
            # A file of the artifact is not the one the freeze record names, or the sensor scored with another artifact.
            (dict(weights="other float32 values"),
             f"the freeze record names the artifact {ARTIFACT_SHA256} with the weights {WEIGHTS_SHA256}"),
            (dict(template="another background"), f"and the template {TEMPLATE_SHA256}"),
            (dict(shown=dict(SHOWN, artifact_sha256="f" * 64)), "the sensor scored with " + "f" * 64),
            (dict(inspected=dict(INSPECTED, weights_sha256="e" * 64)),
             "FAIL  S3.5 a second reader recovers the frozen model and a held-out decision: the freeze record names"),
            (dict(pins="1"), "FAIL  S3.5 sources at the locked revisions after the runs"),
        )
        for number, (settings, line) in enumerate(cases):
            result = self.gate(run_dir=self.root.parent / f"evidence-{number}", **settings)
            with self.subTest(line=line):
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn(line, result.stdout)
                self.assertIn("Sprint 3.5: NOT ACCEPTED on this host", result.stdout)
                self.assertNotIn("Sprint 3.5: ACCEPTED", result.stdout)
        skipped = self.gate(run_dir=self.root.parent / "evidence-build", build="1")
        self.assertIn("SKIP  S3.5 integration driver, artifact recovery and records: no adapter", skipped.stdout)

    def test_the_records_of_the_held_out_episodes_must_exist_and_stay_ungraded(self):
        missing = "FAIL  S3.5 records: cusense_eval.json or cusense_reference.json is missing, graded or incomplete"
        no_median = json.loads(json.dumps(EVAL))
        no_median["arms"]["o3"]["position"]["raw"]["median_m"] = None
        no_count = json.loads(json.dumps(EVAL))
        no_count["arms"]["o3"]["position"]["raw"]["eligible"] = 0
        cases = ((dict(no_eval="1"), "No such file or directory"),
                 (dict(no_reference="1"), "No such file or directory"),
                 (dict(eval=dict(EVAL, graded=True)), "a record of the held-out episodes is marked as graded"),
                 (dict(reference=dict(REFERENCE, graded=True)), "a record of the held-out episodes is marked as graded"),
                 (dict(eval=no_median), "the position error has no mean, median and 90th percentile in metres with a count"),
                 (dict(eval=no_count), "the position error has no mean, median and 90th percentile in metres with a count"))
        for number, (settings, reason) in enumerate(cases):
            result = self.gate(run_dir=self.root.parent / f"evidence-{number}", **settings)
            with self.subTest(reason=reason):
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn(missing, result.stdout)
                self.assertIn(reason, result.stdout)
                self.assertNotIn("Sprint 3.5: ACCEPTED", result.stdout)
        # What the figures are decides nothing: a large position error and a missed episode still close the gate.
        poor = json.loads(json.dumps(EVAL))
        poor["arms"]["o3"]["position"]["raw"].update(mean_m=4.2, median_m=4.0, p90_m=7.5)
        poor["arms"]["o3"]["presence"]["episodes"]["target_present"].update(detected=6, missed=4)
        result = self.gate(run_dir=self.root.parent / "evidence-poor", eval=poor)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("6 of 10 target-present held-out episodes detected", result.stdout)
        self.assertIn("a mean of 4.200 m", result.stdout)

    def test_the_patched_tree_is_verified_when_it_is_there(self):
        (self.root / "build/ocudu-src-o3").mkdir(parents=True)
        result = self.gate()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("PASS  S3.5 patched tree: build/ocudu-src-o3 is the pinned commit with exactly the series applied",
                      result.stdout)
        result = self.gate(run_dir=self.root.parent / "evidence-tree", tree="1")
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S3.5 patched tree: build/ocudu-src-o3 is the pinned commit with exactly the series applied",
                      result.stdout)

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
        for arguments in (("--unknown",), ("--s34-run",), ("--s52-run",), ("--episode-set",)):
            result = self.gate(*arguments, run_dir=self.root.parent / "evidence-usage")
            self.assertEqual(result.returncode, 2)
            self.assertIn("Usage: bash scripts/accept_s3_5.sh", result.stderr)
        (self.bin / "python").unlink()
        missing = self.gate(run_dir=self.root.parent / "evidence-python")
        self.assertEqual(missing.returncode, 2)
        self.assertIn("Run 'make env' first", missing.stderr)


if __name__ == "__main__":
    unittest.main()
