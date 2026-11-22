"""Check the Sprint 3.4 gate runner against tool doubles. No compiler, source, build, NumPy, PyTorch or GPU is needed."""

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

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/accept_s3_4.sh"
TEST = "tests/integrationtests/phy/upper/channel_processors/pusch_e2e_pipeline_test"
WEIGHTS = "nine float32 values: eight weights and the bias"
WEIGHTS_SHA256 = hashlib.sha256(WEIGHTS.encode("ascii")).hexdigest()
ARTIFACT_SHA256 = "a1b2c3d4e5f60718" * 4
GRADE = """PASS  S3.4 control accepted run: the Sprint 3.3 driver graded pilot_features as pass
PASS  S3.4 split assignment: an episode has the group of its trajectory
PASS  S3.4 records account: 16320 observations in 14 runs: 16320 published, 0 dropped
PASS  S3.4 features windows: 16320 records of the rings in sequence order
PASS  S3.4 train threshold: the frozen threshold is the logit -0.412345
PASS  S3.4 freeze no test access: the manifests list 600 training and 80 validation episode keys
PASS  S3.4 live sensor: python -m ocudu_cuda_isac.sensing.presence follow, a second process
PASS  S3.4 parity scores: 20 held-out windows joined by their identifier
NOTE  S3.4 parity held-out episodes at the frozen threshold, read with the labels after the decisions were written
presence_head: 10 of 10 cases passed, 8 of 8 checks"""
GRADE_JSON = {"judgment": "pass", "hardware_gate": "closed", "cases": 10, "cases_passed": 10, "checks": 8,
              "checks_passed": 8, "checks_failed": 0}
REHEARSAL_JSON = dict(GRADE_JSON, judgment="rehearsal",
                      hardware_gate="open: the CUDA processor did not execute and the head scored on the CPU, this is a "
                                    "host rehearsal")
ACCEPTED = {"grant_fixture": {"sha256": "0123456789abcdef" * 4}, "grid_apply": {"sha256": "1111222233334444" * 4},
            "replay_pusch": {"sha256": "5555666677778888" * 4}, "pilot_ls": {"sha256": "9999aaaabbbbcccc" * 4},
            "obs_tail": {"sha256": "ddddeeeeffff0000" * 4}, "obs_feed": {"sha256": "1234123412341234" * 4}}
RECEIPT = {"executables": ACCEPTED, "tests_run": 75, "compiler_warnings": 0, "pin": {"cuda": True, "archives": 50}}
HOST_RECEIPT = dict(RECEIPT, pin={"cuda": False, "archives": 50})
DEVICE = {"device": "cuda:0", "type": "cuda", "torch": "2.15.0", "cuda_runtime": "13.2", "cuda_available": True,
          "name": "NVIDIA GB10", "capability": "12.1"}
CPU_DEVICE = {"device": "cpu", "type": "cpu", "torch": "2.15.0", "cuda_runtime": "13.2", "cuda_available": False}
RECOVER = {"artifact": "train/artifact", "freeze": "train/freeze.json", "sensor": "live/held-out-00/sensor", "window": 0,
           "offline": "parity/held-out-00"}
FREEZE = {"artifact": {"sha256": ARTIFACT_SHA256, "weights_sha256": WEIGHTS_SHA256}, "frozen_utc": "2026-11-22T03:10:00Z"}
INSPECTED = {"head_id": "dd-linear-v1", "artifact_sha256": ARTIFACT_SHA256, "vector": 13,
             "columns": ["peak_to_median_db", "profile_db@-100Hz", "profile_db@-75Hz", "profile_db@-50Hz",
                         "profile_db@-25Hz", "profile_db@+25Hz", "profile_db@+50Hz", "profile_db@+75Hz"],
             "weights_sha256": WEIGHTS_SHA256, "files": ["head.json", "weights.f32"],
             "threshold": {"logit": -0.412345, "negatives": 40, "allowed": 2, "exceeding": 2}}
SHOWN = {"head_id": "dd-linear-v1", "artifact_sha256": ARTIFACT_SHA256, "runtime": "reference", "device": "cuda:0",
         "loads": 1, "scored_windows": 20,
         "window": {"id": "1/89abcdef01234567/7", "contributing": 8, "logit": 2.25, "probability": 0.9047, "present": True}}
S33_SUMMARY = """Sprint 3.3 acceptance on spark-ecf8, 2026-11-22T00:48:01Z
PASS  committed tree
Sprint 3.3: ACCEPTED on this DGX Spark. The reference of the sensing core forms the radar cube.
"""
S33_REHEARSED = """Sprint 3.3 acceptance on devhost, 2026-11-22T00:48:01Z
HOST REHEARSAL: the host path is graded on a build without CUDA. This run cannot accept the sprint.
Sprint 3.3: REHEARSED on this host, NOT ACCEPTED. The features of the rehearsed Sprint 3.2 run agree.
"""
UNIT = "Ran 540 tests in 480.000s\n\nOK"
CARD = "Ran 30 tests in 9.500s\n\nOK"
RULE = "Ran 3 tests in 1.200s\n\nOK"
# The interpreter double answers each gate from FAKE_* variables and logs the call. The driver
# double leaves its grade, the place of what a second reader recovers, the freeze record and the
# weights file.
PYTHON_DOUBLE = """#!/bin/sh
[ "$1" = "-c" ] && exec {real} "$@"
echo "$*" >> "$FAKE_LOG"
case "$*" in
    "-m pip freeze --all") echo "example==1.0" ;;
    *"-m unittest discover"*) printf '%s\\n' "$FAKE_UNIT" >&2; exit "${{FAKE_UNIT_STATUS:-0}}" ;;
    "tests/unit/test_presence_head.py -v") printf '%s\\n' "$FAKE_CARD" >&2; exit "${{FAKE_CARD_STATUS:-0}}" ;;
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
    "-m ocudu_cuda_isac.sensing.presence device")
        [ -z "${{FAKE_NO_TORCH:-}}" ] || {{ echo "No module named 'torch'" >&2; exit 1; }}
        printf '%s\\n' "$FAKE_DEVICE" ;;
    "tests/integration/presence_head.py --config configs/integration/presence_head.json --s33-run "*)
        for out; do :; done
        echo "s33 run: $5" >> "$FAKE_LOG"
        echo "episode set: $7" >> "$FAKE_LOG"
        echo "Run artifacts: $out"
        [ -z "${{FAKE_NO_EVIDENCE:-}}" ] || exit "${{FAKE_DRIVER:-0}}"
        mkdir -p "$out/train/artifact" "$out/live/held-out-00/sensor"
        printf '%s' "$FAKE_WEIGHTS" > "$out/train/artifact/weights.f32"
        printf '%s\\n' "$FAKE_FREEZE" > "$out/train/freeze.json"
        [ -n "${{FAKE_NO_RECOVER:-}}" ] || printf '%s\\n' "$FAKE_RECOVER" > "$out/recover.json"
        printf '%s\\n' "$FAKE_GRADE" > "$out/grade.txt"
        printf '%s\\n' "$FAKE_GRADE_JSON" > "$out/grade.json"
        exit "${{FAKE_DRIVER:-0}}" ;;
    "-m ocudu_cuda_isac.sensing.presence inspect "*)
        [ "${{FAKE_INSPECT:-0}}" = 0 ] || {{ echo "presence: the artifact has another hash, and this one was asked for" >&2; exit 2; }}
        printf '%s\\n' "$FAKE_INSPECTED" ;;
    "-m ocudu_cuda_isac.sensing.presence show "*)
        [ "${{FAKE_SHOW:-0}}" = 0 ] || {{ echo "presence: decision 7: the digest of its input does not agree" >&2; exit 2; }}
        printf '%s\\n' "$FAKE_SHOWN" ;;
    *"check_pins.py --worktrees") echo "ext/ocudu 27e84cb3 (Git worktree)"; exit "${{FAKE_PINS:-0}}" ;;
    *) echo "interpreter double: unexpected $*" >&2; exit 64 ;;
esac
"""


@unittest.skipUnless(shutil.which("bash") and shutil.which("sha256sum"), "the gate runner is a Bash script")
class SprintThreePresenceGateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / "host checkout"
        self.bin = self.root / "isac/bin"
        (self.root / "scripts").mkdir(parents=True)
        self.bin.mkdir(parents=True)
        shutil.copy(SCRIPT, self.root / "scripts/accept_s3_4.sh")
        python = self.bin / "python"
        python.write_text(PYTHON_DOUBLE.format(real=shlex.quote(sys.executable)))
        python.chmod(0o755)
        # The build Sprint 1.1 published, the run that Sprint 3.3 accepted and a traced episode set.
        self.build = self.root / "build/ocudu-spark"
        (self.build / TEST).parent.mkdir(parents=True)
        (self.build / TEST).write_text("#!/bin/sh\n")
        (self.bin / "pusch_e2e_pipeline_test").symlink_to(self.build / TEST)
        (self.build / "isac-native.json").write_text(json.dumps({"target": "all"}))
        self.s33 = self.control_run("s33-spark-ecf8-20261122T004801Z")
        self.set = self.episode_set("demo-set-20261122T020000Z")
        self.calls = self.root.parent / "calls.log"
        self.run_dir = self.root.parent / "evidence"

    def control_run(self, name, summary=S33_SUMMARY, plan=True, executables=None):
        run = self.root / "results/runs" / name
        out = run / "pilot_features"
        (out / "live/run").mkdir(parents=True)
        (out / "grade.json").write_text("{}")
        if plan:
            (out / "live/run/plan.json").write_text("{}")
        (run / "adapter-receipt.json").write_text(json.dumps({"executables": executables or ACCEPTED}))
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
                   FAKE_SHOWN=json.dumps(SHOWN), FAKE_WEIGHTS=WEIGHTS)
        env.update({f"FAKE_{key.upper()}": value if isinstance(value, str) else json.dumps(value)
                    for key, value in settings.items()})
        command = [shutil.which("bash"), str(self.root / "scripts/accept_s3_4.sh"),
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
        for line in ("PASS  unit tests (540 ran, 0 skipped)",
                     "PASS  S3.4 unit gate, run directly: python tests/unit/test_presence_head.py (30 ran, 0 skipped)",
                     "PASS  S3.4 import rule, run directly: python tests/unit/test_core_imports.py (3 ran, 0 skipped)",
                     "PASS  S3.4 control: the Sprint 3.3 run s33-spark-ecf8-20261122T004801Z is ACCEPTED and holds its "
                     "grade and the plan of its live run",
                     "PASS  S3.4 stimulus: the episode set demo-set-20261122T020000Z holds a manifest and its episodes",
                     "PASS  S3.4 references: the Sprint 1.1 build, from the project interpreter (environment.json)",
                     "PASS  S3.4 adapter: built against the Sprint 1.1 build and verified, pilot_ls 9999aaaabbbb, obs_tail "
                     "ddddeeeeffff, with ENABLE_CUDA, 75 adapter tests, 0 compiler warnings",
                     "PASS  S3.4 accepted executables: this sprint changes no native source, and every tool has the bytes "
                     "of the Sprint 3.3 run: grant_fixture 0123456789ab, grid_apply 111122223333, obs_feed 123412341234, "
                     "obs_tail ddddeeeeffff, pilot_ls 9999aaaabbbb, replay_pusch 555566667777",
                     "PASS  S3.4 hardware: the adapter is linked against the CUDA build, so the CUDA processor can execute",
                     "PASS  S3.4 device: PyTorch 2.15.0 reaches cuda:0 (NVIDIA GB10, capability 12.1, CUDA runtime 13.2): "
                     "the head is scored in a second CUDA process (device.json)",
                     "Presence head on the episode set demo-set-20261122T020000Z, on the control "
                     "s33-spark-ecf8-20261122T004801Z, with configs/integration/presence_head.json.",
                     "PASS  S3.4 parity scores: 20 held-out windows joined by their identifier",
                     "NOTE  S3.4 parity held-out episodes at the frozen threshold",
                     "PASS  S3.4 integration grade: 10 of 10 cases, 8 of 8 checks, judgment pass, hardware gate closed",
                     "PASS  S3.4 a second reader recovers the frozen head and a held-out decision: the head dd-linear-v1 "
                     "reads 8 of 13 values, peak_to_median_db, profile_db@-100Hz, profile_db@-75Hz, profile_db@-50Hz, "
                     "profile_db@-25Hz, profile_db@+25Hz, profile_db@+50Hz, profile_db@+75Hz, from the files head.json and "
                     "weights.f32; its threshold is the logit -0.4123, from 40 target-absent validation windows with 2 of 2 "
                     "allowed above it; the sensor loaded it 1 time(s) on cuda:0 in the runtime reference and scored 20 "
                     "windows: window 1/89abcdef01234567/7 from 8 observations has the logit 2.2500 and the probability "
                     f"0.9047, above the threshold; the artifact {ARTIFACT_SHA256[:16]} is the one the freeze record of "
                     f"2026-11-22T03:10:00Z names, and its weights have the SHA-256 {WEIGHTS_SHA256[:16]} by sha256sum",
                     "PASS  S3.4 sources at the locked revisions after the runs",
                     "Sprint 3.4: ACCEPTED on this DGX Spark. One linear presence head on the delay-Doppler features was "
                     "trained on receiver-derived records of the training trajectories",
                     "frozen before a held-out entry was replayed.",
                     "graded in Sprint 4. Next: S5.2 native estimate."):
            self.assertIn(line, summary)
        calls = self.calls.read_text()
        self.assertIn(f"s33 run: {self.s33}", calls)
        self.assertIn(f"episode set: {self.set}", calls)
        self.assertIn(f"scripts/native_adapter.py build --ocudu-build-dir {self.build}", calls)
        self.assertIn(f"-m ocudu_cuda_isac.sensing.presence inspect {self.run_dir}/presence_head/train/artifact "
                      f"--sha256 {ARTIFACT_SHA256}", calls)
        self.assertIn(f"-m ocudu_cuda_isac.sensing.presence show {self.run_dir}/presence_head/live/held-out-00/sensor "
                      "--window 0", calls)
        self.assertNotIn("--host-rehearsal", calls)
        self.assertEqual(summary.count("PASS  "), 21)

    def test_the_newest_accepted_sprint_3_3_run_and_the_newest_set_are_used(self):
        self.control_run("s33-spark-ecf8-20261122T010000Z", S33_SUMMARY.replace("PASS  committed", "FAIL  committed"))
        self.control_run("s33-spark-ecf8-20261122T020000Z", "Sprint 3.3: NOT ACCEPTED on this host\n")
        self.control_run("s33-spark-ecf8-20261122T030000Z", plan=False)
        self.control_run("s33-spark-ecf8-20261122T035000Z", S33_REHEARSED)
        newest = self.control_run("s33-spark-ecf8-20261122T005000Z")
        self.episode_set("demo-set-20261122T030000Z", manifest=False)  # a trace that did not finish
        later = self.episode_set("demo-set-20261122T021500Z")
        self.episode_set("smoke-set-20261122T040000Z")  # only a demo set is taken by default
        result = self.gate()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn(f"s33 run: {newest}", self.calls.read_text())
        self.assertIn(f"episode set: {later}", self.calls.read_text())
        other = self.control_run("s33-devhost-20261119T000000Z", S33_SUMMARY.replace("DGX Spark", "host"))
        named = self.episode_set("smoke-set-20261122T040000Z-named")
        result = self.gate("--s33-run", str(other), "--episode-set", str(named), run_dir=self.root.parent / "evidence-named")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("Sprint 3.4: ACCEPTED on this host.", result.stdout)
        self.assertIn(f"episode set: {named}", self.calls.read_text())

    def test_without_a_control_or_a_set_nothing_is_computed(self):
        shutil.rmtree(self.s33)
        result = self.gate()
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S3.4 control: no Sprint 3.3 ACCEPTED verdict with its grade and the plan of its live run in "
                      "results/runs/s33-*", result.stdout)
        self.assertIn("Run 'make accept-s3.3' on this host first", result.stdout)
        self.assertIn("SKIP  S3.4 reference check, adapter build, integration driver and artifact recovery: no control "
                      "or no episode set", result.stdout)
        self.assertIn("Sprint 3.4: NOT ACCEPTED on this host", result.stdout)
        self.assertNotIn("tests/integration/presence_head.py", self.calls.read_text())
        # With a control and without a traced set the gate says how to get one, and traces none itself.
        self.control_run("s33-spark-ecf8-20261122T004801Z")
        shutil.rmtree(self.set)
        result = self.gate(run_dir=self.root.parent / "evidence-no-set")
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S3.4 stimulus: no episode set with a manifest in data/episodes/demo-set-*", result.stdout)
        self.assertIn("Trace the demonstration set once with 'make episode-set'", result.stdout)
        self.assertNotIn("export_rt_episode", self.calls.read_text())
        self.assertNotIn("tests/integration/presence_head.py", self.calls.read_text())

    def test_a_tool_with_other_bytes_fails(self):
        self.control_run("s33-spark-ecf8-20261122T005500Z", executables=dict(ACCEPTED, pilot_ls={"sha256": "9" * 64}))
        result = self.gate()
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S3.4 accepted executables: this sprint changes no native source, so every tool must keep "
                      "the bytes of the Sprint 3.3 run: pilot_ls is 9999aaaabbbb and was 999999999999", result.stdout)

    def test_without_cuda_the_gate_cannot_be_accepted_and_a_rehearsal_never_is(self):
        result = self.gate(receipt=HOST_RECEIPT)
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S3.4 hardware: the accepted build has no CUDA", result.stdout)
        self.assertNotIn("Sprint 3.4: ACCEPTED", result.stdout)
        # The receiver has CUDA and the interpreter of the sensor reaches no device, or holds no PyTorch.
        result = self.gate(device=CPU_DEVICE, run_dir=self.root.parent / "evidence-cpu")
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S3.4 device: PyTorch 2.15.0 reaches no CUDA device in isac, so the head cannot be scored on "
                      "the selected device", result.stdout)
        result = self.gate(no_torch="1", run_dir=self.root.parent / "evidence-no-torch")
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S3.4 device: PyTorch does not load in isac", result.stdout)
        self.assertIn("No module named 'torch'", result.stdout)
        # A rehearsal needs a rehearsed control: the accepted run is not one.
        result = self.gate("--host-rehearsal", receipt=HOST_RECEIPT, device=CPU_DEVICE, grade_json=REHEARSAL_JSON,
                           run_dir=self.root.parent / "evidence-no-control")
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S3.4 control: no Sprint 3.3 REHEARSED verdict", result.stdout)
        rehearsed = self.control_run("s33-devhost-20261122T020000Z", S33_REHEARSED)
        result = self.gate("--host-rehearsal", receipt=HOST_RECEIPT, device=CPU_DEVICE, grade_json=REHEARSAL_JSON,
                           shown=dict(SHOWN, device="cpu"), run_dir=self.root.parent / "evidence-rehearsal")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertNotIn("FAIL", result.stdout)
        for line in ("HOST REHEARSAL: the host path is graded on a build without CUDA and the head scores on the CPU.",
                     "PASS  S3.4 control: the Sprint 3.3 run s33-devhost-20261122T020000Z is REHEARSED",
                     "NOTE  S3.4 hardware: host rehearsal on an adapter built for the host path; PyTorch 2.15.0 reaches no "
                     "CUDA device in isac.",
                     "PASS  S3.4 integration grade: 10 of 10 cases, 8 of 8 checks, judgment rehearsal, hardware gate open",
                     "the sensor loaded it 1 time(s) on cpu in the runtime reference",
                     "Sprint 3.4: REHEARSED on this host, NOT ACCEPTED."):
            self.assertIn(line, result.stdout)
        self.assertIn(f"s33 run: {rehearsed}", self.calls.read_text())
        self.assertIn("--host-rehearsal --output-dir", self.calls.read_text())
        # A rehearsal whose driver reports a pass was not a rehearsal, and the reverse.
        result = self.gate("--host-rehearsal", receipt=HOST_RECEIPT, device=CPU_DEVICE,
                           run_dir=self.root.parent / "evidence-pass")
        self.assertEqual(result.returncode, 1)
        result = self.gate(grade_json=REHEARSAL_JSON, run_dir=self.root.parent / "evidence-claim")
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S3.4 integration grade: 10 of 10 cases, 8 of 8 checks, judgment rehearsal", result.stdout)

    def test_failures_of_single_gates(self):
        failed = dict(GRADE_JSON, judgment="fail", hardware_gate="open", cases_passed=9, checks_passed=7, checks_failed=1)
        cases = (
            (dict(unit="Ran 540 tests in 4.0s\n\nFAILED (failures=1)", unit_status="1"), "FAIL  unit tests (540 ran"),
            (dict(card="Ran 0 tests in 0.0s\n\nOK"), "tests/unit/test_presence_head.py (none ran)"),
            # Without PyTorch every vector of the unit gate is skipped: such a run cannot close it.
            (dict(card="Ran 30 tests in 0.1s\n\nOK (skipped=27)"),
             "(30 ran, 27 skipped: the head needs NumPy and PyTorch in isac, 'make env-full'"),
            (dict(card="Ran 30 tests in 9.0s\n\nOK (skipped=1)"), "and its live case cmake and a C++17 compiler"),
            (dict(rule="Ran 3 tests in 0.2s\n\nFAILED (failures=1)", rule_status="1"),
             "FAIL  S3.4 import rule, run directly: python tests/unit/test_core_imports.py (3 ran"),
            (dict(environment="1"), "FAIL  S3.4 references: the Sprint 1.1 build"),
            (dict(build="1"), "FAIL  S3.4 adapter: build or verification failed"),
            (dict(verify="1"), "FAIL  S3.4 adapter: build or verification failed"),
            (dict(receipt=dict(RECEIPT, executables={"pilot_ls": ACCEPTED["pilot_ls"]})),
             "FAIL  S3.4 adapter: the receipt names no pilot_ls and obs_tail"),
            (dict(grade=GRADE.replace("PASS  S3.4 parity scores", "FAIL  S3.4 parity scores"), grade_json=failed,
                  driver="1"), "FAIL  S3.4 integration grade: 9 of 10 cases, 7 of 8 checks, judgment fail"),
            (dict(grade=GRADE.replace("PASS  S3.4 freeze no test access", "FAIL  S3.4 freeze no test access")),
             "FAIL  S3.4 freeze no test access"),
            (dict(no_evidence="1", driver="2"), "FAIL  S3.4 integration driver produced no graded checks"),
            (dict(grade_json=dict(GRADE_JSON, hardware_gate="open")), "judgment pass, hardware gate open"),
            (dict(show="1"), "FAIL  S3.4 a second reader recovers the frozen head and a held-out decision (see "
                             "recovered.log)"),
            (dict(inspect="1"), "FAIL  S3.4 a second reader recovers the frozen head and a held-out decision (see "
                                "recovered.log)"),
            (dict(no_recover="1"), "FAIL  S3.4 a second reader recovers the frozen head and a held-out decision (see "
                                   "recovered.log)"),
            # The weights file is not the one the freeze record names, or the sensor scored with another artifact.
            (dict(weights="nine other float32 values"),
             f"the freeze record names the artifact {ARTIFACT_SHA256} with the weights {WEIGHTS_SHA256}"),
            (dict(shown=dict(SHOWN, artifact_sha256="f" * 64)), "the sensor scored with " + "f" * 64),
            (dict(inspected=dict(INSPECTED, weights_sha256="e" * 64)),
             "FAIL  S3.4 a second reader recovers the frozen head and a held-out decision: the freeze record names"),
            (dict(pins="1"), "FAIL  S3.4 sources at the locked revisions after the runs"),
        )
        for number, (settings, line) in enumerate(cases):
            result = self.gate(run_dir=self.root.parent / f"evidence-{number}", **settings)
            with self.subTest(line=line):
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn(line, result.stdout)
                self.assertIn("Sprint 3.4: NOT ACCEPTED on this host", result.stdout)
                self.assertNotIn("Sprint 3.4: ACCEPTED", result.stdout)
        skipped = self.gate(run_dir=self.root.parent / "evidence-build", build="1")
        self.assertIn("SKIP  S3.4 integration driver and artifact recovery: no adapter", skipped.stdout)

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
        for arguments in (("--unknown",), ("--s33-run",), ("--episode-set",)):
            result = self.gate(*arguments, run_dir=self.root.parent / "evidence-usage")
            self.assertEqual(result.returncode, 2)
            self.assertIn("Usage: bash scripts/accept_s3_4.sh", result.stderr)
        (self.bin / "python").unlink()
        missing = self.gate(run_dir=self.root.parent / "evidence-python")
        self.assertEqual(missing.returncode, 2)
        self.assertIn("Run 'make env' first", missing.stderr)


if __name__ == "__main__":
    unittest.main()
