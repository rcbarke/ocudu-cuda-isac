"""Check the Sprint 3.4 integration driver against doubles of the native tool, its ring and the ray tracer.

The control is a Sprint 3.3 run that the driver of that sprint grades on the doubles of its own
driver tests: the small pair of two TDD periods with the store of every grant. The stimulus is a
small episode set on the grant of the demonstration, written by the set module with the
free-space model in place of the ray tracer: two training trajectories with one pair each, one
validation pair and one held-out pair, two periods long. The sensing side, the training script
and the evaluator under the driver are the real ones. The double of this sprint
(tests/unit/presence_head_double.py) only stands in for the native tool and the ring. A window of
two observations has one moving Doppler cell, so the head reads two columns here, and the
quality of what it learns from four windows means nothing: every check of the driver still has a
known answer, and each injected fault must fail the check that names it. NumPy and PyTorch are
needed, as for the driver.
"""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest

try:
    import numpy as np
    import torch
except ImportError:
    np = torch = None

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

if np is not None:
    import test_pilot_features_driver as control_tests  # noqa: E402
    import presence_head_double as double  # noqa: E402
    from ocudu_cuda_isac.sensing import presence  # noqa: E402
    sys.path.insert(0, str(ROOT / "scripts"))
    import train_presence  # noqa: E402

CHECKS = 35
CASES = 10
ORDER = ("control", "split", "records", "features", "train", "freeze", "live", "parity", "causality", "refusals")
ENVIRONMENT = ("PRESENCE_HEAD_FAULT", "PILOT_FEATURES_FAULT", "SNAPSHOT_RING_FAULT", "PILOT_LS_FAULT", "PILOT_LS_BUILD",
               "REPLAY_PUSCH_FAULT", "REPLAY_PUSCH_BUILD")


def load_driver():
    spec = importlib.util.spec_from_file_location("presence_head_driver", ROOT / "tests/integration/presence_head.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def blocked_after(case):
    """The checks that the cases after one failed case report: each is blocked on its control."""
    return [f"{later} control" for later in ORDER[ORDER.index(case) + 1:]]


class Scenarios:
    """The model receiver of the double, once per stimulus: a second run on the same entries copies the first.

    The key is what a run replays and how: the channel files by their hashes, the ring, the evidence
    mode and every injected fault. A copy is what the tool wrote for a plan of the same content.
    """

    def __init__(self, directory, real):
        self.directory, self.real, self.made = directory, real, {}

    def __call__(self, tools, plan_path, native, directory, consumers, extra=()):
        plan = json.loads(Path(plan_path).read_text())
        channel = json.loads(Path(json.loads(Path(plan["replay_plan"]["path"]).read_text())["channel_plan"]["path"]).read_text())
        key = hashlib.sha256(json.dumps([
            [(source["h"]["sha256"], source["w"]["sha256"]) for source in channel["sources"]], plan["ring"],
            plan["evidence"], plan["observer"], plan["run_id"], os.environ.get("SNAPSHOT_RING_FAULT", ""),
            [(consumer["reader"], consumer["wait"], consumer.get("hold_us")) for consumer in consumers]],
            sort_keys=True).encode("ascii")).hexdigest()
        if key not in self.made:
            result = self.real(tools, plan_path, native, directory, consumers, extra)
            home = self.directory / key[:16]
            shutil.copytree(native, home / "native")
            shutil.copytree(consumers[0]["archive"], home / "archive")
            shutil.copy(directory / "native.log", home / "native.log")
            self.made[key] = (home, result)
            return result
        home, result = self.made[key]
        shutil.copytree(home / "native", native)
        shutil.copytree(home / "archive", consumers[0]["archive"])
        shutil.copy(home / "native.log", directory / "native.log")
        return result


def run_in_process(real):
    """The tools of the project that the driver starts as processes, called here in this interpreter.

    A process of its own pays the import of PyTorch each time. The reference run of these tests
    starts the real processes, and the runs with an injected fault call the same command lines
    here: the arguments, the exit status and the two streams are those of the process.
    """
    def run_tool(command, log, timeout=1800):
        words = [str(word) for word in command]
        if words[1:3] == ["-m", "ocudu_cuda_isac.sensing.presence"]:
            main, arguments = presence.main, words[3:]
        elif words[1].endswith("train_presence.py"):
            main, arguments = train_presence.main, words[2:]
        else:
            return real(command, log, timeout)
        said, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(said), contextlib.redirect_stderr(errors):
            try:
                status = main(arguments)
            except SystemExit as stop:
                status = stop.code
        Path(log).write_text(" ".join(words) + "\n" + said.getvalue() + errors.getvalue())
        return types.SimpleNamespace(returncode=status, stdout=said.getvalue(), stderr=errors.getvalue())
    return run_tool


@unittest.skipIf(np is None, "the integration driver needs NumPy and PyTorch")
class DriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.driver = load_driver()
        # The accepted Sprint 3.3 run: the driver of that sprint on the doubles of its tools.
        cls.stage = control_tests.DriverTests
        cls.stage.setUpClass()
        cls.addClassCleanup(cls.release)
        assert cls.stage.reference["status"] == 0, cls.stage.reference["failed"]
        cls.control = cls.stage.reference["out"]
        cls.root = cls.stage.root / "s34"
        cls.root.mkdir()
        cls.grant = ROOT / "configs/grants/demo_51prb_4rx.json"
        cls.set = cls.root / "set"
        double.analytic_set(double.set_config(cls.root / "set.json", cls.grant, double.trajectories(2, 1, 1)), cls.set)
        cls.settings = json.loads((ROOT / "configs/integration/presence_head.json").read_text())
        # The small set: windows of two periods, a ring of four slots, one entry per run, the head on the CPU.
        cls.settings["spec"]["window"].update(periods=2, hop_periods=2)
        cls.settings["ring"].update(window_records=1, burst_records=1, slots=4)
        cls.settings["set"] = {"entries_per_run": 1, "expect": {"set_id": "unit-set", "episodes": {
            "training": 2, "validation": 1, "held_out": 1}}}
        cls.settings["training"].update(min_per_class=1)
        cls.settings["training"]["threshold"].update(min_negatives=1)
        cls.settings["sensor"]["device"] = "cpu"
        cls.config = cls.write_config(cls.root / "config.json", cls.settings)
        cls.tool = cls.root / "tool"
        cls.tool.write_text("a double\n")
        cls.scenarios = Scenarios(cls.root / "scenarios", double.ring_double.scenario)
        cls.reference = cls.run_once(cls.root / "reference", processes=True)

    @classmethod
    def release(cls):
        """Clean up every stage under this one: each holds what the stage below it made."""
        stage = cls.stage
        while stage is not None:
            stage.doClassCleanups()
            stage.tearDownClass()
            stage = getattr(stage, "stage", None)

    @staticmethod
    def write_config(path, settings):
        path.write_text(json.dumps(settings))
        return path

    @classmethod
    def receipt_of(cls):
        return {"executables": {name: {"path": str(cls.tool), "sha256": hashlib.sha256(cls.tool.read_bytes()).hexdigest()}
                                for name in ("pilot_ls", "obs_tail", "obs_feed", "replay_pusch")},
                "ocudu_build": "double", "pin": {"cuda": True}}

    @classmethod
    def run_once(cls, out, fault=None, config=None, control=None, episode_set=None, extra=(), receipt=None, patch=None,
                 processes=False):
        driver = cls.driver
        saved = (driver.native_adapter.verified_receipt, driver.run_capture, driver.run_live, double.ring_double.scenario,
                 driver.run_tool)
        driver.native_adapter.verified_receipt = lambda build_dir: receipt or cls.receipt_of()
        driver.run_capture, driver.run_live = double.capture, double.live
        if not processes:
            driver.run_tool = run_in_process(driver.run_tool)
        double.ring_double.scenario = cls.scenarios
        kept = {key: os.environ.get(key) for key in ENVIRONMENT}
        for key in kept:
            os.environ.pop(key, None)
        if fault:
            os.environ["PRESENCE_HEAD_FAULT"] = fault
        arguments = ["--config", str(config or cls.config), "--s33-run", str(control or cls.control), "--episode-set",
                     str(episode_set or cls.set), "--output-dir", str(out), *extra]
        undo = patch() if patch else None
        loads = presence.Head.LOADS
        try:
            with contextlib.redirect_stdout(io.StringIO()) as said:
                status = driver.main(arguments)
        finally:
            if undo:
                undo()
            presence.Head.LOADS = loads
            (driver.native_adapter.verified_receipt, driver.run_capture, driver.run_live,
             double.ring_double.scenario, driver.run_tool) = saved
            for key, value in kept.items():
                os.environ.pop(key, None)
                if value is not None:
                    os.environ[key] = value
        grade = json.loads((out / "grade.json").read_text())
        failed = [f"{item['case']} {item['check']}" for item in grade["results"] if not item["passed"]]
        return {"status": status, "grade": grade, "failed": failed, "out": out, "said": said.getvalue(),
                "lines": (out / "grade.txt").read_text()}

    def setUp(self):
        scratch = tempfile.TemporaryDirectory(dir=self.root)
        self.addCleanup(scratch.cleanup)
        self.out = Path(scratch.name)

    def faulty(self, fault, **options):
        return self.run_once(self.out / fault, fault=fault, **options)

    def changed_config(self, edit, name="config.json"):
        settings = json.loads(json.dumps(self.settings))
        edit(settings)
        return self.write_config(self.out / name, settings)

    def after_training(self, edit):
        """A patch that lets the training script run and then changes what it froze."""
        driver = self.driver

        def patch():
            real = driver.run_tool

            def run_tool(command, log, timeout=1800):
                result = real(command, log, timeout)
                words = [str(word) for word in command]
                if any(word.endswith("train_presence.py") for word in words) and result.returncode == 0 \
                        and "refusals" not in words[words.index("--output-dir") + 1]:
                    edit(Path(words[words.index("--output-dir") + 1]))
                return result
            driver.run_tool = run_tool
            return lambda: setattr(driver, "run_tool", real)
        return patch

    @staticmethod
    def refreeze(run, artifact=None):
        """Write the freeze record of a changed training run again, so that only the change itself is wrong."""
        freeze = json.loads((run / "freeze.json").read_text())
        if artifact is not None:
            manifest = json.loads((run / "artifact/head.json").read_text())
            freeze["artifact"].update(sha256=manifest["sha256"], weights_sha256=manifest["weights"]["sha256"],
                                      normalizer_sha256=manifest["normalizer"]["sha256"])
            freeze["threshold"] = manifest["threshold"]
        freeze["artifact"]["files"] = {path.name: {"bytes": path.stat().st_size,
                                                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                                       for path in sorted((run / "artifact").iterdir())}
        for role in ("training", "validation"):
            freeze["manifests"][role]["sha256"] = hashlib.sha256((run / f"{role}_manifest.json").read_bytes()).hexdigest()
        (run / "freeze.json").write_text(json.dumps(freeze))

    def rewritten(self, change):
        """The artifact written again, whole and hashed, with one part changed, and the freeze record with it."""
        def edit(run):
            artifact = presence.open_artifact(run / "artifact")
            manifest = json.loads(json.dumps(artifact.manifest))
            weights = [float(value) for value in np.frombuffer(artifact.weights, dtype="<f4")]
            parts = {"normalizer": manifest["normalizer"], "weights": weights[:-1], "bias": weights[-1],
                     "threshold": manifest["threshold"]}
            change(parts)
            shutil.rmtree(run / "artifact")
            presence.write_artifact(run / "artifact", manifest["head_id"], manifest["input"]["spec_id"],
                                    manifest["input"]["spec_sha256"], manifest["input"]["vector"],
                                    manifest["input"]["columns"], parts["normalizer"], parts["weights"], parts["bias"],
                                    parts["threshold"])
            self.refreeze(run, True)
        return self.after_training(edit)

    # ---------------------------------------------------------------- the reference
    def test_the_reference_passes_every_check(self):
        run = self.reference
        grade, lines, out = run["grade"], run["lines"], run["out"]
        self.assertEqual((run["status"], run["failed"]), (0, []), lines)
        self.assertEqual((grade["judgment"], grade["hardware_gate"], grade["cases"], grade["cases_passed"], grade["checks"]),
                         ("pass", "closed", CASES, CASES, CHECKS))
        self.assertIn(f"presence_head: {CASES} of {CASES} cases passed, {CHECKS} of {CHECKS} checks\n", lines)
        for line in (
                "PASS  S3.4 control accepted run: the Sprint 3.3 driver graded reference as pass: 8 of 8 cases",
                "PASS  S3.4 control specification: the head is trained on the vector of the specification dd-uniform-8",
                "PASS  S3.4 control reference entry: the plan of the run live of that gate replays the entry unit-pair, "
                "which verifies and binds the transmit grid of 6 grants",
                # The set and its split.
                "PASS  S3.4 split episode set: the set unit-set (status proposed) verifies with every episode in it: "
                "2 training, 1 validation and 1 held-out pairs",
                "PASS  S3.4 split assignment: an episode has the group of its trajectory",
                "2 training trajectories, 1 validation trajectories, 1 held-out trajectories, no trajectory in two groups",
                "8 realizations under distinct opaque keys, and a key names no group and no case",
                "PASS  S3.4 split replayable: every entry lasts at least the 2 periods of one window and gives 1 per "
                "realization at a hop of 2",
                "an entry of a set binds none itself; 1 grid gain(s) in the set",
                # The receiver on the training and the validation entries.
                "PASS  S3.4 records native runs: 3 runs of pilot_ls in the mode timed on a ring of 4 slots: 2 for the 2 "
                "training entries, 1 for the 1 validation entries, at most 1 entries per run, 36 opportunities in all",
                "PASS  S3.4 records decodes: 72 of 72 decodes gave the original bytes of their transport block with a "
                "CRC pass, on cpu and cuda",
                "PASS  S3.4 records executed path: 36 of 36 decodes requested from the CUDA processor ran its resident "
                "path and 36 of 36 decodes of the host processor ran the host path",
                "PASS  S3.4 records account: 36 observations in 3 runs: 36 published, 0 dropped, 36 released, 0 pending; "
                "the consumers kept 36 records, each the record of its account line",
                # The windows and their join.
                "PASS  S3.4 features windows: 36 records of the rings in sequence order through the stream of the "
                "specification dd-uniform-8: 4 complete windows for 4 training realizations, 2 complete windows for 2 "
                "validation realizations, 1 per realization; 6 answers of a filling window, 24 records outside the "
                "sequence, 0 gaps, 0 invalid, 0 rejected",
                "PASS  S3.4 features samples: each complete window is joined to the split table by the episode key of "
                "its records, after the features were written: 2 target-present and 2 target-absent training windows "
                "from 2 trajectories, 1 and 1 validation windows from 1",
                "a vector holds 7 named values",
                # The head.
                "PASS  S3.4 train script: python scripts/train_presence.py read the two sample files and no other input "
                "and froze the head dd-linear-v1 ",
                "it reads 2 of the 7 values of a vector, peak_to_median_db, profile_db@-100Hz, and not "
                "moving_to_static_db, peak_delay_ns, peak_doppler_hz, doppler_centroid_hz, doppler_spread_hz",
                "PASS  S3.4 train artifact: the deployment artifact is head.json and weights.f32: 12 bytes of weights, "
                "one float32 per column and the bias",
                "PASS  S3.4 train normalizer: mean and deviation of the 2 columns, computed here from the 4 training "
                "windows alone, are those of the artifact; with the 2 validation windows added the mean would differ",
                "PASS  S3.4 train threshold: the frozen threshold is the logit ",
                "with a false-alarm target of 0.05 the rule allows 0 of the 1 target-absent validation windows above "
                "it, and 0 are",
                "PASS  S3.4 train fit: the loss falls from 0.693147, ln 2 at zero weights, to ",
                "a second load gives the same 2 validation logits bit for bit",
                "NOTE  S3.4 train recorded, not graded, on the samples of the fit: training: ",
                # The freeze.
                "PASS  S3.4 freeze hash chain: freeze.json names the artifact ",
                "PASS  S3.4 freeze no test access: the manifests list 4 training and 2 validation episode keys, each of "
                "its own split; none of the 2 held-out keys and no window of one is in either",
                # The held-out run with the sensor.
                "PASS  S3.4 live after the freeze: the first plan of a held-out entry is written ",
                "PASS  S3.4 live native run: 1 run(s) of pilot_ls in the mode timed for the 1 held-out entries, 12 "
                "opportunities, with the presence sensor attached in the mode wait",
                "PASS  S3.4 live decodes: 24 of 24 decodes gave the original bytes of their transport block",
                "PASS  S3.4 live executed path: 12 of 12 decodes requested from the CUDA processor ran its resident path",
                "PASS  S3.4 live sensor: python -m ocudu_cuda_isac.sensing.presence follow, a second process on the "
                "standard-library reader, loaded the frozen artifact ",
                " once and scored on cpu; it ran its feature steps and its first inference before it attached",
                "it read 12 records until the producer said that none follows, with no stale read, and its output "
                "verifies",
                "PASS  S3.4 live account: 12 observations in 1 runs: 12 published, 0 dropped, 12 released, 0 pending",
                "PASS  S3.4 live decisions: 4 decisions, one per answer of the feature stream: 2 scored windows, 1 for "
                "each of the 2 held-out realizations, each answered at the record that completes its window, and 2 of a "
                "filling window without a score",
                "NOTE  S3.4 live device timing, not graded, runtime reference: one inference of the head on cpu",
                # Offline against the device.
                "PASS  S3.4 parity same features: python -m ocudu_cuda_isac.sensing.presence replay, in a process of its "
                "own on the CPU, on the records the sensor kept: the 4 answers of the stream are those of the sensor "
                "line for line, host times aside, with 2 views of the same SHA-256",
                "PASS  S3.4 parity scores: 2 held-out windows joined by their identifier: on both sides the same "
                "contributing observations and the same SHA-256 of the input, 2 values as float32",
                "one artifact scores the same held-out feature offline and on the selected device",
                "PASS  S3.4 parity decisions: at the frozen threshold the 2 decisions of the device are those of the "
                "offline replay",
                "NOTE  S3.4 parity held-out episodes at the frozen threshold, read with the labels after the decisions "
                "were written, recorded and not graded here: ",
                # Causality and what a decision may carry.
                "PASS  S3.4 causality no score without a window: 2 answers of a filling window give a decision with the "
                "status insufficient_history and no score",
                "PASS  S3.4 causality future records: with the payload of the 2 records after the completing one "
                "replaced by 1e+30, the window ",
                "PASS  S3.4 causality line fields: with the peak, the validity counts, the axes and the host times of "
                "the line changed, and with 1000 added to each of the 5 values the head does not read",
                "PASS  S3.4 causality no label: decisions.jsonl, presence.json and features.jsonl of the sensor and "
                "head.json of the artifact name none of the 8 trajectory and episode identifiers of the set",
                "PASS  S3.4 refusals interface: 18 of 18 are refused by the interface with one error: a batch of two "
                "vectors",
                "PASS  S3.4 refusals command line: 8 of 8 are refused by python -m ocudu_cuda_isac.sensing.presence and "
                "by python scripts/train_presence.py with status 2 and one line, and nothing is written"):
            self.assertIn(line, lines)
        self.assertNotIn("rehearsal", run["said"])
        # The evidence.
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual((manifest["selector"], manifest["axes"]["runtime"], manifest["axes"]["model"],
                          manifest["axes"]["detector"], manifest["axes"]["training_domain"], manifest["host_rehearsal"]),
                         ("presence_head", "reference", "linear", "frozen_threshold", "receiver_executed", False))
        self.assertEqual({stage: entry["entry"] for stage, entry in manifest["entries"].items()},
                         {"normalizer": "zscore", "model": "linear", "detector": "frozen_threshold"})
        self.assertEqual([(item["name"], item["role"], item["status"], item["consumer_status"]) for item in manifest["runs"]],
                         [("training-00", "training", 0, 0), ("training-01", "training", 0, 0),
                          ("validation-00", "validation", 0, 0), ("held-out-00", "held_out", 0, 0)])
        self.assertEqual(manifest["episode_set"]["set_id"], "unit-set")
        self.assertEqual((manifest["artifact"]["artifact_sha256"], grade["artifact_sha256"], grade["runtime"]),
                         (grade["artifact_sha256"], manifest["artifact"]["artifact_sha256"], "reference"))
        # The order of the run: every training and validation run ends before the freeze, the freeze before
        # the first plan of a held-out entry, and that before the held-out run.
        order = manifest["order"]
        by_role = {role: [item for item in order["captures"] if item["role"] == role]
                   for role in ("training", "validation", "held_out")}
        self.assertLess(max(item["ended_monotonic_ns"] for item in by_role["training"] + by_role["validation"]),
                        order["frozen_monotonic_ns"])
        self.assertLess(order["frozen_monotonic_ns"], order["first_test_access_monotonic_ns"])
        self.assertLess(order["first_test_access_monotonic_ns"], by_role["held_out"][0]["started_monotonic_ns"])
        for name in ("grade.json", "grade.txt", "comparison.json", "manifest.json", "adapter-receipt.json", "spec.json",
                     "training.json", "split.json", "recover.json", "held_out_evaluation.json", "train.log",
                     "samples/training.jsonl", "samples/validation.jsonl", "train/freeze.json", "train/fit.json",
                     "train/training_manifest.json", "train/validation_manifest.json", "train/artifact/head.json",
                     "train/artifact/weights.f32", "records/training-00/plan.json", "records/training-00/channel-plan.json",
                     "records/training-00/replay-plan.json", "records/training-01/consumer/consumer.json",
                     "records/validation-00/native/ring/publications.jsonl", "features/training-00/features.jsonl",
                     "live/held-out-00/plan.json", "live/held-out-00/sensor/decisions.jsonl",
                     "live/held-out-00/sensor/presence.json", "live/held-out-00/sensor/features/features.jsonl",
                     "live/held-out-00/sensor/features/records/consumer.json", "parity/held-out-00.json",
                     "parity/held-out-00/decisions.jsonl"):
            self.assertTrue((out / name).is_file(), name)
        plan = json.loads((out / "live/held-out-00/plan.json").read_text())
        self.assertEqual((plan["schema"], plan["evidence"], plan["ring"]["slots"], plan["ring"]["attach"], plan["run_id"]),
                         ("pilot-ls-plan/v2", {"mode": "timed"}, 4, "required", "s34-held-out-00"))
        # No sample file and no manifest of the fit holds a held-out key, and the split table is the evaluator's.
        table = json.loads((out / "split.json").read_text())
        held = {key for row in table["episodes"] if row["group"] == "held_out" for key in row["keys"].values()}
        self.assertEqual(len(held), 2)
        for name in ("samples/training.jsonl", "samples/validation.jsonl", "train/training_manifest.json",
                     "train/validation_manifest.json", "train/freeze.json", "train/fit.json"):
            for key in held:
                self.assertNotIn(key, (out / name).read_text(), name)
        evaluation = json.loads((out / "held_out_evaluation.json").read_text())
        self.assertEqual((evaluation["graded"], evaluation["counts"]["target_present"]["episodes"],
                          evaluation["counts"]["target_absent"]["episodes"]), (False, 1, 1))
        comparison = json.loads((out / "comparison.json").read_text())
        self.assertEqual(comparison["parity"]["largest_logit_difference"], 0.0)
        self.assertEqual(comparison["live"]["scored"], 2)

    def test_a_second_reader_recovers_the_frozen_head_and_a_decision(self):
        out = self.reference["out"]
        recover = json.loads((out / "recover.json").read_text())
        self.assertEqual(recover, {"artifact": "train/artifact", "freeze": "train/freeze.json",
                                   "sensor": "live/held-out-00/sensor", "window": 0, "offline": "parity/held-out-00"})
        freeze = json.loads((out / recover["freeze"]).read_text())
        head = presence.inspect(presence.open_artifact(out / recover["artifact"], freeze["artifact"]["sha256"]))
        shown = presence.summary(presence.open_output(out / recover["sensor"]), recover["window"])
        self.assertEqual((shown["artifact_sha256"], shown["loads"], shown["source"], shown["scored_windows"]),
                         (freeze["artifact"]["sha256"], 1, "ring", 2))
        self.assertEqual(head["weights_sha256"],
                         hashlib.sha256((out / recover["artifact"] / "weights.f32").read_bytes()).hexdigest())
        self.assertEqual(head["columns"], ["peak_to_median_db", "profile_db@-100Hz"])
        offline = presence.summary(presence.open_output(out / recover["offline"]), recover["window"])
        self.assertEqual((offline["window"]["id"], offline["window"]["logit"], offline["window"]["input_sha256"]),
                         (shown["window"]["id"], shown["window"]["logit"], shown["window"]["input_sha256"]))
        # What the sensor wrote names nothing of the truth.
        for path in (out / recover["sensor"]).rglob("*"):
            if path.suffix in (".json", ".jsonl"):
                for word in ("position_m", "velocity_m_s", "target_present", "realization", "unit-held-out-line"):
                    self.assertNotIn(word, path.read_text(), path.name)

    # ---------------------------------------------------------------- faults of the receiver and of the sensor
    def test_each_fault_of_the_training_runs_fails_the_check_that_names_it(self):
        for fault, expected in (("capture-drop", ["records account"]), ("capture-crc", ["records decodes"]),
                                ("capture-fallback", ["records executed path"]),
                                ("capture-exit", ["records native runs"])):
            run = self.faulty(fault)
            with self.subTest(fault=fault):
                self.assertEqual((run["status"], run["grade"]["judgment"]), (1, "fail"))
                self.assertEqual(run["failed"], expected + blocked_after("records"))
                # Without the training records nothing is fitted and nothing of the held-out group is read.
                self.assertFalse((run["out"] / "train").exists())
                self.assertFalse((run["out"] / "live").exists())

    def test_each_fault_of_the_held_out_run_fails_the_check_that_names_it(self):
        cases = (("live-crc", ["live decodes"], "24 decodes gave the original bytes"),
                 ("live-fallback", ["live executed path"], "0 of 12 decodes requested from the CUDA processor"),
                 ("live-exit", ["live native run"], "the native tool exited with status 3"),
                 ("sensor-exit", ["live sensor"], "the sensor exited with status 3"),
                 ("no-output", ["live sensor"], "the sensor exited with status 1"),
                 ("wrong-artifact", ["live sensor"], "which is not the frozen one"),
                 ("two-loads", ["live sensor"], "the artifact was loaded 2 times in the process of the sensor"),
                 ("cold-sensor", ["live sensor"], "the sensor was not warm before the first record was published"),
                 ("slow-sensor", ["live account", "live decisions"], " dropped, "),
                 # Two samples that are the same hold no moving part: the leaked window is not even complete.
                 ("future-leak", ["live decisions"], "1 scored windows"))
        for fault, expected, said in cases:
            run = self.faulty(fault)
            with self.subTest(fault=fault):
                self.assertEqual((run["status"], run["grade"]["judgment"]), (1, "fail"))
                self.assertEqual(run["failed"], expected + blocked_after("live"))
                self.assertEqual(run["grade"]["cases_passed"], 6)
                self.assertIn(said, "".join(line for line in run["lines"].splitlines() if line.startswith("FAIL")))

    def test_scores_and_features_that_differ_offline_are_found(self):
        run = self.faulty("device-drift")
        self.assertEqual((run["status"], run["failed"]), (1, ["parity scores"]))
        self.assertIn("the logit differs by 1.000e-03", run["lines"])
        # A window on a sample the ring did not carry: its features and the digest of its input differ offline.
        run = self.faulty("wrong-sample")
        self.assertEqual((run["status"], run["failed"]), (1, ["parity same features", "parity scores"]))
        self.assertIn("the digest of the input differs: the two sides did not score the same feature", run["lines"])

    # ---------------------------------------------------------------- faults of what was frozen
    def test_a_normalizer_or_a_threshold_that_is_not_that_of_its_split_is_found(self):
        def other_mean(parts):
            parts["normalizer"]["mean"][0] += 0.5

        run = self.run_once(self.out / "mean", patch=self.rewritten(other_mean))
        self.assertEqual(run["status"], 1)
        self.assertEqual(run["failed"][0], "train normalizer")
        self.assertNotIn("train artifact", run["failed"])  # the artifact is whole: only its content is wrong

        def other_threshold(parts):
            parts["threshold"]["logit"] -= 1.0
            parts["threshold"]["probability"] = 0.5

        run = self.run_once(self.out / "threshold", patch=self.rewritten(other_threshold))
        self.assertEqual((run["status"], run["failed"]), (1, ["train threshold"] + blocked_after("train")))

    def test_a_freeze_record_that_does_not_bind_what_was_read_is_found(self):
        def another_artifact(run):
            data = bytearray((run / "artifact/weights.f32").read_bytes())
            data[0] ^= 1
            (run / "artifact/weights.f32").write_bytes(bytes(data))

        run = self.run_once(self.out / "weights", patch=self.after_training(another_artifact))
        self.assertEqual((run["status"], run["failed"]), (1, ["train script"] + blocked_after("train")))
        self.assertIn("the checkpoint was altered", run["lines"])

        def stale_manifest(run):
            manifest = json.loads((run / "validation_manifest.json").read_text())
            manifest["windows"] += 1
            (run / "validation_manifest.json").write_text(json.dumps(manifest))

        run = self.run_once(self.out / "stale", patch=self.after_training(stale_manifest))
        self.assertEqual((run["status"], run["failed"]), (1, ["freeze hash chain"] + blocked_after("freeze")))

        def leaked(run):
            table = json.loads((run.parent / "split.json").read_text())
            key = next(row["keys"]["b"] for row in table["episodes"] if row["group"] == "held_out")
            manifest = json.loads((run / "training_manifest.json").read_text())
            manifest["episode_keys"].append(key)
            manifest["window_ids"].append(f"1/{key}/1")
            (run / "training_manifest.json").write_text(json.dumps(manifest))
            self.refreeze(run)

        run = self.run_once(self.out / "leaked", patch=self.after_training(leaked))
        self.assertEqual((run["status"], run["failed"]), (1, ["freeze no test access"] + blocked_after("freeze")))
        self.assertIn("1 keys are in the wrong manifest and 2 held-out identifiers are in a manifest", run["lines"])
        # Nothing of the held-out group is replayed once the freeze does not hold.
        self.assertFalse((run["out"] / "live").exists())

    def test_a_head_that_reads_another_field_of_a_line_is_found(self):
        def leaky():
            real = presence.Head.prepare

            def prepare(head, names, values):
                # Moving over static power lies far below 0 dB in this set. The head must not read it at all.
                x = real(head, names, values)
                return x + np.float32(1.0 if values[names.index("moving_to_static_db")] > 0 else 0.0)
            presence.Head.prepare = prepare
            return lambda: setattr(presence.Head, "prepare", real)

        run = self.run_once(self.out / "leaky", patch=leaky)
        self.assertEqual((run["status"], run["failed"]), (1, ["causality line fields"]))

    # ---------------------------------------------------------------- the configuration and the control grade
    def test_the_expectations_of_the_configuration_are_graded(self):
        cases = ((lambda settings: settings["set"]["expect"]["episodes"].update(training=300),
                  ["split episode set"] + blocked_after("split")),
                 (lambda settings: settings["set"]["expect"].update(set_id="demo-set-v1"),
                  ["split episode set"] + blocked_after("split")),
                 (lambda settings: settings["spec"]["window"].update(periods=4, hop_periods=4),
                  ["control specification"] + blocked_after("control")),
                 (lambda settings: settings["training"].update(columns=["peak_to_mean_db"]),
                  ["train script"] + blocked_after("train")),
                 (lambda settings: settings["training"]["threshold"].update(min_negatives=2),
                  ["train script"] + blocked_after("train")),
                 (lambda settings: settings["training"].update(min_per_class=3), ["train script"] + blocked_after("train")))
        for number, (edit, expected) in enumerate(cases):
            run = self.run_once(self.out / f"expect-{number}", config=self.changed_config(edit, f"config-{number}.json"))
            with self.subTest(expected=expected[0]):
                self.assertEqual((run["status"], run["failed"]), (1, expected))
        self.assertIn("the validation split holds 1 target-absent windows, and the calibration needs 2",
                      self.run_once(self.out / "negatives", config=self.changed_config(
                          lambda settings: settings["training"]["threshold"].update(min_negatives=2), "few.json"))["lines"])

    def test_a_set_that_cannot_be_replayed_is_refused_before_a_grid_is_filled(self):
        short = self.out / "short"
        double.analytic_set(double.set_config(self.out / "short.json", self.grant, double.trajectories(2, 1, 1, periods=1)),
                            short)
        run = self.run_once(self.out / "run-short", episode_set=short)
        self.assertEqual((run["status"], run["failed"]), (1, ["split replayable"] + blocked_after("split")))
        self.assertIn("4 entries are shorter than a window", run["lines"])
        # Another grant than the one whose grids the pair binds.
        foreign = self.out / "foreign"
        double.analytic_set(double.set_config(self.out / "foreign.json", ROOT / "configs/grants/two_prb_offset.json",
                                              double.trajectories(2, 1, 1)), foreign)
        run = self.run_once(self.out / "run-foreign", episode_set=foreign)
        self.assertEqual((run["status"], run["failed"]), (1, ["split replayable"] + blocked_after("split")))
        self.assertFalse((run["out"] / "records").exists())
        # A set with one changed byte does not verify.
        damaged = self.out / "damaged"
        shutil.copytree(self.set, damaged)
        entry = next((damaged / "episodes").iterdir())
        data = bytearray((entry / "b_h.npy").read_bytes())
        data[-1] ^= 1
        (entry / "b_h.npy").write_bytes(bytes(data))
        run = self.run_once(self.out / "run-damaged", episode_set=damaged)
        self.assertEqual(run["failed"], ["split episode set"] + blocked_after("split"))

    def test_the_control_must_be_the_accepted_sprint_3_3_run(self):
        def changed_control(edit):
            copy = self.out / f"control-{len(list(self.out.glob('control-*')))}"
            shutil.copytree(self.control, copy)
            edit(copy)
            return copy

        def failed_grade(copy):
            grade = json.loads((copy / "grade.json").read_text())
            grade.update(judgment="fail", checks_failed=1)
            (copy / "grade.json").write_text(json.dumps(grade))

        run = self.run_once(self.out / "failed", control=changed_control(failed_grade))
        self.assertEqual((run["status"], run["failed"]), (1, ["control accepted run"] + blocked_after("control")))
        run = self.run_once(self.out / "no-plan", control=changed_control(lambda copy: (copy / "live/run/plan.json").unlink()))
        self.assertEqual(run["failed"], ["control reference entry"] + blocked_after("control"))
        run = self.run_once(self.out / "empty", control=changed_control(lambda copy: (copy / "grade.json").unlink()))
        self.assertEqual(run["failed"], ["control accepted run"] + blocked_after("control"))
        # A rehearsal needs a rehearsed control, and it scores on the CPU whatever the configuration selects.
        run = self.run_once(self.out / "rehearsal", extra=("--host-rehearsal",))
        self.assertEqual(run["failed"][0], "control accepted run")
        self.assertIn("this gate needs the judgment rehearsal", run["lines"])
        self.assertEqual(run["grade"]["judgment"], "fail")

    def test_setup_errors_leave_no_evidence(self):
        driver = self.driver

        def refused(arguments, message, receipt=None):
            saved = driver.native_adapter.verified_receipt
            driver.native_adapter.verified_receipt = lambda build_dir: receipt or self.receipt_of()
            try:
                with contextlib.redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit) as stop:
                    driver.main([str(word) for word in arguments])
            finally:
                driver.native_adapter.verified_receipt = saved
            self.assertEqual(stop.exception.code, 2)
            self.assertIn(message, errors.getvalue())

        base = ["--s33-run", self.control, "--episode-set", self.set]
        refused(["--config", self.config, *base, "--output-dir", self.reference["out"]], "Evidence is never overwritten")
        for number, (edit, message) in enumerate((
                (lambda settings: settings["ring"].update(slots=8), "ring.slots is 8"),
                (lambda settings: settings.update(selector="pilot_features"), "for the selector presence_head"),
                (lambda settings: settings["sensor"].update(device="tpu"), 'the device "cuda" or "cpu"'),
                (lambda settings: settings["training"].update(l2=0), "l2 must lie"),
                (lambda settings: settings["set"].update(entries_per_run=0), "set requires entries_per_run"),
                (lambda settings: settings["control"].update(run="stream"), 'its run "live"'),
                (lambda settings: settings["tolerance"].update(logit_absolute=0), "each a positive number"),
                (lambda settings: settings["receiver"].update(keep_grids="no"), "keep_grids, true or false"))):
            config = self.changed_config(edit, f"bad-{number}.json")
            refused(["--config", config, *base, "--output-dir", self.out / f"bad-{number}"], message)
            self.assertFalse((self.out / f"bad-{number}").exists())
        refused(["--config", self.config, "--s33-run", self.control, "--episode-set", self.out / "nowhere", "--output-dir",
                 self.out / "no-set"], "Trace one: make episode-set")
        receipt = self.receipt_of()
        del receipt["executables"]["obs_tail"]
        refused(["--config", self.config, *base, "--output-dir", self.out / "no-tool"], "holds no obs_tail", receipt)


if __name__ == "__main__":
    unittest.main()
