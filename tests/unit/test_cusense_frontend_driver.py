"""Check the Sprint 3.5 integration driver against doubles of the native tool, its two rings and the ray tracer.

The control is a Sprint 3.4 run that the driver of that sprint grades on the doubles of its own
driver tests, and the stimulus is the episode set of those tests: two training trajectories with
one pair each, one validation pair and one held-out pair, two periods long, on the grant of the
demonstration, written with the free-space model in place of the ray tracer. The sensing side,
the training script and the evaluator under the driver are the real ones. The double of this
sprint (tests/unit/cusense_frontend_double.py) only stands in for the native tool and its rings,
with the receiver's estimate of the double of Sprint 5.2. The model has the layers of the
published one at a few channels, and what it learns from eight windows means nothing: every
check of the driver still has a known answer, and each injected fault must fail the check that
names it. The small fit takes all eight windows in one batch, so that its batch statistics are those of
the set. NumPy and PyTorch are needed, as for the driver.
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
    import test_presence_head_driver as control_tests  # noqa: E402
    import cusense_frontend_double as double  # noqa: E402
    from ocudu_cuda_isac.sensing import cusense, presence  # noqa: E402
    sys.path.insert(0, str(ROOT / "scripts"))
    import train_cusense  # noqa: E402

CHECKS = 47
CASES = 12
ORDER = ("control", "split", "records", "windows", "train", "second-input", "freeze", "live", "parity", "evaluation",
         "causality", "refusals")
ENVIRONMENT = ("CUSENSE_FRONTEND_FAULT", "NATIVE_ESTIMATE_FAULT", "PRESENCE_HEAD_FAULT", "PILOT_FEATURES_FAULT",
               "SNAPSHOT_RING_FAULT", "PILOT_LS_FAULT", "PILOT_LS_BUILD", "REPLAY_PUSCH_FAULT", "REPLAY_PUSCH_BUILD")


def load_driver():
    spec = importlib.util.spec_from_file_location("cusense_frontend_driver", ROOT / "tests/integration/cusense_frontend.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def blocked_after(case, done=()):
    """The checks that the cases after one failed case report: each is blocked on its control."""
    return [f"{later} control" for later in ORDER[ORDER.index(case) + 1:] if later not in done]


class Scenarios:
    """The model receiver of the double, once per stimulus: a second run on the same entries copies the first.

    The key is what a run replays and how: the channel files by their hashes, both rings, the
    capture, the evidence mode, the consumers and every injected fault. A copy is what the tool
    and the consumers wrote for a plan of the same content.
    """

    def __init__(self, directory, real):
        self.directory, self.real, self.made = directory, real, {}

    def __call__(self, tools, plan_path, native, directory, consumers, extra=(), tool="pilot_ls"):
        plan = json.loads(Path(plan_path).read_text())
        channel = json.loads(Path(json.loads(Path(plan["replay_plan"]["path"]).read_text())["channel_plan"]["path"]).read_text())
        key = hashlib.sha256(json.dumps([
            [(source["h"]["sha256"], source["w"]["sha256"]) for source in channel["sources"]], plan["ring"],
            plan["estimate"], plan["evidence"], plan["observer"], plan["run_id"],
            os.environ.get("SNAPSHOT_RING_FAULT", ""), os.environ.get("NATIVE_ESTIMATE_FAULT", ""),
            [(consumer["stream"], consumer["reader"], consumer["wait"], consumer.get("hold_us")) for consumer in consumers]],
            sort_keys=True).encode("ascii")).hexdigest()
        if key not in self.made:
            result = self.real(tools, plan_path, native, directory, consumers, extra, tool)
            home = self.directory / key[:16]
            if native.is_dir():
                shutil.copytree(native, home / "native")
            for consumer in consumers:
                if consumer["archive"].is_dir():
                    shutil.copytree(consumer["archive"], home / f"archive-{consumer['stream']}")
            home.mkdir(parents=True, exist_ok=True)
            shutil.copy(directory / "native.log", home / "native.log")
            self.made[key] = (home, result)
            return result
        home, result = self.made[key]
        if (home / "native").is_dir():
            shutil.copytree(home / "native", native)
        for consumer in consumers:
            if (home / f"archive-{consumer['stream']}").is_dir():
                shutil.copytree(home / f"archive-{consumer['stream']}", consumer["archive"])
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
        if words[1:3] == ["-m", "ocudu_cuda_isac.sensing.cusense"]:
            main, arguments = cusense.main, words[3:]
        elif words[1:3] == ["-m", "ocudu_cuda_isac.sensing.presence"]:
            main, arguments = presence.main, words[3:]
        elif words[1].endswith("train_cusense.py"):
            main, arguments = train_cusense.main, words[2:]
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
        # The accepted Sprint 3.4 run: the driver of that sprint on the doubles of its tools.
        cls.stage = control_tests.DriverTests
        cls.stage.setUpClass()
        cls.addClassCleanup(cls.release)
        assert cls.stage.reference["status"] == 0, cls.stage.reference["failed"]
        cls.control = cls.stage.reference["out"]
        cls.set = cls.stage.set
        cls.root = cls.stage.root / "s35"
        cls.root.mkdir()
        grid = json.loads((cls.set / "manifest.json").read_text())["position_grid"]
        cls.settings = json.loads((ROOT / "configs/integration/cusense_frontend.json").read_text())
        # The small set: two windows per realization, one entry per run, a model of a few channels on the CPU.
        cls.settings["set"] = {"entries_per_run": 1, "expect": {
            "set_id": "unit-set", "episodes": {"training": 2, "validation": 1, "held_out": 1},
            "windows_per_realization": 2, "grid": [grid["rows"], grid["cols"]]}}
        cls.settings["training"]["model"].update(stem_channels=4, block_channels=[4, 8, 8], hidden=[8, 8],
                                                 dropout=[0.0, 0.0])
        cls.settings["training"]["optimizer"].update(epochs=60, batch_size=8, learning_rate=0.02)
        cls.settings["training"]["loss"].update(sigma_cells=2.0)
        cls.settings["training"].update(min_per_class=1, device="cpu")
        cls.settings["training"]["threshold"].update(min_negatives=1)
        cls.settings["training"]["smoothing"].update(maps=2)
        cls.settings["sensor"]["device"] = "cpu"
        # Eight training windows of two trajectories say nothing about a third: the floor on the validation
        # windows is graded in a test of its own, and is zero here.
        cls.settings["expect"]["validation_auc_min"] = 0.0
        cls.config = cls.write_config(cls.root / "config.json", cls.settings)
        cls.tool = cls.root / "tool"
        cls.tool.write_text("a double\n")
        cls.scenarios = Scenarios(cls.root / "scenarios", double.estimate_double.scenario)
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
        series = cls.driver.receiver_patch.identity(cls.driver.receiver_patch.load_series())
        return {"executables": {name: {"path": str(cls.tool), "sha256": hashlib.sha256(cls.tool.read_bytes()).hexdigest()}
                                for name in ("pilot_ls", "obs_tail", "obs_feed", "replay_pusch")},
                "ocudu_build": "double-o3", "pin": {"cuda": True}, "patch": dict(series, tree_kind="patched-worktree"),
                "o3_route": {"compiled": True, "tap_interface_version": 1}}

    @classmethod
    def run_once(cls, out, fault=None, config=None, control=None, episode_set=None, extra=(), receipt=None, patch=None,
                 processes=False):
        driver = cls.driver
        saved = (driver.native_adapter.verified_receipt, driver.run_capture, driver.run_live, double.estimate_double.scenario,
                 driver.run_tool)
        driver.native_adapter.verified_receipt = lambda build_dir: receipt or cls.receipt_of()
        driver.run_capture, driver.run_live = double.capture, double.live
        if not processes:
            driver.run_tool = run_in_process(driver.run_tool)
        double.estimate_double.scenario = cls.scenarios
        kept = {key: os.environ.get(key) for key in ENVIRONMENT}
        for key in kept:
            os.environ.pop(key, None)
        if fault:
            os.environ["CUSENSE_FRONTEND_FAULT"] = fault
        arguments = ["--config", str(config or cls.config), "--s34-run", str(control or cls.control), "--episode-set",
                     str(episode_set or cls.set), "--output-dir", str(out), *extra]
        undo = patch() if patch else None
        loads, head_loads = cusense.Model.LOADS, presence.Head.LOADS
        try:
            with contextlib.redirect_stdout(io.StringIO()) as said:
                status = driver.main(arguments)
        finally:
            if undo:
                undo()
            cusense.Model.LOADS, presence.Head.LOADS = loads, head_loads
            (driver.native_adapter.verified_receipt, driver.run_capture, driver.run_live,
             double.estimate_double.scenario, driver.run_tool) = saved
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

    def after_training(self, edit, arm="o3"):
        """A patch that lets the training script run and then changes what it froze for one arm."""
        driver = self.driver

        def patch():
            real = driver.run_tool

            def run_tool(command, log, timeout=1800):
                result = real(command, log, timeout)
                words = [str(word) for word in command]
                if any(word.endswith("train_cusense.py") for word in words) and result.returncode == 0:
                    target = Path(words[words.index("--output-dir") + 1])
                    if target.name == arm and target.parent.name == "train":
                        edit(target)
                return result
            driver.run_tool = run_tool
            return lambda: setattr(driver, "run_tool", real)
        return patch

    @staticmethod
    def refreeze(run, artifact=False):
        """Write the freeze record of a changed training run again, so that only the change itself is wrong."""
        freeze = json.loads((run / "freeze.json").read_text())
        if artifact:
            manifest = json.loads((run / "artifact/model.json").read_text())
            freeze["artifact"].update(sha256=manifest["sha256"], template_count_sha256=manifest["template"]["count_sha256"],
                                      **{f"{part}_sha256": manifest[part]["sha256"]
                                         for part in ("weights", "template", "normalizer", "grid", "threshold")})
            freeze["threshold"] = manifest["threshold"]
        freeze["artifact"]["files"] = {path.name: {"bytes": path.stat().st_size,
                                                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                                       for path in sorted((run / "artifact").iterdir())}
        for role in ("training", "validation"):
            freeze["manifests"][role]["sha256"] = hashlib.sha256((run / f"{role}_manifest.json").read_bytes()).hexdigest()
        (run / "freeze.json").write_text(json.dumps(freeze))

    @classmethod
    def rewrite_artifact(cls, run, change):
        """Write the artifact of a training run again, whole and hashed, with one part changed, and its freeze record."""
        artifact = cusense.open_artifact(run / "artifact")
        manifest = json.loads(json.dumps(artifact.manifest))
        net = cusense.cusense_cnn.unpack(cusense.cusense_cnn.CuSenseCNN(manifest["model"]["config"]), artifact.weights,
                                         manifest["weights"]["tensors"]).eval()
        parts = {"template": dict(artifact.template, fitted_on=manifest["template"]["fitted_on"]),
                 "normalizer": {name: manifest["normalizer"][name] for name in ("mean", "scale", "fitted_on")},
                 "threshold": {name: manifest["threshold"][name] for name in presence.THRESHOLD_FIELDS}}
        change(parts)
        shutil.rmtree(run / "artifact")
        cusense.write_artifact(run / "artifact", manifest["model_id"], manifest["input"]["spec"],
                               manifest["input"]["elements"], manifest["input"]["producer_steps"], parts["template"],
                               parts["normalizer"], net, manifest["grid"], parts["threshold"], manifest["smoothing"],
                               manifest["model"]["target"]["sigma_cells"])
        cls.refreeze(run, True)

    def rewritten(self, change, arm="o3"):
        """A patch that changes one part of what the training script froze for one arm, and nothing else."""
        return self.after_training(lambda run: self.rewrite_artifact(run, change), arm)

    # ---------------------------------------------------------------- the reference
    def test_the_reference_passes_every_check(self):
        run = self.reference
        grade, lines, out = run["grade"], run["lines"], run["out"]
        self.assertEqual((run["status"], run["failed"]), (0, []), lines)
        self.assertEqual((grade["judgment"], grade["hardware_gate"], grade["cases"], grade["cases_passed"], grade["checks"]),
                         ("pass", "closed", CASES, CASES, CHECKS))
        self.assertIn(f"cusense_frontend: {CASES} of {CASES} cases passed, {CHECKS} of {CHECKS} checks\n", lines)
        for line in (
                "PASS  S3.5 control accepted run: the Sprint 3.4 driver graded reference as pass: 10 of 10 cases",
                "PASS  S3.5 control linear head: the head dd-linear-v1 ",
                "it reads the vector of the specification dd-uniform-8 on records of the definition ",
                "PASS  S3.5 control transmit grids: the channel plan of the run training-00 of that gate binds the "
                "transmit grid of 6 grant(s)",
                "PASS  S3.5 control estimate route: the adapter is linked against double-o3, the receiver with the series ",
                "reports the tap interface version 1: the build that Sprint 5.2 accepted",
                # The set and its split.
                "PASS  S3.5 split episode set: the set unit-set (status proposed) verifies with every episode in it: "
                "2 training, 1 validation and 1 held-out pairs",
                "it is the set the accepted Sprint 3.4 run was graded on",
                "PASS  S3.5 split assignment: an episode has the group of its trajectory",
                "8 realizations under distinct opaque keys, and a key names no group and no case",
                "PASS  S3.5 split replayable: every entry schedules the slots [7, 8, 9] of each of its periods of 10 "
                "slots and gives 2 windows of one period per realization",
                "the position grid of the set is 50 rows by 34 columns of cells",
                # The receiver on the training and the validation entries, both stages.
                "PASS  S3.5 records native runs: 3 runs of pilot_ls on plans of version 3 in the mode timed, both stages "
                "on rings of 8 and 8 slots: 2 for the 2 training entries, 1 for the 1 validation entries, at most 1 "
                "entries per run, 36 opportunities in all",
                "PASS  S3.5 records decodes: 72 of 72 decodes gave the original bytes of their transport block with a "
                "CRC pass, on cpu and cuda",
                "PASS  S3.5 records executed path: 36 of 36 decodes requested from the CUDA processor ran its resident "
                "path, the only place where the receiver's estimate exists",
                "PASS  S3.5 records account: 3 runs, 36 opportunities: stage O2: 36 eligible, 36 published, 0 dropped, "
                "36 released, 0 pending, 36 kept; stage O3: 36 eligible, 36 published, 0 dropped, 36 released, 0 pending, "
                "36 kept",
                # Stage 2 up to the window, for both input entries.
                "PASS  S3.5 windows windows: o3_fp16: 36 records of the stage O3 in sequence order through the stream of "
                "cusense-o3-period, 8 complete windows for 4 training realizations, 4 complete windows for 2 validation "
                "realizations, 24 answers of a filling window, 0 gaps, 0 invalid, 0 rejected; o2_comb: 36 records of "
                "the stage O2",
                "2 windows per realization, one per period, and every output verifies with its reader",
                "PASS  S3.5 windows same slots: the 12 windows of the stage O3 and the 12 of the stage O2 have the same "
                "identifiers",
                "PASS  S3.5 windows oracle: 6 of 6 windows of the stage O3 agree with the coherent average computed "
                "again from the payloads, value by value",
                "6 of 6 windows of the stage O2 agree",
                "PASS  S3.5 windows samples: each complete window is joined to the split table by the episode key of "
                "its records, after the windows were written: 4 target-present and 4 target-absent training windows "
                "from 2 trajectories, 2 and 2 validation windows from 1",
                "windows of shape [4, 612, 2] and [4, 306, 2] as complex64 with their counts",
                # The model on the receiver's own estimate.
                "PASS  S3.5 train script: python scripts/train_cusense.py read the two sample sets, the specification, "
                "the grid and the configuration and no other input, and froze the model cusense-o3-v1 ",
                "a plain-block 1D CNN of 16421 parameters from [1, 4, 612] to a map of 50 by 34 cells and a presence "
                "logit, on the input entry o3_fp16",
                "PASS  S3.5 train artifact: the deployment artifact is model.json, template.c64, template.count.u4, "
                "weights.f32: the template ",
                " of shape [4, 612, 2], the normalizer ",
                "66036 bytes of weights ",
                "PASS  S3.5 train statistics: the template H_B, the complex mean of the records behind the 4 "
                "target-absent training windows over their valid elements, 12 to 12 records per element",
                "of the z-score over the 8 training windows, both computed here again from the training sample set "
                "alone, are those of the artifact bit for bit; with the 4 validation windows added the mean would differ",
                "PASS  S3.5 train threshold: the frozen threshold is the logit ",
                "with a false-alarm target of 0.05 the rule of Sprint 3.4 allows 0 of the 2 target-absent validation "
                "windows above it, and 0 are",
                "PASS  S3.5 train fit: the joint loss, binary cross-entropy of the presence logit on every window plus "
                "the KL divergence of the map on the target-present ones, falls from 4.035572, the value of a uniform "
                "map and a logit of zero, to ",
                " in 60 epochs of Adam on cpu; the artifact read back scores the windows of the fit within 0.0e+00",
                "and the floor that the configuration declares for a model that learned presence is 0.0",
                "NOTE  S3.5 train recorded, not graded, on the samples of the fit: training: ",
                # The same on the pilot comb.
                "PASS  S3.5 second-input script: python scripts/train_cusense.py read the two sample sets",
                "froze the model cusense-o2-v1 ",
                "from [1, 4, 306] to a map of 50 by 34 cells and a presence logit, on the input entry o2_comb",
                "PASS  S3.5 second-input artifact: ",
                "PASS  S3.5 second-input statistics: ",
                "PASS  S3.5 second-input threshold: ",
                "PASS  S3.5 second-input fit: ",
                # The freeze.
                "PASS  S3.5 freeze hash chain: train/o3/freeze.json names the artifact ",
                "train/o2/freeze.json names the artifact ",
                "PASS  S3.5 freeze no test access: the manifests of both arms list 4 training and 2 validation episode "
                "keys, each of its own split; none of the 2 held-out keys and no window of one is in any",
                # The held-out run with the sensor.
                "PASS  S3.5 live after the freeze: the first plan of a held-out entry is written ",
                "PASS  S3.5 live native run: 1 run(s) of pilot_ls in the mode timed for the 1 held-out entries, 12 "
                "opportunities, with the cuSense-like sensor attached to the ring of the stage O3 in the mode wait and a "
                "native consumer on the ring of the stage O2",
                "PASS  S3.5 live decodes: 24 of 24 decodes gave the original bytes of their transport block",
                "PASS  S3.5 live executed path: 12 of 12 decodes requested from the CUDA processor ran its resident path",
                "PASS  S3.5 live sensor: python -m ocudu_cuda_isac.sensing.cusense follow, a second process on the "
                "standard-library reader, loaded the frozen artifact ",
                " once and ran the model on cpu, with its front end in NumPy on the CPU",
                "it read 12 records of the stage O3 until the producer said that none follows, with no stale read, and "
                "its output verifies",
                "PASS  S3.5 live account: 1 runs, 12 opportunities: stage O2: 12 eligible, 12 published, 0 dropped, 12 "
                "released, 0 pending, 12 kept; stage O3: 12 eligible, 12 published, 0 dropped, 12 released, 0 pending, "
                "12 kept",
                "PASS  S3.5 live decisions: 12 decisions, one per answer of the window stream: 4 scored windows, 2 for "
                "each of the 2 held-out realizations, each answered at the record that completes its period, and 8 of a "
                "filling window without a score",
                "NOTE  S3.5 live device timing, not graded, runtime reference: one batch-one inference of the model on "
                "cpu",
                # Offline against the device.
                "PASS  S3.5 parity same windows: python -m ocudu_cuda_isac.sensing.cusense replay, in a process of its "
                "own on the CPU, on the records the sensor kept: the 12 answers of the stream are those of the sensor "
                "line for line, host times aside, with 4 window tensors of the same SHA-256",
                "PASS  S3.5 parity scores: 4 held-out windows joined by their identifier: on both sides the same "
                "contributing observations and the same SHA-256 of the input, [1, 4, 612] as float32",
                "one artifact gives the same map and score for the same held-out window offline and on the selected "
                "device",
                "PASS  S3.5 parity decisions: at the frozen threshold the 4 decisions of the device are those of the "
                "offline replay; the cell of the largest value of the map is the same in 4 of 4 windows",
                # The second input, the linear head and the truth.
                "PASS  S3.5 evaluation second input: python -m ocudu_cuda_isac.sensing.cusense replay with the frozen "
                "artifact ",
                "4 scored windows, 2 for each of the 2 realizations, with the identifiers and the contributing "
                "opportunities of the windows that the sensor scored on the stage O3",
                "PASS  S3.5 evaluation linear head: python -m ocudu_cuda_isac.sensing.presence replay with the head ",
                "one decision for each of the 2 held-out realizations at its frozen threshold, none missing; its one "
                "window of 2 periods spans the periods that this branch cuts into 2 windows",
                "PASS  S3.5 evaluation evaluation: cusense_eval.json: presence by label with every denominator, 1 "
                "target-present and 1 target-absent held-out episodes by one prospectively selected window, the last of "
                "each episode, and 2 windows of each by window",
                "PASS  S3.5 evaluation reference: cusense_reference.json records the measured position error beside "
                "the published mean of 73 cm with \"graded\": false, and lists 10 differences between the two",
                "is not compared with the published 241 us: that comparison is made on the native core by S3.6",
                "NOTE  S3.5 evaluation held-out episodes at the frozen thresholds, read with the truth after the "
                "decisions were written, recorded and not graded: o3_fp16: ",
                "; o2_comb: ",
                "; linear head of Sprint 3.4: ",
                # Causality and what a decision may carry.
                "PASS  S3.5 causality no score without a window: 8 answers of a filling window give a decision under "
                "their status and no score, no map, no location and no input digest; 4 complete windows are scored",
                "PASS  S3.5 causality future records: with the payload of the 3 records after the completing one "
                "replaced by 6e+04, the window ",
                "PASS  S3.5 causality record fields: with the cell, the UE, the grant, the valid count and the reserved "
                "field of every record of the realization changed, every decision is the same line",
                "a stream reads 30 named fields of a header, the validity bits and the payload, and not cell, digest, "
                "digest_kind, grant, header_bytes, host_bytes, reserved, schema, ue, valid",
                "PASS  S3.5 causality no label: decisions.jsonl, cusense.json and the window output of the sensor and "
                "model.json of both artifacts name none of the 8 trajectory and episode identifiers of the set",
                "PASS  S3.5 refusals interface: 23 of 23 are refused by the interface with one error: a batch of two "
                "windows",
                "PASS  S3.5 refusals command line: 9 of 9 are refused by python -m ocudu_cuda_isac.sensing.cusense and "
                "by python scripts/train_cusense.py with status 2 and one line, and nothing is written"):
            self.assertIn(line, lines)
        self.assertNotIn("rehearsal", run["said"])
        self.assertTrue(lines.isascii())
        # The evidence.
        manifest = json.loads((out / "manifest.json").read_text())
        axes = manifest["axes"]
        self.assertEqual((manifest["selector"], axes["runtime"], axes["model"], axes["detector"], axes["heads"],
                          axes["clutter"], axes["window"], axes["smoothing"], axes["training_domain"],
                          manifest["host_rehearsal"]),
                         ("cusense_frontend", "reference", "cusense_cnn", "frozen_threshold", ["location_grid", "presence"],
                          "background_magnitude", "coherent_mean", "map_mean_kalman", "receiver_executed", False))
        self.assertEqual({stage: entry["entry"] for stage, entry in manifest["entries"]["o3"].items()},
                         {"input": "o3_fp16", "ingest": "ocudu_o3_fp16", "window": "coherent_mean",
                          "clutter": "background_magnitude", "symbols": "dmrs_mean", "normalizer": "zscore_global",
                          "model": "cusense_cnn", "detector": "frozen_threshold", "smoothing": "map_mean_kalman"})
        self.assertEqual((manifest["entries"]["o2"]["input"]["entry"], manifest["entries"]["o2"]["ingest"]["entry"]),
                         ("o2_comb", "ocudu_o2_comb_c64"))
        self.assertTrue(all(entry["runtime"] == "reference" for arm in manifest["entries"].values() for entry in arm.values()))
        self.assertEqual([(item["name"], item["role"], item["status"], item["consumer_status"]) for item in manifest["runs"]],
                         [("training-00", "training", 0, {"o2": 0, "o3": 0}), ("training-01", "training", 0, {"o2": 0, "o3": 0}),
                          ("validation-00", "validation", 0, {"o2": 0, "o3": 0}),
                          ("held-out-00", "held_out", 0, {"o2": 0, "o3": 0})])
        self.assertEqual(manifest["episode_set"]["set_id"], "unit-set")
        self.assertEqual((manifest["artifacts"]["o3"]["artifact_sha256"], manifest["artifacts"]["o2"]["artifact_sha256"],
                          grade["runtime"]),
                         (grade["artifact_sha256"]["o3"], grade["artifact_sha256"]["o2"], "reference"))
        self.assertEqual(grade["graded_quality"]["validation_auc_min"], 0.0)
        # The order of the run: every training and validation run ends before the freeze, the freeze before
        # the first plan of a held-out entry, and that before the held-out run.
        order = manifest["order"]
        by_role = {role: [item for item in order["captures"] if item["role"] == role]
                   for role in ("training", "validation", "held_out")}
        self.assertLess(max(item["ended_monotonic_ns"] for item in by_role["training"] + by_role["validation"]),
                        order["frozen_monotonic_ns"])
        self.assertLess(order["frozen_monotonic_ns"], order["first_test_access_monotonic_ns"])
        self.assertLess(order["first_test_access_monotonic_ns"], by_role["held_out"][0]["started_monotonic_ns"])
        for name in ("grade.json", "grade.txt", "comparison.json", "manifest.json", "adapter-receipt.json", "specs/o3.json",
                     "specs/o2.json", "grid.json", "training-o3.json", "training-o2.json", "split.json", "recover.json",
                     "cusense_eval.json", "cusense_reference.json", "train-o3.log", "train-o2.log",
                     "samples/o3/training.jsonl", "samples/o3/training.c64", "samples/o3/training.u1",
                     "samples/o2/validation.jsonl", "train/o3/freeze.json", "train/o3/fit.json",
                     "train/o3/training_manifest.json", "train/o3/validation_manifest.json", "train/o3/artifact/model.json",
                     "train/o3/artifact/template.c64", "train/o3/artifact/template.count.u4",
                     "train/o3/artifact/weights.f32", "train/o2/artifact/model.json", "records/training-00/plan.json",
                     "records/training-00/channel-plan.json", "records/training-00/replay-plan.json",
                     "records/training-01/consumer-o2/consumer.json", "records/training-01/consumer-o3/consumer.json",
                     "records/validation-00/native/estimate/captures.jsonl", "windows/o3/training-00/windows.jsonl",
                     "windows/o2/training-00/windows.json", "live/held-out-00/plan.json",
                     "live/held-out-00/sensor/decisions.jsonl", "live/held-out-00/sensor/cusense.json",
                     "live/held-out-00/sensor/inputs.f32", "live/held-out-00/sensor/maps.f32",
                     "live/held-out-00/sensor/windows/windows.jsonl", "live/held-out-00/sensor/windows/records/consumer.json",
                     "live/held-out-00/consumer-o2/consumer.json", "parity/held-out-00.json",
                     "parity/held-out-00/decisions.jsonl", "evaluation/o2/held-out-00/decisions.jsonl",
                     "evaluation/linear/held-out-00/decisions.jsonl"):
            self.assertTrue((out / name).is_file(), name)
        plan = json.loads((out / "live/held-out-00/plan.json").read_text())
        self.assertEqual((plan["schema"], plan["evidence"], plan["ring"]["stream"], plan["ring"]["slots"],
                          plan["estimate"]["ring"]["stream"], plan["estimate"]["ring"]["slots"],
                          plan["estimate"]["ring"]["attach"], plan["run_id"]),
                         ("pilot-ls-plan/v3", {"mode": "timed"}, "o2", 8, "o3", 8, "required", "s35-held-out-00"))
        # No sample set and no manifest of a fit holds a held-out key, and the split table is the evaluator's.
        table = json.loads((out / "split.json").read_text())
        held = {key for row in table["episodes"] if row["group"] == "held_out" for key in row["keys"].values()}
        self.assertEqual(len(held), 2)
        for arm in ("o3", "o2"):
            for name in (f"samples/{arm}/training.jsonl", f"samples/{arm}/validation.jsonl",
                         f"train/{arm}/training_manifest.json", f"train/{arm}/validation_manifest.json",
                         f"train/{arm}/freeze.json", f"train/{arm}/fit.json", f"train/{arm}/artifact/model.json"):
                for key in held:
                    self.assertNotIn(key, (out / name).read_text(), name)
        comparison = json.loads((out / "comparison.json").read_text())
        self.assertEqual((comparison["parity"]["largest_logit_difference"], comparison["parity"]["largest_map_difference"],
                          comparison["live"]["scored"]), (0.0, 0.0, 4))

    def test_the_evaluation_and_the_reference_record_are_written_and_never_graded(self):
        out = self.reference["out"]
        made = json.loads((out / "cusense_eval.json").read_text())
        self.assertEqual((made["schema"], made["graded"], made["role"], sorted(made["arms"])),
                         ("cusense-eval/v1", False, "held_out", ["o2", "o3"]))
        for arm, entry in (("o3", "o3_fp16"), ("o2", "o2_comb")):
            evaluation = made["arms"][arm]
            by_window, by_episode = evaluation["presence"]["windows"], evaluation["presence"]["episodes"]
            # One target-present and one target-absent held-out episode, two windows each: every one in a denominator.
            self.assertEqual((evaluation["input"], by_episode["target_present"]["total"], by_episode["target_absent"]["total"],
                              by_window["target_present"]["total"], by_window["target_absent"]["total"],
                              by_window["target_present"]["missing"], by_episode["target_absent"]["missing"]),
                             (entry, 1, 1, 2, 2, 0, 0))
            position = evaluation["position"]
            self.assertEqual((position["unit"], position["raw"]["eligible"], position["smoothed"]["eligible"],
                              position["kalman"]["eligible"]), ("m", 2, 2, 2))
            for name in ("mean_m", "median_m", "p90_m"):
                self.assertGreater(position["raw"][name], 0.0)
            rows = [row for row in evaluation["rows"] if row["label"] == 1]
            self.assertEqual(position["raw"]["mean_m"], sum(row["error_m"] for row in rows) / len(rows))
            self.assertEqual([row["selected"] for row in sorted(rows, key=lambda row: row["period"])], [False, True])
        linear = made["linear_head"]
        self.assertEqual((linear["head_id"], linear["counts"]["target_present"]["episodes"],
                          linear["counts"]["target_absent"]["episodes"]), ("dd-linear-v1", 1, 1))
        self.assertEqual(sorted(made["side_by_side"]), ["linear_head", "o2", "o3"])
        self.assertEqual((made["latency"]["runtime"], made["latency"]["graded"], made["latency"]["inference_us"]["count"]),
                         ("reference", False, 4))
        reference = json.loads((out / "cusense_reference.json").read_text())
        self.assertEqual((reference["schema"], reference["graded"], reference["published"]["position_error_cm"]["mean"],
                          reference["published"]["live_inference_us"]["value"], reference["measured"]["input"]),
                         ("cusense-reference/v1", False, 73.3, 241, "o3_fp16"))
        self.assertEqual((reference["measured"]["context"]["bandwidth_mhz"], reference["measured"]["context"]["prbs"],
                          reference["measured"]["context"]["rx"], reference["measured"]["context"]["dmrs_symbols"],
                          reference["measured"]["latency"]["runtime"]), (20, 51, 4, 2, "reference"))
        self.assertAlmostEqual(reference["measured"]["position_error"]["raw"]["mean_cm"],
                               100.0 * made["arms"]["o3"]["position"]["raw"]["mean_m"])
        # No check of the grade reads a held-out count or a position error: both are in notes only.
        graded = " ".join(item["check"] for item in self.reference["grade"]["results"])
        for word in ("position error", "detected", "false alarm"):
            self.assertNotIn(word, graded)

    def test_a_second_reader_recovers_the_frozen_model_and_a_decision(self):
        out = self.reference["out"]
        recover = json.loads((out / "recover.json").read_text())
        self.assertEqual(recover, {"artifact": "train/o3/artifact", "freeze": "train/o3/freeze.json",
                                   "sensor": "live/held-out-00/sensor", "window": 0, "offline": "parity/held-out-00",
                                   "second_input": "train/o2/artifact", "evaluation": "cusense_eval.json",
                                   "reference": "cusense_reference.json"})
        freeze = json.loads((out / recover["freeze"]).read_text())
        model = cusense.inspect(cusense.open_artifact(out / recover["artifact"], freeze["artifact"]["sha256"]))
        shown = cusense.summary(cusense.open_output(out / recover["sensor"]), recover["window"])
        self.assertEqual((shown["artifact_sha256"], shown["loads"], shown["source"], shown["scored_windows"], shown["input"],
                          shown["grid"]), (freeze["artifact"]["sha256"], 1, "ring", 4, "o3_fp16", [50, 34]))
        self.assertEqual(model["weights_sha256"],
                         hashlib.sha256((out / recover["artifact"] / "weights.f32").read_bytes()).hexdigest())
        self.assertEqual((model["input_shape"], model["entries"]["model"]), ([1, 4, 612], "cusense_cnn"))
        offline = cusense.summary(cusense.open_output(out / recover["offline"]), recover["window"])
        for name in ("id", "logit", "cell", "input_sha256", "map_sha256"):
            self.assertEqual(offline["window"][name], shown["window"][name], name)
        self.assertAlmostEqual(shown["window"]["map_sum"], 1.0, places=5)
        other = cusense.inspect(cusense.open_artifact(out / recover["second_input"]))
        self.assertEqual((other["input_shape"], other["entries"]["input"]), ([1, 4, 306], "o2_comb"))
        # What the sensor wrote names nothing of the truth.
        for path in (out / recover["sensor"]).rglob("*"):
            if path.suffix in (".json", ".jsonl"):
                for word in ("bistatic", "target_present", "realization", "unit-held-out-line", '"label"'):
                    self.assertNotIn(word, path.read_text(), path.name)

    # ---------------------------------------------------------------- faults of the receiver and of the sensor
    def test_each_fault_of_the_training_runs_fails_the_check_that_names_it(self):
        for fault, expected in (("capture-drop", ["records account"]), ("capture-crc", ["records decodes"]),
                                ("capture-fallback", ["records executed path", "records account"]),
                                ("capture-exit", ["records native runs"])):
            run = self.faulty(fault)
            with self.subTest(fault=fault):
                self.assertEqual((run["status"], run["grade"]["judgment"]), (1, "fail"))
                self.assertEqual(run["failed"], expected + blocked_after("records"))
                # Without the training records nothing is fitted and nothing of the held-out group is read.
                self.assertFalse((run["out"] / "train").exists())
                self.assertFalse((run["out"] / "live").exists())
        self.assertIn("stage O3: 36 eligible, 24 published, 12 dropped",
                      (self.out / "capture-drop" / "grade.txt").read_text())

    def test_each_fault_of_the_held_out_run_fails_the_check_that_names_it(self):
        cases = (("live-crc", ["live decodes"], "12 of 24 decodes gave the original bytes"),
                 ("live-fallback", ["live executed path", "live sensor"], "0 of 12 decodes requested from the CUDA processor"),
                 ("live-exit", ["live native run"], "the native tool exited with status 3"),
                 ("sensor-exit", ["live sensor"], "the sensor exited with status 3"),
                 ("no-output", ["live sensor"], "the sensor exited with status 1"),
                 ("wrong-artifact", ["live sensor"], "which is not the frozen one"),
                 ("two-loads", ["live sensor"], "the artifact was loaded 4 times in the process of the sensor"),
                 ("cold-sensor", ["live sensor"], "the sensor was not warm before the first record was published"),
                 ("slow-sensor", ["live account", "live decisions"], "stage O3: 12 eligible, 8 published, 4 dropped"))
        for fault, expected, said in cases:
            run = self.faulty(fault)
            with self.subTest(fault=fault):
                self.assertEqual((run["status"], run["grade"]["judgment"]), (1, "fail"))
                self.assertEqual(run["failed"], expected + blocked_after("live"))
                self.assertEqual(run["grade"]["cases_passed"], 7)
                self.assertIn(said, "".join(line for line in run["lines"].splitlines() if line.startswith("FAIL")))
                # The truth of the held-out group is not read when its decisions do not stand.
                self.assertFalse((run["out"] / "cusense_eval.json").exists())

    def test_scores_and_windows_that_differ_offline_are_found(self):
        run = self.faulty("device-drift")
        self.assertEqual((run["status"], run["failed"]), (1, ["parity scores"]))
        self.assertIn("the logit differs by 5.000e-03", run["lines"])
        # A window on a slot the ring did not carry, and one with a slot of a later period in it: the window
        # tensors and the digest of the input differ offline.
        for fault in ("wrong-sample", "future-leak"):
            run = self.faulty(fault)
            with self.subTest(fault=fault):
                self.assertEqual((run["status"], run["failed"]), (1, ["parity same windows", "parity scores"]))
                self.assertIn("the digest of the input differs: the two sides did not score the same window", run["lines"])

    # ---------------------------------------------------------------- faults of what was frozen
    def test_statistics_or_a_threshold_that_are_not_those_of_their_split_are_found(self):
        def other_mean(parts):
            parts["normalizer"]["mean"] += 0.5

        run = self.run_once(self.out / "mean", patch=self.rewritten(other_mean))
        self.assertEqual(run["status"], 1)
        self.assertEqual(run["failed"][0], "train statistics")
        self.assertNotIn("train artifact", run["failed"])  # the artifact is whole: only its content is wrong

        def other_template(parts):
            values = parts["template"]["values"].copy()
            values[0, 0, 0] += 0.25
            parts["template"]["values"] = values

        run = self.run_once(self.out / "template", patch=self.rewritten(other_template))
        self.assertEqual(run["failed"][0], "train statistics")

        def other_threshold(parts):
            parts["threshold"]["logit"] -= 1.0
            parts["threshold"]["probability"] = 0.5

        run = self.run_once(self.out / "threshold", patch=self.rewritten(other_threshold))
        self.assertEqual((run["status"], run["failed"]),
                         (1, ["train threshold"] + blocked_after("train", done=("second-input",))))
        # The second input is graded on its own: its statistics are found in its own case.
        run = self.run_once(self.out / "second", patch=self.rewritten(other_mean, arm="o2"))
        self.assertEqual(run["failed"][0], "second-input statistics")
        self.assertFalse(any(name.startswith("train ") for name in run["failed"]))

    def test_a_freeze_record_that_does_not_bind_what_was_read_is_found(self):
        def another_artifact(run):
            data = bytearray((run / "artifact/weights.f32").read_bytes())
            data[0] ^= 1
            (run / "artifact/weights.f32").write_bytes(bytes(data))

        run = self.run_once(self.out / "weights", patch=self.after_training(another_artifact))
        self.assertEqual((run["status"], run["failed"]),
                         (1, ["train script"] + blocked_after("train", done=("second-input",))))
        self.assertIn("the checkpoint was altered", run["lines"])

        def stale_manifest(run):
            manifest = json.loads((run / "validation_manifest.json").read_text())
            manifest["windows"] += 1
            (run / "validation_manifest.json").write_text(json.dumps(manifest))

        run = self.run_once(self.out / "stale", patch=self.after_training(stale_manifest, arm="o2"))
        self.assertEqual((run["status"], run["failed"]), (1, ["freeze hash chain"] + blocked_after("freeze")))

        def leaked(run):
            table = json.loads((run.parent.parent / "split.json").read_text())
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

    def test_a_stream_that_reads_another_field_of_a_record_is_found(self):
        def leaky():
            real = cusense.planes

            def planes(record, spec):
                # The reserved field of a header is 0 in every record of a ring. A stream must not read it at all.
                values, valid = real(record, spec)
                return (values * 1.5 if record.header["reserved"] else values), valid
            cusense.planes = planes
            return lambda: setattr(cusense, "planes", real)

        run = self.run_once(self.out / "leaky", patch=leaky)
        self.assertEqual((run["status"], run["failed"]), (1, ["causality record fields"]))

    def test_a_model_that_did_not_learn_presence_is_not_frozen(self):
        # The floor on the validation windows is declared in the configuration, before the run.
        def no_presence():
            # A fit whose presence logit receives no gradient: it stays at zero for every window, the map is
            # still learned, and the loss still falls.
            model = cusense.cusense_cnn
            real = model.joint_loss

            def joint_loss(grid_logits, presence_logits, targets, labels, grid_weight=1.0):
                _, presence_part, location_part = real(grid_logits, presence_logits, targets, labels, grid_weight)
                return grid_weight * location_part + presence_part.detach(), presence_part, location_part
            model.joint_loss = joint_loss
            return lambda: setattr(model, "joint_loss", real)

        config = self.changed_config(lambda settings: settings["expect"].update(validation_auc_min=0.75), "floor.json")
        run = self.run_once(self.out / "unlearned", config=config, patch=no_presence)
        self.assertEqual((run["status"], run["failed"]),
                         (1, ["train fit", "second-input fit"] + blocked_after("second-input")))
        self.assertIn("on the validation windows the area under the curve is 0.500, and the floor that the configuration "
                      "declares for a model that learned presence is 0.75", run["lines"])
        self.assertFalse((run["out"] / "live").exists())
        # The floor of the demonstration is the one the shipped configuration declares.
        shipped = json.loads((ROOT / "configs/integration/cusense_frontend.json").read_text())
        self.assertEqual(shipped["expect"]["validation_auc_min"], 0.75)

    # ---------------------------------------------------------------- the configuration and the control grade
    def test_the_expectations_of_the_configuration_are_graded(self):
        cases = ((lambda settings: settings["set"]["expect"]["episodes"].update(training=300),
                  ["split episode set"] + blocked_after("split")),
                 (lambda settings: settings["set"]["expect"].update(set_id="demo-set-v1"),
                  ["split episode set"] + blocked_after("split")),
                 (lambda settings: settings["set"]["expect"].update(windows_per_realization=8),
                  ["split replayable"] + blocked_after("split")),
                 (lambda settings: settings["set"]["expect"].update(grid=[10, 10]),
                  ["split replayable"] + blocked_after("split")),
                 (lambda settings: settings["training"].update(min_per_class=5),
                  ["train script", "second-input script"] + blocked_after("second-input")),
                 (lambda settings: settings["training"]["threshold"].update(min_negatives=3),
                  ["train script", "second-input script"] + blocked_after("second-input")))
        for number, (edit, expected) in enumerate(cases):
            run = self.run_once(self.out / f"expect-{number}", config=self.changed_config(edit, f"config-{number}.json"))
            with self.subTest(expected=expected[0]):
                self.assertEqual((run["status"], run["failed"]), (1, expected))
        self.assertIn("the validation split holds 2 target-absent windows, and the calibration needs 3",
                      (self.out / "expect-5" / "grade.txt").read_text())

    def test_a_set_that_is_not_the_one_of_the_control_is_refused_before_a_grid_is_filled(self):
        other = self.out / "other"
        control_tests.double.analytic_set(control_tests.double.set_config(
            self.out / "other.json", self.stage.grant, control_tests.double.trajectories(2, 1, 1), seed=7), other)
        run = self.run_once(self.out / "run-other", episode_set=other)
        self.assertEqual((run["status"], run["failed"]), (1, ["split episode set"] + blocked_after("split")))
        self.assertIn("the linear head cannot be compared on another set", run["lines"])
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

    def test_the_control_must_be_the_accepted_sprint_3_4_run_and_the_build_with_the_route(self):
        def changed_control(edit):
            copy = self.out / f"control-{len(list(self.out.glob('control-*')))}"
            shutil.copytree(self.control, copy)
            edit(copy)
            return copy

        def failed_grade(copy):
            grade = json.loads((copy / "grade.json").read_text())
            grade.update(judgment="fail", checks_failed=1)
            (copy / "grade.json").write_text(json.dumps(grade))

        def rehearsed(copy):
            grade = json.loads((copy / "grade.json").read_text())
            grade.update(judgment="rehearsal", hardware_gate="open: a host rehearsal")
            (copy / "grade.json").write_text(json.dumps(grade))

        def other_head(copy):
            data = bytearray((copy / "train/artifact/weights.f32").read_bytes())
            data[0] ^= 1
            (copy / "train/artifact/weights.f32").write_bytes(bytes(data))

        run = self.run_once(self.out / "failed", control=changed_control(failed_grade))
        self.assertEqual((run["status"], run["failed"]), (1, ["control accepted run"] + blocked_after("control")))
        run = self.run_once(self.out / "rehearsed", control=changed_control(rehearsed))
        self.assertEqual(run["failed"], ["control accepted run"] + blocked_after("control"))
        self.assertIn("this gate needs the judgment pass with the hardware gate closed", run["lines"])
        run = self.run_once(self.out / "head", control=changed_control(other_head))
        self.assertEqual(run["failed"], ["control linear head"] + blocked_after("control"))
        run = self.run_once(self.out / "no-plan",
                            control=changed_control(lambda copy: (copy / "records/training-00/plan.json").unlink()))
        self.assertEqual(run["failed"], ["control transmit grids"] + blocked_after("control"))
        run = self.run_once(self.out / "empty", control=changed_control(lambda copy: (copy / "grade.json").unlink()))
        self.assertEqual(run["failed"], ["control accepted run"] + blocked_after("control"))
        # The adapter must be the one linked against the patched receiver, with the series of this tree.
        plain = self.receipt_of()
        del plain["o3_route"], plain["patch"]
        run = self.run_once(self.out / "no-route", receipt=plain)
        self.assertEqual(run["failed"], ["control estimate route"] + blocked_after("control"))
        self.assertIn("Build it: make adapter-build-o3", run["lines"])
        stale = self.receipt_of()
        stale["patch"]["series_sha256"] = "0" * 64
        run = self.run_once(self.out / "stale-series", receipt=stale)
        self.assertEqual(run["failed"], ["control estimate route"] + blocked_after("control"))
        self.assertFalse((run["out"] / "records").exists())

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

        base = ["--s34-run", self.control, "--episode-set", self.set]
        refused(["--config", self.config, *base, "--output-dir", self.reference["out"]], "Evidence is never overwritten")
        for number, (edit, message) in enumerate((
                (lambda settings: settings["rings"]["o3"].update(slots=4), "rings.o3.slots is 4"),
                (lambda settings: settings["rings"]["o3"].update(stream="o2"), "rings.o3 requires its stream o3"),
                (lambda settings: settings.update(selector="presence_head"), "for the selector cusense_frontend"),
                (lambda settings: settings["sensor"].update(device="tpu"), 'the device "cuda" or "cpu"'),
                (lambda settings: settings["training"]["optimizer"].update(entry="sgd"), 'optimizer.entry is "adam"'),
                (lambda settings: settings["set"].update(entries_per_run=0), "set requires entries_per_run"),
                (lambda settings: settings["control"].update(driver="pilot_features"), "the driver presence_head"),
                (lambda settings: settings["tolerance"].update(logit_absolute=0), "each a positive number"),
                (lambda settings: settings["receiver"].update(keep_grids="no"), "keep_grids, true or false"),
                (lambda settings: settings["arms"]["o2"]["spec"]["window"].update(slots_in_period=[8]),
                 "the two arms must cut the same window"),
                (lambda settings: settings["arms"]["o2"].update(spec=settings["arms"]["o3"]["spec"]),
                 "arms.o2 reads the input entry o2_comb"),
                (lambda settings: settings["arms"]["o2"].update(model_id="cusense-o3-v1"), "different model identifiers"),
                (lambda settings: settings["selection"].update(window="best"), "selection requires window: last"),
                (lambda settings: settings["expect"].update(validation_auc_min=1.5), "validation_auc_min from 0 to 1"),
                (lambda settings: settings["estimate"].update(landing="nowhere"), "the estimate"))):
            config = self.changed_config(edit, f"bad-{number}.json")
            refused(["--config", config, *base, "--output-dir", self.out / f"bad-{number}"], message)
            self.assertFalse((self.out / f"bad-{number}").exists())
        refused(["--config", self.config, "--s34-run", self.control, "--episode-set", self.out / "nowhere", "--output-dir",
                 self.out / "no-set"], "Trace one: make episode-set")
        receipt = self.receipt_of()
        del receipt["executables"]["obs_tail"]
        refused(["--config", self.config, *base, "--output-dir", self.out / "no-tool"], "holds no obs_tail", receipt)
        # There is no host rehearsal: the receiver's estimate exists only on a CUDA device.
        with contextlib.redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit):
            driver.main(["--config", str(self.config), "--s34-run", str(self.control), "--episode-set", str(self.set),
                         "--output-dir", str(self.out / "rehearsal"), "--host-rehearsal"])
        self.assertIn("unrecognized arguments: --host-rehearsal", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
