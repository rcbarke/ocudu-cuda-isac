#!/usr/bin/env python3
"""Sprint 3.5 integration driver (selector cusense_frontend). Hardware command, not part of unit discovery.

It trains the cuSense-like branch on receiver-derived records, freezes it and scores held-out
windows with it in a second process on the selected device. The input is the receiver's own
estimate, the stage O3 of Sprint 5.2, and the pilot comb of the stage O2 is the second input
entry: every replay publishes both stages, each on a ring of its own. The stimulus is the
episode set that the accepted Sprint 3.4 run was graded on, split by trajectory before an episode
was traced. The control is that run: its grade, its frozen linear head, and the plan whose
channel plan binds the transmit grid of every grant. Twelve cases, each graded on its own:

  control       the accepted Sprint 3.4 run, its frozen linear head, the stores of the grids its
                plans bind, and the adapter build with the route to the receiver's estimate
  split         the episode set is that of the control and verifies, every trajectory is in one
                group, and every entry gives the windows of one TDD period that the gate expects
  records       the tapped receiver replay of the training and of the validation entries in the
                mode timed, with a native consumer on each ring that keeps every record
  windows       stage 2 up to the window on those records, for both input entries: one complete
                window per period, checked against an element-by-element oracle and joined to
                the split and to the position of the target
  train         scripts/train_cusense.py on the O3 sample sets: the template and the statistics
                of the z-score from the training windows, the weights by the joint loss, the
                threshold from the target-absent validation windows, one artifact of four files
  second-input  the same script and settings on the O2 sample sets: the same pipeline on the comb
  freeze        the freeze records name the hashes of both artifacts and of every manifest, no
                manifest names a held-out episode, and nothing of the held-out group was replayed
  live          the held-out entries are replayed and the sensor follows the O3 ring in a second
                process: the frozen artifact, loaded once, on the selected device, one decision
                per answer of the stream, while a native consumer keeps the O2 ring
  parity        the same records scored again offline on the CPU: the same windows, the same
                input digests, the logit, the probability and the map within the tolerance
  evaluation    the O2 arm and the linear head of Sprint 3.4 on the records of the same held-out
                run, then the truth: cusense_eval.json and cusense_reference.json
  causality     a window that is filling has no score, records of the future and fields outside
                the input leave a decision unchanged, and no output names a label
  refusals      wrong shape, wrong ordering, altered artifact, another specification, statistics
                of another split and a scene in two splits, by the interface and the command line

    python tests/integration/cusense_frontend.py --config configs/integration/cusense_frontend.json \\
        --s34-run results/runs/s34-spark-001 --episode-set data/episodes/demo-set-001 \\
        --output-dir results/runs/s35-spark-001

The output directory must be new. The exit status is 0 only when every check of every case passed.
There is no host rehearsal: the stage O3 exists only where the CUDA processor ran its resident
path. Every number of the branch carries runtime "reference": NumPy and PyTorch in a Python
sensor. The position error and the presence counts on the held-out episodes are recorded beside
the published figures and are not graded, and the latency comparison with the published 241 us
belongs to Sprint 3.6.
"""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from ocudu_cuda_isac.episodes import episode_set as sets  # noqa: E402
from ocudu_cuda_isac.episodes import grant as grants  # noqa: E402
from ocudu_cuda_isac.episodes import grid_apply as replay  # noqa: E402
from ocudu_cuda_isac.episodes import native_estimate as ne  # noqa: E402
from ocudu_cuda_isac.episodes import replay_pusch as rp  # noqa: E402
from ocudu_cuda_isac.episodes import snapshot_ring as sr  # noqa: E402
from ocudu_cuda_isac.episodes.episode import EpisodeError  # noqa: E402
from ocudu_cuda_isac.episodes.grant import GrantError  # noqa: E402
from ocudu_cuda_isac.episodes.grid_apply import GridApplyError  # noqa: E402
from ocudu_cuda_isac.episodes.native_estimate import NativeEstimatePlanError  # noqa: E402
from ocudu_cuda_isac.episodes.replay_pusch import ReplayPuschError  # noqa: E402
from ocudu_cuda_isac.episodes.snapshot_ring import SnapshotRingError  # noqa: E402
from ocudu_cuda_isac.replay import ContractError, load_json  # noqa: E402
from ocudu_cuda_isac.sensing import obs  # noqa: E402
from ocudu_cuda_isac.sensing.obs import ObsError  # noqa: E402
import native_adapter  # noqa: E402
import receiver_patch  # noqa: E402

CONFIG_SCHEMA = "integration-config/cusense_frontend/v1"
GRADE_SCHEMA = "integration-grade/cusense_frontend/v1"
CONTROL_GRADE_SCHEMA = "integration-grade/presence_head/v1"
CASES = ("control", "split", "records", "windows", "train", "second-input", "freeze", "live", "parity", "evaluation",
         "causality", "refusals")
TOOLS = ("pilot_ls", "obs_tail")
ROLES = ("training", "validation")
ARMS = ("o3", "o2")          # an arm reads the stream of its name
TRAIN_CASE = {"o3": "train", "o2": "second-input"}
NATIVE_TIMEOUT_S = 3600
CONSUMER_GRACE_S = 180
TRAIN_TIMEOUT_S = 6 * 3600
SENTINEL = 6.0e4             # a large value that FP16 holds
ORACLE_WINDOWS = 2           # per run and arm: the first and the last complete window
EVALUATOR_KEYS = ('"label"', '"scene"', '"truth"', '"target_present"', '"case"', '"pair"', "held_out")
EVAL_FILE, REFERENCE_FILE = "cusense_eval.json", "cusense_reference.json"


class ConfigError(ValueError):
    """The configuration or the run setup is unusable. Nothing was graded."""


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ring_driver = load_script("snapshot_ring_driver", ROOT / "tests/integration/snapshot_ring.py")
digest, write_json, Grader = ring_driver.digest, ring_driver.write_json, ring_driver.Grader
last_line, git_commit = ring_driver.last_line, ring_driver.git_commit
Run, run_is_clean = ring_driver.Run, ring_driver.run_is_clean


class Grade(Grader):
    """The Sprint 2.1 grader with this sprint's label."""

    def check(self, case, name, passed, detail):
        self.results.append({"case": case, "check": name, "passed": bool(passed), "detail": detail})
        self._line(("PASS" if passed else "FAIL") + f"  S3.5 {case} {name}: {detail}")
        return bool(passed)

    def note(self, case, text):
        self._line(f"NOTE  S3.5 {case} {text}")


def load_config(path, cusense, evaluator):
    try:
        config = load_json(Path(path))
    except (ContractError, OSError) as error:
        raise ConfigError(f"cannot read the configuration: {error}") from None
    needed = {"schema", "selector", "adapter_build", "backends", "control", "set", "receiver", "rings", "estimate", "arms",
              "training", "sensor", "selection", "tolerance", "expect"}
    if not isinstance(config, dict) or set(config) != needed or config["schema"] != CONFIG_SCHEMA \
            or config["selector"] != "cusense_frontend":
        raise ConfigError(f"the configuration must be a {CONFIG_SCHEMA} object for the selector cusense_frontend")
    if list(config["backends"]) != list(rp.BACKENDS):
        raise ConfigError('backends must be ["cpu", "cuda"]: every grid is decoded with both')
    if config["control"] != {"driver": "presence_head"}:
        raise ConfigError("control requires the driver presence_head: the accepted Sprint 3.4 run")
    chosen = config["set"]
    if not isinstance(chosen, dict) or set(chosen) != {"entries_per_run", "expect"} \
            or type(chosen["entries_per_run"]) is not int or chosen["entries_per_run"] < 1 \
            or not isinstance(chosen["expect"], dict) \
            or set(chosen["expect"]) != {"set_id", "episodes", "windows_per_realization", "grid"} \
            or not isinstance(chosen["expect"]["episodes"], dict) or set(chosen["expect"]["episodes"]) != set(sets.GROUPS) \
            or type(chosen["expect"]["windows_per_realization"]) is not int or chosen["expect"]["windows_per_realization"] < 1:
        raise ConfigError("set requires entries_per_run and expect with set_id, the episodes of every group, "
                          "windows_per_realization and grid")
    receiver = config["receiver"]
    if not isinstance(receiver, dict) or set(receiver) != {"keep_grids"} or type(receiver["keep_grids"]) is not bool:
        raise ConfigError("receiver requires keep_grids, true or false")
    rings = config["rings"]
    if not isinstance(rings, dict) or set(rings) != {"o2", "o3"}:
        raise ConfigError("rings requires o2 and o3")
    try:
        for name, ring in rings.items():
            if set(ring) != {"stream", "window_records", "burst_records", "slots", "wait", "attach_timeout_ms", "drain_ms"} \
                    or ring["wait"] not in sr.WAITS or ring["stream"] != name:
                raise ConfigError(f"rings.{name} requires its stream {name}, window_records, burst_records, slots, wait, "
                                  "attach_timeout_ms and drain_ms")
            if ring["slots"] != obs.slots_for(ring["window_records"], ring["burst_records"]):
                raise ConfigError(f"rings.{name}.slots is {ring['slots']}, and the window and the burst of the "
                                  f"configuration need {obs.slots_for(ring['window_records'], ring['burst_records'])}")
        estimate = config["estimate"]
        if not isinstance(estimate, dict) or set(estimate) != {"copy_stream", "landing", "timeout_us"}:
            raise ConfigError("estimate requires copy_stream, landing and timeout_us")
        if estimate["copy_stream"] not in ne.COPY_STREAMS or estimate["landing"] not in ne.LANDINGS \
                or type(estimate["timeout_us"]) is not int or estimate["timeout_us"] < 1:
            raise ConfigError("the estimate names a copy stream of " + ", ".join(ne.COPY_STREAMS) + ", a landing of "
                              + ", ".join(ne.LANDINGS) + " and a positive timeout_us")
        arms = config["arms"]
        if not isinstance(arms, dict) or set(arms) != set(ARMS):
            raise ConfigError("arms requires o3 and o2")
        entries = {"o3": "o3_fp16", "o2": "o2_comb"}
        for name, arm in arms.items():
            if not isinstance(arm, dict) or set(arm) != {"spec", "model_id"} or not isinstance(arm["model_id"], str) \
                    or not arm["model_id"]:
                raise ConfigError(f"arms.{name} requires spec and model_id")
            if cusense.check_spec(arm["spec"])["input"]["entry"] != entries[name]:
                raise ConfigError(f"arms.{name} reads the input entry {entries[name]}")
        if arms["o3"]["model_id"] == arms["o2"]["model_id"]:
            raise ConfigError("the two arms need different model identifiers")
        if arms["o3"]["spec"]["window"] != arms["o2"]["spec"]["window"] \
                or arms["o3"]["spec"]["radio"] != arms["o2"]["spec"]["radio"]:
            raise ConfigError("the two arms must cut the same window: the same slots of the same period")
        cusense.check_settings(config["training"])
    except (KeyError, TypeError, ObsError, NativeEstimatePlanError, SnapshotRingError, cusense.CuSenseError) as error:
        raise ConfigError(f"the rings, the estimate, the arms or the training configuration are unusable: {error}") from None
    sensor = config["sensor"]
    if not isinstance(sensor, dict) or set(sensor) != {"device", "ready_timeout_ms"} \
            or sensor["device"] not in ("cuda", "cpu") or type(sensor["ready_timeout_ms"]) is not int \
            or sensor["ready_timeout_ms"] < 1000:
        raise ConfigError('sensor requires the device "cuda" or "cpu" and ready_timeout_ms of at least 1000')
    selection = config["selection"]
    if not isinstance(selection, dict) or set(selection) != {"window"} or selection["window"] not in evaluator.SELECTIONS:
        raise ConfigError("selection requires window: " + ", ".join(evaluator.SELECTIONS))
    tolerance = config["tolerance"]
    if not isinstance(tolerance, dict) or set(tolerance) != set(evaluator.TOLERANCE) \
            or any(type(value) is not float or not value > 0 for value in tolerance.values()):
        raise ConfigError("tolerance requires " + ", ".join(sorted(evaluator.TOLERANCE)) + ", each a positive number")
    expect = config["expect"]
    if not isinstance(expect, dict) or set(expect) != {"validation_auc_min", "published"} \
            or type(expect["validation_auc_min"]) is not float or not 0 <= expect["validation_auc_min"] <= 1 \
            or not isinstance(expect["published"], dict) \
            or set(expect["published"]) != {"mean_position_error_cm", "live_inference_us"}:
        raise ConfigError("expect requires validation_auc_min from 0 to 1 and published with mean_position_error_cm "
                          "and live_inference_us")
    return config


# ---------------------------------------------------------------- processes
def _tail_command(tools, socket, archive, wait, capacity):
    return [str(tools["obs_tail"]), "--socket", str(socket), "--output-dir", str(archive), "--wait", wait,
            "--connect-ms", "60000", "--idle-ms", "60000", "--capacity", str(capacity)]


def _wait(process, limit, stream=None):
    try:
        return process.wait(timeout=limit)
    except subprocess.TimeoutExpired:
        process.kill()
        if stream is not None:
            stream.write("TimeoutExpired: the process was stopped\n")
        return -1


def _report(native):
    try:
        return json.loads((native / "native.json").read_text())
    except (OSError, ValueError):
        return None


def _tool_command(tools, plan_path, native, keep_grids):
    return [str(tools["pilot_ls"]), "--plan", str(plan_path), "--output-dir", str(native)] \
        + ([] if keep_grids else ["--no-grids"])


def run_capture(tools, plan_path, native, directory, archives, wait, capacity, keep_grids=True):
    """Run pilot_ls on one plan with a native consumer on each ring. Returns (status, report, consumer statuses).

    archives maps a stream, "o2" or "o3", to the directory of its consumer. Each consumer is
    started before the tool and retries until the tool has opened the channel of its stream.
    native and the archives are created empty.
    """
    native.mkdir(parents=True)
    plan = json.loads(Path(plan_path).read_text())
    sockets = {"o2": sr.socket_path(native, plan), "o3": ne.socket_path(native, plan)}
    started, logs = {}, []
    try:
        for stream, archive in archives.items():
            archive.mkdir(parents=True)
            command = _tail_command(tools, sockets[stream], archive, wait, capacity)
            log = (directory / f"consumer-{stream}.log").open("w")
            logs.append(log)
            log.write(shlex.join(command) + "\n")
            log.flush()
            started[stream] = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        command = _tool_command(tools, plan_path, native, keep_grids)
        with (directory / "native.log").open("w") as stream:
            stream.write(shlex.join(command) + "\n")
            stream.flush()
            status = _wait(subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT), NATIVE_TIMEOUT_S, stream)
        statuses = {name: _wait(process, CONSUMER_GRACE_S) for name, process in started.items()}
    except OSError as error:
        (directory / "native.log").open("a").write(f"{type(error).__name__}: {error}\n")
        for process in started.values():
            process.kill()
        return -1, None, dict.fromkeys(archives, -1)
    finally:
        for log in logs:
            log.close()
    return status, _report(native), statuses


def run_live(tools, plan_path, native, directory, sensor_dir, artifact_dir, device, wait, ready_timeout_s, archive, capacity,
             keep_grids=True):
    """Run pilot_ls on one plan with the cuSense-like sensor on the O3 ring. Returns (status, report, statuses).

    The sensor is started first. It loads the artifact, creates the context of its device, runs
    its steps once and then writes sensor-ready.json. Only then are the consumer that keeps the
    O2 ring and the tool started, so that the receiver never waits for a sensor that is loading.
    native, sensor_dir and archive are created empty.
    """
    native.mkdir(parents=True)
    sensor_dir.mkdir(parents=True)
    archive.mkdir(parents=True)
    plan = json.loads(Path(plan_path).read_text())
    ready = directory / "sensor-ready.json"
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    sensor_command = [sys.executable, "-m", "ocudu_cuda_isac.sensing.cusense", "follow", "--socket",
                      str(ne.socket_path(native, plan)), "--artifact", str(artifact_dir), "--output-dir", str(sensor_dir),
                      "--wait", wait, "--connect-ms", "60000", "--idle-ms", "60000", "--ready-file", str(ready)] \
        + ([] if device is None else ["--device", device])
    tail_command = _tail_command(tools, sr.socket_path(native, plan), archive, wait, capacity)
    command = _tool_command(tools, plan_path, native, keep_grids)
    sensor = consumer = None
    try:
        with (directory / "sensor.log").open("w") as sensor_log, (directory / "consumer-o2.log").open("w") as tail_log, \
                (directory / "native.log").open("w") as stream:
            sensor_log.write(shlex.join(sensor_command) + "\n")
            sensor_log.flush()
            sensor = subprocess.Popen(sensor_command, stdout=sensor_log, stderr=subprocess.STDOUT, env=environment)
            deadline = time.monotonic() + ready_timeout_s
            while not ready.is_file() and sensor.poll() is None and time.monotonic() < deadline:
                time.sleep(0.05)
            stream.write(shlex.join(command) + "\n")
            if not ready.is_file():
                stream.write("the sensor did not become ready: the tool was not started\n")
                if sensor.poll() is None:
                    sensor.kill()
                return -1, None, {"o2": -1, "o3": sensor.wait(timeout=CONSUMER_GRACE_S)}
            stream.flush()
            tail_log.write(shlex.join(tail_command) + "\n")
            tail_log.flush()
            consumer = subprocess.Popen(tail_command, stdout=tail_log, stderr=subprocess.STDOUT)
            status = _wait(subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT), NATIVE_TIMEOUT_S, stream)
            statuses = {"o3": _wait(sensor, CONSUMER_GRACE_S), "o2": _wait(consumer, CONSUMER_GRACE_S)}
    except OSError as error:
        (directory / "native.log").open("a").write(f"{type(error).__name__}: {error}\n")
        for process in (sensor, consumer):
            if process is not None:
                process.kill()
        return -1, None, {"o2": -1, "o3": -1}
    return status, _report(native), statuses


def run_tool(command, log, timeout=1800):
    """Run a Python tool of the project with the source tree on its path. Returns the completed process."""
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    result = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=timeout, check=False)
    Path(log).write_text(shlex.join([str(word) for word in command]) + "\n" + result.stdout + result.stderr)
    return result


# ---------------------------------------------------------------- helpers
class Control:
    """The accepted Sprint 3.4 run: its set, its frozen linear head and the grids its plans bind."""

    def __init__(self):
        self.manifest = self.bindings = self.observer = self.head = self.spec = self.set = self.directory = None


class Capture:
    """One receiver replay of some entries of the set: its plan, its run and what the consumers of its rings kept."""

    def __init__(self, role, name, rows, directory):
        self.role, self.name, self.rows, self.directory = role, name, rows, directory
        self.run = Run(name, directory, None)
        self.channel = self.output = None
        self.archives, self.statuses, self.keys = {}, {}, []
        self.started_ns = self.ended_ns = None

    def opportunities(self):
        return 2 * sum(row["opportunities"] for row in self.rows)


def text(value, digits=1):
    return "none" if value is None else f"{value:.{digits}f}"


def changed_record(record, payload=None, **header):
    """A copy of a record with its payload or some header fields replaced."""
    made = dict(record.header, **header)
    payload = record.payload if payload is None else payload
    made.update(payload_bytes=len(payload))
    return obs.Record(made, record.host, record.mask, payload, depth=record.depth)


def plan_capture(config, control, evaluator, capture):
    """Write the three plans of a capture into its directory."""
    capture.directory.mkdir(parents=True)
    capture.channel, capture.run.plan = evaluator.write_plans(
        f"s35-{capture.name}", capture.rows, control.bindings, capture.directory, config["rings"], config["estimate"],
        epoch=control.observer["epoch"], source=control.observer["source"])
    capture.keys = evaluator.keys_of(capture.channel)


def verify_capture(capture):
    """Open what the tool wrote. Sets the problem of the run when it does not verify."""
    run = capture.run
    if run.status == 0 and run.report:
        try:
            run.output = ne.open_output(run.native)
        except (NativeEstimatePlanError, SnapshotRingError) as error:
            run.problem = f"the native output does not verify: {error}"


def clean(capture, backends):
    return run_is_clean(capture.run, backends) and capture.run.output.mode == "timed"


def decoded(captures):
    """(decodes that gave the original bytes, radio records) over some captures."""
    records = [record for capture in captures for record in capture.run.output.records]
    return sum(record["outcome"] == "decoded" for record in records), len(records)


def path_check(grader, name, captures, check="executed path"):
    """The CUDA processor ran its resident path and the host processor the host path, in every run."""
    records = [record for capture in captures for record in capture.run.output.records]
    cuda = [record for record in records if record["backend"] == "cuda"]
    host = [record for record in records if record["backend"] == "cpu"]
    resident = sum((record["path"] or {}).get("executed") == "resident" for record in cuda)
    on_host = sum((record["path"] or {}).get("executed") == "host" for record in host)
    built = all(capture.run.report.get("built_with_cuda") is True for capture in captures)
    return grader.check(name, check, built and cuda and resident == len(cuda) and on_host == len(host),
                        f"{resident} of {len(cuda)} decodes requested from the CUDA processor ran its resident path, "
                        f"the only place where the receiver's estimate exists, and {on_host} of {len(host)} decodes of "
                        "the host processor ran the host path; the adapter is built with CUDA" if built else
                        "the adapter is not built with CUDA: the CUDA processor cannot execute")


def account_check(grader, name, captures, check="account"):
    """Every observation of both stages was published, read and kept, and each kept record is that of its line."""
    fields = ("eligible", "published", "dropped", "released", "pending")
    total = {stream: dict.fromkeys(fields, 0) for stream in ("o2", "o3")}
    kept, problems, balanced = {"o2": 0, "o3": 0}, [], True
    for capture in captures:
        output = capture.run.output
        states = {"o2": sr.account(output.base), "o3": ne.account(output)}
        joins = {"o2": sr.join_archive(output.base, capture.archives["o2"]),
                 "o3": ne.join_archive(output, capture.archives["o3"])}
        for stream in total:
            balanced = balanced and states[stream]["balanced"]
            for key in fields:
                total[stream][key] += states[stream].get(key, 0)
            pairs, wrong = joins[stream]
            kept[stream] += len(pairs)
            problems += [f"{capture.name} {stream}: {problem}" for problem in wrong]
        if output.mode != "timed":
            problems.append(f"{capture.name}: the run is not in the mode timed")
    wanted = sum(capture.opportunities() for capture in captures)
    whole = balanced and not problems and all(
        total[stream]["dropped"] == 0 and total[stream]["pending"] == 0
        and kept[stream] == total[stream]["published"] == wanted for stream in total)
    grader.check(name, check, whole,
                 f"{len(captures)} runs, {wanted} opportunities: " + "; ".join(
                     f"stage {stream.upper()}: {total[stream]['eligible']} eligible, {total[stream]['published']} "
                     f"published, {total[stream]['dropped']} dropped, {total[stream]['released']} released, "
                     f"{total[stream]['pending']} pending, {kept[stream]} kept" for stream in ("o2", "o3"))
                 + "; each kept record is the record of its account line: the epoch, the sequence, the join keys and "
                   "the ticket of its grid; a consumer that keeps the pace of the receiver loses no observation"
                 if not problems else "; ".join(problems[:3]))
    return whole, total


def open_archives(capture, names):
    """Open the archives of a capture whose consumers ended clean. Returns a problem or None."""
    for stream, directory in names.items():
        try:
            if capture.statuses.get(stream) != 0:
                log = capture.directory / ("sensor.log" if directory.name == "records" else f"consumer-{stream}.log")
                raise ObsError(f"its consumer exited with status {capture.statuses.get(stream)}: {last_line(log)}")
            capture.archives[stream] = obs.open_archive(directory)
        except ObsError as error:
            return f"{capture.name}: the archive of the stage {stream.upper()} does not verify: {error}"
    return None


def slots_in_frame(spec):
    return 10 * spec["radio"]["scs_khz"] // 15


# ---------------------------------------------------------------- control
def run_control(config, grader, s34_out, receipt, features, presence, evaluator):
    """The accepted Sprint 3.4 run, its frozen head, the grids its plans bind and the adapter with the route."""
    name, control, measured = "control", Control(), {}
    try:
        grade = load_json(s34_out / "grade.json")
        manifest = load_json(s34_out / "manifest.json")
    except (ContractError, OSError) as error:
        grader.check(name, "accepted run", False, f"{s34_out} holds no grade and manifest of the Sprint 3.4 driver: {error}")
        return None, measured
    accepted = isinstance(grade, dict) and grade.get("schema") == CONTROL_GRADE_SCHEMA and grade.get("judgment") == "pass" \
        and grade.get("cases_passed") == grade.get("cases") and grade.get("checks_failed") == 0 \
        and grade.get("hardware_gate") == "closed"
    control.manifest, control.directory = manifest, s34_out
    control.set = manifest.get("episode_set") or {}
    measured["s34"] = {"run_id": manifest.get("run_id"), "judgment": grade.get("judgment"),
                       "hardware_gate": grade.get("hardware_gate"), "parent_commit": manifest.get("parent_commit"),
                       "artifact_sha256": grade.get("artifact_sha256"), "spec_sha256": grade.get("spec_sha256"),
                       "episode_set": {key: control.set.get(key) for key in ("set_id", "directory", "config_sha256",
                                                                               "assignment_sha256")}}
    grader.check(name, "accepted run", accepted,
                 f"the Sprint 3.4 driver graded {s34_out.name} as {grade.get('judgment')}: {grade.get('cases_passed')} of "
                 f"{grade.get('cases')} cases, {grade.get('checks_failed')} failed checks, hardware gate "
                 f"{str(grade.get('hardware_gate')).split(':')[0]}, at commit {str(manifest.get('parent_commit'))[:7]}"
                 + ("" if accepted else "; this gate needs the judgment pass with the hardware gate closed"))
    # ---- The linear head of that run: the baseline that is scored on the records of the same held-out run.
    try:
        control.spec = features.check_spec(load_json(s34_out / "spec.json"))
        control.head = presence.open_artifact(s34_out / "train" / "artifact", grade.get("artifact_sha256"))
        head = control.head.manifest
        same = features.spec_sha256(control.spec) == grade.get("spec_sha256") == head["input"]["spec_sha256"] \
            and grade.get("record_definition") == f"{obs.DEFINITION:016x}"
        grader.check(name, "linear head", same,
                     f"the head {head['head_id']} {control.head.sha256[:16]} that run froze verifies with the hash its "
                     f"grade names; it reads the vector of the specification {control.spec['spec_id']} on records of the "
                     f"definition {obs.DEFINITION:016x}, and is scored here on the records of the same held-out run"
                     if same else
                     f"the head of that run names the specification {head['input']['spec_sha256'][:16]} and its grade "
                     f"{str(grade.get('spec_sha256'))[:16]} on the record definition {grade.get('record_definition')}")
    except (features.FeaturesError, presence.PresenceError, ContractError, OSError, KeyError, TypeError) as error:
        grader.check(name, "linear head", False, f"that run holds no frozen head that verifies: {error}")
        same = False
    # ---- The channel plan of a run of that gate binds the transmit grid of every grant.
    try:
        listed = [item for item in manifest.get("runs", []) if isinstance(item, dict) and item.get("status") == 0]
        if not listed:
            raise SnapshotRingError("the Sprint 3.4 run holds no finished run")
        directory = s34_out / listed[0]["directory"]
        if digest(directory / "plan.json") != listed[0].get("plan_sha256"):
            raise SnapshotRingError("the plan of that run does not have the hash its manifest names")
        plan = sr.load_plan(directory / "plan.json")
        replay_path = Path(plan["replay_plan"]["path"])
        if digest(replay_path) != plan["replay_plan"]["sha256"]:
            raise SnapshotRingError("the replay plan of that run does not have the hash its plan binds")
        channel = replay.load_plan(rp.load_plan(replay_path)["channel_plan"]["path"])
        control.bindings = {item["grant_id"]: {"x_sha256": item["x"]["sha256"], "store": str(Path(item["x"]["path"]).parent),
                                               "record_sha256": item["record_sha256"]} for item in channel["grants"]}
        for grant_id, binding in control.bindings.items():
            store = grants.open_store(binding["store"])
            if store.manifest["files"].get("x.npy", {}).get("sha256") != binding["x_sha256"]:
                raise GrantError(f"the store of grant {grant_id} holds another transmit grid than the plan binds")
        control.observer = plan["observer"]
    except (SnapshotRingError, ReplayPuschError, GridApplyError, GrantError, OSError, KeyError, TypeError) as error:
        grader.check(name, "transmit grids", False, f"{error}")
        return None, measured
    measured["grants"] = {grant_id: binding["x_sha256"] for grant_id, binding in control.bindings.items()}
    grader.check(name, "transmit grids", True,
                 f"the channel plan of the run {listed[0]['name']} of that gate binds the transmit grid of "
                 f"{len(control.bindings)} grant(s): {', '.join(sorted(control.bindings))}; every store entry verifies "
                 f"and holds the grid of that hash; observer epoch {control.observer['epoch']}, source "
                 f"{control.observer['source']}")
    # ---- The adapter that this gate runs is the one linked against the patched receiver.
    try:
        current = receiver_patch.identity(receiver_patch.load_series())
        problem = None
    except receiver_patch.PatchError as error:
        current, problem = None, str(error)
    carried, route = receipt.get("patch") or {}, receipt.get("o3_route") or {}
    routed = current is not None and receiver_patch.same_series(carried, current) and route.get("compiled") is True \
        and receipt["pin"]["cuda"] is True
    measured["adapter"] = {"ocudu_build": receipt["ocudu_build"], "route": route,
                           "series_sha256": (current or {}).get("series_sha256")}
    grader.check(name, "estimate route", routed,
                 problem or (f"the adapter is linked against {receipt['ocudu_build']}, the receiver with the series "
                             f"{current['series_sha256'][:16]} of patches/ocudu, built with ENABLE_CUDA; pilot_ls "
                             f"{receipt['executables']['pilot_ls']['sha256'][:12]} reports the tap interface version "
                             f"{route.get('tap_interface_version')}: the build that Sprint 5.2 accepted" if routed else
                             "the receipt of the adapter names no route to the receiver's estimate, or another series "
                             "than patches/ocudu holds. Build it: make adapter-build-o3"))
    return (control if accepted and same and routed else None), measured


# ---------------------------------------------------------------- split
def run_split(config, out, grader, control, set_dir, cusense, evaluator):
    """The episode set, its split by trajectory, and whether its entries give the windows this gate expects."""
    name, expect = "split", config["set"]["expect"]
    spec = config["arms"]["o3"]["spec"]
    try:
        episode_set = sets.open_set(set_dir)
        table = evaluator.split_table(episode_set)
        grid = cusense.grid_record(episode_set.manifest["position_grid"])
    except (EpisodeError, evaluator.PresenceHeadError, GridApplyError, cusense.CuSenseError, KeyError) as error:
        grader.check(name, "episode set", False, f"{set_dir} is no episode set that verifies: {error}")
        return None, {}
    manifest, counts = episode_set.manifest, table["counts"]
    known = control.set
    same = (manifest["config_sha256"], manifest["assignment_sha256"]) == (known.get("config_sha256"),
                                                                         known.get("assignment_sha256"))
    grader.check(name, "episode set", manifest["set_id"] == expect["set_id"] and counts == expect["episodes"] and same,
                 f"the set {manifest['set_id']} (status {manifest['status']}) verifies with every episode in it: "
                 f"{counts['training']} training, {counts['validation']} validation and {counts['held_out']} held-out "
                 f"pairs; the configuration expects {expect['set_id']} with {expect['episodes']['training']}, "
                 f"{expect['episodes']['validation']} and {expect['episodes']['held_out']}; "
                 + (f"it is the set the accepted Sprint 3.4 run was graded on, by the hash of its configuration "
                    f"{manifest['config_sha256'][:16]} and of its assignment {manifest['assignment_sha256'][:16]}, so "
                    "both gates score the same held-out episodes" if same else
                    f"the accepted Sprint 3.4 run was graded on the set {known.get('set_id')} with the configuration "
                    f"{str(known.get('config_sha256'))[:16]}: the linear head cannot be compared on another set"))
    lookup = evaluator.by_key(table)
    groups = table["groups"]
    scenes = {group: {row["trajectory"] for row in table["episodes"] if row["group"] == group} for group in sets.GROUPS}
    disjoint = not (scenes["training"] & scenes["validation"] or scenes["training"] & scenes["held_out"]
                    or scenes["validation"] & scenes["held_out"])
    paired = all(lookup[row["keys"]["a"]]["split"] == lookup[row["keys"]["b"]]["split"] for row in table["episodes"])
    grader.check(name, "assignment", disjoint and paired and len(lookup) == 2 * len(table["episodes"])
                 and manifest["assignment_sha256"] == sets.assignment_sha256(manifest["config"]["trajectories"]),
                 f"an episode has the group of its trajectory, and the set fixed that assignment "
                 f"{manifest['assignment_sha256'][:16]} before an episode was traced or a window was cut: "
                 + ", ".join(f"{len(groups[group])} {group.replace('_', '-')} trajectories" for group in sets.GROUPS)
                 + f", no trajectory in two groups; both members of each matched pair are in one group; "
                 f"{len(lookup)} realizations under distinct opaque keys, and a key names no group and no case")
    # Every entry gives one window per period: the uplink slots of the window, on the grants the control binds.
    slots, period, frame = spec["window"]["slots_in_period"], spec["radio"]["period_slots"], slots_in_frame(spec)
    foreign, uneven = [], []
    for entry in episode_set.episodes():
        for grant_id, described in entry.manifest["grants"].items():
            if control.bindings.get(grant_id, {}).get("record_sha256") != described["record_sha256"]:
                foreign.append(f"{entry.episode_id} {grant_id}")
        found = sorted((item.sfn * frame + item.slot) % period for item in entry.schedule.opportunities)
        if found != sorted(slots * entry.schedule.periods):
            uneven.append(entry.episode_id)
    per_case = {row["periods"] for row in table["episodes"]}
    gains = sorted({row["grid_gain"] for row in table["episodes"]})
    shape = [grid["rows"], grid["cols"]]
    fits = not foreign and not uneven and per_case == {expect["windows_per_realization"]} and shape == expect["grid"]
    grader.check(name, "replayable", fits,
                 f"every entry schedules the slots {slots} of each of its periods of {period} slots and gives "
                 f"{'/'.join(str(value) for value in sorted(per_case))} windows of one period per realization, as the "
                 f"configuration expects; every grant of every entry has the record hash of the grant of that name in "
                 f"the control, so the entries are replayed with the transmit grids that the control binds; the position "
                 f"grid of the set is {shape[0]} rows by {shape[1]} columns of cells, {grid['cell_m'][0]:.3f} m along y by "
                 f"{grid['cell_m'][1]:.3f} m along x; {len(gains)} grid gain(s) in the set" if fits else
                 f"{len(uneven)} entries do not schedule the slots {slots} in every period (first: {uneven[:1]}), "
                 f"{len(foreign)} grants are not those of the control (first: {foreign[:1]}), the entries give "
                 f"{sorted(per_case)} windows per realization and the grid is {shape}; the configuration expects "
                 f"{expect['windows_per_realization']} and {expect['grid']}")
    if not fits or not disjoint:
        return None, {"counts": counts}
    write_json(out / "split.json", table)
    write_json(out / "grid.json", manifest["position_grid"])
    measured = {"set_id": manifest["set_id"], "status": manifest["status"], "directory": str(Path(set_dir).resolve()),
                "config_sha256": manifest["config_sha256"], "assignment_sha256": manifest["assignment_sha256"],
                "counts": counts, "windows_per_realization": expect["windows_per_realization"], "grid_gains": gains,
                "grid": {key: grid[key] for key in ("rows", "cols", "cell_m", "origin_m", "sha256")},
                "split_sha256": digest(out / "split.json")}
    return table, measured


# ---------------------------------------------------------------- records
def run_records(config, out, grader, tools, control, table, evaluator, captures):
    """The tapped receiver replay of the training and of the validation entries, with a consumer on each ring."""
    name, backends, size = "records", config["backends"], config["set"]["entries_per_run"]
    made = []
    for role in ROLES:
        for index, rows in enumerate(evaluator.chunks(table, role, size)):
            capture = Capture(role, f"{role}-{index:02d}", rows, out / "records" / f"{role}-{index:02d}")
            captures.append(capture)
            made.append(capture)
            plan_capture(config, control, evaluator, capture)
            run = capture.run
            names = {stream: capture.directory / f"consumer-{stream}" for stream in ("o2", "o3")}
            capture.started_ns = time.monotonic_ns()
            run.status, run.report, capture.statuses = run_capture(
                tools, capture.directory / evaluator.PLAN, run.native, capture.directory, names,
                config["rings"]["o3"]["wait"], capture.opportunities() + 64, config["receiver"]["keep_grids"])
            capture.ended_ns = time.monotonic_ns()
            verify_capture(capture)
    bad = [capture for capture in made if not clean(capture, backends)]
    opportunities = sum(capture.opportunities() for capture in made)
    grader.check(name, "native runs", not bad and made,
                 f"{len(made)} runs of pilot_ls on plans of version 3 in the mode timed, both stages on rings of "
                 f"{config['rings']['o2']['slots']} and {config['rings']['o3']['slots']} slots: "
                 + ", ".join(f"{sum(capture.role == role for capture in made)} for the "
                             f"{sum(len(capture.rows) for capture in made if capture.role == role)} {role} entries"
                             for role in ROLES)
                 + f", at most {size} entries per run, {opportunities} opportunities in all; one radio record per "
                 "planned decode under a unique key, no anomaly, every grid released" if not bad else
                 f"{len(bad)} of {len(made)} runs did not finish clean, first {bad[0].name}: {bad[0].run.said()}")
    if bad or not made:
        return {}
    good, planned = decoded(made)
    grader.check(name, "decodes", good == planned == opportunities * len(backends),
                 f"{good} of {planned} decodes gave the original bytes of their transport block with a CRC pass, on "
                 f"{' and '.join(backends)}: the receiver decodes every grid of the set while both of its estimates "
                 "are exported")
    path_check(grader, name, made)
    for capture in made:
        problem = open_archives(capture, {stream: capture.directory / f"consumer-{stream}" for stream in ("o2", "o3")})
        if problem:
            grader.check(name, "account", False, problem)
            return {}
    _, total = account_check(grader, name, made)
    periods = [(capture.ended_ns - capture.started_ns) / 1e9 for capture in made]
    estimates = [capture.run.output.estimate for capture in made]
    grader.note(name, f"recorded, not graded: {sum(periods):.1f} s of wall time for {len(made)} runs, tool start and "
                      f"verification of the stimulus included; ring depth at most "
                      f"{max(capture.run.output.base.ring['counts']['max_depth'] for capture in made)} of "
                      f"{config['rings']['o2']['slots']} slots for the stage O2 and "
                      f"{max(item['ring']['counts']['max_depth'] for item in estimates)} of "
                      f"{config['rings']['o3']['slots']} for the stage O3; landing "
                      f"{estimates[0]['landing']['effective']}, copy stream {estimates[0]['copy_stream']}")
    return {"runs": len(made), "opportunities": opportunities, "decodes": planned, "account": total,
            "wall_seconds": sum(periods)}


# ---------------------------------------------------------------- windows
def run_windows(config, out, grader, table, captures, per_case, cusense, evaluator):
    """Stage 2 up to the window on the kept records, for both input entries, and the join of its windows."""
    name, tolerance = "windows", config["tolerance"]["window_relative"]
    np = cusense.np
    parts = {arm: {role: [] for role in ROLES} for arm in ARMS}
    accounts = {arm: {role: [] for role in ROLES} for arm in ARMS}
    counts = {arm: {} for arm in ARMS}
    slots = {arm: {} for arm in ARMS}
    oracle = {arm: {"checked": 0, "agree": 0, "worst": 0.0} for arm in ARMS}
    for capture in captures:
        if capture.role not in ROLES or set(capture.archives) != {"o2", "o3"}:
            continue
        try:
            truth = evaluator.positions(capture.channel)
            for arm in ARMS:
                spec = config["arms"][arm]["spec"]
                directory = out / "windows" / arm / capture.name
                directory.mkdir(parents=True)
                report, stream = cusense.windows(capture.archives[arm].directory, spec, directory)
                made = cusense.open_windows(directory)
                means, held = cusense.stacked(stream) if made.ok else (np.zeros([0] + cusense.shape_of(spec), "<c8"),
                                                                       np.zeros([0] + cusense.shape_of(spec), "u1"))
                rows, means, held, account = evaluator.samples(made.lines, means, held, table, capture.role, truth,
                                                               capture.keys)
                parts[arm][capture.role].append((rows, means, held))
                accounts[arm][capture.role].append(account)
                for key, value in report["counts"].items():
                    counts[arm][key] = counts[arm].get(key, 0) + value
                for line in made.ok:
                    slots[arm][line["window"]["id"]] = [item["opportunity"] for item in line["contributing"]]
                # The first and the last complete window again, element by element, with the reader of their stage.
                by_sequence = {record.sequence: record for record in capture.archives[arm].records}
                for line in ([made.ok[0], made.ok[-1]] if made.ok else [])[:ORACLE_WINDOWS]:
                    mean, count = evaluator.oracle_window([by_sequence[item["sequence"]] for item in line["contributing"]],
                                                          spec)
                    _, kept_mean, kept_count = stream.tensors[line["mean"]["index"]]
                    worst = float((np.abs(mean - kept_mean.astype(np.complex128)) / np.maximum(np.abs(mean), 1.0)).max())
                    oracle[arm]["checked"] += 1
                    oracle[arm]["agree"] += bool(np.array_equal(count, kept_count) and worst <= tolerance)
                    oracle[arm]["worst"] = max(oracle[arm]["worst"], worst)
        except (cusense.CuSenseError, ObsError, evaluator.CuSenseFrontendError) as error:
            grader.check(name, "windows", False, f"{capture.name}: {error}")
            return {}
    realizations = {arm: {role: sum(item["realizations"] for item in accounts[arm][role]) for role in ROLES} for arm in ARMS}
    windows = {arm: {role: sum(item["windows"] for item in accounts[arm][role]) for role in ROLES} for arm in ARMS}
    missing = [dict(item, arm=arm) for arm in ARMS for role in ROLES for account in accounts[arm][role]
               for item in account["missing"]]
    whole = all(windows[arm][role] == per_case * realizations[arm][role] and realizations[arm][role]
                for arm in ARMS for role in ROLES) and not missing \
        and not any(counts[arm].get(key) for arm in ARMS for key in ("gap", "invalid", "rejected"))
    grader.check(name, "windows", whole,
                 "; ".join(
                     f"{config['arms'][arm]['spec']['input']['entry']}: {counts[arm].get('records', 0)} records of the "
                     f"stage {arm.upper()} in sequence order through the stream of {config['arms'][arm]['spec']['spec_id']}, "
                     + ", ".join(f"{windows[arm][role]} complete windows for {realizations[arm][role]} {role} realizations"
                                 for role in ROLES)
                     + f", {counts[arm].get('insufficient_history', 0)} answers of a filling window, "
                     f"{counts[arm].get('gap', 0)} gaps, {counts[arm].get('invalid', 0)} invalid, "
                     f"{counts[arm].get('rejected', 0)} rejected" for arm in ARMS)
                 + f"; {per_case} windows per realization, one per period, and every output verifies with its reader"
                 + ("" if not missing else f"; without a window: {len(missing)} realizations, first "
                                           f"{missing[0]['episode']} ({missing[0]['last_answer']})"))
    if not whole:
        return {}
    grader.check(name, "same slots", slots["o3"] == slots["o2"] and len(slots["o3"]) > 0,
                 f"the {len(slots['o3'])} windows of the stage O3 and the {len(slots['o2'])} of the stage O2 have the "
                 "same identifiers, and each pair names the same contributing opportunities in the same order: the two "
                 "input entries are computed from the same uplink slots of the same receiver replay")
    grader.check(name, "oracle", all(oracle[arm]["checked"] == oracle[arm]["agree"] > 0 for arm in ARMS),
                 "; ".join(f"{oracle[arm]['agree']} of {oracle[arm]['checked']} windows of the stage {arm.upper()} agree "
                           f"with the coherent average computed again from the payloads, value by value, with the "
                           f"reader of that stage and none of the code of the stream: the same counts of valid records "
                           f"at every element and a mean within {oracle[arm]['worst']:.1e} of it (tolerance "
                           f"{tolerance:.0e}, the stored mean is complex64)" for arm in ARMS))
    files, balance, facts, rows = {arm: {} for arm in ARMS}, {arm: {} for arm in ARMS}, {}, {arm: {} for arm in ARMS}
    try:
        for arm in ARMS:
            for role in ROLES:
                merged, means, held = evaluator.merge_samples(parts[arm][role])
                path = out / "samples" / arm / f"{role}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                cusense.write_samples(path, merged, means, held)
                listed = cusense.sample_files(path)
                files[arm][role] = {"file": str(path.relative_to(out)),
                                    "files": {item.name: digest(item) for item in listed}}
                balance[arm][role] = (sum(row["label"] for row in merged), sum(1 - row["label"] for row in merged))
                rows[arm][role] = merged
            facts[arm] = cusense.check_split(rows[arm]["training"], rows[arm]["validation"])
    except (cusense.CuSenseError, evaluator.CuSenseFrontendError) as error:
        grader.check(name, "samples", False, f"{error}")
        return {}
    placed = all(row["cell"] is not None and len(row["position_m"]) == 2 for arm in ARMS for role in ROLES
                 for row in rows[arm][role] if row["label"])
    grader.check(name, "samples", placed and all(balance[arm][role][0] == balance[arm][role][1] > 0
                                                for arm in ARMS for role in ROLES),
                 f"each complete window is joined to the split table by the episode key of its records, after the "
                 f"windows were written: {balance['o3']['training'][0]} target-present and "
                 f"{balance['o3']['training'][1]} target-absent training windows from "
                 f"{len(facts['o3']['scenes']['training'])} trajectories, {balance['o3']['validation'][0]} and "
                 f"{balance['o3']['validation'][1]} validation windows from {len(facts['o3']['scenes']['validation'])}, "
                 "the same counts for the stage O2; a target-present window carries the cell and the position of its "
                 "target at the last opportunity of the window, a target-absent window none; no window, key, pair or "
                 f"trajectory is in both sets; windows of shape {facts['o3']['shape']} and {facts['o2']['shape']} as "
                 "complex64 with their counts (samples/o3, samples/o2)")
    return {"windows": windows, "counts": counts, "oracle": oracle, "samples": files, "rows": rows,
            "shape": {arm: facts[arm]["shape"] for arm in ARMS}}


# ---------------------------------------------------------------- train
def train_arm(config, out, grader, arm, found, cusense, presence, evaluator):
    """The training script on the sample sets of one input entry, and what it froze, checked here a second time."""
    name, tolerance, expect = TRAIN_CASE[arm], config["tolerance"], config["expect"]
    np = cusense.np
    spec = config["arms"][arm]["spec"]
    settings = dict(config["training"], model_id=config["arms"][arm]["model_id"])
    directory = out / "train" / arm
    write_json(out / f"training-{arm}.json", settings)
    paths = {role: out / "samples" / arm / f"{role}.jsonl" for role in ROLES}
    command = [sys.executable, str(ROOT / "scripts/train_cusense.py"), "--training", str(paths["training"]),
               "--validation", str(paths["validation"]), "--spec", str(out / "specs" / f"{arm}.json"), "--grid",
               str(out / "grid.json"), "--config", str(out / f"training-{arm}.json"), "--output-dir", str(directory)]
    result = run_tool(command, out / f"train-{arm}.log", timeout=TRAIN_TIMEOUT_S)
    try:
        if result.returncode != 0:
            raise cusense.CuSenseError(f"exit status {result.returncode}: {last_line(out / f'train-{arm}.log')}")
        freeze = load_json(directory / "freeze.json")
        fit = load_json(directory / "fit.json")
        artifact = cusense.open_artifact(directory / "artifact", freeze["artifact"]["sha256"])
        grid = cusense.grid_record(load_json(out / "grid.json"))
        sets_read = {role: cusense.read_samples(paths[role]) for role in ROLES}
    except (cusense.CuSenseError, ContractError, OSError, KeyError) as error:
        grader.check(name, "script", False, f"scripts/train_cusense.py left no frozen model: {error}")
        return {}, None
    manifest = artifact.manifest
    rx, positions, symbols = cusense.shape_of(spec)
    wanted = cusense.model_config(settings, rx, grid)
    grader.check(name, "script", manifest["model_id"] == settings["model_id"] and manifest["input"]["spec"] == spec
                 and manifest["model"]["config"] == wanted and manifest["smoothing"] == settings["smoothing"]
                 and manifest["grid"]["sha256"] == grid["sha256"],
                 f"python scripts/train_cusense.py read the two sample sets, the specification, the grid and the "
                 f"configuration and no other input, and froze the model {manifest['model_id']} {artifact.sha256[:16]}: "
                 f"the entry {manifest['entries']['model']}, a plain-block 1D CNN of {manifest['model']['parameters']} "
                 f"parameters from [1, {rx}, {positions}] to a map of {grid['rows']} by {grid['cols']} cells and a presence "
                 f"logit, on the input entry {manifest['entries']['input']} and the grid of the set")
    files = sorted(path.name for path in (directory / "artifact").iterdir())
    words = [key for key in EVALUATOR_KEYS if key in (directory / "artifact" / cusense.MODEL_FILE).read_text()]
    parts = {part: manifest[part]["sha256"] for part in ("weights", "template", "normalizer", "grid", "threshold")}
    hashed = parts["weights"] == digest(directory / "artifact" / cusense.WEIGHTS_FILE) \
        and parts["template"] == digest(directory / "artifact" / cusense.TEMPLATE_FILE) \
        and manifest["template"]["count_sha256"] == digest(directory / "artifact" / cusense.TEMPLATE_COUNT_FILE) \
        and parts["normalizer"] == cusense.normalizer_sha256(manifest["normalizer"]) \
        and parts["threshold"] == cusense.part_sha256(manifest["threshold"]) \
        and parts["grid"] == cusense.part_sha256(manifest["grid"])
    grader.check(name, "artifact", files == sorted(cusense.ARTIFACT_FILES) and hashed and not words,
                 f"the deployment artifact is {', '.join(files)}: the template "
                 f"{parts['template'][:16]} of shape {manifest['template']['shape']}, the normalizer "
                 f"{parts['normalizer'][:16]}, {manifest['weights']['bytes']} bytes of weights {parts['weights'][:16]} in "
                 f"{len(manifest['weights']['tensors'])} tensors, the threshold {parts['threshold'][:16]} and the grid "
                 f"geometry {parts['grid'][:16]} with its units, each hashed here again; it holds no sample, no label and "
                 "no file name of the training run, and it verifies with its reader"
                 if not words else f"model.json holds evaluator words: {words}")
    # ---- Stage 1 and the statistics of stage 2, computed here again from the training windows alone.
    training, validation = sets_read["training"], sets_read["validation"]
    absent = [index for index, row in enumerate(training.rows) if row["label"] == 0]
    template = cusense.fit_template([training.rows[index] for index in absent], training.means[absent],
                                    training.counts[absent])
    normalizer = cusense.fit_normalizer(training.rows, training.means, training.counts, template)
    moved = [dict(row, split="training") for row in validation.rows]
    other = cusense.fit_normalizer(training.rows + moved, np.concatenate([training.means, validation.means]),
                                   np.concatenate([training.counts, validation.counts]), template)
    stored = manifest["normalizer"]
    close = np.array_equal(artifact.template["values"], template["values"]) \
        and np.array_equal(artifact.template["count"], template["count"]) \
        and (stored["mean"], stored["scale"]) == (normalizer["mean"], normalizer["scale"]) \
        and manifest["template"]["fitted_on"] == template["fitted_on"] \
        and stored["fitted_on"] == normalizer["fitted_on"]
    grader.check(name, "statistics", close and template["fitted_on"]["split"] == "training",
                 f"the template H_B, the complex mean of the records behind the {len(absent)} target-absent training "
                 f"windows over their valid elements, {manifest['template']['records_min']} to "
                 f"{manifest['template']['records_max']} records per element, and the one mean {stored['mean']:+.6f} and "
                 f"deviation {stored['scale']:.6f} of the z-score over the {len(training.rows)} training windows, both "
                 f"computed here again from the training sample set alone, are those of the artifact bit for bit; with "
                 f"the {len(validation.rows)} validation windows added the mean would differ by "
                 f"{abs(other['mean'] - normalizer['mean']):.3e}: no validation window entered either, and the artifact "
                 "names the windows they were fitted on by a hash")
    # ---- The threshold, from the target-absent validation windows, by the rule of Sprint 3.4.
    model = cusense.Model(artifact, "cpu")
    logits = [model.score(mean, count)["logit"] for mean, count in zip(validation.means, validation.counts)]
    again = presence.calibrate_threshold(validation.rows, logits, settings["threshold"]["target_false_alarm"],
                                         settings["threshold"]["min_negatives"])
    threshold = manifest["threshold"]
    counted = all(again[key] == threshold[key] for key in ("entry", "rule", "target_false_alarm", "negatives", "allowed",
                                                            "exceeding"))
    grader.check(name, "threshold", counted and threshold["exceeding"] <= threshold["allowed"]
                 and abs(again["logit"] - threshold["logit"]) <= tolerance["reload_absolute"]
                 and abs(again["probability"] - threshold["probability"]) <= tolerance["reload_absolute"],
                 f"the frozen threshold is the logit {threshold['logit']:.6f} (probability "
                 f"{threshold['probability']:.4f}): with a false-alarm target of {threshold['target_false_alarm']} the "
                 f"rule of Sprint 3.4 allows {threshold['allowed']} of the {threshold['negatives']} target-absent "
                 f"validation windows above it, and {threshold['exceeding']} are; recomputed here from the artifact "
                 f"read back, one window at a time on the CPU, it has the same counts and a logit within "
                 f"{abs(again['logit'] - threshold['logit']):.1e} of it (tolerance {tolerance['reload_absolute']:.0e}); no "
                 "target-present window and no training window enters it")
    history = fit["history"]
    reload = freeze["reload"]["largest_logit_difference"]
    second = cusense.Model(cusense.open_artifact(directory / "artifact"), "cpu")
    some = range(0, len(validation.rows), max(1, len(validation.rows) // 32))
    same = all(second.score(validation.means[index], validation.counts[index])["logit"] == logits[index] for index in some)
    area = evaluator.area_under_curve(logits, [row["label"] for row in validation.rows])
    rates = fit["at_the_frozen_threshold"]
    learned = area is not None and area >= expect["validation_auc_min"] \
        and math.isclose(rates["validation"]["area_under_curve"], area, abs_tol=1e-9)
    grader.check(name, "fit", history["final_loss"] < history["initial_loss"] and reload <= tolerance["reload_absolute"]
                 and same and learned
                 and math.isclose(history["initial_loss"], history["initial_expected"]["loss"], abs_tol=1e-4),
                 f"the joint loss, binary cross-entropy of the presence logit on every window plus the KL divergence "
                 f"of the map on the target-present ones, falls from {history['initial_loss']:.6f}, the value of a "
                 f"uniform map and a logit of zero, to {history['final_loss']:.6f} in {history['epochs']} epochs of Adam "
                 f"on {history['device']}; the artifact read back scores the windows of the fit within {reload:.1e} of it "
                 f"(tolerance {tolerance['reload_absolute']:.0e}), and a second load gives the same validation logits bit "
                 f"for bit; on the validation windows the area under the curve is {text(area, 3)}, and the floor that "
                 f"the configuration declares for a model that learned presence is {expect['validation_auc_min']}")
    grader.note(name, "recorded, not graded, on the samples of the fit: "
                + "; ".join(f"{role}: {rates[role]['detected']} of {rates[role]['target_present']} target-present "
                            f"windows above the threshold, {rates[role]['false_alarms']} of "
                            f"{rates[role]['target_absent']} target-absent, area under the curve "
                            f"{text(rates[role]['area_under_curve'], 3)}, position error of the raw map mean "
                            f"{text(rates[role]['position_error']['mean_m'], 3)} m and median "
                            f"{text(rates[role]['position_error']['median_m'], 3)} m over "
                            f"{rates[role]['position_error']['eligible']} windows" for role in ROLES)
                + f"; loss parts at the end: presence {history['final']['presence']:.4f}, location "
                  f"{history['final']['location']:.4f} nats")
    measured = {"artifact_sha256": artifact.sha256, "parts": parts, "parameters": manifest["model"]["parameters"],
                "threshold": threshold, "normalizer": {"mean": stored["mean"], "scale": stored["scale"]},
                "fit": {"initial_loss": history["initial_loss"], "final_loss": history["final_loss"],
                        "epochs": history["epochs"], "device": history["device"]}, "reload": reload, "rates": rates,
                "validation_auc": area}
    return measured, {"freeze": freeze, "artifact": artifact, "directory": directory, "settings": settings}


# ---------------------------------------------------------------- freeze
def run_freeze(config, out, grader, table, found, trained, evaluator):
    """What the freeze records bind, and that nothing of the held-out group was read before them."""
    name = "freeze"
    lookup = evaluator.by_key(table)
    held = {key for key, known in lookup.items() if known["split"] == "held_out"}
    chains, wrong, leaked, listed = [], [], [], {}
    for arm in ARMS:
        freeze, directory, artifact = trained[arm]["freeze"], trained[arm]["directory"], trained[arm]["artifact"]
        manifests = {role: load_json(directory / f"{role}_manifest.json") for role in ROLES}
        files = {path.name: digest(path) for path in (directory / "artifact").iterdir()}
        chains.append(
            freeze["artifact"]["sha256"] == artifact.sha256
            and {name_: entry["sha256"] for name_, entry in freeze["artifact"]["files"].items()} == files
            and all(freeze["artifact"][f"{part}_sha256"] == artifact.manifest[part]["sha256"]
                    for part in ("weights", "template", "normalizer", "grid", "threshold"))
            and all(freeze["manifests"][role]["sha256"] == digest(directory / f"{role}_manifest.json")
                    and {name_: entry["sha256"] for name_, entry in manifests[role]["samples"]["files"].items()}
                    == found["samples"][arm][role]["files"] for role in ROLES))
        wrong += [f"{arm} {role} {key}" for role in ROLES for key in manifests[role]["episode_keys"]
                  if lookup.get(key, {}).get("split") != role]
        leaked += [key for role in ROLES for key in manifests[role]["episode_keys"] if key in held]
        leaked += [window for role in ROLES for window in manifests[role]["window_ids"] if window.split("/")[1] in held]
        listed[arm] = {role: len(manifests[role]["episode_keys"]) for role in ROLES}
    grader.check(name, "hash chain", all(chains),
                 "; ".join(f"train/{arm}/freeze.json names the artifact {trained[arm]['artifact'].sha256[:16]}, its four "
                           f"files and its five parts by SHA-256 and both manifests, and each manifest the three files of "
                           f"the sample set this driver wrote" for arm in ARMS) + "; all were hashed here again")
    untouched = not any((out / later).exists() for later in ("live", "parity", "evaluation"))
    grader.check(name, "no test access", not wrong and not leaked and untouched and len(held) > 0,
                 f"the manifests of both arms list {listed['o3']['training']} training and "
                 f"{listed['o3']['validation']} validation episode keys, each of its own split; none of the {len(held)} "
                 "held-out keys and no window of one is in any; when the models were frozen this run held no plan, no "
                 "record and no window of a held-out entry: the files of the set were verified by their hashes, and no "
                 "estimate of a held-out grid existed" if not wrong and not leaked else
                 f"{len(wrong)} keys are in the wrong manifest and {len(leaked)} held-out identifiers are in a manifest")
    frozen = max(trained[arm]["freeze"]["frozen_monotonic_ns"] for arm in ARMS)
    return {"frozen_utc": {arm: trained[arm]["freeze"]["frozen_utc"] for arm in ARMS}, "frozen_monotonic_ns": frozen,
            "freeze_sha256": {arm: digest(trained[arm]["directory"] / "freeze.json") for arm in ARMS},
            "held_out_keys": len(held)}


# ---------------------------------------------------------------- live
def run_live_case(config, out, grader, tools, control, table, trained, frozen, per_case, cusense, evaluator, captures):
    """The held-out entries, replayed for the first time, with the sensor on the O3 ring in a second process."""
    name, backends, settings = "live", config["backends"], config["rings"]["o3"]
    arm = trained["o3"]
    freeze, directory = arm["freeze"], arm["directory"]
    accessed = time.monotonic_ns()
    intact = all(digest(trained[item]["directory"] / "freeze.json") == frozen["freeze_sha256"][item]
                 and cusense.open_artifact(trained[item]["directory"] / "artifact").sha256
                 == trained[item]["freeze"]["artifact"]["sha256"] for item in ARMS)
    grader.check(name, "after the freeze", intact and frozen["frozen_monotonic_ns"] < accessed,
                 f"the first plan of a held-out entry is written {(accessed - frozen['frozen_monotonic_ns']) / 1e9:.1f} s "
                 f"after the later of the two freeze records, on the monotonic clock of this host; both freeze records "
                 "and both artifacts have the hashes of the case freeze")
    device = config["sensor"]["device"]
    made = []
    for index, rows in enumerate(evaluator.chunks(table, "held_out", config["set"]["entries_per_run"])):
        capture = Capture("held_out", f"held-out-{index:02d}", rows, out / "live" / f"held-out-{index:02d}")
        captures.append(capture)
        made.append(capture)
        plan_capture(config, control, evaluator, capture)
        run = capture.run
        capture.started_ns = time.monotonic_ns()
        run.status, run.report, capture.statuses = run_live(
            tools, capture.directory / evaluator.PLAN, run.native, capture.directory, capture.directory / "sensor",
            directory / "artifact", device, settings["wait"], config["sensor"]["ready_timeout_ms"] / 1000,
            capture.directory / "consumer-o2", capture.opportunities() + 64, config["receiver"]["keep_grids"])
        capture.ended_ns = time.monotonic_ns()
        verify_capture(capture)
    bad = [capture for capture in made if not clean(capture, backends)]
    opportunities = sum(capture.opportunities() for capture in made)
    grader.check(name, "native run", not bad and made,
                 f"{len(made)} run(s) of pilot_ls in the mode timed for the {sum(len(capture.rows) for capture in made)} "
                 f"held-out entries, {opportunities} opportunities, with the cuSense-like sensor attached to the ring of "
                 f"the stage O3 in the mode {settings['wait']} and a native consumer on the ring of the stage O2; one "
                 "radio record per planned decode under a unique key, no anomaly, every grid released" if not bad else
                 f"{len(bad)} of {len(made)} runs did not finish clean, first {bad[0].name}: {bad[0].run.said()}")
    if bad or not made:
        return {}
    good, planned = decoded(made)
    grader.check(name, "decodes", good == planned == opportunities * len(backends),
                 f"{good} of {planned} decodes gave the original bytes of their transport block with a CRC pass while "
                 "the sensor scored on the ring")
    path_check(grader, name, made)
    # ---- The sensor: what it wrote, what it loaded and where it ran.
    problems, facts = [], []
    for capture in made:
        sensor_dir = capture.directory / "sensor"
        try:
            if capture.statuses.get("o3") != 0:
                raise cusense.CuSenseError(f"the sensor exited with status {capture.statuses.get('o3')}: "
                                           f"{last_line(capture.directory / 'sensor.log')}")
            capture.output = cusense.open_output(sensor_dir)
            ready = load_json(capture.directory / "sensor-ready.json")
            problem = open_archives(capture, {"o3": sensor_dir / cusense.WINDOWS_DIR / cusense.RECORDS_DIR,
                                              "o2": capture.directory / "consumer-o2"})
            if problem:
                raise ObsError(problem)
        except (cusense.CuSenseError, ObsError, ContractError, OSError) as error:
            grader.check(name, "sensor", False, f"{capture.name}: the sensor left no output that verifies: {error}")
            return {}
        report, source = capture.output.report, capture.output.report["source"]
        records = capture.archives["o3"].records
        first = min((record.header["t_publish_ns"] for record in records), default=None)
        facts.append({"run": capture.name, "device": report["runtime"]["device"], "device_name": report["runtime"].get("name"),
                      "loads": report["loads"], "warm_up_ms": source.get("warm_up_ms"), "records": source.get("records"),
                      "ready_before_first_record_ms": None if first is None else (first - ready["t_ready_ns"]) / 1e6})
        if source.get("kind") != "ring" or source.get("ended") != "closed" or source.get("stale") != 0 \
                or source.get("records") != len(records):
            problems.append(f"{capture.name}: the sensor did not read the ring to its end without a stale read")
        if report["loads"] != 1:
            problems.append(f"{capture.name}: the artifact was loaded {report['loads']} times in the process of the sensor")
        if capture.output.manifest["sha256"] != freeze["artifact"]["sha256"]:
            problems.append(f"{capture.name}: the sensor scored with the artifact {capture.output.manifest['sha256'][:16]}, "
                            "which is not the frozen one")
        if report["runtime"]["type"] != device:
            problems.append(f"{capture.name}: the model ran on {report['runtime']['device']}, and the gate selects {device}")
        if first is None or not ready["t_ready_ns"] < first or ready["artifact_sha256"] != freeze["artifact"]["sha256"]:
            problems.append(f"{capture.name}: the sensor was not warm before the first record was published")
    shown = facts[0]
    warm = shown["warm_up_ms"] or {}
    grader.check(name, "sensor", not problems,
                 f"python -m ocudu_cuda_isac.sensing.cusense follow, a second process on the standard-library reader, "
                 f"loaded the frozen artifact {freeze['artifact']['sha256'][:16]} once and ran the model on "
                 f"{shown['device']}" + (f" ({shown['device_name']})" if shown["device_name"] else "")
                 + f", with its front end in NumPy on the CPU; it ran its stages and its first inference before it "
                 f"attached, {text(warm.get('frontend'), 2)} and {text(warm.get('model'), 2)} ms, and was ready "
                 f"{text(shown['ready_before_first_record_ms'], 1)} ms before the first record was published; it read "
                 f"{sum(item['records'] for item in facts)} records of the stage O3 until the producer said that none "
                 "follows, with no stale read, and its output verifies" if not problems else "; ".join(problems[:3]))
    if problems:
        return {}
    _, total = account_check(grader, name, made)
    # ---- One decision per answer of the stream, and one scored window per period of every held-out realization.
    lookup = evaluator.by_key(table)
    scored = [line for capture in made for line in capture.output.scored]
    keys = [key for capture in made for key in capture.keys]
    by_key = {}
    for line in scored:
        by_key.setdefault(line["window"]["episode"], []).append(line)
    others = [line for capture in made for line in capture.output.decisions
              if line["status"] not in ("ok", "insufficient_history")]
    complete = all(len(by_key.get(key, [])) == per_case for key in keys) and len(by_key) == len(keys) \
        and all(lookup[key]["split"] == "held_out" for key in by_key) and not others \
        and all(line["trigger"]["sequence"] == line["contributing"][-1]["sequence"] for line in scored)
    decisions = sum(len(capture.output.decisions) for capture in made)
    grader.check(name, "decisions", complete,
                 f"{decisions} decisions, one per answer of the window stream: {len(scored)} scored windows, {per_case} "
                 f"for each of the {len(keys)} held-out realizations, each answered at the record that completes its "
                 f"period, and {decisions - len(scored)} of a filling window without a score; every decision names its "
                 "contributing observations, the first and the last sample of its window, the digests of its input and "
                 "of its map, the cell and the position in metres, and the hashes of the frozen model"
                 + ("" if not others else f"; {len(others)} decisions with the status {others[0]['status']}"))
    infer = [line["infer_ns"] / 1000.0 for line in scored]
    front = [line["frontend_ns"] / 1000.0 for line in scored if line.get("frontend_ns") is not None]
    delays = [(line["t_decision_ns"] - line["t_publish_ns"]) / 1000.0 for line in scored
              if line["t_decision_ns"] is not None and line["t_publish_ns"]]
    grader.note(name, f"device timing, not graded, runtime reference: one batch-one inference of the model on "
                      f"{shown['device']}, from the input in host memory to the map and the logit in host memory, median "
                      f"{text(obs.percentile(infer, 0.5))} us and largest {text(max(infer, default=None))} us over "
                      f"{len(infer)} windows; the template subtraction and the z-score of one window in NumPy on the "
                      f"CPU, median {text(obs.percentile(front, 0.5))} us; from the publication store of the completing "
                      f"record to the decision, "
                      f"median {text(obs.percentile(delays, 0.5))} us and largest {text(max(delays, default=None))} us; "
                      "this is PyTorch in a Python sensor behind a reader that sleeps, beside a receiver that shares "
                      "the device without MPS, and no latency of the sensing core: S3.6 times the engine by the "
                      "boundary of the published 241 us")
    return {"runs": len(made), "opportunities": opportunities, "account": total, "sensor": facts,
            "scored": len(scored), "decisions": decisions, "infer_us": sr.spread(infer),
            "frontend_us": sr.spread(front), "publish_to_decision_us": sr.spread(delays),
            "first_test_access_monotonic_ns": accessed}


# ---------------------------------------------------------------- parity
def run_parity(config, out, grader, trained, captures, cusense, evaluator):
    """The records the sensor kept, scored again offline on the CPU, against what the sensor decided on its device."""
    name, tolerance = "parity", config["tolerance"]
    made = [capture for capture in captures if capture.role == "held_out" and capture.output is not None]
    problems, flips, moved, lines, hashes, joined = [], [], [], 0, 0, 0
    worst = {"logit": 0.0, "probability": 0.0, "map": 0.0}
    for capture in made:
        directory = out / "parity" / capture.name
        directory.mkdir(parents=True)
        log = out / "parity" / f"{capture.name}.log"
        result = run_tool([sys.executable, "-m", "ocudu_cuda_isac.sensing.cusense", "replay", "--archive",
                           str(capture.archives["o3"].directory), "--artifact", str(trained["o3"]["directory"] / "artifact"),
                           "--device", "cpu", "--output-dir", str(directory)], log)
        try:
            if result.returncode != 0:
                raise cusense.CuSenseError(f"exit status {result.returncode}: {last_line(log)}")
            offline = cusense.open_output(directory)
        except cusense.CuSenseError as error:
            grader.check(name, "offline", False, f"{capture.name}: the offline replay left no output that verifies: {error}")
            return {}
        capture.offline = offline
        live_lines, again = capture.output.windows.lines, offline.windows.lines
        if [cusense.comparable_window(line) for line in live_lines] != [cusense.comparable_window(line) for line in again]:
            problems.append(f"{capture.name}: the window lines of the sensor differ from those of the replay")
        lines += len(live_lines)
        hashes += sum(a["mean"]["sha256"] == b["mean"]["sha256"] and a["count"]["sha256"] == b["count"]["sha256"]
                      for a, b in zip(capture.output.windows.ok, offline.windows.ok))
        wrong, facts = evaluator.parity(capture.output, offline, tolerance)
        problems += [f"{capture.name}: {problem}" for problem in wrong]
        flips += facts["flips"]
        moved += facts["locations_moved"]
        joined += facts["joined"]
        worst = {"logit": max(worst["logit"], facts["largest_logit_difference"]),
                 "probability": max(worst["probability"], facts["largest_probability_difference"]),
                 "map": max(worst["map"], facts["largest_map_difference"])}
        write_json(out / "parity" / f"{capture.name}.json", facts)
    window_problems = [problem for problem in problems if "window lines" in problem]
    grader.check(name, "same windows", not window_problems and lines and hashes == joined,
                 f"python -m ocudu_cuda_isac.sensing.cusense replay, in a process of its own on the CPU, on the records "
                 f"the sensor kept: the {lines} answers of the stream are those of the sensor line for line, host times "
                 f"aside, with {hashes} window tensors of the same SHA-256" if not window_problems and hashes == joined
                 else (window_problems or [f"{hashes} of {joined} window tensors have the same SHA-256"])[0])
    decision_problems = [problem for problem in problems if "the decision differs" in problem]
    score_problems = [problem for problem in problems if problem not in window_problems + decision_problems]
    device = made[0].output.report["runtime"]["device"] if made else "none"
    shape = trained["o3"]["artifact"].manifest["input"]["shape"]
    grader.check(name, "scores", not score_problems and joined > 0,
                 f"{joined} held-out windows joined by their identifier: on both sides the same contributing "
                 f"observations and the same SHA-256 of the input, {shape} as float32; the logit on {device} and offline "
                 f"on the cpu differ by at most {worst['logit']:.2e} (tolerance {tolerance['logit_absolute']:.0e} plus "
                 f"{tolerance['logit_relative']:.0e} of the logit), the "
                 f"probability by at most {worst['probability']:.2e} ({tolerance['probability_absolute']:.1e}) and a "
                 f"cell of the map by at most {worst['map']:.2e} ({tolerance['map_absolute']:.0e}): one artifact gives "
                 "the same map and score for the same held-out window offline and on the selected device"
                 if not score_problems else f"{len(score_problems)} differences, first: {score_problems[0]}")
    grader.check(name, "decisions", not decision_problems and joined > 0,
                 f"at the frozen threshold the {joined} decisions of the device are those of the offline replay"
                 + (f", {len(flips)} with a score within the tolerance of a logit of the threshold aside" if flips else "")
                 + f"; the cell of the largest value of the map is the same in {joined - len(moved)} of {joined} windows"
                 + ("" if not moved else ", and where it differs two cells hold nearly the same value: first "
                                         f"{moved[0]['window']}, {moved[0]['device_peak']:.6f} against "
                                         f"{moved[0]['offline_peak']:.6f}")
                 if not decision_problems else f"{len(decision_problems)} decisions differ at the threshold beyond the "
                                               f"tolerance of a logit, first: {decision_problems[0]}")
    return {"joined": joined, "largest_logit_difference": worst["logit"],
            "largest_probability_difference": worst["probability"], "largest_map_difference": worst["map"],
            "flips": flips, "locations_moved": len(moved)}


# ---------------------------------------------------------------- evaluation
def run_evaluation(config, out, grader, control, table, trained, captures, per_case, measured, cusense, presence,
                   evaluator):
    """The second input and the linear head on the records of the same held-out run, and then the truth."""
    name, selection, expect = "evaluation", config["selection"]["window"], config["expect"]
    made = [capture for capture in captures if capture.role == "held_out" and capture.output is not None]
    keys = [key for capture in made for key in capture.keys]
    # ---- The O2 arm, offline, on the records of the stage O2 that the native consumer kept.
    second, problems = [], []
    for capture in made:
        directory = out / "evaluation" / "o2" / capture.name
        directory.mkdir(parents=True)
        log = out / "evaluation" / "o2" / f"{capture.name}.log"
        result = run_tool([sys.executable, "-m", "ocudu_cuda_isac.sensing.cusense", "replay", "--archive",
                           str(capture.archives["o2"].directory), "--artifact", str(trained["o2"]["directory"] / "artifact"),
                           "--device", "cpu", "--output-dir", str(directory)], log)
        try:
            if result.returncode != 0:
                raise cusense.CuSenseError(f"exit status {result.returncode}: {last_line(log)}")
            second.append(cusense.open_output(directory))
        except cusense.CuSenseError as error:
            problems.append(f"{capture.name}: the replay of the O2 arm left no output that verifies: {error}")
    scored = {}
    for output in second:
        for line in output.scored:
            scored.setdefault(line["window"]["episode"], []).append(line)
        if output.manifest["sha256"] != trained["o2"]["artifact"].sha256:
            problems.append("the O2 arm was not scored with its frozen artifact")
    slots = {line["window"]["id"]: [item["opportunity"] for item in line["contributing"]]
             for capture in made for line in capture.output.scored}
    other = {line["window"]["id"]: [item["opportunity"] for item in line["contributing"]]
             for output in second for line in output.scored}
    whole = not problems and len(second) == len(made) and all(len(scored.get(key, [])) == per_case for key in keys) \
        and slots == other
    grader.check(name, "second input", whole,
                 f"python -m ocudu_cuda_isac.sensing.cusense replay with the frozen artifact "
                 f"{trained['o2']['artifact'].sha256[:16]} of the O2 arm on the records of the stage O2 that the native "
                 f"consumer kept during the same held-out run: {len(other)} scored windows, {per_case} for each of the "
                 f"{len(keys)} realizations, with the identifiers and the contributing opportunities of the windows that "
                 "the sensor scored on the stage O3: both input entries decide on the same slots"
                 if whole else (problems or ["the O2 arm did not score the windows that the sensor scored"])[0])
    # ---- The linear head of Sprint 3.4, on the same records of the stage O2.
    baselines, problems = [], []
    for capture in made:
        directory = out / "evaluation" / "linear" / capture.name
        directory.mkdir(parents=True)
        log = out / "evaluation" / "linear" / f"{capture.name}.log"
        result = run_tool([sys.executable, "-m", "ocudu_cuda_isac.sensing.presence", "replay", "--archive",
                           str(capture.archives["o2"].directory), "--spec", str(control.directory / "spec.json"),
                           "--artifact", str(control.directory / "train" / "artifact"), "--device", "cpu", "--output-dir",
                           str(directory)], log)
        try:
            if result.returncode != 0:
                raise presence.PresenceError(f"exit status {result.returncode}: {last_line(log)}")
            output = presence.open_output(directory)
            if output.manifest["sha256"] != control.head.sha256:
                raise presence.PresenceError("the replay did not score with the head of the control")
            baselines.append(evaluator.baseline(output, table, "held_out", capture.keys))
        except (presence.PresenceError, evaluator.CuSenseFrontendError) as error:
            problems.append(f"{capture.name}: {error}")
    linear = None
    if baselines and not problems:
        counts = {"target_present": dict.fromkeys(("episodes", "detected", "missed", "missing"), 0),
                  "target_absent": dict.fromkeys(("episodes", "false_alarm", "quiet", "missing"), 0)}
        rows = []
        for item in baselines:
            rows += item["rows"]
            for side in counts:
                for key in counts[side]:
                    counts[side][key] += item["counts"][side][key]
        scores = [(row["logit"], row["label"]) for row in rows if row["logit"] is not None]
        linear = {"head_id": control.head.manifest["head_id"], "artifact_sha256": control.head.sha256,
                  "spec_id": control.spec["spec_id"], "counts": counts, "rows": rows,
                  "area_under_curve": evaluator.area_under_curve([logit for logit, _ in scores],
                                                                 [label for _, label in scores])}
    grader.check(name, "linear head", linear is not None
                 and linear["counts"]["target_present"]["missing"] == linear["counts"]["target_absent"]["missing"] == 0
                 and linear["counts"]["target_present"]["episodes"] + linear["counts"]["target_absent"]["episodes"]
                 == len(keys),
                 f"python -m ocudu_cuda_isac.sensing.presence replay with the head {control.head.sha256[:16]} that the "
                 f"accepted Sprint 3.4 run froze, on the same records of the stage O2: one decision for each of the "
                 f"{len(keys)} held-out realizations at its frozen threshold, none missing; its one window of "
                 f"{control.spec['window']['periods']} periods spans the periods that this branch cuts into {per_case} "
                 "windows"
                 if linear is not None else (problems or ["no baseline"])[0])
    # ---- The truth, read after every decision of every arm was written.
    evaluations = {"o3": [], "o2": []}
    for capture, output in zip(made, second):
        truth = evaluator.positions(capture.channel)
        evaluations["o3"].append(evaluator.evaluate(capture.output, table, truth, "held_out", capture.keys, per_case,
                                                    selection))
        evaluations["o2"].append(evaluator.evaluate(output, table, truth, "held_out", capture.keys, per_case, selection))
    arms = {arm: evaluator.merge_evaluations(items) for arm, items in evaluations.items() if items}
    live = measured.get("live") or {}
    sensor = (live.get("sensor") or [{}])[0]
    latency = {"runtime": cusense.RUNTIME, "device": sensor.get("device"), "device_name": sensor.get("device_name"),
               "boundary": "one batch-one inference of the model, from the input in host memory to the map and the "
                           "logit in host memory; the front end in NumPy on the CPU is outside it",
               "inference_us": live.get("infer_us"), "frontend_us": live.get("frontend_us"),
               "publish_to_decision_us": live.get("publish_to_decision_us"),
               "warm_up_ms": sensor.get("warm_up_ms"), "graded": False}
    present = sum(1 for key in keys if evaluator.by_key(table)[key]["label"])
    sound = set(arms) == set(ARMS)
    for arm, made_arm in arms.items():
        by_window, by_episode = made_arm["presence"]["windows"], made_arm["presence"]["episodes"]
        for side, wanted in ((by_episode, 1), (by_window, per_case)):
            first, other_side = side["target_present"], side["target_absent"]
            sound = sound and first["total"] == wanted * present and other_side["total"] == wanted * (len(keys) - present) \
                and first["detected"] + first["missed"] + first["missing"] == first["total"] \
                and other_side["false_alarm"] + other_side["quiet"] + other_side["missing"] == other_side["total"]
        position = made_arm["position"]
        scored_present = sum(1 for row in made_arm["rows"] if row["label"] and row["window"] is not None)
        sound = sound and position["unit"] == "m" and position["raw"]["eligible"] == scored_present > 0 \
            and all(position["raw"][key] is not None and math.isfinite(position["raw"][key])
                    for key in ("mean_m", "median_m", "p90_m"))
    record = {"schema": evaluator.EVAL_SCHEMA, "graded": False, "role": "held_out",
              "why": "the position error and the presence counts on held-out episodes are recorded beside the "
                     "published figures; this gate grades the three stages, the artifact, its freeze and its parity",
              "set": measured.get("split"), "selection": evaluator.SELECTION_RULE[selection],
              "threshold_rule": "frozen before the first plan of a held-out entry was written",
              "arms": arms, "linear_head": linear, "side_by_side": evaluator.compare_arms(arms, linear),
              "latency": latency}
    write_json(out / EVAL_FILE, record)
    shown = arms.get("o3")
    grader.check(name, "evaluation", sound and shown is not None,
                 f"{EVAL_FILE}: presence by label with every denominator, {present} target-present and "
                 f"{len(keys) - present} target-absent held-out episodes by one prospectively selected window, the last "
                 f"of each episode, and {per_case} windows of each by window; a realization or a window without a scored "
                 "decision counts as missing and never as correct; the position error in metres with its mean, its "
                 "median and its 90th percentile over the scored windows of the target-present episodes, for the raw "
                 "map, the mean of the newest maps and the Kalman filter, each with its count; both input entries and "
                 "the linear head side by side" if sound else
                 "a denominator of the evaluation does not add up, or a position statistic is missing")
    if shown is None:
        return {}
    # ---- The published system, beside what this run measured. Recorded, never graded.
    context = {"bandwidth_mhz": 20, "prbs": cusense.shape_of(config["arms"]["o3"]["spec"])[1] // 12,
               "rx": cusense.shape_of(config["arms"]["o3"]["spec"])[0],
               "dmrs_symbols": cusense.shape_of(config["arms"]["o3"]["spec"])[2], "device_name": sensor.get("device_name"),
               "set_id": (measured.get("split") or {}).get("set_id"),
               "windows": {"training": (measured.get("windows") or {}).get("windows", {}).get("o3", {}).get("training"),
                           "validation": (measured.get("windows") or {}).get("windows", {}).get("o3", {}).get("validation"),
                           "held_out": shown["presence"]["scored_windows"]}}
    reference = evaluator.reference_record(shown, latency, context)
    write_json(out / REFERENCE_FILE, reference)
    published = reference["published"]
    said = " ".join(reference["differences"])
    grader.check(name, "reference", reference["graded"] is False
                 and round(published["position_error_cm"]["mean"]) == expect["published"]["mean_position_error_cm"]
                 and published["live_inference_us"]["value"] == expect["published"]["live_inference_us"]
                 and all(word in said for word in ("simulated scene", "20 MHz", "100 MHz", "GB10", "GH200"))
                 and reference["measured"]["latency"]["runtime"] == cusense.RUNTIME,
                 f"{REFERENCE_FILE} records the measured position error beside the published mean of "
                 f"{expect['published']['mean_position_error_cm']} cm with \"graded\": false, and lists "
                 f"{len(reference['differences'])} differences between the two, the simulated scene, 20 MHz against "
                 "100 MHz and the GB10 against the GH200 among them; the latency of the sensor is recorded with its "
                 f"runtime entry, {cusense.RUNTIME}, and is not compared with the published "
                 f"{expect['published']['live_inference_us']} us: that comparison is made on the native core by S3.6")

    def side(arm):
        made_arm = arms[arm]
        episodes_of, position = made_arm["presence"]["episodes"], made_arm["position"]
        return (f"{made_arm['input']}: {episodes_of['target_present']['detected']} of "
                f"{episodes_of['target_present']['total']} target-present episodes detected and "
                f"{episodes_of['target_absent']['false_alarm']} of {episodes_of['target_absent']['total']} target-absent "
                f"alarmed, area under the curve {text(made_arm['presence']['area_under_curve'], 3)} over "
                f"{made_arm['presence']['scored_windows']} windows; position error mean "
                f"{text(position['raw']['mean_m'], 3)} m, median {text(position['raw']['median_m'], 3)} m, 90th "
                f"percentile {text(position['raw']['p90_m'], 3)} m raw over {position['raw']['eligible']} windows, mean "
                f"{text(position['smoothed']['mean_m'], 3)} m after the mean of the maps and "
                f"{text(position['kalman']['mean_m'], 3)} m after the Kalman filter")

    grader.note(name, "held-out episodes at the frozen thresholds, read with the truth after the decisions were written, "
                      "recorded and not graded: " + "; ".join(side(arm) for arm in ARMS if arm in arms)
                + ("" if linear is None else
                   f"; linear head of Sprint 3.4: {linear['counts']['target_present']['detected']} of "
                   f"{linear['counts']['target_present']['episodes']} detected and "
                   f"{linear['counts']['target_absent']['false_alarm']} of {linear['counts']['target_absent']['episodes']} "
                   f"alarmed, area under the curve {text(linear['area_under_curve'], 3)}")
                + f"; a cell is {shown['position']['cell_m'][0]:.2f} m along y by {shown['position']['cell_m'][1]:.2f} m "
                  f"along x; the "
                  f"published mean is {published['position_error_cm']['mean']} cm in another room, at five times the "
                  "bandwidth, over the air")
    return {"arms": {arm: {"presence": {key: value for key, value in made_arm["presence"].items()},
                           "position": {key: made_arm["position"][key] for key in ("raw", "smoothed", "kalman")}}
                     for arm, made_arm in arms.items()},
            "linear_head": None if linear is None else {key: linear[key] for key in ("counts", "area_under_curve")},
            "eval_sha256": digest(out / EVAL_FILE), "reference_sha256": digest(out / REFERENCE_FILE)}


# ---------------------------------------------------------------- causality
def run_causality(config, out, grader, table, trained, captures, cusense):
    """What may not change a decision, and what a decision may not carry."""
    name = "causality"
    np = cusense.np
    capture = next((item for item in captures if item.role == "held_out" and item.output is not None), None)
    if capture is None or not capture.output.scored:
        grader.check(name, "window", False, "the sensor scored no held-out window")
        return {}
    model = cusense.Model(trained["o3"]["artifact"], "cpu")
    records = capture.archives["o3"].records
    reference = cusense.run_model(records, model)[2]
    scored = [line for line in reference if line["status"] == "ok"]
    waiting = [line for line in reference if line["status"] != "ok"]
    grader.check(name, "no score without a window",
                 scored and all(line["score"] is None and line["decision"] is None and line["input"] is None
                                and line["location"] is None for line in waiting),
                 f"{len(waiting)} answers of a filling window give a decision under their status and no score, no "
                 f"map, no location and no input digest; {len(scored)} complete windows are scored")
    window = scored[0]
    episode = int(window["window"]["episode"], 16)
    own = [record for record in records if record.header["episode"] == episode]
    last = max(index for index, record in enumerate(own) if record.sequence == window["contributing"][-1]["sequence"])

    def answer(decisions, wanted=window):
        found = [line for line in decisions if line["window"] and line["window"]["id"] == wanted["window"]["id"]
                 and line["status"] == "ok"]
        if not found:
            return None
        made = cusense.comparable(found[0])
        del made["line"]
        return made

    base = answer(reference)
    big = np.full(len(own[0].payload) // 2, SENTINEL, dtype="<f2").tobytes()
    future = own[:last + 1] + [changed_record(record, payload=big) for record in own[last + 1:]]
    grader.check(name, "future records", answer(cusense.run_model(future, model)[2]) == base and len(own) > last + 1,
                 f"with the payload of the {len(own) - last - 1} records after the completing one replaced by "
                 f"{SENTINEL:.0e}, the window {window['window']['id']} is decided with the same line: the same input "
                 "digest, the same map digest, the same cell, the same logit and the same decision, bit for bit. The "
                 "model scored before it saw them")
    # ---- A stream reads the fields of INPUT_FIELDS, the validity bits and the payload of a record.
    unread = sorted(field for field, _, _ in obs.FIELDS if field not in cusense.INPUT_FIELDS)
    hidden = [changed_record(record, cell=record.header["cell"] + 7, ue=record.header["ue"] ^ 0x5555,
                             grant=record.header["grant"] ^ 0xFFFF, valid=1, reserved=9) for record in own]
    place = next(index for index, record in enumerate(own) if record.sequence == window["contributing"][0]["sequence"])
    scaled = np.frombuffer(own[place].payload, dtype="<f2").astype(np.float32) * 1.5
    touched = own[:place] + [changed_record(own[place], payload=scaled.astype("<f2").tobytes())] + own[place + 1:]
    other, moved = answer(cusense.run_model(hidden, model)[2]), answer(cusense.run_model(touched, model)[2])
    all_same = [cusense.comparable(line) for line in cusense.run_model(hidden, model)[2]] \
        == [cusense.comparable(line) for line in cusense.run_model(own, model)[2]]
    grader.check(name, "record fields", other == base and all_same and moved is not None
                 and moved["input"]["sha256"] != base["input"]["sha256"],
                 f"with the cell, the UE, the grant, the valid count and the reserved field of every record of the "
                 f"realization changed, every decision is the same line; with the payload of one contributing slot "
                 f"scaled by 1.5 the digest of the input changes: a stream reads {len(cusense.INPUT_FIELDS)} named "
                 f"fields of a header, the validity bits and the payload, and not {', '.join(unread)}; a record "
                 "carries no label, no split and no position, so no field of the truth can reach a window")
    # ---- No output of the sensing side names what only the evaluator knows.
    names = {row["trajectory"] for row in table["episodes"]} | {row["episode_id"] for row in table["episodes"]}
    sensor_dir = capture.directory / "sensor"
    files = [sensor_dir / cusense.DECISIONS_FILE, sensor_dir / cusense.REPORT_FILE,
             sensor_dir / cusense.WINDOWS_DIR / cusense.WINDOWS_FILE,
             sensor_dir / cusense.WINDOWS_DIR / cusense.WINDOWS_REPORT_FILE,
             trained["o3"]["directory"] / "artifact" / cusense.MODEL_FILE,
             trained["o2"]["directory"] / "artifact" / cusense.MODEL_FILE]
    found = []
    for path in files:
        content = path.read_text(encoding="ascii")
        found += [f"{path.name}: {word}" for word in sorted(names) + list(EVALUATOR_KEYS) if word in content]
    grader.check(name, "no label", not found,
                 f"decisions.jsonl, cusense.json and the window output of the sensor and model.json of both artifacts "
                 f"name none of the {len(names)} trajectory and episode identifiers of the set and carry no field of a "
                 "label, a scene, a pair or a case: a decision is joined to the truth by its episode key, after the run, "
                 "by the evaluator" if not found else f"{len(found)} evaluator words in the outputs, first: {found[0]}")
    return {"window": window["window"]["id"], "records": len(own), "unread_fields": unread}


# ---------------------------------------------------------------- refusals
def run_refusals(config, out, grader, found, trained, captures, cusense, presence):
    """Every refusal of the card, by the interface and by the command line."""
    name = "refusals"
    directory = out / "refusals"
    directory.mkdir(parents=True)
    np = cusense.np
    artifact, artifact_dir = trained["o3"]["artifact"], trained["o3"]["directory"] / "artifact"
    model = cusense.Model(artifact, "cpu")
    _, rx, positions = artifact.manifest["input"]["shape"]
    training = cusense.read_samples(out / "samples" / "o3" / "training.jsonl")
    validation = cusense.read_samples(out / "samples" / "o3" / "validation.jsonl")
    other_arm = cusense.read_samples(out / "samples" / "o2" / "validation.jsonl")
    capture = next(item for item in captures if item.role == "held_out" and item.output is not None)
    line = capture.output.windows.ok[0]
    tensors = (training.means[0], training.counts[0])
    o2_record = capture.archives["o2"].records[0]
    spec = config["arms"]["o3"]["spec"]

    def copy_artifact(label, change):
        target = directory / label
        shutil.copytree(artifact_dir, target)
        change(target)
        return target

    def flip(file_name):
        def change(target):
            data = bytearray((target / file_name).read_bytes())
            data[0] ^= 1
            (target / file_name).write_bytes(bytes(data))
        return change

    def rewrite(edit):
        def change(target):
            manifest = json.loads((target / cusense.MODEL_FILE).read_text())
            edit(manifest)
            (target / cusense.MODEL_FILE).write_text(json.dumps(manifest))
        return change

    def not_an_input():
        reason = cusense.refusal(o2_record, spec)
        if reason:
            raise cusense.CuSenseError(reason)

    flipped = copy_artifact("flipped-weight", flip(cusense.WEIGHTS_FILE))
    background = copy_artifact("flipped-template", flip(cusense.TEMPLATE_FILE))
    moved = copy_artifact("moved-threshold", rewrite(lambda manifest: manifest["threshold"].update(
        logit=manifest["threshold"]["logit"] - 1.0)))
    shifted = copy_artifact("moved-mean", rewrite(lambda manifest: manifest["normalizer"].update(
        mean=manifest["normalizer"]["mean"] + 1.0)))
    resplit = copy_artifact("other-split", rewrite(lambda manifest: manifest["normalizer"]["fitted_on"].update(
        split="validation")))
    extra = copy_artifact("extra-file", lambda target: (target / "training.jsonl").write_text("{}\n"))
    other_spec = json.loads(json.dumps(spec))
    other_spec["window"]["min_valid_fraction"] = 0.5
    other_spec["spec_id"] = "another-window"
    absent = next(index for index, row in enumerate(validation.rows) if row["label"] == 0)
    held = dict(training.rows[0], split="held_out")
    crossed = dict(validation.rows[0], scene=training.rows[0]["scene"])
    few = min(len(training.rows), 4)
    zeros = np.zeros((few + 1, rx, positions), dtype=np.float32)
    grid = artifact.grid
    api = (("a batch of two windows", lambda: model.predict(np.zeros((2, rx, positions), dtype=np.float32))),
           ("a window without its batch axis", lambda: model.predict(np.zeros((rx, positions), dtype=np.float32))),
           ("a window of another array", lambda: model.predict(np.zeros((1, rx + 1, positions), dtype=np.float32))),
           ("the axes in another order", lambda: model.predict(np.zeros((1, positions, rx), dtype=np.float32))),
           ("an input as float64", lambda: model.predict(np.zeros((1, rx, positions), dtype=np.float64))),
           ("a value that is not finite", lambda: model.predict(np.full((1, rx, positions), np.nan, dtype=np.float32))),
           ("a window of the other input entry", lambda: model.prepare(other_arm.means[0], other_arm.counts[0])),
           ("a window of another specification",
            lambda: model.decide(dict(line, spec_sha256=cusense.spec_sha256(other_spec)), tensors)),
           ("a window of other array elements",
            lambda: model.decide(dict(line, rx=dict(line["rx"], elements=[element + 4 for element in line["rx"]["elements"]])),
                                 tensors)),
           ("a record of the stage O2 on the stream of the stage O3", not_an_input),
           ("a weight with one bit changed", lambda: cusense.open_artifact(flipped)),
           ("a template with one bit changed", lambda: cusense.open_artifact(background)),
           ("a threshold moved in model.json", lambda: cusense.open_artifact(moved)),
           ("a normalizer moved in model.json", lambda: cusense.open_artifact(shifted)),
           ("statistics that name the validation split", lambda: cusense.open_artifact(resplit)),
           ("a fifth file in the artifact", lambda: cusense.open_artifact(extra)),
           ("another artifact than the frozen one", lambda: cusense.open_artifact(artifact_dir, "0" * 64)),
           ("a validation window in the fit of the template",
            lambda: cusense.fit_template([validation.rows[absent]], validation.means[[absent]],
                                         validation.counts[[absent]])),
           ("a held-out window in the fit of the normalizer",
            lambda: cusense.fit_normalizer([held], training.means[:1], training.counts[:1], artifact.template)),
           ("a validation window in the fit of the weights",
            lambda: cusense.fit_model(training.rows[:few] + [validation.rows[0]], zeros, grid, trained["o3"]["settings"])),
           ("a training window in the calibration of the threshold",
            lambda: presence.calibrate_threshold(training.rows[:2], [0.0, 0.0], 0.05)),
           ("a held-out window among the training samples",
            lambda: cusense.check_split(training.rows + [held], validation.rows)),
           ("a trajectory in the training and in the validation split",
            lambda: cusense.check_split(training.rows, validation.rows[1:] + [crossed])))
    refused, kept = 0, []
    for label, action in api:
        try:
            action()
        except (cusense.CuSenseError, presence.PresenceError):
            refused += 1
        else:
            kept.append(label)
    grader.check(name, "interface", refused == len(api),
                 f"{refused} of {len(api)} are refused by the interface with one error: " + ", ".join(label for label, _ in api)
                 + ("" if not kept else f"; not refused: {', '.join(kept)}"))
    # ---- The same through the command lines: status 2, one line, nothing written.
    write_json(directory / "other-spec.json", other_spec)
    cusense.write_samples(directory / "held-out-row.jsonl", [held] + training.rows[1:few], training.means[:few],
                          training.counts[:few])
    some = min(len(validation.rows), 4)
    cusense.write_samples(directory / "crossed-scene.jsonl", [crossed] + validation.rows[1:some], validation.means[:some],
                          validation.counts[:some])
    module = [sys.executable, "-m", "ocudu_cuda_isac.sensing.cusense"]
    script = [sys.executable, str(ROOT / "scripts/train_cusense.py")]
    archive = str(capture.archives["o3"].directory)
    paths = {role: str(out / "samples" / "o3" / f"{role}.jsonl") for role in ROLES}

    def replay_with(artifact_path, records=archive):
        return module + ["replay", "--archive", records, "--artifact", str(artifact_path), "--device", "cpu"]

    def train_with(first, second, spec_path=out / "specs" / "o3.json"):
        return script + ["--training", str(first), "--validation", str(second), "--spec", str(spec_path), "--grid",
                         str(out / "grid.json"), "--config", str(out / "training-o3.json")]

    commands = (("a weight with one bit changed", replay_with(flipped), "cusense: ", True),
                ("a template with one bit changed", replay_with(background), "cusense: ", True),
                ("a threshold moved in model.json", replay_with(moved), "cusense: ", True),
                ("a fifth file in the artifact", replay_with(extra), "cusense: ", True),
                ("another artifact than the frozen one", module + ["inspect", str(artifact_dir), "--sha256", "0" * 64],
                 "cusense: ", False),
                ("a held-out window among the training samples",
                 train_with(directory / "held-out-row.jsonl", paths["validation"]), "train_cusense: ", None),
                ("a trajectory in the training and in the validation split",
                 train_with(paths["training"], directory / "crossed-scene.jsonl"), "train_cusense: ", None),
                ("the validation samples given as the training samples",
                 train_with(paths["validation"], paths["training"]), "train_cusense: ", None),
                ("another specification than the samples were computed with",
                 train_with(paths["training"], paths["validation"], directory / "other-spec.json"), "train_cusense: ", None))
    by_tool, unrefused = 0, []
    for number, (label, command, prefix, needs_empty) in enumerate(commands):
        target = directory / f"out-{number:02d}"
        if needs_empty is True:
            target.mkdir()
        if needs_empty is not False:
            command = command + ["--output-dir", str(target)]
        result = run_tool(command, directory / f"tool-{number:02d}.log", timeout=1800)
        untouched = not target.exists() or not any(target.iterdir())
        if result.returncode == 2 and untouched and result.stderr.startswith(prefix) and result.stderr.count("\n") == 1:
            by_tool += 1
        else:
            unrefused.append(label)
    grader.check(name, "command line", by_tool == len(commands),
                 f"{by_tool} of {len(commands)} are refused by python -m ocudu_cuda_isac.sensing.cusense and by "
                 "python scripts/train_cusense.py with status 2 and one line, and nothing is written: "
                 + ", ".join(label for label, _, _, _ in commands)
                 + ("" if not unrefused else f"; not refused: {', '.join(unrefused)}"))
    return {"interface": len(api), "refused": refused, "commands": len(commands), "refused_by_tool": by_tool}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="new directory for the run")
    parser.add_argument("--s34-run", required=True, type=Path,
                        help="run directory of the accepted Sprint 3.4 gate: its frozen linear head, its episode set and "
                             "the plans that bind the transmit grids")
    parser.add_argument("--episode-set", required=True, type=Path,
                        help="directory of the channel-episode-set/v1 set that the Sprint 3.4 run was graded on")
    parser.add_argument("--adapter-build", type=Path,
                        help="adapter build against the patched receiver (default: from the configuration)")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.exit(2, f"cusense_frontend: {args.output_dir} already exists. Evidence is never overwritten.\n")
    try:
        from ocudu_cuda_isac.episodes import cusense_frontend as evaluator
        from ocudu_cuda_isac.sensing import cusense, features, presence
    except ImportError as error:
        parser.exit(2, f"cusense_frontend: the branch needs NumPy and PyTorch in this interpreter: {error}\n")
    try:
        config = load_config(args.config, cusense, evaluator)
        receipt = native_adapter.verified_receipt(args.adapter_build or ROOT / config["adapter_build"])
        for needed in TOOLS:
            if needed not in receipt["executables"]:
                raise ConfigError(f"the adapter build holds no {needed}. Rebuild it: make adapter-build-o3")
        if not (args.episode_set / "manifest.json").is_file():
            raise ConfigError(f"{args.episode_set} holds no manifest.json of an episode set. Trace one: make episode-set")
    except (ConfigError, native_adapter.AdapterError) as error:
        parser.exit(2, f"cusense_frontend: {error}\n")
    tools = {name: Path(receipt["executables"][name]["path"]) for name in TOOLS}
    s34_run = args.s34_run.resolve()
    s34_out = s34_run / config["control"]["driver"] if (s34_run / config["control"]["driver"] / "grade.json").is_file() \
        else s34_run
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = args.output_dir.resolve()
    out.mkdir(parents=True)
    write_json(out / "adapter-receipt.json", receipt)
    (out / "specs").mkdir()
    for arm in ARMS:
        write_json(out / "specs" / f"{arm}.json", config["arms"][arm]["spec"])

    grader, measured, captures = Grade(), {}, []

    def guarded(case, action, default=None):
        before = len(grader.results)
        try:
            return action()
        except Exception as error:  # an unexpected fault fails its case and keeps the remaining cases
            (out / f"driver-error-{case}.txt").write_text(traceback.format_exc())
            grader.check(case, "driver", False, f"{type(error).__name__}: {error} (driver-error-{case}.txt)")
            return default
        finally:
            if len(grader.results) == before:
                grader.check(case, "driver", False, "the case produced no check")

    def passed(case):
        return any(item["case"] == case for item in grader.results) \
            and all(item["passed"] for item in grader.results if item["case"] == case)

    def blocked(cases, reason):
        for case in cases:
            grader.check(case, "control", False, reason)

    control, measured["control"] = guarded(
        "control", lambda: run_control(config, grader, s34_out, receipt, features, presence, evaluator), (None, {}))
    table = found = frozen = None
    trained = {}
    if control is None:
        blocked(CASES[1:], "the Sprint 3.4 run or the adapter build does not hold the control this case builds on")
    else:
        table, measured["split"] = guarded(
            "split", lambda: run_split(config, out, grader, control, args.episode_set, cusense, evaluator), (None, {}))
        if table is None or not passed("split"):
            table = None
            blocked(CASES[2:], "the episode set cannot be split and replayed as the case split requires")
    if table is not None:
        per_case = measured["split"]["windows_per_realization"]
        measured["records"] = guarded("records", lambda: run_records(config, out, grader, tools, control, table,
                                                                     evaluator, captures), {})
        if passed("records"):
            found = guarded("windows", lambda: run_windows(config, out, grader, table, captures, per_case, cusense,
                                                           evaluator), {})
            measured["windows"] = {key: value for key, value in (found or {}).items() if key != "rows"}
        if not passed("records") or not passed("windows"):
            found = None
            blocked(CASES[3:] if not passed("records") else CASES[4:],
                    "the training and the validation records did not give the samples this case builds on")
    if found:
        for arm in ARMS:
            case = TRAIN_CASE[arm]
            measured[case], made = guarded(case, lambda arm=arm: train_arm(config, out, grader, arm, found, cusense,
                                                                           presence, evaluator), ({}, None))
            if made is not None and passed(case):
                trained[arm] = made
        if set(trained) != set(ARMS):
            done = [TRAIN_CASE[arm] for arm in ARMS if any(item["case"] == TRAIN_CASE[arm] for item in grader.results)]
            blocked([case for case in CASES[4:] if case not in done],
                    "a model was not frozen: nothing of the held-out group is read")
            trained = {}
    if trained:
        frozen = guarded("freeze", lambda: run_freeze(config, out, grader, table, found, trained, evaluator), None)
        measured["freeze"] = frozen
        if frozen is None or not passed("freeze"):
            frozen = None
            blocked(CASES[7:], "the freeze records do not hold: nothing of the held-out group is read")
    if frozen is not None:
        measured["live"] = guarded("live", lambda: run_live_case(
            config, out, grader, tools, control, table, trained, frozen, per_case, cusense, evaluator, captures), {})
        if passed("live"):
            measured["parity"] = guarded("parity", lambda: run_parity(config, out, grader, trained, captures, cusense,
                                                                      evaluator), {})
            measured["evaluation"] = guarded("evaluation", lambda: run_evaluation(
                config, out, grader, control, table, trained, captures, per_case, measured, cusense, presence, evaluator),
                {})
            measured["causality"] = guarded("causality", lambda: run_causality(config, out, grader, table, trained,
                                                                               captures, cusense), {})
            measured["refusals"] = guarded("refusals", lambda: run_refusals(config, out, grader, found, trained, captures,
                                                                            cusense, presence), {})
        else:
            blocked(CASES[8:], "the sensor left no held-out decisions this case is compared with")

    failed = [item for item in grader.results if not item["passed"]]
    passed_cases = [case for case in CASES if passed(case)]
    whole = not failed and len(passed_cases) == len(CASES)
    artifacts = {arm: trained[arm]["artifact"] for arm in trained}
    grade = {
        "schema": GRADE_SCHEMA, "selector": "cusense_frontend", "run_id": out.name, "cases": len(CASES),
        "cases_passed": len(passed_cases), "cases_failed": len(CASES) - len(passed_cases),
        "checks": len(grader.results), "checks_passed": len(grader.results) - len(failed), "checks_failed": len(failed),
        "judgment": "pass" if whole else "fail", "hardware_gate": "closed" if whole else "open",
        "reason": "every check of every case passed" if not failed else
                  f"{len(failed)} check(s) failed, first: {failed[0]['case']} {failed[0]['check']}",
        "backends": config["backends"],
        "spec_sha256": {arm: cusense.spec_sha256(config["arms"][arm]["spec"]) for arm in ARMS},
        "tolerance": config["tolerance"], "runtime": cusense.RUNTIME,
        "artifact_sha256": {arm: artifacts[arm].sha256 for arm in artifacts},
        "threshold_logit": {arm: artifacts[arm].threshold for arm in artifacts},
        "graded_quality": {"validation_auc_min": config["expect"]["validation_auc_min"],
                           "held_out": "recorded in cusense_eval.json and cusense_reference.json, not graded"},
        "record_definition": f"{obs.DEFINITION:016x}", "results": grader.results}
    write_json(out / "grade.json", grade)
    write_json(out / "comparison.json", measured)
    # What a second reader starts from: the frozen artifact, its freeze record and the output of the sensor.
    live = [capture for capture in captures if capture.role == "held_out" and capture.output is not None]
    if trained and live:
        write_json(out / "recover.json", {
            "artifact": str((trained["o3"]["directory"] / "artifact").relative_to(out)),
            "freeze": str((trained["o3"]["directory"] / "freeze.json").relative_to(out)),
            "sensor": str((live[0].directory / "sensor").relative_to(out)), "window": 0,
            "offline": str((out / "parity" / live[0].name).relative_to(out)),
            "second_input": str((trained["o2"]["directory"] / "artifact").relative_to(out)),
            "evaluation": EVAL_FILE, "reference": REFERENCE_FILE})
    lock = {item["name"]: item["commit"] for item in json.loads((ROOT / "ext/lock.json").read_text())["dependencies"]}
    write_json(out / "manifest.json", {
        "run_id": out.name, "selector": "cusense_frontend",
        "purpose": "Sprint 3.5 integration: the cuSense-like branch on the receiver's own estimate, with the pilot "
                   "comb as the second input: a background template and z-score statistics fitted on the training "
                   "trajectories of an episode set, a plain-block 1D CNN trained on a location map and a presence "
                   "logit, a threshold from the target-absent validation windows, frozen, and then scored on held-out "
                   "records in a second process on the selected device and again offline",
        "command": shlex.join([sys.executable, *sys.argv]), "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": platform.node(), "machine": platform.machine(), "python": platform.python_version(),
        "numpy": cusense.np.__version__, "torch": cusense.torch.__version__,
        "cuda_runtime": cusense.torch.version.cuda, "cuda_available": bool(cusense.torch.cuda.is_available()),
        "cpus": os.cpu_count(), "parent_commit": git_commit(), "locked_sources": lock,
        "inputs": {str(args.config): digest(args.config)}, "driver_sha256": digest(__file__),
        "cusense_sha256": digest(ROOT / "src/ocudu_cuda_isac/sensing/cusense.py"),
        "model_sha256": digest(ROOT / "src/ocudu_cuda_isac/sensing/models/cusense_cnn.py"),
        "evaluator_sha256": digest(ROOT / "src/ocudu_cuda_isac/episodes/cusense_frontend.py"),
        "training_script_sha256": digest(ROOT / "scripts/train_cusense.py"),
        "s34_run": str(s34_run), "control": measured["control"].get("s34"),
        "episode_set": measured.get("split"),
        "specs": {arm: config["arms"][arm]["spec"] for arm in ARMS},
        "entries": {arm: cusense.resolve(config["arms"][arm]["spec"], config["training"]["smoothing"]["entry"])
                    for arm in ARMS},
        "training": config["training"],
        "artifacts": {arm: artifacts[arm].model() for arm in artifacts},
        "order": {"rule": "training and validation records, windows, fits, freeze, and only then the first plan of a "
                          "held-out entry",
                  "captures": [{"name": capture.name, "role": capture.role, "started_monotonic_ns": capture.started_ns,
                                "ended_monotonic_ns": capture.ended_ns} for capture in captures],
                  "frozen_monotonic_ns": frozen["frozen_monotonic_ns"] if frozen else None,
                  "first_test_access_monotonic_ns": (measured.get("live") or {}).get("first_test_access_monotonic_ns")},
        "axes": {"split": "staged_harness", "receiver_stage": "o3_native_estimate, with o2_pilot_ls as the second input",
                 "ue_realization": "replayed channel, no UE", "host_stack": "DeepSig-OCUDU",
                 "sensing_source": config["arms"]["o3"]["spec"]["input"]["source"],
                 "placement": "one host, two processes", "transport": "obs-ring/v1 in host shared memory, one ring per stage",
                 "runtime": cusense.RUNTIME, "clutter": config["arms"]["o3"]["spec"]["steps"]["clutter"],
                 "window": config["arms"]["o3"]["spec"]["steps"]["window"], "model": "cusense_cnn",
                 "detector": "frozen_threshold", "heads": ["location_grid", "presence"],
                 "smoothing": config["training"]["smoothing"]["entry"], "training_domain": "receiver_executed",
                 "gpu_sharing": "the receiver and the sensor share the device by its default time slicing: MPS is S4.1"},
        "runs": [{"name": capture.name, "role": capture.role, "directory": str(capture.directory.relative_to(out)),
                  "status": capture.run.status, "consumer_status": capture.statuses,
                  "entries": [row["episode_id"] for row in capture.rows],
                  "plan_sha256": digest(capture.directory / "plan.json") if (capture.directory / "plan.json").is_file()
                  else None,
                  "rings": {"o2": capture.run.plan["ring"], "o3": capture.run.plan["estimate"]["ring"]}
                  if capture.run.plan else None} for capture in captures],
        "adapter": {"executables": {name: {"path": str(tools[name]), "sha256": receipt["executables"][name]["sha256"]}
                                    for name in TOOLS},
                    "receipt": "adapter-receipt.json", "ocudu_build": receipt["ocudu_build"],
                    "built_with_cuda": receipt["pin"]["cuda"], "o3_route": receipt.get("o3_route")},
        "host_rehearsal": False,
        "environment": {key: os.environ[key] for key in ("VIRTUAL_ENV", "CUDA_VISIBLE_DEVICES") if key in os.environ},
        "judgment": grade["judgment"], "reason": grade["reason"]})
    summary = (f"cusense_frontend: {grade['cases_passed']} of {grade['cases']} cases passed, "
               f"{grade['checks_passed']} of {grade['checks']} checks")
    (out / "grade.txt").write_text("\n".join(grader.lines + [summary]) + "\n", encoding="ascii")
    print(summary)
    print(f"Run artifacts: {out}")
    return 0 if whole else 1


if __name__ == "__main__":
    sys.exit(main())
