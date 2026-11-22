#!/usr/bin/env python3
"""Sprint 3.4 integration driver (selector presence_head). Hardware command, not part of unit discovery.

It trains one presence head on receiver-derived records, freezes it and scores held-out windows
with it in a second process on the selected device. The stimulus is an episode set of Sprint 2.2,
split by trajectory before an episode was traced. The control is the accepted Sprint 3.3 run: its
feature specification, its executables and the reference pair whose entry binds the transmit grid
of every grant. Ten cases, each graded on its own, in this order:

  control    the accepted Sprint 3.3 run, its specification, and the entry of the pair with the
             store of every grant
  split      the episode set verifies, every trajectory is in one group, and every entry can be
             replayed with the grids of the pair
  records    the tapped receiver replay of the training and of the validation entries, in the mode
             timed, with a native consumer that keeps every record of the ring
  features   the causal stream of Sprint 3.3 on those records: one complete window per realization,
             joined to the split by its episode key
  train      scripts/train_presence.py on the two sample files: the normalizer and the weights from
             the training windows, the threshold from the target-absent validation windows, one
             artifact of two files
  freeze     the freeze record names the hashes of the artifact and of both manifests, no manifest
             names a held-out episode, and nothing of the held-out group was replayed yet
  live       the held-out entries are replayed and the presence sensor follows the ring in a second
             process: the frozen artifact, loaded once, on the selected device, one decision per
             answer of the stream
  parity     the same records scored again offline on the CPU: the same features, the same input
             digests, the logit and the probability within the declared tolerance
  causality  a window that is filling has no score, records of the future and fields outside the
             vector leave a decision unchanged, and no output names a label
  refusals   wrong shape, wrong ordering, altered checkpoint, another specification, a scene in two
             splits and samples of the wrong split, by the interface and by the command line

    python tests/integration/presence_head.py --config configs/integration/presence_head.json \\
        --s33-run results/runs/s33-spark-001 --episode-set data/episodes/demo-set-001 \\
        --output-dir results/runs/s34-spark-001

The output directory must be new. The exit status is 0 only when every check of every case passed.
On a host without a CUDA build, --host-rehearsal runs the same cases on a rehearsed Sprint 3.3 run
and scores on the CPU. Such a run is graded as a rehearsal and cannot close the hardware gate.
Every number of the head carries runtime "reference": PyTorch in a Python sensor. No latency of
this driver is a result, and the quality of the head on the held-out episodes is recorded and not
graded: both belong to Sprint 4.
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
from ocudu_cuda_isac.episodes import episode as episodes  # noqa: E402
from ocudu_cuda_isac.episodes import episode_set as sets  # noqa: E402
from ocudu_cuda_isac.episodes import grant as grants  # noqa: E402
from ocudu_cuda_isac.episodes import grid_apply as replay  # noqa: E402
from ocudu_cuda_isac.episodes import replay_pusch as rp  # noqa: E402
from ocudu_cuda_isac.episodes import snapshot_ring as sr  # noqa: E402
from ocudu_cuda_isac.episodes.episode import EpisodeError  # noqa: E402
from ocudu_cuda_isac.episodes.grant import GrantError  # noqa: E402
from ocudu_cuda_isac.episodes.grid_apply import GridApplyError  # noqa: E402
from ocudu_cuda_isac.episodes.replay_pusch import ReplayPuschError  # noqa: E402
from ocudu_cuda_isac.episodes.snapshot_ring import SnapshotRingError  # noqa: E402
from ocudu_cuda_isac.replay import ContractError, load_json  # noqa: E402
from ocudu_cuda_isac.sensing import obs  # noqa: E402
from ocudu_cuda_isac.sensing.obs import ObsError  # noqa: E402
import native_adapter  # noqa: E402

CONFIG_SCHEMA = "integration-config/presence_head/v1"
GRADE_SCHEMA = "integration-grade/presence_head/v1"
CONTROL_GRADE_SCHEMA = "integration-grade/pilot_features/v1"
CASES = ("control", "split", "records", "features", "train", "freeze", "live", "parity", "causality", "refusals")
TOOLS = ("pilot_ls", "obs_tail")
ROLES = ("training", "validation")
NATIVE_TIMEOUT_S = 3600
CONSUMER_GRACE_S = 180
SENTINEL = 1e30
EVALUATOR_KEYS = ('"label"', '"split"', '"scene"', '"truth"', '"target_present"', '"case"', '"pair"')


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
Run, run_is_clean, account_text = ring_driver.Run, ring_driver.run_is_clean, ring_driver.account_text


class Grade(Grader):
    """The Sprint 2.1 grader with this sprint's label."""

    def check(self, case, name, passed, detail):
        self.results.append({"case": case, "check": name, "passed": bool(passed), "detail": detail})
        self._line(("PASS" if passed else "FAIL") + f"  S3.4 {case} {name}: {detail}")
        return bool(passed)

    def note(self, case, text):
        self._line(f"NOTE  S3.4 {case} {text}")


def load_config(path, features, presence):
    try:
        config = load_json(Path(path))
    except (ContractError, OSError) as error:
        raise ConfigError(f"cannot read the configuration: {error}") from None
    needed = {"schema", "selector", "adapter_build", "backends", "control", "set", "receiver", "ring", "spec", "training",
              "sensor", "tolerance"}
    if not isinstance(config, dict) or set(config) != needed or config["schema"] != CONFIG_SCHEMA \
            or config["selector"] != "presence_head":
        raise ConfigError(f"the configuration must be a {CONFIG_SCHEMA} object for the selector presence_head")
    if list(config["backends"]) != list(rp.BACKENDS):
        raise ConfigError('backends must be ["cpu", "cuda"]: every grid is decoded with both')
    if config["control"] != {"driver": "pilot_features", "run": "live"}:
        raise ConfigError('control requires the driver pilot_features and its run "live", whose plan names the pair')
    chosen = config["set"]
    if not isinstance(chosen, dict) or set(chosen) != {"entries_per_run", "expect"} \
            or type(chosen["entries_per_run"]) is not int or chosen["entries_per_run"] < 1 \
            or not isinstance(chosen["expect"], dict) or set(chosen["expect"]) != {"set_id", "episodes"} \
            or not isinstance(chosen["expect"]["episodes"], dict) or set(chosen["expect"]["episodes"]) != set(sets.GROUPS):
        raise ConfigError("set requires entries_per_run and expect with set_id and the episodes of every group")
    receiver = config["receiver"]
    if not isinstance(receiver, dict) or set(receiver) != {"keep_grids"} or type(receiver["keep_grids"]) is not bool:
        raise ConfigError("receiver requires keep_grids, true or false")
    ring = config["ring"]
    if not isinstance(ring, dict) or set(ring) != {"stream", "window_records", "burst_records", "slots", "wait",
                                                   "attach_timeout_ms", "drain_ms"} or ring["wait"] not in sr.WAITS:
        raise ConfigError("ring requires stream, window_records, burst_records, slots, wait, attach_timeout_ms and "
                          "drain_ms")
    sensor = config["sensor"]
    if not isinstance(sensor, dict) or set(sensor) != {"device", "ready_timeout_ms"} \
            or sensor["device"] not in ("cuda", "cpu") or type(sensor["ready_timeout_ms"]) is not int \
            or sensor["ready_timeout_ms"] < 1000:
        raise ConfigError('sensor requires the device "cuda" or "cpu" and ready_timeout_ms of at least 1000')
    try:
        if ring["slots"] != obs.slots_for(ring["window_records"], ring["burst_records"]):
            raise ConfigError(f"ring.slots is {ring['slots']}, and the window and the burst of the configuration need "
                              f"{obs.slots_for(ring['window_records'], ring['burst_records'])}")
        features.check_spec(config["spec"])
        presence.check_settings(config["training"])
    except (ObsError, features.FeaturesError, presence.PresenceError) as error:
        raise ConfigError(str(error)) from None
    tolerance = config["tolerance"]
    if not isinstance(tolerance, dict) or set(tolerance) != {"logit_absolute", "probability_absolute", "reload_absolute"} \
            or any(type(value) is not float or not value > 0 for value in tolerance.values()):
        raise ConfigError("tolerance requires logit_absolute, probability_absolute and reload_absolute, each a "
                          "positive number")
    return config


# ---------------------------------------------------------------- processes
def _finish(tool, consumer, stream):
    """Wait for the tool, then for its consumer. Returns (tool status, consumer status)."""
    try:
        status = tool.wait(timeout=NATIVE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        tool.kill()
        stream.write("TimeoutExpired: the tool was stopped\n")
        status = -1
    try:
        return status, consumer.wait(timeout=CONSUMER_GRACE_S)
    except subprocess.TimeoutExpired:
        consumer.kill()
        return status, -1


def _report(native):
    try:
        return json.loads((native / "native.json").read_text())
    except (OSError, ValueError):
        return None


def run_capture(tools, plan_path, native, directory, archive, wait, capacity, keep_grids=True):
    """Run pilot_ls on one plan with the native consumer that keeps every record. Returns (status, report, consumer status).

    The consumer is started before the tool and retries until the tool has opened its channel.
    native and archive are created empty.
    """
    native.mkdir(parents=True)
    archive.mkdir(parents=True)
    plan = json.loads(Path(plan_path).read_text())
    consumer_command = [str(tools["obs_tail"]), "--socket", str(sr.socket_path(native, plan)), "--output-dir",
                        str(archive), "--wait", wait, "--connect-ms", "60000", "--idle-ms", "60000", "--capacity",
                        str(capacity)]
    command = [str(tools["pilot_ls"]), "--plan", str(plan_path), "--output-dir", str(native)] \
        + ([] if keep_grids else ["--no-grids"])
    consumer = None
    try:
        with (directory / "consumer.log").open("w") as consumer_log, (directory / "native.log").open("w") as stream:
            consumer_log.write(shlex.join(consumer_command) + "\n")
            consumer_log.flush()
            consumer = subprocess.Popen(consumer_command, stdout=consumer_log, stderr=subprocess.STDOUT)
            stream.write(shlex.join(command) + "\n")
            stream.flush()
            tool = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            status, consumer_status = _finish(tool, consumer, stream)
    except OSError as error:
        (directory / "native.log").open("a").write(f"{type(error).__name__}: {error}\n")
        if consumer is not None:
            consumer.kill()
        return -1, None, -1
    return status, _report(native), consumer_status


def run_live(tools, plan_path, native, directory, sensor_dir, spec_path, artifact_dir, device, wait, ready_timeout_s,
             keep_grids=True):
    """Run pilot_ls on one plan with the presence sensor as its consumer. Returns (status, report, sensor status).

    The sensor is started first. It loads the artifact, creates the context of its device, runs
    its steps once and then writes sensor-ready.json. Only then is the tool started, so that the
    receiver never waits for a sensor that is loading. native and sensor_dir are created empty.
    """
    native.mkdir(parents=True)
    sensor_dir.mkdir(parents=True)
    plan = json.loads(Path(plan_path).read_text())
    ready = directory / "sensor-ready.json"
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    sensor_command = [sys.executable, "-m", "ocudu_cuda_isac.sensing.presence", "follow", "--socket",
                      str(sr.socket_path(native, plan)), "--spec", str(spec_path), "--artifact", str(artifact_dir),
                      "--output-dir", str(sensor_dir), "--wait", wait, "--connect-ms", "60000", "--idle-ms", "60000",
                      "--ready-file", str(ready)] + ([] if device is None else ["--device", device])
    command = [str(tools["pilot_ls"]), "--plan", str(plan_path), "--output-dir", str(native)] \
        + ([] if keep_grids else ["--no-grids"])
    sensor = None
    try:
        with (directory / "sensor.log").open("w") as sensor_log, (directory / "native.log").open("w") as stream:
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
                return -1, None, sensor.wait(timeout=CONSUMER_GRACE_S)
            stream.flush()
            tool = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            status, sensor_status = _finish(tool, sensor, stream)
    except OSError as error:
        (directory / "native.log").open("a").write(f"{type(error).__name__}: {error}\n")
        if sensor is not None:
            sensor.kill()
        return -1, None, -1
    return status, _report(native), sensor_status


def run_tool(command, log, timeout=1800):
    """Run a Python tool of the project with the source tree on its path. Returns the completed process."""
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    result = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=timeout, check=False)
    Path(log).write_text(shlex.join([str(word) for word in command]) + "\n" + result.stdout + result.stderr)
    return result


# ---------------------------------------------------------------- helpers
class Control:
    """The accepted Sprint 3.3 run: what it was measured with and the pair it replayed."""

    def __init__(self):
        self.manifest = self.plan = self.entry = self.bindings = self.observer = None


class Capture:
    """One receiver replay of some entries of the set: its plan, its run and what its consumer kept."""

    def __init__(self, role, name, rows, directory):
        self.role, self.name, self.rows, self.directory = role, name, rows, directory
        self.run = Run(name, directory, None)
        self.channel = self.archive = self.problem = self.sensor_status = self.output = None
        self.keys, self.started_ns, self.ended_ns = [], None, None

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


def plan_capture(config, control, evaluator, capture, wait=None):
    """Write the three plans of a capture into its directory."""
    settings = config["ring"]
    capture.directory.mkdir(parents=True)
    ring = sr.ring_spec(settings["window_records"], settings["burst_records"], wait or settings["wait"], "required",
                        settings["attach_timeout_ms"], settings["drain_ms"], settings["stream"])
    capture.channel, capture.run.plan = evaluator.write_plans(
        f"s34-{capture.name}", capture.rows, control.bindings, capture.directory, ring,
        epoch=control.observer["epoch"], source=control.observer["source"])
    capture.keys = evaluator.keys_of(capture.channel)


def verify_capture(capture):
    """Open what the tool wrote. Sets the problem of the capture when it does not verify."""
    run = capture.run
    if run.status == 0 and run.report:
        try:
            run.output = sr.open_output(run.native)
        except SnapshotRingError as error:
            run.problem = f"the native output does not verify: {error}"


def decoded(captures):
    """(decodes that gave the original bytes, radio records) over some captures."""
    records = [record for capture in captures for record in capture.run.output.records]
    return sum(record["outcome"] == "decoded" for record in records), len(records)


def path_check(grader, name, captures, rehearsal, check="executed path"):
    """The CUDA processor ran its resident path and the host processor the host path, in every run."""
    records = [record for capture in captures for record in capture.run.output.records]
    cuda = [record for record in records if record["backend"] == "cuda"]
    host = [record for record in records if record["backend"] == "cpu"]
    resident = sum((record["path"] or {}).get("executed") == "resident" for record in cuda)
    on_host = sum((record["path"] or {}).get("executed") == "host" for record in host)
    if rehearsal:
        return grader.check(name, check, on_host == len(host) and len(host) > 0,
                            f"host rehearsal: {on_host} of {len(host)} decodes of the host processor ran the host "
                            f"path; the {len(cuda)} decodes requested from the CUDA processor are not graded on a "
                            "build without CUDA")
    built = all(capture.run.report.get("built_with_cuda") is True for capture in captures)
    return grader.check(name, check, built and cuda and resident == len(cuda) and on_host == len(host),
                        f"{resident} of {len(cuda)} decodes requested from the CUDA processor ran its resident path "
                        f"and {on_host} of {len(host)} decodes of the host processor ran the host path; the adapter "
                        "is built with CUDA" if built else
                        "the adapter is not built with CUDA: the CUDA processor cannot execute")


def account_check(grader, name, captures, archives, check="account"):
    """Every observation of every run was published, read and kept, and each kept record is that of its line."""
    states, problems, kept = [], [], 0
    for capture, archive in zip(captures, archives):
        found = sr.account(capture.run.output)
        states.append(found)
        pairs, wrong = sr.join_archive(capture.run.output, archive)
        problems += [f"{capture.name}: {problem}" for problem in wrong]
        kept += len(pairs)
        if capture.run.output.mode != "timed":
            problems.append(f"{capture.name}: the run is not in the mode timed")
    total = {key: sum(state[key] for state in states) for key in ("eligible", "published", "dropped", "released",
                                                                 "pending")}
    whole = all(state["balanced"] for state in states) and total["dropped"] == 0 and total["pending"] == 0 \
        and not problems and kept == total["published"] == sum(capture.opportunities() for capture in captures)
    grader.check(name, check, whole,
                 f"{total['eligible']} observations in {len(captures)} runs: {total['published']} published, "
                 f"{total['dropped']} dropped, {total['released']} released, {total['pending']} pending; the consumers "
                 f"kept {kept} records, each the record of its account line: the epoch, the sequence, the opportunity "
                 "key and the ticket of its grid; a consumer that keeps the pace of the receiver loses no observation"
                 if not problems else "; ".join(problems[:3]))
    return whole, total


# ---------------------------------------------------------------- control
def run_control(config, grader, s33_out, rehearsal, features, evaluator):
    """The accepted Sprint 3.3 run, its specification and the entry of the pair with its grant stores."""
    name, control, measured = "control", Control(), {}
    try:
        grade = load_json(s33_out / "grade.json")
        manifest = load_json(s33_out / "manifest.json")
    except (ContractError, OSError) as error:
        grader.check(name, "accepted run", False, f"{s33_out} holds no grade and manifest of the Sprint 3.3 driver: {error}")
        return None, measured
    wanted = "rehearsal" if rehearsal else "pass"
    accepted = isinstance(grade, dict) and grade.get("schema") == CONTROL_GRADE_SCHEMA \
        and grade.get("judgment") == wanted and grade.get("cases_passed") == grade.get("cases") \
        and grade.get("checks_failed") == 0 and (rehearsal or grade.get("hardware_gate") == "closed")
    control.manifest = manifest
    measured["s33"] = {"run_id": manifest.get("run_id"), "judgment": grade.get("judgment"),
                       "hardware_gate": grade.get("hardware_gate"), "parent_commit": manifest.get("parent_commit"),
                       "spec_sha256": grade.get("spec_sha256"), "record_definition": grade.get("record_definition"),
                       "s32_run": manifest.get("s32_run"), "adapter": (manifest.get("adapter") or {}).get("executables")}
    grader.check(name, "accepted run", accepted,
                 f"the Sprint 3.3 driver graded {s33_out.name} as {grade.get('judgment')}: {grade.get('cases_passed')} of "
                 f"{grade.get('cases')} cases, {grade.get('checks_failed')} failed checks, hardware gate "
                 f"{str(grade.get('hardware_gate')).split(':')[0]}, at commit {str(manifest.get('parent_commit'))[:7]}"
                 + ("" if accepted else f"; this gate needs the judgment {wanted}"
                    + ("" if rehearsal else " with the hardware gate closed")))
    mine = features.spec_sha256(config["spec"])
    same = grade.get("spec_sha256") == mine and manifest.get("spec") == config["spec"] \
        and grade.get("record_definition") == f"{obs.DEFINITION:016x}"
    grader.check(name, "specification", same,
                 f"the head is trained on the vector of the specification {config['spec']['spec_id']} {mine[:16]}, the "
                 f"one that run was graded with, on records of the definition {obs.DEFINITION:016x}" if same else
                 f"that run was graded with the specification {grade.get('spec_sha256')} on the record definition "
                 f"{grade.get('record_definition')}, and this configuration has {mine} and {obs.DEFINITION:016x}")
    # The plan of the live run of that gate names the replay of the pair, and the pair its grant stores.
    try:
        listed = {item["name"]: item for item in manifest.get("runs", []) if isinstance(item, dict)}
        item = listed.get(config["control"]["run"])
        if item is None or item.get("status") != 0:
            raise SnapshotRingError(f"the Sprint 3.3 run holds no finished run named {config['control']['run']}")
        directory = s33_out / item["directory"]
        if digest(directory / "plan.json") != item.get("plan_sha256"):
            raise SnapshotRingError("the plan of that run does not have the hash its manifest names")
        control.plan = sr.load_plan(directory / "plan.json")
        replay_path = Path(control.plan["replay_plan"]["path"])
        if digest(replay_path) != control.plan["replay_plan"]["sha256"]:
            raise SnapshotRingError("the replay plan of the Sprint 2.4 run does not have the hash that plan binds")
        channel = replay.load_plan(rp.load_plan(replay_path)["channel_plan"]["path"])
        entries = {str(Path(source["h"]["path"]).parent) for source in channel["sources"]}
        if len(entries) != 1:
            raise SnapshotRingError("the channel plan of the pair does not replay one episode entry")
        control.entry = episodes.open_episode(Path(entries.pop()))
        control.bindings = evaluator.bindings_of(control.entry)
        for grant_id, binding in control.bindings.items():
            store = grants.open_store(binding["store"])
            if store.manifest["files"].get("x.npy", {}).get("sha256") != binding["x_sha256"]:
                raise GrantError(f"the store of grant {grant_id} holds another transmit grid than the entry binds")
        control.observer = control.plan["observer"]
    except (SnapshotRingError, ReplayPuschError, GridApplyError, EpisodeError, GrantError, OSError, KeyError,
            evaluator.PresenceHeadError) as error:
        grader.check(name, "reference entry", False, f"{error}")
        return None, measured
    measured["entry"] = {"episode_id": control.entry.episode_id, "directory": str(control.entry.directory),
                         "grants": {grant_id: binding["x_sha256"] for grant_id, binding in control.bindings.items()}}
    grader.check(name, "reference entry", True,
                 f"the plan of the run {config['control']['run']} of that gate replays the entry "
                 f"{control.entry.episode_id}, which verifies and binds the transmit grid of {len(control.bindings)} "
                 f"grants: {', '.join(sorted(control.bindings))}; every store entry verifies and holds the grid of that "
                 f"hash; observer epoch {control.observer['epoch']}, source {control.observer['source']}")
    return (control if accepted and same else None), measured


# ---------------------------------------------------------------- split
def run_split(config, out, grader, control, set_dir, evaluator):
    """The episode set, its split by trajectory, and whether its entries can be replayed with the grids of the pair."""
    name, expect, spec = "split", config["set"]["expect"], config["spec"]
    try:
        episode_set = sets.open_set(set_dir)
        table = evaluator.split_table(episode_set)
    except (EpisodeError, evaluator.PresenceHeadError, GridApplyError) as error:
        grader.check(name, "episode set", False, f"{set_dir} is no episode set that verifies: {error}")
        return None, {}
    manifest = episode_set.manifest
    counts = table["counts"]
    grader.check(name, "episode set", manifest["set_id"] == expect["set_id"] and counts == expect["episodes"],
                 f"the set {manifest['set_id']} (status {manifest['status']}) verifies with every episode in it: "
                 f"{counts['training']} training, {counts['validation']} validation and {counts['held_out']} held-out "
                 f"pairs, {manifest['counts']['opportunities']} opportunities per case in all; the configuration expects "
                 f"{expect['set_id']} with {expect['episodes']['training']}, {expect['episodes']['validation']} and "
                 f"{expect['episodes']['held_out']}")
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
    # Every entry must give at least one window, on the grants whose grids the pair binds.
    window, hop = spec["window"]["periods"], spec["window"]["hop_periods"]
    short = [row["episode_id"] for row in table["episodes"] if row["periods"] < window]
    foreign = []
    for entry in episode_set.episodes():
        for grant_id, described in entry.manifest["grants"].items():
            if control.bindings.get(grant_id, {}).get("record_sha256") != described["record_sha256"]:
                foreign.append(f"{entry.episode_id} {grant_id}")
    per_case = {(row["periods"] - window) // hop + 1 for row in table["episodes"] if row["periods"] >= window}
    gains = sorted({row["grid_gain"] for row in table["episodes"]})
    grader.check(name, "replayable", not short and not foreign and len(per_case) == 1,
                 f"every entry lasts at least the {window} periods of one window and gives "
                 f"{'/'.join(str(value) for value in sorted(per_case))} per realization at a hop of {hop}; every grant "
                 f"of every entry has the record hash of the grant of that name in the pair, so the entries are "
                 f"replayed with the transmit grids that the pair binds: an entry of a set binds none itself; "
                 f"{len(gains)} grid gain(s) in the set" if not short and not foreign else
                 f"{len(short)} entries are shorter than a window (first: {short[:1]}) and {len(foreign)} grants are "
                 f"not those of the pair (first: {foreign[:1]})")
    if short or foreign or len(per_case) != 1 or not disjoint:
        return None, {"counts": counts}
    write_json(out / "split.json", table)
    measured = {"set_id": manifest["set_id"], "status": manifest["status"], "directory": str(Path(set_dir).resolve()),
                "config_sha256": manifest["config_sha256"], "assignment_sha256": manifest["assignment_sha256"],
                "counts": counts, "windows_per_realization": per_case.pop(), "grid_gains": gains,
                "split_sha256": digest(out / "split.json")}
    return table, measured


# ---------------------------------------------------------------- records
def run_records(config, out, grader, tools, control, table, evaluator, rehearsal, captures):
    """The tapped receiver replay of the training and of the validation entries, each run with a consumer that keeps."""
    name, backends, size = "records", config["backends"], config["set"]["entries_per_run"]
    made = []
    for role in ROLES:
        for index, rows in enumerate(evaluator.chunks(table, role, size)):
            capture = Capture(role, f"{role}-{index:02d}", rows, out / "records" / f"{role}-{index:02d}")
            captures.append(capture)
            made.append(capture)
            plan_capture(config, control, evaluator, capture)
            run = capture.run
            capture.started_ns = time.monotonic_ns()
            run.status, run.report, capture.sensor_status = run_capture(
                tools, capture.directory / evaluator.PLAN, run.native, capture.directory, capture.directory / "consumer",
                config["ring"]["wait"], capture.opportunities() + 64, config["receiver"]["keep_grids"])
            capture.ended_ns = time.monotonic_ns()
            verify_capture(capture)
    bad = [capture for capture in made if not (run_is_clean(capture.run, backends) and capture.run.output.mode == "timed")]
    opportunities = sum(capture.opportunities() for capture in made)
    grader.check(name, "native runs", not bad and made,
                 f"{len(made)} runs of pilot_ls in the mode timed on a ring of {config['ring']['slots']} slots: "
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
                 f"{' and '.join(backends)}: the receiver decodes every grid of the set while its estimate is exported")
    path_check(grader, name, made, rehearsal)
    archives = []
    for capture in made:
        try:
            if capture.sensor_status != 0:
                raise ObsError(f"the consumer exited with status {capture.sensor_status}: "
                               f"{last_line(capture.directory / 'consumer.log')}")
            capture.archive = obs.open_archive(capture.directory / "consumer")
        except ObsError as error:
            grader.check(name, "account", False, f"{capture.name}: the archive of the consumer does not verify: {error}")
            return {}
        archives.append(capture.archive)
    _, total = account_check(grader, name, made, archives)
    periods = [(capture.ended_ns - capture.started_ns) / 1e9 for capture in made]
    grader.note(name, f"recorded, not graded: {sum(periods):.1f} s of wall time for {len(made)} runs, tool start and "
                      f"verification of the stimulus included; ring depth at most "
                      f"{max(capture.run.output.ring['counts']['max_depth'] for capture in made)} of "
                      f"{config['ring']['slots']} slots")
    return {"runs": len(made), "opportunities": opportunities, "decodes": planned, "account": total,
            "wall_seconds": sum(periods)}


# ---------------------------------------------------------------- features
def run_features(config, out, grader, table, captures, per_case, features, presence, evaluator):
    """The causal stream of Sprint 3.3 on the kept records, and the join of its windows to the split."""
    name, spec = "features", config["spec"]
    rows, accounts, counts = {role: [] for role in ROLES}, {role: [] for role in ROLES}, {}
    for capture in captures:
        if capture.role not in ROLES or capture.archive is None:
            continue
        directory = out / "features" / capture.name
        directory.mkdir(parents=True)
        try:
            features.replay(capture.archive.directory, spec, directory)
            output = features.open_output(directory)
            made, account = evaluator.samples(output, table, capture.role, capture.keys)
        except (features.FeaturesError, ObsError, evaluator.PresenceHeadError) as error:
            grader.check(name, "windows", False, f"{capture.name}: {error}")
            return {}
        capture.output = output
        rows[capture.role] += made
        accounts[capture.role].append(account)
        for key, value in output.report["counts"].items():
            counts[key] = counts.get(key, 0) + value
    realizations = {role: sum(account["realizations"] for account in accounts[role]) for role in ROLES}
    missing = [dict(item, run=role) for role in ROLES for account in accounts[role] for item in account["missing"]]
    whole = all(len(rows[role]) == per_case * realizations[role] and realizations[role] for role in ROLES) \
        and not missing and not counts.get("gap") and not counts.get("invalid") and not counts.get("rejected")
    grader.check(name, "windows", whole,
                 f"{counts.get('records', 0)} records of the rings in sequence order through the stream of the "
                 f"specification {spec['spec_id']}: " + ", ".join(
                     f"{len(rows[role])} complete windows for {realizations[role]} {role} realizations" for role in ROLES)
                 + f", {per_case} per realization; {counts.get('insufficient_history', 0)} answers of a filling window, "
                 f"{counts.get('passed', 0)} records outside the sequence, {counts.get('gap', 0)} gaps, "
                 f"{counts.get('invalid', 0)} invalid, {counts.get('rejected', 0)} rejected; every output verifies with "
                 "its reader" + ("" if not missing else f"; without a window: {len(missing)} realizations, first "
                                                        f"{missing[0]['episode']} ({missing[0]['last_answer']})"))
    if not whole:
        return {}
    paths, balance = {}, {}
    for role in ROLES:
        paths[role] = presence.write_samples(out / "samples" / f"{role}.jsonl", rows[role])
        balance[role] = (sum(row["label"] for row in rows[role]), sum(1 - row["label"] for row in rows[role]))
    try:
        facts = presence.check_split(rows["training"], rows["validation"])
    except presence.PresenceError as error:
        grader.check(name, "samples", False, f"{error}")
        return {}
    vector = facts["vector"]
    grader.check(name, "samples", all(balance[role][0] == balance[role][1] > 0 for role in ROLES),
                 f"each complete window is joined to the split table by the episode key of its records, after the "
                 f"features were written: {balance['training'][0]} target-present and {balance['training'][1]} "
                 f"target-absent training windows from {len(facts['scenes']['training'])} trajectories, "
                 f"{balance['validation'][0]} and {balance['validation'][1]} validation windows from "
                 f"{len(facts['scenes']['validation'])}; no window, key, pair or trajectory is in both files; a vector "
                 f"holds {len(vector)} named values (samples/training.jsonl, samples/validation.jsonl)")
    return {"vector": vector, "windows": {role: len(rows[role]) for role in ROLES}, "counts": counts,
            "samples": {role: {"file": str(paths[role].relative_to(out)), "sha256": digest(paths[role])} for role in ROLES},
            "rows": rows}


# ---------------------------------------------------------------- train
def run_train(config, out, grader, found, presence):
    """The training script on the two sample files, and what it froze, checked here a second time."""
    name, settings, tolerance = "train", config["training"], config["tolerance"]
    directory = out / "train"
    write_json(out / "training.json", settings)
    command = [sys.executable, str(ROOT / "scripts/train_presence.py"), "--training", str(out / "samples/training.jsonl"),
               "--validation", str(out / "samples/validation.jsonl"), "--spec", str(out / "spec.json"), "--config",
               str(out / "training.json"), "--output-dir", str(directory)]
    result = run_tool(command, out / "train.log")
    try:
        if result.returncode != 0:
            raise presence.PresenceError(f"exit status {result.returncode}: {last_line(out / 'train.log')}")
        freeze = load_json(directory / "freeze.json")
        fit = load_json(directory / "fit.json")
        artifact = presence.open_artifact(directory / "artifact", freeze["artifact"]["sha256"])
    except (presence.PresenceError, ContractError, OSError, KeyError) as error:
        grader.check(name, "script", False, f"scripts/train_presence.py left no frozen head: {error}")
        return {}, None
    manifest = artifact.manifest
    rows, vector = found["rows"], found["vector"]
    columns = presence.select_columns(vector, settings["columns"])
    grader.check(name, "script", manifest["input"]["columns"] == columns and manifest["input"]["vector"] == vector
                 and manifest["head_id"] == settings["head_id"],
                 f"python scripts/train_presence.py read the two sample files and no other input and froze the head "
                 f"{manifest['head_id']} {artifact.sha256[:16]}; the head is bound to names: it reads {len(columns)} of "
                 f"the {len(vector)} values of a vector, {', '.join(columns)}, and not "
                 f"{', '.join(value for value in vector if value not in columns)}")
    files = sorted(path.name for path in (directory / "artifact").iterdir())
    weights = artifact.weights
    words = [key for key in EVALUATOR_KEYS if key in (directory / "artifact" / presence.HEAD_FILE).read_text()]
    grader.check(name, "artifact", files == [presence.HEAD_FILE, presence.WEIGHTS_FILE] and len(weights) == 4 * (len(columns) + 1)
                 and not words,
                 f"the deployment artifact is {' and '.join(files)}: {len(weights)} bytes of weights, one float32 per "
                 f"column and the bias, and the input contract, the normalizer and the threshold with their hashes; it "
                 f"holds no sample, no label, no split and no file name of the training run, and it verifies with its "
                 f"reader: weights {manifest['weights']['sha256'][:16]}, normalizer {manifest['normalizer']['sha256'][:16]}")
    # ---- The normalizer, computed here again from the training windows alone.
    np = presence.np
    places = [vector.index(column) for column in columns]

    def columns_of(samples):
        return np.array([[row["values"][place] for place in places] for row in samples], dtype=np.float32).astype(np.float64)

    training = columns_of(rows["training"])
    both = np.concatenate([training, columns_of(rows["validation"])])
    mean, scale = training.mean(axis=0), training.std(axis=0)
    stored_mean, stored_scale = np.array(manifest["normalizer"]["mean"]), np.array(manifest["normalizer"]["scale"])
    close = bool(np.allclose(stored_mean, mean, rtol=1e-12, atol=1e-12)
                 and np.allclose(stored_scale, np.where(scale > 1e-9 * np.maximum(1.0, np.abs(mean)), scale, 1.0),
                                 rtol=1e-12, atol=1e-12))
    other = float(np.abs(stored_mean - both.mean(axis=0)).max())
    grader.check(name, "normalizer", close,
                 f"mean and deviation of the {len(columns)} columns, computed here from the {len(training)} training "
                 f"windows alone, are those of the artifact; with the {len(both) - len(training)} validation windows "
                 f"added the mean would differ by up to {other:.3e}: the validation split did not enter the fit")
    # ---- The threshold, from the target-absent validation windows, by its rule.
    head = presence.Head(artifact, "cpu")
    logits = {role: [head.score(row["names"], row["values"])["logit"] for row in rows[role]] for role in ROLES}
    again = presence.calibrate_threshold(rows["validation"], logits["validation"],
                                         settings["threshold"]["target_false_alarm"],
                                         settings["threshold"]["min_negatives"])
    stored = manifest["threshold"]
    grader.check(name, "threshold", again == stored and stored["exceeding"] <= stored["allowed"],
                 f"the frozen threshold is the logit {stored['logit']:.6f} (probability {stored['probability']:.4f}): "
                 f"with a false-alarm target of {stored['target_false_alarm']} the rule allows {stored['allowed']} of the "
                 f"{stored['negatives']} target-absent validation windows above it, and {stored['exceeding']} are; "
                 "recomputed here from the artifact read back, it is the same record; no target-present window and no "
                 "training window enters it")
    history = fit["history"]
    reload = freeze["reload"]["largest_logit_difference"]
    second = presence.Head(presence.open_artifact(directory / "artifact"), "cpu")
    same = all(second.score(row["names"], row["values"])["logit"] == logit
               for row, logit in zip(rows["validation"], logits["validation"]))
    grader.check(name, "fit", history["final_loss"] < history["initial_loss"] and reload <= tolerance["reload_absolute"]
                 and same and math.isclose(history["initial_loss"], math.log(2.0), rel_tol=1e-9),
                 f"the loss falls from {history['initial_loss']:.6f}, ln 2 at zero weights, to {history['final_loss']:.6f} "
                 f"in {history['evaluations']} evaluations of L-BFGS in float64 on the CPU; the artifact read back scores "
                 f"every training and validation window within {reload:.1e} of the fit (tolerance "
                 f"{tolerance['reload_absolute']:.0e}), and a second load gives the same {len(logits['validation'])} "
                 "validation logits bit for bit")
    rates = fit["at_the_frozen_threshold"]
    grader.note(name, "recorded, not graded, on the samples of the fit: "
                + "; ".join(f"{role}: {rates[role]['detected']} of {rates[role]['target_present']} target-present "
                            f"windows above the threshold, {rates[role]['false_alarms']} of "
                            f"{rates[role]['target_absent']} target-absent, area under the curve "
                            f"{text(rates[role]['area_under_curve'], 3)}" for role in ROLES)
                + "; weights by column: " + ", ".join(f"{column} {value:+.3f}" for column, value in fit["weights"].items())
                + f", bias {fit['bias']:+.3f}")
    measured = {"artifact_sha256": artifact.sha256, "weights_sha256": manifest["weights"]["sha256"],
                "normalizer_sha256": manifest["normalizer"]["sha256"], "columns": columns, "threshold": stored,
                "fit": {"initial_loss": history["initial_loss"], "final_loss": history["final_loss"],
                        "evaluations": history["evaluations"]}, "reload": reload, "rates": rates,
                "weights": fit["weights"], "bias": fit["bias"]}
    return measured, {"freeze": freeze, "artifact": artifact, "directory": directory}


# ---------------------------------------------------------------- freeze
def run_freeze(config, out, grader, table, found, trained, evaluator):
    """What the freeze record binds, and that nothing of the held-out group was read before it."""
    name, freeze, directory = "freeze", trained["freeze"], trained["directory"]
    artifact = trained["artifact"]
    manifests = {role: load_json(directory / f"{role}_manifest.json") for role in ROLES}
    files = {path.name: digest(path) for path in (directory / "artifact").iterdir()}
    chain = freeze["artifact"]["sha256"] == artifact.sha256 \
        and freeze["artifact"]["weights_sha256"] == files.get("weights.f32") \
        and {name_: entry["sha256"] for name_, entry in freeze["artifact"]["files"].items()} == files \
        and all(freeze["manifests"][role]["sha256"] == digest(directory / f"{role}_manifest.json")
                and manifests[role]["samples"]["sha256"] == found["samples"][role]["sha256"]
                == freeze["manifests"][role]["samples_sha256"] for role in ROLES)
    grader.check(name, "hash chain", chain,
                 f"freeze.json names the artifact {artifact.sha256[:16]}, its two files by SHA-256 and both manifests; "
                 f"each manifest names the hash of the sample file this driver wrote; all were hashed here again: "
                 f"weights {files.get('weights.f32', 'missing')[:16]}, training manifest "
                 f"{freeze['manifests']['training']['sha256'][:16]}, validation manifest "
                 f"{freeze['manifests']['validation']['sha256'][:16]}")
    lookup = evaluator.by_key(table)
    held = {key for key, known in lookup.items() if known["split"] == "held_out"}
    wrong = [f"{role} {key}" for role in ROLES for key in manifests[role]["episode_keys"]
             if lookup.get(key, {}).get("split") != role]
    leaked = [key for role in ROLES for key in manifests[role]["episode_keys"] if key in held]
    leaked += [window for role in ROLES for window in manifests[role]["window_ids"] if window.split("/")[1] in held]
    untouched = not (out / "live").exists() and not (out / "parity").exists()
    grader.check(name, "no test access", not wrong and not leaked and untouched and len(held) > 0,
                 f"the manifests list {len(manifests['training']['episode_keys'])} training and "
                 f"{len(manifests['validation']['episode_keys'])} validation episode keys, each of its own split; none "
                 f"of the {len(held)} held-out keys and no window of one is in either; when the head was frozen this "
                 "run held no plan, no record and no feature of a held-out entry: the files of the set were verified by "
                 "their hashes, and no estimate of a held-out grid existed" if not wrong and not leaked else
                 f"{len(wrong)} keys are in the wrong manifest and {len(leaked)} held-out identifiers are in a manifest")
    return {"frozen_utc": freeze["frozen_utc"], "frozen_monotonic_ns": freeze["frozen_monotonic_ns"],
            "freeze_sha256": digest(directory / "freeze.json"), "held_out_keys": len(held)}


# ---------------------------------------------------------------- live
def run_live_case(config, out, grader, tools, control, table, trained, frozen, per_case, features, presence, evaluator,
                  rehearsal, captures):
    """The held-out entries, replayed for the first time, with the presence sensor on the ring in a second process."""
    name, backends, settings = "live", config["backends"], config["ring"]
    freeze, directory = trained["freeze"], trained["directory"]
    accessed = time.monotonic_ns()
    intact = digest(directory / "freeze.json") == frozen["freeze_sha256"] \
        and presence.open_artifact(directory / "artifact").sha256 == freeze["artifact"]["sha256"]
    grader.check(name, "after the freeze", intact and freeze["frozen_monotonic_ns"] < accessed,
                 f"the first plan of a held-out entry is written {(accessed - freeze['frozen_monotonic_ns']) / 1e9:.1f} s "
                 f"after the freeze record of {freeze['frozen_utc']}, on the monotonic clock of this host; freeze.json "
                 f"and the artifact have the hashes of the case freeze")
    device = "cpu" if rehearsal else config["sensor"]["device"]
    spec_path = out / "spec.json"
    made = []
    for index, rows in enumerate(evaluator.chunks(table, "held_out", config["set"]["entries_per_run"])):
        capture = Capture("held_out", f"held-out-{index:02d}", rows, out / "live" / f"held-out-{index:02d}")
        captures.append(capture)
        made.append(capture)
        plan_capture(config, control, evaluator, capture)
        run = capture.run
        capture.started_ns = time.monotonic_ns()
        run.status, run.report, capture.sensor_status = run_live(
            tools, capture.directory / evaluator.PLAN, run.native, capture.directory, capture.directory / "sensor",
            spec_path, directory / "artifact", device, settings["wait"], config["sensor"]["ready_timeout_ms"] / 1000,
            config["receiver"]["keep_grids"])
        capture.ended_ns = time.monotonic_ns()
        verify_capture(capture)
    bad = [capture for capture in made if not (run_is_clean(capture.run, backends) and capture.run.output.mode == "timed")]
    opportunities = sum(capture.opportunities() for capture in made)
    grader.check(name, "native run", not bad and made,
                 f"{len(made)} run(s) of pilot_ls in the mode timed for the {sum(len(capture.rows) for capture in made)} "
                 f"held-out entries, {opportunities} opportunities, with the presence sensor attached in the mode "
                 f"{settings['wait']}; one radio record per planned decode under a unique key, no anomaly, every grid "
                 "released" if not bad else
                 f"{len(bad)} of {len(made)} runs did not finish clean, first {bad[0].name}: {bad[0].run.said()}")
    if bad or not made:
        return {}
    good, planned = decoded(made)
    grader.check(name, "decodes", good == planned == opportunities * len(backends),
                 f"{good} of {planned} decodes gave the original bytes of their transport block with a CRC pass while "
                 "the sensor scored on the ring")
    path_check(grader, name, made, rehearsal)
    # ---- The sensor: what it wrote, what it loaded and where it ran.
    archives, problems, facts = [], [], []
    for capture in made:
        sensor_dir = capture.directory / "sensor"
        try:
            if capture.sensor_status != 0:
                raise presence.PresenceError(f"the sensor exited with status {capture.sensor_status}: "
                                             f"{last_line(capture.directory / 'sensor.log')}")
            capture.output = presence.open_output(sensor_dir)
            capture.archive = obs.open_archive(sensor_dir / presence.FEATURES_DIR / features.RECORDS_DIR)
            ready = load_json(capture.directory / "sensor-ready.json")
        except (presence.PresenceError, ObsError, ContractError, OSError) as error:
            grader.check(name, "sensor", False, f"{capture.name}: the presence sensor left no output that verifies: {error}")
            return {}
        archives.append(capture.archive)
        report, source = capture.output.report, capture.output.report["source"]
        first = min((record.header["t_publish_ns"] for record in capture.archive.records), default=None)
        facts.append({"run": capture.name, "device": report["runtime"]["device"], "device_name": report["runtime"].get("name"),
                      "loads": report["loads"], "warm_up_ms": source["warm_up_ms"], "records": source["records"],
                      "ready_before_first_record_ms": None if first is None else (first - ready["t_ready_ns"]) / 1e6})
        if source["kind"] != "ring" or source["ended"] != "closed" or source["stale"] != 0 \
                or source["records"] != len(capture.archive.records):
            problems.append(f"{capture.name}: the sensor did not read the ring to its end without a stale read")
        if report["loads"] != 1:
            problems.append(f"{capture.name}: the artifact was loaded {report['loads']} times in the process of the sensor")
        if capture.output.manifest["sha256"] != freeze["artifact"]["sha256"]:
            problems.append(f"{capture.name}: the sensor scored with the artifact {capture.output.manifest['sha256'][:16]}, "
                            "which is not the frozen one")
        if report["runtime"]["type"] != device:
            problems.append(f"{capture.name}: the head ran on {report['runtime']['device']}, and the gate selects {device}")
        if first is None or not ready["t_ready_ns"] < first or ready["artifact_sha256"] != freeze["artifact"]["sha256"]:
            problems.append(f"{capture.name}: the sensor was not warm before the first record was published")
    shown = facts[0]
    grader.check(name, "sensor", not problems,
                 f"python -m ocudu_cuda_isac.sensing.presence follow, a second process on the standard-library reader, "
                 f"loaded the frozen artifact {freeze['artifact']['sha256'][:16]} once and scored on {shown['device']}"
                 + (f" ({shown['device_name']})" if shown["device_name"] else "")
                 + f"; it ran its feature steps and its first inference before it attached, "
                 f"{text(shown['warm_up_ms']['features'], 2)} and {text(shown['warm_up_ms']['head'], 2)} ms, and was ready "
                 f"{text(shown['ready_before_first_record_ms'], 1)} ms before the first record was published; it read "
                 f"{sum(item['records'] for item in facts)} records until the producer said that none follows, with no "
                 "stale read, and its output verifies" if not problems else "; ".join(problems[:3]))
    if problems:
        return {}
    _, total = account_check(grader, name, made, archives)
    # ---- One decision per answer of the stream, and one scored window per realization of the held-out group.
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
                 f"{decisions} decisions, one per answer of the feature stream: {len(scored)} scored windows, {per_case} "
                 f"for each of the {len(keys)} held-out realizations, each answered at the record that completes its "
                 f"window, and {decisions - len(scored)} of a filling window without a score; every decision names its "
                 "contributing observations, the first and the last sample of its window, the digest of its input and "
                 "the hashes of the frozen model")
    infer = [line["infer_ns"] / 1000.0 for line in scored]
    delays = [(line["t_decision_ns"] - line["t_publish_ns"]) / 1000.0 for line in scored
              if line["t_decision_ns"] is not None and line["t_publish_ns"]]
    grader.note(name, f"device timing, not graded, runtime reference: one inference of the head on {shown['device']}, "
                      f"from the vector in host memory to the score in host memory, median "
                      f"{text(obs.percentile(infer, 0.5))} us and largest {text(max(infer, default=None))} us over "
                      f"{len(infer)} windows; from the publication store of the completing record to the decision, "
                      f"median {text(obs.percentile(delays, 0.5))} us and largest {text(max(delays, default=None))} us; "
                      "this is PyTorch in a Python sensor behind a reader that sleeps, beside a receiver that shares "
                      "the device without MPS, and no latency of the sensing core: S3.6 times the engine and S4.1 the "
                      "two clients")
    return {"runs": len(made), "opportunities": opportunities, "account": total, "sensor": facts,
            "scored": len(scored), "decisions": decisions, "infer_us": sr.spread(infer),
            "publish_to_decision_us": sr.spread(delays), "first_test_access_monotonic_ns": accessed}


# ---------------------------------------------------------------- parity
def run_parity(config, out, grader, table, trained, captures, features, presence, evaluator):
    """The records the sensor kept, scored again offline on the CPU, against what the sensor decided on its device."""
    name, tolerance = "parity", config["tolerance"]
    made = [capture for capture in captures if capture.role == "held_out" and capture.output is not None]
    problems, flips, lines, hashes, worst, joined, evaluations = [], [], 0, 0, [0.0, 0.0], 0, []
    offline_outputs = []
    for capture in made:
        directory = out / "parity" / capture.name
        directory.mkdir(parents=True)
        result = run_tool([sys.executable, "-m", "ocudu_cuda_isac.sensing.presence", "replay", "--archive",
                           str(capture.archive.directory), "--spec", str(out / "spec.json"), "--artifact",
                           str(trained["directory"] / "artifact"), "--device", "cpu", "--output-dir", str(directory)],
                          out / "parity" / f"{capture.name}.log")
        try:
            if result.returncode != 0:
                raise presence.PresenceError(f"exit status {result.returncode}: "
                                             f"{last_line(out / 'parity' / (capture.name + '.log'))}")
            offline = presence.open_output(directory)
        except presence.PresenceError as error:
            grader.check(name, "offline", False, f"{capture.name}: the offline replay left no output that verifies: {error}")
            return {}
        offline_outputs.append(offline)
        live_lines, again = capture.output.features.lines, offline.features.lines
        if [features.comparable(line) for line in live_lines] != [features.comparable(line) for line in again]:
            problems.append(f"{capture.name}: the feature lines of the sensor differ from those of the replay")
        lines += len(live_lines)
        hashes += sum(a["view"]["sha256"] == b["view"]["sha256"]
                      for a, b in zip(capture.output.features.ok, offline.features.ok))
        wrong, facts = evaluator.parity(capture.output, offline, tolerance)
        problems += [f"{capture.name}: {problem}" for problem in wrong]
        flips += facts["flips"]
        joined += facts["joined"]
        worst = [max(worst[0], facts["largest_logit_difference"]), max(worst[1], facts["largest_probability_difference"])]
        write_json(out / "parity" / f"{capture.name}.json", facts)
        evaluations.append(evaluator.evaluate(capture.output, table, "held_out", capture.keys))
    feature_problems = [problem for problem in problems if "feature lines" in problem]
    grader.check(name, "same features", not feature_problems and lines and hashes == joined,
                 f"python -m ocudu_cuda_isac.sensing.presence replay, in a process of its own on the CPU, on the records "
                 f"the sensor kept: the {lines} answers of the stream are those of the sensor line for line, host times "
                 f"aside, with {hashes} views of the same SHA-256" if not feature_problems else feature_problems[0])
    score_problems = [problem for problem in problems if problem not in feature_problems]
    device = made[0].output.report["runtime"]["device"] if made else "none"
    grader.check(name, "scores", not score_problems and joined > 0,
                 f"{joined} held-out windows joined by their identifier: on both sides the same contributing "
                 f"observations and the same SHA-256 of the input, {len(trained['artifact'].columns)} values as float32; "
                 f"the logit on {device} and offline on the cpu differ by at most {worst[0]:.2e} (tolerance "
                 f"{tolerance['logit_absolute']:.0e}) and the probability by at most {worst[1]:.2e} (tolerance "
                 f"{tolerance['probability_absolute']:.0e}): one artifact scores the same held-out feature offline and "
                 "on the selected device" if not score_problems else f"{len(score_problems)} differences, first: "
                                                                     f"{score_problems[0]}")
    grader.check(name, "decisions", not flips and joined > 0,
                 f"at the frozen threshold the {joined} decisions of the device are those of the offline replay"
                 if not flips else f"{len(flips)} decisions differ at the threshold, first: window {flips[0]['window']} "
                                   f"with a margin of {flips[0]['margin']:+.2e}")
    # ---- The truth, read after every decision was written. Recorded, and graded in Sprint 4.
    counts = {"target_present": {"episodes": 0, "detected": 0, "missed": 0, "missing": 0},
              "target_absent": {"episodes": 0, "false_alarm": 0, "quiet": 0, "missing": 0}}
    rows = []
    for evaluation in evaluations:
        rows += evaluation["rows"]
        for side in counts:
            for key in counts[side]:
                counts[side][key] += evaluation["counts"][side][key]
    scores = [(row["logit"], row["label"]) for row in rows if row["logit"] is not None]
    area = evaluator.area_under_curve([logit for logit, _ in scores], [label for _, label in scores])
    threshold = trained["artifact"].threshold
    write_json(out / "held_out_evaluation.json", {
        "schema": evaluator.EVALUATION_SCHEMA, "role": "held_out", "graded": False,
        "why": "the quality of the head on held-out episodes is a criterion of Sprint 4, at the calibrated operating "
               "point; this gate grades the artifact, its freeze and its parity", "threshold_logit": threshold,
        "counts": counts, "area_under_curve": area, "rows": rows})
    present, absent = counts["target_present"], counts["target_absent"]
    grader.note(name, f"held-out episodes at the frozen threshold, read with the labels after the decisions were "
                      f"written, recorded and not graded here: {present['detected']} of {present['episodes']} "
                      f"target-present episodes detected, {present['missed']} missed, {present['missing']} without a "
                      f"decision; {absent['false_alarm']} of {absent['episodes']} target-absent episodes alarmed, "
                      f"{absent['missing']} without a decision; area under the curve {text(area, 3)} over {len(scores)} "
                      "windows (held_out_evaluation.json)")
    return {"joined": joined, "largest_logit_difference": worst[0], "largest_probability_difference": worst[1],
            "flips": flips, "held_out": {"counts": counts, "area_under_curve": area}}


# ---------------------------------------------------------------- causality
def run_causality(config, out, grader, table, trained, captures, features, presence):
    """What may not change a decision, and what a decision may not carry."""
    name, spec = "causality", config["spec"]
    capture = next((item for item in captures if item.role == "held_out" and item.output is not None), None)
    if capture is None or not capture.output.scored:
        grader.check(name, "window", False, "the sensor scored no held-out window")
        return {}
    head = presence.Head(trained["artifact"], "cpu")
    records = capture.archive.records
    _, _, _, reference = presence.run_records(records, spec, head)
    scored = [line for line in reference if line["status"] == "ok"]
    waiting = [line for line in reference if line["status"] != "ok"]
    grader.check(name, "no score without a window",
                 scored and all(line["score"] is None and line["decision"] is None and line["input"] is None
                                for line in waiting),
                 f"{len(waiting)} answers of a filling window give a decision with the status insufficient_history and "
                 f"no score, no decision and no input digest; {len(scored)} complete windows are scored")
    window = scored[0]
    episode = int(window["window"]["episode"], 16)
    own = [record for record in records if record.header["episode"] == episode]
    last = max(index for index, record in enumerate(own) if record.sequence == window["contributing"][-1]["sequence"])
    wanted = presence.comparable(window)
    del wanted["line"]

    def answer(decisions):
        found = [line for line in decisions if line["window"] and line["window"]["id"] == window["window"]["id"]
                 and line["status"] == "ok"]
        if not found:
            return None
        made = presence.comparable(found[0])
        del made["line"]
        return made

    big = presence.np.full(len(own[0].payload) // 8, SENTINEL * (1 + 1j), dtype="<c8").tobytes()
    future = own[:last + 1] + [changed_record(record, payload=big) for record in own[last + 1:]]
    grader.check(name, "future records", answer(presence.run_records(future, spec, head)[3]) == wanted
                 and len(own) > last + 1,
                 f"with the payload of the {len(own) - last - 1} records after the completing one replaced by "
                 f"{SENTINEL:.0e}, the window {window['window']['id']} is decided with the same line: the same input "
                 "digest, the same logit and the same decision, bit for bit. The head scored before it saw them")
    # ---- A head reads the vector of a line and the hash of its specification, and of the vector its columns.
    lines, _, _ = features.run_records(own, spec)
    line = next(item for item in lines if item["status"] == "ok" and item["window"]["id"] == window["window"]["id"])
    base = head.decide(line)
    hidden = json.loads(json.dumps(line))
    hidden["view"]["peak"]["delay_bin"] += 5
    hidden["validity"]["holes"] = 99
    hidden["axes"]["delay"]["resolution_ns"] = 1.0
    hidden["t_publish_ns"] = 1
    vector, columns = trained["artifact"].vector, trained["artifact"].columns
    unused = [value for value in vector if value not in columns]
    for value in unused:
        hidden["features"]["values"][vector.index(value)] += 1000.0
    other = head.decide(hidden)
    touched = json.loads(json.dumps(line))
    touched["features"]["values"][vector.index(columns[0])] += 1.0
    moved = head.decide(touched)
    grader.check(name, "line fields", other["score"] == base["score"] and other["input"] == base["input"]
                 and moved["score"] != base["score"] and moved["input"]["sha256"] != base["input"]["sha256"],
                 f"with the peak, the validity counts, the axes and the host times of the line changed, and with 1000 "
                 f"added to each of the {len(unused)} values the head does not read ({', '.join(unused)}), the logit and "
                 f"the digest of the input are the same; with 1 added to {columns[0]} both change: a head reads "
                 f"{', '.join(presence.LINE_FIELDS)} and of the vector its {len(columns)} columns, by name")
    # ---- No output of the sensing side names what only the evaluator knows.
    names = {row["trajectory"] for row in table["episodes"]} | {row["episode_id"] for row in table["episodes"]}
    sensor_dir = capture.directory / "sensor"
    files = [sensor_dir / presence.DECISIONS_FILE, sensor_dir / presence.REPORT_FILE,
             sensor_dir / presence.FEATURES_DIR / features.LINES_FILE,
             trained["directory"] / "artifact" / presence.HEAD_FILE]
    found = []
    for path in files:
        content = path.read_text(encoding="ascii")
        found += [f"{path.name}: {word}" for word in sorted(names) + list(EVALUATOR_KEYS) if word in content]
    grader.check(name, "no label", not found,
                 f"decisions.jsonl, presence.json and features.jsonl of the sensor and head.json of the artifact name "
                 f"none of the {len(names)} trajectory and episode identifiers of the set and carry no field of a label, "
                 "a split, a scene, a pair or a case: a decision is joined to the truth by its episode key, after the "
                 "run, by the evaluator" if not found else f"{len(found)} evaluator words in the outputs, first: {found[0]}")
    return {"window": window["window"]["id"], "records": len(own), "unused_columns": unused}


# ---------------------------------------------------------------- refusals
def run_refusals(config, out, grader, found, trained, captures, features, presence):
    """Every refusal of the card, by the interface and by the command line."""
    name = "refusals"
    directory = out / "refusals"
    directory.mkdir(parents=True)
    np = presence.np
    artifact, artifact_dir = trained["artifact"], trained["directory"] / "artifact"
    head = presence.Head(artifact, "cpu")
    vector, columns = artifact.vector, artifact.columns
    rows = found["rows"]
    sample = rows["validation"][0]
    count = len(columns)
    capture = next(item for item in captures if item.role == "held_out" and item.output is not None)
    line = capture.output.features.ok[0]

    def copy_artifact(label, change):
        target = directory / label
        shutil.copytree(artifact_dir, target)
        change(target)
        return target

    def flip_weight(target):
        data = bytearray((target / presence.WEIGHTS_FILE).read_bytes())
        data[0] ^= 1
        (target / presence.WEIGHTS_FILE).write_bytes(bytes(data))

    def move_threshold(target):
        manifest = json.loads((target / presence.HEAD_FILE).read_text())
        manifest["threshold"]["logit"] -= 1.0
        (target / presence.HEAD_FILE).write_text(json.dumps(manifest))

    def move_mean(target):
        manifest = json.loads((target / presence.HEAD_FILE).read_text())
        manifest["normalizer"]["mean"][0] += 1.0
        (target / presence.HEAD_FILE).write_text(json.dumps(manifest))

    flipped = copy_artifact("flipped-weight", flip_weight)
    moved = copy_artifact("moved-threshold", move_threshold)
    shifted = copy_artifact("moved-mean", move_mean)
    extra = copy_artifact("extra-file", lambda target: (target / "training.jsonl").write_text("{}\n"))
    reversed_names = list(reversed(sample["names"]))
    other_spec = json.loads(json.dumps(config["spec"]))
    other_spec["view"]["delay_bins"] += 1
    other_line = dict(line, spec_sha256=features.spec_sha256(other_spec))
    held = dict(rows["training"][0], split="held_out")
    crossed = dict(rows["validation"][0], scene=rows["training"][0]["scene"])
    api = (("a batch of two vectors", lambda: head.predict(np.zeros((2, count), dtype=np.float32))),
           ("a vector without its batch axis", lambda: head.predict(np.zeros(count, dtype=np.float32))),
           ("a vector of another width", lambda: head.predict(np.zeros((1, count + 1), dtype=np.float32))),
           ("a vector as float64", lambda: head.predict(np.zeros((1, count), dtype=np.float64))),
           ("a value that is not finite", lambda: head.predict(np.full((1, count), np.nan, dtype=np.float32))),
           ("the names in another order", lambda: head.score(reversed_names, list(reversed(sample["values"])))),
           ("a vector without one of its names", lambda: head.score(sample["names"][:-1], sample["values"][:-1])),
           ("a line of another specification", lambda: head.decide(other_line)),
           ("a weight with one bit changed", lambda: presence.open_artifact(flipped)),
           ("a threshold moved in head.json", lambda: presence.open_artifact(moved)),
           ("a normalizer moved in head.json", lambda: presence.open_artifact(shifted)),
           ("a third file in the artifact", lambda: presence.open_artifact(extra)),
           ("another artifact than the frozen one", lambda: presence.open_artifact(artifact_dir, "0" * 64)),
           ("a validation window in the fit of the normalizer",
            lambda: presence.fit_normalizer(rows["training"] + rows["validation"][:1], vector, columns)),
           ("a validation window in the fit of the weights",
            lambda: presence.fit_head(rows["training"] + rows["validation"][:1], vector, columns,
                                      artifact.manifest["normalizer"], 0.01)),
           ("a training window in the calibration of the threshold",
            lambda: presence.calibrate_threshold(rows["training"][:2], [0.0, 0.0], 0.05)),
           ("a held-out window among the training samples", lambda: presence.check_split(rows["training"] + [held],
                                                                                         rows["validation"])),
           ("a trajectory in the training and in the validation split",
            lambda: presence.check_split(rows["training"], rows["validation"][1:] + [crossed])))
    refused, kept = 0, []
    for label, action in api:
        try:
            action()
        except presence.PresenceError:
            refused += 1
        else:
            kept.append(label)
    grader.check(name, "interface", refused == len(api),
                 f"{refused} of {len(api)} are refused by the interface with one error: " + ", ".join(label for label, _ in api)
                 + ("" if not kept else f"; not refused: {', '.join(kept)}"))
    # ---- The same through the command lines: status 2, one line, nothing written.
    write_json(directory / "other-spec.json", other_spec)
    presence.write_samples(directory / "held-out-row.jsonl", rows["training"] + [held])
    presence.write_samples(directory / "crossed-scene.jsonl", rows["validation"][1:] + [crossed])
    module = [sys.executable, "-m", "ocudu_cuda_isac.sensing.presence"]
    script = [sys.executable, str(ROOT / "scripts/train_presence.py")]
    archive, spec_path = str(capture.archive.directory), str(out / "spec.json")
    training, validation = str(out / "samples/training.jsonl"), str(out / "samples/validation.jsonl")

    def replay_with(artifact_path, spec=spec_path):
        return module + ["replay", "--archive", archive, "--spec", spec, "--artifact", str(artifact_path), "--device", "cpu"]

    def train_with(first, second):
        return script + ["--training", str(first), "--validation", str(second), "--spec", spec_path, "--config",
                         str(out / "training.json")]

    commands = (("a weight with one bit changed", replay_with(flipped), "presence: ", True),
                ("a threshold moved in head.json", replay_with(moved), "presence: ", True),
                ("a third file in the artifact", replay_with(extra), "presence: ", True),
                ("another specification than the head was trained on",
                 replay_with(artifact_dir, str(directory / "other-spec.json")), "presence: ", True),
                ("another artifact than the frozen one", module + ["inspect", str(artifact_dir), "--sha256", "0" * 64],
                 "presence: ", False),
                ("a held-out window among the training samples",
                 train_with(directory / "held-out-row.jsonl", validation), "train_presence: ", None),
                ("a trajectory in the training and in the validation split",
                 train_with(training, directory / "crossed-scene.jsonl"), "train_presence: ", None),
                ("the validation samples given as the training samples", train_with(validation, training),
                 "train_presence: ", None))
    by_tool, unrefused = 0, []
    for number, (label, command, prefix, needs_empty) in enumerate(commands):
        target = directory / f"out-{number:02d}"
        if needs_empty is True:
            target.mkdir()
        if needs_empty is not False:
            command = command + ["--output-dir", str(target)]
        result = run_tool(command, directory / f"tool-{number:02d}.log", timeout=600)
        untouched = not target.exists() or not any(target.iterdir())
        if result.returncode == 2 and untouched and result.stderr.startswith(prefix) and result.stderr.count("\n") == 1:
            by_tool += 1
        else:
            unrefused.append(label)
    grader.check(name, "command line", by_tool == len(commands),
                 f"{by_tool} of {len(commands)} are refused by python -m ocudu_cuda_isac.sensing.presence and by "
                 "python scripts/train_presence.py with status 2 and one line, and nothing is written: "
                 + ", ".join(label for label, _, _, _ in commands)
                 + ("" if not unrefused else f"; not refused: {', '.join(unrefused)}"))
    return {"interface": len(api), "refused": refused, "commands": len(commands), "refused_by_tool": by_tool}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="new directory for the run")
    parser.add_argument("--s33-run", required=True, type=Path,
                        help="run directory of the accepted Sprint 3.3 gate: its specification, its executables and the "
                             "plan that names the reference pair")
    parser.add_argument("--episode-set", required=True, type=Path,
                        help="directory of a channel-episode-set/v1 set, as make episode-set writes it")
    parser.add_argument("--adapter-build", type=Path, help="adapter build directory (default: from the configuration)")
    parser.add_argument("--host-rehearsal", action="store_true",
                        help="development host without a CUDA build: grade the host path on a rehearsed Sprint 3.3 run, "
                             "score on the CPU and leave the hardware gate open")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.exit(2, f"presence_head: {args.output_dir} already exists. Evidence is never overwritten.\n")
    try:
        from ocudu_cuda_isac.episodes import presence_head as evaluator
        from ocudu_cuda_isac.sensing import features, presence
    except ImportError as error:
        parser.exit(2, f"presence_head: the head needs NumPy and PyTorch in this interpreter: {error}\n")
    try:
        config = load_config(args.config, features, presence)
        receipt = native_adapter.verified_receipt(args.adapter_build or ROOT / config["adapter_build"])
        for needed in TOOLS:
            if needed not in receipt["executables"]:
                raise ConfigError(f"the adapter build holds no {needed}. Rebuild it: python scripts/native_adapter.py build")
        if not (args.episode_set / "manifest.json").is_file():
            raise ConfigError(f"{args.episode_set} holds no manifest.json of an episode set. Trace one: make episode-set")
    except (ConfigError, native_adapter.AdapterError) as error:
        parser.exit(2, f"presence_head: {error}\n")
    tools = {name: Path(receipt["executables"][name]["path"]) for name in TOOLS}
    s33_run = args.s33_run.resolve()
    s33_out = s33_run / config["control"]["driver"] if (s33_run / config["control"]["driver"] / "grade.json").is_file() \
        else s33_run
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = args.output_dir.resolve()
    out.mkdir(parents=True)
    write_json(out / "adapter-receipt.json", receipt)
    write_json(out / "spec.json", config["spec"])

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
        "control", lambda: run_control(config, grader, s33_out, args.host_rehearsal, features, evaluator), (None, {}))
    table = found = trained = frozen = None
    if control is None:
        blocked(CASES[1:], "the Sprint 3.3 run does not hold the control this case builds on")
    else:
        table, measured["split"] = guarded(
            "split", lambda: run_split(config, out, grader, control, args.episode_set, evaluator), (None, {}))
        if table is None or not passed("split"):
            table = None
            blocked(CASES[2:], "the episode set cannot be split and replayed as the case split requires")
    if table is not None:
        per_case = measured["split"]["windows_per_realization"]
        measured["records"] = guarded("records", lambda: run_records(config, out, grader, tools, control, table,
                                                                     evaluator, args.host_rehearsal, captures), {})
        if passed("records"):
            found = guarded("features", lambda: run_features(config, out, grader, table, captures, per_case, features,
                                                             presence, evaluator), {})
            measured["features"] = {key: value for key, value in (found or {}).items() if key != "rows"}
        if not passed("records") or not passed("features"):
            found = None
            blocked(CASES[3:] if not passed("records") else CASES[4:],
                    "the training and the validation records did not give the samples this case builds on")
    if found:
        measured["train"], trained = guarded("train", lambda: run_train(config, out, grader, found, presence), ({}, None))
        if trained is None or not passed("train"):
            trained = None
            blocked(CASES[5:], "no head was frozen: nothing of the held-out group is read")
    if trained is not None:
        frozen = guarded("freeze", lambda: run_freeze(config, out, grader, table, found, trained, evaluator), None)
        measured["freeze"] = frozen
        if frozen is None or not passed("freeze"):
            frozen = None
            blocked(CASES[6:], "the freeze record does not hold: nothing of the held-out group is read")
    if frozen is not None:
        measured["live"] = guarded("live", lambda: run_live_case(
            config, out, grader, tools, control, table, trained, frozen, per_case, features, presence, evaluator,
            args.host_rehearsal, captures), {})
        if passed("live"):
            measured["parity"] = guarded("parity", lambda: run_parity(config, out, grader, table, trained, captures,
                                                                      features, presence, evaluator), {})
            measured["causality"] = guarded("causality", lambda: run_causality(config, out, grader, table, trained,
                                                                               captures, features, presence), {})
            measured["refusals"] = guarded("refusals", lambda: run_refusals(config, out, grader, found, trained, captures,
                                                                            features, presence), {})
        else:
            blocked(CASES[7:], "the presence sensor left no held-out decisions this case is compared with")

    failed = [item for item in grader.results if not item["passed"]]
    passed_cases = [case for case in CASES if passed(case)]
    whole = not failed and len(passed_cases) == len(CASES)
    artifact = trained["artifact"] if trained else None
    grade = {
        "schema": GRADE_SCHEMA, "selector": "presence_head", "run_id": out.name, "cases": len(CASES),
        "cases_passed": len(passed_cases), "cases_failed": len(CASES) - len(passed_cases),
        "checks": len(grader.results), "checks_passed": len(grader.results) - len(failed), "checks_failed": len(failed),
        "judgment": ("rehearsal" if args.host_rehearsal else "pass") if whole else "fail",
        "hardware_gate": "open: the CUDA processor did not execute and the head scored on the CPU, this is a host "
                         "rehearsal" if args.host_rehearsal else ("closed" if whole else "open"),
        "reason": "every check of every case passed" if not failed else
                  f"{len(failed)} check(s) failed, first: {failed[0]['case']} {failed[0]['check']}",
        "backends": config["backends"], "spec_sha256": features.spec_sha256(config["spec"]),
        "tolerance": config["tolerance"], "runtime": presence.RUNTIME,
        "artifact_sha256": artifact.sha256 if artifact else None,
        "threshold_logit": artifact.threshold if artifact else None,
        "record_definition": f"{obs.DEFINITION:016x}", "results": grader.results}
    write_json(out / "grade.json", grade)
    write_json(out / "comparison.json", measured)
    # What a second reader starts from: the frozen artifact, its freeze record and the output of the sensor.
    live = [capture for capture in captures if capture.role == "held_out" and capture.output is not None]
    if trained is not None and live:
        write_json(out / "recover.json", {
            "artifact": str((trained["directory"] / "artifact").relative_to(out)),
            "freeze": str((trained["directory"] / "freeze.json").relative_to(out)),
            "sensor": str((live[0].directory / "sensor").relative_to(out)), "window": 0,
            "offline": str((out / "parity" / live[0].name).relative_to(out))})
    lock = {item["name"]: item["commit"] for item in json.loads((ROOT / "ext/lock.json").read_text())["dependencies"]}
    write_json(out / "manifest.json", {
        "run_id": out.name, "selector": "presence_head",
        "purpose": "Sprint 3.4 integration: one linear presence head on the delay-Doppler features, trained on "
                   "receiver-derived records of the training trajectories of an episode set, with its threshold from "
                   "the target-absent validation windows, frozen, and then scored on held-out records in a second "
                   "process on the selected device and again offline",
        "command": shlex.join([sys.executable, *sys.argv]), "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": platform.node(), "machine": platform.machine(), "python": platform.python_version(),
        "numpy": presence.np.__version__, "torch": presence.torch.__version__,
        "cuda_runtime": presence.torch.version.cuda, "cuda_available": bool(presence.torch.cuda.is_available()),
        "cpus": os.cpu_count(), "parent_commit": git_commit(), "locked_sources": lock,
        "inputs": {str(args.config): digest(args.config)}, "driver_sha256": digest(__file__),
        "presence_sha256": digest(ROOT / "src/ocudu_cuda_isac/sensing/presence.py"),
        "features_sha256": digest(ROOT / "src/ocudu_cuda_isac/sensing/features.py"),
        "evaluator_sha256": digest(ROOT / "src/ocudu_cuda_isac/episodes/presence_head.py"),
        "training_script_sha256": digest(ROOT / "scripts/train_presence.py"),
        "s33_run": str(s33_run), "control": measured["control"].get("s33"),
        "episode_set": measured.get("split"), "spec": config["spec"], "steps": features.resolve(config["spec"]),
        "entries": presence.resolve(), "training": config["training"],
        "artifact": None if artifact is None else artifact.model(),
        "order": {"rule": "training and validation records, features, fit, freeze, and only then the first plan of a "
                          "held-out entry",
                  "captures": [{"name": capture.name, "role": capture.role, "started_monotonic_ns": capture.started_ns,
                                "ended_monotonic_ns": capture.ended_ns} for capture in captures],
                  "frozen_monotonic_ns": frozen["frozen_monotonic_ns"] if frozen else None,
                  "first_test_access_monotonic_ns": (measured.get("live") or {}).get("first_test_access_monotonic_ns")},
        "axes": {"split": "staged_harness", "receiver_stage": "o2_pilot_ls", "ue_realization": "replayed channel, no UE",
                 "host_stack": "DeepSig-OCUDU", "sensing_source": config["spec"]["input"]["source"],
                 "placement": "one host, two processes", "transport": "obs-ring/v1 in host shared memory",
                 "runtime": "reference", "clutter": f"static removal {config['spec']['steps']['static']}",
                 "model": "linear", "detector": "frozen_threshold", "heads": ["presence"],
                 "training_domain": "receiver_executed",
                 "gpu_sharing": "the receiver and the sensor share the device by its default time slicing: MPS is "
                                "S4.1" if not args.host_rehearsal else "the sensor is not a CUDA client"},
        "runs": [{"name": capture.name, "role": capture.role, "directory": str(capture.directory.relative_to(out)),
                  "status": capture.run.status, "consumer_status": capture.sensor_status,
                  "entries": [row["episode_id"] for row in capture.rows],
                  "plan_sha256": digest(capture.directory / "plan.json") if (capture.directory / "plan.json").is_file()
                  else None, "ring": capture.run.plan["ring"] if capture.run.plan else None} for capture in captures],
        "adapter": {"executables": {name: {"path": str(tools[name]), "sha256": receipt["executables"][name]["sha256"]}
                                    for name in TOOLS},
                    "receipt": "adapter-receipt.json", "ocudu_build": receipt["ocudu_build"],
                    "built_with_cuda": receipt["pin"]["cuda"]},
        "host_rehearsal": args.host_rehearsal,
        "environment": {key: os.environ[key] for key in ("VIRTUAL_ENV", "CUDA_VISIBLE_DEVICES") if key in os.environ},
        "judgment": grade["judgment"], "reason": grade["reason"]})
    summary = (f"presence_head: {grade['cases_passed']} of {grade['cases']} cases passed, "
               f"{grade['checks_passed']} of {grade['checks']} checks"
               + (", host rehearsal: the hardware gate stays open" if args.host_rehearsal else ""))
    (out / "grade.txt").write_text("\n".join(grader.lines + [summary]) + "\n", encoding="ascii")
    print(summary)
    print(f"Run artifacts: {out}")
    return 0 if whole else 1


if __name__ == "__main__":
    sys.exit(main())
