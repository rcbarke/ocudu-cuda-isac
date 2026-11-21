#!/usr/bin/env python3
"""Sprint 3.2 integration driver (selector snapshot_ring). Hardware command, not part of unit discovery.

It runs the tapped receiver replay as the producer of one observation ring, with a consumer in a
second process, on the plan of the reference pair that the accepted Sprint 3.1 run archived. The
controls are the radio records of the accepted Sprint 2.4 run, written without an observer, and
the observations of the accepted Sprint 3.1 run. Ten cases, each graded on its own:

  control              the accepted Sprint 3.1 run, the plan it tapped and both sets of control records
  audit                every digest, as Sprint 3.1 ran, and every observation published on the ring:
                       the estimates are those of the accepted run, the consumer reads what the
                       observer wrote, and the parts of the 1.47 ms of an audited observation are named
  timed                no SHA-256 between the first fill and the drain, no system call between an
                       admission and its publication: the payloads, hashed offline, and the decoded
                       bytes are those of the audit run, and the tap stays below its ceiling
  wait                 the same with a consumer that sleeps on its eventfd
  reference-reader     the same read by the standard-library reader: the records of the native consumer
  slow-consumer        a consumer that is slower than the receiver: observations are dropped and
                       counted, and every decode is the control's
  terminated-consumer  a consumer that is killed while it holds a credit, and one that takes over:
                       a new epoch, the abandoned records reclaimed, every decode the control's
  tap-off              the timed harness without the tap: no observation, no hash, the same decodes
  refusals             every rule of the plan, refused by the plan check and by the native tool
  backends             the path every decode executed, and the count of every planned decode

    python tests/integration/snapshot_ring.py --config configs/integration/snapshot_ring.json \\
        --s31-run results/runs/s31-spark-001 --output-dir results/runs/s32-spark-001

The output directory must be new. The exit status is 0 only when every check of every case passed.
On a host without a CUDA build, --host-rehearsal runs the same cases on a rehearsed Sprint 3.1 run.
Such a run is graded as a rehearsal and cannot close the hardware gate.
"""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from ocudu_cuda_isac.episodes import grid_apply as replay  # noqa: E402
from ocudu_cuda_isac.episodes import pilot_ls as pl  # noqa: E402
from ocudu_cuda_isac.episodes import replay_pusch as rp  # noqa: E402
from ocudu_cuda_isac.episodes import snapshot_ring as sr  # noqa: E402
from ocudu_cuda_isac.episodes.grid_apply import GridApplyError  # noqa: E402
from ocudu_cuda_isac.episodes.pilot_ls import PilotLsPlanError  # noqa: E402
from ocudu_cuda_isac.episodes.replay_pusch import ReplayPuschError  # noqa: E402
from ocudu_cuda_isac.episodes.snapshot_ring import SnapshotRingError  # noqa: E402
from ocudu_cuda_isac.replay import ContractError, load_json  # noqa: E402
from ocudu_cuda_isac.sensing import obs  # noqa: E402
from ocudu_cuda_isac.sensing.obs import ObsError  # noqa: E402
from ocudu_cuda_isac.sensing.pilot_ls import PilotLsError  # noqa: E402
import native_adapter  # noqa: E402

CONFIG_SCHEMA = "integration-config/snapshot_ring/v1"
GRADE_SCHEMA = "integration-grade/snapshot_ring/v1"
CONTROL_GRADE_SCHEMA = "integration-grade/pilot_ls/v1"
CASES = ("control", "audit", "timed", "wait", "reference-reader", "slow-consumer", "terminated-consumer", "tap-off",
         "refusals", "backends")
TOOLS = ("pilot_ls", "obs_tail")
NATIVE_TIMEOUT_S = 1800
CONSUMER_GRACE_S = 60


class ConfigError(ValueError):
    """The configuration or the run setup is unusable. Nothing was graded."""


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tap_driver = load_script("pilot_ls_driver", ROOT / "tests/integration/pilot_ls.py")
digest, write_json, Grader = tap_driver.digest, tap_driver.write_json, tap_driver.Grader
last_line, git_commit = tap_driver.last_line, tap_driver.git_commit


class Grade(Grader):
    """The Sprint 2.1 grader with this sprint's label."""

    def check(self, case, name, passed, detail):
        self.results.append({"case": case, "check": name, "passed": bool(passed), "detail": detail})
        self._line(("PASS" if passed else "FAIL") + f"  S3.2 {case} {name}: {detail}")
        return bool(passed)

    def note(self, case, text):
        self._line(f"NOTE  S3.2 {case} {text}")


def load_config(path):
    try:
        config = load_json(Path(path))
    except (ContractError, OSError) as error:
        raise ConfigError(f"cannot read the configuration: {error}") from None
    needed = {"schema", "selector", "adapter_build", "backends", "observer", "control", "ring", "tap", "slow_consumer",
              "terminated_consumer", "pair"}
    if not isinstance(config, dict) or set(config) != needed or config["schema"] != CONFIG_SCHEMA \
            or config["selector"] != "snapshot_ring":
        raise ConfigError(f"the configuration must be a {CONFIG_SCHEMA} object for the selector snapshot_ring")
    if list(config["backends"]) != list(rp.BACKENDS):
        raise ConfigError('backends must be ["cpu", "cuda"]: the gate decodes every grid with both')
    observer = config["observer"]
    if not isinstance(observer, dict) or set(observer) != {"tap", "stage", "source", "epoch"} \
            or observer["tap"] != "pilot_ls" or type(observer["epoch"]) is not int or observer["epoch"] < 1:
        raise ConfigError("observer requires the tap pilot_ls, a stage, a source and an epoch from 1")
    control = config["control"]
    if not isinstance(control, dict) or set(control) != {"driver", "run"} or control["driver"] != "pilot_ls" \
            or not isinstance(control["run"], str) or not control["run"]:
        raise ConfigError("control requires the driver pilot_ls and the name of the run it tapped")
    ring = config["ring"]
    if not isinstance(ring, dict) or set(ring) != {"stream", "window_records", "burst_records", "slots",
                                                   "attach_timeout_ms", "drain_ms"}:
        raise ConfigError("ring requires stream, window_records, burst_records, slots, attach_timeout_ms and drain_ms")
    try:
        if ring["slots"] != obs.slots_for(ring["window_records"], ring["burst_records"]):
            raise ConfigError(f"ring.slots is {ring['slots']}, and the window and the burst of the configuration need "
                              f"{obs.slots_for(ring['window_records'], ring['burst_records'])}")
    except ObsError as error:
        raise ConfigError(str(error)) from None
    tap = config["tap"]
    if not isinstance(tap, dict) or set(tap) != {"ceiling_us", "explain_us", "expected_us"} \
            or any(type(value) not in (int, float) or not value > 0 for value in tap.values()):
        raise ConfigError("tap requires ceiling_us, explain_us and expected_us, each positive")
    slow, ended = config["slow_consumer"], config["terminated_consumer"]
    if not isinstance(slow, dict) or set(slow) != {"hold_us"} or type(slow["hold_us"]) is not int or slow["hold_us"] < 1:
        raise ConfigError("slow_consumer requires hold_us, a positive integer")
    if not isinstance(ended, dict) or set(ended) != {"exit_after"} or type(ended["exit_after"]) is not int \
            or ended["exit_after"] < 1:
        raise ConfigError("terminated_consumer requires exit_after, a positive integer")
    pair = config["pair"]
    if not isinstance(pair, dict) or set(pair) != {"expect"} or not isinstance(pair["expect"], dict) \
            or set(pair["expect"]) != {"opportunities", "payload_bytes"}:
        raise ConfigError("pair requires expect with opportunities and payload_bytes")
    return config


# ---------------------------------------------------------------- processes
def consumer_command(tools, consumer, socket, after_pid):
    """The command of one consumer process. A reference reader is this interpreter with the project source."""
    if consumer["reader"] == "reference":
        command = [sys.executable, "-m", "ocudu_cuda_isac.sensing.obs", "tail", "--socket", str(socket), "--output-dir",
                   str(consumer["archive"]), "--wait", consumer["wait"], "--connect-ms", "30000", "--idle-ms", "60000"]
        return command + (["--hold-us", str(consumer["hold_us"])] if consumer.get("hold_us") else [])
    command = [str(tools["obs_tail"]), "--socket", str(socket), "--output-dir", str(consumer["archive"]), "--wait",
               consumer["wait"], "--connect-ms", "30000", "--idle-ms", "60000", "--capacity", "512"]
    for option in ("hold_us", "exit_after", "cpu"):
        if consumer.get(option) is not None:
            command += ["--" + option.replace("_", "-"), str(consumer[option])]
    return command + (["--after-pid", str(after_pid)] if after_pid else [])


def run_scenario(tools, plan_path, native, directory, consumers, extra=()):
    """Run pilot_ls on one plan with its consumers. Returns (exit status, report or None, consumer statuses).

    Every consumer is started before the tool and retries until the tool has opened its channel.
    A consumer after one that ends on purpose waits for that one to end, so the order in which
    they attach is fixed. native is created empty.
    """
    native.mkdir(parents=True)
    plan = json.loads(Path(plan_path).read_text())
    socket = sr.socket_path(native, plan) if plan.get("ring") else None
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    started, logs, previous = [], [], None
    try:
        for index, consumer in enumerate(consumers):
            consumer["archive"].mkdir(parents=True)
            command = consumer_command(tools, consumer, socket, previous)
            log = (directory / f"consumer-{index}.log").open("w")
            log.write(shlex.join(command) + "\n")
            log.flush()
            logs.append(log)
            started.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment))
            previous = started[-1].pid if consumer.get("exit_after") is not None else None
        command = [str(tools["pilot_ls"]), "--plan", str(plan_path), "--output-dir", str(native), *extra]
        with (directory / "native.log").open("w") as stream:
            stream.write(shlex.join(command) + "\n")
            stream.flush()
            tool = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            # A consumer that ends on purpose is reaped at once, so that its successor sees it gone.
            for process, consumer in zip(started, consumers):
                if consumer.get("exit_after") is not None:
                    try:
                        process.wait(timeout=NATIVE_TIMEOUT_S)
                    except subprocess.TimeoutExpired:
                        process.kill()
            try:
                status = tool.wait(timeout=NATIVE_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                tool.kill()
                stream.write("TimeoutExpired: the tool was stopped\n")
                status = -1
        statuses = []
        for process in started:
            try:
                statuses.append(process.wait(timeout=CONSUMER_GRACE_S))
            except subprocess.TimeoutExpired:
                process.kill()
                statuses.append(-1)
    except OSError as error:
        (directory / "native.log").open("a").write(f"{type(error).__name__}: {error}\n")
        for process in started:
            process.kill()
        return -1, None, [-1] * len(consumers)
    finally:
        for log in logs:
            log.close()
    try:
        return status, json.loads((native / "native.json").read_text()), statuses
    except (OSError, ValueError):
        return status, None, statuses


class Control:
    """The plan of the accepted Sprint 3.1 run and both sets of control records."""

    def __init__(self):
        self.name = self.replay_plan_path = self.radio = self.plan = self.channel = self.accepted = None
        self.estimates = {}


class Run:
    """One run of this gate: its plan, its verified output, the archives of its consumers and its control."""

    def __init__(self, name, directory, control):
        self.name, self.directory, self.control = name, directory, control
        self.plan = self.report = self.output = None
        self.status, self.problem, self.consumers, self.statuses, self.archives = None, None, [], [], []

    @property
    def native(self):
        return self.directory / "native"

    def ok(self):
        return self.status == 0 and self.output is not None and self.problem is None

    def said(self):
        return self.problem or f"the native tool exited with status {self.status}: " \
                               f"{last_line(self.directory / 'native.log')}"

    def differences(self):
        """Radio records of this run against those the accepted replay_pusch wrote for the same plan."""
        return pl.radio_differences(self.output.records, self.control.radio.records)

    def counts(self):
        totals = {}
        for record in self.output.records:
            total = totals.setdefault(record["backend"], [0, 0])
            total[0] += record["outcome"] == "decoded"
            total[1] += 1
        return ", ".join(f"{name} {decoded} of {planned} decoded" for name, (decoded, planned) in totals.items())

    def estimates(self):
        """Payload and mask digests of an audited run by opportunity, as its observer wrote them."""
        return {record["host"]["opportunity_id"]: (record["payload"]["digest"], record["validity"]["sha256"])
                for record in self.output.observer.observations}


def ring_run(runs, directory, name, tools, run_id, control, config, mode, wait=None, tap="pilot_ls", consumers=(),
             extra=()):
    """Write the plan of a run on the control plan, run the tool with its consumers and verify what they wrote."""
    run = Run(name, directory, control)
    runs.append(run)
    directory.mkdir(parents=True)
    ring = None
    if wait is not None:
        settings = config["ring"]
        ring = sr.ring_spec(settings["window_records"], settings["burst_records"], wait,
                            "required" if consumers else "none", settings["attach_timeout_ms"], settings["drain_ms"],
                            settings["stream"])
    run.plan = sr.build_plan(run_id, control.replay_plan_path, mode, ring, tap, epoch=config["observer"]["epoch"],
                             source=config["observer"]["source"])
    sr.write_plan(run.plan, directory / "plan.json")
    run.consumers = [dict(consumer, archive=directory / f"consumer-{index}") for index, consumer in enumerate(consumers)]
    run.status, run.report, run.statuses = run_scenario(tools, directory / "plan.json", run.native, directory,
                                                        run.consumers, extra)
    if run.status != 0 or not run.report:
        return run
    try:
        run.output = sr.open_output(run.native)
    except SnapshotRingError as error:
        run.problem = f"the native output does not verify: {error}"
        return run
    for consumer, status in zip(run.consumers, run.statuses):
        if consumer.get("exit_after") is not None:
            run.archives.append(None)  # killed on purpose: it kept nothing
            continue
        try:
            if status != 0:
                raise ObsError(f"the consumer exited with status {status}")
            run.archives.append(obs.open_archive(consumer["archive"]))
        except ObsError as error:
            run.problem = f"the archive of consumer {len(run.archives)} does not verify: {error}"
            run.archives.append(None)
    return run


def run_is_clean(run, backends):
    """True when a run finished, verifies and holds one record per planned decode with every grid released."""
    if not run.ok():
        return False
    totals, opportunities = run.report["totals"], len(run.report["opportunities"])
    return len(run.output.records) == opportunities * len(backends) == totals["planned"] \
        and not run.report["anomalies"] and totals["grids_not_released"] == 0 and totals["retained"] is True \
        and len({record["key"] for record in run.output.records}) == len(run.output.records)


def decodes_check(grader, name, run, check="decodes"):
    """Every radio record of a run against the control, field by field, and the decoded bytes file by file."""
    found = run.differences()
    control = run.control.radio
    same = sum(run.output.radio.decoded(record) == control.decoded(control.by_key[record["key"]])
               for record in run.output.records if record["key"] in control.by_key)
    passes = sum(record["crc_ok"] for record in run.output.records)
    grader.check(name, check, not found and same == len(run.output.records),
                 f"{len(run.output.records)} radio records equal those the accepted replay_pusch wrote without an "
                 f"observer, in the key, the grant, the allocation, the grid digest, the outcome, the CRC flag, the hash "
                 f"of the decoded bytes and the executed path; {passes} decodes pass the CRC and {same} of "
                 f"{len(run.output.records)} files of decoded bytes are the control's, byte for byte"
                 if not found else f"{len(found)} differences from the control, first: {found[0]}")
    return not found and same == len(run.output.records)


def archive_check(grader, name, run, archive, reference, check="records"):
    """The records a consumer kept against the account of the producer and the estimates of the audit run."""
    if archive is None:
        grader.check(name, check, False, run.problem or "the consumer kept no archive")
        return [], {}
    pairs, problems = sr.join_archive(run.output, archive)
    digests = sr.payload_digests(pairs)
    equal = sum(reference.get(opportunity) == found for opportunity, found in digests.items())
    sequences = [record.sequence for record in archive.records]
    ordered = sequences == list(range(len(sequences)))
    grader.check(name, check, not problems and ordered and equal == len(pairs) and pairs,
                 f"the {archive.report['reader']} consumer kept {len(pairs)} records of epoch {archive.report['epoch']}, "
                 f"sequences 0 to {len(sequences) - 1} in order; each is the record of its account line: the join keys, "
                 f"the slot, the times, the valid count and the ticket of its grid; {equal} of {len(pairs)} payloads and "
                 "validity masks, hashed here after the run, have the SHA-256 of the audited estimate of the same "
                 "opportunity" if not problems and ordered else
                 "; ".join(problems[:3]) or f"the sequences of the archive are not 0 to {len(sequences) - 1} in order")
    return pairs, digests


def account_text(states):
    return (f"{states['eligible']} eligible observations: {states['published']} published and {states['dropped']} "
            f"dropped; of the published {states['released']} released, {states['reclaimed']} reclaimed and "
            f"{states['pending']} pending at the cutoff; {states['faults']} faults, {states['epochs']} epochs")


def text(value, digits=1):
    return "none" if value is None else f"{value:.{digits}f}"


def spread_text(stats, unit="us"):
    if not stats.get("count"):
        return "no value"
    return (f"median {text(stats['p50'])} {unit}, 99th percentile {text(stats['p99'])} {unit}, 99.9th percentile "
            f"{text(stats['p999'])} {unit}, largest {text(stats['max'])} {unit} over {stats['count']} values")


def delays_of(archive):
    """Delay of every read of a consumer from the publication store, in microseconds, the first one excluded."""
    return sr.spread([line["delay_ns"] / 1000.0 for line in archive.report["records"][1:]])


# ---------------------------------------------------------------- control
def run_control(config, grader, s31_out, rehearsal):
    """The accepted Sprint 3.1 run, the plan it tapped and both sets of control records."""
    name, control, measured = "control", Control(), {}
    try:
        grade = load_json(s31_out / "grade.json")
        manifest = load_json(s31_out / "manifest.json")
    except (ContractError, OSError) as error:
        grader.check(name, "accepted run", False, f"{s31_out} holds no grade and manifest of the Sprint 3.1 driver: {error}")
        return None, measured
    wanted = "rehearsal" if rehearsal else "pass"
    accepted = isinstance(grade, dict) and grade.get("schema") == CONTROL_GRADE_SCHEMA \
        and grade.get("judgment") == wanted and grade.get("cases_passed") == grade.get("cases") \
        and grade.get("checks_failed") == 0 and (rehearsal or grade.get("hardware_gate") == "closed")
    measured["s31"] = {"run_id": manifest.get("run_id"), "judgment": grade.get("judgment"),
                       "hardware_gate": grade.get("hardware_gate"), "parent_commit": manifest.get("parent_commit"),
                       "adapter_sha256": (manifest.get("adapter") or {}).get("sha256"), "s24_run": manifest.get("s24_run")}
    grader.check(name, "accepted run", accepted,
                 f"the Sprint 3.1 driver graded {s31_out.name} as {grade.get('judgment')}: {grade.get('cases_passed')} of "
                 f"{grade.get('cases')} cases, {grade.get('checks_failed')} failed checks, hardware gate "
                 f"{str(grade.get('hardware_gate')).split(':')[0]}, at commit {str(manifest.get('parent_commit'))[:7]}"
                 + ("" if accepted else f"; this gate needs the judgment {wanted}"
                    + ("" if rehearsal else " with the hardware gate closed")))
    listed = {item["name"]: item for item in manifest.get("runs", []) if isinstance(item, dict)}
    item = listed.get(config["control"]["run"])
    try:
        if item is None or item.get("status") != 0:
            raise PilotLsPlanError(f"the Sprint 3.1 run holds no finished run named {config['control']['run']}")
        directory = s31_out / item["directory"]
        if digest(directory / "plan.json") != item.get("plan_sha256"):
            raise PilotLsPlanError("the plan of the tapped run does not have the hash its manifest names")
        tapped_plan = pl.load_plan(directory / "plan.json")
        control.accepted = pl.open_output(directory / "native")
        control.replay_plan_path = Path(tapped_plan["replay_plan"]["path"])
        if digest(control.replay_plan_path) != tapped_plan["replay_plan"]["sha256"]:
            raise PilotLsPlanError("the replay plan of the Sprint 2.4 run does not have the hash the tapped plan binds")
        control.plan = rp.load_plan(control.replay_plan_path)
        control.channel = replay.load_plan(control.plan["channel_plan"]["path"])
        control.radio = rp.open_output(control.replay_plan_path.parent / "native")
    except (PilotLsPlanError, PilotLsError, ReplayPuschError, GridApplyError, OSError) as error:
        grader.check(name, "records", False, f"{error}")
        return None, measured
    control.name = config["control"]["run"]
    control.estimates = {record["host"]["opportunity_id"]: (record["payload"]["digest"], record["validity"]["sha256"])
                         for record in control.accepted.observations}
    untapped = all(record["tap"] == "none" for record in control.radio.records)
    measured.update(records=len(control.radio.records), observations=len(control.accepted.observations),
                    replay_plan=str(control.replay_plan_path))
    grader.check(name, "records", untapped and control.estimates,
                 f"the run {control.name} of the Sprint 3.1 gate verifies with the reader of that sprint: "
                 f"{len(control.accepted.observations)} accepted observations; the replay plan it tapped has the hash "
                 f"it binds, and the Sprint 2.4 run beside that plan holds {len(control.radio.records)} radio records, "
                 "each written without a tap by the accepted replay_pusch")
    return (control if untapped and control.estimates else None), measured


# ---------------------------------------------------------------- audit
def run_audit(config, out, grader, tools, runs, control):
    name, backends, expect = "audit", config["backends"], config["pair"]["expect"]
    run = ring_run(runs, out / "audit" / "run", "audit", tools, "s32-audit", control, config, "audit", "spin",
                   consumers=[{"reader": "native", "wait": "spin"}])
    clean = run_is_clean(run, backends) and run.output.mode == "audit" and run.output.observer is not None
    count = len(run.output.observer.observations) if clean else 0
    grader.check(name, "native run", clean and count == expect["opportunities"],
                 f"both members of the pair in the mode audit, every observation published on a ring of "
                 f"{run.output.ring['slot_count']} slots of {run.output.ring['slot_bytes']} bytes to a native consumer: "
                 f"{count} observations, {run.counts()}; expected {expect['opportunities']} observations"
                 if run.ok() else run.said())
    if not run.ok() or run.output.observer is None:
        return {}, None
    measured = {}
    # ---- The corrected observer computes what the accepted one computed.
    mine = run.estimates()
    ingress = {record["host"]["opportunity_id"]: record["host"]["ingress"]["readback_sha256"]
               for record in run.output.observer.observations}
    accepted = {record["host"]["opportunity_id"]: record["host"]["ingress"]["readback_sha256"]
                for record in control.accepted.observations}
    equal = sum(control.estimates.get(opportunity) == found for opportunity, found in mine.items())
    grids = sum(accepted.get(opportunity) == found for opportunity, found in ingress.items())
    grader.check(name, "accepted estimates", equal == grids == count == len(control.estimates),
                 f"{equal} of {count} payloads and validity masks have the SHA-256 that the accepted Sprint 3.1 run "
                 f"recorded for the same opportunity, and {grids} of {count} were computed from a grid with the readback "
                 "digest of that run: the cores that this sprint amended compute what the accepted ones computed")
    decodes_check(grader, name, run)
    # ---- The consumer read what the observer wrote, and the account balances.
    archive = run.archives[0] if run.archives else None
    pairs, _ = archive_check(grader, name, run, archive, mine)
    files = sum(record.payload == run.output.observer.payload(by_id[line["opportunity_id"]])
                and record.mask == run.output.observer.mask(by_id[line["opportunity_id"]])
                for by_id in [{item["host"]["opportunity_id"]: item for item in run.output.observer.observations}]
                for line, record in pairs)
    states = sr.account(run.output)
    grader.check(name, "account", states["balanced"] and states["dropped"] == 0 and states["pending"] == 0
                 and files == len(pairs) == count,
                 f"{account_text(states)}; {files} of {len(pairs)} records in the consumer's archive hold the bytes of "
                 "the payload file and of the mask file that the observer wrote for the same opportunity")
    # ---- Where the time of an audited observation goes. Recorded beside the breakdown of 2026-11-21.
    parts = run.report["observer"]["parts"] or {}
    measured["parts_us"] = parts
    measured["hash_in_window"] = run.output.hash_in_window()
    whole = run.report["observer"]["service_us"]
    if whole.get("count"):
        expected = sr.SPARK_BREAKDOWN_US
        grader.note(name, f"parts of one audited observation, not graded, mean over {whole['count']} observations, the "
                          f"first one excluded: estimate {text(parts['estimate_us'].get('mean'))} us, digest of the grid "
                          f"{text(parts['grid_digest_us'].get('mean'))} us (expected {expected['grid_digest']:.0f} us on "
                          f"the Spark), record fields, keys and digest of the mask {text(parts['keys_mask_us'].get('mean'))} "
                          f"us, digest of the payload {text(parts['payload_digest_us'].get('mean'))} us (expected "
                          f"{expected['payload_digest']:.0f} us), whole observation {text(whole.get('mean'))} us (accepted "
                          f"run: {expected['accepted_observation']:.0f} us); the copy into the slot and the publication "
                          f"{text((parts.get('publish_us') or {}).get('mean'))} us")
        window = run.output.hash_in_window()
        grader.note(name, f"SHA-256 between the first fill and the drain in the mode audit, not graded: {window['digests']} "
                          f"digests over {window['bytes']} bytes")
    return measured, run


# ---------------------------------------------------------------- timed
def timed_checks(config, grader, name, run, reference, expect_wait):
    """What every timed run with a consumer that keeps up must show. Returns what was measured."""
    measured = {}
    window = run.output.hash_in_window()
    counts = run.output.ring["counts"]
    measured["hash_in_window"] = window
    grader.check(name, "no checksum", window == {"digests": 0, "updates": 0, "bytes": 0},
                 f"between the first fill and the drain of the ring the hash of the core computed {window['digests']} "
                 f"digests, took {window['updates']} calls with data and hashed {window['bytes']} bytes; the grids, the "
                 f"records and the decoded bytes were written and digested after the drain "
                 f"({run.report['evidence']['hash_after_drain']['digests']} digests there)")
    calls = sum(line["window_calls"] for line in run.output.publications)
    waited = run.output.ring["wait"]["consumer"]
    spin = expect_wait == "spin"
    grader.check(name, "no system call", calls == 0 == counts["window_calls"] and waited == expect_wait
                 and (counts["wakes"] == 0 if spin else counts["wakes"] <= counts["published"]),
                 f"the producer made {counts['window_calls']} system calls between an admission and its publication "
                 f"store over {counts['published']} publications; the consumer attached in the mode {waited}, and the "
                 f"producer wrote its eventfd {counts['wakes']} times"
                 + ("" if spin else ", each after a publication store that found the waiting flag set"))
    decodes_check(grader, name, run)
    archive = run.archives[0] if run.archives else None
    pairs, digests = archive_check(grader, name, run, archive, reference)
    states = sr.account(run.output)
    undigested = sum(record.header["digest_kind"] == 0 for _, record in pairs)
    grader.check(name, "account", states["balanced"] and states["dropped"] == 0 and states["pending"] == 0
                 and len(pairs) == states["published"] == len(reference) and undigested == len(pairs),
                 f"{account_text(states)}; {len(pairs)} of {states['published']} published records reached the consumer, "
                 f"{undigested} of them without a digest in the record, as a timed run publishes them")
    measured.update(account=states, consumer_delay_us=delays_of(archive) if archive else None)
    return measured


def run_timed(config, out, grader, tools, runs, control, audit):
    name, backends, limits = "timed", config["backends"], config["tap"]
    run = ring_run(runs, out / "timed" / "run", "timed", tools, "s32-timed", control, config, "timed", "spin",
                   consumers=[{"reader": "native", "wait": "spin"}])
    clean = run_is_clean(run, backends) and run.output.mode == "timed"
    grader.check(name, "native run", clean,
                 f"both members of the pair in the mode timed, from the plan of the audit run, with a spinning native "
                 f"consumer: {len(run.output.publications)} observations, {run.counts()}; one record per planned decode "
                 "under a unique key, no anomaly, every grid released" if run.ok() else run.said())
    if not run.ok() or audit is None:
        if run.ok():
            grader.check(name, "audit run", False, "the audit run did not finish, so the timed run has no reference")
        return {}, None
    measured = timed_checks(config, grader, name, run, audit.estimates(), "spin")
    # ---- The cost of the tap, with its ceiling. The estimate and the publication are reported apart.
    times = sr.tap_times(run.output)
    measured["tap_us"] = times
    service, estimate, publish = times["service_us"], times["estimate_us"], times["publish_us"]
    under = bool(service.get("count")) and service["mean"] <= limits["ceiling_us"] and service["max"] <= limits["ceiling_us"]
    grader.check(name, "tap cost", under,
                 f"service time of the tap from the read of the grid to the publication store, over {service.get('count')} "
                 f"observations of {run.output.published[0]['payload_bytes']} bytes, the first one excluded: mean "
                 f"{text(service.get('mean'))} us, largest {text(service.get('max'))} us, at or below the ceiling of "
                 f"{limits['ceiling_us']:g} us; the estimate takes {text(estimate.get('mean'))} us in the mean and "
                 f"{text(estimate.get('max'))} us at most, the admission, the header and the publication "
                 f"{text(publish.get('mean'))} us and {text(publish.get('max'))} us" if service.get("count") else
                 "the run published no observation after the first")
    if service.get("count"):
        grader.note(name, f"expected before the publication: about {limits['expected_us']:g} us, from "
                          f"{sr.SPARK_BREAKDOWN_US['no_hash']} us measured on the Spark without a hash on 2026-11-21; "
                          f"measured here {text(estimate.get('mean'))} us for the estimate"
                          + (f"; the mean of the whole tap is above {limits['explain_us']:g} us and needs an explanation "
                             "before this number is used" if service["mean"] > limits["explain_us"] else
                             f"; the mean of the whole tap is at or below {limits['explain_us']:g} us"))
    if run.archives and run.archives[0] is not None:
        delay = delays_of(run.archives[0])
        grader.note(name, f"delay of the spinning consumer from the publication store to its read, not graded: "
                          f"{spread_text(delay)}; publication alone: {spread_text(publish)}")
    published = sr.PUBLISHED_RING_US
    measured["published_ring_us"] = published
    grader.note(name, f"recorded beside these and not graded: the supervised ring of the OCUDU dApp platform on a GB10 "
                      f"publishes {published['payload_bytes']} bytes in {published['producer_p50']} us at the median, "
                      f"{published['producer_p999_quiet']} us quiet against {published['producer_p999_cell']:g} us with a "
                      f"cell on the air at the 99.9th percentile, with {published['system_calls_per_publication']} system "
                      f"calls per publication, and wakes its consumer in {published['consumer_wake_p999_quiet']:g} us "
                      f"against {published['consumer_wake_p999_cell']:g} us ({published['source']})")
    return measured, run


def run_wait(config, out, grader, tools, runs, control, audit):
    name, backends = "wait", config["backends"]
    run = ring_run(runs, out / "wait" / "run", "wait", tools, "s32-wait", control, config, "timed", "wait",
                   consumers=[{"reader": "native", "wait": "wait"}])
    clean = run_is_clean(run, backends) and run.output.mode == "timed"
    grader.check(name, "native run", clean,
                 f"the timed run with a native consumer that sleeps on its eventfd: {len(run.output.publications)} "
                 f"observations, {run.counts()}" if run.ok() else run.said())
    if not run.ok() or audit is None:
        return {}
    measured = timed_checks(config, grader, name, run, audit.estimates(), "wait")
    if run.archives and run.archives[0] is not None:
        delay, sleeps = delays_of(run.archives[0]), run.archives[0].report["counts"]["sleeps"]
        measured["sleeps"] = sleeps
        grader.note(name, f"delay of the waiting consumer from the publication store to its read, not graded: "
                          f"{spread_text(delay)}; it slept {sleeps} times; publication alone: "
                          f"{spread_text(sr.tap_times(run.output)['publish_us'])}")
    return measured


def run_reference_reader(config, out, grader, tools, runs, control, audit, timed):
    name, backends = "reference-reader", config["backends"]
    run = ring_run(runs, out / "reference_reader" / "run", "reference-reader", tools, "s32-reference", control, config,
                   "timed", "wait", consumers=[{"reader": "reference", "wait": "wait"}])
    clean = run_is_clean(run, backends) and run.output.mode == "timed"
    grader.check(name, "native run", clean,
                 f"the timed run read by ocudu_cuda_isac.sensing.obs, the standard-library reader, in the mode wait: "
                 f"{len(run.output.publications)} observations, {run.counts()}" if run.ok() else run.said())
    if not run.ok() or audit is None:
        return {}
    archive = run.archives[0] if run.archives else None
    pairs, _ = archive_check(grader, name, run, archive, audit.estimates())
    decodes_check(grader, name, run)
    # ---- The same slot contents by both readers: the records of the native consumer of the timed run.
    native = timed.archives[0] if timed is not None and timed.archives else None
    varying = ("t_ready_ns", "t_publish_ns")
    same, compared = 0, 0
    if native is not None and archive is not None:
        other = {record.header["opportunity"]: record for record in native.records}
        for record in archive.records:
            twin = other.get(record.header["opportunity"])
            compared += 1
            same += twin is not None and record.payload == twin.payload and record.mask == twin.mask \
                and {key: value for key, value in record.header.items() if key not in varying} \
                == {key: value for key, value in twin.header.items() if key not in varying} \
                and record.host[:32] == twin.host[:32]
    grader.check(name, "both readers", native is not None and compared and same == compared == len(pairs),
                 f"{same} of {compared} records have the header fields, the host block, the validity bits and the payload "
                 "bytes that the native consumer kept for the same opportunity in the timed run; only the two host times "
                 "of a run differ" if native is not None else "the timed run kept no archive to compare with")
    return {"records": compared, "equal": same,
            "consumer_delay_us": delays_of(archive) if archive is not None else None}


# ---------------------------------------------------------------- overload and restart
def run_slow(config, out, grader, tools, runs, control, audit):
    name, backends, hold = "slow-consumer", config["backends"], config["slow_consumer"]["hold_us"]
    run = ring_run(runs, out / "slow_consumer" / "run", "slow-consumer", tools, "s32-slow", control, config, "timed",
                   "spin", consumers=[{"reader": "native", "wait": "spin", "hold_us": hold}])
    clean = run_is_clean(run, backends)
    grader.check(name, "native run", clean,
                 f"the timed run with a consumer that works {hold} us on every record: "
                 f"{len(run.output.publications)} observations, {run.counts()}" if run.ok() else run.said())
    if not run.ok() or audit is None:
        return {}
    states = sr.account(run.output)
    waits = [line["service_us"] for line in run.output.dropped]
    slots = run.output.ring["slot_count"]
    depth = run.output.ring["counts"]["max_depth"]
    grader.check(name, "overload", states["dropped"] > 0 and depth == slots - 1 and max(waits, default=0.0) < 1000.0,
                 f"{states['dropped']} observations found no credit and were dropped at once and counted, the longest in "
                 f"{text(max(waits, default=None), 2)} us; the ring held at most {depth + 1} of its {slots} slots: nothing "
                 "was overwritten and the producer never waited" if states["dropped"] else
                 "no observation was dropped: the consumer was not slower than the receiver, so the overload was not "
                 "reached. Raise slow_consumer.hold_us")
    decodes_check(grader, name, run, "receiver continues")
    archive = run.archives[0] if run.archives else None
    pairs, _ = archive_check(grader, name, run, archive, audit.estimates())
    window = run.output.hash_in_window()
    grader.check(name, "account", states["balanced"] and len(pairs) == states["released"]
                 and window == {"digests": 0, "updates": 0, "bytes": 0}
                 and run.output.ring["counts"]["window_calls"] == 0,
                 f"{account_text(states)}; the consumer kept the {len(pairs)} records it released; no hash in the window "
                 "and no system call between an admission and its publication, under overload as well")
    return {"account": states, "max_depth": depth, "drop_us": sr.spread(waits)}


def run_terminated(config, out, grader, tools, runs, control, audit):
    name, backends, after = "terminated-consumer", config["backends"], config["terminated_consumer"]["exit_after"]
    run = ring_run(runs, out / "terminated_consumer" / "run", "terminated-consumer", tools, "s32-terminated", control,
                   config, "timed", "spin",
                   consumers=[{"reader": "native", "wait": "spin", "exit_after": after},
                              {"reader": "native", "wait": "spin"}])
    clean = run_is_clean(run, backends)
    killed = run.statuses[0] if run.statuses else None
    grader.check(name, "native run", clean and killed == 9,
                 f"the timed run with a consumer that ends without a word after {after} records, while it holds a "
                 f"credit, and a second one that attaches when the first is gone: {len(run.output.publications)} "
                 f"observations, {run.counts()}; the first consumer ended with status {killed}"
                 if run.ok() else run.said())
    if not run.ok() or audit is None:
        return {}
    states = sr.account(run.output)
    ring = run.output.ring
    epochs = sorted({line["epoch"] for line in run.output.published})
    grader.check(name, "new epoch", states["epochs"] == 2 and len(epochs) == 2 and ring["last_epoch"] == epochs[-1]
                 and states["reclaimed"] >= 1 and states["faults"] == 0,
                 f"{states['epochs']} consumers attached; the records were published in the epochs {epochs}, the second "
                 f"from sequence 0 again; {states['reclaimed']} records that the first consumer held or never read were "
                 "reclaimed when the second one attached, after the channel of the first was closed")
    decodes_check(grader, name, run, "receiver continues")
    archive = run.archives[1] if len(run.archives) > 1 else None
    pairs, _ = archive_check(grader, name, run, archive, audit.estimates())
    second = [line for line in run.output.published if line["epoch"] == epochs[-1]] if epochs else []
    grader.check(name, "account", states["balanced"] and states["pending"] == 0 and len(pairs) == len(second)
                 and not any(path.suffix == ".obs" for path in run.consumers[0]["archive"].iterdir()),
                 f"{account_text(states)}; the second consumer kept all {len(pairs)} records of its epoch, and the "
                 "first, which was killed, kept none")
    return {"account": states, "epochs": epochs}


def run_tap_off(config, out, grader, tools, runs, control):
    name, backends = "tap-off", config["backends"]
    run = ring_run(runs, out / "tap_off" / "run", "tap-off", tools, "s32-tap-off", control, config, "timed", None,
                   tap="none")
    clean = run_is_clean(run, backends)
    window = run.output.hash_in_window() if run.ok() else None
    grader.check(name, "native run", clean and run.output.ring is None and not run.output.publications
                 and run.report["totals"]["observations"] == 0 and window == {"digests": 0, "updates": 0, "bytes": 0},
                 f"the timed harness without the tap and without a ring: no observation, {run.counts()}, and "
                 f"{window['digests']} digests between the first fill and the last decode" if run.ok() else run.said())
    if run.ok():
        decodes_check(grader, name, run)
    return {"hash_in_window": window}


# ---------------------------------------------------------------- refusals
def run_refusals(config, out, grader, tools, control):
    """Every rule of the plan: refused by the plan check, and by the native tool before a file is written."""
    name, case_dir = "refusals", out / "refusals"
    case_dir.mkdir()
    ring = sr.ring_spec(config["ring"]["window_records"], config["ring"]["burst_records"], "spin", "none")
    good = sr.build_plan("s32-refusal", control.replay_plan_path, "timed", ring)
    edits = (
        ("an unknown evidence mode", lambda plan: plan["evidence"].update(mode="fast"), "evidence mode"),
        ("a timed tap without a ring", lambda plan: plan.update(ring=None), "names a tap and no ring"),
        ("a ring without a tap", lambda plan: (plan["observer"].update(tap="none", stage=None, source=None, layout=None),
                                               plan.update(sources=[])), "names a ring and no tap"),
        ("a slot count that is not the rule's", lambda plan: plan["ring"].update(slots=16), "rounded up to a power of two"),
        ("an unknown wait mode", lambda plan: plan["ring"].update(wait="sleep"), "ring.wait"),
        ("an unknown attach rule", lambda plan: plan["ring"].update(attach="maybe"), "ring.attach"),
        ("a stream name with a path", lambda plan: plan["ring"].update(stream="../o2"), "ring.stream"),
    )
    checked = native = 0
    said = []
    for index, (label, edit, words) in enumerate(edits):
        plan = json.loads(json.dumps(good))
        edit(plan)
        try:
            sr.check_plan(plan)
        except SnapshotRingError as error:
            checked += words in str(error)
        path = case_dir / f"plan-{index}.json"
        path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="ascii")
        status, report, _ = run_scenario(tools, path, case_dir / f"native-{index}", case_dir, [])
        log = (case_dir / "native.log").read_text(errors="replace")
        (case_dir / f"native-{index}.log").write_text(log)
        wrote = sorted(item.name for item in (case_dir / f"native-{index}").iterdir())
        if status == 2 and report is None and not wrote and words in log:
            native += 1
        else:
            said.append(f"{label}: status {status}, files {wrote}, last line {last_line(case_dir / 'native.log')!r}")
    grader.check(name, "plan check", checked == len(edits),
                 f"{checked} of {len(edits)} faulty plans are refused by the plan check with the rule they break: "
                 + ", ".join(label for label, _, _ in edits))
    grader.check(name, "native tool", native == len(edits),
                 f"{native} of {len(edits)} faulty plans are refused by the native tool with status 2 and the same rule, "
                 "before a processor is created, a ring is opened or a file is written" if not said else "; ".join(said[:3]))
    return {"plans": len(edits), "refused_by_check": checked, "refused_by_tool": native}


# ---------------------------------------------------------------- backends
def run_backends(config, grader, runs, rehearsal):
    """The path every decode of every run executed. A decode that fell back to the host path cannot close the gate."""
    name, measured = "backends", {}
    finished = [run for run in runs if run.ok()]
    records = [record for run in finished for record in run.output.records]
    planned = sum(len(run.report["opportunities"]) * len(config["backends"]) for run in finished)
    built = {run.report.get("built_with_cuda") for run in finished}
    cuda = [record for record in records if record["backend"] == "cuda"]
    resident = sum((record["path"] or {}).get("executed") == "resident" for record in cuda)
    host = [record for record in records if record["backend"] == "cpu"]
    on_host = sum((record["path"] or {}).get("executed") == "host" for record in host)
    decoded = sum(record["outcome"] == "decoded" for record in records)
    measured.update(runs=len(finished), records=len(records), planned=planned, cuda=len(cuda), resident=resident,
                    host=len(host), decoded=decoded, modes=sorted({run.output.mode for run in finished}))
    grader.check(name, "decode count", len(finished) == len(runs) and len(records) == planned and decoded == len(records),
                 f"{len(finished)} of {len(runs)} runs finished; {len(records)} radio records for {planned} planned decodes, "
                 f"{decoded} of them decoded with every byte equal to the original")
    if rehearsal:
        grader.check(name, "executed path", on_host == len(host) and len(host) > 0,
                     f"host rehearsal: {on_host} of {len(host)} decodes of the host processor ran the host path; the "
                     f"{len(cuda)} decodes requested from the CUDA processor are not graded on a build without CUDA")
    else:
        grader.check(name, "executed path", built == {True} and cuda and resident == len(cuda) and on_host == len(host),
                     f"{resident} of {len(cuda)} decodes requested from the CUDA processor ran its resident path, in the "
                     f"modes audit and timed, with the ring attached, and {on_host} of {len(host)} decodes of the host "
                     "processor ran the host path; the adapter is built with CUDA" if built == {True} else
                     "the adapter is not built with CUDA: the CUDA processor cannot execute")
    for backend in config["backends"]:
        for mode in ("audit", "timed"):
            times = sorted(record["timing"]["service_us"] for run in finished if run.output.mode == mode
                           and run.name in ("audit", "timed") for record in run.output.records
                           if record["backend"] == backend and record["timing"]["service_us"] is not None
                           and not record["timing"]["first_use"])
            if times:
                measured[f"service_us_{backend}_{mode}"] = sr.spread(times)
                grader.note(name, f"PUSCH service time of {backend} in the mode {mode}, not graded: mean "
                                  f"{sum(times) / len(times):.0f} us, median {times[len(times) // 2]:.0f} us, largest "
                                  f"{times[-1]:.0f} us over {len(times)} decodes; the tap and every digest lie outside "
                                  "this time; the host is not tuned")
    return measured


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="new directory for the run")
    parser.add_argument("--s31-run", required=True, type=Path,
                        help="run directory of the accepted Sprint 3.1 gate: it names the plan and holds the accepted "
                             "observations; the Sprint 2.4 run that plan lies in holds the control decodes")
    parser.add_argument("--adapter-build", type=Path, help="adapter build directory (default: from the configuration)")
    parser.add_argument("--host-rehearsal", action="store_true",
                        help="development host without a CUDA build: grade the host path on a rehearsed Sprint 3.1 run "
                             "and leave the hardware gate open")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.exit(2, f"snapshot_ring: {args.output_dir} already exists. Evidence is never overwritten.\n")
    try:
        config = load_config(args.config)
        receipt = native_adapter.verified_receipt(args.adapter_build or ROOT / config["adapter_build"])
        for needed in TOOLS:
            if needed not in receipt["executables"]:
                raise ConfigError(f"the adapter build holds no {needed}. Rebuild it: python scripts/native_adapter.py build")
    except (ConfigError, native_adapter.AdapterError) as error:
        parser.exit(2, f"snapshot_ring: {error}\n")
    tools = {name: Path(receipt["executables"][name]["path"]) for name in TOOLS}
    s31_run = args.s31_run.resolve()
    s31_out = s31_run / config["control"]["driver"] if (s31_run / config["control"]["driver"] / "grade.json").is_file() \
        else s31_run
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = args.output_dir.resolve()
    out.mkdir(parents=True)
    write_json(out / "adapter-receipt.json", receipt)

    grader, measured, runs = Grade(), {}, []

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

    control, measured["control"] = guarded("control", lambda: run_control(config, grader, s31_out, args.host_rehearsal),
                                           (None, {}))
    if control is not None:
        measured["audit"], audit = guarded("audit", lambda: run_audit(config, out, grader, tools, runs, control),
                                           ({}, None))
        measured["timed"], timed = guarded("timed", lambda: run_timed(config, out, grader, tools, runs, control, audit),
                                           ({}, None))
        measured["wait"] = guarded("wait", lambda: run_wait(config, out, grader, tools, runs, control, audit))
        measured["reference-reader"] = guarded(
            "reference-reader", lambda: run_reference_reader(config, out, grader, tools, runs, control, audit, timed))
        measured["slow-consumer"] = guarded("slow-consumer",
                                            lambda: run_slow(config, out, grader, tools, runs, control, audit))
        measured["terminated-consumer"] = guarded(
            "terminated-consumer", lambda: run_terminated(config, out, grader, tools, runs, control, audit))
        measured["tap-off"] = guarded("tap-off", lambda: run_tap_off(config, out, grader, tools, runs, control))
        measured["refusals"] = guarded("refusals", lambda: run_refusals(config, out, grader, tools, control))
        measured["backends"] = guarded("backends", lambda: run_backends(config, grader, runs, args.host_rehearsal))
    else:
        for case in CASES[1:]:
            grader.check(case, "control", False, "the Sprint 3.1 run does not hold the control this case is compared with")

    failed = [item for item in grader.results if not item["passed"]]
    passed_cases = [case for case in CASES if any(item["case"] == case for item in grader.results)
                    and all(item["passed"] for item in grader.results if item["case"] == case)]
    complete = not failed and len(passed_cases) == len(CASES)
    grade = {
        "schema": GRADE_SCHEMA, "selector": "snapshot_ring", "run_id": out.name, "cases": len(CASES),
        "cases_passed": len(passed_cases), "cases_failed": len(CASES) - len(passed_cases),
        "checks": len(grader.results), "checks_passed": len(grader.results) - len(failed), "checks_failed": len(failed),
        "judgment": ("rehearsal" if args.host_rehearsal else "pass") if complete else "fail",
        "hardware_gate": "open: the CUDA processor did not execute, this is a host rehearsal" if args.host_rehearsal
                         else ("closed" if complete else "open"),
        "reason": "every check of every case passed" if not failed else
                  f"{len(failed)} check(s) failed, first: {failed[0]['case']} {failed[0]['check']}",
        "backends": config["backends"], "observer": config["observer"], "ring": config["ring"], "tap": config["tap"],
        "record_definition": f"{obs.DEFINITION:016x}", "results": grader.results}
    write_json(out / "grade.json", grade)
    write_json(out / "comparison.json", measured)
    lock = {item["name"]: item["commit"] for item in json.loads((ROOT / "ext/lock.json").read_text())["dependencies"]}
    clock = Path("/sys/devices/system/clocksource/clocksource0/current_clocksource")
    write_json(out / "manifest.json", {
        "run_id": out.name, "selector": "snapshot_ring",
        "purpose": "Sprint 3.2 integration: the reference pair of the accepted Sprint 3.1 run decoded again in the modes "
                   "audit and timed, with every observation published on the ring obs-ring/v1 to a consumer in a second "
                   "process, compared with the decodes of the accepted Sprint 2.4 run and the estimates of the accepted "
                   "Sprint 3.1 run",
        "command": shlex.join([sys.executable, *sys.argv]), "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": platform.node(), "machine": platform.machine(), "python": platform.python_version(),
        "clock_source": clock.read_text().strip() if clock.is_file() else None, "cpus": os.cpu_count(),
        "parent_commit": git_commit(), "locked_sources": lock,
        "inputs": {str(args.config): digest(args.config)}, "driver_sha256": digest(__file__),
        "reader_sha256": digest(ROOT / "src/ocudu_cuda_isac/episodes/snapshot_ring.py"),
        "reference_reader_sha256": digest(ROOT / "src/ocudu_cuda_isac/sensing/obs.py"),
        "s31_run": str(s31_run), "control": measured["control"].get("s31"),
        "axes": {"split": "staged_harness", "receiver_stage": "o2_pilot_ls", "ue_realization": "replayed channel, no UE",
                 "host_stack": "DeepSig-OCUDU", "sensing_source": config["observer"]["source"],
                 "placement": "one host, two processes", "transport": "obs-ring/v1 in host shared memory",
                 "consumer": "obs_tail, which keeps what it reads and runs no sensing stage",
                 "gpu_sharing": "the consumer is not a CUDA client"},
        "runs": [{"name": run.name, "directory": str(run.directory.relative_to(out)), "status": run.status,
                  "plan_sha256": digest(run.directory / "plan.json") if (run.directory / "plan.json").is_file() else None,
                  "mode": run.plan["evidence"]["mode"] if run.plan else None,
                  "tap": run.plan["observer"]["tap"] if run.plan else None,
                  "ring": run.plan["ring"] if run.plan else None,
                  "consumers": [{key: (str(value) if key == "archive" else value) for key, value in consumer.items()}
                                for consumer in run.consumers],
                  "consumer_statuses": run.statuses} for run in runs],
        "adapter": {"executables": {name: {"path": str(tools[name]), "sha256": receipt["executables"][name]["sha256"]}
                                    for name in TOOLS},
                    "receipt": "adapter-receipt.json", "ocudu_build": receipt["ocudu_build"],
                    "built_with_cuda": receipt["pin"]["cuda"]},
        "host_rehearsal": args.host_rehearsal,
        "environment": {key: os.environ[key] for key in ("VIRTUAL_ENV", "CUDA_VISIBLE_DEVICES",
                                                         "OCUDU_PUSCH_ACCELERATION_TIMING") if key in os.environ},
        "judgment": grade["judgment"], "reason": grade["reason"]})
    summary = (f"snapshot_ring: {grade['cases_passed']} of {grade['cases']} cases passed, "
               f"{grade['checks_passed']} of {grade['checks']} checks"
               + (", host rehearsal: the hardware gate stays open" if args.host_rehearsal else ""))
    (out / "grade.txt").write_text("\n".join(grader.lines + [summary]) + "\n", encoding="ascii")
    print(summary)
    print(f"Run artifacts: {out}")
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
