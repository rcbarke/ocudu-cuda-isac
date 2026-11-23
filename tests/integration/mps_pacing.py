#!/usr/bin/env python3
"""Sprint 4.1 integration driver (selector mps_pacing). Hardware command, not part of unit discovery.

It runs the receiver and the sensor as two processes and two MPS clients on one GPU. The
receiver is the staged harness pilot_ls on a paced plan of version 4: validation entries of the
episode set of the accepted Sprint 3.5 run, released at the absolute times of their slots. The
sensor is sense_host with the TensorRT engine that the accepted Sprint 3.6 run built from the
artifact of Sprint 3.5. The launcher is scripts/run_m1.py. Every arm offers the same realization
on the same schedule. Twelve cases, each graded on its own:

  build         the adapter with the route to the receiver's estimate and with the pacing core,
                and the build of the sensing core, verify against their receipts and the tree
  control       the accepted Sprint 3.6 run with its engine, the accepted Sprint 3.5 run with its
                artifact, its split and its transmit grids, and the entries this gate replays:
                none of the held-out group; every tool of the adapter but pilot_ls has the bytes
                of the adapter that the Sprint 3.5 run verified
  mps           one MPS control daemon on one device named by its UUID, no client before the
                first arm, and a process that is no client is refused the label with the list
  schedule      one channel plan and one replay plan for every arm, two plans of version 4 around
                them, and one offered schedule: a release per uplink slot
  dsp_only      the receiver alone, tap off
  capture_only  both receiver stages exported and published, read and released by obs_tail,
                nothing inferred
  ai_only       no receiver: the records the capture-only arm kept, published again at the pace
                of their slots, to sense_host
  concurrent    the receiver and sense_host together: both process identifiers in one reading of
                the client list, the sensor's context before the receiver, the same decisions as
                the ai-only arm, and both at work over a common interval
  contention    the arrival counts of every arm, and the PUSCH service-time tail of the
                capture-only arm against the concurrent arm by the rule of the configuration;
                where the rule is exceeded the sensor-side controls are applied in their order
  runtime       the native latency beside the receiver, by the cuSense boundary and from the
                release of the first slot of a window, and the same branch on the reference
                runtime as a third arm
  faults        a sensor that is killed while the receiver runs, and a consumer that takes over:
                the receiver decodes every released grid, and what was lost is counted
  records       mps_pacing.json: what every arm offered, delivered and took, recorded and not
                graded; the process inventory; time slicing as its own labeled control

    python tests/integration/mps_pacing.py --config configs/integration/mps_pacing.json \\
        --s36-run results/runs/s36-spark-001 --output-dir results/runs/s41-spark-001/mps_pacing

The output directory must be new. The exit status is 0 only when every check of every case
passed. Latencies and tails are recorded, never graded: Sprint 4.6 calibrates on these arms.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import native_adapter  # noqa: E402
import native_core  # noqa: E402
import receiver_patch  # noqa: E402
import run_m1  # noqa: E402
from ocudu_cuda_isac.episodes import grid_apply as replay  # noqa: E402
from ocudu_cuda_isac.episodes import mps_pacing as mp  # noqa: E402
from ocudu_cuda_isac.episodes import native_estimate as ne  # noqa: E402
from ocudu_cuda_isac.episodes import replay_pusch as rp  # noqa: E402
from ocudu_cuda_isac.episodes import snapshot_ring as sr  # noqa: E402
from ocudu_cuda_isac.episodes.mps_pacing import MpsPacingError  # noqa: E402

CONFIG_SCHEMA = "integration-config/mps_pacing/v1"
GRADE_SCHEMA = "integration-grade/mps_pacing/v1"
RECORD_SCHEMA = "mps-pacing/v1"
RECORD_FILE = "mps_pacing.json"
CASES = ("build", "control", "mps", "schedule", "dsp_only", "capture_only", "ai_only", "concurrent", "contention",
         "runtime", "faults", "records")
CONFIG_FIELDS = {"schema", "selector", "adapter_build", "core_build", "control", "episodes", "repetitions", "warmup",
                 "pacing", "rings", "estimate", "receiver", "sensor", "cpus", "mps", "contention", "faults",
                 "time_slicing_control", "timeouts", "expect"}
ADAPTER_TOOLS = ("pilot_ls", "obs_tail")
CORE_TOOLS = ("sense_host", "sense_feed")
UNTOUCHED = ("grant_fixture", "grid_apply", "replay_pusch", "obs_tail", "obs_feed")
ORDERS = (("dsp_only", "capture_only", "ai_only", "concurrent"), ("concurrent", "capture_only", "ai_only", "dsp_only"),
          ("capture_only", "concurrent", "dsp_only", "ai_only"))
CHANNEL_PLAN, REPLAY_PLAN, PLAN_TAP, PLAN_DSP = "channel-plan.json", "replay-plan.json", "plan-tap.json", "plan-dsp.json"


class ConfigError(ValueError):
    """The configuration or the run setup is unusable. Nothing was graded."""


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text(value, digits=1):
    return "none" if value is None else f"{value:.{digits}f}"


class Grade:
    """Collects PASS, FAIL and NOTE lines. A case passes only when it has no FAIL line."""

    def __init__(self):
        self.lines, self.results = [], []

    def check(self, case, name, passed, detail):
        self.results.append({"case": case, "check": name, "passed": bool(passed), "detail": detail})
        self._line(("PASS" if passed else "FAIL") + f"  S4.1 {case} {name}: {detail}")
        return bool(passed)

    def note(self, case, said):
        self._line(f"NOTE  S4.1 {case} {said}")

    def _line(self, line):
        self.lines.append(line)
        print(line, flush=True)


def load_config(path):
    try:
        config = json.loads(Path(path).read_text(encoding="ascii"))
    except (OSError, ValueError) as error:
        raise ConfigError(f"cannot read the configuration: {error}") from None
    if not isinstance(config, dict) or set(config) != CONFIG_FIELDS or config["schema"] != CONFIG_SCHEMA \
            or config["selector"] != "mps_pacing":
        raise ConfigError(f"the configuration must be a {CONFIG_SCHEMA} object with exactly: "
                          + ", ".join(sorted(CONFIG_FIELDS)))
    try:
        episodes, pacing, expect = config["episodes"], config["pacing"], config["expect"]
        if episodes["group"] not in ("training", "validation"):
            raise ConfigError("episodes.group is training or validation: this gate reads no held-out episode before "
                              "Sprint 4.6 has frozen the profile")
        numbers = [episodes["pairs"], config["repetitions"], config["warmup"], pacing["slot_us"], pacing["gap_slots"],
                   pacing["allowance_us"], pacing["latest_start_us"], pacing["start_delay_us"],
                   config["faults"]["after_ms"], config["mps"]["inventory_period_ms"], config["mps"]["thread_pct"],
                   config["contention"]["control_repetitions"], expect["min_windows"], expect["max_skipped_releases"],
                   config["sensor"]["ready_timeout_ms"]] + list(config["sensor"]["budget_us"])
        if any(type(value) is not int or value < 0 for value in numbers) or episodes["pairs"] < 1 \
                or config["repetitions"] < 1 or len(config["sensor"]["budget_us"]) != 4:
            raise ConfigError("a count of the configuration is no integer that is not negative, or no repetition is run")
        if pacing["status"] != "proposed" or pacing["latest_start_us"] > pacing["allowance_us"]:
            raise ConfigError('pacing.status is "proposed" until Sprint 4.6 calibrates it, and the latest start lies '
                              "inside the allowance")
        lead = pacing.get("lead_in")
        if lead is not None and (not isinstance(lead, dict) or set(lead) != set(mp.LEAD_IN_FIELDS)
                                 or any(type(lead[key]) is not int or lead[key] < 1 for key in mp.LEAD_IN_FIELDS)):
            raise ConfigError("pacing.lead_in is null, or names how many of the first releases are decoded ahead of the "
                              "schedule and by what lead: " + ", ".join(mp.LEAD_IN_FIELDS))
        mp.check_rule(config["contention"]["rule"])
        if tuple(config["contention"]["order"]) != mp.CONTROLS:
            raise ConfigError("contention.order is the order of the controls: " + ", ".join(mp.CONTROLS))
        if config["receiver"] != {"backends": ["cuda"], "keep_grids": False}:
            raise ConfigError('a timed arm decodes on the CUDA processor alone and keeps no grid: receiver is '
                              '{"backends": ["cuda"], "keep_grids": false}')
        for name, ring in config["rings"].items():
            if ring["stream"] != name or ring["wait"] != "spin" \
                    or ring["slots"] != sr.ring_spec(ring["window_records"], ring["burst_records"])["slots"]:
                raise ConfigError(f"rings.{name} names its stream, the wait mode spin of a timed run, and the slot count "
                                  "that follows from its window and its burst: no arm enlarges a ring")
        if set(config["rings"]) != {"o2", "o3"} or set(config["cpus"]) != {"receiver", "sensor", "o2", "tools"}:
            raise ConfigError("rings names o2 and o3, and cpus names receiver, sensor, o2 and tools")
        phy = set(run_m1.parse_cpus(config["cpus"]["receiver"]))
        if any(phy & set(run_m1.parse_cpus(config["cpus"][role])) for role in ("sensor", "o2", "tools")):
            raise ConfigError("the sensor, the consumer of the O2 ring and the tools run outside the CPU set of the "
                              "receiver")
        if not 1 <= config["mps"]["thread_pct"] <= 100 or not 0 <= expect["min_decision_agreement"] <= 1 \
                or not expect["logit_tolerance"] > 0 or type(config["time_slicing_control"]) is not bool \
                or config["sensor"]["o2_consumer"] not in ("presence", "obs_tail") \
                or config["sensor"]["reference_device"] not in ("cuda", "cpu"):
            raise ConfigError("a setting of the configuration is outside its range or is not one of its choices")
    except (KeyError, TypeError, MpsPacingError, run_m1.LaunchError, sr.SnapshotRingError) as error:
        raise ConfigError(f"the configuration is incomplete or unusable: {error!r}") from None
    return config


# ---------------------------------------------------------------- setup and one arm
class Setup:
    """What every case shares: the configuration, the builds, the controls, the plans and the arms that ran."""

    def __init__(self, config, out, adapter, core, rehearsal):
        self.config, self.out, self.adapter, self.core, self.rehearsal = config, out, adapter, core, rehearsal
        self.tools = {name: Path(adapter["executables"][name]["path"]) for name in ADAPTER_TOOLS}
        self.tools.update({name: Path(core["executables"][name]["path"]) for name in CORE_TOOLS})
        self.device = "host" if rehearsal else "cuda"
        self.engine = "host_reference" if rehearsal else "tensorrt"
        self.host, self.mps = run_m1.Host(), None
        self.s36_out = self.s35_out = self.engine_dir = self.artifact_dir = self.artifact = None
        self.table = self.rows = self.bindings = self.observer = self.head = None
        self.plans, self.opportunities, self.windows = {}, 0, 0
        self.arms, self.inventory = [], None

    def of(self, arm, control="mps_default", fault=None, sharing="mps"):
        return [item for item in self.arms if item.arm == arm and item.control == control and item.fault == fault
                and item.sharing == sharing and not item.warmup]


class Arm:
    """One launch of one arm: its specification, its record and what its processes wrote, verified."""

    def __init__(self, arm, label, directory, control="mps_default", fault=None, sharing="mps", warmup=False):
        self.arm, self.label, self.directory, self.control = arm, label, directory, control
        self.fault, self.sharing, self.warmup = fault, sharing, warmup
        self.spec = self.launch = self.output = self.sensor = self.feed = None
        self.archives, self.problems = {}, []

    def ok(self):
        return not self.problems and self.launch is not None

    def said(self):
        return "; ".join(self.problems[:2]) or "clean"


def last_json(path):
    """The last line of a log that is a JSON object, or None."""
    try:
        lines = Path(path).read_text(errors="replace").strip().splitlines()
        return json.loads(lines[-1]) if len(lines) > 1 else None
    except (OSError, ValueError, IndexError):
        return None


def build_spec(setup, item, feed_archive=None):
    """The specification of one launch. Everything of it is known before a process exists."""
    config, tools, where, cpus = setup.config, setup.tools, item.directory, setup.config["cpus"]
    count = setup.opportunities + 64
    native, plan = where / "native", setup.plans["dsp" if item.arm == "dsp_only" else "tap"]
    env, release, allowance = run_m1.control_settings(item.control, config["mps"]["thread_pct"],
                                                      setup.plans.get("after_radio_us", 0))
    consumers = []

    def sense_host(role, socket, output, ready):
        command = run_m1.sense_host_command(
            tools["sense_host"], socket, setup.artifact_dir, output, ready, setup.artifact.sha256, setup.engine_dir,
            setup.device, setup.engine, config["sensor"]["wait"], release, allowance, config["sensor"]["budget_us"],
            2 * count + 1024, count)
        return run_m1.process_spec(role, "sense_host", command, not setup.rehearsal, cpus["sensor"], env, ready)

    def tail(role, socket, archive):
        return run_m1.process_spec(role, "obs_tail", run_m1.tail_command(tools["obs_tail"], socket, archive, count), False,
                                   cpus["o2"] if role == "o2" else cpus["sensor"])

    if item.arm == "ai_only":
        socket = where / "obs.sock"
        ring = config["rings"]["o3"]
        producer = run_m1.process_spec("producer", "sense_feed", run_m1.feed_command(
            tools["sense_feed"], socket, feed_archive, ring["slots"], config["pacing"]["slot_us"],
            config["pacing"]["gap_slots"]), False, cpus["tools"])
        consumers.append(sense_host("sensor", socket, where / "sensor", where / "sensor.ready"))
    else:
        producer = run_m1.process_spec("producer", "pilot_ls", run_m1.receiver_command(tools["pilot_ls"], plan["path"], native),
                                       not setup.rehearsal, cpus["receiver"])
        o2, o3 = sr.socket_path(native, plan["plan"]) if plan["plan"]["ring"] else None, None
        if plan["plan"]["estimate"]:
            o3 = ne.socket_path(native, plan["plan"])
        if item.arm == "capture_only":
            consumers += [tail("o3", o3, where / "tail-o3"), tail("o2", o2, where / "tail-o2")]
        elif item.arm == "concurrent":
            consumers.append(sense_host("sensor", o3, where / "sensor", where / "sensor.ready"))
        elif item.arm == "concurrent_reference":
            device = None if setup.rehearsal else config["sensor"]["reference_device"]
            command = run_m1.cusense_command(sys.executable, o3, setup.artifact_dir, where / "sensor",
                                             where / "sensor.ready", device, config["sensor"]["wait"])
            consumers.append(run_m1.process_spec("sensor", "cusense", command, device == "cuda", cpus["sensor"], env,
                                                 where / "sensor.ready", python_path=True))
        if item.arm in ("concurrent", "concurrent_reference"):
            if setup.head is not None and config["sensor"]["o2_consumer"] == "presence" and item.arm == "concurrent":
                command = run_m1.presence_command(sys.executable, o2, setup.head["spec"], setup.head["artifact"],
                                                  where / "delay-doppler", where / "delay-doppler.ready", "cpu",
                                                  config["sensor"]["wait"])
                consumers.append(run_m1.process_spec("o2", "presence", command, False, cpus["o2"], None,
                                                     where / "delay-doppler.ready", python_path=True))
            else:
                consumers.append(tail("o2", o2, where / "tail-o2"))
    fault = None
    if item.fault is not None:
        again = None
        if item.fault == "consumer_restart":
            again = tail("takeover", ne.socket_path(native, plan["plan"]), where / "tail-takeover")
        fault = {"kind": item.fault, "role": "sensor", "after_ms": config["faults"]["after_ms"], "restart": again}
    return run_m1.arm_spec(item.arm, item.label, producer, consumers, item.sharing, fault, config["timeouts"],
                           config["mps"]["inventory_period_ms"])


def execute(setup, item, feed_archive=None, cusense=None, obs=None, presence=None):
    """Launch one arm and verify what it wrote. Whatever does not verify becomes a problem of the arm."""
    where = item.directory
    where.mkdir(parents=True)
    try:
        item.spec = build_spec(setup, item, feed_archive)
        for process in item.spec["consumers"]:
            if "--output-dir" in process["command"]:
                Path(process["command"][process["command"].index("--output-dir") + 1]).mkdir(parents=True)
        if item.arm != "ai_only":
            (where / "native").mkdir()
        if item.spec["fault"] and item.spec["fault"]["restart"]:
            (where / "tail-takeover").mkdir()
        write_json(where / "spec.json", item.spec)
        item.launch = run_m1.launch(item.spec, where, setup.host, setup.mps if item.sharing == "mps" else None)
        write_json(where / "launch.json", item.launch)
    except (run_m1.LaunchError, MpsPacingError, OSError) as error:
        item.problems.append(f"the arm was not launched: {error}")
        return item
    allowed = ("sensor",) if item.fault else ()
    statuses = {entry["role"]: entry["status"] for entry in item.launch["processes"]}
    item.problems += item.launch["problems"]
    for role, status in statuses.items():
        if status != 0 and role not in allowed:
            item.problems.append(f"the {role} process ended with status {status} ({where.name}/{role}.log)")
    try:
        if item.arm != "ai_only" and statuses.get("producer") == 0:
            item.output = mp.open_output(where / "native")
            wrong = mp.offered_matches(item.output, setup.plans["dsp" if item.arm == "dsp_only" else "tap"]["plan"])
            item.problems += wrong
        elif item.arm == "ai_only":
            item.feed = last_json(where / "producer.log")
            if item.feed is None or "published" not in item.feed:
                item.problems.append("sense_feed printed no report")
        for entry in item.launch["processes"]:
            role, kind = entry["role"], entry["kind"]
            if entry["status"] != 0:
                continue
            if kind == "obs_tail":
                item.archives[role] = obs.open_archive(where / ("tail-" + role))
            elif kind in ("sense_host", "cusense"):
                item.sensor = cusense.open_output(where / "sensor")
            elif kind == "presence":
                item.archives["delay-doppler"] = presence.open_output(where / "delay-doppler")
    except Exception as error:  # every reader has its own error type: whatever one raises is a problem of the arm
        item.problems.append(f"an output does not verify: {type(error).__name__}: {error}")
    return item


# ---------------------------------------------------------------- cases before the first arm
def run_build(setup, grader):
    adapter, core, name = setup.adapter, setup.core, "build"
    try:
        current, problem = receiver_patch.identity(receiver_patch.load_series()), None
    except receiver_patch.PatchError as error:
        current, problem = None, str(error)
    route = adapter.get("o3_route") or {}
    routed = current is not None and receiver_patch.same_series(adapter.get("patch") or {}, current) \
        and route.get("compiled") is True and adapter["pin"]["cuda"] is True
    grader.check(name, "adapter", routed,
                 problem or (f"the adapter {adapter['adapter_build']} verifies against its receipt and the tree: linked "
                             f"against the receiver with the series {current['series_sha256'][:16]} of patches/ocudu, "
                             f"built with ENABLE_CUDA, pilot_ls {adapter['executables']['pilot_ls']['sha256'][:12]} with "
                             f"the route to the estimate, {adapter['tests_run']} adapter tests, "
                             f"{adapter['compiler_warnings']} compiler warnings" if routed else
                             "the receipt of the adapter names no route to the receiver's estimate, or another series "
                             "than patches/ocudu holds. Build it: make adapter-build-o3"))
    sources = [path for path in adapter.get("sources", {}) if "/pacing/" in path]
    status, said = setup.host.run([setup.tools["pilot_ls"], "--pacing"], timeout=120)
    # A plan with a lead-in is read only by a tool that says it knows one.
    reads = status == 0 and mp.PLAN_SCHEMA in said \
        and (setup.config["pacing"].get("lead_in") is None or "lead-in" in said)
    grader.check(name, "pacing core", len(sources) >= 4 and reads,
                 f"the receipt binds {len(sources)} files of native/ocudu_adapter/pacing, and pilot_ls --pacing answers: "
                 f"{said.strip()[:200]}" if reads else
                 f"pilot_ls --pacing ended with status {status} and names no plan of version 4, or no lead-in although "
                 f"the configuration asks for one: {said.strip()[:200]}. Rebuild: make adapter-build-o3")
    if setup.rehearsal:
        grader.note(name, f"rehearsal: the core is built for the device {core['device']}; no kernel runs on a GPU")
    else:
        wanted = setup.config["expect"]["cuda_architecture"]
        grader.check(name, "core", core["device"] == "cuda" and bool(core["tensorrt"])
                     and str(wanted) in str(core["cuda_architectures"]),
                     f"the sensing core {core['build_dir']} verifies against its receipt and the tree: sources "
                     f"{core['sources_sha256'][:16]}, device {core['device']}, CUDA architecture "
                     f"{core['cuda_architectures']} (expected {wanted}), TensorRT "
                     f"{(core['tensorrt'] or {}).get('version')}, sense_host "
                     f"{core['executables']['sense_host']['sha256'][:12]}, sense_feed "
                     f"{core['executables']['sense_feed']['sha256'][:12]}")
    return {"adapter": {"build": adapter["adapter_build"], "pilot_ls": adapter["executables"]["pilot_ls"]["sha256"],
                        "pacing_sources": len(sources)},
            "core": {"sources_sha256": core["sources_sha256"], "device": core["device"]}}


def run_control(setup, grader, s36_run, s35_run, cusense, presence, evaluator):
    name, config, measured = "control", setup.config, {}
    # ---- The accepted Sprint 3.6 run: its engine, and the Sprint 3.5 run it was built from.
    if s36_run is not None:
        out = s36_run / config["control"]["sense_core"]
        out = out if (out / "grade.json").is_file() else s36_run
        grade, manifest = json.loads((out / "grade.json").read_text()), json.loads((out / "manifest.json").read_text())
        summary = (s36_run / "summary.txt").read_text(errors="replace") if (s36_run / "summary.txt").is_file() else ""
        accepted = grade.get("judgment") == "pass" and grade.get("hardware_gate") == "closed" \
            and "Sprint 3.6: ACCEPTED on this DGX Spark" in summary and "\nFAIL  " not in summary
        engine = out / "engine" / "engine"
        named = json.loads((engine / "engine.json").read_text()) if (engine / "engine.json").is_file() else {}
        setup.s36_out, setup.engine_dir = out, engine if named else None
        same = grade.get("core_sources_sha256") == setup.core["sources_sha256"]
        grader.check(name, "sense core run", accepted and bool(named),
                     f"the Sprint 3.6 run {s36_run.name} is ACCEPTED on this DGX Spark with its hardware gate closed; its "
                     f"engine {str(named.get('sha256'))[:16]} was built with TensorRT "
                     f"{(named.get('library') or {}).get('version')} as {named.get('precision')} for the artifact "
                     f"{str(named.get('artifact_sha256'))[:16]}; the core of this gate has "
                     + ("the sources of that run" if same else
                        f"other sources than that run, {setup.core['sources_sha256'][:16]} against "
                        f"{str(grade.get('core_sources_sha256'))[:16]}: sense_host loads the engine by its manifest")
                     if accepted and named else
                     f"{s36_run} holds no verdict ACCEPTED on this DGX Spark with a closed hardware gate and an engine")
        s35_run = s35_run or Path(manifest["s35_run"])
        measured["s36"] = {"run": s36_run.name, "engine_sha256": named.get("sha256"), "same_core_sources": same}
    else:
        grader.note(name, "rehearsal: no Sprint 3.6 run and no engine: the sensor runs the model in C++ on the CPU")
    # ---- The accepted Sprint 3.5 run: the artifact, the split and the transmit grids.
    out = s35_run / config["control"]["cusense_frontend"]
    out = out if (out / "grade.json").is_file() else s35_run
    setup.s35_out = out
    grade, manifest = json.loads((out / "grade.json").read_text()), json.loads((out / "manifest.json").read_text())
    recover = json.loads((out / "recover.json").read_text())
    freeze = json.loads((out / recover["freeze"]).read_text())
    setup.artifact_dir = out / recover["artifact"]
    setup.artifact = cusense.open_artifact(setup.artifact_dir, freeze["artifact"]["sha256"])
    passed = grade.get("judgment") == "pass" and grade.get("hardware_gate") == "closed"
    engine_of = None if setup.engine_dir is None else json.loads((setup.engine_dir / "engine.json").read_text())
    bound = engine_of is None or engine_of.get("artifact_sha256") == setup.artifact.sha256
    card = setup.artifact.manifest
    grader.check(name, "artifact", passed and bound,
                 f"the Sprint 3.5 driver graded {s35_run.name} as {grade.get('judgment')} with its hardware gate "
                 f"{str(grade.get('hardware_gate')).split(':')[0]}; the artifact {card['model_id']} "
                 f"{setup.artifact.sha256[:16]} is the one its freeze record names, input {card['entries']['input']} of "
                 f"shape {card['input']['shape']}, threshold logit {card['threshold']['logit']:.6f}"
                 + ("" if engine_of is None else ", and the engine of the Sprint 3.6 run was built from it")
                 if passed and bound else
                 f"the Sprint 3.5 run is not graded pass with a closed gate, or the engine names the artifact "
                 f"{str((engine_of or {}).get('artifact_sha256'))[:16]} and the freeze record {setup.artifact.sha256[:16]}")
    # ---- This sprint edits one tool of the adapter. Every other one is rebuilt with the bytes of that run.
    earlier = json.loads((out / "adapter-receipt.json").read_text())["executables"] \
        if (out / "adapter-receipt.json").is_file() else {}
    now = setup.adapter["executables"]
    changed = [tool for tool in UNTOUCHED if tool in earlier and tool in now and earlier[tool]["sha256"] != now[tool]["sha256"]]
    compared = [tool for tool in UNTOUCHED if tool in earlier and tool in now]
    if compared and not setup.rehearsal:
        grader.check(name, "receiver side", not changed,
                     f"{len(compared)} tools of the adapter, {', '.join(compared)}, have the bytes of the adapter that the "
                     f"Sprint 3.5 run verified; pilot_ls, the one tool this sprint edits, has "
                     f"{now['pilot_ls']['sha256'][:12]} against {str(earlier.get('pilot_ls', {}).get('sha256'))[:12]}"
                     if not changed else
                     f"{', '.join(changed)} differ from the adapter of the Sprint 3.5 run: this sprint edits pilot_ls only")
    else:
        grader.note(name, "the adapter receipt of the Sprint 3.5 run is not compared: " +
                    ("a rehearsal runs doubles of the tools" if setup.rehearsal else "that run holds none"))
    # ---- The entries: one run of one grid gain from the configured group, never the held-out group.
    setup.table = evaluator.check_table(json.loads((out / "split.json").read_text()))
    group, pairs = config["episodes"]["group"], config["episodes"]["pairs"]
    chunks = evaluator.chunks(setup.table, group, pairs)
    setup.rows = chunks[0] if chunks else []
    held = [row["episode_id"] for row in setup.rows if row["group"] == "held_out"]
    setup.opportunities = 2 * sum(row["opportunities"] for row in setup.rows)
    setup.windows = 2 * sum(row["periods"] for row in setup.rows)
    grader.check(name, "episodes", setup.rows and not held and all(Path(row["directory"]).is_dir() for row in setup.rows),
                 f"{len(setup.rows)} {group} pairs of the set {setup.table['set_id']} with one grid gain, "
                 f"{setup.opportunities} opportunities and {setup.windows} windows of one period; "
                 f"{setup.table['counts']['held_out']} held-out pairs of the set are not read by this gate"
                 if setup.rows and not held else f"the set gives no {group} entry on this host, or a held-out one")
    # ---- The transmit grids: those the plans of a run of that gate bind.
    runs = [run for run in manifest.get("runs", []) if run.get("status") == 0 and run.get("plan_sha256")]
    directory = out / runs[0]["directory"]
    plan = ne.load_plan(directory / "plan.json")
    channel = replay.load_plan(rp.load_plan(plan["replay_plan"]["path"])["channel_plan"]["path"])
    setup.bindings = {grant["grant_id"]: {"x_sha256": grant["x"]["sha256"], "store": str(Path(grant["x"]["path"]).parent)}
                      for grant in channel["grants"]}
    setup.observer = plan["observer"]
    whole = digest(directory / "plan.json") == runs[0]["plan_sha256"] \
        and all(Path(item["store"]).is_dir() for item in setup.bindings.values())
    grader.check(name, "transmit grids", whole,
                 f"the plan of the run {runs[0]['name']} of that gate has the hash its manifest names and binds the "
                 f"transmit grid of {len(setup.bindings)} grant(s): {', '.join(sorted(setup.bindings))}; observer epoch "
                 f"{setup.observer['epoch']}, source {setup.observer['source']}")
    # ---- The delay-Doppler branch: the frozen head of the Sprint 3.4 run, where this host holds it.
    try:
        s34 = Path(manifest["s34_run"])
        s34 = s34 / config["control"]["presence_head"] if (s34 / config["control"]["presence_head"] / "spec.json").is_file() \
            else s34
        head = presence.open_artifact(s34 / "train" / "artifact")
        setup.head = {"spec": s34 / "spec.json", "artifact": s34 / "train" / "artifact", "sha256": head.sha256}
        grader.note(name, f"the delay-Doppler branch reads the O2 ring with the head {head.manifest['head_id']} "
                          f"{head.sha256[:16]} of the Sprint 3.4 run {Path(manifest['s34_run']).name}, on the CPU")
    except Exception as error:  # the branch is optional here: the O2 ring is then read and released by obs_tail
        grader.note(name, f"no frozen head of a Sprint 3.4 run on this host ({type(error).__name__}): the O2 ring of the "
                          "concurrent arm is read and released by obs_tail")
    measured.update(s35={"run": s35_run.name, "artifact_sha256": setup.artifact.sha256},
                    episodes={"group": group, "pairs": len(setup.rows), "opportunities": setup.opportunities,
                              "windows": setup.windows, "entries": [row["episode_id"] for row in setup.rows]},
                    head=None if setup.head is None else setup.head["sha256"])
    return measured


def run_mps(setup, grader):
    name, config = "mps", setup.config["mps"]
    setup.mps = run_m1.Mps(setup.host, ROOT / config["pipe_dir"], ROOT / config["log_dir"])
    state = setup.mps.start()
    found = setup.inventory = setup.mps.inventory()
    problems = [problem for problem in mp.inventory_problems(found) if "servers are listed" not in problem]
    clients = [pid for listed in found["clients"].values() for pid in listed]
    grader.check(name, "daemon", not problems and not clients,
                 f"the MPS control daemon on {found['daemon']['pipe_directory']} was {state} and answers; it serves one "
                 f"device, {found['devices'][0]['name']} {found['devices'][0]['uuid']}; {len(found['servers'])} server(s) "
                 "and no client before the first arm" if not problems and not clients else
                 "; ".join(problems) or f"{len(clients)} process(es) are clients of the server before the first arm: "
                                        f"{clients}")
    # A process that is no client is refused the label, with the client list in the message.
    try:
        mp.verify_clients(dict(found, servers=found["servers"] or [0], clients=found["clients"] or {"0": []}),
                          {"probe": os.getpid()})
        refused = None
    except MpsPacingError as error:
        refused = str(error)
    grader.check(name, "refusal", refused is not None and "Client list of the server" in refused
                 and f"process {os.getpid()} is not a client" in refused,
                 f"this driver, process {os.getpid()}, is no client of the server, and the label is refused: "
                 f"\"{(refused or '')[:260]}\"")
    settings = found["active_thread_percentage"]
    priority = found["default_client_priority"] or {}
    grader.check(name, "settings", settings["default"] is not None,
                 f"read back from the control interface: default active thread percentage {settings['default']}, default "
                 f"client priority " + (f"{priority.get('text')}" if priority.get("status") == 0 else
                                        f"not readable on this release (status {priority.get('status')}: "
                                        f"{str(priority.get('text'))[:80]})"))
    return {"state": state, "devices": found["devices"], "default_active_thread_percentage": settings["default"],
            "default_client_priority": priority, "pipe_directory": found["daemon"]["pipe_directory"]}


def run_schedule(setup, grader):
    name, config, out = "schedule", setup.config, setup.out / "plans"
    out.mkdir()
    run_id, pacing = setup.out.parent.name if setup.out.name == "mps_pacing" else setup.out.name, config["pacing"]
    channel = replay.entry_plan(run_id, [row["directory"] for row in setup.rows], bindings=setup.bindings)
    replay.write_plan(channel, out / CHANNEL_PLAN)
    rp.write_plan(rp.build_plan(run_id, out / CHANNEL_PLAN, backends=config["receiver"]["backends"]), out / REPLAY_PLAN)
    opportunities = mp.opportunities_of(out / REPLAY_PLAN)
    frame = 10 * setup.artifact.manifest["input"]["spec"]["radio"]["scs_khz"] // 15
    offsets = mp.release_offsets(opportunities, pacing["slot_us"], frame, pacing["gap_slots"])
    lead = pacing.get("lead_in")
    schedule = mp.pacing_spec(offsets, pacing["allowance_us"], pacing["latest_start_us"], pacing["start_delay_us"], lead)
    rings = {stream: sr.ring_spec(ring["window_records"], ring["burst_records"], ring["wait"], "required",
                                  ring["attach_timeout_ms"], ring["drain_ms"], stream)
             for stream, ring in config["rings"].items()}
    estimate = ne.estimate_spec(config["estimate"]["copy_stream"], config["estimate"]["landing"],
                                config["estimate"]["timeout_us"], rings["o3"], setup.observer["source"])
    made = {"tap": mp.build_plan(run_id, out / REPLAY_PLAN, schedule, rings["o2"], estimate,
                                 epoch=setup.observer["epoch"], source=setup.observer["source"]),
            "dsp": mp.build_plan(run_id, out / REPLAY_PLAN, schedule, None, None, tap="none",
                                 epoch=setup.observer["epoch"])}
    for key, file in (("tap", PLAN_TAP), ("dsp", PLAN_DSP)):
        mp.write_plan(made[key], out / file)
        setup.plans[key] = {"path": out / file, "plan": made[key], "sha256": digest(out / file)}
    setup.opportunities = len(opportunities)
    gaps = sorted({b - a for a, b in zip(offsets, offsets[1:])})
    grader.check(name, "plans", len(offsets) == len(opportunities) == 2 * sum(row["opportunities"] for row in setup.rows),
                 f"two plans of version 4 in the mode timed around one replay plan on the CUDA processor: {PLAN_DSP} with "
                 f"the tap off and {PLAN_TAP} with both stages on rings of {rings['o2']['slots']} and "
                 f"{rings['o3']['slots']} slots in the wait mode spin; {len(opportunities)} opportunities, one release "
                 f"each, over {offsets[-1] / 1000.0:.1f} ms of offered schedule with distances of "
                 f"{', '.join(str(gap) for gap in gaps)} us; processing allowance {pacing['allowance_us']} us and latest "
                 f"start {pacing['latest_start_us']} us, both {pacing['status']}; "
                 + (f"a lead-in of {lead['releases']} decodes, the first releases once more, {lead['lead_us']} us ahead "
                    f"of their own release times, from {text(-mp.lead_offsets(schedule)[0] / 1000.0)} to "
                    f"{text(-mp.lead_offsets(schedule)[-1] / 1000.0)} ms before the schedule starts, {pacing['status']}"
                    if lead else "no lead-in: the first decode of a run is the first use of its processor"))
    same = made["tap"]["replay_plan"] == made["dsp"]["replay_plan"] and made["tap"]["pacing"] == made["dsp"]["pacing"]
    offered = hashlib.sha256(json.dumps(schedule, sort_keys=True).encode()).hexdigest()
    grader.check(name, "one realization", same,
                 f"both plans bind the replay plan {made['tap']['replay_plan']['sha256'][:16]} and the offered schedule "
                 f"{offered[:16]}: every arm decodes the same realization of the same entries at the same release times, "
                 "and the arm that replays buffered records publishes them at the pace of the same slots")
    return {"run_id": run_id, "opportunities": len(opportunities), "duration_us": offsets[-1], "distances_us": gaps,
            "offered_sha256": offered, "replay_plan_sha256": made["tap"]["replay_plan"]["sha256"],
            "plans": {key: setup.plans[key]["sha256"] for key in ("tap", "dsp")}, "pacing": dict(pacing)}


# ---------------------------------------------------------------- the arms
def campaign(setup, readers):
    """The warm-up and the repetitions of the four timed arms. The order of the arms rotates."""
    config, arms = setup.config, setup.out / "arms"
    arms.mkdir()
    for number in range(config["warmup"]):
        item = Arm("concurrent", f"warmup-{number}", arms / f"warmup-{number}-concurrent", warmup=True)
        setup.arms.append(execute(setup, item, **readers))
    for repetition in range(config["repetitions"]):
        kept = None
        for arm in ORDERS[repetition % len(ORDERS)]:
            item = Arm(arm, f"r{repetition}-{arm}", arms / f"r{repetition}-{arm}")
            if arm == "ai_only":
                if kept is None or "o3" not in kept.archives:
                    item.problems.append("the capture-only arm of this repetition kept no records of the O3 ring")
                    setup.arms.append(item)
                    continue
                setup.arms.append(execute(setup, item, kept.directory / "tail-o3", **readers))
                continue
            setup.arms.append(execute(setup, item, **readers))
            if arm == "capture_only":
                kept = item


def launch_check(grader, name, arms, roles, setup):
    """Every repetition launched clean and carries the label MPS on the proof of one reading."""
    bad = [item for item in arms if not item.ok()]
    proven = [item for item in arms if item.launch and item.launch["sharing"]["label"] == "mps"
              and (item.launch["sharing"]["verified"] or setup.rehearsal)]
    whole = arms and not bad and len(proven) == len(arms)
    proof = (proven[0].launch["sharing"]["proof"] if proven else None) or {}
    return grader.check(name, "launch", whole,
                        f"{len(arms)} repetition(s) launched by scripts/run_m1.py, every process ended with status 0; "
                        f"each carries the label MPS on one reading of the client list that names "
                        f"{' and '.join(roles)} as client(s) of the server {proof.get('server')} on the device "
                        f"{(proof.get('device') or {}).get('uuid')}" if whole else
                        f"no repetition ran" if not arms else
                        f"{len(bad)} of {len(arms)} repetition(s) did not run clean, first {(bad or arms)[0].label}: "
                        f"{(bad or arms)[0].said()}; {len(proven)} carry the label MPS")


def radio_check(setup, grader, name, arms):
    """The receiver decoded every grid it released, on its resident path, on the offered schedule."""
    expect, total = setup.config["expect"], {"released": 0, "decoded": 0, "resident": 0, "skipped": 0, "late": 0,
                                             "overruns": 0, "anomalies": 0}
    for item in arms:
        summary, counts = mp.radio_summary(item.output), item.output.pacing["counts"]
        total["released"] += counts["on_time"] + counts["late"]
        total["decoded"] += summary["decoded"]
        total["resident"] += summary["resident"]
        total["anomalies"] += len(item.output.report["anomalies"])
        for key in ("skipped", "late", "overruns"):
            total[key] += counts[key]
    offered = setup.opportunities * len(arms)
    good = total["decoded"] == total["released"] and total["released"] + total["skipped"] == offered \
        and (setup.rehearsal or total["resident"] == total["decoded"]) and not total["anomalies"]
    grader.check(name, "decodes", good,
                 f"{total['decoded']} of {total['released']} released grids gave the original bytes of their transport "
                 f"block with a CRC pass, {total['resident']} on the resident CUDA path, {total['anomalies']} anomalies; "
                 f"{total['skipped']} of {offered} offered releases were skipped and stay in the denominator")
    lateness = mp.statistics([(line["t_start_ns"] - line["t_release_ns"]) / 1000.0 for item in arms
                              for line in item.output.ran()])
    grader.check(name, "schedule kept", total["skipped"] <= expect["max_skipped_releases"] * len(arms),
                 f"the offered schedule of every repetition is the one of the plan and no job started before its "
                 f"release: {total['released'] - total['late']} on time, {total['late']} late, {total['skipped']} skipped "
                 f"(at most {expect['max_skipped_releases']} per repetition), {total['overruns']} overruns of the "
                 f"proposed allowance; a start lies {text(lateness['p50_us'], 2)} us after its release in the median, "
                 f"{text(lateness['p99_us'], 2)} us at the 99th percentile and {text(lateness['max_us'], 2)} us at most")
    total["cold_start"] = lead_in_check(setup, grader, name, arms)
    return total


def lead_in_check(setup, grader, name, arms):
    """The cold start of the receiver is a decode of the lead-in, apart from the schedule, and costs it no release."""
    lead = setup.config["pacing"].get("lead_in")
    found = [dict(mp.cold_start(item.output), label=item.label) for item in arms]
    median = mp.statistics(pooled(arms))["p50_us"]
    if lead is None:
        cold = [item["service_us"][0] for item in found if item["service_us"]]
        grader.note(name, f"no lead-in: in each of {len(found)} repetition(s) the first release is the first use of the "
                          f"processor, {text(min(cold, default=None))} to {text(max(cold, default=None))} us against "
                          f"{text(median)} us in the median, and {sum(item['skipped_first_releases'] for item in found)} "
                          "of the releases behind it were skipped. The service statistics leave that decode out")
        return found
    count = lead["releases"]
    whole = all(item["lead_in"] == lead and item["decoded"] == count and len(item["service_us"]) == count
                for item in found)
    margins = [item["before_first_release_us"] for item in found]
    ahead = whole and all(margin is not None and margin > 0 for margin in margins)
    apart = all(not item["first_use_in_schedule"] for item in found)
    skipped = sum(item["skipped_first_releases"] for item in found)
    by_decode = [[item["service_us"][k] for item in found if len(item["service_us"]) > k] for k in range(count)]
    first = [item["first_releases"][0]["service_us"] for item in found
             if item["first_releases"] and item["first_releases"][0]["service_us"] is not None]
    overran = sum(bool(item["first_releases"]) and item["first_releases"][0]["overrun"] for item in found)
    good = whole and ahead and apart and skipped == 0
    spread = "; ".join(f"decode {k + 1} {text(min(values))} to {text(max(values))} us"
                       for k, values in enumerate(by_decode[:4]) if values)
    grader.check(name, "lead-in", good,
                 f"in every repetition the {count} decodes of the lead-in gave the original bytes of their block and "
                 f"were done {text(min(margins))} us or more before the first release; no release is the first use of "
                 f"the processor, and none of the first {count} releases of a schedule was skipped. The cold start, "
                 f"recorded apart in pacing/lead_in.jsonl and a sample of no statistic: {spread}. Behind it the first "
                 f"release of the schedule took {text(min(first, default=None))} to {text(max(first, default=None))} us "
                 f"against {text(median)} us in the median, {overran} of them an overrun" if good else
                 "; ".join(reason for reason, wrong in (
                     (f"a repetition did not decode its {count} lead-in decodes to the original bytes", not whole),
                     ("a lead-in was not done before the first release of its schedule: it ended "
                      f"{text(-min((m for m in margins if m is not None), default=0.0))} us after it", whole and not ahead),
                     ("a release of a schedule is the first use of its processor although the lead-in ran", not apart),
                     (f"{skipped} of the first {count} releases of a schedule were skipped: the cold start still costs "
                      "the schedule", skipped != 0)) if wrong))
    return found


def pooled(arms):
    return [value for item in arms for value in mp.service_times(item.output)]


def service_note(grader, name, arms):
    made = mp.statistics(pooled(arms))
    busy = mp.statistics([(line["t_done_ns"] - line["t_release_ns"]) / 1000.0 for item in arms for line in item.output.ran()])
    grader.note(name, f"recorded, not graded: PUSCH service time, start of the decode to its callback, over "
                      f"{made['count']} decodes: {text(made['p50_us'])} us in the median, {text(made['p99_us'])} us at "
                      f"the 99th percentile, {text(made['max_us'])} us at most; from the release of a slot to the end of "
                      f"its job {text(busy['p50_us'])} / {text(busy['p99_us'])} / {text(busy['max_us'])} us")
    return {"service": made, "release_to_done": busy}


def run_dsp_only(setup, grader):
    name, arms = "dsp_only", setup.of("dsp_only")
    if not launch_check(grader, name, arms, ["the receiver"], setup):
        return {}
    total = radio_check(setup, grader, name, arms)
    quiet = all(item.output.base.ring is None and item.output.estimate is None for item in arms)
    grader.check(name, "tap off", quiet,
                 "the plan names no tap, no ring and no estimate: nothing is observed, copied or published, and the "
                 "receiver patch has no observer registered")
    return dict(total, **service_note(grader, name, arms))


def account_of(arms):
    total = {stream: dict.fromkeys(("eligible", "published", "dropped", "released", "pending"), 0) for stream in ("o2", "o3")}
    for item in arms:
        found = mp.arrivals(item.output)
        for stream in total:
            for key in total[stream]:
                total[stream][key] += (found[stream] or {}).get(key) or 0
    return total


def run_capture_only(setup, grader):
    name, arms = "capture_only", setup.of("capture_only")
    if not launch_check(grader, name, arms, ["the receiver"], setup):
        return {}
    total = radio_check(setup, grader, name, arms)
    account, kept, problems = account_of(arms), {"o2": 0, "o3": 0}, []
    for item in arms:
        for stream, join, output in (("o2", sr.join_archive, item.output.base), ("o3", ne.join_archive, item.output.native)):
            if stream not in item.archives:
                problems.append(f"{item.label}: no archive of the stage {stream.upper()}")
                continue
            pairs, wrong = join(output, item.archives[stream])
            kept[stream] += len(pairs)
            problems += [f"{item.label} {stream}: {problem}" for problem in wrong]
    whole = not problems and all(account[stream]["published"] == total["released"] == kept[stream]
                                 and account[stream]["dropped"] == 0 and account[stream]["pending"] == 0
                                 for stream in account)
    grader.check(name, "account", whole,
                 "; ".join(f"stage {stream.upper()}: {account[stream]['published']} published, "
                           f"{account[stream]['dropped']} dropped, {account[stream]['released']} released, "
                           f"{account[stream]['pending']} pending, {kept[stream]} kept by obs_tail" for stream in account)
                 + f", for {total['released']} released grids: the real export of both stages, read and released, with "
                   "nothing inferred; each kept record is the record of its account line" if not problems else
                 "; ".join(problems[:3]))
    return dict(total, account=account, **service_note(grader, name, arms))


def sensor_counts(arms):
    total = {"scored": 0, "gap": 0, "stale": 0, "stale_skipped": 0, "late": 0, "submissions": 0}
    for item in arms:
        counts = item.sensor.report["counts"]
        total["scored"] += len(item.sensor.scored)
        for key in ("gap", "stale", "stale_skipped", "late", "submissions"):
            total[key] += counts.get(key) or 0
    return total


def run_ai_only(setup, grader):
    name, arms = "ai_only", setup.of("ai_only")
    if not launch_check(grader, name, arms, ["the sensor"], setup):
        return {}
    fed = [item.feed for item in arms]
    offered = sum(item["offered"] for item in fed)
    grader.check(name, "buffered records", all(item["dropped"] == 0 and item["pending_at_cutoff"] == 0
                                               and item["published"] == item["offered"] for item in fed),
                 f"sense_feed published {sum(item['published'] for item in fed)} of {offered} records that the "
                 f"capture-only arm of the same repetition kept of the O3 ring, at the pace of their slots, dropped "
                 f"{sum(item['dropped'] for item in fed)} and had {sum(item['pending_at_cutoff'] for item in fed)} credits "
                 "out at its end: the real estimate of the receiver, buffered, with no receiver on the device")
    counts, runtime = sensor_counts(arms), arms[0].sensor.report["runtime"]
    wanted = setup.config["expect"]["min_windows"] * len(arms)
    native = runtime["entry"] == "native" and (setup.rehearsal or (runtime["engine_backend"] == "tensorrt"
                                                                  and runtime["timing_valid"]))
    grader.check(name, "decisions", native and counts["scored"] >= wanted and counts["stale"] == 0,
                 f"sense_host on {runtime['device']} with the {runtime['engine_backend']} backend scored "
                 f"{counts['scored']} windows (at least {wanted}) in {counts['submissions']} submissions; "
                 f"{counts['gap']} gaps, {counts['stale_skipped']} overtaken, {counts['stale']} stale; every output "
                 "verifies with the reader of the reference")
    return {"feed": fed, "sensor": counts}


def run_concurrent(setup, grader, evaluator_latency):
    name, arms, expect = "concurrent", setup.of("concurrent"), setup.config["expect"]
    if not launch_check(grader, name, arms, ["the receiver", "the sensor"], setup):
        return {}
    before = all(item.launch["sharing"]["context_before_receiver"] for item in arms)
    order = [[(event["event"], event["role"]) for event in item.launch["order"]] for item in arms]
    ready_first = all(sequence.index(("ready", "sensor")) < sequence.index(("started", "producer")) for sequence in order)
    sensors = [entry for item in arms for entry in item.launch["processes"] if entry["role"] == "sensor"]
    grader.check(name, "start order", before and ready_first and not any(entry["maps_python"] for entry in sensors),
                 "in every repetition sense_host loaded its artifact, created its CUDA context and warmed its engine, and "
                 "was a client of the server, before the receiver was started; the warm sensor maps no Python library; "
                 f"it runs on CPU {sensors[0]['cpus']['applied']} and the receiver on "
                 f"{setup.config['cpus']['receiver']}" if before and ready_first else
                 "in a repetition the sensor's context was no client of the server before the receiver was started, or "
                 "the receiver was started before the sensor was ready")
    total = radio_check(setup, grader, name, arms)
    account = account_of(arms)
    counts = sensor_counts(arms)
    grader.check(name, "account", account["o3"]["published"] == total["released"] and account["o3"]["dropped"] == 0
                 and account["o3"]["pending"] == 0 and counts["scored"] >= expect["min_windows"] * len(arms)
                 and counts["stale"] == 0,
                 f"stage O3: {account['o3']['published']} estimates published for {total['released']} released grids, "
                 f"{account['o3']['dropped']} dropped, {account['o3']['pending']} pending; sense_host scored "
                 f"{counts['scored']} windows (at least {expect['min_windows'] * len(arms)}) in {counts['submissions']} "
                 f"submissions, {counts['gap']} gaps, {counts['stale_skipped']} overtaken; stage O2: "
                 f"{account['o2']['published']} published and {account['o2']['dropped']} dropped to "
                 + ("the delay-Doppler branch on the CPU" if any("delay-doppler" in item.archives for item in arms)
                    else "obs_tail"))
    # ---- The same decisions as the sensor alone on the buffered records of the same repetition.
    alone = {item.label.split("-")[0]: item for item in setup.of("ai_only") if item.ok()}
    joined = agree = 0
    worst, flips = 0.0, []
    for item in arms:
        other = alone.get(item.label.split("-")[0])
        if other is None:
            continue
        found = mp.decision_agreement(item.sensor.scored, other.sensor.scored, expect["logit_tolerance"])
        joined, agree = joined + found["joined"], agree + found["agree"]
        worst, flips = max(worst, found["largest_logit_difference"]), flips + found["flips"]
    grader.check(name, "same decisions", joined >= 1 and agree >= expect["min_decision_agreement"] * joined,
                 f"{joined} windows were scored beside the receiver and, from the buffered records of the capture-only "
                 f"arm, by the sensor alone: {agree} decide alike, {len(flips)} flip(s), and the largest logit difference "
                 f"is {worst:.2e}. The two arms read two decodes of one grid, and the receiver's estimate is not "
                 "bit-reproducible, so the scores are compared by decision and never by digest")
    # ---- Both clients at work over a common interval, from their own marks on one clock.
    found = [mp.overlap(item.output, item.sensor.scored) for item in arms]
    common = all(entry["common_interval_ns"] is not None and entry["sensor_inside_run"] >= 1 for entry in found)
    grader.check(name, "common interval", common or setup.rehearsal,
                 f"{sum(entry['radio_intervals'] for entry in found)} decodes and "
                 f"{sum(entry['sensor_intervals'] for entry in found)} submissions of the sensor lie on one monotonic "
                 f"clock; {sum(entry['sensor_inside_run'] for entry in found)} submissions ran between the first and the "
                 f"last decode of their run, over a common interval of "
                 f"{sum((entry['common_interval_ns'] or [0, 0])[1] - (entry['common_interval_ns'] or [0, 0])[0] for entry in found) / 1e6:.0f} "
                 f"ms; {sum(entry['sensor_overlapping'] for entry in found)} submissions overlap a decode in time: a "
                 "window closes with the last uplink slot of its period, so its submission falls behind the burst"
                 if common else "rehearsal: the marks of an archive replay are not those of a live ring" if setup.rehearsal
                 else "the marks of the two clients give no common interval")
    return dict(total, account=account, sensor=counts, agreement={"joined": joined, "agree": agree, "flips": flips,
                                                                  "largest_logit_difference": worst},
                overlap=found, **service_note(grader, name, arms))


# ---------------------------------------------------------------- comparison of the arms
def run_contention(setup, grader, readers):
    name, config = "contention", setup.config["contention"]
    rule = config["rule"]
    offered = {arm: [len(item.output.releases) for item in setup.of(arm)] for arm in ("dsp_only", "capture_only", "concurrent")}
    released = {arm: [item.output.pacing["counts"]["on_time"] + item.output.pacing["counts"]["late"]
                      for item in setup.of(arm)] for arm in offered}
    survive = all(counts and set(counts) == {setup.opportunities} for counts in offered.values())
    grader.check(name, "arrival counts", survive,
                 f"every repetition of every arm with a receiver was offered the {setup.opportunities} releases of the one "
                 f"schedule; released: " + ", ".join(f"{arm} {sum(counts)}" for arm, counts in released.items())
                 + "; a release that was not started stays in its denominator")
    base = pooled(setup.of("capture_only"))
    shifts = {"mps_default": mp.tail_shift(base, pooled(setup.of("concurrent")), rule)}
    planned = {}
    for control in mp.CONTROLS[1:]:
        if shifts[mp.CONTROLS[mp.CONTROLS.index(control) - 1]] is not None \
                and shifts[mp.CONTROLS[mp.CONTROLS.index(control) - 1]]["exceeded"] is not True:
            break
        if control == "release_after_radio":
            done = [(line["t_done_ns"] - line["t_start_ns"]) / 1000.0 for item in setup.of("capture_only")
                    for line in item.output.ran()]
            setup.plans["after_radio_us"] = int(mp.nearest_rank(done, mp.QUANTILES[config["after_radio_quantile"]]) or 0)
        made = []
        for repetition in range(config["control_repetitions"]):
            item = Arm("concurrent", f"{control}-{repetition}", setup.out / "arms" / f"{control}-{repetition}-concurrent",
                       control=control)
            setup.arms.append(execute(setup, item, **readers))
            made.append(item)
        if all(item.ok() for item in made) and made:
            shifts[control] = mp.tail_shift(base, pooled(made), rule)
        else:
            shifts[control] = None
            planned[control] = made[0].said() if made else "no repetition was configured"
    chosen, why = mp.select_control(shifts, config["order"])
    first = shifts["mps_default"]
    stated = first["exceeded"] is not None
    grader.check(name, "tail", stated,
                 f"the rule {first['rule_sha256'][:16]} ({rule['status']}, declared in the configuration before a run): the "
                 f"{rule['quantile']} of the PUSCH service time may move by {rule['allowed_shift_us']} us or "
                 f"{100 * rule['allowed_shift_fraction']:.0f} percent of the capture-only arm, whichever is larger. "
                 f"Capture only: {text(first['base']['value_us'])} us over {first['base']['count']} decodes; concurrent: "
                 f"{text(first['other']['value_us'])} us over {first['other']['count']}; shift "
                 f"{text(first['shift_us'])} us against {text(first['allowed_us'])} us allowed" if stated else
                 f"the rule gives no verdict: {first['reason']}")
    grader.check(name, "control in effect", stated,
                 (f"the control in effect is {chosen}: {why}" if chosen else f"no control is in effect: {why}")
                 + "; the receiver was not slowed, queued or reconfigured in any arm, and no ring was enlarged"
                 + ("".join(f"; {control} is a planned cell on this host: {reason[:120]}"
                            for control, reason in planned.items())))
    isolated = {arm: mp.statistics(pooled(setup.of(arm))) for arm in ("dsp_only", "capture_only", "concurrent")}
    grader.note(name, "recorded, not graded: PUSCH service time by arm, median / 99th percentile / most in us: "
                + "; ".join(f"{arm} {text(made['p50_us'])} / {text(made['p99_us'])} / {text(made['max_us'])} over "
                            f"{made['count']}" for arm, made in isolated.items()))
    return {"rule": rule, "rule_sha256": first["rule_sha256"], "shifts": shifts, "control_in_effect": chosen,
            "reason": why, "planned_cells": planned, "after_radio_us": setup.plans.get("after_radio_us"),
            "service": isolated, "offered": offered, "released": released}


def sensor_latency(arms, evaluator, obs):
    """The latency definitions of the sensing core over the scored windows of each of some arms.

    An output whose decisions carry no host times is no live output and has no latency: None.
    """
    made = []
    for item in arms:
        rows, kept = None, item.directory / "sensor" / "windows" / "records"
        if item.sensor.report["runtime"]["entry"] != "native" and (kept / "consumer.json").is_file():
            # The reference sensor marks no notification: the time it read each record stands for it.
            rows = obs.open_archive(kept).report["records"]
        try:
            made.append(evaluator.latency(item.sensor, rows))
        except evaluator.SenseCoreError:
            made.append(None)
    return made


def run_runtime(setup, grader, readers, evaluator, obs):
    name, measured = "runtime", {}
    beside, alone = setup.of("concurrent"), setup.of("ai_only")

    def line(found, key):
        values = [entry[key] for entry in found]
        count = sum(value["count"] for value in values)
        return (f"{text(max(value['p50_us'] or 0 for value in values))} us in the median and "
                f"{text(max(value['p99_us'] or 0 for value in values))} us at the 99th percentile of the slowest "
                f"repetition, {count} windows")

    native, single = sensor_latency(beside, evaluator, obs), sensor_latency(alone, evaluator, obs)
    live = native and single and None not in native and None not in single
    valid = live and (all(entry["timing_valid"] and entry["engine"] == "tensorrt" for entry in native) or setup.rehearsal)
    if not live:
        grader.check(name, "native", False, "a sensor output of a timed arm carries no host times: it is no live output")
        return measured
    grader.check(name, "native", valid,
                 f"sense_host beside the receiver under MPS, by the cuSense boundary: {line(native, 'cusense_boundary')}; "
                 f"alone under MPS on the buffered records: {line(single, 'cusense_boundary')}; post-window beside the "
                 f"receiver: {line(native, 'post_window')}. Recorded, not graded"
                 + ("" if not setup.rehearsal else ": a rehearsal on the host twin times nothing"))
    whole = [mp.window_latency(item.output, item.sensor.scored) for item in beside]
    unjoined = sum(entry["unjoined"] for entry in whole)
    grader.check(name, "end to end", unjoined == 0 or setup.rehearsal,
                 f"the paced receiver fixes the map between the radio clock and the host clock: from the release of the "
                 f"first slot of a window to its decision {text(max(entry['end_to_end']['p50_us'] or 0 for entry in whole))} "
                 f"us in the median and {text(max(entry['end_to_end']['p99_us'] or 0 for entry in whole))} us at the 99th "
                 f"percentile of the slowest repetition; from the end of the radio job of its last slot "
                 f"{text(max(entry['after_radio']['p50_us'] or 0 for entry in whole))} us in the median; {unjoined} "
                 "window(s) name a release that did not run")
    measured.update(native_beside_receiver=native, native_alone=single, end_to_end=whole)
    # ---- The same branch on the reference runtime, as a third arm on the same offered trace.
    item = Arm("concurrent_reference", "reference-0", setup.out / "arms" / "reference-0-concurrent_reference")
    setup.arms.append(execute(setup, item, **readers))
    if not item.ok() or item.sensor is None:
        grader.check(name, "reference", False, f"the arm with the reference sensor did not run clean: {item.said()}")
        return measured
    sensor = [entry for entry in item.launch["processes"] if entry["role"] == "sensor"][0]
    found = sensor_latency([item], evaluator, obs)[0]
    label = item.launch["sharing"]
    if found is None:
        # Only a rehearsal gets here: its double replays an archive, and a replay marks no host time.
        grader.check(name, "reference", setup.rehearsal and len(item.sensor.scored) >= 1,
                     f"the reference sensor scored {len(item.sensor.scored)} windows, and its decisions carry no host "
                     "times" + (": a rehearsal replays an archive in place of a live ring" if setup.rehearsal else ""))
        return measured
    grader.check(name, "reference", len(item.sensor.scored) >= 1 and sensor["maps_python"] is not False,
                 f"the same branch on the reference runtime, PyTorch in a Python sensor on "
                 f"{item.sensor.report['runtime']['device']}, beside the same receiver on the same schedule: "
                 f"{len(item.sensor.scored)} windows scored, sharing {label['label']}"
                 + (" with both process identifiers in the client list" if label["verified"] else "")
                 + f"; by the cuSense boundary {text(found['cusense_boundary']['p50_us'])} us in the median and "
                 f"{text(found['cusense_boundary']['p99_us'])} us at the 99th percentile; the process maps a Python "
                 "library, and its latency supports no deadline. Recorded, not graded")
    measured["reference_beside_receiver"] = found
    measured["reference_service"] = mp.statistics(mp.service_times(item.output))
    return measured


def run_faults(setup, grader, readers):
    name, measured = "faults", {}
    for kind, said in (("consumer_exit", "is killed"), ("consumer_restart", "is killed and obs_tail takes over the ring")):
        item = Arm("concurrent", f"{kind}-0", setup.out / "arms" / f"{kind}-0-concurrent", fault=kind)
        setup.arms.append(execute(setup, item, **readers))
        note = (item.launch or {}).get("fault")
        if not item.ok() or note is None or item.output is None:
            grader.check(name, kind, False, f"the arm did not run as planned: {item.said()}")
            continue
        summary, counts = mp.radio_summary(item.output), item.output.pacing["counts"]
        released = counts["on_time"] + counts["late"]
        state = ne.account(item.output.native)
        producer = [entry for entry in item.launch["processes"] if entry["role"] == "producer"][0]
        lost = state["dropped"] + state.get("reclaimed", 0) + state.get("pending", 0)
        whole = summary["decoded"] == released and producer["ended_by"] == "itself" and note["producer_alive_after"] \
            and note.get("clock") == "schedule_start" and state["balanced"] and state["published"] + state["dropped"] == released
        taken = len(item.archives["takeover"].records) if "takeover" in item.archives else None
        if kind == "consumer_restart":
            whole = whole and taken is not None and (taken >= 1 or setup.rehearsal)
        grader.check(name, kind, whole,
                     f"sense_host {said} {note['after_ms']} ms after the start of the offered schedule: the receiver was never "
                     f"signalled, ended by itself and decoded {summary['decoded']} of {released} released grids with the "
                     f"original bytes; of the {state['eligible']} estimates {state['published']} were published and "
                     f"{state['dropped']} found no credit and were dropped and counted, {state.get('reclaimed', 0)} were "
                     f"reclaimed with the closed epoch and {state.get('pending', 0)} pending at the cutoff: {lost} lost, "
                     "bounded by the ring and accounted for"
                     + ("" if taken is None else f"; the consumer that took over kept {taken} records of a new epoch"))
        measured[kind] = {"fault": note, "account": state, "decoded": summary["decoded"], "released": released,
                          "taken_over": taken, "service": summary["service"]}
    return measured


def cold_record(config, measured):
    """The cold start of the receiver, kept apart from every figure of the schedule."""
    lead = config["pacing"].get("lead_in")
    runs = {arm: (measured.get(arm) or {}).get("cold_start") or [] for arm in ("dsp_only", "capture_only", "concurrent")}
    every = [item for items in runs.values() for item in items]
    depth = max((len(item["service_us"]) for item in every), default=0)
    stages = sorted({name for item in every for name in (item["stages_us"] or {})})
    first = [item["first_releases"][0]["service_us"] for item in every
             if item["first_releases"] and item["first_releases"][0]["service_us"] is not None]
    return {
        "isolated": ("the first use of the receiver is a decode of the lead-in: it is no release, no radio record, no "
                     "observation and no sample of a service-time statistic, and it is recorded in pacing/lead_in.jsonl "
                     "of every run") if lead else
                    ("no lead-in: the first release of every run is the first use of the receiver. Its service time is "
                     "left out of the service statistics, and the releases it costs are counted in the schedule"),
        "lead_in": None if lead is None else dict(lead, status=config["pacing"]["status"]),
        "runs": len(every),
        "by_decode_us": [mp.statistics([item["service_us"][k] for item in every if len(item["service_us"]) > k])
                         for k in range(depth)],
        "first_decode_stages_us": {name: mp.statistics([item["stages_us"][name] for item in every
                                                        if item["stages_us"] and name in item["stages_us"]])
                                   for name in stages},
        "before_first_release_us": mp.statistics([item["before_first_release_us"] for item in every
                                                  if item["before_first_release_us"] is not None]),
        "first_release_us": mp.statistics(first),
        "skipped_first_releases": sum(item["skipped_first_releases"] for item in every),
        "first_use_in_schedule": sum(len(item["first_use_in_schedule"]) for item in every),
        "arms": runs,
        "measured_without_lead_in": config["pacing"].get("cold_start_without_lead_in")}


def run_records(setup, grader, measured, readers, evaluator, obs):
    name, config = "records", setup.config
    sliced = None
    if config["time_slicing_control"]:
        item = Arm("concurrent", "time_slicing-0", setup.out / "arms" / "time_slicing-0-concurrent", sharing="time_slicing")
        setup.arms.append(execute(setup, item, **readers))
        found = sensor_latency([item], evaluator, obs)[0] if item.ok() and item.sensor is not None else None
        if found is not None and item.output is not None:
            sliced = {"label": item.launch["sharing"]["label"], "service": mp.statistics(mp.service_times(item.output)),
                      "cusense_boundary": found["cusense_boundary"], "scored": len(item.sensor.scored)}
            grader.check(name, "time slicing", sliced["label"] == "time_slicing",
                         f"the same arm with both processes outside the daemon carries the label time_slicing and never "
                         f"MPS: by the cuSense boundary {text(found['cusense_boundary']['p50_us'])} us in the median and "
                         f"{text(found['cusense_boundary']['p99_us'])} us at the 99th percentile, PUSCH service time "
                         f"{text(sliced['service']['p50_us'])} / {text(sliced['service']['p99_us'])} us. A labeled "
                         "control, recorded and not graded")
        else:
            sliced = {"label": "planned", "reason": item.said()}
            grader.note(name, f"time slicing is a planned cell on this host: the arm outside the daemon did not run "
                              f"clean ({item.said()[:200]})")
    processes = [dict(entry, arm=item.label) for item in setup.arms if item.launch for entry in item.launch["processes"]]
    native = [entry for entry in processes if entry["kind"] == "sense_host" and entry["maps_python"] is not None]
    grader.check(name, "process inventory", native and not any(entry["maps_python"] for entry in native),
                 f"{len(processes)} processes of {len([item for item in setup.arms if item.launch])} launches are "
                 f"recorded with their command, their CPUs, their environment, their times and their status; "
                 f"{len(native)} warm sense_host processes map no Python library")
    arms = {}
    for item in setup.arms:
        if item.launch is None:
            continue
        entry = {"label": item.label, "arm": item.arm, "control": item.control, "fault": item.fault, "warmup": item.warmup,
                 "sharing": item.launch["sharing"], "problems": item.problems, "directory": item.directory.name,
                 "radio": None if item.output is None else mp.radio_summary(item.output),
                 "arrivals": None if item.output is None else mp.arrivals(item.output),
                 "sensor": None if item.sensor is None else {"runtime": item.sensor.report["runtime"],
                                                             "counts": item.sensor.report["counts"]},
                 "feed": item.feed}
        arms.setdefault(item.arm, []).append(entry)
    record = {
        "schema": RECORD_SCHEMA, "graded": False, "run_id": setup.out.name,
        "meaning": "what every arm of Sprint 4.1 offered, delivered and took on this host. Counts are graded by the gate. "
                   "Service times, tails and latencies are recorded for Sprint 4.6, which calibrates on these arms, and "
                   "are valid for the recorded host state only",
        "load": {"scripted": True, "receiver": "the staged harness on an offered schedule: one release per uplink slot "
                                               "of DDDDDDSUUU, three per 5 ms",
                 "core_network": "none: no UE, no 5GC and no PRACH procedure is attached",
                 "traffic": "one known transport block per release, replayed through a traced channel"},
        "host_state": {"first": setup.arms[0].launch["host"]["before"] if setup.arms and setup.arms[0].launch else None,
                       "last": setup.arms[-1].launch["host"]["after"] if setup.arms and setup.arms[-1].launch else None},
        "frozen": {"cpus": config["cpus"], "rings": config["rings"], "pacing": config["pacing"],
                   "estimate": config["estimate"], "warmup_arms": config["warmup"], "repetitions": config["repetitions"],
                   "order": [list(order) for order in ORDERS[:config["repetitions"]]],
                   "tools": {name: digest(path) for name, path in setup.tools.items()},
                   "plans": {key: setup.plans[key]["sha256"] for key in ("tap", "dsp") if key in setup.plans}},
        "mps": measured.get("mps"), "schedule": measured.get("schedule"), "arms": arms,
        "cold_start": cold_record(config, measured),
        "contention": measured.get("contention"), "runtime": measured.get("runtime"), "faults": measured.get("faults"),
        "time_slicing": sliced, "rehearsal": setup.rehearsal}
    write_json(setup.out / RECORD_FILE, record)
    needed = [arm for arm in mp.TIMED_ARMS if arm not in arms]
    grader.check(name, "record", not needed and record["graded"] is False,
                 f"{RECORD_FILE} holds {sum(len(items) for items in arms.values())} launches of "
                 f"{len(arms)} arms with their sharing label, their offered and delivered counts, their service-time "
                 f"tails and the state of the host; the load is scripted and no core network is attached; nothing of it "
                 "is graded as a latency" if not needed else f"{RECORD_FILE} lacks the arm(s) {', '.join(needed)}")
    return {"launches": sum(len(items) for items in arms.values()), "time_slicing": sliced}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="new directory for the run")
    parser.add_argument("--s36-run", type=Path, help="run directory of the accepted Sprint 3.6 gate: its engine, and the "
                                                     "Sprint 3.5 run it names")
    parser.add_argument("--s35-run", type=Path, help="run directory of the accepted Sprint 3.5 gate (default: the one "
                                                     "the Sprint 3.6 run names)")
    parser.add_argument("--adapter-build", type=Path, help="adapter build against the patched receiver")
    parser.add_argument("--core-build", type=Path, help="build of the sensing core")
    parser.add_argument("--rehearsal", action="store_true",
                        help="development host: the host twin of the core and no GPU; the hardware gate stays open")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.exit(2, f"mps_pacing: {args.output_dir} already exists. Evidence is never overwritten.\n")
    try:
        from ocudu_cuda_isac.episodes import presence_head as split
        from ocudu_cuda_isac.episodes import sense_core as evaluator
        from ocudu_cuda_isac.sensing import cusense, obs, presence
    except ImportError as error:
        parser.exit(2, f"mps_pacing: the gate needs NumPy and PyTorch in this interpreter: {error}\n")
    try:
        config = load_config(args.config)
        adapter = native_adapter.verified_receipt(args.adapter_build or ROOT / config["adapter_build"])
        core = native_core.verified_receipt(args.core_build or ROOT / config["core_build"])
        for needed, receipt in ((ADAPTER_TOOLS, adapter), (CORE_TOOLS, core)):
            for tool in needed:
                if tool not in receipt["executables"]:
                    raise ConfigError(f"a build holds no {tool}. Rebuild: make adapter-build-o3, make core-build")
        if args.s36_run is None and not (args.rehearsal and args.s35_run):
            raise ConfigError("the gate needs the accepted Sprint 3.6 run: --s36-run. A rehearsal names --s35-run")
        if not args.rehearsal and core["device"] != "cuda":
            raise ConfigError("the build of the core is for the host twin. Build the device: make core-build")
    except (ConfigError, native_adapter.AdapterError, native_core.CoreError) as error:
        parser.exit(2, f"mps_pacing: {error}\n")
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = args.output_dir.resolve()
    out.mkdir(parents=True)
    write_json(out / "adapter-receipt.json", adapter)
    write_json(out / native_core.RECEIPT, core)
    # The driver and its readers stay off the CPUs of the receiver.
    tools_cpus = [cpu for cpu in run_m1.parse_cpus(config["cpus"]["tools"]) if cpu in os.sched_getaffinity(0)]
    if tools_cpus:
        os.sched_setaffinity(0, tools_cpus)
    setup = Setup(config, out, adapter, core, args.rehearsal)
    readers = {"cusense": cusense, "obs": obs, "presence": presence}
    grader, measured = Grade(), {}

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

    measured["build"] = guarded("build", lambda: run_build(setup, grader), {})
    measured["control"] = guarded("control", lambda: run_control(
        setup, grader, args.s36_run and args.s36_run.resolve(), args.s35_run and args.s35_run.resolve(), cusense, presence,
        split), {})
    measured["mps"] = guarded("mps", lambda: run_mps(setup, grader), {})
    if not (passed("build") and passed("control") and passed("mps")):
        blocked(CASES[3:], "the builds, the controls or the MPS daemon do not hold what this case builds on")
    else:
        measured["schedule"] = guarded("schedule", lambda: run_schedule(setup, grader), {})
        if not passed("schedule"):
            blocked(CASES[4:], "no plan of version 4 was written: no arm is launched")
        else:
            try:
                campaign(setup, readers)
            except Exception as error:  # what a launch could not survive: every arm case then says so
                (out / "driver-error-campaign.txt").write_text(traceback.format_exc())
                grader.note("schedule", f"the campaign stopped: {type(error).__name__}: {error}")
            measured["dsp_only"] = guarded("dsp_only", lambda: run_dsp_only(setup, grader), {})
            measured["capture_only"] = guarded("capture_only", lambda: run_capture_only(setup, grader), {})
            measured["ai_only"] = guarded("ai_only", lambda: run_ai_only(setup, grader), {})
            measured["concurrent"] = guarded("concurrent", lambda: run_concurrent(setup, grader, evaluator), {})
            if all(passed(case) for case in ("dsp_only", "capture_only", "ai_only", "concurrent")):
                measured["contention"] = guarded("contention", lambda: run_contention(setup, grader, readers), {})
                measured["runtime"] = guarded("runtime", lambda: run_runtime(setup, grader, readers, evaluator, obs), {})
                measured["faults"] = guarded("faults", lambda: run_faults(setup, grader, readers), {})
                measured["records"] = guarded("records", lambda: run_records(setup, grader, measured, readers, evaluator,
                                                                             obs), {})
            else:
                blocked(CASES[8:], "an arm did not run clean: nothing is compared")

    failed = [item for item in grader.results if not item["passed"]]
    passed_cases = [case for case in CASES if passed(case)]
    whole = not failed and len(passed_cases) == len(CASES)
    gate = "closed" if whole and not args.rehearsal else \
        "open: a rehearsal runs no receiver, no kernel on a device and no MPS daemon" if args.rehearsal else "open"
    grade = {"schema": GRADE_SCHEMA, "selector": "mps_pacing", "run_id": out.name, "cases": len(CASES),
             "cases_passed": len(passed_cases), "cases_failed": len(CASES) - len(passed_cases),
             "checks": len(grader.results), "checks_passed": len(grader.results) - len(failed), "checks_failed": len(failed),
             "judgment": "pass" if whole else "fail", "hardware_gate": gate, "rehearsal": args.rehearsal,
             "reason": "every check of every case passed" if not failed else
                       f"{len(failed)} check(s) failed, first: {failed[0]['case']} {failed[0]['check']}",
             "artifact_sha256": None if setup.artifact is None else setup.artifact.sha256,
             "core_sources_sha256": core["sources_sha256"], "pilot_ls_sha256": adapter["executables"]["pilot_ls"]["sha256"],
             "control_in_effect": (measured.get("contention") or {}).get("control_in_effect"),
             "graded_latency": "none: service times, tails and latencies are recorded in mps_pacing.json for Sprint 4.6",
             "results": grader.results}
    write_json(out / "grade.json", grade)
    write_json(out / "comparison.json", measured)
    (out / "grade.txt").write_text("\n".join(grader.lines) + "\n", encoding="ascii")
    write_json(out / "manifest.json", {
        "schema": "integration-manifest/mps_pacing/v1", "run_id": out.name, "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": shlex.join([sys.executable, *sys.argv]), "config": config, "config_sha256": digest(args.config),
        "s36_run": None if args.s36_run is None else str(args.s36_run.resolve()),
        "s35_run": None if setup.s35_out is None else str(setup.s35_out),
        "adapter_build": adapter["adapter_build"], "core_build": core["build_dir"], "rehearsal": args.rehearsal,
        "host": platform.node(), "machine": platform.machine(), "python": platform.python_version(),
        "driver_sha256": digest(__file__), "launcher_sha256": digest(ROOT / "scripts/run_m1.py"),
        "evaluator_sha256": digest(ROOT / "src/ocudu_cuda_isac/episodes/mps_pacing.py"),
        "launches": [{"label": item.label, "arm": item.arm, "control": item.control, "fault": item.fault,
                      "sharing": item.sharing, "warmup": item.warmup, "directory": str(item.directory.relative_to(out)),
                      "problems": item.problems} for item in setup.arms],
        "axes": {"split": "staged_harness", "receiver_stage": "o3_native_estimate, with o2_pilot_ls beside it",
                 "ue_realization": "replayed channel, no UE", "host_stack": "DeepSig-OCUDU", "placement": "one_spark",
                 "gpu_sharing": "mps_default", "runtime": "native", "memory_path": "pinned_host_ring",
                 "model": "cusense_cnn", "transport": "obs-ring/v1, one first-in first-out ring per stage, wait mode spin"},
        "files": [name for name in ("adapter-receipt.json", native_core.RECEIPT, RECORD_FILE, "grade.json", "grade.txt",
                                    "comparison.json") if (out / name).is_file()]})
    print(f"mps_pacing: {grade['cases_passed']} of {grade['cases']} cases, {grade['checks_passed']} of {grade['checks']} "
          f"checks, judgment {grade['judgment']}, hardware gate {gate.split(':')[0]} ({out / 'grade.json'})")
    return 0 if whole else 1


if __name__ == "__main__":
    sys.exit(main())
