#!/usr/bin/env python3
"""Launcher of the receiver and the sensor as two processes on one GPU (Sprint 4.1, selector mps_pacing).

One arm is one producer and the consumers of its rings. The producer is the staged harness
pilot_ls on a paced plan, or, in the arm that replays buffered records, sense_feed. A consumer
is sense_host, the reference sensor in the interpreter of isac, or obs_tail, which reads and
releases every record and infers nothing. The launcher

  starts every consumer first and waits until each one says it is ready: a sensor loads its
  artifact, creates its CUDA context and warms its engine before the receiver exists;
  starts the producer, which waits for the consumers of its rings and then runs its schedule;
  reads the client list of the MPS server while the processes run, and labels the arm MPS only
  when one reading names every CUDA process of the arm as a client of the one server on the one
  device. A refusal carries the client list. The readings while the producer runs are two
  commands of the control interface each, and they stop with the first one that proves the label;
  injects at most one fault: it ends a consumer at a given time after the producer started, and
  may start another one in its place. It never signals the producer;
  ends nothing else before the producer has ended, and records what every process was: its
  command, its CPUs, what it added to its environment, its status, its times and whether it
  maps a Python library.

    python scripts/run_m1.py mps start|status|stop [--pipe-dir DIR] [--log-dir DIR]
    python scripts/run_m1.py arm --spec FILE --record FILE

The driver of the gate, tests/integration/mps_pacing.py, builds the arm specifications and
calls launch() directly. Everything the launcher asks of the host goes through one object, so
that a unit case replaces it with a fake clock and a fake process inventory. Standard library
only. No process of an arm is started with a shell.
"""

import argparse
import json
import os
from pathlib import Path
import platform
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ocudu_cuda_isac.episodes import mps_pacing as mp  # noqa: E402
from ocudu_cuda_isac.episodes.mps_pacing import MpsPacingError  # noqa: E402

SPEC_SCHEMA = "m1-arm/v1"
LAUNCH_SCHEMA = "m1-launch/v1"
MPS_CONTROL = "nvidia-cuda-mps-control"
DEFAULT_MPS_DIR = ROOT / "build" / "mps"
KINDS = ("pilot_ls", "sense_feed", "sense_host", "obs_tail", "presence", "cusense")
FAULTS = ("consumer_exit", "consumer_restart")
SPEC_FIELDS = {"schema", "arm", "label", "sharing", "producer", "consumers", "fault", "timeouts", "inventory_period_ms",
               "extra_clients"}
PROCESS_FIELDS = {"role", "kind", "command", "cpus", "env", "cuda", "ready_file", "log", "python_path"}
TIMEOUT_FIELDS = {"ready_s", "run_s", "grace_s"}
MPS_ENV = ("CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY")
CLIENT_ENV = ("CUDA_MPS_CLIENT_PRIORITY", "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE")


class LaunchError(ValueError):
    """An arm cannot be started as it was specified. Nothing of it runs."""


# ---------------------------------------------------------------- the host
def parse_cpus(text):
    """The CPUs of a list such as "5-9,15-19". None or an empty text names none."""
    found = set()
    for part in str(text or "").split(","):
        part = part.strip()
        if not part:
            continue
        first, _, last = part.partition("-")
        try:
            low, high = int(first), int(last or first)
        except ValueError:
            raise LaunchError(f"{text!r} is no list of CPUs") from None
        if low < 0 or high < low:
            raise LaunchError(f"{text!r} is no list of CPUs")
        found.update(range(low, high + 1))
    return sorted(found)


class Host:
    """What the launcher asks of the machine it runs on. A unit case passes a double of this class."""

    def now_ns(self):
        return time.monotonic_ns()

    def sleep(self, seconds):
        time.sleep(seconds)

    def cpus(self):
        return sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))

    def spawn(self, command, log, env, cpus):
        """Start one process with its output in a log. cpus is the list it may run on, or an empty list."""
        stream = open(log, "wb")
        stream.write((shlex.join(str(word) for word in command) + "\n").encode())
        stream.flush()

        def pin():
            os.sched_setaffinity(0, cpus)

        try:
            return subprocess.Popen([str(word) for word in command], stdout=stream, stderr=subprocess.STDOUT, env=env,
                                    preexec_fn=pin if cpus else None, start_new_session=True)
        finally:
            stream.close()

    def run(self, command, text="", env=None, timeout=30):
        """One short command with its input. Returns (status, output). A missing tool has the status 127."""
        try:
            done = subprocess.run([str(word) for word in command], input=text, capture_output=True, text=True, env=env,
                                  timeout=timeout, check=False)
            return done.returncode, done.stdout + done.stderr
        except FileNotFoundError as error:
            return 127, str(error)
        except subprocess.TimeoutExpired:
            return 124, "no answer within %d s" % timeout

    def exists(self, path):
        return Path(path).exists()

    def maps(self, pid):
        """The files a process maps, or None when they cannot be read."""
        try:
            lines = Path(f"/proc/{pid}/maps").read_text().splitlines()
        except OSError:
            return None
        return sorted({line.split(None, 5)[5] for line in lines if len(line.split(None, 5)) == 6
                       and line.split(None, 5)[5].startswith("/")})

    def state(self):
        """What the host was when an arm ran. Recorded with it."""
        made = {"host": platform.node(), "machine": platform.machine(), "kernel": platform.release(),
                "cpus": os.cpu_count(), "loadavg": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
                "monotonic_ns": self.now_ns()}
        for name, path in (("cpu_governor", "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
                           ("clocksource", "/sys/devices/system/clocksource/clocksource0/current_clocksource")):
            made[name] = Path(path).read_text().strip() if Path(path).is_file() else None
        zones = {}
        for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp")) if Path("/sys/class/thermal").is_dir() else ():
            try:
                zones[zone.parent.name] = int(zone.read_text().strip()) / 1000.0
            except (OSError, ValueError):
                continue
        made["thermal_zones_c"] = zones or None
        status, text = self.run(["nvidia-smi", "--query-gpu=name,driver_version,utilization.gpu,memory.used,temperature.gpu,"
                                 "power.draw", "--format=csv,noheader"], timeout=60)
        made["gpu"] = text.strip().splitlines()[0] if status == 0 and text.strip() else None
        return made


# ---------------------------------------------------------------- the MPS control interface
class Mps:
    """One MPS control daemon, named by its pipe directory."""

    def __init__(self, host, pipe_dir=None, log_dir=None, control=MPS_CONTROL, visible=None, base_env=None):
        self.host, self.control, self.visible = host, control, visible
        self.pipe_dir = Path(pipe_dir or DEFAULT_MPS_DIR / "pipe")
        self.log_dir = Path(log_dir or DEFAULT_MPS_DIR / "log")
        self.base_env = dict(os.environ if base_env is None else base_env)

    def env(self):
        """What a process needs in its environment to find this daemon, and to be found by it."""
        made = {"CUDA_MPS_PIPE_DIRECTORY": str(self.pipe_dir), "CUDA_MPS_LOG_DIRECTORY": str(self.log_dir)}
        if self.visible is not None:
            made["CUDA_VISIBLE_DEVICES"] = str(self.visible)
        return made

    def ask(self, command):
        """One command of the control interface. Returns (status, text)."""
        return self.host.run([self.control], text=command + "\n", env=dict(self.base_env, **self.env()), timeout=30)

    def alive(self):
        status, text = self.ask("get_default_active_thread_percentage")
        return status == 0 and "Cannot find MPS control daemon" not in text

    def start(self, wait_s=10.0):
        """Start the daemon unless it answers already. Returns "running" or "started"."""
        if self.alive():
            return "running"
        self.pipe_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        status, text = self.host.run([self.control, "-d"], env=dict(self.base_env, **self.env()), timeout=60)
        if status != 0:
            raise LaunchError(f"{self.control} -d ended with status {status}: {text.strip()[:300]}")
        deadline = self.host.now_ns() + int(wait_s * 1e9)
        while not self.alive():
            if self.host.now_ns() > deadline:
                raise LaunchError(f"the MPS control daemon does not answer on {self.pipe_dir} within {wait_s:.0f} s")
            self.host.sleep(0.1)
        return "started"

    def stop(self):
        status, text = self.ask("quit")
        return status == 0, text.strip()

    def devices(self):
        """The devices the daemon serves: every GPU of the host, or those CUDA_VISIBLE_DEVICES names."""
        status, text = self.host.run(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"], timeout=60)
        found = []
        for line in text.splitlines() if status == 0 else ():
            parts = [part.strip() for part in line.split(",", 2)]
            if len(parts) == 3 and parts[0].isdigit():
                found.append({"index": int(parts[0]), "uuid": parts[1], "name": parts[2]})
        if self.visible is not None:
            wanted = [word.strip() for word in str(self.visible).split(",") if word.strip()]
            found = [device for device in found
                     if any(word == str(device["index"]) or device["uuid"].startswith(word) for word in wanted)]
        return found

    def inventory(self):
        """One reading of the daemon: its servers, their clients, its settings and the devices it serves."""
        made = {"schema": mp.INVENTORY_SCHEMA, "t_monotonic_ns": self.host.now_ns(),
                "daemon": {"alive": self.alive(), "pipe_directory": str(self.pipe_dir),
                           "log_directory": str(self.log_dir), "visible_devices": self.visible},
                "servers": [], "clients": {}, "active_thread_percentage": {"default": None, "servers": {}},
                "default_client_priority": None, "devices": self.devices(), "compute_apps": None}
        if not made["daemon"]["alive"]:
            return made
        status, text = self.ask("get_server_list")
        made["servers"] = mp.parse_pids(text) if status == 0 else []
        for server in made["servers"]:
            status, text = self.ask(f"get_client_list {server}")
            made["clients"][str(server)] = mp.parse_pids(text) if status == 0 else []
            status, text = self.ask(f"get_active_thread_percentage {server}")
            made["active_thread_percentage"]["servers"][str(server)] = text.strip() if status == 0 else None
        status, text = self.ask("get_default_active_thread_percentage")
        made["active_thread_percentage"]["default"] = text.strip() if status == 0 else None
        # A release of MPS without client priorities answers with an error text: it is recorded as it came.
        status, text = self.ask("get_default_client_priority")
        made["default_client_priority"] = {"status": status, "text": text.strip()[:200]}
        status, text = self.host.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                      "--format=csv,noheader"], timeout=60)
        made["compute_apps"] = text.strip().splitlines() if status == 0 else None
        return made

    def clients(self, base):
        """A light reading for the time a producer runs: the servers and their clients, and nothing else.

        The devices and the settings are those of base, a full reading taken before the producer
        started. Two commands of the control interface are sent, and nvidia-smi is not run: a
        reading must not disturb what it reads.
        """
        made = dict(base, t_monotonic_ns=self.host.now_ns(), light=True, servers=[], clients={})
        status, text = self.ask("get_server_list")
        made["daemon"] = dict(base["daemon"], alive=status == 0 and "Cannot find MPS control daemon" not in text)
        made["servers"] = mp.parse_pids(text) if made["daemon"]["alive"] else []
        for server in made["servers"]:
            status, text = self.ask(f"get_client_list {server}")
            made["clients"][str(server)] = mp.parse_pids(text) if status == 0 else []
        return made


# ---------------------------------------------------------------- arm specification
def process_spec(role, kind, command, cuda, cpus=None, env=None, ready_file=None, log=None, python_path=False):
    """One process of an arm."""
    return {"role": role, "kind": kind, "command": [str(word) for word in command], "cpus": cpus, "env": dict(env or {}),
            "cuda": bool(cuda), "ready_file": None if ready_file is None else str(ready_file),
            "log": str(log) if log else None, "python_path": bool(python_path)}


def arm_spec(arm, label, producer, consumers, sharing="mps", fault=None, timeouts=None, inventory_period_ms=100,
             extra_clients=0):
    """The specification of one arm, checked."""
    return check_spec({"schema": SPEC_SCHEMA, "arm": arm, "label": label, "sharing": sharing, "producer": producer,
                       "consumers": list(consumers), "fault": fault,
                       "timeouts": dict({"ready_s": 180, "run_s": 1800, "grace_s": 90}, **(timeouts or {})),
                       "inventory_period_ms": inventory_period_ms, "extra_clients": extra_clients})


def check_spec(spec):
    """Apply every rule of an arm specification. Raises LaunchError before a process exists."""
    if not isinstance(spec, dict) or set(spec) != SPEC_FIELDS or spec["schema"] != SPEC_SCHEMA:
        raise LaunchError(f"an arm is a {SPEC_SCHEMA} object with exactly: " + ", ".join(sorted(SPEC_FIELDS)))
    if spec["arm"] not in mp.ARMS:
        raise LaunchError(f"unknown arm {spec['arm']!r}: an arm is one of " + ", ".join(mp.ARMS))
    if spec["sharing"] not in mp.SHARING:
        raise LaunchError(f"unknown sharing {spec['sharing']!r}: " + " or ".join(mp.SHARING))
    if not isinstance(spec["label"], str) or not spec["label"]:
        raise LaunchError("an arm has a label")
    timeouts = spec["timeouts"]
    if not isinstance(timeouts, dict) or set(timeouts) != TIMEOUT_FIELDS \
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0 for value in timeouts.values()):
        raise LaunchError("timeouts names ready_s, run_s and grace_s, each a positive number")
    if type(spec["inventory_period_ms"]) is not int or spec["inventory_period_ms"] < 1 \
            or type(spec["extra_clients"]) is not int or spec["extra_clients"] < 0:
        raise LaunchError("inventory_period_ms is a positive integer and extra_clients a count")
    processes = [spec["producer"]] + list(spec["consumers"] if isinstance(spec["consumers"], list) else [None])
    roles = []
    for item in processes:
        if not isinstance(item, dict) or set(item) != PROCESS_FIELDS or item["kind"] not in KINDS \
                or not isinstance(item["command"], list) or not item["command"] \
                or not isinstance(item["env"], dict) or not isinstance(item["role"], str) or not item["role"]:
            raise LaunchError("a process of an arm names exactly: " + ", ".join(sorted(PROCESS_FIELDS))
                              + ", with a kind of " + ", ".join(KINDS))
        parse_cpus(item["cpus"])
        roles.append(item["role"])
    if len(set(roles)) != len(roles):
        raise LaunchError("two processes of an arm have one role: " + ", ".join(roles))
    producer = spec["producer"]
    if producer["kind"] not in ("pilot_ls", "sense_feed") or producer["role"] != "producer":
        raise LaunchError("the producer of an arm is pilot_ls or sense_feed, in the role producer")
    if any(item["kind"] in ("pilot_ls", "sense_feed") for item in spec["consumers"]):
        raise LaunchError("a consumer of an arm is a sensor or obs_tail")
    receiver = producer["kind"] == "pilot_ls"
    sensors = [item for item in spec["consumers"] if item["kind"] in ("sense_host", "cusense", "presence")]
    wanted = {"dsp_only": (True, 0, 0), "capture_only": (True, 0, None), "ai_only": (False, 1, 1),
              "concurrent": (True, 1, None), "concurrent_reference": (True, 1, None)}[spec["arm"]]
    if receiver != wanted[0]:
        raise LaunchError(f"the arm {spec['arm']} " + ("runs the receiver" if wanted[0] else "replays buffered records "
                          "with sense_feed and runs no receiver"))
    if spec["arm"] == "dsp_only" and spec["consumers"]:
        raise LaunchError("the arm dsp_only runs the receiver alone: its plan names no tap and no ring")
    if spec["arm"] == "capture_only" and (sensors or not spec["consumers"]):
        raise LaunchError("the arm capture_only reads and releases every record with obs_tail and runs no sensor")
    if wanted[1] and not sensors:
        raise LaunchError(f"the arm {spec['arm']} runs a sensor")
    if wanted[2] is not None and len(spec["consumers"]) != wanted[2]:
        raise LaunchError(f"the arm {spec['arm']} has {wanted[2]} consumer(s)")
    for item in spec["consumers"]:
        if item["cuda"] and item["ready_file"] is None:
            raise LaunchError(f"the {item['role']} process creates a CUDA context and names no ready file: the receiver "
                              "must not start before that context exists")
        if set(item["env"]) & set(MPS_ENV) or set(producer["env"]) & set(MPS_ENV):
            raise LaunchError("a process names no MPS directory of its own: the launcher gives every process of an arm "
                              "the daemon of the arm")
    if set(producer["env"]) & set(CLIENT_ENV):
        raise LaunchError("no arm reconfigures the receiver for the sensor: a contention control belongs to the sensor "
                          "client only")
    fault = spec["fault"]
    if fault is not None:
        if not isinstance(fault, dict) or set(fault) != {"kind", "role", "after_ms", "restart"} \
                or fault["kind"] not in FAULTS or type(fault["after_ms"]) is not int or fault["after_ms"] < 0:
            raise LaunchError("a fault names its kind, " + " or ".join(FAULTS) + ", a role, after_ms and restart")
        if fault["role"] not in [item["role"] for item in spec["consumers"]]:
            raise LaunchError(f"the fault names the role {fault['role']!r}, which is no consumer of the arm: a fault "
                              "never ends the producer")
        if (fault["kind"] == "consumer_restart") != isinstance(fault["restart"], dict):
            raise LaunchError("a consumer_restart names the process that takes over, and a consumer_exit names none")
        if fault["restart"] is not None:
            again = fault["restart"]
            if set(again) != PROCESS_FIELDS or again["kind"] in ("pilot_ls", "sense_feed") or again["role"] in roles:
                raise LaunchError("the process that takes over is a consumer with a role of its own")
    return spec


# ---------------------------------------------------------------- launch
def _environment(base, spec, item, mps):
    """The environment of one process: the base, the daemon of the arm, and what the process adds."""
    env = dict(base)
    for name in MPS_ENV + CLIENT_ENV:
        env.pop(name, None)
    if spec["sharing"] == "mps" and mps is not None:
        env.update(mps.env())
    env.update({key: str(value) for key, value in item["env"].items()})
    if item["python_path"]:
        env["PYTHONPATH"] = str(ROOT / "src")
    return env


def launch(spec, directory, host=None, mps=None, base_env=None):
    """Run one arm and return its m1-launch/v1 record. Logs are written into directory.

    The record says what ran and whether the arm may carry the label it asked for. It never
    raises for something a process did: a process that did not become ready, a producer that ran
    out of time and a refused label are problems of the record. LaunchError is raised only for a
    specification that cannot be started.
    """
    host = host or Host()
    check_spec(spec)
    directory = Path(directory)
    base_env = dict(os.environ if base_env is None else base_env)
    if spec["sharing"] == "mps" and mps is None:
        raise LaunchError("an arm that shares the device by MPS names the daemon it runs on")
    available = set(host.cpus())
    period_ns = spec["inventory_period_ms"] * 1_000_000
    timeouts = spec["timeouts"]
    record = {"schema": LAUNCH_SCHEMA, "arm": spec["arm"], "label": spec["label"],
              "sharing": {"requested": spec["sharing"], "label": None, "verified": False, "reason": None, "proof": None,
                          "context_before_receiver": None},
              "processes": [], "order": [], "inventories": [], "fault": None, "problems": [],
              "host": {"before": host.state(), "after": None}}
    running = {}

    def start(item):
        cpus = [cpu for cpu in parse_cpus(item["cpus"]) if cpu in available]
        log = directory / (item["log"] or f"{item['role']}.log")
        process = host.spawn(item["command"], log, _environment(base_env, spec, item, mps), cpus)
        entry = {"role": item["role"], "kind": item["kind"], "command": item["command"], "pid": process.pid,
                 "cpus": {"requested": item["cpus"], "applied": cpus or None},
                 "env": dict(item["env"]), "cuda": item["cuda"], "log": log.name, "status": None,
                 "started_ns": host.now_ns(), "ready_ns": None, "ended_ns": None, "ended_by": None, "maps_python": None}
        record["processes"].append(entry)
        record["order"].append({"event": "started", "role": item["role"], "pid": process.pid, "t_ns": entry["started_ns"]})
        running[item["role"]] = (process, entry, item)
        return process, entry

    def reap(role, by=None):
        process, entry, _ = running[role]
        if entry["status"] is None and process.poll() is not None:
            entry.update(status=process.poll(), ended_ns=host.now_ns(), ended_by=entry["ended_by"] or by or "itself")
            record["order"].append({"event": "ended", "role": role, "pid": entry["pid"], "t_ns": entry["ended_ns"]})
        return entry["status"]

    def end(role, by):
        process, entry, _ = running[role]
        if process.poll() is None:
            entry["ended_by"] = by
            process.kill()
            try:
                process.wait(timeout=10)
            except Exception:  # a process that does not die is recorded and left
                record["problems"].append(f"the {role} process {entry['pid']} did not end after it was killed")
        reap(role, by)

    full = {}

    def inventory(light=False):
        if spec["sharing"] != "mps":
            return None
        found = mps.clients(full["reading"]) if light and full else mps.inventory()
        full.setdefault("reading", found)
        record["inventories"].append(found)
        return found

    def cuda_roles():
        return {role: entry["pid"] for role, (process, entry, item) in running.items() if item["cuda"]}

    def proof(found, needed):
        try:
            return mp.verify_clients(found, needed, spec["extra_clients"]), None
        except MpsPacingError as error:
            return None, str(error)

    try:
        # ---- Every consumer first. A sensor is warm, and its context exists, before the producer does.
        for item in spec["consumers"]:
            process, entry = start(item)
            if item["ready_file"] is None:
                continue
            deadline = host.now_ns() + int(timeouts["ready_s"] * 1e9)
            while not host.exists(item["ready_file"]) and process.poll() is None and host.now_ns() < deadline:
                host.sleep(0.02)
            if not host.exists(item["ready_file"]):
                record["problems"].append(f"the {item['role']} process did not become ready within "
                                          f"{timeouts['ready_s']} s: the producer was not started")
                return record
            entry["ready_ns"] = host.now_ns()
            entry["maps_python"] = _maps_python(host, entry["pid"])
            record["order"].append({"event": "ready", "role": item["role"], "pid": entry["pid"], "t_ns": entry["ready_ns"]})
        # One full reading before the producer exists: the devices, the settings, and whether the
        # context of every sensor is a client of the server already.
        found = inventory()
        if spec["sharing"] == "mps" and cuda_roles():
            _, refused = proof(found, cuda_roles())
            record["sharing"]["context_before_receiver"] = refused is None
            if refused is not None:
                record["sharing"]["reason"] = refused

        # ---- The producer. Nothing below signals it but the end of its time.
        producer, produced = start(spec["producer"])
        fault, applied = spec["fault"], False
        began = produced["started_ns"]
        next_reading = began
        deadline = began + int(timeouts["run_s"] * 1e9)
        best, last_refusal = None, record["sharing"]["reason"]
        while producer.poll() is None:
            now = host.now_ns()
            if now >= next_reading and best is None and spec["sharing"] == "mps" and cuda_roles():
                found = inventory(light=True)
                next_reading = now + period_ns
                made, refused = proof(found, cuda_roles())
                if made is not None:
                    best = dict(made, t_monotonic_ns=found["t_monotonic_ns"])
                else:
                    last_refusal = refused
            if fault is not None and not applied and now >= began + fault["after_ms"] * 1_000_000:
                applied = True
                end(fault["role"], "fault")
                note = {"kind": fault["kind"], "role": fault["role"], "after_ms": fault["after_ms"],
                        "applied_ns": host.now_ns(), "pid": running[fault["role"]][1]["pid"], "restarted": None,
                        "producer_alive_after": producer.poll() is None}
                if fault["restart"] is not None:
                    _, again = start(fault["restart"])
                    note["restarted"] = {"role": again["role"], "pid": again["pid"], "started_ns": again["started_ns"]}
                record["fault"] = note
            for role in list(running):
                reap(role)
            if now > deadline:
                record["problems"].append(f"the producer did not end within {timeouts['run_s']} s and was stopped")
                end("producer", "timeout")
                break
            host.sleep(min(0.01, spec["inventory_period_ms"] / 1000.0))
        reap("producer")
        if fault is not None and not applied:
            record["problems"].append(f"the fault after {fault['after_ms']} ms was never applied: the producer ended first")

        # ---- The consumers end on their own when the producer said that no record follows.
        deadline = host.now_ns() + int(timeouts["grace_s"] * 1e9)
        for role, (process, entry, _) in list(running.items()):
            if role == "producer":
                continue
            while process.poll() is None and host.now_ns() < deadline:
                host.sleep(0.02)
            if process.poll() is None:
                record["problems"].append(f"the {role} process did not end within {timeouts['grace_s']} s of the "
                                          "producer and was stopped")
                end(role, "timeout")
            reap(role)

        # ---- The label. MPS is proven by one reading that names every CUDA process of the arm.
        sharing = record["sharing"]
        if spec["sharing"] == "mps":
            needed = cuda_roles()
            if not needed:
                sharing.update(label="mps", verified=False, reason="no process of the arm creates a CUDA context")
            elif best is not None:
                sharing.update(label="mps", verified=True, proof=best, reason=None)
            else:
                sharing.update(label="refused", verified=False,
                               reason=last_refusal or "no reading of the client list was taken while the producer ran")
                record["problems"].append(sharing["reason"])
        else:
            sharing.update(label="time_slicing", verified=False,
                           reason="the processes of this arm were started outside the MPS daemon: a labeled control")
        return record
    finally:
        for role in list(running):
            process, entry, _ = running[role]
            if process.poll() is None:
                end(role, "launcher")
            reap(role)
        record["host"]["after"] = host.state()


def _maps_python(host, pid):
    names = host.maps(pid)
    if names is None:
        return None
    return any("python" in Path(name).name.lower() for name in names)


def clean(record, allow=()):
    """True when every process of a launch ended with the status zero, apart from the roles of allow."""
    return not record["problems"] and all(entry["status"] == 0 or entry["role"] in allow for entry in record["processes"])


# ---------------------------------------------------------------- commands of the processes
def receiver_command(pilot_ls, plan, native, keep_grids=False):
    return [pilot_ls, "--plan", plan, "--output-dir", native] + ([] if keep_grids else ["--no-grids"])


def tail_command(obs_tail, socket, archive, capacity, wait="spin", record_bytes=32768):
    return [obs_tail, "--socket", socket, "--output-dir", archive, "--wait", wait, "--connect-ms", 120000, "--idle-ms",
            120000, "--capacity", capacity, "--record-bytes", record_bytes]


def feed_command(sense_feed, socket, archive, slots, slot_us, period_slots, repeat=1, attach_ms=60000, drain_ms=5000):
    return [sense_feed, "--socket", socket, "--archive", archive, "--slots", slots, "--pace", "slots", "--slot-us",
            slot_us, "--period-slots", period_slots, "--repeat", repeat, "--attach-ms", attach_ms, "--drain-ms", drain_ms]


def sense_host_command(sense_host, socket, artifact, output, ready, artifact_sha256, engine_dir=None, device="cuda",
                       engine="tensorrt", wait="spin", release="immediate", radio_allowance_us=0, budget_us=(0, 0, 0, 0),
                       lines=16384, windows=4096, keep_records=False, capacity=4096):
    command = [sense_host, "--socket", socket, "--artifact", artifact, "--output-dir", output, "--device", device,
               "--engine", engine, "--evidence", "timed", "--wait", wait, "--expect-sha256", artifact_sha256,
               "--ready-file", ready, "--connect-ms", 120000, "--idle-ms", 120000, "--release", release,
               "--budget-us", ",".join(str(value) for value in budget_us), "--lines", lines, "--windows", windows]
    if release == "after_radio":
        command += ["--radio-allowance-us", radio_allowance_us]
    if engine_dir is not None:
        command += ["--engine-dir", engine_dir]
    if keep_records:
        command += ["--keep-records", "--capacity", capacity]
    return command


def cusense_command(python, socket, artifact, output, ready, device, wait="spin"):
    return [python, "-m", "ocudu_cuda_isac.sensing.cusense", "follow", "--socket", socket, "--artifact", artifact,
            "--output-dir", output, "--wait", wait, "--connect-ms", 120000, "--idle-ms", 120000, "--ready-file", ready] \
        + ([] if device is None else ["--device", device])


def presence_command(python, socket, spec, artifact, output, ready, device="cpu", wait="spin"):
    return [python, "-m", "ocudu_cuda_isac.sensing.presence", "follow", "--socket", socket, "--spec", spec, "--artifact",
            artifact, "--output-dir", output, "--wait", wait, "--connect-ms", 120000, "--idle-ms", 120000,
            "--ready-file", ready, "--device", device]


def control_settings(control, thread_pct=50, radio_allowance_us=0):
    """What a contention control changes, on the sensor client only: (environment, release, allowance)."""
    if control not in mp.CONTROLS:
        raise LaunchError(f"unknown contention control {control!r}: " + ", ".join(mp.CONTROLS))
    if control == "mps_priority":
        return {"CUDA_MPS_CLIENT_PRIORITY": "1"}, "immediate", 0
    if control == "mps_thread_pct":
        return {"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(thread_pct)}, "immediate", 0
    if control == "release_after_radio":
        return {}, "after_radio", radio_allowance_us
    return {}, "immediate", 0


# ---------------------------------------------------------------- command line
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    daemon = commands.add_parser("mps", help="start, read or stop the MPS control daemon of this checkout")
    daemon.add_argument("action", choices=("start", "status", "stop"))
    daemon.add_argument("--pipe-dir", type=Path, help=f"default: {DEFAULT_MPS_DIR / 'pipe'}")
    daemon.add_argument("--log-dir", type=Path, help=f"default: {DEFAULT_MPS_DIR / 'log'}")
    daemon.add_argument("--visible", help="CUDA_VISIBLE_DEVICES of the daemon: one device, by index or UUID")
    one = commands.add_parser("arm", help="run one arm from its specification and write its record")
    one.add_argument("--spec", required=True, type=Path)
    one.add_argument("--record", required=True, type=Path, help="new file for the m1-launch/v1 record")
    one.add_argument("--pipe-dir", type=Path)
    one.add_argument("--log-dir", type=Path)
    args = parser.parse_args(argv)
    host = Host()
    try:
        if args.command == "mps":
            mps = Mps(host, args.pipe_dir, args.log_dir, visible=args.visible)
            if args.action == "start":
                print(f"run_m1: the MPS control daemon on {mps.pipe_dir} is {mps.start()}")
            elif args.action == "stop":
                stopped, text = mps.stop()
                print(f"run_m1: the MPS control daemon on {mps.pipe_dir} " + ("ended" if stopped else f"did not answer: {text}"))
                return 0 if stopped else 1
            found = mps.inventory()
            print(json.dumps(found, indent=2, sort_keys=True))
            problems = mp.inventory_problems(found) if found["servers"] else \
                [problem for problem in mp.inventory_problems(found) if "servers are listed" not in problem]
            for problem in problems:
                print(f"run_m1: {problem}", file=sys.stderr)
            return 1 if problems else 0
        if args.record.exists():
            parser.exit(2, f"run_m1: {args.record} already exists. Evidence is never overwritten.\n")
        spec = json.loads(args.spec.read_text(encoding="ascii"))
        mps = Mps(host, args.pipe_dir, args.log_dir) if spec.get("sharing") == "mps" else None
        record = launch(spec, args.record.parent, host, mps)
        args.record.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="ascii")
        print(f"run_m1: arm {record['arm']} ({record['label']}): sharing {record['sharing']['label']}, "
              f"{len(record['processes'])} processes, {len(record['problems'])} problem(s) ({args.record})")
        for problem in record["problems"]:
            print(f"run_m1: {problem}", file=sys.stderr)
        return 0 if clean(record) and record["sharing"]["label"] != "refused" else 1
    except (LaunchError, MpsPacingError, OSError, ValueError) as error:
        parser.exit(2, f"run_m1: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
