#!/usr/bin/env python3
"""Double of the launcher of Sprint 4.1 and of the processes it starts, for hosts without an OCUDU build, a GPU or MPS.

It stands in for scripts/run_m1.py launch(): one arm specification in, one m1-launch/v1 record
out, with everything the processes of the arm would have written.

  pilot_ls     the double of Sprint 5.2 (tests/unit/native_estimate_double.py) on the plan of
               version 4 without its schedule, and then what a paced run adds: the report of
               version 4, the offered schedule and pacing/releases.jsonl, written by the release
               rule of the reference (mps_pacing.Pacer) on a clock of the double. The service time
               of a decode is the one its radio record names.
  obs_tail     the archive that the consumer of the double kept of its ring.
  sense_host   the real sensor of the host twin of the core, when the caller names its build, on
               the records the double kept of the O3 ring: an archive replay, not a live ring.
  cusense      the reference sensor, in this interpreter, on the same records.
  presence     the reference of the delay-Doppler branch, in this interpreter, on the O2 records.
  sense_feed   no ring: the sensor reads the archive the feed was given.
  MPS          an inventory of one device with the processes of the arm as clients.

MPS_PACING_FAULT names one fault to inject, so that each check of the driver can be shown to fail:

    skipped-release    the receiver with the tap off skips the release of one opportunity behind the first ones
    early-start        a job starts before its offered release time
    shifted-schedule   every release after the first lies 100 us later than the plan offers
    other-service      a release names another service time than its radio record
    no-client          the sensor is no client of the MPS server
    late-context       the sensor's context is a client only after the receiver started
    drop               the consumer of the O3 ring is slow: records are dropped
    sensor-exit        the sensor ends with status 3
    receiver-exit      the receiver ends with status 3 after writing
    no-lead-in         the receiver runs a plan with a lead-in as a plan without one
    cold-schedule      a release is the first use of the processor although the lead-in ran
    late-lead-in       the first decode of the lead-in takes so long that the lead-in ends inside the schedule

It is not a test module: discovery skips it.
"""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ocudu_cuda_isac.episodes import mps_pacing as mp  # noqa: E402
from ocudu_cuda_isac.episodes import native_estimate as ne  # noqa: E402
import native_estimate_double as estimate_double  # noqa: E402
import run_m1  # noqa: E402

START_NS = 50_000_000_000
DEVICE = {"index": 0, "uuid": "GPU-00000000-double", "name": "double GB10"}
SENSE_HOST = None  # the sense_host of a host twin of the core, set by the caller
SCENARIO = None    # the scenario of the double of Sprint 5.2, or a cache in front of it, set by the caller


def _argument(command, flag, default=None):
    return command[command.index(flag) + 1] if flag in command else default


def _stream(socket):
    return "o3" if Path(socket).parent.name == ne.ESTIMATE_DIR else "o2"


def pace(native, plan, plan_path, fault):
    """Turn the output of the double of Sprint 5.2 into the output of a paced run of the plan of version 4."""
    report = json.loads((native / "native.json").read_text())
    pacing = plan["pacing"]
    radio_path = native / "radio" / "radio.jsonl"
    radio = [json.loads(line) for line in radio_path.read_text().splitlines() if line.strip()]
    for record in radio:
        if record["timing"]["service_us"] is not None:
            record["timing"]["service_us"] = round(record["timing"]["service_us"], 3)
    by_sequence = {}
    for record in radio:
        by_sequence.setdefault(record["opportunity_sequence"], []).append(record)
    moved = 100 if fault == "shifted-schedule" else 0
    offsets = [(offset + (moved if position else 0)) * 1000 for position, offset in enumerate(pacing["release_us"])]
    pacer = mp.Pacer(offsets, pacing["allowance_us"] * 1000, pacing["latest_start_us"] * 1000)
    start = START_NS + pacing["start_delay_us"] * 1000
    pacer.begin(start)
    now, lines = START_NS, []
    # The lead-in of the plan: the first releases once more, ahead of the schedule, on the clock of the double.
    # A decode of it is no release and no radio record. Its first decode is cold, as a processor's first use is.
    lead = None if fault == "no-lead-in" else mp.lead_in_of(pacing)
    ahead, names = [], [backend["name"] for backend in report["backends"]]
    service_ahead = {name: [] for name in names}
    cold = (1310.0, 236.0, 200.5, 185.25, 181.0, 180.5)
    for k in range(lead["releases"] if lead else 0):
        item, offset = report["opportunities"][k], offsets[k] // 1000  # the schedule the double runs, shifted or not
        issue = start + (offset - lead["lead_us"]) * 1000
        now += 80_000
        ready = now
        began_job = now = max(now, issue)
        decodes = []
        for record in by_sequence[item["sequence"]]:
            # The double keeps a ring whole, so only the plan with the tap off can lose releases to a slow lead-in.
            slow = fault == "late-lead-in" and k == 0 and plan["observer"]["tap"] == "none"
            took = 9950.0 if slow else cold[min(k, len(cold) - 1)]
            began, now = now, now + int(round(took * 1000))
            decodes.append({"backend": record["backend"], "key": record["key"], "outcome": "decoded", "crc_ok": True,
                            "payload_match": True, "t_start_ns": began, "t_callback_ns": now, "service_us": took,
                            "path": record["path"]})
            service_ahead[record["backend"]].append(took)
        now += 6_000
        ahead.append({"schema": mp.LEAD_IN_SCHEMA, "run_id": report["run_id"], "sequence": k,
                      "opportunity_sequence": item["sequence"], "opportunity_id": item["opportunity_id"],
                      "sfn": item["sfn"], "slot": item["slot"], "grant_id": item["grant_id"], "offset_us": offset,
                      "lead_us": lead["lead_us"], "issued": True, "t_issue_ns": issue, "t_ready_ns": ready,
                      "t_start_ns": began_job, "t_done_ns": now, "fill_us": 76.5, "released": True, "decodes": decodes})
    for position, item in enumerate(report["opportunities"]):
        records = by_sequence[item["sequence"]]
        # Behind the first releases, which a lead-in answers for: this fault is about the schedule alone.
        skipped = fault == "skipped-release" and position == 7 and plan["observer"]["tap"] == "none"
        if skipped:
            now = pacer.release_time(position) + pacing["latest_start_us"] * 1000 + 1
        taken = pacer.take(position, now)
        decodes = []
        if not taken:
            # What a skipped release leaves: no grid, and every decode of it not dispatched.
            item.update(grid_index=None)
            for name in ("generation", "readback_sha256", "fill_us", "retained", "released"):
                item.pop(name, None)
            for record, decode in zip(records, item["decodes"]):
                record.update(outcome="not_dispatched", crc_ok=False, payload_match=False, decoded_sha256=None,
                              decoded_file=None, ingress=None, error="skipped release: the receiver was free after the "
                                                                     "latest start of this job", path=None, decoder=None)
                record["timing"]["service_us"] = None
                decode.update(outcome="not_dispatched", crc_ok=False, payload_match=False, service_us=None)
                decodes.append({"backend": record["backend"], "key": record["key"], "outcome": "not_dispatched",
                                "skip": "missed_release", "t_start_ns": None, "t_callback_ns": None, "service_us": None})
        else:
            now += 80_000
            pacer.ready(position, now)
            now = max(now, pacer.release_time(position)) + 40
            if fault == "early-start" and position == 2:
                now = pacer.release_time(position) - 5
            pacer.start(position, now)
            now += 12_000
            for record, decode in zip(records, item["decodes"]):
                service = record["timing"]["service_us"]
                decode["service_us"] = service
                began, called = now, None
                if service is not None:
                    called = began + int(round(service * 1000))
                    now = called
                shown = service + 1.0 if fault == "other-service" and position == 0 and service is not None else service
                decodes.append({"backend": record["backend"], "key": record["key"], "outcome": record["outcome"],
                                "skip": None, "t_start_ns": began, "t_callback_ns": called, "service_us": shown})
            now += 6_000
            pacer.done(position, now)
        mark = pacer.marks[position]
        offset = offsets[position] // 1000
        lines.append({"schema": mp.RELEASE_SCHEMA, "run_id": report["run_id"], "sequence": item["sequence"],
                      "opportunity_id": item["opportunity_id"], "sfn": item["sfn"], "slot": item["slot"],
                      "offset_us": offset, "outcome": mark["outcome"], "overrun": mark["overrun"],
                      "t_release_ns": start + offset * 1000, "t_free_ns": mark["free"], "t_ready_ns": mark["ready"],
                      "t_start_ns": mark["start"], "t_done_ns": mark["done"], "fill_us": 76.5 if taken else None,
                      "decodes": decodes})
    if any(line["outcome"] == "skipped" for line in lines):
        kept = [json.loads(line) for line in (native / "ingress" / "ingress.jsonl").read_text().splitlines() if line.strip()]
        gone = {line["sequence"] for line in lines if line["outcome"] == "skipped"}
        (native / "ingress" / "ingress.jsonl").write_text(
            "".join(json.dumps(line) + "\n" for line in kept if line["sequence"] not in gone))
        for record in radio:
            if record["opportunity_sequence"] in gone:
                for name in list((native / "radio").glob(f"{record['opportunity_sequence']:04d}.*.bin")):
                    name.unlink()
        outcomes = report["totals"]["outcomes"]
        outcomes["decoded"] -= len(gone) * len(report["backends"])
        outcomes["not_dispatched"] = outcomes.get("not_dispatched", 0) + len(gone) * len(report["backends"])
    if lead and fault != "cold-schedule":
        for record in radio:
            record["timing"]["first_use"] = False  # the first use of the processor was a decode of the lead-in
    radio_path.write_text("".join(json.dumps(record) + "\n" for record in radio))
    folder = native / mp.PACING_DIR
    folder.mkdir()
    (folder / mp.RELEASES_FILE).write_text("".join(json.dumps(line) + "\n" for line in lines))
    led = None
    if lead:
        (folder / mp.LEAD_IN_FILE).write_text("".join(json.dumps(line) + "\n" for line in ahead))
        led = {"schema": mp.LEAD_IN_SCHEMA, "file": f"{mp.PACING_DIR}/{mp.LEAD_IN_FILE}", "releases": lead["releases"],
               "lead_us": lead["lead_us"], "rule": "the lead-in of the reference", "observer": "none",
               "issued": len(ahead), "decodes": len(ahead) * len(names), "decoded": len(ahead) * len(names),
               "t_first_start_ns": ahead[0]["t_start_ns"], "t_last_done_ns": ahead[-1]["t_done_ns"],
               "before_first_release_us": (lines[0]["t_release_ns"] - ahead[-1]["t_done_ns"]) / 1000.0,
               "service_us": service_ahead}
    report.update(schema=mp.NATIVE_SCHEMA, plan={"file": str(plan_path), "sha256": hashlib.sha256(Path(plan_path).read_bytes()).hexdigest()})
    report["evidence"]["plan_version"] = 4
    report["access"]["pacing"] = "pacing/ holds one pace-release/v1 line per opportunity of a paced run"
    report["pacing"] = {"schema": mp.PACING_SCHEMA, "file": f"{mp.PACING_DIR}/{mp.RELEASES_FILE}", "clock": mp.CLOCK,
                        "clock_source": "a clock of the double", "rule": "the release rule of the reference",
                        "wait": "spin", "t_start_ns": start, "start_delay_us": pacing["start_delay_us"],
                        "allowance_us": pacing["allowance_us"], "latest_start_us": pacing["latest_start_us"],
                        "first_offset_us": lines[0]["offset_us"], "last_offset_us": lines[-1]["offset_us"],
                        "counts": pacer.counts(), "book": "the double keeps no book"}
    if mp.lead_in_of(pacing) is not None and fault != "no-lead-in":
        report["pacing"]["lead_in"] = led
    (native / "native.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return report


def receiver(spec, directory, fault):
    """The producer pilot_ls and the consumers of its rings. Returns (status, {role: status}, {role: records kept})."""
    command = spec["producer"]["command"]
    plan_path, native = Path(_argument(command, "--plan")), Path(_argument(command, "--output-dir"))
    plan = mp.load_plan(plan_path)
    inner = {key: plan[key] for key in ne.PLAN_FIELDS}
    inner["schema"] = ne.PLAN_SCHEMA
    inner_path = directory / "plan-v3.json"
    inner_path.write_text(json.dumps(inner, indent=2, sort_keys=True) + "\n", encoding="ascii")
    consumers, kept = [], {}
    listed = list(spec["consumers"])
    if spec["fault"] and spec["fault"]["restart"]:
        listed.append(spec["fault"]["restart"])
    for item in listed:
        socket = _argument(item["command"], "--socket")
        where = Path(_argument(item["command"], "--output-dir")) if item["kind"] == "obs_tail" \
            else directory / f"double-records-{item['role']}"
        if where.is_dir() and not any(where.iterdir()):
            where.rmdir()
        kept[item["role"]] = where
        if item["role"] == "takeover":
            continue  # the double has one consumer per ring: what took over is copied from the first below
        consumer = {"reader": "native", "wait": "spin", "stream": _stream(socket), "archive": where}
        if fault == "drop" and consumer["stream"] == "o3" or spec["fault"] and item["role"] == spec["fault"]["role"]:
            consumer["hold_us"] = 5000
        consumers.append(consumer)
    native.rmdir()
    scenario = SCENARIO or estimate_double.scenario
    status, report, _ = scenario({}, inner_path, native, directory, consumers)
    inner_path.unlink(missing_ok=True)
    if status == 0 and report is not None:
        pace(native, plan, plan_path, fault)
    if "takeover" in kept and kept.get("o2") and kept["o2"].is_dir():
        shutil.copytree(kept["o2"], kept["takeover"])
    return (3 if fault == "receiver-exit" else status), kept


def sensor(item, records, fault):
    """One sensor on the records the double kept. Returns its status."""
    command, kind = item["command"], item["kind"]
    output, artifact = Path(_argument(command, "--output-dir")), _argument(command, "--artifact")
    ready = _argument(command, "--ready-file")
    if fault == "sensor-exit" and kind in ("sense_host", "cusense"):
        return 3
    if ready:
        Path(ready).write_text("{}\n")
    if not Path(records).is_dir() or not any(Path(records).glob("*.obs")):
        return 2
    if kind == "sense_host":
        if SENSE_HOST is None:
            raise RuntimeError("the double runs the real sense_host of a host twin: name its path in SENSE_HOST")
        run = [str(SENSE_HOST), "--archive", str(records), "--artifact", artifact, "--output-dir", str(output), "--device",
               "host", "--engine", "host_reference", "--evidence", "timed", "--expect-sha256",
               _argument(command, "--expect-sha256"), "--lines", _argument(command, "--lines", "16384"), "--windows",
               _argument(command, "--windows", "4096")]
        done = subprocess.run(run, capture_output=True, text=True, timeout=600, check=False)
        (output.parent / f"{item['role']}.log").write_text(" ".join(run) + "\n" + done.stdout + done.stderr)
        return done.returncode
    from ocudu_cuda_isac.sensing import cusense, presence
    if kind == "cusense":
        main, arguments = cusense.main, ["replay", "--archive", str(records), "--artifact", artifact, "--output-dir",
                                         str(output), "--device", "cpu"]
    else:
        main, arguments = presence.main, ["replay", "--archive", str(records), "--spec", _argument(command, "--spec"),
                                          "--artifact", artifact, "--output-dir", str(output), "--device", "cpu"]
    said = io.StringIO()
    loads = (cusense.Model.LOADS, presence.Head.LOADS)
    with contextlib.redirect_stdout(said), contextlib.redirect_stderr(said):
        try:
            status = main(arguments)
        except SystemExit as stop:
            status = stop.code
    cusense.Model.LOADS, presence.Head.LOADS = loads
    (output.parent / f"{item['role']}.log").write_text(" ".join(arguments) + "\n" + said.getvalue())
    return status or 0


def launch(spec, directory, host=None, mps=None, base_env=None):
    """What run_m1.launch returns for an arm, with the outputs of its processes written by doubles."""
    run_m1.check_spec(spec)
    directory = Path(directory)
    fault = os.environ.get("MPS_PACING_FAULT", "")
    clock, pid = [START_NS - 3_000_000_000], [4700]
    record = {"schema": run_m1.LAUNCH_SCHEMA, "arm": spec["arm"], "label": spec["label"],
              "sharing": {"requested": spec["sharing"], "label": None, "verified": False, "reason": None, "proof": None,
                          "context_before_receiver": None},
              "processes": [], "order": [], "inventories": [], "fault": None, "problems": [],
              "host": {"before": {"host": "double", "monotonic_ns": clock[0]}, "after": None}, "double": True}

    def tick(amount=100_000_000):
        clock[0] += amount
        return clock[0]

    def started(item):
        pid[0] += 1
        python = item["kind"] in ("cusense", "presence")
        entry = {"role": item["role"], "kind": item["kind"], "command": item["command"], "pid": pid[0],
                 "cpus": {"requested": item["cpus"], "applied": run_m1.parse_cpus(item["cpus"]) or None},
                 "env": dict(item["env"]), "cuda": item["cuda"], "log": f"{item['role']}.log", "status": None,
                 "started_ns": tick(), "ready_ns": None, "ended_ns": None, "ended_by": None,
                 "maps_python": python if item["ready_file"] else None}
        record["processes"].append(entry)
        record["order"].append({"event": "started", "role": item["role"], "pid": entry["pid"], "t_ns": entry["started_ns"]})
        if item["ready_file"]:
            entry["ready_ns"] = tick()
            record["order"].append({"event": "ready", "role": item["role"], "pid": entry["pid"], "t_ns": entry["ready_ns"]})
        return entry

    def ended(entry, status, by="itself"):
        entry.update(status=status, ended_ns=tick(), ended_by=by)
        record["order"].append({"event": "ended", "role": entry["role"], "pid": entry["pid"], "t_ns": entry["ended_ns"]})

    entries = {item["role"]: started(item) for item in spec["consumers"]}
    producer = started(spec["producer"])
    statuses = {}
    if spec["producer"]["kind"] == "pilot_ls":
        status, kept = receiver(spec, directory, fault)
        (directory / "producer.log").write_text("pilot_ls double\n")
        for item in spec["consumers"]:
            if item["kind"] == "obs_tail":
                statuses[item["role"]] = 0 if kept[item["role"]].is_dir() else 2
            else:
                statuses[item["role"]] = sensor(item, kept[item["role"]], fault)
    else:
        archive = Path(_argument(spec["producer"]["command"], "--archive"))
        count = len(list(archive.glob("*.obs")))
        (directory / "producer.log").write_text("sense_feed double\n" + json.dumps(
            {"offered": count, "published": count, "admitted": count, "dropped": 0, "released": count,
             "pending_at_cutoff": 0, "faults": 0, "pace": "slots"}) + "\n")
        status = 0
        for item in spec["consumers"]:
            statuses[item["role"]] = sensor(item, archive, fault)
    if spec["fault"] is not None:
        note = spec["fault"]
        killed = entries[note["role"]]
        ended(killed, -9, "fault")
        statuses.pop(note["role"])
        record["fault"] = {"kind": note["kind"], "role": note["role"], "after_ms": note["after_ms"],
                           "clock": "schedule_start", "clock_start_ns": killed["ended_ns"] - note["after_ms"] * 1_000_000,
                           "applied_ns": killed["ended_ns"], "pid": killed["pid"], "restarted": None,
                           "producer_alive_after": True}
        if note["restart"] is not None:
            again = started(note["restart"])
            entries[again["role"]] = again
            statuses[again["role"]] = 0
            record["fault"]["restarted"] = {"role": again["role"], "pid": again["pid"], "started_ns": again["started_ns"]}
    ended(producer, status)
    for role, state in statuses.items():
        ended(entries[role], state)
    needed = {entry["role"]: entry["pid"] for entry in record["processes"] if entry["cuda"]}
    sharing = record["sharing"]
    if spec["sharing"] != "mps":
        sharing.update(label="time_slicing", reason="the processes of this arm were started outside the MPS daemon: a "
                                                    "labeled control")
    else:
        clients = [entry["pid"] for entry in record["processes"]
                   if entry["cuda"] or entry["kind"] in ("sense_host", "pilot_ls", "cusense")]
        if fault == "no-client":
            clients = [client for client in clients if client != entries.get("sensor", {}).get("pid")]
        roles = needed or {entry["role"]: entry["pid"] for entry in record["processes"]
                           if entry["kind"] in ("sense_host", "pilot_ls", "cusense")}
        found = {"schema": mp.INVENTORY_SCHEMA, "t_monotonic_ns": tick(), "daemon": {"alive": True}, "servers": [900],
                 "clients": {"900": clients}, "devices": [DEVICE]}
        record["inventories"].append(found)
        try:
            proof = mp.verify_clients(found, roles, spec["extra_clients"])
            sharing.update(label="mps", verified=True, proof=dict(proof, t_monotonic_ns=found["t_monotonic_ns"]),
                           context_before_receiver=fault != "late-context")
        except mp.MpsPacingError as error:
            sharing.update(label="refused", reason=str(error), context_before_receiver=False)
            record["problems"].append(str(error))
    record["host"]["after"] = {"host": "double", "monotonic_ns": tick()}
    return record


class Mps:
    """A double of the MPS control daemon of an arm: one device, no client before the first arm."""

    def __init__(self, host, pipe_dir=None, log_dir=None, control=None, visible=None, base_env=None):
        self.pipe_dir, self.log_dir = Path(pipe_dir or "/nowhere/pipe"), Path(log_dir or "/nowhere/log")

    def start(self, wait_s=10.0):
        return "running"

    def inventory(self):
        clients = {"900": [1]} if os.environ.get("MPS_PACING_FAULT") == "foreign-client" else {}
        return {"schema": mp.INVENTORY_SCHEMA, "t_monotonic_ns": START_NS,
                "daemon": {"alive": True, "pipe_directory": str(self.pipe_dir), "log_directory": str(self.log_dir),
                           "visible_devices": None},
                "servers": [900] if clients else [], "clients": clients,
                "active_thread_percentage": {"default": "100.0", "servers": {}},
                "default_client_priority": {"status": 0, "text": "0"}, "devices": [DEVICE], "compute_apps": None}
