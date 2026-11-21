#!/usr/bin/env python3
"""Double of the native pilot_ls tool with its ring and its consumers, for hosts without an OCUDU build.

It stands in for one whole scenario of the Sprint 3.2 gate: the tool on a pilot-ls-plan/v2 file
and the consumer processes beside it. The receiver and the estimates are those of the double of
Sprint 3.1 (tests/unit/pilot_ls_double.py), which computes each observation with the reference.
On top of them this double models the ring: a slot count, admission by credit, publication in
sequence order, a consumer that keeps up, one that is slow, one that is killed while it holds a
credit and one that takes over in a new epoch. It writes what the native tool writes for either
evidence mode, and for every consumer the archive that obs_tail or the reference reader writes,
with the records packed by ocudu_cuda_isac.sensing.obs.

The unit tests use it to check the reader of the native output and every check of the
integration driver. SNAPSHOT_RING_FAULT names one fault to inject, so that each check can be
shown to fail:

    timed-hash      a timed run hashes between its first fill and its drain
    window-call     the producer makes a system call between an admission and its publication
    timed-payload   one payload of a timed run differs from the audited estimate in one bit
    timed-digest    the records of a timed run carry a digest
    lost-record     the consumer misses the last record that the producer counts as released
    reorder         two records reach the consumer in the wrong order
    unbalanced      a published record is neither released, reclaimed nor pending
    slow-tap        one observation takes 400 us
    changed-decode  with the ring attached, the CUDA processor returns a wrong byte
    no-drop         the slow consumer is not slow: the overload is never reached
    blocking        the producer waits 5 ms for a credit before it drops
    same-epoch      the consumer that takes over continues the epoch of the one that was killed
    host-fallback   the CUDA processor executes the host path
    accept-all      no plan is refused
    exit            the tool exits with status 3 after writing

It is not a test module: discovery skips it.
"""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ocudu_cuda_isac.episodes import pilot_ls as pl  # noqa: E402
from ocudu_cuda_isac.episodes import snapshot_ring as sr  # noqa: E402
from ocudu_cuda_isac.episodes.snapshot_ring import SnapshotRingError  # noqa: E402
from ocudu_cuda_isac.sensing import obs  # noqa: E402
import pilot_ls_double as tap_double  # noqa: E402

# One run of the Sprint 3.1 double per replay plan, tap and receiver fault. Every scenario of a
# gate decodes the same plan, so the model receiver runs once and its output is copied.
CACHE = {}


def base_output(plan, directory, native, fault):
    """The files of the Sprint 3.1 double for the first-version part of a plan, in native. Returns its status."""
    inner = {key: plan[key] for key in pl.PLAN_FIELDS}
    inner["schema"] = pl.PLAN_SCHEMA
    inner["run_id"] = "cached"
    key = (plan["replay_plan"]["sha256"], plan["observer"]["tap"], fault, os.environ.get("PILOT_LS_BUILD", ""))
    if key not in CACHE:
        home = Path(directory).parent / f".cache-{len(CACHE)}-{os.getpid()}"
        while home.exists():
            home = home.with_name(home.name + "x")
        (home / "native").mkdir(parents=True)
        (home / "plan.json").write_text(json.dumps(inner, indent=2, sort_keys=True) + "\n", encoding="ascii")
        with contextlib.redirect_stdout(io.StringIO()):
            status = tap_double.run(home / "plan.json", home / "native", [], True, fault)
        CACHE[key] = (home / "native", status)
    source, status = CACHE[key]
    shutil.copytree(source, native, dirs_exist_ok=True)
    return status


def header_of(record, mode, epoch, sequence, payload, mask, times):
    """The obs/v1 header of one observation record of Sprint 3.1, as the producer glue builds it."""
    shape, allocation = record["shape"], record["allocation"]
    symbols = 0
    for symbol in allocation["dmrs_symbols"]:
        symbols |= 1 << symbol
    return obs.new_header(
        epoch=epoch, sequence=sequence, episode=int(record["keys"]["episode"], 16),
        opportunity=int(record["keys"]["opportunity"], 16), grant=int(record["keys"]["grant"], 16),
        absolute_slot=record["radio"]["absolute_slot"], cell=record["radio"]["cell"], ue=record["radio"]["ue"],
        stage=2, source=1, layout=2, dtype=1, producer_steps=1, scale=record["what"]["scale"], rank=3,
        shape=shape["shape"], strides=shape["strides"], first_subcarrier=allocation["first_subcarrier"],
        subcarrier_step=allocation["subcarrier_step"], subcarrier_count=allocation["subcarrier_count"],
        dmrs_symbols=symbols, rx_count=allocation["rx_count"], layer_count=1, antenna_map=allocation["antenna_map"],
        mask_bytes=len(mask), payload_bytes=len(payload), host_bytes=obs.STAGED_HOST.size, time_unit=1,
        digest_kind=1 if mode == "audit" else 0, evidence=1 if mode == "audit" else 2, port=record["what"]["port"],
        t_first_sample=record["time"]["t_first_sample"], t_last_sample=record["time"]["t_last_sample"],
        t_ready_ns=times["t_ready_ns"], t_publish_ns=times["t_publish_ns"],
        digest=hashlib.sha256(payload).digest() if mode == "audit" else b"", valid=record["validity"]["valid"])


def scenario(tools, plan_path, native, directory, consumers, extra=()):
    """One scenario of the gate, as tests/integration/snapshot_ring.py run_scenario runs it."""
    fault = os.environ.get("SNAPSHOT_RING_FAULT", "")
    native.mkdir(parents=True)
    for consumer in consumers:
        consumer["archive"].mkdir(parents=True)
    log = directory / "native.log"
    statuses = [9 if consumer.get("exit_after") is not None else 0 for consumer in consumers]
    raw = Path(plan_path).read_bytes()
    try:
        plan = json.loads(raw)
        if fault != "accept-all":
            sr.check_plan(plan)
        elif plan.get("ring") is not None and plan["observer"]["tap"] == "none":
            plan["ring"] = None  # a tool that refuses nothing still has nothing to publish
    except (ValueError, SnapshotRingError) as error:
        log.write_text(f"pilot_ls --plan {plan_path} --output-dir {native}\npilot_ls: input error: {error}\n")
        return 2, None, statuses
    mode, ring, tap = plan["evidence"]["mode"], plan["ring"], plan["observer"]["tap"]
    timed, tapped = mode == "timed", tap == "pilot_ls"
    inner = "changed-decode" if fault == "changed-decode" and ring else fault if fault == "host-fallback" else ""
    try:
        status = base_output(plan, directory, native, inner)
    except Exception as error:  # a process that crashes exits with status 1
        log.write_text(f"pilot_ls --plan {plan_path} --output-dir {native}\n{type(error).__name__}: {error}\n")
        return 1, None, statuses
    report = json.loads((native / "native.json").read_text())
    radio = [json.loads(line) for line in (native / "radio/radio.jsonl").read_text().splitlines() if line.strip()]
    for record in radio:
        record["run_id"] = plan["run_id"]
    (native / "radio/radio.jsonl").write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in radio),
                                              encoding="ascii")
    observations, payloads, masks = [], {}, {}
    if tapped:
        folder = native / pl.OBSERVER_DIR
        observations = [json.loads(line) for line in (folder / pl.RECORDS_FILE).read_text().splitlines() if line.strip()]
        for record in observations:
            record["run_id"] = plan["run_id"]
            payloads[record["host"]["opportunity_id"]] = (folder / record["payload"]["file"]).read_bytes()
            masks[record["host"]["opportunity_id"]] = (folder / record["validity"]["file"]).read_bytes()
        (folder / pl.RECORDS_FILE).write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in observations),
                                              encoding="ascii")

    # ---- The ring: admission by credit, sequence order, epochs.
    lines, kept, ring_report = [], [[] for _ in consumers], None
    if ring is not None:
        slots, epoch, published, released = ring["slots"], plan["observer"]["epoch"], 0, 0
        totals = dict.fromkeys(sr.COUNT_FIELDS, 0)
        reader = 0 if consumers else None  # the consumer that is attached
        totals["epochs"] = 1 if consumers else 0
        read_by_first, since_death, clock = 0, None, 1_000_000_000
        entries = {item["sequence"]: item for item in report["opportunities"]}
        for position, record in enumerate(observations):
            host, opportunity = record["host"], record["host"]["opportunity_id"]
            entry = entries[host["opportunity_sequence"]]
            clock += 1_000_000
            # A consumer that takes over: two opportunities after the first one is gone.
            if since_death is not None and len(consumers) > 1 and reader is None:
                since_death += 1
                if since_death > 2:
                    reader, since_death = 1, None
                    totals["epochs"] += 1
                    if fault != "same-epoch":
                        totals["reclaimed"] += published - released
                        totals["released"] += released
                        epoch, published, released = epoch + 1, 0, 0
            # A slow consumer finishes one record in the time of three opportunities, published or not.
            if reader is not None and consumers[reader].get("hold_us") and fault != "no-drop" and position % 3 == 2 \
                    and released < published:
                released += 1
            held = published - released
            base = {"schema": sr.PUBLICATION_SCHEMA, "run_id": plan["run_id"], "mode": mode,
                    "opportunity_id": opportunity, "opportunity_sequence": host["opportunity_sequence"],
                    "keys": {name: record["keys"][name] for name in ("episode", "opportunity", "grant")},
                    "absolute_slot": record["radio"]["absolute_slot"], "payload_bytes": record["payload"]["bytes"],
                    "t_read_ns": clock, "window_calls": 0}
            if held >= slots:
                waited = 5000.0 if fault == "blocking" else 0.05
                lines.append(dict(base, outcome="dropped", reason="no_credit", epoch=None, sequence=None, depth=None,
                                  bytes=None, grid=None, valid=None, holes=None, t_ready_ns=None, t_publish_ns=None,
                                  t_done_ns=clock + int(waited * 1000), service_us=waited, estimate_us=None,
                                  publish_us=waited))
                totals["dropped"] += 1
                continue
            estimate = 1400.0 if not timed else 7.0
            service = estimate + (0.4 if timed else 3.0)
            if fault == "slow-tap" and timed and position == 3:
                service = 400.0
            line = dict(base, outcome="published", reason=None, epoch=epoch, sequence=published, depth=held,
                        bytes=obs.RECORD_HEADER_BYTES + obs.STAGED_HOST.size + record["validity"]["mask_bytes"]
                        + record["payload"]["bytes"],
                        grid={"grid_index": entry["grid_index"], "generation": entry["generation"]},
                        valid=record["validity"]["valid"], holes=record["validity"]["holes"],
                        t_ready_ns=clock + int(estimate * 1000), t_publish_ns=clock + int(service * 1000) - 100,
                        t_done_ns=clock + int(service * 1000), service_us=service, estimate_us=estimate,
                        publish_us=service - estimate)
            if fault == "window-call" and not lines:
                line["window_calls"] = totals["window_calls"] = 1
            lines.append(line)
            published += 1
            totals["published"] += 1
            totals["admitted"] += 1
            totals["max_depth"] = max(totals["max_depth"], held)
            if reader is None:
                continue
            consumer = consumers[reader]
            if ring["wait"] == "wait":
                totals["wakes"] += 1
            if consumer.get("exit_after") is not None:
                read_by_first += 1
                if read_by_first >= consumer["exit_after"]:
                    released, reader, since_death = published - 1, None, 0  # killed while it holds one credit
                else:
                    released = published
                continue
            kept[reader].append((line, record))
            if not consumer.get("hold_us") or fault == "no-drop":
                released = published
        # The drain: an attached consumer reads and releases what it holds.
        if reader is not None:
            released = published
        totals["released"] += released
        totals["pending"] = published - released
        if fault == "unbalanced":
            totals["released"] -= 1
        ring_report = {
            "schema": sr.RING_SCHEMA, "layout": "obs-ring/v1", "definition": f"{obs.DEFINITION:016x}",
            "stream": ring["stream"], "socket": f"{sr.RING_DIR}/{ring['stream']}.sock",
            "window_records": ring["window_records"], "burst_records": ring["burst_records"], "slot_count": slots,
            "slot_bytes": 20224, "segment_bytes": 4096 + slots * 20224, "payload_capacity": 19584, "mask_capacity": 306,
            "host_capacity": obs.STAGED_HOST.size, "memory_path": "host_shared", "clock": "model clock of the double",
            "wait": {"planned": ring["wait"], "consumer": consumers[0]["wait"] if consumers else None},
            "attach": ring["attach"], "attached_at_start": bool(consumers), "attached_at_drain": reader is not None,
            "first_epoch": plan["observer"]["epoch"], "last_epoch": epoch, "drain_ms": ring["drain_ms"],
            "pending_at_cutoff": totals["pending"], "eligible": len(lines), "counts": totals}
        (native / sr.RING_DIR).mkdir()
        (native / sr.RING_DIR / sr.PUBLICATIONS_FILE).write_text(
            "".join(json.dumps(line, sort_keys=True) + "\n" for line in lines), encoding="ascii")
        (native / sr.RING_DIR / sr.RING_FILE).write_text(json.dumps(ring_report, indent=2, sort_keys=True) + "\n",
                                                         encoding="ascii")

    # ---- What each consumer kept.
    for index, (consumer, pairs) in enumerate(zip(consumers, kept)):
        if consumer.get("exit_after") is not None:
            continue
        if fault == "lost-record" and timed:
            pairs = pairs[:-1]
        rows, written = [], []
        for line, record in pairs:
            opportunity = record["host"]["opportunity_id"]
            payload, mask = payloads[opportunity], masks[opportunity]
            if fault == "timed-payload" and timed and not rows:
                payload = bytes([payload[0] ^ 1]) + payload[1:]
            header = header_of(record, "audit" if fault == "timed-digest" and timed else mode, line["epoch"],
                               line["sequence"], payload, mask, line)
            host = obs.STAGED_HOST.pack(1, line["opportunity_sequence"], line["grid"]["grid_index"], record["radio"]["sfn"],
                                        line["grid"]["generation"], record["radio"]["slot"], 0, line["t_read_ns"])
            written.append(obs.Record(header, host, mask, payload, depth=line["depth"]))
            delay = 60_000 if consumer["wait"] == "wait" else 400
            rows.append({"epoch": line["epoch"], "sequence": line["sequence"], "opportunity": line["keys"]["opportunity"],
                         "depth": line["depth"], "bytes": len(written[-1].to_bytes()),
                         "t_publish_ns": line["t_publish_ns"], "t_read_ns": line["t_publish_ns"] + delay,
                         "delay_ns": delay, "intact": True, "file": True})
        if fault == "reorder" and timed and len(written) > 2:
            written[1], written[2] = written[2], written[1]
            names = [obs.record_name(row["epoch"], row["sequence"]) for row in rows]
        else:
            names = [obs.record_name(record.header["epoch"], record.sequence) for record in written]
        for name, record in zip(names, written):
            (consumer["archive"] / name).write_bytes(record.to_bytes())
        (consumer["archive"] / "consumer.json").write_text(json.dumps({
            "schema": obs.CONSUMER_SCHEMA, "reader": consumer["reader"], "definition": f"{obs.DEFINITION:016x}",
            "wait": consumer["wait"], "epoch": rows[0]["epoch"] if rows else plan["observer"]["epoch"], "pid": 1,
            "cpu": -1, "hold_us": consumer.get("hold_us", 0), "window": 0, "slot_count": ring["slots"],
            "slot_bytes": 20224, "attached_ns": 1, "ended": "closed",
            "counts": {"read": len(rows), "stale": 0, "released": len(rows), "skipped": 0,
                       "sleeps": len(rows) if consumer["wait"] == "wait" else 0, "kept": len(rows), "not_kept": 0,
                       "system_calls": 17}, "records": rows}, indent=2) + "\n", encoding="ascii")

    # ---- The report of a plan of version 2.
    zero = {"digests": 0, "updates": 0, "bytes": 0}
    hashed = {"digests": 9 * len(report["opportunities"]), "updates": 300 * len(report["opportunities"]),
              "bytes": 2_200_000 * len(report["opportunities"])}
    report.update(schema=sr.NATIVE_SCHEMA, run_id=plan["run_id"], ring=ring_report,
                  plan={"file": str(plan_path), "sha256": hashlib.sha256(raw).hexdigest()})
    report["evidence"] = {
        "mode": mode, "plan_version": 2, "window": "from before the first fill to the drain of the ring",
        "hash_in_window": (dict(zero, digests=3, updates=3, bytes=64) if fault == "timed-hash" else zero) if timed else hashed,
        "hash_after_drain": hashed if timed else zero, "hash_before_first_fill": {"digests": 40, "updates": 40, "bytes": 9000},
        "ownership": "the ticket of the fill" if timed else "the readback digest of the grid",
        "records_written": "after the last opportunity" if timed else "as each opportunity is done"}
    by_sequence = {line["opportunity_sequence"]: line for line in lines}
    first = "the first observation of the run"
    times = [line for line in lines if line["outcome"] == "published"][1:]

    def stats(values):
        made = {"count": len(values), "excluded": first}
        if values:
            made.update(mean=sum(values) / len(values), min=min(values), max=max(values))
        return made

    report["observer"]["mode"] = mode
    if tapped and timed:
        shutil.rmtree(native / pl.OBSERVER_DIR)
        for entry in report["opportunities"]:
            line = by_sequence.get(entry["sequence"])
            if line is not None:
                entry["observer"] = {"sequence": line["sequence"], "ring": line["outcome"], "payload_sha256": None,
                                     "mask_sha256": None, "ingress_sha256": None, "valid": line["valid"],
                                     "holes": line["holes"], "service_us": line["service_us"]}
        report["observer"].update(records=len(lines), service_us=stats([line["service_us"] for line in times]),
                                  memory="a slot of the model ring")
        report["observer"]["parts"] = {"estimate_us": stats([line["estimate_us"] for line in times]),
                                       "publish_us": stats([line["publish_us"] for line in times])}
        report["totals"]["observations"] = len(lines)
    elif tapped:
        for entry in report["opportunities"]:
            line = by_sequence.get(entry["sequence"])
            if line is not None and entry.get("observer"):
                entry["observer"]["ring"] = line["outcome"]
        count = max(len(observations) - 1, 0)
        report["observer"]["service_us"] = dict(stats([1471.0] * count))
        report["observer"]["parts"] = {
            "estimate_us": stats([6.7] * count), "grid_digest_us": stats([1356.4] * count),
            "keys_mask_us": stats([3.4] * count), "payload_digest_us": stats([97.0] * count),
            "publish_us": stats([line["publish_us"] for line in times]) if ring is not None else None}
    else:
        report["observer"]["parts"] = None
    (native / "native.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii")
    log.write_text(f"pilot_ls --plan {plan_path} --output-dir {native}\npilot_ls double: {plan['run_id']}, tap {tap}, "
                   f"evidence {mode}, {len(lines)} observations on the ring\n")
    return (3 if fault == "exit" else status), report, statuses
