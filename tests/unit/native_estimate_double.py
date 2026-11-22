#!/usr/bin/env python3
"""Double of the native pilot_ls tool with the route to the receiver's estimate, for hosts without an OCUDU build.

It stands in for one scenario of the Sprint 5.2 gate: the tool on a pilot-ls-plan/v3 file and the
consumers of its two streams. Everything a plan of version 2 asks for is done by the double of
Sprint 3.2 (tests/unit/snapshot_ring_double.py). On top of it this double models the receiver
patch: for every decode that the model receiver ran on its resident path it makes the estimate
that the matched estimator makes of the stage O2 of the same slot, which is what the kernels of
the pinned receiver compute, and accounts for it as the capture core does. The tool named
"pilot_ls" is the one of the adapter build against the patched receiver. Any other name is a
build without the route, which refuses a plan that names an estimate.

The unit tests use it to check the reader of the native output and every check of the
integration driver. NATIVE_ESTIMATE_FAULT names one fault to inject, so that each check can be
shown to fail:

    zero-buffer        the tool exports the zeros of the host stub
    no-smoothing       the tool exports an estimate that was never smoothed
    wrong-scale        the estimate is scaled by the ratio of the reference amplitudes once more
    stale              the tool exports the estimate of the previous PDU
    port-swap          two receive ports are exchanged in the payload
    early-publish      a record is published before its copy completed
    lost-record        the consumer of the O3 stream misses the last record
    host-fallback      the CUDA processor executes the host path
    unpatched-accepts  the build without the route accepts a plan that names an estimate
    exit               the tool exits with status 3 after writing

It is not a test module: discovery skips it.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ocudu_cuda_isac.episodes import native_estimate as ne  # noqa: E402
from ocudu_cuda_isac.episodes import pilot_ls as pl  # noqa: E402
from ocudu_cuda_isac.episodes import snapshot_ring as sr  # noqa: E402
from ocudu_cuda_isac.episodes.native_estimate import NativeEstimatePlanError  # noqa: E402
from ocudu_cuda_isac.sensing import native_estimate as reference  # noqa: E402
from ocudu_cuda_isac.sensing import obs  # noqa: E402
from ocudu_cuda_isac.sensing import pilot_ls as o2_reference  # noqa: E402
import snapshot_ring_double as ring_double  # noqa: E402

PATCHED = "pilot_ls"
SCALE = 1.4125375747680664  # 10^(3/20) as a single-precision value


def stage_o2(plan, directory):
    """The stage O2 of every opportunity of a plan, as the double of Sprint 3.1 computes it: {opportunity: (record, values)}."""
    scratch = Path(directory) / ".o2"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    ring_double.base_output(plan, directory, scratch, "")
    folder = scratch / pl.OBSERVER_DIR
    out = {}
    for line in (folder / pl.RECORDS_FILE).read_text().splitlines():
        record = json.loads(line)
        values = o2_reference.payload_values((folder / record["payload"]["file"]).read_bytes())
        out[record["host"]["opportunity_id"]] = (record, values)
    shutil.rmtree(scratch)
    return out


def estimate_of(record, values, fault, previous):
    """The payload the model receiver leaves for one PDU, with the fault of the run."""
    planes, symbols, pilots = record["shape"]["shape"]
    blocks = pilots // reference.PILOTS_PER_PRB
    made, shape = reference.matched(values, [planes, symbols, pilots], smoothing=fault != "no-smoothing")
    if fault == "zero-buffer":
        made = [0j] * len(made)
    elif fault == "wrong-scale":
        made = [reference.chalf(value * 1.001186490058899) for value in made]
    elif fault == "stale" and previous is not None and len(previous) == len(made):
        made = previous
    elif fault == "port-swap" and planes > 1:
        swapped = list(made)
        for symbol in range(symbols):
            for block in range(blocks):
                first = reference.element(shape, symbol, block, 0, 0)
                second = reference.element(shape, symbol, block, 1, 0)
                swapped[first:first + 12], swapped[second:second + 12] = made[second:second + 12], made[first:first + 12]
        made = swapped
    return made, shape


def scenario(tools, plan_path, native, directory, consumers, extra=(), tool=PATCHED):
    """One scenario of the gate, as tests/integration/native_estimate.py run_scenario runs it."""
    fault = os.environ.get("NATIVE_ESTIMATE_FAULT", "")
    patched = tool == PATCHED
    log = Path(directory) / "native.log"
    statuses = [0] * len(consumers)
    raw = Path(plan_path).read_bytes()

    def refuse(reason):
        native.mkdir(parents=True, exist_ok=True)
        for consumer in consumers:
            consumer["archive"].mkdir(parents=True, exist_ok=True)
        log.write_text(f"pilot_ls --plan {plan_path} --output-dir {native}\npilot_ls: input error: {reason}\n")
        return 2, None, statuses

    try:
        plan = json.loads(raw)
        ne.check_plan(plan)
    except (ValueError, NativeEstimatePlanError) as error:
        return refuse(error)
    estimate = plan["estimate"]
    if estimate is not None and not patched and fault != "unpatched-accepts":
        return refuse("the plan names the receiver's own estimate, and this build has no route to it: it is linked "
                      "against the receiver without the patch of patches/ocudu/")
    # ---- Everything of a plan of version 2, by the double of Sprint 3.2.
    inner = {key: plan[key] for key in sr.PLAN_FIELDS}
    inner["schema"] = sr.PLAN_SCHEMA
    inner_path = Path(directory) / "plan-v2.json"
    inner_path.write_text(json.dumps(inner, indent=2, sort_keys=True) + "\n", encoding="ascii")
    first = [consumer for consumer in consumers if consumer.get("stream", "o2") == "o2"]
    second = [consumer for consumer in consumers if consumer.get("stream") == "o3"]
    kept = os.environ.get("SNAPSHOT_RING_FAULT")
    if fault == "host-fallback":
        os.environ["SNAPSHOT_RING_FAULT"] = "host-fallback"
    try:
        status, report, _ = ring_double.scenario(tools, inner_path, native, directory, first, extra)
    finally:
        os.environ.pop("SNAPSHOT_RING_FAULT", None)
        if kept is not None:
            os.environ["SNAPSHOT_RING_FAULT"] = kept
    inner_path.unlink()
    for consumer in second:
        consumer["archive"].mkdir(parents=True)
    if status != 0 or report is None:
        return status, report, statuses
    mode = plan["evidence"]["mode"]
    report.update(schema=ne.NATIVE_SCHEMA, plan={"file": str(plan_path), "sha256": hashlib.sha256(raw).hexdigest()},
                  estimate=None, o3_route=patched)
    report["evidence"]["plan_version"] = 3
    report["access"]["estimate"] = "estimate/ holds the account of the capture and, in the mode audit, its records"
    for item in report["opportunities"]:
        item["estimate"] = None
    if estimate is None:
        (native / "native.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii")
        return (3 if fault == "exit" else 0), report, statuses

    # ---- The capture: one estimate per decode of the CUDA processor that ran its resident path.
    o2 = stage_o2(plan, directory)
    radio = [json.loads(line) for line in (native / "radio/radio.jsonl").read_text().splitlines() if line.strip()]
    cuda = {record["opportunity_id"]: record for record in radio if record["backend"] == ne.BACKEND}
    folder = native / ne.ESTIMATE_DIR
    folder.mkdir()
    ring = estimate["ring"]
    slots = ring["slots"] if ring else 0
    epoch, published, released = plan["observer"]["epoch"], 0, 0
    totals = dict.fromkeys(sr.COUNT_FIELDS, 0)
    totals["epochs"] = 1 if second else 0
    lines, records, archive, previous = [], [], [], None
    outcomes = dict.fromkeys(ne.OUTCOMES, 0)
    clock, launch = 2_000_000_000, 0
    for item in report["opportunities"]:
        name = item["opportunity_id"]
        decode = cuda.get(name)
        if item.get("grid_index") is None or decode is None or decode["outcome"] == "not_dispatched":
            continue
        clock += 1_000_000
        record, values = o2[name]
        resident = decode["path"]["executed"] == "resident"
        keys = {key: record["keys"][key] for key in ("episode", "opportunity", "grant")}
        shape = [record["shape"]["shape"][1], record["shape"]["shape"][2] // 6, record["shape"]["shape"][0], 12]
        size = reference.elements(shape) * reference.ELEMENT_BYTES
        line = {"schema": ne.CAPTURE_SCHEMA, "run_id": plan["run_id"], "mode": mode, "stage": reference.STAGE,
                "backend": ne.BACKEND, "opportunity_id": name, "opportunity_sequence": item["sequence"],
                "outcome": "absent", "reason": "no_estimate", "admitted": False, "seen": False, "landed": False,
                "producer": None, "copy": None, "ring": None, "keys": keys,
                "absolute_slot": record["radio"]["absolute_slot"],
                "grid": {"grid_index": item["grid_index"], "generation": item["generation"]}, "payload_bytes": size,
                "valid": None, "holes": None, "nonzero": None, "t_expect_ns": clock, "t_tap_ns": None,
                "t_submit_ns": None, "t_resolve_ns": clock + 900_000, "t_complete_ns": None, "t_publish_ns": None,
                "t_done_ns": clock + 900_100, "resolve_us": 0.1}
        entry = {"outcome": "absent", "reason": "no_estimate", "backend": ne.BACKEND, "ring": None, "launch": None,
                 "valid": None, "holes": None, "nonzero": None, "payload_sha256": None, "mask_sha256": None,
                 "submit_us": None, "resolve_us": 0.1}
        admitted = False
        if ring is not None:
            # A consumer that keeps up releases what it read before the next estimate is admitted.
            if second and not second[0].get("hold_us"):
                released = published
            admitted = published - released < slots
            totals["admitted" if admitted else "dropped"] += 1
        if not resident:
            totals["cancelled"] += admitted
            line["admitted"] = admitted
        else:
            launch += 1
            made, shape = estimate_of(record, values, fault, previous)
            previous = reference.matched(values, record["shape"]["shape"])[0]
            payload = reference.payload_bytes(made)
            mask, valid, holes, nonzero = reference.scan(payload)
            steps = reference.STEPS if shape[1] > 1 else reference.STEPS[:2]
            producer = {"launch": launch, "device": 0, "backend": ne.BACKEND, "resident_mode": True,
                        "rnti": record["radio"]["ue"], "sfn": record["radio"]["sfn"], "slot": record["radio"]["slot"],
                        "numerology": 1, "start_prb": record["allocation"]["first_subcarrier"] // 12,
                        "prb_count": shape[1], "contiguous": True, "dmrs_symbols": record["allocation"]["dmrs_symbols"],
                        "rx_ports": shape[2], "layers": 1, "dmrs_type": 1, "scale": SCALE, "steps": steps,
                        "time_interpolation": False, "bytes": size}
            wanted = admitted or mode == "audit"
            landing = "slot" if mode == "timed" and estimate["landing"] == "slot" else "staging"
            line.update(seen=True, admitted=admitted, producer=producer, t_tap_ns=clock + 300_000,
                        t_submit_ns=clock + 302_000)
            entry.update(launch=launch, submit_us=2.0)
            if not wanted:
                line.update(outcome="dropped", reason="no_credit")
            else:
                complete = clock + 900_050
                line.update(landed=True, valid=valid, holes=holes, nonzero=nonzero, t_complete_ns=complete,
                            resolve_us=3.0, t_done_ns=clock + 903_000,
                            copy={"stream": estimate["copy_stream"], "landing": landing, "completion": "cuda_event",
                                  "polls": 1, "submit_us": 2.0})
                if admitted:
                    publish = complete - 500 if fault == "early-publish" else complete + 800
                    line.update(outcome="published", reason=None, t_publish_ns=publish,
                                ring={"epoch": epoch, "sequence": published, "depth": published - released,
                                      "bytes": obs.RECORD_HEADER_BYTES + obs.STAGED_HOST.size + len(mask) + size,
                                      "window_calls": 0})
                    entry["ring"] = published
                    totals["max_depth"] = max(totals["max_depth"], published - released)
                    archive.append((line, record, payload, mask, shape, steps))
                    published += 1
                    totals["published"] += 1
                else:
                    line.update(outcome="captured" if ring is None else "dropped",
                                reason=None if ring is None else "no_credit")
                entry.update(valid=valid, holes=holes, nonzero=nonzero, resolve_us=3.0)
                if mode == "audit":
                    digest, mask_digest = hashlib.sha256(payload).hexdigest(), hashlib.sha256(mask).hexdigest()
                    files = (f"{item['sequence']:04d}.o3.fp16", f"{item['sequence']:04d}.o3.mask")
                    (folder / files[0]).write_bytes(payload)
                    (folder / files[1]).write_bytes(mask)
                    entry.update(payload_sha256=digest, mask_sha256=mask_digest)
                    records.append({
                        "schema": reference.DESCRIPTOR_SCHEMA, "run_id": plan["run_id"],
                        "keys": dict(keys, epoch=plan["observer"]["epoch"], sequence=len(records)),
                        "radio": record["radio"],
                        "what": {"stage": reference.STAGE, "producer": reference.PRODUCER, "producer_steps": steps,
                                 "source": estimate["source"], "port": record["what"]["port"], "layout": reference.LAYOUT,
                                 "dtype": reference.DTYPE, "scale": SCALE, "rule": reference.RULE,
                                 "precision": reference.PRECISION},
                        "shape": {"rank": 4, "axes": reference.AXES, "shape": shape,
                                  "strides": [shape[1] * shape[2] * 12, shape[2] * 12, 12, 1]},
                        "allocation": {"first_subcarrier": record["allocation"]["first_subcarrier"], "subcarrier_step": 1,
                                       "subcarrier_count": shape[1] * 12,
                                       "dmrs_symbols": record["allocation"]["dmrs_symbols"], "rx_count": shape[2],
                                       "layer_count": 1, "antenna_map": record["allocation"]["antenna_map"],
                                       "coordinates_sha256": None},
                        "validity": {"mask_bytes": len(mask), "file": files[1], "sha256": mask_digest, "valid": valid,
                                     "holes": holes},
                        "time": dict(record["time"], t_ready_ns=complete, t_publish_ns=line["t_publish_ns"]),
                        "payload": {"bytes": size, "file": files[0], "digest_kind": "sha256", "digest": digest},
                        "host": {"name": record["host"]["name"], "opportunity_id": name,
                                 "opportunity_sequence": item["sequence"], "grant_id": record["host"]["grant_id"],
                                 "grant_record_sha256": record["host"]["grant_record_sha256"],
                                 "ingress": {"grid_index": item["grid_index"], "generation": item["generation"],
                                             "readback_sha256": item["readback_sha256"]},
                                 "completion": "cuda_event", "producer": producer,
                                 "copy": {"stream": estimate["copy_stream"], "landing": landing, "polls": 1},
                                 "ring_sequence": entry["ring"], "nonzero": nonzero, "service_us": 5.0}})
        entry.update(outcome=line["outcome"], reason=line["reason"])
        outcomes[line["outcome"]] += 1
        item["estimate"] = entry
        lines.append(line)
    if second:
        released = published
    totals["released"], totals["pending"] = released, published - released
    ring_report = None
    if ring is not None:
        ring_report = {
            "schema": sr.RING_SCHEMA, "layout": "obs-ring/v1", "definition": f"{obs.DEFINITION:016x}",
            "stream": ring["stream"], "socket": f"{ne.ESTIMATE_DIR}/{ring['stream']}.sock",
            "window_records": ring["window_records"], "burst_records": ring["burst_records"], "slot_count": slots,
            "slot_bytes": 20288, "segment_bytes": 4096 + slots * 20288, "payload_capacity": 19584, "mask_capacity": 612,
            "host_capacity": obs.STAGED_HOST.size, "memory_path": "host_shared", "clock": "model clock of the double",
            "wait": {"planned": ring["wait"], "consumer": second[0]["wait"] if second else None},
            "attach": ring["attach"], "attached_at_start": bool(second), "attached_at_drain": bool(second),
            "first_epoch": epoch, "last_epoch": epoch, "drain_ms": ring["drain_ms"],
            "pending_at_cutoff": totals["pending"], "eligible": len(lines), "counts": totals}
        (folder / ne.RING_FILE).write_text(json.dumps(ring_report, indent=2, sort_keys=True) + "\n", encoding="ascii")
    (folder / ne.CAPTURES_FILE).write_text("".join(json.dumps(line, sort_keys=True) + "\n" for line in lines),
                                           encoding="ascii")
    if mode == "audit":
        (folder / ne.RECORDS_FILE).write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
                                              encoding="ascii")
    # ---- What the consumer of the O3 stream kept.
    for consumer in second:
        pairs = archive[:-1] if fault == "lost-record" else archive
        rows = []
        for line, record, payload, mask, shape, steps in pairs:
            symbols = 0
            for symbol in record["allocation"]["dmrs_symbols"]:
                symbols |= 1 << symbol
            header = obs.new_header(
                epoch=line["ring"]["epoch"], sequence=line["ring"]["sequence"], episode=int(line["keys"]["episode"], 16),
                opportunity=int(line["keys"]["opportunity"], 16), grant=int(line["keys"]["grant"], 16),
                absolute_slot=line["absolute_slot"], cell=record["radio"]["cell"], ue=record["radio"]["ue"], stage=3,
                source=1, layout=1, dtype=2, producer_steps=7 if len(steps) == 3 else 3, scale=SCALE, rank=4, shape=shape,
                strides=[shape[1] * shape[2] * 12, shape[2] * 12, 12, 1],
                first_subcarrier=record["allocation"]["first_subcarrier"], subcarrier_step=1,
                subcarrier_count=shape[1] * 12, dmrs_symbols=symbols, rx_count=shape[2], layer_count=1,
                antenna_map=record["allocation"]["antenna_map"], mask_bytes=len(mask), payload_bytes=len(payload),
                host_bytes=obs.STAGED_HOST.size, time_unit=1, digest_kind=1 if mode == "audit" else 0,
                evidence=1 if mode == "audit" else 2, port=record["what"]["port"],
                t_first_sample=record["time"]["t_first_sample"], t_last_sample=record["time"]["t_last_sample"],
                t_ready_ns=line["t_complete_ns"], t_publish_ns=line["t_publish_ns"],
                digest=hashlib.sha256(payload).digest() if mode == "audit" else b"", valid=line["valid"])
            host = obs.STAGED_HOST.pack(1, line["opportunity_sequence"], line["grid"]["grid_index"], record["radio"]["sfn"],
                                        line["grid"]["generation"], record["radio"]["slot"], 0, line["t_tap_ns"])
            made = obs.Record(header, host, mask, payload, depth=line["ring"]["depth"])
            (consumer["archive"] / obs.record_name(header["epoch"], header["sequence"])).write_bytes(made.to_bytes())
            rows.append({"epoch": header["epoch"], "sequence": header["sequence"], "opportunity": line["keys"]["opportunity"],
                         "depth": line["ring"]["depth"], "bytes": len(made.to_bytes()),
                         "t_publish_ns": line["t_publish_ns"], "t_read_ns": line["t_publish_ns"] + 400, "delay_ns": 400,
                         "intact": True, "file": True})
        (consumer["archive"] / "consumer.json").write_text(json.dumps({
            "schema": obs.CONSUMER_SCHEMA, "reader": consumer["reader"], "definition": f"{obs.DEFINITION:016x}",
            "wait": consumer["wait"], "epoch": epoch, "pid": 1, "cpu": -1, "hold_us": consumer.get("hold_us", 0),
            "window": 0, "slot_count": slots, "slot_bytes": 20288, "attached_ns": 1, "ended": "closed",
            "counts": {"read": len(rows), "stale": 0, "released": len(rows), "skipped": 0, "sleeps": 0, "kept": len(rows),
                       "not_kept": 0, "system_calls": 17}, "records": rows}, indent=2) + "\n", encoding="ascii")
    copied = [line for line in lines if line["copy"] is not None][1:]

    def stats(values):
        made = {"count": len(values), "excluded": "the first copy of the run"}
        if values:
            made.update(mean=sum(values) / len(values), min=min(values), max=max(values))
        return made

    landed = [line for line in lines if line["landed"]]
    report["estimate"] = {
        "tap": ne.TAP, "stage": reference.STAGE, "source": estimate["source"], "layout": reference.LAYOUT,
        "dtype": reference.DTYPE, "backend": ne.BACKEND, "producer": reference.PRODUCER, "rule": reference.RULE,
        "precision": reference.PRECISION,
        "interface": {"header": "ocudu/phy/upper/channel_processors/pusch/pusch_resident_estimate_tap.h", "version": 1},
        "copy_stream": estimate["copy_stream"],
        "landing": {"planned": estimate["landing"],
                    "effective": "slot" if mode == "timed" and estimate["landing"] == "slot" else "staging",
                    "refusal": None},
        "timeout_us": estimate["timeout_us"], "completion": "one CUDA event, modelled by the double",
        "device": {"id": 0, "copies_submitted": len(copied) + 1 if lines else 0, "last_error": None}, "mode": mode,
        "eligible": len(lines), "outcomes": outcomes, "reasons": {}, "landed": len(landed), "archived": len(records),
        "valid": sum(line["valid"] for line in landed), "holes": sum(line["holes"] for line in landed),
        "nonzero": sum(line["nonzero"] for line in landed), "payload_bytes": sum(line["payload_bytes"] for line in landed),
        "unexpected": 0, "overruns": 0, "max_polls": 1 if copied else 0,
        "submit_us": stats([line["copy"]["submit_us"] for line in copied]),
        "resolve_us": stats([line["resolve_us"] for line in copied]), "ring": ring_report}
    (native / "native.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return (3 if fault == "exit" else 0), report, statuses
