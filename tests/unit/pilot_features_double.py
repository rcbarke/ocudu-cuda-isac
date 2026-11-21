#!/usr/bin/env python3
"""Double of the live case of the Sprint 3.3 gate, for hosts without an OCUDU build.

It stands in for what tests/integration/pilot_features.py run_live starts: the native tool
pilot_ls on a pilot-ls-plan/v2 file and the reference sensor on its ring. The tool, the ring and
the records are those of the double of Sprint 3.2 (tests/unit/snapshot_ring_double.py), which
computes each observation with the reference and writes the archive that a consumer keeps. The
sensor is the real one: the records of that archive go through ocudu_cuda_isac.sensing.features
in the order of the ring, and the output is written as the sensor writes it. Only the ring
between the two is a model.

PILOT_FEATURES_FAULT names one fault to inject, so that each check of the live case can be shown
to fail:

    future-leak     the sensor computes a window with the sample of a later record in it
    slow-sensor     the sensor is slower than the receiver: observations are dropped at the ring
    sensor-exit     the sensor exits with status 3 after writing
    no-output       the sensor writes nothing
    changed-decode  with the ring attached, the CUDA processor returns a wrong byte
    host-fallback   the CUDA processor executes the host path
    exit            the tool exits with status 3 after writing

It is not a test module: discovery skips it.
"""

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ocudu_cuda_isac.sensing import features  # noqa: E402
from ocudu_cuda_isac.sensing import obs  # noqa: E402
import snapshot_ring_double as ring_double  # noqa: E402

RING_FAULTS = ("changed-decode", "host-fallback", "exit")


def live(tools, plan_path, native, directory, sensor_dir, spec_path, wait):
    """One live run of the gate, as tests/integration/pilot_features.py run_live runs it."""
    fault = os.environ.get("PILOT_FEATURES_FAULT", "")
    consumer = {"reader": "reference", "wait": wait, "archive": sensor_dir / features.RECORDS_DIR}
    if fault == "slow-sensor":
        consumer["hold_us"] = 5000
    saved = os.environ.get("SNAPSHOT_RING_FAULT")
    if fault in RING_FAULTS:
        os.environ["SNAPSHOT_RING_FAULT"] = fault
    try:
        status, report, _ = ring_double.scenario(tools, plan_path, native, directory, [consumer])
    finally:
        os.environ.pop("SNAPSHOT_RING_FAULT", None)
        if saved is not None:
            os.environ["SNAPSHOT_RING_FAULT"] = saved
    log = directory / "sensor.log"
    if report is None:
        log.write_text("features: no producer opened the control channel\n")
        return status, report, 2
    if fault == "no-output":
        log.write_text("features: stopped before its output\n")
        return status, report, 1
    spec = json.loads(Path(spec_path).read_text())
    archive = obs.open_archive(consumer["archive"])
    records = list(archive.records)
    if fault == "future-leak":
        # The sample of a contributing record is replaced by that of the next record of the sequence.
        stream = features.Stream(spec)
        chosen = [index for index, record in enumerate(records)
                  if record.header["absolute_slot"] % stream.period_slots in stream.slots]
        early, late = chosen[0], chosen[1]
        records[early] = obs.Record(dict(records[early].header), records[early].host, records[late].mask,
                                    records[late].payload, depth=records[early].depth)
    lines, views, stream = features.run_records(records, spec, clock=lambda: 2_000_000_000)
    features.write_output(sensor_dir, spec, lines, views, stream,
                          {"kind": "ring", "socket": "model ring of the double", "reader": "reference", "wait": wait,
                           "epoch": archive.report["epoch"], "records": len(archive.records), "stale": 0,
                           "ended": "closed"})
    counts = stream.counts
    log.write_text(f"features: {counts['records']} records, {counts['ok']} complete windows\n")
    return status, report, 3 if fault == "sensor-exit" else 0
