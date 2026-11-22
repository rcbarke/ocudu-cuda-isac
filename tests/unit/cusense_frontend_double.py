#!/usr/bin/env python3
"""Doubles of the Sprint 3.5 gate, for hosts without an OCUDU build, without a GPU and without a ray tracer.

capture and live stand in for what tests/integration/cusense_frontend.py starts: the native tool
pilot_ls on a pilot-ls-plan/v3 file with one consumer on each of its two rings. The tool, the
rings and the records of both receiver stages are those of the double of Sprint 5.2
(tests/unit/native_estimate_double.py): the stage O2 of the model receiver and, for every decode
that ran its resident path, the estimate that the matched estimator makes of it. The sensing side
is the real one: in the live case the records of the O3 archive go through
ocudu_cuda_isac.sensing.cusense in the order of the ring, with the artifact the gate froze, and
the output is written as the sensor writes it. Only the rings between the two are a model. The
episode set of these tests is the analytic set of tests/unit/presence_head_double.py.

CUSENSE_FRONTEND_FAULT names one fault to inject, so that each check of the driver can be shown
to fail. The first four act on the runs that keep the training and the validation records, the
others on the held-out run with the sensor:

    capture-drop      the consumer of the O3 stream is slow: estimates are dropped at its ring
    capture-crc       with the rings attached, the CUDA processor returns a wrong byte
    capture-fallback  the CUDA processor executes the host path: no estimate exists
    capture-exit      the tool exits with status 3 after writing
    live-crc          as capture-crc, in the held-out run
    live-fallback     as capture-fallback, in the held-out run
    live-exit         as capture-exit, in the held-out run
    sensor-exit       the sensor exits with status 3 after writing
    no-output         the sensor writes nothing
    wrong-artifact    the sensor scores with another model than the frozen one
    two-loads         the sensor loads the artifact once per window
    cold-sensor       the sensor attaches before it is warm
    slow-sensor       the sensor is slower than the receiver: estimates are dropped at its ring
    future-leak       the sensor computes a window with the slot of a later period in it
    wrong-sample      the sensor computes a window on a slot that the ring did not carry
    device-drift      the scores of the sensor differ from the offline ones by 5e-3

It is not a test module: discovery skips it.
"""

import json
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ocudu_cuda_isac.sensing import cusense  # noqa: E402
from ocudu_cuda_isac.sensing import obs  # noqa: E402
import native_estimate_double as estimate_double  # noqa: E402

RING_FAULTS = {"crc": ("SNAPSHOT_RING_FAULT", "changed-decode"), "fallback": ("NATIVE_ESTIMATE_FAULT", "host-fallback"),
               "exit": ("NATIVE_ESTIMATE_FAULT", "exit")}
DRIFT = 5e-3


def _scenario(tools, plan_path, native, directory, consumers, fault):
    """The scenario of Sprint 5.2 with one fault of its own doubles set for the call."""
    name, value = RING_FAULTS.get(fault, (None, None))
    saved = {key: os.environ.get(key) for key in ("SNAPSHOT_RING_FAULT", "NATIVE_ESTIMATE_FAULT")}
    if name:
        os.environ[name] = value
    try:
        return estimate_double.scenario(tools, plan_path, native, directory, consumers)
    finally:
        for key, kept in saved.items():
            os.environ.pop(key, None)
            if kept is not None:
                os.environ[key] = kept


def capture(tools, plan_path, native, directory, archives, wait, capacity, keep_grids=True):
    """One receiver replay with a consumer on each ring that keeps the records, as the driver's capture runs it."""
    fault = os.environ.get("CUSENSE_FRONTEND_FAULT", "")
    consumers = [{"reader": "native", "wait": wait, "stream": stream, "archive": archives[stream]} for stream in ("o2", "o3")]
    if fault == "capture-drop":
        consumers[1]["hold_us"] = 5000
    status, report, statuses = _scenario(tools, plan_path, native, directory, consumers,
                                         fault[len("capture-"):] if fault.startswith("capture-") else None)
    for stream in archives:
        (directory / f"consumer-{stream}.log").write_text(f"obs_tail double: {capacity} records of capacity\n")
    return status, report, dict(zip(("o2", "o3"), statuses))


def _window_slots(records, spec):
    """The places, in a list of records, of the slots that the window of a specification reads."""
    period, slots = spec["radio"]["period_slots"], spec["window"]["slots_in_period"]
    return [index for index, record in enumerate(records) if record.header["absolute_slot"] % period in slots]


def live(tools, plan_path, native, directory, sensor_dir, artifact_dir, device, wait, ready_timeout_s, archive, capacity,
         keep_grids=True):
    """The held-out replay with the cuSense-like sensor on the O3 ring and a consumer that keeps the O2 ring."""
    fault = os.environ.get("CUSENSE_FRONTEND_FAULT", "")
    scratch = directory / "double-records"
    consumers = [{"reader": "native", "wait": wait, "stream": "o2", "archive": archive},
                 {"reader": "reference", "wait": wait, "stream": "o3", "archive": scratch}]
    if fault == "slow-sensor":
        consumers[1]["hold_us"] = 5000
    status, report, statuses = _scenario(tools, plan_path, native, directory, consumers,
                                         fault[len("live-"):] if fault.startswith("live-") else None)
    (directory / "consumer-o2.log").write_text(f"obs_tail double: {capacity} records of capacity\n")
    log = directory / "sensor.log"
    made = {"o2": statuses[0], "o3": 0}
    if report is None or not (scratch / "consumer.json").is_file():
        log.write_text("cusense: no producer opened the control channel\n")
        return status, report, dict(made, o3=2)
    if fault == "no-output":
        log.write_text("cusense: stopped before its output\n")
        return status, report, dict(made, o3=1)
    kept = obs.open_archive(scratch)
    records = list(kept.records)
    artifact = cusense.open_artifact(artifact_dir)
    if fault == "wrong-artifact":
        # The same model with its threshold moved, written whole: an artifact that verifies and is not the frozen one.
        manifest = artifact.manifest
        other = directory / "double-artifact"
        net = cusense.cusense_cnn.unpack(cusense.cusense_cnn.CuSenseCNN(manifest["model"]["config"]), artifact.weights,
                                         manifest["weights"]["tensors"]).eval()
        threshold = {name: manifest["threshold"][name] for name in cusense.presence.THRESHOLD_FIELDS}
        threshold.update(logit=threshold["logit"] + 1.0,
                         probability=1.0 / (1.0 + math.exp(-(threshold["logit"] + 1.0))))
        cusense.write_artifact(other, manifest["model_id"], manifest["input"]["spec"], manifest["input"]["elements"],
                               manifest["input"]["producer_steps"],
                               dict(artifact.template, fitted_on=manifest["template"]["fitted_on"]),
                               {name: manifest["normalizer"][name] for name in ("mean", "scale", "fitted_on")}, net,
                               manifest["grid"], threshold, manifest["smoothing"], manifest["model"]["target"]["sigma_cells"])
        artifact = cusense.open_artifact(other)
    loads = cusense.Model.LOADS
    model = cusense.Model(artifact, "cpu" if device is None else device)
    warm = {"frontend": cusense.warm_up(model.spec), "model": model.warm_up()}
    chosen = _window_slots(records, model.spec)
    need = len(model.spec["window"]["slots_in_period"])
    if fault == "future-leak" and len(chosen) > need:
        # The first slot of the first window holds the payload of the first slot of the next period.
        early, late = records[chosen[0]], records[chosen[need]]
        records[chosen[0]] = obs.Record(dict(early.header), early.host, late.mask, late.payload, depth=early.depth)
    if fault == "wrong-sample" and chosen:
        # The completing slot of the first window is scaled: a sample the ring did not carry.
        place = chosen[min(need, len(chosen)) - 1]
        values = np.frombuffer(records[place].payload, dtype="<f2").astype(np.float32) * 1.5
        records[place] = obs.Record(dict(records[place].header), records[place].host, records[place].mask,
                                    values.astype("<f2").tobytes(), depth=records[place].depth)
    first = min((record.header["t_publish_ns"] for record in records), default=2_000_000_000)
    ready = first + 1000 if fault == "cold-sensor" else first - 1_000_000
    (directory / "sensor-ready.json").write_text(json.dumps(
        {"device": str(model.device), "artifact_sha256": model.artifact.sha256, "warm_up_ms": warm, "t_ready_ns": ready}))
    clock = iter(range(3_000_000_000, 4_000_000_000, 1000))
    lines, stream, decisions, inputs, maps = cusense.run_model(records, model, clock=lambda: next(clock))
    if fault == "device-drift":
        for decision in decisions:
            if decision["status"] == "ok":
                decision["score"]["logit"] += DRIFT
                decision["score"]["probability"] = 1.0 / (1.0 + math.exp(-decision["score"]["logit"]))
                decision["decision"]["present"] = decision["score"]["logit"] > decision["decision"]["threshold_logit"]
    source = {"kind": "ring", "socket": "model ring of the double", "reader": "reference", "wait": wait,
              "epoch": kept.report["epoch"], "records": len(kept.records), "stale": 0, "ended": "closed",
              "warm_up_ms": warm}
    window_dir = sensor_dir / cusense.WINDOWS_DIR
    window_dir.mkdir(parents=True)
    scratch.rename(window_dir / cusense.RECORDS_DIR)
    cusense.write_windows(window_dir, model.spec, lines, stream, source)
    # The sensor is a process of its own: its count of loads starts at zero.
    cusense.Model.LOADS = max(len(inputs), 2) if fault == "two-loads" else 1
    cusense.write_output(sensor_dir, model, decisions, inputs, maps, source)
    cusense.Model.LOADS = loads + 1
    counts = stream.counts
    log.write_text(f"cusense: {counts['records']} records, {counts['ok']} scored windows\n")
    return status, report, dict(made, o3=3 if fault == "sensor-exit" else 0)
