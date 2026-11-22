#!/usr/bin/env python3
"""Doubles of the Sprint 3.4 gate, for hosts without an OCUDU build and without a ray tracer.

Two things stand in for what a development host does not have.

analytic_set writes an episode set as the exporter of Sprint 2.2 writes it, with the free-space
point-target model in place of the ray tracer: the trajectories, the groups, the identifiers, the
labels and the noise seeds are those of the set module, and every entry verifies with its reader.
As the exporter does, it binds no transmit grid to an entry.

capture and live stand in for what tests/integration/presence_head.py starts: the native tool
pilot_ls on a pilot-ls-plan/v2 file with a consumer on its ring. The tool, the ring and the
records are those of the double of Sprint 3.2 (tests/unit/snapshot_ring_double.py). The sensing
side is the real one: in the live case the records of the archive go through
ocudu_cuda_isac.sensing.presence in the order of the ring, with the artifact the gate froze, and
the output is written as the sensor writes it. Only the ring between the two is a model.

PRESENCE_HEAD_FAULT names one fault to inject, so that each check of the driver can be shown to
fail. The first four act on the runs that keep the training and the validation records, the
others on the held-out run with the sensor:

    capture-drop      the consumer is slow: observations are dropped at the ring
    capture-crc       with the ring attached, the CUDA processor returns a wrong byte
    capture-fallback  the CUDA processor executes the host path
    capture-exit      the tool exits with status 3 after writing
    live-crc          as capture-crc, in the held-out run
    live-fallback     as capture-fallback, in the held-out run
    live-exit         as capture-exit, in the held-out run
    sensor-exit       the sensor exits with status 3 after writing
    no-output         the sensor writes nothing
    wrong-artifact    the sensor scores with another head than the frozen one
    two-loads         the sensor loads the artifact once per window
    cold-sensor       the sensor attaches before it is warm
    slow-sensor       the sensor is slower than the receiver: observations are dropped at the ring
    future-leak       the sensor computes a window with the sample of a later record in it
    wrong-sample      the sensor computes a window on a sample that the ring did not carry
    device-drift      the scores of the sensor differ from the offline ones by 1e-3

It is not a test module: discovery skips it.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ocudu_cuda_isac.episodes import episode as episodes  # noqa: E402
from ocudu_cuda_isac.episodes import episode_set as sets  # noqa: E402
from ocudu_cuda_isac.episodes.episode import TC_PER_SECOND  # noqa: E402
from ocudu_cuda_isac.sensing import features  # noqa: E402
from ocudu_cuda_isac.sensing import obs  # noqa: E402
from ocudu_cuda_isac.sensing import presence  # noqa: E402
import snapshot_ring_double as ring_double  # noqa: E402

RING_FAULTS = {"crc": "changed-decode", "fallback": "host-fallback", "exit": "exit"}
TARGET = {"model": "constant_rcs_cuboid", "center_height_m": 0.875, "sigma_m2": 1.0, "size_m": [0.3, 0.3, 0.3]}


def set_config(path, grant, trajectories, set_id="unit-set", seed=20261122, snr_db=30):
    """Write an episode-set-config/v1 file on the scene of the shipped sets, with the analytic target."""
    config = json.loads((ROOT / "configs/episodes/smoke_set.json").read_text())
    config.update(set_id=set_id, grant=str(grant), target=dict(TARGET), trajectories=trajectories,
                  noise={"snr_db": snr_db, "seed": seed}, min_training_coverage=0.0001)
    Path(path).write_text(json.dumps(config))
    return Path(path)


def trajectories(training=2, validation=2, held_out=2, periods=2):
    """Four walks: two for training, one for validation and one held out. Episodes of one walk are 2 s apart."""
    def walk(name, group, kind, count, **shape):
        return dict({"id": name, "group": group, "kind": kind, "speed_m_s": 1.2, "start_s": 0.5, "episodes": count,
                     "periods": periods, "stride_s": 2.0}, **shape)

    first = (training + 1) // 2
    return [walk("unit-training-lanes", "training", "lawnmower", first, axis="y", lanes=3, margin_m=0.5),
            walk("unit-training-line", "training", "line", training - first, from_m=[1.5, 3.0], to_m=[6.0, 10.0]),
            walk("unit-validation-random", "validation", "random", validation, seed=7, waypoints=8, margin_m=0.5),
            walk("unit-held-out-line", "held_out", "line", held_out, from_m=[6.5, 2.5], to_m=[1.2, 10.5])]


def analytic_set(config_path, directory):
    """Write the set of a configuration with the analytic model in place of the ray tracer. Returns its manifest."""
    config, base = sets.load_set_config(config_path)
    plans = sets.plan(config, base)
    wavelength = episodes.SPEED_OF_LIGHT_M_S / base.carrier_hz
    count = len(base.rx_ports)
    geometry = {"ue_tx_m": [float(v) for v in config["endpoints"]["ue_tx_m"]],
                "gnb_rx_m": [float(v) for v in config["endpoints"]["gnb_rx_m"]],
                "rx_element_offsets_m": [[(n - (count - 1) / 2) * 0.5 * wavelength, 0.0, 0.0] for n in range(count)]}
    offsets = np.array([base.subcarrier_offset_hz(k) for k in range(base.grid_subcarriers)], dtype=np.float64)
    directory = Path(directory)
    (directory / "episodes").mkdir(parents=True)
    manifests = {}
    for item in plans:
        schedule = item["schedule"]
        target = dict(config["target"], motion=item["motion"])
        h_shape, w_shape = episodes.shapes(schedule)
        h_a, h_b = np.zeros(h_shape, dtype="<c8"), np.zeros(h_shape, dtype="<c8")
        truth = []
        for opportunity in schedule.opportunities:
            for symbol in range(14):
                paths = episodes.free_space_paths(geometry, target, opportunity.symbol_time_tc(symbol) / TC_PER_SECOND,
                                                  base.carrier_hz, True)
                tones = [amplitude * np.exp(1j * np.array(phases))[:, None]
                         * np.exp(-2j * np.pi * (base.carrier_hz + offsets) * delay)[None, :]
                         for amplitude, delay, phases in paths]
                h_a[opportunity.index, :, 0, symbol, :] = tones[0]
                h_b[opportunity.index, :, 0, symbol, :] = tones[0] + tones[1]
            position, velocity = episodes.target_state(item["motion"], opportunity.slot_start_tc / TC_PER_SECOND)
            truth.append({"opportunity": opportunity.index, "position_m": position, "velocity_m_s": velocity,
                          "bistatic": episodes.bistatic(geometry, position, velocity, base.carrier_hz)})
        power = float(np.mean(np.abs(h_a) ** 2))
        sigma = math.sqrt(power * 10 ** (-item["noise"]["snr_db"] / 10))
        generator = np.random.Generator(np.random.PCG64(item["noise"]["seed"]))
        w = (generator.standard_normal(w_shape) + 1j * generator.standard_normal(w_shape)) * sigma / math.sqrt(2)
        sets.label(config, item, truth)
        description = {"geometry": geometry, "array": {}, "scene": config["scene"], "target": target,
                       "solver": {"double": "free-space point target"}, "time_evolution": {},
                       "noise": {"snr_db": float(item["noise"]["snr_db"]), "sigma": sigma, "seed": item["noise"]["seed"]},
                       "calibration": {"reference_power": power, "grid_gain": 1.0 / math.sqrt(power)},
                       "software": {}, "provenance": {}, "labels": item["labels"]}
        manifests[item["episode_id"]] = episodes.export(
            directory / "episodes" / item["episode_id"], item["episode_id"], schedule, description,
            {"a": h_a.tobytes(), "b": h_b.tobytes(), "w": w.astype("<c8").tobytes()}, truth)
    return sets.write_manifest(directory, config, hashlib.sha256(Path(config_path).read_bytes()).hexdigest(), plans,
                               manifests)


def _scenario(tools, plan_path, native, directory, consumer, ring_fault):
    saved = os.environ.get("SNAPSHOT_RING_FAULT")
    if ring_fault:
        os.environ["SNAPSHOT_RING_FAULT"] = ring_fault
    try:
        status, report, statuses = ring_double.scenario(tools, plan_path, native, directory, [consumer])
    finally:
        os.environ.pop("SNAPSHOT_RING_FAULT", None)
        if saved is not None:
            os.environ["SNAPSHOT_RING_FAULT"] = saved
    return status, report, statuses[0]


def capture(tools, plan_path, native, directory, archive, wait, capacity, keep_grids=True):
    """One receiver replay with a consumer that keeps the records, as the driver's capture runs it."""
    fault = os.environ.get("PRESENCE_HEAD_FAULT", "")
    consumer = {"reader": "native", "wait": wait, "archive": archive}
    if fault == "capture-drop":
        consumer["hold_us"] = 5000
    ring_fault = RING_FAULTS.get(fault[len("capture-"):]) if fault.startswith("capture-") else None
    status, report, consumer_status = _scenario(tools, plan_path, native, directory, consumer, ring_fault)
    (directory / "consumer.log").write_text(f"obs_tail double: {capacity} records of capacity\n")
    return status, report, consumer_status


def live(tools, plan_path, native, directory, sensor_dir, spec_path, artifact_dir, device, wait, ready_timeout_s,
         keep_grids=True):
    """The held-out replay with the presence sensor on its ring, as the driver's live case runs it."""
    fault = os.environ.get("PRESENCE_HEAD_FAULT", "")
    scratch = directory / "double-records"
    consumer = {"reader": "reference", "wait": wait, "archive": scratch}
    if fault == "slow-sensor":
        consumer["hold_us"] = 5000
    ring_fault = RING_FAULTS.get(fault[len("live-"):]) if fault.startswith("live-") else None
    status, report, _ = _scenario(tools, plan_path, native, directory, consumer, ring_fault)
    log = directory / "sensor.log"
    if report is None:
        log.write_text("presence: no producer opened the control channel\n")
        return status, report, 2
    if fault == "no-output":
        log.write_text("presence: stopped before its output\n")
        return status, report, 1
    spec = json.loads(Path(spec_path).read_text())
    archive = obs.open_archive(scratch)
    records = list(archive.records)
    artifact = presence.open_artifact(artifact_dir)
    if fault == "wrong-artifact":
        manifest = artifact.manifest
        other = directory / "double-artifact"
        presence.write_artifact(other, manifest["head_id"], manifest["input"]["spec_id"], manifest["input"]["spec_sha256"],
                                manifest["input"]["vector"], manifest["input"]["columns"], manifest["normalizer"],
                                [0.5] * len(manifest["input"]["columns"]), 0.25, manifest["threshold"])
        artifact = presence.open_artifact(other)
    loads = presence.Head.LOADS
    head = presence.Head(artifact, "cpu" if device is None else device)
    warm = {"features": features.warm_up(spec), "head": head.warm_up()}
    if fault == "future-leak":
        stream = features.Stream(spec)
        chosen = [index for index, record in enumerate(records)
                  if record.header["absolute_slot"] % stream.period_slots in stream.slots]
        early, late = chosen[0], chosen[1]
        records[early] = obs.Record(dict(records[early].header), records[early].host, records[late].mask,
                                    records[late].payload, depth=records[early].depth)
    if fault == "wrong-sample":
        # The completing record of the first window is scaled: a sample the ring did not carry.
        stream = features.Stream(spec)
        chosen = [index for index, record in enumerate(records)
                  if record.header["absolute_slot"] % stream.period_slots in stream.slots]
        place = chosen[stream.need - 1]
        values = np.frombuffer(records[place].payload, dtype="<c8") * 1.5
        records[place] = obs.Record(dict(records[place].header), records[place].host, records[place].mask,
                                    values.astype("<c8").tobytes(), depth=records[place].depth)
    first = min((record.header["t_publish_ns"] for record in records), default=2_000_000_000)
    ready = first + 1000 if fault == "cold-sensor" else first - 1_000_000
    (directory / "sensor-ready.json").write_text(json.dumps(
        {"device": str(head.device), "artifact_sha256": head.artifact.sha256, "warm_up_ms": warm, "t_ready_ns": ready}))
    clock = iter(range(3_000_000_000, 4_000_000_000, 1000))
    lines, views, stream, decisions = presence.run_records(records, spec, head, clock=lambda: next(clock))
    if fault == "device-drift":
        for decision in decisions:
            if decision["status"] == "ok":
                decision["score"]["logit"] += 1e-3
                decision["score"]["probability"] = 1.0 / (1.0 + math.exp(-decision["score"]["logit"]))
                decision["decision"]["present"] = decision["score"]["logit"] > decision["decision"]["threshold_logit"]
    source = {"kind": "ring", "socket": "model ring of the double", "reader": "reference", "wait": wait,
              "epoch": archive.report["epoch"], "records": len(archive.records), "stale": 0, "ended": "closed",
              "warm_up_ms": warm["features"]}
    feature_dir = sensor_dir / presence.FEATURES_DIR
    feature_dir.mkdir(parents=True)
    scratch.rename(feature_dir / features.RECORDS_DIR)
    features.write_output(feature_dir, spec, lines, views, stream, source)
    # The sensor is a process of its own: its count of loads starts at zero.
    presence.Head.LOADS = max(len([line for line in lines if line["status"] == "ok"]), 2) if fault == "two-loads" else 1
    presence.write_output(sensor_dir, head, decisions, dict(source, warm_up_ms=warm))
    presence.Head.LOADS = loads + 1
    counts = stream.counts
    log.write_text(f"presence: {counts['records']} records, {counts['ok']} scored windows\n")
    return status, report, 3 if fault == "sensor-exit" else 0
