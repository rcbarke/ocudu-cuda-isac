"""A stand-in for an accepted Sprint 3.5 run, for the rehearsal of the Sprint 3.6 driver.

The Sprint 3.6 gate reads three things of the run that Sprint 3.5 accepted: the frozen artifact
with its freeze record, the output of the reference sensor on the held-out records, and the
records that sensor kept. make_control() makes all three on a development host, small: a model
with random weights on an array of two antennas and 48 subcarriers, and a few episodes of records
that sense_feed publishes on a ring for the reference sensor, as a receiver would.

It needs NumPy, PyTorch and a build of the core for the two tools. It is no evidence of anything:
the records are synthetic and the model is not trained.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ocudu_cuda_isac.sensing import obs  # noqa: E402
import test_cusense_frontend as fixtures  # noqa: E402

DRIVER = "cusense_frontend"
SMOOTHING = {"entry": "map_mean_kalman", "maps": 3, "process_noise_m2_s3": 1.0, "measurement_noise_m2": 0.25,
             "initial_speed_m_s": 1.5}


def records_of(episodes=4, periods=10, seed=7):
    """The records of a few episodes, target-absent and target-present in turn, in the order of one ring."""
    generator = np.random.Generator(np.random.PCG64(seed))
    records = []
    for episode in range(episodes):
        cell = (episode % 4, episode % 3) if episode % 2 else None
        records += fixtures.periods_of(periods, generator, cell, episode=0x3000 + episode, first_sequence=len(records))
    return records


def write_archive(directory, records, epoch=1):
    """The records as the files of a consumer archive, without its report: what sense_feed publishes."""
    directory = Path(directory)
    directory.mkdir(parents=True)
    for number, record in enumerate(records):
        (directory / obs.record_name(epoch, number)).write_bytes(record.to_bytes())
    return directory


def follow(sense_feed, archive, artifact, output, wait="spin", slot_us=3000, timeout=120.0):
    """The reference sensor on a ring that sense_feed fills from an archive. Returns the report of the feed.

    The slots are stretched, so that an interpreter on a busy development host keeps up with the
    producer and the control holds every window.
    """
    output = Path(output)
    output.mkdir(parents=True)
    socket, ready = output.parent / "obs.sock", output.parent / "ready.json"
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    sensor = subprocess.Popen([sys.executable, "-m", "ocudu_cuda_isac.sensing.cusense", "follow", "--socket", str(socket),
                               "--artifact", str(artifact), "--output-dir", str(output), "--device", "cpu", "--wait", wait,
                               "--ready-file", str(ready)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment)
    deadline = time.monotonic() + timeout
    while not ready.exists() and sensor.poll() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    if not ready.exists():
        sensor.kill()
        raise RuntimeError("the reference sensor was not warm in time: " + sensor.communicate()[0].decode(errors="replace"))
    # A ring of many slots: an interpreter that stalls for some periods still finds every record.
    feed = subprocess.run([str(sense_feed), "--socket", str(socket), "--archive", str(archive), "--slot-us", str(slot_us),
                           "--slots", "128"], capture_output=True, text=True, timeout=timeout)
    said = sensor.communicate(timeout=timeout)[0].decode(errors="replace")
    report = json.loads(feed.stdout.strip().splitlines()[-1]) if feed.returncode == 0 else {}
    if feed.returncode != 0 or sensor.returncode != 0 or report.get("dropped") != 0:
        raise RuntimeError(f"the feed ended with {feed.returncode} ({feed.stderr.strip()}), dropped "
                           f"{report.get('dropped')} records, and the sensor ended with {sensor.returncode} "
                           f"({said.strip()[-300:]})")
    ready.unlink()
    return report


def make_control(directory, sense_feed, episodes=4, periods=10, captures=1, seed=7):
    """A run directory with what the Sprint 3.6 driver reads of a Sprint 3.5 run. Returns its path."""
    run = Path(directory) / "s35-double"
    out = run / DRIVER
    (out / "train" / "o3").mkdir(parents=True)
    artifact = fixtures.make_artifact(out / "train" / "o3" / "artifact", mean=0.1, scale=0.3, smoothing=SMOOTHING,
                                      logit=-0.2)
    manifest = artifact.manifest
    (out / "train" / "o3" / "freeze.json").write_text(json.dumps({
        "schema": "cusense-freeze/double", "frozen_utc": "2026-11-22T00:00:00Z",
        "artifact": {"sha256": artifact.sha256, "weights_sha256": manifest["weights"]["sha256"],
                     "template_sha256": manifest["template"]["sha256"]}}, indent=2) + "\n")
    names = []
    for index in range(captures):
        name = f"held-out-{index:02d}"
        archive = write_archive(out / "offered" / name, records_of(episodes, periods, seed + index))
        follow(sense_feed, archive, artifact.directory, out / "live" / name / "sensor")
        names.append(name)
    (out / "recover.json").write_text(json.dumps({
        "artifact": "train/o3/artifact", "freeze": "train/o3/freeze.json", "sensor": f"live/{names[0]}/sensor",
        "window": 0, "offline": f"parity/{names[0]}"}, indent=2) + "\n")
    (run / "summary.txt").write_text("A stand-in for a Sprint 3.5 run on a development host. It is no accepted run.\n")
    return run


def rehearsal_config(core_build, min_windows=20):
    """The configuration of the driver for a rehearsal on the double: small counts, the same cases."""
    return {"schema": "integration-config/sense_core/v1", "selector": "sense_core", "core_build": str(core_build),
            "control": {"driver": DRIVER, "min_windows": min_windows}, "engine": {"precision": "fp32"},
            "units": {"windows": 6},
            "live": {"repeat": 2, "ring_slots": 8, "slot_us": 500, "wait": "spin", "min_windows": 2 * min_windows,
                     "ready_timeout_ms": 120000, "budget_us": [0, 0, 0, 100000]},
            "overload": {"repeat": 2, "ring_slots": 4, "period_us": 100, "hold_us": 3000, "budget_us": [0, 0, 0, 1]},
            "expect": {"cuda_architecture": 121, "published_live_inference_us": 241}}
