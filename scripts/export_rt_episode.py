#!/usr/bin/env python3
"""Offline Sionna RT 2.2.0 exporter for a paired bistatic channel episode."""

import argparse
import hashlib
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory

RT_VERSION = "2.2.0"
CARRIER_HZ = 3_500_000_000
TIME_STEP_NS = 500_000
TIME_COUNT = 4
OFFSETS_HZ = [index * 30_000 for index in range(-6, 6)]
TX_POSITION_M = [0, 0, 1.5]
RX_POSITION_M = [20, 0, 3]
TARGET = {
    "name": "scatterer",
    "sigma": 1.0,  # square meters
    "length": 0.3, "width": 0.3, "height": 0.3,  # meters
    "position": [9, 4, 1.5],
    "velocity": [0, 1, 0],  # meters per second in world coordinates
}
COMMON_SOLVER = {
    "synthetic_array": True, "los": True, "specular_reflection": False,
    "refraction": False, "seed": 42,
}
BACKGROUND = dict(COMMON_SOLVER, max_depth=0, samples_per_src=64,
                  max_num_paths_per_src=64)
SCATTERED = dict(COMMON_SOLVER, max_depth=1, samples_per_sp=64,
                 buffer_size_per_sp=64)
AXES = ["rx_element", "tx_element", "time", "frequency_offset"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def store_cfr(directory, filename, raw):
    """Map the one-gNB/one-UE v2.2.0 NumPy CFR to the replay antenna axes."""
    import numpy as np

    expected = (1, 4, 1, 1, TIME_COUNT, len(OFFSETS_HZ))
    if raw.shape != expected or not np.iscomplexobj(raw):
        raise ValueError(f"Expected complex CFR in RT axis order {expected}, got {raw.shape} {raw.dtype}")
    channel = np.ascontiguousarray(raw[0, :, 0, :, :, :], dtype="<c8")
    if not np.isfinite(channel).all():
        raise ValueError("Nonfinite channel coefficient")
    path = directory / filename
    with path.open("xb") as file:
        np.save(file, channel, allow_pickle=False)
    return {"file": filename, "sha256": sha256(path), "axes": AXES,
            "shape": list(channel.shape), "dtype": channel.dtype.str,
            "units": "linear complex baseband amplitude gain", "order": "C"}


def trace_scene(present):
    from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene
    from sionna.rt.rcs import ConstantRCSSensingTarget, RCSSolver

    scene = load_scene()
    scene.frequency = CARRIER_HZ
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=4, horizontal_spacing=0.5,
                                 pattern="iso", polarization="V")
    scene.add(Transmitter("ue", position=TX_POSITION_M))
    scene.add(Receiver("gnb", position=RX_POSITION_M))
    if present:
        scene.add(ConstantRCSSensingTarget(**TARGET))

    paths = PathSolver(deterministic=True)(scene, **BACKGROUND)
    if not paths.valid.numpy().any():
        raise RuntimeError("The unobstructed direct path is missing")
    if present:
        scattered = RCSSolver(deterministic=True)(scene, **SCATTERED)
        if not scattered.valid.numpy().any():
            raise RuntimeError("The unobstructed bistatic scattering path is missing")
        paths = paths.concat(scattered)

    raw = paths.cfr(frequencies=OFFSETS_HZ, sampling_frequency=1e9 / TIME_STEP_NS,
                    num_time_steps=TIME_COUNT, normalize_delays=False, normalize=False,
                    reverse_direction=False, out_type="numpy")
    return raw


def write_pair(destination, absent_cfr, present_cfr, software, variant):
    import numpy as np

    manifest = {
        "schema": "rt-cfr-pair/v1", "access": "offline_stimulus_and_evaluator",
        "software": software, "mitsuba_variant": variant,
        "exporter_sha256": sha256(Path(__file__)),
        "geometry": {
            "frame": "Sionna world Cartesian (x,y,z) in meters, z up",
            "time_reference_ns": 0, "scene": "free space; procedural cuboid when present",
            "ue_tx_m": TX_POSITION_M, "gnb_rx_m": RX_POSITION_M,
            "endpoint_orientation_rad": [0, 0, 0], "endpoint_velocity_m_s": [0, 0, 0],
            "tx_offsets_lambda": [[0, 0, 0]],
            "rx_offsets_lambda": [[0, y, 0] for y in (-0.75, -0.25, 0.25, 0.75)],
            "pattern": "iso", "polarization": "V",
        },
        "target_when_present": {"model": "ConstantRCSSensingTarget", "sigma_m2": TARGET["sigma"],
                                "pose_xyz_m": TARGET["position"], "velocity_xyz_m_s": TARGET["velocity"],
                                "dimensions_xyz_m": [TARGET[k] for k in ("length", "width", "height")],
                                "orientation_rad": [0, 0, 0], "xpr_db": None},
        "solver": {"deterministic": True, "background": BACKGROUND, "scattered": SCATTERED},
        "grid": {"carrier_hz": CARRIER_HZ, "frequency_offsets_hz": OFFSETS_HZ,
                 "episode_times_ns": [i * TIME_STEP_NS for i in range(TIME_COUNT)],
                 "normalize_delays": False, "normalize_gain": False,
                 "time_evolution": "single geometry snapshot with path Doppler from Paths.cfr"},
        "cases": [],
    }

    # Stage the pair and its provenance together so readers only see completed exports.
    if destination.exists():
        raise FileExistsError(f"Output directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as temporary:
        staging = Path(temporary) / "episode"
        staging.mkdir()
        for prefix, present, raw in (("a", False, absent_cfr), ("b", True, present_cfr)):
            description = store_cfr(staging, f"{prefix}_h.npy", raw)
            manifest["cases"].append({"id": prefix, "target_present": present, "h": description})
        absent = np.load(staging / "a_h.npy", allow_pickle=False)
        present = np.load(staging / "b_h.npy", allow_pickle=False)
        if not np.any(absent) or np.array_equal(absent, present):
            raise ValueError("Expected a direct-link baseline and a distinct target-present channel")
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staging.rename(destination)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory for the episode")
    args = parser.parse_args(argv)
    try:
        installed = version("sionna-rt")
        if installed != RT_VERSION:
            parser.error(f"this exporter requires sionna-rt=={RT_VERSION}; installed {installed}")
        if args.output_dir.exists():
            parser.error(f"output directory already exists: {args.output_dir}")
        import mitsuba as mi

        software = {name: version(name) for name in ("sionna-rt", "mitsuba", "drjit", "numpy")}
        absent = trace_scene(False)
        present = trace_scene(True)
        write_pair(args.output_dir, absent, present, software, mi.variant())
    except PackageNotFoundError as exc:
        parser.exit(2, f"Missing offline runtime dependency ({exc}). Use a provisioned sionna-rt=={RT_VERSION} environment.\n")
    except (FileExistsError, ImportError) as exc:
        parser.exit(2, f"{exc}\n")
    print(args.output_dir / "manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
