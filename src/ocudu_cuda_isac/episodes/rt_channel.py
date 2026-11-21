"""Ray-traced channels of an episode on the grant's subcarriers and symbol times (Sprint 2.2).

Offline only. It needs the full environment (Sionna RT 2.2.0, Mitsuba, Dr.Jit, NumPy). The scene
without the target and the scene with it are separate scenes, so the background of the
target-absent case is computed on its own and the target takes part in every visibility test of
the target-present case. PathSolver gives the background paths, RCSSolver the paths scattered by
the target, and Paths.concat joins them.

Time. Paths.cfr evaluates a uniform time grid that starts at the geometry snapshot. The symbols of
a grant are not on a uniform grid, so the path phase is evaluated at the real symbol times:

    H[rx, t, k] = sum over paths of a[rx] exp(-j 2 pi (f_c + f_k) tau) exp(j 2 pi nu (t - t_snapshot))

with a, tau and nu taken from the concatenated Paths object, in double precision. Every snapshot
is compared with Paths.cfr at the snapshot instant, and once per case with the uniform-grid time
evolution of Paths.cfr on a grid that contains the DM-RS symbol times. The geometry is traced
again at the start of every opportunity's slot, so a path phase is extrapolated over at most one
slot. The error of extrapolating to the next opportunity is measured and recorded.
"""

from importlib.metadata import version
import hashlib
import math
from pathlib import Path
import platform
import tempfile
import time

from . import episode as episodes
from . import episode_set as sets
from .episode import EpisodeError, SPEED_OF_LIGHT_M_S, TC_PER_SECOND
from .grant import STORAGE

RT_VERSION = "2.2.0"
VARIANTS = ("llvm_ad_mono_polarized", "cuda_ad_mono_polarized")
# A uniform solver grid that contains every DM-RS symbol time is only built up to this size.
MAX_UNIFORM_STEPS = 2048
# Rounding error of the receiver grid's complex bf16 storage relative to the stored value: the RMS
# level measured in Sprint 2.1 on values of arbitrary amplitude. The hard bound is STORAGE["relative"].
GRID_ROUNDING_RMS_DB = -55.6
ROOM_FACES = ("floor", "ceiling", "wall_y0", "wall_y1", "wall_x0", "wall_x1")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime(variant):
    """Select the Mitsuba variant, then import Sionna RT. Returns the modules the tracer uses."""
    import mitsuba as mi

    installed = version("sionna-rt")
    if installed != RT_VERSION:
        raise EpisodeError(f"the episode exporter requires sionna-rt=={RT_VERSION}; installed {installed}")
    if variant not in VARIANTS:
        raise EpisodeError(f"the Mitsuba variant must be one of {', '.join(VARIANTS)}")
    mi.set_variant(variant)
    import numpy as np
    import sionna.rt as rt
    from sionna.rt import rcs

    return {"mi": mi, "np": np, "rt": rt, "rcs": rcs, "variant": mi.variant()}


def room_files(directory, scene):
    """Write the procedural room (floor, ceiling, four walls) as a Mitsuba scene and return its file hashes."""
    lx, ly, lz = scene["size_m"]
    corners = {
        "floor": [(0, 0, 0), (lx, 0, 0), (lx, ly, 0), (0, ly, 0)],
        "ceiling": [(0, 0, lz), (0, ly, lz), (lx, ly, lz), (lx, 0, lz)],
        "wall_y0": [(0, 0, 0), (0, 0, lz), (lx, 0, lz), (lx, 0, 0)],
        "wall_y1": [(0, ly, 0), (lx, ly, 0), (lx, ly, lz), (0, ly, lz)],
        "wall_x0": [(0, 0, 0), (0, ly, 0), (0, ly, lz), (0, 0, lz)],
        "wall_x1": [(lx, 0, 0), (lx, 0, lz), (lx, ly, lz), (lx, ly, 0)]}
    materials = {"floor": scene["materials"]["floor"], "ceiling": scene["materials"]["ceiling"]}
    materials.update({face: scene["materials"]["walls"] for face in ROOM_FACES[2:]})
    directory = Path(directory)
    (directory / "meshes").mkdir(parents=True)
    xml = ['<scene version="2.1.0">']
    for material in sorted(set(materials.values())):
        xml += [f'  <bsdf type="itu-radio-material" id="mat-{material}">',
                f'    <string name="type" value="{material}"/>',
                f'    <float name="thickness" value="{scene["thickness_m"]:.6f}"/>', "  </bsdf>"]
    for face in ROOM_FACES:
        ply = ["ply", "format ascii 1.0", "element vertex 4", "property float x", "property float y",
               "property float z", "element face 2", "property list uchar int vertex_indices", "end_header"]
        ply += ["%.6f %.6f %.6f" % corner for corner in corners[face]] + ["3 0 1 2", "3 0 2 3"]
        (directory / "meshes" / f"{face}.ply").write_text("\n".join(ply) + "\n", encoding="ascii")
        xml += [f'  <shape type="ply" id="mesh-{face}">', f'    <string name="filename" value="meshes/{face}.ply"/>',
                '    <boolean name="face_normals" value="true"/>', f'    <ref id="mat-{materials[face]}" name="bsdf"/>',
                "  </shape>"]
    (directory / "scene.xml").write_text("\n".join(xml + ["</scene>"]) + "\n", encoding="ascii")
    files = {str(path.relative_to(directory)): sha256(path) for path in sorted(directory.rglob("*")) if path.is_file()}
    return files


def synthesize(np, paths, offsets_hz, times_s, carrier_hz, select=None):
    """H[rx, time, subcarrier] of a path set at times measured from its snapshot, in double precision."""
    keep = slice(None) if select is None else select
    a, tau, doppler = paths["a"][:, keep], paths["tau"][keep], paths["doppler"][keep]
    shape = (a.shape[0], len(times_s), len(offsets_hz))
    if a.shape[1] == 0:
        return np.zeros(shape, dtype=np.complex128)
    frequency = np.exp(-2j * np.pi * np.outer(tau, carrier_hz + np.asarray(offsets_hz, dtype=np.float64)))
    evolution = np.exp(2j * np.pi * np.outer(doppler, np.asarray(times_s, dtype=np.float64)))
    return np.einsum("rp,pt,pk->rtk", a, evolution, frequency, optimize=True)


class Tracer:
    """Both scenes of one static configuration. Reused for every episode that shares them."""

    def __init__(self, config, base, variant):
        self.config, self.base = config, base
        self.modules = runtime(variant)
        mi, np, rt, rcs = (self.modules[key] for key in ("mi", "np", "rt", "rcs"))
        self.np = np
        self.offsets = [base.subcarrier_offset_hz(k) for k in range(base.grid_subcarriers)]
        self._staging = tempfile.TemporaryDirectory()
        scene = config["scene"]
        if scene["kind"] == "open_room":
            files = room_files(Path(self._staging.name) / "room", scene)
            source = str(Path(self._staging.name) / "room" / "scene.xml")
            self.scene_record = dict(scene, files=files, sha256=hashlib.sha256(
                "".join(f"{name}:{digest}\n" for name, digest in sorted(files.items())).encode("ascii")).hexdigest())
        else:
            source = None
            self.scene_record = dict(scene, sha256=hashlib.sha256(b"free_space").hexdigest())
        endpoints, array = config["endpoints"], config["array"]
        self.scenes = {}
        for present in (False, True):
            item = rt.load_scene(source) if source else rt.load_scene()
            item.frequency = base.carrier_hz
            item.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern=array["pattern"],
                                           polarization=array["polarization"])
            item.rx_array = rt.PlanarArray(num_rows=1, num_cols=len(base.rx_ports),
                                           horizontal_spacing=array["spacing_wavelengths"], pattern=array["pattern"],
                                           polarization=array["polarization"])
            item.add(rt.Transmitter("ue", position=endpoints["ue_tx_m"]))
            item.add(rt.Receiver("gnb", position=endpoints["gnb_rx_m"], orientation=[endpoints["gnb_yaw_rad"], 0, 0]))
            self.scenes[present] = item
        target = config["target"]
        if target["model"] == "constant_rcs_cuboid":
            length, width, height = target["size_m"]
            self.target = rcs.ConstantRCSSensingTarget(name="target", sigma=target["sigma_m2"], length=length,
                                                       width=width, height=height, position=[0, 0, 0])
        else:
            self.target = rcs.TR38901SensingTarget(name="target", object_type="human", position=[0, 0, 0])
        self.scenes[True].add(self.target)
        self.wavelength = SPEED_OF_LIGHT_M_S / base.carrier_hz
        local = np.asarray(self.scenes[False].rx_array.positions(self.wavelength).numpy(), dtype=np.float64)
        local = local.reshape(3, -1).T  # [element, xyz] in the array's own frame
        yaw = endpoints["gnb_yaw_rad"]
        self.rx_offsets = [[float(x * math.cos(yaw) - y * math.sin(yaw)), float(x * math.sin(yaw) + y * math.cos(yaw)),
                            float(z)] for x, y, z in local]
        self.rx_offsets_wavelengths = [[float(part / self.wavelength) for part in row] for row in local]
        self.path_solver = rt.PathSolver(deterministic=True)
        self.rcs_solver = rcs.RCSSolver(deterministic=True)

    def close(self):
        self._staging.cleanup()

    def geometry(self):
        endpoints = self.config["endpoints"]
        return {"frame": "Sionna world Cartesian (x, y, z) in meters, z up",
                "ue_tx_m": [float(v) for v in endpoints["ue_tx_m"]], "gnb_rx_m": [float(v) for v in endpoints["gnb_rx_m"]],
                "gnb_yaw_rad": float(endpoints["gnb_yaw_rad"]), "endpoint_velocity_m_s": [0.0, 0.0, 0.0],
                "tx_element_offsets_m": [[0.0, 0.0, 0.0]], "rx_element_offsets_m": self.rx_offsets,
                "rx_element_offsets_wavelengths_array_frame": self.rx_offsets_wavelengths}

    def _arrays(self, paths, background_count):
        np = self.np
        valid = np.asarray(paths.valid.numpy()).reshape(-1)
        real, imag = paths.a
        a = (np.asarray(real.numpy(), dtype=np.float64) + 1j * np.asarray(imag.numpy(), dtype=np.float64))
        a = a.reshape(a.shape[1], -1)  # [rx element, path]: one receiver, one transmit element
        scattered = np.arange(valid.size) >= background_count
        return {"a": a[:, valid], "tau": np.asarray(paths.tau.numpy(), dtype=np.float64).reshape(-1)[valid],
                "doppler": np.asarray(paths.doppler.numpy(), dtype=np.float64).reshape(-1)[valid],
                "scattered": scattered[valid]}

    def snapshot(self, state=None):
        """Trace one geometry. state is (position, velocity) of the target, or None for the scene without it."""
        mi, np = self.modules["mi"], self.np
        solver = self.config["solver"]
        present = state is not None
        scene = self.scenes[present]
        common = {"synthetic_array": True, "seed": solver["seed"]}
        started = time.perf_counter()
        if present:
            position, velocity = state
            self.target.position = mi.Point3f(*[float(v) for v in position])
            self.target.velocity = mi.Vector3f(*[float(v) for v in velocity])
            if self.config["target"]["model"] == "tr38901_human" and math.hypot(velocity[0], velocity[1]) > 0:
                self.target.orientation = mi.Point3f(math.atan2(velocity[1], velocity[0]), 0.0, 0.0)
        paths = self.path_solver(scene, diffuse_reflection=False, diffraction=False, **solver["background"], **common)
        background_count = int(np.asarray(paths.valid.numpy()).reshape(-1).size)
        if not np.asarray(paths.valid.numpy()).any():
            raise EpisodeError("no valid background path: the direct link is missing")
        if present:
            scattered = self.rcs_solver(scene, **solver["scattered"], **common)
            if not np.asarray(scattered.valid.numpy()).any():
                raise EpisodeError(f"no valid scattering path with the target at {list(position)}")
            paths = paths.concat(scattered)
        seconds = time.perf_counter() - started
        result = self._arrays(paths, background_count)
        if np.any(result["scattered"]) != present or not np.isfinite(result["a"]).all():
            raise EpisodeError("the traced paths are inconsistent with the scene")
        # Sionna's own evaluation of the same paths at the snapshot instant.
        solver_h = paths.cfr(frequencies=[float(f) for f in self.offsets], sampling_frequency=1.0, num_time_steps=1,
                             normalize_delays=False, normalize=False, reverse_direction=False, out_type="numpy")
        solver_h = np.asarray(solver_h).reshape(result["a"].shape[0], len(self.offsets))
        own = synthesize(np, result, self.offsets, [0.0], self.base.carrier_hz)[:, 0, :]
        result["solver_cfr_error"] = float(np.max(np.abs(own - solver_h)) / np.sqrt(np.mean(np.abs(solver_h) ** 2)))
        result["seconds"], result["paths_object"] = seconds, paths
        return result

    def uniform_grid_error(self, paths, symbol_times_s):
        """Largest difference to the time evolution of Paths.cfr on a uniform grid through the given times."""
        np = self.np
        ticks = [round(value * TC_PER_SECOND) for value in symbol_times_s if value > 0]
        step = math.gcd(*ticks) if ticks else 0
        if not step or max(ticks) // step + 1 > MAX_UNIFORM_STEPS:
            return None
        count = max(ticks) // step + 1
        edges = [self.offsets[0], self.offsets[len(self.offsets) // 2], self.offsets[-1]]
        solver_h = paths["paths_object"].cfr(frequencies=[float(f) for f in edges],
                                             sampling_frequency=TC_PER_SECOND / step, num_time_steps=count,
                                             normalize_delays=False, normalize=False, reverse_direction=False,
                                             out_type="numpy")
        solver_h = np.asarray(solver_h).reshape(paths["a"].shape[0], count, len(edges))[:, [tick // step for tick in ticks]]
        own = synthesize(np, paths, edges, [tick / TC_PER_SECOND for tick in ticks], self.base.carrier_hz)
        return {"step_tc": step, "steps": count, "times_checked": len(ticks),
                "max_relative_error": float(np.max(np.abs(own - solver_h)) / np.sqrt(np.mean(np.abs(solver_h) ** 2)))}

    def pair(self, schedule, motion, noise):
        """Trace both cases of one episode. Returns (arrays, truth, measured, description parts)."""
        np, base = self.np, self.base
        carrier = base.carrier_hz
        opportunities = schedule.opportunities
        h_shape, w_shape = episodes.shapes(schedule)
        h = {case: np.zeros(h_shape, dtype="<c8") for case in ("a", "b")}

        # ---- Target absent: one geometry, nothing moves. ----
        absent = self.snapshot(None)
        if np.any(absent["doppler"] != 0):
            raise EpisodeError("a path of the target-free scene has a Doppler shift")
        static = synthesize(np, absent, self.offsets, [0.0], carrier)[:, 0, :]
        h["a"][:] = static[None, :, None, None, :]
        first = int(np.argmin(absent["tau"]))
        measured = {"a": {"snapshots": 1, "paths": int(absent["tau"].size),
                          "solver_cfr_max_relative_error": absent["solver_cfr_error"],
                          "direct_path": {"delay_ns": float(1e9 * absent["tau"][first]),
                                          "amplitude": [float(v) for v in np.abs(absent["a"][:, first])],
                                          "phase_rad": [float(v) for v in np.angle(absent["a"][:, first])]},
                          "solve_seconds": absent["seconds"]}}

        # ---- Target present: the geometry is traced again at the start of every opportunity's slot. ----
        truth, ratios, counts, cfr_errors, seconds, validity, previous, uniform = [], [], [], [], [], [], None, None
        for opportunity in opportunities:
            grant = opportunity.grant
            reference = opportunity.slot_start_tc / TC_PER_SECOND
            position, velocity = episodes.target_state(motion, reference)
            traced = self.snapshot((position, velocity))
            times = [entry["useful_start_tc"] / TC_PER_SECOND for entry in grant.symbol_times()]
            is_target = traced["scattered"]
            scattered = synthesize(np, traced, self.offsets, times, carrier, is_target)
            background = synthesize(np, traced, self.offsets, times, carrier, ~is_target)
            h["b"][opportunity.index, :, 0] = scattered + background
            ratio = np.abs(scattered) ** 2 / np.abs(background) ** 2
            ratios.append(ratio)
            if previous is not None:
                gap = reference - previous[1]
                ahead = synthesize(np, previous[0], self.offsets, [gap], carrier, previous[0]["scattered"])
                now = synthesize(np, traced, self.offsets, [0.0], carrier, is_target)
                validity.append((gap, float(np.sqrt(np.mean(np.abs(ahead - now) ** 2) / np.mean(np.abs(now) ** 2)))))
            if uniform is None:
                uniform = self.uniform_grid_error(traced, [times[symbol] for symbol in grant.dmrs["symbols"]])
            previous = (traced, reference)
            counts.append((int((~is_target).sum()), int(is_target.sum())))
            cfr_errors.append(traced["solver_cfr_error"])
            seconds.append(traced["seconds"])
            truth.append({"opportunity": opportunity.index, "time_s": reference,
                          "position_m": [float(v) for v in position], "velocity_m_s": [float(v) for v in velocity],
                          "bistatic": episodes.bistatic(self.geometry(), position, velocity, carrier),
                          "scattered_to_direct_db": float(10 * np.log10(np.mean(ratio)))})
        del previous
        all_ratios = 10 * np.log10(np.concatenate([ratio.reshape(-1) for ratio in ratios]))
        floor_db = 20 * math.log10(STORAGE["relative"])
        stats = {"definition": "10 log10(|H of the target's paths|^2 / |H of the background paths|^2) per resource "
                               "element, receive port and opportunity of the target-present case",
                 "elements": int(all_ratios.size), "min": float(all_ratios.min()),
                 "p05": float(np.percentile(all_ratios, 5)), "median": float(np.median(all_ratios)),
                 "p95": float(np.percentile(all_ratios, 95)), "max": float(all_ratios.max()),
                 "mean_power": float(10 * np.log10(np.mean(np.concatenate([r.reshape(-1) for r in ratios])))),
                 "grid_rounding_bound_db": floor_db, "grid_rounding_rms_db": GRID_ROUNDING_RMS_DB}
        stats["median_margin_to_rounding_bound_db"] = stats["median"] - floor_db
        stats["median_margin_to_rounding_rms_db"] = stats["median"] - GRID_ROUNDING_RMS_DB
        measured["b"] = {
            "snapshots": len(opportunities), "background_paths": [min(c[0] for c in counts), max(c[0] for c in counts)],
            "scattered_paths": [min(c[1] for c in counts), max(c[1] for c in counts)],
            "solver_cfr_max_relative_error": max(cfr_errors), "uniform_grid_check": uniform,
            "scattered_to_direct_db": stats,
            "solve_seconds": {"mean": float(np.mean(seconds)), "max": float(np.max(seconds)), "total": float(np.sum(seconds))},
            "path_validity": {
                "definition": "relative RMS difference of the target's paths, extrapolated by their Doppler shift from "
                              "one opportunity's snapshot to the next opportunity's slot start, against the paths "
                              "traced there",
                "gaps_ms": sorted({round(1e3 * gap, 6) for gap, _ in validity}),
                "max_relative_error": max((error for _, error in validity), default=None),
                "by_gap_ms": {str(round(1e3 * gap, 6)): max(e for g, e in validity if g == gap)
                              for gap in sorted({g for g, _ in validity})}}}

        # ---- One noise realization for both cases, in the units of H times unit-power data. ----
        grant0 = opportunities[0].grant
        symbols, subcarriers = zip(*sorted(grant0.occupied()))
        reference_power = float(np.mean(np.abs(h["a"][0, :, 0][:, list(symbols), list(subcarriers)]) ** 2))
        sigma = math.sqrt(reference_power * 10 ** (-noise["snr_db"] / 10))
        generator = np.random.Generator(np.random.PCG64(noise["seed"]))
        w = generator.standard_normal(w_shape) + 1j * generator.standard_normal(w_shape)
        w = (w * (sigma / math.sqrt(2.0))).astype("<c8")
        parts = {
            "noise": {"snr_db": float(noise["snr_db"]), "sigma": sigma, "seed": noise["seed"],
                      "generator": "numpy.random.Generator(PCG64(seed)).standard_normal, real part drawn first",
                      "numpy": np.__version__,
                      "definition": "circular complex Gaussian with E|W|^2 = sigma^2 = reference_power / 10^(snr_db/10) "
                                    "on every resource element of the carrier grid and receive port"},
            "calibration": {
                "reference_power": reference_power,
                "reference_power_definition": "mean |H|^2 of the target-absent case over the receive ports and the "
                                              "occupied resource elements of opportunity 0",
                "grid_gain": 1.0 / math.sqrt(reference_power),
                "grid_gain_rule": "a real factor on Y = sum(H X) + W that gives the receiver grid unit mean signal "
                                  "power; recorded for the consumer and not applied to the stored arrays"},
            "time_evolution": {
                "method": "geometry traced at the slot start of every opportunity; inside the slot each path phase "
                          "advances by its Doppler shift, evaluated at the useful start of every symbol",
                "snapshot": "slot_start_tc of the opportunity", "target_absent": "one snapshot, no moving object",
                "online_updates": False,
                "update_cost_seconds": measured["b"]["solve_seconds"],
                "path_validity": measured["b"]["path_validity"]}}
        arrays = {"a": h["a"].tobytes(), "b": h["b"].tobytes(), "w": w.tobytes()}
        return arrays, truth, measured, parts

    def description(self, config_sha256, parts, labels=None, motion=None):
        """The scene, solver and software part of an episode manifest."""
        modules, config = self.modules, self.config
        software = {name: version(name) for name in ("sionna-rt", "mitsuba", "drjit", "numpy")}
        software.update(python=platform.python_version(), machine=platform.machine())
        here = Path(__file__).resolve().parent
        result = {
            "geometry": self.geometry(),
            "array": dict(config["array"], rx_elements=len(self.base.rx_ports), synthetic_array=True,
                          order="element i of the receive array is receive plane i"),
            "scene": self.scene_record,
            "target": dict(config["target"], motion=motion) if motion else config["target"],
            "solver": {"deterministic": True, "mitsuba_variant": modules["variant"], **config["solver"],
                       "classes": "PathSolver for the background, RCSSolver for the target, Paths.concat"},
            "software": software,
            "provenance": {"config_sha256": config_sha256,
                           "modules": {name: sha256(here / name) for name in
                                       ("episode.py", "episode_set.py", "rt_channel.py", "schedule.py", "grant.py")}},
            **parts}
        if config.get("analytic_model"):
            result["analytic_model"] = config["analytic_model"]
        if labels is not None:
            result["labels"] = labels
        return result


def export_pair(config_path, output_dir, variant, waveforms=None):
    """Trace and write the matched pair of one episode-pair-config/v1 file. Returns its manifest."""
    config, base, schedule = sets.load_pair_config(config_path)
    tracer = Tracer(config, base, variant)
    try:
        arrays, truth, measured, parts = tracer.pair(schedule, config["target"]["motion"], config["noise"])
        description = tracer.description(sha256(config_path), parts)
        return episodes.export(output_dir, config["episode_id"], schedule, description, arrays, truth, measured,
                               waveforms)
    finally:
        tracer.close()


def export_set(config_path, output_dir, variant, progress=None):
    """Trace and write every episode of one episode-set-config/v1 file. Returns the set manifest."""
    config, base = sets.load_set_config(config_path)
    plans = sets.plan(config, base)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise EpisodeError(f"{output_dir} already exists. An episode set is never overwritten.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    tracer = Tracer(config, base, variant)
    try:
        with tempfile.TemporaryDirectory(dir=output_dir.parent) as temporary:
            staging = Path(temporary) / "set"
            (staging / "episodes").mkdir(parents=True)
            manifests = {}
            for number, item in enumerate(plans):
                arrays, truth, measured, parts = tracer.pair(item["schedule"], item["motion"], item["noise"])
                sets.label(config, item, truth)
                description = tracer.description(sha256(config_path), parts, labels=item["labels"],
                                                 motion=item["motion"])
                manifests[item["episode_id"]] = episodes.export(staging / "episodes" / item["episode_id"],
                                                                item["episode_id"], item["schedule"], description,
                                                                arrays, truth, measured)
                if progress:
                    progress(number + 1, len(plans), item)
            manifest = sets.write_manifest(staging, config, sha256(config_path), plans, manifests)
            staging.rename(output_dir)
        return manifest
    finally:
        tracer.close()
