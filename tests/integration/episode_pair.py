#!/usr/bin/env python3
"""Sprint 2.2 integration driver (selector episode_pair). Hardware command, not part of unit discovery.

Four cases, each graded on its own:

  diagnostic       the four-time, twelve-tone exporter, kept as a regression of the axis order
  grant-fixture    the uplink grant fixture of the pair: one legal grant per scheduled uplink slot,
                   the base grant bound to the record that Sprint 2.1 archived and every other slot
                   realized by the accepted native fixture adapter, with a receiver check
  reference-pair   the matched target-absent and target-present pair on that schedule, traced with
                   Sionna RT, stored, read back and compared with the analytic two-path model at
                   every grant coordinate and symbol time
  episode-set      a small demonstration set: a room, a walking person, whole trajectories assigned
                   to training, validation and held-out groups

    python tests/integration/episode_pair.py --config configs/integration/episode_pair.json \\
        --grant-store results/runs/s21-spark-001/grant_fixture --output-dir results/runs/s22-spark-001

The output directory must be new. The exit status is 0 only when every check of every case passed.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from ocudu_cuda_isac.episodes import episode as episodes  # noqa: E402
from ocudu_cuda_isac.episodes import episode_set as sets  # noqa: E402
from ocudu_cuda_isac.episodes import grant as grants  # noqa: E402
from ocudu_cuda_isac.episodes import rt_channel  # noqa: E402
from ocudu_cuda_isac.episodes.episode import EpisodeError, TC_PER_SECOND  # noqa: E402
from ocudu_cuda_isac.episodes.grant import GrantError  # noqa: E402
from ocudu_cuda_isac.replay import ContractError, load_json  # noqa: E402
import native_adapter  # noqa: E402

CONFIG_SCHEMA = "integration-config/episode_pair/v1"
GRADE_SCHEMA = "integration-grade/episode_pair/v1"
CASES = ("diagnostic", "grant-fixture", "reference-pair", "episode-set")
TOLERANCES = ("solver_cfr_relative", "direct_relative", "echo_relative", "phase_rad", "path_validity_relative",
              "diagnostic_delay_ns", "diagnostic_doppler_hz", "peak_snr_db_min", "ratio_db")


class ConfigError(ValueError):
    """The configuration or the run setup is unusable. Nothing was graded."""


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fixture = load_script("grant_fixture_driver", ROOT / "tests/integration/grant_fixture.py")
exporter = load_script("rt_export", ROOT / "scripts/export_rt_episode.py")
digest, write_json, Grader = fixture.digest, fixture.write_json, fixture.Grader


class Grade(Grader):
    """The Sprint 2.1 grader with this sprint's label."""

    def check(self, case, name, passed, detail):
        self.results.append({"case": case, "check": name, "passed": bool(passed), "detail": detail})
        self._line(("PASS" if passed else "FAIL") + f"  S2.2 {case} {name}: {detail}")
        return bool(passed)

    def note(self, case, text):
        self._line(f"NOTE  S2.2 {case} {text}")


def load_config(path):
    try:
        config = load_json(Path(path))
    except (ContractError, OSError) as error:
        raise ConfigError(f"cannot read the configuration: {error}") from None
    needed = {"schema", "selector", "adapter_build", "variant", "tolerance", "diagnostic", "pair", "set"}
    if not isinstance(config, dict) or set(config) != needed or config["schema"] != CONFIG_SCHEMA \
            or config["selector"] != "episode_pair":
        raise ConfigError(f"the configuration must be a {CONFIG_SCHEMA} object for the selector episode_pair")
    if not isinstance(config["tolerance"], dict) or set(config["tolerance"]) != set(TOLERANCES):
        raise ConfigError(f"tolerance requires exactly {', '.join(TOLERANCES)}")
    if config["variant"] not in rt_channel.VARIANTS:
        raise ConfigError(f"variant must be one of {', '.join(rt_channel.VARIANTS)}")
    for key, fields in (("diagnostic", {"expect"}), ("pair", {"config", "expect"}), ("set", {"config", "expect"})):
        if not isinstance(config[key], dict) or set(config[key]) != fields or not isinstance(config[key]["expect"], dict):
            raise ConfigError(f"{key} requires exactly {', '.join(sorted(fields))}")
    return config


def worst(np, value, reference, scale):
    """Largest |value - reference| relative to a scale."""
    return float(np.max(np.abs(value - reference)) / scale)


# ---------------------------------------------------------------- diagnostic
def run_diagnostic(config, out, grader, np):
    name, tolerance, expect = "diagnostic", config["tolerance"], config["diagnostic"]["expect"]
    target = out / "diagnostic" / "rt-pair"
    command = [sys.executable, str(ROOT / "scripts/export_rt_episode.py"), "--output-dir", str(target),
               "--variant", config["variant"]]
    (out / "diagnostic").mkdir()
    with (out / "diagnostic" / "export.log").open("w") as log:
        log.write(shlex.join(command) + "\n")
        log.flush()
        status = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False).returncode
    try:
        manifest = json.loads((target / "manifest.json").read_text())
        arrays = {case["id"]: np.load(target / case["h"]["file"], allow_pickle=False) for case in manifest["cases"]}
        hashes = all(digest(target / case["h"]["file"]) == case["h"]["sha256"] for case in manifest["cases"])
    except (OSError, ValueError, KeyError) as error:
        grader.check(name, "exporter", False, f"exit status {status}, no readable pair: {error} (diagnostic/export.log)")
        return {}
    shape = [4, 1, exporter.TIME_COUNT, len(exporter.OFFSETS_HZ)]
    grader.check(name, "exporter", status == 0 and hashes and manifest["schema"] == "rt-cfr-pair/v1"
                 and all(list(array.shape) == shape and array.dtype.str == "<c8" for array in arrays.values()),
                 f"scripts/export_rt_episode.py wrote {manifest['schema']} with both cases as complex64 "
                 f"{shape}, hashes verified, variant {manifest['mitsuba_variant']} (diagnostic/rt-pair)")
    try:
        direct = exporter.check_direct_path(arrays["a"][None, :, None, :, :, :].astype(np.complex64))
        grader.check(name, "direct path", True,
                     f"free-space gain within {direct['amplitude_relative_error']:.1e} and delay "
                     f"{direct['delay_ns_observed']:.3f} ns within {direct['delay_ns_max_error']:.3f} ns of "
                     f"{direct['delay_ns_expected']:.3f} ns")
    except ValueError as error:
        direct = {"error": str(error)}
        grader.check(name, "direct path", False, str(error))
    # The return alone: its phase slope over the twelve tones is the delay, over the four times the Doppler shift.
    echo = arrays["b"].astype(np.complex128) - arrays["a"].astype(np.complex128)
    step_hz = exporter.OFFSETS_HZ[1] - exporter.OFFSETS_HZ[0]
    delay_ns = float(-1e9 * np.angle(np.sum(echo[..., 1:] * np.conj(echo[..., :-1]))) / (2 * math.pi * step_hz))
    doppler = float(np.angle(np.sum(echo[:, :, 1:, :] * np.conj(echo[:, :, :-1, :])))
                    / (2 * math.pi * exporter.TIME_STEP_NS * 1e-9))
    geometry = {"ue_tx_m": exporter.TX_POSITION_M, "gnb_rx_m": exporter.RX_POSITION_M}
    model = episodes.bistatic(geometry, exporter.TARGET["position"], exporter.TARGET["velocity"], exporter.CARRIER_HZ)
    excess = delay_ns - exporter.direct_path_reference()["delay_ns"]
    grader.check(name, "scattered return",
                 abs(excess - model["excess_delay_ns"]) <= tolerance["diagnostic_delay_ns"]
                 and abs(doppler - model["doppler_hz"]) <= tolerance["diagnostic_doppler_hz"]
                 and abs(model["excess_delay_ns"] - expect["excess_delay_ns"]) <= 0.01
                 and abs(model["doppler_hz"] - expect["doppler_hz"]) <= 0.01,
                 f"excess delay {excess:.3f} ns and Doppler {doppler:+.3f} Hz, geometry gives "
                 f"{model['excess_delay_ns']:.3f} ns and {model['doppler_hz']:+.3f} Hz: an axis test only, far below "
                 "the delay resolution of the demonstration bandwidth")
    return {"exit_status": status, "direct": direct, "excess_delay_ns": excess, "doppler_hz": doppler, "model": model}


# ------------------------------------------------------------- grant fixture
def run_grant_fixture(config, out, grader, adapter, receipt, store_root):
    """Bind the base grant to its archived record and realize the grant of every other scheduled slot."""
    name, expect = "grant-fixture", config["pair"]["expect"]
    case_dir = out / "grant_fixture"
    case_dir.mkdir()
    pair_config, base, schedule = sets.load_pair_config(ROOT / config["pair"]["config"])
    opportunities, measured, waveforms = schedule.opportunities, {}, {}
    times = base.symbol_times()
    first, last = base.dmrs["symbols"][0], base.dmrs["symbols"][-1]
    spacing = 1e6 * (times[last]["useful_start_tc"] - times[first]["useful_start_tc"]) / TC_PER_SECOND
    gap = 1e6 * (opportunities[1].symbol_time_tc(first) - opportunities[0].symbol_time_tc(last)) / TC_PER_SECOND
    span = [base.subcarrier_offset_hz(0) / 1e6, base.subcarrier_offset_hz(base.grid_subcarriers - 1) / 1e6]
    legal = all(schedule.pattern[item.absolute_slot % len(schedule.pattern)] == "U" for item in opportunities) \
        and all(b.slot_start_tc > a.slot_start_tc for a, b in zip(opportunities, opportunities[1:]))
    measured["schedule"] = {"opportunities": len(opportunities), "grants": sorted(schedule.grants),
                            "dmrs_spacing_us": spacing, "next_slot_gap_us": gap, "span_mhz": span,
                            "duration_ms": 1e3 * schedule.duration_tc() / TC_PER_SECOND}
    grader.check(name, "schedule", legal and len(opportunities) == expect["opportunities"]
                 and len(schedule.grants) == expect["grants"] and abs(spacing - expect["dmrs_spacing_us"]) < 0.01
                 and abs(gap - expect["next_slot_gap_us"]) < 0.01 and span == expect["span_mhz"],
                 f"{len(opportunities)} opportunities on the uplink slots {schedule.ul_slots} of {schedule.pattern} over "
                 f"{schedule.periods} periods, {len(schedule.grants)} grants; from the record: subcarriers "
                 f"{span[0]:+.2f} to {span[1]:+.2f} MHz, DM-RS symbols {spacing:.2f} us apart, next slot's first "
                 f"DM-RS {gap:.2f} us later")

    # ---- The base grant: the record and the transmit grid that Sprint 2.1 archived. ----
    base_x = None
    try:
        store = grants.open_store(store_root / base.grant_id / "store")
        same = store.grant.record() == base.record() and store.x is not None
        base_x = store.x
        x_sha = store.manifest["files"]["x.npy"]["sha256"]
        waveforms[base.grant_id] = {"x_sha256": x_sha, "store": str(store.directory)}
        measured["base"] = {"store": str(store.directory), "x_sha256": x_sha,
                            "tb_sha256": store.manifest["derived"]["tb_sha256"]}
        grader.check(name, "base grant record", same,
                     f"{base.grant_id} of {config['pair']['config']} equals the archived record, every file verified; "
                     f"its transmit grid x.npy {x_sha[:16]} is the waveform of slot {base.slot}" if same else
                     "the archived record differs from the grant of the pair configuration")
        if expect.get("base_x_sha256"):
            grader.note(name, f"x.npy of {base.grant_id} against the Sprint 2.1 accepted run (not graded): "
                              f"{'identical' if x_sha == expect['base_x_sha256'] else 'different'}")
    except (GrantError, OSError) as error:
        grader.check(name, "base grant record", False, f"no verified Sprint 2.1 record of {base.grant_id}: {error}")

    # ---- Every other slot index: the same transport block with that slot's DM-RS, through the accepted adapter. ----
    width = base.grid_subcarriers
    for grant_id, grant in sorted(schedule.grants.items()):
        if grant is base:
            continue
        label = f"slot {grant.slot} grant"
        try:
            slot_dir = case_dir / grant_id
            slot_dir.mkdir()
            grants.export(grant, slot_dir / "request")
            status, report = fixture.run_adapter(adapter, slot_dir / "request", slot_dir / "native", "crb0",
                                                 slot_dir / "native.log")
            if status != 0 or not report or report.get("schema") != fixture.NATIVE_SCHEMA:
                grader.check(name, label, False, f"the adapter exited with status {status} "
                                                 f"(grant_fixture/{grant_id}/native.log)")
                continue
            native = fixture.read_grid(slot_dir / "native" / "x_native.c64",
                                       grant.antenna_ports * grants.SYMBOLS_PER_SLOT * width)
            checked = grants.check_reference(grant, native)
            loop = report.get("loopback", {})
            decoded = bool(loop.get("executed") and loop.get("tb_crc_ok") and loop.get("bytes_equal")
                           and loop.get("bytes_compared") == len(grant.tb))
            data_equal = base_x is not None and all(
                native[symbol * width + subcarrier] == base_x[symbol * width + subcarrier]
                for symbol, subcarrier in grant.data_index)
            moved = base_x is not None and sum(native[symbol * width + subcarrier] != base_x[symbol * width + subcarrier]
                                               for symbol, subcarrier in grant.pilot_index)
            manifest = grants.export(grant, slot_dir / "store", {"x": {"values": native, "provenance": {
                "reference": "pinned OCUDU transmit chain at the Sprint 1 native boundary, read back from the "
                             "resource grid", "dmrs_reference": "crb0", "schedule": schedule.schedule_id,
                "adapter_sha256": receipt["executables"]["grant_fixture"]["sha256"],
                "ocudu_build": receipt["ocudu_build"], "built_with_cuda": report.get("built_with_cuda"),
                "storage": report["grid"]["storage"]}}})
            grants.open_store(slot_dir / "store")
            x_sha = manifest["files"]["x.npy"]["sha256"]
            waveforms[grant_id] = {"x_sha256": x_sha, "store": str(slot_dir / "store")}
            measured[grant_id] = {"x_sha256": x_sha, "pilots": checked, "loopback": loop,
                                  "data_equal_to_base": data_equal, "pilots_differing_from_base": moved}
            ok = fixture.pdu_matches(grant, report) and decoded and data_equal and bool(moved)
            grader.check(name, label, ok,
                         f"{grant_id}: the adapter echoed the grant's PDU, {checked['occupied_re']} occupied elements "
                         f"and {checked['pilots_checked']} pilots equal the record, {len(grant.data_index)} data symbols "
                         f"{'equal' if data_equal else 'DIFFER from'} those of slot {base.slot}, {moved} pilots differ "
                         f"from slot {base.slot}, host receiver CRC {'pass' if loop.get('tb_crc_ok') else 'FAIL'} with "
                         f"{loop.get('bytes_compared')} bytes {'equal' if loop.get('bytes_equal') else 'DIFFERENT'}; "
                         f"x.npy {x_sha[:16]} archived (grant_fixture/{grant_id}/store)")
        except (GrantError, OSError, KeyError) as error:
            grader.check(name, label, False, f"{grant_id}: {error}")
    return measured, waveforms


# ------------------------------------------------------------ reference pair
def analytic_grids(np, episode):
    """Analytic direct and scattered H at every coordinate, with the exact geometry of each symbol time."""
    manifest, grant = episode.manifest, episode.grant
    offsets = np.array([episode.offset_hz(k) for k in range(grant.grid_subcarriers)], dtype=np.float64)
    shape = (episode.h_shape[0], episode.h_shape[1], episode.h_shape[3], episode.h_shape[4])
    direct, echo = np.zeros(shape, dtype=np.complex128), np.zeros(shape, dtype=np.complex128)
    for opportunity in range(shape[0]):
        for symbol in range(shape[2]):
            paths = episodes.free_space_paths(manifest["geometry"], manifest["target"],
                                              episode.time_s(opportunity, symbol), grant.carrier_hz, True)
            for grid, (amplitude, delay, phases) in zip((direct, echo), paths):
                tones = amplitude * np.exp(-2j * np.pi * (grant.carrier_hz + offsets) * delay)
                grid[opportunity, :, symbol, :] = np.exp(1j * np.array(phases))[:, None] * tones[None, :]
    return direct, echo


def delay_doppler(np, episode, h_b, w):
    """Delay-Doppler map of the pilot LS on the first DM-RS symbol of each period's first uplink slot."""
    schedule = episode.schedule
    picks = [item for item in schedule.opportunities if item.absolute_slot % len(schedule.pattern) == schedule.ul_slots[0]]
    symbol = episode.grant.dmrs["symbols"][0]
    rows = []
    for item in picks:
        pilots = [(index, k) for index, (s, k) in enumerate(item.grant.pilot_index) if s == symbol]
        columns = [k for _, k in pilots]
        reference = np.array([item.grant.pilot_reference[0][index] for index, _ in pilots])
        received = h_b[item.index][:, symbol, columns] * reference[None, :] + w[item.index][:, symbol, columns]
        rows.append(received * np.conj(reference)[None, :] / np.abs(reference[None, :]) ** 2)
    observed = np.stack(rows)                                  # [slow time, rx, pilot]
    observed = observed - observed.mean(axis=0, keepdims=True)  # static removal
    cube = np.fft.fftshift(np.fft.fft(np.fft.ifft(observed, axis=2), axis=0), axes=0)
    power = np.sum(np.abs(cube) ** 2, axis=1)                   # [Doppler bin, delay bin]
    times = [schedule.opportunities[item.index].symbol_time_tc(symbol) / TC_PER_SECOND for item in picks]
    comb = columns[1] - columns[0]
    delay_bin_ns = 1e9 / (len(columns) * comb * episode.grant.scs_khz * 1e3)
    doppler_bin_hz = 1.0 / (len(times) * (times[1] - times[0]))
    row, column = np.unravel_index(int(np.argmax(power)), power.shape)
    peak = float(power[row, column])

    def sidelobe(cut, index):
        rest = [float(value) for position, value in enumerate(cut)
                if min(abs(position - index), len(cut) - abs(position - index)) > 1]
        return 10 * math.log10(max(rest) / peak)

    return {"slow_time_samples": len(times), "uniform_spacing_ms": 1e3 * (times[1] - times[0]),
            "pilots_per_symbol": len(columns), "occupied_bandwidth_mhz": len(columns) * comb * episode.grant.scs_khz / 1e3,
            "delay_resolution_ns": delay_bin_ns, "doppler_resolution_hz": doppler_bin_hz,
            "unambiguous_delay_us": 1e-3 * delay_bin_ns * len(columns),
            "unambiguous_doppler_hz": [-doppler_bin_hz * len(times) / 2, doppler_bin_hz * len(times) / 2],
            "peak_delay_bin": int(column), "peak_delay_ns": float(column * delay_bin_ns),
            "peak_doppler_bin": int(row - len(times) // 2),
            "peak_doppler_hz": float((row - len(times) // 2) * doppler_bin_hz),
            "peak_to_median_db": 10 * math.log10(peak / float(np.median(power))),
            "delay_sidelobe_db": sidelobe(power[row, :], column), "doppler_sidelobe_db": sidelobe(power[:, column], row)}


def refusals(np, entry, scratch):
    """Damage four copies of a stored pair. The loader must refuse each one."""
    def flipped(copy):
        data = bytearray((copy / "b_h.npy").read_bytes())
        data[len(data) // 2] ^= 0x01
        (copy / "b_h.npy").write_bytes(bytes(data))

    def missing(copy):
        (copy / "w.npy").unlink()

    def truncated(copy):
        data = (copy / "a_h.npy").read_bytes()
        (copy / "a_h.npy").write_bytes(data[:-8])

    def swapped(copy):
        array = np.load(copy / "a_h.npy", allow_pickle=False)
        with (copy / "a_h.npy").open("wb") as stream:
            np.save(stream, np.ascontiguousarray(array.swapaxes(1, 3)), allow_pickle=False)
        manifest = json.loads((copy / "manifest.json").read_text())
        manifest["files"]["a_h.npy"].update(sha256=digest(copy / "a_h.npy"), bytes=(copy / "a_h.npy").stat().st_size)
        write_json(copy / "manifest.json", manifest)

    results = {}
    for label, damage in (("one changed byte", flipped), ("a missing array", missing), ("a partial write", truncated),
                          ("a swapped axis", swapped)):
        copy = scratch / label.replace(" ", "-")
        shutil.copytree(entry, copy)
        damage(copy)
        try:
            episodes.open_episode(copy)
            results[label] = None
        except EpisodeError as error:
            results[label] = str(error)
        shutil.rmtree(copy)
    return results


def run_pair(config, out, grader, np, waveforms):
    name, tolerance, expect = "reference-pair", config["tolerance"], config["pair"]["expect"]
    case_dir = out / "reference_pair"
    case_dir.mkdir()
    entry = case_dir / "store"
    try:
        manifest = rt_channel.export_pair(ROOT / config["pair"]["config"], entry, config["variant"], waveforms)
        episode = episodes.open_episode(entry)
    except Exception as error:  # a generator that cannot run is a failed check, with its cause on file
        (case_dir / "export-error.txt").write_text(traceback.format_exc())
        grader.check(name, "export", False, f"{type(error).__name__}: {error} (reference_pair/export-error.txt)")
        return {}
    absent, present = (case["measured"] for case in manifest["cases"])
    bound = sum(1 for item in manifest["grants"].values() if item["x"])
    grader.check(name, "export", present["scattered_paths"][0] >= 1 and absent["paths"] >= 1,
                 f"PathSolver, RCSSolver and Paths.concat on {manifest['solver']['mitsuba_variant']}: {absent['paths']} "
                 f"path without the target, {present['background_paths'][1]} background and "
                 f"{present['scattered_paths'][0]} scattered with it, traced at {present['snapshots']} slot starts in "
                 f"{present['solve_seconds']['total']:.2f} s (reference_pair/store)")
    grader.check(name, "stored pair", episode.h_shape == [expect["opportunities"], len(episode.grant.rx_ports), 1, 14,
                                                           episode.grant.grid_subcarriers]
                 and manifest["grant"] == sets.load_pair_config(ROOT / config["pair"]["config"])[1].record()
                 and bound == len(manifest["grants"]),
                 f"read back with the standard library: a_h.npy, b_h.npy and w.npy verified, H is complex64 "
                 f"{episode.h_shape}, the opportunities, symbol times and subcarrier offsets equal the grant "
                 f"schedule, {bound} of {len(manifest['grants'])} grants bound to an archived transmit grid")
    # Recorded, not graded: whether this host reproduces the files of the development-host rehearsal.
    earlier = expect.get("rehearsal_sha256", {})
    if earlier:
        same = ", ".join(f"{name} {'identical' if earlier.get(name) == entry['sha256'] else 'different'}"
                         for name, entry in sorted(manifest["files"].items()))
        grader.note(name, f"against the x86_64 rehearsal (not graded): {same}")
    uniform = present["uniform_grid_check"] or {}
    solver_error = max(absent["solver_cfr_max_relative_error"], present["solver_cfr_max_relative_error"])
    grader.check(name, "solver agreement",
                 solver_error <= tolerance["solver_cfr_relative"]
                 and uniform.get("max_relative_error", math.inf) <= tolerance["solver_cfr_relative"],
                 f"path phases evaluated at the real symbol times differ from Paths.cfr by {solver_error:.1e} at the "
                 f"snapshot and by {uniform.get('max_relative_error', math.nan):.1e} on its uniform grid of "
                 f"{uniform.get('steps')} steps through the DM-RS symbol times (relative to the RMS of H)")

    h_a = np.load(entry / "a_h.npy", allow_pickle=False)[:, :, 0].astype(np.complex128)
    h_b = np.load(entry / "b_h.npy", allow_pickle=False)[:, :, 0].astype(np.complex128)
    w = np.load(entry / "w.npy", allow_pickle=False).astype(np.complex128)
    direct, echo_model = analytic_grids(np, episode)
    a0, as_ = float(np.abs(direct).max()), float(np.abs(echo_model).max())
    direct_error = worst(np, h_a, direct, a0)
    grader.check(name, "direct path", np.isfinite(h_a).all() and np.all(h_a != 0)
                 and direct_error <= tolerance["direct_relative"],
                 f"target absent: all {h_a.size} coefficients finite and nonzero, within {direct_error:.1e} of the "
                 f"free-space path (gain {a0:.4e}, delay {absent['direct_path']['delay_ns']:.3f} ns) at every "
                 "opportunity, port, symbol and subcarrier")
    echo = h_b - h_a
    echo_error = worst(np, echo, echo_model, as_)
    phase_error = float(np.max(np.abs(np.angle(echo * np.conj(echo_model)))))
    grader.check(name, "scattered path", echo_error <= tolerance["echo_relative"],
                 f"target present minus absent equals the bistatic radar return of the moving target, computed from "
                 f"the geometry at each symbol time, within {echo_error:.1e} of its amplitude {as_:.4e} at all "
                 f"{echo.size} coordinates (largest phase error {phase_error:.1e} rad, receive order included)")

    # ---- Phase of the return at two unequal intervals and across one period. A path whose delay changes
    # at the rate -nu / f_c advances by 2 pi nu (1 + f / f_c) dt at the subcarrier offset f. ----
    truth = [item["bistatic"] for item in manifest["cases"][1]["truth"]]
    symbols = episode.grant.dmrs["symbols"]
    pilots = [k for s, k in episode.grant.pilot_index if s == symbols[0]]
    scale = 1 + np.array([episode.offset_hz(k) for k in pilots]) / episode.grant.carrier_hz
    per_period = len(episode.schedule.ul_slots)
    steps, largest = [], 0.0
    for label, first, second in (("within a slot", (0, symbols[0]), (0, symbols[-1])),
                                 ("to the next slot", (0, symbols[-1]), (1, symbols[0])),
                                 ("across one period", (0, symbols[0]), (per_period, symbols[0]))):
        interval = episode.time_s(*second) - episode.time_s(*first)
        measured = np.angle(echo[second[0]][:, second[1], pilots] * np.conj(echo[first[0]][:, first[1], pilots]))
        wanted = 2 * math.pi * 0.5 * (truth[first[0]]["doppler_hz"] + truth[second[0]]["doppler_hz"]) * interval
        error = float(np.max(np.abs(np.angle(np.exp(1j * (measured - wanted * scale[None, :]))))))
        largest = max(largest, error)
        steps.append({"interval": label, "seconds": interval, "expected_rad": wanted,
                      "measured_rad": float(np.mean(measured)), "max_error_rad": error})
    doppler = truth[0]["doppler_hz"]
    grader.check(name, "phase at unequal intervals",
                 largest <= tolerance["phase_rad"] and doppler < 0 and all(step["measured_rad"] < 0 for step in steps),
                 "the return's phase advances by 2 pi nu (1 + f / f_c) dt: " + ", ".join(
                     f"{1e6 * step['seconds']:.2f} us {step['measured_rad']:+.5f} rad (at the carrier "
                     f"{step['expected_rad']:+.5f})" for step in steps)
                 + f"; largest error {largest:.1e} rad on every port and pilot; nu = {doppler:+.2f} Hz is negative for "
                   "the receding target")

    view = delay_doppler(np, episode, h_b, w)
    delay_bins = truth[0]["delay_ns"] / view["delay_resolution_ns"]
    doppler_bins = doppler / view["doppler_resolution_hz"]
    grader.check(name, "delay-Doppler on the pilot mask",
                 abs(view["peak_delay_bin"] - delay_bins) < 1 and abs(view["peak_doppler_bin"] - doppler_bins) < 1
                 and view["peak_doppler_bin"] < 0 and view["peak_to_median_db"] >= tolerance["peak_snr_db_min"]
                 and abs(view["occupied_bandwidth_mhz"] - expect["occupied_bandwidth_mhz"]) < 1e-9,
                 f"pilot LS of Y = H X + W on symbol {symbols[0]} of {view['slow_time_samples']} slots "
                 f"{view['uniform_spacing_ms']:.3f} ms apart, static part removed: peak at {view['peak_delay_ns']:.1f} ns "
                 f"and {view['peak_doppler_hz']:+.1f} Hz, within one bin of {truth[0]['delay_ns']:.1f} ns and "
                 f"{doppler:+.1f} Hz, {view['peak_to_median_db']:.1f} dB above the median of the map")
    grader.note(name, f"resolution {view['delay_resolution_ns']:.1f} ns over {view['occupied_bandwidth_mhz']:.2f} MHz "
                      f"and {view['doppler_resolution_hz']:.1f} Hz over {view['slow_time_samples']} slots; unambiguous "
                      f"{view['unambiguous_delay_us']:.2f} us and {view['unambiguous_doppler_hz'][0]:+.0f} to "
                      f"{view['unambiguous_doppler_hz'][1]:+.0f} Hz; sidelobes {view['delay_sidelobe_db']:.1f} dB in "
                      f"delay, {view['doppler_sidelobe_db']:.1f} dB in Doppler (rectangular windows)")

    ratio = present["scattered_to_direct_db"]
    expected_ratio = 20 * math.log10(as_ / a0)
    noise = manifest["noise"]
    grader.check(name, "power margins", abs(ratio["median"] - expected_ratio) <= tolerance["ratio_db"]
                 and abs(expected_ratio - expect["scattered_to_direct_db"]) <= tolerance["ratio_db"],
                 f"scattered to direct power per resource element {ratio['median']:.2f} dB (radar equation "
                 f"{expected_ratio:.2f} dB), {ratio['median_margin_to_rounding_rms_db']:.1f} dB above the RMS rounding "
                 f"error of the receiver grid and {ratio['median_margin_to_rounding_bound_db']:.1f} dB above its bound; "
                 f"noise at {-noise['snr_db']:.0f} dB, sigma {noise['sigma']:.4e}, covers the rounding")
    validity = present["path_validity"]
    grader.check(name, "time evolution", validity["max_relative_error"] is not None
                 and validity["max_relative_error"] <= tolerance["path_validity_relative"],
                 f"geometry traced again at every slot start; extrapolating the target's paths by their Doppler shift "
                 f"to the next opportunity ({', '.join(str(gap) for gap in validity['gaps_ms'])} ms ahead) differs "
                 f"from the paths traced there by at most {validity['max_relative_error']:.1e}; "
                 f"{1e3 * present['solve_seconds']['mean']:.1f} ms per snapshot")

    refused = refusals(np, entry, case_dir)
    grader.check(name, "refusals", all(refused.values()),
                 "the loader refuses " + ", ".join(label for label, reason in refused.items() if reason)
                 + ("" if all(refused.values()) else "; ACCEPTED: "
                    + ", ".join(label for label, reason in refused.items() if not reason)))
    reader = subprocess.run([sys.executable, "-m", "ocudu_cuda_isac.episodes.episode", "show", str(entry),
                             "--rx", str(len(episode.grant.rx_ports) - 1)], capture_output=True, text=True, check=False,
                            env=dict(os.environ, PYTHONPATH=str(ROOT / "src")))
    try:
        shown = json.loads(reader.stdout)
        write_json(case_dir / "second-reader.json", shown)
        coefficient, advance = shown["coefficient"], shown["phase_advance"]
        ok = reader.returncode == 0 and coefficient["is_pilot"] \
            and coefficient["analytic_relative_error"] <= tolerance["direct_relative"] + tolerance["echo_relative"] * as_ / a0 \
            and all(step["error_rad"] <= tolerance["phase_rad"] for step in advance)
        grader.check(name, "second reader", ok,
                     f"a separate process recovers H at opportunity {coefficient['opportunity']}, port "
                     f"{coefficient['rx_port']}, symbol {coefficient['symbol']}, subcarrier {coefficient['subcarrier']} "
                     f"({coefficient['offset_hz'] / 1e6:+.2f} MHz, {1e3 * coefficient['time_s']:.6f} ms) = "
                     f"{coefficient['value'][0]:+.6e}{coefficient['value'][1]:+.6e}j, {coefficient['analytic_relative_error']:.1e} "
                     f"from the analytic value, and the return's phase steps {advance[0]['measured_rad']:+.5f} and "
                     f"{advance[1]['measured_rad']:+.5f} rad (reference_pair/second-reader.json)")
    except (ValueError, KeyError) as error:
        grader.check(name, "second reader", False, f"exit status {reader.returncode}: {error} {reader.stderr[-300:]}")
    return {"measured": {"a": absent, "b": {key: value for key, value in present.items()}},
            "direct_relative_error": direct_error, "echo_relative_error": echo_error, "echo_phase_error_rad": phase_error,
            "phase_steps": steps, "delay_doppler": view, "refusals": refused,
            "sha256": {key: value["sha256"] for key, value in manifest["files"].items()},
            "calibration": manifest["calibration"], "noise": noise}


# --------------------------------------------------------------- episode set
def run_set(config, out, grader, np):
    name, tolerance, expect = "episode-set", config["tolerance"], config["set"]["expect"]
    case_dir = out / "episode_set"
    case_dir.mkdir()
    try:
        rt_channel.export_set(ROOT / config["set"]["config"], case_dir / "store", config["variant"])
        stored = sets.open_set(case_dir / "store")
    except Exception as error:
        (case_dir / "export-error.txt").write_text(traceback.format_exc())
        grader.check(name, "export", False, f"{type(error).__name__}: {error} (episode_set/export-error.txt)")
        return {}
    manifest = stored.manifest
    counts, entries = manifest["counts"], stored.episodes()
    target = manifest["config"]["target"]["model"]
    grader.check(name, "export", counts["episodes"] == expect["episodes"] and target == expect["target"],
                 f"{counts['episodes']} matched pairs of set {manifest['set_id']} with the {target} target in the "
                 f"{manifest['config']['scene']['kind']} scene, {counts['opportunities']} opportunities; the set and every "
                 "episode verified on reading (episode_set/store)")
    owners = {}
    for row in manifest["episodes"]:
        owners.setdefault(row["trajectory"], set()).add(row["group"])
    by_group = ", ".join(f"{group} {counts['by_group'][group]}" for group in sets.GROUPS)
    grader.check(name, "groups", counts["by_group"] == expect["by_group"] and all(len(v) == 1 for v in owners.values())
                 and counts["target_present_runs"] == counts["target_absent_runs"]
                 and all(row["episode_id"].startswith("ep-") for row in manifest["episodes"]),
                 f"whole trajectories assigned before tracing: {by_group} "
                 f"episodes; {counts['target_present_runs']} target-present and {counts['target_absent_runs']} "
                 f"target-absent runs; assignment {manifest['assignment_sha256'][:16]}; opaque episode identifiers")
    coverage = manifest["coverage"]
    grid = manifest["position_grid"]
    labelled = sum(len(entry.manifest["cases"][1]["truth"]) for entry in entries)
    grader.check(name, "position labels", coverage["training_fraction"] >= coverage["minimum_training_fraction"]
                 and labelled == counts["opportunities"],
                 f"{labelled} target positions labelled on the frozen {grid['rows']} by {grid['cols']} grid "
                 f"({grid['cell_m'][0]:.3f} by {grid['cell_m'][1]:.3f} m cells); training covers "
                 f"{coverage['by_group']['training']} of {coverage['cells']} cells, at least "
                 f"{coverage['minimum_training_fraction']:.4f} required")
    worst_direct, worst_solver, fewest, ratios = 0.0, 0.0, None, []
    for entry in entries:
        absent, present = (case["measured"] for case in entry.manifest["cases"])
        amplitude, delay, phases = episodes.free_space_paths(entry.manifest["geometry"], entry.manifest["target"], 0.0,
                                                             entry.grant.carrier_hz, False)[0]
        got = absent["direct_path"]
        errors = [abs(got["delay_ns"] - 1e9 * delay) / (1e9 * delay)]
        errors += [abs(value - amplitude) / amplitude for value in got["amplitude"]]
        errors += [abs(math.remainder(value - wanted, 2 * math.pi)) for value, wanted in zip(got["phase_rad"], phases)]
        worst_direct = max(worst_direct, *errors)
        worst_solver = max(worst_solver, absent["solver_cfr_max_relative_error"], present["solver_cfr_max_relative_error"])
        fewest = present["scattered_paths"][0] if fewest is None else min(fewest, present["scattered_paths"][0])
        ratios.append(present["scattered_to_direct_db"]["median"])
    grader.check(name, "direct path and array", worst_direct <= 10 * tolerance["direct_relative"],
                 f"the shortest path of every target-absent case has the free-space delay, gain and per-element phase "
                 f"of the recorded array geometry within {worst_direct:.1e}")
    grader.check(name, "scattering", fewest is not None and fewest >= 1 and worst_solver <= tolerance["solver_cfr_relative"],
                 f"at least {fewest} scattered path(s) at every opportunity of every target-present case; Paths.cfr "
                 f"agreement {worst_solver:.1e}; scattered to background power per resource element "
                 f"{min(ratios):.1f} to {max(ratios):.1f} dB (episode medians)")
    return {"counts": counts, "coverage": coverage, "assignment_sha256": manifest["assignment_sha256"],
            "direct_error": worst_direct, "solver_error": worst_solver, "ratio_medians_db": ratios}


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True,
                              timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="new directory for the run")
    parser.add_argument("--grant-store", required=True, type=Path,
                        help="grant_fixture directory of the accepted Sprint 2.1 run, which holds <grant_id>/store")
    parser.add_argument("--adapter-build", type=Path, help="adapter build directory (default: from the configuration)")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.exit(2, f"episode_pair: {args.output_dir} already exists. Evidence is never overwritten.\n")
    try:
        config = load_config(args.config)
        receipt = native_adapter.verified_receipt(args.adapter_build or ROOT / config["adapter_build"])
        import numpy as np
    except (ConfigError, native_adapter.AdapterError, ImportError) as error:
        parser.exit(2, f"episode_pair: {error}\n")
    adapter = Path(receipt["executables"]["grant_fixture"]["path"])
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = args.output_dir.resolve()
    out.mkdir(parents=True)
    write_json(out / "adapter-receipt.json", receipt)

    grader, measured, waveforms = Grade(), {}, {}

    def guarded(case, action):
        before = len(grader.results)
        try:
            return action()
        except Exception as error:  # an unexpected fault fails its case and keeps the remaining cases
            (out / f"driver-error-{case}.txt").write_text(traceback.format_exc())
            grader.check(case, "driver", False, f"{type(error).__name__}: {error} (driver-error-{case}.txt)")
            return {}
        finally:
            if len(grader.results) == before:
                grader.check(case, "driver", False, "the case produced no check")

    measured["diagnostic"] = guarded("diagnostic", lambda: run_diagnostic(config, out, grader, np))
    result = guarded("grant-fixture", lambda: run_grant_fixture(config, out, grader, adapter, receipt,
                                                                args.grant_store.resolve()))
    measured["grant-fixture"], waveforms = result if isinstance(result, tuple) else ({}, {})
    measured["reference-pair"] = guarded("reference-pair", lambda: run_pair(config, out, grader, np, waveforms))
    measured["episode-set"] = guarded("episode-set", lambda: run_set(config, out, grader, np))

    failed = [item for item in grader.results if not item["passed"]]
    passed_cases = [case for case in CASES if any(item["case"] == case for item in grader.results)
                    and all(item["passed"] for item in grader.results if item["case"] == case)]
    grade = {
        "schema": GRADE_SCHEMA, "selector": "episode_pair", "run_id": out.name, "cases": len(CASES),
        "cases_passed": len(passed_cases), "cases_failed": len(CASES) - len(passed_cases),
        "checks": len(grader.results), "checks_passed": len(grader.results) - len(failed), "checks_failed": len(failed),
        "judgment": "pass" if not failed and len(passed_cases) == len(CASES) else "fail",
        "reason": "every check of every case passed" if not failed else
                  f"{len(failed)} check(s) failed, first: {failed[0]['case']} {failed[0]['check']}",
        "tolerance": config["tolerance"], "results": grader.results}
    write_json(out / "grade.json", grade)
    write_json(out / "comparison.json", measured)
    lock = {item["name"]: item["commit"] for item in json.loads((ROOT / "ext/lock.json").read_text())["dependencies"]}
    inputs = [args.config, ROOT / config["pair"]["config"], ROOT / config["set"]["config"]]
    write_json(out / "manifest.json", {
        "run_id": out.name, "selector": "episode_pair",
        "purpose": "Sprint 2.2 integration: the uplink grant fixture of an episode, a matched target-absent and "
                   "target-present pair on it, and a small demonstration episode set",
        "command": shlex.join([sys.executable, *sys.argv]), "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": platform.node(), "machine": platform.machine(), "python": platform.python_version(),
        "parent_commit": git_commit(), "locked_sources": lock,
        "inputs": {str(path): digest(path) for path in inputs if Path(path).is_file()},
        "driver_sha256": digest(__file__), "grant_store": str(args.grant_store.resolve()),
        "adapter": {"executable": str(adapter), "sha256": receipt["executables"]["grant_fixture"]["sha256"],
                    "receipt": "adapter-receipt.json", "ocudu_build": receipt["ocudu_build"]},
        "mitsuba_variant": config["variant"],
        "environment": {key: os.environ[key] for key in ("VIRTUAL_ENV", "CUDA_VISIBLE_DEVICES") if key in os.environ},
        "judgment": grade["judgment"], "reason": grade["reason"]})
    summary = (f"episode_pair: {grade['cases_passed']} of {grade['cases']} cases passed, "
               f"{grade['checks_passed']} of {grade['checks']} checks")
    (out / "grade.txt").write_text("\n".join(grader.lines + [summary]) + "\n", encoding="ascii")
    print(summary)
    print(f"Run artifacts: {out}")
    return 0 if grade["judgment"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
