"""Radar cube, delay-Doppler view and causal features on pilot observations (Sprint 3.3, selector pilot_features).

A sensor reads obs/v1 records in the order of the ring. This module is the reference of what it
makes of them:

  cube      X[frequency, slow time, rx]: the pilot estimates of the reference symbols that a
            specification selects, over the periods of one window. Frequency is the pilot comb of
            the record, slow time the start of the useful part of each selected symbol, rx the
            array elements of the antenna map in rising order. Every element has a validity bit,
            and an element without one is zero and is never read.
  view      the delay-Doppler power of the cube after static removal, summed over rx. It is a
            diagnostic: a known return must peak in the cell its delay and its Doppler shift name.
  features  a short vector of that view, in a fixed order, for the presence head of Sprint 3.4.

Sign conventions, fixed here and tested: a path of delay tau and Doppler shift nu contributes
exp(-j 2 pi f tau) exp(+j 2 pi nu t). A later arrival lies at a higher delay bin. A phase that
advances in slow time, a path that shortens, lies at a positive Doppler bin, and a receding target
at a negative one.

Causality. A Stream takes one record at a time and answers at once, with what it has seen up to
that record: "ok" with the features of a complete window, "insufficient_history" while a window
fills, "gap" for a window that closed without one of its samples, "invalid" for a window whose
valid support is too small, "rejected" for a record that is no input of the specification. A
history never spans two epochs or two episode keys, and nothing of a record is read but the
fields named in INPUT_FIELDS, the validity bits and the payload.

Every step is a registry entry with its execution device. All of them run in NumPy on the CPU
and carry runtime "reference": this module defines the steps and checks the native units of
Sprint 6.12, and no latency it measures is a result. It imports the record reader and the slot
times of the sensing package and nothing of a host, a simulator or an episode.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np

from . import obs
from . import pilot_ls as estimator
from .obs import ObsError

SPEC_SCHEMA = "pilot-features-spec/v1"
LINE_SCHEMA = "pilot-features/v1"
REPORT_SCHEMA = "pilot-features-report/v1"
LINES_FILE = "features.jsonl"
REPORT_FILE = "sensor.json"
VIEWS_DIR = "views"
RECORDS_DIR = "records"
RUNTIME = "reference"
STATUSES = ("ok", "insufficient_history", "gap", "invalid", "rejected")
TC_PER_SECOND = estimator.TC_PER_SECOND
SPEC_FIELDS = ("schema", "spec_id", "input", "radio", "sequence", "window", "steps", "view")
INPUT = ("stage", "source", "layout", "dtype")
RADIO = ("scs_khz", "period_slots")
SEQUENCE = ("slots_in_period", "reference_symbols")
WINDOW = ("periods", "hop_periods", "min_valid_fraction")
STEPS = ("cube", "static", "delay", "doppler", "features")
VIEW = ("delay_bins", "doppler_grid", "floor_db")
# Everything a stream reads of a record header. A field outside this list cannot change a feature.
INPUT_FIELDS = ("epoch", "sequence", "episode", "opportunity", "absolute_slot", "stage", "source", "layout", "dtype",
                "scale", "rank", "shape", "strides", "first_subcarrier", "subcarrier_step", "subcarrier_count",
                "dmrs_symbols", "rx_count", "layer_count", "antenna_map", "mask_bytes", "payload_bytes", "time_unit",
                "evidence", "port", "t_first_sample", "t_last_sample", "t_ready_ns", "t_publish_ns")
# Host times of a run. Two runs on the same records agree on every other field of a line.
VARYING = ("t_ready_ns", "t_publish_ns", "t_features_ns")
LEADING = ("peak_to_median_db", "moving_to_static_db", "peak_delay_ns", "peak_doppler_hz", "doppler_centroid_hz",
           "doppler_spread_hz")
# The steps of this ticket. Each entry states where it runs. Sprint 6.12 adds the native unit of
# every entry under the same name.
_CPU = {"device": "cpu", "library": "numpy", "runtime": RUNTIME}
REGISTRY = {
    "cube": {"comb_cube": dict(_CPU, does="pilot comb by slow time by array element, complex128, with validity bits; "
                                          "an element without a bit is zero")},
    "static": {"none": dict(_CPU, does="nothing is removed"),
               "window_mean": dict(_CPU, does="per frequency and element, the mean over the valid samples of the window "
                                              "is subtracted: the zero-Doppler part of this window and of no other")},
    "delay": {"ifft": dict(_CPU, does="inverse FFT over a regular comb, rectangular taper, a hole is zero")},
    "doppler": {"fft": dict(_CPU, does="FFT over a homogeneous sequence, one sample per period, rectangular taper"),
                "ndft": dict(_CPU, does="discrete Fourier sum at the actual sample times on a declared Doppler grid, "
                                        "rectangular taper")},
    "features": {"dd_compact": dict(_CPU, does="peak over median, moving over static power, peak cell, Doppler centroid "
                                               "and spread, and the strongest cell of every Doppler bin over the median")},
}


class FeaturesError(ValueError):
    """A specification, a record sequence or a sensor output violates the documented interface."""


# ---------------------------------------------------------------- specification
def _fields(value, fields, name):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise FeaturesError(f"{name} requires exactly {', '.join(fields)}")
    return value


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise FeaturesError(f"{name} must be an integer from {low} to {high}")
    return value


def check_spec(spec):
    """Check a pilot-features-spec/v1 object and return it. Every refusal is made before a record is read."""
    _fields(spec, SPEC_FIELDS, "a feature specification")
    if spec["schema"] != SPEC_SCHEMA:
        raise FeaturesError(f"unsupported specification schema {spec['schema']!r}")
    if not isinstance(spec["spec_id"], str) or not spec["spec_id"] or not spec["spec_id"].isascii():
        raise FeaturesError("spec_id must be a nonempty ASCII string")
    wanted = _fields(spec["input"], INPUT, "input")
    if wanted["stage"] not in obs.STAGES.values() or wanted["source"] not in obs.SOURCES.values():
        raise FeaturesError(f"unknown receiver stage {wanted['stage']!r} or sensing source {wanted['source']!r}")
    if (wanted["layout"], wanted["dtype"]) != ("ocudu_o2_comb_c64", "<c8"):
        raise FeaturesError(f"no cube is defined here for the layout {wanted['layout']!r} with the dtype "
                            f"{wanted['dtype']!r}: this ticket reads ocudu_o2_comb_c64 as <c8")
    radio = _fields(spec["radio"], RADIO, "radio")
    try:
        estimator.numerology(radio["scs_khz"])
    except estimator.PilotLsError as error:
        raise FeaturesError(str(error)) from None
    period = _integer(radio["period_slots"], "radio.period_slots", 1, 1 << 16)
    if period % (1 << estimator.numerology(radio["scs_khz"])):
        raise FeaturesError("radio.period_slots must be a whole number of subframes, so that a period has one length")
    sequence = _fields(spec["sequence"], SEQUENCE, "sequence")
    for name, high in (("slots_in_period", period - 1), ("reference_symbols", 13)):
        values = sequence[name]
        if not isinstance(values, list) or not values or values != sorted(set(values)):
            raise FeaturesError(f"sequence.{name} must be a nonempty rising list")
        for value in values:
            _integer(value, f"an entry of sequence.{name}", 0, high)
    window = _fields(spec["window"], WINDOW, "window")
    periods = _integer(window["periods"], "window.periods", 1, 4096)
    _integer(window["hop_periods"], "window.hop_periods", 1, 4096)
    share = window["min_valid_fraction"]
    if type(share) not in (int, float) or not 0 < share <= 1:
        raise FeaturesError("window.min_valid_fraction must lie above 0 and at or below 1")
    steps = _fields(spec["steps"], STEPS, "steps")
    for stage in STEPS:
        if steps[stage] not in REGISTRY[stage]:
            raise FeaturesError(f"unknown {stage} step {steps[stage]!r} (registered: {', '.join(sorted(REGISTRY[stage]))})")
    view = _fields(spec["view"], VIEW, "view")
    _integer(view["delay_bins"], "view.delay_bins", 1, 3300)
    if type(view["floor_db"]) not in (int, float) or not 20 <= view["floor_db"] <= 300:
        raise FeaturesError("view.floor_db must lie from 20 to 300")
    samples = periods * len(sequence["slots_in_period"]) * len(sequence["reference_symbols"])
    grid = view["doppler_grid"]
    if steps["doppler"] == "fft":
        if len(sequence["slots_in_period"]) != 1 or len(sequence["reference_symbols"]) != 1:
            raise FeaturesError("the fft Doppler step needs a homogeneous sequence: one slot of the period and one "
                                "reference symbol. The symbols of a slot and the slots of a burst are not on one grid: "
                                "use ndft with their actual times")
        if grid is not None:
            raise FeaturesError("the fft Doppler step has the grid of its window: view.doppler_grid must be null")
        if samples < 2:
            raise FeaturesError("a Doppler transform needs two samples")
    else:
        _fields(grid, ("bins", "step_hz"), "view.doppler_grid of the ndft step")
        _integer(grid["bins"], "view.doppler_grid.bins", 2, 4096)
        if type(grid["step_hz"]) not in (int, float) or not grid["step_hz"] > 0 or not math.isfinite(grid["step_hz"]):
            raise FeaturesError("view.doppler_grid.step_hz must be positive")
        if samples < 2:
            raise FeaturesError("a Doppler transform needs two samples")
    return spec


def spec_sha256(spec):
    """The SHA-256 of a specification in its canonical form. Taken once, when a stream is bound."""
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()


def demo_spec(spec_id="dd-uniform-8", periods=8, hop_periods=8, slot=7, delay_bins=32):
    """The delay-Doppler branch of the demonstration: symbol row 0 of one uplink slot per 5 ms period."""
    return check_spec({
        "schema": SPEC_SCHEMA, "spec_id": spec_id,
        "input": {"stage": "O2", "source": "pusch_dmrs", "layout": "ocudu_o2_comb_c64", "dtype": "<c8"},
        "radio": {"scs_khz": 30, "period_slots": 10},
        "sequence": {"slots_in_period": [slot], "reference_symbols": [0]},
        "window": {"periods": periods, "hop_periods": hop_periods, "min_valid_fraction": 0.9},
        "steps": {"cube": "comb_cube", "static": "window_mean", "delay": "ifft", "doppler": "fft",
                  "features": "dd_compact"},
        "view": {"delay_bins": delay_bins, "doppler_grid": None, "floor_db": 120.0}})


def load_spec(path):
    try:
        return check_spec(json.loads(Path(path).read_text(encoding="ascii")))
    except (OSError, ValueError) as error:
        if isinstance(error, FeaturesError):
            raise
        raise FeaturesError(f"cannot read the specification: {error}") from None


def resolve(spec):
    """The registry entries a specification selects, by stage."""
    return {stage: dict(REGISTRY[stage][spec["steps"][stage]], entry=spec["steps"][stage]) for stage in STEPS}


# ---------------------------------------------------------------- time
def symbol_time_tc(scs_khz, absolute_slot, symbol):
    """Start of the useful part of one symbol in T_c from slot 0 of the epoch (TS 38.211 5.3.1)."""
    if type(symbol) is not int or not 0 <= symbol < estimator.SYMBOLS_PER_SLOT:
        raise FeaturesError("a symbol of a slot lies from 0 to 13")
    mu = estimator.numerology(scs_khz)
    useful, prefix, boundary = (2048 * estimator.KAPPA) >> mu, (144 * estimator.KAPPA) >> mu, 7 << mu
    first = (absolute_slot % (1 << mu)) * estimator.SYMBOLS_PER_SLOT
    start = estimator.slot_start_tc(scs_khz, absolute_slot)
    for index in range(first, first + symbol):
        start += useful + prefix + (16 * estimator.KAPPA if index % boundary == 0 else 0)
    return start + prefix + (16 * estimator.KAPPA if (first + symbol) % boundary == 0 else 0)


def period_tc(scs_khz, period_slots):
    """Length of one period in T_c. A period is a whole number of subframes."""
    return (period_slots >> estimator.numerology(scs_khz)) * (TC_PER_SECOND // 1000)


# ---------------------------------------------------------------- record
def planes(record):
    """The payload and the validity bits of an ocudu_o2_comb_c64 record: two arrays [rx_plane][symbol][pilot]."""
    shape = tuple(record.shape)
    elements = int(np.prod(shape))
    if len(record.payload) != 8 * elements or len(record.mask) != (elements + 7) // 8:
        raise FeaturesError("the payload or the validity bits do not have the size of the shape")
    values = np.frombuffer(record.payload, dtype="<c8").reshape(shape)
    bits = np.unpackbits(np.frombuffer(record.mask, dtype=np.uint8), bitorder="little")[:elements]
    return values, bits.reshape(shape).astype(bool)


def refusal(record, spec):
    """Why a record is no input of a specification, or None. Nothing of the payload is read."""
    header, wanted = record.header, spec["input"]
    named = (obs.STAGES.get(header["stage"]), obs.SOURCES.get(header["source"]), obs.LAYOUTS.get(header["layout"]),
             obs.DTYPES.get(header["dtype"]))
    if named != tuple(wanted[name] for name in INPUT):
        return f"the record is {named[0]} of {named[1]} in the layout {named[2]} as {named[3]}, and the " \
               f"specification reads {wanted['stage']} of {wanted['source']} in the layout {wanted['layout']}"
    shape, rx = record.shape, header["rx_count"]
    symbols = [symbol for symbol in range(estimator.SYMBOLS_PER_SLOT) if header["dmrs_symbols"] >> symbol & 1]
    if not symbols or header["rank"] != 3 or shape != [rx, len(symbols), header["subcarrier_count"]] \
            or record.strides != [shape[1] * shape[2], shape[2], 1] or header["layer_count"] != 1 \
            or not 1 <= rx <= obs.MAX_PLANES or len(set(header["antenna_map"][:rx])) != rx:
        return "the shape is not [rx_plane][reference_symbol][pilot] of one layer with one array element per plane"
    elements = shape[0] * shape[1] * shape[2]
    if header["payload_bytes"] != 8 * elements or header["mask_bytes"] != (elements + 7) // 8:
        return "the payload or the validity bits do not have the size of the shape"
    if header["subcarrier_step"] < 1:
        return "the pilot axis is not a regular comb: no delay transform is defined here for it"
    if spec["sequence"]["reference_symbols"][-1] >= len(symbols):
        return f"the record holds {len(symbols)} reference symbols, and the specification selects row " \
               f"{spec['sequence']['reference_symbols'][-1]}"
    scs, slot = spec["radio"]["scs_khz"], header["absolute_slot"]
    if obs.TIME_UNITS.get(header["time_unit"]) != "tc" \
            or header["t_first_sample"] != symbol_time_tc(scs, slot, symbols[0]) \
            or header["t_last_sample"] != symbol_time_tc(scs, slot, symbols[-1]) + estimator.useful_tc(scs):
        return "the sample times of the record are not those of its slot and its reference symbols at the subcarrier " \
               "spacing of the specification"
    return None


# ---------------------------------------------------------------- arithmetic
def axes_of(spec, lattice, times_tc):
    """The signed axes of a window, with their resolution and their alias limits.

    lattice is (first_subcarrier, subcarrier_step, subcarrier_count, elements) of the contributing
    records, times_tc the start of each slow-time sample. Returns None for the fft step on times
    that are not equally spaced.
    """
    first, step, count, elements = lattice
    spacing_hz = step * spec["radio"]["scs_khz"] * 1000.0
    bins = min(spec["view"]["delay_bins"], count)
    resolution_ns = 1e9 / (count * spacing_hz)
    samples = len(times_tc)
    if spec["steps"]["doppler"] == "fft":
        gaps = {b - a for a, b in zip(times_tc, times_tc[1:])}
        if len(gaps) != 1:
            return None
        interval_s = gaps.pop() / TC_PER_SECOND
        doppler_bins = [index - samples // 2 for index in range(samples)]
        resolution_hz = 1.0 / (samples * interval_s)
        doppler = {"bins": doppler_bins, "hz": [index * resolution_hz for index in doppler_bins],
                   "resolution_hz": resolution_hz, "sample_interval_s": interval_s,
                   "unambiguous_hz": [doppler_bins[0] * resolution_hz, doppler_bins[-1] * resolution_hz]}
    else:
        grid = spec["view"]["doppler_grid"]
        doppler_bins = [index - grid["bins"] // 2 for index in range(grid["bins"])]
        span_s = (times_tc[-1] - times_tc[0]) / TC_PER_SECOND
        doppler = {"bins": doppler_bins, "hz": [index * grid["step_hz"] for index in doppler_bins],
                   "resolution_hz": 1.0 / span_s if span_s > 0 else None, "sample_interval_s": None,
                   "unambiguous_hz": [doppler_bins[0] * grid["step_hz"], doppler_bins[-1] * grid["step_hz"]]}
    return {
        "order": ["frequency", "slow_time", "rx"],
        "frequency": {"first_subcarrier": first, "subcarrier_step": step, "count": count, "spacing_hz": spacing_hz,
                      "span_hz": count * spacing_hz, "unit": "Hz from subcarrier 0 of the carrier grid"},
        "slow_time": {"samples": samples, "unit": "s from the first sample of the window",
                      "s": [(value - times_tc[0]) / TC_PER_SECOND for value in times_tc]},
        "rx": {"elements": list(elements), "unit": "array element of the antenna map, rising"},
        "delay": {"bins": bins, "resolution_ns": resolution_ns, "unambiguous_ns": 1e9 / spacing_hz,
                  "kept_ns": bins * resolution_ns, "sign": "a later arrival lies at a higher bin",
                  "taper": "rectangular"},
        "doppler": dict(doppler, sign="a phase that advances in slow time lies at a positive bin: an approaching "
                                      "target is positive, a receding one negative", taper="rectangular"),
        "phase": "exp(-j 2 pi f tau) exp(+j 2 pi nu t)"}


def view_of(spec, cube, valid, axes):
    """The delay-Doppler power of one cube and what is measured beside it.

    cube and valid are [frequency, slow time, rx]. Only valid elements are read. Returns the power
    [delay][doppler] summed over rx as float64, and a dict of the scalars the features use.
    """
    cube = np.where(valid, cube, 0)
    counts = valid.sum(axis=1)
    if spec["steps"]["static"] == "window_mean":
        mean = cube.sum(axis=1) / np.maximum(counts, 1)
        moving = np.where(valid, cube - mean[:, None, :], 0)
    else:
        mean = np.zeros(cube.shape[::2], dtype=np.complex128)
        moving = cube
    static_power = float((counts * np.abs(mean) ** 2).sum())
    moving_power = float((np.abs(moving) ** 2).sum())
    bins = axes["delay"]["bins"]
    profile = np.fft.ifft(moving, axis=0)[:bins]
    samples = cube.shape[1]
    if spec["steps"]["doppler"] == "fft":
        spectrum = np.fft.fftshift(np.fft.fft(profile, axis=1), axes=1) / samples
    else:
        kernel = np.exp(-2j * np.pi * np.outer(np.array(axes["doppler"]["hz"]), np.array(axes["slow_time"]["s"])))
        spectrum = np.einsum("mnr,dn->mdr", profile, kernel) / samples
    power = (np.abs(spectrum) ** 2).sum(axis=2)
    static_profile = (np.abs(np.fft.ifft(mean, axis=0)[:bins]) ** 2).sum(axis=1)
    return np.ascontiguousarray(power, dtype=np.float64), {
        "static_power": static_power, "moving_power": moving_power,
        "static_peak_bin": int(np.argmax(static_profile)) if static_power > 0 else None}


def feature_names(axes):
    """The order of the vector for a window with these axes."""
    return list(LEADING) + [f"profile_db@{value:+g}Hz" for value in axes["doppler"]["hz"] if value != 0]


def features_of(spec, power, scalars, axes):
    """The compact vector of one view and the cells it names. Returns (names, values, cells) or a reason.

    The zero-Doppler column holds what static removal left and is excluded from every statistic
    when a static step runs. The median is that of the remaining cells, floored at floor_db under
    the peak, so that a view without noise has finite features: its peak reads floor_db exactly.
    """
    hz = np.array(axes["doppler"]["hz"])
    moving = hz != 0 if spec["steps"]["static"] != "none" else np.ones(len(hz), dtype=bool)
    cells = power[:, moving]
    if not np.isfinite(power).all() or cells.size == 0 or not cells.max() > 0:
        return "the view holds no finite power outside the static column"
    row, column = (int(index) for index in np.unravel_index(int(np.argmax(cells)), cells.shape))
    column = int(np.flatnonzero(moving)[column])
    peak = float(power[row, column])
    floor = peak * 10.0 ** (-spec["view"]["floor_db"] / 10.0)
    median = max(float(np.median(cells)), floor)
    static = max(scalars["static_power"], scalars["moving_power"] * 10.0 ** (-spec["view"]["floor_db"] / 10.0))
    by_bin = cells.sum(axis=0)
    centroid = float((hz[moving] * by_bin).sum() / by_bin.sum())
    spread = float(math.sqrt(max(float((((hz[moving] - centroid) ** 2) * by_bin).sum() / by_bin.sum()), 0.0)))
    resolution = axes["delay"]["resolution_ns"]
    values = [10.0 * math.log10(peak / median), 10.0 * math.log10(scalars["moving_power"] / static),
              row * resolution, float(hz[column]), centroid, spread]
    values += [10.0 * math.log10(max(float(power[:, index].max()), floor) / median)
               for index in np.flatnonzero(hz != 0)]
    if not all(math.isfinite(value) for value in values):
        return "a feature is not finite"
    static_bin = scalars["static_peak_bin"]
    found = {"peak": {"delay_bin": row, "doppler_bin": axes["doppler"]["bins"][column], "delay_ns": row * resolution,
                      "doppler_hz": float(hz[column]), "power": peak},
             "median_power": median,
             "static_peak": None if static_bin is None else {"delay_bin": static_bin, "delay_ns": static_bin * resolution},
             "excess_delay_ns": None if static_bin is None else (row - static_bin) * resolution,
             # Leakage: the share of the moving power of the kept view that lies in the peak cell. One
             # return on a bin of a complete lattice reads 1. A return between bins, a hole and noise
             # lower it.
             "leakage": {"peak_fraction": peak / float(cells.sum())}}
    return feature_names(axes), values, found


# ---------------------------------------------------------------- stream
class Stream:
    """The causal feature stream of one specification. push() takes the records in the order of the ring."""

    def __init__(self, spec, clock=None):
        self.spec = check_spec(spec)
        self.clock = clock  # host time of an answer, read when the answer is complete. It is stored, never used.
        self.sha256 = spec_sha256(spec)
        radio, sequence, window = spec["radio"], spec["sequence"], spec["window"]
        self.scs, self.period_slots = radio["scs_khz"], radio["period_slots"]
        self.slots, self.rows = sequence["slots_in_period"], sequence["reference_symbols"]
        self.periods, self.hop = window["periods"], window["hop_periods"]
        self.need = self.periods * len(self.slots) * len(self.rows)
        self.counts = dict.fromkeys(("records", "passed", "resets") + STATUSES, 0)
        self.views = []
        self.last = None  # the cube of the newest complete window, for a reader of the reference
        self._lines = 0
        self._reset()

    def _reset(self):
        self.key = self.lattice = self.origin = self.due = None
        self.last_slot = -1
        self.samples = {}

    def _expected(self, last_period):
        return [(period, slot, row) for period in range(last_period - self.periods + 1, last_period + 1)
                for slot in self.slots for row in self.rows]

    def _line(self, status, reason, last_period, record, have):
        header = None if record is None else record.header
        self.counts[status] += 1
        self._lines += 1
        window = None
        if last_period is not None:
            epoch, episode = self.key
            window = {"id": f"{epoch}/{episode:016x}/{last_period}", "epoch": epoch, "episode": f"{episode:016x}",
                      "first_period": last_period - self.periods + 1, "last_period": last_period}
        return {"schema": LINE_SCHEMA, "line": self._lines - 1, "status": status, "reason": reason, "window": window,
                "trigger": None if header is None else {"epoch": header["epoch"], "sequence": header["sequence"],
                                                        "opportunity": f"{header['opportunity']:016x}",
                                                        "absolute_slot": header["absolute_slot"]},
                "history": {"have": have, "need": self.need}, "spec_sha256": self.sha256, "runtime": RUNTIME,
                "t_ready_ns": None if header is None else header["t_ready_ns"],
                "t_publish_ns": None if header is None else header["t_publish_ns"], "t_features_ns": None if self.clock is None else self.clock()}

    def _close(self, last_period, record):
        """The outcome of the window that ends with last_period, from the samples held."""
        expected = self._expected(last_period)
        held = [self.samples.get(place) for place in expected]
        have = sum(sample is not None for sample in held)
        if have != len(expected):
            line = self._line("gap", f"{len(expected) - have} of {len(expected)} samples of the window never arrived",
                              last_period, record, have)
            line["missing"] = [{"period": period, "slot_in_period": slot, "reference_symbol": row}
                               for (period, slot, row), sample in zip(expected, held) if sample is None]
            return line
        first, step, count, elements = self.lattice[:4]
        times = [sample["time_tc"] for sample in held]
        axes = axes_of(self.spec, (first, step, count, elements), times)
        cube = np.stack([sample["values"] for sample in held], axis=1)
        valid = np.stack([sample["valid"] for sample in held], axis=1)
        total, good = int(valid.size), int(valid.sum())
        self.last = {"last_period": last_period, "cube": cube, "valid": valid, "axes": axes}
        contributing, seen = [], set()
        for sample in held:
            if sample["sequence"] not in seen:
                seen.add(sample["sequence"])
                contributing.append({"sequence": sample["sequence"], "opportunity": sample["opportunity"],
                                     "absolute_slot": sample["absolute_slot"]})
        facts = {"contributing": contributing,
                 "validity": {"elements": total, "valid": good, "holes": total - good, "fill": good / total,
                              # A line of the cube is one frequency on one element over slow time. The mean
                              # of a line with a hole is taken over its valid samples.
                              "lines_with_holes": int((~valid).any(axis=1).sum()),
                              "interpolation": "none: a hole is zero in both transforms"},
                 "time": {"unit": "tc", "tc_per_second": TC_PER_SECOND, "t_first_sample": times[0],
                          "t_last_sample": times[-1] + estimator.useful_tc(self.scs),
                          "acquisition_us": 1e6 * (times[-1] + estimator.useful_tc(self.scs) - times[0]) / TC_PER_SECOND},
                 "input": dict(self.spec["input"], port=self.lattice[4], scale=self.lattice[5],
                               evidence=sorted({sample["evidence"] for sample in held}))}
        if axes is None:
            line = self._line("invalid", "the samples of the window are not equally spaced, and the fft Doppler step "
                                         "needs a homogeneous sequence", last_period, record, have)
        elif good / total < self.spec["window"]["min_valid_fraction"]:
            line = self._line("invalid", f"{good} of {total} elements of the cube are valid, below the share "
                                         f"{self.spec['window']['min_valid_fraction']} of the specification",
                              last_period, record, have)
        else:
            power, scalars = view_of(self.spec, cube, valid, axes)
            made = features_of(self.spec, power, scalars, axes)
            if isinstance(made, str):
                line = self._line("invalid", made, last_period, record, have)
            else:
                names, values, found = made
                line = self._line("ok", None, last_period, record, have)
                line.update(axes=axes, view=dict(found, entry=f"{self.spec['steps']['delay']}+{self.spec['steps']['doppler']}",
                                                 static=self.spec["steps"]["static"], shape=list(power.shape),
                                                 axes=["delay", "doppler"], dtype="<f8", file=None, sha256=None,
                                                 moving_power=scalars["moving_power"],
                                                 static_power=scalars["static_power"]),
                            features={"entry": self.spec["steps"]["features"], "names": names, "values": values})
                self.views.append((line["line"], power))
        line.update(facts)
        return line

    def push(self, record):
        """Take the next record of the ring. Returns the outcome lines it causes, oldest first.

        A record outside the sequence of the specification causes none. An answer carries the
        time of the stream's clock at which it was complete, after its view and its vector.
        """
        self.counts["records"] += 1
        problem = refusal(record, self.spec)
        if problem is not None:
            return [self._line("rejected", problem, None, record, 0)]
        header = record.header
        rx = header["rx_count"]
        order = sorted(range(rx), key=lambda plane: header["antenna_map"][plane])
        lattice = (header["first_subcarrier"], header["subcarrier_step"], header["subcarrier_count"],
                   tuple(header["antenna_map"][plane] for plane in order), header["port"], header["scale"],
                   header["dmrs_symbols"])
        key, slot = (header["epoch"], header["episode"]), header["absolute_slot"]
        if key != self.key or slot <= self.last_slot:
            # Another epoch, another episode or a time that went back: a new history.
            if self.key is not None:
                self.counts["resets"] += 1
            self._reset()
            self.key = key
        self.last_slot = slot
        period, place = divmod(slot, self.period_slots)
        if place not in self.slots:
            # A slot outside the sequence may carry another allocation. It is not looked at further.
            self.counts["passed"] += 1
            return []
        if self.lattice is not None and lattice != self.lattice:
            # Another allocation inside the sequence: its samples lie on another lattice.
            self.counts["resets"] += 1
            self._reset()
            self.key, self.last_slot = key, slot
        self.lattice = lattice
        lines = []
        while self.due is not None and period > self.due:
            # A window whose last sample never came is closed by the first record after it.
            lines.append(self._close(self.due, None))
            self.due += self.hop
        if self.origin is None:
            self.origin, self.due = period, period + self.periods - 1
        values, valid = planes(record)
        symbols = [symbol for symbol in range(estimator.SYMBOLS_PER_SLOT) if header["dmrs_symbols"] >> symbol & 1]
        for row in self.rows:
            part, good = values[order, row, :].T, valid[order, row, :].T
            good = good & np.isfinite(part)
            self.samples[(period, place, row)] = {
                "values": np.where(good, part, 0).astype(np.complex128), "valid": good.copy(),
                "time_tc": symbol_time_tc(self.scs, slot, symbols[row]), "sequence": header["sequence"],
                "opportunity": f"{header['opportunity']:016x}", "absolute_slot": slot,
                "evidence": obs.EVIDENCE.get(header["evidence"], "unknown")}
        first = self.due - self.periods + 1
        for held in [held for held in self.samples if held[0] < first]:
            del self.samples[held]
        if period == self.due and place == self.slots[-1]:
            lines.append(self._close(period, record))
            self.due += self.hop
            first = self.due - self.periods + 1
            for held in [held for held in self.samples if held[0] < first]:
                del self.samples[held]
        else:
            lines.append(self._line("insufficient_history", "the window is filling", self.due, record,
                                    len(self.samples)))
        return lines

    def open_samples(self):
        """Samples held for a window that did not close: (have, need)."""
        return len(self.samples), self.need


def comparable(line):
    """A line without the host times of its run: what two runs on the same records must agree on."""
    return {key: value for key, value in line.items() if key not in VARYING}


def run_records(records, spec, clock=None):
    """Push records through a new stream in the given order. Returns (lines, views, stream)."""
    stream, lines = Stream(spec, clock), []
    for record in records:
        lines += stream.push(record)
    return lines, stream.views, stream


# ---------------------------------------------------------------- output
def view_name(line):
    return f"{line:06d}.dd.f64"


def write_output(directory, spec, lines, views, stream, source):
    """Write what a sensor leaves: features.jsonl, one view per complete window and sensor.json.

    The directory must exist and must hold none of the three. The digest of a view is taken
    here, after the last record was read.
    """
    directory = Path(directory)
    if not directory.is_dir() or any((directory / name).exists() for name in (LINES_FILE, REPORT_FILE, VIEWS_DIR)):
        raise FeaturesError(f"the output directory {directory} does not exist or holds a result already")
    (directory / VIEWS_DIR).mkdir()
    by_line = {line["line"]: line for line in lines}
    for number, power in views:
        data = power.astype("<f8").tobytes()
        (directory / VIEWS_DIR / view_name(number)).write_bytes(data)
        by_line[number]["view"].update(file=f"{VIEWS_DIR}/{view_name(number)}", sha256=hashlib.sha256(data).hexdigest())
    (directory / LINES_FILE).write_text("".join(json.dumps(line, sort_keys=True) + "\n" for line in lines),
                                        encoding="ascii")
    have, need = stream.open_samples()
    report = {"schema": REPORT_SCHEMA, "spec": spec, "spec_sha256": stream.sha256, "steps": resolve(spec),
              "runtime": {"entry": RUNTIME, "device": "cpu", "python": platform.python_version(), "numpy": np.__version__,
                          "machine": platform.machine(),
                          "module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              "input_fields": list(INPUT_FIELDS), "source": source, "counts": dict(stream.counts), "lines": len(lines),
              "views": len(views), "open_at_end": {"have": have, "need": need}}
    (directory / REPORT_FILE).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return report


class Output:
    """A verified sensor output."""

    def __init__(self, directory, report, lines):
        self.directory, self.report, self.lines = directory, report, lines
        self.spec = report["spec"]
        self.ok = [line for line in lines if line["status"] == "ok"]
        self.by_window = {line["window"]["id"]: line for line in self.ok}

    def view(self, line):
        """The delay-Doppler power of a complete window, [delay][doppler]."""
        data = (self.directory / line["view"]["file"]).read_bytes()
        return np.frombuffer(data, dtype="<f8").reshape(line["view"]["shape"])


def open_output(directory):
    """Verify a sensor output and return it.

    The report names the specification with its hash and the steps of the registry. The lines
    count from zero, each names a known status and the hash of the specification, and their
    counts are those of the report. A complete window has one view file with the hash and the
    size its line names, a vector with one finite value per name, and a peak that is the largest
    cell of the view outside the static column. The directory holds nothing else but the archive
    of the records.
    """
    directory = Path(directory)
    try:
        report = json.loads((directory / REPORT_FILE).read_text(encoding="ascii"))
        lines = [json.loads(text) for text in (directory / LINES_FILE).read_text(encoding="ascii").splitlines()
                 if text.strip()]
    except (OSError, ValueError) as error:
        raise FeaturesError(f"{directory} is not a complete sensor output: {error}") from None
    if not isinstance(report, dict) or report.get("schema") != REPORT_SCHEMA:
        raise FeaturesError(f"{directory / REPORT_FILE} is not a {REPORT_SCHEMA} report")
    spec = check_spec(report.get("spec"))
    if report.get("spec_sha256") != spec_sha256(spec) or report.get("steps") != resolve(spec) \
            or (report.get("runtime") or {}).get("entry") != RUNTIME:
        raise FeaturesError("the report does not name its specification, the steps of the registry and the reference "
                            "runtime")
    counts, names = dict.fromkeys(STATUSES, 0), {LINES_FILE, REPORT_FILE, VIEWS_DIR, RECORDS_DIR}
    views = set()
    for number, line in enumerate(lines):
        if not isinstance(line, dict) or line.get("schema") != LINE_SCHEMA or line.get("line") != number \
                or line.get("status") not in STATUSES or line.get("spec_sha256") != report["spec_sha256"] \
                or line.get("runtime") != RUNTIME:
            raise FeaturesError(f"line {number} of {LINES_FILE} is not a {LINE_SCHEMA} line of this specification")
        counts[line["status"]] += 1
        if line["status"] != "ok":
            if "features" in line or "view" in line:
                raise FeaturesError(f"line {number} has the status {line['status']} and carries features")
            continue
        try:
            view, vector, axes = line["view"], line["features"], line["axes"]
            data = (directory / view["file"]).read_bytes()
            shape = view["shape"]
            good = hashlib.sha256(data).hexdigest() == view["sha256"] and len(data) == 8 * shape[0] * shape[1] \
                and shape == [axes["delay"]["bins"], len(axes["doppler"]["bins"])] \
                and vector["names"] == feature_names(axes) and len(vector["values"]) == len(vector["names"]) \
                and all(type(value) is float and math.isfinite(value) for value in vector["values"]) \
                and len(line["contributing"]) >= 1 and line["history"]["have"] == line["history"]["need"]
            power = np.frombuffer(data, dtype="<f8").reshape(shape)
            moving = [index for index, value in enumerate(axes["doppler"]["hz"])
                      if value != 0 or view["static"] == "none"]
            row, column = np.unravel_index(int(np.argmax(power[:, moving])), (shape[0], len(moving)))
            peak = view["peak"]
            good = good and (peak["delay_bin"], peak["doppler_bin"]) == (int(row), axes["doppler"]["bins"][moving[column]]) \
                and peak["power"] == float(power[row, moving[column]])
        except (KeyError, TypeError, ValueError, OSError, IndexError) as error:
            raise FeaturesError(f"line {number}: the complete window is not whole: {error}") from None
        if not good:
            raise FeaturesError(f"line {number}: the view, its hash, its peak or the vector do not agree")
        views.add(Path(view["file"]).name)
    if any(report["counts"].get(status) != counts[status] for status in STATUSES) or report.get("lines") != len(lines) \
            or report.get("views") != len(views):
        raise FeaturesError("the counts of the report are not those of its lines")
    extra = sorted(path.name for path in directory.iterdir() if path.name not in names)
    extra += sorted(path.name for path in (directory / VIEWS_DIR).iterdir() if path.name not in views)
    if extra:
        raise FeaturesError(f"the output holds {extra[0]}, which its report does not list")
    return Output(directory, report, lines)


def warm_up(spec):
    """Run one small window of noise through every step of a specification. Returns the time it took in ms.

    The first transform of an interpreter loads the FFT module and builds its tables: a stall of
    8 ms was seen on a DGX Spark, against half a millisecond for every later window. A sensor that paid this at
    its first complete window would hold its ring for that long, and the producer drops what
    finds no credit. A sensor therefore warms up before it attaches. Nothing of this window is kept.
    """
    stream = Stream(spec)
    started = time.perf_counter()
    samples = stream.need
    if spec["steps"]["doppler"] == "fft":
        times = [index * period_tc(stream.scs, stream.period_slots) for index in range(samples)]
    else:
        times = [index * estimator.useful_tc(stream.scs) * (2 + index % 3) for index in range(samples)]
    shape = (max(2 * spec["view"]["delay_bins"], 16), samples, 2)
    generator = np.random.Generator(np.random.PCG64(0))
    cube = generator.standard_normal(shape) + 1j * generator.standard_normal(shape)
    valid = np.isfinite(cube)
    bits = np.packbits(valid.reshape(-1), bitorder="little").tobytes()
    np.unpackbits(np.frombuffer(bits, dtype=np.uint8), bitorder="little")
    axes = axes_of(spec, (0, 2, shape[0], (0, 1)), times)
    power, scalars = view_of(spec, np.where(valid, cube, 0), valid, axes)
    features_of(spec, power, scalars, axes)
    return 1000.0 * (time.perf_counter() - started)


def replay(archive_dir, spec, output_dir):
    """Push the records of a consumer archive through a stream, in sequence order, and write the output."""
    archive = obs.open_archive(archive_dir)
    lines, views, stream = run_records(archive.records, spec)
    return write_output(output_dir, spec, lines, views, stream,
                        {"kind": "archive", "directory": str(archive_dir), "reader": archive.report["reader"],
                         "records": len(archive.records)})


def follow(socket_path, spec, output_dir, wait="wait", connect_timeout=30.0, idle_timeout=60.0):
    """Attach to a ring as its consumer, push every record through a stream as it is read, and write the output.

    This is the reference sensor: a second process on the standard-library reader. It copies what
    it needs of a record and releases the record at once, so it holds no credit. The records it
    read are kept under records/, as the reader of Sprint 3.2 keeps them. Before it attaches it
    runs one window of noise through its steps (warm_up), so that no record waits for a first use.
    """
    output_dir = Path(output_dir)
    if not output_dir.is_dir() or any(output_dir.iterdir()):
        raise FeaturesError(f"the output directory {output_dir} does not exist or is not empty")
    warm_ms = warm_up(spec)  # before the attach: the ring is not held while the transforms load
    stream, lines = Stream(spec, time.monotonic_ns), []
    (output_dir / RECORDS_DIR).mkdir()
    consumer = obs.tail(socket_path, output_dir / RECORDS_DIR, wait, connect_timeout, idle_timeout,
                        on_record=lambda record: lines.extend(stream.push(record)))
    return write_output(output_dir, spec, lines, stream.views, stream,
                        {"kind": "ring", "socket": str(socket_path), "reader": consumer["reader"], "wait": wait,
                         "epoch": consumer["epoch"], "records": consumer["counts"]["read"],
                         "stale": consumer["counts"]["stale"], "ended": consumer["ended"],
                         "warm_up_ms": warm_ms})


def summary(output, window=0):
    """What a second reader recovers from a sensor output: the counts and one complete window."""
    made = {"spec_id": output.spec["spec_id"], "spec_sha256": output.report["spec_sha256"],
            "runtime": output.report["runtime"]["entry"], "source": output.report["source"]["kind"],
            "steps": {stage: f"{entry['entry']} ({entry['library']} on the {entry['device']})"
                      for stage, entry in output.report["steps"].items()},
            "counts": output.report["counts"], "complete_windows": len(output.ok)}
    if not 0 <= window < len(output.ok):
        raise FeaturesError(f"the output holds {len(output.ok)} complete windows, and window {window} was asked for")
    line = output.ok[window]
    axes, view = line["axes"], line["view"]
    made["window"] = {
        "id": line["window"]["id"], "contributing": len(line["contributing"]),
        "first_opportunity": line["contributing"][0]["opportunity"], "acquisition_us": line["time"]["acquisition_us"],
        "delay_resolution_ns": axes["delay"]["resolution_ns"], "delay_unambiguous_ns": axes["delay"]["unambiguous_ns"],
        "doppler_resolution_hz": axes["doppler"]["resolution_hz"], "doppler_unambiguous_hz": axes["doppler"]["unambiguous_hz"],
        "peak": view["peak"], "static_peak": view["static_peak"], "fill": line["validity"]["fill"],
        "view_file": view["file"], "view_sha256": view["sha256"],
        "features": dict(zip(line["features"]["names"], line["features"]["values"]))}
    return made


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reference features on pilot observations: the cube, the "
                                                 "delay-Doppler view and the compact vector.")
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("spec", help="write the specification of the delay-Doppler branch of the demonstration")
    make.add_argument("--output", required=True, type=Path)
    make.add_argument("--spec-id", default="dd-uniform-8")
    make.add_argument("--periods", type=int, default=8)
    make.add_argument("--hop-periods", type=int, default=8)
    make.add_argument("--slot", type=int, default=7, help="slot of the period whose records enter the sequence")
    make.add_argument("--delay-bins", type=int, default=32)
    live = commands.add_parser("follow", help="attach to a ring and compute features as the records arrive")
    live.add_argument("--socket", required=True, type=Path, help="control channel of the producer's ring")
    live.add_argument("--wait", choices=sorted(obs.WAIT_MODES), default="wait")
    live.add_argument("--connect-ms", type=int, default=30000)
    live.add_argument("--idle-ms", type=int, default=60000)
    again = commands.add_parser("replay", help="compute features from the records of a consumer archive")
    again.add_argument("--archive", required=True, type=Path)
    for command in (live, again):
        command.add_argument("--spec", required=True, type=Path)
        command.add_argument("--output-dir", required=True, type=Path, help="existing, empty directory")
    show = commands.add_parser("show", help="verify a sensor output and print one complete window")
    show.add_argument("directory", type=Path)
    show.add_argument("--window", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        if args.command == "spec":
            if args.output.exists():
                raise FeaturesError(f"{args.output} already exists. A specification is never overwritten.")
            spec = demo_spec(args.spec_id, args.periods, args.hop_periods, args.slot, args.delay_bins)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="ascii")
            print(json.dumps({"spec": str(args.output), "sha256": spec_sha256(spec)}, indent=2))
            return 0
        if args.command == "show":
            print(json.dumps(summary(open_output(args.directory), args.window), indent=2))
            return 0
        spec = load_spec(args.spec)
        if args.command == "follow":
            report = follow(args.socket, spec, args.output_dir, args.wait, args.connect_ms / 1000, args.idle_ms / 1000)
        else:
            report = replay(args.archive, spec, args.output_dir)
        counts = report["counts"]
        print(f"features: {counts['records']} records, {counts['ok']} complete windows, "
              f"{counts['insufficient_history']} lines of a filling window, {counts['gap']} gaps, {counts['invalid']} "
              f"invalid, {counts['rejected']} rejected, {counts['passed']} outside the sequence")
        return 0
    except (FeaturesError, ObsError) as error:
        parser.exit(2, f"features: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
