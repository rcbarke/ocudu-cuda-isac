"""cuSense-like front end and location model on the receiver's estimate (Sprint 3.5, selector cusense_frontend).

A sensor reads obs/v1 records in the order of the ring. This module is the reference of the three
stages that the cuSense paper applies to them (Villa et al., arXiv:2512.06493v3, section 5.2), on
the receiver's own estimate, the stage O3, and on the observer's pilot least squares, the stage O2:

  stage 1   the background template H_B[a, k, s]: the complex average of the estimates of
            target-absent calibration records over antenna, subcarrier and DM-RS symbol, taken
            over the valid elements only.
  stage 2   the causal window: the uplink slots of one TDD period, each placed at the subcarrier
            offset of its allocation and averaged coherently over the valid elements. The
            template is subtracted in the magnitude domain, |H_t| - |H_B|, the result is averaged
            over the DM-RS symbols, and a z-score with one mean and one deviation gives the input
            X[a, k] of the model.
  stage 3   the model cusense_cnn: a location map over the position grid of the episode set, and
            a presence logit that a frozen threshold turns into a decision.

Two things differ from the published design, and both are stated wherever a result is reported.
The template and the statistics of the z-score are fitted on the training split and on no other
sample. And a presence logit is trained with the map, so that a target-absent window is scored.

Input entries. o3_fp16 reads the stage O3 in the layout ocudu_o3_fp16: every subcarrier of the
allocation on every DM-RS symbol, after frequency interpolation and smoothing and before time
interpolation, widened from FP16 once. o2_comb reads the stage O2 in the layout ocudu_o2_comb_c64:
the pilot comb. The two give the same pipeline on K = 612 and on K = 306 subcarriers at the
demonstration profile.

Causality. A WindowStream takes one record at a time and answers at once: "ok" with the window
that the record completes, "insufficient_history" while a window fills, "gap" for a window that
closed without one of its slots, "invalid" for a window whose valid support is too small and
"rejected" for a record that is no input of the specification. A window never spans two epochs,
two episode keys or two periods, and nothing of a record is read but the fields of INPUT_FIELDS,
the validity bits and the payload. A decision is made from one window, when it is complete.

The artifact, cusense-artifact/v1, is one directory of four files: model.json with the
specification of the windows, the normalizer, the grid geometry with its units, the threshold and
the hashes of every part, template.c64 and template.count.u4 with the template and the count of
records behind each of its elements, and weights.f32 with every tensor of the model. It holds no
sample, no label and no split.

An optional stage, map_mean_kalman, averages the newest maps of one history and filters the
position. It has its own output and never changes the raw map, the score or the decision.

Every entry carries runtime "reference": NumPy on the CPU for the front end and PyTorch for the
model, on the CPU or on a CUDA device that is selected when the model is loaded. This module
defines the stages and checks the kernels and the engine of Sprint 3.6, and no latency it
measures is a result. It imports the readers of the sensing package and nothing of a host, a
simulator or an episode.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import tempfile
import time

import numpy as np
import torch

from . import features
from . import native_estimate
from . import obs
from . import pilot_ls as estimator
from . import presence
from .features import FeaturesError
from .models import cusense_cnn
from .models.cusense_cnn import ModelError
from .obs import ObsError
from .presence import PresenceError

SPEC_SCHEMA = "cusense-spec/v1"
WINDOW_SCHEMA = "cusense-window/v1"
WINDOWS_REPORT_SCHEMA = "cusense-windows-report/v1"
ARTIFACT_SCHEMA = "cusense-artifact/v1"
DECISION_SCHEMA = "cusense-decision/v1"
REPORT_SCHEMA = "cusense-report/v1"
SAMPLE_SCHEMA = "cusense-sample/v1"
SETTINGS_SCHEMA = "cusense-train-config/v1"
MODEL_FILE = "model.json"
TEMPLATE_FILE = "template.c64"
TEMPLATE_COUNT_FILE = "template.count.u4"
WEIGHTS_FILE = "weights.f32"
ARTIFACT_FILES = (MODEL_FILE, TEMPLATE_FILE, TEMPLATE_COUNT_FILE, WEIGHTS_FILE)
WINDOWS_FILE = "windows.jsonl"
WINDOWS_REPORT_FILE = "windows.json"
DECISIONS_FILE = "decisions.jsonl"
REPORT_FILE = "cusense.json"
INPUTS_FILE = "inputs.f32"
MAPS_FILE = "maps.f32"
WINDOWS_DIR = "windows"
RECORDS_DIR = "records"
RUNTIME = features.RUNTIME
STATUSES = features.STATUSES
# The native core of Sprint 3.6 writes the same two files under its own runtime entry. It has two
# statuses that only a live core has: a record of the window was rewritten while it was read, and
# the window was overtaken by a newer complete one before it ran. A reader of an output accepts
# both runtimes, and the reference writes its own entry only.
NATIVE_RUNTIME = "native"
RUNTIMES = (RUNTIME, NATIVE_RUNTIME)
NATIVE_STATUSES = ("stale", "stale_skipped")
SPLITS = presence.SPLITS
TC_PER_SECOND = estimator.TC_PER_SECOND
EPSILON = 1e-8
PUBLISHED_TAU = 1e-10
SPEC_FIELDS = ("schema", "spec_id", "input", "radio", "window", "support", "steps", "validity")
INPUT = ("entry", "stage", "source", "layout", "dtype")
RADIO = ("scs_khz", "period_slots")
WINDOW = ("slots_in_period", "min_valid_fraction")
SUPPORT = ("first_subcarrier", "subcarrier_step", "subcarriers", "dmrs_symbols", "rx")
STEPS = ("ingest", "window", "clutter", "symbols", "normalizer")
# Everything a stream reads of a record header. A field outside this list cannot change a window.
INPUT_FIELDS = ("epoch", "sequence", "episode", "opportunity", "absolute_slot", "stage", "source", "layout", "dtype",
                "producer_steps", "scale", "rank", "shape", "strides", "first_subcarrier", "subcarrier_step",
                "subcarrier_count", "dmrs_symbols", "rx_count", "layer_count", "antenna_map", "mask_bytes",
                "payload_bytes", "time_unit", "evidence", "port", "t_first_sample", "t_last_sample", "t_ready_ns",
                "t_publish_ns")
# Host times, the device and the measured durations of a run. Two runs on the same records agree on every other field.
VARYING_WINDOW = ("t_ready_ns", "t_publish_ns", "t_window_ns")
VARYING = VARYING_WINDOW + ("t_decision_ns", "frontend_ns", "infer_ns", "device")
# What a model reads of a window. A field outside this list cannot change a score.
WINDOW_FIELDS = ("status", "spec_sha256", "rx.elements", "input.producer_steps", "the mean and the count of the window")
ARTIFACT_FIELDS = ("schema", "model_id", "entries", "input", "template", "normalizer", "model", "weights", "grid",
                   "threshold", "smoothing", "output", "sha256")
ENTRY_FIELDS = ("input", "ingest", "window", "clutter", "symbols", "normalizer", "model", "detector", "smoothing")
ARTIFACT_INPUT = ("spec", "spec_sha256", "elements", "producer_steps", "shape", "dtype")
TEMPLATE_FIELDS = ("file", "dtype", "shape", "bytes", "sha256", "count_file", "count_dtype", "count_bytes",
                   "count_sha256", "valid_elements", "records_min", "records_max", "fitted_on")
NORMALIZER_FIELDS = ("entry", "mean", "scale", "epsilon", "sha256", "fitted_on")
FITTED_ON = ("split", "windows", "windows_sha256")
MODEL_FIELDS = ("config", "config_sha256", "parameters", "heads", "losses", "target")
WEIGHTS_FIELDS = ("file", "dtype", "layout", "tensors", "count", "bytes", "sha256")
GRID_FIELDS = ("rows", "cols", "cell_m", "origin_m", "area", "rule", "units", "sha256")
THRESHOLD_FIELDS = presence.THRESHOLD_FIELDS + ("sha256",)
SMOOTHING_FIELDS = ("entry", "maps", "process_noise_m2_s3", "measurement_noise_m2", "initial_speed_m_s")
SAMPLE_FIELDS = ("schema", "window", "episode", "split", "scene", "pair", "label", "cell", "position_m", "spec_sha256",
                 "elements", "producer_steps", "shape", "index", "mean_sha256", "count_sha256")
SETTINGS_FIELDS = ("schema", "model_id", "model", "loss", "optimizer", "seed", "device", "min_per_class", "threshold",
                   "smoothing")
MODEL_SETTINGS = ("stem_channels", "stem_kernel", "block_channels", "block_kernel", "hidden", "dropout")
LOSS_SETTINGS = ("grid_weight", "sigma_cells")
OPTIMIZER_SETTINGS = ("entry", "learning_rate", "batch_size", "epochs", "weight_decay")
WEIGHTS_LAYOUT = "one little-endian float32 per value, tensor after tensor in the order of weights.tensors"
GRID_UNITS = {"cell": "[row, column] of the position grid: rows along y, columns along x",
              "cell_m": "[height along y, width along x] of one cell in metres",
              "origin_m": "[x, y] of the corner of cell [0, 0] in metres, in the frame of the scene",
              "position_m": "[x, y] in metres, in the frame of the scene: the centre of a cell for an estimate"}
TARGET_RULE = "the cell of the target at the last opportunity of the window, smoothed by a Gaussian and normalized"
OUTPUT = {"input": "X[a, k] = (the mean over the DM-RS symbols of |H_t[a, k, s]| - |H_B[a, k, s]| - mean) / (scale + "
                   "epsilon), float32 of shape [1, A, K]; an element without a valid sample is 0",
          "map": "the softmax of the grid logits, float32 of shape [rows, cols], summing to one",
          "location": "the cell of the largest value of the map, the first one in row order, and the centre of that "
                      "cell in metres",
          "presence": "the presence logit and probability = 1 / (1 + exp(-logit))",
          "decision": "present when the logit lies above threshold.logit"}
_NUMPY = {"device": "cpu", "library": "numpy", "runtime": RUNTIME}
_TORCH = {"device": "selected when the model is loaded: cpu or a cuda device", "library": "torch", "runtime": RUNTIME}
_PYTHON = {"device": "cpu", "library": "python", "runtime": RUNTIME}
INPUTS = {"o3_fp16": {"stage": "O3", "source": "pusch_dmrs", "layout": native_estimate.LAYOUT,
                      "dtype": native_estimate.DTYPE},
          "o2_comb": {"stage": "O2", "source": "pusch_dmrs", "layout": estimator.LAYOUT, "dtype": estimator.DTYPE}}
# The entries of this ticket. Each states where it runs. Sprint 3.6 adds the native unit of the
# ingest and of the front end under the same names, and the engine of the model.
REGISTRY = {
    "input": {"o3_fp16": dict(_NUMPY, does="the receiver's own estimate, the stage O3: every subcarrier of the "
                                           "allocation on every DM-RS symbol, after frequency interpolation and before "
                                           "time interpolation"),
              "o2_comb": dict(_NUMPY, does="the observer's pilot least squares, the stage O2: the pilot comb of the "
                                           "allocation on every DM-RS symbol")},
    "ingest": {native_estimate.LAYOUT: dict(_NUMPY, does="[dmrs_symbol][prb][rx_port][subcarrier_in_prb] as complex "
                                                         "FP16, widened once and without a scale, to [rx, subcarrier, "
                                                         "dmrs_symbol] with the array elements in rising order"),
               estimator.LAYOUT: dict(_NUMPY, does="[rx_plane][reference_symbol][pilot] as complex64 to [rx, pilot, "
                                                   "dmrs_symbol] with the array elements in rising order")},
    "window": {"coherent_mean": dict(_NUMPY, does="the uplink slots of one period, each placed at the subcarrier "
                                                  "offset of its allocation and averaged in the complex domain over "
                                                  "the valid elements")},
    "clutter": {"background_magnitude": dict(_NUMPY, does="|H_t| - |H_B| with the complex template of the target-absent "
                                                          "calibration records of the training split")},
    "symbols": {"dmrs_mean": dict(_NUMPY, does="the mean over the DM-RS symbols with a valid sample")},
    "normalizer": {"zscore_global": dict(_NUMPY, does="(x - mean) / (scale + epsilon) with one mean and one deviation "
                                                      "over every window, antenna and valid subcarrier of the "
                                                      "training split")},
    "model": {cusense_cnn.ENTRY: dict(_TORCH, does="plain-block 1D CNN from [1, A, K] to a location map over the "
                                                   "position grid and a presence logit, float32, one window at a time")},
    "detector": {"frozen_threshold": dict(_PYTHON, does="present when the presence logit lies above a threshold that "
                                                        "was calibrated on the target-absent windows of the validation "
                                                        "split and frozen")},
    "smoothing": {"none": dict(_PYTHON, does="no optional stage: a decision carries its raw map and nothing else"),
                  "map_mean_kalman": dict(_NUMPY, does="the mean of the newest maps of one history, the cell of its "
                                                       "largest value, and a constant-velocity Kalman filter on that "
                                                       "position; an output of its own")},
}


class CuSenseError(ValueError):
    """A specification, a record sequence, an artifact, a sample set or a sensor output violates the documented interface."""


# ---------------------------------------------------------------- helpers
def canonical(value):
    """The bytes a hash of a JSON value is taken over."""
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def part_sha256(part):
    """The hash of one part of an artifact: SHA-256 over every field of it but its own hash, in canonical form."""
    return _sha256(canonical({key: value for key, value in part.items() if key != "sha256"}))


def ids_sha256(values):
    """The hash of a set of identifiers: SHA-256 over their sorted list."""
    return _sha256(json.dumps(sorted(values), separators=(",", ":")).encode("ascii"))


def _fields(value, fields, name):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise CuSenseError(f"{name} requires exactly {', '.join(fields)}")
    return value


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise CuSenseError(f"{name} must be an integer from {low} to {high}")
    return value


def _number(value, name, low=None, high=None, above=None):
    if type(value) not in (int, float) or not math.isfinite(value) or (low is not None and value < low) \
            or (high is not None and value > high) or (above is not None and not value > above):
        bound = (f" above {above}" if above is not None else "") + (f" from {low}" if low is not None else "") \
            + (f" to {high}" if high is not None else "")
        raise CuSenseError(f"{name} must be a finite number{bound}")
    return float(value)


default_device, device_of, device_facts = presence.default_device, presence.device_of, presence.device_facts


def _device(name):
    try:
        return name if isinstance(name, torch.device) else device_of(name)
    except PresenceError as error:
        raise CuSenseError(str(error).replace("a head runs", "a model runs")) from None


# ---------------------------------------------------------------- specification
def check_spec(spec):
    """Check a cusense-spec/v1 object and return it. Every refusal is made before a record is read."""
    _fields(spec, SPEC_FIELDS, "a window specification")
    if spec["schema"] != SPEC_SCHEMA:
        raise CuSenseError(f"unsupported specification schema {spec['schema']!r}")
    if not isinstance(spec["spec_id"], str) or not spec["spec_id"] or not spec["spec_id"].isascii():
        raise CuSenseError("spec_id must be a nonempty ASCII string")
    wanted = _fields(spec["input"], INPUT, "input")
    if wanted["entry"] not in INPUTS:
        raise CuSenseError(f"unknown input entry {wanted['entry']!r} (registered: {', '.join(sorted(INPUTS))})")
    if {name: wanted[name] for name in INPUT if name != "entry"} != INPUTS[wanted["entry"]]:
        known = INPUTS[wanted["entry"]]
        raise CuSenseError(f"the input entry {wanted['entry']} reads the stage {known['stage']} of {known['source']} in "
                           f"the layout {known['layout']} as {known['dtype']}")
    radio = _fields(spec["radio"], RADIO, "radio")
    try:
        mu = estimator.numerology(radio["scs_khz"])
    except estimator.PilotLsError as error:
        raise CuSenseError(str(error)) from None
    period = _integer(radio["period_slots"], "radio.period_slots", 1, 1 << 16)
    if period % (1 << mu):
        raise CuSenseError("radio.period_slots must be a whole number of subframes, so that a period has one length")
    window = _fields(spec["window"], WINDOW, "window")
    slots = window["slots_in_period"]
    if not isinstance(slots, list) or not slots or slots != sorted(set(slots)):
        raise CuSenseError("window.slots_in_period must be a nonempty rising list")
    for value in slots:
        _integer(value, "an entry of window.slots_in_period", 0, period - 1)
    _number(window["min_valid_fraction"], "window.min_valid_fraction", high=1, above=0)
    support = _fields(spec["support"], SUPPORT, "support")
    _integer(support["first_subcarrier"], "support.first_subcarrier", 0, 1 << 16)
    step = _integer(support["subcarrier_step"], "support.subcarrier_step", 1, 12)
    _integer(support["subcarriers"], "support.subcarriers", 16, 1 << 16)
    _integer(support["dmrs_symbols"], "support.dmrs_symbols", 1, estimator.SYMBOLS_PER_SLOT)
    _integer(support["rx"], "support.rx", 1, obs.MAX_PLANES)
    if wanted["entry"] == "o3_fp16" and step != 1:
        raise CuSenseError("the stage O3 holds every subcarrier of its allocation: support.subcarrier_step is 1")
    steps = _fields(spec["steps"], STEPS, "steps")
    for stage in STEPS:
        if steps[stage] not in REGISTRY[stage]:
            raise CuSenseError(f"unknown {stage} step {steps[stage]!r} (registered: "
                               f"{', '.join(sorted(REGISTRY[stage]))})")
    if steps["ingest"] != wanted["layout"]:
        raise CuSenseError(f"the ingest step of the input entry {wanted['entry']} is {wanted['layout']}")
    _number(_fields(spec["validity"], ("tau",), "validity")["tau"], "validity.tau", low=0)
    return spec


def spec_sha256(spec):
    """The SHA-256 of a specification in its canonical form. Taken once, when a stream is bound."""
    return _sha256(canonical(spec))


def demo_spec(entry="o3_fp16", spec_id=None, slots=(7, 8, 9), subcarriers=None, rx=4, dmrs_symbols=2, first_subcarrier=0):
    """The cuSense-like branch of the demonstration: the three uplink slots of one 5 ms period, 51 blocks, four RX."""
    if entry not in INPUTS:
        raise CuSenseError(f"unknown input entry {entry!r} (registered: {', '.join(sorted(INPUTS))})")
    step = 1 if entry == "o3_fp16" else 2
    count = (612 if step == 1 else 306) if subcarriers is None else subcarriers
    return check_spec({
        "schema": SPEC_SCHEMA, "spec_id": spec_id or ("cusense-o3-period" if entry == "o3_fp16" else "cusense-o2-period"),
        "input": dict(INPUTS[entry], entry=entry), "radio": {"scs_khz": 30, "period_slots": 10},
        "window": {"slots_in_period": list(slots), "min_valid_fraction": 0.9},
        "support": {"first_subcarrier": first_subcarrier, "subcarrier_step": step, "subcarriers": count,
                    "dmrs_symbols": dmrs_symbols, "rx": rx},
        "steps": {"ingest": INPUTS[entry]["layout"], "window": "coherent_mean", "clutter": "background_magnitude",
                  "symbols": "dmrs_mean", "normalizer": "zscore_global"},
        "validity": {"tau": PUBLISHED_TAU}})


def load_spec(path):
    try:
        return check_spec(json.loads(Path(path).read_text(encoding="ascii")))
    except (OSError, ValueError) as error:
        if isinstance(error, CuSenseError):
            raise
        raise CuSenseError(f"cannot read the specification: {error}") from None


def resolve(spec, smoothing="none"):
    """The registry entries a specification selects, by stage, with the model, the detector and the optional stage."""
    made = {"input": dict(REGISTRY["input"][spec["input"]["entry"]], entry=spec["input"]["entry"])}
    made.update({stage: dict(REGISTRY[stage][spec["steps"][stage]], entry=spec["steps"][stage]) for stage in STEPS})
    for stage, entry in (("model", cusense_cnn.ENTRY), ("detector", "frozen_threshold"), ("smoothing", smoothing)):
        made[stage] = dict(REGISTRY[stage][entry], entry=entry)
    return made


def shape_of(spec):
    """[A, K, S] of a window of a specification: array elements, subcarriers of the support and DM-RS symbols."""
    support = spec["support"]
    return [support["rx"], support["subcarriers"], support["dmrs_symbols"]]


def period_s(spec):
    """Length of one period in seconds: the time between two windows of one history."""
    return features.period_tc(spec["radio"]["scs_khz"], spec["radio"]["period_slots"]) / TC_PER_SECOND


# ---------------------------------------------------------------- record
def place(first_subcarrier, support, count=1):
    """Where an allocation that starts at a subcarrier lies on the subcarrier axis of a window.

    The axis of a window is the support of its specification: support.subcarriers positions from
    support.first_subcarrier at support.subcarrier_step. An allocation that starts at resource
    block 2 lands on subcarrier 24 of an axis that starts at subcarrier 0 with step 1. An
    allocation off the lattice or beyond the support is refused.
    """
    offset, rest = divmod(first_subcarrier - support["first_subcarrier"], support["subcarrier_step"])
    if rest or offset < 0 or offset + count > support["subcarriers"]:
        raise CuSenseError(f"an allocation of {count} positions from subcarrier {first_subcarrier} does not lie on the "
                           f"support: {support['subcarriers']} positions from subcarrier {support['first_subcarrier']} "
                           f"at a step of {support['subcarrier_step']}")
    return offset


def symbols_of(header):
    return [symbol for symbol in range(estimator.SYMBOLS_PER_SLOT) if header["dmrs_symbols"] >> symbol & 1]


def steps_of(header):
    return [name for bit, name in obs.STEPS.items() if header["producer_steps"] & bit]


def refusal(record, spec):
    """Why a record is no input of a specification, or None. Nothing of the payload is read."""
    header, wanted, support = record.header, spec["input"], spec["support"]
    named = (obs.STAGES.get(header["stage"]), obs.SOURCES.get(header["source"]), obs.LAYOUTS.get(header["layout"]),
             obs.DTYPES.get(header["dtype"]))
    if named != (wanted["stage"], wanted["source"], wanted["layout"], wanted["dtype"]):
        return f"the record is {named[0]} of {named[1]} in the layout {named[2]} as {named[3]}, and the " \
               f"specification reads {wanted['stage']} of {wanted['source']} in the layout {wanted['layout']}"
    shape, rx, symbols = record.shape, header["rx_count"], symbols_of(header)
    if not symbols or header["layer_count"] != 1 or not 1 <= rx <= obs.MAX_PLANES \
            or len(set(header["antenna_map"][:rx])) != rx:
        return "the record does not hold one layer with one array element per receive plane"
    if wanted["layout"] == native_estimate.LAYOUT:
        blocks = header["subcarrier_count"] // native_estimate.SUBCARRIERS_PER_PRB
        if header["rank"] != 4 or shape != [len(symbols), blocks, rx, native_estimate.SUBCARRIERS_PER_PRB] \
                or header["subcarrier_count"] != blocks * native_estimate.SUBCARRIERS_PER_PRB or blocks < 1 \
                or record.strides != [shape[1] * shape[2] * shape[3], shape[2] * shape[3], shape[3], 1]:
            return "the shape is not [dmrs_symbol][prb][rx_port][subcarrier_in_prb] of whole resource blocks"
        elements, width = shape[0] * shape[1] * shape[2] * shape[3], native_estimate.ELEMENT_BYTES
        made = steps_of(header)
        if made[:2] != native_estimate.STEPS[:2] or "time_interpolation" in made:
            return "the estimate is not the stage after frequency interpolation and before time interpolation"
    else:
        if header["rank"] != 3 or shape != [rx, len(symbols), header["subcarrier_count"]] \
                or record.strides != [shape[1] * shape[2], shape[2], 1]:
            return "the shape is not [rx_plane][reference_symbol][pilot]"
        elements, width = shape[0] * shape[1] * shape[2], 8
    if header["payload_bytes"] != width * elements or header["mask_bytes"] != (elements + 7) // 8:
        return "the payload or the validity bits do not have the size of the shape"
    if rx != support["rx"] or len(symbols) != support["dmrs_symbols"]:
        return f"the record holds {rx} receive planes and {len(symbols)} DM-RS symbols, and the specification " \
               f"{support['rx']} and {support['dmrs_symbols']}"
    if header["subcarrier_step"] != support["subcarrier_step"]:
        return f"the record has a subcarrier step of {header['subcarrier_step']}, and the support of the " \
               f"specification {support['subcarrier_step']}"
    try:
        place(header["first_subcarrier"], support, header["subcarrier_count"])
    except CuSenseError as error:
        return str(error)
    scs, slot = spec["radio"]["scs_khz"], header["absolute_slot"]
    if obs.TIME_UNITS.get(header["time_unit"]) != "tc" \
            or header["t_first_sample"] != features.symbol_time_tc(scs, slot, symbols[0]) \
            or header["t_last_sample"] != features.symbol_time_tc(scs, slot, symbols[-1]) + estimator.useful_tc(scs):
        return "the sample times of the record are not those of its slot and its DM-RS symbols at the subcarrier " \
               "spacing of the specification"
    return None


def planes(record, spec):
    """The estimate of one record as [rx, position, dmrs_symbol]: (values complex128, valid bool).

    The receive planes are sorted by their array element. An element is valid when its validity
    bit is set, both parts are finite and its magnitude lies above validity.tau: the rule by
    which the published pipeline leaves zero padding out of every average.
    """
    header, shape = record.header, tuple(record.shape)
    elements = int(np.prod(shape))
    rx = header["rx_count"]
    order = sorted(range(rx), key=lambda plane: header["antenna_map"][plane])
    bits = np.unpackbits(np.frombuffer(record.mask, dtype=np.uint8), bitorder="little")[:elements].astype(bool)
    if spec["input"]["layout"] == native_estimate.LAYOUT:
        if len(record.payload) != native_estimate.ELEMENT_BYTES * elements:
            raise CuSenseError("the payload does not have the size of the shape")
        parts = np.frombuffer(record.payload, dtype="<f2").astype(np.float64).reshape(shape + (2,))
        values = (parts[..., 0] + 1j * parts[..., 1]).transpose(2, 1, 3, 0).reshape(rx, shape[1] * shape[3], shape[0])
        valid = bits.reshape(shape).transpose(2, 1, 3, 0).reshape(rx, shape[1] * shape[3], shape[0])
    else:
        if len(record.payload) != 8 * elements:
            raise CuSenseError("the payload does not have the size of the shape")
        values = np.frombuffer(record.payload, dtype="<c8").astype(np.complex128).reshape(shape).transpose(0, 2, 1)
        valid = bits.reshape(shape).transpose(0, 2, 1)
    values, valid = values[order], valid[order]
    finite = np.isfinite(values.real) & np.isfinite(values.imag)
    valid = valid & finite
    values = np.where(valid, values, 0)
    return values, valid & (np.abs(values) > spec["validity"]["tau"])


# ---------------------------------------------------------------- arithmetic
def coherent_mean(values, valid):
    """The complex mean over the first axis, taken over the valid elements: (mean, count).

    values is complex [N, ...] and valid boolean of the same shape. An element without a valid
    sample has the mean 0 and the count 0. This is the average of the template (stage 1) and of
    the window (stage 2): the same formula on calibration records and on the slots of a period.
    """
    values, valid = np.asarray(values, dtype=np.complex128), np.asarray(valid, dtype=bool)
    if values.shape != valid.shape or values.ndim < 1:
        raise CuSenseError("the values and their validity bits do not have one shape")
    count = valid.sum(axis=0)
    total = np.where(valid, values, 0).sum(axis=0)
    return np.where(count > 0, total / np.maximum(count, 1), 0), count


def merge_means(means, counts):
    """The mean of the records behind several means, each with its count per element: (mean, count).

    A mean with the count n stands for n records. The result is what coherent_mean gives on all
    of those records at once.
    """
    means, counts = np.asarray(means, dtype=np.complex128), np.asarray(counts, dtype=np.uint64)
    if means.shape != counts.shape or means.ndim < 1:
        raise CuSenseError("the means and their counts do not have one shape")
    count = counts.sum(axis=0)
    total = (means * counts).sum(axis=0)
    return np.where(count > 0, total / np.maximum(count, 1), 0), count


def background_subtract(mean, count, template, template_count):
    """Stage 2 after the window average: |H_t| - |H_B|, averaged over the DM-RS symbols. Returns (d, valid).

    mean and template are complex [A, K, S] with their counts. An element takes part where the
    window and the template both hold a valid sample. d is real [A, K], and valid marks the
    positions with a DM-RS symbol that took part.
    """
    mean, template = np.asarray(mean, dtype=np.complex128), np.asarray(template, dtype=np.complex128)
    if mean.shape != template.shape or mean.ndim != 3 or np.shape(count) != mean.shape \
            or np.shape(template_count) != mean.shape:
        raise CuSenseError(f"a window of the shape {list(mean.shape)} and a template of the shape "
                           f"{list(template.shape)} are not one [A, K, S] support")
    usable = (np.asarray(count) > 0) & (np.asarray(template_count) > 0)
    difference = np.where(usable, np.abs(mean) - np.abs(template), 0.0)
    symbols = usable.sum(axis=2)
    return np.where(symbols > 0, difference.sum(axis=2) / np.maximum(symbols, 1), 0.0), symbols > 0


def zscore(d, valid, mean, scale, epsilon=EPSILON):
    """The input of the model: (d - mean) / (scale + epsilon) as float32. A position without a valid sample is 0."""
    return np.where(valid, (np.asarray(d, dtype=np.float64) - mean) / (scale + epsilon), 0.0).astype(np.float32)


# ---------------------------------------------------------------- stream
class WindowStream:
    """The causal window stream of one specification. push() takes the records in the order of the ring."""

    def __init__(self, spec, clock=None):
        self.spec = check_spec(spec)
        self.clock = clock  # host time of an answer, read when the answer is complete. It is stored, never used.
        self.sha256 = spec_sha256(spec)
        self.scs, self.period_slots = spec["radio"]["scs_khz"], spec["radio"]["period_slots"]
        self.slots = spec["window"]["slots_in_period"]
        self.need = len(self.slots)
        self.shape = shape_of(spec)
        self.counts = dict.fromkeys(("records", "passed", "resets") + STATUSES, 0)
        self.tensors = []  # (line, mean as complex64, count as uint8) of every complete window, in order
        self.last = None   # the tensors of the newest complete window
        self._lines = 0
        self._reset()

    def _reset(self):
        self.key = self.period = None
        self.last_slot = -1
        self.samples = {}

    def _line(self, status, reason, period, record, have):
        header = None if record is None else record.header
        self.counts[status] += 1
        self._lines += 1
        window = None
        if period is not None:
            epoch, episode = self.key
            window = {"id": f"{epoch}/{episode:016x}/{period}", "epoch": epoch, "episode": f"{episode:016x}",
                      "period": period}
        return {"schema": WINDOW_SCHEMA, "line": self._lines - 1, "status": status, "reason": reason, "window": window,
                "trigger": None if header is None else {"epoch": header["epoch"], "sequence": header["sequence"],
                                                        "opportunity": f"{header['opportunity']:016x}",
                                                        "absolute_slot": header["absolute_slot"]},
                "history": {"have": have, "need": self.need}, "spec_sha256": self.sha256, "runtime": RUNTIME,
                "t_ready_ns": None if header is None else header["t_ready_ns"],
                "t_publish_ns": None if header is None else header["t_publish_ns"],
                "t_window_ns": None if self.clock is None else self.clock()}

    def _close(self, period, record):
        """The outcome of the window of one period, from the slots held."""
        held = [self.samples.get(slot) for slot in self.slots]
        have = sum(sample is not None for sample in held)
        if have != self.need:
            line = self._line("gap", f"{self.need - have} of {self.need} slots of the window never arrived", period,
                              record, have)
            line["missing"] = [{"period": period, "slot_in_period": slot}
                               for slot, sample in zip(self.slots, held) if sample is None]
            return line
        rx, positions, symbols = self.shape
        values = np.zeros((self.need, rx, positions, symbols), dtype=np.complex128)
        valid = np.zeros((self.need, rx, positions, symbols), dtype=bool)
        for index, sample in enumerate(held):
            start, width = sample["offset"], sample["values"].shape[1]
            values[index, :, start:start + width, :] = sample["values"]
            valid[index, :, start:start + width, :] = sample["valid"]
        mean, count = coherent_mean(values, valid)
        total, good = int(count.size), int((count > 0).sum())
        elements, steps = {sample["elements"] for sample in held}, {tuple(sample["steps"]) for sample in held}
        first, last = held[0], held[-1]
        facts = {"contributing": [{"sequence": sample["sequence"], "opportunity": sample["opportunity"],
                                   "absolute_slot": sample["absolute_slot"], "first_position": sample["offset"],
                                   "positions": int(sample["values"].shape[1])} for sample in held],
                 "validity": {"elements": total, "valid": good, "holes": total - good, "fill": good / total,
                              "records_per_element": {"min": int(count.min()), "max": int(count.max())},
                              "interpolation": "none: an element without a valid sample is left out of every average"},
                 "time": {"unit": "tc", "tc_per_second": TC_PER_SECOND, "t_first_sample": first["t_first_sample"],
                          "t_last_sample": last["t_last_sample"],
                          "acquisition_us": 1e6 * (last["t_last_sample"] - first["t_first_sample"]) / TC_PER_SECOND},
                 "input": dict(self.spec["input"], port=first["port"], scale=first["scale"],
                               producer_steps=list(first["steps"]),
                               evidence=sorted({sample["evidence"] for sample in held})),
                 "rx": {"elements": list(first["elements"]), "unit": "array element of the antenna map, rising"}}
        if len(elements) != 1 or len(steps) != 1:
            line = self._line("invalid", "the slots of the window were not estimated on one array or by one chain of "
                                         "steps", period, record, have)
        elif good / total < self.spec["window"]["min_valid_fraction"]:
            line = self._line("invalid", f"{good} of {total} elements of the window hold a valid sample, below the "
                                         f"share {self.spec['window']['min_valid_fraction']} of the specification",
                              period, record, have)
        else:
            stored, counted = np.ascontiguousarray(mean.astype("<c8")), np.ascontiguousarray(count.astype("u1"))
            line = self._line("ok", None, period, record, have)
            # The digests of the two tensors are taken by seal(), after the last record was read.
            line.update(mean={"index": len(self.tensors), "shape": list(self.shape), "dtype": "<c8", "sha256": None},
                        count={"dtype": "u1", "sha256": None})
            self.last = (stored, counted)
            self.tensors.append((line["line"], stored, counted))
        line.update(facts)
        return line

    def push(self, record):
        """Take the next record of the ring. Returns the outcome lines it causes, oldest first.

        A record outside the slots of the specification causes none. An answer carries the time of
        the stream's clock at which it was complete.
        """
        self.counts["records"] += 1
        problem = refusal(record, self.spec)
        if problem is not None:
            return [self._line("rejected", problem, None, record, 0)]
        header = record.header
        key, slot = (header["epoch"], header["episode"]), header["absolute_slot"]
        if key != self.key or slot <= self.last_slot:
            # Another epoch, another episode or a time that went back: a new history.
            if self.key is not None:
                self.counts["resets"] += 1
            self._reset()
            self.key = key
        self.last_slot = slot
        period, position = divmod(slot, self.period_slots)
        if position not in self.slots:
            # A slot outside the window may carry another allocation. It is not looked at further.
            self.counts["passed"] += 1
            return []
        lines = []
        if self.period is not None and period != self.period:
            # A window whose last slot never came is closed by the first record of a later period.
            lines.append(self._close(self.period, None))
            self.samples = {}
        self.period = period
        values, valid = planes(record, self.spec)
        rx = header["rx_count"]
        self.samples[position] = {
            "values": values, "valid": valid,
            "offset": place(header["first_subcarrier"], self.spec["support"], header["subcarrier_count"]),
            "elements": tuple(sorted(header["antenna_map"][:rx])), "steps": steps_of(header), "port": header["port"],
            "scale": header["scale"], "sequence": header["sequence"], "opportunity": f"{header['opportunity']:016x}",
            "absolute_slot": slot, "t_first_sample": header["t_first_sample"], "t_last_sample": header["t_last_sample"],
            "evidence": obs.EVIDENCE.get(header["evidence"], "unknown")}
        if position == self.slots[-1]:
            lines.append(self._close(period, record))
            self.samples, self.period = {}, None
        else:
            lines.append(self._line("insufficient_history", "the window is filling", period, record, len(self.samples)))
        return lines

    def open_samples(self):
        """Slots held for a window that did not close: (have, need)."""
        return len(self.samples), self.need


def comparable_window(line):
    """A window line without the host times of its run: what two runs on the same records must agree on."""
    return {key: value for key, value in line.items() if key not in VARYING_WINDOW}


def seal(lines, stream):
    """Take the digests of the tensors of every complete window and write them into its line. Returns the lines.

    A stream computes no digest while it reads: a hash proves a window after the run and has no
    place between a record and a decision. This is called once the last record was read.
    """
    by_line = {line["line"]: line for line in lines}
    for number, mean, count in stream.tensors:
        line = by_line.get(number)
        if line is not None:
            line["mean"]["sha256"], line["count"]["sha256"] = _sha256(mean.tobytes()), _sha256(count.tobytes())
    return lines


def run_records(records, spec, clock=None):
    """Push records through a new stream in the given order and seal the lines. Returns (lines, stream)."""
    stream, lines = WindowStream(spec, clock), []
    for record in records:
        lines += stream.push(record)
    return seal(lines, stream), stream


def stacked(stream):
    """The tensors of every complete window of a stream: (means [N, A, K, S] complex64, counts [N, A, K, S] uint8)."""
    shape = tuple(stream.shape)
    if not stream.tensors:
        return np.zeros((0,) + shape, dtype="<c8"), np.zeros((0,) + shape, dtype="u1")
    return np.stack([mean for _, mean, _ in stream.tensors]), np.stack([count for _, _, count in stream.tensors])


def warm_up(spec):
    """Run one window of noise through every step of the front end. Returns the time it took in ms.

    The first use of an operator loads its code. A sensor pays that before it attaches, so that
    no record waits for it. Nothing of this window is kept.
    """
    started = time.perf_counter()
    rx, positions, symbols = shape_of(spec)
    generator = np.random.Generator(np.random.PCG64(0))
    values = generator.standard_normal((3, rx, positions, symbols)) + 1j * generator.standard_normal((3, rx, positions,
                                                                                                    symbols))
    valid = np.isfinite(values.real)
    np.unpackbits(np.packbits(valid.reshape(-1), bitorder="little"), bitorder="little")
    mean, count = coherent_mean(values, valid)
    stored = mean.astype("<c8")
    _sha256(stored.tobytes())
    d, good = background_subtract(stored, count, stored * 0.5, count)
    zscore(d, good, 0.0, 1.0)
    return 1000.0 * (time.perf_counter() - started)


# ---------------------------------------------------------------- window output
def write_windows(directory, spec, lines, stream, source):
    """Write what the window stream of a run leaves: windows.jsonl and windows.json.

    The directory must exist and hold neither. The digests of the window tensors are taken here,
    after the last record was read. The tensors themselves are not written: a line names their
    SHA-256, and a sample set or a sensor output holds them.
    """
    directory = Path(directory)
    if not directory.is_dir() or any((directory / name).exists() for name in (WINDOWS_FILE, WINDOWS_REPORT_FILE)):
        raise CuSenseError(f"the output directory {directory} does not exist or holds a result already")
    seal(lines, stream)
    (directory / WINDOWS_FILE).write_text("".join(json.dumps(line, sort_keys=True) + "\n" for line in lines),
                                          encoding="ascii")
    have, need = stream.open_samples()
    report = {"schema": WINDOWS_REPORT_SCHEMA, "spec": spec, "spec_sha256": stream.sha256,
              "steps": {stage: entry for stage, entry in resolve(spec).items() if stage in ("input",) + STEPS[:2]},
              "runtime": {"entry": RUNTIME, "device": "cpu", "python": platform.python_version(), "numpy": np.__version__,
                          "machine": platform.machine(), "module_sha256": _sha256(Path(__file__).read_bytes())},
              "input_fields": list(INPUT_FIELDS), "source": source, "counts": dict(stream.counts), "lines": len(lines),
              "windows": len(stream.tensors), "shape": list(stream.shape), "open_at_end": {"have": have, "need": need}}
    (directory / WINDOWS_REPORT_FILE).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return report


class Windows:
    """A verified window output."""

    def __init__(self, directory, report, lines):
        self.directory, self.report, self.lines = directory, report, lines
        self.spec = report["spec"]
        self.runtime = report["runtime"]["entry"]
        self.ok = [line for line in lines if line["status"] == "ok"]
        self.by_window = {line["window"]["id"]: line for line in self.ok}


def open_windows(directory):
    """Verify a window output and return it.

    The report names the specification with its hash and its runtime: the reference, or the
    native core of Sprint 3.6. The lines count from zero, each names a known status of that
    runtime and the hash of the specification, and their counts are those of the report. A
    complete window names the slots of its period in order, one record per slot, and the hash of
    its mean and of its count. A native sensor in the evidence mode timed leaves the mean on its
    device: its report says so, and its lines name no digest of it. The directory holds nothing
    else but the archive of the records.
    """
    directory = Path(directory)
    try:
        report = json.loads((directory / WINDOWS_REPORT_FILE).read_text(encoding="ascii"))
        lines = [json.loads(text) for text in (directory / WINDOWS_FILE).read_text(encoding="ascii").splitlines()
                 if text.strip()]
    except (OSError, ValueError) as error:
        raise CuSenseError(f"{directory} is not a complete window output: {error}") from None
    if not isinstance(report, dict) or report.get("schema") != WINDOWS_REPORT_SCHEMA:
        raise CuSenseError(f"{directory / WINDOWS_REPORT_FILE} is not a {WINDOWS_REPORT_SCHEMA} report")
    spec = check_spec(report.get("spec"))
    runtime = (report.get("runtime") or {}).get("entry")
    if report.get("spec_sha256") != spec_sha256(spec) or runtime not in RUNTIMES:
        raise CuSenseError("the report does not name its specification and the reference runtime or the native one")
    statuses = STATUSES + (NATIVE_STATUSES if runtime == NATIVE_RUNTIME else ())
    timed = runtime == NATIVE_RUNTIME and (report.get("source") or {}).get("evidence") == "timed"
    counts, complete = dict.fromkeys(statuses, 0), 0
    slots, period_slots = spec["window"]["slots_in_period"], spec["radio"]["period_slots"]
    for number, line in enumerate(lines):
        if not isinstance(line, dict) or line.get("schema") != WINDOW_SCHEMA or line.get("line") != number \
                or line.get("status") not in statuses or line.get("spec_sha256") != report["spec_sha256"] \
                or line.get("runtime") != runtime:
            raise CuSenseError(f"line {number} of {WINDOWS_FILE} is not a {WINDOW_SCHEMA} line of this specification")
        counts[line["status"]] += 1
        if line["status"] != "ok":
            if "mean" in line or "count" in line:
                raise CuSenseError(f"line {number} has the status {line['status']} and names a window tensor")
            continue
        try:
            digests = (line["mean"]["sha256"], line["count"]["sha256"])
            good = line["mean"]["index"] == complete and line["mean"]["shape"] == shape_of(spec) \
                and (digests == (None, None) if timed else all(len(value) == 64 for value in digests)) \
                and [item["absolute_slot"] for item in line["contributing"]] \
                == [line["window"]["period"] * period_slots + slot for slot in slots] \
                and line["history"] == {"have": len(slots), "need": len(slots)} \
                and line["trigger"]["sequence"] == line["contributing"][-1]["sequence"] \
                and line["validity"]["valid"] + line["validity"]["holes"] == line["validity"]["elements"]
        except (KeyError, TypeError, IndexError) as error:
            raise CuSenseError(f"line {number}: the complete window is not whole: {error!r}") from None
        if not good:
            raise CuSenseError(f"line {number}: the window does not name the slots of its period in order with the "
                               "hashes of its tensors")
        complete += 1
    if any(report["counts"].get(status) != counts[status] for status in statuses) or report.get("lines") != len(lines) \
            or report.get("windows") != complete:
        raise CuSenseError("the counts of the report are not those of its lines")
    extra = sorted(path.name for path in directory.iterdir()
                   if path.name not in (WINDOWS_FILE, WINDOWS_REPORT_FILE, RECORDS_DIR))
    if extra:
        raise CuSenseError(f"the output holds {extra[0]}, which its report does not list")
    return Windows(directory, report, lines)


# ---------------------------------------------------------------- samples
def check_sample(sample):
    """Check one cusense-sample/v1 row and return it."""
    _fields(sample, SAMPLE_FIELDS, "a sample")
    if sample["schema"] != SAMPLE_SCHEMA or sample["split"] not in SPLITS or type(sample["label"]) is not int \
            or sample["label"] not in (0, 1):
        raise CuSenseError("a sample is a cusense-sample/v1 row with a split and a label of 0 or 1")
    for name in ("window", "episode", "scene", "pair", "spec_sha256", "mean_sha256", "count_sha256"):
        if not isinstance(sample[name], str) or not sample[name]:
            raise CuSenseError(f"the {name} of a sample must be a nonempty string")
    cell, position = sample["cell"], sample["position_m"]
    if sample["label"] == 1:
        if not isinstance(cell, list) or len(cell) != 2 or any(type(value) is not int or value < 0 for value in cell) \
                or not isinstance(position, list) or len(position) != 2 \
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in position):
            raise CuSenseError(f"the target-present window {sample['window']} needs the cell and the position of its "
                               "target")
    elif cell is not None or position is not None:
        raise CuSenseError(f"the target-absent window {sample['window']} names a position: it has none")
    if not isinstance(sample["shape"], list) or len(sample["shape"]) != 3 \
            or any(type(value) is not int or value < 1 for value in sample["shape"]) \
            or type(sample["index"]) is not int or sample["index"] < 0 \
            or not isinstance(sample["elements"], list) or len(sample["elements"]) != sample["shape"][0] \
            or not isinstance(sample["producer_steps"], list):
        raise CuSenseError(f"the window {sample['window']} does not name the shape, the place, the array elements and "
                           "the producer steps of its tensors")
    return sample


class SampleSet:
    """The rows of one sample file with the tensors of their windows."""

    def __init__(self, path, rows, means, counts, files):
        self.path, self.rows, self.means, self.counts, self.files = path, rows, means, counts, files


def sample_files(path):
    """The three files of a sample set: the rows, the means and the counts."""
    path = Path(path)
    return path, path.with_suffix(".c64"), path.with_suffix(".u1")


def write_samples(path, rows, means, counts):
    """Write a sample set: the rows, one per line, and the tensors of their windows. No file may exist.

    means is complex64 [N, A, K, S] and counts uint8 of the same shape, in the order of the rows:
    row n names index n and the SHA-256 of its two tensors.
    """
    names = sample_files(path)
    if any(name.exists() for name in names):
        raise CuSenseError(f"{names[0]} or one of its tensor files already exists. A sample set is never overwritten.")
    means, counts = np.ascontiguousarray(means, dtype="<c8"), np.ascontiguousarray(counts, dtype="u1")
    if means.shape != counts.shape or means.ndim != 4 or len(rows) != means.shape[0]:
        raise CuSenseError("a sample set holds one mean and one count of the shape [A, K, S] per row")
    for index, row in enumerate(rows):
        check_sample(row)
        if row["index"] != index or row["shape"] != list(means.shape[1:]) \
                or row["mean_sha256"] != _sha256(means[index].tobytes()) \
                or row["count_sha256"] != _sha256(counts[index].tobytes()):
            raise CuSenseError(f"the row of the window {row['window']} does not name its place and the hashes of its "
                               "tensors")
    names[0].parent.mkdir(parents=True, exist_ok=True)
    names[1].write_bytes(means.tobytes())
    names[2].write_bytes(counts.tobytes())
    names[0].write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="ascii")
    return names[0]


def read_samples(path):
    """Read a sample set and verify every tensor against the hash its row names."""
    names = sample_files(path)
    try:
        rows = [json.loads(text) for text in names[0].read_text(encoding="ascii").splitlines() if text.strip()]
        data, bits = names[1].read_bytes(), names[2].read_bytes()
    except (OSError, ValueError) as error:
        raise CuSenseError(f"cannot read the samples {path}: {error}") from None
    for row in rows:
        check_sample(row)
    if not rows:
        raise CuSenseError(f"the sample file {path} holds no row")
    shape = tuple(rows[0]["shape"])
    size = int(np.prod(shape))
    if any(tuple(row["shape"]) != shape for row in rows) or [row["index"] for row in rows] != list(range(len(rows))) \
            or len(data) != 8 * size * len(rows) or len(bits) != size * len(rows):
        raise CuSenseError(f"the tensor files of {path} do not hold one window of the shape {list(shape)} per row")
    means = np.frombuffer(data, dtype="<c8").reshape((len(rows),) + shape)
    counts = np.frombuffer(bits, dtype="u1").reshape((len(rows),) + shape)
    for index, row in enumerate(rows):
        if row["mean_sha256"] != _sha256(means[index].tobytes()) or row["count_sha256"] != _sha256(counts[index].tobytes()):
            raise CuSenseError(f"the tensors of the window {row['window']} do not have the hashes its row names")
    files = {name.name: {"bytes": name.stat().st_size, "sha256": _sha256(name.read_bytes())} for name in names}
    return SampleSet(names[0], rows, means, counts, files)


def check_split(training, validation):
    """Refuse a sample set that is not split before its windows. Returns the facts of the split.

    Every row of a file carries the split of that file. No window, episode key, pair or scene
    appears in both, every row was computed with one specification on one array, and no row of
    the held-out split is in either.
    """
    seen = {}
    for role, rows in (("training", training), ("validation", validation)):
        for row in rows:
            check_sample(row)
            if row["split"] != role:
                raise CuSenseError(f"the window {row['window']} of the {row['split']} split is among the {role} "
                                   "samples" + (": a held-out sample is read after the freeze and never before"
                                                if row["split"] == "held_out" else ""))
        seen[role] = {name: {row[name] for row in rows} for name in ("window", "episode", "pair", "scene")}
        if len(seen[role]["window"]) != len(rows):
            raise CuSenseError(f"a window is among the {role} samples twice")
    for name, text in (("scene", "scene"), ("pair", "matched pair"), ("episode", "episode key"), ("window", "window")):
        both = sorted(seen["training"][name] & seen["validation"][name])
        if both:
            raise CuSenseError(f"the {text} {both[0]} is in the training and in the validation split: a split is made "
                               "by scene before any window is cut")
    rows = list(training) + list(validation)
    if not rows:
        raise CuSenseError("there is no sample")
    first = rows[0]
    if any((row["spec_sha256"], row["shape"], row["elements"], row["producer_steps"])
           != (first["spec_sha256"], first["shape"], first["elements"], first["producer_steps"]) for row in rows):
        raise CuSenseError("the samples were not computed with one specification, on one array and by one chain of "
                           "producer steps")
    return {"spec_sha256": first["spec_sha256"], "shape": first["shape"], "elements": first["elements"],
            "producer_steps": first["producer_steps"],
            "scenes": {role: sorted(seen[role]["scene"]) for role in seen},
            "pairs": {role: len(seen[role]["pair"]) for role in seen}}


def _only(samples, split, purpose):
    for sample in samples:
        if sample["split"] != split:
            raise CuSenseError(f"the window {sample['window']} of the {sample['split']} split was given to "
                               f"{purpose}, which reads the {split} split and no other sample")


def _fitted_on(samples):
    return {"split": "training", "windows": len(samples), "windows_sha256": ids_sha256(row["window"] for row in samples)}


def check_fitted_on(record, name):
    """Refuse statistics that were fitted on anything but the training split."""
    _fields(record, FITTED_ON, f"{name}.fitted_on")
    if record["split"] != "training" or type(record["windows"]) is not int or record["windows"] < 1 \
            or not isinstance(record["windows_sha256"], str) or len(record["windows_sha256"]) != 64:
        raise CuSenseError(f"{name} names the {record['split']} split: the template and the statistics of the z-score "
                           "are fitted on windows of the training split and on no other record")
    return record


# ---------------------------------------------------------------- stage 1 and the statistics of stage 2
def fit_template(samples, means, counts):
    """The background template from the target-absent windows of the training split.

    Returns {"values", "count", "fitted_on"}: the complex template [A, K, S] as complex64, the
    count of records behind each element as uint32, and what it was fitted on. A window holds
    the mean and the count of its records per element, so the count-weighted mean of the windows
    is the mean of the calibration records, over their valid elements. A sample of another
    split and a target-present sample are refused.
    """
    _only(samples, "training", "the fit of the template")
    for sample in samples:
        if sample["label"] != 0:
            raise CuSenseError(f"the target-present window {sample['window']} was given to the fit of the template, "
                               "which reads target-absent calibration windows")
    if not len(samples) or len(samples) != len(means) or len(samples) != len(counts):
        raise CuSenseError("the template needs a calibration window, with one mean and one count per window")
    values, count = merge_means(means, counts)
    if count.max() >= 1 << 32:
        raise CuSenseError("the template counts more records than a uint32 holds")
    return {"values": np.ascontiguousarray(values.astype("<c8")), "count": np.ascontiguousarray(count.astype("<u4")),
            "fitted_on": _fitted_on(samples)}


def differences_of(means, counts, template):
    """Stage 2 before the z-score for a set of windows: (d [N, A, K] float64, valid [N, A, K] bool)."""
    made = [background_subtract(mean, count, template["values"], template["count"]) for mean, count in zip(means, counts)]
    if not made:
        raise CuSenseError("there is no window")
    return np.stack([d for d, _ in made]), np.stack([valid for _, valid in made])


def fit_normalizer(samples, means, counts, template):
    """Mean and deviation of the z-score over the training split. Returns the normalizer record without its hash.

    One mean and one population deviation over every window, antenna and valid position of the
    training split, target-present and target-absent. A sample of another split is refused. A
    set that does not vary keeps the scale 1 and says so.
    """
    _only(samples, "training", "the fit of the normalizer")
    if not len(samples) or len(samples) != len(means):
        raise CuSenseError("the normalizer needs a training window")
    d, valid = differences_of(means, counts, template)
    values = d[valid]
    if values.size == 0:
        raise CuSenseError("no position of a training window holds a valid sample")
    mean, deviation = float(values.mean()), float(values.std())
    constant = not deviation > 1e-12 * max(1.0, abs(mean))
    return {"entry": "zscore_global", "mean": mean, "scale": 1.0 if constant else deviation, "epsilon": EPSILON,
            "constant": constant, "elements": int(values.size), "fitted_on": _fitted_on(samples)}


def inputs_of(means, counts, template, normalizer):
    """The input of the model for a set of windows: float32 [N, A, K]."""
    d, valid = differences_of(means, counts, template)
    return zscore(d, valid, normalizer["mean"], normalizer["scale"], normalizer["epsilon"])


def normalizer_sha256(normalizer):
    return _sha256(canonical({name: normalizer[name] for name in ("entry", "mean", "scale", "epsilon")}))


# ---------------------------------------------------------------- grid
def grid_record(geometry):
    """The grid part of an artifact from the position grid of an episode set, with its units and its hash."""
    try:
        made = {"rows": geometry["rows"], "cols": geometry["cols"], "cell_m": [float(v) for v in geometry["cell_m"]],
                "origin_m": [float(v) for v in geometry["origin_m"]],
                "area": {axis: [float(v) for v in geometry["area"][axis]] for axis in ("x_m", "y_m")},
                "rule": geometry["rule"], "units": dict(GRID_UNITS)}
    except (KeyError, TypeError, ValueError) as error:
        raise CuSenseError(f"the position grid is incomplete: {error!r}") from None
    made["sha256"] = part_sha256(made)
    return check_grid(made)


def check_grid(grid):
    _fields(grid, GRID_FIELDS, "grid")
    rows, cols = _integer(grid["rows"], "grid.rows", 1, 4096), _integer(grid["cols"], "grid.cols", 1, 4096)
    try:
        (x0, x1), (y0, y1) = grid["area"]["x_m"], grid["area"]["y_m"]
        height, width = grid["cell_m"]
        origin = list(grid["origin_m"])
    except (KeyError, TypeError, ValueError):
        raise CuSenseError("grid requires area.x_m, area.y_m, cell_m and origin_m, each of two numbers") from None
    if not (x0 < x1 and y0 < y1) or origin != [x0, y0] or not math.isclose(height, (y1 - y0) / rows, rel_tol=1e-9) \
            or not math.isclose(width, (x1 - x0) / cols, rel_tol=1e-9) or grid["units"] != GRID_UNITS \
            or not isinstance(grid["rule"], str):
        raise CuSenseError("the grid does not follow from its area: rows along y, columns along x, the origin at the "
                           "lower corner, each in metres")
    if grid["sha256"] != part_sha256(grid):
        raise CuSenseError("the grid does not have the hash it names: the geometry changed")
    return grid


def cell_centre(grid, cell):
    """The centre of a cell in metres: [x, y]."""
    row, col = cell
    if not (0 <= row < grid["rows"] and 0 <= col < grid["cols"]):
        raise CuSenseError(f"the cell {list(cell)} lies outside the grid of {grid['rows']} by {grid['cols']}")
    return [grid["origin_m"][0] + (col + 0.5) * grid["cell_m"][1], grid["origin_m"][1] + (row + 0.5) * grid["cell_m"][0]]


def locate(probability_map, grid):
    """The estimate of one map: the cell of its largest value, the first one in row order, and its centre."""
    values = np.asarray(probability_map)
    if values.shape != (grid["rows"], grid["cols"]):
        raise CuSenseError(f"a map of the shape {list(values.shape)} is not the grid of {grid['rows']} by {grid['cols']}")
    row, col = (int(index) for index in np.unravel_index(int(np.argmax(values)), values.shape))
    return {"cell": [row, col], "position_m": cell_centre(grid, (row, col)), "probability": float(values[row, col])}


# ---------------------------------------------------------------- optional stage
def check_smoothing(smoothing):
    """Check the settings of the optional stage and return them. The entry "none" has no other field."""
    if isinstance(smoothing, dict) and smoothing.get("entry") == "none":
        if set(smoothing) != {"entry"}:
            raise CuSenseError('the smoothing entry "none" takes no setting')
        return smoothing
    _fields(smoothing, SMOOTHING_FIELDS, "smoothing")
    if smoothing["entry"] != "map_mean_kalman":
        raise CuSenseError(f"unknown smoothing entry {smoothing['entry']!r} (registered: "
                           f"{', '.join(sorted(REGISTRY['smoothing']))})")
    _integer(smoothing["maps"], "smoothing.maps", 1, 4096)
    _number(smoothing["process_noise_m2_s3"], "smoothing.process_noise_m2_s3", above=0)
    _number(smoothing["measurement_noise_m2"], "smoothing.measurement_noise_m2", above=0)
    _number(smoothing["initial_speed_m_s"], "smoothing.initial_speed_m_s", above=0)
    return smoothing


class Smoother:
    """The optional stage map_mean_kalman on the maps of one history, one window at a time.

    update() takes the map of a window and returns the mean of the newest maps of its history,
    the cell of the largest value of that mean, and the position after a constant-velocity Kalman
    filter with that cell's centre as its measurement. A new epoch, a new episode key and a period
    that does not follow start a new history. It reads maps of the present and the past only.
    """

    def __init__(self, settings, grid, period_seconds):
        self.settings, self.grid, self.period_s = check_smoothing(settings), grid, period_seconds
        self.active = settings["entry"] != "none"
        self.reset()

    def reset(self):
        self.key = self.period = self.state = self.covariance = None
        self.maps = []

    def update(self, key, period, probability_map):
        if not self.active:
            return None
        settings = self.settings
        if key != self.key or self.period is None or period <= self.period:
            self.reset()
            self.key = key
        steps = 0 if self.period is None else period - self.period
        self.period = period
        self.maps = (self.maps + [np.asarray(probability_map, dtype=np.float64)])[-settings["maps"]:]
        mean = np.mean(self.maps, axis=0)
        found = locate(mean, self.grid)
        z = np.array(found["position_m"])
        r = settings["measurement_noise_m2"]
        if self.state is None:
            self.state = np.array([z[0], z[1], 0.0, 0.0])
            self.covariance = np.diag([r, r, settings["initial_speed_m_s"] ** 2, settings["initial_speed_m_s"] ** 2])
        else:
            dt, q = steps * self.period_s, settings["process_noise_m2_s3"]
            move = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
            noise = q * np.array([[dt ** 3 / 3, 0, dt ** 2 / 2, 0], [0, dt ** 3 / 3, 0, dt ** 2 / 2],
                                  [dt ** 2 / 2, 0, dt, 0], [0, dt ** 2 / 2, 0, dt]])
            state, covariance = move @ self.state, move @ self.covariance @ move.T + noise
            look = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
            gain = covariance @ look.T @ np.linalg.inv(look @ covariance @ look.T + r * np.eye(2))
            self.state = state + gain @ (z - look @ state)
            self.covariance = (np.eye(4) - gain @ look) @ covariance
        return {"entry": settings["entry"], "maps": len(self.maps), "cell": found["cell"],
                "position_m": found["position_m"], "probability": found["probability"],
                "kalman_position_m": [float(self.state[0]), float(self.state[1])],
                "kalman_velocity_m_s": [float(self.state[2]), float(self.state[3])]}


# ---------------------------------------------------------------- artifact
def identity(manifest):
    """The hash of an artifact: SHA-256 over every field of model.json but this one, in canonical form."""
    return part_sha256(manifest)


def check_manifest(manifest):
    """Check a cusense-artifact/v1 object and return it. The three binary files are checked by open_artifact."""
    _fields(manifest, ARTIFACT_FIELDS, "an artifact")
    if manifest["schema"] != ARTIFACT_SCHEMA:
        raise CuSenseError(f"unsupported artifact schema {manifest['schema']!r}")
    if not isinstance(manifest["model_id"], str) or not manifest["model_id"] or not manifest["model_id"].isascii():
        raise CuSenseError("model_id must be a nonempty ASCII string")
    entries = _fields(manifest["entries"], ENTRY_FIELDS, "entries")
    for stage in ENTRY_FIELDS:
        if entries[stage] not in REGISTRY[stage]:
            raise CuSenseError(f"unknown {stage} entry {entries[stage]!r} (registered: "
                               f"{', '.join(sorted(REGISTRY[stage]))})")
    wanted = _fields(manifest["input"], ARTIFACT_INPUT, "input")
    spec = check_spec(wanted["spec"])
    rx, positions, symbols = shape_of(spec)
    steps = spec["steps"]
    if wanted["spec_sha256"] != spec_sha256(spec) or wanted["shape"] != [1, rx, positions] or wanted["dtype"] != "<f4" \
            or not isinstance(wanted["elements"], list) or len(wanted["elements"]) != rx \
            or wanted["elements"] != sorted(set(wanted["elements"])) or not isinstance(wanted["producer_steps"], list):
        raise CuSenseError(f"the input of a model is one window of its specification as <f4: shape [1, {rx}, "
                           f"{positions}], on {rx} array elements in rising order")
    if (entries["input"], entries["ingest"], entries["window"], entries["clutter"], entries["symbols"],
            entries["normalizer"]) != (spec["input"]["entry"], steps["ingest"], steps["window"], steps["clutter"],
                                       steps["symbols"], steps["normalizer"]):
        raise CuSenseError("the entries of the artifact are not the steps of its specification")
    template = _fields(manifest["template"], TEMPLATE_FIELDS, "template")
    size = rx * positions * symbols
    if (template["file"], template["dtype"], template["shape"], template["bytes"], template["count_file"],
            template["count_dtype"], template["count_bytes"]) != (TEMPLATE_FILE, "<c8", [rx, positions, symbols],
                                                                  8 * size, TEMPLATE_COUNT_FILE, "<u4", 4 * size):
        raise CuSenseError(f"the template of a model is [{rx}, {positions}, {symbols}] as <c8 in {TEMPLATE_FILE} with "
                           f"its counts as <u4 in {TEMPLATE_COUNT_FILE}")
    check_fitted_on(template["fitted_on"], "the template")
    normalizer = _fields(manifest["normalizer"], NORMALIZER_FIELDS, "normalizer")
    _number(normalizer["mean"], "normalizer.mean")
    _number(normalizer["scale"], "normalizer.scale", above=0)
    if normalizer["entry"] != entries["normalizer"] or normalizer["epsilon"] != EPSILON:
        raise CuSenseError("the normalizer must be the entry of the artifact with its epsilon")
    if normalizer["sha256"] != normalizer_sha256(normalizer):
        raise CuSenseError("the normalizer does not have the hash the artifact names: the preprocessing state changed")
    check_fitted_on(normalizer["fitted_on"], "the normalizer")
    grid = check_grid(manifest["grid"])
    model = _fields(manifest["model"], MODEL_FIELDS, "model")
    try:
        config = cusense_cnn.check_config(model["config"])
    except ModelError as error:
        raise CuSenseError(str(error)) from None
    if (config["antennas"], config["rows"], config["cols"]) != (rx, grid["rows"], grid["cols"]) \
            or config["entry"] != entries["model"] or model["config_sha256"] != cusense_cnn.config_sha256(config) \
            or model["heads"] != list(cusense_cnn.HEADS) or model["losses"] != cusense_cnn.LOSSES:
        raise CuSenseError("the model is not the entry of the artifact on the array of its specification and on its "
                           "grid, with the heads and the losses of this module")
    target = _fields(model["target"], ("rule", "sigma_cells", "sigma_m"), "model.target")
    if target["rule"] != TARGET_RULE or type(target["sigma_cells"]) not in (int, float) or target["sigma_cells"] < 0:
        raise CuSenseError("model.target names the rule of the training target and its smoothing in cells")
    weights = _fields(manifest["weights"], WEIGHTS_FIELDS, "weights")
    tensors = weights["tensors"]
    if (weights["file"], weights["dtype"], weights["layout"]) != (WEIGHTS_FILE, "<f4", WEIGHTS_LAYOUT) \
            or not isinstance(tensors, list) or not tensors or type(weights["count"]) is not int \
            or weights["count"] != sum(item.get("count", 0) for item in tensors if isinstance(item, dict)) \
            or weights["bytes"] != 4 * weights["count"] or type(model["parameters"]) is not int:
        raise CuSenseError(f"the weights of a model are its tensors as <f4 in {WEIGHTS_FILE}, one after another")
    threshold = _fields(manifest["threshold"], THRESHOLD_FIELDS, "threshold")
    if threshold["entry"] != entries["detector"] or threshold["rule"] != presence.THRESHOLD_RULE \
            or any(type(threshold[name]) not in (int, float) or not math.isfinite(threshold[name])
                   for name in ("logit", "probability", "target_false_alarm")) \
            or not 0 <= threshold["target_false_alarm"] < 1 or not 0 <= threshold["probability"] <= 1 \
            or any(type(threshold[name]) is not int or threshold[name] < 0 for name in ("negatives", "allowed", "exceeding")) \
            or threshold["exceeding"] > threshold["allowed"] or threshold["negatives"] < 1 \
            or threshold["allowed"] != math.floor(threshold["target_false_alarm"] * threshold["negatives"]):
        raise CuSenseError("the threshold does not follow its rule: " + presence.THRESHOLD_RULE)
    if threshold["sha256"] != part_sha256(threshold):
        raise CuSenseError("the threshold does not have the hash it names: it was moved after it was frozen")
    smoothing = check_smoothing(manifest["smoothing"])
    if smoothing["entry"] != entries["smoothing"]:
        raise CuSenseError("the optional stage is not the entry of the artifact")
    if manifest["output"] != OUTPUT:
        raise CuSenseError("the output of a model is the input, the map, the location, the presence and the decision "
                           "of this module")
    if manifest["sha256"] != identity(manifest):
        raise CuSenseError("model.json does not have the hash it names: the artifact was altered after it was frozen")
    return manifest


class Artifact:
    """A verified cusense-artifact/v1 artifact: its manifest, its template and the bytes of its weights."""

    def __init__(self, directory, manifest, template, template_count, weights):
        self.directory, self.manifest, self.weights = directory, manifest, weights
        self.template = {"values": template, "count": template_count}
        self.sha256 = manifest["sha256"]
        self.spec, self.grid = manifest["input"]["spec"], manifest["grid"]
        self.spec_sha256 = manifest["input"]["spec_sha256"]
        self.threshold = manifest["threshold"]["logit"]

    def model(self):
        """What a decision names of its model."""
        manifest = self.manifest
        return {"model_id": manifest["model_id"], "entry": manifest["entries"]["model"], "artifact_sha256": self.sha256,
                "weights_sha256": manifest["weights"]["sha256"], "template_sha256": manifest["template"]["sha256"],
                "normalizer_sha256": manifest["normalizer"]["sha256"], "grid_sha256": manifest["grid"]["sha256"],
                "threshold_sha256": manifest["threshold"]["sha256"]}


def threshold_record(threshold):
    """The threshold part of an artifact from the record that the calibration returns, with its hash."""
    made = {name: threshold[name] for name in presence.THRESHOLD_FIELDS}
    made["sha256"] = part_sha256(made)
    return made


def write_artifact(directory, model_id, spec, elements, producer_steps, template, normalizer, net, grid, threshold,
                   smoothing, sigma_cells):
    """Write a cusense-artifact/v1 artifact into a new directory and return its manifest.

    template is what fit_template returns, normalizer what fit_normalizer returns, net a model of
    cusense_cnn, grid a grid record, threshold the record of presence.calibrate_threshold and
    smoothing the settings of the optional stage. The directory appears only when it is complete.
    """
    directory = Path(directory)
    if directory.exists():
        raise CuSenseError(f"{directory} already exists. An artifact is never overwritten.")
    check_spec(spec)
    values = np.ascontiguousarray(template["values"], dtype="<c8")
    count = np.ascontiguousarray(template["count"], dtype="<u4")
    if list(values.shape) != shape_of(spec) or values.shape != count.shape:
        raise CuSenseError(f"the template has the shape {list(values.shape)}, and a window of the specification "
                           f"{shape_of(spec)}")
    if not np.isfinite(values.real).all() or not np.isfinite(values.imag).all():
        raise CuSenseError("the template holds a value that is not finite")
    try:
        data, layout = cusense_cnn.pack(net)
        config = cusense_cnn.check_config(net.config)
    except ModelError as error:
        raise CuSenseError(str(error)) from None
    held = count[count > 0]
    mean, scale = _number(normalizer["mean"], "normalizer.mean"), _number(normalizer["scale"], "normalizer.scale")
    described = {"entry": "zscore_global", "mean": mean, "scale": scale, "epsilon": EPSILON}
    manifest = {
        "schema": ARTIFACT_SCHEMA, "model_id": model_id,
        "entries": {"input": spec["input"]["entry"], "ingest": spec["steps"]["ingest"], "window": spec["steps"]["window"],
                    "clutter": spec["steps"]["clutter"], "symbols": spec["steps"]["symbols"],
                    "normalizer": spec["steps"]["normalizer"], "model": config["entry"], "detector": "frozen_threshold",
                    "smoothing": smoothing["entry"]},
        "input": {"spec": spec, "spec_sha256": spec_sha256(spec), "elements": list(elements),
                  "producer_steps": list(producer_steps), "shape": [1] + shape_of(spec)[:2], "dtype": "<f4"},
        "template": {"file": TEMPLATE_FILE, "dtype": "<c8", "shape": list(values.shape), "bytes": values.nbytes,
                     "sha256": _sha256(values.tobytes()), "count_file": TEMPLATE_COUNT_FILE, "count_dtype": "<u4",
                     "count_bytes": count.nbytes, "count_sha256": _sha256(count.tobytes()),
                     "valid_elements": int((count > 0).sum()), "records_min": int(held.min()) if held.size else 0,
                     "records_max": int(count.max()) if count.size else 0, "fitted_on": dict(template["fitted_on"])},
        "normalizer": dict(described, sha256=normalizer_sha256(described), fitted_on=dict(normalizer["fitted_on"])),
        "model": {"config": config, "config_sha256": cusense_cnn.config_sha256(config),
                  "parameters": cusense_cnn.parameters_of(net), "heads": list(cusense_cnn.HEADS),
                  "losses": cusense_cnn.LOSSES,
                  "target": {"rule": TARGET_RULE, "sigma_cells": sigma_cells,
                             "sigma_m": [sigma_cells * grid["cell_m"][0], sigma_cells * grid["cell_m"][1]]}},
        "weights": {"file": WEIGHTS_FILE, "dtype": "<f4", "layout": WEIGHTS_LAYOUT, "tensors": layout,
                    "count": len(data) // 4, "bytes": len(data), "sha256": _sha256(data)},
        "grid": dict(grid), "threshold": threshold_record(threshold), "smoothing": dict(smoothing),
        "output": dict(OUTPUT)}
    manifest["sha256"] = identity(manifest)
    check_manifest(manifest)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory.parent) as temporary:
        staging = Path(temporary) / "artifact"
        staging.mkdir()
        (staging / TEMPLATE_FILE).write_bytes(values.tobytes())
        (staging / TEMPLATE_COUNT_FILE).write_bytes(count.tobytes())
        (staging / WEIGHTS_FILE).write_bytes(data)
        (staging / MODEL_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="ascii")
        staging.rename(directory)
    return manifest


def open_artifact(directory, sha256=None):
    """Verify an artifact and return it.

    model.json must be a cusense-artifact/v1 object with the hash it names. The template, its
    counts and the weights must have the sizes and the hashes that model.json names and hold
    finite values, and the directory must hold nothing else. With sha256 the artifact must be the
    one that a freeze record names.
    """
    directory = Path(directory)
    try:
        manifest = json.loads((directory / MODEL_FILE).read_text(encoding="ascii"))
        values = (directory / TEMPLATE_FILE).read_bytes()
        count = (directory / TEMPLATE_COUNT_FILE).read_bytes()
        data = (directory / WEIGHTS_FILE).read_bytes()
    except (OSError, ValueError) as error:
        raise CuSenseError(f"{directory} is not a complete model: {error}") from None
    check_manifest(manifest)
    template, weights = manifest["template"], manifest["weights"]
    if len(values) != template["bytes"] or _sha256(values) != template["sha256"] or len(count) != template["count_bytes"] \
            or _sha256(count) != template["count_sha256"]:
        raise CuSenseError("the template or its counts do not have the size and the hash that model.json names: the "
                           "background was altered")
    if len(data) != weights["bytes"] or _sha256(data) != weights["sha256"]:
        raise CuSenseError("weights.f32 does not have the size and the hash that model.json names: the checkpoint was "
                           "altered")
    shape = tuple(template["shape"])
    stored = np.frombuffer(values, dtype="<c8").reshape(shape)
    counted = np.frombuffer(count, dtype="<u4").reshape(shape)
    if not np.isfinite(stored.real).all() or not np.isfinite(stored.imag).all() \
            or not np.isfinite(np.frombuffer(data, dtype="<f4")).all():
        raise CuSenseError("the template or the weights hold a value that is not finite")
    extra = sorted(path.name for path in directory.iterdir() if path.name not in ARTIFACT_FILES)
    if extra:
        raise CuSenseError(f"the artifact holds {extra[0]}: a deployed model is {', '.join(ARTIFACT_FILES)} and nothing "
                           "else")
    if sha256 is not None and manifest["sha256"] != sha256:
        raise CuSenseError(f"the artifact has the hash {manifest['sha256']}, and {sha256} was asked for")
    return Artifact(directory, manifest, stored, counted, data)


# ---------------------------------------------------------------- model
class Scorer:
    """The arithmetic of a model on one device: one window of shape [1, A, K] as float32 at a time."""

    def __init__(self, config, data, layout, device="cpu"):
        self.device = _device(device)
        try:
            net = cusense_cnn.unpack(cusense_cnn.CuSenseCNN(config), data, layout)
        except ModelError as error:
            raise CuSenseError(str(error)) from None
        self.net = net.eval().to(self.device)
        self.antennas, self.rows, self.cols = net.antennas, net.rows, net.cols

    def predict(self, x):
        """Score one window. Returns (map float32 [rows, cols], logit, probability, time in ns).

        The time runs from the input in host memory to the map and both values back in host
        memory, so on a CUDA device it holds the transfer in both directions and the wait for
        the device.
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if not isinstance(x, np.ndarray):
            raise CuSenseError("a model scores a NumPy array or a tensor of shape [1, A, K] as float32")
        if x.dtype != np.float32 or x.ndim != 3 or x.shape[0] != 1 or x.shape[1] != self.antennas:
            raise CuSenseError(f"a model scores one window of shape [1, {self.antennas}, K] as float32, and this one "
                               f"has the shape {list(x.shape)} as {x.dtype}")
        if not np.isfinite(x).all():
            raise CuSenseError("the window holds a value that is not finite")
        started = time.perf_counter_ns()
        try:
            with torch.inference_mode():
                logits, presence_logit = self.net(torch.from_numpy(np.ascontiguousarray(x)).to(self.device))
                both = torch.cat((torch.softmax(logits, dim=1).reshape(-1), presence_logit,
                                  torch.sigmoid(presence_logit))).cpu()
        except ModelError as error:
            raise CuSenseError(str(error)) from None
        elapsed = time.perf_counter_ns() - started
        values = both.numpy()
        logit, probability = float(values[-2]), float(values[-1])
        if not math.isfinite(logit) or not math.isfinite(probability) or not np.isfinite(values).all():
            raise CuSenseError("the score of the window is not finite")
        return np.ascontiguousarray(values[:-2].reshape(self.rows, self.cols), dtype="<f4"), logit, probability, elapsed


class Model:
    """One frozen cuSense-like model, loaded once in a process and bound to one device."""

    LOADS = 0  # how often this process built a model from an artifact

    def __init__(self, artifact, device="cpu"):
        self.artifact = artifact if isinstance(artifact, Artifact) else open_artifact(artifact)
        manifest = self.artifact.manifest
        self.scorer = Scorer(manifest["model"]["config"], self.artifact.weights, manifest["weights"]["tensors"], device)
        self.device = self.scorer.device
        self.spec, self.grid, self.threshold = self.artifact.spec, self.artifact.grid, self.artifact.threshold
        self.shape = shape_of(self.spec)
        Model.LOADS += 1

    def smoother(self):
        """A new optional stage with the settings of the artifact."""
        return Smoother(self.artifact.manifest["smoothing"], self.grid, period_s(self.spec))

    def prepare(self, mean, count):
        """The input of the model from the tensors of one window: float32 of shape [1, A, K]."""
        mean, count = np.asarray(mean), np.asarray(count)
        if list(mean.shape) != self.shape or mean.shape != count.shape:
            raise CuSenseError(f"the window has the shape {list(mean.shape)}, and the model was trained on "
                               f"{self.shape}")
        normalizer = self.artifact.manifest["normalizer"]
        d, valid = background_subtract(mean, count, self.artifact.template["values"], self.artifact.template["count"])
        x = zscore(d, valid, normalizer["mean"], normalizer["scale"], normalizer["epsilon"])[None]
        if not np.isfinite(x).all():
            raise CuSenseError("the window holds a value that is not finite as float32")
        return np.ascontiguousarray(x, dtype="<f4")

    def predict(self, x):
        return self.scorer.predict(x)

    def score(self, mean, count):
        """Score the tensors of one window: the input, the map, the logit and the decision. No digest is taken."""
        started = time.perf_counter_ns()
        x = self.prepare(mean, count)
        frontend = time.perf_counter_ns() - started
        probability_map, logit, probability, elapsed = self.scorer.predict(x)
        return {"x": x, "map": probability_map, "logit": logit, "probability": probability,
                "present": logit > self.threshold, "frontend_ns": frontend, "infer_ns": elapsed}

    def warm_up(self, rounds=3):
        """Score a window that equals the template a few times. Returns the time it took in ms.

        The first inference of a process creates the device context and loads the libraries of
        the operators: seconds on a CUDA device. A sensor pays it before it attaches.
        """
        started = time.perf_counter()
        template = self.artifact.template
        count = np.minimum(template["count"], 1).astype("u1")
        for _ in range(max(rounds, 1)):
            self.score(template["values"], count)
        return 1000.0 * (time.perf_counter() - started)

    def decide(self, line, tensors=None, clock=None, smoother=None):
        """The decision of one line of the window stream.

        A complete window is scored from its tensors, (mean, count), and its decision names the
        contributing records, the times of its window and the frozen model. Any other line gives
        a decision without a score under the status of the line. The result is (decision, input
        or None, map or None). The digests of the input and of the map are not taken here:
        seal_decisions() takes them after the last record was read.
        """
        made = {"schema": DECISION_SCHEMA, "line": line["line"], "status": line["status"], "reason": line["reason"],
                "window": line["window"], "trigger": line["trigger"], "runtime": RUNTIME, "device": str(self.device),
                "model": self.artifact.model(), "score": None, "decision": None, "location": None, "smoothed": None,
                "input": None, "frontend_ns": None, "infer_ns": None, "t_ready_ns": line["t_ready_ns"],
                "t_publish_ns": line["t_publish_ns"], "t_window_ns": line["t_window_ns"], "t_decision_ns": None}
        x = probability_map = None
        if line["status"] == "ok":
            wanted = self.artifact.manifest["input"]
            if line["spec_sha256"] != wanted["spec_sha256"]:
                raise CuSenseError(f"the window was computed with the specification {line['spec_sha256']}, and the "
                                   f"model was trained on {wanted['spec_sha256']} ({wanted['spec']['spec_id']})")
            if line["rx"]["elements"] != wanted["elements"] or line["input"]["producer_steps"] != wanted["producer_steps"]:
                raise CuSenseError(f"the window was estimated on the array elements {line['rx']['elements']} by the "
                                   f"steps {line['input']['producer_steps']}, and the model was trained on "
                                   f"{wanted['elements']} and {wanted['producer_steps']}")
            if tensors is None:
                raise CuSenseError("a complete window is decided from its mean and its count")
            scored = self.score(*tensors)
            x, probability_map = scored["x"], scored["map"]
            found = locate(probability_map, self.grid)
            made.update(score={"logit": scored["logit"], "probability": scored["probability"]},
                        decision={"present": scored["present"], "threshold_logit": self.threshold,
                                  "detector": self.artifact.manifest["entries"]["detector"]},
                        location=dict(found, map={"shape": list(probability_map.shape), "dtype": "<f4",
                                                  "sha256": None}),
                        input={"spec_sha256": line["spec_sha256"], "entry": wanted["spec"]["input"]["entry"],
                               "shape": list(x.shape), "sha256": None},
                        frontend_ns=scored["frontend_ns"], infer_ns=scored["infer_ns"],
                        contributing=line["contributing"], time=line["time"])
            if smoother is not None and smoother.active:
                key = (line["window"]["epoch"], line["window"]["episode"])
                made["smoothed"] = smoother.update(key, line["window"]["period"], probability_map)
        made["t_decision_ns"] = None if clock is None else clock()
        return made, x, probability_map


def comparable(decision):
    """A decision without the host times, the durations and the device of its run."""
    return {key: value for key, value in decision.items() if key not in VARYING}


def input_sha256(x):
    """The digest of one input of a model: SHA-256 over its values as little-endian float32, shape [1, A, K]."""
    return _sha256(np.ascontiguousarray(x, dtype="<f4").tobytes())


def seal_decisions(decisions, inputs, maps):
    """Take the digests of the input and of the map of every scored decision, and name their place. Returns the decisions.

    inputs and maps are those of the scored decisions, in order: [A, K] and [rows, cols] as
    float32. As for the windows, no digest is taken while a sensor reads.
    """
    scored = [line for line in decisions if line["status"] == "ok"]
    if len(scored) != len(inputs) or len(scored) != len(maps):
        raise CuSenseError("there is one input and one map per scored decision")
    for index, line in enumerate(scored):
        line["input"].update(index=index, sha256=input_sha256(np.asarray(inputs[index])[None]))
        line["location"]["map"].update(index=index, sha256=_sha256(np.ascontiguousarray(maps[index], dtype="<f4").tobytes()))
    return decisions


def run_model(records, model, clock=None):
    """Push records through a new stream and score every complete window at once.

    Returns (lines, stream, decisions, inputs, maps): the inputs and the maps of the scored
    windows in the order of their decisions, with every digest taken after the last record.
    """
    stream, smoother = WindowStream(model.spec, clock), model.smoother()
    lines, decisions, inputs, maps = [], [], [], []
    for record in records:
        for line in stream.push(record):
            lines.append(line)
            decision, x, probability_map = model.decide(line, stream.last if line["status"] == "ok" else None, clock,
                                                        smoother)
            decisions.append(decision)
            if x is not None:
                inputs.append(x[0])
                maps.append(probability_map)
    return seal(lines, stream), stream, seal_decisions(decisions, inputs, maps), inputs, maps


# ---------------------------------------------------------------- training
def check_settings(settings):
    """Check a cusense-train-config/v1 object and return it."""
    _fields(settings, SETTINGS_FIELDS, "a training configuration")
    if settings["schema"] != SETTINGS_SCHEMA:
        raise CuSenseError(f"unsupported training configuration schema {settings['schema']!r}")
    if not isinstance(settings["model_id"], str) or not settings["model_id"] or not settings["model_id"].isascii():
        raise CuSenseError("model_id must be a nonempty ASCII string")
    _fields(settings["model"], MODEL_SETTINGS, "model")
    loss = _fields(settings["loss"], LOSS_SETTINGS, "loss")
    _number(loss["grid_weight"], "loss.grid_weight", low=0, high=1e6)
    _number(loss["sigma_cells"], "loss.sigma_cells", low=0, high=1e4)
    optimizer = _fields(settings["optimizer"], OPTIMIZER_SETTINGS, "optimizer")
    if optimizer["entry"] != "adam":
        raise CuSenseError('optimizer.entry is "adam"')
    _number(optimizer["learning_rate"], "optimizer.learning_rate", high=10, above=0)
    _number(optimizer["weight_decay"], "optimizer.weight_decay", low=0, high=10)
    _integer(optimizer["batch_size"], "optimizer.batch_size", 1, 1 << 20)
    _integer(optimizer["epochs"], "optimizer.epochs", 1, 100000)
    _integer(settings["seed"], "seed", 0, 2 ** 31 - 1)
    if settings["device"] not in ("auto", "cpu", "cuda"):
        raise CuSenseError('device is "auto", "cpu" or "cuda"')
    _integer(settings["min_per_class"], "min_per_class", 1, 10 ** 6)
    threshold = _fields(settings["threshold"], ("target_false_alarm", "min_negatives"), "threshold")
    if type(threshold["target_false_alarm"]) not in (int, float) or not 0 <= threshold["target_false_alarm"] < 1:
        raise CuSenseError("threshold.target_false_alarm must lie from 0 to below 1")
    _integer(threshold["min_negatives"], "threshold.min_negatives", 1, 10 ** 9)
    check_smoothing(settings["smoothing"])
    return settings


def demo_settings(model_id="cusense-o3-v1"):
    """The training configuration of the cuSense-like branch of the demonstration."""
    return check_settings({
        "schema": SETTINGS_SCHEMA, "model_id": model_id,
        "model": {name: json.loads(json.dumps(cusense_cnn.PUBLISHED[name])) for name in MODEL_SETTINGS},
        "loss": {"grid_weight": 1.0, "sigma_cells": 8.0},
        "optimizer": {"entry": "adam", "learning_rate": 0.001, "batch_size": 256, "epochs": 100, "weight_decay": 0.0},
        "seed": 20261122, "device": "auto", "min_per_class": 20,
        "threshold": {"target_false_alarm": 0.05, "min_negatives": 20},
        "smoothing": {"entry": "map_mean_kalman", "maps": 10, "process_noise_m2_s3": 1.0, "measurement_noise_m2": 0.25,
                      "initial_speed_m_s": 1.5}})


def model_config(settings, antennas, grid):
    """The configuration of the model a training configuration asks for, on an array and a grid."""
    try:
        return cusense_cnn.check_config(dict(json.loads(json.dumps(settings["model"])), entry=cusense_cnn.ENTRY,
                                             antennas=antennas, rows=grid["rows"], cols=grid["cols"],
                                             presence_head=True))
    except ModelError as error:
        raise CuSenseError(str(error)) from None


def training_device(settings):
    """The device a fit runs on: the one the configuration names, or a CUDA device when PyTorch reaches one."""
    return _device(default_device() if settings["device"] == "auto" else settings["device"])


def _cells(samples, grid):
    cells = []
    for sample in samples:
        cell = sample["cell"] if sample["label"] == 1 else [0, 0]
        if not (0 <= cell[0] < grid["rows"] and 0 <= cell[1] < grid["cols"]):
            raise CuSenseError(f"the cell {cell} of the window {sample['window']} lies outside the grid of "
                               f"{grid['rows']} by {grid['cols']}")
        cells.append(cell)
    return torch.tensor(cells, dtype=torch.long)


def measure_loss(net, x, labels, cells, grid, loss, batch_size=256):
    """The loss of a model on a set of windows, in evaluation mode: {"loss", "presence", "location"}.

    The presence part is the mean over every window and the location part the mean over the
    target-present windows, as joint_loss defines them on one batch.
    """
    was_training = net.training
    net.eval()
    presence_sum = location_sum = 0.0
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            part = slice(start, start + batch_size)
            grid_logits, presence_logits = net(x[part])
            marks = labels[part].to(presence_logits.dtype)
            targets = cusense_cnn.smoothed_target(cells[part], grid["rows"], grid["cols"], loss["sigma_cells"])
            presence_sum += float(torch.nn.functional.binary_cross_entropy_with_logits(presence_logits, marks,
                                                                                       reduction="sum"))
            location_sum += float((cusense_cnn.kl_divergence(targets, grid_logits) * marks).sum())
    net.train(was_training)
    present = int(labels.sum())
    parts = {"presence": presence_sum / max(x.shape[0], 1), "location": location_sum / max(present, 1)}
    return dict(parts, loss=parts["presence"] + loss["grid_weight"] * parts["location"])


def fit_model(samples, x, grid, settings, device="cpu", monitor=None):
    """Fit the model on the training split. Returns (net on the CPU, history).

    samples are the rows of the training split and x their inputs, float32 [N, A, K]. The loss
    is the binary cross-entropy of the presence logit on every window plus loss.grid_weight times
    the KL divergence of the map on the target-present windows, minimised by Adam from the seed
    of the configuration. The weights are those after the last epoch: no sample of another split
    selects them. monitor, (rows, inputs) of the validation split, is measured after every epoch
    and changes nothing. A sample of another split is refused.
    """
    _only(samples, "training", "the fit of the weights")
    check_settings(settings)
    labels = [sample["label"] for sample in samples]
    counts = (labels.count(0), labels.count(1))
    if min(counts) < settings["min_per_class"]:
        raise CuSenseError(f"the training split holds {counts[1]} target-present and {counts[0]} target-absent "
                           f"windows, and the fit needs {settings['min_per_class']} of each")
    x = np.asarray(x)
    if x.dtype != np.float32 or x.ndim != 3 or x.shape[0] != len(samples):
        raise CuSenseError("the fit needs one input of shape [A, K] as float32 per training window")
    device = _device(device)
    optimizer_settings, loss = settings["optimizer"], settings["loss"]
    torch.manual_seed(settings["seed"])
    order = torch.Generator().manual_seed(settings["seed"])
    net = cusense_cnn.CuSenseCNN(model_config(settings, x.shape[1], grid)).to(device)
    data = torch.from_numpy(np.ascontiguousarray(x)).to(device)
    marks, cells = torch.tensor(labels, dtype=torch.long).to(device), _cells(samples, grid).to(device)
    watched = None
    if monitor is not None:
        rows, inputs = monitor
        _only(rows, "validation", "the monitor of the fit")
        watched = (torch.from_numpy(np.ascontiguousarray(inputs)).to(device),
                   torch.tensor([row["label"] for row in rows], dtype=torch.long).to(device),
                   _cells(rows, grid).to(device))
    batch = optimizer_settings["batch_size"]
    initial = measure_loss(net, data, marks, cells, grid, loss, batch)
    optimizer = torch.optim.Adam(net.parameters(), lr=optimizer_settings["learning_rate"],
                                 weight_decay=optimizer_settings["weight_decay"])
    epochs, started = [], time.perf_counter()
    for epoch in range(optimizer_settings["epochs"]):
        net.train()
        totals, seen = [0.0, 0.0, 0.0], 0
        for chosen in torch.randperm(len(samples), generator=order).split(batch):
            chosen = chosen.to(device)
            optimizer.zero_grad()
            grid_logits, presence_logits = net(data[chosen])
            targets = cusense_cnn.smoothed_target(cells[chosen], grid["rows"], grid["cols"], loss["sigma_cells"])
            total, presence_part, location_part = cusense_cnn.joint_loss(grid_logits, presence_logits, targets,
                                                                        marks[chosen], loss["grid_weight"])
            total.backward()
            optimizer.step()
            size = len(chosen)
            seen += size
            for index, value in enumerate((total, presence_part, location_part)):
                totals[index] += float(value.detach()) * size
        entry = {"epoch": epoch + 1, "loss": totals[0] / seen, "presence": totals[1] / seen, "location": totals[2] / seen}
        if watched is not None:
            entry["validation"] = measure_loss(net, watched[0], watched[1], watched[2], grid, loss, batch)
        epochs.append(entry)
        if not math.isfinite(entry["loss"]):
            raise CuSenseError(f"the loss of epoch {epoch + 1} is not finite")
    final = measure_loss(net, data, marks, cells, grid, loss, batch)
    net = net.to("cpu").eval()
    if not all(bool(torch.isfinite(tensor).all()) for _, tensor in cusense_cnn.tensors_of(net)):
        raise CuSenseError("the fit did not give finite weights")
    # At zero output layers the map is uniform and the presence logit is 0, whatever the windows hold.
    target = cusense_cnn.smoothed_target(_cells([sample for sample in samples if sample["label"] == 1], grid),
                                         grid["rows"], grid["cols"], loss["sigma_cells"]).double().reshape(counts[1], -1)
    entropy = -(target * torch.where(target > 0, torch.log(target.clamp_min(1e-300)), torch.zeros_like(target))).sum(dim=1)
    uniform = float((math.log(grid["rows"] * grid["cols"]) - entropy).mean())
    history = {"optimizer": "Adam from the seed of the configuration, float32", "device": str(device),
               "loss": "mean binary cross-entropy of the presence logit over every window plus grid_weight times the "
                       "mean KL divergence of the map over the target-present windows, in nats",
               "selection": "the weights after the last epoch: no sample of another split selects them",
               "settings": {"optimizer": optimizer_settings, "loss": loss, "seed": settings["seed"]},
               "epochs": len(epochs), "steps": len(epochs) * math.ceil(len(samples) / batch),
               "seconds": time.perf_counter() - started,
               "initial": initial, "final": final, "initial_loss": initial["loss"], "final_loss": final["loss"],
               "initial_expected": {"presence": math.log(2.0), "location": uniform,
                                    "loss": math.log(2.0) + loss["grid_weight"] * uniform,
                                    "why": "the output layers start at zero: a uniform map and the presence logit 0"},
               "per_epoch": epochs, "samples": {"target_present": counts[1], "target_absent": counts[0]},
               "repeatable": "the same rows, configuration and seed give the same weights on the CPU of one host; a "
                             "CUDA device does not promise the last bit, and the hash of the artifact names the fit "
                             "that was frozen"}
    return net, history


def scores_of(config, data, layout, x, device="cpu"):
    """The map and the presence logit of every window as a deployed model computes them: one window at a time.

    Returns (logits, maps): a list of logits and float32 [N, rows, cols].
    """
    scorer = Scorer(config, data, layout, device)
    x = np.asarray(x)
    made = [scorer.predict(x[index:index + 1]) for index in range(x.shape[0])]
    return [logit for _, logit, _, _ in made], np.stack([probability_map for probability_map, _, _, _ in made])


# ---------------------------------------------------------------- sensor output
def write_output(directory, model, decisions, inputs, maps, source, windows_dir=WINDOWS_DIR):
    """Write decisions.jsonl, inputs.f32, maps.f32 and cusense.json beside a window output. Returns the report.

    The directory must exist and hold none of them. inputs and maps are those of the scored
    decisions, in order, and their digests are taken here. windows_dir names the window output
    the decisions were made from: a directory inside this one, or a path.
    """
    directory = Path(directory)
    names = (DECISIONS_FILE, REPORT_FILE, INPUTS_FILE, MAPS_FILE)
    if not directory.is_dir() or any((directory / name).exists() for name in names):
        raise CuSenseError(f"the output directory {directory} does not exist or holds a result already")
    seal_decisions(decisions, inputs, maps)
    scored = [line for line in decisions if line["status"] == "ok"]
    (directory / INPUTS_FILE).write_bytes(b"".join(np.ascontiguousarray(x, dtype="<f4").tobytes() for x in inputs))
    (directory / MAPS_FILE).write_bytes(b"".join(np.ascontiguousarray(item, dtype="<f4").tobytes() for item in maps))
    (directory / DECISIONS_FILE).write_text("".join(json.dumps(line, sort_keys=True) + "\n" for line in decisions),
                                            encoding="ascii")
    counts = {status: sum(line["status"] == status for line in decisions) for status in STATUSES}
    counts.update(decisions=len(decisions), scored=len(scored),
                  present=sum(line["decision"]["present"] for line in scored),
                  absent=sum(not line["decision"]["present"] for line in scored))
    manifest = model.artifact.manifest
    report = {"schema": REPORT_SCHEMA, "artifact": {"directory": str(model.artifact.directory), "manifest": manifest,
                                                    "model": model.artifact.model()},
              "entries": resolve(model.spec, manifest["smoothing"]["entry"]), "window_fields": list(WINDOW_FIELDS),
              "runtime": dict(device_facts(model.device), entry=RUNTIME, python=platform.python_version(),
                              numpy=np.__version__, machine=platform.machine(),
                              module_sha256=_sha256(Path(__file__).read_bytes())),
              "loads": Model.LOADS, "windows": str(windows_dir), "source": source, "counts": counts,
              "files": {INPUTS_FILE: {"shape": [len(inputs)] + manifest["input"]["shape"][1:], "dtype": "<f4"},
                        MAPS_FILE: {"shape": [len(maps), manifest["grid"]["rows"], manifest["grid"]["cols"]],
                                    "dtype": "<f4"}}}
    (directory / REPORT_FILE).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return report


class Output:
    """A verified output of a cuSense-like sensor."""

    def __init__(self, directory, report, decisions, windows, inputs, maps):
        self.directory, self.report, self.decisions, self.windows = directory, report, decisions, windows
        self.inputs, self.maps = inputs, maps
        self.manifest = report["artifact"]["manifest"]
        self.scored = [line for line in decisions if line["status"] == "ok"]
        self.by_window = {line["window"]["id"]: line for line in self.scored}

    def input_of(self, decision):
        return self.inputs[decision["input"]["index"]]

    def map_of(self, decision):
        return self.maps[decision["location"]["map"]["index"]]


def open_output(directory):
    """Verify the output of a cuSense-like sensor and return it.

    The window output it names verifies with its own reader, and the output is of the same
    runtime: the reference, or the native core of Sprint 3.6. The report embeds an artifact
    manifest that has the hash it names. There is one decision per window line, with the number,
    the status and the window of that line. A scored decision names the model of the report, an
    input and a map that have the hashes it names, a map that sums to one with the location at
    its largest value, a finite logit with the probability that follows from it, and the decision
    that the frozen threshold gives. No other decision carries a score.
    """
    directory = Path(directory)
    try:
        report = json.loads((directory / REPORT_FILE).read_text(encoding="ascii"))
        decisions = [json.loads(text) for text in (directory / DECISIONS_FILE).read_text(encoding="ascii").splitlines()
                     if text.strip()]
        data, drawn = (directory / INPUTS_FILE).read_bytes(), (directory / MAPS_FILE).read_bytes()
    except (OSError, ValueError) as error:
        raise CuSenseError(f"{directory} is not a complete sensor output: {error}") from None
    if not isinstance(report, dict) or report.get("schema") != REPORT_SCHEMA:
        raise CuSenseError(f"{directory / REPORT_FILE} is not a {REPORT_SCHEMA} report")
    try:
        manifest = check_manifest(report["artifact"]["manifest"])
        model = report["artifact"]["model"]
        where = Path(report["windows"])
        windows = open_windows(where if where.is_absolute() else directory / where)
    except (KeyError, TypeError) as error:
        raise CuSenseError(f"the report of {directory} is incomplete: {error!r}") from None
    wanted = Artifact(directory, manifest, None, None, None).model()
    runtime = (report.get("runtime") or {}).get("entry")
    entries, named = resolve(manifest["input"]["spec"], manifest["smoothing"]["entry"]), report.get("entries")
    if runtime == NATIVE_RUNTIME:
        # The native core names the same entries, each with the unit that runs it and where.
        same = isinstance(named, dict) and set(named) == set(entries) \
            and all(isinstance(named[stage], dict) and named[stage].get("entry") == entries[stage]["entry"]
                    and named[stage].get("runtime") == NATIVE_RUNTIME for stage in entries)
    else:
        same = named == entries
    if model != wanted or not same or runtime not in RUNTIMES or runtime != windows.runtime:
        raise CuSenseError("the report does not name the model of its artifact, the entries of the registry and the "
                           "runtime of its windows")
    if windows.report["spec_sha256"] != manifest["input"]["spec_sha256"]:
        raise CuSenseError("the windows were computed with another specification than the model was trained on")
    if len(decisions) != len(windows.lines):
        raise CuSenseError(f"{len(decisions)} decisions for {len(windows.lines)} lines of the window stream")
    scored = [line for line in decisions if isinstance(line, dict) and line.get("status") == "ok"]
    shape, grid = tuple(manifest["input"]["shape"][1:]), manifest["grid"]
    cells = (grid["rows"], grid["cols"])
    if len(data) != 4 * len(scored) * int(np.prod(shape)) or len(drawn) != 4 * len(scored) * int(np.prod(cells)):
        raise CuSenseError("inputs.f32 or maps.f32 do not hold one tensor per scored decision")
    inputs = np.frombuffer(data, dtype="<f4").reshape((len(scored),) + shape)
    maps = np.frombuffer(drawn, dtype="<f4").reshape((len(scored),) + cells)
    threshold, position = manifest["threshold"]["logit"], 0
    for number, (decision, line) in enumerate(zip(decisions, windows.lines)):
        if not isinstance(decision, dict) or decision.get("schema") != DECISION_SCHEMA or decision.get("line") != number \
                or decision.get("status") != line["status"] or decision.get("window") != line["window"] \
                or decision.get("trigger") != line["trigger"] or decision.get("runtime") != runtime \
                or decision.get("model") != wanted:
            raise CuSenseError(f"decision {number} is not the {DECISION_SCHEMA} line of window line {number} with the "
                               "model of the report")
        if line["status"] != "ok":
            if any(decision.get(name) is not None for name in ("score", "decision", "location", "smoothed", "input")):
                raise CuSenseError(f"decision {number} has the status {line['status']} and carries a score")
            continue
        try:
            score, verdict, given, location = decision["score"], decision["decision"], decision["input"], decision["location"]
            x, drawing = inputs[position], maps[position]
            found = locate(drawing, grid)
            good = given["index"] == position and location["map"]["index"] == position \
                and given["sha256"] == _sha256(x[None].tobytes()) and given["shape"] == [1] + list(shape) \
                and given["spec_sha256"] == line["spec_sha256"] \
                and location["map"]["sha256"] == _sha256(drawing.tobytes()) \
                and abs(float(drawing.astype(np.float64).sum()) - 1.0) <= 1e-4 and bool((drawing >= 0).all()) \
                and location["cell"] == found["cell"] and location["position_m"] == found["position_m"] \
                and type(score["logit"]) is float and math.isfinite(score["logit"]) \
                and abs(score["probability"] - 1.0 / (1.0 + math.exp(-score["logit"]))) <= 1e-6 \
                and verdict["threshold_logit"] == threshold and verdict["present"] is (score["logit"] > threshold) \
                and decision["contributing"] == line["contributing"] and decision["time"] == line["time"] \
                and (decision["smoothed"] is None) == (manifest["smoothing"]["entry"] == "none")
        except (KeyError, TypeError, IndexError, OverflowError, CuSenseError) as error:
            raise CuSenseError(f"decision {number}: the scored decision is not whole: {error!r}") from None
        if not good:
            raise CuSenseError(f"decision {number}: the digest of its input, its map, its location, its score or its "
                               "decision do not agree with its window and with the frozen threshold")
        position += 1
    counts = report.get("counts") or {}
    if counts.get("decisions") != len(decisions) or counts.get("scored") != len(scored) \
            or counts.get("present") != sum(line["decision"]["present"] for line in scored):
        raise CuSenseError("the counts of the report are not those of its decisions")
    extra = sorted(path.name for path in directory.iterdir()
                   if path.name not in (DECISIONS_FILE, REPORT_FILE, INPUTS_FILE, MAPS_FILE, WINDOWS_DIR))
    if extra:
        raise CuSenseError(f"the output holds {extra[0]}, which its report does not list")
    return Output(directory, report, decisions, windows, inputs, maps)


def _empty(output_dir):
    output_dir = Path(output_dir)
    if not output_dir.is_dir() or any(output_dir.iterdir()):
        raise CuSenseError(f"the output directory {output_dir} does not exist or is not empty")
    return output_dir


def windows(archive_dir, spec, output_dir):
    """Push the records of a consumer archive through a window stream and write its output. Returns (report, stream)."""
    output_dir = _empty(output_dir)
    archive = obs.open_archive(archive_dir)
    lines, stream = run_records(archive.records, spec)
    report = write_windows(output_dir, spec, lines, stream,
                           {"kind": "archive", "directory": str(archive_dir), "reader": archive.report["reader"],
                            "records": len(archive.records)})
    return report, stream


def replay(archive_dir, artifact_dir, output_dir, device="cpu"):
    """Push the records of a consumer archive through a stream and a model, and write the output."""
    output_dir = _empty(output_dir)
    model = Model(artifact_dir, device)
    archive = obs.open_archive(archive_dir)
    lines, stream, decisions, inputs, maps = run_model(archive.records, model)
    source = {"kind": "archive", "directory": str(archive_dir), "reader": archive.report["reader"],
              "records": len(archive.records)}
    (output_dir / WINDOWS_DIR).mkdir()
    write_windows(output_dir / WINDOWS_DIR, model.spec, lines, stream, source)
    return write_output(output_dir, model, decisions, inputs, maps, source)


def follow(socket_path, artifact_dir, output_dir, device=None, wait="wait", connect_timeout=30.0, idle_timeout=60.0,
           ready_file=None):
    """Attach to a ring as its consumer and score every complete window as its last record is read.

    This is the reference sensor of the cuSense-like branch: a second process on the
    standard-library reader. The artifact is loaded once, before the attach. The steps of the
    front end run once on noise and the model once on its template before the attach as well, so
    that no record waits for a first use or for the context of the device. With ready_file the
    sensor writes that file when it is warm, and a launcher starts the receiver after it. The
    records it read are kept under windows/records/.
    """
    output_dir = _empty(output_dir)
    model = Model(artifact_dir, default_device() if device is None else device)
    warm_front = warm_up(model.spec)
    warm_model = model.warm_up()
    stream, smoother = WindowStream(model.spec, time.monotonic_ns), model.smoother()
    lines, decisions, inputs, maps = [], [], [], []

    def on_record(record):
        for line in stream.push(record):
            lines.append(line)
            decision, x, probability_map = model.decide(line, stream.last if line["status"] == "ok" else None,
                                                        time.monotonic_ns, smoother)
            decisions.append(decision)
            if x is not None:
                inputs.append(x[0])
                maps.append(probability_map)

    window_dir = output_dir / WINDOWS_DIR
    (window_dir / RECORDS_DIR).mkdir(parents=True)
    warm = {"frontend": warm_front, "model": warm_model}
    if ready_file is not None:
        Path(ready_file).write_text(json.dumps({"device": str(model.device), "artifact_sha256": model.artifact.sha256,
                                                "warm_up_ms": warm, "t_ready_ns": time.monotonic_ns()}) + "\n",
                                    encoding="ascii")
    consumer = obs.tail(socket_path, window_dir / RECORDS_DIR, wait, connect_timeout, idle_timeout, on_record=on_record)
    source = {"kind": "ring", "socket": str(socket_path), "reader": consumer["reader"], "wait": wait,
              "epoch": consumer["epoch"], "records": consumer["counts"]["read"], "stale": consumer["counts"]["stale"],
              "ended": consumer["ended"], "warm_up_ms": warm}
    write_windows(window_dir, model.spec, lines, stream, source)
    return write_output(output_dir, model, decisions, inputs, maps, source)


def summary(output, window=0):
    """What a second reader recovers from a sensor output: the model, the counts and one scored window."""
    manifest = output.manifest
    made = {"model_id": manifest["model_id"], "artifact_sha256": manifest["sha256"],
            "weights_sha256": manifest["weights"]["sha256"], "template_sha256": manifest["template"]["sha256"],
            "normalizer_sha256": manifest["normalizer"]["sha256"], "input": manifest["entries"]["input"],
            "input_shape": manifest["input"]["shape"], "grid": [manifest["grid"]["rows"], manifest["grid"]["cols"]],
            "threshold_logit": manifest["threshold"]["logit"], "runtime": output.report["runtime"]["entry"],
            "device": output.report["runtime"]["device"], "loads": output.report["loads"],
            "source": output.report["source"]["kind"], "counts": output.report["counts"],
            "scored_windows": len(output.scored)}
    if not 0 <= window < len(output.scored):
        raise CuSenseError(f"the output holds {len(output.scored)} scored windows, and window {window} was asked for")
    line = output.scored[window]
    made["window"] = {"id": line["window"]["id"], "contributing": len(line["contributing"]),
                      "logit": line["score"]["logit"], "probability": line["score"]["probability"],
                      "present": line["decision"]["present"], "cell": line["location"]["cell"],
                      "position_m": line["location"]["position_m"], "map_peak": line["location"]["probability"],
                      "map_sum": float(output.map_of(line).astype(np.float64).sum()),
                      "smoothed": line["smoothed"], "input_sha256": line["input"]["sha256"],
                      "map_sha256": line["location"]["map"]["sha256"], "infer_ns": line["infer_ns"]}
    return made


def inspect(artifact):
    """What a second reader recovers from an artifact: its hashes, its input, its grid and its threshold."""
    manifest = artifact.manifest
    template = manifest["template"]
    return {"model_id": manifest["model_id"], "artifact_sha256": manifest["sha256"], "entries": manifest["entries"],
            "spec_id": manifest["input"]["spec"]["spec_id"], "spec_sha256": manifest["input"]["spec_sha256"],
            "input_shape": manifest["input"]["shape"], "elements": manifest["input"]["elements"],
            "template": {"sha256": template["sha256"], "shape": template["shape"],
                         "valid_elements": template["valid_elements"], "records_min": template["records_min"],
                         "records_max": template["records_max"], "fitted_on": template["fitted_on"]},
            "normalizer": {name: manifest["normalizer"][name] for name in ("mean", "scale", "epsilon", "sha256")},
            "weights_sha256": manifest["weights"]["sha256"], "weights_bytes": manifest["weights"]["bytes"],
            "parameters": manifest["model"]["parameters"], "tensors": len(manifest["weights"]["tensors"]),
            "grid": {name: manifest["grid"][name] for name in ("rows", "cols", "cell_m", "origin_m", "sha256")},
            "threshold": manifest["threshold"], "smoothing": manifest["smoothing"],
            "files": sorted(path.name for path in artifact.directory.iterdir())}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reference cuSense-like front end and location model on receiver "
                                                 "estimates: one frozen model, loaded once, scoring one window of one "
                                                 "TDD period at a time.")
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("spec", help="write the window specification of the demonstration")
    make.add_argument("--output", required=True, type=Path)
    make.add_argument("--entry", choices=sorted(INPUTS), default="o3_fp16")
    make.add_argument("--spec-id")
    tune = commands.add_parser("settings", help="write the training configuration of the demonstration model")
    tune.add_argument("--output", required=True, type=Path)
    tune.add_argument("--model-id", default="cusense-o3-v1")
    where = commands.add_parser("device", help="print the device a model would run on in this interpreter")
    where.add_argument("--device", help="cpu, cuda or cuda:N (default: cuda when PyTorch reaches one)")
    cut = commands.add_parser("windows", help="compute the windows of a consumer archive, without a model")
    cut.add_argument("--archive", required=True, type=Path)
    cut.add_argument("--spec", required=True, type=Path)
    cut.add_argument("--output-dir", required=True, type=Path, help="existing, empty directory")
    live = commands.add_parser("follow", help="attach to a ring and score every complete window")
    live.add_argument("--socket", required=True, type=Path, help="control channel of the producer's ring")
    live.add_argument("--wait", choices=sorted(obs.WAIT_MODES), default="wait")
    live.add_argument("--connect-ms", type=int, default=30000)
    live.add_argument("--idle-ms", type=int, default=60000)
    live.add_argument("--ready-file", type=Path, help="written when the model is loaded and warm, before the attach")
    again = commands.add_parser("replay", help="compute windows and decisions from the records of a consumer archive")
    again.add_argument("--archive", required=True, type=Path)
    for command in (live, again):
        command.add_argument("--artifact", required=True, type=Path, help="directory of a cusense-artifact/v1 artifact")
        command.add_argument("--device", help="cpu, cuda or cuda:N (default: cuda when PyTorch reaches one, for "
                                              "follow; cpu otherwise)")
        command.add_argument("--output-dir", required=True, type=Path, help="existing, empty directory")
    show = commands.add_parser("show", help="verify a sensor output and print one scored window")
    show.add_argument("directory", type=Path)
    show.add_argument("--window", type=int, default=0)
    look = commands.add_parser("inspect", help="verify an artifact and print its hashes, its grid and its threshold")
    look.add_argument("artifact", type=Path)
    look.add_argument("--sha256", help="the hash a freeze record names for the artifact")
    args = parser.parse_args(argv)
    try:
        if args.command in ("spec", "settings"):
            if args.output.exists():
                raise CuSenseError(f"{args.output} already exists. A configuration is never overwritten.")
            made = demo_spec(args.entry, args.spec_id) if args.command == "spec" else demo_settings(args.model_id)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(made, indent=2, sort_keys=True) + "\n", encoding="ascii")
            print(json.dumps({args.command: str(args.output), "sha256": _sha256(canonical(made))}, indent=2))
            return 0
        if args.command == "device":
            print(json.dumps(device_facts(_device(args.device or default_device())), indent=2))
            return 0
        if args.command == "show":
            print(json.dumps(summary(open_output(args.directory), args.window), indent=2))
            return 0
        if args.command == "inspect":
            print(json.dumps(inspect(open_artifact(args.artifact, args.sha256)), indent=2))
            return 0
        if args.command == "windows":
            report, _ = windows(args.archive, load_spec(args.spec), args.output_dir)
            counts = report["counts"]
            print(f"cusense: {counts['records']} records, {counts['ok']} complete windows, "
                  f"{counts['insufficient_history']} lines of a filling window, {counts['gap']} gaps, "
                  f"{counts['invalid']} invalid, {counts['rejected']} rejected, {counts['passed']} outside the window")
            return 0
        if args.command == "follow":
            report = follow(args.socket, args.artifact, args.output_dir, args.device, args.wait, args.connect_ms / 1000,
                            args.idle_ms / 1000, args.ready_file)
        else:
            report = replay(args.archive, args.artifact, args.output_dir, args.device or "cpu")
        counts = report["counts"]
        print(f"cusense: {counts['decisions']} decisions on {report['runtime']['device']}, {counts['scored']} scored: "
              f"{counts['present']} present, {counts['absent']} absent; {counts['insufficient_history']} of a filling "
              f"window, {counts['gap']} gaps, {counts['invalid']} invalid, {counts['rejected']} rejected; the model "
              f"{report['artifact']['model']['model_id']} {report['artifact']['model']['artifact_sha256'][:12]} was "
              f"loaded {report['loads']} time(s)")
        return 0
    except (CuSenseError, PresenceError, FeaturesError, ObsError) as error:
        parser.exit(2, f"cusense: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
