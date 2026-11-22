"""Evaluator side of the cuSense-like branch (Sprint 3.5, selector cusense_frontend).

The sensing core turns records into windows, a location map and a presence decision, and knows
nothing of what the records were computed from. This module holds what an evaluator adds around
it, offline. The split of an episode set, its replay in runs of one grid gain and the transmit
grids of its entries are those of Sprint 3.4 (episodes/presence_head.py) and are reused here.

  plans        the receiver replay of some entries of a set as a plan of version 3: both receiver
               stages, O2 and O3, each on a ring of its own, in the evidence mode timed.
  positions    where the target of a realization is at every opportunity: the cell and the position
               that the set recorded. That is truth, read after the windows are written.
  samples      the complete windows of a window stream, joined to the split by episode key and to
               the position of the target at the last opportunity of the window.
  oracle       the window average and the input of the model again, element by element, with the
               reader of the stage they come from and no code of the stream.
  parity       the decisions of a sensor on a device against those of the same artifact on the CPU,
               joined by window and by the digest of the input.
  evaluation   presence counts by label with their denominators, and the position error in metres
               on the target-present windows, raw and after the optional stage.
  reference    the measured figures beside the published figures of the cuSense system, with every
               difference between the two listed. Recorded, never graded.

Nothing here runs in a slot loop, and nothing here is imported by the sensing package.
"""

import math
from pathlib import Path

import numpy as np

from ..sensing import cusense
from ..sensing import native_estimate as o3_reference
from ..sensing import pilot_ls as o2_reference
from . import episode as episodes
from . import grid_apply as replay
from . import native_estimate as ne
from . import pilot_features
from . import presence_head
from . import replay_pusch as rp
from . import snapshot_ring as sr
from .episode import EpisodeError
from .grid_apply import GridApplyError
from .native_estimate import NativeEstimatePlanError
from .pilot_features import PilotFeaturesError
from .presence_head import PresenceHeadError
from .replay_pusch import ReplayPuschError
from .snapshot_ring import SnapshotRingError

EVALUATION_SCHEMA = "cusense-evaluation/v1"
EVAL_SCHEMA = "cusense-eval/v1"
REFERENCE_SCHEMA = "cusense-reference/v1"
PARITY_SCHEMA = "cusense-parity/v1"
GROUPS = presence_head.GROUPS
CHANNEL_PLAN, REPLAY_PLAN, PLAN = presence_head.CHANNEL_PLAN, presence_head.REPLAY_PLAN, presence_head.PLAN
# A logit may differ by logit_absolute plus logit_relative of its magnitude: a saturated presence logit is a large
# float32 value. The probability and the cells of the map are bounded and take an absolute tolerance.
TOLERANCE = {"logit_absolute": 1e-3, "logit_relative": 1e-4, "probability_absolute": 2.5e-4, "map_absolute": 1e-4,
             "reload_absolute": 1e-5, "window_relative": 1e-6}
SELECTIONS = ("last",)
SELECTION_RULE = {"last": "one prospectively selected window per realization: the last complete window of the episode"}
# The published system: Villa et al., arXiv:2512.06493v3. Position error on two unseen runs after
# averaging ten maps and Kalman filtering (table 2), and the live inference of the native serving
# option. Recorded beside what this project measures, and never graded.
PUBLISHED = {
    "source": "Villa et al., arXiv:2512.06493v3, section 6.2 and table 2",
    "position_error_cm": {"mean": 73.3, "median": 54.9, "p90": 148.3, "rmse": 95.0,
                          "of": "46,417 predictions of two unseen runs, after the mean of ten maps and Kalman filtering"},
    "error_at_95_percent_m": {"raw": 3.00, "averaged": 2.69, "kalman": 2.10},
    "live_inference_us": {"value": 241, "serving_option": "native: CUDA preprocessing and a TensorRT engine in the "
                                                         "dApp, GH200 with MIG"},
    "system": {"bandwidth_mhz": 100, "prbs": 273, "rx": 4, "dmrs_symbols": 3, "gpu": "GH200",
               "room_m": [6.78, 10.06], "training_samples": 362318, "calibration_samples": 54400,
               "label_sigma_cells": 8.0, "epochs": 100, "batch_size": 256, "learning_rate": 0.001,
               "window_ms": 2.0, "labels": "camera-derived positions"}}
DIFFERENCES = (
    "a simulated scene, traced with Sionna RT, against an over-the-air room",
    "20 MHz, 51 resource blocks at 30 kHz, against 100 MHz, 273 resource blocks",
    "a GB10 in a DGX Spark against a GH200",
    "the receiver's estimate on two DM-RS symbols of DeepSig-OCUDU against three DM-RS symbols of the Aerial L1",
    "positions of the traced target, exact, against camera-derived labels",
    "a few thousand windows of a few hundred traced episodes against 362,318 slot samples of five runs",
    "a split by whole trajectories against contiguous blocks inside the training runs and two unseen runs",
    "a template and z-score statistics fitted on the training split only against statistics over all samples",
    "a presence logit trained with the map, and target-absent windows scored, against no detection stage",
    "the reference runtime, PyTorch in a Python sensor, against the serving options of the paper: the latency "
    "comparison is made on the native core by S3.6",
)


class CuSenseFrontendError(ValueError):
    """A set, a plan, a window stream or a decision set cannot be evaluated as documented."""


split_table, check_table, by_key, chunks = (presence_head.split_table, presence_head.check_table, presence_head.by_key,
                                            presence_head.chunks)
bindings_of, keys_of, area_under_curve = presence_head.bindings_of, presence_head.keys_of, presence_head.area_under_curve


# ---------------------------------------------------------------- plans
def ring_of(settings, attach="required"):
    """The ring object of a plan from the settings of one stream."""
    try:
        return sr.ring_spec(settings["window_records"], settings["burst_records"], settings["wait"], attach,
                            settings["attach_timeout_ms"], settings["drain_ms"], settings["stream"])
    except (SnapshotRingError, KeyError, TypeError) as error:
        raise CuSenseFrontendError(f"the ring of a stream is unusable: {error}") from None


def write_plans(run_id, rows, bindings, directory, rings, estimate, epoch=1, source="pusch_dmrs"):
    """The three plans of one tapped receiver replay of some entries of a set, written into directory.

    Returns (channel plan, plan of the tapped run). The plan of the run is a plan of version 3:
    the stage O2 on the ring rings["o2"] and the receiver's own estimate, the stage O3, on the
    ring rings["o3"], in the evidence mode timed. estimate names the copy stream, the landing and
    the timeout of the capture. The transmit grids are those of bindings, as in Sprint 3.4.
    """
    directory = Path(directory)
    try:
        channel = replay.entry_plan(run_id, [row["directory"] for row in rows],
                                    bindings={name: {"x_sha256": item["x_sha256"], "store": item["store"]}
                                              for name, item in bindings.items()})
        replay.write_plan(channel, directory / CHANNEL_PLAN)
        rp.write_plan(rp.build_plan(run_id, directory / CHANNEL_PLAN), directory / REPLAY_PLAN)
        wanted = ne.estimate_spec(estimate["copy_stream"], estimate["landing"], estimate["timeout_us"],
                                  ring_of(rings["o3"]), source)
        plan = ne.build_plan(run_id, directory / REPLAY_PLAN, "timed", ring_of(rings["o2"]), wanted, epoch=epoch,
                             source=source)
        ne.write_plan(plan, directory / PLAN)
    except (GridApplyError, ReplayPuschError, SnapshotRingError, NativeEstimatePlanError, KeyError, TypeError) as error:
        raise CuSenseFrontendError(f"the entries cannot be replayed: {error}") from None
    return channel, plan


# ---------------------------------------------------------------- truth, for the evaluator only
def positions(channel_plan):
    """Where the target of every realization of a run is, by opportunity. Evaluator-only.

    Returns {episode key: {"present", "entry", "opportunities": {opportunity key: {"index",
    "cell", "position_m"}}}}. A target-absent realization has no cell and no position.
    """
    try:
        found = pilot_features.realizations(channel_plan)
    except (PilotFeaturesError, KeyError, TypeError) as error:
        raise CuSenseFrontendError(f"the channel plan does not replay episode entries: {error}") from None
    opened, made = {}, {}
    for key, item in found.items():
        directory = str(item["entry"])
        try:
            if directory not in opened:
                opened[directory] = episodes.open_episode(item["entry"])
            entry = opened[directory]
            places = {}
            for opportunity, index in item["opportunities"].items():
                truth = entry.truth(index) if item["present"] else None
                places[opportunity] = {"index": index, "cell": None if truth is None else [int(v) for v in truth["cell"]],
                                       "position_m": None if truth is None else [float(v) for v in truth["position_m"][:2]]}
        except (EpisodeError, KeyError, IndexError, TypeError) as error:
            raise CuSenseFrontendError(f"the entry {directory} gives no position for its opportunities: {error!r}") from None
        made[key] = {"present": item["present"], "entry": directory, "opportunities": places}
    return made


def position_of(truth, line):
    """The position of the target of a complete window: that of the last opportunity of the window, or None."""
    known = truth.get(line["window"]["episode"])
    if known is None:
        raise CuSenseFrontendError(f"the window {line['window']['id']} belongs to no realization of the run")
    place = known["opportunities"].get(line["contributing"][-1]["opportunity"])
    if place is None:
        raise CuSenseFrontendError(f"the window {line['window']['id']} ends at an opportunity its realization does not "
                                   "hold")
    return place


# ---------------------------------------------------------------- samples
def samples(lines, means, counts, table, role, truth, expected=None):
    """The complete windows of a window stream as samples of one split. Returns (rows, means, counts, account).

    lines are the sealed lines of a stream and means and counts the tensors of its complete
    windows, in order. Every complete window must belong to an episode key of the table whose
    split is role: a window of another split is refused. A target-present window carries the
    cell and the position of its target at the last opportunity of the window. expected lists
    the episode keys that the run replayed. The account names every key without a window.
    """
    lookup = by_key(table)
    rows, seen, last = [], {}, {}
    for line in lines:
        if line["window"] is not None:
            last[line["window"]["episode"]] = line["status"]
    complete = [line for line in lines if line["status"] == "ok"]
    if len(complete) != len(means) or len(complete) != len(counts):
        raise CuSenseFrontendError("the stream holds another number of complete windows than tensors")
    for index, line in enumerate(complete):
        key = line["window"]["episode"]
        known = lookup.get(key)
        if known is None:
            raise CuSenseFrontendError(f"the window {line['window']['id']} belongs to no realization of the set")
        if known["split"] != role:
            raise CuSenseFrontendError(f"the window {line['window']['id']} belongs to the {known['split']} split, and "
                                       f"this stream was read as {role}")
        place = position_of(truth, line)
        if (place["cell"] is not None) != bool(known["label"]):
            raise CuSenseFrontendError(f"the window {line['window']['id']} has the label {known['label']} and "
                                       f"{'a' if place['cell'] is not None else 'no'} position")
        seen[key] = seen.get(key, 0) + 1
        rows.append({"schema": cusense.SAMPLE_SCHEMA, "window": line["window"]["id"], "episode": key,
                     "split": known["split"], "scene": known["scene"], "pair": known["pair"], "label": known["label"],
                     "cell": place["cell"], "position_m": place["position_m"], "spec_sha256": line["spec_sha256"],
                     "elements": line["rx"]["elements"], "producer_steps": line["input"]["producer_steps"],
                     "shape": line["mean"]["shape"], "index": index, "mean_sha256": line["mean"]["sha256"],
                     "count_sha256": line["count"]["sha256"]})
    wanted = list(expected) if expected is not None else sorted(seen)
    missing = [{"episode": key, "pair": lookup[key]["pair"] if key in lookup else None, "last_answer": last.get(key)}
               for key in wanted if key not in seen]
    account = {"role": role, "realizations": len(wanted), "with_a_window": sum(key in seen for key in wanted),
               "windows": len(rows), "per_realization": sorted({seen.get(key, 0) for key in wanted}),
               "target_present": sum(row["label"] for row in rows),
               "target_absent": sum(1 - row["label"] for row in rows), "missing": missing}
    return rows, np.asarray(means), np.asarray(counts), account


def merge_samples(parts):
    """The samples of several runs of one split as one set: (rows, means, counts), with the rows numbered again."""
    rows, means, counts = [], [], []
    for some, mean, count in parts:
        for row in some:
            rows.append(dict(row, index=len(rows)))
        means.append(mean)
        counts.append(count)
    if not rows:
        raise CuSenseFrontendError("there is no window to make a sample set of")
    return rows, np.concatenate(means), np.concatenate(counts)


# ---------------------------------------------------------------- oracle
def oracle_window(records, spec):
    """The window average of the records of one period, element by element. Returns (mean, count).

    The values are read with the reader of their stage: native_estimate.payload_values and
    coordinate for the stage O3, pilot_ls.payload_values for the stage O2. Each is placed at
    [array element in rising order, position on the support, DM-RS symbol] and summed in double
    precision. It shares no code with the stream: no transposition and no array operator.
    """
    support, tau = spec["support"], spec["validity"]["tau"]
    shape = cusense.shape_of(spec)
    total = np.zeros(shape, dtype=np.complex128)
    count = np.zeros(shape, dtype=np.int64)
    for record in records:
        header, sizes = record.header, list(record.shape)
        rx = header["rx_count"]
        rank = {element: number for number, element in enumerate(sorted(header["antenna_map"][:rx]))}
        offset = (header["first_subcarrier"] - support["first_subcarrier"]) // support["subcarrier_step"]
        if spec["input"]["layout"] == o3_reference.LAYOUT:
            values = o3_reference.payload_values(record.payload)
            for index, value in enumerate(values):
                where = o3_reference.coordinate(sizes, index)
                place = (rank[header["antenna_map"][where["rx_port"]]], offset + where["subcarrier"], where["dmrs_symbol"])
                if record.is_valid(index) and math.isfinite(value.real) and math.isfinite(value.imag) and abs(value) > tau:
                    total[place] += value
                    count[place] += 1
        else:
            values = o2_reference.payload_values(record.payload)
            planes, symbols, pilots = sizes
            for index, value in enumerate(values):
                plane, rest = divmod(index, symbols * pilots)
                symbol, pilot = divmod(rest, pilots)
                place = (rank[header["antenna_map"][plane]], offset + pilot, symbol)
                if record.is_valid(index) and math.isfinite(value.real) and math.isfinite(value.imag) and abs(value) > tau:
                    total[place] += value
                    count[place] += 1
    return np.where(count > 0, total / np.maximum(count, 1), 0), count


def oracle_input(mean, count, template, template_count, normalizer, step=37):
    """The input of the model at some positions of one window, each from the formula of the card.

    Returns [(array element, position, value)] for every step-th position: the mean over the
    DM-RS symbols of |H_t| - |H_B| where both hold a sample, minus the mean of the normalizer and
    divided by its scale plus epsilon, with Python floats.
    """
    made = []
    for element in range(mean.shape[0]):
        for position in range(0, mean.shape[1], step):
            parts = [abs(complex(mean[element, position, symbol])) - abs(complex(template[element, position, symbol]))
                     for symbol in range(mean.shape[2])
                     if count[element, position, symbol] > 0 and template_count[element, position, symbol] > 0]
            value = 0.0 if not parts else (sum(parts) / len(parts) - normalizer["mean"]) \
                / (normalizer["scale"] + normalizer["epsilon"])
            made.append((element, position, value))
    return made


# ---------------------------------------------------------------- parity
def parity(device, offline, tolerance=None):
    """The scored decisions of a sensor on its device against those of the same artifact offline.

    device and offline are verified sensor outputs. They are joined by window identifier, and a
    pair must name the same digest of its input, the same contributing opportunities and the
    same model. The logit, the probability and every cell of the map must agree within the
    declared tolerance, which for a logit is logit_absolute plus logit_relative of the offline
    value. A decision that differs at the frozen threshold is listed with the margin of its
    score, and is a problem when that margin is larger than the tolerance of its logit. A
    location that differs is listed. Returns (problems, facts).
    """
    tolerance = dict(TOLERANCE, **(tolerance or {}))
    mine, theirs = device.by_window, offline.by_window
    problems, flips, moved, worst = [], [], [], {"logit": 0.0, "probability": 0.0, "map": 0.0}
    if set(mine) != set(theirs):
        only = sorted(set(mine) ^ set(theirs))
        problems.append(f"{len(only)} windows are scored on one side only, first: {only[0]}")
    if device.manifest["sha256"] != offline.manifest["sha256"]:
        problems.append("the two outputs were scored with different artifacts")
    for window in sorted(set(mine) & set(theirs)):
        first, second = mine[window], theirs[window]
        if first["input"]["sha256"] != second["input"]["sha256"]:
            problems.append(f"{window}: the digest of the input differs: the two sides did not score the same window")
            continue
        if [item["opportunity"] for item in first["contributing"]] != [item["opportunity"] for item in second["contributing"]]:
            problems.append(f"{window}: the contributing observations differ")
        logit = abs(first["score"]["logit"] - second["score"]["logit"])
        probability = abs(first["score"]["probability"] - second["score"]["probability"])
        cells = float(np.abs(device.map_of(first).astype(np.float64) - offline.map_of(second).astype(np.float64)).max())
        worst = {"logit": max(worst["logit"], logit), "probability": max(worst["probability"], probability),
                 "map": max(worst["map"], cells)}
        allowed = tolerance["logit_absolute"] + tolerance["logit_relative"] * abs(second["score"]["logit"])
        if logit > allowed or probability > tolerance["probability_absolute"] or cells > tolerance["map_absolute"]:
            problems.append(f"{window}: the logit differs by {logit:.3e}, the probability by {probability:.3e} and a "
                            f"cell of the map by {cells:.3e}")
        if first["decision"]["present"] != second["decision"]["present"]:
            margin = first["score"]["logit"] - first["decision"]["threshold_logit"]
            flips.append({"window": window, "device_logit": first["score"]["logit"],
                          "offline_logit": second["score"]["logit"], "margin": margin})
            if abs(margin) > allowed:
                problems.append(f"{window}: the decision differs at the threshold with a margin of {margin:+.3e}")
        if first["location"]["cell"] != second["location"]["cell"]:
            moved.append({"window": window, "device_cell": first["location"]["cell"],
                          "offline_cell": second["location"]["cell"],
                          "device_peak": first["location"]["probability"],
                          "offline_peak": second["location"]["probability"]})
    facts = {"schema": PARITY_SCHEMA, "joined": len(set(mine) & set(theirs)), "device": device.report["runtime"]["device"],
             "offline": offline.report["runtime"]["device"], "largest_logit_difference": worst["logit"],
             "largest_probability_difference": worst["probability"], "largest_map_difference": worst["map"],
             "tolerance": tolerance, "flips": flips, "locations_moved": moved,
             "artifact_sha256": device.manifest["sha256"]}
    return problems, facts


# ---------------------------------------------------------------- evaluation, with the truth
def nearest_rank(values, share):
    """The value at a share of a sorted copy of values, by nearest rank: the smallest value with that share at or below it."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(max(math.ceil(share * len(ordered)), 1), len(ordered)) - 1]


def error_statistics(values):
    """Mean, median, 90th percentile and root mean square of position errors in metres, with their count."""
    if not values:
        return {"eligible": 0, "mean_m": None, "median_m": None, "p90_m": None, "rmse_m": None, "max_m": None}
    return {"eligible": len(values), "mean_m": sum(values) / len(values), "median_m": nearest_rank(values, 0.5),
            "p90_m": nearest_rank(values, 0.9), "rmse_m": math.sqrt(sum(value * value for value in values) / len(values)),
            "max_m": max(values)}


def _counts():
    return {"target_present": {"total": 0, "detected": 0, "missed": 0, "missing": 0},
            "target_absent": {"total": 0, "false_alarm": 0, "quiet": 0, "missing": 0}}


def _tally(side, label, present):
    side["total"] += 1
    if present is None:
        side["missing"] += 1
    elif label:
        side["detected" if present else "missed"] += 1
    else:
        side["false_alarm" if present else "quiet"] += 1


def evaluate(output, table, truth, role, expected, windows_per_realization, selection="last"):
    """The decisions of a sensor output against the truth: a cusense-evaluation/v1 object of one arm.

    This is truth, read after the decisions are written. Presence is counted twice, each with
    its denominator: over every window that a realization should have given, and over one
    prospectively selected window per realization. A window or a realization without a scored
    decision is counted as missing and never as correct. The position error is the distance in
    metres between the estimate and the position of the target at the last opportunity of the
    window, on the scored windows of target-present realizations: of the raw map, of the mean of
    the newest maps and of the Kalman filter.
    """
    if selection not in SELECTIONS:
        raise CuSenseFrontendError(f"unknown selection {selection!r}: " + ", ".join(SELECTIONS))
    lookup = by_key(table)
    scored = {}
    for line in output.scored:
        key = line["window"]["episode"]
        known = lookup.get(key)
        if known is None or known["split"] != role:
            raise CuSenseFrontendError(f"the window {line['window']['id']} is no window of the {role} split")
        scored.setdefault(key, []).append(line)
    by_window, by_episode = _counts(), _counts()
    errors, rows, logits, labels = {"raw": [], "smoothed": [], "kalman": []}, [], [], []
    for key in expected:
        known = lookup.get(key)
        if known is None:
            raise CuSenseFrontendError(f"the episode key {key} names no realization of the set")
        label = known["label"]
        side = "target_present" if label else "target_absent"
        lines = sorted(scored.get(key, []), key=lambda line: line["window"]["period"])
        chosen = lines[-1] if lines else None
        _tally(by_episode[side], label, None if chosen is None else chosen["decision"]["present"])
        for line in lines:
            _tally(by_window[side], label, line["decision"]["present"])
            logits.append(line["score"]["logit"])
            labels.append(label)
            row = {"episode": key, "pair": known["pair"], "label": label, "window": line["window"]["id"],
                   "period": line["window"]["period"], "logit": line["score"]["logit"],
                   "present": line["decision"]["present"], "selected": line is chosen,
                   "cell": line["location"]["cell"], "position_m": line["location"]["position_m"], "truth_m": None,
                   "error_m": None, "smoothed_error_m": None, "kalman_error_m": None}
            if label:
                place = position_of(truth, line)
                row["truth_m"] = place["position_m"]
                row["error_m"] = math.dist(line["location"]["position_m"], place["position_m"])
                errors["raw"].append(row["error_m"])
                if line["smoothed"] is not None:
                    row["smoothed_error_m"] = math.dist(line["smoothed"]["position_m"], place["position_m"])
                    row["kalman_error_m"] = math.dist(line["smoothed"]["kalman_position_m"], place["position_m"])
                    errors["smoothed"].append(row["smoothed_error_m"])
                    errors["kalman"].append(row["kalman_error_m"])
            rows.append(row)
        for _ in range(max(windows_per_realization - len(lines), 0)):
            _tally(by_window[side], label, None)
        if not lines:
            rows.append({"episode": key, "pair": known["pair"], "label": label, "window": None, "period": None,
                         "logit": None, "present": None, "selected": False, "cell": None, "position_m": None,
                         "truth_m": None, "error_m": None, "smoothed_error_m": None, "kalman_error_m": None})
    manifest = output.manifest
    return {"schema": EVALUATION_SCHEMA, "role": role, "graded": False, "input": manifest["entries"]["input"],
            "artifact_sha256": manifest["sha256"], "threshold_logit": manifest["threshold"]["logit"],
            "presence": {"windows": dict(by_window, rule="every window a realization should give: "
                                                         f"{windows_per_realization} per realization, one per period"),
                         "episodes": dict(by_episode, rule=SELECTION_RULE[selection]),
                         "area_under_curve": area_under_curve(logits, labels), "scored_windows": len(logits)},
            "position": {"unit": "m", "eligible": "scored windows of target-present realizations, whatever their "
                                                  "presence decision",
                         "estimate": "the centre of a cell of the grid: a perfect cell still leaves up to half a cell "
                                     "diagonal",
                         "cell_m": manifest["grid"]["cell_m"], "raw": error_statistics(errors["raw"]),
                         "smoothed": error_statistics(errors["smoothed"]), "kalman": error_statistics(errors["kalman"]),
                         "smoothing": manifest["smoothing"]},
            "rows": rows}


def merge_evaluations(evaluations):
    """The evaluations of several runs of one role, one artifact and one input as one: counts added, rows joined."""
    if not evaluations:
        raise CuSenseFrontendError("there is no evaluation to merge")
    first = evaluations[0]
    if any((item["role"], item["artifact_sha256"], item["input"]) != (first["role"], first["artifact_sha256"], first["input"])
           for item in evaluations):
        raise CuSenseFrontendError("the evaluations are not of one role, one artifact and one input")
    rows = [row for item in evaluations for row in item["rows"]]
    presence = {}
    for side in ("windows", "episodes"):
        merged = _counts()
        for item in evaluations:
            for label in merged:
                for key in merged[label]:
                    merged[label][key] += item["presence"][side][label][key]
        presence[side] = dict(merged, rule=first["presence"][side]["rule"])
    scored = [row for row in rows if row["logit"] is not None]
    presence.update(area_under_curve=area_under_curve([row["logit"] for row in scored], [row["label"] for row in scored]),
                    scored_windows=len(scored))
    position = dict(first["position"])
    for name, field in (("raw", "error_m"), ("smoothed", "smoothed_error_m"), ("kalman", "kalman_error_m")):
        position[name] = error_statistics([row[field] for row in rows if row[field] is not None])
    return dict(first, presence=presence, position=position, rows=rows)


def baseline(output, table, role, expected):
    """The decisions of the linear presence head of Sprint 3.4 on the same realizations, by episode."""
    try:
        return presence_head.evaluate(output, table, role, expected)
    except PresenceHeadError as error:
        raise CuSenseFrontendError(str(error)) from None


def compare_arms(evaluations, linear):
    """What the arms decided on the same held-out realizations, side by side. Recorded, never graded."""
    made = {}
    for name, evaluation in evaluations.items():
        episodes_of, position = evaluation["presence"]["episodes"], evaluation["position"]
        made[name] = {"detected": episodes_of["target_present"]["detected"],
                      "target_present": episodes_of["target_present"]["total"],
                      "false_alarm": episodes_of["target_absent"]["false_alarm"],
                      "target_absent": episodes_of["target_absent"]["total"],
                      "area_under_curve": evaluation["presence"]["area_under_curve"],
                      "position_mean_m": position["raw"]["mean_m"], "position_kalman_mean_m": position["kalman"]["mean_m"]}
    if linear is not None:
        counts = linear["counts"]
        made["linear_head"] = {"detected": counts["target_present"]["detected"],
                               "target_present": counts["target_present"]["episodes"],
                               "false_alarm": counts["target_absent"]["false_alarm"],
                               "target_absent": counts["target_absent"]["episodes"],
                               "area_under_curve": linear["area_under_curve"], "position_mean_m": None,
                               "position_kalman_mean_m": None}
    return made


def reference_record(evaluation, latency, context):
    """The measured figures beside the published ones: a cusense-reference/v1 object. Never graded.

    evaluation is the evaluation of the arm on the receiver's own estimate, latency the
    batch-one times of the sensor with their runtime entry, and context what the run was: the
    bandwidth, the device and the set.
    """
    position = evaluation["position"]

    def centimetres(statistics):
        return {name[:-2] + "_cm": None if statistics[name] is None else 100.0 * statistics[name]
                for name in ("mean_m", "median_m", "p90_m", "rmse_m")} | {"eligible": statistics["eligible"]}

    return {"schema": REFERENCE_SCHEMA, "graded": False,
            "aim": "comparable performance in software, judged as Sprint 1.2 judged the timing of DeepSig Table II: "
                   "the published position error belongs to one room and one over-the-air system and is no target",
            "published": PUBLISHED,
            "measured": {"input": evaluation["input"], "artifact_sha256": evaluation["artifact_sha256"],
                         "position_error": {"raw": centimetres(position["raw"]),
                                            "smoothed": centimetres(position["smoothed"]),
                                            "kalman": centimetres(position["kalman"]),
                                            "like_for_like": "the published figure is after the mean of ten maps and "
                                                             "Kalman filtering: compare it with kalman"},
                         "latency": latency, "context": context},
            "differences": list(DIFFERENCES),
            "latency_comparison": "not made here: the published 241 us belongs to the native serving option, CUDA "
                                  "preprocessing with a TensorRT engine, and this run is the reference runtime. S3.6 "
                                  "compares the native core by the same boundary",
            "resolution": "at 18.36 MHz the nominal delay resolution is 54.5 ns, about 16 m of bistatic path: a "
                          "position at room scale comes from structure the model learns across antennas and "
                          "frequency in the trained scene"}
