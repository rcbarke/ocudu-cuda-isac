"""Evaluator side of the native sensing core (Sprint 3.6, selector sense_core).

The core, native/sense_core, turns the same records into the same windows, maps and decisions
as the reference of Sprint 3.5, with its units compiled for the device. This module holds what
an evaluator adds around the two runtimes, offline, from verified sensor outputs:

  input_bound   what the FP32 units may differ from the reference by at every element of an
                input: the declared number of FP32 spacings at the magnitude the element was
                computed from, divided by the divisor of the z-score.
  parity        the output of the core against the output of the reference on the same records:
                the windows, the statuses, the inputs within their bound, the scores and the maps
                within the declared tolerance, the decisions at the frozen threshold, and every
                flip with its margin. A sense-runtime-parity/v1 object.
  latency       the three latency definitions of docs/architecture/RUNTIME.md over the scored
                decisions of one output, with their counts and tails, and the stages of a
                submission where the runtime marks them.
  reference     the native latency beside the published figure of the cuSense system, by the
                boundary of that system, with every difference listed. Recorded, never graded.

Nothing here runs in a slot loop, and nothing here is imported by the sensing package.
"""

import numpy as np

from ..sensing import cusense
from . import cusense_frontend

PARITY_SCHEMA = "sense-runtime-parity/v1"
LATENCY_SCHEMA = "sense-runtime-latency/v1"
REFERENCE_SCHEMA = cusense_frontend.REFERENCE_SCHEMA
TC_PER_SECOND = cusense.TC_PER_SECOND
DEFINITIONS = {
    "end_to_end": "t_decision minus t_first_sample of the window, through the map between the radio clock and the host "
                  "clock that the run records",
    "post_window": "t_decision minus t_ready of the last record of the window",
    "cusense_boundary": "t_decision minus the notification of the last record of the window at the sensor",
}
# What the published 241 us was measured on, against this gate. Stated with the figure, always.
DIFFERENCES = (
    "20 MHz, 51 resource blocks at 30 kHz and 612 subcarriers per antenna, against 100 MHz and 273 resource blocks",
    "a GB10 in a DGX Spark against a GH200 with MIG",
    "a simulated scene, traced with Sionna RT and replayed, against an over-the-air room",
    "the receiver's estimate on two DM-RS symbols of DeepSig-OCUDU against three DM-RS symbols of the Aerial L1",
    "a window of three uplink slots, averaged in the complex domain, against the window of the published system",
    "a cuSense-like implementation written for this project: cuSense is not open source, and its kernels and its "
    "engine were not used",
    "a sensor in a process of its own that reads the observation ring, against a dApp in the framework of the L1",
)


class SenseCoreError(ValueError):
    """Two outputs cannot be compared, or an output carries no times."""


def nearest_rank(values, share):
    return cusense_frontend.nearest_rank(values, share)


def statistics(values):
    """Count, mean, median, tails and extremes of a list of durations in microseconds."""
    if not values:
        return {"count": 0, "mean_us": None, "p50_us": None, "p90_us": None, "p99_us": None, "min_us": None, "max_us": None}
    return {"count": len(values), "mean_us": sum(values) / len(values), "p50_us": nearest_rank(values, 0.5),
            "p90_us": nearest_rank(values, 0.9), "p99_us": nearest_rank(values, 0.99), "min_us": min(values),
            "max_us": max(values)}


# ---------------------------------------------------------------- parity
def input_bound(x, template, template_count, normalizer, tolerance):
    """The largest difference the FP32 units may show at each element of an input [A, K].

    The front end computes |H_t| - |H_B| per DM-RS symbol, its mean over the symbols and the
    z-score. Each step rounds once at the magnitude of its operands, which is at most the larger
    of the window magnitude and the template magnitude. The bound is input_ulps spacings of FP32
    at that magnitude, divided by the divisor of the z-score. The window magnitude is recovered
    from the input itself, so the bound needs no tensor that a timed run does not keep.
    """
    count = np.asarray(template_count)
    background = np.abs(np.asarray(template).astype(np.complex128))
    held = (count > 0).sum(axis=2)
    base = np.where(held > 0, (background * (count > 0)).sum(axis=2) / np.maximum(held, 1), 0.0)
    divisor = normalizer["scale"] + normalizer["epsilon"]
    window = np.abs(np.asarray(x, dtype=np.float64) * divisor + normalizer["mean"]) + base
    magnitude = np.maximum(np.maximum(window, base), abs(normalizer["mean"]))
    return tolerance["input_ulps"] * tolerance["spacing"] * magnitude / divisor


def parity(native, reference, artifact, tolerance):
    """The output of the native core against the output of the reference on the same records.

    native and reference are verified sensor outputs of one artifact. artifact is that artifact,
    opened, and tolerance the fp32 part of what the core declares (runtime.tolerance). The two
    are joined by window identifier. Returns (problems, facts): facts is a
    sense-runtime-parity/v1 object without the names of the two outputs.
    """
    problems = []
    if native.report["runtime"]["entry"] != cusense.NATIVE_RUNTIME or reference.report["runtime"]["entry"] != cusense.RUNTIME:
        raise SenseCoreError("parity compares an output of the native core with an output of the reference")
    if native.manifest["sha256"] != reference.manifest["sha256"] or artifact.sha256 != native.manifest["sha256"]:
        raise SenseCoreError("the two outputs and the artifact are not of one frozen artifact")
    mine, theirs = native.by_window, reference.by_window
    only = sorted(set(mine) ^ set(theirs))
    if only:
        problems.append(f"{len(only)} windows are scored by one runtime only, first: {only[0]}")
    # Every answer of the two streams, by its window where it has one: the same status.
    statuses = {"native": {}, "reference": {}}
    for name, output in (("native", native), ("reference", reference)):
        for line in output.decisions:
            statuses[name][line["status"]] = statuses[name].get(line["status"], 0) + 1
    live_only = {status: statuses["native"].get(status, 0) for status in cusense.NATIVE_STATUSES}
    if any(live_only.values()):
        problems.append(f"the native run skipped or lost windows ({live_only}): a parity run replays every record")
    if [line["status"] for line in native.decisions] != [line["status"] for line in reference.decisions]:
        problems.append("the two runtimes do not answer the records with the same statuses in the same order")
    normalizer = artifact.manifest["normalizer"]
    worst = {"input": 0.0, "input_of_bound": 0.0, "logit": 0.0, "probability": 0.0, "map": 0.0}
    flips, moved, agree, identical_inputs = [], [], 0, 0
    for window in sorted(set(mine) & set(theirs)):
        first, second = mine[window], theirs[window]
        if [item["opportunity"] for item in first["contributing"]] != [item["opportunity"] for item in second["contributing"]]:
            problems.append(f"{window}: the contributing observations differ")
            continue
        x, wanted = native.input_of(first).astype(np.float64), reference.input_of(second).astype(np.float64)
        bound = input_bound(wanted, artifact.template["values"], artifact.template["count"], normalizer, tolerance)
        difference = np.abs(x - wanted)
        identical_inputs += first["input"]["sha256"] == second["input"]["sha256"]
        logit = abs(first["score"]["logit"] - second["score"]["logit"])
        probability = abs(first["score"]["probability"] - second["score"]["probability"])
        cells = float(np.abs(native.map_of(first).astype(np.float64) - reference.map_of(second).astype(np.float64)).max())
        ratio = float((difference / np.maximum(bound, 1e-300)).max())
        worst = {"input": max(worst["input"], float(difference.max())), "input_of_bound": max(worst["input_of_bound"], ratio),
                 "logit": max(worst["logit"], logit), "probability": max(worst["probability"], probability),
                 "map": max(worst["map"], cells)}
        allowed = tolerance["logit_absolute"] + tolerance["logit_relative"] * abs(second["score"]["logit"])
        if ratio > 1.0:
            place = np.unravel_index(int(np.argmax(difference / np.maximum(bound, 1e-300))), difference.shape)
            problems.append(f"{window}: the input differs by {float(difference[place]):.3e} at antenna {place[0]}, "
                            f"subcarrier {place[1]}, {ratio:.2f} times its bound")
        if logit > allowed or probability > tolerance["probability_absolute"] or cells > tolerance["map_absolute"]:
            problems.append(f"{window}: the logit differs by {logit:.3e}, the probability by {probability:.3e} and a "
                            f"cell of the map by {cells:.3e}")
        if first["decision"]["present"] == second["decision"]["present"]:
            agree += 1
        else:
            margin = second["score"]["logit"] - second["decision"]["threshold_logit"]
            flips.append({"window": window, "native_logit": first["score"]["logit"],
                          "reference_logit": second["score"]["logit"], "threshold_logit": second["decision"]["threshold_logit"],
                          "margin": margin, "within_tolerance": abs(margin) <= allowed})
            if abs(margin) > allowed:
                problems.append(f"{window}: the decision differs at the threshold with a margin of {margin:+.3e}")
        if first["location"]["cell"] != second["location"]["cell"]:
            peak = abs(first["location"]["probability"] - second["location"]["probability"])
            moved.append({"window": window, "native_cell": first["location"]["cell"],
                          "reference_cell": second["location"]["cell"], "peak_difference": peak})
            if peak > tolerance["map_absolute"]:
                problems.append(f"{window}: the location differs with peaks {peak:.3e} apart")
    joined = len(set(mine) & set(theirs))
    facts = {"schema": PARITY_SCHEMA, "artifact_sha256": artifact.sha256, "joined": joined,
             "scored": {"native": len(mine), "reference": len(theirs)}, "statuses": statuses,
             "tolerance": {name: tolerance[name] for name in ("spacing", "input_ulps", "input_rule", "logit_absolute",
                                                              "logit_relative", "probability_absolute", "map_absolute")},
             "largest_input_difference": worst["input"], "largest_input_difference_of_its_bound": worst["input_of_bound"],
             "inputs_bit_identical": identical_inputs, "largest_logit_difference": worst["logit"],
             "largest_probability_difference": worst["probability"], "largest_map_difference": worst["map"],
             "decisions": {"compared": joined, "agree": agree, "flips": len(flips)}, "flips": flips,
             "locations_moved": moved,
             "native": {"device": native.report["runtime"]["device"], "engine": native.report["runtime"]["engine_backend"],
                        "engine_detail": native.report["runtime"]["engine_detail"],
                        "timing_valid": native.report["runtime"]["timing_valid"]},
             "reference": {"device": reference.report["runtime"]["device"]}}
    return problems, facts


# ---------------------------------------------------------------- latency
def notifications(output, records):
    """When the sensor learned of the last record of each scored window: host times in ns, in the order of output.scored.

    The native core marks it in every decision. The reference sensor reads its records through
    a consumer that keeps the time it read each one: records is the list of that consumer's
    report, and the time of a window is that of its last record. A window whose time is not
    known has None. A window identifier is not used here: a producer that publishes an archive
    twice gives the same window twice.
    """
    if output.report["runtime"]["entry"] == cusense.NATIVE_RUNTIME:
        return [line["marks"]["t_notify_ns"] for line in output.scored]
    read = {(item["epoch"], item["sequence"]): item.get("t_read_ns") for item in records or []}
    return [read.get((line["trigger"]["epoch"], line["trigger"]["sequence"])) for line in output.scored]


def latency(output, records=None, clock_map=None):
    """The three latency definitions over the scored decisions of one live output. A part of a sense-runtime-latency/v1 object.

    clock_map says how the run relates the radio clock to the host clock. With the map
    "ready_is_last_sample" the host time at which the last record of a window was ready stands
    for the end of its last sample: that is the map of a producer that publishes a record when
    its slot ends, and a receiver adds its own processing time to it. Without a map the
    end-to-end latency is not stated.
    """
    if clock_map not in (None, "ready_is_last_sample"):
        raise SenseCoreError(f"unknown clock map {clock_map!r}")
    native = output.report["runtime"]["entry"] == cusense.NATIVE_RUNTIME
    notified = notifications(output, records)
    post, boundary, whole, acquisition, late = [], [], [], [], 0
    stages = {name: [] for name in (("notify_to_submit", "ingest", "kernels", "engine", "collect") if native
                                    else ("frontend", "inference"))}
    for line, when in zip(output.scored, notified):
        if line["t_decision_ns"] is None or line["t_ready_ns"] is None:
            raise SenseCoreError(f"the decision of {line['window']['id']} carries no host times: it is no live decision")
        post.append((line["t_decision_ns"] - line["t_ready_ns"]) / 1e3)
        span = 1e6 * (line["time"]["t_last_sample"] - line["time"]["t_first_sample"]) / TC_PER_SECOND
        acquisition.append(span)
        if clock_map is not None:
            whole.append(span + post[-1])
        if when is not None:
            boundary.append((line["t_decision_ns"] - when) / 1e3)
        if native:
            marks = line["marks"]
            if when is not None:
                stages["notify_to_submit"].append((marks["t_submit_ns"] - when) / 1e3)
            stages["ingest"].append((marks["t_ingest_ns"] - marks["t_submit_ns"]) / 1e3)
            stages["kernels"].append((marks["t_kernels_ns"] - marks["t_ingest_ns"]) / 1e3)
            stages["engine"].append((marks["t_engine_ns"] - marks["t_kernels_ns"]) / 1e3)
            stages["collect"].append((line["t_decision_ns"] - marks["t_engine_ns"]) / 1e3)
            late += (line.get("deadline") or {}).get("outcome") == "late"
        else:
            stages["frontend"].append(line["frontend_ns"] / 1e3)
            stages["inference"].append(line["infer_ns"] / 1e3)
    runtime = output.report["runtime"]
    return {"runtime": runtime["entry"], "device": runtime["device"],
            "engine": runtime.get("engine_backend", "torch"), "engine_detail": runtime.get("engine_detail"),
            "timing_valid": bool(runtime.get("timing_valid", True)),
            "scored": len(output.scored), "counts": output.report["counts"],
            "post_window": dict(statistics(post), definition=DEFINITIONS["post_window"]),
            "cusense_boundary": dict(statistics(boundary), definition=DEFINITIONS["cusense_boundary"],
                                     notification="the mark t_notify of the core" if native else
                                     "the time the consumer of the reference sensor read the record"),
            "end_to_end": dict(statistics(whole), definition=DEFINITIONS["end_to_end"], clock_map=clock_map,
                               stated=clock_map is not None,
                               note="the map of this run: a record is ready when its slot ends, so the latency is the "
                                    "acquisition time of the window plus the post-window time. A receiver adds its "
                                    "own processing time to it: that run is S4.1's" if clock_map else
                                    "not stated: the run records no map between the radio clock and the host clock"),
            "acquisition": statistics(acquisition), "stages": {name: statistics(values) for name, values in stages.items()},
            "late": late if native else None}


def speedup(native, reference):
    """How the medians and the tails of two latency records compare, definition by definition. Not graded."""
    made = {}
    for name in ("post_window", "cusense_boundary", "end_to_end"):
        made[name] = {}
        for key in ("p50_us", "p99_us"):
            first, second = native[name][key], reference[name][key]
            made[name][key] = None if not first or second is None else {"native": first, "reference": second,
                                                                        "reference_over_native": second / first}
    return made


def reference_record(native, reference, context, earlier=None):
    """The native latency beside the published one: a cusense-reference/v1 object. Never graded.

    native and reference are latency records of the two runtimes on the same offered records,
    context says what the run was, and earlier is the cusense-reference/v1 object of the
    Sprint 3.5 run this model comes from, whose position error it carries along.
    """
    published = cusense_frontend.PUBLISHED
    boundary = native["cusense_boundary"]
    return {"schema": REFERENCE_SCHEMA, "graded": False,
            "aim": "comparable performance in software: the published latency belongs to one over-the-air system on "
                   "other hardware at five times the bandwidth, and is no target",
            "published": published,
            "measured": {
                "position_error": None if earlier is None else earlier["measured"]["position_error"],
                "position_error_from": None if earlier is None else "the Sprint 3.5 run of this artifact: the native "
                                                                     "core scores the same windows within tolerance",
                "latency": {
                    "boundary": "the cuSense boundary: " + DEFINITIONS["cusense_boundary"],
                    "published_us": published["live_inference_us"]["value"],
                    "published_serving_option": published["live_inference_us"]["serving_option"],
                    "native": {"runtime": native["runtime"], "device": native["device"], "engine": native["engine"],
                               "engine_detail": native["engine_detail"], "timing_valid": native["timing_valid"],
                               "cusense_boundary": boundary, "post_window": native["post_window"],
                               "stages": native["stages"]},
                    "reference": None if reference is None else {
                        "runtime": reference["runtime"], "device": reference["device"],
                        "cusense_boundary": reference["cusense_boundary"], "post_window": reference["post_window"],
                        "stages": reference["stages"]},
                    "comparison": None if reference is None else speedup(native, reference),
                    "comparable": bool(native["timing_valid"]),
                    "why_not_comparable": None if native["timing_valid"] else
                    "this run is not on the CUDA device with the TensorRT backend: the native showing is blocked"},
                "context": context},
            "differences": list(DIFFERENCES),
            "latency_comparison": "made here by the boundary of the published figure, from the notification of the last "
                                  "record of a window to the published decision, with the sensor alone on the device. "
                                  "The arms with a receiver beside it on one GPU belong to S4.1",
            "resolution": "the published 241 us covers 273 resource blocks on three DM-RS symbols, and this figure 51 "
                          "resource blocks on two: the kernels of this gate read about a ninth of the samples"}
