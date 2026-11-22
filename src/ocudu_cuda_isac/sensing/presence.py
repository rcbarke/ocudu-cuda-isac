"""Presence head on the causal features (Sprint 3.4, selector presence_head).

A sensor of Sprint 3.3 answers every record with a line, and a complete window carries a vector
of named values. This module is the reference of what turns that vector into a decision:

  columns     the head is bound to names. Its artifact lists the names of the whole vector, in
              the order of the specification it was trained on, and the columns it reads. A
              vector with other names, or the same names in another order, is refused.
  normalizer  zscore: per column, (x - mean) / scale with the mean and the deviation of the
              training split and of no other sample.
  model       linear: logit = w . z + b and probability = 1 / (1 + exp(-logit)), in float32, one
              vector at a time.
  detector    frozen_threshold: present when the logit lies above a threshold that was calibrated
              on the target-absent windows of the validation split and frozen.

The artifact, presence-head/v1, is one directory of two files. head.json holds the input
contract, the normalizer, the threshold and the hash of the weights, and its own hash over all of
them. weights.f32 holds one float32 per column and the bias. Nothing else of a training run is in
it: no sample, no label, no split.

Causality. A decision is made from one line of the feature stream, when that line is complete,
and is never revised. A line that carries no vector gives a decision without a score, under the
status of the line. A head reads the vector of a line and the hash of its specification and
nothing else of it.

Training is offline and is split before any window exists: every sample carries the split, the
scene and the pair that an evaluator assigned to it, and the fit refuses a scene in two splits, a
sample outside the training split in the normalizer or in the weights, and a sample outside the
validation split in the threshold.

Every entry carries runtime "reference": PyTorch in the interpreter of a sensor, on the CPU or on
a CUDA device that is selected when the head is loaded. This module defines the head and checks
the engine that the export route of Sprint 3.6 builds from the artifact, and no latency it
measures is a result. It imports the feature reference and the record reader of the sensing
package and nothing of a host, a simulator or an episode.
"""

import argparse
import fnmatch
import hashlib
import json
import math
from pathlib import Path
import platform
import struct
import sys
import tempfile
import time

import numpy as np
import torch

from . import features
from . import obs
from .features import FeaturesError
from .obs import ObsError

ARTIFACT_SCHEMA = "presence-head/v1"
DECISION_SCHEMA = "presence-decision/v1"
REPORT_SCHEMA = "presence-report/v1"
SAMPLE_SCHEMA = "presence-sample/v1"
SETTINGS_SCHEMA = "presence-train-config/v1"
HEAD_FILE = "head.json"
WEIGHTS_FILE = "weights.f32"
DECISIONS_FILE = "decisions.jsonl"
REPORT_FILE = "presence.json"
FEATURES_DIR = "features"
RUNTIME = features.RUNTIME
SPLITS = ("training", "validation", "held_out")
ARTIFACT_FIELDS = ("schema", "head_id", "entries", "input", "normalizer", "weights", "output", "threshold", "sha256")
ENTRY_FIELDS = ("normalizer", "model", "detector")
INPUT_FIELDS = ("features_entry", "spec_id", "spec_sha256", "vector", "columns", "shape", "dtype")
NORMALIZER_FIELDS = ("entry", "mean", "scale", "sha256")
WEIGHTS_FIELDS = ("file", "dtype", "layout", "count", "bytes", "sha256")
THRESHOLD_FIELDS = ("entry", "logit", "probability", "rule", "target_false_alarm", "negatives", "allowed", "exceeding")
SAMPLE_FIELDS = ("schema", "window", "episode", "split", "scene", "pair", "label", "spec_sha256", "names", "values")
SETTINGS_FIELDS = ("schema", "head_id", "columns", "l2", "max_iterations", "min_per_class", "threshold")
# Host times and the device of a run. Two runs on the same lines agree on every other field of a decision.
VARYING = ("t_ready_ns", "t_publish_ns", "t_features_ns", "t_decision_ns", "infer_ns", "device")
# What a head reads of a feature line. A field outside this list cannot change a score.
LINE_FIELDS = ("status", "spec_sha256", "features.names", "features.values")
OUTPUT = {"logit": "w . z + b with z = (x - mean) / scale, float32",
          "probability": "1 / (1 + exp(-logit))",
          "decision": "present when the logit lies above threshold.logit"}
LAYOUT = "one little-endian float32 per column of input.columns, in that order, then the bias"
THRESHOLD_RULE = ("allowed = floor(target_false_alarm * negatives); the threshold is the logit of the validation "
                  "negative that ranks allowed + 1 from the top, so that at most allowed negatives lie above it")
# The entries of this ticket. Each states where it runs. The export route of Sprint 3.6 builds the
# engine of the model under the same name.
_TORCH = {"device": "selected when the head is loaded: cpu or a cuda device", "library": "torch", "runtime": RUNTIME}
REGISTRY = {
    "normalizer": {"zscore": dict(_TORCH, does="per column, (x - mean) / scale in float32 with the mean and the "
                                              "deviation of the training split; a constant column has the scale 1")},
    "model": {"linear": dict(_TORCH, does="one weight per column and a bias: logit = w . z + b and probability = "
                                          "1 / (1 + exp(-logit)), float32, one vector at a time")},
    "detector": {"frozen_threshold": {"device": "cpu", "library": "python", "runtime": RUNTIME,
                                      "does": "present when the logit lies above a threshold that was calibrated on "
                                              "the target-absent windows of the validation split and frozen"}},
}
DEMO_COLUMNS = ("peak_to_median_db", "profile_db@*")


class PresenceError(ValueError):
    """An artifact, a vector, a sample set or a sensor output violates the documented interface."""


# ---------------------------------------------------------------- helpers
def canonical(value):
    """The bytes a hash of a JSON value is taken over."""
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _fields(value, fields, name):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise PresenceError(f"{name} requires exactly {', '.join(fields)}")
    return value


def _numbers(values, name, count=None):
    if not isinstance(values, list) or (count is not None and len(values) != count) \
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise PresenceError(f"{name} must be a list of {'' if count is None else f'{count} '}finite numbers")
    return [float(value) for value in values]


def _names(values, name):
    if not isinstance(values, list) or not values or any(not isinstance(value, str) or not value or not value.isascii()
                                                         for value in values) or len(set(values)) != len(values):
        raise PresenceError(f"{name} must be a nonempty list of distinct ASCII names")
    return values


def resolve():
    """The registry entries of a head, by stage."""
    return {stage: dict(REGISTRY[stage][entry], entry=entry)
            for stage, entry in (("normalizer", "zscore"), ("model", "linear"), ("detector", "frozen_threshold"))}


def default_device():
    """The CUDA device of this process when PyTorch reaches one, and the CPU otherwise."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def device_of(name):
    """A torch device from its name. A CUDA device that this process cannot reach is refused."""
    try:
        device = torch.device(name)
    except (RuntimeError, TypeError) as error:
        raise PresenceError(f"unknown device {name!r}: {error}") from None
    if device.type not in ("cpu", "cuda"):
        raise PresenceError(f"a head runs on the cpu or on a cuda device, and {name!r} is neither")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise PresenceError(f"the device {name} was asked for and PyTorch reaches no CUDA device in this process")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise PresenceError(f"the device {name} was asked for and this process sees {torch.cuda.device_count()} "
                                "CUDA devices")
    return device


def device_facts(device):
    """What a report records of the device of a head."""
    facts = {"device": str(device), "type": device.type, "torch": torch.__version__,
             "cuda_runtime": torch.version.cuda, "cuda_available": bool(torch.cuda.is_available())}
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        facts.update(device=f"cuda:{index}", name=torch.cuda.get_device_name(index),
                     capability=".".join(str(part) for part in torch.cuda.get_device_capability(index)))
    return facts


# ---------------------------------------------------------------- columns
def select_columns(vector, patterns):
    """The names of a vector that a list of patterns selects, in the order of the vector.

    A pattern is a name, or a name with * for any run of characters. Every pattern must select at
    least one name, so that a misspelt column is refused and not dropped.
    """
    _names(vector, "the vector of a specification")
    if not isinstance(patterns, (list, tuple)) or not patterns or any(not isinstance(item, str) or not item
                                                                      for item in patterns):
        raise PresenceError("columns must be a nonempty list of names or patterns")
    for pattern in patterns:
        if not any(fnmatch.fnmatchcase(name, pattern) for name in vector):
            raise PresenceError(f"the column {pattern!r} selects no value of the vector ({', '.join(vector)})")
    return [name for name in vector if any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)]


def bind(vector, columns, names):
    """Where the columns of a head lie in a vector that arrives with its names.

    The names must be those of the vector the head was trained on, in that order: the names of a
    specification are a contract, and a vector of another specification has other names.
    """
    if list(names) != list(vector):
        if sorted(names) == sorted(vector):
            place = next(index for index, (mine, theirs) in enumerate(zip(vector, names)) if mine != theirs)
            raise PresenceError(f"the vector has its values in another order than the head was trained on: position "
                                f"{place} is {names[place]!r} and the head expects {vector[place]!r}")
        missing = [name for name in vector if name not in names]
        extra = [name for name in names if name not in vector]
        raise PresenceError(f"the vector is not that of the head: {len(names)} names against {len(vector)}"
                            + (f", without {missing[0]!r}" if missing else "")
                            + (f", with {extra[0]!r}" if extra else ""))
    return [vector.index(name) for name in columns]


# ---------------------------------------------------------------- artifact
def normalizer_sha256(mean, scale):
    return _sha256(canonical({"entry": "zscore", "mean": mean, "scale": scale}))


def identity(manifest):
    """The hash of an artifact: SHA-256 over every field of head.json but this one, in canonical form."""
    return _sha256(canonical({key: value for key, value in manifest.items() if key != "sha256"}))


def pack_weights(weights, bias):
    """The bytes of weights.f32. A value that float32 cannot hold is refused."""
    values = _numbers(list(weights), "the weights") + _numbers([bias], "the bias")
    try:
        data = struct.pack(f"<{len(values)}f", *values)
    except (OverflowError, struct.error):
        raise PresenceError("a weight does not fit a float32") from None
    if not all(math.isfinite(value) for value in struct.unpack(f"<{len(values)}f", data)):
        raise PresenceError("a weight does not fit a float32")
    return data


def check_manifest(manifest):
    """Check a presence-head/v1 object and return it. The weights file is checked by open_artifact."""
    _fields(manifest, ARTIFACT_FIELDS, "an artifact")
    if manifest["schema"] != ARTIFACT_SCHEMA:
        raise PresenceError(f"unsupported artifact schema {manifest['schema']!r}")
    if not isinstance(manifest["head_id"], str) or not manifest["head_id"] or not manifest["head_id"].isascii():
        raise PresenceError("head_id must be a nonempty ASCII string")
    entries = _fields(manifest["entries"], ENTRY_FIELDS, "entries")
    for stage in ENTRY_FIELDS:
        if entries[stage] not in REGISTRY[stage]:
            raise PresenceError(f"unknown {stage} entry {entries[stage]!r} (registered: "
                                f"{', '.join(sorted(REGISTRY[stage]))})")
    wanted = _fields(manifest["input"], INPUT_FIELDS, "input")
    vector, columns = _names(wanted["vector"], "input.vector"), _names(wanted["columns"], "input.columns")
    if [name for name in vector if name in columns] != columns:
        raise PresenceError("input.columns must be names of input.vector, in the order of the vector")
    if wanted["shape"] != [1, len(columns)] or wanted["dtype"] != "<f4":
        raise PresenceError(f"the input of a head is one vector of its columns as <f4: shape [1, {len(columns)}]")
    for name in ("features_entry", "spec_id", "spec_sha256"):
        if not isinstance(wanted[name], str) or not wanted[name]:
            raise PresenceError(f"input.{name} must be a nonempty string")
    normalizer = _fields(manifest["normalizer"], NORMALIZER_FIELDS, "normalizer")
    mean = _numbers(normalizer["mean"], "normalizer.mean", len(columns))
    scale = _numbers(normalizer["scale"], "normalizer.scale", len(columns))
    if normalizer["entry"] != entries["normalizer"] or any(not value > 0 for value in scale):
        raise PresenceError("the normalizer must be the entry of the artifact with a positive scale per column")
    if normalizer["sha256"] != normalizer_sha256(mean, scale):
        raise PresenceError("the normalizer does not have the hash the artifact names: the preprocessing state changed")
    weights = _fields(manifest["weights"], WEIGHTS_FIELDS, "weights")
    count = len(columns) + 1
    if (weights["file"], weights["dtype"], weights["layout"], weights["count"], weights["bytes"]) \
            != (WEIGHTS_FILE, "<f4", LAYOUT, count, 4 * count):
        raise PresenceError(f"the weights of a head are {count} values as <f4 in {WEIGHTS_FILE}: one per column and "
                            "the bias")
    if manifest["output"] != OUTPUT:
        raise PresenceError("the output of a head is the logit, the probability and the decision of this module")
    threshold = _fields(manifest["threshold"], THRESHOLD_FIELDS, "threshold")
    _numbers([threshold["logit"], threshold["probability"], threshold["target_false_alarm"]], "the threshold")
    if threshold["entry"] != entries["detector"] or threshold["rule"] != THRESHOLD_RULE \
            or not 0 <= threshold["target_false_alarm"] < 1 or not 0 <= threshold["probability"] <= 1 \
            or any(type(threshold[name]) is not int or threshold[name] < 0 for name in ("negatives", "allowed", "exceeding")) \
            or threshold["exceeding"] > threshold["allowed"] or threshold["negatives"] < 1 \
            or threshold["allowed"] != math.floor(threshold["target_false_alarm"] * threshold["negatives"]):
        raise PresenceError("the threshold does not follow its rule: " + THRESHOLD_RULE)
    if manifest["sha256"] != identity(manifest):
        raise PresenceError("head.json does not have the hash it names: the artifact was altered after it was frozen")
    return manifest


class Artifact:
    """A verified presence-head/v1 artifact: its manifest and the bytes of its weights."""

    def __init__(self, directory, manifest, weights):
        self.directory, self.manifest, self.weights = directory, manifest, weights
        self.sha256 = manifest["sha256"]
        self.vector, self.columns = manifest["input"]["vector"], manifest["input"]["columns"]
        self.threshold = manifest["threshold"]["logit"]

    def model(self):
        """What a decision names of its model."""
        manifest = self.manifest
        return {"head_id": manifest["head_id"], "entry": manifest["entries"]["model"],
                "artifact_sha256": self.sha256, "weights_sha256": manifest["weights"]["sha256"],
                "normalizer_sha256": manifest["normalizer"]["sha256"]}


def write_artifact(directory, head_id, spec_id, spec_sha256, vector, columns, normalizer, weights, bias, threshold,
                   features_entry="dd_compact"):
    """Write a presence-head/v1 artifact into a new directory and return its manifest.

    normalizer is {"mean", "scale"} with one value per column, threshold the record that
    calibrate_threshold returns. The directory appears only when it is complete.
    """
    directory = Path(directory)
    if directory.exists():
        raise PresenceError(f"{directory} already exists. An artifact is never overwritten.")
    data = pack_weights(weights, bias)
    mean = _numbers(list(normalizer["mean"]), "normalizer.mean")
    scale = _numbers(list(normalizer["scale"]), "normalizer.scale")
    manifest = {"schema": ARTIFACT_SCHEMA, "head_id": head_id,
                "entries": {"normalizer": "zscore", "model": "linear", "detector": "frozen_threshold"},
                "input": {"features_entry": features_entry, "spec_id": spec_id, "spec_sha256": spec_sha256,
                          "vector": list(vector), "columns": list(columns), "shape": [1, len(columns)], "dtype": "<f4"},
                "normalizer": {"entry": "zscore", "mean": mean, "scale": scale, "sha256": normalizer_sha256(mean, scale)},
                "weights": {"file": WEIGHTS_FILE, "dtype": "<f4", "layout": LAYOUT, "count": len(data) // 4,
                            "bytes": len(data), "sha256": _sha256(data)},
                "output": dict(OUTPUT), "threshold": dict(threshold)}
    manifest["sha256"] = identity(manifest)
    check_manifest(manifest)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory.parent) as temporary:
        staging = Path(temporary) / "artifact"
        staging.mkdir()
        (staging / WEIGHTS_FILE).write_bytes(data)
        (staging / HEAD_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="ascii")
        staging.rename(directory)
    return manifest


def open_artifact(directory, sha256=None):
    """Verify an artifact and return it.

    head.json must be a presence-head/v1 object with the hash it names, weights.f32 must have the
    size and the hash that head.json names and hold finite values, and the directory must hold
    nothing else. With sha256 the artifact must be the one that a freeze record names.
    """
    directory = Path(directory)
    try:
        manifest = json.loads((directory / HEAD_FILE).read_text(encoding="ascii"))
        data = (directory / WEIGHTS_FILE).read_bytes()
    except (OSError, ValueError) as error:
        raise PresenceError(f"{directory} is not a complete presence head: {error}") from None
    check_manifest(manifest)
    if len(data) != manifest["weights"]["bytes"] or _sha256(data) != manifest["weights"]["sha256"]:
        raise PresenceError("weights.f32 does not have the size and the hash that head.json names: the checkpoint was "
                            "altered")
    if not all(math.isfinite(value) for value in struct.unpack(f"<{len(data) // 4}f", data)):
        raise PresenceError("weights.f32 holds a value that is not finite")
    extra = sorted(path.name for path in directory.iterdir() if path.name not in (HEAD_FILE, WEIGHTS_FILE))
    if extra:
        raise PresenceError(f"the artifact holds {extra[0]}: a deployed head is head.json and weights.f32 and nothing "
                            "else")
    if sha256 is not None and manifest["sha256"] != sha256:
        raise PresenceError(f"the artifact has the hash {manifest['sha256']}, and {sha256} was asked for")
    return Artifact(directory, manifest, data)


# ---------------------------------------------------------------- head
class Scorer:
    """The arithmetic of a head on one device: normalizer and linear model in float32, one vector at a time."""

    def __init__(self, mean, scale, weights, bias, device="cpu"):
        self.device = device if isinstance(device, torch.device) else device_of(device)
        self.columns = len(mean)
        if len(scale) != self.columns or len(weights) != self.columns:
            raise PresenceError("the normalizer and the weights do not have one value per column")
        self._mean = torch.tensor(mean, dtype=torch.float32).reshape(1, -1).to(self.device)
        self._scale = torch.tensor(scale, dtype=torch.float32).reshape(1, -1).to(self.device)
        self._weights = torch.tensor(weights, dtype=torch.float32).reshape(1, -1).to(self.device)
        self._bias = torch.tensor([bias], dtype=torch.float32).to(self.device)

    def predict(self, x):
        """Score one vector. x is float32 of shape [1, columns]. Returns (logit, probability, time in ns).

        The time runs from the vector in host memory to both values back in host memory, so on a
        CUDA device it holds the transfer in both directions and the wait for the device.
        """
        if isinstance(x, np.ndarray):
            if x.dtype != np.float32 or x.shape != (1, self.columns):
                raise PresenceError(f"a head scores one vector of shape [1, {self.columns}] as float32, and this one "
                                    f"has the shape {list(x.shape)} as {x.dtype}")
            x = torch.from_numpy(np.ascontiguousarray(x))
        elif isinstance(x, torch.Tensor):
            if x.dtype != torch.float32 or tuple(x.shape) != (1, self.columns):
                raise PresenceError(f"a head scores one vector of shape [1, {self.columns}] as float32, and this one "
                                    f"has the shape {list(x.shape)} as {str(x.dtype).replace('torch.', '')}")
        else:
            raise PresenceError("a head scores a NumPy array or a tensor of shape [1, columns] as float32")
        if not bool(torch.isfinite(x).all()):
            raise PresenceError("the vector holds a value that is not finite")
        started = time.perf_counter_ns()
        with torch.inference_mode():
            z = (x.to(self.device) - self._mean) / self._scale
            logit = torch.nn.functional.linear(z, self._weights, self._bias).reshape(1)
            both = torch.cat((logit, torch.sigmoid(logit))).cpu()
        elapsed = time.perf_counter_ns() - started
        logit, probability = float(both[0]), float(both[1])
        if not math.isfinite(logit) or not math.isfinite(probability):
            raise PresenceError("the score of the vector is not finite")
        return logit, probability, elapsed


class Head:
    """One frozen presence head, loaded once in a process and bound to one device."""

    LOADS = 0  # how often this process built a head from an artifact

    def __init__(self, artifact, device="cpu"):
        self.artifact = artifact if isinstance(artifact, Artifact) else open_artifact(artifact)
        manifest = self.artifact.manifest
        values = struct.unpack(f"<{manifest['weights']['count']}f", self.artifact.weights)
        self.scorer = Scorer(manifest["normalizer"]["mean"], manifest["normalizer"]["scale"], values[:-1], values[-1],
                             device)
        self.device = self.scorer.device
        self.vector, self.columns = self.artifact.vector, self.artifact.columns
        self.threshold = self.artifact.threshold
        self._places = [self.vector.index(name) for name in self.columns]
        self._bound = tuple(self.vector)
        Head.LOADS += 1

    def prepare(self, names, values):
        """The input of the head from a vector that arrives with its names: float32 of shape [1, columns]."""
        if tuple(names) != self._bound:
            bind(self.vector, self.columns, list(names))
        if len(values) != len(names):
            raise PresenceError(f"the vector names {len(names)} values and holds {len(values)}")
        try:
            with np.errstate(over="ignore"):
                x = np.array([[values[place] for place in self._places]], dtype=np.float32)
        except (TypeError, ValueError, OverflowError):
            raise PresenceError("the vector holds a value that is no number") from None
        if not np.isfinite(x).all():
            raise PresenceError("the vector holds a value that is not finite as float32")
        return x

    def predict(self, x):
        return self.scorer.predict(x)

    def score(self, names, values):
        """Score one named vector. Returns the logit, the probability, the decision and the digest of the input."""
        x = self.prepare(names, values)
        logit, probability, elapsed = self.scorer.predict(x)
        return {"logit": logit, "probability": probability, "present": logit > self.threshold,
                "input_sha256": _sha256(x.astype("<f4").tobytes()), "infer_ns": elapsed}

    def warm_up(self, rounds=3):
        """Score the mean of the training split a few times. Returns the time it took in ms.

        The first inference of a process creates the device context and loads the libraries of
        the operators: seconds on a CUDA device. A sensor pays it before it attaches.
        """
        started = time.perf_counter()
        x = np.array([self.artifact.manifest["normalizer"]["mean"]], dtype=np.float32)
        for _ in range(max(rounds, 1)):
            self.scorer.predict(x)
        return 1000.0 * (time.perf_counter() - started)

    def decide(self, line, clock=None):
        """The decision of one line of the feature stream.

        A line with a vector is scored, and its decision names the contributing records, the
        times of its window, the digest of its input and the frozen model. Any other line gives a
        decision without a score under the status of the line.
        """
        made = {"schema": DECISION_SCHEMA, "line": line["line"], "status": line["status"], "reason": line["reason"],
                "window": line["window"], "trigger": line["trigger"], "runtime": RUNTIME, "device": str(self.device),
                "model": self.artifact.model(), "score": None, "decision": None, "input": None, "infer_ns": None,
                "t_ready_ns": line["t_ready_ns"], "t_publish_ns": line["t_publish_ns"],
                "t_features_ns": line["t_features_ns"], "t_decision_ns": None}
        if line["status"] == "ok":
            wanted = self.artifact.manifest["input"]
            if line["spec_sha256"] != wanted["spec_sha256"]:
                raise PresenceError(f"the line was computed with the specification {line['spec_sha256']}, and the head "
                                    f"was trained on {wanted['spec_sha256']} ({wanted['spec_id']})")
            scored = self.score(line["features"]["names"], line["features"]["values"])
            made.update(score={"logit": scored["logit"], "probability": scored["probability"]},
                        decision={"present": scored["present"], "threshold_logit": self.threshold,
                                  "detector": self.artifact.manifest["entries"]["detector"]},
                        input={"spec_sha256": line["spec_sha256"], "features_entry": wanted["features_entry"],
                               "columns": len(self.columns), "sha256": scored["input_sha256"]},
                        infer_ns=scored["infer_ns"], contributing=line["contributing"], time=line["time"])
        made["t_decision_ns"] = None if clock is None else clock()
        return made


def comparable(decision):
    """A decision without the host times and the device of its run."""
    return {key: value for key, value in decision.items() if key not in VARYING}


def decide_lines(head, lines, clock=None):
    """One decision per line of a feature stream, in order."""
    return [head.decide(line, clock) for line in lines]


def run_records(records, spec, head, clock=None):
    """Push records through a new stream and score every answer at once. Returns (lines, views, stream, decisions)."""
    stream, lines, decisions = features.Stream(spec, clock), [], []
    for record in records:
        for line in stream.push(record):
            lines.append(line)
            decisions.append(head.decide(line, clock))
    return lines, stream.views, stream, decisions


# ---------------------------------------------------------------- training
def check_settings(settings):
    """Check a presence-train-config/v1 object and return it."""
    _fields(settings, SETTINGS_FIELDS, "a training configuration")
    if settings["schema"] != SETTINGS_SCHEMA:
        raise PresenceError(f"unsupported training configuration schema {settings['schema']!r}")
    if not isinstance(settings["head_id"], str) or not settings["head_id"] or not settings["head_id"].isascii():
        raise PresenceError("head_id must be a nonempty ASCII string")
    if not isinstance(settings["columns"], list) or not settings["columns"] \
            or any(not isinstance(item, str) or not item for item in settings["columns"]):
        raise PresenceError("columns must be a nonempty list of names or patterns")
    if type(settings["l2"]) not in (int, float) or not 0 < settings["l2"] <= 1e3:
        raise PresenceError("l2 must lie above 0 and at or below 1000: without it a separable split has no finite fit")
    for name, low, high in (("max_iterations", 1, 100000), ("min_per_class", 1, 10**6)):
        if type(settings[name]) is not int or not low <= settings[name] <= high:
            raise PresenceError(f"{name} must be an integer from {low} to {high}")
    threshold = _fields(settings["threshold"], ("target_false_alarm", "min_negatives"), "threshold")
    if type(threshold["target_false_alarm"]) not in (int, float) or not 0 <= threshold["target_false_alarm"] < 1:
        raise PresenceError("threshold.target_false_alarm must lie from 0 to below 1")
    if type(threshold["min_negatives"]) is not int or threshold["min_negatives"] < 1:
        raise PresenceError("threshold.min_negatives must be a positive integer")
    return settings


def demo_settings(head_id="dd-linear-v1"):
    """The training configuration of the delay-Doppler branch of the demonstration."""
    return check_settings({"schema": SETTINGS_SCHEMA, "head_id": head_id, "columns": list(DEMO_COLUMNS), "l2": 0.01,
                           "max_iterations": 200, "min_per_class": 20,
                           "threshold": {"target_false_alarm": 0.05, "min_negatives": 20}})


def check_sample(sample):
    """Check one presence-sample/v1 row and return it."""
    _fields(sample, SAMPLE_FIELDS, "a sample")
    if sample["schema"] != SAMPLE_SCHEMA or sample["split"] not in SPLITS or sample["label"] not in (0, 1) \
            or type(sample["label"]) is not int:
        raise PresenceError("a sample is a presence-sample/v1 row with a split and a label of 0 or 1")
    for name in ("window", "episode", "scene", "pair", "spec_sha256"):
        if not isinstance(sample[name], str) or not sample[name]:
            raise PresenceError(f"the {name} of a sample must be a nonempty string")
    _names(sample["names"], "the names of a sample")
    _numbers(sample["values"], "the values of a sample", len(sample["names"]))
    return sample


def check_split(training, validation):
    """Refuse a sample set that is not split before its windows. Returns the facts of the split.

    Every row of a file carries the split of that file. No window, episode key, pair or scene
    appears in both, every row has the same specification and the same names, and no row of the
    held-out split is in either.
    """
    seen = {}
    for role, rows in (("training", training), ("validation", validation)):
        for row in rows:
            check_sample(row)
            if row["split"] != role:
                raise PresenceError(f"the window {row['window']} of the {row['split']} split is among the {role} "
                                    "samples" + (": a held-out sample is read after the freeze and never before"
                                                 if row["split"] == "held_out" else ""))
        seen[role] = {name: {row[name] for row in rows} for name in ("window", "episode", "pair", "scene")}
        if len(seen[role]["window"]) != len(rows):
            raise PresenceError(f"a window is among the {role} samples twice")
    for name, text in (("scene", "scene"), ("pair", "matched pair"), ("episode", "episode key"), ("window", "window")):
        both = sorted(seen["training"][name] & seen["validation"][name])
        if both:
            raise PresenceError(f"the {text} {both[0]} is in the training and in the validation split: a split is made "
                                "by scene before any window is cut")
    rows = list(training) + list(validation)
    if not rows:
        raise PresenceError("there is no sample")
    if len({row["spec_sha256"] for row in rows}) != 1 or any(row["names"] != rows[0]["names"] for row in rows):
        raise PresenceError("the samples were not computed with one specification: their names or its hash differ")
    return {"spec_sha256": rows[0]["spec_sha256"], "vector": rows[0]["names"],
            "scenes": {role: sorted(seen[role]["scene"]) for role in seen},
            "pairs": {role: len(seen[role]["pair"]) for role in seen}}


def _matrix(samples, vector, columns):
    places = bind(vector, columns, vector)
    for sample in samples:
        if sample["names"] != vector:
            bind(vector, columns, sample["names"])
    return np.array([[sample["values"][place] for place in places] for sample in samples], dtype=np.float64)


def _only(samples, split, purpose):
    for sample in samples:
        if sample["split"] != split:
            raise PresenceError(f"the window {sample['window']} of the {sample['split']} split was given to "
                                f"{purpose}, which reads the {split} split and no other sample")


def fit_normalizer(samples, vector, columns):
    """Mean and deviation of every column over the training split. Returns {"entry", "mean", "scale", "constant"}.

    A sample of another split is refused. A column that does not vary keeps the scale 1 and is
    listed, so that the head stays finite on it.
    """
    _only(samples, "training", "the fit of the normalizer")
    if not samples:
        raise PresenceError("the normalizer needs a training sample")
    x = _matrix(samples, vector, columns).astype(np.float32).astype(np.float64)
    mean, deviation = x.mean(axis=0), x.std(axis=0)
    constant = [name for name, value, centre in zip(columns, deviation, mean)
                if not value > 1e-9 * max(1.0, abs(centre))]
    scale = [1.0 if name in constant else float(value) for name, value in zip(columns, deviation)]
    return {"entry": "zscore", "mean": [float(value) for value in mean], "scale": scale, "constant": constant}


def normalized(samples, vector, columns, normalizer):
    """The columns of samples after the normalizer, in the float32 arithmetic of a head, as float64."""
    x = _matrix(samples, vector, columns).astype(np.float32)
    mean = np.array(normalizer["mean"], dtype=np.float32)
    scale = np.array(normalizer["scale"], dtype=np.float32)
    return ((x - mean) / scale).astype(np.float64)


def fit_head(samples, vector, columns, normalizer, l2, max_iterations=200, min_per_class=1):
    """Fit the linear head on the training split. Returns (weights, bias, history).

    The loss is the mean binary cross-entropy of the logits plus l2 / 2 times the squared norm of
    the weights, minimised from zero by L-BFGS in float64 on the CPU: the same samples give the
    same weights. A sample of another split is refused.
    """
    _only(samples, "training", "the fit of the weights")
    labels = [sample["label"] for sample in samples]
    counts = (labels.count(0), labels.count(1))
    if min(counts) < min_per_class:
        raise PresenceError(f"the training split holds {counts[1]} target-present and {counts[0]} target-absent "
                            f"windows, and the fit needs {min_per_class} of each")
    z = torch.tensor(normalized(samples, vector, columns, normalizer), dtype=torch.float64)
    y = torch.tensor(labels, dtype=torch.float64)
    weights = torch.zeros(z.shape[1], dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([weights, bias], lr=1.0, max_iter=max_iterations, max_eval=4 * max_iterations,
                                  tolerance_grad=1e-10, tolerance_change=1e-14, history_size=20,
                                  line_search_fn="strong_wolfe")
    losses = []

    def objective():
        fit = torch.nn.functional.binary_cross_entropy_with_logits(z @ weights + bias, y)
        return fit, fit + 0.5 * l2 * (weights * weights).sum()

    def closure():
        optimizer.zero_grad()
        _, loss = objective()
        loss.backward()
        losses.append(float(loss.detach()))
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        fit, loss = objective()
    weights.grad = bias.grad = None
    _, last = objective()
    last.backward()
    gradient = float(torch.sqrt((weights.grad ** 2).sum() + (bias.grad ** 2).sum()))
    history = {"optimizer": "L-BFGS with a strong Wolfe line search, from zero, float64 on the cpu",
               "loss": "mean binary cross-entropy of the logits plus l2 / 2 times the squared norm of the weights",
               "l2": l2, "evaluations": len(losses), "max_iterations": max_iterations,
               "initial_loss": losses[0] if losses else None, "final_loss": float(loss),
               "final_cross_entropy": float(fit), "gradient_norm": gradient,
               "samples": {"target_present": counts[1], "target_absent": counts[0]}}
    made = [float(value) for value in weights.detach()], float(bias.detach()[0])
    if not all(math.isfinite(value) for value in made[0] + [made[1]]):
        raise PresenceError("the fit did not give finite weights")
    return made[0], made[1], history


def logits_of(samples, vector, columns, normalizer, weights, bias):
    """The logit of every sample as a head on the CPU computes it: float32, one vector at a time."""
    data = struct.unpack(f"<{len(weights) + 1}f", pack_weights(weights, bias))
    scorer = Scorer(normalizer["mean"], normalizer["scale"], data[:-1], data[-1], "cpu")
    x = _matrix(samples, vector, columns).astype(np.float32)
    return [scorer.predict(x[index:index + 1])[0] for index in range(len(samples))]


def calibrate_threshold(samples, logits, target_false_alarm, min_negatives=1):
    """The frozen threshold from the target-absent windows of the validation split.

    samples and logits are the validation split and the logit of each sample, as the deployed
    arithmetic gives it. A sample of another split is refused. Returns the threshold record of an
    artifact. Rule: allowed = floor(target_false_alarm * negatives), and the threshold is the
    logit of the negative that ranks allowed + 1 from the top. A positive is never read.
    """
    _only(samples, "validation", "the calibration of the threshold")
    if len(samples) != len(logits):
        raise PresenceError("the calibration needs one logit per validation sample")
    if type(target_false_alarm) not in (int, float) or not 0 <= target_false_alarm < 1:
        raise PresenceError("the false-alarm target must lie from 0 to below 1")
    negatives = sorted((logit for sample, logit in zip(samples, logits) if sample["label"] == 0), reverse=True)
    if len(negatives) < max(min_negatives, 1):
        raise PresenceError(f"the validation split holds {len(negatives)} target-absent windows, and the calibration "
                            f"needs {max(min_negatives, 1)}")
    allowed = math.floor(target_false_alarm * len(negatives))
    logit = float(negatives[allowed])
    return {"entry": "frozen_threshold", "logit": logit, "probability": 1.0 / (1.0 + math.exp(-logit)),
            "rule": THRESHOLD_RULE, "target_false_alarm": float(target_false_alarm), "negatives": len(negatives),
            "allowed": allowed, "exceeding": sum(value > logit for value in negatives)}


def read_samples(path):
    """The rows of a sample file: one presence-sample/v1 object per line."""
    try:
        rows = [json.loads(text) for text in Path(path).read_text(encoding="ascii").splitlines() if text.strip()]
    except (OSError, ValueError) as error:
        raise PresenceError(f"cannot read the samples {path}: {error}") from None
    return [check_sample(row) for row in rows]


def write_samples(path, rows):
    """Write sample rows, one per line. The file must not exist."""
    path = Path(path)
    if path.exists():
        raise PresenceError(f"{path} already exists. A sample file is never overwritten.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(check_sample(row), sort_keys=True) + "\n" for row in rows), encoding="ascii")
    return path


# ---------------------------------------------------------------- output
def write_output(directory, head, decisions, source, features_dir=FEATURES_DIR):
    """Write decisions.jsonl and presence.json beside a feature output. Returns the report.

    The directory must exist and hold neither file. features_dir names the feature output the
    decisions were made from: a directory inside this one, or a path.
    """
    directory = Path(directory)
    if not directory.is_dir() or any((directory / name).exists() for name in (DECISIONS_FILE, REPORT_FILE)):
        raise PresenceError(f"the output directory {directory} does not exist or holds a result already")
    (directory / DECISIONS_FILE).write_text("".join(json.dumps(line, sort_keys=True) + "\n" for line in decisions),
                                            encoding="ascii")
    scored = [line for line in decisions if line["status"] == "ok"]
    counts = {status: sum(line["status"] == status for line in decisions) for status in features.STATUSES}
    counts.update(decisions=len(decisions), scored=len(scored),
                  present=sum(line["decision"]["present"] for line in scored),
                  absent=sum(not line["decision"]["present"] for line in scored))
    report = {"schema": REPORT_SCHEMA, "artifact": {"directory": str(head.artifact.directory),
                                                    "manifest": head.artifact.manifest, "model": head.artifact.model()},
              "entries": resolve(), "line_fields": list(LINE_FIELDS),
              "runtime": dict(device_facts(head.device), entry=RUNTIME, python=platform.python_version(),
                              numpy=np.__version__, machine=platform.machine(),
                              module_sha256=_sha256(Path(__file__).read_bytes())),
              "loads": Head.LOADS, "features": str(features_dir), "source": source, "counts": counts}
    (directory / REPORT_FILE).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return report


class Output:
    """A verified output of a presence sensor."""

    def __init__(self, directory, report, decisions, feature_output):
        self.directory, self.report, self.decisions, self.features = directory, report, decisions, feature_output
        self.manifest = report["artifact"]["manifest"]
        self.scored = [line for line in decisions if line["status"] == "ok"]
        self.by_window = {line["window"]["id"]: line for line in self.scored}


def open_output(directory):
    """Verify the output of a presence sensor and return it.

    The feature output it names verifies with its own reader. The report embeds an artifact
    manifest that has the hash it names. There is one decision per feature line, with the number,
    the status and the window of that line. A scored decision names the model of the report, the
    digest of the columns that its line holds, a finite logit with the probability that follows
    from it, and the decision that the frozen threshold gives. No other decision carries a score.
    """
    directory = Path(directory)
    try:
        report = json.loads((directory / REPORT_FILE).read_text(encoding="ascii"))
        decisions = [json.loads(text) for text in (directory / DECISIONS_FILE).read_text(encoding="ascii").splitlines()
                     if text.strip()]
    except (OSError, ValueError) as error:
        raise PresenceError(f"{directory} is not a complete presence output: {error}") from None
    if not isinstance(report, dict) or report.get("schema") != REPORT_SCHEMA:
        raise PresenceError(f"{directory / REPORT_FILE} is not a {REPORT_SCHEMA} report")
    try:
        manifest = check_manifest(report["artifact"]["manifest"])
        model = report["artifact"]["model"]
        place = Path(report["features"])
        feature_output = features.open_output(place if place.is_absolute() else directory / place)
    except (KeyError, TypeError) as error:
        raise PresenceError(f"the report of {directory} is incomplete: {error!r}") from None
    except FeaturesError as error:
        raise PresenceError(f"the feature output of {directory} does not verify: {error}") from None
    wanted = {"head_id": manifest["head_id"], "entry": manifest["entries"]["model"], "artifact_sha256": manifest["sha256"],
              "weights_sha256": manifest["weights"]["sha256"], "normalizer_sha256": manifest["normalizer"]["sha256"]}
    if model != wanted or report.get("entries") != resolve() or (report.get("runtime") or {}).get("entry") != RUNTIME:
        raise PresenceError("the report does not name the model of its artifact, the entries of the registry and the "
                            "reference runtime")
    if feature_output.report["spec_sha256"] != manifest["input"]["spec_sha256"]:
        raise PresenceError("the features were computed with another specification than the head was trained on")
    if len(decisions) != len(feature_output.lines):
        raise PresenceError(f"{len(decisions)} decisions for {len(feature_output.lines)} lines of the feature stream")
    places = [manifest["input"]["vector"].index(name) for name in manifest["input"]["columns"]]
    threshold = manifest["threshold"]["logit"]
    for number, (decision, line) in enumerate(zip(decisions, feature_output.lines)):
        if not isinstance(decision, dict) or decision.get("schema") != DECISION_SCHEMA or decision.get("line") != number \
                or decision.get("status") != line["status"] or decision.get("window") != line["window"] \
                or decision.get("trigger") != line["trigger"] or decision.get("runtime") != RUNTIME \
                or decision.get("model") != wanted:
            raise PresenceError(f"decision {number} is not the {DECISION_SCHEMA} line of feature line {number} with the "
                                "model of the report")
        if line["status"] != "ok":
            if any(decision.get(name) is not None for name in ("score", "decision", "input")):
                raise PresenceError(f"decision {number} has the status {line['status']} and carries a score")
            continue
        try:
            score, verdict, given = decision["score"], decision["decision"], decision["input"]
            if line["features"]["names"] != manifest["input"]["vector"]:
                raise PresenceError(f"decision {number}: the vector of its line is not that of the head")
            x = np.array([[line["features"]["values"][index] for index in places]], dtype="<f4")
            good = given["sha256"] == _sha256(x.tobytes()) and given["columns"] == len(places) \
                and given["spec_sha256"] == line["spec_sha256"] \
                and type(score["logit"]) is float and math.isfinite(score["logit"]) \
                and abs(score["probability"] - 1.0 / (1.0 + math.exp(-score["logit"]))) <= 1e-6 \
                and verdict["threshold_logit"] == threshold and verdict["present"] is (score["logit"] > threshold) \
                and decision["contributing"] == line["contributing"] and decision["time"] == line["time"]
        except (KeyError, TypeError, IndexError, OverflowError) as error:
            raise PresenceError(f"decision {number}: the scored decision is not whole: {error!r}") from None
        if not good:
            raise PresenceError(f"decision {number}: the digest of its input, its score or its decision do not agree "
                                "with its line and with the frozen threshold")
    counts = report.get("counts") or {}
    scored = [line for line in decisions if line["status"] == "ok"]
    if counts.get("decisions") != len(decisions) or counts.get("scored") != len(scored) \
            or counts.get("present") != sum(line["decision"]["present"] for line in scored):
        raise PresenceError("the counts of the report are not those of its decisions")
    extra = sorted(path.name for path in directory.iterdir()
                   if path.name not in (DECISIONS_FILE, REPORT_FILE, FEATURES_DIR))
    if extra:
        raise PresenceError(f"the output holds {extra[0]}, which its report does not list")
    return Output(directory, report, decisions, feature_output)


def _checked(spec, head):
    wanted = head.artifact.manifest["input"]
    if features.spec_sha256(spec) != wanted["spec_sha256"]:
        raise PresenceError(f"the specification {spec['spec_id']} has the hash {features.spec_sha256(spec)}, and the "
                            f"head was trained on {wanted['spec_sha256']} ({wanted['spec_id']})")


def _empty(output_dir):
    output_dir = Path(output_dir)
    if not output_dir.is_dir() or any(output_dir.iterdir()):
        raise PresenceError(f"the output directory {output_dir} does not exist or is not empty")
    return output_dir


def replay(archive_dir, spec, artifact_dir, output_dir, device="cpu"):
    """Push the records of a consumer archive through a stream and a head, and write the output."""
    output_dir = _empty(output_dir)
    head = Head(artifact_dir, device)
    _checked(spec, head)
    archive = obs.open_archive(archive_dir)
    lines, views, stream, decisions = run_records(archive.records, spec, head)
    (output_dir / FEATURES_DIR).mkdir()
    features.write_output(output_dir / FEATURES_DIR, spec, lines, views, stream,
                          {"kind": "archive", "directory": str(archive_dir), "reader": archive.report["reader"],
                           "records": len(archive.records)})
    return write_output(output_dir, head, decisions, {"kind": "archive", "directory": str(archive_dir),
                                                      "records": len(archive.records)})


def score(features_dir, artifact_dir, output_dir, device="cpu"):
    """Score the lines of a feature output that a sensor wrote before, and write the decisions."""
    output_dir = _empty(output_dir)
    head = Head(artifact_dir, device)
    try:
        made = features.open_output(features_dir)
    except FeaturesError as error:
        raise PresenceError(f"the feature output {features_dir} does not verify: {error}") from None
    _checked(made.spec, head)
    decisions = decide_lines(head, made.lines)
    return write_output(output_dir, head, decisions, {"kind": "features", "directory": str(Path(features_dir).resolve()),
                                                      "records": made.report["counts"]["records"]},
                        Path(features_dir).resolve())


def follow(socket_path, spec, artifact_dir, output_dir, device=None, wait="wait", connect_timeout=30.0,
           idle_timeout=60.0, ready_file=None):
    """Attach to a ring as its consumer, compute features and score every complete window as its record is read.

    This is the reference sensor with a head: a second process on the standard-library reader.
    The artifact is loaded once, before the attach. The feature steps and the head run once on
    noise and on the training mean before the attach as well, so that no record waits for the
    first transform, the first inference or the context of the device. With ready_file the sensor
    writes that file when it is warm, and a launcher starts the receiver after it. The records it
    read are kept under features/records/.
    """
    output_dir = _empty(output_dir)
    head = Head(artifact_dir, default_device() if device is None else device)
    _checked(spec, head)
    warm_features = features.warm_up(spec)
    warm_head = head.warm_up()
    stream, lines, decisions = features.Stream(spec, time.monotonic_ns), [], []

    def on_record(record):
        for line in stream.push(record):
            lines.append(line)
            decisions.append(head.decide(line, time.monotonic_ns))

    feature_dir = output_dir / FEATURES_DIR
    (feature_dir / features.RECORDS_DIR).mkdir(parents=True)
    if ready_file is not None:
        Path(ready_file).write_text(json.dumps({"device": str(head.device), "artifact_sha256": head.artifact.sha256,
                                                "warm_up_ms": {"features": warm_features, "head": warm_head},
                                                "t_ready_ns": time.monotonic_ns()}) + "\n", encoding="ascii")
    consumer = obs.tail(socket_path, feature_dir / features.RECORDS_DIR, wait, connect_timeout, idle_timeout,
                        on_record=on_record)
    source = {"kind": "ring", "socket": str(socket_path), "reader": consumer["reader"], "wait": wait,
              "epoch": consumer["epoch"], "records": consumer["counts"]["read"], "stale": consumer["counts"]["stale"],
              "ended": consumer["ended"], "warm_up_ms": warm_features}
    features.write_output(feature_dir, spec, lines, stream.views, stream, source)
    return write_output(output_dir, head, decisions, dict(source, warm_up_ms={"features": warm_features,
                                                                              "head": warm_head}))


def summary(output, window=0):
    """What a second reader recovers from a presence output: the model, the counts and one scored window."""
    manifest = output.manifest
    made = {"head_id": manifest["head_id"], "artifact_sha256": manifest["sha256"],
            "weights_sha256": manifest["weights"]["sha256"], "normalizer_sha256": manifest["normalizer"]["sha256"],
            "columns": manifest["input"]["columns"], "threshold_logit": manifest["threshold"]["logit"],
            "runtime": output.report["runtime"]["entry"], "device": output.report["runtime"]["device"],
            "loads": output.report["loads"], "source": output.report["source"]["kind"],
            "counts": output.report["counts"], "scored_windows": len(output.scored)}
    if not 0 <= window < len(output.scored):
        raise PresenceError(f"the output holds {len(output.scored)} scored windows, and window {window} was asked for")
    line = output.scored[window]
    made["window"] = {"id": line["window"]["id"], "contributing": len(line["contributing"]),
                      "logit": line["score"]["logit"], "probability": line["score"]["probability"],
                      "present": line["decision"]["present"], "input_sha256": line["input"]["sha256"],
                      "infer_ns": line["infer_ns"]}
    return made


def inspect(artifact):
    """What a second reader recovers from an artifact: its hashes, its columns and its threshold."""
    manifest = artifact.manifest
    values = struct.unpack(f"<{manifest['weights']['count']}f", artifact.weights)
    return {"head_id": manifest["head_id"], "artifact_sha256": manifest["sha256"], "entries": manifest["entries"],
            "spec_id": manifest["input"]["spec_id"], "spec_sha256": manifest["input"]["spec_sha256"],
            "columns": manifest["input"]["columns"], "vector": len(manifest["input"]["vector"]),
            "weights_sha256": manifest["weights"]["sha256"], "weights_bytes": manifest["weights"]["bytes"],
            "weights": dict(zip(manifest["input"]["columns"], values[:-1])), "bias": values[-1],
            "normalizer_sha256": manifest["normalizer"]["sha256"], "threshold": manifest["threshold"],
            "files": sorted(path.name for path in artifact.directory.iterdir())}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reference presence head on the causal features: one frozen linear "
                                                 "head, loaded once, scoring one window at a time.")
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("settings", help="write the training configuration of the demonstration head")
    make.add_argument("--output", required=True, type=Path)
    make.add_argument("--head-id", default="dd-linear-v1")
    where = commands.add_parser("device", help="print the device a head would run on in this interpreter")
    where.add_argument("--device", help="cpu, cuda or cuda:N (default: cuda when PyTorch reaches one)")
    live = commands.add_parser("follow", help="attach to a ring, compute features and score every complete window")
    live.add_argument("--socket", required=True, type=Path, help="control channel of the producer's ring")
    live.add_argument("--wait", choices=sorted(obs.WAIT_MODES), default="wait")
    live.add_argument("--connect-ms", type=int, default=30000)
    live.add_argument("--idle-ms", type=int, default=60000)
    live.add_argument("--ready-file", type=Path, help="written when the head is loaded and warm, before the attach")
    again = commands.add_parser("replay", help="compute features and decisions from the records of a consumer archive")
    again.add_argument("--archive", required=True, type=Path)
    for command in (live, again):
        command.add_argument("--spec", required=True, type=Path)
    offline = commands.add_parser("score", help="score the lines of a feature output that a sensor wrote before")
    offline.add_argument("--features", required=True, type=Path, help="directory of a verified feature output")
    for command in (live, again, offline):
        command.add_argument("--artifact", required=True, type=Path, help="directory of a presence-head/v1 artifact")
        command.add_argument("--device", help="cpu, cuda or cuda:N (default: cuda when PyTorch reaches one, for "
                                              "follow; cpu otherwise)")
        command.add_argument("--output-dir", required=True, type=Path, help="existing, empty directory")
    show = commands.add_parser("show", help="verify a presence output and print one scored window")
    show.add_argument("directory", type=Path)
    show.add_argument("--window", type=int, default=0)
    look = commands.add_parser("inspect", help="verify an artifact and print its hashes, columns and threshold")
    look.add_argument("artifact", type=Path)
    look.add_argument("--sha256", help="the hash a freeze record names for the artifact")
    args = parser.parse_args(argv)
    try:
        if args.command == "settings":
            if args.output.exists():
                raise PresenceError(f"{args.output} already exists. A configuration is never overwritten.")
            settings = demo_settings(args.head_id)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(settings, indent=2, sort_keys=True) + "\n", encoding="ascii")
            print(json.dumps({"settings": str(args.output), "sha256": _sha256(canonical(settings))}, indent=2))
            return 0
        if args.command == "device":
            print(json.dumps(device_facts(device_of(args.device or default_device())), indent=2))
            return 0
        if args.command == "show":
            print(json.dumps(summary(open_output(args.directory), args.window), indent=2))
            return 0
        if args.command == "inspect":
            print(json.dumps(inspect(open_artifact(args.artifact, args.sha256)), indent=2))
            return 0
        if args.command == "score":
            report = score(args.features, args.artifact, args.output_dir, args.device or "cpu")
        else:
            spec = features.load_spec(args.spec)
            if args.command == "follow":
                report = follow(args.socket, spec, args.artifact, args.output_dir, args.device, args.wait,
                                args.connect_ms / 1000, args.idle_ms / 1000, args.ready_file)
            else:
                report = replay(args.archive, spec, args.artifact, args.output_dir, args.device or "cpu")
        counts = report["counts"]
        print(f"presence: {counts['decisions']} decisions on {report['runtime']['device']}, {counts['scored']} scored: "
              f"{counts['present']} present, {counts['absent']} absent; {counts['insufficient_history']} of a filling "
              f"window, {counts['gap']} gaps, {counts['invalid']} invalid, {counts['rejected']} rejected; the head "
              f"{report['artifact']['model']['head_id']} {report['artifact']['model']['artifact_sha256'][:12]} was "
              f"loaded {report['loads']} time(s)")
        return 0
    except (PresenceError, FeaturesError, ObsError) as error:
        parser.exit(2, f"presence: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
