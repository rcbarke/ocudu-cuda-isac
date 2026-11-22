"""The native sensing core from Python (Sprint 3.6, selector sense_core).

native/sense_core builds a pybind11 extension, _sense_core, beside its library. This module
finds that extension in a build of the core and gives it to tests, parity checks and offline
scoring. No timed run imports it: the sensor process is sense_host, which holds no interpreter.

    core = runtime.load()                     # build/sense-core, then build/sense-core-host
    plan = core.Plan(artifact_dir, device="host", engine="host_reference")
    for record in records:                    # the bytes of one obs/v1 record each
        for answer in plan.offer(record):
            ...
        if plan.ready():
            decision = plan.run()             # one submission: ingest, kernels, engine

  load        the extension of a build directory, by the path of its file
  record_bytes  a record as the core takes it: header, host block, validity bits, payload
  score       records through a plan, as the sensor loop of an archive does: every answer and
              every decision in order, with the input and the map of each scored window
  build, registry, tolerance   what a build says of itself, parsed

A build is named by its directory, by the environment variable ISAC_SENSE_CORE_BUILD, or found
in build/ of the checkout. Nothing here imports a module outside the standard library and NumPy.
"""

import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BUILDS = ("build/sense-core", "build/sense-core-host")
ENVIRONMENT = "ISAC_SENSE_CORE_BUILD"
MODULE = "_sense_core"
RUNTIME = "native"


class RuntimeUnavailable(ImportError):
    """No build of the native core with its binding was found for this interpreter."""


def find(build_dir=None):
    """The file of the extension: in the named build, in the one the environment names, or in build/."""
    places = [Path(build_dir)] if build_dir is not None else \
        ([Path(os.environ[ENVIRONMENT])] if os.environ.get(ENVIRONMENT) else []) + [ROOT / name for name in BUILDS]
    suffixes = importlib.machinery.EXTENSION_SUFFIXES
    for place in places:
        for suffix in suffixes:
            path = place / "python" / (MODULE + suffix)
            if path.is_file():
                return path
    raise RuntimeUnavailable("no build of the native sensing core holds its binding for this interpreter in "
                             + ", ".join(str(place) for place in places)
                             + ". Build it: python scripts/native_core.py build (or build --host-only)")


_loaded = {}


def load(build_dir=None):
    """The extension module of a build. One interpreter holds one build of it."""
    path = find(build_dir).resolve()
    if path in _loaded:
        return _loaded[path]
    if _loaded:
        raise RuntimeUnavailable(f"this interpreter already holds the binding of {next(iter(_loaded))}: one process "
                                 f"loads one build of the core, and {path} is another")
    spec = importlib.util.spec_from_file_location(MODULE, path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as error:
        raise RuntimeUnavailable(f"{path} does not load in this interpreter: {error}") from None
    module.__build_dir__ = str(path.parents[1])
    _loaded[path] = module
    return module


def build(core):
    return json.loads(core.build())


def registry(core):
    return json.loads(core.registry())


def tolerance(core):
    return json.loads(core.tolerance())


def record_bytes(record):
    """One record as the core takes it. Accepts bytes, or a record of ocudu_cuda_isac.sensing.obs."""
    if isinstance(record, (bytes, bytearray, memoryview)):
        return bytes(record)
    return record.to_bytes()


def archive_bytes(directory):
    """The records of a consumer archive, in the order of their names: the order of epoch and sequence."""
    return [path.read_bytes() for path in sorted(Path(directory).glob("*.obs"))]


def score(plan, records, clock=None):
    """Records through a plan, in order, as an archive is replayed: every window that completes runs.

    Returns (lines, inputs, maps). A line is ("answer", answer) for an answer without a score, or
    ("decision", decision, smoothed) for the decision of a complete window. inputs and maps hold
    the input [A, K] and the map [rows, cols] of every scored decision, in order.
    """
    lines, inputs, maps = [], [], []
    for record in records:
        notified = 0 if clock is None else clock()
        for answer in plan.offer(record_bytes(record), notified):
            if answer["kind"] != "ready":
                lines.append(("answer", answer))
        if plan.ready():
            decision = plan.run()
            smoothed = plan.smooth()
            lines.append(("decision", decision, smoothed))
            if decision["status"] == "ok":
                inputs.append(np.array(decision["input"], dtype="<f4"))
                maps.append(np.array(decision["map"], dtype="<f4"))
    plan.finish()
    return lines, inputs, maps
