"""Channel application for receiver replay: the plan and the reference (Sprint 2.3, selector grid_apply).

The native replay (native/ocudu_adapter/replay) computes the receiver grid of every opportunity as

    Y[rx, symbol, subcarrier] = g * (sum over tx of H * X + W)

over the whole carrier grid, once, in double precision. X is the transmit grid of the opportunity's
grant (a pusch-grant-store/v1 entry), H and W are one opportunity of a channel-episode/v1 entry, and
g is calibration.grid_gain of that entry: real, positive, one value for a replay run, applied to
the signal and to the noise before the value is converted to the grid's storage format. A stored
grid is never rescaled.

This module is the offline side of that step. It

  * defines the arithmetic as a scalar reference, with the order of every operation fixed, so that
    the native code and a NumPy reference reproduce the same bits;
  * holds the gain rules and builds the grid-apply-plan/v1 file the native tool reads: from episode
    entries, with every transmit grid bound by the hash the entry names, or from a fixture that
    has no entry, with a recorded gain override;
  * reads the native output back: the o1-grid/v1 descriptors and the stored grids of ingress/.

Standard library only. The plan is an offline stimulus record. It names no transport block and no
label, and the native tool reads neither.
"""

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

from . import episode as episodes
from . import grant as grants
from . import schedule as schedules
from .episode import EpisodeError
from .grant import GrantError, bf16
from ..replay import ContractError, load_json

PLAN_SCHEMA = "grid-apply-plan/v1"
NATIVE_SCHEMA = "grid-apply-native/v1"
INGRESS_SCHEMA = "o1-grid/v1"
RULE = "Y = g (sum over tx of H X + W)"
# The rule g = 1 / sqrt(reference_power) holds within this relative tolerance.
GAIN_RULE_TOLERANCE = 1e-6
# Values handed to the grid, before storage: abs(error) <= absolute + relative * abs(reference).
# The tolerance applies after the gain, where the signal has unit scale.
UNIT_SCALE = {"absolute": 1e-6, "relative": 1e-5}
PLAN_FIELDS = ("schema", "run_id", "access", "carrier", "gain", "grants", "sources", "opportunities")
CARRIER_FIELDS = ("rx_ports", "tx_ports", "symbols", "subcarriers")
GRANT_FIELDS = ("grant_id", "record_sha256", "slot", "x", "active")
SOURCE_FIELDS = ("realization_id", "h", "w", "entry")
ENTRY_FIELDS = ("episode_id", "grid_gain", "reference_power")
OPPORTUNITY_FIELDS = ("opportunity_id", "realization_id", "index", "grant_id", "sfn", "slot")
# Everything the receiver side may read of one grid. A field outside this list fails the access check.
INGRESS_FIELDS = ("schema", "run_id", "sequence", "opportunity_id", "sfn", "slot", "grant_id", "grant_record_sha256",
                  "active_re", "active_sha256", "grid_index", "generation", "state", "grid", "amplitude")
INGRESS_GRID_FIELDS = ("file", "axes", "shape", "dtype", "order", "storage", "sha256")
INGRESS_AMPLITUDE_FIELDS = ("grid_gain", "convention")
GRID_AXES = ["rx_port", "symbol", "subcarrier"]


class GridApplyError(ValueError):
    """A plan, a gain, a binding or a native output violates the documented interface."""


# ---------------------------------------------------------------- arithmetic
def to_single(value):
    """One real value converted to single precision: round to nearest, ties to even."""
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error):
        raise GridApplyError("channel application produced a value that does not fit single precision") from None


def single(values):
    """Complex values as a complex64 array holds them: each part converted to single precision."""
    return [complex(to_single(value.real), to_single(value.imag)) for value in values]


def apply(shape, x, h, w, gain):
    """The values handed to the receiver grid for one opportunity, in the order [rx, symbol, subcarrier].

    shape is (rx_ports, tx_ports, symbols, subcarriers). x holds [tx, symbol, subcarrier], h holds
    [rx, tx, symbol, subcarrier] and w holds [rx, symbol, subcarrier] as flat complex sequences,
    last axis fastest, with the values that the complex64 arrays hold (see single). Each product
    is formed as (hr xr - hi xi, hr xi + hi xr), the products are summed in transmit-port order,
    the noise is added, each part is multiplied by the gain and converted to single precision. The
    native kernel performs the same operations in that order.
    """
    gain = check_gain(gain)
    rx_ports, tx_ports, symbols, subcarriers = shape
    plane = symbols * subcarriers
    if min(shape) < 1 or len(x) != tx_ports * plane or len(h) != rx_ports * tx_ports * plane \
            or len(w) != rx_ports * plane:
        raise GridApplyError("x, h and w do not have the sizes of the carrier grid")
    result = []
    for rx in range(rx_ports):
        for element in range(plane):
            real = imag = 0.0
            for tx in range(tx_ports):
                channel, symbol = h[(rx * tx_ports + tx) * plane + element], x[tx * plane + element]
                product_real = channel.real * symbol.real - channel.imag * symbol.imag
                product_imag = channel.real * symbol.imag + channel.imag * symbol.real
                real, imag = (product_real, product_imag) if tx == 0 else (real + product_real, imag + product_imag)
            noise = w[rx * plane + element]
            real, imag = (real + noise.real) * gain, (imag + noise.imag) * gain
            if not (math.isfinite(real) and math.isfinite(imag)):
                raise GridApplyError("channel application produced a nonfinite value")
            result.append(complex(to_single(real), to_single(imag)))
    return result


def stored(values):
    """What the receiver grid keeps of the values handed to it: complex bf16, widened without change."""
    return [complex(bf16(value.real), bf16(value.imag)) for value in values]


def within_unit_scale(value, reference):
    """True when a value handed to the grid equals its reference within the unit-scale tolerance."""
    return abs(value - reference) <= UNIT_SCALE["absolute"] + UNIT_SCALE["relative"] * abs(reference)


def half_precision(value):
    """One real value as the pinned CUDA receiver keeps its estimates: an IEEE 16-bit float."""
    return struct.unpack("<e", struct.pack("<e", value))[0]


def grid_bytes(values):
    """Little-endian complex64 bytes of a grid. Their SHA-256 is the readback digest of a stored grid."""
    return struct.pack(f"<{2 * len(values)}f", *[part for value in values for part in (value.real, value.imag)])


# ---------------------------------------------------------------- gain
def check_gain(value):
    """A grid gain is a finite, positive real number. Anything else is refused."""
    if value is None:
        raise GridApplyError("the run names no grid gain. An episode entry records it as calibration.grid_gain.")
    if type(value) not in (int, float):
        raise GridApplyError("the grid gain must be a real number")
    if not math.isfinite(value):
        raise GridApplyError("the grid gain is not finite")
    if not value > 0:
        raise GridApplyError(f"the grid gain is {value!r}. It must be positive.")
    return float(value)


def follows_rule(gain, reference_power):
    """True when gain equals 1 / sqrt(reference_power) within the declared tolerance."""
    if type(reference_power) not in (int, float) or not math.isfinite(reference_power) or not reference_power > 0:
        return False
    rule = 1.0 / math.sqrt(reference_power)
    return abs(gain - rule) <= GAIN_RULE_TOLERANCE * rule


def entry_calibration(manifest):
    """(grid_gain, reference_power) of an episode entry, with the gain checked against its rule."""
    calibration = manifest.get("calibration") if isinstance(manifest, dict) else None
    if not isinstance(calibration, dict) or "grid_gain" not in calibration:
        raise GridApplyError("the entry records no calibration.grid_gain")
    gain = check_gain(calibration["grid_gain"])
    power = calibration.get("reference_power")
    if not follows_rule(gain, power):
        raise GridApplyError(f"the grid gain {gain!r} differs from 1 / sqrt(reference_power) of the entry "
                             f"({power!r}) by more than {GAIN_RULE_TOLERANCE:g}")
    return gain, float(power)


# ---------------------------------------------------------------- files
def sha256_file(path):
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 22), b""):
                digest.update(block)
    except OSError as error:
        raise GridApplyError(f"cannot read {path}: {error}") from None
    return digest.hexdigest()


def file_ref(path):
    """A file with the hash that binds it, as a plan names it."""
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def array_header(path):
    """(descr, shape, data offset) of a version 1.0 .npy file, read from its header alone."""
    try:
        with Path(path).open("rb") as stream:
            start = stream.read(10)
            if start[:8] != b"\x93NUMPY\x01\x00" or len(start) != 10:
                raise GridApplyError(f"{path} is not a version 1.0 .npy file")
            size = struct.unpack("<H", start[8:10])[0]
            text = stream.read(size).decode("latin1")
            total = Path(path).stat().st_size
    except OSError as error:
        raise GridApplyError(f"cannot read {path}: {error}") from None
    try:
        header = ast.literal_eval(text)
        descr, shape, fortran = header["descr"], tuple(header["shape"]), header["fortran_order"]
    except (ValueError, SyntaxError, KeyError, TypeError):
        raise GridApplyError(f"{path} has an unreadable array header") from None
    item = {"<c8": 8, "<i4": 4}.get(descr)
    if fortran or item is None or total != 10 + size + item * math.prod(shape):
        raise GridApplyError(f"{path} is not a C-ordered little-endian complex64 or int32 array of its declared shape")
    return descr, list(shape), 10 + size


def write_array(path, descr, shape, flat):
    """Write a fixture array as a .npy file. The file must not exist."""
    path = Path(path)
    if path.exists():
        raise GridApplyError(f"{path} already exists. A fixture array is never overwritten.")
    if math.prod(shape) != len(flat):
        raise GridApplyError(f"{len(flat)} values do not fill an array of shape {list(shape)}")
    try:
        path.write_bytes(grants._npy_bytes(descr, shape, flat))
    except GrantError as error:
        raise GridApplyError(str(error)) from None
    return file_ref(path)


def write_complex(path, shape, values):
    return write_array(path, "<c8", shape, values)


def write_index(path, pairs):
    """Occupied resource elements as int32 rows of (symbol, subcarrier)."""
    return write_array(path, "<i4", (len(pairs), 2), [part for pair in pairs for part in pair])


# ---------------------------------------------------------------- plan
def _keys(value, fields, name, optional=()):
    if not isinstance(value, dict) or not set(fields) - set(optional) <= set(value) or set(value) - set(fields):
        raise GridApplyError(f"{name} requires exactly {', '.join(fields)}")
    return value


def _name(value, name):
    if not isinstance(value, str) or not value or not value.isascii() or value.strip() != value:
        raise GridApplyError(f"{name} must be a nonempty ASCII string")
    return value


def _count(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise GridApplyError(f"{name} must be an integer from {low} to {high}")
    return value


def _ref(value, name, bound=True):
    """A file reference. bound: the reference must carry the hash that binds the file."""
    if not isinstance(value, dict) or set(value) - {"path", "sha256"} or "path" not in value:
        raise GridApplyError(f"{name} requires path and sha256")
    _name(value["path"], f"{name}.path")
    digest = value.get("sha256")
    if digest is None:
        if bound:
            raise GridApplyError(f"{name} is bound by no hash")
        return value
    if not isinstance(digest, str) or len(digest) != 64 or set(digest) - set("0123456789abcdef"):
        raise GridApplyError(f"{name}.sha256 must be 64 hexadecimal digits")
    return value


def check_plan(plan):
    """Apply every rule of a plan that needs no array: the fields, the gain and the bindings.

    The native tool applies the same rules again, and then verifies every file against its hash.
    """
    _keys(plan, PLAN_FIELDS, "a plan", optional=("gain",))
    if plan["schema"] != PLAN_SCHEMA:
        raise GridApplyError(f"unsupported plan schema {plan['schema']!r}")
    _name(plan["run_id"], "run_id")
    if plan["access"] != "offline_stimulus":
        raise GridApplyError('access must be "offline_stimulus"')
    carrier = _keys(plan["carrier"], CARRIER_FIELDS, "carrier")
    for key in CARRIER_FIELDS:
        _count(carrier[key], f"carrier.{key}", 1, 1 << 16)
    sources, grant_list, opportunities = plan["sources"], plan["grants"], plan["opportunities"]
    if not all(isinstance(item, list) and item for item in (sources, grant_list, opportunities)):
        raise GridApplyError("a plan needs at least one grant, one channel realization and one opportunity")

    # ---- Realizations first: the gain rules need to know whether the run reads an entry. ----
    realizations = {}
    for source in sources:
        _keys(source, SOURCE_FIELDS, "a source")
        name = _name(source["realization_id"], "realization_id")
        if name in realizations:
            raise GridApplyError(f"the realization {name} is listed twice")
        _ref(source["h"], f"the channel of realization {name}")
        _ref(source["w"], f"the noise of realization {name}")
        if source["entry"] is not None:
            _keys(source["entry"], ENTRY_FIELDS, f"the entry of realization {name}")
        realizations[name] = source

    # ---- The gain: one value for the run. ----
    gain = plan.get("gain")
    if gain is None:
        check_gain(None)
    if not isinstance(gain, dict) or not {"value", "source"} <= set(gain) \
            or set(gain) - {"value", "source", "reference_power", "reason", "rule"}:
        raise GridApplyError("gain requires value and source, and optionally reference_power, reason and rule")
    value = check_gain(gain["value"])
    entries = [source["entry"] for source in sources if source["entry"] is not None]
    if gain["source"] == "override":
        if entries:
            raise GridApplyError("a gain override is allowed only for a fixture that has no entry, and this run "
                                 "reads an episode entry")
        if not isinstance(gain.get("reason"), str) or not gain["reason"].strip():
            raise GridApplyError("a gain override records its reason, and this one names none")
    elif gain["source"] == "entry":
        if not follows_rule(value, gain.get("reference_power")):
            raise GridApplyError(f"the grid gain {value!r} differs from 1 / sqrt(reference_power) by more than "
                                 f"{GAIN_RULE_TOLERANCE:g}, or the plan names no reference power")
        if len(entries) != len(sources):
            raise GridApplyError("a run reads episode entries or a fixture, not both")
        for source in sources:
            recorded = check_gain(source["entry"]["grid_gain"])
            if recorded != value:
                raise GridApplyError(f"two gains in one run: {value!r} and {recorded!r} "
                                     f"(realization {source['realization_id']})")
            if not follows_rule(recorded, source["entry"]["reference_power"]):
                raise GridApplyError(f"the grid gain of realization {source['realization_id']} differs from "
                                     "1 / sqrt(reference_power) of its entry")
    else:
        raise GridApplyError('gain.source must be "entry" or "override"')

    # ---- Grants: each transmit grid is bound by a hash. ----
    bound = {}
    for grant in grant_list:
        _keys(grant, GRANT_FIELDS, "a grant", optional=("slot",))
        name = _name(grant["grant_id"], "grant_id")
        if name in bound:
            raise GridApplyError(f"the grant {name} is listed twice")
        _name(grant["record_sha256"], f"record_sha256 of grant {name}")
        if not isinstance(grant["x"], dict) or grant["x"].get("sha256") is None:
            raise GridApplyError(f"the grant {name} has no bound transmit grid: no hash binds it")
        _ref(grant["x"], f"the transmit grid of grant {name}")
        if not isinstance(grant["active"], list) or not grant["active"]:
            raise GridApplyError(f"the grant {name} lists no occupied resource elements")
        for index in grant["active"]:
            _ref(index, f"the occupied elements of grant {name}")
        bound[name] = grant

    seen = set()
    for opportunity in opportunities:
        _keys(opportunity, OPPORTUNITY_FIELDS, "an opportunity")
        name = _name(opportunity["opportunity_id"], "opportunity_id")
        if name in seen:
            raise GridApplyError(f"the opportunity {name} is listed twice")
        seen.add(name)
        if opportunity["realization_id"] not in realizations:
            raise GridApplyError(f"the opportunity {name} names a realization that the run does not hold")
        _count(opportunity["index"], f"index of opportunity {name}", 0, 1 << 30)
        _count(opportunity["sfn"], f"sfn of opportunity {name}", 0, 1023)
        _count(opportunity["slot"], f"slot of opportunity {name}", 0, 159)
        if opportunity["grant_id"] is None:
            raise GridApplyError(f"the opportunity {name} has no bound grid")
        if opportunity["grant_id"] not in bound:
            raise GridApplyError(f"the opportunity {name} names the grant {opportunity['grant_id']!r}, which the "
                                 "run does not bind")
    return plan


def check_files(plan):
    """Verify every file of a plan: its hash, its element type and its shape on the carrier grid."""
    check_plan(plan)
    carrier = plan["carrier"]
    grid = [carrier["symbols"], carrier["subcarriers"]]

    def verify(ref, name, descr, shape=None):
        if sha256_file(ref["path"]) != ref["sha256"]:
            raise GridApplyError(f"{name} does not have the hash its record binds ({ref['path']})")
        found, found_shape, _ = array_header(ref["path"])
        if found != descr or (shape is not None and found_shape != shape):
            raise GridApplyError(f"{name} is {found} {found_shape}, expected {descr} {shape}")
        return found_shape

    for grant in plan["grants"]:
        try:
            verify(grant["x"], f"the transmit grid of grant {grant['grant_id']}", "<c8", [carrier["tx_ports"]] + grid)
        except GridApplyError as error:
            if "expected" not in str(error):
                raise
            # The grid lies on another carrier grid than the run: it cannot be remapped.
            raise GridApplyError(f"changed allocation: {error}") from None
        for index in grant["active"]:
            shape = verify(index, f"the occupied elements of grant {grant['grant_id']}", "<i4")
            if len(shape) != 2 or shape[1] != 2:
                raise GridApplyError(f"{index['path']} is not an array of (symbol, subcarrier) rows")
    counts = {}
    for source in plan["sources"]:
        name = source["realization_id"]
        h_shape = verify(source["h"], f"the channel of realization {name}", "<c8")
        if len(h_shape) != 5 or h_shape[1:] != [carrier["rx_ports"], carrier["tx_ports"]] + grid:
            raise GridApplyError(f"the channel of realization {name} has the shape {h_shape}, not that of the carrier")
        verify(source["w"], f"the noise of realization {name}", "<c8", [h_shape[0], carrier["rx_ports"]] + grid)
        counts[name] = h_shape[0]
    for opportunity in plan["opportunities"]:
        if opportunity["index"] >= counts[opportunity["realization_id"]]:
            raise GridApplyError(f"the opportunity {opportunity['opportunity_id']} reads index {opportunity['index']} "
                                 f"of a realization with {counts[opportunity['realization_id']]} opportunities")
    return plan


def grant_from_store(store):
    """The plan entry of one grant from its verified store entry: its grid and its occupied elements."""
    if store.x is None:
        raise GridApplyError(f"the store entry of grant {store.grant.grant_id} holds no transmit grid")
    files, directory = store.manifest["files"], Path(store.directory).resolve()
    return {"grant_id": store.grant.grant_id, "record_sha256": schedules.record_sha256(store.grant.record()),
            "slot": store.grant.slot,
            "x": {"path": str(directory / "x.npy"), "sha256": files["x.npy"]["sha256"]},
            "active": [{"path": str(directory / name), "sha256": files[name]["sha256"]}
                       for name in ("pilot_index.npy", "data_index.npy")]}


def fixture_plan(run_id, carrier, gain, reason, grant_list, sources, opportunities):
    """A plan for a fixture that has no episode entry, such as the unit vectors and the unity check.

    The gain is an override, recorded with its reason. sources is a list of {"realization_id", "h",
    "w"} with file references, opportunities a list of the plan's opportunity records.
    """
    plan = {"schema": PLAN_SCHEMA, "run_id": run_id, "access": "offline_stimulus",
            "carrier": dict(zip(CARRIER_FIELDS, carrier)),
            "gain": {"value": gain, "source": "override", "reason": reason, "rule": RULE},
            "grants": list(grant_list),
            "sources": [dict(source, entry=None) for source in sources],
            "opportunities": list(opportunities)}
    return check_files(plan)


def _bound_store(grant_id, binding, roots):
    """The verified store entry whose transmit grid has the hash an episode entry binds."""
    if not isinstance(binding, dict) or not isinstance(binding.get("x_sha256"), str):
        raise GridApplyError(f"the grant {grant_id} has no bound grid: the entry names no transmit grid for it")
    candidates = [Path(binding["store"])] if isinstance(binding.get("store"), str) else []
    candidates += [Path(root) / grant_id / "store" for root in roots]
    other = None
    for candidate in candidates:
        if not (candidate / "manifest.json").is_file():
            continue
        try:
            store = grants.open_store(candidate)
        except GrantError as error:
            raise GridApplyError(f"the store entry of grant {grant_id} at {candidate} is unusable: {error}") from None
        digest = store.manifest["files"].get("x.npy", {}).get("sha256")
        if digest == binding["x_sha256"]:
            return store
        other = (candidate, digest)
    if other:
        raise GridApplyError(f"the transmit grid of grant {grant_id} at {other[0]} has the hash {other[1]}, and the "
                             f"entry binds {binding['x_sha256']}")
    raise GridApplyError(f"no store entry of grant {grant_id} was found. Name the directory that holds "
                         f"{grant_id}/store with --grant-store.")


def entry_plan(run_id, entries, stores=(), cases=("a", "b")):
    """A plan that replays episode entries: one or more channel-episode/v1 directories.

    Every entry is verified by open_episode. Its gain is calibration.grid_gain, and all entries of
    one run must record the same value. The transmit grid of every grant comes from a verified
    store entry whose hash the entry binds: the directory the entry recorded, or
    <root>/<grant_id>/store under one of stores. Opportunities follow in the order of the entries,
    the cases and the schedule, under opaque identifiers.
    """
    if not entries:
        raise GridApplyError("a plan needs at least one episode entry")
    if not cases or set(cases) - set(episodes.CASES):
        raise GridApplyError('cases are "a" and "b"')
    gain = power = carrier = None
    bound, sources, opportunities = {}, [], []
    for directory in entries:
        try:
            episode = episodes.open_episode(directory)
        except EpisodeError as error:
            raise GridApplyError(f"{directory} is not a usable episode entry: {error}") from None
        manifest = episode.manifest
        entry_gain, entry_power = entry_calibration(manifest)
        if gain is None:
            gain, power = entry_gain, entry_power
        elif entry_gain != gain:
            raise GridApplyError(f"two gains in one run: {gain!r} and {entry_gain!r} (episode {episode.episode_id})")
        shape = [episode.h_shape[1], episode.h_shape[2], episode.h_shape[3], episode.h_shape[4]]
        if carrier is None:
            carrier = shape
        elif shape != carrier:
            raise GridApplyError(f"episode {episode.episode_id} has the carrier grid {shape}, the run has {carrier}")
        for grant_id, grant in sorted(episode.schedule.grants.items()):
            described = manifest["grants"][grant_id]
            store = _bound_store(grant_id, described.get("x"), stores)
            item = grant_from_store(store)
            if item["record_sha256"] != described["record_sha256"] or store.grant.record() != grant.record():
                raise GridApplyError(f"the store entry of grant {grant_id} holds another grant record than the entry")
            if bound.setdefault(grant_id, item) != item:
                raise GridApplyError(f"two entries bind different transmit grids to the grant {grant_id}")
        recorded = {case["id"]: case for case in manifest["cases"]}
        for case in cases:
            realization = recorded[case]["realization_id"]
            sources.append({
                "realization_id": realization,
                "h": {"path": str(Path(episode.directory).resolve() / episodes.FILES[case]),
                      "sha256": manifest["files"][episodes.FILES[case]]["sha256"]},
                "w": {"path": str(Path(episode.directory).resolve() / episodes.FILES["w"]),
                      "sha256": manifest["files"][episodes.FILES["w"]]["sha256"]},
                "entry": {"episode_id": episode.episode_id, "grid_gain": entry_gain, "reference_power": entry_power}})
            for item in episode.schedule.opportunities:
                opportunities.append({"opportunity_id": f"{realization}.{item.index:04d}", "realization_id": realization,
                                      "index": item.index, "grant_id": item.grant.grant_id, "sfn": item.sfn,
                                      "slot": item.slot})
    plan = {"schema": PLAN_SCHEMA, "run_id": run_id, "access": "offline_stimulus",
            "carrier": dict(zip(CARRIER_FIELDS, carrier)),
            "gain": {"value": gain, "source": "entry", "reference_power": power,
                     "rule": "calibration.grid_gain of the entry, 1 / sqrt(reference_power); " + RULE},
            "grants": [bound[grant_id] for grant_id in sorted(bound)], "sources": sources,
            "opportunities": opportunities}
    return check_plan(plan)


def write_plan(plan, path):
    """Check a plan and write it. The file must not exist."""
    path = Path(path)
    if path.exists():
        raise GridApplyError(f"{path} already exists. A plan is never overwritten.")
    check_plan(plan)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return path


def load_plan(path):
    try:
        return check_plan(load_json(Path(path)))
    except (ContractError, OSError) as error:
        raise GridApplyError(f"cannot read the plan: {error}") from None


# ---------------------------------------------------------------- native output
class Output:
    """A verified native output: the report and the descriptors and stored grids of ingress/."""

    def __init__(self, directory, report, descriptors):
        self.directory, self.report, self.descriptors = directory, report, descriptors
        self.run_id = report["run_id"]

    def grid_path(self, sequence):
        return self.directory / "ingress" / self.descriptors[sequence]["grid"]["file"]

    def element(self, sequence, rx, symbol, subcarrier):
        """One stored value of one grid, read by its offset."""
        shape = self.descriptors[sequence]["grid"]["shape"]
        flat = 0
        for position, size in zip((rx, symbol, subcarrier), shape):
            if type(position) is not int or not 0 <= position < size:
                raise GridApplyError(f"index {[rx, symbol, subcarrier]} is outside a grid of shape {shape}")
            flat = flat * size + position
        with self.grid_path(sequence).open("rb") as stream:
            stream.seek(8 * flat)
            return complex(*struct.unpack("<2f", stream.read(8)))


def ingress_violations(directory, forbidden=()):
    """What the receiver side could read that it must not. An empty list means the access rule holds.

    ingress/ may hold ingress.jsonl and the grid files its descriptors name, and nothing else. A
    descriptor may hold the fields of INGRESS_FIELDS only. forbidden lists strings that must not
    appear anywhere in the descriptors: hashes and paths of H, W and transport blocks, and labels.
    """
    ingress = Path(directory) / "ingress"
    found = []
    try:
        text = (ingress / "ingress.jsonl").read_text(encoding="ascii")
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    except (OSError, ValueError) as error:
        return [f"ingress.jsonl is unreadable: {error}"]
    allowed = {"ingress.jsonl"}
    for record in records:
        if not isinstance(record, dict):
            found.append("a descriptor is not an object")
            continue
        found += [f"descriptor field {key!r}" for key in sorted(set(record) - set(INGRESS_FIELDS))]
        grid, amplitude = record.get("grid"), record.get("amplitude")
        if isinstance(grid, dict):
            found += [f"descriptor field grid.{key}" for key in sorted(set(grid) - set(INGRESS_GRID_FIELDS))]
            allowed.add(str(grid.get("file")))
        if isinstance(amplitude, dict):
            found += [f"descriptor field amplitude.{key}"
                      for key in sorted(set(amplitude) - set(INGRESS_AMPLITUDE_FIELDS))]
    try:
        found += [f"file ingress/{path.name}" for path in sorted(ingress.iterdir()) if path.name not in allowed
                  or not path.is_file()]
    except OSError as error:
        found.append(f"ingress/ is unreadable: {error}")
    found += [f"the descriptors hold {item!r}" for item in forbidden if item and item in text]
    return found


def open_output(directory):
    """Verify a native output and return it: the report, every descriptor and the hash of every stored grid."""
    directory = Path(directory)
    try:
        report = load_json(directory / "native.json")
        lines = (directory / "ingress" / "ingress.jsonl").read_text(encoding="ascii").splitlines()
        descriptors = [json.loads(line) for line in lines if line.strip()]
    except (ContractError, OSError, ValueError) as error:
        raise GridApplyError(f"{directory} is not a complete native output: {error}") from None
    if not isinstance(report, dict) or report.get("schema") != NATIVE_SCHEMA \
            or not isinstance(report.get("opportunities"), list):
        raise GridApplyError(f"{directory / 'native.json'} is not a {NATIVE_SCHEMA} report")
    if len(descriptors) != len(report["opportunities"]) or not descriptors:
        raise GridApplyError("the report and the descriptors list different numbers of opportunities")
    carrier = report["carrier"]
    shape = [carrier["rx_ports"], carrier["symbols"], carrier["subcarriers"]]
    for sequence, (descriptor, entry) in enumerate(zip(descriptors, report["opportunities"])):
        grid = descriptor.get("grid") if isinstance(descriptor, dict) else None
        if not isinstance(grid, dict) or set(descriptor) != set(INGRESS_FIELDS) \
                or set(grid) != set(INGRESS_GRID_FIELDS) or descriptor["schema"] != INGRESS_SCHEMA:
            raise GridApplyError(f"descriptor {sequence} is not an {INGRESS_SCHEMA} record")
        if descriptor["sequence"] != sequence or entry.get("sequence") != sequence \
                or descriptor["opportunity_id"] != entry.get("opportunity_id") \
                or descriptor["run_id"] != report["run_id"]:
            raise GridApplyError(f"descriptor {sequence} does not belong to opportunity {sequence} of the report")
        if grid["shape"] != shape or grid["axes"] != GRID_AXES or grid["dtype"] != "<c8" or grid["order"] != "C" \
                or descriptor["state"] != "ready":
            raise GridApplyError(f"descriptor {sequence} does not describe a ready grid on the carrier of the run")
        name = grid["file"]
        if not isinstance(name, str) or Path(name).name != name:
            raise GridApplyError(f"descriptor {sequence} names a file outside ingress/")
        path = directory / "ingress" / name
        try:
            size = path.stat().st_size
        except OSError as error:
            raise GridApplyError(f"cannot read the grid of opportunity {sequence}: {error}") from None
        if size != 8 * math.prod(shape) or sha256_file(path) != grid["sha256"] \
                or entry.get("readback_sha256") != grid["sha256"]:
            raise GridApplyError(f"the stored grid of opportunity {sequence} differs from its readback digest")
    return Output(directory, report, descriptors)


def summary(output, sequence=0, rx=0, symbol=0, subcarrier=0):
    """What a second reader recovers from a native output: one stored value and the digest of its grid."""
    if type(sequence) is not int or not 0 <= sequence < len(output.descriptors):
        raise GridApplyError(f"the output has opportunities 0 to {len(output.descriptors) - 1}")
    descriptor, gain = output.descriptors[sequence], output.report["gain"]
    value = output.element(sequence, rx, symbol, subcarrier)
    return {"run_id": output.run_id, "opportunities": len(output.descriptors), "verified_grids": len(output.descriptors),
            "gain": {key: gain[key] for key in ("value", "source", "reference_power", "reason") if key in gain},
            "sequence": sequence, "opportunity_id": descriptor["opportunity_id"], "grant_id": descriptor["grant_id"],
            "sfn": descriptor["sfn"], "slot": descriptor["slot"], "generation": descriptor["generation"],
            "state": descriptor["state"], "grid_file": descriptor["grid"]["file"],
            "readback_sha256": descriptor["grid"]["sha256"], "storage": descriptor["grid"]["storage"],
            "element": {"rx_port": rx, "symbol": symbol, "subcarrier": subcarrier,
                        "value": [value.real, value.imag], "magnitude": abs(value)}}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build a replay plan from episode entries, or read a native output.")
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("plan", help="verify episode entries and their grant stores and write the plan")
    make.add_argument("--episode", required=True, action="append", type=Path, help="episode entry directory")
    make.add_argument("--grant-store", action="append", type=Path, default=[],
                      help="directory that holds <grant_id>/store, searched after the path the entry recorded")
    make.add_argument("--run-id", required=True)
    make.add_argument("--output", required=True, type=Path, help="new plan file")
    show = commands.add_parser("show", help="verify a native output and print one stored value and its grid digest")
    show.add_argument("directory", type=Path)
    show.add_argument("--sequence", type=int, default=0, help="opportunity in the order of the plan (default 0)")
    show.add_argument("--rx", type=int, default=0)
    show.add_argument("--symbol", type=int, default=0)
    show.add_argument("--subcarrier", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            plan = entry_plan(args.run_id, args.episode, args.grant_store)
            write_plan(plan, args.output)
            print(json.dumps({"plan": str(args.output), "run_id": plan["run_id"], "gain": plan["gain"]["value"],
                              "grants": len(plan["grants"]), "realizations": len(plan["sources"]),
                              "opportunities": len(plan["opportunities"])}, indent=2))
            return 0
        print(json.dumps(summary(open_output(args.directory), args.sequence, args.rx, args.symbol, args.subcarrier),
                         indent=2))
        return 0
    except GridApplyError as error:
        parser.exit(2, f"grid_apply: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
