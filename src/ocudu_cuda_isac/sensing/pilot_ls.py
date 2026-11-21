"""Pilot least squares on a receiver grid: the reference of the stage O2 (Sprint 3.1, selector pilot_ls).

For every pilot of a sensing source and every receive plane

    H_ls = Y * conj(X) / |X|^2

where Y is the stored value of the receiver grid and X the reference symbol of the source record.
The native observer (native/ocudu_adapter/observer/observer_core.h) computes the same values with
the same order of arithmetic, so the two agree bit for bit:

  1. Y and X are single-precision values, widened to double precision without change;
  2. d = xr * xr + xi * xi, nr = yr * xr + yi * xi, ni = yi * xr - yr * xi;
  3. hr = nr / d, hi = ni / d;
  4. each part is converted to single precision, round to nearest, ties to even.

Payload layout ocudu_o2_comb_c64: [rx_plane][reference_symbol][pilot], last axis fastest. The pilot
axis is the sorted set of subcarriers that carry a pilot on any reference symbol of the source.
One validity bit per element, in payload order, bit (n mod 8) of byte (n // 8): set for a pilot
with a finite estimate, a measured zero included, and cleared for a hole.

Standard library only. This module imports the registry of sensing sources and nothing of a host,
a simulator or an episode: it belongs to the sensing side.
"""

import hashlib
from math import isfinite
import struct

from . import sources

DESCRIPTOR_SCHEMA = "obs-descriptor/v1"
STAGE = "O2"
PRODUCER = "observer"
STEP = "pilot_ls"
LAYOUT = "ocudu_o2_comb_c64"
DTYPE = "<c8"
AXES = ["rx_plane", "reference_symbol", "pilot"]
RULE = "H = Y conj(X) / |X|^2"
PRECISION = "complex64, computed in double precision from single-precision operands"
TIME_UNIT = "tc"
TC_PER_SECOND = 480_000 * 4096
SFN_PERIOD = 1024
SYMBOLS_PER_SLOT = 14
KAPPA = 64
# The groups of an obs-descriptor/v1 record and the fields of each. A record holds these and no other.
GROUPS = {
    "keys": ("epoch", "sequence", "episode", "opportunity", "grant"),
    "radio": ("cell", "ue", "sfn", "slot", "absolute_slot"),
    "what": ("stage", "producer", "producer_steps", "source", "port", "layout", "dtype", "scale", "rule", "precision"),
    "shape": ("rank", "axes", "shape", "strides"),
    "allocation": ("first_subcarrier", "subcarrier_step", "subcarrier_count", "dmrs_symbols", "rx_count",
                   "layer_count", "antenna_map", "coordinates_sha256"),
    "validity": ("mask_bytes", "file", "sha256", "valid", "holes"),
    "time": ("unit", "tc_per_second", "t_first_sample", "t_last_sample", "t_ready_ns", "t_publish_ns"),
    "payload": ("bytes", "file", "digest_kind", "digest"),
    "host": ("name", "opportunity_id", "opportunity_sequence", "grant_id", "grant_record_sha256", "ingress",
             "completion", "service_us"),
}
RECORD_FIELDS = ("schema", "run_id") + tuple(GROUPS)
INGRESS_FIELDS = ("grid_index", "generation", "readback_sha256")


class PilotLsError(ValueError):
    """A sensing source, an estimate or an observation record violates the documented interface."""


# ---------------------------------------------------------------- kernel
def single(value):
    """A real value rounded to single precision, as a float. Raises OverflowError beyond its range."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def ls(received, reference):
    """The estimate of one resource element and whether it is finite: (complex, bool).

    received and reference are complex numbers whose parts are single-precision values.
    """
    yr, yi, xr, xi = received.real, received.imag, reference.real, reference.imag
    d = xr * xr + xi * xi
    if d == 0:
        raise PilotLsError("a configured pilot has zero reference power")
    nr = yr * xr + yi * xi
    ni = yi * xr - yr * xi
    try:
        hr, hi = single(nr / d), single(ni / d)
    except OverflowError:
        return 0j, False
    if not (isfinite(hr) and isfinite(hi)):
        return 0j, False
    return complex(hr, hi), True


# ---------------------------------------------------------------- source
class Source:
    """A validated single-port sensing source, compiled into the rectangle of the payload."""

    def __init__(self, record, antenna_map):
        try:
            sources.validate(record)
        except sources.SourceError as error:
            raise PilotLsError(str(error)) from None
        if len(record["ports"]) != 1:
            raise PilotLsError(f"the sensing source {record['source']} has {len(record['ports'])} ports: this observer "
                               "estimates one port. Register one source record per port.")
        slot = record["slot"]
        if not {"tc_per_second", "symbol_useful_start_tc"} <= set(slot) or slot["tc_per_second"] != TC_PER_SECOND \
                or not isinstance(slot["symbol_useful_start_tc"], list) \
                or len(slot["symbol_useful_start_tc"]) != len(slot["symbols"]):
            raise PilotLsError("the sensing source names no start time for each reference symbol in T_c")
        if not isinstance(antenna_map, (list, tuple)) or not antenna_map \
                or any(type(element) is not int or not 0 <= element <= 4095 for element in antenna_map):
            raise PilotLsError("an antenna map names one array element from 0 to 4095 per receive plane")
        if len(set(antenna_map)) != len(antenna_map):
            raise PilotLsError("the antenna map names an array element twice")
        slots_per_frame(slot["scs_khz"])
        self.record = record
        self.name = record["source"]
        self.port = record["ports"][0]
        self.amplitude = float(record["reference"]["amplitude"])
        self.antenna_map = list(antenna_map)
        self.index = [tuple(entry) for entry in record["mask"]["index"]]
        self.reference = [complex(single(pair[0]), single(pair[1])) for pair in record["reference"]["values"][0]]
        self.symbols = sorted({symbol for symbol, _ in self.index})
        self.subcarriers = sorted({subcarrier for _, subcarrier in self.index})
        self.grid_symbols = record["grid"]["symbols"]
        self.grid_subcarriers = record["grid"]["subcarriers"]
        self.scs_khz, self.slot = slot["scs_khz"], slot["slot"]
        self.symbol_useful_start_tc = list(slot["symbol_useful_start_tc"])
        where = {coordinate: number for number, coordinate in enumerate(self.index)}
        # cells[row][column] is the pilot of the record at that element of the rectangle, or None for a hole.
        self.cells = [[where.get((symbol, subcarrier)) for subcarrier in self.subcarriers] for symbol in self.symbols]
        self.holes = sum(cell is None for row in self.cells for cell in row)
        steps = {b - a for a, b in zip(self.subcarriers, self.subcarriers[1:])}
        self.subcarrier_step = steps.pop() if len(steps) == 1 else 0
        self.coordinates_sha256 = hashlib.sha256(
            struct.pack(f"<{len(self.symbols) + len(self.subcarriers)}I", *self.symbols, *self.subcarriers)).hexdigest()

    @property
    def rx_planes(self):
        return len(self.antenna_map)

    @property
    def shape(self):
        return [self.rx_planes, len(self.symbols), len(self.subcarriers)]

    @property
    def strides(self):
        return [len(self.symbols) * len(self.subcarriers), len(self.subcarriers), 1]

    def plane_of(self, element):
        """The receive plane that holds an array element."""
        if element not in self.antenna_map:
            raise PilotLsError(f"the antenna map names no receive plane for the array element {element}")
        return self.antenna_map.index(element)

    def element(self, plane, row, column):
        """Position of one element in the payload."""
        strides = self.strides
        return plane * strides[0] + row * strides[1] + column * strides[2]

    def coordinate(self, row, column):
        """(symbol, subcarrier) of one element of a receive plane, in carrier coordinates."""
        return self.symbols[row], self.subcarriers[column]


class Estimate:
    """The values, the validity bits and the counts of one estimate. Immutable once made."""

    def __init__(self, source, values, mask, valid):
        self.source, self.values, self.mask, self.valid = source, tuple(values), bytes(mask), valid
        self.holes = len(values) - valid

    def is_valid(self, element):
        return bool(self.mask[element // 8] >> (element % 8) & 1)

    def payload(self):
        """The payload bytes: little-endian complex64 in payload order."""
        return payload_bytes(self.values)

    @property
    def payload_sha256(self):
        return hashlib.sha256(self.payload()).hexdigest()

    @property
    def mask_sha256(self):
        return hashlib.sha256(self.mask).hexdigest()


def payload_bytes(values):
    return struct.pack(f"<{2 * len(values)}f", *[part for value in values for part in (value.real, value.imag)])


def payload_values(data):
    """The complex values of a payload."""
    if len(data) % 8:
        raise PilotLsError("a payload holds whole complex64 values")
    parts = struct.unpack(f"<{len(data) // 4}f", data)
    return [complex(parts[i], parts[i + 1]) for i in range(0, len(parts), 2)]


def estimate(source, grid):
    """Estimate one source on a grid. grid(plane, symbol, subcarrier) returns the stored value."""
    values, mask, valid = [], bytearray((source.rx_planes * len(source.symbols) * len(source.subcarriers) + 7) // 8), 0
    for plane in range(source.rx_planes):
        for row, symbol in enumerate(source.symbols):
            for column, subcarrier in enumerate(source.subcarriers):
                which = source.cells[row][column]
                value, ok = (0j, False) if which is None else ls(grid(plane, symbol, subcarrier), source.reference[which])
                if ok:
                    mask[len(values) // 8] |= 1 << (len(values) % 8)
                    valid += 1
                values.append(value if ok else 0j)
    return Estimate(source, values, mask, valid)


def stored_grid(data, planes, symbols, subcarriers):
    """Read access to a stored grid file: little-endian complex64 [rx_port, symbol, subcarrier] in C order."""
    if len(data) != 8 * planes * symbols * subcarriers:
        raise PilotLsError(f"the stored grid holds {len(data)} bytes, expected {8 * planes * symbols * subcarriers}")

    def grid(plane, symbol, subcarrier):
        if not (0 <= plane < planes and 0 <= symbol < symbols and 0 <= subcarrier < subcarriers):
            raise PilotLsError("a read leaves the carrier grid")
        real, imag = struct.unpack_from("<2f", data, 8 * ((plane * symbols + symbol) * subcarriers + subcarrier))
        return complex(real, imag)
    return grid


# ---------------------------------------------------------------- time and keys
def numerology(scs_khz):
    if scs_khz not in (15, 30, 60, 120):
        raise PilotLsError("the subcarrier spacing must be 15, 30, 60 or 120 kHz")
    return (15, 30, 60, 120).index(scs_khz)


def slots_per_frame(scs_khz):
    return 10 << numerology(scs_khz)


def useful_tc(scs_khz):
    """Length of the useful part of a symbol in T_c."""
    return (2048 * KAPPA) >> numerology(scs_khz)


def slot_start_tc(scs_khz, absolute_slot):
    """Start of a slot in T_c, counted from slot 0 of the epoch.

    The first symbol of every half subframe carries the longer cyclic prefix (TS 38.211 5.3.1), so
    the slots of a subframe are summed and not multiplied.
    """
    mu = numerology(scs_khz)
    per = 1 << mu
    start = (absolute_slot // per) * (TC_PER_SECOND // 1000)
    useful, prefix, boundary = (2048 * KAPPA) >> mu, (144 * KAPPA) >> mu, 7 << mu
    for symbol in range((absolute_slot % per) * SYMBOLS_PER_SLOT):
        start += useful + prefix + (16 * KAPPA if symbol % boundary == 0 else 0)
    return start


def opaque_key(text):
    """The leading 64 bits of the SHA-256 of a text, as 16 hexadecimal digits: an opaque join key."""
    return hashlib.sha256(text.encode("ascii")).hexdigest()[:16]


class SlotClock:
    """The absolute slot of the opportunities of one run: it does not wrap with the frame number."""

    def __init__(self):
        self._episodes = {}

    def absolute_slot(self, episode_id, scs_khz, sfn, slot):
        wraps, last = self._episodes.get(episode_id, (0, None))
        if last is not None and (sfn, slot) < last:
            wraps += 1
        self._episodes[episode_id] = (wraps, (sfn, slot))
        return (wraps * SFN_PERIOD + sfn) * slots_per_frame(scs_khz) + slot


def sample_times(source, absolute_slot):
    """Radio time of the first and of the last contributing sample, in T_c from the epoch start."""
    origin = slot_start_tc(source.scs_khz, absolute_slot)
    return (origin + source.symbol_useful_start_tc[0],
            origin + source.symbol_useful_start_tc[-1] + useful_tc(source.scs_khz))


# ---------------------------------------------------------------- record
def _hash(value):
    return isinstance(value, str) and len(value) == 64 and not set(value) - set("0123456789abcdef")


def _key(value):
    return isinstance(value, str) and len(value) == 16 and not set(value) - set("0123456789abcdef")


def _count(value, low=0):
    return type(value) is int and value >= low


def record_violations(record):
    """What is wrong with one obs-descriptor/v1 record on its own. An empty list means it is consistent."""
    if not isinstance(record, dict) or set(record) != set(RECORD_FIELDS):
        return ["the record does not hold exactly the groups of an obs-descriptor/v1 record"]
    for group, fields in GROUPS.items():
        if not isinstance(record[group], dict) or set(record[group]) != set(fields):
            return [f"the group {group} does not hold exactly its fields"]
    found = []
    if record["schema"] != DESCRIPTOR_SCHEMA:
        found.append(f"schema {record['schema']!r}")
    if not isinstance(record["run_id"], str) or not record["run_id"]:
        found.append("no run")
    keys, radio, what, shape = record["keys"], record["radio"], record["what"], record["shape"]
    allocation, validity, time, payload, host = (record[group] for group in
                                                 ("allocation", "validity", "time", "payload", "host"))
    if not _count(keys["epoch"], 1) or not _count(keys["sequence"]) \
            or not all(_key(keys[name]) for name in ("episode", "opportunity", "grant")):
        found.append("the join keys are an epoch from 1, a sequence and three opaque 64-bit keys")
    if not all(_count(radio[name]) for name in GROUPS["radio"]) or radio["sfn"] >= SFN_PERIOD:
        found.append("the radio identity holds non-negative integers")
    if (what["stage"], what["producer"], what["producer_steps"], what["layout"], what["dtype"]) \
            != (STAGE, PRODUCER, [STEP], LAYOUT, DTYPE):
        found.append("the record is not an observer-derived O2 estimate in the layout ocudu_o2_comb_c64")
    if (what["rule"], what["precision"]) != (RULE, PRECISION):
        found.append("the record names another rule or another precision than the estimator uses")
    if not isinstance(what["source"], str) or not what["source"] or not _count(what["port"]) \
            or type(what["scale"]) not in (int, float) or not what["scale"] > 0:
        found.append("the record names no source, no port or no reference amplitude")
    sizes = shape["shape"]
    if shape["rank"] != 3 or shape["axes"] != AXES or not isinstance(sizes, list) or len(sizes) != 3 \
            or not all(_count(size, 1) for size in sizes) or shape["strides"] != [sizes[1] * sizes[2], sizes[2], 1]:
        return found + ["the shape is not three contiguous axes"]
    elements = sizes[0] * sizes[1] * sizes[2]
    symbols = allocation["dmrs_symbols"]
    if not isinstance(symbols, list) or symbols != sorted(set(symbols)) or len(symbols) != sizes[1] \
            or allocation["subcarrier_count"] != sizes[2] or allocation["rx_count"] != sizes[0] \
            or allocation["layer_count"] != 1 or not _hash(allocation["coordinates_sha256"]) \
            or not _count(allocation["first_subcarrier"]) or not _count(allocation["subcarrier_step"]):
        found.append("the allocation does not describe the axes of the payload")
    antenna_map = allocation["antenna_map"]
    if not isinstance(antenna_map, list) or len(antenna_map) != sizes[0] or len(set(antenna_map)) != len(antenna_map):
        found.append("the antenna map does not name one array element per receive plane")
    if validity["mask_bytes"] != (elements + 7) // 8 or not _hash(validity["sha256"]) \
            or not _count(validity["valid"]) or not _count(validity["holes"]) \
            or validity["valid"] + validity["holes"] != elements or not isinstance(validity["file"], str):
        found.append("the validity bits do not cover every element of the payload")
    if (time["unit"], time["tc_per_second"]) != (TIME_UNIT, TC_PER_SECOND) or not _count(time["t_first_sample"]) \
            or not _count(time["t_last_sample"]) or time["t_last_sample"] <= time["t_first_sample"] \
            or not _count(time["t_ready_ns"], 1) or time["t_publish_ns"] is not None:
        found.append("the times are not T_c from the epoch start with a completion time and no publication")
    if payload["bytes"] != 8 * elements or payload["digest_kind"] != "sha256" or not _hash(payload["digest"]) \
            or not isinstance(payload["file"], str):
        found.append("the payload is not complex64 of the stated shape with a SHA-256 digest")
    ingress = host["ingress"]
    if not isinstance(ingress, dict) or set(ingress) != set(INGRESS_FIELDS) or not _hash(ingress["readback_sha256"]) \
            or not isinstance(host["opportunity_id"], str) or not _hash(host["grant_record_sha256"]) \
            or not _count(host["opportunity_sequence"]):
        found.append("the host block names no opportunity, no grant record or no grid digest")
    elif keys["opportunity"] != opaque_key(host["opportunity_id"]) or keys["grant"] != opaque_key(host["grant_record_sha256"]):
        found.append("a join key is not the key of the text in the host block")
    return found


def describe(source, made, context):
    """The obs-descriptor/v1 record of one estimate, as the native tool writes it.

    context holds run_id, epoch, sequence, opportunity_id, opportunity_sequence, episode_id, grant_id,
    grant_record_sha256, cell, ue, sfn, slot, absolute_slot, ingress (grid_index, generation,
    readback_sha256), payload_file, mask_file, t_ready_ns and service_us.
    """
    first, last = sample_times(source, context["absolute_slot"])
    return {
        "schema": DESCRIPTOR_SCHEMA, "run_id": context["run_id"],
        "keys": {"epoch": context["epoch"], "sequence": context["sequence"],
                 "episode": opaque_key(context["episode_id"]), "opportunity": opaque_key(context["opportunity_id"]),
                 "grant": opaque_key(context["grant_record_sha256"])},
        "radio": {"cell": context["cell"], "ue": context["ue"], "sfn": context["sfn"], "slot": context["slot"],
                  "absolute_slot": context["absolute_slot"]},
        "what": {"stage": STAGE, "producer": PRODUCER, "producer_steps": [STEP], "source": source.name,
                 "port": source.port, "layout": LAYOUT, "dtype": DTYPE, "scale": source.amplitude, "rule": RULE,
                 "precision": PRECISION},
        "shape": {"rank": 3, "axes": list(AXES), "shape": source.shape, "strides": source.strides},
        "allocation": {"first_subcarrier": source.subcarriers[0], "subcarrier_step": source.subcarrier_step,
                       "subcarrier_count": len(source.subcarriers), "dmrs_symbols": list(source.symbols),
                       "rx_count": source.rx_planes, "layer_count": 1, "antenna_map": list(source.antenna_map),
                       "coordinates_sha256": source.coordinates_sha256},
        "validity": {"mask_bytes": len(made.mask), "file": context["mask_file"], "sha256": made.mask_sha256,
                     "valid": made.valid, "holes": made.holes},
        "time": {"unit": TIME_UNIT, "tc_per_second": TC_PER_SECOND, "t_first_sample": first, "t_last_sample": last,
                 "t_ready_ns": context["t_ready_ns"], "t_publish_ns": None},
        "payload": {"bytes": 8 * len(made.values), "file": context["payload_file"], "digest_kind": "sha256",
                    "digest": made.payload_sha256},
        "host": {"name": context.get("host", "ocudu-deepsig staged harness"),
                 "opportunity_id": context["opportunity_id"], "opportunity_sequence": context["opportunity_sequence"],
                 "grant_id": context["grant_id"], "grant_record_sha256": context["grant_record_sha256"],
                 "ingress": dict(context["ingress"]), "completion": "host_synchronous",
                 "service_us": context["service_us"]},
    }
