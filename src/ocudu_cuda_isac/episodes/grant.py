"""One legal uplink grant and its transmit reference (Sprint 2.1, selector grant_fixture).

A pusch-grant/v1 record names everything a receiver and a transmit reference need for one PUSCH
transmission: the original transport block, RNTI, cell and scrambling identities, numerology,
bandwidth part, slot, symbols, DM-RS configuration, layers and port mapping. This module validates
the record, derives the transport block size, the resource elements in mapping order and the DM-RS
reference symbols, and keeps them with the transmit grid in a checksummed store entry.

Coordinates are carrier coordinates everywhere: subcarrier k = 12 * CRB + resource element, where
CRB 0 is the first resource block of the carrier grid and the DM-RS reference point. Subcarrier k
lies at (k - 6 * grid_size_rb) * subcarrier spacing from the carrier frequency.

Standard library only, so the record is readable on any host. Data symbols are not computed here:
they come from the transmit references (the pinned OCUDU chain and Sionna NR) and are stored.
"""

import argparse
import ast
from fractions import Fraction
import hashlib
import json
from math import floor, isfinite
from pathlib import Path
import re
import struct
import sys
import tempfile

from ..replay import ContractError, load_json
from ..sensing import sources
from ..sensing.sources import pusch_dmrs

SCHEMA = "pusch-grant/v1"
STORE_SCHEMA = "pusch-grant-store/v1"
TB_GENERATOR = "sha256-counter/v1"
SYMBOLS_PER_SLOT = 14
SUBCARRIERS_PER_RB = 12
RECORD_FIELDS = ("schema", "grant_id", "carrier", "bwp", "slot", "rnti", "n_id", "allocation", "mcs", "codeword",
                 "layers", "dmrs", "tx", "rx_ports", "tb")
# MCS index tables for PUSCH without transform precoding (TS 38.214 Tables 5.1.3.1-1 to 5.1.3.1-3):
# bits per symbol and target code rate times 1024. Reserved indices are absent.
MCS = {
    "qam64": tuple([(2, r) for r in (120, 157, 193, 251, 308, 379, 449, 526, 602, 679)]
                   + [(4, r) for r in (340, 378, 434, 490, 553, 616, 658)]
                   + [(6, r) for r in (438, 466, 517, 567, 616, 666, 719, 772, 822, 873, 910, 948)]),
    "qam256": tuple([(2, r) for r in (120, 193, 308, 449, 602)]
                    + [(4, r) for r in (378, 434, 490, 553, 616, 658)]
                    + [(6, r) for r in (466, 517, 567, 616, 666, 719, 772, 822, 873)]
                    + [(8, r) for r in (682.5, 711, 754, 797, 841, 885, 916.5, 948)]),
    "qam64LowSe": tuple([(2, r) for r in (30, 40, 50, 64, 78, 99, 120, 157, 193, 251, 308, 379, 449, 526, 602)]
                        + [(4, r) for r in (340, 378, 434, 490, 553, 616)]
                        + [(6, r) for r in (438, 466, 517, 567, 616, 666, 719, 772)]),
}
MODULATION = {2: "QPSK", 4: "16QAM", 6: "64QAM", 8: "256QAM"}
# Transport block sizes for N_info <= 3824 (TS 38.214 Table 5.1.3.2-1).
TBS_TABLE = (
    24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 208, 224,
    240, 256, 272, 288, 304, 320, 336, 352, 368, 384, 408, 432, 456, 480, 504, 528, 552, 576, 608, 640, 672, 704,
    736, 768, 808, 848, 888, 928, 984, 1032, 1064, 1128, 1160, 1192, 1224, 1256, 1288, 1320, 1352, 1416, 1480,
    1544, 1608, 1672, 1736, 1800, 1864, 1928, 2024, 2088, 2152, 2216, 2280, 2408, 2472, 2536, 2600, 2664, 2728,
    2792, 2856, 2976, 3104, 3240, 3368, 3496, 3624, 3752, 3824)
# The pinned resource grid stores each component as a 16-bit brain float: the upper half of an
# IEEE single, eight significant bits, rounded to nearest even. A stored value therefore lies
# within 2**-8 of its magnitude of the value written. The absolute term covers single-precision
# arithmetic before the rounding.
STORAGE = {"name": "complex bf16", "relative": 2.0 ** -8, "absolute": 1e-6}
GRID_STEM = r"x(_[a-z0-9]+)*"
ENTRY_FILES = ("tb.bin", "grant.json", "source.json", "pilot_index.npy", "pilot_ref.npy", "data_index.npy")
INT64_MAX = 2**63 - 1


class GrantError(ValueError):
    """A grant record, transport block or store entry violates the documented interface."""


def _fields(value, fields, name):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise GrantError(f"{name} requires exactly {', '.join(fields)}")
    return value


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise GrantError(f"{name} must be an integer from {low} to {high}")
    return value


def _integers(value, name, low, high):
    if not isinstance(value, list) or not value:
        raise GrantError(f"{name} must be a nonempty list")
    return [_integer(item, name, low, high) for item in value]


def transport_block_size(prbs, symbols, dmrs_re_per_prb, bits_per_symbol, rate_x1024, layers=1):
    """Transport block size in bits for a PUSCH allocation (TS 38.214 6.1.4.2, exact arithmetic)."""
    re_per_prb = min(156, SUBCARRIERS_PER_RB * symbols - dmrs_re_per_prb)
    rate = Fraction(rate_x1024) / 1024
    info = re_per_prb * prbs * rate * bits_per_symbol * layers
    if info <= 0:
        raise GrantError("the allocation carries no information bits")
    if info <= 3824:
        step = 2 ** max(3, (info.numerator // info.denominator).bit_length() - 1 - 6)
        quantized = max(24, step * (info // step))
        return next(size for size in TBS_TABLE if size >= quantized)
    reduced = info - 24
    step = 2 ** ((reduced.numerator // reduced.denominator).bit_length() - 1 - 5)
    quantized = max(3840, step * floor(reduced / step + Fraction(1, 2)))  # round half up
    if rate <= Fraction(1, 4):
        blocks = -(-(quantized + 24) // 3816)
    elif quantized > 8424:
        blocks = -(-(quantized + 24) // 8424)
    else:
        blocks = 1
    return 8 * blocks * -(-(quantized + 24) // (8 * blocks)) - 24


def coding(tbs_bits, rate_x1024):
    """LDPC base graph, transport block CRC length and code block count (TS 38.212 6.2.2, 5.2.2)."""
    rate = Fraction(rate_x1024) / 1024
    second = tbs_bits <= 292 or rate <= Fraction(1, 4) or (tbs_bits <= 3824 and rate <= Fraction(67, 100))
    crc = 16 if tbs_bits <= 3824 else 24
    largest = 3840 if second else 8448
    total = tbs_bits + crc
    blocks = 1 if total <= largest else -(-total // (largest - 24))
    return {"ldpc_base_graph": "BG2" if second else "BG1", "tb_crc_bits": crc, "codeblocks": blocks}


def generated_block(seed, size):
    """Deterministic payload: SHA-256 of 'seed:counter' blocks, truncated to size bytes."""
    if not isinstance(seed, str) or not seed or not seed.isascii():
        raise GrantError("a generated transport block needs a nonempty ASCII seed")
    count = -(-size // 32)
    return b"".join(hashlib.sha256(f"{seed}:{index}".encode("ascii")).digest() for index in range(count))[:size]


def bf16(value):
    """Round one real value the way the pinned resource grid stores it."""
    raw = struct.unpack("<I", struct.pack("<f", value))[0]
    raw = (raw + 0x7FFF + ((raw >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", raw))[0]


def within_storage(value, reference):
    """True when a stored complex value equals a reference to the declared representation tolerance."""
    return all(abs(got - want) <= STORAGE["relative"] * abs(want) + STORAGE["absolute"]
               for got, want in ((value.real, reference.real), (value.imag, reference.imag)))


class Grant:
    """A validated grant. Build it with build() or load(); every derived value is fixed at creation."""

    def __init__(self, record, tb, tb_origin):
        self._record = record
        self.tb = tb
        self.tb_origin = tb_origin
        self.grant_id = record["grant_id"]
        carrier, bwp, slot, allocation, dmrs = (record[key] for key in ("carrier", "bwp", "slot", "allocation", "dmrs"))
        self.scs_khz, self.grid_size_rb = carrier["scs_khz"], carrier["grid_size_rb"]
        self.carrier_hz, self.n_cell_id = carrier["carrier_hz"], carrier["n_cell_id"]
        self.bwp_start_rb, self.bwp_size_rb = bwp["start_rb"], bwp["size_rb"]
        self.sfn, self.slot = slot["sfn"], slot["slot"]
        self.rnti, self.n_id = record["rnti"], record["n_id"]
        self.mapping_type = allocation["mapping_type"]
        self.prb_start, self.prb_count = allocation["prb_start"], allocation["prb_count"]
        self.symbol_start, self.symbol_count = allocation["symbol_start"], allocation["symbol_count"]
        self.mcs_table, self.mcs_index = record["mcs"]["table"], record["mcs"]["index"]
        self.layers = record["layers"]
        self.dmrs = dict(dmrs)
        self.antenna_ports = record["tx"]["antenna_ports"]
        self.rx_ports = list(record["rx_ports"])
        self.bits_per_symbol, self.rate_x1024 = MCS[self.mcs_table][self.mcs_index]
        self.crb_start = self.bwp_start_rb + self.prb_start
        self.subcarrier_start = SUBCARRIERS_PER_RB * self.crb_start
        self.subcarrier_count = SUBCARRIERS_PER_RB * self.prb_count
        self.grid_subcarriers = SUBCARRIERS_PER_RB * self.grid_size_rb
        groups = dmrs["cdm_groups_without_data"]
        self.tbs_bits = transport_block_size(self.prb_count, self.symbol_count, 6 * groups * len(dmrs["symbols"]),
                                             self.bits_per_symbol, self.rate_x1024, self.layers)
        self.source = sources.resolve(pusch_dmrs.NAME)({
            "scs_khz": self.scs_khz, "sfn": self.sfn, "slot": self.slot, "grid_size_rb": self.grid_size_rb,
            "crb_start": self.crb_start, "crb_count": self.prb_count, "config_type": dmrs["config_type"],
            "symbols": list(dmrs["symbols"]), "ports": list(dmrs["ports"]), "scrambling_id": dmrs["scrambling_id"],
            "n_scid": dmrs["n_scid"], "cdm_groups_without_data": groups})
        self.pilot_index = [tuple(entry) for entry in self.source["mask"]["index"]]
        self.pilot_reference = [[complex(*pair) for pair in row] for row in self.source["reference"]["values"]]
        # Data follows the mapping order of TS 38.211 6.3.1.7: increasing subcarrier, then symbol.
        # On a DM-RS symbol the subcarriers of the CDM groups without data stay empty.
        occupied = range(self.subcarrier_start, self.subcarrier_start + self.subcarrier_count)
        self.data_index = [(symbol, k)
                           for symbol in range(self.symbol_start, self.symbol_start + self.symbol_count)
                           for k in occupied
                           if symbol not in dmrs["symbols"] or (k - self.subcarrier_start) % 2 >= groups]

    def record(self, tb=None):
        """The grant as a JSON object. tb replaces the transport block entry, for a store entry."""
        result = json.loads(json.dumps(self._record))
        result["tb"] = tb if tb is not None else {"hex": self.tb.hex()}
        return result

    def derived(self):
        """Values that follow from the record. Each transmit reference must reproduce them."""
        amplitude = pusch_dmrs.amplitude(self.dmrs["cdm_groups_without_data"])
        return {
            "tbs_bits": self.tbs_bits, "tb_bytes": len(self.tb), "tb_sha256": hashlib.sha256(self.tb).hexdigest(),
            "modulation": MODULATION[self.bits_per_symbol], "bits_per_symbol": self.bits_per_symbol,
            "target_code_rate_x1024": self.rate_x1024, **coding(self.tbs_bits, self.rate_x1024),
            "crb_start": self.crb_start, "subcarrier_start": self.subcarrier_start,
            "subcarrier_count": self.subcarrier_count, "grid_subcarriers": self.grid_subcarriers,
            "data_re": len(self.data_index), "pilot_re": len(self.pilot_index),
            "coded_bits": len(self.data_index) * self.bits_per_symbol * self.layers,
            "tx_scale": {"data_rms": 1.0, "dmrs_amplitude": amplitude,
                         "dmrs_epre_boost_db": -pusch_dmrs.EPRE_RATIO_DB[self.dmrs["cdm_groups_without_data"]],
                         "rule": "TS 38.214 Table 6.2.2-1, as the pinned receiver assumes"},
            "port_mapping": {"layers_to_dmrs_ports": list(self.dmrs["ports"]),
                             "layers_to_antenna_ports": "identity", "rx_ports": list(self.rx_ports)},
            "bit_order": "byte 0 first, most significant bit of each byte first",
            "re_order": "data symbols fill increasing subcarriers of a symbol, then the next symbol",
        }

    def tb_bits(self):
        """The transport block as a_0 ... a_(A-1): byte 0 first, most significant bit first."""
        return [(byte >> shift) & 1 for byte in self.tb for shift in range(7, -1, -1)]

    def subcarrier_offset_hz(self, subcarrier):
        """Frequency of a carrier-coordinate subcarrier relative to the carrier frequency."""
        return (subcarrier - self.grid_subcarriers // 2) * self.scs_khz * 1000

    def symbol_times(self):
        """Per-symbol start times in T_c from the slot start (see pusch_dmrs.symbol_times)."""
        return pusch_dmrs.symbol_times(self.scs_khz, self.slot)

    def occupied(self):
        """Every resource element the grant transmits on, as a set of (symbol, subcarrier)."""
        return set(self.data_index) | set(self.pilot_index)


def _transport_block(entry, base):
    """Resolve the three ways a record can carry its transport block."""
    if not isinstance(entry, dict):
        raise GrantError("tb must be an object holding hex, a generator or a file reference")
    if set(entry) == {"hex"}:
        if not isinstance(entry["hex"], str):
            raise GrantError("tb.hex must be a string of hexadecimal digits")
        try:
            return bytes.fromhex(entry["hex"]), {"form": "hex"}
        except ValueError:
            raise GrantError("tb.hex is not hexadecimal") from None
    if set(entry) == {"generator", "seed", "bytes"}:
        if entry["generator"] != TB_GENERATOR:
            raise GrantError(f"unknown transport block generator {entry['generator']!r}")
        size = _integer(entry["bytes"], "tb.bytes", 1, 2**24)
        return generated_block(entry["seed"], size), {"form": "generator", "generator": TB_GENERATOR,
                                                      "seed": entry["seed"]}
    if set(entry) == {"file", "bytes", "sha256"}:
        name = entry["file"]
        if base is None or not isinstance(name, str) or Path(name).name != name:
            raise GrantError("tb.file must name a file beside the grant record")
        try:
            data = (Path(base) / name).read_bytes()
        except OSError as error:
            raise GrantError(f"cannot read the transport block: {error}") from None
        if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise GrantError("the transport block file differs from the size or hash in the record")
        return data, {"form": "file"}
    raise GrantError("tb must hold exactly hex, or generator/seed/bytes, or file/bytes/sha256")


def build(record, base=None):
    """Validate one pusch-grant/v1 record and return its Grant.

    Every rule is checked here, before any grid or array exists: an invalid port, a resource block
    outside the bandwidth part, a duplicate pilot or a payload that does not match the transport
    block size raises GrantError and nothing is allocated.
    """
    _fields(record, RECORD_FIELDS, "grant")
    if record["schema"] != SCHEMA:
        raise GrantError(f"unsupported grant schema {record['schema']!r}")
    if not isinstance(record["grant_id"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}",
                                                                   record["grant_id"]):
        raise GrantError("grant_id must be 1 to 64 characters from A-Z, a-z, 0-9, dot, underscore and hyphen")

    carrier = _fields(record["carrier"], ("scs_khz", "cyclic_prefix", "grid_size_rb", "carrier_hz", "n_cell_id"),
                      "carrier")
    if carrier["scs_khz"] not in (15, 30, 60, 120) or type(carrier["scs_khz"]) is not int:
        raise GrantError("carrier.scs_khz must be 15, 30, 60 or 120")
    if carrier["cyclic_prefix"] != "normal":
        raise GrantError("only the normal cyclic prefix is supported")
    grid = _integer(carrier["grid_size_rb"], "carrier.grid_size_rb", 1, 275)
    _integer(carrier["carrier_hz"], "carrier.carrier_hz", 1, INT64_MAX)
    _integer(carrier["n_cell_id"], "carrier.n_cell_id", 0, 1007)

    bwp = _fields(record["bwp"], ("start_rb", "size_rb"), "bwp")
    bwp_start = _integer(bwp["start_rb"], "bwp.start_rb", 0, 274)
    bwp_size = _integer(bwp["size_rb"], "bwp.size_rb", 1, 275)
    if bwp_start + bwp_size > grid:
        raise GrantError(f"the bandwidth part (CRB {bwp_start} to {bwp_start + bwp_size - 1}) leaves the "
                         f"{grid}-block carrier grid")

    slot = _fields(record["slot"], ("sfn", "slot"), "slot")
    _integer(slot["sfn"], "slot.sfn", 0, 1023)
    _integer(slot["slot"], "slot.slot", 0, 10 * carrier["scs_khz"] // 15 - 1)
    _integer(record["rnti"], "rnti", 1, 65535)
    _integer(record["n_id"], "n_id", 0, 1023)

    allocation = _fields(record["allocation"],
                         ("mapping_type", "prb_start", "prb_count", "symbol_start", "symbol_count"), "allocation")
    prb_start = _integer(allocation["prb_start"], "allocation.prb_start", 0, 274)
    prb_count = _integer(allocation["prb_count"], "allocation.prb_count", 1, 275)
    if prb_start + prb_count > bwp_size:
        raise GrantError(f"out-of-BWP PRB: the allocation (PRB {prb_start} to {prb_start + prb_count - 1}) leaves "
                         f"the {bwp_size}-block bandwidth part")
    symbol_start = _integer(allocation["symbol_start"], "allocation.symbol_start", 0, SYMBOLS_PER_SLOT - 1)
    symbol_count = _integer(allocation["symbol_count"], "allocation.symbol_count", 1, SYMBOLS_PER_SLOT)
    if symbol_start + symbol_count > SYMBOLS_PER_SLOT:
        raise GrantError("the symbol allocation leaves the slot")

    mcs = _fields(record["mcs"], ("table", "index"), "mcs")
    if mcs["table"] not in MCS:
        raise GrantError(f"mcs.table must be one of {', '.join(sorted(MCS))}")
    _integer(mcs["index"], "mcs.index", 0, len(MCS[mcs["table"]]) - 1)

    codeword = _fields(record["codeword"], ("rv", "new_data"), "codeword")
    if codeword["rv"] != 0 or type(codeword["rv"]) is not int or codeword["new_data"] is not True:
        raise GrantError("only a new transmission with redundancy version 0 is supported")
    if record["layers"] != 1 or type(record["layers"]) is not int:
        raise GrantError("only one transmission layer is supported")
    tx = _fields(record["tx"], ("antenna_ports", "precoding"), "tx")
    if tx["antenna_ports"] != 1 or type(tx["antenna_ports"]) is not int or tx["precoding"] != "identity":
        raise GrantError("only one antenna port with identity precoding is supported")

    dmrs = _fields(record["dmrs"], ("config_type", "length", "type_a_position", "additional_position", "symbols",
                                    "ports", "scrambling_id", "n_scid", "cdm_groups_without_data"), "dmrs")
    if dmrs["config_type"] != 1 or dmrs["length"] != 1:
        raise GrantError("only single-symbol DM-RS of configuration type 1 is supported")
    _integer(dmrs["type_a_position"], "dmrs.type_a_position", 2, 3)
    _integer(dmrs["additional_position"], "dmrs.additional_position", 0, 3)
    ports = _integers(dmrs["ports"], "dmrs.ports", 0, 4095)
    for port in ports:
        if port not in pusch_dmrs.TYPE1_PORTS:
            raise GrantError(f"invalid port {port}: single-symbol DM-RS type 1 has ports 0 to 3")
    if len(set(ports)) != len(ports):
        raise GrantError("duplicate pilot: a DM-RS port is listed twice")
    if ports != list(range(record["layers"])):
        raise GrantError(f"invalid port mapping {ports}: the pinned receiver maps layer i to DM-RS port i")
    symbols = _integers(dmrs["symbols"], "dmrs.symbols", 0, SYMBOLS_PER_SLOT - 1)
    if len(set(symbols)) != len(symbols):
        raise GrantError("duplicate pilot: a DM-RS symbol is listed twice")
    _integer(dmrs["scrambling_id"], "dmrs.scrambling_id", 0, 65535)
    _integer(dmrs["n_scid"], "dmrs.n_scid", 0, 1)
    _integer(dmrs["cdm_groups_without_data"], "dmrs.cdm_groups_without_data", 1, 2)
    try:
        legal = pusch_dmrs.symbol_positions(allocation["mapping_type"], dmrs["type_a_position"],
                                            dmrs["additional_position"], symbol_start, symbol_count)
    except sources.SourceError as error:
        raise GrantError(str(error)) from None
    if symbols != legal:
        raise GrantError(f"dmrs.symbols {symbols} are not the positions {legal} that TS 38.211 Table 6.4.1.1.3-3 "
                         "gives for this allocation")

    rx_ports = _integers(record["rx_ports"], "rx_ports", 0, 15)
    if len(set(rx_ports)) != len(rx_ports):
        raise GrantError("invalid port: a receive port is listed twice")
    if rx_ports != list(range(len(rx_ports))):
        # pusch_demodulator_impl::get_ch_data_re of the pinned source reads grid port i for the
        # i-th listed port when the data mask is contiguous, so another list decodes the wrong ports.
        raise GrantError(f"invalid port list {rx_ports}: the pinned receiver reads receive ports 0 to N-1 in order")

    tb, origin = _transport_block(record["tb"], base)
    bits_per_symbol, rate = MCS[mcs["table"]][mcs["index"]]
    tbs = transport_block_size(prb_count, symbol_count, 6 * dmrs["cdm_groups_without_data"] * len(symbols),
                               bits_per_symbol, rate, record["layers"])
    if 8 * len(tb) != tbs:
        raise GrantError(f"mismatched payload: the transport block holds {8 * len(tb)} bits and the grant "
                         f"carries {tbs}")
    try:
        return Grant(json.loads(json.dumps({key: record[key] for key in RECORD_FIELDS if key != "tb"})), tb, origin)
    except sources.SourceError as error:
        raise GrantError(str(error)) from None


def load(path):
    """Read and validate a grant record file. A file reference resolves beside the record."""
    path = Path(path)
    try:
        return build(load_json(path), base=path.parent)
    except ContractError as error:
        raise GrantError(str(error)) from None
    except OSError as error:
        raise GrantError(f"cannot read the grant record: {error}") from None


def realize(record, make_grid, data=None, base=None):
    """Validate a record, then allocate a grid and write the transmit reference into it.

    make_grid(ports, symbols, subcarriers) returns an object with put(port, symbol, subcarrier,
    value). It is called only after the record and the data length passed every check. Pilots are
    always written. data, when given, holds one complex symbol per data resource element in mapping
    order. Returns (grant, grid).
    """
    grant = build(record, base)
    if data is not None and len(data) != len(grant.data_index):
        raise GrantError(f"mismatched payload: {len(data)} data symbols for {len(grant.data_index)} resource elements")
    grid = make_grid(grant.antenna_ports, SYMBOLS_PER_SLOT, grant.grid_subcarriers)
    for (symbol, subcarrier), value in zip(grant.pilot_index, grant.pilot_reference[0]):
        grid.put(0, symbol, subcarrier, value)
    if data is not None:
        for (symbol, subcarrier), value in zip(grant.data_index, data):
            grid.put(0, symbol, subcarrier, value)
    return grant, grid


def check_reference(grant, x):
    """Compare a transmit grid with the grant and return what was measured.

    x holds antenna_ports * 14 * grid_subcarriers complex values, last axis fastest. The set of
    nonzero resource elements must equal the grant's data and pilot elements exactly, and every
    pilot must equal its reference symbol to the storage tolerance.
    """
    expected = grant.antenna_ports * SYMBOLS_PER_SLOT * grant.grid_subcarriers
    if len(x) != expected:
        raise GrantError(f"the transmit grid holds {len(x)} values, the grant's carrier grid has {expected}")
    if any(not (isfinite(value.real) and isfinite(value.imag)) for value in x):
        raise GrantError("the transmit grid holds a nonfinite value")
    width = grant.grid_subcarriers
    found = {(index // width % SYMBOLS_PER_SLOT, index % width) for index, value in enumerate(x) if value != 0}
    wanted = grant.occupied()
    if found != wanted:
        extra, missing = sorted(found - wanted), sorted(wanted - found)
        where = (extra or missing)[0]
        raise GrantError(f"the transmit grid differs from the grant's resource elements: {len(extra)} unexpected, "
                         f"{len(missing)} empty, first at symbol {where[0]}, subcarrier {where[1]}")
    worst, exact = 0.0, 0
    for (symbol, subcarrier), reference in zip(grant.pilot_index, grant.pilot_reference[0]):
        value = x[symbol * width + subcarrier]
        if not within_storage(value, reference):
            raise GrantError(f"the pilot at symbol {symbol}, subcarrier {subcarrier} is {value}, expected {reference}")
        worst = max(worst, abs(value - reference))
        exact += value == complex(bf16(reference.real), bf16(reference.imag))
    return {"occupied_re": len(found), "pilots_checked": len(grant.pilot_index), "pilots_equal_after_rounding": exact,
            "pilot_max_abs_error": worst}


def _npy_bytes(descr, shape, flat):
    """Serialize a little-endian array in NumPy's .npy format, version 1.0, C order."""
    header = "{'descr': %r, 'fortran_order': False, 'shape': %r, }" % (descr, tuple(shape))
    header += " " * (-(10 + len(header) + 1) % 64) + "\n"
    try:
        if descr == "<c8":
            body = struct.pack(f"<{2 * len(flat)}f", *[part for value in flat for part in (value.real, value.imag)])
        elif descr == "<i4":
            body = struct.pack(f"<{len(flat)}i", *flat)
        else:
            raise GrantError(f"unsupported array type {descr}")
    except (OverflowError, struct.error):
        raise GrantError(f"a value does not fit the array type {descr}") from None
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode("latin1") + body


def read_npy(path):
    """Read a little-endian complex64 or int32 .npy file. Returns (descr, shape, flat values)."""
    data = Path(path).read_bytes()
    if data[:8] != b"\x93NUMPY\x01\x00":
        raise GrantError(f"{Path(path).name} is not a version 1.0 .npy file")
    size = struct.unpack("<H", data[8:10])[0]
    try:
        header = ast.literal_eval(data[10:10 + size].decode("latin1"))
        descr, shape, fortran = header["descr"], tuple(header["shape"]), header["fortran_order"]
    except (ValueError, SyntaxError, KeyError, TypeError):
        raise GrantError(f"{Path(path).name} has an unreadable array header") from None
    count = 1
    for dimension in shape:
        count *= dimension
    body = data[10 + size:]
    if fortran or descr not in ("<c8", "<i4") or len(body) != count * (8 if descr == "<c8" else 4):
        raise GrantError(f"{Path(path).name} is not a C-ordered little-endian complex64 or int32 array")
    if descr == "<i4":
        return descr, shape, list(struct.unpack(f"<{count}i", body))
    parts = struct.unpack(f"<{2 * count}f", body)
    return descr, shape, [complex(parts[i], parts[i + 1]) for i in range(0, 2 * count, 2)]


def export(grant, directory, grids=None):
    """Write the stimulus-store entry of one grant into a new directory and return its manifest.

    The entry holds the grant, the original transport block, the sensing source with the pilot
    coordinates and reference symbols and the data coordinates in mapping order. grids maps a file
    stem to {"values": X[tx_port, symbol, subcarrier] as a flat sequence, "provenance": ...}. The
    stem "x" is the transmit grid at the native boundary. Further references are named "x_<name>".
    Every grid is checked against the grant first. The directory appears only when complete.
    """
    directory = Path(directory)
    if directory.exists():
        raise GrantError(f"{directory} already exists. A store entry is never overwritten.")
    grids = dict(grids or {})
    if grids and "x" not in grids:
        raise GrantError('a store entry with grids needs the native-boundary grid under the name "x"')
    references = {}
    for stem, grid in grids.items():
        if not re.fullmatch(GRID_STEM, stem) or not isinstance(grid, dict) or set(grid) != {"values", "provenance"}:
            raise GrantError('a grid is named "x" or "x_<name>" and holds exactly values and provenance')
        references[stem + ".npy"] = {"provenance": grid["provenance"], "check": check_reference(grant, grid["values"])}
    files = {}

    def add(staging, name, data, **description):
        (staging / name).write_bytes(data)
        files[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), **description}

    def text(value):
        return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("ascii")

    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory.parent) as temporary:
        staging = Path(temporary) / "entry"
        staging.mkdir()
        add(staging, "tb.bin", grant.tb, contents="original transport block, byte 0 first")
        add(staging, "grant.json", text(grant.record({"file": "tb.bin", "bytes": len(grant.tb),
                                                      "sha256": files["tb.bin"]["sha256"]})), contents=SCHEMA)
        add(staging, "pilot_index.npy",
            _npy_bytes("<i4", (len(grant.pilot_index), 2), [part for entry in grant.pilot_index for part in entry]),
            axes=["pilot", "coordinate"], columns=["symbol", "subcarrier"], shape=[len(grant.pilot_index), 2],
            dtype="<i4", order="C", units="slot symbol index and carrier-coordinate subcarrier index")
        add(staging, "pilot_ref.npy",
            _npy_bytes("<c8", (len(grant.pilot_reference), len(grant.pilot_index)),
                       [value for row in grant.pilot_reference for value in row]),
            axes=["dmrs_port", "pilot"], shape=[len(grant.pilot_reference), len(grant.pilot_index)], dtype="<c8",
            order="C", units="transmitted DM-RS symbol, linear amplitude relative to unit-power data")
        add(staging, "data_index.npy",
            _npy_bytes("<i4", (len(grant.data_index), 2), [part for entry in grant.data_index for part in entry]),
            axes=["data_re in mapping order", "coordinate"], columns=["symbol", "subcarrier"],
            shape=[len(grant.data_index), 2], dtype="<i4", order="C",
            units="slot symbol index and carrier-coordinate subcarrier index")
        source = json.loads(json.dumps(grant.source))
        source["mask"]["index"] = {"file": "pilot_index.npy", "sha256": files["pilot_index.npy"]["sha256"]}
        source["reference"]["values"] = {"file": "pilot_ref.npy", "sha256": files["pilot_ref.npy"]["sha256"]}
        add(staging, "source.json", text(source), contents=sources.SCHEMA)
        shape = (grant.antenna_ports, SYMBOLS_PER_SLOT, grant.grid_subcarriers)
        for stem, grid in grids.items():
            add(staging, stem + ".npy", _npy_bytes("<c8", shape, grid["values"]),
                axes=["tx_port", "symbol", "subcarrier"], shape=list(shape), dtype="<c8", order="C",
                units="transmitted resource element, linear amplitude, unit-power data")
        manifest = {
            "schema": STORE_SCHEMA, "grant_id": grant.grant_id, "access": "offline_stimulus_and_evaluator",
            "coordinates": {
                "subcarrier": "carrier coordinates: k = 12 * CRB + resource element, CRB 0 at grid subcarrier 0",
                "subcarrier_offset_hz": "(k - 6 * grid_size_rb) * scs_khz * 1000 from carrier_hz",
                "symbol": "index in the slot, 0 to 13; symbol_times gives each start in T_c from the slot start",
                "dmrs_reference_point": "subcarrier 0 of CRB 0 (TS 38.211 6.4.1.1.3)"},
            "tc_per_second": pusch_dmrs.TC_PER_SECOND, "symbol_times": grant.symbol_times(),
            "tb_origin": grant.tb_origin, "derived": grant.derived(), "representation": dict(STORAGE),
            "transmit_references": references, "files": files,
        }
        (staging / "manifest.json").write_bytes(text(manifest))
        staging.rename(directory)
    return manifest


class Store:
    """A verified store entry: the grant, its manifest and the transmit grids it holds, by file stem."""

    def __init__(self, directory, grant, manifest, grids):
        self.directory, self.grant, self.manifest, self.grids = directory, grant, manifest, grids
        self.x = grids.get("x")

    def pilot(self, index):
        """Coordinate and reference symbol of one pilot, with the transmitted value when a grid is stored."""
        grant = self.grant
        if type(index) is not int or not 0 <= index < len(grant.pilot_index):
            raise GrantError(f"the record has pilots 0 to {len(grant.pilot_index) - 1}")
        symbol, subcarrier = grant.pilot_index[index]
        reference = grant.pilot_reference[0][index]
        result = {"index": index, "symbol": symbol, "subcarrier": subcarrier,
                  "offset_hz": grant.subcarrier_offset_hz(subcarrier), "dmrs_port": grant.dmrs["ports"][0],
                  "reference": [reference.real, reference.imag]}
        if self.x is not None:
            value = self.x[symbol * grant.grid_subcarriers + subcarrier]
            result["transmitted"] = [value.real, value.imag]
        return result

    def data(self, index):
        """Coordinate and transmitted value of one data symbol, counted in mapping order."""
        grant = self.grant
        if self.x is None:
            raise GrantError("this record holds no transmit grid, so it has no data symbols yet")
        if type(index) is not int or not 0 <= index < len(grant.data_index):
            raise GrantError(f"the record has data symbols 0 to {len(grant.data_index) - 1}")
        symbol, subcarrier = grant.data_index[index]
        value = self.x[symbol * grant.grid_subcarriers + subcarrier]
        return {"index": index, "symbol": symbol, "subcarrier": subcarrier,
                "offset_hz": grant.subcarrier_offset_hz(subcarrier), "value": [value.real, value.imag]}


def open_store(directory):
    """Verify every file of a store entry against its manifest and against the grant, then return it."""
    directory = Path(directory)
    try:
        manifest = load_json(directory / "manifest.json")
    except (ContractError, OSError) as error:
        raise GrantError(f"{directory} is not a complete store entry: {error}") from None
    if not isinstance(manifest, dict) or manifest.get("schema") != STORE_SCHEMA \
            or not isinstance(manifest.get("files"), dict):
        raise GrantError(f"{directory / 'manifest.json'} is not a {STORE_SCHEMA} manifest")
    listed = set(manifest["files"])
    extra = listed - set(ENTRY_FILES)
    if not set(ENTRY_FILES) <= listed or any(not re.fullmatch(GRID_STEM + r"\.npy", name) for name in extra) \
            or (extra and "x.npy" not in extra):
        raise GrantError("the manifest does not list exactly the files of a store entry")
    for name, entry in manifest["files"].items():
        try:
            data = (directory / name).read_bytes()
        except OSError as error:
            raise GrantError(f"cannot read {name}: {error}") from None
        if not isinstance(entry, dict) or len(data) != entry.get("bytes") \
                or hashlib.sha256(data).hexdigest() != entry.get("sha256"):
            raise GrantError(f"{name} differs from the size or hash in the manifest")
    grant = load(directory / "grant.json")
    if manifest.get("grant_id") != grant.grant_id or manifest.get("derived") != json.loads(json.dumps(grant.derived())):
        raise GrantError("the manifest's derived values differ from the grant")
    # The stored arrays must be the ones this grant yields: a consumer may read them directly.
    _, shape, flat = read_npy(directory / "pilot_index.npy")
    stored = [tuple(flat[i:i + 2]) for i in range(0, len(flat), 2)]
    if len(set(stored)) != len(stored):
        raise GrantError("duplicate pilot in pilot_index.npy")
    if shape != (len(grant.pilot_index), 2) or stored != grant.pilot_index:
        raise GrantError("pilot_index.npy differs from the pilots of the grant")
    _, shape, flat = read_npy(directory / "data_index.npy")
    if shape != (len(grant.data_index), 2) or [tuple(flat[i:i + 2]) for i in range(0, len(flat), 2)] != grant.data_index:
        raise GrantError("data_index.npy differs from the data resource elements of the grant")
    _, shape, flat = read_npy(directory / "pilot_ref.npy")
    wanted = [complex(*struct.unpack("<2f", struct.pack("<2f", value.real, value.imag)))
              for row in grant.pilot_reference for value in row]
    if shape != (len(grant.pilot_reference), len(grant.pilot_index)) or flat != wanted:
        raise GrantError("pilot_ref.npy differs from the reference symbols of the grant")
    source = json.loads(json.dumps(grant.source))
    source["mask"]["index"] = {"file": "pilot_index.npy", "sha256": manifest["files"]["pilot_index.npy"]["sha256"]}
    source["reference"]["values"] = {"file": "pilot_ref.npy", "sha256": manifest["files"]["pilot_ref.npy"]["sha256"]}
    try:
        stored_source = load_json(directory / "source.json")
    except ContractError as error:
        raise GrantError(str(error)) from None
    if stored_source != source:
        raise GrantError("source.json differs from the sensing source of the grant")
    grids = {}
    for name in sorted(extra):
        descr, shape, values = read_npy(directory / name)
        if descr != "<c8" or shape != (grant.antenna_ports, SYMBOLS_PER_SLOT, grant.grid_subcarriers):
            raise GrantError(f"{name} does not have the grant's carrier grid shape")
        check_reference(grant, values)
        grids[name[:-4]] = values
    return Store(directory, grant, manifest, grids)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export a grant record into a store entry, or read one back.")
    commands = parser.add_subparsers(dest="command", required=True)
    write = commands.add_parser("export", help="validate a grant record and write its store entry")
    write.add_argument("--grant", required=True, type=Path)
    write.add_argument("--output-dir", required=True, type=Path, help="new directory for the entry")
    show = commands.add_parser("show", help="verify a store entry and print a pilot, a data symbol and the TB hash")
    show.add_argument("directory", type=Path)
    show.add_argument("--pilot", type=int, default=0, help="pilot number in the record's order (default 0)")
    show.add_argument("--data", type=int, default=0, help="data symbol number in mapping order (default 0)")
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            manifest = export(load(args.grant), args.output_dir)
            print(json.dumps({"store": str(args.output_dir), "grant_id": manifest["grant_id"],
                              "tbs_bits": manifest["derived"]["tbs_bits"],
                              "tb_sha256": manifest["derived"]["tb_sha256"]}, indent=2))
            return 0
        store = open_store(args.directory)
        result = {"grant_id": store.grant.grant_id, "verified_files": sorted(store.manifest["files"]),
                  "tb_bytes": len(store.grant.tb), "tb_sha256": hashlib.sha256(store.grant.tb).hexdigest(),
                  "tbs_bits": store.grant.tbs_bits, "pilot": store.pilot(args.pilot),
                  "data": store.data(args.data) if store.x is not None else None,
                  "transmit_grids": sorted(store.grids)}
        print(json.dumps(result, indent=2))
        return 0
    except GrantError as error:
        parser.exit(2, f"grant: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
