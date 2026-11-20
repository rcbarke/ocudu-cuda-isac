"""Sensing source pusch_dmrs: the DM-RS of one CP-OFDM PUSCH allocation (TS 38.211 6.4.1.1).

Standard library only. The sequence is referenced to subcarrier 0 of common resource block 0, as
the specification requires for PUSCH without transform precoding and as the pinned OCUDU PUSCH
estimator generates it. Configuration type 1 with single-symbol DM-RS is implemented, which is what
the pinned resident receiver path executes.
"""

from math import sqrt

from . import SCHEMA, SourceError, validate

NAME = "pusch_dmrs"
SYMBOLS_PER_SLOT = 14  # normal cyclic prefix
GOLD_OFFSET = 1600     # N_C of TS 38.211 5.2.1
# Basic time unit T_c = 1 / (480 kHz * 4096) and the constant kappa = T_s / T_c (TS 38.211 4.1).
TC_PER_SECOND = 480_000 * 4096
KAPPA = 64
# Ratio of PUSCH EPRE to DM-RS EPRE in dB by the number of CDM groups without data
# (TS 38.214 Table 6.2.2-1). The pinned receiver assumes this amplitude for the reference symbols.
EPRE_RATIO_DB = {1: 0.0, 2: -3.0, 3: -4.77}
# Configuration type 1, single-symbol DM-RS (TS 38.211 Table 6.4.1.1.3-1): CDM group offset and
# frequency weights per port. The time weight is +1 for every port of a single-symbol DM-RS.
TYPE1_PORTS = {0: (0, (1, 1)), 1: (0, (1, -1)), 2: (1, (1, 1)), 3: (1, (1, -1))}
# DM-RS positions for single-symbol DM-RS without frequency hopping (TS 38.211 Table 6.4.1.1.3-3),
# indexed by duration then by dmrs-AdditionalPosition. "0" stands for l_0.
_A = {4: ((0,),) * 4, 5: ((0,),) * 4, 6: ((0,),) * 4, 7: ((0,),) * 4,
      8: ((0,), (0, 7), (0, 7), (0, 7)), 9: ((0,), (0, 7), (0, 7), (0, 7)),
      10: ((0,), (0, 9), (0, 6, 9), (0, 6, 9)), 11: ((0,), (0, 9), (0, 6, 9), (0, 6, 9)),
      12: ((0,), (0, 9), (0, 6, 9), (0, 5, 8, 11)), 13: ((0,), (0, 11), (0, 7, 11), (0, 5, 8, 11)),
      14: ((0,), (0, 11), (0, 7, 11), (0, 5, 8, 11))}
_B = {1: ((0,),) * 4, 2: ((0,),) * 4, 3: ((0,),) * 4, 4: ((0,),) * 4,
      5: ((0,), (0, 4), (0, 4), (0, 4)), 6: ((0,), (0, 4), (0, 4), (0, 4)), 7: ((0,), (0, 4), (0, 4), (0, 4)),
      8: ((0,), (0, 6), (0, 3, 6), (0, 3, 6)), 9: ((0,), (0, 6), (0, 3, 6), (0, 3, 6)),
      10: ((0,), (0, 8), (0, 4, 8), (0, 3, 6, 9)), 11: ((0,), (0, 8), (0, 4, 8), (0, 3, 6, 9)),
      12: ((0,), (0, 10), (0, 5, 10), (0, 3, 6, 9)), 13: ((0,), (0, 10), (0, 5, 10), (0, 3, 6, 9)),
      14: ((0,), (0, 10), (0, 5, 10), (0, 3, 6, 9))}


def gold_sequence(c_init, length):
    """Return c(0) ... c(length - 1) of the length-31 Gold sequence of TS 38.211 5.2.1."""
    if type(c_init) is not int or not 0 <= c_init < 2**31 or type(length) is not int or length < 0:
        raise SourceError("the sequence needs an initial state below 2**31 and a length")
    x1, x2 = 1, c_init  # bit n of each integer is x(n) of the current window
    for _ in range(GOLD_OFFSET):
        x1 = (x1 >> 1) | (((x1 ^ (x1 >> 3)) & 1) << 30)
        x2 = (x2 >> 1) | (((x2 ^ (x2 >> 1) ^ (x2 >> 2) ^ (x2 >> 3)) & 1) << 30)
    sequence = []
    for _ in range(length):
        sequence.append((x1 ^ x2) & 1)
        x1 = (x1 >> 1) | (((x1 ^ (x1 >> 3)) & 1) << 30)
        x2 = (x2 >> 1) | (((x2 ^ (x2 >> 1) ^ (x2 >> 2) ^ (x2 >> 3)) & 1) << 30)
    return sequence


def sequence_seed(slot, symbol, scrambling_id, n_scid):
    """c_init of TS 38.211 6.4.1.1.1.1 for slot n_s,f, symbol l, N_ID^n_SCID and n_SCID."""
    return ((SYMBOLS_PER_SLOT * slot + symbol + 1) * (2 * scrambling_id + 1) * 2**17
            + 2 * scrambling_id + n_scid) % 2**31


def amplitude(cdm_groups_without_data):
    """DM-RS amplitude relative to unit-power data symbols (beta of TS 38.211 6.4.1.1.3)."""
    if cdm_groups_without_data not in EPRE_RATIO_DB:
        raise SourceError("the number of CDM groups without data must be 1, 2 or 3")
    return 10.0 ** (-EPRE_RATIO_DB[cdm_groups_without_data] / 20.0)


def symbol_positions(mapping_type, type_a_position, additional_position, symbol_start, symbol_count):
    """Slot symbol indices that carry single-symbol DM-RS for one PUSCH time allocation."""
    if mapping_type == "A":
        if symbol_start != 0 or type_a_position not in (2, 3):
            raise SourceError("mapping type A starts at symbol 0 with dmrs-TypeA-Position 2 or 3")
        table, first, origin = _A, type_a_position, 0
    elif mapping_type == "B":
        table, first, origin = _B, 0, symbol_start
    else:
        raise SourceError("the PUSCH mapping type must be A or B")
    if additional_position not in (0, 1, 2, 3) or symbol_count not in table:
        raise SourceError(f"no DM-RS position is defined for mapping type {mapping_type}, duration {symbol_count} "
                          f"and additional position {additional_position}")
    if mapping_type == "A" and additional_position == 3 and type_a_position != 2:
        raise SourceError("additional position 3 requires dmrs-TypeA-Position 2")
    return [origin + (first if position == 0 else position) for position in table[symbol_count][additional_position]]


def symbol_times(scs_khz, slot):
    """Start of the cyclic prefix and of the useful part of each symbol, in T_c from the slot start.

    The first symbol of every half subframe carries the longer prefix (TS 38.211 5.3.1), so the
    symbol times are not uniform. A channel must be evaluated at these times, not on a regular grid.
    """
    if scs_khz not in (15, 30, 60, 120):
        raise SourceError("the subcarrier spacing must be 15, 30, 60 or 120 kHz")
    mu = (15, 30, 60, 120).index(scs_khz)
    useful = (2048 * KAPPA) >> mu
    first = SYMBOLS_PER_SLOT * (slot % (1 << mu))  # index of the slot's first symbol in its subframe
    times, cursor = [], 0
    for symbol in range(SYMBOLS_PER_SLOT):
        prefix = (144 * KAPPA) >> mu
        if (first + symbol) % (7 << mu) == 0:
            prefix += 16 * KAPPA
        times.append({"symbol": symbol, "cp_start_tc": cursor, "useful_start_tc": cursor + prefix,
                      "useful_tc": useful})
        cursor += prefix + useful
    return times


def build(parameters):
    """Return the sensing-source record of one PUSCH DM-RS allocation.

    parameters: scs_khz, sfn, slot, grid_size_rb, crb_start, crb_count, config_type, symbols, ports,
    scrambling_id, n_scid, cdm_groups_without_data. Coordinates are carrier coordinates:
    subcarrier k = 12 * CRB + resource element, with CRB 0 at subcarrier 0 of the grid.
    """
    needed = ("scs_khz", "sfn", "slot", "grid_size_rb", "crb_start", "crb_count", "config_type", "symbols",
              "ports", "scrambling_id", "n_scid", "cdm_groups_without_data")
    if not isinstance(parameters, dict) or set(parameters) != set(needed):
        raise SourceError(f"{NAME} requires exactly {', '.join(needed)}")
    p = parameters
    if p["config_type"] != 1:
        raise SourceError("only DM-RS configuration type 1 is implemented")
    symbols, ports = p["symbols"], p["ports"]
    if not isinstance(symbols, list) or not symbols or any(type(s) is not int for s in symbols):
        raise SourceError("DM-RS symbols must be a nonempty list of integers")
    if len(set(symbols)) != len(symbols):
        raise SourceError("duplicate pilot: a DM-RS symbol is listed twice")
    if symbols != sorted(symbols) or symbols[0] < 0 or symbols[-1] >= SYMBOLS_PER_SLOT:
        raise SourceError("DM-RS symbols must increase and lie inside the slot")
    if not isinstance(ports, list) or not ports or any(type(port) is not int for port in ports):
        raise SourceError("DM-RS ports must be a nonempty list of integers")
    if len(set(ports)) != len(ports):
        raise SourceError("duplicate pilot: a DM-RS port is listed twice")
    for port in ports:
        if port not in TYPE1_PORTS:
            raise SourceError(f"invalid port {port}: single-symbol DM-RS type 1 has ports 0 to 3")
    crb_start, crb_count, grid = p["crb_start"], p["crb_count"], p["grid_size_rb"]
    if any(type(value) is not int for value in (crb_start, crb_count, grid)) \
            or crb_start < 0 or crb_count < 1 or crb_start + crb_count > grid:
        raise SourceError("the allocated resource blocks leave the carrier grid")
    scale = amplitude(p["cdm_groups_without_data"]) / sqrt(2.0)
    deltas = {TYPE1_PORTS[port][0] for port in ports}
    if len(deltas) != 1:
        raise SourceError("ports of different CDM groups need one source record per group")
    delta = deltas.pop()
    if delta >= p["cdm_groups_without_data"]:
        raise SourceError("a DM-RS port lies in a CDM group that the grant leaves to data")

    # Ports of one CDM group share their resource elements and differ by their weights, so the
    # mask lists each pilot once and the reference keeps one row per port.
    index, values = [], [[] for _ in ports]
    last = 6 * (crb_start + crb_count)  # r(0) ... r(last - 1) reach the end of the allocation
    for symbol in symbols:
        bits = gold_sequence(sequence_seed(p["slot"], symbol, p["scrambling_id"], p["n_scid"]), 2 * last)
        for element in range(6 * crb_start, last):  # r(element) sits on subcarrier 2 * element + delta
            index.append([symbol, 2 * element + delta])
            for row, port in zip(values, ports):
                weight = TYPE1_PORTS[port][1][element % 2] * scale
                row.append([weight * (1 - 2 * bits[2 * element]), weight * (1 - 2 * bits[2 * element + 1])])
    times = symbol_times(p["scs_khz"], p["slot"])
    return validate({
        "schema": SCHEMA, "source": NAME, "channel": "PUSCH", "link_direction": "uplink",
        "reference_signal": "DM-RS configuration type 1, single symbol, CP-OFDM (TS 38.211 6.4.1.1)",
        "ports": list(ports),
        # The receiver's PUSCH processor executes this signal. A run records which backend did.
        "processor": {"name": "pusch", "backend": None},
        "slot": {"scs_khz": p["scs_khz"], "sfn": p["sfn"], "slot": p["slot"], "symbols": list(symbols),
                 "tc_per_second": TC_PER_SECOND,
                 "symbol_useful_start_tc": [times[symbol]["useful_start_tc"] for symbol in symbols]},
        "grid": {"coordinates": "carrier", "symbols": SYMBOLS_PER_SLOT, "subcarriers": 12 * grid},
        "mask": {"axes": ["pilot", ["symbol", "subcarrier"]], "index": index},
        "reference": {"axes": ["port", "pilot", ["real", "imaginary"]],
                      "amplitude": amplitude(p["cdm_groups_without_data"]), "values": values},
    })
