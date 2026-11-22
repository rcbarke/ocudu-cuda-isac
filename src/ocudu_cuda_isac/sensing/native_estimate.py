"""The receiver's own channel estimate: the reference of the stage O3 (Sprint 5.2, selector native_estimate).

The resident CUDA kernels of the pinned receiver leave one estimate per PDU on the device, after
three steps and before time interpolation. This module states what those steps compute, on the
CPU and without a device, so that an exported estimate can be compared with an estimator that is
matched to the receiver's and fed with the observer's own pilot least squares, the stage O2.

  1. Pilot least squares, h = y conj(p) / |p|^2, in single precision, stored as FP16. The stage
     O2 holds the same quantity in complex64, so its value rounded to FP16 is the model of this
     step.
  2. Frequency interpolation inside each resource block, in FP16 arithmetic: a pilot subcarrier
     keeps its pilot, a subcarrier between two pilots is half of each, and the last subcarrier of
     a block keeps the last pilot.
  3. Frequency smoothing over the pilots of the whole allocation, in single precision: a FIR
     filter of 15 raised-cosine taps, where a tap beyond the allocation is left out and the sum
     is divided by the sum of the taps that were used. The twelve subcarriers of each block are
     then written back by the rule of step 2, in single precision, and stored as FP16.

Step 3 filters the buffer in place: one CUDA block per resource block and DM-RS symbol reads the
pilots of up to two neighbouring blocks on each side and writes its own. Whether a block reads a
neighbour before or after that neighbour was written is not fixed by the kernel. The matched
estimator is the order in which every block reads the unfiltered pilots ("jacobi"). The other
orders are modelled here so that the parity tolerance can be declared for what the device may do.

Payload layout ocudu_o3_fp16: [dmrs_symbol][prb][rx_port][subcarrier_in_prb], last axis fastest,
complex FP16 pairs {real, imaginary}, little-endian. One validity bit per element, in payload
order, bit (n mod 8) of byte (n // 8): cleared only where a part is not finite.

Standard library only. This module imports nothing of a host, a simulator or an episode: it
belongs to the sensing side.
"""

from math import floor, isfinite, log2
import struct

DESCRIPTOR_SCHEMA = "obs-descriptor/v1"
STAGE = "O3"
PRODUCER = "receiver"
STEPS = ["pilot_ls", "frequency_interpolation", "smoothing"]
LAYOUT = "ocudu_o3_fp16"
DTYPE = "<cfp16"
AXES = ["dmrs_symbol", "prb", "rx_port", "subcarrier_in_prb"]
RULE = "pilot LS, frequency interpolation and frequency smoothing of the resident PUSCH kernels, per DM-RS symbol"
PRECISION = "complex FP16, the bytes of the device buffer"
SUBCARRIERS_PER_PRB = 12
PILOTS_PER_PRB = 6  # DM-RS configuration type 1, one CDM group: every second subcarrier
ELEMENT_BYTES = 4
# d_rc_filter of lib/phy/cuda/src/pusch_e2e.cu at the pinned commit, as the single-precision values it holds.
RC_TAPS = (-0.0391585909, -0.0287990728, 0.0, 0.0445245383, 0.0970724536, 0.1467045930, 0.1821531663, 0.1950058248,
           0.1821531663, 0.1467045930, 0.0970724536, 0.0445245383, 0.0, -0.0287990728, -0.0391585909)
ORDERS = ("jacobi", "ascending", "descending")
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
             "completion", "producer", "copy", "ring_sequence", "nonzero", "service_us"),
}
RECORD_FIELDS = ("schema", "run_id") + tuple(GROUPS)
PRODUCER_FIELDS = ("launch", "device", "backend", "resident_mode", "rnti", "sfn", "slot", "numerology", "start_prb",
                   "prb_count", "contiguous", "dmrs_symbols", "rx_ports", "layers", "dmrs_type", "scale", "steps",
                   "time_interpolation", "bytes")


class NativeEstimateError(ValueError):
    """A payload, a record or an estimate of the stage O3 violates the documented interface."""


# ---------------------------------------------------------------- precision
def single(value):
    """A real value rounded to single precision, as a float."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def half(value):
    """A real value rounded to FP16, round to nearest and ties to even, as a float. Beyond the range: infinite."""
    try:
        return struct.unpack("<e", struct.pack("<e", value))[0]
    except OverflowError:
        return float("inf") if value > 0 else float("-inf")


def half_bits(value):
    """The 16 bits of the FP16 number nearest to a value."""
    try:
        return struct.unpack("<H", struct.pack("<e", value))[0]
    except OverflowError:
        return 0x7C00 if value > 0 else 0xFC00


def chalf(value):
    return complex(half(value.real), half(value.imag))


def ulp(scale):
    """The spacing of FP16 numbers at a magnitude. Below the normal range it is that of the subnormal numbers."""
    return 2.0 ** (max(floor(log2(scale)), -14) - 10) if scale > 0 else 2.0 ** -24


_TAPS = tuple(single(tap) for tap in RC_TAPS)


# ---------------------------------------------------------------- layout
def elements(shape):
    return shape[0] * shape[1] * shape[2] * shape[3]


def element(shape, dmrs_symbol, prb, rx_port, subcarrier_in_prb):
    """Position of one element in a payload of the given shape [dmrs_symbols, prbs, rx_ports, 12]."""
    index = (dmrs_symbol, prb, rx_port, subcarrier_in_prb)
    if len(shape) != 4 or shape[3] != SUBCARRIERS_PER_PRB or any(not 0 <= i < n for i, n in zip(index, shape)):
        raise NativeEstimateError(f"the index {list(index)} is outside a payload of the shape {list(shape)}")
    return ((dmrs_symbol * shape[1] + prb) * shape[2] + rx_port) * SUBCARRIERS_PER_PRB + subcarrier_in_prb


def coordinate(shape, index):
    """What one element is: the DM-RS symbol, the block, the receive port and the subcarrier of the allocation."""
    if len(shape) != 4 or shape[3] != SUBCARRIERS_PER_PRB or not 0 <= index < elements(shape):
        raise NativeEstimateError(f"the element {index} is outside a payload of the shape {list(shape)}")
    rest, subcarrier_in_prb = divmod(index, SUBCARRIERS_PER_PRB)
    rest, rx_port = divmod(rest, shape[2])
    dmrs_symbol, prb = divmod(rest, shape[1])
    return {"dmrs_symbol": dmrs_symbol, "prb": prb, "rx_port": rx_port, "subcarrier_in_prb": subcarrier_in_prb,
            "subcarrier": prb * SUBCARRIERS_PER_PRB + subcarrier_in_prb}


def payload_values(data):
    """The complex values of a payload, each widened from FP16 without change."""
    if len(data) % ELEMENT_BYTES:
        raise NativeEstimateError(f"a payload of {len(data)} bytes holds no whole number of complex FP16 values")
    parts = struct.unpack(f"<{len(data) // 2}e", data)
    return [complex(parts[2 * n], parts[2 * n + 1]) for n in range(len(parts) // 2)]


def payload_bytes(values):
    """The payload of a list of complex values: each part rounded to FP16."""
    return b"".join(struct.pack("<2H", half_bits(value.real), half_bits(value.imag)) for value in values)


def scan(data):
    """The validity bits of a payload and its counts: (mask, valid, holes, nonzero)."""
    words = struct.unpack(f"<{len(data) // 2}H", data)
    count = len(words) // 2
    mask = bytearray((count + 7) // 8)
    valid = nonzero = 0
    for n in range(count):
        real, imag = words[2 * n], words[2 * n + 1]
        if real & 0x7C00 != 0x7C00 and imag & 0x7C00 != 0x7C00:
            mask[n // 8] |= 1 << (n % 8)
            valid += 1
            nonzero += ((real | imag) & 0x7FFF) != 0
    return bytes(mask), valid, count - valid, nonzero


# ---------------------------------------------------------------- matched estimator
def interpolate(pilots):
    """Step 2 for one resource block: six FP16 pilots to twelve subcarriers, in FP16 arithmetic."""
    out = []
    for subcarrier in range(SUBCARRIERS_PER_PRB):
        left = pilots[subcarrier // 2]
        if subcarrier % 2 == 0 or subcarrier == SUBCARRIERS_PER_PRB - 1:
            out.append(left)
        else:
            right = pilots[subcarrier // 2 + 1]
            out.append(complex(half(half(left.real * 0.5) + half(right.real * 0.5)),
                               half(half(left.imag * 0.5) + half(right.imag * 0.5))))
    return out


def _filter(buffer, pilot, fused):
    """The filtered value of one pilot of a row, in single precision, before it is stored."""
    real = imag = total = 0.0
    for tap, weight in enumerate(_TAPS):
        source = pilot + tap - 7
        if not 0 <= source < len(buffer):
            continue
        value = buffer[source]
        if fused:  # one rounding per term, as a fused multiply-add gives
            real, imag = single(real + weight * value.real), single(imag + weight * value.imag)
        else:
            real, imag = single(real + single(weight * value.real)), single(imag + single(weight * value.imag))
        total = single(total + weight)
    if total > 1e-6:
        inverse = single(1.0 / total)
        real, imag = single(real * inverse), single(imag * inverse)
    return complex(real, imag)


def _write_back(smoothed):
    """The twelve stored subcarriers of one block from its six filtered pilots."""
    out = []
    for subcarrier in range(SUBCARRIERS_PER_PRB):
        left = smoothed[subcarrier // 2]
        if subcarrier % 2 == 0 or subcarrier == SUBCARRIERS_PER_PRB - 1:
            out.append(chalf(left))
        else:
            right = smoothed[subcarrier // 2 + 1]
            out.append(complex(half(single(0.5 * left.real + 0.5 * right.real)),
                               half(single(0.5 * left.imag + 0.5 * right.imag))))
    return out


def smooth(pilots, order="jacobi", fused=True, waves=None):
    """Step 3 for one receive port and one DM-RS symbol. Returns [prb][12] stored values.

    pilots are the FP16 pilots of the allocation in subcarrier order, six per block. order names
    which pilots a block reads of its neighbours: "jacobi" the unfiltered ones, "ascending" and
    "descending" the filtered ones of every block that ran before it in that order. waves, a list
    of lists of block numbers, runs the blocks of one list at once and the lists one after another.
    """
    if len(pilots) % PILOTS_PER_PRB or not pilots:
        raise NativeEstimateError(f"{len(pilots)} pilots are no whole number of resource blocks")
    blocks = len(pilots) // PILOTS_PER_PRB
    if waves is None:
        if order not in ORDERS:
            raise NativeEstimateError(f"unknown order {order!r}: " + ", ".join(ORDERS))
        waves = [list(range(blocks))] if order == "jacobi" else \
            [[block] for block in (range(blocks) if order == "ascending" else reversed(range(blocks)))]
    elif sorted(block for wave in waves for block in wave) != list(range(blocks)):
        raise NativeEstimateError("the waves do not name every resource block once")
    buffer, out = list(pilots), [None] * blocks
    for wave in waves:
        done = {block: [_filter(buffer, block * PILOTS_PER_PRB + pilot, fused) for pilot in range(PILOTS_PER_PRB)]
                for block in wave}
        for block, smoothed in done.items():
            out[block] = _write_back(smoothed)
            for pilot in range(PILOTS_PER_PRB):
                buffer[block * PILOTS_PER_PRB + pilot] = out[block][2 * pilot]
    return out


def matched(o2_values, o2_shape, smoothing=True, order="jacobi", fused=True):
    """The estimate that the matched estimator makes of a payload of the stage O2.

    o2_values is the comb payload [rx_plane][reference_symbol][pilot] of one allocation without a
    hole: six pilots per resource block on every reference symbol. The result is a list of complex
    values in the layout ocudu_o3_fp16, with the shape [reference_symbols, prbs, rx_planes, 12].
    A single resource block is not smoothed by the receiver, and smoothing=False models that.
    """
    planes, symbols, pilots = o2_shape
    if pilots % PILOTS_PER_PRB or len(o2_values) != planes * symbols * pilots:
        raise NativeEstimateError(f"a comb payload of the shape {list(o2_shape)} is not six pilots per resource block "
                                  "on every reference symbol")
    blocks = pilots // PILOTS_PER_PRB
    shape = [symbols, blocks, planes, SUBCARRIERS_PER_PRB]
    out = [0j] * elements(shape)
    for plane in range(planes):
        for symbol in range(symbols):
            base = (plane * symbols + symbol) * pilots
            row = [chalf(value) for value in o2_values[base:base + pilots]]
            if smoothing and blocks > 1:
                stored = smooth(row, order, fused)
            else:
                stored = [interpolate(row[block * PILOTS_PER_PRB:(block + 1) * PILOTS_PER_PRB]) for block in range(blocks)]
            for block in range(blocks):
                start = element(shape, symbol, block, plane, 0)
                out[start:start + SUBCARRIERS_PER_PRB] = stored[block]
    return out, shape


# ---------------------------------------------------------------- parity
def deviation(value, reference):
    """How far a value is from its reference, in FP16 spacings at the magnitude of the reference."""
    scale = max(abs(reference.real), abs(reference.imag))
    return max(abs(value.real - reference.real), abs(value.imag - reference.imag)) / ulp(scale)


def parity(values, reference, shape, support="pilot"):
    """Compare an estimate with its reference. support "pilot" takes the pilot subcarriers, "all" every element.

    Returns the counts the acceptance names: the numerator sum |value - reference|^2, the
    denominator sum |reference|^2 and the sample count, with the NMSE, the largest absolute and
    relative error, the largest deviation in FP16 spacings and the element it lies at.
    """
    if len(values) != len(reference) or len(values) != elements(shape):
        raise NativeEstimateError("an estimate and its reference hold another number of elements than their shape")
    if support not in ("pilot", "all"):
        raise NativeEstimateError('support is "pilot" or "all"')
    numerator = denominator = largest = relative = spacing = 0.0
    samples = nonzero_reference = zero_values = 0
    worst = None
    for index, (value, wanted) in enumerate(zip(values, reference)):
        if support == "pilot" and (index % SUBCARRIERS_PER_PRB) % 2:
            continue
        samples += 1
        error = abs(value - wanted)
        numerator += error * error
        denominator += abs(wanted) ** 2
        nonzero_reference += wanted != 0
        zero_values += value == 0
        largest = max(largest, error)
        if wanted != 0:
            relative = max(relative, error / abs(wanted))
        steps = deviation(value, wanted)
        if steps > spacing or worst is None:
            spacing, worst = max(spacing, steps), index
    place = None
    if worst is not None:
        place = dict(coordinate(shape, worst), element=worst, value=[values[worst].real, values[worst].imag],
                     reference=[reference[worst].real, reference[worst].imag])
    return {"support": support, "samples": samples, "numerator": numerator, "denominator": denominator,
            "nmse": numerator / denominator if denominator else (0.0 if numerator == 0 else float("inf")),
            "max_abs": largest, "max_rel": relative, "max_fp16_ulps": spacing, "worst": place,
            "nonzero_reference": nonzero_reference, "zero_values": zero_values}


def beyond(values, reference, limit, support="pilot"):
    """How many compared elements deviate from their reference by more than limit FP16 spacings."""
    return sum(1 for index, (value, wanted) in enumerate(zip(values, reference))
               if (support == "all" or not (index % SUBCARRIERS_PER_PRB) % 2) and deviation(value, wanted) > limit)


def within(result, tolerance):
    """Whether a parity result meets a declared tolerance: abs_ulps, relative and nmse. A null bound is not applied.

    A reference that is not zero cannot be met by an estimate of zeros: that is the stub of a
    host, not an estimate. An estimate of zeros meets a reference of zeros.
    """
    if result["nonzero_reference"] and result["zero_values"] == result["samples"]:
        return False
    checks = (("abs_ulps", "max_fp16_ulps"), ("relative", "max_rel"), ("nmse", "nmse"))
    return all(tolerance.get(bound) is None or result[measure] <= tolerance[bound] for bound, measure in checks)


# ---------------------------------------------------------------- record
def record_violations(record):
    """What is wrong with one obs-descriptor/v1 record of the stage O3 on its own. Empty: nothing."""
    if not isinstance(record, dict) or set(record) != set(RECORD_FIELDS):
        return ["the record does not hold exactly the groups of an obs-descriptor/v1 record"]
    found = []
    for group, fields in GROUPS.items():
        if not isinstance(record[group], dict) or set(record[group]) != set(fields):
            return [f"the group {group} does not hold exactly its fields"]
    what, shape, allocation = record["what"], record["shape"], record["allocation"]
    validity, payload, host = record["validity"], record["payload"], record["host"]
    if record["schema"] != DESCRIPTOR_SCHEMA:
        found.append("another schema")
    if (what["stage"], what["producer"], what["layout"], what["dtype"]) != (STAGE, PRODUCER, LAYOUT, DTYPE) \
            or what["rule"] != RULE or what["precision"] != PRECISION:
        found.append(f"the record is not the stage {STAGE} of the producer {PRODUCER} in the layout {LAYOUT}")
    steps = what["producer_steps"]
    if not isinstance(steps, list) or steps[:2] != STEPS[:2] or steps not in (STEPS, STEPS[:2]):
        found.append("the producer steps are pilot_ls and frequency_interpolation, with smoothing for more than one block")
    sizes = shape["shape"]
    if shape["rank"] != 4 or shape["axes"] != AXES or not isinstance(sizes, list) or len(sizes) != 4 \
            or any(type(size) is not int or size < 1 for size in sizes) or sizes[3] != SUBCARRIERS_PER_PRB:
        return found + ["the shape is not [dmrs_symbol, prb, rx_port, 12]"]
    count = elements(sizes)
    if shape["strides"] != [sizes[1] * sizes[2] * 12, sizes[2] * 12, 12, 1]:
        found.append("the strides are not those of a contiguous payload with the last axis fastest")
    if payload["bytes"] != count * ELEMENT_BYTES or validity["mask_bytes"] != (count + 7) // 8 \
            or payload["digest_kind"] != "sha256":
        found.append("the payload or the validity bits have another size than the shape gives")
    if type(validity["valid"]) is not int or validity["valid"] + validity["holes"] != count \
            or not 0 <= host["nonzero"] <= validity["valid"]:
        found.append("valid and holes do not add up to the elements of the shape")
    if allocation["subcarrier_step"] != 1 or allocation["subcarrier_count"] != sizes[1] * 12 \
            or allocation["first_subcarrier"] % 12 or allocation["rx_count"] != sizes[2] \
            or allocation["layer_count"] != 1 or len(allocation["dmrs_symbols"]) != sizes[0] \
            or len(allocation["antenna_map"]) != sizes[2] or allocation["coordinates_sha256"] is not None:
        found.append("the allocation is not every subcarrier of whole resource blocks on the DM-RS symbols of the shape")
    if (steps == STEPS) != (sizes[1] > 1):
        found.append("the receiver smooths an allocation of more than one resource block and no other")
    producer = host["producer"]
    if not isinstance(producer, dict) or set(producer) != set(PRODUCER_FIELDS):
        return found + ["host.producer does not hold exactly the fields of the trace of the producer"]
    if (producer["prb_count"], producer["rx_ports"], producer["layers"], producer["bytes"]) \
            != (sizes[1], sizes[2], 1, count * ELEMENT_BYTES) or producer["dmrs_symbols"] != allocation["dmrs_symbols"] \
            or producer["start_prb"] * 12 != allocation["first_subcarrier"] or producer["steps"] != steps \
            or producer["scale"] != what["scale"] or producer["backend"] != "cuda" or not producer["contiguous"] \
            or producer["time_interpolation"] is not False or type(producer["launch"]) is not int or producer["launch"] < 1:
        found.append("the trace of the producer names another estimate than the record")
    copy = host["copy"]
    if host["completion"] != "cuda_event" or not isinstance(copy, dict) or set(copy) != {"stream", "landing", "polls"} \
            or copy["stream"] not in ("producer", "side") or copy["landing"] not in ("slot", "staging") \
            or type(copy["polls"]) is not int or copy["polls"] < 1:
        found.append("the record does not name the completion event of an ordered copy")
    if not isinstance(what["scale"], (int, float)) or not isfinite(what["scale"]) or what["scale"] <= 0:
        found.append("the scale is the reference amplitude of the producer, a positive number")
    return found
