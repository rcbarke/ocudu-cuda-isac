"""A matched target-absent and target-present channel episode on one grant schedule (Sprint 2.2).

A channel-episode/v1 entry binds two channel realizations to the uplink grant fixture of
schedule.py: case "a" is the scene without the target, case "b" the same scene with it. Both share
the grants, hence the transmit grids, and one noise realization. The entry holds

  a_h.npy, b_h.npy   H[opportunity, rx_port, tx_port, symbol, subcarrier], complex64, the physical
                     channel frequency response with absolute delay and amplitude
  w.npy              W[opportunity, rx_port, symbol, subcarrier], complex64, in the units of H
  manifest.json      axes, times, grants, geometry, array, scene, solver, noise, calibration,
                     truth of the present case, and the SHA-256 of every file

Axes are the grant's: subcarrier k is the carrier-coordinate subcarrier of the grant record, at
Grant.subcarrier_offset_hz(k) from the carrier, and symbol l of an opportunity lies at its slot
start plus the useful start that Grant.symbol_times() gives. Receive plane i is antenna i. The
receiver observes Y = sum over tx of H * X + W on the grant's resource elements.

Standard library only: a second host verifies an entry, reads any coefficient and repeats the
analytic check without NumPy, Sionna or a GPU. Generation is in rt_channel.py.
"""

import argparse
from array import array
import cmath
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import tempfile

from . import grant as grants
from . import schedule as schedules
from .grant import GrantError
from .schedule import ScheduleError
from ..replay import ContractError, load_json
from ..sensing.sources import pusch_dmrs

SCHEMA = "channel-episode/v1"
ANALYTIC_MODEL = "free-space-point-target/v1"
SPEED_OF_LIGHT_M_S = 299_792_458.0
TC_PER_SECOND = pusch_dmrs.TC_PER_SECOND
H_AXES = ["opportunity", "rx_port", "tx_port", "symbol", "subcarrier"]
W_AXES = ["opportunity", "rx_port", "symbol", "subcarrier"]
CASES = {"a": False, "b": True}
FILES = {"a": "a_h.npy", "b": "b_h.npy", "w": "w.npy"}
DESCRIPTION = ("geometry", "array", "scene", "target", "solver", "time_evolution", "noise", "calibration",
               "software", "provenance")
OPTIONAL = ("analytic_model", "labels")
# A complex64 component keeps 24 significant bits: a stored value lies within 2**-24 of its
# magnitude of the value written, and a difference of two stored values within twice that.
C64_RELATIVE = 2.0 ** -24
CHUNK = 1 << 22


class EpisodeError(ValueError):
    """An episode entry or its inputs violate the documented interface."""


def c64_bound(*magnitudes):
    """Largest error of a sum or difference of stored complex64 values with these magnitudes.

    The bound scales with the stored terms, not with the result: an echo far below the direct
    path is recovered from b - a with an error set by the direct path's magnitude.
    """
    return 2.0 * math.sqrt(2.0) * C64_RELATIVE * sum(abs(value) for value in magnitudes)


def _vector(value, name):
    if not isinstance(value, list) or len(value) != 3 \
            or any(type(part) not in (int, float) or not math.isfinite(part) for part in value):
        raise EpisodeError(f"{name} must be three finite numbers")
    return [float(part) for part in value]


def target_state(motion, time_s):
    """Position and velocity of the target at an episode time, from the motion record of the manifest."""
    if not isinstance(motion, dict) or motion.get("kind") not in ("constant_velocity", "waypoints"):
        raise EpisodeError("target motion must be constant_velocity or waypoints")
    if motion["kind"] == "constant_velocity":
        position = _vector(motion.get("position_m"), "position_m")
        velocity = _vector(motion.get("velocity_m_s"), "velocity_m_s")
        return [p + v * time_s for p, v in zip(position, velocity)], velocity
    points, speed, start = motion.get("points"), motion.get("speed_m_s"), motion.get("start_s")
    if not isinstance(points, list) or len(points) < 2 or type(speed) not in (int, float) or not speed > 0 \
            or type(start) not in (int, float) or not math.isfinite(speed * (start + time_s)) or start + time_s < 0:
        raise EpisodeError("a waypoint motion needs two points, a positive speed and a nonnegative walk time")
    points = [_vector(point, "waypoint") for point in points]
    travelled = speed * (start + time_s)
    for first, second in zip(points, points[1:]):
        length = math.dist(first, second)
        if length <= 0:
            raise EpisodeError("two successive waypoints coincide")
        if travelled <= length:
            unit = [(b - a) / length for a, b in zip(first, second)]
            return [a + u * travelled for a, u in zip(first, unit)], [u * speed for u in unit]
        travelled -= length
    raise EpisodeError("the episode runs past the end of its trajectory")


def bistatic(geometry, position, velocity, carrier_hz):
    """Delay and Doppler shift of the first-order return tx -> target -> rx, from geometry alone."""
    tx, rx = geometry["ue_tx_m"], geometry["gnb_rx_m"]
    d1, d2 = math.dist(tx, position), math.dist(rx, position)
    rate = sum(v * ((p - t) / d1 + (p - r) / d2) for v, p, t, r in zip(velocity, position, tx, rx))
    direct = math.dist(tx, rx)
    return {"path_m": d1 + d2, "delay_ns": 1e9 * (d1 + d2) / SPEED_OF_LIGHT_M_S,
            "excess_delay_ns": 1e9 * (d1 + d2 - direct) / SPEED_OF_LIGHT_M_S,
            "doppler_hz": -rate * carrier_hz / SPEED_OF_LIGHT_M_S, "leg_tx_m": d1, "leg_rx_m": d2}


def free_space_paths(geometry, target, time_s, carrier_hz, present):
    """The analytic paths of the free-space point-target model at one instant.

    Returns (amplitude, delay_s, [phase per receive element]) per path: the direct path with the
    Friis amplitude, and with the target the bistatic radar return of a constant cross-section.
    Receive elements see a plane wave: element n adds 2 pi / lambda times the projection of its
    offset on the direction of arrival, as a synthetic array does.
    """
    wavelength = SPEED_OF_LIGHT_M_S / carrier_hz
    tx, rx, offsets = geometry["ue_tx_m"], geometry["gnb_rx_m"], geometry["rx_element_offsets_m"]

    def phases(source):
        distance = math.dist(rx, source)
        arrival = [(s - r) / distance for s, r in zip(source, rx)]
        return [2 * math.pi / wavelength * sum(o * a for o, a in zip(offset, arrival)) for offset in offsets]

    direct = math.dist(tx, rx)
    paths = [(wavelength / (4 * math.pi * direct), direct / SPEED_OF_LIGHT_M_S, phases(tx))]
    if present:
        position, _ = target_state(target["motion"], time_s)
        d1, d2 = math.dist(tx, position), math.dist(rx, position)
        amplitude = wavelength * math.sqrt(target["sigma_m2"]) / ((4 * math.pi) ** 1.5 * d1 * d2)
        paths.append((amplitude, (d1 + d2) / SPEED_OF_LIGHT_M_S, phases(position)))
    return paths


def path_response(paths, carrier_hz, offset_hz, rx):
    """H of analytic paths at one frequency offset and receive element: sum a exp(j(phi - 2 pi f tau))."""
    return sum(amplitude * cmath.exp(1j * (phase[rx] - 2 * math.pi * (carrier_hz + offset_hz) * delay))
               for amplitude, delay, phase in paths)


def _npy_header(shape):
    header = "{'descr': '<c8', 'fortran_order': False, 'shape': %r, }" % (tuple(shape),)
    header += " " * (-(10 + len(header) + 1) % 64) + "\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode("latin1")


def _payload(values, count, name):
    """Little-endian complex64 bytes of count values, from a bytes-like object or a flat sequence."""
    if isinstance(values, (bytes, bytearray, memoryview)):
        data = bytes(values)
    else:
        try:
            data = struct.pack(f"<{2 * len(values)}f", *[part for value in values
                                                         for part in (complex(value).real, complex(value).imag)])
        except (OverflowError, struct.error, TypeError):
            raise EpisodeError(f"{name} holds a value that is not a finite complex64") from None
    if len(data) != 8 * count:
        raise EpisodeError(f"{name} holds {len(data) // 8} complex values, the grant schedule needs {count}")
    return data


def _scan(data):
    """(all finite, any nonzero) of little-endian float32 bytes."""
    finite, nonzero = True, False
    for start in range(0, len(data), CHUNK):
        block = array("f")
        block.frombytes(data[start:start + CHUNK])
        if sys.byteorder != "little":
            block.byteswap()
        finite = finite and math.isfinite(sum(block))
        nonzero = nonzero or any(block)
    return finite, nonzero


def shapes(schedule):
    """Shapes of H and W for a schedule: every opportunity, receive port and carrier-grid element."""
    base = schedule.base
    count = len(schedule.opportunities)
    return ([count, len(base.rx_ports), base.antenna_ports, grants.SYMBOLS_PER_SLOT, base.grid_subcarriers],
            [count, len(base.rx_ports), grants.SYMBOLS_PER_SLOT, base.grid_subcarriers])


def axes(schedule):
    """The axis record of an entry. The loader recomputes it from the grant and compares."""
    base = schedule.base
    h_shape, w_shape = shapes(schedule)
    return {
        "h": H_AXES, "h_shape": h_shape, "w": W_AXES, "w_shape": w_shape, "dtype": "<c8", "order": "C",
        "subcarrier": "carrier coordinates of the grant: k = 12 * CRB + resource element",
        "subcarrier_offset_hz": {"first": base.subcarrier_offset_hz(0), "step": base.scs_khz * 1000,
                                 "count": base.grid_subcarriers, "rule": "Grant.subcarrier_offset_hz(k)"},
        "carrier_hz": base.carrier_hz, "tc_per_second": TC_PER_SECOND,
        "time": "symbol l of an opportunity lies at slot_start_tc + symbol_useful_start_tc[l] of its grant, in T_c "
                "from the start of the schedule; H is the channel at that instant",
        "rx_port": "receive plane i is antenna i of the array, the order of the grant's rx_ports 0 to N-1",
        "tx_port": "antenna port of the grant, one",
        "units": {"h": "linear complex baseband amplitude gain, absolute delay and amplitude preserved",
                  "w": "additive noise in the units of H times unit-power data"},
        "application": "Y[rx, symbol, subcarrier] = sum over tx of H * X + W, on the grant's resource elements"}


def _check_description(description, schedule, truth):
    if not isinstance(description, dict) or not set(DESCRIPTION) <= set(description) \
            or set(description) - set(DESCRIPTION) - set(OPTIONAL):
        raise EpisodeError(f"an episode description requires {', '.join(DESCRIPTION)} and optionally "
                           f"{', '.join(OPTIONAL)}")
    for key in DESCRIPTION:
        if not isinstance(description[key], dict):
            raise EpisodeError(f"{key} must be an object")
    geometry, base = description["geometry"], schedule.base
    for key in ("ue_tx_m", "gnb_rx_m"):
        _vector(geometry.get(key), f"geometry.{key}")
    offsets = geometry.get("rx_element_offsets_m")
    if not isinstance(offsets, list) or len(offsets) != len(base.rx_ports):
        raise EpisodeError(f"geometry.rx_element_offsets_m needs one offset per receive port of the grant "
                           f"({len(base.rx_ports)})")
    for offset in offsets:
        _vector(offset, "a receive element offset")
    if description.get("analytic_model") not in (None, ANALYTIC_MODEL):
        raise EpisodeError(f"unknown analytic model {description.get('analytic_model')!r}")
    if description.get("analytic_model") and (description["target"].get("motion", {}).get("kind") != "constant_velocity"
                                              or type(description["target"].get("sigma_m2")) not in (int, float)):
        raise EpisodeError("the analytic model needs a constant cross-section and a constant velocity")
    noise = description["noise"]
    for key in ("snr_db", "sigma", "seed"):
        if type(noise.get(key)) not in (int, float) or not math.isfinite(noise[key]):
            raise EpisodeError(f"noise.{key} must be a finite number")
    if not isinstance(truth, list) or len(truth) != len(schedule.opportunities):
        raise EpisodeError("the present case needs one truth entry per opportunity")
    for index, entry in enumerate(truth):
        if not isinstance(entry, dict) or entry.get("opportunity") != index:
            raise EpisodeError("truth entries are objects in opportunity order")
        _vector(entry.get("position_m"), "truth position_m")
        _vector(entry.get("velocity_m_s"), "truth velocity_m_s")


def export(directory, episode_id, schedule, description, arrays, truth, measured=None, waveforms=None):
    """Write one episode entry into a new directory and return its manifest.

    arrays maps "a", "b" and "w" to complex64 values in C order, as little-endian bytes or as a
    flat sequence. truth lists the target state of the present case per opportunity. measured
    holds what the generator observed per case. waveforms binds grants to archived transmit grids
    as {grant_id: {"x_sha256": ..., "store": ...}}. Everything is checked before a file is
    written, and the directory appears only when complete.
    """
    directory = Path(directory)
    if directory.exists():
        raise EpisodeError(f"{directory} already exists. An episode entry is never overwritten.")
    if not isinstance(schedule, schedules.Schedule):
        raise EpisodeError("an episode is exported on a validated grant schedule")
    if not isinstance(episode_id, str) or not episode_id or not episode_id.isascii() \
            or any(character in episode_id for character in "/\\ \t\n"):
        raise EpisodeError("episode_id must be a nonempty ASCII identifier without separators")
    _check_description(description, schedule, truth)
    if not isinstance(arrays, dict) or set(arrays) != set(FILES):
        raise EpisodeError('arrays must hold exactly "a", "b" and "w"')
    h_shape, w_shape = shapes(schedule)
    data = {key: _payload(arrays[key], math.prod(w_shape if key == "w" else h_shape), FILES[key]) for key in FILES}
    for key in FILES:
        finite, nonzero = _scan(data[key])
        if not finite:
            raise EpisodeError(f"{FILES[key]} holds a nonfinite value")
        if not nonzero and key != "w":
            raise EpisodeError(f"{FILES[key]} is all zero: the channel has no path")
    if data["a"] == data["b"]:
        raise EpisodeError("the target-present channel equals the target-absent one")
    waveforms = dict(waveforms or {})
    if set(waveforms) - set(schedule.grants):
        raise EpisodeError("a waveform is bound to a grant that the schedule does not use")
    described = schedule.describe_grants()
    for grant_id, entry in described.items():
        entry["x"] = waveforms.get(grant_id)
    measured = measured or {}
    files, cases = {}, []
    for key in FILES:
        shape = w_shape if key == "w" else h_shape
        blob = _npy_header(shape) + data[key]
        files[FILES[key]] = {"sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob),
                             "axes": W_AXES if key == "w" else H_AXES, "shape": shape, "dtype": "<c8", "order": "C"}
        data[key] = blob
    for case, present in CASES.items():
        seed = f"{episode_id}:{case}:{files[FILES[case]]['sha256']}".encode("ascii")
        entry = {"id": case, "target_present": present, "h": FILES[case], "w": FILES["w"],
                 "realization_id": "rz-" + hashlib.sha256(seed).hexdigest()[:12],
                 "measured": measured.get(case, {})}
        if present:
            entry["truth"] = truth
        cases.append(entry)
    manifest = {
        "schema": SCHEMA, "episode_id": episode_id, "access": "offline_stimulus_and_evaluator",
        "grant": schedule.base.record(), "schedule": schedule.record(), "axes": axes(schedule),
        "opportunities": [opportunity.describe() for opportunity in schedule.opportunities],
        "grants": described,
        "mask": {"occupied": "data_index and pilot_index of each opportunity's grant record",
                 "pilots": "sensing source pusch_dmrs of each opportunity's grant",
                 "support": "H and W cover every resource element of the carrier grid"},
        "cases": cases, "files": files,
        **{key: description[key] for key in DESCRIPTION + OPTIONAL if key in description}}
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory.parent) as temporary:
        staging = Path(temporary) / "entry"
        staging.mkdir()
        for key in FILES:
            (staging / FILES[key]).write_bytes(data[key])
        (staging / "manifest.json").write_bytes((json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("ascii"))
        staging.rename(directory)
    return manifest


class Episode:
    """A verified entry. Coefficients are read from the files on demand, so any size opens quickly."""

    def __init__(self, directory, manifest, schedule, offsets):
        self.directory, self.manifest, self.schedule, self._offsets = directory, manifest, schedule, offsets
        self.episode_id = manifest["episode_id"]
        self.grant = schedule.base
        self.h_shape, self.w_shape = shapes(schedule)

    def _read(self, name, shape, index):
        flat = 0
        for position, size in zip(index, shape):
            if type(position) is not int or not 0 <= position < size:
                raise EpisodeError(f"index {list(index)} is outside {name} of shape {shape}")
            flat = flat * size + position
        with (self.directory / name).open("rb") as stream:
            stream.seek(self._offsets[name] + 8 * flat)
            return complex(*struct.unpack("<2f", stream.read(8)))

    def h(self, case, opportunity, rx, symbol, subcarrier, tx=0):
        if case not in CASES:
            raise EpisodeError('a case is "a" (target absent) or "b" (target present)')
        return self._read(FILES[case], self.h_shape, (opportunity, rx, tx, symbol, subcarrier))

    def w(self, opportunity, rx, symbol, subcarrier):
        return self._read(FILES["w"], self.w_shape, (opportunity, rx, symbol, subcarrier))

    def echo(self, opportunity, rx, symbol, subcarrier):
        """What the target adds at one coordinate: H of the present case minus H of the absent case."""
        return self.h("b", opportunity, rx, symbol, subcarrier) - self.h("a", opportunity, rx, symbol, subcarrier)

    def offset_hz(self, subcarrier):
        return self.grant.subcarrier_offset_hz(subcarrier)

    def time_tc(self, opportunity, symbol):
        return self.schedule.opportunities[opportunity].symbol_time_tc(symbol)

    def time_s(self, opportunity, symbol):
        return self.time_tc(opportunity, symbol) / TC_PER_SECOND

    def truth(self, opportunity):
        return self.manifest["cases"][1]["truth"][opportunity]

    def analytic(self, case, opportunity, rx, symbol, subcarrier):
        """H of the analytic model at one coordinate, with the exact geometry of that instant. None without a model."""
        if self.manifest.get("analytic_model") != ANALYTIC_MODEL:
            return None
        paths = free_space_paths(self.manifest["geometry"], self.manifest["target"], self.time_s(opportunity, symbol),
                                 self.grant.carrier_hz, CASES[case])
        return path_response(paths, self.grant.carrier_hz, self.offset_hz(subcarrier), rx)

    def analytic_echo(self, opportunity, rx, symbol, subcarrier):
        if self.manifest.get("analytic_model") != ANALYTIC_MODEL:
            return None
        paths = free_space_paths(self.manifest["geometry"], self.manifest["target"], self.time_s(opportunity, symbol),
                                 self.grant.carrier_hz, True)
        return path_response(paths[1:], self.grant.carrier_hz, self.offset_hz(subcarrier), rx)

    def phase_advance(self, first, second, rx, subcarrier):
        """Phase advance of the echo between two (opportunity, symbol) instants, measured and expected."""
        stored = [self.echo(opportunity, rx, symbol, subcarrier) for opportunity, symbol in (first, second)]
        if any(value == 0 for value in stored):
            raise EpisodeError("the echo is zero at a requested coordinate")
        result = {"from": list(first), "to": list(second),
                  "interval_s": self.time_s(*second) - self.time_s(*first),
                  "measured_rad": cmath.phase(stored[1] * stored[0].conjugate())}
        model = [self.analytic_echo(opportunity, rx, symbol, subcarrier) for opportunity, symbol in (first, second)]
        if None not in model:
            result["analytic_rad"] = cmath.phase(model[1] * model[0].conjugate())
            result["error_rad"] = abs(cmath.phase(stored[1] * stored[0].conjugate()
                                                  * (model[1] * model[0].conjugate()).conjugate()))
        return result

    def coefficient(self, case, opportunity, rx, symbol, subcarrier):
        """One stored coefficient with its coordinates, for a second reader."""
        value = self.h(case, opportunity, rx, symbol, subcarrier)
        entry = self.schedule.opportunities[opportunity]
        result = {"case": case, "target_present": CASES[case], "opportunity": opportunity, "rx_port": rx,
                  "symbol": symbol, "subcarrier": subcarrier, "offset_hz": self.offset_hz(subcarrier),
                  "time_tc": self.time_tc(opportunity, symbol), "time_s": self.time_s(opportunity, symbol),
                  "sfn": entry.sfn, "slot": entry.slot, "grant_id": entry.grant.grant_id,
                  "is_pilot": (symbol, subcarrier) in set(entry.grant.pilot_index),
                  "value": [value.real, value.imag], "magnitude": abs(value)}
        model = self.analytic(case, opportunity, rx, symbol, subcarrier)
        if model is not None:
            result["analytic"] = [model.real, model.imag]
            result["analytic_relative_error"] = abs(value - model) / abs(model)
        return result


def open_episode(directory):
    """Verify an entry against its manifest and against its grant, then return it.

    Every file must have the recorded size and hash and a complex64 C-order header of the shape
    that the grant schedule gives. The opportunities, the grants and the axes are recomputed from
    the embedded grant and schedule records and must equal the manifest. A changed byte, a
    transposed or missing array and an incomplete entry are refused.
    """
    directory = Path(directory)
    try:
        manifest = load_json(directory / "manifest.json")
    except (ContractError, OSError) as error:
        raise EpisodeError(f"{directory} is not a complete episode entry: {error}") from None
    needed = {"schema", "episode_id", "access", "grant", "schedule", "axes", "opportunities", "grants", "mask",
              "cases", "files", *DESCRIPTION}
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA or not needed <= set(manifest) \
            or set(manifest) - needed - set(OPTIONAL):
        raise EpisodeError(f"{directory / 'manifest.json'} is not a {SCHEMA} manifest")
    try:
        base = grants.build(manifest["grant"])
        schedule = schedules.build(manifest["schedule"], base)
    except (GrantError, ScheduleError) as error:
        raise EpisodeError(f"the embedded grant schedule is invalid: {error}") from None
    if manifest["opportunities"] != [opportunity.describe() for opportunity in schedule.opportunities]:
        raise EpisodeError("the opportunities differ from the grant schedule")
    described = schedule.describe_grants()
    stored = manifest["grants"]
    if not isinstance(stored, dict) or set(stored) != set(described) or any(
            not isinstance(stored[key], dict) or {name: value for name, value in stored[key].items() if name != "x"}
            != described[key] for key in described):
        raise EpisodeError("the grants differ from the grant schedule")
    if manifest["axes"] != json.loads(json.dumps(axes(schedule))):
        raise EpisodeError("the axes differ from the coordinates of the grant")
    h_shape, w_shape = shapes(schedule)
    files = manifest["files"]
    if not isinstance(files, dict) or set(files) != set(FILES.values()):
        raise EpisodeError("the manifest does not list exactly a_h.npy, b_h.npy and w.npy")
    offsets, digests = {}, {}
    for key, name in FILES.items():
        shape = w_shape if key == "w" else h_shape
        header = _npy_header(shape)
        try:
            data = (directory / name).read_bytes()
        except OSError as error:
            raise EpisodeError(f"cannot read {name}: {error}") from None
        entry = files[name]
        digests[name] = hashlib.sha256(data).hexdigest()
        if not isinstance(entry, dict) or len(data) != entry.get("bytes") or digests[name] != entry.get("sha256"):
            raise EpisodeError(f"{name} differs from the size or hash in the manifest")
        if entry.get("shape") != shape or entry.get("axes") != (W_AXES if key == "w" else H_AXES) \
                or data[:len(header)] != header or len(data) != len(header) + 8 * math.prod(shape):
            raise EpisodeError(f"{name} is not a C-ordered complex64 array of shape {shape} with the declared axes")
        finite, nonzero = _scan(data[len(header):])
        if not finite or (key != "w" and not nonzero):
            raise EpisodeError(f"{name} holds a nonfinite value or no path")
        offsets[name] = len(header)
    if digests[FILES["a"]] == digests[FILES["b"]]:
        raise EpisodeError("the target-present channel equals the target-absent one")
    cases = manifest["cases"]
    if not isinstance(cases, list) or [(case.get("id"), case.get("target_present"), case.get("h"), case.get("w"))
                                       for case in cases if isinstance(case, dict)] != \
            [(case, present, FILES[case], FILES["w"]) for case, present in CASES.items()]:
        raise EpisodeError('the cases must be "a" (target absent) and "b" (target present) on one noise file')
    _check_description({key: manifest[key] for key in DESCRIPTION + OPTIONAL if key in manifest}, schedule,
                       cases[1].get("truth"))
    return Episode(directory, manifest, schedule, offsets)


def summary(episode, case="b", opportunity=0, rx=0, symbol=None, subcarrier=None):
    """What a second reader recovers: the verified files, one coefficient and the phase checks."""
    grant = episode.schedule.opportunities[opportunity].grant
    if symbol is None:
        symbol = grant.pilot_index[0][0]
    if subcarrier is None:
        subcarrier = grant.pilot_index[0][1]
    manifest = episode.manifest
    result = {"episode_id": episode.episode_id, "verified_files": sorted(manifest["files"]),
              "sha256": {name: entry["sha256"] for name, entry in sorted(manifest["files"].items())},
              "h_shape": episode.h_shape, "opportunities": len(episode.schedule.opportunities),
              "grants": sorted(episode.schedule.grants), "analytic_model": manifest.get("analytic_model"),
              "coefficient": episode.coefficient(case, opportunity, rx, symbol, subcarrier),
              "truth": episode.truth(opportunity)}
    pilots = sorted({entry[0] for entry in grant.pilot_index})
    later = [index for index, entry in enumerate(episode.schedule.opportunities) if index > opportunity]
    if len(pilots) >= 2 and later:
        # Two unequal intervals: between the DM-RS symbols of one slot, then on to the next opportunity.
        result["phase_advance"] = [
            episode.phase_advance((opportunity, pilots[0]), (opportunity, pilots[-1]), rx, subcarrier),
            episode.phase_advance((opportunity, pilots[-1]), (later[0], pilots[0]), rx, subcarrier)]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Verify an episode entry and read one coefficient back.")
    commands = parser.add_subparsers(dest="command", required=True)
    show = commands.add_parser("show", help="verify every file, then print one coefficient and the phase checks")
    show.add_argument("directory", type=Path)
    show.add_argument("--case", default="b", choices=sorted(CASES), help="a: target absent, b: target present")
    show.add_argument("--opportunity", type=int, default=0)
    show.add_argument("--rx", type=int, default=0)
    show.add_argument("--symbol", type=int, help="default: the first DM-RS symbol of the opportunity's grant")
    show.add_argument("--subcarrier", type=int, help="default: the subcarrier of the grant's first pilot")
    args = parser.parse_args(argv)
    try:
        episode = open_episode(args.directory)
        print(json.dumps(summary(episode, args.case, args.opportunity, args.rx, args.symbol, args.subcarrier),
                         indent=2))
        return 0
    except (EpisodeError, IndexError) as error:
        parser.exit(2, f"episode: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
