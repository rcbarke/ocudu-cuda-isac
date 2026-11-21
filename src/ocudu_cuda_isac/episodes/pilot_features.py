"""Features on pilot observations: the evaluator side (Sprint 3.3, selector pilot_features).

The sensing side, ocudu_cuda_isac.sensing.features, turns the records of the ring into windows,
a delay-Doppler view and a feature vector, one record at a time. This module is what an
evaluator holds against it after a run. It shares no window logic and no transform with the
sensing side:

  * observations() reads the estimates that an audited run of the observer wrote as files, the
    same O2 that the ring carried;
  * offline_windows() sorts them and cuts complete windows by slot arithmetic, with all records
    in hand, where the stream decides at each record;
  * oracle() computes the view of a window as explicit Fourier sums at the physical frequencies
    and at the sample times of the records, in double precision, and the vector from it, where
    the stream uses FFT operators;
  * realizations() and expected_return() join a window to the episode entry it was replayed
    from: the case of each realization and the delay and Doppler shift of the first-order return
    from geometry. That is truth. Only an evaluator reads it, after the features are written.

NumPy is needed. The module imports the episode readers, so the sensing package never imports it.
"""

import math
from pathlib import Path

import numpy as np

from . import episode as episodes
from .episode import EpisodeError
from ..sensing import pilot_ls as reference

TC_PER_SECOND = reference.TC_PER_SECOND
LEADING = ("peak_to_median_db", "moving_to_static_db", "peak_delay_ns", "peak_doppler_hz", "doppler_centroid_hz",
           "doppler_spread_hz")
# Agreement of the stream with the oracle: the power of every cell relative to the peak of the
# view, and each feature in its own unit (dB, ns or Hz).
TOLERANCE = {"power_relative": 1e-9, "feature_absolute": 1e-6}


class PilotFeaturesError(ValueError):
    """An audited output, a window or an episode entry cannot be joined as documented."""


# ---------------------------------------------------------------- the same O2, from files
def observations(observer):
    """The estimates of an audited observer output, as arrays. observer is a reader of Sprint 3.1.

    Returns one dict per observation in the order of the file: the join keys, the slot, the
    allocation and the times of its obs-descriptor/v1 record, the values [rx_plane][symbol][pilot]
    in double precision and the validity bits as booleans.
    """
    made = []
    for record in observer.observations:
        shape = record["shape"]["shape"]
        elements = shape[0] * shape[1] * shape[2]
        payload, mask = observer.payload(record), observer.mask(record)
        if len(payload) != 8 * elements or len(mask) != (elements + 7) // 8:
            raise PilotFeaturesError(f"the files of observation {record['keys']['sequence']} do not have the size of "
                                     "its shape")
        bits = np.unpackbits(np.frombuffer(mask, dtype=np.uint8), bitorder="little")[:elements]
        allocation = record["allocation"]
        made.append({
            "epoch": record["keys"]["epoch"], "sequence": record["keys"]["sequence"],
            "episode": record["keys"]["episode"], "opportunity": record["keys"]["opportunity"],
            "opportunity_id": record["host"]["opportunity_id"], "absolute_slot": record["radio"]["absolute_slot"],
            "first_subcarrier": allocation["first_subcarrier"], "subcarrier_step": allocation["subcarrier_step"],
            "antenna_map": list(allocation["antenna_map"]), "symbols": list(allocation["dmrs_symbols"]),
            "t_first_sample": record["time"]["t_first_sample"], "t_last_sample": record["time"]["t_last_sample"],
            "values": np.frombuffer(payload, dtype="<c8").reshape(shape).astype(np.complex128),
            "valid": bits.reshape(shape).astype(bool)})
    return made


def offline_windows(found, spec):
    """Cut the windows of a specification from all observations of a run, with every record in hand.

    Observations are grouped by epoch and episode key and sorted by slot. The first window of a
    group starts with the period of its first observation in the sequence, and the next ones
    follow at the hop. A window is eligible when every sample of the sequence is present over
    all its periods. Returns the windows of every group, eligible or not, in the order in which
    a stream closes them.
    """
    radio, sequence, rule = spec["radio"], spec["sequence"], spec["window"]
    slots, rows, periods, hop = sequence["slots_in_period"], sequence["reference_symbols"], rule["periods"], rule["hop_periods"]
    useful = reference.useful_tc(radio["scs_khz"])
    groups = {}
    for item in found:
        period, place = divmod(item["absolute_slot"], radio["period_slots"])
        if place in slots:
            groups.setdefault((item["epoch"], item["episode"]), {})[(period, place)] = item
    windows = []
    for (epoch, episode), held in groups.items():
        first, last = min(held)[0], max(held)[0]
        end = first + periods - 1
        while end <= last:
            places = [(period, slot) for period in range(end - periods + 1, end + 1) for slot in slots]
            members = [held.get(place) for place in places]
            window = {"id": f"{epoch}/{episode}/{end}", "epoch": epoch, "episode": episode,
                      "first_period": end - periods + 1, "last_period": end,
                      "missing": sum(member is None for member in members), "close_slot": None}
            closing = held.get((end, slots[-1]))
            # A stream closes a window at its last sample, or at the first later record of the sequence.
            later = [item["absolute_slot"] for place, item in held.items() if place > (end, slots[-1])]
            window["close_slot"] = closing["absolute_slot"] if closing is not None else (min(later) if later else None)
            window["eligible"] = window["missing"] == 0
            if window["eligible"]:
                cube, valid, times = [], [], []
                for member in members:
                    order = sorted(range(len(member["antenna_map"])), key=lambda plane: member["antenna_map"][plane])
                    for row in rows:
                        if row == 0:
                            times.append(member["t_first_sample"])
                        elif row == len(member["symbols"]) - 1:
                            times.append(member["t_last_sample"] - useful)
                        else:
                            raise PilotFeaturesError("the record names the time of its first and of its last reference "
                                                     "symbol only")
                        cube.append(member["values"][order, row, :].T)
                        valid.append(member["valid"][order, row, :].T)
                window.update(
                    contributing=[member["opportunity"] for member in members], times_tc=times,
                    cube=np.stack(cube, axis=1), valid=np.stack(valid, axis=1),
                    first_subcarrier=members[0]["first_subcarrier"], subcarrier_step=members[0]["subcarrier_step"],
                    same_comb=len({(member["first_subcarrier"], member["subcarrier_step"], member["values"].shape[2])
                                   for member in members}) == 1)
            if window["close_slot"] is not None:
                windows.append(window)
            end += hop
    return sorted(windows, key=lambda window: (window["epoch"], window["episode"], window["last_period"]))


def oracle(window, spec):
    """The view and the vector of one eligible window, by explicit Fourier sums in double precision.

    Delay cell m lies at tau = m / (K df) and sums exp(+j 2 pi (f - f0) tau) over the comb.
    Doppler cell d lies at nu and sums exp(-j 2 pi nu (t - t0)) over the sample times of the
    records. For the fft step nu = d / (N dt) with the interval dt taken from those times.
    """
    cube, valid = np.where(window["valid"], window["cube"], 0), window["valid"]
    pilots, samples, _ = cube.shape
    spacing = window["subcarrier_step"] * spec["radio"]["scs_khz"] * 1000.0
    seconds = (np.array(window["times_tc"], dtype=np.float64) - window["times_tc"][0]) / TC_PER_SECOND
    counts = valid.sum(axis=1)
    if spec["steps"]["static"] == "window_mean":
        mean = cube.sum(axis=1) / np.maximum(counts, 1)
        moving = np.where(valid, cube - mean[:, None, :], 0)
    else:
        mean, moving = np.zeros((pilots, cube.shape[2]), dtype=np.complex128), cube
    bins = min(spec["view"]["delay_bins"], pilots)
    resolution_ns = 1e9 / (pilots * spacing)
    offsets = np.arange(pilots) * spacing
    delays = np.arange(bins) / (pilots * spacing)
    to_delay = np.exp(2j * np.pi * np.outer(delays, offsets)) / pilots
    if spec["steps"]["doppler"] == "fft":
        steps = {later - earlier for earlier, later in zip(window["times_tc"], window["times_tc"][1:])}
        if len(steps) != 1:
            raise PilotFeaturesError("the samples of the window are not equally spaced")
        indices = np.arange(samples) - samples // 2
        hz = indices / (samples * (steps.pop() / TC_PER_SECOND))
    else:
        grid = spec["view"]["doppler_grid"]
        indices = np.arange(grid["bins"]) - grid["bins"] // 2
        hz = indices * grid["step_hz"]
    to_doppler = np.exp(-2j * np.pi * np.outer(hz, seconds)) / samples
    spectrum = np.einsum("mk,knr,dn->mdr", to_delay, moving, to_doppler)
    power = (np.abs(spectrum) ** 2).sum(axis=2)
    moves = hz != 0 if spec["steps"]["static"] != "none" else np.ones(len(hz), dtype=bool)
    cells = power[:, moves]
    row, column = (int(index) for index in np.unravel_index(int(np.argmax(cells)), cells.shape))
    column = int(np.flatnonzero(moves)[column])
    peak = float(power[row, column])
    scale = 10.0 ** (-spec["view"]["floor_db"] / 10.0)
    floor = peak * scale
    median = max(float(np.median(cells)), floor)
    moving_power = float((np.abs(moving) ** 2).sum())
    static_power = max(float((counts * np.abs(mean) ** 2).sum()), moving_power * scale)
    weights = cells.sum(axis=0)
    centroid = float((hz[moves] * weights).sum() / weights.sum())
    spread = math.sqrt(max(float((((hz[moves] - centroid) ** 2) * weights).sum() / weights.sum()), 0.0))
    values = [10.0 * math.log10(peak / median), 10.0 * math.log10(moving_power / static_power), row * resolution_ns,
              float(hz[column]), centroid, spread]
    values += [10.0 * math.log10(max(float(power[:, index].max()), floor) / median) for index in np.flatnonzero(hz != 0)]
    names = list(LEADING) + [f"profile_db@{value:+g}Hz" for value in hz if value != 0]
    return {"power": power, "names": names, "values": values, "hz": [float(value) for value in hz],
            "resolution_ns": resolution_ns, "resolution_hz": float(hz[1] - hz[0]) if len(hz) > 1 else None,
            "peak": {"delay_bin": row, "doppler_bin": int(indices[column]), "delay_ns": row * resolution_ns,
                     "doppler_hz": float(hz[column]), "power": peak}}


def differences(line, power, made, tolerance=None):
    """Where a complete window of the stream differs from the oracle of the same window. Empty when it does not."""
    tolerance = tolerance or TOLERANCE
    found = []
    if list(power.shape) != list(made["power"].shape):
        return [f"the view has the shape {list(power.shape)}, the oracle {list(made['power'].shape)}"]
    worst = float(np.abs(power - made["power"]).max() / made["power"].max())
    if not worst <= tolerance["power_relative"]:
        found.append(f"a cell of the view differs from the oracle by {worst:.3g} of the peak")
    if line["features"]["names"] != made["names"]:
        found.append("the vector has other names or another order than the oracle")
    else:
        for name, mine, other in zip(made["names"], line["features"]["values"], made["values"]):
            if not abs(mine - other) <= tolerance["feature_absolute"]:
                found.append(f"{name} is {mine!r}, the oracle has {other!r}")
    peak = line["view"]["peak"]
    if (peak["delay_bin"], peak["doppler_bin"]) != (made["peak"]["delay_bin"], made["peak"]["doppler_bin"]):
        found.append(f"the peak lies at delay bin {peak['delay_bin']} and Doppler bin {peak['doppler_bin']}, the oracle "
                     f"has {made['peak']['delay_bin']} and {made['peak']['doppler_bin']}")
    return found


# ---------------------------------------------------------------- truth, for the evaluator only
def realizations(channel_plan):
    """What each episode key of a run stands for: the case, the entry and the opportunities, from the channel plan.

    Returns {episode key: {"realization_id", "case", "present", "entry", "opportunities"}}, where
    opportunities maps the opaque key of an opportunity to its index in the entry.
    """
    names = {name: case for case, name in episodes.FILES.items() if case in episodes.CASES}
    made = {}
    for source in channel_plan["sources"]:
        path = Path(source["h"]["path"])
        case = names.get(path.name)
        if case is None or source.get("entry") is None:
            raise PilotFeaturesError(f"the realization {source['realization_id']} is not a case of an episode entry")
        made[reference.opaque_key(source["realization_id"])] = {
            "realization_id": source["realization_id"], "case": case, "present": episodes.CASES[case],
            "entry": path.parent, "opportunities": {}}
    for opportunity in channel_plan["opportunities"]:
        made[reference.opaque_key(opportunity["realization_id"])]["opportunities"][
            reference.opaque_key(opportunity["opportunity_id"])] = opportunity["index"]
    return made


def expected_return(entry, index):
    """Delay and Doppler shift of the first-order return at one opportunity of an entry, from geometry alone."""
    try:
        episode = entry if isinstance(entry, episodes.Episode) else episodes.open_episode(entry)
        truth = episode.truth(index)
        found = episodes.bistatic(episode.manifest["geometry"], truth["position_m"], truth["velocity_m_s"],
                                  episode.grant.carrier_hz)
    except (EpisodeError, KeyError, IndexError, TypeError) as error:
        raise PilotFeaturesError(f"the entry {getattr(entry, 'directory', entry)} gives no truth for opportunity {index}: "
                                 f"{error}") from None
    geometry = episode.manifest["geometry"]
    direct_ns = 1e9 * math.dist(geometry["ue_tx_m"], geometry["gnb_rx_m"]) / episodes.SPEED_OF_LIGHT_M_S
    return {"delay_ns": found["delay_ns"], "doppler_hz": found["doppler_hz"], "excess_delay_ns": found["excess_delay_ns"],
            "direct_delay_ns": direct_ns, "scattered_to_direct_db": truth.get("scattered_to_direct_db")}


def fold(value, low, span):
    """A Doppler shift folded into the unambiguous interval that starts at low and is span wide."""
    return low + (value - low) % span


def resolved(line, expected):
    """Whether the peak of a complete window lies within one resolvable bin of an expected return.

    The delay is compared as it is. The Doppler shift is folded into the unambiguous interval of
    the window first, and the fold is reported. Returns (passed, facts).
    """
    axes, peak = line["axes"], line["view"]["peak"]
    delay_bin, doppler_bin = axes["delay"]["resolution_ns"], axes["doppler"]["resolution_hz"]
    cells = axes["doppler"]["hz"]
    span = (cells[-1] - cells[0]) + (cells[1] - cells[0])
    folded = fold(expected["doppler_hz"], cells[0] - (cells[1] - cells[0]) / 2.0, span)
    facts = {"peak_delay_ns": peak["delay_ns"], "expected_delay_ns": expected["delay_ns"],
             "delay_error_ns": peak["delay_ns"] - expected["delay_ns"], "delay_resolution_ns": delay_bin,
             "peak_doppler_hz": peak["doppler_hz"], "expected_doppler_hz": expected["doppler_hz"],
             "expected_doppler_folded_hz": folded, "aliased": abs(folded - expected["doppler_hz"]) > 1e-9,
             "doppler_error_hz": peak["doppler_hz"] - folded, "doppler_resolution_hz": doppler_bin,
             "inside_kept_delay": expected["delay_ns"] < axes["delay"]["kept_ns"]}
    passed = abs(facts["delay_error_ns"]) <= delay_bin and abs(facts["doppler_error_hz"]) <= doppler_bin \
        and facts["inside_kept_delay"]
    return passed, facts
