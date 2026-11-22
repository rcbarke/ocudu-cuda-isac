"""Evaluator side of the presence head (Sprint 3.4, selector presence_head).

The sensing core scores a vector and knows nothing of what the vector was computed from. This
module holds what an evaluator adds around it, offline:

  split table   which episodes of a set are training, validation and held-out. The assignment is
                that of the set: an episode has the group of its trajectory, and the set fixed it
                before an episode was traced. The table maps the opaque episode key of every
                realization to its split, its scene (the trajectory), its matched pair (the
                episode) and its label. It is evaluator-only: no record and no feature line
                carries any of it.
  plans         the receiver replay of some entries of a set: the channel plan, the replay plan
                and the plan of the tapped run. An entry of a set binds no transmit grid, so the
                grids are those that an accepted entry of the same grants recorded.
  samples       the complete windows of a feature output, joined to the table by episode key.
  parity        the decisions of a sensor on a device against the decisions of the same artifact
                on the CPU, joined by window and by the digest of the input.
  evaluation    the counts of a decision set at the frozen threshold, by label.

Nothing here runs in a slot loop, and nothing here is imported by the sensing package.
"""

import hashlib
import json
from pathlib import Path

from ..sensing import pilot_ls as reference
from ..sensing import presence
from . import episode as episodes
from . import episode_set as sets
from . import grid_apply as replay
from . import replay_pusch as rp
from . import snapshot_ring as sr
from .episode import EpisodeError
from .grid_apply import GridApplyError
from .replay_pusch import ReplayPuschError
from .snapshot_ring import SnapshotRingError

SPLIT_SCHEMA = "presence-split/v1"
EVALUATION_SCHEMA = "presence-evaluation/v1"
PARITY_SCHEMA = "presence-parity/v1"
GROUPS = sets.GROUPS
TOLERANCE = {"logit_absolute": 1e-4, "probability_absolute": 1e-5, "reload_absolute": 1e-6}
CHANNEL_PLAN = "channel-plan.json"
REPLAY_PLAN = "replay-plan.json"
PLAN = "plan.json"


class PresenceHeadError(ValueError):
    """A set, a table, a plan or a decision set cannot be evaluated as documented."""


# ---------------------------------------------------------------- split table
def split_table(episode_set):
    """The split of a verified episode set, by episode: a presence-split/v1 object.

    Every episode has the group of its trajectory, as the set recorded it before it was traced.
    Each row names the keys of its two realizations: the opaque episode keys that the records of
    a replay carry. A key in two rows and a trajectory in two groups are refused.
    """
    manifest = episode_set.manifest
    rows, keys, owner = [], {}, {}
    for row, entry in zip(manifest["episodes"], episode_set.episodes()):
        group, trajectory = row["group"], row["trajectory"]
        if owner.setdefault(trajectory, group) != group or group not in GROUPS:
            raise PresenceHeadError(f"the trajectory {trajectory} is in two groups: a split is made by trajectory")
        made = {}
        for case in entry.manifest["cases"]:
            key = reference.opaque_key(case["realization_id"])
            if keys.setdefault(key, row["episode_id"]) != row["episode_id"] or key in made.values():
                raise PresenceHeadError(f"the episode key {key} names two realizations")
            made[case["id"]] = key
        if sorted(made) != sorted(episodes.CASES):
            raise PresenceHeadError(f"episode {row['episode_id']} is not a matched pair of the cases a and b")
        rows.append({"episode_id": row["episode_id"], "trajectory": trajectory, "group": group,
                     "directory": str(Path(entry.directory).resolve()), "periods": entry.schedule.periods,
                     "opportunities": len(entry.schedule.opportunities), "keys": made,
                     "grid_gain": replay.entry_calibration(entry.manifest)[0]})
    table = {"schema": SPLIT_SCHEMA, "set_id": manifest["set_id"], "set_status": manifest["status"],
             "config_sha256": manifest["config_sha256"], "assignment_sha256": manifest["assignment_sha256"],
             "unit": "trajectory", "rule": manifest["assignment_rule"],
             "groups": {group: list(manifest["groups"][group]) for group in GROUPS},
             "labels": {case: int(present) for case, present in episodes.CASES.items()},
             "counts": {group: sum(row["group"] == group for row in rows) for group in GROUPS}, "episodes": rows}
    return check_table(table)


def check_table(table):
    """Check a presence-split/v1 object and return it."""
    needed = {"schema", "set_id", "set_status", "config_sha256", "assignment_sha256", "unit", "rule", "groups", "labels",
              "counts", "episodes"}
    if not isinstance(table, dict) or set(table) != needed or table["schema"] != SPLIT_SCHEMA:
        raise PresenceHeadError(f"a split table is a {SPLIT_SCHEMA} object")
    groups = table["groups"]
    if set(groups) != set(GROUPS) or any(not groups[group] for group in GROUPS):
        raise PresenceHeadError("every group of a split holds at least one trajectory")
    owner = {name: group for group in GROUPS for name in groups[group]}
    if len(owner) != sum(len(groups[group]) for group in GROUPS):
        raise PresenceHeadError("a trajectory is in two groups: a split is made by trajectory")
    keys = set()
    for row in table["episodes"]:
        if owner.get(row["trajectory"]) != row["group"]:
            raise PresenceHeadError(f"episode {row['episode_id']} is not in the group of its trajectory")
        for key in row["keys"].values():
            if key in keys:
                raise PresenceHeadError(f"the episode key {key} names two realizations")
            keys.add(key)
    if table["counts"] != {group: sum(row["group"] == group for row in table["episodes"]) for group in GROUPS}:
        raise PresenceHeadError("the counts of the split table are not those of its episodes")
    return table


def by_key(table):
    """What each episode key of a table stands for: {key: {"split", "scene", "pair", "case", "label"}}."""
    return {key: {"split": row["group"], "scene": row["trajectory"], "pair": row["episode_id"], "case": case,
                  "label": table["labels"][case]}
            for row in table["episodes"] for case, key in row["keys"].items()}


def chunks(table, group, size):
    """The episodes of one group in runs of at most size entries. One run holds one grid gain.

    The channel application takes one gain per run. The entries of a set record the same gain
    when their target-absent channels are the same to the last bit, and are grouped by it here
    so that a set whose traces differ in the last bit is still replayed.
    """
    if type(size) is not int or size < 1:
        raise PresenceHeadError("a run holds at least one entry")
    by_gain = {}
    for row in table["episodes"]:
        if row["group"] == group:
            by_gain.setdefault(row["grid_gain"], []).append(row)
    made = []
    for gain in sorted(by_gain):
        rows = by_gain[gain]
        made += [rows[start:start + size] for start in range(0, len(rows), size)]
    return made


# ---------------------------------------------------------------- plans
def bindings_of(entry):
    """The transmit grid that an accepted entry recorded for each of its grants: {grant: {"x_sha256", "store"}}."""
    try:
        episode = entry if isinstance(entry, episodes.Episode) else episodes.open_episode(entry)
    except EpisodeError as error:
        raise PresenceHeadError(f"{entry} is not a usable episode entry: {error}") from None
    made = {}
    for grant_id, described in sorted(episode.manifest["grants"].items()):
        binding = described.get("x")
        if not isinstance(binding, dict) or not isinstance(binding.get("x_sha256"), str) \
                or not isinstance(binding.get("store"), str):
            raise PresenceHeadError(f"the entry {episode.episode_id} records no transmit grid for the grant {grant_id}")
        made[grant_id] = {"x_sha256": binding["x_sha256"], "store": binding["store"],
                          "record_sha256": described["record_sha256"]}
    return made


def write_plans(run_id, rows, bindings, directory, ring, epoch=1, source="pusch_dmrs"):
    """The three plans of one tapped receiver replay of some entries of a set, written into directory.

    Returns (channel plan, plan of the tapped run). The transmit grids are those of bindings: the
    store of each grant must hold the grant record that the entry of the set schedules. The
    payload rule is fixed, the evidence mode is timed and the tap is pilot_ls.
    """
    directory = Path(directory)
    try:
        channel = replay.entry_plan(run_id, [row["directory"] for row in rows],
                                    bindings={name: {"x_sha256": item["x_sha256"], "store": item["store"]}
                                              for name, item in bindings.items()})
        replay.write_plan(channel, directory / CHANNEL_PLAN)
        rp.write_plan(rp.build_plan(run_id, directory / CHANNEL_PLAN), directory / REPLAY_PLAN)
        plan = sr.build_plan(run_id, directory / REPLAY_PLAN, "timed", ring, "pilot_ls", epoch=epoch, source=source)
        sr.write_plan(plan, directory / PLAN)
    except (GridApplyError, ReplayPuschError, SnapshotRingError) as error:
        raise PresenceHeadError(f"the entries cannot be replayed: {error}") from None
    return channel, plan


def keys_of(channel_plan):
    """The episode key of every realization that a channel plan replays, in the order of the plan."""
    return [reference.opaque_key(source["realization_id"]) for source in channel_plan["sources"]]


# ---------------------------------------------------------------- samples
def samples(output, table, role, expected=None):
    """The complete windows of a feature output as samples of one split. Returns (rows, account).

    Every complete window must belong to an episode key of the table whose split is role: a
    window of another split in the output is refused. expected lists the episode keys that the
    run replayed. The account names every key without a complete window and the last answer the
    stream gave for it.
    """
    lookup = by_key(table)
    rows, seen, last = [], {}, {}
    for line in output.lines:
        if line["window"] is not None:
            last[line["window"]["episode"]] = line["status"]
    for line in output.ok:
        key = line["window"]["episode"]
        known = lookup.get(key)
        if known is None:
            raise PresenceHeadError(f"the window {line['window']['id']} belongs to no realization of the set")
        if known["split"] != role:
            raise PresenceHeadError(f"the window {line['window']['id']} belongs to the {known['split']} split, and "
                                    f"this output was read as {role}")
        seen[key] = seen.get(key, 0) + 1
        rows.append({"schema": presence.SAMPLE_SCHEMA, "window": line["window"]["id"], "episode": key,
                     "split": known["split"], "scene": known["scene"], "pair": known["pair"], "label": known["label"],
                     "spec_sha256": line["spec_sha256"], "names": line["features"]["names"],
                     "values": line["features"]["values"]})
    wanted = list(expected) if expected is not None else sorted(seen)
    missing = [{"episode": key, "pair": lookup[key]["pair"] if key in lookup else None, "last_answer": last.get(key)}
               for key in wanted if key not in seen]
    account = {"role": role, "realizations": len(wanted), "with_a_window": sum(key in seen for key in wanted),
               "windows": len(rows), "target_present": sum(row["label"] for row in rows),
               "target_absent": sum(1 - row["label"] for row in rows), "missing": missing,
               "counts": dict(output.report["counts"])}
    return rows, account


def ids_sha256(values):
    """The hash of a set of identifiers: SHA-256 over their sorted list."""
    return hashlib.sha256(json.dumps(sorted(values), separators=(",", ":")).encode("ascii")).hexdigest()


# ---------------------------------------------------------------- parity
def parity(device, offline, tolerance=None):
    """The scored decisions of a sensor on its device against those of the same artifact offline.

    device and offline are verified presence outputs. They are joined by window identifier, and a
    pair must name the same digest of its input, the same contributing opportunities and the
    same model. The logit and the probability must agree within the declared tolerance. A
    decision that differs at the frozen threshold is listed with the margin of its score.
    Returns (problems, facts).
    """
    tolerance = dict(TOLERANCE, **(tolerance or {}))
    mine, theirs = device.by_window, offline.by_window
    problems, flips, worst_logit, worst_probability = [], [], 0.0, 0.0
    if set(mine) != set(theirs):
        only = sorted(set(mine) ^ set(theirs))
        problems.append(f"{len(only)} windows are scored on one side only, first: {only[0]}")
    if device.manifest["sha256"] != offline.manifest["sha256"]:
        problems.append("the two outputs were scored with different artifacts")
    for window in sorted(set(mine) & set(theirs)):
        first, second = mine[window], theirs[window]
        if first["input"]["sha256"] != second["input"]["sha256"]:
            problems.append(f"{window}: the digest of the input differs: the two sides did not score the same feature")
            continue
        if [item["opportunity"] for item in first["contributing"]] != [item["opportunity"] for item in second["contributing"]]:
            problems.append(f"{window}: the contributing observations differ")
        logit = abs(first["score"]["logit"] - second["score"]["logit"])
        probability = abs(first["score"]["probability"] - second["score"]["probability"])
        worst_logit, worst_probability = max(worst_logit, logit), max(worst_probability, probability)
        if logit > tolerance["logit_absolute"] or probability > tolerance["probability_absolute"]:
            problems.append(f"{window}: the logit differs by {logit:.3e} and the probability by {probability:.3e}")
        if first["decision"]["present"] != second["decision"]["present"]:
            flips.append({"window": window, "device_logit": first["score"]["logit"],
                          "offline_logit": second["score"]["logit"],
                          "margin": first["score"]["logit"] - first["decision"]["threshold_logit"]})
    facts = {"schema": PARITY_SCHEMA, "joined": len(set(mine) & set(theirs)), "device": device.report["runtime"]["device"],
             "offline": offline.report["runtime"]["device"], "largest_logit_difference": worst_logit,
             "largest_probability_difference": worst_probability, "tolerance": tolerance, "flips": flips,
             "artifact_sha256": device.manifest["sha256"]}
    return problems, facts


# ---------------------------------------------------------------- evaluation, with the truth
def area_under_curve(logits, labels):
    """The probability that a target-present window scores above a target-absent one. Ties count one half."""
    positives = [value for value, label in zip(logits, labels) if label == 1]
    negatives = [value for value, label in zip(logits, labels) if label == 0]
    if not positives or not negatives:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def evaluate(output, table, role, expected=None):
    """The decisions of a presence output at the frozen threshold, by label: a presence-evaluation/v1 object.

    This is truth, read after the decisions are written. Every realization of expected stays in
    its denominator: one without a scored window is counted as missing and never as correct.
    """
    lookup = by_key(table)
    scored = {}
    for line in output.scored:
        key = line["window"]["episode"]
        known = lookup.get(key)
        if known is None or known["split"] != role:
            raise PresenceHeadError(f"the window {line['window']['id']} is no window of the {role} split")
        scored.setdefault(key, []).append(line)
    wanted = list(expected) if expected is not None else sorted(scored)
    rows, counts = [], {"target_present": {"episodes": 0, "detected": 0, "missed": 0, "missing": 0},
                        "target_absent": {"episodes": 0, "false_alarm": 0, "quiet": 0, "missing": 0}}
    logits, labels = [], []
    for key in wanted:
        known = lookup[key]
        side = counts["target_present" if known["label"] else "target_absent"]
        side["episodes"] += 1
        lines = scored.get(key, [])
        if not lines:
            side["missing"] += 1
            rows.append({"episode": key, "pair": known["pair"], "label": known["label"], "window": None, "logit": None,
                         "present": None})
            continue
        for line in lines:
            logits.append(line["score"]["logit"])
            labels.append(known["label"])
            rows.append({"episode": key, "pair": known["pair"], "label": known["label"], "window": line["window"]["id"],
                         "logit": line["score"]["logit"], "present": line["decision"]["present"]})
        positive = any(line["decision"]["present"] for line in lines)
        if known["label"]:
            side["detected" if positive else "missed"] += 1
        else:
            side["false_alarm" if positive else "quiet"] += 1
    return {"schema": EVALUATION_SCHEMA, "role": role, "graded": False, "opportunity": "one episode: a realization "
            "counts as positive when any of its scored windows is positive", "threshold_logit":
            output.manifest["threshold"]["logit"], "counts": counts, "area_under_curve": area_under_curve(logits, labels),
            "windows": len(logits), "rows": rows}
