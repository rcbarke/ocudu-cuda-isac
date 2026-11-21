"""Episode configurations and the demonstration episode set (Sprint 2.2).

Two configuration records select what rt_channel.py traces:

  episode-pair-config/v1   one matched pair with an explicit target motion
  episode-set-config/v1    a set of matched pairs: walking trajectories inside an area of interest,
                           each cut into episodes, each trajectory assigned to one group

A set is a channel-episode-set/v1 directory. Its manifest freezes the position grid and the
assignment of whole trajectories to the training, validation and held-out groups before any window
is cut: an episode inherits the group of its trajectory and nothing else decides it. Every episode
is a matched pair, so a set holds as many target-absent runs as target-present runs. The absent
members of the training group are the calibration episodes of the background template.

Standard library only. A grant path in a configuration is relative to the configuration file.
"""

import hashlib
import json
import math
from pathlib import Path
import random

from . import episode as episodes
from . import grant as grants
from . import schedule as schedules
from .episode import EpisodeError, TC_PER_SECOND
from .grant import GrantError
from .schedule import ScheduleError
from ..replay import ContractError, load_json

PAIR_CONFIG = "episode-pair-config/v1"
SET_CONFIG = "episode-set-config/v1"
SET_SCHEMA = "channel-episode-set/v1"
GROUPS = ("training", "validation", "held_out")
STATIC = ("grant", "schedule", "scene", "endpoints", "array", "target", "solver", "noise")
MATERIALS = ("concrete", "brick", "plasterboard", "wood", "glass", "ceiling_board", "chipboard", "plywood",
             "marble", "floorboard", "metal")
BACKGROUND_FLAGS = ("max_depth", "samples_per_src", "max_num_paths_per_src", "los", "specular_reflection", "refraction")
SCATTERED_FLAGS = ("max_depth", "samples_per_sp", "buffer_size_per_sp", "los", "specular_reflection", "refraction")
TRAJECTORY = {"line": ("from_m", "to_m"), "lawnmower": ("axis", "lanes", "margin_m"),
              "spiral": ("turns", "margin_m"), "random": ("seed", "waypoints", "margin_m")}
TRAJECTORY_COMMON = ("id", "group", "kind", "speed_m_s", "start_s", "episodes", "periods", "stride_s")


def _fields(value, fields, name, optional=()):
    if not isinstance(value, dict) or not set(fields) <= set(value) or set(value) - set(fields) - set(optional):
        raise EpisodeError(f"{name} requires exactly {', '.join(fields)}"
                           + (f" and optionally {', '.join(optional)}" if optional else ""))
    return value


def _number(value, name, low=None, high=None):
    if type(value) not in (int, float) or not math.isfinite(value) or (low is not None and value < low) \
            or (high is not None and value > high):
        raise EpisodeError(f"{name} must be a finite number" + (f" from {low}" if low is not None else "")
                           + (f" to {high}" if high is not None else ""))
    return value


def _count(value, name, low=1, high=10**7):
    if type(value) is not int or not low <= value <= high:
        raise EpisodeError(f"{name} must be an integer from {low} to {high}")
    return value


def _vector(value, name, size=3):
    if not isinstance(value, list) or len(value) != size:
        raise EpisodeError(f"{name} must hold {size} numbers")
    return [float(_number(part, name)) for part in value]


def _static(config, name):
    """Validate the part that a pair and a set share: scene, endpoints, array, target, solver, noise."""
    scene = config["scene"]
    if not isinstance(scene, dict) or scene.get("kind") not in ("free_space", "open_room"):
        raise EpisodeError(f"{name}: scene.kind must be free_space or open_room")
    if scene["kind"] == "free_space":
        _fields(scene, ("kind",), "scene")
    else:
        _fields(scene, ("kind", "size_m", "materials", "thickness_m"), "scene")
        size = _vector(scene["size_m"], "scene.size_m")
        if min(size) <= 0:
            raise EpisodeError("scene.size_m must be positive")
        materials = _fields(scene["materials"], ("floor", "ceiling", "walls"), "scene.materials")
        if any(material not in MATERIALS for material in materials.values()):
            raise EpisodeError(f"a room material must be one of {', '.join(MATERIALS)}")
        _number(scene["thickness_m"], "scene.thickness_m", 0.001, 10)
    endpoints = _fields(config["endpoints"], ("ue_tx_m", "gnb_rx_m", "gnb_yaw_rad"), "endpoints")
    tx, rx = _vector(endpoints["ue_tx_m"], "endpoints.ue_tx_m"), _vector(endpoints["gnb_rx_m"], "endpoints.gnb_rx_m")
    _number(endpoints["gnb_yaw_rad"], "endpoints.gnb_yaw_rad", -math.pi, math.pi)
    if math.dist(tx, rx) <= 0:
        raise EpisodeError("the UE and the gNB coincide")
    if scene["kind"] == "open_room" and any(not 0 < value < limit for point in (tx, rx)
                                            for value, limit in zip(point, scene["size_m"])):
        raise EpisodeError("an endpoint lies outside the room")
    array = _fields(config["array"], ("spacing_wavelengths", "pattern", "polarization"), "array")
    _number(array["spacing_wavelengths"], "array.spacing_wavelengths", 0.01, 100)
    if array["pattern"] not in ("iso", "dipole", "hw_dipole", "tr38901") or array["polarization"] not in ("V", "H"):
        raise EpisodeError("array.pattern must be iso, dipole, hw_dipole or tr38901 and array.polarization V or H")
    target = config["target"]
    if not isinstance(target, dict) or target.get("model") not in ("constant_rcs_cuboid", "tr38901_human"):
        raise EpisodeError("target.model must be constant_rcs_cuboid or tr38901_human")
    solver = _fields(config["solver"], ("seed", "background", "scattered"), "solver")
    _count(solver["seed"], "solver.seed", 0, 2**31 - 1)
    for key, flags in (("background", BACKGROUND_FLAGS), ("scattered", SCATTERED_FLAGS)):
        _fields(solver[key], flags, f"solver.{key}")
        for flag in flags:
            if flag in ("los", "specular_reflection", "refraction"):
                if type(solver[key][flag]) is not bool:
                    raise EpisodeError(f"solver.{key}.{flag} must be true or false")
            else:
                _count(solver[key][flag], f"solver.{key}.{flag}", 0 if key == "background" and flag == "max_depth" else 1)
    noise = _fields(config["noise"], ("snr_db", "seed"), "noise")
    _number(noise["snr_db"], "noise.snr_db", -50, 200)
    _count(noise["seed"], "noise.seed", 0, 2**63 - 1)


def _grant(config, path):
    if not isinstance(config["grant"], str):
        raise EpisodeError("grant must be a path relative to the configuration file")
    try:
        return grants.load(Path(path).parent / config["grant"])
    except GrantError as error:
        raise EpisodeError(f"the grant of the configuration is invalid: {error}") from None


def _schedule(record, schedule_id, periods, base, cache=None):
    _fields(record, ("tdd", "start_sfn", "payload"), "schedule", optional=("periods",))
    try:
        return schedules.build({"schema": schedules.SCHEMA, "schedule_id": schedule_id, "tdd": record["tdd"],
                                "start_sfn": record["start_sfn"], "periods": periods,
                                "payload": record["payload"]}, base, cache)
    except ScheduleError as error:
        raise EpisodeError(f"the schedule of the configuration is invalid: {error}") from None


def _read(path, schema):
    try:
        config = load_json(Path(path))
    except (ContractError, OSError) as error:
        raise EpisodeError(f"cannot read the configuration: {error}") from None
    if not isinstance(config, dict) or config.get("schema") != schema:
        raise EpisodeError(f"{path} is not a {schema} configuration")
    return config


def load_pair_config(path):
    """Read one episode-pair-config/v1 file. Returns (configuration, base grant, schedule)."""
    config = _fields(_read(path, PAIR_CONFIG), ("schema", "episode_id", *STATIC), "a pair configuration",
                     optional=("analytic_model",))
    _static(config, "pair")
    target = config["target"]
    if target["model"] == "constant_rcs_cuboid":
        _fields(target, ("model", "sigma_m2", "size_m", "motion"), "target")
        _number(target["sigma_m2"], "target.sigma_m2", 1e-12, 1e12)
        if min(_vector(target["size_m"], "target.size_m")) <= 0:
            raise EpisodeError("target.size_m must be positive")
    else:
        _fields(target, ("model", "motion"), "target")
    episodes.target_state(target["motion"], 0.0)
    if "periods" not in config["schedule"]:
        raise EpisodeError("the schedule of a pair needs its number of periods")
    base = _grant(config, path)
    schedule = _schedule(config["schedule"], config["episode_id"], config["schedule"]["periods"], base)
    if config.get("analytic_model") is not None:
        if config["analytic_model"] != episodes.ANALYTIC_MODEL or config["scene"]["kind"] != "free_space" \
                or target["model"] != "constant_rcs_cuboid" or target["motion"]["kind"] != "constant_velocity" \
                or config["array"]["pattern"] != "iso":
            raise EpisodeError(f"{episodes.ANALYTIC_MODEL} needs free space, isotropic elements and a constant "
                               "cross-section moving at a constant velocity")
    return config, base, schedule


def polyline(trajectory, area, height):
    """Waypoints of one trajectory as [x, y, z] points inside the area of interest."""
    (x0, x1), (y0, y1) = area["x_m"], area["y_m"]
    kind = trajectory["kind"]
    if kind == "line":
        points = [_vector(trajectory["from_m"], "from_m", 2), _vector(trajectory["to_m"], "to_m", 2)]
    else:
        margin = _number(trajectory["margin_m"], "margin_m", 0)
        x0, x1, y0, y1 = x0 + margin, x1 - margin, y0 + margin, y1 - margin
        if x0 >= x1 or y0 >= y1:
            raise EpisodeError(f"the margin of trajectory {trajectory['id']} leaves no area")
    if kind == "lawnmower":
        lanes = _count(trajectory["lanes"], "lanes", 2, 1000)
        if trajectory["axis"] not in ("x", "y"):
            raise EpisodeError("a lawnmower axis is x or y")
        points = []
        for lane in range(lanes):
            across = lane / (lanes - 1)
            ends = (0.0, 1.0) if lane % 2 == 0 else (1.0, 0.0)
            for along in ends:
                points.append([x0 + (x1 - x0) * along, y0 + (y1 - y0) * across] if trajectory["axis"] == "x"
                              else [x0 + (x1 - x0) * across, y0 + (y1 - y0) * along])
    elif kind == "spiral":
        turns = _count(trajectory["turns"], "turns", 1, 1000)
        points, step_x, step_y = [], (x1 - x0) / (2 * turns + 1), (y1 - y0) / (2 * turns + 1)
        for turn in range(turns):
            left, right = x0 + turn * step_x, x1 - turn * step_x
            low, high = y0 + turn * step_y, y1 - turn * step_y
            points += [[left, low], [right, low], [right, high], [left, high], [left, low + step_y]]
    elif kind == "random":
        generator = random.Random(_count(trajectory["seed"], "seed", 0, 2**63 - 1))
        points = [[x0 + (x1 - x0) * generator.random(), y0 + (y1 - y0) * generator.random()]
                  for _ in range(_count(trajectory["waypoints"], "waypoints", 2, 100000))]
    (ax0, ax1), (ay0, ay1) = area["x_m"], area["y_m"]
    for x, y in points:
        if not (ax0 <= x <= ax1 and ay0 <= y <= ay1):
            raise EpisodeError(f"trajectory {trajectory['id']} leaves the area of interest at ({x}, {y})")
    return [[float(x), float(y), float(height)] for x, y in points]


def cell(config, x, y):
    """Row and column of a position in the frozen grid: rows along y, columns along x."""
    (x0, x1), (y0, y1) = config["area"]["x_m"], config["area"]["y_m"]
    rows, cols = config["position_grid"]["rows"], config["position_grid"]["cols"]
    if not (x0 <= x <= x1 and y0 <= y <= y1):
        raise EpisodeError(f"the position ({x}, {y}) lies outside the area of interest")
    return [min(rows - 1, int((y - y0) / (y1 - y0) * rows)), min(cols - 1, int((x - x0) / (x1 - x0) * cols))]


def grid_record(config):
    (x0, x1), (y0, y1) = config["area"]["x_m"], config["area"]["y_m"]
    rows, cols = config["position_grid"]["rows"], config["position_grid"]["cols"]
    return {"rows": rows, "cols": cols, "cell_m": [(y1 - y0) / rows, (x1 - x0) / cols],
            "origin_m": [x0, y0], "area": config["area"],
            "rule": "row = floor((y - y_min) / cell height), column = floor((x - x_min) / cell width), the upper "
                    "edge belongs to the last cell"}


def load_set_config(path):
    """Read one episode-set-config/v1 file. Returns (configuration, base grant)."""
    config = _fields(_read(path, SET_CONFIG), ("schema", "set_id", "status", *STATIC, "area", "position_grid",
                                               "min_training_coverage", "trajectories"), "a set configuration")
    _static(config, "set")
    if not isinstance(config["set_id"], str) or not config["set_id"].isascii() or not config["set_id"] \
            or any(character in config["set_id"] for character in "/\\ \t\n"):
        raise EpisodeError("set_id must be a nonempty ASCII identifier without separators")
    if config["status"] not in ("proposed", "frozen"):
        raise EpisodeError("status must be proposed or frozen")
    target = _fields(config["target"], ("model", "center_height_m"), "the target of a set", optional=("sigma_m2", "size_m"))
    _number(target["center_height_m"], "target.center_height_m", 0.01, 100)
    if target["model"] == "constant_rcs_cuboid":
        _number(target.get("sigma_m2"), "target.sigma_m2", 1e-12, 1e12)
        _vector(target.get("size_m"), "target.size_m")
    area = _fields(config["area"], ("x_m", "y_m"), "area")
    for key in ("x_m", "y_m"):
        low, high = _vector(area[key], f"area.{key}", 2)
        if low >= high:
            raise EpisodeError(f"area.{key} must increase")
    if config["scene"]["kind"] == "open_room" and (
            area["x_m"][0] < 0 or area["x_m"][1] > config["scene"]["size_m"][0]
            or area["y_m"][0] < 0 or area["y_m"][1] > config["scene"]["size_m"][1]):
        raise EpisodeError("the area of interest leaves the room")
    grid = _fields(config["position_grid"], ("rows", "cols"), "position_grid")
    _count(grid["rows"], "position_grid.rows", 1, 10000)
    _count(grid["cols"], "position_grid.cols", 1, 10000)
    _number(config["min_training_coverage"], "min_training_coverage", 0, 1)
    trajectories = config["trajectories"]
    if not isinstance(trajectories, list) or not trajectories:
        raise EpisodeError("a set needs trajectories")
    seen = set()
    for trajectory in trajectories:
        if not isinstance(trajectory, dict) or trajectory.get("kind") not in TRAJECTORY:
            raise EpisodeError(f"a trajectory kind is one of {', '.join(sorted(TRAJECTORY))}")
        _fields(trajectory, TRAJECTORY_COMMON + TRAJECTORY[trajectory["kind"]], f"a {trajectory['kind']} trajectory")
        name = trajectory["id"]
        if not isinstance(name, str) or not name or not name.isascii() or name in seen:
            raise EpisodeError("every trajectory needs its own ASCII id")
        seen.add(name)
        if trajectory["group"] not in GROUPS:
            raise EpisodeError(f"trajectory {name}: the group must be one of {', '.join(GROUPS)}")
        _number(trajectory["speed_m_s"], "speed_m_s", 0.01, 100)
        _number(trajectory["start_s"], "start_s", 0)
        _number(trajectory["stride_s"], "stride_s", 0)
        _count(trajectory["episodes"], "episodes")
        _count(trajectory["periods"], "periods")
    missing = [group for group in GROUPS if not any(t["group"] == group for t in trajectories)]
    if missing:
        raise EpisodeError(f"no trajectory is assigned to the group {missing[0]}")
    return config, _grant(config, path)


def assignment_sha256(trajectories):
    """Hash of the trajectory-to-group assignment, fixed before any episode or window exists."""
    pairs = sorted((trajectory["id"], trajectory["group"]) for trajectory in trajectories)
    return hashlib.sha256(json.dumps(pairs, separators=(",", ":")).encode("ascii")).hexdigest()


def plan(config, base):
    """Every episode of a set, in order: its opaque id, group, schedule, target motion and noise seed."""
    plans, cache = [], {}
    for trajectory in config["trajectories"]:
        points = polyline(trajectory, config["area"], config["target"]["center_height_m"])
        length = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
        for index in range(trajectory["episodes"]):
            identity = hashlib.sha256(f"{config['set_id']}:{trajectory['id']}:{index}".encode("ascii")).hexdigest()
            episode_id = "ep-" + identity[:12]
            schedule = _schedule(config["schedule"], episode_id, trajectory["periods"], base, cache)
            start = trajectory["start_s"] + index * trajectory["stride_s"]
            if trajectory["speed_m_s"] * (start + schedule.duration_tc() / TC_PER_SECOND) > length:
                raise EpisodeError(f"trajectory {trajectory['id']} is {length:.2f} m long and episode {index} "
                                   "runs past its end")
            seed = int.from_bytes(hashlib.sha256(f"{config['noise']['seed']}:{episode_id}".encode("ascii")).digest()[:6],
                                  "big")
            plans.append({"episode_id": episode_id, "trajectory": trajectory["id"], "group": trajectory["group"],
                          "index": index, "walk_start_s": start, "schedule": schedule,
                          "motion": {"kind": "waypoints", "points": points, "speed_m_s": trajectory["speed_m_s"],
                                     "start_s": start},
                          "noise": {"snr_db": config["noise"]["snr_db"], "seed": seed}, "labels": None})
    if len({item["episode_id"] for item in plans}) != len(plans):
        raise EpisodeError("two episodes received the same identifier")
    return plans


def label(config, item, truth):
    """Add the grid cell to every truth entry of one planned episode and set the episode's labels."""
    cells = []
    for entry in truth:
        entry["cell"] = cell(config, entry["position_m"][0], entry["position_m"][1])
        cells.append(tuple(entry["cell"]))
    item["labels"] = {"set_id": config["set_id"], "trajectory": item["trajectory"], "group": item["group"],
                      "episode_index": item["index"], "walk_start_s": item["walk_start_s"],
                      "position_grid": grid_record(config), "cells": [list(value) for value in sorted(set(cells))]}
    return item["labels"]


def _text(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("ascii")


def write_manifest(directory, config, config_sha256, plans, manifests):
    """Write manifest.json of a set whose episodes are complete under directory/episodes."""
    directory = Path(directory)
    rows = []
    visited = {group: set() for group in GROUPS}
    for item in plans:
        manifest = manifests[item["episode_id"]]
        cells = manifest["labels"]["cells"]
        visited[item["group"]].update(tuple(value) for value in cells)
        path = directory / "episodes" / item["episode_id"] / "manifest.json"
        rows.append({"episode_id": item["episode_id"], "trajectory": item["trajectory"], "group": item["group"],
                     "index": item["index"], "directory": f"episodes/{item['episode_id']}",
                     "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "files": {name: entry["sha256"] for name, entry in sorted(manifest["files"].items())},
                     "opportunities": len(manifest["opportunities"]), "cells": cells})
    total = config["position_grid"]["rows"] * config["position_grid"]["cols"]
    counts = {group: sum(row["group"] == group for row in rows) for group in GROUPS}
    manifest = {
        "schema": SET_SCHEMA, "set_id": config["set_id"], "status": config["status"],
        "access": "offline_stimulus_and_evaluator", "config_sha256": config_sha256, "config": config,
        "position_grid": grid_record(config),
        "groups": {group: [t["id"] for t in config["trajectories"] if t["group"] == group] for group in GROUPS},
        "assignment_sha256": assignment_sha256(config["trajectories"]),
        "assignment_rule": "an episode has the group of its trajectory; the assignment is part of the configuration "
                           "and was fixed before any episode was traced or any window was cut",
        "trajectories": [{"id": t["id"], "group": t["group"], "kind": t["kind"],
                          "points_m": polyline(t, config["area"], config["target"]["center_height_m"]),
                          "episodes": t["episodes"]} for t in config["trajectories"]],
        "episodes": rows,
        "counts": {"episodes": len(rows), "target_present_runs": len(rows), "target_absent_runs": len(rows),
                   "by_group": counts, "opportunities": sum(row["opportunities"] for row in rows)},
        "coverage": {"cells": total, "by_group": {group: len(visited[group]) for group in GROUPS},
                     "training_fraction": len(visited["training"]) / total,
                     "minimum_training_fraction": config["min_training_coverage"]},
        "calibration_episodes": "case a of every episode of the training group"}
    (directory / "manifest.json").write_bytes(_text(manifest))
    return manifest


class EpisodeSet:
    """A verified set: its manifest and its episodes, each a verified entry."""

    def __init__(self, directory, manifest, entries):
        self.directory, self.manifest, self._entries = directory, manifest, entries

    def episodes(self, group=None):
        return [entry for row, entry in zip(self.manifest["episodes"], self._entries)
                if group is None or row["group"] == group]


def open_set(directory):
    """Verify a set and every episode in it, then return it.

    Refused: an episode whose files or manifest differ from the set manifest, an episode whose
    group is not the group of its trajectory, a trajectory in two groups, a group without a
    trajectory, a changed assignment and a count that the episodes do not reproduce.
    """
    directory = Path(directory)
    try:
        manifest = load_json(directory / "manifest.json")
    except (ContractError, OSError) as error:
        raise EpisodeError(f"{directory} is not a complete episode set: {error}") from None
    if not isinstance(manifest, dict) or manifest.get("schema") != SET_SCHEMA:
        raise EpisodeError(f"{directory / 'manifest.json'} is not a {SET_SCHEMA} manifest")
    try:
        groups, rows, trajectories = manifest["groups"], manifest["episodes"], manifest["config"]["trajectories"]
        declared = {trajectory["id"]: trajectory["group"] for trajectory in trajectories}
        if len(declared) != len(trajectories) or set(groups) != set(GROUPS) or any(not groups[g] for g in GROUPS) \
                or {name: group for group in GROUPS for name in groups[group]} != declared \
                or sum(len(groups[group]) for group in GROUPS) != len(declared):
            raise EpisodeError("the groups are not a partition of the trajectories")
        if manifest["assignment_sha256"] != assignment_sha256(trajectories):
            raise EpisodeError("the trajectory assignment changed after it was fixed")
        entries, visited = [], set()
        for row in rows:
            path = directory / row["directory"]
            if hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest() != row["manifest_sha256"]:
                raise EpisodeError(f"the manifest of episode {row['episode_id']} differs from the set manifest")
            entry = episodes.open_episode(path)
            labels = entry.manifest.get("labels") or {}
            if entry.episode_id != row["episode_id"] or labels.get("trajectory") != row["trajectory"] \
                    or labels.get("group") != row["group"] or declared.get(row["trajectory"]) != row["group"]:
                raise EpisodeError(f"episode {row['episode_id']} is not in the group of its trajectory")
            if {name: item["sha256"] for name, item in entry.manifest["files"].items()} != row["files"]:
                raise EpisodeError(f"the files of episode {row['episode_id']} differ from the set manifest")
            for truth in entry.manifest["cases"][1]["truth"]:
                if truth.get("cell") != cell(manifest["config"], truth["position_m"][0], truth["position_m"][1]):
                    raise EpisodeError(f"a position label of episode {row['episode_id']} is not its grid cell")
            if row["group"] == "training":
                visited.update(tuple(value) for value in labels["cells"])
            entries.append(entry)
        counts = manifest["counts"]
        if counts["episodes"] != len(rows) or counts["target_present_runs"] != counts["target_absent_runs"] \
                or counts["by_group"] != {group: sum(row["group"] == group for row in rows) for group in GROUPS} \
                or len({row["episode_id"] for row in rows}) != len(rows) \
                or manifest["coverage"]["by_group"]["training"] != len(visited):
            raise EpisodeError("the counts of the set manifest differ from its episodes")
    except (KeyError, TypeError, OSError) as error:
        raise EpisodeError(f"the set manifest is incomplete: {error!r}") from None
    return EpisodeSet(directory, manifest, entries)
