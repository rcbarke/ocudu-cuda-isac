#!/usr/bin/env python3
"""Sprint 3.3 integration driver (selector pilot_features). Hardware command, not part of unit discovery.

It turns the observations of the reference pair into radar cubes, delay-Doppler views and feature
vectors with the reference of the sensing core, three ways, and holds them against each other and
against the geometry of the episode. The control is the accepted Sprint 3.2 run: the records that
its native consumer kept from the ring in the mode timed, and the files that its observer wrote
for the same estimates in the mode audit. Eight cases, each graded on its own:

  control       the accepted Sprint 3.2 run, its two runs of the pair, and the episode entry they replayed
  stream        the records the consumer kept, pushed through the causal stream in the order of the
                ring: the windows, their outcomes and the registry entries that computed them
  offline       the same O2 from the files of the audited run, cut into complete windows with every
                record in hand and transformed by explicit Fourier sums: the same eligible windows,
                the same view and the same vector
  known-return  the return of the target-present member lies within one resolvable bin of the delay
                and the Doppler shift that the geometry of the episode gives, in the cell the card
                names, and the target-absent member holds no such cell
  causality     a window that is filling carries no vector, and records of the future, fields outside
                the input list and an earlier epoch leave a window unchanged
  mask          a masked element behind a sentinel changes nothing, a hole is counted, and a missing
                observation is a gap without a vector
  live          the tapped receiver replay publishes the pair again and the reference sensor follows
                the ring in a second process: the features of the recorded stream, bit for bit, and
                the decodes of the control
  refusals      every rule of the specification, refused by the check and by the command line

    python tests/integration/pilot_features.py --config configs/integration/pilot_features.json \\
        --s32-run results/runs/s32-spark-001 --output-dir results/runs/s33-spark-001

The output directory must be new. The exit status is 0 only when every check of every case passed.
On a host without a CUDA build, --host-rehearsal runs the same cases on a rehearsed Sprint 3.2 run.
Such a run is graded as a rehearsal and cannot close the hardware gate. Every number of a feature
here carries runtime "reference": NumPy on the CPU. No latency of this driver is a result.
"""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from ocudu_cuda_isac.episodes import episode as episodes  # noqa: E402
from ocudu_cuda_isac.episodes import grid_apply as replay  # noqa: E402
from ocudu_cuda_isac.episodes import replay_pusch as rp  # noqa: E402
from ocudu_cuda_isac.episodes import snapshot_ring as sr  # noqa: E402
from ocudu_cuda_isac.episodes.episode import EpisodeError  # noqa: E402
from ocudu_cuda_isac.episodes.grid_apply import GridApplyError  # noqa: E402
from ocudu_cuda_isac.episodes.replay_pusch import ReplayPuschError  # noqa: E402
from ocudu_cuda_isac.episodes.snapshot_ring import SnapshotRingError  # noqa: E402
from ocudu_cuda_isac.replay import ContractError, load_json  # noqa: E402
from ocudu_cuda_isac.sensing import obs  # noqa: E402
from ocudu_cuda_isac.sensing.obs import ObsError  # noqa: E402
import native_adapter  # noqa: E402

CONFIG_SCHEMA = "integration-config/pilot_features/v1"
GRADE_SCHEMA = "integration-grade/pilot_features/v1"
CONTROL_GRADE_SCHEMA = "integration-grade/snapshot_ring/v1"
CASES = ("control", "stream", "offline", "known-return", "causality", "mask", "live", "refusals")
TOOLS = ("pilot_ls",)
NATIVE_TIMEOUT_S = 1800
SENSOR_GRACE_S = 120
SENTINEL = 1e30


class ConfigError(ValueError):
    """The configuration or the run setup is unusable. Nothing was graded."""


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ring_driver = load_script("snapshot_ring_driver", ROOT / "tests/integration/snapshot_ring.py")
digest, write_json, Grader = ring_driver.digest, ring_driver.write_json, ring_driver.Grader
last_line, git_commit = ring_driver.last_line, ring_driver.git_commit
Run, run_is_clean, decodes_check, account_text = (ring_driver.Run, ring_driver.run_is_clean, ring_driver.decodes_check,
                                                  ring_driver.account_text)


class Grade(Grader):
    """The Sprint 2.1 grader with this sprint's label."""

    def check(self, case, name, passed, detail):
        self.results.append({"case": case, "check": name, "passed": bool(passed), "detail": detail})
        self._line(("PASS" if passed else "FAIL") + f"  S3.3 {case} {name}: {detail}")
        return bool(passed)

    def note(self, case, text):
        self._line(f"NOTE  S3.3 {case} {text}")


def load_config(path, features):
    try:
        config = load_json(Path(path))
    except (ContractError, OSError) as error:
        raise ConfigError(f"cannot read the configuration: {error}") from None
    needed = {"schema", "selector", "adapter_build", "backends", "control", "ring", "spec", "tolerance", "known_return",
              "mask", "pair"}
    if not isinstance(config, dict) or set(config) != needed or config["schema"] != CONFIG_SCHEMA \
            or config["selector"] != "pilot_features":
        raise ConfigError(f"the configuration must be a {CONFIG_SCHEMA} object for the selector pilot_features")
    if list(config["backends"]) != list(rp.BACKENDS):
        raise ConfigError('backends must be ["cpu", "cuda"]: the live case decodes every grid with both')
    control = config["control"]
    if not isinstance(control, dict) or set(control) != {"driver", "audit_run", "timed_run"} \
            or control["driver"] != "snapshot_ring" or not all(isinstance(control[name], str) and control[name]
                                                               for name in ("audit_run", "timed_run")):
        raise ConfigError("control requires the driver snapshot_ring and the names of its audited and its timed run")
    ring = config["ring"]
    if not isinstance(ring, dict) or set(ring) != {"stream", "window_records", "burst_records", "slots", "wait",
                                                   "attach_timeout_ms", "drain_ms"} or ring["wait"] not in sr.WAITS:
        raise ConfigError("ring requires stream, window_records, burst_records, slots, wait, attach_timeout_ms and "
                          "drain_ms")
    try:
        if ring["slots"] != obs.slots_for(ring["window_records"], ring["burst_records"]):
            raise ConfigError(f"ring.slots is {ring['slots']}, and the window and the burst of the configuration need "
                              f"{obs.slots_for(ring['window_records'], ring['burst_records'])}")
        features.check_spec(config["spec"])
    except (ObsError, features.FeaturesError) as error:
        raise ConfigError(str(error)) from None
    tolerance = config["tolerance"]
    if not isinstance(tolerance, dict) or set(tolerance) != {"power_relative", "feature_absolute"} \
            or any(type(value) is not float or not value > 0 for value in tolerance.values()):
        raise ConfigError("tolerance requires power_relative and feature_absolute, each a positive number")
    known = config["known_return"]
    if not isinstance(known, dict) or set(known) != {"delay_bin", "doppler_bin", "min_peak_to_median_db",
                                                     "max_absent_peak_to_median_db", "min_mirror_db"} \
            or type(known["delay_bin"]) is not int or type(known["doppler_bin"]) is not int:
        raise ConfigError("known_return requires delay_bin, doppler_bin, min_peak_to_median_db, "
                          "max_absent_peak_to_median_db and min_mirror_db")
    mask = config["mask"]
    if not isinstance(mask, dict) or set(mask) != {"elements"} or type(mask["elements"]) is not int or mask["elements"] < 1:
        raise ConfigError("mask requires elements, the number of elements to mask in one observation")
    pair = config["pair"]
    if not isinstance(pair, dict) or set(pair) != {"expect"} or not isinstance(pair["expect"], dict) \
            or set(pair["expect"]) != {"opportunities", "complete_windows", "filling", "outside_sequence"}:
        raise ConfigError("pair requires expect with opportunities, complete_windows, filling and outside_sequence")
    return config


# ---------------------------------------------------------------- processes
def run_live(tools, plan_path, native, directory, sensor_dir, spec_path, wait):
    """Run pilot_ls on one plan with the reference sensor as its consumer. Returns (status, report, sensor status).

    The sensor is started before the tool and retries until the tool has opened its channel.
    native and sensor_dir are created empty.
    """
    native.mkdir(parents=True)
    sensor_dir.mkdir(parents=True)
    plan = json.loads(Path(plan_path).read_text())
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    sensor_command = [sys.executable, "-m", "ocudu_cuda_isac.sensing.features", "follow", "--socket",
                      str(sr.socket_path(native, plan)), "--spec", str(spec_path), "--output-dir", str(sensor_dir),
                      "--wait", wait, "--connect-ms", "30000", "--idle-ms", "60000"]
    command = [str(tools["pilot_ls"]), "--plan", str(plan_path), "--output-dir", str(native)]
    sensor = None
    try:
        with (directory / "sensor.log").open("w") as sensor_log, (directory / "native.log").open("w") as stream:
            sensor_log.write(shlex.join(sensor_command) + "\n")
            sensor_log.flush()
            sensor = subprocess.Popen(sensor_command, stdout=sensor_log, stderr=subprocess.STDOUT, env=environment)
            stream.write(shlex.join(command) + "\n")
            stream.flush()
            try:
                status = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=NATIVE_TIMEOUT_S,
                                        check=False).returncode
            except subprocess.TimeoutExpired:
                stream.write("TimeoutExpired: the tool was stopped\n")
                status = -1
            try:
                sensor_status = sensor.wait(timeout=SENSOR_GRACE_S)
            except subprocess.TimeoutExpired:
                sensor.kill()
                sensor_status = -1
    except OSError as error:
        (directory / "native.log").open("a").write(f"{type(error).__name__}: {error}\n")
        if sensor is not None:
            sensor.kill()
        return -1, None, -1
    try:
        return status, json.loads((native / "native.json").read_text()), sensor_status
    except (OSError, ValueError):
        return status, None, sensor_status


class Control:
    """The accepted Sprint 3.2 run: its audited and its timed run of the pair, and what they replayed."""

    def __init__(self):
        self.audit = self.timed = self.archive = self.radio = self.replay_plan_path = self.channel = None
        self.timed_plan = self.entry = self.cases = None
        self.estimates = {}


def text(value, digits=1):
    return "none" if value is None else f"{value:.{digits}f}"


def named(line):
    return dict(zip(line["features"]["names"], line["features"]["values"]))


def complete(lines):
    return [line for line in lines if line["status"] == "ok"]


def statuses(lines):
    counts = {}
    for line in lines:
        counts[line["status"]] = counts.get(line["status"], 0) + 1
    return counts


def same_lines(features, mine, other):
    """The first difference between two streams on the same records, host times aside, or None."""
    if len(mine) != len(other):
        return f"{len(mine)} lines against {len(other)}"
    for number, (first, second) in enumerate(zip(mine, other)):
        first, second = features.comparable(first), features.comparable(second)
        if first != second:
            keys = sorted(key for key in set(first) | set(second) if first.get(key) != second.get(key))
            return f"line {number} differs in {', '.join(keys)}"
    return None


def changed_record(record, payload=None, mask=None, host=None, **header):
    """A copy of a record with some of its parts replaced."""
    made = dict(record.header, **header)
    payload = record.payload if payload is None else payload
    mask = record.mask if mask is None else mask
    host = record.host if host is None else host
    made.update(payload_bytes=len(payload), mask_bytes=len(mask), host_bytes=len(host))
    return obs.Record(made, host, mask, payload, depth=record.depth)


# ---------------------------------------------------------------- control
def run_control(config, grader, s32_out, rehearsal, evaluator):
    """The accepted Sprint 3.2 run, its two runs of the pair and the episode entry they replayed."""
    name, control, measured = "control", Control(), {}
    try:
        grade = load_json(s32_out / "grade.json")
        manifest = load_json(s32_out / "manifest.json")
    except (ContractError, OSError) as error:
        grader.check(name, "accepted run", False, f"{s32_out} holds no grade and manifest of the Sprint 3.2 driver: {error}")
        return None, measured
    wanted = "rehearsal" if rehearsal else "pass"
    accepted = isinstance(grade, dict) and grade.get("schema") == CONTROL_GRADE_SCHEMA \
        and grade.get("judgment") == wanted and grade.get("cases_passed") == grade.get("cases") \
        and grade.get("checks_failed") == 0 and (rehearsal or grade.get("hardware_gate") == "closed")
    measured["s32"] = {"run_id": manifest.get("run_id"), "judgment": grade.get("judgment"),
                       "hardware_gate": grade.get("hardware_gate"), "parent_commit": manifest.get("parent_commit"),
                       "record_definition": grade.get("record_definition"), "s31_run": manifest.get("s31_run"),
                       "adapter": (manifest.get("adapter") or {}).get("executables")}
    grader.check(name, "accepted run", accepted,
                 f"the Sprint 3.2 driver graded {s32_out.name} as {grade.get('judgment')}: {grade.get('cases_passed')} of "
                 f"{grade.get('cases')} cases, {grade.get('checks_failed')} failed checks, hardware gate "
                 f"{str(grade.get('hardware_gate')).split(':')[0]}, at commit {str(manifest.get('parent_commit'))[:7]}"
                 + ("" if accepted else f"; this gate needs the judgment {wanted}"
                    + ("" if rehearsal else " with the hardware gate closed")))
    listed = {item["name"]: item for item in manifest.get("runs", []) if isinstance(item, dict)}
    try:
        found = {}
        for role in ("audit_run", "timed_run"):
            item = listed.get(config["control"][role])
            if item is None or item.get("status") != 0:
                raise SnapshotRingError(f"the Sprint 3.2 run holds no finished run named {config['control'][role]}")
            directory = s32_out / item["directory"]
            if digest(directory / "plan.json") != item.get("plan_sha256"):
                raise SnapshotRingError(f"the plan of the run {item['name']} does not have the hash its manifest names")
            found[role] = (directory, sr.load_plan(directory / "plan.json"), sr.open_output(directory / "native"))
        (_, audit_plan, control.audit), (timed_dir, control.timed_plan, control.timed) = found["audit_run"], found["timed_run"]
        if control.audit.mode != "audit" or control.audit.observer is None or control.timed.mode != "timed":
            raise SnapshotRingError("the two runs are not an audited run with its observer files and a timed run")
        if audit_plan["replay_plan"] != control.timed_plan["replay_plan"]:
            raise SnapshotRingError("the audited and the timed run did not decode the same replay plan")
        control.replay_plan_path = Path(control.timed_plan["replay_plan"]["path"])
        if digest(control.replay_plan_path) != control.timed_plan["replay_plan"]["sha256"]:
            raise SnapshotRingError("the replay plan of the Sprint 2.4 run does not have the hash the timed plan binds")
        plan = rp.load_plan(control.replay_plan_path)
        control.channel = replay.load_plan(plan["channel_plan"]["path"])
        control.radio = rp.open_output(control.replay_plan_path.parent / "native")
        control.archive = obs.open_archive(timed_dir / "consumer-0")
    except (SnapshotRingError, ReplayPuschError, GridApplyError, ObsError, OSError, KeyError) as error:
        grader.check(name, "records", False, f"{error}")
        return None, measured
    control.estimates = {record["host"]["opportunity_id"]: (record["payload"]["digest"], record["validity"]["sha256"])
                         for record in control.audit.observer.observations}
    pairs, problems = sr.join_archive(control.timed, control.archive)
    digests = sr.payload_digests(pairs)
    equal = sum(control.estimates.get(opportunity) == made for opportunity, made in digests.items())
    expected = config["pair"]["expect"]["opportunities"]
    whole = not problems and equal == len(pairs) == len(control.estimates) == expected
    measured.update(records=len(control.archive.records), observations=len(control.estimates),
                    replay_plan=str(control.replay_plan_path))
    grader.check(name, "records", whole,
                 f"the runs {config['control']['audit_run']} and {config['control']['timed_run']} of the Sprint 3.2 gate "
                 f"verify with the reader of that sprint and decoded one replay plan; the native consumer of the timed "
                 f"run kept {len(pairs)} records, each the record of its account line, and {equal} of {len(pairs)} "
                 f"payloads and validity masks, hashed here, have the SHA-256 of the file that the observer of the "
                 f"audited run wrote for the same opportunity: one O2, once from the ring and once from files; expected "
                 f"{expected} observations" if not problems else "; ".join(problems[:3]))
    if not whole:
        return None, measured
    # The episode entry behind the plan. It holds the truth, and only the cases below that say so read it.
    try:
        control.cases = evaluator.realizations(control.channel)
        entries = {str(item["entry"]) for item in control.cases.values()}
        if len(entries) != 1 or sorted(item["case"] for item in control.cases.values()) != ["a", "b"]:
            raise evaluator.PilotFeaturesError("the channel plan does not name the two cases of one episode entry")
        control.entry = episodes.open_episode(Path(entries.pop()))
    except (evaluator.PilotFeaturesError, EpisodeError, OSError) as error:
        grader.check(name, "episode entry", False, f"{error}")
        return None, measured
    keys = {f"{record.header['episode']:016x}" for record in control.archive.records}
    measured["entry"] = {"episode_id": control.entry.episode_id, "directory": str(control.entry.directory),
                         "opportunities": len(control.entry.schedule.opportunities)}
    joined = keys == set(control.cases)
    grader.check(name, "episode entry", joined,
                 f"the channel plan names the cases a and b of the entry {control.entry.episode_id}, which verifies: "
                 f"{len(control.entry.schedule.opportunities)} opportunities per case; the {len(keys)} episode keys of "
                 "the records are the keys of its two realizations, and a key names no case: the join is the "
                 "evaluator's")
    return (control if joined else None), measured


# ---------------------------------------------------------------- stream
def run_stream(config, out, grader, control, features):
    """The recorded ring through the causal stream, as a sensor would have seen it."""
    name, spec, expect = "stream", config["spec"], config["pair"]["expect"]
    directory = out / "stream" / "sensor"
    directory.mkdir(parents=True)
    try:
        report = features.replay(control.archive.directory, spec, directory)
        output = features.open_output(directory)
    except (features.FeaturesError, ObsError) as error:
        grader.check(name, "output", False, f"the stream on the recorded ring left no output that verifies: {error}")
        return {}, None
    counts = report["counts"]
    whole = (counts["records"], counts["ok"], counts["insufficient_history"], counts["passed"]) \
        == (expect["opportunities"], expect["complete_windows"], expect["filling"], expect["outside_sequence"]) \
        and not counts["gap"] and not counts["invalid"] and not counts["rejected"]
    grader.check(name, "output", whole,
                 f"{counts['records']} records of the ring in sequence order: {counts['ok']} complete windows, "
                 f"{counts['insufficient_history']} answers of a filling window, {counts['passed']} records outside the "
                 f"sequence, {counts['gap']} gaps, {counts['invalid']} invalid, {counts['rejected']} rejected, "
                 f"histories started anew: {counts['resets']}; expected {expect['complete_windows']}, {expect['filling']} and "
                 f"{expect['outside_sequence']}; the output verifies with its reader: every view has the hash and the "
                 "peak its line names")
    if not output.ok:
        return {"counts": counts}, output
    line = output.ok[0]
    axes, need = line["axes"], line["history"]["need"]
    answered = all(len({item["sequence"] for item in window["contributing"]}) == len(window["contributing"])
                   and window["history"] == {"have": need, "need": need}
                   and window["trigger"]["sequence"] == window["contributing"][-1]["sequence"]
                   and window["validity"]["holes"] == 0 for window in output.ok)
    episodes_seen = {window["window"]["episode"] for window in output.ok}
    grader.check(name, "windows", answered and len(episodes_seen) == len(output.ok),
                 f"each of the {len(output.ok)} complete windows was answered at the record that completed it, from "
                 f"{need} samples of {len(line['contributing'])} observations with every element valid, "
                 f"{line['time']['acquisition_us'] / 1000:.2f} ms from the first to the last sample; one window per "
                 f"episode key, and no window spans two keys")
    steps = report["steps"]
    devices = {(entry["device"], entry["library"], entry["runtime"]) for entry in steps.values()}
    grader.check(name, "registry", devices == {("cpu", "numpy", "reference")} and report["runtime"]["entry"] == "reference",
                 "the steps of the specification are registry entries, each with its execution device: "
                 + ", ".join(f"{stage} {entry['entry']}" for stage, entry in steps.items())
                 + f", all in NumPy {report['runtime']['numpy']} on the CPU with the runtime reference")
    grader.note(name, f"axes of a window, signed: delay {axes['delay']['bins']} bins of "
                      f"{axes['delay']['resolution_ns']:.2f} ns, unambiguous to {axes['delay']['unambiguous_ns'] / 1000:.2f} "
                      f"us, {axes['delay']['kept_ns']:.0f} ns kept; Doppler bins {axes['doppler']['bins'][0]} to "
                      f"{axes['doppler']['bins'][-1]} of {axes['doppler']['resolution_hz']:.2f} Hz, unambiguous from "
                      f"{axes['doppler']['unambiguous_hz'][0]:.1f} to {axes['doppler']['unambiguous_hz'][1]:.1f} Hz; "
                      f"{axes['frequency']['count']} pilots {axes['frequency']['spacing_hz'] / 1000:.0f} kHz apart on "
                      f"elements {axes['rx']['elements']}; a later arrival lies at a higher delay bin, a receding target "
                      "at a negative Doppler bin")
    return {"counts": counts, "spec_sha256": report["spec_sha256"], "axes": axes,
            "windows": [{"id": window["window"]["id"], "peak": window["view"]["peak"], "features": named(window),
                         "view_sha256": window["view"]["sha256"]} for window in output.ok]}, output


# ---------------------------------------------------------------- offline
def run_offline(config, grader, control, stream, features, evaluator):
    """The same O2 from the files of the audited run, with every record in hand and explicit Fourier sums."""
    name, spec, tolerance = "offline", config["spec"], config["tolerance"]
    found = evaluator.observations(control.audit.observer)
    windows = evaluator.offline_windows(found, spec)
    eligible = [window for window in windows if window["eligible"]]
    mine = {line["window"]["id"]: [item["opportunity"] for item in line["contributing"]] for line in stream.ok}
    theirs = {window["id"]: window["contributing"] for window in eligible}
    grader.check(name, "eligible windows", mine == theirs and len(eligible) == len(windows) and mine,
                 f"{len(found)} observations read from the files of the audited run and cut by slot arithmetic give "
                 f"{len(eligible)} eligible windows of {len(windows)}; the stream answered {len(mine)} complete windows "
                 "at the records of the ring: the same window identifiers, each with the same contributing "
                 "opportunities in the same order" if mine == theirs else
                 f"the stream has the windows {sorted(mine)}, the offline cut {sorted(theirs)}, or their contributing "
                 "opportunities differ")
    if mine != theirs or not mine:
        return {}
    problems, worst, largest = [], 0.0, 0.0
    np = features.np
    for window in eligible:
        line = stream.by_window[window["id"]]
        made = evaluator.oracle(window, spec)
        power = stream.view(line)
        problems += [f"{window['id']}: {problem}" for problem in evaluator.differences(line, power, made, tolerance)]
        if power.shape == made["power"].shape:
            worst = max(worst, float(np.abs(power - made["power"]).max() / made["power"].max()))
            if line["features"]["names"] == made["names"]:
                largest = max([largest] + [abs(a - b) for a, b in zip(line["features"]["values"], made["values"])])
    count = len(stream.ok[0]["features"]["names"])
    grader.check(name, "feature values", not problems,
                 f"the view and the vector of each window against explicit Fourier sums in double precision at the "
                 f"physical frequencies and at the sample times of the records: the largest difference of a cell is "
                 f"{worst:.1e} of the peak (tolerance {tolerance['power_relative']:.0e}) and of one of the {count} "
                 f"features {largest:.1e} in its unit (tolerance {tolerance['feature_absolute']:.0e}); the peaks lie in "
                 "the same cells" if not problems else f"{len(problems)} differences, first: {problems[0]}")
    return {"windows": len(eligible), "largest_cell_difference": worst, "largest_feature_difference": largest}


# ---------------------------------------------------------------- known return
def run_known_return(config, out, grader, control, stream, features, evaluator):
    """The return of the present member against the geometry of the episode. This case reads the truth."""
    name, known = "known-return", config["known_return"]
    by_case = {}
    for line in stream.ok:
        realization = control.cases.get(line["window"]["episode"])
        if realization is not None:
            by_case.setdefault(realization["case"], []).append((line, realization))
    if sorted(by_case) != ["a", "b"]:
        grader.check(name, "members", False, "the complete windows do not cover both members of the pair")
        return {}
    (present, realization), (absent, _) = by_case["b"][0], by_case["a"][0]
    indices = [realization["opportunities"][item["opportunity"]] for item in present["contributing"]]
    first, last = (evaluator.expected_return(control.entry, index) for index in (indices[0], indices[-1]))
    peak, axes = present["view"]["peak"], present["axes"]
    at_cell = (peak["delay_bin"], peak["doppler_bin"]) == (known["delay_bin"], known["doppler_bin"])
    grader.check(name, "declared cell", at_cell,
                 f"the strongest moving cell of the target-present member lies at delay bin {peak['delay_bin']} and "
                 f"Doppler bin {peak['doppler_bin']}, {peak['delay_ns']:.1f} ns and {peak['doppler_hz']:+.1f} Hz; the "
                 f"card names delay bin {known['delay_bin']} and Doppler bin {known['doppler_bin']}")
    passed, facts = evaluator.resolved(present, first)
    grader.check(name, "one resolvable bin", passed,
                 f"the geometry of the entry gives a return at {first['delay_ns']:.1f} ns and "
                 f"{first['doppler_hz']:+.1f} Hz at opportunity {indices[0]}, "
                 f"{first['excess_delay_ns']:.1f} ns behind the direct path"
                 + (f", folded to {facts['expected_doppler_folded_hz']:+.1f} Hz in the unambiguous interval"
                    if facts["aliased"] else "")
                 + f"; the peak is {facts['delay_error_ns']:+.1f} ns and {facts['doppler_error_hz']:+.1f} Hz from it, "
                 f"within one bin of {facts['delay_resolution_ns']:.1f} ns for {axes['frequency']['span_hz'] / 1e6:.2f} MHz "
                 f"and of {facts['doppler_resolution_hz']:.1f} Hz for {axes['slow_time']['samples']} samples over "
                 f"{axes['slow_time']['s'][-1] * 1000:.0f} ms")
    strong, weak = named(present)["peak_to_median_db"], named(absent)["peak_to_median_db"]
    grader.check(name, "present against absent",
                 strong >= known["min_peak_to_median_db"] and weak <= known["max_absent_peak_to_median_db"],
                 f"the peak of the present member stands {strong:.1f} dB above the median of its view (at least "
                 f"{known['min_peak_to_median_db']:.1f}), and the strongest cell of the absent member {weak:.1f} dB above "
                 f"its own (at most {known['max_absent_peak_to_median_db']:.1f}): the absent spectrum is kept and holds no "
                 "such cell")
    power, bins = stream.view(present), axes["doppler"]["bins"]
    measured = {"present": {"window": present["window"]["id"], "peak": peak, "features": named(present)},
                "absent": {"window": absent["window"]["id"], "peak": absent["view"]["peak"], "features": named(absent)},
                "expected_first": first, "expected_last": last, "resolved": facts}
    # The sign of the Doppler axis: the peak has the sign that the geometry gives, and where the axis
    # holds the mirrored cell, that cell is far weaker. A window of two samples has no mirrored cell.
    mirror, margin = -peak["doppler_bin"], None
    signed = peak["doppler_hz"] * facts["expected_doppler_folded_hz"] > 0
    if mirror in bins and mirror != 0 and mirror != peak["doppler_bin"]:
        margin = 10 * math.log10(float(power[peak["delay_bin"], bins.index(peak["doppler_bin"])])
                                 / float(power[peak["delay_bin"], bins.index(mirror)]))
        measured["mirror_db"] = margin
    grader.check(name, "signed axis", signed and (margin is None or margin >= known["min_mirror_db"]),
                 f"the peak lies at {peak['doppler_hz']:+.1f} Hz, on the side of the {first['doppler_hz']:+.1f} Hz that "
                 "the geometry gives, " + (
                     f"and is {margin:.1f} dB above the cell at Doppler bin {mirror:+d} of the same delay (at least "
                     f"{known['min_mirror_db']:.1f}): a transposed Doppler axis would show the return in that cell"
                     if margin is not None else
                     f"and the axis of this window has no cell at bin {mirror:+d} to compare with: the mirrored cell is "
                     "graded on a window of eight periods"))
    static = present["view"]["static_peak"]
    grader.note(name, f"recorded, not graded: the static part peaks at delay bin "
                      f"{static['delay_bin'] if static else 'none'}, where the direct path of {first['direct_delay_ns']:.1f} "
                      f"ns lies; the return moves {last['delay_ns'] - first['delay_ns']:+.2f} ns and "
                      f"{last['doppler_hz'] - first['doppler_hz']:+.2f} Hz over the window; moving over static power "
                      f"{named(present)['moving_to_static_db']:.1f} dB with the target and "
                      f"{named(absent)['moving_to_static_db']:.1f} dB without; Doppler centroid "
                      f"{named(present)['doppler_centroid_hz']:+.1f} Hz; the share of the moving power in the peak cell "
                      f"is {present['view']['leakage']['peak_fraction']:.2f}, the rest is leakage of a return between "
                      "bins and noise")
    write_json(out / "known_return.json", measured)
    return measured


# ---------------------------------------------------------------- causality
def run_causality(config, grader, control, stream, features):
    """What may not change a window: the future, fields outside the input list, an earlier epoch."""
    name, spec = "causality", config["spec"]
    records = control.archive.records
    reference, _, _ = features.run_records(records, spec)
    present = [line for line in complete(reference)
               if (control.cases.get(line["window"]["episode"]) or {}).get("case") == "b"]
    if not present:
        grader.check(name, "window", False, "the stream answered no complete window of the target-present member")
        return {}
    window = present[0]
    members = [item["sequence"] for item in window["contributing"]]
    episode = int(window["window"]["episode"], 16)
    own = [record for record in records if record.header["episode"] == episode]
    last = max(index for index, record in enumerate(own) if record.sequence == members[-1])

    def answer(lines):
        found = [line for line in lines if line["window"] and line["window"]["id"] == window["window"]["id"]
                 and line["status"] == "ok"]
        return features.comparable(found[0]) if found else None

    wanted = features.comparable(window)
    del wanted["line"]

    def same(lines):
        made = answer(lines)
        return made is not None and {key: value for key, value in made.items() if key != "line"} == wanted

    # ---- A window that is filling answers at every sample and carries no vector.
    early, _, held = features.run_records(own[:last], spec)
    filling = [line for line in early if line["status"] == "insufficient_history"]
    need = window["history"]["need"]
    grader.check(name, "insufficient history",
                 len(filling) == len(early) == len(members) - 1 and not any("features" in line for line in early)
                 and [line["history"]["have"] for line in early] == list(range(1, len(members))),
                 f"the records of the present member before the one that completes its window are answered "
                 f"insufficient_history {len(filling)} times, with {', '.join(str(line['history']['have']) for line in early)} "
                 f"of {need} samples, and no answer carries a view or a vector; samples left open: "
                 f"{held.open_samples()[0]}")
    # ---- Records of the future: everything after the completing record is replaced by a sentinel.
    big = features.np.full(len(own[0].payload) // 8, SENTINEL * (1 + 1j), dtype="<c8").tobytes()
    future = own[:last + 1] + [changed_record(record, payload=big) for record in own[last + 1:]]
    lines, _, _ = features.run_records(future, spec)
    grader.check(name, "future samples", same(lines) and len(own) > last + 1,
                 f"with the payload of the {len(own) - last - 1} records after the completing one replaced by "
                 f"{SENTINEL:.0e}, the window is answered with the same line: the same contributing records, the same "
                 "peak and the same vector, bit for bit. The stream answered before it saw them")
    # ---- Fields that a stream does not read.
    hidden = [changed_record(record, host=b"evaluator only: target present", grant=record.header["grant"] ^ 0xFFFF,
                             cell=record.header["cell"] + 1, ue=record.header["ue"] + 1, producer_steps=15,
                             digest_kind=1, digest=bytes(range(32)), valid=0, reserved=1) for record in own]
    lines, _, _ = features.run_records(hidden, spec)
    outside = [field for field, _, _ in obs.FIELDS if field not in features.INPUT_FIELDS]
    grader.check(name, "input fields", same(lines),
                 f"with the host block, the grant key, the cell, the UE, the producer steps, the digest and the valid "
                 f"count of every record changed, the window is the same: a stream reads {len(features.INPUT_FIELDS)} "
                 f"fields of the header, the validity bits and the payload, and not {', '.join(outside)}")
    # ---- An earlier epoch: its records fill most of a window and are never used.
    epoch = own[0].header["epoch"]
    older = [changed_record(record, payload=big, epoch=epoch - 1) for record in own[:last]]
    lines, _, later = features.run_records(older + own, spec)
    used = {item["sequence"] for line in lines if line["status"] == "ok" for item in line["contributing"]}
    before = [line for line in lines if line["window"] and line["window"]["epoch"] == epoch - 1]
    grader.check(name, "earlier epoch", same(lines) and later.counts["resets"] == 1
                 and all(line["status"] == "insufficient_history" for line in before) and len(before) == len(members) - 1,
                 f"{len(older)} records of epoch {epoch - 1} with a payload of {SENTINEL:.0e} fill "
                 f"{len(before)} of {len(members)} observations of a window; the first record of epoch {epoch} starts a "
                 f"new history, and the window of epoch {epoch} is the same, from the sequences "
                 f"{sorted(used)[0]} to {sorted(used)[-1]} of its own epoch")
    # ---- The check is not empty: a contributing sample does change the window.
    values = features.np.frombuffer(own[last].payload, dtype="<c8") * 1.5
    touched = own[:last] + [changed_record(own[last], payload=values.astype("<c8").tobytes())] + own[last + 1:]
    lines, _, _ = features.run_records(touched, spec)
    made = answer(lines)
    grader.check(name, "sensitivity", made is not None and made["features"]["values"] != window["features"]["values"],
                 "with the payload of the completing record scaled by 1.5 the vector changes: the comparisons above can "
                 "tell a window that read another sample")
    return {"window": window["window"]["id"], "contributing": members, "records": len(own)}


# ---------------------------------------------------------------- mask
def run_mask(config, grader, control, features):
    """A masked element behind a sentinel, a counted hole, and a missing observation."""
    name, spec, count = "mask", config["spec"], config["mask"]["elements"]
    np = features.np
    records = control.archive.records
    reference, views, _ = features.run_records(records, spec)
    present = [(line, power) for line, (_, power) in zip(complete(reference), views)
               if (control.cases.get(line["window"]["episode"]) or {}).get("case") == "b"]
    if not present:
        grader.check(name, "window", False, "the stream answered no complete window of the target-present member")
        return {}
    window, full = present[0]
    position = (len(window["contributing"]) - 1) // 2
    interior = 0 < position < len(window["contributing"]) - 1
    target = window["contributing"][position]["sequence"]
    index = next(index for index, record in enumerate(records)
                 if record.sequence == target and f"{record.header['episode']:016x}" == window["window"]["episode"])
    record = records[index]
    shape = record.shape
    row = spec["sequence"]["reference_symbols"][0]
    # The elements to mask lie on the reference symbol that the sequence selects, spread over the comb.
    chosen = [(plane % shape[0], row, (37 * plane + 11) % shape[2]) for plane in range(count)]
    flat = [(plane * shape[1] + symbol) * shape[2] + pilot for plane, symbol, pilot in chosen]
    bits = bytearray(record.mask)
    for element in flat:
        bits[element // 8] &= ~(1 << (element % 8)) & 0xFF
    outcomes = []
    for stored in (0.0, SENTINEL):
        values = np.frombuffer(record.payload, dtype="<c8").copy()
        values[flat] = stored * (1 + 1j)
        changed = records[:index] + [changed_record(record, payload=values.tobytes(), mask=bytes(bits))] + records[index + 1:]
        lines, made, _ = features.run_records(changed, spec)
        line = [line for line in lines if line["window"] and line["window"]["id"] == window["window"]["id"]][-1]
        power = [power for number, power in made if number == line["line"]]
        outcomes.append((features.comparable(line), power[0] if power else None))
    (zero, zero_power), (big, big_power) = outcomes
    equal = zero == big and zero_power is not None and np.array_equal(zero_power, big_power)
    validity = zero.get("validity", {})
    grader.check(name, "sentinel", equal and zero["status"] == "ok" and validity.get("holes") == len(set(flat)),
                 f"{len(set(flat))} elements of observation {target} of the present window lose their validity bit; "
                 f"with {SENTINEL:.0e} stored behind them the line, the view and the vector are those with zero stored, "
                 f"bit for bit: a masked value is never read; the window counts {validity.get('holes')} holes on "
                 f"{validity.get('lines_with_holes')} lines of the cube and stays complete at a fill of "
                 f"{validity.get('fill', 0):.4f}")
    if zero["status"] == "ok":
        same_cell = (zero["view"]["peak"]["delay_bin"], zero["view"]["peak"]["doppler_bin"]) \
            == (window["view"]["peak"]["delay_bin"], window["view"]["peak"]["doppler_bin"])
        shift = float(np.abs(zero_power - full).max() / full.max())
        grader.check(name, "valid support", same_cell and 0 < shift < 0.5,
                     f"on the valid support the return stays in its cell; the holes change the view by at most "
                     f"{shift:.1e} of its peak, which is their leakage, and the share of the moving power in the peak "
                     f"cell goes from {window['view']['leakage']['peak_fraction']:.3f} to "
                     f"{zero['view']['leakage']['peak_fraction']:.3f}; interpolation: "
                     f"{validity.get('interpolation')}")
    # ---- A missing observation: the record never reached the sensor.
    lines, _, stream = features.run_records(records[:index] + records[index + 1:], spec)
    gaps = [line for line in lines if line["status"] == "gap"]
    period, place = divmod(record.header["absolute_slot"], spec["radio"]["period_slots"])
    others = [line["window"]["id"] for line in complete(lines)]
    if interior:
        told = len(gaps) == 1 and gaps[0]["window"]["id"] == window["window"]["id"] and "features" not in gaps[0] \
            and gaps[0]["missing"] == [{"period": period, "slot_in_period": place, "reference_symbol": symbol}
                                       for symbol in spec["sequence"]["reference_symbols"]]
        said = f"the window closes as a gap that names period {period}, slot {place} as missing and carries no vector"
    else:
        # A window of two observations has no interior one. Without its first observation the
        # history starts a period later and stays a filling window, which is no answer either.
        told = not gaps and stream.counts["insufficient_history"] >= 1
        said = "the history of that member starts one period later and its window stays a filling window without a " \
               "vector: a window of this length has no interior observation, and a gap is graded by the unit gate"
    grader.check(name, "missing observation", told and window["window"]["id"] not in others
                 and len(others) == len(complete(reference)) - 1,
                 f"without observation {target} {said}; other complete windows, answered as before: {len(others)}"
                 if told else f"the stream answered {statuses(lines)} without observation {target}")
    return {"masked": chosen, "observation": target, "holes": validity.get("holes"), "fill": validity.get("fill")}


# ---------------------------------------------------------------- live
def run_live_case(config, out, grader, tools, control, stream, features, rehearsal, runs):
    """The pair published again, with the reference sensor on the ring in a second process."""
    name, backends, settings = "live", config["backends"], config["ring"]
    directory = out / "live" / "run"
    directory.mkdir(parents=True)
    run = Run("live", directory, control)
    runs.append(run)
    sensor_dir, spec_path = directory / "sensor", directory / "spec.json"
    write_json(spec_path, config["spec"])
    ring = sr.ring_spec(settings["window_records"], settings["burst_records"], settings["wait"], "required",
                        settings["attach_timeout_ms"], settings["drain_ms"], settings["stream"])
    observer = control.timed_plan["observer"]
    run.plan = sr.build_plan("s33-live", control.replay_plan_path, "timed", ring, "pilot_ls", epoch=observer["epoch"],
                             source=observer["source"])
    sr.write_plan(run.plan, directory / "plan.json")
    run.status, run.report, sensor_status = run_live(tools, directory / "plan.json", run.native, directory, sensor_dir,
                                                     spec_path, settings["wait"])
    run.statuses = [sensor_status]
    run.consumers = [{"reader": "features", "wait": settings["wait"], "archive": sensor_dir / features.RECORDS_DIR}]
    if run.status == 0 and run.report:
        try:
            run.output = sr.open_output(run.native)
        except SnapshotRingError as error:
            run.problem = f"the native output does not verify: {error}"
    clean = run_is_clean(run, backends) and run.output.mode == "timed"
    grader.check(name, "native run", clean,
                 f"both members of the pair in the mode timed on a ring of {run.output.ring['slot_count']} slots, with "
                 f"the reference sensor attached in the mode {settings['wait']}: {len(run.output.publications)} "
                 f"observations, {run.counts()}; one record per planned decode under a unique key, no anomaly, every "
                 "grid released" if run.ok() else run.said())
    if not run.ok():
        return {}
    measured = {}
    decodes_check(grader, name, run)
    records = run.output.records
    cuda = [record for record in records if record["backend"] == "cuda"]
    host = [record for record in records if record["backend"] == "cpu"]
    resident = sum((record["path"] or {}).get("executed") == "resident" for record in cuda)
    on_host = sum((record["path"] or {}).get("executed") == "host" for record in host)
    if rehearsal:
        grader.check(name, "executed path", on_host == len(host) and len(host) > 0,
                     f"host rehearsal: {on_host} of {len(host)} decodes of the host processor ran the host path; the "
                     f"{len(cuda)} decodes requested from the CUDA processor are not graded on a build without CUDA")
    else:
        built = run.report.get("built_with_cuda") is True
        grader.check(name, "executed path", built and cuda and resident == len(cuda) and on_host == len(host),
                     f"{resident} of {len(cuda)} decodes requested from the CUDA processor ran its resident path with "
                     f"the sensor attached, and {on_host} of {len(host)} decodes of the host processor ran the host "
                     "path; the adapter is built with CUDA" if built else
                     "the adapter is not built with CUDA: the CUDA processor cannot execute")
    # ---- The sensor: what it wrote, what it read, and the account of the producer.
    try:
        if sensor_status != 0:
            raise features.FeaturesError(f"the sensor exited with status {sensor_status}: "
                                         f"{last_line(directory / 'sensor.log')}")
        output = features.open_output(sensor_dir)
        archive = obs.open_archive(sensor_dir / features.RECORDS_DIR)
    except (features.FeaturesError, ObsError) as error:
        grader.check(name, "sensor", False, f"the reference sensor left no output that verifies: {error}")
        return measured
    source = output.report["source"]
    grader.check(name, "sensor", source["kind"] == "ring" and source["ended"] == "closed" and source["stale"] == 0
                 and source["records"] == len(archive.records),
                 f"python -m ocudu_cuda_isac.sensing.features follow, a second process on the standard-library reader, "
                 f"read {source['records']} records of epoch {source['epoch']} from the ring until the producer said "
                 f"that none follows, with {source['stale']} stale reads, and its output verifies")
    states = sr.account(run.output)
    pairs, problems = sr.join_archive(run.output, archive)
    digests = sr.payload_digests(pairs)
    equal = sum(control.estimates.get(opportunity) == made for opportunity, made in digests.items())
    measured["account"] = states
    grader.check(name, "account", states["balanced"] and states["dropped"] == 0 and states["pending"] == 0
                 and not problems and equal == len(pairs) == states["published"],
                 f"{account_text(states)}; the sensor kept {len(pairs)} records, each the record of its account line, "
                 f"and {equal} of {len(pairs)} payloads and validity masks have the SHA-256 of the audited estimate of "
                 "the same opportunity: the sensor computed on the O2 of the control"
                 if not problems else "; ".join(problems[:3]))
    difference = same_lines(features, output.lines, stream.lines)
    hashes = [line["view"]["sha256"] for line in output.ok] == [line["view"]["sha256"] for line in stream.ok]
    grader.check(name, "same features", difference is None and hashes and output.ok,
                 f"the {len(output.lines)} answers of the live sensor are those of the recorded stream of the accepted "
                 f"Sprint 3.2 run, line for line, host times aside: the same windows from the same sequences, "
                 f"{len(output.ok)} views with the same SHA-256 and {len(output.ok)} vectors equal bit for bit"
                 if difference is None else f"the live sensor and the recorded stream differ: {difference}")
    delays = [(line["t_features_ns"] - line["t_publish_ns"]) / 1000.0 for line in output.ok
              if line["t_features_ns"] is not None and line["t_publish_ns"]]
    if delays:
        measured["reference_delay_us"] = sr.spread(delays)
        grader.note(name, f"from the publication store of the completing record to the complete vector, not graded, "
                          f"runtime reference: median {text(obs.percentile(delays, 0.5))} us, largest "
                          f"{text(max(delays))} us over {len(delays)} windows; this is Python and NumPy on the CPU "
                          "behind a reader that sleeps, and no latency of the sensing core: S6.12 times the native units")
    measured.update(windows=len(output.ok), lines=len(output.lines), sensor=str(sensor_dir))
    return measured


# ---------------------------------------------------------------- refusals
def run_refusals(config, out, grader, control, features):
    """Every rule of the specification, by the check and by the command line."""
    name = "refusals"
    directory = out / "refusals"
    directory.mkdir(parents=True)

    def changed(group, key, value):
        spec = json.loads(json.dumps(config["spec"]))
        if group is None:
            spec[key] = value
        else:
            spec[group][key] = value
        return spec

    cases = (("another schema", changed(None, "schema", "pilot-features-spec/v0")),
             ("an unknown receiver stage", changed("input", "stage", "O7")),
             ("a layout without a cube", changed("input", "layout", "ocudu_o3_fp16")),
             ("an unknown subcarrier spacing", changed("radio", "scs_khz", 45)),
             ("a period that is no whole number of subframes", changed("radio", "period_slots", 5)),
             ("a slot outside the period", changed("sequence", "slots_in_period", [config["spec"]["radio"]["period_slots"]])),
             ("a burst of slots under the fft step", changed("sequence", "slots_in_period", [7, 8, 9])),
             ("two reference symbols under the fft step", changed("sequence", "reference_symbols", [0, 1])),
             ("a window of no period", changed("window", "periods", 0)),
             ("a hop of no period", changed("window", "hop_periods", 0)),
             ("a valid share of zero", changed("window", "min_valid_fraction", 0)),
             ("an unregistered static step", changed("steps", "static", "mti_2")),
             ("an unregistered Doppler step", changed("steps", "doppler", "music")),
             ("the ndft step without a Doppler grid", changed("steps", "doppler", "ndft")),
             ("a view without a delay bin", changed("view", "delay_bins", 0)))
    refused, by_tool, kept = 0, 0, []
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    for number, (label, spec) in enumerate(cases):
        try:
            features.check_spec(spec)
        except features.FeaturesError:
            refused += 1
        else:
            kept.append(label)
        path, target = directory / f"spec-{number:02d}.json", directory / f"out-{number:02d}"
        write_json(path, spec)
        target.mkdir()
        result = subprocess.run([sys.executable, "-m", "ocudu_cuda_isac.sensing.features", "replay", "--archive",
                                 str(control.archive.directory), "--spec", str(path), "--output-dir", str(target)],
                                capture_output=True, text=True, env=environment, timeout=300, check=False)
        (directory / f"tool-{number:02d}.log").write_text(result.stdout + result.stderr)
        if result.returncode == 2 and not any(target.iterdir()) and result.stderr.startswith("features: "):
            by_tool += 1
        elif label not in kept:
            kept.append(label)
    grader.check(name, "specification check", refused == len(cases),
                 f"{refused} of {len(cases)} faulty specifications are refused by the check before a record is read: "
                 + ", ".join(label for label, _ in cases))
    grader.check(name, "command line", by_tool == len(cases),
                 f"{by_tool} of {len(cases)} are refused by python -m ocudu_cuda_isac.sensing.features replay with "
                 "status 2 and one line, and the output directory stays empty"
                 + ("" if by_tool == len(cases) else f"; not refused: {', '.join(kept)}"))
    return {"cases": len(cases), "refused": refused, "refused_by_tool": by_tool}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="new directory for the run")
    parser.add_argument("--s32-run", required=True, type=Path,
                        help="run directory of the accepted Sprint 3.2 gate: it holds the records its consumer kept from "
                             "the ring and the files of the audited observer")
    parser.add_argument("--adapter-build", type=Path, help="adapter build directory (default: from the configuration)")
    parser.add_argument("--host-rehearsal", action="store_true",
                        help="development host without a CUDA build: grade the host path on a rehearsed Sprint 3.2 run "
                             "and leave the hardware gate open")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.exit(2, f"pilot_features: {args.output_dir} already exists. Evidence is never overwritten.\n")
    try:
        from ocudu_cuda_isac.episodes import pilot_features as evaluator
        from ocudu_cuda_isac.sensing import features
    except ImportError as error:
        parser.exit(2, f"pilot_features: the reference of the features needs NumPy in this interpreter: {error}\n")
    try:
        config = load_config(args.config, features)
        receipt = native_adapter.verified_receipt(args.adapter_build or ROOT / config["adapter_build"])
        for needed in TOOLS:
            if needed not in receipt["executables"]:
                raise ConfigError(f"the adapter build holds no {needed}. Rebuild it: python scripts/native_adapter.py build")
    except (ConfigError, native_adapter.AdapterError) as error:
        parser.exit(2, f"pilot_features: {error}\n")
    tools = {name: Path(receipt["executables"][name]["path"]) for name in TOOLS}
    s32_run = args.s32_run.resolve()
    s32_out = s32_run / config["control"]["driver"] if (s32_run / config["control"]["driver"] / "grade.json").is_file() \
        else s32_run
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = args.output_dir.resolve()
    out.mkdir(parents=True)
    write_json(out / "adapter-receipt.json", receipt)
    write_json(out / "spec.json", config["spec"])

    grader, measured, runs = Grade(), {}, []

    def guarded(case, action, default=None):
        before = len(grader.results)
        try:
            return action()
        except Exception as error:  # an unexpected fault fails its case and keeps the remaining cases
            (out / f"driver-error-{case}.txt").write_text(traceback.format_exc())
            grader.check(case, "driver", False, f"{type(error).__name__}: {error} (driver-error-{case}.txt)")
            return default
        finally:
            if len(grader.results) == before:
                grader.check(case, "driver", False, "the case produced no check")

    control, measured["control"] = guarded(
        "control", lambda: run_control(config, grader, s32_out, args.host_rehearsal, evaluator), (None, {}))
    stream = None
    if control is not None:
        measured["stream"], stream = guarded("stream", lambda: run_stream(config, out, grader, control, features),
                                             ({}, None))
    if control is not None and stream is not None and stream.ok:
        measured["offline"] = guarded("offline", lambda: run_offline(config, grader, control, stream, features, evaluator))
        measured["known-return"] = guarded(
            "known-return", lambda: run_known_return(config, out, grader, control, stream, features, evaluator))
        measured["causality"] = guarded("causality", lambda: run_causality(config, grader, control, stream, features))
        measured["mask"] = guarded("mask", lambda: run_mask(config, grader, control, features))
        measured["live"] = guarded("live", lambda: run_live_case(config, out, grader, tools, control, stream, features,
                                                                 args.host_rehearsal, runs))
        measured["refusals"] = guarded("refusals", lambda: run_refusals(config, out, grader, control, features))
    else:
        for case in CASES[1:] if control is None else CASES[2:]:
            grader.check(case, "control", False,
                         "the Sprint 3.2 run does not hold the control this case is compared with" if control is None
                         else "the stream answered no complete window on the recorded ring")

    failed = [item for item in grader.results if not item["passed"]]
    passed_cases = [case for case in CASES if any(item["case"] == case for item in grader.results)
                    and all(item["passed"] for item in grader.results if item["case"] == case)]
    whole = not failed and len(passed_cases) == len(CASES)
    grade = {
        "schema": GRADE_SCHEMA, "selector": "pilot_features", "run_id": out.name, "cases": len(CASES),
        "cases_passed": len(passed_cases), "cases_failed": len(CASES) - len(passed_cases),
        "checks": len(grader.results), "checks_passed": len(grader.results) - len(failed), "checks_failed": len(failed),
        "judgment": ("rehearsal" if args.host_rehearsal else "pass") if whole else "fail",
        "hardware_gate": "open: the CUDA processor did not execute, this is a host rehearsal" if args.host_rehearsal
                         else ("closed" if whole else "open"),
        "reason": "every check of every case passed" if not failed else
                  f"{len(failed)} check(s) failed, first: {failed[0]['case']} {failed[0]['check']}",
        "backends": config["backends"], "spec_sha256": features.spec_sha256(config["spec"]),
        "tolerance": config["tolerance"], "known_return": config["known_return"], "runtime": features.RUNTIME,
        "record_definition": f"{obs.DEFINITION:016x}", "results": grader.results}
    write_json(out / "grade.json", grade)
    write_json(out / "comparison.json", measured)
    # What a second reader starts from: the live output, its target-present window and the same view of the stream.
    live = out / "live" / "run" / "sensor"
    if stream is not None and control is not None and (live / features.REPORT_FILE).is_file():
        present = [index for index, line in enumerate(stream.ok)
                   if (control.cases.get(line["window"]["episode"]) or {}).get("case") == "b"]
        if present:
            write_json(out / "recover.json", {
                "sensor": str(live.relative_to(out)), "window": present[0],
                "stream_view": str((stream.directory / stream.ok[present[0]]["view"]["file"]).relative_to(out))})
    lock = {item["name"]: item["commit"] for item in json.loads((ROOT / "ext/lock.json").read_text())["dependencies"]}
    write_json(out / "manifest.json", {
        "run_id": out.name, "selector": "pilot_features",
        "purpose": "Sprint 3.3 integration: the radar cube, the delay-Doppler view and the causal features of the "
                   "reference pair, from the records of the ring of the accepted Sprint 3.2 run, from the files of its "
                   "audited observer and from a live ring, compared with each other and with the geometry of the episode",
        "command": shlex.join([sys.executable, *sys.argv]), "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": platform.node(), "machine": platform.machine(), "python": platform.python_version(),
        "numpy": features.np.__version__, "cpus": os.cpu_count(), "parent_commit": git_commit(), "locked_sources": lock,
        "inputs": {str(args.config): digest(args.config)}, "driver_sha256": digest(__file__),
        "features_sha256": digest(ROOT / "src/ocudu_cuda_isac/sensing/features.py"),
        "evaluator_sha256": digest(ROOT / "src/ocudu_cuda_isac/episodes/pilot_features.py"),
        "reference_reader_sha256": digest(ROOT / "src/ocudu_cuda_isac/sensing/obs.py"),
        "s32_run": str(s32_run), "control": measured["control"].get("s32"),
        "spec": config["spec"], "steps": features.resolve(config["spec"]),
        "axes": {"split": "staged_harness", "receiver_stage": "o2_pilot_ls", "ue_realization": "replayed channel, no UE",
                 "host_stack": "DeepSig-OCUDU", "sensing_source": config["spec"]["input"]["source"],
                 "placement": "one host, two processes", "transport": "obs-ring/v1 in host shared memory",
                 "runtime": "reference", "clutter": f"static removal {config['spec']['steps']['static']}",
                 "detector": "none: this ticket publishes features, S3.4 the decision",
                 "gpu_sharing": "the sensor is not a CUDA client"},
        "runs": [{"name": run.name, "directory": str(run.directory.relative_to(out)), "status": run.status,
                  "plan_sha256": digest(run.directory / "plan.json") if (run.directory / "plan.json").is_file() else None,
                  "mode": run.plan["evidence"]["mode"] if run.plan else None, "ring": run.plan["ring"] if run.plan else None,
                  "consumers": [{key: str(value) if key == "archive" else value for key, value in consumer.items()}
                                for consumer in run.consumers], "consumer_statuses": run.statuses} for run in runs],
        "adapter": {"executables": {name: {"path": str(tools[name]), "sha256": receipt["executables"][name]["sha256"]}
                                    for name in TOOLS},
                    "receipt": "adapter-receipt.json", "ocudu_build": receipt["ocudu_build"],
                    "built_with_cuda": receipt["pin"]["cuda"]},
        "host_rehearsal": args.host_rehearsal,
        "environment": {key: os.environ[key] for key in ("VIRTUAL_ENV", "CUDA_VISIBLE_DEVICES") if key in os.environ},
        "judgment": grade["judgment"], "reason": grade["reason"]})
    summary = (f"pilot_features: {grade['cases_passed']} of {grade['cases']} cases passed, "
               f"{grade['checks_passed']} of {grade['checks']} checks"
               + (", host rehearsal: the hardware gate stays open" if args.host_rehearsal else ""))
    (out / "grade.txt").write_text("\n".join(grader.lines + [summary]) + "\n", encoding="ascii")
    print(summary)
    print(f"Run artifacts: {out}")
    return 0 if whole else 1


if __name__ == "__main__":
    sys.exit(main())
