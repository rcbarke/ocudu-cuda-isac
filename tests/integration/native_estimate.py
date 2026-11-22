#!/usr/bin/env python3
"""Sprint 5.2 integration driver (selector native_estimate). Hardware command, not part of unit discovery.

It runs the tapped receiver replay of the adapter build against the patched receiver on the plans
that the accepted Sprint 3.1 run tapped, exports the receiver's own channel estimate, the stage
O3, beside the stage O2, and compares the two. The controls are the radio records of the accepted
Sprint 2.4 run, which the unpatched build wrote without an observer, and the stage O2 of the
accepted Sprint 3.1 run. Ten cases, each graded on its own:

  control     the accepted Sprint 3.1 run and, for every plan of this gate, both sets of control records
  patch       the series of patches/ocudu/, the receipt of the patched receiver and the two adapters
  flat        a flat channel without noise, three grants: every decode is the unpatched build's, the
              stage O2 is the accepted one, and the exported estimate is the matched estimator's
              within one FP16 spacing, with the layout and the element the card names
  varying     the known frequency-varying channel without noise: the same, within the declared NMSE
  noisy       the seeded noisy channels of the reference pair: the same, with the counts of the parity
  hook-off    the patched build without an observer: the decodes of the unpatched build
  side-stream the copy on a stream of the observer, ordered by an event: the same estimate
  timed       both stages on their rings to consumers in other processes, with no checksum between
              the first fill and the drain and no system call in a publication
  refusals    every rule of the plan, refused by the plan check, by the patched tool and, for a
              plan that names an estimate, by the build without the route
  backends    the path every decode executed and what became of its estimate, branch by branch

    python tests/integration/native_estimate.py --config configs/integration/native_estimate.json \\
        --s31-run results/runs/s31-spark-001 --output-dir results/runs/s52-spark-001

The output directory must be new. The exit status is 0 only when every check of every case passed.
This gate has no host rehearsal: the estimate exists only on a CUDA device.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
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
from ocudu_cuda_isac.episodes import native_estimate as ne  # noqa: E402
from ocudu_cuda_isac.episodes import pilot_ls as pl  # noqa: E402
from ocudu_cuda_isac.episodes import replay_pusch as rp  # noqa: E402
from ocudu_cuda_isac.episodes import snapshot_ring as sr  # noqa: E402
from ocudu_cuda_isac.episodes.native_estimate import NativeEstimatePlanError  # noqa: E402
from ocudu_cuda_isac.episodes.pilot_ls import PilotLsPlanError  # noqa: E402
from ocudu_cuda_isac.episodes.replay_pusch import ReplayPuschError  # noqa: E402
from ocudu_cuda_isac.replay import ContractError, load_json  # noqa: E402
from ocudu_cuda_isac.sensing import native_estimate as reference  # noqa: E402
from ocudu_cuda_isac.sensing import obs  # noqa: E402
from ocudu_cuda_isac.sensing import pilot_ls as o2_reference  # noqa: E402
from ocudu_cuda_isac.sensing.native_estimate import NativeEstimateError  # noqa: E402
from ocudu_cuda_isac.sensing.obs import ObsError  # noqa: E402
from ocudu_cuda_isac.sensing.pilot_ls import PilotLsError  # noqa: E402
import native_adapter  # noqa: E402
import receiver_patch  # noqa: E402

CONFIG_SCHEMA = "integration-config/native_estimate/v1"
GRADE_SCHEMA = "integration-grade/native_estimate/v1"
CONTROL_GRADE_SCHEMA = "integration-grade/pilot_ls/v1"
CASES = ("control", "patch", "flat", "varying", "noisy", "hook-off", "side-stream", "timed", "refusals", "backends")
TOOLS = ("pilot_ls", "obs_tail")
CLASSES = ("flat", "varying", "noisy")
NATIVE_TIMEOUT_S = 1800
CONSUMER_GRACE_S = 60


class ConfigError(ValueError):
    """The configuration or the run setup is unusable. Nothing was graded."""


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ring_driver = load_script("snapshot_ring_driver", ROOT / "tests/integration/snapshot_ring.py")
digest, write_json, Grader = ring_driver.digest, ring_driver.write_json, ring_driver.Grader
last_line, git_commit, consumer_command = ring_driver.last_line, ring_driver.git_commit, ring_driver.consumer_command
run_is_clean, decodes_check = ring_driver.run_is_clean, ring_driver.decodes_check


class Grade(Grader):
    """The Sprint 2.1 grader with this sprint's label."""

    def check(self, case, name, passed, detail):
        self.results.append({"case": case, "check": name, "passed": bool(passed), "detail": detail})
        self._line(("PASS" if passed else "FAIL") + f"  S5.2 {case} {name}: {detail}")
        return bool(passed)

    def note(self, case, text):
        self._line(f"NOTE  S5.2 {case} {text}")


def load_config(path):
    try:
        config = load_json(Path(path))
    except (ContractError, OSError) as error:
        raise ConfigError(f"cannot read the configuration: {error}") from None
    needed = {"schema", "selector", "adapter_build", "o3_adapter_build", "backends", "observer", "estimate", "control",
              "rings", "tolerance", "study", "expect"}
    if not isinstance(config, dict) or set(config) != needed or config["schema"] != CONFIG_SCHEMA \
            or config["selector"] != "native_estimate":
        raise ConfigError(f"the configuration must be a {CONFIG_SCHEMA} object for the selector native_estimate")
    if list(config["backends"]) != list(rp.BACKENDS):
        raise ConfigError('backends must be ["cpu", "cuda"]: the gate decodes every grid with both')
    estimate = config["estimate"]
    if not isinstance(estimate, dict) or set(estimate) != {"tap", "stage", "copy_stream", "landing", "timeout_us"} \
            or estimate["tap"] != ne.TAP or estimate["stage"] != reference.STAGE:
        raise ConfigError("estimate requires the tap native_estimate, the stage O3, copy_stream, landing and timeout_us")
    control = config["control"]
    if not isinstance(control, dict) or set(control) != {"driver", *CLASSES} or control["driver"] != "pilot_ls":
        raise ConfigError("control requires the driver pilot_ls and the runs it tapped: flat, varying and noisy")
    rings = config["rings"]
    if not isinstance(rings, dict) or set(rings) != {"o2", "o3"}:
        raise ConfigError("rings requires o2 and o3")
    try:
        for name, ring in rings.items():
            if ring["slots"] != obs.slots_for(ring["window_records"], ring["burst_records"]) or ring["stream"] != name:
                raise ConfigError(f"rings.{name} names another stream or another slot count than its window and its "
                                  "burst need")
        tolerance = config["tolerance"]
        if set(tolerance) != {*CLASSES, "race_free_ulps"} or not tolerance["race_free_ulps"] > 0:
            raise ConfigError("tolerance requires flat, varying, noisy and race_free_ulps")
        for name in CLASSES:
            ne.check_tolerance(tolerance[name], f"tolerance.{name}")
    except (KeyError, TypeError, ObsError, NativeEstimatePlanError) as error:
        raise ConfigError(f"the rings or the tolerances are unusable: {error}") from None
    expect = config["expect"]
    if not isinstance(expect, dict) or set(expect) != {"o2_unity", "o3_unity", "payload_bytes", "pair_opportunities",
                                                       "shape", "first_subcarrier", "element"}:
        raise ConfigError("expect requires o2_unity, o3_unity, payload_bytes, pair_opportunities, shape, "
                          "first_subcarrier and element")
    return config


# ---------------------------------------------------------------- processes
def run_scenario(tools, plan_path, native, directory, consumers, extra=(), tool="pilot_ls"):
    """Run one pilot_ls on one plan with the consumers of its streams. Returns (status, report or None, statuses).

    A consumer names its stream, "o2" or "o3". Each is started before the tool and retries until
    the tool has opened the channel of its stream. native is created empty.
    """
    native.mkdir(parents=True)
    plan = json.loads(Path(plan_path).read_text())
    sockets = {"o2": sr.socket_path(native, plan) if plan.get("ring") else None,
               "o3": ne.socket_path(native, plan) if (plan.get("estimate") or {}).get("ring") else None}
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    started, logs = [], []
    try:
        for index, consumer in enumerate(consumers):
            consumer["archive"].mkdir(parents=True)
            command = consumer_command(tools, consumer, sockets[consumer["stream"]], None)
            log = (directory / f"consumer-{index}.log").open("w")
            log.write(shlex.join(command) + "\n")
            log.flush()
            logs.append(log)
            started.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment))
        command = [str(tools[tool]), "--plan", str(plan_path), "--output-dir", str(native), *extra]
        with (directory / "native.log").open("w") as stream:
            stream.write(shlex.join(command) + "\n")
            stream.flush()
            try:
                status = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=NATIVE_TIMEOUT_S,
                                        check=False).returncode
            except subprocess.TimeoutExpired:
                stream.write("TimeoutExpired: the tool was stopped\n")
                status = -1
        statuses = []
        for process in started:
            try:
                statuses.append(process.wait(timeout=CONSUMER_GRACE_S))
            except subprocess.TimeoutExpired:
                process.kill()
                statuses.append(-1)
    except OSError as error:
        (directory / "native.log").open("a").write(f"{type(error).__name__}: {error}\n")
        for process in started:
            process.kill()
        return -1, None, [-1] * len(consumers)
    finally:
        for log in logs:
            log.close()
    try:
        return status, json.loads((native / "native.json").read_text()), statuses
    except (OSError, ValueError):
        return status, None, statuses


class Control:
    """One run of the accepted Sprint 3.1 gate: the plan it tapped, its stage O2 and the decodes of Sprint 2.4."""

    def __init__(self, name):
        self.name, self.label = name, name.split("/")[-1]
        self.replay_plan_path = self.radio = self.accepted = None
        self.estimates = {}


class Run(ring_driver.Run):
    """One run of this gate."""

    def o2(self):
        """Payload and mask digests of the stage O2 of an audited run, by opportunity."""
        return {record["host"]["opportunity_id"]: (record["payload"]["digest"], record["validity"]["sha256"])
                for record in self.output.observer.observations}


def estimate_run(runs, directory, name, tools, run_id, control, config, mode="audit", estimate=True, copy_stream=None,
                 rings=False, consumers=(), tool="pilot_ls"):
    """Write the plan of a run on a control plan, run the tool with its consumers and verify what they wrote."""
    run = Run(name, directory, control)
    runs.append(run)
    directory.mkdir(parents=True)
    settings, wanted = config["rings"], config["estimate"]

    def ring(stream):
        item = settings[stream]
        return sr.ring_spec(item["window_records"], item["burst_records"], "spin", "required" if consumers else "none",
                            item["attach_timeout_ms"], item["drain_ms"], item["stream"])

    spec = ne.estimate_spec(copy_stream or wanted["copy_stream"], wanted["landing"], wanted["timeout_us"],
                            ring("o3") if rings else None, config["observer"]["source"]) if estimate else None
    run.plan = ne.build_plan(run_id, control.replay_plan_path, mode, ring("o2") if rings else None, spec,
                             epoch=config["observer"]["epoch"], source=config["observer"]["source"])
    ne.write_plan(run.plan, directory / "plan.json")
    run.consumers = [dict(consumer, archive=directory / f"consumer-{consumer['stream']}") for consumer in consumers]
    run.status, run.report, run.statuses = run_scenario(tools, directory / "plan.json", run.native, directory,
                                                        run.consumers, tool=tool)
    if run.status != 0 or not run.report:
        return run
    try:
        run.output = ne.open_output(run.native)
    except NativeEstimatePlanError as error:
        run.problem = f"the native output does not verify: {error}"
        return run
    for consumer, status in zip(run.consumers, run.statuses):
        try:
            if status != 0:
                raise ObsError(f"the consumer exited with status {status}")
            run.archives.append(obs.open_archive(consumer["archive"]))
        except ObsError as error:
            run.problem = f"the archive of the consumer of {consumer['stream']} does not verify: {error}"
            run.archives.append(None)
    return run


def text(value, digits=2):
    return "none" if value is None else f"{value:.{digits}e}"


def account_text(states):
    return (f"{states['eligible']} decodes of the CUDA processor: " + ", ".join(
        f"{states[name]} {name}" for name in ne.OUTCOMES) + f"; {states['unexpected']} estimates outside a capture")


# ---------------------------------------------------------------- control and patch
def run_control(config, grader, s31_out):
    """The accepted Sprint 3.1 run and, for every plan of this gate, both sets of control records."""
    name, controls, measured = "control", {}, {}
    try:
        grade, manifest = load_json(s31_out / "grade.json"), load_json(s31_out / "manifest.json")
    except (ContractError, OSError) as error:
        grader.check(name, "accepted run", False, f"{s31_out} holds no grade and manifest of the Sprint 3.1 driver: {error}")
        return None, measured
    accepted = isinstance(grade, dict) and grade.get("schema") == CONTROL_GRADE_SCHEMA and grade.get("judgment") == "pass" \
        and grade.get("cases_passed") == grade.get("cases") and grade.get("checks_failed") == 0 \
        and grade.get("hardware_gate") == "closed"
    measured["s31"] = {"run_id": manifest.get("run_id"), "judgment": grade.get("judgment"),
                       "hardware_gate": grade.get("hardware_gate"), "parent_commit": manifest.get("parent_commit")}
    grader.check(name, "accepted run", accepted,
                 f"the Sprint 3.1 driver graded {s31_out.name} as {grade.get('judgment')}: {grade.get('cases_passed')} of "
                 f"{grade.get('cases')} cases, {grade.get('checks_failed')} failed checks, hardware gate "
                 f"{str(grade.get('hardware_gate')).split(':')[0]}" + ("" if accepted else
                                                                      "; this gate needs the judgment pass with the "
                                                                      "hardware gate closed"))
    listed = {item["name"]: item for item in manifest.get("runs", []) if isinstance(item, dict)}
    wanted = [run for run in listed if run.startswith(config["control"]["flat"])]
    wanted += [config["control"]["varying"], config["control"]["noisy"]]
    try:
        if len(wanted) < 3:
            raise PilotLsPlanError(f"the Sprint 3.1 run holds no run whose name starts with {config['control']['flat']}")
        for run in wanted:
            item = listed.get(run)
            if item is None or item.get("status") != 0:
                raise PilotLsPlanError(f"the Sprint 3.1 run holds no finished run named {run}")
            directory = s31_out / item["directory"]
            if digest(directory / "plan.json") != item.get("plan_sha256"):
                raise PilotLsPlanError(f"the plan of the run {run} does not have the hash its manifest names")
            tapped = pl.load_plan(directory / "plan.json")
            control = Control(run)
            control.accepted = pl.open_output(directory / "native")
            control.replay_plan_path = Path(tapped["replay_plan"]["path"])
            if digest(control.replay_plan_path) != tapped["replay_plan"]["sha256"]:
                raise PilotLsPlanError(f"the replay plan of the run {run} does not have the hash the tapped plan binds")
            control.radio = rp.open_output(control.replay_plan_path.parent / "native")
            control.estimates = {record["host"]["opportunity_id"]: (record["payload"]["digest"], record["validity"]["sha256"])
                                 for record in control.accepted.observations}
            if any(record["tap"] != "none" for record in control.radio.records) or not control.estimates:
                raise PilotLsPlanError(f"the Sprint 2.4 run beside the plan of {run} was not written without a tap")
            controls[run] = control
    except (PilotLsPlanError, PilotLsError, ReplayPuschError, OSError) as error:
        grader.check(name, "records", False, f"{error}")
        return None, measured
    measured["runs"] = {run: {"observations": len(control.estimates), "radio_records": len(control.radio.records),
                              "replay_plan": str(control.replay_plan_path)} for run, control in controls.items()}
    grader.check(name, "records", True, "; ".join(
        f"{run}: {len(control.estimates)} accepted observations of the stage O2 and {len(control.radio.records)} radio "
        "records of the unpatched build without a tap" for run, control in controls.items()))
    return (controls if accepted else None), measured


def run_patch(grader, receipts):
    """The series, the receipt of the patched receiver and the two adapter builds."""
    name, control, patched = "patch", receipts["control"], receipts["o3"]
    try:
        series = receiver_patch.load_series()
        current = receiver_patch.identity(series)
        problem = None
    except receiver_patch.PatchError as error:
        series, current, problem = None, None, str(error)
    grader.check(name, "series", series is not None, problem or
                 f"patches/ocudu/series.json {current['series_sha256'][:16]} on the pinned commit "
                 f"{current['base_commit'][:12]}: " + ", ".join(f"{item['file']} with the SHA-256 {item['sha256'][:16]}"
                                                               for item in current["patches"])
                 + f"; {len(current['files'])} files touched, each named with its hash after the series")
    carried = patched.get("patch") or {}
    route = patched.get("o3_route") or {}
    same = series is not None and receiver_patch.same_series(carried, current)
    grader.check(name, "patched receiver", same and route.get("compiled") is True and patched["pin"]["cuda"],
                 f"the adapter with the O3 route is linked against {patched['ocudu_build']}, built from a "
                 f"{carried.get('tree_kind')} with exactly this series, with ENABLE_CUDA; pilot_ls "
                 f"{patched['executables']['pilot_ls']['sha256'][:12]} reports the tap interface version "
                 f"{route.get('tap_interface_version')}" if same and route.get("compiled") else
                 "the receipt of the O3 adapter names no route, or another series than patches/ocudu/ holds")
    plain = not control.get("patch") and not control.get("o3_route") and control["ocudu_build"] != patched["ocudu_build"]
    grader.check(name, "control build", plain,
                 f"the adapter without the route is linked against {control['ocudu_build']}, the accepted build of the "
                 f"unpatched receiver; pilot_ls {control['executables']['pilot_ls']['sha256'][:12]}" if plain else
                 "the control adapter carries a patch or is linked against the patched receiver")
    return {"series": current, "o3_adapter": {"ocudu_build": patched["ocudu_build"], "route": route},
            "control_adapter": {"ocudu_build": control["ocudu_build"]}}


# ---------------------------------------------------------------- the estimate
def estimate_checks(config, grader, name, run, kind, parity_out, label=None):
    """The checks every audited run with an estimate shares. Returns its parity or None."""
    label = label or run.name
    if not grader.check(name, f"native run {label}" if label != name else "native run",
                        run_is_clean(run, config["backends"]),
                        f"{run.counts()}; one record per planned decode, no anomaly, every grid released"
                        if run.ok() else run.said()):
        return None
    decodes_check(grader, name, run, f"decodes {label}" if label != name else "decodes")
    found = run.o2()
    equal = sum(run.control.estimates.get(opportunity) == pair for opportunity, pair in found.items())
    grader.check(name, f"stage O2 {label}" if label != name else "stage O2",
                 equal == len(found) == len(run.control.estimates),
                 f"{equal} of {len(found)} payloads and validity masks of the stage O2 have the SHA-256 that the accepted "
                 "Sprint 3.1 run recorded for the same opportunity: the patched receiver hands the observer the same grid")
    states = ne.account(run.output)
    resident = sum(record["backend"] == ne.BACKEND and record["path"]["executed"] == "resident"
                   for record in run.output.radio.records)
    captured = states["captured"] + states["published"]
    grader.check(name, f"capture {label}" if label != name else "capture",
                 states["balanced"] and captured == resident == states["eligible"] and len(run.output.estimates) == captured
                 and states["unexpected"] == 0,
                 account_text(states) + f"; {resident} decodes ran the resident path, and each left one estimate, copied "
                 f"on the stream {run.output.estimate['copy_stream']} and published on its completion event")
    try:
        result = ne.parity_of(run.output, config["tolerance"][kind], config["tolerance"]["race_free_ulps"])
    except (NativeEstimatePlanError, NativeEstimateError) as error:
        grader.check(name, f"parity {label}" if label != name else "parity", False, str(error))
        return None
    total, rows = result["totals"], result["rows"]
    worst = max(rows, key=lambda row: row["pilot_support"]["nmse"]) if rows else None
    grader.check(name, f"parity {label}" if label != name else "parity", result["passed"],
                 f"{sum(row['passed'] for row in rows)} of {len(rows)} estimates meet the declared tolerance against the "
                 f"matched estimator on the stage O2 of their slot, on the common pilot support: numerator "
                 f"{text(total['numerator'])}, denominator {text(total['denominator'])}, {total['samples']} samples, NMSE "
                 f"{text(total['nmse'])}" + ("" if worst is None else
                                           f"; the largest NMSE of one estimate is {text(worst['pilot_support']['nmse'])} "
                                           f"against its bound {text(worst['nmse_bound'])}, and the largest deviation is "
                                           f"{max(row['pilot_support']['max_fp16_ulps'] for row in rows):.2f} FP16 spacings"))
    if rows:
        effect = max(row["smoothing_effect_nmse"] for row in rows)
        grader.note(name, f"{label}, not graded: {result['race_free_records']} of {len(rows)} estimates lie within "
                    f"{config['tolerance']['race_free_ulps']:g} FP16 spacing of the order in which every block reads "
                    f"unfiltered pilots; smoothing changes the pilots by an NMSE of up to {text(effect)}, the estimate "
                    f"without smoothing is {text(max(row['unsmoothed_o2']['nmse'] for row in rows))} away from the "
                    "exported one, and the in-place orders of the model are up to "
                    f"{text(max([v['nmse'] for row in rows for v in row['in_place_model'].values()] or [0.0]))} away from "
                    "the matched one")
    parity_out.append(dict(result, case=name, run=run.name, kind=kind))
    return result


def run_flat(config, out, grader, tools, runs, controls, parity_out):
    name, expect, measured, element = "flat", config["expect"], {}, None
    for control in [item for run, item in controls.items() if run.startswith(config["control"]["flat"])]:
        run = estimate_run(runs, out / "flat" / control.label / "run", f"flat/{control.label}", tools,
                           f"s52-flat-{control.label}", control, config)
        result = estimate_checks(config, grader, name, run, "flat", parity_out, control.label)
        if result is None:
            continue
        # Every estimate against what its grant is expected to give. A grant of another slot keeps its base name.
        values, odd, seen = set(), [], {}
        for record in run.output.estimates:
            grant = record["host"]["grant_id"].split(".")[0]
            values.update(run.output.values(record))
            seen[grant] = record["shape"]["shape"]
            if record["shape"]["shape"] != expect["shape"].get(grant) \
                    or record["allocation"]["first_subcarrier"] != expect["first_subcarrier"].get(grant):
                odd.append(f"{record['host']['grant_id']} has the shape {record['shape']['shape']} from subcarrier "
                           f"{record['allocation']['first_subcarrier']}")
            if grant == expect["element"]["grant"] and element is None:
                element = record
        wanted = complex(expect["o3_unity"], 0.0)
        grader.check(name, f"value {control.label}", values == {wanted} and not odd,
                     f"every element of {len(run.output.estimates)} estimates is {sorted(values, key=abs)[0]}: the unity "
                     f"channel reads {expect['o2_unity']:.7f} in the stage O2 and {expect['o3_unity']} in the FP16 of "
                     "the receiver, with a set validity bit on every element; "
                     + ", ".join(f"{grant} in the shape {shape} from subcarrier {expect['first_subcarrier'][grant]}"
                                 for grant, shape in sorted(seen.items())) if values and not odd else
                     "; ".join(odd[:3]) or f"the estimates hold {len(values)} different values")
        measured[control.label] = {"value": [sorted(values, key=abs)[0].real, sorted(values, key=abs)[0].imag]
                                   if values else None, "records": len(run.output.estimates)}
    index = expect["element"]["index"]
    if element is None:
        grader.check(name, "element", False, f"no flat run holds an estimate of the grant {expect['element']['grant']}")
    else:
        shape = element["shape"]["shape"]
        position = reference.element(shape, *index)
        where = reference.coordinate(shape, position)
        grader.check(name, "element", position == expect["element"]["position"]
                     and where["subcarrier"] == expect["element"]["subcarrier"] and where["rx_port"] == index[2]
                     and where["dmrs_symbol"] == index[0],
                     f"element {index} of the payload is position {position}: DM-RS symbol {where['dmrs_symbol']} "
                     f"(symbol {element['allocation']['dmrs_symbols'][where['dmrs_symbol']]} of the slot), receive "
                     f"port {where['rx_port']} (array element {element['allocation']['antenna_map'][where['rx_port']]}) "
                     f"and subcarrier {where['subcarrier']} of the allocation")
    return measured


def run_class(config, out, grader, tools, runs, controls, parity_out, name, kind, **options):
    """One audited run on the varying or the noisy control, or on the copy stream of the observer."""
    control = controls[config["control"][kind]]
    run = estimate_run(runs, out / name.replace("-", "_") / "run", name, tools, f"s52-{name}", control, config, **options)
    result = estimate_checks(config, grader, name, run, kind, parity_out)
    return ({"records": result["records"], "totals": result["totals"],
             "race_free_records": result["race_free_records"]} if result else {}), run


def service_of(report, backend=ne.BACKEND):
    for item in report["backends"]:
        if item["name"] == backend:
            return item["service_us"].get("mean")
    return None


def run_hook_off(config, out, grader, tools, runs, controls, noisy):
    """The patched receiver without an observer: the hook-disabled decode control."""
    name, control = "hook-off", controls[config["control"]["noisy"]]
    run = estimate_run(runs, out / "hook_off" / "run", name, tools, "s52-hook-off", control, config, estimate=False)
    if not grader.check(name, "native run", run_is_clean(run, config["backends"]) and run.output.estimate is None,
                        f"{run.counts()}; the patched build with a plan that names no estimate registers no observer and "
                        "writes no estimate" if run.ok() else run.said()):
        return {}
    decodes_check(grader, name, run)
    equal = sum(control.estimates.get(opportunity) == pair for opportunity, pair in run.o2().items())
    grader.check(name, "stage O2", equal == len(control.estimates),
                 f"{equal} of {len(control.estimates)} payloads of the stage O2 are the accepted ones")
    times = {"unpatched": service_of(control.radio.report), "hook_off": service_of(run.report),
             "hook_on": service_of(noisy.report) if noisy is not None and noisy.ok() else None}
    grader.note(name, "CUDA PUSCH service time, mean over the decodes after the first, not graded: "
                + ", ".join(f"{key.replace('_', ' ')} {'none' if value is None else f'{value:.1f} us'}"
                            for key, value in times.items())
                + ". Every run here is audited, so the time includes its digests. The timed distribution is S5.5's")
    return {"service_us": times}


def run_timed(config, out, grader, tools, runs, controls, noisy):
    """Both stages on their rings, in the mode timed, to one consumer per stream."""
    name, control = "timed", controls[config["control"]["noisy"]]
    consumers = [{"reader": "native", "wait": "spin", "stream": "o2"}, {"reader": "native", "wait": "spin", "stream": "o3"}]
    run = estimate_run(runs, out / "timed" / "run", name, tools, "s52-timed", control, config, mode="timed", rings=True,
                       consumers=consumers)
    if not grader.check(name, "native run", run_is_clean(run, config["backends"]) and all(run.archives),
                        f"{run.counts()}; one consumer per stream attached before the first fill" if run.ok() else
                        run.said()):
        return {}
    decodes_check(grader, name, run)
    window, estimate = run.output.base.hash_in_window(), run.output.estimate
    ring = estimate["ring"]
    grader.check(name, "no checksum", window == {"digests": 0, "updates": 0, "bytes": 0},
                 f"between the first fill and the drain of both rings the hash of the core computed {window['digests']} "
                 f"digests over {window['bytes']} bytes")
    states = ne.account(run.output)
    grader.check(name, "account", states["balanced"] and states["published"] == states["eligible"] > 0
                 and ring["counts"]["window_calls"] == 0 and ring["counts"]["faults"] == 0,
                 account_text(states) + f"; of the published {states['released']} released, {states['reclaimed']} "
                 f"reclaimed and {states['pending']} pending at the cutoff; the producer made "
                 f"{ring['counts']['window_calls']} system calls between an admission and its publication store")
    pairs, problems = ne.join_archive(run.output, run.archives[1])
    tolerance, checked, passed = config["tolerance"]["noisy"], 0, 0
    o2 = {f"{record.header['opportunity']:016x}": record for record in run.archives[0].records}
    same = {}
    audited = {record["host"]["opportunity_id"]: record["payload"]["digest"] for record in noisy.output.estimates} \
        if noisy is not None and noisy.ok() else {}
    for line, record in pairs:
        other = o2.get(line["keys"]["opportunity"])
        if other is None:
            problems.append(f"the opportunity {line['opportunity_id']} has no record of the stage O2")
            continue
        same[line["opportunity_id"]] = hashlib.sha256(record.payload).hexdigest() == audited.get(line["opportunity_id"])
        result = ne.compare(reference.payload_values(record.payload), list(record.shape),
                            o2_reference.payload_values(other.payload), list(other.shape), tolerance)
        checked += 1
        passed += result["passed"]
    sequences = [record.sequence for record in run.archives[1].records]
    grader.check(name, "records", not problems and checked == passed == len(pairs) == states["published"]
                 and sequences == list(range(len(sequences))),
                 f"the consumer of the O3 stream kept {len(pairs)} records, sequences 0 to {len(sequences) - 1} in order; "
                 "each is the record of its account line: the join keys, the slot, the completion and publication "
                 f"times, the shape and the scale its producer reported; {passed} of {checked} meet the declared "
                 "tolerance against the matched estimator on the O2 record of the same opportunity, which the other "
                 "consumer kept" if not problems else "; ".join(problems[:3]))
    grader.note(name, f"not graded: {sum(same.values())} of {len(same)} timed payloads have the bytes of the audited run "
                "of the same plan. The in-place filter of the receiver does not fix the order of its reads, so two "
                "decodes of one grid may differ within the declared tolerance; landing "
                f"{estimate['landing']['effective']}" + (f" ({estimate['landing']['refusal']})"
                                                         if estimate["landing"]["refusal"] else "")
                + f", submit {estimate['submit_us'].get('mean', 0):.1f} us inside the decode and resolve "
                f"{estimate['resolve_us'].get('mean', 0):.1f} us after it, mean over the copies after the first")
    return {"account": states, "identical_to_audit": sum(same.values()), "landing": estimate["landing"],
            "submit_us": estimate["submit_us"], "resolve_us": estimate["resolve_us"]}


def run_refusals(config, out, grader, tools, controls):
    """Every rule of the plan, by the plan check, by the patched tool and by the build without the route."""
    name, control = "refusals", controls[config["control"]["noisy"]]
    case_dir = out / "refusals"
    case_dir.mkdir()
    good = ne.build_plan("s52-refusal", control.replay_plan_path, estimate=ne.estimate_spec())
    rules = (("another tap", lambda plan: plan["estimate"].update(tap="pilot_ls")),
             ("the host backend", lambda plan: plan["estimate"].update(backend="cpu")),
             ("an unknown copy stream", lambda plan: plan["estimate"].update(copy_stream="any")),
             ("an unknown landing", lambda plan: plan["estimate"].update(landing="pageable")),
             ("a timed estimate without a ring", lambda plan: (plan["evidence"].update(mode="timed"),
                                                              plan.update(ring=sr.ring_spec(3, 3, attach="none")))),
             ("a ring with a wrong slot count", lambda plan: plan["estimate"].update(
                 ring=dict(sr.ring_spec(3, 3, attach="none", stream="o3"), slots=4))))
    refused, said = 0, []
    for index, (label, change) in enumerate(rules):
        plan = json.loads(json.dumps(good))
        change(plan)
        try:
            ne.check_plan(plan)
            offline = False
        except NativeEstimatePlanError:
            offline = True
        directory = case_dir / f"rule-{index}"
        directory.mkdir()
        (directory / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        status, report, _ = run_scenario(tools, directory / "plan.json", directory / "native", directory, [])
        native = status == 2 and report is None
        refused += offline and native
        if not (offline and native):
            said.append(f"{label}: the plan check {'refuses' if offline else 'accepts'} it and the tool exits with {status}")
    grader.check(name, "rules", refused == len(rules),
                 f"{refused} of {len(rules)} broken plans are refused by the plan check and by the patched tool with exit "
                 "status 2 before a file is written: " + ", ".join(label for label, _ in rules) if not said else
                 "; ".join(said))
    directory = case_dir / "unpatched"
    directory.mkdir()
    (directory / "plan.json").write_text(json.dumps(good, indent=2, sort_keys=True) + "\n")
    status, report, _ = run_scenario(tools, directory / "plan.json", directory / "native", directory, [],
                                     tool="pilot_ls_unpatched")
    words = last_line(directory / "native.log")
    grader.check(name, "no route", status == 2 and report is None and "no route" in words,
                 f"the pilot_ls of the build against the unpatched receiver refuses a plan that names an estimate with "
                 f"exit status {status}: {words[:160]}")
    return {"rules": len(rules), "refused": refused}


def run_backends(config, grader, runs):
    """The path every decode executed and what became of its estimate, branch by branch."""
    name, branches, clean = "backends", {}, True
    for run in runs:
        if not run.ok():
            continue
        outcomes = {line["opportunity_id"]: line["outcome"] for line in run.output.captures}
        for record in run.output.radio.records:
            executed = record["path"]["executed"]
            estimate = outcomes.get(record["opportunity_id"], "none") if record["backend"] == ne.BACKEND else "none"
            key = (record["backend"], executed, estimate)
            branches[key] = branches.get(key, 0) + 1
            wanted = run.output.estimate is not None and record["backend"] == ne.BACKEND
            clean = clean and (executed == "resident") == (record["backend"] == ne.BACKEND) \
                and (estimate in ("captured", "published")) == wanted
    listed = "; ".join(f"{backend} on the {executed} path, estimate {estimate}: {count} decodes"
                       for (backend, executed, estimate), count in sorted(branches.items()))
    grader.check(name, "provenance", clean and bool(branches),
                 f"{listed}. Supported: the resident CUDA path with one layer and DM-RS type 1. Recorded as unsupported: "
                 "the host path, which has no device buffer and reports no estimate; transform precoding, which the "
                 "patch does not tap; more than one layer, which the observer refuses")
    return {"branches": [{"backend": backend, "executed": executed, "estimate": estimate, "decodes": count}
                         for (backend, executed, estimate), count in sorted(branches.items())]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="new directory for the run")
    parser.add_argument("--s31-run", required=True, type=Path,
                        help="run directory of the accepted Sprint 3.1 gate: it names the plans and holds the accepted "
                             "stage O2; the Sprint 2.4 run those plans lie in holds the control decodes")
    parser.add_argument("--adapter-build", type=Path, help="adapter build against the unpatched receiver")
    parser.add_argument("--o3-adapter-build", type=Path, help="adapter build against the patched receiver")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.exit(2, f"native_estimate: {args.output_dir} already exists. Evidence is never overwritten.\n")
    try:
        config = load_config(args.config)
        receipts = {"control": native_adapter.verified_receipt(args.adapter_build or ROOT / config["adapter_build"]),
                    "o3": native_adapter.verified_receipt(args.o3_adapter_build or ROOT / config["o3_adapter_build"])}
        for receipt in receipts.values():
            for needed in TOOLS:
                if needed not in receipt["executables"]:
                    raise ConfigError(f"an adapter build holds no {needed}. Rebuild it.")
    except (ConfigError, native_adapter.AdapterError) as error:
        parser.exit(2, f"native_estimate: {error}\n")
    tools = {name: Path(receipts["o3"]["executables"][name]["path"]) for name in TOOLS}
    tools["pilot_ls_unpatched"] = Path(receipts["control"]["executables"]["pilot_ls"]["path"])
    s31_run = args.s31_run.resolve()
    s31_out = s31_run / config["control"]["driver"] if (s31_run / config["control"]["driver"] / "grade.json").is_file() \
        else s31_run
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = args.output_dir.resolve()
    out.mkdir(parents=True)
    write_json(out / "adapter-receipt.json", receipts["control"])
    write_json(out / "o3-adapter-receipt.json", receipts["o3"])
    grader, measured, runs, parity = Grade(), {}, [], []

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

    controls, measured["control"] = guarded("control", lambda: run_control(config, grader, s31_out), (None, {}))
    measured["patch"] = guarded("patch", lambda: run_patch(grader, receipts), {})
    if controls is not None:
        measured["flat"] = guarded("flat", lambda: run_flat(config, out, grader, tools, runs, controls, parity))
        measured["varying"], _ = guarded("varying", lambda: run_class(config, out, grader, tools, runs, controls, parity,
                                                                    "varying", "varying"), ({}, None))
        measured["noisy"], noisy = guarded("noisy", lambda: run_class(config, out, grader, tools, runs, controls, parity,
                                                                    "noisy", "noisy"), ({}, None))
        measured["hook-off"] = guarded("hook-off", lambda: run_hook_off(config, out, grader, tools, runs, controls, noisy))
        measured["side-stream"], _ = guarded("side-stream", lambda: run_class(
            config, out, grader, tools, runs, controls, parity, "side-stream", "noisy", copy_stream="side"), ({}, None))
        measured["timed"] = guarded("timed", lambda: run_timed(config, out, grader, tools, runs, controls, noisy))
        measured["refusals"] = guarded("refusals", lambda: run_refusals(config, out, grader, tools, controls))
        measured["backends"] = guarded("backends", lambda: run_backends(config, grader, runs))
    else:
        for case in CASES[2:]:
            grader.check(case, "control", False, "the Sprint 3.1 run does not hold the control this case is compared with")

    failed = [item for item in grader.results if not item["passed"]]
    passed_cases = [case for case in CASES if any(item["case"] == case for item in grader.results)
                    and all(item["passed"] for item in grader.results if item["case"] == case)]
    complete = not failed and len(passed_cases) == len(CASES)
    grade = {
        "schema": GRADE_SCHEMA, "selector": "native_estimate", "run_id": out.name, "cases": len(CASES),
        "cases_passed": len(passed_cases), "cases_failed": len(CASES) - len(passed_cases),
        "checks": len(grader.results), "checks_passed": len(grader.results) - len(failed), "checks_failed": len(failed),
        "judgment": "pass" if complete else "fail", "hardware_gate": "closed" if complete else "open",
        "reason": "every check of every case passed" if not failed else
                  f"{len(failed)} check(s) failed, first: {failed[0]['case']} {failed[0]['check']}",
        "backends": config["backends"], "estimate": config["estimate"], "tolerance": config["tolerance"],
        "record_definition": f"{obs.DEFINITION:016x}", "results": grader.results}
    write_json(out / "grade.json", grade)
    write_json(out / "comparison.json", measured)
    totals = {"numerator": sum(item["totals"]["numerator"] for item in parity),
              "denominator": sum(item["totals"]["denominator"] for item in parity),
              "samples": sum(item["totals"]["samples"] for item in parity)}
    write_json(out / "o3_parity.json", {
        "schema": ne.PARITY_SCHEMA, "run_id": out.name, "stage": reference.STAGE, "reference": "the matched estimator "
        "of ocudu_cuda_isac.sensing.native_estimate on the stage O2 of the same opportunity, on the pilot subcarriers",
        "rule": reference.RULE, "tolerance": config["tolerance"], "study": config["study"],
        "patch": measured.get("patch", {}).get("series"), "passed": bool(parity) and all(item["passed"] for item in parity),
        "totals": dict(totals, nmse=totals["numerator"] / totals["denominator"] if totals["denominator"] else None),
        "runs": parity})
    lock = {item["name"]: item["commit"] for item in json.loads((ROOT / "ext/lock.json").read_text())["dependencies"]}
    write_json(out / "manifest.json", {
        "run_id": out.name, "selector": "native_estimate",
        "purpose": "Sprint 5.2 integration: the plans of the accepted Sprint 3.1 run decoded by the patched receiver, "
                   "with its own channel estimate exported as the stage O3 beside the stage O2, compared with the matched "
                   "estimator, with the decodes of the unpatched build and with the accepted stage O2",
        "command": shlex.join([sys.executable, *sys.argv]), "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": platform.node(), "machine": platform.machine(), "python": platform.python_version(),
        "parent_commit": git_commit(), "locked_sources": lock,
        "inputs": {str(args.config): digest(args.config)}, "driver_sha256": digest(__file__),
        "reader_sha256": digest(ROOT / "src/ocudu_cuda_isac/episodes/native_estimate.py"),
        "reference_sha256": digest(ROOT / "src/ocudu_cuda_isac/sensing/native_estimate.py"),
        "s31_run": str(s31_run), "control": measured["control"].get("s31"), "patch": measured.get("patch"),
        "axes": {"split": "staged_harness", "receiver_stage": "o3_native_estimate", "ue_realization": "replayed channel, "
                 "no UE", "host_stack": "DeepSig-OCUDU with the patch series", "sensing_source": config["observer"]["source"],
                 "placement": "one host, the receiver and one consumer per stream", "transport": "obs-ring/v1 in host "
                 "shared memory", "export_path": "host copy from the device on the completion event of the copy"},
        "runs": [{"name": run.name, "directory": str(run.directory.relative_to(out)), "status": run.status,
                  "plan_sha256": digest(run.directory / "plan.json") if (run.directory / "plan.json").is_file() else None,
                  "mode": run.plan["evidence"]["mode"] if run.plan else None,
                  "estimate": run.plan["estimate"] if run.plan else None,
                  "trace": str((run.native / ne.ESTIMATE_DIR / ne.CAPTURES_FILE).relative_to(out))
                  if (run.native / ne.ESTIMATE_DIR / ne.CAPTURES_FILE).is_file() else None} for run in runs],
        "adapters": {"o3": {"receipt": "o3-adapter-receipt.json", "ocudu_build": receipts["o3"]["ocudu_build"],
                            "pilot_ls_sha256": receipts["o3"]["executables"]["pilot_ls"]["sha256"]},
                     "control": {"receipt": "adapter-receipt.json", "ocudu_build": receipts["control"]["ocudu_build"],
                                 "pilot_ls_sha256": receipts["control"]["executables"]["pilot_ls"]["sha256"]}},
        "environment": {key: os.environ[key] for key in ("VIRTUAL_ENV", "CUDA_VISIBLE_DEVICES",
                                                         "OCUDU_PUSCH_ACCELERATION_TIMING") if key in os.environ},
        "judgment": grade["judgment"], "reason": grade["reason"]})
    summary = (f"native_estimate: {grade['cases_passed']} of {grade['cases']} cases passed, "
               f"{grade['checks_passed']} of {grade['checks']} checks")
    (out / "grade.txt").write_text("\n".join(grader.lines + [summary]) + "\n", encoding="ascii")
    print(summary)
    print(f"Run artifacts: {out}")
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
