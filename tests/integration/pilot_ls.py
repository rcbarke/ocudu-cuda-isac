#!/usr/bin/env python3
"""Sprint 3.1 integration driver (selector pilot_ls). Hardware command, not part of unit discovery.

It runs the tapped receiver replay, built against the accepted OCUDU build, on the plans that the
accepted Sprint 2.4 run archived. Every grid is filled as in that run, the observer computes the
pilot least squares of the grant's sensing source on the ready grid (the stage O2), and the host
and the CUDA PUSCH processor then decode the same grid. The radio records of the accepted Sprint
2.4 run are the control: they were written by the accepted replay_pusch without an observer. Nine
cases, each graded on its own:

  control          the accepted Sprint 2.4 run, the plans it archived and its radio records
  unity            the unity grids of the shipped grants: every pilot of every receive plane reads
                   as 1.00119, at the coordinates of its grant, and the decodes are unchanged
  direct           the direct path without noise: the estimate is the known channel times the gain
                   and the pilot ratio on all receive planes, within the rounding of the grid
  reference-pair   both members of the pair: the observation names the grid digest the control
                   decoded, the reference recomputes every estimate bit for bit, a sparse oracle
                   in double precision agrees, and every radio record equals the control's
  payload          a transport block per opportunity: other data, another grid, the same estimate
  tap-off          the same executable without the tap: no observation, the same radio records
  rerun            one opportunity repeated alone from the archived plan, with the tap
  refusals         every plan rule, refused by the plan check and by the native tool before a file
                   is written
  backends         the path every tapped decode executed, and the count of every planned decode

    python tests/integration/pilot_ls.py --config configs/integration/pilot_ls.json \\
        --s24-run results/runs/s24-spark-001 --output-dir results/runs/s31-spark-001

The output directory must be new. The exit status is 0 only when every check of every case passed.
On a host without a CUDA build, --host-rehearsal runs the same cases on a rehearsed Sprint 2.4 run.
Such a run is graded as a rehearsal and cannot close the hardware gate.
"""

import argparse
import copy
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
from ocudu_cuda_isac.episodes import grant as grants  # noqa: E402
from ocudu_cuda_isac.episodes import grid_apply as replay  # noqa: E402
from ocudu_cuda_isac.episodes import pilot_ls as pl  # noqa: E402
from ocudu_cuda_isac.episodes import replay_pusch as rp  # noqa: E402
from ocudu_cuda_isac.episodes.grid_apply import GridApplyError  # noqa: E402
from ocudu_cuda_isac.episodes.pilot_ls import PilotLsPlanError  # noqa: E402
from ocudu_cuda_isac.episodes.replay_pusch import ReplayPuschError  # noqa: E402
from ocudu_cuda_isac.replay import ContractError, load_json  # noqa: E402
from ocudu_cuda_isac.sensing import pilot_ls as reference  # noqa: E402
from ocudu_cuda_isac.sensing.pilot_ls import PilotLsError  # noqa: E402
import native_adapter  # noqa: E402

CONFIG_SCHEMA = "integration-config/pilot_ls/v1"
GRADE_SCHEMA = "integration-grade/pilot_ls/v1"
CONTROL_GRADE_SCHEMA = "integration-grade/replay_pusch/v1"
CASES = ("control", "unity", "direct", "reference-pair", "payload", "tap-off", "rerun", "refusals", "backends")
TOLERANCES = ("storage_relative", "absolute", "oracle_relative", "phase_rad")
CONTROL_RUNS = ("unity", "direct", "pair", "payload")
NATIVE_TIMEOUT_S = 1800


class ConfigError(ValueError):
    """The configuration or the run setup is unusable. Nothing was graded."""


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


apply_driver = load_script("grid_apply_driver", ROOT / "tests/integration/grid_apply.py")
fixture = apply_driver.fixture
digest, write_json, Grader = fixture.digest, fixture.write_json, fixture.Grader
last_line, EVALUATOR_WORDS = apply_driver.last_line, apply_driver.EVALUATOR_WORDS


class Grade(Grader):
    """The Sprint 2.1 grader with this sprint's label."""

    def check(self, case, name, passed, detail):
        self.results.append({"case": case, "check": name, "passed": bool(passed), "detail": detail})
        self._line(("PASS" if passed else "FAIL") + f"  S3.1 {case} {name}: {detail}")
        return bool(passed)

    def note(self, case, text):
        self._line(f"NOTE  S3.1 {case} {text}")


def load_config(path):
    try:
        config = load_json(Path(path))
    except (ContractError, OSError) as error:
        raise ConfigError(f"cannot read the configuration: {error}") from None
    needed = {"schema", "selector", "adapter_build", "backends", "observer", "control", "tolerance", "unity", "pair"}
    if not isinstance(config, dict) or set(config) != needed or config["schema"] != CONFIG_SCHEMA \
            or config["selector"] != "pilot_ls":
        raise ConfigError(f"the configuration must be a {CONFIG_SCHEMA} object for the selector pilot_ls")
    tolerance = config["tolerance"]
    if not isinstance(tolerance, dict) or set(tolerance) != set(TOLERANCES) \
            or any(type(value) not in (int, float) or not value > 0 for value in tolerance.values()):
        raise ConfigError(f"tolerance requires exactly {', '.join(TOLERANCES)}, each positive")
    # Both backends decode every tapped grid, as in the control.
    if list(config["backends"]) != list(rp.BACKENDS):
        raise ConfigError('backends must be ["cpu", "cuda"]: the gate decodes every tapped grid with both')
    observer = config["observer"]
    if not isinstance(observer, dict) or set(observer) != {"tap", "stage", "source", "epoch"} \
            or (observer["tap"], observer["stage"]) != ("pilot_ls", reference.STAGE) \
            or type(observer["epoch"]) is not int or observer["epoch"] < 1 or not isinstance(observer["source"], str):
        raise ConfigError("observer requires the tap pilot_ls, the stage O2, a source and an epoch from 1")
    control = config["control"]
    if not isinstance(control, dict) or set(control) != {"driver", "runs"} or control["driver"] != "replay_pusch" \
            or not isinstance(control["runs"], dict) or set(control["runs"]) != set(CONTROL_RUNS) \
            or not all(isinstance(name, str) and name for name in control["runs"].values()):
        raise ConfigError(f"control requires the driver replay_pusch and the runs {', '.join(CONTROL_RUNS)}")
    unity = config["unity"]
    if not isinstance(unity, dict) or set(unity) != {"expect"} or not isinstance(unity["expect"], dict) \
            or set(unity["expect"]) != {"estimate", "shape", "first_subcarrier"}:
        raise ConfigError("unity requires expect with estimate, shape and first_subcarrier")
    pair = config["pair"]
    if not isinstance(pair, dict) or set(pair) != {"expect"} or not isinstance(pair["expect"], dict) \
            or set(pair["expect"]) != {"opportunities", "shape", "payload_bytes"}:
        raise ConfigError("pair requires expect with opportunities, shape and payload_bytes")
    return config


# ---------------------------------------------------------------- native tool
def run_native(tool, plan_path, output, log, extra=()):
    """Run pilot_ls on one plan. Returns (exit status, report or None). output is created empty."""
    output.mkdir(parents=True)
    command = [str(tool), "--plan", str(plan_path), "--output-dir", str(output), *extra]
    with log.open("w") as stream:
        stream.write(shlex.join(command) + "\n")
        stream.flush()
        try:
            status = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=NATIVE_TIMEOUT_S,
                                    check=False).returncode
        except (OSError, subprocess.TimeoutExpired) as error:
            stream.write(f"{type(error).__name__}: {error}\n")
            return -1, None
    try:
        return status, json.loads((output / "native.json").read_text())
    except (OSError, ValueError):
        return status, None


class Control:
    """One run of the accepted Sprint 2.4 gate: its archived plans and its radio records without a tap."""

    def __init__(self, name, directory, plan_sha256):
        self.name, self.directory, self.plan_sha256 = name, directory, plan_sha256
        self.plan_path = directory / "plan.json"
        self.output = self.plan = self.channel = self.problem = None

    def open(self):
        try:
            if digest(self.plan_path) != self.plan_sha256:
                raise ReplayPuschError("plan.json does not have the hash the manifest of the Sprint 2.4 run names")
            self.plan = rp.load_plan(self.plan_path)
            self.channel = replay.load_plan(self.plan["channel_plan"]["path"])
            self.output = rp.open_output(self.directory / "native")
        except (ReplayPuschError, GridApplyError, OSError) as error:
            self.problem = f"{self.name}: {error}"
        return self.problem is None


class Run:
    """One native run of this gate: its plan, its verified output, its sources and its control."""

    def __init__(self, name, directory, control):
        self.name, self.directory, self.control = name, directory, control
        self.plan = self.report = self.output = None
        self.status, self.problem, self.sources = None, None, {}

    @property
    def native(self):
        return self.directory / "native"

    def ok(self):
        return self.status == 0 and self.output is not None and self.problem is None

    def said(self):
        return self.problem or f"the native tool exited with status {self.status}: " \
                               f"{last_line(self.directory / 'native.log')}"

    def differences(self):
        """Radio records of this run against those the accepted replay_pusch wrote for the same plan."""
        return pl.radio_differences(self.output.records, self.control.output.records)

    def counts(self):
        totals = {}
        for record in self.output.records:
            total = totals.setdefault(record["backend"], [0, 0])
            total[0] += record["outcome"] == "decoded"
            total[1] += 1
        return ", ".join(f"{name} {decoded} of {planned} decoded" for name, (decoded, planned) in totals.items())


def tapped_run(runs, directory, name, tool, run_id, control, config, tap="pilot_ls", extra=()):
    """Write the plan of a run on a control plan, run the tool and verify its output."""
    run = Run(name, directory, control)
    runs.append(run)
    directory.mkdir(parents=True)
    run.plan = pl.build_plan(run_id, control.plan_path, tap, epoch=config["observer"]["epoch"],
                             source=config["observer"]["source"])
    pl.write_plan(run.plan, directory / "plan.json")
    run.status, run.report = run_native(tool, directory / "plan.json", run.native, directory / "native.log", extra)
    if run.status != 0 or not run.report:
        return run
    try:
        run.output = pl.open_output(run.native)
        _, _, run.sources = pl.check_files(run.plan)
    except (PilotLsPlanError, PilotLsError) as error:
        run.problem = f"the native output does not verify: {error}"
    return run


def run_is_clean(run, backends):
    """True when a run finished, verifies, and holds one record per planned decode and one observation per grid."""
    if not run.ok():
        return False
    totals, opportunities = run.report["totals"], len(run.report["opportunities"])
    wanted = opportunities if run.output.tap == "pilot_ls" else 0
    return len(run.output.records) == opportunities * len(backends) == totals["planned"] \
        and not run.report["anomalies"] and totals["grids_not_released"] == 0 and totals["retained"] is True \
        and len(run.output.observations) == wanted == totals["observations"] \
        and len({record["key"] for record in run.output.records}) == len(run.output.records)


def clean_text(run):
    return (f"{len(run.output.observations)} observations on {len(run.report['opportunities'])} grids, {run.counts()}; "
            "one record per planned decode under a unique key, no anomaly, every grid released") if run.ok() \
        else run.said()


def unchanged_text(run):
    found = run.differences()
    return found, (f"{len(run.output.records)} radio records equal those of the control {run.control.name} in the key, "
                   "the grant, the allocation, the grid digest, the outcome, the CRC flag, the hash of the decoded "
                   "bytes and the executed path" if not found else
                   f"{len(found)} differences from the control {run.control.name}, first: {found[0]}")


def base_grant(grant_id):
    """The shipped grant a slot grant or a payload grant was derived from."""
    return grant_id.split(".s")[0].split(".p")[0]


def values_of(np, run, record):
    """The payload of one observation as complex64 [rx_plane, reference_symbol, pilot]."""
    return np.frombuffer(run.output.payload(record), dtype="<c8").reshape(record["shape"]["shape"])


def text(value, digits=2):
    return "none" if value is None else f"{value:.{digits}f}"


# ---------------------------------------------------------------- control
def run_control(config, grader, s24_out, rehearsal):
    """The accepted Sprint 2.4 run and the runs of it that this gate taps. Returns ({role: [Control]}, measured)."""
    name, controls, measured = "control", {}, {}
    try:
        grade = load_json(s24_out / "grade.json")
        manifest = load_json(s24_out / "manifest.json")
    except (ContractError, OSError) as error:
        grader.check(name, "accepted run", False, f"{s24_out} holds no grade and manifest of the Sprint 2.4 driver: {error}")
        return controls, measured
    wanted = "rehearsal" if rehearsal else "pass"
    accepted = isinstance(grade, dict) and grade.get("schema") == CONTROL_GRADE_SCHEMA \
        and grade.get("judgment") == wanted and grade.get("cases_passed") == grade.get("cases") \
        and grade.get("checks_failed") == 0 and (rehearsal or grade.get("hardware_gate") == "closed")
    measured["s24"] = {"run_id": manifest.get("run_id"), "judgment": grade.get("judgment"),
                       "hardware_gate": grade.get("hardware_gate"), "parent_commit": manifest.get("parent_commit"),
                       "adapter_sha256": (manifest.get("adapter") or {}).get("sha256")}
    grader.check(name, "accepted run", accepted,
                 f"the Sprint 2.4 driver graded {s24_out.name} as {grade.get('judgment')}: {grade.get('cases_passed')} of "
                 f"{grade.get('cases')} cases, {grade.get('checks_failed')} failed checks, hardware gate "
                 f"{str(grade.get('hardware_gate')).split(':')[0]}; its replay_pusch is "
                 f"{str(measured['s24']['adapter_sha256'])[:12]}"
                 + ("" if accepted else f"; this gate needs the judgment {wanted}"
                    + ("" if rehearsal else " with the hardware gate closed")))
    listed = {item["name"]: item for item in manifest.get("runs", []) if isinstance(item, dict)}
    problems, opened = [], 0
    for role, wanted_name in config["control"]["runs"].items():
        names = sorted(run for run in listed if run == wanted_name or run.startswith(wanted_name + "/"))
        controls[role] = []
        if not names:
            problems.append(f"the Sprint 2.4 run holds no run named {wanted_name}")
        for run_name in names:
            item = listed[run_name]
            control = Control(run_name, s24_out / item["directory"], item.get("plan_sha256"))
            if item.get("status") == 0 and control.open():
                controls[role].append(control)
                opened += 1
            else:
                problems.append(control.problem or f"{run_name}: the tool of Sprint 2.4 exited with status "
                                                   f"{item.get('status')}")
    records = sum(len(control.output.records) for group in controls.values() for control in group)
    untapped = all(record["tap"] == "none" for group in controls.values() for control in group
                   for record in control.output.records)
    measured["runs"] = {role: [control.name for control in group] for role, group in controls.items()}
    measured["records"] = records
    grader.check(name, "records", not problems and untapped and opened > 0,
                 f"{opened} runs of the Sprint 2.4 gate verify with the reader of that sprint: each plan has the hash "
                 f"its manifest names, {records} radio records, each written without a tap by the accepted "
                 "replay_pusch (" + ", ".join(control.name for group in controls.values() for control in group) + ")"
                 if not problems and untapped else "; ".join(problems) or "a control record names a tap")
    if problems:
        return {}, measured
    return controls, measured


# ---------------------------------------------------------------- unity
def run_unity(config, out, grader, np, tool, runs, controls):
    name, expect, backends = "unity", config["unity"]["expect"], config["backends"]
    case_dir = out / "unity"
    case_dir.mkdir()
    measured, done = {}, []
    for control in controls:
        label = control.name.split("/")[-1]
        run = tapped_run(runs, case_dir / label / "run", f"unity/{label}", tool, f"s31-unity-{label}", control, config)
        done.append((label, run))
    clean = all(run_is_clean(run, backends) and not run.differences() for _, run in done) and bool(done)
    grader.check(name, "native run", clean,
                 "; ".join(f"{run.name}: {clean_text(run)}" + ("" if not run.ok() or not run.differences() else
                                                               f"; {unchanged_text(run)[1]}") for _, run in done)
                 + "; a unity channel with zero noise and g = 1, tapped before each decode; every radio record equals "
                   "the control's")
    unity = np.float32(expect["estimate"])
    for label, run in done:
        if not run.ok():
            continue
        rows = pl.oracle(run.output, run.sources)
        seen, ok = {}, all(row["payload_equal"] and row["mask_equal"] and row["coordinates_equal"] for row in rows)
        for record in run.output.observations:
            values = values_of(np, run, record)
            grant = base_grant(record["host"]["grant_id"])
            exact = bool(np.all(values.real == unity) and np.all(values.imag == 0.0))
            shape_ok = record["shape"]["shape"] == expect["shape"].get(grant) \
                and record["allocation"]["first_subcarrier"] == expect["first_subcarrier"].get(grant) \
                and record["allocation"]["subcarrier_step"] == 2 and record["validity"]["holes"] == 0 \
                and record["validity"]["valid"] == values.size
            ok = ok and exact and shape_ok
            entry = seen.setdefault(grant, {"grids": 0, "pilots": 0, "shape": record["shape"]["shape"],
                                            "first_subcarrier": record["allocation"]["first_subcarrier"],
                                            "min": float(values.real.min()), "max": float(values.real.max())})
            entry["grids"] += 1
            entry["pilots"] += int(values.size)
            entry["min"], entry["max"] = min(entry["min"], float(values.real.min())), max(entry["max"], float(values.real.max()))
        measured[label] = seen
        grader.check(name, f"{label} estimate", ok and bool(seen),
                     "; ".join(f"{grant}: {entry['grids']} grids, {entry['pilots']} estimates of shape {entry['shape']} "
                               f"from subcarrier {entry['first_subcarrier']} in steps of two, each between "
                               f"{entry['min']:.7f} and {entry['max']:.7f} with a zero imaginary part and a set "
                               "validity bit" for grant, entry in sorted(seen.items()))
                     + f"; expected {float(unity):.7f}, the stored pilot of magnitude one per component against the "
                       f"reference of {1.0 / float(unity):.6f}; the reference recomputes {len(rows)} of {len(rows)} "
                       "payloads bit for bit" if ok else
                     "an estimate of a unity grid is not the pilot ratio on every pilot, has another shape or another "
                     "first subcarrier than its grant, or differs from the reference: "
                     + "; ".join(f"{grant}: shape {entry['shape']}, first subcarrier {entry['first_subcarrier']}, values "
                                 f"{entry['min']:.7f} to {entry['max']:.7f}" for grant, entry in sorted(seen.items())))
    return measured


# ---------------------------------------------------------------- direct path
def run_direct(config, out, grader, np, tool, runs, control):
    name, backends, tolerance = "direct", config["backends"], config["tolerance"]
    run = tapped_run(runs, out / "direct" / "run", "direct", tool, "s31-direct", control, config)
    found, said = unchanged_text(run) if run.ok() else ([], "")
    grader.check(name, "native run", run_is_clean(run, backends) and not found,
                 f"the direct path of the pair without noise, tapped before each decode: {clean_text(run)}"
                 + (f"; {said}" if run.ok() else ""))
    if not run.ok():
        return {}
    channel = control.channel
    gain = channel["gain"]["value"]
    x_of = {item["grant_id"]: np.load(item["x"]["path"], allow_pickle=False) for item in channel["grants"]}
    h_of = {item["realization_id"]: np.load(item["h"]["path"], mmap_mode="r", allow_pickle=False)
            for item in channel["sources"]}
    planned = {item["opportunity_id"]: item for item in channel["opportunities"]}
    inside = total = swapped_inside = 0
    worst, phase_error, planes = 0.0, 0.0, 0
    for record in run.output.observations:
        item = planned[record["host"]["opportunity_id"]]
        source = run.sources[record["host"]["grant_record_sha256"]]
        values = values_of(np, run, record).astype(np.complex128)
        symbols, subcarriers = np.asarray(source.symbols), np.asarray(source.subcarriers)
        h = np.asarray(h_of[item["realization_id"]][item["index"]])[:, 0][:, symbols][:, :, subcarriers].astype(np.complex128)
        sent = x_of[item["grant_id"]][0][symbols][:, subcarriers].astype(np.complex128)
        pilots = np.asarray(source.reference, dtype=np.complex128).reshape(len(symbols), len(subcarriers))
        # The grid holds g H X after rounding, and the estimate divides by the reference of the source.
        wanted = gain * h * (sent / pilots)[None, :, :]
        allowed = tolerance["storage_relative"] * np.abs(wanted) + tolerance["absolute"]
        error = np.abs(values - wanted)
        inside += int(np.count_nonzero(error <= allowed))
        total += int(values.size)
        worst = max(worst, float(np.max(error / np.maximum(np.abs(wanted), 1e-30))))
        planes = values.shape[0]
        if planes > 1:
            # The phase from one receive plane to the next, of the estimate and of the known channel.
            for plane in range(planes - 1):
                measured_step = np.angle(np.sum(values[plane + 1] * np.conj(values[plane])))
                known_step = np.angle(np.sum(wanted[plane + 1] * np.conj(wanted[plane])))
                phase_error = max(phase_error, float(abs(np.angle(np.exp(1j * (measured_step - known_step))))))
            swapped_inside += int(np.count_nonzero(np.abs(values[::-1] - wanted) <= allowed))
    measured = {"estimates": total, "inside": inside, "max_relative_error": worst, "gain": gain, "planes": planes,
                "max_phase_step_error_rad": phase_error, "inside_with_reversed_planes": swapped_inside}
    grader.check(name, "known channel", total > 0 and inside == total,
                 f"{inside} of {total} estimates on {len(run.output.observations)} grids and {planes} receive planes "
                 f"equal g H times the pilot ratio, with g = {gain:.4f}, H of the archived channel and the stored pilot "
                 f"over the reference of the source, within the rounding of the grid ({tolerance['storage_relative']:g} "
                 f"of the magnitude plus {tolerance['absolute']:g}); the largest relative error is {worst:.6f}")
    if planes > 1:
        grader.check(name, "receive planes", phase_error <= tolerance["phase_rad"] and swapped_inside < total // 2,
                     f"the phase from each receive plane to the next agrees with the known channel within "
                     f"{phase_error:.6f} rad (allowed {tolerance['phase_rad']:g}), and with the planes in reverse order "
                     f"only {swapped_inside} of {total} estimates stay inside the tolerance: every plane of the payload "
                     "is the plane of the grid that its antenna map names")
    else:
        grader.check(name, "receive planes", False, "the direct path of the gate has one receive plane, expected the "
                                                    "four of the demonstration")
    return measured


# ---------------------------------------------------------------- reference pair
def run_pair(config, out, grader, np, tool, runs, control):
    name, backends, tolerance, expect = "reference-pair", config["backends"], config["tolerance"], config["pair"]["expect"]
    run = tapped_run(runs, out / "reference_pair" / "run", "reference-pair", tool, "s31-pair", control, config)
    clean = run_is_clean(run, backends)
    shapes = sorted({tuple(record["shape"]["shape"]) for record in run.output.observations}) if run.ok() else []
    sizes = sorted({record["payload"]["bytes"] for record in run.output.observations}) if run.ok() else []
    grader.check(name, "native run", clean and len(run.output.observations) == expect["opportunities"]
                 and shapes == [tuple(expect["shape"])] and sizes == [expect["payload_bytes"]],
                 f"both members of the pair with the gain and the noise of the entry, tapped before each decode: "
                 f"{clean_text(run)}; every payload has the shape {list(shapes[0]) if shapes else None} and "
                 f"{sizes[0] if sizes else 0} bytes, in the layout {reference.LAYOUT}" if run.ok() else run.said())
    if not run.ok():
        return {}, None
    output, measured = run.output, {}

    # ---- The observation was computed from the grid that the control decoded.
    by_opportunity = {}
    for record in control.output.records:
        by_opportunity.setdefault(record["opportunity_id"], set()).add(record["ingress"]["readback_sha256"])
    stored = {descriptor["opportunity_id"]: descriptor["grid"]["sha256"] for descriptor in control.output.descriptors}
    joined = sum(by_opportunity.get(record["host"]["opportunity_id"]) == {record["host"]["ingress"]["readback_sha256"]}
                 and stored.get(record["host"]["opportunity_id"]) == record["host"]["ingress"]["readback_sha256"]
                 for record in output.observations)
    count = len(output.observations)
    grader.check(name, "ingress", joined == count,
                 f"{joined} of {count} observations name the readback digest of the grid that the accepted replay_pusch "
                 f"decoded for the same opportunity on both backends (first {output.observations[0]['host']['ingress']['readback_sha256'][:16]}): "
                 "O2 is taken from the exact O1 grid that was passed to PUSCH")

    # ---- The reference recomputes every estimate from the stored grid and the source record.
    rows = pl.oracle(output, run.sources)
    exact = sum(row["payload_equal"] and row["mask_equal"] for row in rows)
    described = sum(row["coordinates_equal"] and row["identity_equal"] for row in rows)
    late = pl.expected_times(output, run.sources)
    first = output.observations[0]
    grader.check(name, "reference", exact == described == count and not late,
                 f"{exact} of {count} payloads and validity masks equal the reference of ocudu_cuda_isac.sensing.pilot_ls "
                 f"bit for bit, computed from the stored grid and source.json of the grant; {described} of {count} "
                 f"records name the source {first['what']['source']}, port {first['what']['port']}, the reference "
                 f"amplitude {first['what']['scale']:.6f}, the stage {first['what']['stage']}, the producer "
                 f"{first['what']['producer']} and the coordinates of the source; {count - len(late)} of {count} carry "
                 "the slot count and the sample times the reference derives; precision: " + first["what"]["precision"])

    # ---- A separate sparse oracle in double precision, and nonzero estimates at the declared pilots.
    worst, nonzero, elements, valid = 0.0, 0, 0, 0
    descriptors = output.descriptors_by_sequence()
    for record in output.observations:
        descriptor = descriptors[record["host"]["opportunity_sequence"]]
        source = run.sources[record["host"]["grant_record_sha256"]]
        grid = np.fromfile(run.native / "ingress" / descriptor["grid"]["file"], dtype="<c8").reshape(descriptor["grid"]["shape"])
        symbols, subcarriers = np.asarray(source.symbols), np.asarray(source.subcarriers)
        pilots = np.asarray(source.reference, dtype=np.complex128).reshape(len(symbols), len(subcarriers))
        received = grid[:, symbols][:, :, subcarriers].astype(np.complex128)
        oracle = received * np.conj(pilots)[None] / (np.abs(pilots) ** 2)[None]
        values = values_of(np, run, record)
        worst = max(worst, float(np.max(np.abs(values - oracle) / np.maximum(np.abs(oracle), 1e-30))))
        nonzero += int(np.count_nonzero(values))
        elements += int(values.size)
        valid += record["validity"]["valid"]
    measured.update(oracle_max_relative_error=worst, estimates=elements, nonzero=nonzero, valid=valid)
    grader.check(name, "sparse oracle", worst <= tolerance["oracle_relative"] and nonzero == elements == valid,
                 f"a least-squares oracle in double precision on the pilots of the stored grids agrees within "
                 f"{worst:.2e} of the magnitude (allowed {tolerance['oracle_relative']:g}); {nonzero} of {elements} "
                 f"estimates at the declared pilots are nonzero and {valid} validity bits are set")

    # ---- The decodes with the tap on are the decodes of the control.
    found, said = unchanged_text(run)
    same_bytes = sum(output.radio.decoded(record) == control.output.decoded(control.output.by_key[record["key"]])
                     for record in output.records if record["key"] in control.output.by_key)
    passes = sum(record["crc_ok"] for record in output.records)
    measured.update(records=len(output.records), crc_pass=passes, differences=len(found))
    grader.check(name, "decodes", not found and same_bytes == len(output.records),
                 f"with the tap on, {said}; {passes} decodes pass the CRC and {same_bytes} of {len(output.records)} "
                 "files of decoded bytes are those of the control, byte for byte")

    # ---- Access: nothing of the evaluator reaches the observer.
    forbidden = list(EVALUATOR_WORDS)
    forbidden += [item[key]["sha256"] for item in control.channel["sources"] for key in ("h", "w")]
    forbidden += [grant["tb"]["sha256"] for grant in control.plan["grants"]]
    forbidden += [Path(item[key]["path"]).name for item in control.channel["sources"] for key in ("h", "w")]
    violations = pl.observer_violations(run.native, forbidden) + replay.ingress_violations(run.native, forbidden)
    grader.check(name, "access", not violations,
                 "observer/ holds one record, one payload and one validity mask per opportunity, with the fields of an "
                 f"{reference.DESCRIPTOR_SCHEMA} record and no name or hash of H, W or a transport block and no label; "
                 "ingress/ is as the control leaves it" if not violations else "; ".join(violations[:4]))

    # ---- The two members, recorded and not graded.
    members = {}
    for record in output.observations:
        members.setdefault(record["keys"]["episode"], []).append(values_of(np, run, record).astype(np.complex128))
    if len(members) == 2:
        first_member, second_member = (np.stack(values) for values in members.values())
        if first_member.shape == second_member.shape:
            power = float(np.mean(np.abs(first_member) ** 2))
            change = float(np.mean(np.abs(second_member - first_member) ** 2))
            measured["member_difference_db"] = 10 * math.log10(change / power) if change > 0 and power > 0 else None
            grader.note(name, f"the estimates of the two members differ by {text(measured['member_difference_db'], 1)} dB "
                              "of the estimate's power over the pilots of all opportunities, not graded: the members "
                              "share the noise, so this is the return of the target and the rounding of the grid")
    times = [record["host"]["service_us"] for record in output.observations[1:]]
    if times:
        times.sort()
        measured["tap_service_us"] = {"count": len(times), "mean": sum(times) / len(times),
                                      "p50": times[len(times) // 2], "max": times[-1]}
        grader.note(name, f"service time of the tap, not graded: mean {sum(times) / len(times):.0f} us, median "
                          f"{times[len(times) // 2]:.0f} us, largest {times[-1]:.0f} us over {len(times)} observations "
                          f"of {first['shape']['shape']}, on the host, the first one excluded; the host is not tuned")
    return measured, run


# ---------------------------------------------------------------- payload
def run_payload(config, out, grader, tool, runs, control, pair_run):
    name, backends = "payload", config["backends"]
    run = tapped_run(runs, out / "payload" / "run", "payload", tool, "s31-payload", control, config)
    found, said = unchanged_text(run) if run.ok() else ([], "")
    grader.check(name, "native run", run_is_clean(run, backends) and not found,
                 f"the pair with a transport block per opportunity, tapped before each decode: {clean_text(run)}"
                 + (f"; {said}" if run.ok() else ""))
    if not run.ok() or pair_run is None or not pair_run.ok():
        grader.check(name, "data invariance", False, "the tapped run of the pair or of the payload did not finish")
        return {}
    fixed = {record["host"]["opportunity_id"]: record for record in pair_run.output.observations}
    same = other_grid = 0
    for record in run.output.observations:
        earlier = fixed.get(record["host"]["opportunity_id"])
        if earlier is None:
            continue
        same += record["payload"]["digest"] == earlier["payload"]["digest"] \
            and record["validity"]["sha256"] == earlier["validity"]["sha256"]
        other_grid += record["host"]["ingress"]["readback_sha256"] != earlier["host"]["ingress"]["readback_sha256"]
    count = len(run.output.observations)
    blocks = len({record["decoded_sha256"] for record in run.output.records if record["crc_ok"]})
    grader.check(name, "data invariance", same == other_grid == count == len(fixed),
                 f"{other_grid} of {count} grids differ from the grid of the same opportunity with the fixed block, and "
                 f"{same} of {count} estimates are the estimate of that grid bit for bit: the data elements changed "
                 f"({blocks} distinct decoded blocks) and O2 did not")
    return {"observations": count, "same_estimate": same, "other_grid": other_grid, "blocks": blocks}


# ---------------------------------------------------------------- tap off
def run_tap_off(config, out, grader, tool, runs, control, pair_run):
    name, backends = "tap-off", config["backends"]
    run = tapped_run(runs, out / "tap_off" / "run", "tap-off", tool, "s31-pair-off", control, config, tap="none")
    grader.check(name, "native run", run_is_clean(run, backends) and not (run.native / pl.OBSERVER_DIR).exists(),
                 f"the pair through the same executable without the tap: {clean_text(run)}; the output holds no "
                 "observer directory" if run.ok() else run.said())
    if not run.ok():
        grader.check(name, "records", False, "the run without the tap did not finish")
        return {}
    found, said = unchanged_text(run)
    between = pl.radio_differences(pair_run.output.records, run.output.records) if pair_run is not None and pair_run.ok() \
        else ["the tapped run of the pair did not finish"]
    grader.check(name, "records", not found and not between,
                 f"without the tap, {said}; and the records with the tap on equal the records with the tap off of the "
                 "same executable" if not found and not between else
                 f"{len(found)} differences from the control and {len(between)} between tap on and tap off, first: "
                 f"{(found + between)[0]}")
    return {"differences_from_control": len(found), "differences_on_off": len(between)}


# ---------------------------------------------------------------- rerun
def run_rerun(config, out, grader, tool, pair_run):
    name = "rerun"
    if pair_run is None or not pair_run.ok():
        grader.check(name, "one opportunity", False, "the tapped run of the pair did not finish")
        return {}
    chosen = pair_run.output.observations[-1]
    opportunity = chosen["host"]["opportunity_id"]
    directory = out / "rerun"
    directory.mkdir()
    status, _ = run_native(tool, pair_run.directory / "plan.json", directory / "native", directory / "native.log",
                           ("--only", opportunity))
    try:
        again = pl.open_output(directory / "native") if status == 0 else None
        problem = None if again else f"the native tool exited with status {status}: {last_line(directory / 'native.log')}"
    except PilotLsPlanError as error:
        again, problem = None, f"the native output does not verify: {error}"
    if again is None:
        grader.check(name, "one opportunity", False, problem)
        return {}
    record = again.observations[0] if len(again.observations) == 1 else None
    earlier = [item for item in pair_run.output.records if item["opportunity_id"] == opportunity]
    found = pl.radio_differences([dict(item, opportunity_sequence=item["opportunity_sequence"]) for item in again.records],
                                 earlier)
    # A repeated opportunity is filled into the first grid: its index and its generation are those of a new run.
    found = [item for item in found if "ingress.grid_index" not in item and "ingress.generation" not in item]
    same = record is not None and record["payload"]["digest"] == chosen["payload"]["digest"] \
        and record["validity"]["sha256"] == chosen["validity"]["sha256"] \
        and record["host"]["ingress"]["readback_sha256"] == chosen["host"]["ingress"]["readback_sha256"] \
        and record["radio"]["absolute_slot"] == chosen["radio"]["absolute_slot"] \
        and record["time"]["t_first_sample"] == chosen["time"]["t_first_sample"]
    grader.check(name, "one opportunity", same and not found and len(again.records) == len(earlier),
                 f"{opportunity} alone, from the archived plan with --only: the same grid "
                 f"({chosen['host']['ingress']['readback_sha256'][:16]}), the same estimate "
                 f"({chosen['payload']['digest'][:16]}), the same slot count and sample times, and the same outcome and "
                 f"bytes on {len(again.records)} backends" if same and not found else
                 f"the repeated opportunity differs from the full run: {found[0] if found else 'the observation'}")
    return {"opportunity_id": opportunity, "same": same, "differences": len(found)}


# ---------------------------------------------------------------- refusals
def run_refusals(config, out, grader, tool, pair_run):
    name = "refusals"
    case_dir = out / "refusals"
    case_dir.mkdir()
    if pair_run is None or pair_run.plan is None:
        grader.check(name, "plan rules", False, "the plan of the reference pair was not built")
        return {}
    plan = pair_run.plan
    first, second = plan["sources"][0], plan["sources"][1]

    def tapped_replay(item, directory):
        replay_plan = json.loads(Path(plan["replay_plan"]["path"]).read_text())
        replay_plan["receiver"]["tap"] = "pilot_ls"
        (directory / "replay-plan.json").write_text(json.dumps(replay_plan, indent=2, sort_keys=True) + "\n")
        item["replay_plan"] = replay.file_ref(directory / "replay-plan.json")

    def zero_pilot(item, directory):
        _, shape, flat = grants.read_npy(first["pilot_ref"]["path"])
        flat[3] = 0j
        (directory / "pilot_ref.npy").write_bytes(grants._npy_bytes("<c8", shape, flat))
        record = json.loads(Path(first["source"]["path"]).read_text())
        record["reference"]["values"]["sha256"] = replay.sha256_file(directory / "pilot_ref.npy")
        (directory / "source.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        item["sources"][0].update(source=replay.file_ref(directory / "source.json"),
                                  pilot_ref=replay.file_ref(directory / "pilot_ref.npy"))

    def without_tap(item, _):
        item["observer"].update(tap="none", stage=None, source=None, layout=None)

    # Each rule with the words its refusal must carry: in the plan check, then in the native tool.
    rules = [
        ("a tap that does not exist", lambda item, _: item["observer"].update(tap="o3"), "unknown tap", "unknown tap"),
        ("a stage the tap does not export", lambda item, _: item["observer"].update(stage="O3"),
         "exports the stage O2", "exports the stage O2"),
        ("an epoch of zero", lambda item, _: item["observer"].update(epoch=0), "epoch is at least 1", "epoch is at least 1"),
        ("a replay plan with another hash", lambda item, _: item["replay_plan"].update(sha256="0" * 64),
         "replay plan does not have the hash", "replay plan does not have the hash"),
        ("a replay plan that names a tap itself", tapped_replay, "replay plan is refused", "replay plan is refused"),
        ("a grant without a sensing source", lambda item, _: item["sources"].pop(1),
         "binds no sensing source", "binds no sensing source"),
        ("a sensing source listed twice", lambda item, _: item["sources"].append(copy.deepcopy(first)),
         "listed twice", "listed twice"),
        ("a source bound to another grant record", lambda item, _: item["sources"][0].update(record_sha256="0" * 64),
         "changed allocation", "changed allocation"),
        ("a source record with another hash", lambda item, _: item["sources"][0]["source"].update(sha256="0" * 64),
         "record binds", "record binds"),
        ("the reference symbols of another grant", lambda item, _: item["sources"][0].update(pilot_ref=second["pilot_ref"]),
         "names other arrays than the plan binds", "names other arrays than the plan binds"),
        ("a configured pilot with zero reference power", zero_pilot, "zero reference power", "zero reference power"),
        ("an antenna map for another number of receive planes",
         lambda item, _: item["sources"][0].update(antenna_map=[0]), "receive planes", "receive planes"),
        ("an antenna map that names an element twice",
         lambda item, _: item["sources"][0].update(antenna_map=[0] * len(first["antenna_map"])),
         "names an array element twice", "names an array element twice"),
        ("a run without a tap that binds a source", without_tap, "without a tap", "without a tap"),
    ]
    measured, refused, accepted = {}, [], []
    for number, (what, change, checker_words, native_words) in enumerate(rules):
        directory = case_dir / f"plan-{number}"
        directory.mkdir()
        mutated = copy.deepcopy(plan)
        mutated["run_id"] = f"s31-refusal-{number}"
        change(mutated, directory)
        try:
            pl.check_files(mutated)
            checker = None
        except (PilotLsPlanError, PilotLsError) as error:
            checker = str(error)
        # The plan is written as it is, checks bypassed, so that the native tool meets it.
        (directory / "plan.json").write_text(json.dumps(mutated, indent=2, sort_keys=True) + "\n")
        status, _ = run_native(tool, directory / "plan.json", directory / "native", directory / "native.log")
        said = last_line(directory / "native.log")
        untouched = not any((directory / "native").iterdir())
        measured[what] = {"plan_check": checker, "native_status": status, "native_message": said,
                          "no_file_written": untouched}
        named = bool(checker) and checker_words in checker
        (refused if named and native_words in said and status == 2 and untouched else accepted).append(what)
    grader.check(name, "plan rules", not accepted and len(refused) == len(rules),
                 "refused by the plan check and by the native tool, each naming the rule, with status 2 before a "
                 "processor is created or a file is written: " + ", ".join(refused)
                 + ("" if not accepted else "; NOT REFUSED AS REQUIRED: " + ", ".join(accepted)))
    return measured


# ---------------------------------------------------------------- backends
def run_backends(config, grader, runs, rehearsal):
    name, backends = "backends", config["backends"]
    finished = [run for run in runs if run.ok()]
    records = [record for run in finished for record in run.output.records]
    measured = {"runs": len(runs), "runs_finished": len(finished), "records": len(records)}
    if not finished:
        grader.check(name, "executed path", False, "no native run finished")
        grader.check(name, "accounting", False, "no native run finished")
        return measured
    built = all(run.report["built_with_cuda"] is True for run in finished)
    available = all(backend["acceleration_available"] is True for run in finished for backend in run.report["backends"]
                    if backend["name"] == "cuda")
    paths = {backend: {} for backend in backends}
    wrong = []
    for record in records:
        executed = record["path"]["executed"] if isinstance(record["path"], dict) else None
        paths[record["backend"]][executed] = paths[record["backend"]].get(executed, 0) + 1
        ports = len(record["allocation"]["rx_ports"])
        wanted = "resident" if record["backend"] == "cuda" and ports in rp.RESIDENT_PORTS else "host"
        if record["outcome"] in rp.COMPLETED and executed != wanted:
            wrong.append(record)
    fallbacks = sorted({record["path"]["fallback"] for record in wrong if record["path"].get("fallback")})
    measured["paths"], measured["built_with_cuda"], measured["acceleration_available"] = paths, built, available
    shown = "; ".join(f"{backend}: " + ", ".join(f"{count} on the {path} path" for path, count in sorted(found.items(), key=str))
                      for backend, found in paths.items())
    if rehearsal:
        host_only = not built and all(set(found) <= {"host"} for found in paths.values())
        grader.check(name, "executed path", host_only,
                     f"REHEARSAL on a build without CUDA, not a hardware result: {shown}; every decode requested from "
                     f"the CUDA processor fell back to the host path and says so ({'; '.join(fallbacks) or 'no reason'}). "
                     "The CUDA decodes of this gate are open." if host_only else
                     f"--host-rehearsal was given on a build with CUDA: {shown}. Run the gate without it.")
    else:
        grader.check(name, "executed path", built and available and not wrong,
                     f"{shown}; the adapter is built with CUDA, the device is usable, and with the observer attached "
                     "every decode requested from the CUDA processor ran its resident path and every decode requested "
                     "from the host processor the host path" if built and available and not wrong else
                     f"{shown}; built with CUDA: {built}, device usable: {available}; {len(wrong)} decodes did not "
                     f"execute the requested path ({'; '.join(fallbacks) or 'no reason recorded'}). A fallback cannot "
                     "close this gate.")
    planned = sum(run.report["totals"]["planned"] for run in finished)
    outcomes = {outcome: sum(record["outcome"] == outcome for record in records) for outcome in rp.OUTCOMES}
    anomalies = sum(len(run.report["anomalies"]) for run in finished)
    held = sum(run.report["totals"]["grids_not_released"] for run in finished)
    observations = sum(len(run.output.observations) for run in finished)
    grids = sum(len(run.report["opportunities"]) for run in finished if run.output.tap == "pilot_ls")
    measured.update(outcomes=outcomes, planned=planned, observations=observations, tapped_grids=grids)
    grader.check(name, "accounting", len(finished) == len(runs) and planned == len(records)
                 and sum(outcomes.values()) == planned and anomalies == 0 and held == 0 and observations == grids
                 and outcomes["timeout"] == outcomes["not_dispatched"] == outcomes["decoder_error"] == 0
                 and outcomes["payload_mismatch"] == 0,
                 f"{planned} decodes planned in {len(runs)} runs: {len(records)} records, each with one terminal "
                 "outcome: " + ", ".join(f"{count} {outcome}" for outcome, count in outcomes.items())
                 + f"; {observations} observations for {grids} tapped grids; {anomalies} anomalies, {held} grids held "
                   "after the run")
    for backend in backends:
        for label, source in (("with the tap", [record for run in finished if run.output.tap == "pilot_ls"
                                               for record in run.output.records]),
                              ("in the control", [record for run in finished if run.output.tap == "pilot_ls"
                                                  for record in run.control.output.records])):
            times = sorted(record["timing"]["service_us"] for record in source if record["backend"] == backend
                           and record["timing"]["service_us"] is not None and not record["timing"]["first_use"]
                           and len(record["allocation"]["rx_ports"]) == 4 and record["allocation"]["prb_count"] == 51)
            if times:
                key = f"service_us_{backend}_{'tap' if label == 'with the tap' else 'control'}"
                measured[key] = {"count": len(times), "mean": sum(times) / len(times), "p50": times[len(times) // 2],
                                 "max": times[-1]}
                grader.note(name, f"PUSCH service time of {backend} {label}, not graded: mean "
                                  f"{sum(times) / len(times):.0f} us, median {times[len(times) // 2]:.0f} us, largest "
                                  f"{times[-1]:.0f} us over {len(times)} decodes of 51 blocks on four receive ports; the "
                                  "tap runs before the decode and is not inside this time; runs at different times on "
                                  "a host that is not tuned")
    return measured


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True,
                              timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="new directory for the run")
    parser.add_argument("--s24-run", required=True, type=Path,
                        help="run directory of the accepted Sprint 2.4 gate: it holds the plans and the control records")
    parser.add_argument("--adapter-build", type=Path, help="adapter build directory (default: from the configuration)")
    parser.add_argument("--host-rehearsal", action="store_true",
                        help="development host without a CUDA build: grade the host path on a rehearsed Sprint 2.4 run "
                             "and leave the hardware gate open")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.exit(2, f"pilot_ls: {args.output_dir} already exists. Evidence is never overwritten.\n")
    try:
        config = load_config(args.config)
        receipt = native_adapter.verified_receipt(args.adapter_build or ROOT / config["adapter_build"])
        for needed in ("pilot_ls", "replay_pusch"):
            if needed not in receipt["executables"]:
                raise ConfigError(f"the adapter build holds no {needed}. Rebuild it: python scripts/native_adapter.py build")
        import numpy as np
    except (ConfigError, native_adapter.AdapterError, ImportError) as error:
        parser.exit(2, f"pilot_ls: {error}\n")
    tool = Path(receipt["executables"]["pilot_ls"]["path"])
    s24_run = args.s24_run.resolve()
    s24_out = s24_run / config["control"]["driver"] if (s24_run / config["control"]["driver"] / "grade.json").is_file() \
        else s24_run
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = args.output_dir.resolve()
    out.mkdir(parents=True)
    write_json(out / "adapter-receipt.json", receipt)

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

    controls, measured["control"] = guarded("control", lambda: run_control(config, grader, s24_out, args.host_rehearsal),
                                            ({}, {}))
    if controls and all(controls.get(role) for role in CONTROL_RUNS):
        pair_control = controls["pair"][0]
        measured["unity"] = guarded("unity", lambda: run_unity(config, out, grader, np, tool, runs, controls["unity"]))
        measured["direct"] = guarded("direct", lambda: run_direct(config, out, grader, np, tool, runs,
                                                                  controls["direct"][0]))
        measured["reference-pair"], pair_run = guarded(
            "reference-pair", lambda: run_pair(config, out, grader, np, tool, runs, pair_control), ({}, None))
        measured["payload"] = guarded("payload", lambda: run_payload(config, out, grader, tool, runs,
                                                                     controls["payload"][0], pair_run))
        measured["tap-off"] = guarded("tap-off", lambda: run_tap_off(config, out, grader, tool, runs, pair_control,
                                                                     pair_run))
        measured["rerun"] = guarded("rerun", lambda: run_rerun(config, out, grader, tool, pair_run))
        measured["refusals"] = guarded("refusals", lambda: run_refusals(config, out, grader, tool, pair_run))
        measured["backends"] = guarded("backends", lambda: run_backends(config, grader, runs, args.host_rehearsal))
    else:
        for case in CASES[1:]:
            grader.check(case, "control", False, "the Sprint 2.4 run does not hold the control this case is compared with")

    failed = [item for item in grader.results if not item["passed"]]
    passed_cases = [case for case in CASES if any(item["case"] == case for item in grader.results)
                    and all(item["passed"] for item in grader.results if item["case"] == case)]
    complete = not failed and len(passed_cases) == len(CASES)
    grade = {
        "schema": GRADE_SCHEMA, "selector": "pilot_ls", "run_id": out.name, "cases": len(CASES),
        "cases_passed": len(passed_cases), "cases_failed": len(CASES) - len(passed_cases),
        "checks": len(grader.results), "checks_passed": len(grader.results) - len(failed), "checks_failed": len(failed),
        "judgment": ("rehearsal" if args.host_rehearsal else "pass") if complete else "fail",
        "hardware_gate": "open: the CUDA processor did not execute, this is a host rehearsal" if args.host_rehearsal
                         else ("closed" if complete else "open"),
        "reason": "every check of every case passed" if not failed else
                  f"{len(failed)} check(s) failed, first: {failed[0]['case']} {failed[0]['check']}",
        "backends": config["backends"], "observer": config["observer"], "tolerance": config["tolerance"],
        "estimator_precision": reference.PRECISION, "results": grader.results}
    write_json(out / "grade.json", grade)
    write_json(out / "comparison.json", measured)
    lock = {item["name"]: item["commit"] for item in json.loads((ROOT / "ext/lock.json").read_text())["dependencies"]}
    write_json(out / "manifest.json", {
        "run_id": out.name, "selector": "pilot_ls",
        "purpose": "Sprint 3.1 integration: the plans of the accepted Sprint 2.4 run decoded again with an observer "
                   "that exports the pilot least squares of the grant's sensing source from each ready grid (O2), "
                   "compared with the radio records of that run",
        "command": shlex.join([sys.executable, *sys.argv]), "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": platform.node(), "machine": platform.machine(), "python": platform.python_version(),
        "numpy": np.__version__, "parent_commit": git_commit(), "locked_sources": lock,
        "inputs": {str(args.config): digest(args.config)}, "driver_sha256": digest(__file__),
        "reference_sha256": digest(ROOT / "src/ocudu_cuda_isac/sensing/pilot_ls.py"),
        "reader_sha256": digest(ROOT / "src/ocudu_cuda_isac/episodes/pilot_ls.py"),
        "s24_run": str(s24_run), "control": measured["control"].get("s24"),
        "axes": {"split": "staged_harness", "receiver_stage": "o2_pilot_ls", "ue_realization": "replayed channel, no UE",
                 "host_stack": "DeepSig-OCUDU", "grid_memory": "host resource grid", "sensing_source": config["observer"]["source"],
                 "observer_memory": "host buffer owned by the observer", "placement": "in the receiver process, before the decode"},
        "runs": [{"name": run.name, "directory": str(run.directory.relative_to(out)), "status": run.status,
                  "plan_sha256": digest(run.directory / "plan.json") if (run.directory / "plan.json").is_file() else None,
                  "tap": run.plan["observer"]["tap"] if run.plan else None,
                  "control": run.control.name, "control_plan_sha256": run.control.plan_sha256} for run in runs],
        "adapter": {"executable": str(tool), "sha256": receipt["executables"]["pilot_ls"]["sha256"],
                    "replay_pusch_sha256": receipt["executables"]["replay_pusch"]["sha256"],
                    "receipt": "adapter-receipt.json", "ocudu_build": receipt["ocudu_build"],
                    "built_with_cuda": receipt["pin"]["cuda"]},
        "host_rehearsal": args.host_rehearsal,
        "environment": {key: os.environ[key] for key in ("VIRTUAL_ENV", "CUDA_VISIBLE_DEVICES",
                                                         "OCUDU_PUSCH_ACCELERATION_TIMING") if key in os.environ},
        "judgment": grade["judgment"], "reason": grade["reason"]})
    summary = (f"pilot_ls: {grade['cases_passed']} of {grade['cases']} cases passed, "
               f"{grade['checks_passed']} of {grade['checks']} checks"
               + (", host rehearsal: the hardware gate stays open" if args.host_rehearsal else ""))
    (out / "grade.txt").write_text("\n".join(grader.lines + [summary]) + "\n", encoding="ascii")
    print(summary)
    print(f"Run artifacts: {out}")
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
