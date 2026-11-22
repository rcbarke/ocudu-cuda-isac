#!/usr/bin/env python3
"""Sprint 3.6 integration driver (selector sense_core). Hardware command, not part of unit discovery.

It runs the native sensing core beside the reference of Sprint 3.5 on the held-out records of
the accepted Sprint 3.5 run, with the artifact that run froze. No receiver runs here and no
launcher: the records are those the reference sensor kept, replayed from their archive and, for
the live cases, published again on an observation ring at the pace of their slots. Eight cases,
each graded on its own:

  core       the build of the core verifies against its receipt and the tree: the device, the
             CUDA architecture, TensorRT, the unit cases with the kernels on the device, link
             lines without a receiver, and a binding that names the same sources
  control    the accepted Sprint 3.5 run: its frozen artifact with the hash of its freeze record,
             and the held-out outputs of its sensor with the records it kept
  engine     the ONNX export of the artifact, byte for byte again, and its TensorRT engine built
             on this GPU; an engine of another GPU, of another TensorRT version and of another
             artifact is refused by the sensor
  units      held-out records through the binding: the cube of the ingest unit against the
             reference reader bit for bit, and the window mean, its counts and the input of the
             model against the reference within the declared tolerance, with no device call
             while a window is open and no allocation after the plan was made
  parity     every held-out archive through sense_host and through the reference: the same
             windows and statuses, the inputs within their bound, the scores and the maps within
             the declared tolerance, the decisions at the frozen threshold, every flip with its
             margin: runtime_parity.json
  live       the sensor alone on the device: sense_feed publishes the held-out records on a
             ring at the pace of their slots, once for sense_host and once for the reference
             sensor. Both outputs verify, nothing is dropped, and the three latency definitions
             are recorded for both runtimes with their tails: runtime_latency.json
  overload   a producer that is faster than the sensor: records without a credit are dropped and
             counted at the ring, the producer never waits, windows that were overtaken are
             skipped and counted, a stage over its budget is counted and its decision published
             as late, and every credit is back when the producer ends
  reference  the native latency beside the published 241 us of the cuSense system, by its
             boundary, not graded, with every difference stated: cusense_reference.json

    python tests/integration/sense_core.py --config configs/integration/sense_core.json \\
        --s35-run results/runs/s35-spark-001 --output-dir results/runs/s36-spark-001/sense_core

The output directory must be new. The exit status is 0 only when every check of every case
passed. With --rehearsal the driver runs on the host twin with the engine host_reference, on a
development host without a GPU: it grades the same checks but those of the device, and its
hardware gate stays open. The timed arms with a receiver beside the sensor are Sprint 4.1's.
"""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import native_core  # noqa: E402

CONFIG_SCHEMA = "integration-config/sense_core/v1"
GRADE_SCHEMA = "integration-grade/sense_core/v1"
CASES = ("core", "control", "engine", "units", "parity", "live", "overload", "reference")
PARITY_FILE, LATENCY_FILE, REFERENCE_FILE = "runtime_parity.json", "runtime_latency.json", "cusense_reference.json"
CONFIG_FIELDS = {"schema", "selector", "core_build", "control", "engine", "units", "live", "overload", "expect"}
SENSOR_TIMEOUT_S = 1800


class ConfigError(ValueError):
    """The configuration or the run setup is unusable. Nothing was graded."""


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii")


class Grade:
    """Collects PASS, FAIL and NOTE lines. A case passes only when it has no FAIL line."""

    def __init__(self):
        self.lines, self.results = [], []

    def check(self, case, name, passed, detail):
        self.results.append({"case": case, "check": name, "passed": bool(passed), "detail": detail})
        self._line(("PASS" if passed else "FAIL") + f"  S3.6 {case} {name}: {detail}")
        return bool(passed)

    def note(self, case, text):
        self._line(f"NOTE  S3.6 {case} {text}")

    def _line(self, line):
        self.lines.append(line)
        print(line, flush=True)


def load_config(path):
    try:
        config = json.loads(Path(path).read_text(encoding="ascii"))
    except (OSError, ValueError) as error:
        raise ConfigError(f"cannot read the configuration: {error}") from None
    if not isinstance(config, dict) or set(config) != CONFIG_FIELDS or config["schema"] != CONFIG_SCHEMA \
            or config["selector"] != "sense_core":
        raise ConfigError(f"the configuration must be a {CONFIG_SCHEMA} object with exactly: "
                          + ", ".join(sorted(CONFIG_FIELDS)))
    try:
        live, overload = config["live"], config["overload"]
        numbers = [config["units"]["windows"], live["repeat"], live["ring_slots"], live["slot_us"], live["min_windows"],
                   live["ready_timeout_ms"], overload["ring_slots"], overload["period_us"], overload["hold_us"],
                   overload["repeat"]] + list(live["budget_us"]) + list(overload["budget_us"])
        if any(type(value) is not int or value < 0 for value in numbers) or len(live["budget_us"]) != 4 \
                or len(overload["budget_us"]) != 4 or config["engine"]["precision"] not in ("fp32", "fp16") \
                or config["control"]["driver"] != "cusense_frontend" or live["wait"] not in ("spin", "wait") \
                or type(config["expect"]["published_live_inference_us"]) is not int:
            raise ConfigError("a count of the configuration is no integer, a budget does not name four stages, or a "
                              "name is not one of its choices")
    except (KeyError, TypeError) as error:
        raise ConfigError(f"the configuration is incomplete: {error!r}") from None
    return config


def run_logged(args, log, timeout=SENSOR_TIMEOUT_S, env=None):
    """One command with its output in a log. Returns (status, last line that is JSON or text)."""
    with open(log, "wb") as stream:
        stream.write((" ".join(str(arg) for arg in args) + "\n").encode())
        stream.flush()
        try:
            done = subprocess.run([str(arg) for arg in args], stdout=stream, stderr=subprocess.STDOUT, timeout=timeout,
                                  env=env)
            status = done.returncode
        except subprocess.TimeoutExpired:
            status = 124
    text = Path(log).read_text(errors="replace").strip().splitlines()
    return status, (text[-1] if len(text) > 1 else "")


def host_state():
    """What the host was when the latencies were taken. Recorded with them."""
    state = {"host": platform.node(), "machine": platform.machine(), "kernel": platform.release(),
             "cpus": os.cpu_count(), "loadavg": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
             "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    status, text = native_core.run(["nvidia-smi", "--query-gpu=name,driver_version,utilization.gpu,memory.used,"
                                    "temperature.gpu,power.draw", "--format=csv,noheader"], 60)
    state["gpu"] = text.strip().splitlines()[0] if status == 0 and text.strip() else None
    status, text = native_core.run(["nvidia-smi", "--query-compute-apps=pid,name,used_memory", "--format=csv,noheader"], 60)
    state["gpu_processes"] = text.strip().splitlines() if status == 0 else None
    governor = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    state["cpu_governor"] = governor.read_text().strip() if governor.is_file() else None
    return state


class Setup:
    """What every case shares: the configuration, the build, the artifact and the held-out captures."""

    def __init__(self, config, out, receipt, rehearsal, s35_run):
        self.config, self.out, self.receipt, self.rehearsal, self.s35_run = config, out, receipt, rehearsal, s35_run
        self.tools = {name: Path(item["path"]) for name, item in receipt["executables"].items()}
        self.device = "host" if rehearsal else "cuda"
        self.engine = "host_reference" if rehearsal else "tensorrt"
        self.engine_dir = None
        self.artifact_dir = self.artifact = self.freeze = None
        self.captures = []  # (name, archive directory, verified reference output of Sprint 3.5)
        self.reference = {}  # name -> verified reference output on the CPU, computed here
        self.native = {}     # name -> verified native output of the archive replay

    def sensor(self, extra, output_dir, evidence, log):
        """One run of sense_host on the device and engine of this gate."""
        command = [self.tools["sense_host"], "--artifact", self.artifact_dir, "--output-dir", output_dir, "--device",
                   self.device, "--engine", self.engine, "--evidence", evidence, "--expect-sha256", self.artifact.sha256]
        if self.engine_dir is not None:
            command += ["--engine-dir", self.engine_dir]
        return run_logged(command + list(extra), log)


# ---------------------------------------------------------------- cases
def run_core(setup, grader, runtime):
    receipt, config = setup.receipt, setup.config
    files = len(receipt["sources"])
    grader.check("core", "receipt", True,
                 f"the build {receipt['build_dir']} verifies against its receipt and the tree: {files} source files, "
                 f"sources {receipt['sources_sha256'][:16]}, library {receipt['library']['sha256'][:12]}, sense_host "
                 f"{receipt['executables']['sense_host']['sha256'][:12]}, sense_feed "
                 f"{receipt['executables']['sense_feed']['sha256'][:12]} (isac-core.json)")
    if setup.rehearsal:
        grader.note("core", f"rehearsal: the build is for the device {receipt['device']}; the kernels are not on a GPU")
    else:
        wanted = config["expect"]["cuda_architecture"]
        grader.check("core", "device", receipt["device"] == "cuda" and str(wanted) in str(receipt["cuda_architectures"]),
                     f"the core is built for the device {receipt['device']} with the CUDA architecture "
                     f"{receipt['cuda_architectures']} (expected {wanted}) by {receipt['toolchain']['cuda_compiler']}")
        tensorrt = receipt["tensorrt"]
        grader.check("core", "tensorrt", bool(tensorrt) and "sense_engine" in receipt["executables"],
                     "no TensorRT backend in this build: run the probe and rebuild" if not tensorrt else
                     f"the TensorRT backend is built against {tensorrt['version']} from {tensorrt['origin']}, with "
                     f"sense_engine {receipt['executables'].get('sense_engine', {}).get('sha256', 'missing')[:12]}")
        grader.check("core", "kernels", receipt["tests"]["device_vectors"] == "ran",
                     f"the unit case of the kernels on the device {receipt['tests']['device_vectors']}")
    tests = receipt["tests"]
    grader.check("core", "unit-cases", tests["ran"] >= 40 and tests["failed"] == 0,
                 f"{tests['ran']} unit cases ran in the build, {tests['failed']} failed, skipped: "
                 f"{', '.join(tests['skipped']) or 'none'}; {receipt['compiler_warnings']} compiler warnings")
    found = [name for name in ("sense_host", "sense_core") if native_core.forbidden_in(receipt["link"].get(name))]
    grader.check("core", "link", not found and receipt["link"]["sense_host"],
                 "the link line of the library and of sense_host names no receiver archive, no Aerial library and no "
                 "Python library" if not found else f"the link line of {found[0]} names a library the core must not link")
    core = runtime.load(receipt["build_dir"])
    said = runtime.build(core)
    grader.check("core", "binding", said["sources_sha256"] == receipt["sources_sha256"],
                 f"the binding of {receipt['binding']['path'] if receipt['binding'] else 'no file'} names the sources "
                 f"{said['sources_sha256'][:16]} of the receipt and the API version {said['api_version']}")
    return core, {"sources_sha256": receipt["sources_sha256"], "device": receipt["device"],
                  "tensorrt": receipt["tensorrt"], "tests": tests}


def run_control(setup, grader, cusense, obs):
    run, config = setup.s35_run, setup.config
    out = run / config["control"]["driver"]
    summary = (run / "summary.txt").read_text(errors="replace") if (run / "summary.txt").is_file() else ""
    accepted = "Sprint 3.5: ACCEPTED on this DGX Spark" in summary and "\nFAIL  " not in summary
    if setup.rehearsal:
        grader.note("control", "rehearsal: the control is a set of records and an artifact of this host, not an accepted "
                               "Sprint 3.5 run")
    else:
        grader.check("control", "accepted", accepted,
                     f"the Sprint 3.5 run {run.name} " + ("is ACCEPTED on this DGX Spark" if accepted else
                                                         "holds no verdict ACCEPTED on this DGX Spark without a failed line"))
    recover = json.loads((out / "recover.json").read_text())
    freeze = json.loads((out / recover["freeze"]).read_text())
    setup.artifact_dir, setup.freeze = out / recover["artifact"], freeze
    setup.artifact = cusense.open_artifact(setup.artifact_dir, freeze["artifact"]["sha256"])
    manifest = setup.artifact.manifest
    grader.check("control", "artifact", True,
                 f"the artifact {manifest['model_id']} {setup.artifact.sha256[:16]} is the one the freeze record of "
                 f"{freeze.get('frozen_utc')} names: input {manifest['entries']['input']} of shape "
                 f"{manifest['input']['shape']}, a grid of {manifest['grid']['rows']} by {manifest['grid']['cols']} "
                 f"cells, threshold logit {manifest['threshold']['logit']:.6f}")
    live = sorted(path.parent for path in (out / "live").glob("*/sensor/cusense.json"))
    windows = 0
    for sensor in live:
        output = cusense.open_output(sensor)
        archive = obs.open_archive(sensor / "windows" / "records")
        if output.manifest["sha256"] != setup.artifact.sha256:
            raise ConfigError(f"{sensor} was scored with another artifact")
        setup.captures.append((sensor.parent.name, sensor / "windows" / "records", output, archive))
        windows += len(output.scored)
    wanted = config["control"]["min_windows"]
    grader.check("control", "held-out", len(setup.captures) >= 1 and windows >= wanted,
                 f"{len(setup.captures)} held-out output(s) of the reference sensor verify, with the records the sensor "
                 f"kept: {sum(len(item[3].records) for item in setup.captures)} records and {windows} scored windows "
                 f"(at least {wanted})")
    return {"run": run.name, "artifact_sha256": setup.artifact.sha256, "captures": [item[0] for item in setup.captures],
            "windows": windows}


def run_engine(setup, grader, export):
    out, config = setup.out / "engine", setup.config
    out.mkdir()
    (out / "onnx").mkdir()
    record = export.write_onnx(setup.artifact, out / "onnx")
    export.check_onnx(out / "onnx", setup.artifact)
    grader.check("engine", "export", True,
                 f"the ONNX export of the artifact is {record['bytes']} bytes, {record['sha256'][:16]}, at operator set "
                 f"{record['opset']}, written from weights.f32 alone, and a second export gives the same bytes "
                 "(engine/onnx/export.json)")
    measured = {"onnx_sha256": record["sha256"]}
    try:
        import numpy as np
        offline = setup.captures[0][2]
        x = offline.inputs[:4]
        worst, beyond = 0.0, 0
        for index in range(len(x)):
            drawn, logit = export.run_onnx(out / "onnx" / record["file"], x[index][None])
            wanted = offline.scored[index]["score"]["logit"]
            cells = float(np.abs(drawn.reshape(offline.maps[index].shape) - offline.maps[index]).max())
            score = abs(float(logit[0]) - wanted)
            # Between the device of the Sprint 3.5 sensor and a CPU: the tolerance of that sprint.
            beyond += score > 1e-3 + 1e-4 * abs(wanted) or cells > 1e-4
            worst = max(worst, score, cells)
        grader.check("engine", "export-scores", beyond == 0,
                     f"the graph in ONNX Runtime on the CPU gives the scores that the Sprint 3.5 sensor wrote for "
                     f"{len(x)} windows on its device, within {worst:.2e}")
        measured["onnxruntime_difference"] = worst
    except export.ExportError as error:
        grader.note("engine", f"the graph was not run on the CPU: {error}")
    if setup.rehearsal:
        grader.note("engine", "rehearsal: no TensorRT engine is built; the engine is host_reference, the model in C++ "
                              "on the CPU from weights.f32")
        return measured
    (out / "engine").mkdir()
    command = [setup.tools["sense_engine"], "--onnx", out / "onnx" / record["file"], "--artifact", setup.artifact_dir,
               "--output-dir", out / "engine", "--precision", config["engine"]["precision"]]
    status, last = run_logged(command, out / "sense-engine.log")
    built = status == 0 and (out / "engine/engine.json").is_file()
    manifest = json.loads((out / "engine/engine.json").read_text()) if built else {}
    grader.check("engine", "built", built and manifest.get("artifact_sha256") == setup.artifact.sha256,
                 f"sense_engine ended with status {status}: {last[:200]} (engine/sense-engine.log)" if not built else
                 f"the engine was built on this GPU, {manifest['device']['name']} (capability "
                 f"{manifest['device']['capability']}), with TensorRT {manifest['library']['version']} as "
                 f"{manifest['precision']}: {manifest['bytes']} bytes, {manifest['sha256'][:16]}, {manifest['layers']} "
                 f"layers, in {manifest['build_ms']:.0f} ms; it names the artifact {manifest['artifact_sha256'][:16]}")
    if not built:
        return measured
    setup.engine_dir = out / "engine"
    measured.update(engine_sha256=manifest["sha256"], tensorrt=manifest["library"]["version"], device=manifest["device"])
    # An engine is refused anywhere but where it was built, and for any other artifact.
    archive = setup.captures[0][1]
    for name, change, word in (
            ("other-gpu", lambda m: m["device"].update(name="NVIDIA GH200", capability="9.0", uuid="0" * 32), "GPU"),
            ("other-version", lambda m: m["library"].update(version="10.0.0.0"), "rebuild"),
            ("other-artifact", lambda m: m.update(artifact_sha256="0" * 64), "artifact")):
        where = out / name
        shutil.copytree(out / "engine", where)
        altered = json.loads((where / "engine.json").read_text())
        change(altered)
        write_json(where / "engine.json", altered)
        (out / f"{name}-output").mkdir()
        command = [setup.tools["sense_host"], "--archive", archive, "--artifact", setup.artifact_dir, "--output-dir",
                   out / f"{name}-output", "--device", "cuda", "--engine", "tensorrt", "--engine-dir", where]
        status, last = run_logged(command, out / f"{name}.log", 600)
        grader.check("engine", f"refused-{name}", status != 0 and word in last and not any((out / f"{name}-output").iterdir()),
                     f"an engine whose manifest names {name.replace('-', ' ')} is refused before a record is read, "
                     f"status {status}: {last[:160]}")
    return measured


def run_units(setup, grader, core, runtime, cusense, evaluator, np):
    config, artifact = setup.config, setup.artifact
    name, archive_dir, _, archive = setup.captures[0]
    spec, tolerance = artifact.spec, runtime.tolerance(core)["fp32"]
    plan = core.Plan(str(setup.artifact_dir), device=setup.device, engine="host_reference", keep_mean=True)
    plan.warm_up(2)
    configured = plan.counters()
    stream, model = cusense.WindowStream(spec), cusense.Model(artifact)
    wanted, cubes, worst = config["units"]["windows"], 0, {"mean": 0.0, "input_of_bound": 0.0}
    same_cube = same_mean = same_count = windows = 0
    for record in archive.records:
        data = record.to_bytes()
        if cusense.refusal(record, spec) is None and cubes < 3 * wanted:
            values, valid = cusense.planes(record, spec)
            cube, bits, _ = plan.ingest(data)
            cubes += 1
            # A value that is no sample is zero in the cube of the core, whatever its payload holds.
            same_cube += bool(np.array_equal(bits.astype(bool), valid)
                              and np.array_equal(cube[valid], np.asarray(values).astype("<c8")[valid])
                              and not cube[~valid].any())
        lines = stream.push(record)
        plan.offer(data)
        if plan.ready():
            decision = plan.run()
            if lines and lines[-1]["status"] == "ok" and decision["status"] == "ok":
                mean, count = stream.last
                windows += 1
                same_count += bool(np.array_equal(decision["count"], count))
                same_mean += bool(np.array_equal(decision["mean"], mean))
                # In spacings of FP32 at the magnitude of each element.
                parts = np.abs(decision["mean"].astype(np.complex128) - mean.astype(np.complex128))
                scale = np.maximum(np.abs(mean.astype(np.complex128)), 1e-300) * tolerance["spacing"]
                worst["mean"] = max(worst["mean"], float((parts / scale).max()))
                x = model.prepare(mean, count)[0]
                bound = evaluator.input_bound(x, artifact.template["values"], artifact.template["count"],
                                              artifact.manifest["normalizer"], tolerance)
                ratio = np.abs(decision["input"].astype(np.float64) - x.astype(np.float64)) / np.maximum(bound, 1e-300)
                worst["input_of_bound"] = max(worst["input_of_bound"], float(ratio.max()))
        if windows >= wanted:
            break
    counters = plan.counters()
    where = "the kernels on the device" if setup.device == "cuda" else "the host twin of the kernels"
    grader.check("units", "ingest", cubes >= 3 and same_cube == cubes,
                 f"{same_cube} of {cubes} held-out records of {name} through the ingest unit on {where} give the cube of "
                 "the reference reader bit for bit, values and validity")
    grader.check("units", "window", windows >= 1 and same_count == windows and worst["mean"] <= 2 * tolerance["mean_ulps"],
                 f"{windows} windows: the counts equal those of the reference in {same_count}, the window mean is "
                 f"bit-identical to the complex64 mean of the reference in {same_mean} and differs by at most "
                 f"{worst['mean']:.2f} spacings of FP32 at the magnitude of an element (declared: "
                 f"{tolerance['mean_ulps']:.0f} on each part)")
    grader.check("units", "input", windows >= 1 and worst["input_of_bound"] <= 1.0,
                 f"the input of the model differs from the reference by at most {worst['input_of_bound']:.3f} of its "
                 f"declared bound ({tolerance['input_ulps']:.0f} spacings of FP32 at the magnitude of the element, "
                 "over the divisor of the z-score)")
    grader.check("units", "bounded", counters["device_calls_open"] == 0
                 and counters["device_allocations"] == configured["device_allocations"] and counters["views_held"] == 0,
                 f"{counters['submissions']} submissions made {counters['device_calls'] - configured['device_calls']} "
                 f"device calls, none while a window was open; {counters['device_allocations']} allocations, all when "
                 "the plan was configured; every view was released")
    return {"windows": windows, "cubes": cubes, "largest_mean_difference": worst["mean"],
            "largest_input_of_bound": worst["input_of_bound"], "mean_bit_identical": same_mean}


def run_parity(setup, grader, core, runtime, cusense, evaluator):
    out = setup.out / "parity"
    out.mkdir()
    tolerance = runtime.tolerance(core)["fp32"]
    parts, problems_total = {}, 0
    totals = {"joined": 0, "agree": 0, "flips": [], "moved": 0, "input": 0.0, "input_of_bound": 0.0, "logit": 0.0,
              "probability": 0.0, "map": 0.0, "bit_identical_inputs": 0}
    for name, archive_dir, offline, _ in setup.captures:
        (out / f"{name}-native").mkdir()
        status, last = setup.sensor(["--archive", archive_dir], out / f"{name}-native", "audit", out / f"{name}-native.log")
        if status != 0:
            grader.check("parity", f"{name}-sensor", False, f"sense_host ended with status {status}: {last[:200]}")
            problems_total += 1
            continue
        native = cusense.open_output(out / f"{name}-native")
        (out / f"{name}-reference").mkdir()
        cusense.replay(archive_dir, setup.artifact_dir, out / f"{name}-reference", "cpu")
        reference = cusense.open_output(out / f"{name}-reference")
        setup.native[name], setup.reference[name] = native, reference
        problems, facts = evaluator.parity(native, reference, setup.artifact, tolerance)
        parts[name] = dict(facts, problems=problems, native_output=str((out / f"{name}-native").relative_to(setup.out)),
                           reference_output=str((out / f"{name}-reference").relative_to(setup.out)))
        problems_total += len(problems)
        totals["joined"] += facts["joined"]
        totals["agree"] += facts["decisions"]["agree"]
        totals["flips"] += facts["flips"]
        totals["moved"] += len(facts["locations_moved"])
        totals["bit_identical_inputs"] += facts["inputs_bit_identical"]
        for key, field in (("input", "largest_input_difference"), ("input_of_bound", "largest_input_difference_of_its_bound"),
                           ("logit", "largest_logit_difference"), ("probability", "largest_probability_difference"),
                           ("map", "largest_map_difference")):
            totals[key] = max(totals[key], facts[field])
        counts = native.report["counts"]
        grader.check("parity", name, not problems,
                     f"{facts['joined']} windows of {name} by both runtimes on the same records: " +
                     (f"the inputs within {facts['largest_input_difference_of_its_bound']:.3f} of their bound, the logit "
                      f"within {facts['largest_logit_difference']:.2e}, the probability within "
                      f"{facts['largest_probability_difference']:.2e}, the map within {facts['largest_map_difference']:.2e}, "
                      f"{facts['decisions']['agree']} of {facts['decisions']['compared']} decisions agree at the frozen "
                      f"threshold, {len(facts['flips'])} flip(s), {len(facts['locations_moved'])} location(s) moved; "
                      f"{counts['submissions']} submissions, {counts['device_calls_in_an_open_window']} device calls in an "
                      "open window" if not problems else f"{len(problems)} problem(s), first: {problems[0]}"))
    native_runtime = next(iter(setup.native.values())).report["runtime"] if setup.native else {}
    record = {"schema": evaluator.PARITY_SCHEMA, "artifact_sha256": setup.artifact.sha256,
              "tolerance": runtime.tolerance(core), "native": {key: native_runtime.get(key) for key in (
                  "device", "device_name", "engine_backend", "engine_detail", "engine_sha256", "engine_graph",
                  "timing_valid")},
              "reference": {"device": "cpu", "runtime": cusense.RUNTIME}, "captures": parts,
              "totals": {"windows": totals["joined"], "decisions_agree": totals["agree"], "flips": totals["flips"],
                         "locations_moved": totals["moved"], "inputs_bit_identical": totals["bit_identical_inputs"],
                         "largest_input_difference": totals["input"],
                         "largest_input_difference_of_its_bound": totals["input_of_bound"],
                         "largest_logit_difference": totals["logit"],
                         "largest_probability_difference": totals["probability"],
                         "largest_map_difference": totals["map"], "problems": problems_total}}
    write_json(setup.out / PARITY_FILE, record)
    wanted = setup.config["control"]["min_windows"]
    grader.check("parity", "record", problems_total == 0 and totals["joined"] >= wanted,
                 f"{PARITY_FILE} holds {totals['joined']} windows (at least {wanted}) scored by the native core on "
                 f"{native_runtime.get('device')} with the {native_runtime.get('engine_backend')} backend and by the "
                 f"reference on the CPU: {totals['agree']} decisions agree, {len(totals['flips'])} flip(s), each listed "
                 f"with its margin, the largest logit difference is {totals['logit']:.2e} and the largest map difference "
                 f"{totals['map']:.2e}")
    return record["totals"]


def feed_and_sense(setup, name, archive_dir, sensor_command, ready, log, feed_extra, timeout_ms):
    """Start a sensor, wait until it is warm, publish an archive on its ring. Returns (sensor status, feed report)."""
    out = setup.out
    with open(log, "wb") as stream:
        sensor = subprocess.Popen([str(arg) for arg in sensor_command], stdout=stream, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + timeout_ms / 1000.0
        while not ready.exists() and sensor.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        if not ready.exists():
            sensor.kill()
            sensor.wait()
            return 1, {"problem": "the sensor was not warm in time"}
        try:  # the process inventory of the warm sensor: what it maps
            mapped = sorted({line.split()[-1] for line in Path(f"/proc/{sensor.pid}/maps").read_text().splitlines()
                             if "/" in line.split()[-1]})
        except (OSError, IndexError):
            mapped = None
        feed = subprocess.run([str(setup.tools["sense_feed"]), "--archive", str(archive_dir)] + [str(a) for a in feed_extra],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=SENSOR_TIMEOUT_S)
        (out / f"{name}-feed.log").write_text(feed.stdout + feed.stderr)
        try:
            status = sensor.wait(timeout=SENSOR_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            sensor.kill()
            status = 124
    try:
        report = json.loads(feed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        report = {"problem": (feed.stderr.strip().splitlines() or ["sense_feed printed no report"])[-1]}
    report["sensor_maps"] = mapped
    return status, report


def run_live(setup, grader, cusense, evaluator, obs):
    config, out = setup.config["live"], setup.out / "live"
    out.mkdir()
    name, archive_dir, _, archive = setup.captures[0]
    socket = out / "obs.sock"
    offered = len(archive.records) * config["repeat"]
    feed = ["--socket", socket, "--slots", config["ring_slots"], "--slot-us", config["slot_us"], "--repeat",
            config["repeat"], "--pace", "slots"]
    budget = ",".join(str(value) for value in config["budget_us"])
    state = {"before": host_state()}
    # The native core.
    (out / "native").mkdir()
    command = [setup.tools["sense_host"], "--socket", socket, "--artifact", setup.artifact_dir, "--output-dir",
               out / "native", "--device", setup.device, "--engine", setup.engine, "--evidence", "timed", "--wait",
               config["wait"], "--keep-records", "--ready-file", out / "native-ready.json", "--budget-us", budget,
               "--capacity", offered + 64, "--lines", 2 * offered + 1024, "--windows", offered + 64]
    if setup.engine_dir is not None:
        command += ["--engine-dir", setup.engine_dir]
    status, fed = feed_and_sense(setup, "live/native", archive_dir, command, out / "native-ready.json", out / "native.log",
                                 feed, config["ready_timeout_ms"])
    native = cusense.open_output(out / "native") if status == 0 else None
    clean = native is not None and fed.get("dropped") == 0 and fed.get("pending_at_cutoff") == 0 \
        and native.report["counts"]["stale_skipped"] == 0 and native.report["counts"]["stale"] == 0
    grader.check("live", "native", clean and len(native.scored) >= config["min_windows"],
                 f"sense_host ended with status {status}, the feed says {fed}" if native is None else
                 f"sense_host on {native.report['runtime']['device']} with the {native.report['runtime']['engine_backend']} "
                 f"backend read {native.report['source']['records']} records of the ring in place and scored "
                 f"{len(native.scored)} windows (at least {config['min_windows']}): the producer published "
                 f"{fed.get('published')}, dropped {fed.get('dropped')} and had {fed.get('pending_at_cutoff')} credits out "
                 f"at its end; {native.report['counts']['stale_skipped']} windows skipped, "
                 f"{native.report['counts']['late']} late against a window budget of {config['budget_us'][3]} us; the "
                 "output verifies")
    if native is None:
        return {}
    mapped = fed.get("sensor_maps")
    python = [name for name in mapped or [] if "python" in Path(name).name.lower()]
    grader.check("live", "no-interpreter", mapped is not None and not python,
                 "the process inventory of the sensor could not be read" if mapped is None else
                 f"the warm sensor maps {len(mapped)} files and no Python library" if not python else
                 f"the sensor maps {python[0]}")
    ordered = all(line["marks"]["t_notify_ns"] <= line["marks"]["t_submit_ns"] <= line["marks"]["t_ingest_ns"]
                  <= line["marks"]["t_kernels_ns"] <= line["marks"]["t_engine_ns"] <= line["t_decision_ns"]
                  for line in native.scored)
    counts = native.report["counts"]
    grader.check("live", "one-submission", ordered and counts["device_calls_in_an_open_window"] == 0
                 and len(native.scored) <= counts["submissions"] <= len(native.scored) + counts["invalid"],
                 f"{counts['submissions']} submissions for {len(native.scored)} scored windows, each with its marks in "
                 f"order from the notification to the decision; {counts['device_calls_in_an_open_window']} device calls "
                 f"while a window was open; {counts['device_allocations']} allocations, all before the first record")
    # The same records as an archive replay of the same engine: the same scores.
    # The ring gives its own epoch to a record, so the windows are joined by episode and period.
    replayed = setup.native.get(name)
    if replayed is not None:
        tolerance, worst, joined = 1e-6, 0.0, 0
        archived = {(line["window"]["episode"], line["window"]["period"]): line for line in replayed.scored}
        for line in native.scored:
            other = archived.get((line["window"]["episode"], line["window"]["period"]))
            if other is not None:
                joined += 1
                worst = max(worst, abs(line["score"]["logit"] - other["score"]["logit"]))
        grader.check("live", "same-scores", joined >= 1 and worst <= tolerance,
                     f"{joined} windows of the live run have the logit of the archive replay of the same engine within "
                     f"{worst:.2e}: a record read in place on the ring is the record of the archive")
    # The reference sensor on the same offered records.
    reference = None
    (out / "reference").mkdir()
    command = [sys.executable, "-m", "ocudu_cuda_isac.sensing.cusense", "follow", "--socket", socket, "--artifact",
               setup.artifact_dir, "--output-dir", out / "reference", "--wait", config["wait"], "--ready-file",
               out / "reference-ready.json"] + ([] if setup.rehearsal else ["--device", "cuda"])
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    with open(out / "reference.log", "wb") as stream:
        sensor = subprocess.Popen([str(arg) for arg in command], stdout=stream, stderr=subprocess.STDOUT, env=environment)
        deadline = time.monotonic() + config["ready_timeout_ms"] / 1000.0
        while not (out / "reference-ready.json").exists() and sensor.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        other = subprocess.run([str(setup.tools["sense_feed"]), "--archive", str(archive_dir)] + [str(a) for a in feed],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=SENSOR_TIMEOUT_S)
        (out / "reference-feed.log").write_text(other.stdout + other.stderr)
        try:
            status = sensor.wait(timeout=SENSOR_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            sensor.kill()
            status = 124
    try:
        other_fed = json.loads(other.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        other_fed = {"problem": "sense_feed printed no report"}
    if status == 0:
        reference = cusense.open_output(out / "reference")
    grader.check("live", "reference", reference is not None and len(reference.scored) >= 1,
                 f"the reference sensor ended with status {status} (live/reference.log)" if reference is None else
                 f"the reference sensor on {reference.report['runtime']['device']} read the same records from a ring of "
                 f"its own and scored {len(reference.scored)} windows; its producer published {other_fed.get('published')} "
                 f"and dropped {other_fed.get('dropped')}: a record the reference could not take in time is a counted "
                 "drop, and its windows are gaps")
    state["after"] = host_state()
    first = evaluator.latency(native, clock_map="ready_is_last_sample")
    second = None
    if reference is not None:
        rows = obs.open_archive(out / "reference" / "windows" / "records").report["records"]
        second = evaluator.latency(reference, rows, "ready_is_last_sample")
    record = {"schema": evaluator.LATENCY_SCHEMA, "artifact_sha256": setup.artifact.sha256,
              "arm": "the sensor alone on the device: no receiver runs, and the records of a held-out archive are "
                     "published on a ring at the pace of their slots. The arms with a receiver beside the sensor are "
                     "Sprint 4.1's",
              "definitions": evaluator.DEFINITIONS, "offered": {"archive": str(archive_dir), "capture": name,
                                                               "repeat": config["repeat"], "slot_us": config["slot_us"],
                                                               "ring_slots": config["ring_slots"], "wait": config["wait"]},
              "native": dict(first, feed=fed), "reference": None if second is None else dict(second, feed=other_fed),
              "comparison": None if second is None else evaluator.speedup(first, second), "graded": False,
              "host_state": state}
    write_json(setup.out / LATENCY_FILE, record)
    boundary, post = first["cusense_boundary"], first["post_window"]
    stated = boundary["count"] == len(native.scored) and post["count"] == len(native.scored) and first["end_to_end"]["stated"]
    text = "" if second is None else (
        f"; the reference runtime on {second['device']}: {second['cusense_boundary']['p50_us']:.0f} us in the median and "
        f"{second['cusense_boundary']['p99_us']:.0f} us at the 99th percentile over {second['cusense_boundary']['count']} "
        "windows")
    grader.check("live", "latency", stated and (setup.rehearsal or first["timing_valid"]),
                 f"{LATENCY_FILE} holds the three latency definitions for both runtimes with counts and tails, and the "
                 f"state of the host. Native, by the cuSense boundary: {boundary['p50_us']:.1f} us in the median, "
                 f"{boundary['p99_us']:.1f} us at the 99th percentile and {boundary['max_us']:.1f} us at most over "
                 f"{boundary['count']} windows; post-window {post['p50_us']:.1f} us in the median; the stages in the "
                 f"median: ingest {first['stages']['ingest']['p50_us']:.1f}, kernels "
                 f"{first['stages']['kernels']['p50_us']:.1f}, engine {first['stages']['engine']['p50_us']:.1f} us{text}. "
                 "Recorded, not graded" + ("" if first["timing_valid"] else
                                           ": this run is not on the CUDA device with TensorRT and times nothing"))
    return {"native": first, "reference": second}


def run_overload(setup, grader, cusense):
    config, out = setup.config["overload"], setup.out / "overload"
    out.mkdir()
    name, archive_dir, _, archive = setup.captures[0]
    socket = out / "obs.sock"
    (out / "native").mkdir()
    offered = len(archive.records) * config["repeat"]
    command = [setup.tools["sense_host"], "--socket", socket, "--artifact", setup.artifact_dir, "--output-dir",
               out / "native", "--device", setup.device, "--engine", setup.engine, "--evidence", "timed", "--ready-file",
               out / "ready.json", "--hold-us", config["hold_us"], "--budget-us",
               ",".join(str(value) for value in config["budget_us"]), "--lines", 2 * offered + 1024, "--windows",
               offered + 64]
    if setup.engine_dir is not None:
        command += ["--engine-dir", setup.engine_dir]
    feed = ["--socket", socket, "--slots", config["ring_slots"], "--pace", "fixed", "--period-us", config["period_us"],
            "--repeat", config["repeat"]]
    status, fed = feed_and_sense(setup, "overload/native", archive_dir, command, out / "ready.json", out / "native.log",
                                 feed, setup.config["live"]["ready_timeout_ms"])
    output = cusense.open_output(out / "native") if status == 0 else None
    if output is None:
        grader.check("overload", "sensor", False, f"sense_host ended with status {status}, the feed says {fed}")
        return {}
    counts = output.report["counts"]
    offered, pace = fed["offered"], fed["duration_ns"] / 1e3 / max(fed["offered"], 1)
    grader.check("overload", "drops", fed["dropped"] > 0 and fed["admitted"] + fed["dropped"] == offered
                 and fed["published"] == fed["admitted"] and fed["faults"] == 0,
                 f"a producer at one record per {config['period_us']} us against a sensor that works {config['hold_us']} "
                 f"us on a window, on a ring of {config['ring_slots']} slots: of {offered} records {fed['admitted']} were "
                 f"admitted and {fed['dropped']} found no credit, were dropped and counted")
    grader.check("overload", "producer-never-waits", fed["window_calls"] == 0 and pace < 4 * config["period_us"],
                 f"the producer kept its own pace, {pace:.1f} us per record over {offered} records, with a publication "
                 f"of {fed['fill_and_publish_ns']['p99'] / 1e3:.2f} us at the 99th percentile and no system call "
                 "between an admission and its publication")
    grader.check("overload", "credits", fed["pending_at_cutoff"] == 0 and fed["released"] == fed["published"],
                 f"every credit was back when the producer ended: {fed['released']} of {fed['published']} released, "
                 f"{fed['pending_at_cutoff']} out; the deepest backlog on the ring was {fed['max_depth']} records")
    lost = counts["gap"] + counts["stale_skipped"]
    grader.check("overload", "skipped", lost > 0 and counts["device_calls_in_an_open_window"] == 0,
                 f"the sensor scored {counts['scored']} windows and answered {counts['gap']} with a gap and "
                 f"{counts['stale_skipped']} as overtaken by a newer complete window ({counts['stale_skipped_records']} "
                 "records skipped): it never worked through a backlog")
    late = [line for line in output.scored if (line.get("deadline") or {}).get("outcome") == "late"]
    grader.check("overload", "late", len(late) >= 1 and counts["late"] == len(late) and len(late) <= counts["scored"],
                 f"{counts['late']} of {counts['scored']} decisions overran the window budget of "
                 f"{config['budget_us'][3]} us: each was counted and published with the outcome late, and the output "
                 "verifies")
    return {"feed": fed, "counts": counts}


def run_reference(setup, grader, measured, evaluator):
    latencies = measured.get("live") or {}
    native, reference = latencies.get("native"), latencies.get("reference")
    if native is None:
        grader.check("reference", "record", False, "the live case left no native latency")
        return {}
    earlier, path = None, setup.s35_run / setup.config["control"]["driver"] / REFERENCE_FILE
    if path.is_file():
        earlier = json.loads(path.read_text())
    manifest = setup.artifact.manifest
    context = {"artifact_sha256": setup.artifact.sha256, "model_id": manifest["model_id"],
               "input_shape": manifest["input"]["shape"], "grid": [manifest["grid"]["rows"], manifest["grid"]["cols"]],
               "control_run": setup.s35_run.name, "arm": "the sensor alone on the device", "rehearsal": setup.rehearsal,
               "host": platform.node()}
    record = evaluator.reference_record(native, reference, context, earlier)
    write_json(setup.out / REFERENCE_FILE, record)
    published = setup.config["expect"]["published_live_inference_us"]
    boundary = native["cusense_boundary"]
    whole = record["graded"] is False and record["published"]["live_inference_us"]["value"] == published \
        and len(record["differences"]) >= 3 and boundary["count"] >= 1
    grader.check("reference", "record", whole,
                 f"{REFERENCE_FILE} puts the native latency by the cuSense boundary, {boundary['p50_us']:.1f} us in the "
                 f"median and {boundary['p99_us']:.1f} us at the 99th percentile over {boundary['count']} windows on "
                 f"{native['device']} with the {native['engine']} backend, beside the published {published} us with "
                 f"\"graded\": false and lists {len(record['differences'])} differences" +
                 ("" if native["timing_valid"] else "; this run is not on the CUDA device with TensorRT, and the record "
                                                    "says that the native showing is blocked"))
    return {"published_us": published, "native_boundary": boundary, "comparable": record["measured"]["latency"]["comparable"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="new directory for the run")
    parser.add_argument("--s35-run", required=True, type=Path, help="run directory of the accepted Sprint 3.5 gate")
    parser.add_argument("--core-build", type=Path, help="build of the core (default: from the configuration)")
    parser.add_argument("--rehearsal", action="store_true",
                        help="development host: the host twin and the engine host_reference; the hardware gate stays open")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.exit(2, f"sense_core: {args.output_dir} already exists. Evidence is never overwritten.\n")
    try:
        import numpy as np
        from ocudu_cuda_isac.episodes import sense_core as evaluator
        from ocudu_cuda_isac.sensing import cusense, export, obs, runtime
    except ImportError as error:
        parser.exit(2, f"sense_core: the gate needs NumPy and PyTorch in this interpreter: {error}\n")
    try:
        config = load_config(args.config)
        receipt = native_core.verified_receipt(args.core_build or ROOT / config["core_build"])
        if not (args.s35_run / config["control"]["driver"] / "recover.json").is_file():
            raise ConfigError(f"{args.s35_run} holds no {config['control']['driver']}/recover.json of a Sprint 3.5 run")
        if receipt["binding"] is None:
            raise ConfigError("the build of the core holds no binding: install pybind11 in isac and rebuild")
        if not args.rehearsal and receipt["device"] != "cuda":
            raise ConfigError("the build of the core is for the host twin. Build the device: python "
                              "scripts/native_core.py build, or rehearse with --rehearsal")
    except (ConfigError, native_core.CoreError) as error:
        parser.exit(2, f"sense_core: {error}\n")
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = args.output_dir.resolve()
    out.mkdir(parents=True)
    write_json(out / native_core.RECEIPT, receipt)
    setup = Setup(config, out, receipt, args.rehearsal, args.s35_run.resolve())
    grader, measured = Grade(), {}

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

    def passed(case):
        return any(item["case"] == case for item in grader.results) \
            and all(item["passed"] for item in grader.results if item["case"] == case)

    def blocked(cases, reason):
        for case in cases:
            grader.check(case, "control", False, reason)

    core, measured["core"] = guarded("core", lambda: run_core(setup, grader, runtime), (None, {}))
    measured["control"] = guarded("control", lambda: run_control(setup, grader, cusense, obs), {})
    if core is None or not passed("control") or not setup.captures:
        blocked(CASES[2:], "the build of the core or the Sprint 3.5 run does not hold what this case builds on")
    else:
        measured["engine"] = guarded("engine", lambda: run_engine(setup, grader, export), {})
        measured["units"] = guarded("units", lambda: run_units(setup, grader, core, runtime, cusense, evaluator, np), {})
        if not passed("engine"):
            blocked(CASES[4:], "no engine of the artifact runs on this device")
        else:
            measured["parity"] = guarded("parity", lambda: run_parity(setup, grader, core, runtime, cusense, evaluator), {})
            measured["live"] = guarded("live", lambda: run_live(setup, grader, cusense, evaluator, obs), {})
            measured["overload"] = guarded("overload", lambda: run_overload(setup, grader, cusense), {})
            measured["reference"] = guarded("reference", lambda: run_reference(setup, grader, measured, evaluator), {})

    failed = [item for item in grader.results if not item["passed"]]
    passed_cases = [case for case in CASES if passed(case)]
    whole = not failed and len(passed_cases) == len(CASES)
    gate = "closed" if whole and not args.rehearsal else \
        "open: a rehearsal on the host twin times nothing and runs no kernel on a device" if args.rehearsal else "open"
    grade = {"schema": GRADE_SCHEMA, "selector": "sense_core", "run_id": out.name, "cases": len(CASES),
             "cases_passed": len(passed_cases), "cases_failed": len(CASES) - len(passed_cases),
             "checks": len(grader.results), "checks_passed": len(grader.results) - len(failed), "checks_failed": len(failed),
             "judgment": "pass" if whole else "fail", "hardware_gate": gate, "rehearsal": args.rehearsal,
             "reason": "every check of every case passed" if not failed else
                       f"{len(failed)} check(s) failed, first: {failed[0]['case']} {failed[0]['check']}",
             "runtime": cusense.NATIVE_RUNTIME, "device": setup.device, "engine": setup.engine,
             "artifact_sha256": None if setup.artifact is None else setup.artifact.sha256,
             "core_sources_sha256": receipt["sources_sha256"],
             "timed_arms": "pending Sprint 4.1: sense_host as the second MPS client beside the receiver",
             "results": grader.results}
    write_json(out / "grade.json", grade)
    write_json(out / "comparison.json", measured)
    (out / "grade.txt").write_text("\n".join(grader.lines) + "\n")
    write_json(out / "manifest.json", {
        "schema": "integration-manifest/sense_core/v1", "run_id": out.name, "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "command": sys.argv,
        "config": config, "config_sha256": native_core.digest(args.config), "s35_run": str(setup.s35_run),
        "core_build": receipt["build_dir"], "rehearsal": args.rehearsal, "python": platform.python_version(),
        "files": [name for name in (native_core.RECEIPT, PARITY_FILE, LATENCY_FILE, REFERENCE_FILE, "grade.json",
                                    "grade.txt", "comparison.json") if (out / name).is_file()]})
    print(f"sense_core: {grade['cases_passed']} of {grade['cases']} cases, {grade['checks_passed']} of {grade['checks']} "
          f"checks, judgment {grade['judgment']}, hardware gate {gate.split(':')[0]} ({out / 'grade.json'})")
    return 0 if whole else 1


if __name__ == "__main__":
    sys.exit(main())
