#!/usr/bin/env python3
"""Sprint 2.3 integration driver (selector grid_apply). Hardware command, not part of unit discovery.

It runs the native channel application, built against the accepted OCUDU build, on the stimulus
that Sprint 2.2 archived, and compares every stored resource element with a reference computed
here. Five cases, each graded on its own:

  unit-vectors     the two-transmit vector of the Sprint 0.3 fixture with g = 1 and g = 2 and a zero
                   channel, through the pinned resource grid, against the scalar reference
  unity            the six archived transmit grids through a unity channel with zero noise and
                   g = 1: every receive plane must return X bit for bit
  reference-pair   the accepted pair on all opportunities and both cases with the grid gain of the
                   entry: values before storage, stored grids, digests, gain, noise, the difference
                   of the two cases, access of the receiver side, a second reader
  allocation       the grant changes on one channel: X and the occupied elements follow the grant
                   and H is read at the new coordinates; a grid of another carrier is refused
  refusals         every gain rule and every binding rule, refused by the plan checker and by the
                   native tool before a file is written

    python tests/integration/grid_apply.py --config configs/integration/grid_apply.json \\
        --s22-run results/runs/s22-spark-001 --output-dir results/runs/s23-spark-001

The output directory must be new. The exit status is 0 only when every check of every case passed.
"""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shlex
import shutil
import struct
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from ocudu_cuda_isac.episodes import episode as episodes  # noqa: E402
from ocudu_cuda_isac.episodes import grant as grants  # noqa: E402
from ocudu_cuda_isac.episodes import grid_apply as replay  # noqa: E402
from ocudu_cuda_isac.episodes.episode import EpisodeError  # noqa: E402
from ocudu_cuda_isac.episodes.grant import GrantError  # noqa: E402
from ocudu_cuda_isac.episodes.grid_apply import GridApplyError  # noqa: E402
from ocudu_cuda_isac.replay import ContractError, load_json  # noqa: E402
import native_adapter  # noqa: E402

CONFIG_SCHEMA = "integration-config/grid_apply/v1"
GRADE_SCHEMA = "integration-grade/grid_apply/v1"
CASES = ("unit-vectors", "unity", "reference-pair", "allocation", "refusals")
TOLERANCES = ("unit_scale_absolute", "unit_scale_relative", "gain_rule_relative", "storage_relative", "unit_power",
              "noise_db", "power_relative")
NATIVE_TIMEOUT_S = 900
# Words that belong to the evaluator. None may appear in what the receiver side reads.
EVALUATOR_WORDS = ("target_present", "truth", "position_m", "velocity_m_s", "bistatic", "labels", "tb.bin",
                   "tb_sha256", "a_h.npy", "b_h.npy", "w.npy")


class ConfigError(ValueError):
    """The configuration or the run setup is unusable. Nothing was graded."""


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fixture = load_script("grant_fixture_driver", ROOT / "tests/integration/grant_fixture.py")
digest, write_json, Grader = fixture.digest, fixture.write_json, fixture.Grader


class Grade(Grader):
    """The Sprint 2.1 grader with this sprint's label."""

    def check(self, case, name, passed, detail):
        self.results.append({"case": case, "check": name, "passed": bool(passed), "detail": detail})
        self._line(("PASS" if passed else "FAIL") + f"  S2.3 {case} {name}: {detail}")
        return bool(passed)

    def note(self, case, text):
        self._line(f"NOTE  S2.3 {case} {text}")


def load_config(path):
    try:
        config = load_json(Path(path))
    except (ContractError, OSError) as error:
        raise ConfigError(f"cannot read the configuration: {error}") from None
    needed = {"schema", "selector", "adapter_build", "tolerance", "pair", "allocation"}
    if not isinstance(config, dict) or set(config) != needed or config["schema"] != CONFIG_SCHEMA \
            or config["selector"] != "grid_apply":
        raise ConfigError(f"the configuration must be a {CONFIG_SCHEMA} object for the selector grid_apply")
    tolerance = config["tolerance"]
    if not isinstance(tolerance, dict) or set(tolerance) != set(TOLERANCES):
        raise ConfigError(f"tolerance requires exactly {', '.join(TOLERANCES)}")
    # The module declares these tolerances and the native core applies them. A second value in the
    # configuration would let the comparison and the code disagree.
    declared = (replay.UNIT_SCALE["absolute"], replay.UNIT_SCALE["relative"], replay.GAIN_RULE_TOLERANCE,
                grants.STORAGE["relative"])
    if (tolerance["unit_scale_absolute"], tolerance["unit_scale_relative"], tolerance["gain_rule_relative"],
            tolerance["storage_relative"]) != declared:
        raise ConfigError(f"the unit-scale, gain-rule and storage tolerances must be the declared ones: {declared}")
    pair = config["pair"]
    if not isinstance(pair, dict) or set(pair) != {"entry", "grant_stores", "expect"} \
            or not isinstance(pair["expect"], dict) or not isinstance(pair["grant_stores"], list):
        raise ConfigError("pair requires exactly entry, grant_stores and expect")
    if not isinstance(config["allocation"], dict) or set(config["allocation"]) != {"changed_grant"}:
        raise ConfigError("allocation requires exactly changed_grant")
    return config


# ---------------------------------------------------------------- native tool
def run_native(tool, plan_path, output, log):
    """Run grid_apply on one plan. Returns (exit status, report or None). output is created empty."""
    output.mkdir(parents=True)
    command = [str(tool), "--plan", str(plan_path), "--output-dir", str(output)]
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


def last_line(log):
    lines = [line for line in Path(log).read_text(errors="replace").splitlines() if line.strip()]
    return lines[-1][:240] if lines else ""


def read_values(path):
    data = Path(path).read_bytes()
    parts = struct.unpack(f"<{len(data) // 4}f", data)
    return [complex(parts[index], parts[index + 1]) for index in range(0, len(parts), 2)]


def same_bits(first, second):
    """True when two complex sequences hold the same single-precision bits, signed zeros included."""
    return replay.grid_bytes(first) == replay.grid_bytes(second)


# ---------------------------------------------------------------- NumPy reference
def reference(np, x, h, w, gain):
    """Y of one opportunity in double precision, with the operations of the kernel in their order.

    x is [tx, symbol, subcarrier], h is [rx, tx, symbol, subcarrier] and w is [rx, symbol,
    subcarrier], all complex64. Returns the real and imaginary parts of g (sum(H X) + W) and of the
    signal part g sum(H X) as float64 [rx, symbol, subcarrier]. Every product, sum and scaling is a
    separate array operation, so no step is fused with another.
    """
    real = imag = None
    for tx in range(h.shape[1]):
        hr, hi = h[:, tx].real.astype(np.float64), h[:, tx].imag.astype(np.float64)
        xr, xi = x[tx].real.astype(np.float64), x[tx].imag.astype(np.float64)
        product_real = hr * xr - hi * xi
        product_imag = hr * xi + hi * xr
        real, imag = (product_real, product_imag) if tx == 0 else (real + product_real, imag + product_imag)
    total_real = (real + w.real.astype(np.float64)) * gain
    total_imag = (imag + w.imag.astype(np.float64)) * gain
    return total_real, total_imag, real * gain, imag * gain


def bf16_array(np, values):
    """The storage rule of the receiver grid on a float32 array: upper 16 bits, nearest, ties to even."""
    raw = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    raw = (raw + np.uint32(0x7FFF) + ((raw >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xFFFF0000)
    return raw.view(np.float32)


def bits(np, values):
    return np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)


def as_complex(np, real, imag):
    out = np.empty(real.shape, dtype=np.complex64)
    out.real, out.imag = real, imag
    return out


def occupied_mask(np, grant_entry, symbols, subcarriers):
    mask = np.zeros((symbols, subcarriers), dtype=bool)
    for index in grant_entry["active"]:
        rows = np.load(index["path"], allow_pickle=False)
        mask[rows[:, 0], rows[:, 1]] = True
    return mask


def load_grids(np, output, folder, count, shape):
    return np.stack([np.fromfile(output / folder / f"{sequence:04d}.c64", dtype="<c8").reshape(shape)
                     for sequence in range(count)])


def grid_digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


# ---------------------------------------------------------------- unit vectors
def fixture_run(case_dir, name, tool, shape, x, h, w, gain, reason):
    """Write one fixture, run the native tool on it and return what it delivered beside the scalar reference."""
    rx, tx, symbols, subcarriers = shape
    directory = case_dir / name
    (directory / "arrays").mkdir(parents=True)
    plane = symbols * subcarriers
    active = [(element // subcarriers, element % subcarriers) for element in range(plane)
              if any(x[port * plane + element] != 0 for port in range(tx))]
    x_ref = replay.write_complex(directory / "arrays" / "x.npy", (tx, symbols, subcarriers), x)
    plan = replay.fixture_plan(
        f"s23-unit-{name}", shape, gain, reason,
        [{"grant_id": f"fixture-{name}", "record_sha256": "fixture:" + x_ref["sha256"], "x": x_ref,
          "active": [replay.write_index(directory / "arrays" / "active.npy", active)]}],
        [{"realization_id": f"fx-{name}",
          "h": replay.write_complex(directory / "arrays" / "h.npy", (1, rx, tx, symbols, subcarriers), h),
          "w": replay.write_complex(directory / "arrays" / "w.npy", (1, rx, symbols, subcarriers), w)}],
        [{"opportunity_id": f"fx-{name}.0000", "realization_id": f"fx-{name}", "index": 0,
          "grant_id": f"fixture-{name}", "sfn": 0, "slot": 0}])
    replay.write_plan(plan, directory / "plan.json")
    status, report = run_native(tool, directory / "plan.json", directory / "native", directory / "native.log")
    result = {"status": status, "report": report, "log": directory / "native.log"}
    if status == 0 and report:
        result["written"] = read_values(directory / "native" / "prestore" / "0000.c64")
        result["stored"] = read_values(directory / "native" / "ingress" / "0000.c64")
        result["digest"] = report["opportunities"][0]["readback_sha256"]
        expected = replay.apply(shape, replay.single(x), replay.single(h), replay.single(w), gain)
        result["expected"], result["expected_stored"] = expected, replay.stored(expected)
        result["expected_digest"] = hashlib.sha256(replay.grid_bytes(result["expected_stored"])).hexdigest()
    return result


def run_unit_vectors(out, grader, tool):
    name = "unit-vectors"
    case_dir = out / "unit_vectors"
    case_dir.mkdir()
    measured = {}
    # The two-transmit, two-symbol vector of tests/test_replay.py and its four answers.
    shape = (2, 2, 2, 1)
    x = [complex(v) for v in (1, 2, 3, 4)]
    h = [1, 1, 1j, -1j, 0.5, 1, 2, 3]
    answers = [1 + 3j, 2 - 4j, 6.5 + 0j, 14 + 0j]
    ok, shown = True, []
    for gain in (1.0, 2.0):
        run = fixture_run(case_dir, f"s03-g{gain:.0f}", tool, shape, x, [complex(v) for v in h], [0j] * 4, gain,
                          "unit vector of the Sprint 0.3 replay fixture: it has no episode entry")
        if run["status"] != 0 or not run["report"]:
            grader.check(name, "Sprint 0.3 vector", False, f"the native tool exited with status {run['status']}: "
                                                           f"{last_line(run['log'])}")
            return measured
        wanted = [gain * value for value in answers]
        gain_record = run["report"]["gain"]
        ok = ok and run["written"] == wanted and run["stored"] == wanted and same_bits(run["stored"], run["expected_stored"]) \
            and run["digest"] == run["expected_digest"] and gain_record["source"] == "override" \
            and gain_record["value"] == gain and bool(gain_record.get("reason"))
        shown.append(f"g = {gain:g}: " + ", ".join(f"{v.real:g}{v.imag:+g}j" for v in run["stored"])
                     + f" (digest {run['digest'][:16]}, reference {run['expected_digest'][:16]})")
        measured[f"s03-g{gain:.0f}"] = {"stored": [[v.real, v.imag] for v in run["stored"]], "digest": run["digest"],
                                        "reference_digest": run["expected_digest"], "gain": gain_record}
    grader.check(name, "Sprint 0.3 vector", ok,
                 "through the pinned resource grid, " + "; ".join(shown) + ": the four answers and exactly twice them; "
                 "the gain is an override, recorded with its reason, because the fixture has no entry")

    # Zero H gives g W for any X, with the gain on the noise and the storage rule after it.
    shape = (2, 1, 2, 4)
    x = [complex(0.31640625 * (1 + n % 3), -0.94921875 + 0.25 * n) for n in range(8)]
    w = [complex(1.0747184e-05 * (n - 7.5), 3.3986e-04 * math.sin(n + 1.0)) for n in range(16)]
    run = fixture_run(case_dir, "zero-channel", tool, shape, x, [0j] * 16, w, 2942.4243526773034,
                      "unit vector with a zero channel: it has no episode entry")
    if run["status"] != 0 or not run["report"]:
        grader.check(name, "zero channel", False, f"the native tool exited with status {run['status']}: "
                                                  f"{last_line(run['log'])}")
        return measured
    scaled = [complex(replay.to_single(2942.4243526773034 * value.real), replay.to_single(2942.4243526773034 * value.imag))
              for value in replay.single(w)]
    changed = sum(a != b for a, b in zip(run["written"], run["stored"]))
    grader.check(name, "zero channel", same_bits(run["written"], scaled) and same_bits(run["written"], run["expected"])
                 and same_bits(run["stored"], run["expected_stored"]) and run["digest"] == run["expected_digest"]
                 and changed > 0,
                 f"zero H with g = 2942.42: all {len(scaled)} values handed to the grid equal g W, and the stored ones "
                 f"equal g W rounded to the storage format ({changed} of {len(scaled)} changed by the rounding); digest "
                 f"{run['digest'][:16]} equals the reference's")
    measured["zero-channel"] = {"digest": run["digest"], "reference_digest": run["expected_digest"],
                                "changed_by_storage": changed}
    return measured


# ---------------------------------------------------------------- stores
def locate_stores(config, s22_run, roots):
    """Open the episode entry, build its plan and open the store entry of every grant the plan binds."""
    entry = s22_run / config["pair"]["entry"]
    search = [s22_run / item for item in config["pair"]["grant_stores"]] + list(roots)
    plan = replay.entry_plan(f"s23-pair-{s22_run.name}", [entry], search)
    episode = episodes.open_episode(entry)
    stores = {item["grant_id"]: grants.open_store(Path(item["x"]["path"]).parent) for item in plan["grants"]}
    return entry, episode, plan, stores, search


# ---------------------------------------------------------------- unity
def run_unity(out, grader, np, tool, plan_grants, stores, carrier):
    name = "unity"
    case_dir = out / "unity"
    (case_dir / "arrays").mkdir(parents=True)
    rx, tx, symbols, subcarriers = carrier
    plane = symbols * subcarriers
    source = {"realization_id": "fx-unity",
              "h": replay.write_complex(case_dir / "arrays" / "h.npy", (1, rx, tx, symbols, subcarriers),
                                        [1 + 0j] * (rx * tx * plane)),
              "w": replay.write_complex(case_dir / "arrays" / "w.npy", (1, rx, symbols, subcarriers), [0j] * (rx * plane))}
    opportunities = [{"opportunity_id": f"fx-unity.{number:04d}", "realization_id": "fx-unity", "index": 0,
                      "grant_id": item["grant_id"], "sfn": 0, "slot": item["slot"]}
                     for number, item in enumerate(plan_grants)]
    reason = "unity check of the archived transmit grids: a fixture without an episode entry"
    plan = replay.fixture_plan("s23-unity", carrier, 1.0, reason, plan_grants, [source], opportunities)
    replay.write_plan(plan, case_dir / "plan.json")
    status, report = run_native(tool, case_dir / "plan.json", case_dir / "native", case_dir / "native.log")
    if status != 0 or not report:
        grader.check(name, "native run", False, f"the native tool exited with status {status}: "
                                                f"{last_line(case_dir / 'native.log')}")
        return {}
    measured = {"gain": report["gain"], "grants": {}}
    grader.check(name, "native run", report["gain"]["source"] == "override" and report["gain"]["value"] == 1.0
                 and report["gain"].get("reason") == reason and report["totals"]["model_mismatches"] == 0
                 and report["totals"]["components_changed"] == 0 and report["totals"]["retained"] is True,
                 f"{len(opportunities)} grids through a unity channel on {rx} receive planes with zero noise and g = 1, "
                 f"an override recorded with its reason; storage changed {report['totals']['components_changed']} of "
                 f"{report['totals']['components']} components")
    for sequence, item in enumerate(plan_grants):
        grant_id = item["grant_id"]
        x = np.load(item["x"]["path"], allow_pickle=False)
        payload = x.tobytes()
        stored = np.fromfile(case_dir / "native" / "ingress" / f"{sequence:04d}.c64", dtype="<c8").reshape(rx, symbols, subcarriers)
        written = np.fromfile(case_dir / "native" / "prestore" / f"{sequence:04d}.c64", dtype="<c8").reshape(stored.shape)
        planes = sum(stored[port].tobytes() == payload and written[port].tobytes() == payload for port in range(rx))
        occupied = len(stores[grant_id].grant.occupied())
        record = report["opportunities"][sequence]
        expected = hashlib.sha256(payload * rx).hexdigest()
        measured["grants"][grant_id] = {"planes_equal": planes, "readback_sha256": record["readback_sha256"],
                                        "x_sha256": item["x"]["sha256"]}
        grader.check(name, f"slot {item['slot']} grid", planes == rx and record["readback_sha256"] == expected
                     and record["grant_id"] == grant_id,
                     f"{grant_id}: {planes} of {rx} receive planes return x.npy ({item['x']['sha256'][:16]}) bit for bit, "
                     f"{x.size} values each with {occupied} occupied; readback digest {record['readback_sha256'][:16]}")
    return measured


# ---------------------------------------------------------------- reference pair
def run_pair(config, out, grader, np, tool, entry, episode, plan, stores):
    name, tolerance, expect = "reference-pair", config["tolerance"], config["pair"]["expect"]
    case_dir = out / "reference_pair"
    case_dir.mkdir()
    manifest = episode.manifest
    gain, power = plan["gain"]["value"], plan["gain"]["reference_power"]
    sigma = manifest["noise"]["sigma"]
    carrier = plan["carrier"]
    rx, symbols, subcarriers = carrier["rx_ports"], carrier["symbols"], carrier["subcarriers"]
    shape = (rx, symbols, subcarriers)
    opportunities = plan["opportunities"]
    count = len(opportunities)
    case_of = {case["realization_id"]: case["id"] for case in manifest["cases"]}
    measured = {"gain": gain, "reference_power": power, "g_sigma": gain * sigma, "sigma": sigma}

    bound = all(item["x"]["sha256"] == manifest["grants"][item["grant_id"]]["x"]["x_sha256"] for item in plan["grants"])
    rule = abs(gain * math.sqrt(power) - 1.0)
    grader.check(name, "plan", bound and count == 2 * expect["opportunities"] and len(plan["grants"]) == expect["grants"]
                 and len(plan["sources"]) == 2 and rule <= tolerance["gain_rule_relative"]
                 and abs(gain - expect["grid_gain"]) <= 1e-9 * expect["grid_gain"]
                 and gain == manifest["calibration"]["grid_gain"],
                 f"from the verified entry {episode.episode_id}: {count} opportunities of {len(plan['sources'])} "
                 f"realizations, {len(plan['grants'])} grants, each transmit grid bound by the hash the entry names; one "
                 f"gain g = {gain:.4f} = calibration.grid_gain, {rule:.1e} from 1 / sqrt(reference_power = {power:.6e})")
    replay.write_plan(plan, case_dir / "plan.json")
    status, report = run_native(tool, case_dir / "plan.json", case_dir / "native", case_dir / "native.log")
    if status != 0 or not report:
        grader.check(name, "native run", False, f"the native tool exited with status {status}: "
                                                f"{last_line(case_dir / 'native.log')}")
        return measured
    native = case_dir / "native"
    try:
        output = replay.open_output(native)
        opened = True
    except GridApplyError as error:
        output, opened = None, str(error)
    totals, records = report["totals"], report["opportunities"]
    generations = [record["generation"] for record in records]
    in_order = all(record["state"] == "ready" and record["retained"] is True for record in records) \
        and all(later["generation"] > earlier["generation"] for earlier, later in zip(records, records[2:])
                if later["grid_index"] == earlier["grid_index"])
    grader.check(name, "native run", opened is True and totals["opportunities"] == count and in_order
                 and report["gain"]["value"] == gain and report["gain"]["source"] == "entry"
                 and len(report["files"]) >= len(plan["grants"]) * 3 + 4,
                 f"{count} grids applied in double precision, written through the pinned writer, made ready, read "
                 f"back and released; two owned grids, generations 1 to {max(generations)}, each grid kept its digest "
                 f"while the next one was filled; {len(report['files'])} input files verified against their hashes "
                 f"(reference_pair/native)" if opened is True else f"the native output does not verify: {opened}")
    if opened is not True:
        return measured

    # ---- The reference, in complex128, for every opportunity. ----
    x_of = {item["grant_id"]: np.load(item["x"]["path"], allow_pickle=False) for item in plan["grants"]}
    mask_of = {item["grant_id"]: occupied_mask(np, item, symbols, subcarriers) for item in plan["grants"]}
    h_of = {item["realization_id"]: np.load(item["h"]["path"], mmap_mode="r", allow_pickle=False) for item in plan["sources"]}
    w_of = {item["realization_id"]: np.load(item["w"]["path"], mmap_mode="r", allow_pickle=False) for item in plan["sources"]}
    real, imag = np.empty((count,) + shape), np.empty((count,) + shape)
    signal_real, signal_imag = np.empty((count,) + shape), np.empty((count,) + shape)
    masks = np.empty((count, symbols, subcarriers), dtype=bool)
    for sequence, item in enumerate(opportunities):
        real[sequence], imag[sequence], signal_real[sequence], signal_imag[sequence] = reference(
            np, x_of[item["grant_id"]], np.asarray(h_of[item["realization_id"]][item["index"]]),
            np.asarray(w_of[item["realization_id"]][item["index"]]), gain)
        masks[sequence] = mask_of[item["grant_id"]]
    written = load_grids(np, native, "prestore", count, shape)
    stored = load_grids(np, native, "ingress", count, shape)
    single_real, single_imag = real.astype(np.float32), imag.astype(np.float32)
    kept_real, kept_imag = bf16_array(np, single_real), bf16_array(np, single_imag)
    elements = int(real.size)
    occupied = masks[:, None, :, :] & np.ones((1, rx, 1, 1), dtype=bool)
    occupied_count = int(occupied.sum())

    # ---- Values handed to the grid: the unit-scale tolerance applies here, after the gain. ----
    error = np.hypot(written.real.astype(np.float64) - real, written.imag.astype(np.float64) - imag)
    magnitude = np.hypot(real, imag)
    within = error <= tolerance["unit_scale_absolute"] + tolerance["unit_scale_relative"] * magnitude
    identical = int(np.count_nonzero((bits(np, written.real) == bits(np, single_real))
                                     & (bits(np, written.imag) == bits(np, single_imag))))
    measured["before_storage"] = {"elements": elements, "within_tolerance": int(within.sum()),
                                  "max_error": float(error.max()), "max_relative_error": float((error / magnitude).max()),
                                  "identical_to_single_precision_reference": identical}
    grader.check(name, "values before storage", bool(within.all()) and identical == elements,
                 f"native against the complex128 reference in Python on all {elements} elements of {count} opportunities: "
                 f"every one within 1e-6 + 1e-5 |reference| after the gain (largest error {float(error.max()):.1e}), and "
                 f"{identical} of {elements} equal the reference converted to single precision bit for bit")

    # ---- The stored grid: the reference rounded to the storage format, element by element. ----
    different = (bits(np, stored.real) != bits(np, kept_real)) | (bits(np, stored.imag) != bits(np, kept_imag))
    mismatched, mismatched_occupied = int(different.sum()), int((different & occupied).sum())
    reference_grids = as_complex(np, kept_real, kept_imag)
    digests = [grid_digest(stored[sequence]) for sequence in range(count)]
    digests_equal = sum(digests[sequence] == records[sequence]["readback_sha256"]
                        == grid_digest(reference_grids[sequence]) for sequence in range(count))
    step = np.maximum(*(np.abs(kept.astype(np.float64) - handed) / np.maximum(np.abs(handed.astype(np.float64)), 1e-300)
                        for kept, handed in ((stored.real, written.real), (stored.imag, written.imag))))
    run_digest = hashlib.sha256("\n".join(digests).encode("ascii")).hexdigest()
    measured["stored"] = {"elements": elements, "occupied_elements": occupied_count, "mismatched": mismatched,
                          "mismatched_occupied": mismatched_occupied, "digests_equal": digests_equal,
                          "max_storage_step": float(step.max()), "run_digest": run_digest,
                          "first_digest": digests[0], "native_model_mismatches": totals["model_mismatches"]}
    grader.check(name, "stored grids", mismatched == 0 and digests_equal == count
                 and float(step.max()) <= tolerance["storage_relative"] and totals["model_mismatches"] == 0,
                 f"read back from the pinned grid: {elements - mismatched} of {elements} elements, the {occupied_count} "
                 f"occupied ones included ({mismatched_occupied} differ), equal the reference rounded to the storage "
                 f"format; {digests_equal} of {count} readback digests equal the reference's; storage moved a component "
                 f"by at most {float(step.max()):.5f} of its magnitude (bound 2^-8 = {tolerance['storage_relative']:.5f})")
    earlier = expect.get("rehearsal_sha256", {})
    if earlier:
        grader.note(name, f"against the x86_64 rehearsal (not graded): digest over the {count} stored grids "
                          f"{'identical' if earlier.get('run_digest') == run_digest else 'different'} "
                          f"({run_digest[:16]}), first grid "
                          f"{'identical' if earlier.get('first_grid') == digests[0] else 'different'} ({digests[0][:16]})")

    # ---- The gain: unit mean power of g H of the target-absent case on opportunity 0. ----
    absent = next(item["realization_id"] for item in plan["sources"] if case_of[item["realization_id"]] == "a")
    first = next(sequence for sequence, item in enumerate(opportunities)
                 if item["realization_id"] == absent and item["index"] == 0)
    h0 = np.asarray(h_of[absent][0]).astype(np.complex128)[:, :, masks[first]]
    unit = float(gain * gain * np.mean(np.abs(h0) ** 2))
    native_unit = next(item["first"] for item in report["realizations"] if item["realization_id"] == absent)
    same_noise = len({item["w"]["sha256"] for item in plan["sources"]}) == 1
    measured["unit_power"] = {"python": unit, "native": native_unit["unit_power"], "g_sigma": gain * sigma}
    grader.check(name, "gain", abs(unit - 1.0) <= tolerance["unit_power"] and native_unit["sequence"] == first
                 and abs(native_unit["unit_power"] - 1.0) <= tolerance["unit_power"]
                 and abs(gain * sigma - expect["g_sigma"]) <= 1e-6 and same_noise
                 and all(line["amplitude"]["grid_gain"] == gain for line in output.descriptors),
                 f"g = {gain:.4f}: the mean power of g H of the target-absent case on opportunity 0, over {rx} receive "
                 f"ports and {int(masks[first].sum())} occupied elements, is 1 {unit - 1.0:+.1e} in Python and 1 "
                 f"{native_unit['unit_power'] - 1.0:+.1e} in the native tool; both cases carry this gain and one noise "
                 f"file ({plan['sources'][0]['w']['sha256'][:16]}); g sigma = {gain * sigma:.6f}")

    # ---- Both cases: the difference of the two stored grids is the return of the target. ----
    index_of = {(item["realization_id"], item["index"]): sequence for sequence, item in enumerate(opportunities)}
    present = next(item["realization_id"] for item in plan["sources"] if case_of[item["realization_id"]] == "b")
    pairs = [(index_of[(absent, index)], index_of[(present, index)]) for index in range(count // 2)]
    a, b = [p[0] for p in pairs], [p[1] for p in pairs]
    worst_ratio, compared, echo_power, rounding_power, prestore_gap = 0.0, 0, 0.0, 0.0, 0.0
    for kept, exact, handed, total in ((stored.real, signal_real, written.real, real),
                                       (stored.imag, signal_imag, written.imag, imag)):
        between = kept[b].astype(np.float64) - kept[a].astype(np.float64)
        wanted = exact[b] - exact[a]  # g (H_b - H_a) X: the noise is the same in both and cancels
        limit = (2.0 ** -8 + 2.0 ** -22) * (np.abs(total[a]) + np.abs(total[b]))
        gap = np.abs(between - wanted)
        worst_ratio = max(worst_ratio, float((gap / np.maximum(limit, 1e-300)).max()))
        compared += int(gap.size)
        echo_power += float(np.mean(wanted[occupied[a]] ** 2))
        rounding_power += float(np.mean(gap[occupied[a]] ** 2))
        # Before storage the two cases differ by the return alone: same gain, same noise.
        early = np.abs((handed[b].astype(np.float64) - handed[a].astype(np.float64)) - wanted)
        prestore_gap = max(prestore_gap, float((early / (np.abs(total[a]) + np.abs(total[b]))).max()))
    measured["difference"] = {"components": compared, "largest_error_over_bound": worst_ratio,
                              "return_rms": math.sqrt(echo_power), "rounding_rms": math.sqrt(rounding_power),
                              "before_storage_relative_gap": prestore_gap}
    grader.check(name, "difference of the two cases", worst_ratio <= 1.0 and prestore_gap <= 2.0 ** -22,
                 f"stored grid of the target-present case minus that of the target-absent case equals g (H_b - H_a) X "
                 f"within the rounding bound on all {compared} components (largest error {worst_ratio:.2f} of the "
                 f"bound); on the occupied elements the return has an RMS of {math.sqrt(echo_power):.2e} and the "
                 f"rounding of the two grids {math.sqrt(rounding_power):.2e}; before storage the two cases differ by "
                 f"the return alone, to {prestore_gap:.1e} of their magnitude")

    # ---- The noise the grid holds after storage, and the power of every receive plane. ----
    held = (stored.real.astype(np.float64) - signal_real) ** 2 + (stored.imag.astype(np.float64) - signal_imag) ** 2
    held_db = 10 * math.log10(float(held.mean()))
    wanted_db = 20 * math.log10(gain * sigma)
    rounding = (stored.real.astype(np.float64) - written.real) ** 2 + (stored.imag.astype(np.float64) - written.imag) ** 2
    sqnr_db = 10 * math.log10(float((real ** 2 + imag ** 2).mean()) / float(rounding.mean())) if rounding.any() \
        else math.inf
    measured["noise"] = {"held_db": held_db, "expected_db": wanted_db, "rounding_sqnr_db": sqnr_db,
                         "snr_db": manifest["noise"]["snr_db"]}
    grader.check(name, "noise after storage", abs(held_db - wanted_db) <= tolerance["noise_db"]
                 and abs(wanted_db + expect["snr_db"]) <= 1e-6 and sqnr_db > expect["snr_db"],
                 f"the stored grid minus the signal part g sum(H X) has a power of {held_db:.3f} dB relative to unit "
                 f"signal power on all {elements} elements, against g sigma = {gain * sigma:.4f} ({wanted_db:.3f} dB); "
                 f"the rounding of the grid lies {sqnr_db:.1f} dB below the stored values, so the noise is held")
    plane_rows, plane_gap = {}, 0.0
    for realization in (absent, present):
        chosen = [sequence for sequence, item in enumerate(opportunities) if item["realization_id"] == realization]
        ours, theirs = [], []
        for port in range(rx):
            power_here = [float(np.mean(np.abs(reference_grids[sequence, port][masks[sequence]].astype(np.complex128)) ** 2))
                          for sequence in chosen]
            ours.append(sum(power_here) / len(power_here))
            theirs.append(sum(records[sequence]["planes"][port]["power_active"] for sequence in chosen) / len(chosen))
            plane_gap = max(plane_gap, max(abs(records[sequence]["planes"][port]["power_active"] - value) / value
                                           for sequence, value in zip(chosen, power_here)))
        plane_rows[case_of[realization]] = {"python": ours, "native": theirs}
    measured["planes"] = plane_rows
    grader.check(name, "receive planes", plane_gap <= tolerance["power_relative"]
                 and all(value > 0 and math.isfinite(value) for row in plane_rows.values() for value in row["native"]),
                 "mean power of the stored grid on the occupied elements, per receive plane 0 to "
                 f"{rx - 1}: target absent " + ", ".join(f"{value:.5f}" for value in plane_rows["a"]["native"])
                 + "; target present " + ", ".join(f"{value:.5f}" for value in plane_rows["b"]["native"])
                 + f"; the native measurement and Python agree within {plane_gap:.1e} on every opportunity")

    # ---- X depends on the slot: each opportunity must carry the grid of its own grant. ----
    base_id = episode.grant.grant_id
    moved, misplaced = {}, 0
    for sequence, item in enumerate(opportunities):
        if item["grant_id"] == base_id or item["realization_id"] != absent:
            continue
        other_real, other_imag, _, _ = reference(np, x_of[base_id], np.asarray(h_of[absent][item["index"]]),
                                                 np.asarray(w_of[absent][item["index"]]), gain)
        differs = np.any((bits(np, stored[sequence].real) != bits(np, bf16_array(np, other_real.astype(np.float32))))
                         | (bits(np, stored[sequence].imag) != bits(np, bf16_array(np, other_imag.astype(np.float32)))),
                         axis=0)
        pilots = np.zeros_like(differs)
        rows = np.array(stores[item["grant_id"]].grant.pilot_index)
        pilots[rows[:, 0], rows[:, 1]] = True
        misplaced += int((differs & ~pilots).sum())
        moved.setdefault(item["grant_id"], set()).add(int(differs.sum()))
    measured["grants"] = {grant_id: sorted(values) for grant_id, values in moved.items()}
    grader.check(name, "grant of each opportunity", bool(moved) and misplaced == 0
                 and all(len(values) == 1 and min(values) > 0 for values in moved.values())
                 and len(moved) == len(plan["grants"]) - 1,
                 "the transmit grid follows the grant of the slot: a reference that keeps the grid of "
                 f"{base_id} at every opportunity differs from the stored grid on "
                 + ", ".join(f"{min(values)} pilots of {grant_id.rsplit('.', 1)[-1]}" for grant_id, values in sorted(moved.items()))
                 + f" and on {misplaced} other elements")

    # ---- Access: what the receiver side reads names no channel, no noise, no block and no label. ----
    secrets = [entry_file["sha256"] for entry_file in manifest["files"].values()]
    secrets += [store.manifest["derived"]["tb_sha256"] for store in stores.values()]
    secrets += [store.manifest["files"]["tb.bin"]["sha256"] for store in stores.values()]
    secrets += list(EVALUATOR_WORDS) + [case["id"] + "_h" for case in manifest["cases"]]
    violations = replay.ingress_violations(native, secrets)
    roles = sorted({item["role"].split(" of ")[0] for item in report["files"]})
    opaque = all(line["opportunity_id"].split(".")[0] in case_of for line in output.descriptors) \
        and len({line["opportunity_id"] for line in output.descriptors}) == count
    measured["access"] = {"violations": violations, "roles_read": roles,
                          "ingress_files": len(list((native / "ingress").iterdir()))}
    grader.check(name, "access", not violations and opaque and not any("block" in role for role in roles)
                 and report["access"]["transport_block"].startswith("not read"),
                 f"ingress/ holds {count} stored grids and their {count} descriptors and nothing else; a descriptor has "
                 f"the {len(replay.INGRESS_FIELDS)} declared fields, opaque opportunity identifiers, and no hash or name "
                 "of H, W or a transport block and no label; the tool read " + ", ".join(roles)
                 if not violations else "the receiver side could read: " + "; ".join(violations[:5]))

    # ---- A second reader: a separate process, standard library only. ----
    pilot_symbol, pilot_subcarrier = stores[opportunities[first]["grant_id"]].grant.pilot_index[0]
    reader = subprocess.run([sys.executable, "-m", "ocudu_cuda_isac.episodes.grid_apply", "show", str(native),
                             "--sequence", str(first), "--rx", str(rx - 1), "--symbol", str(pilot_symbol),
                             "--subcarrier", str(pilot_subcarrier)], capture_output=True, text=True, check=False,
                            env=dict(os.environ, PYTHONPATH=str(ROOT / "src")))
    try:
        shown = json.loads(reader.stdout)
        write_json(case_dir / "second-reader.json", shown)
        value = complex(*shown["element"]["value"])
        wanted = complex(float(kept_real[first, rx - 1, pilot_symbol, pilot_subcarrier]),
                         float(kept_imag[first, rx - 1, pilot_symbol, pilot_subcarrier]))
        grader.check(name, "second reader", reader.returncode == 0 and value == wanted
                     and shown["readback_sha256"] == digests[first] and shown["verified_grids"] == count,
                     f"a separate process verifies the {count} stored grids against their digests and recovers Y at "
                     f"opportunity {shown['opportunity_id']}, port {rx - 1}, symbol {pilot_symbol}, subcarrier "
                     f"{pilot_subcarrier} = {value.real:+.6f}{value.imag:+.6f}j, the reference rounded to the grid "
                     f"(reference_pair/second-reader.json)")
    except (ValueError, KeyError, TypeError) as error:
        grader.check(name, "second reader", False, f"exit status {reader.returncode}: {error} {reader.stderr[-300:]}")

    fill = [record["fill_us"] for record in records]
    grader.note(name, f"fill time of one grid, not graded: mean {sum(fill) / len(fill):.0f} us, largest {max(fill):.0f} us "
                      f"for {rx} by {symbols} by {subcarriers} elements on one core; the stimulus is prepared ahead of "
                      "the slot it is used in")
    measured["fill_us"] = {"mean": sum(fill) / len(fill), "max": max(fill), "min": min(fill)}
    measured["built_with_cuda"] = report.get("built_with_cuda")
    return measured


# ---------------------------------------------------------------- allocation
def run_allocation(config, out, grader, np, tool, plan, stores, search):
    name = "allocation"
    case_dir = out / "allocation"
    (case_dir / "arrays").mkdir(parents=True)
    carrier = plan["carrier"]
    rx, tx, symbols, subcarriers = (carrier[key] for key in replay.CARRIER_FIELDS)
    changed_id = config["allocation"]["changed_grant"]
    base = plan["grants"][0]
    try:
        changed_store = next(grants.open_store(root / changed_id / "store") for root in search
                             if (root / changed_id / "store" / "manifest.json").is_file())
    except (StopIteration, GrantError) as error:
        grader.check(name, "changed grant", False, f"no verified store entry of the grant {changed_id} under "
                                                   f"{', '.join(str(root) for root in search)}: {error}")
        return {}
    changed = replay.grant_from_store(changed_store)
    # A channel that holds its own coordinates: 1 + port / 4 in the real part, the element number in the imaginary part.
    plane = symbols * subcarriers
    h = np.empty((1, rx, tx, symbols, subcarriers), dtype="<c8")
    h.real = (1.0 + np.arange(rx) / 4.0)[None, :, None, None, None]
    h.imag = (np.arange(plane, dtype=np.float64) / plane).reshape(symbols, subcarriers)[None, None, None]
    np.save(case_dir / "arrays" / "h.npy", h, allow_pickle=False)
    np.save(case_dir / "arrays" / "w.npy", np.zeros((1, rx, symbols, subcarriers), dtype="<c8"), allow_pickle=False)
    source = {"realization_id": "fx-coordinates", "h": replay.file_ref(case_dir / "arrays" / "h.npy"),
              "w": replay.file_ref(case_dir / "arrays" / "w.npy")}
    order = [base, changed]
    opportunities = [{"opportunity_id": f"fx-coordinates.{number:04d}", "realization_id": "fx-coordinates", "index": 0,
                      "grant_id": item["grant_id"], "sfn": 0, "slot": item["slot"]} for number, item in enumerate(order)]
    fixture_plan = replay.fixture_plan("s23-allocation", (rx, tx, symbols, subcarriers), 1.0,
                                       "changed allocation on a coordinate-coded channel: a fixture without an entry",
                                       order, [source], opportunities)
    replay.write_plan(fixture_plan, case_dir / "plan.json")
    status, report = run_native(tool, case_dir / "plan.json", case_dir / "native", case_dir / "native.log")
    if status != 0 or not report:
        grader.check(name, "changed grant", False, f"the native tool exited with status {status}: "
                                                   f"{last_line(case_dir / 'native.log')}")
        return {}
    measured, ok, parts = {}, True, []
    for sequence, item in enumerate(order):
        x = np.load(item["x"]["path"], allow_pickle=False)
        mask = occupied_mask(np, item, symbols, subcarriers)
        stored = np.fromfile(case_dir / "native" / "ingress" / f"{sequence:04d}.c64", dtype="<c8").reshape(rx, symbols, subcarriers)
        real, imag, _, _ = reference(np, x, h[0], np.zeros((rx, symbols, subcarriers), dtype="<c8"), 1.0)
        equal = np.array_equal(bits(np, stored.real), bits(np, bf16_array(np, real.astype(np.float32)))) \
            and np.array_equal(bits(np, stored.imag), bits(np, bf16_array(np, imag.astype(np.float32))))
        support = np.array_equal(np.all(stored != 0, axis=0), mask) and np.array_equal(np.any(stored != 0, axis=0), mask)
        columns = np.nonzero(mask.any(axis=0))[0]
        ok = ok and equal and support and report["opportunities"][sequence]["grant_id"] == item["grant_id"]
        measured[item["grant_id"]] = {"occupied": int(mask.sum()), "subcarriers": [int(columns[0]), int(columns[-1])],
                                      "equal_to_reference": bool(equal), "support_equal": bool(support)}
        parts.append(f"{item['grant_id']} occupies {int(mask.sum())} elements on subcarriers {int(columns[0])} to "
                     f"{int(columns[-1])}")
    records = report["opportunities"]
    ok = ok and records[0]["readback_sha256"] != records[1]["readback_sha256"]
    grader.check(name, "changed grant", ok,
                 "the grant changes on one channel that holds its own coordinates: " + ", then ".join(parts)
                 + f"; each stored grid is nonzero on exactly those elements of all {rx} planes and equals H at those "
                   "coordinates times X of the grant, rounded to the grid")

    # A transmit grid on another carrier grid cannot be remapped: it is refused before a file is written.
    narrow = np.load(base["x"]["path"], allow_pickle=False)[:, :, :-12]
    np.save(case_dir / "arrays" / "x_other_carrier.npy", np.ascontiguousarray(narrow), allow_pickle=False)
    other = copy.deepcopy(fixture_plan)
    other["run_id"] = "s23-allocation-other-carrier"
    other["grants"][0]["x"] = replay.file_ref(case_dir / "arrays" / "x_other_carrier.npy")
    write_json(case_dir / "other-carrier-plan.json", other)
    try:
        replay.check_files(other)
        refused = None
    except GridApplyError as error:
        refused = str(error)
    status, _ = run_native(tool, case_dir / "other-carrier-plan.json", case_dir / "other-carrier",
                           case_dir / "other-carrier.log")
    said = last_line(case_dir / "other-carrier.log")
    untouched = not any((case_dir / "other-carrier").iterdir())
    measured["other_carrier"] = {"plan_check": refused, "native_status": status, "native_message": said}
    grader.check(name, "another carrier", bool(refused) and status == 2 and untouched and "changed allocation" in said,
                 f"a transmit grid of {narrow.shape[2]} subcarriers on the {subcarriers}-subcarrier carrier is refused by "
                 f"the plan check and by the native tool with status {status} and no file written: {said}")
    return measured


# ---------------------------------------------------------------- refusals
def damaged_entry(entry, scratch, change):
    """A copy of an episode entry, the arrays linked and the manifest changed."""
    copy_dir = scratch / "entry"
    copy_dir.mkdir(parents=True)
    for path in entry.iterdir():
        if path.name == "manifest.json":
            continue
        try:
            os.link(path, copy_dir / path.name)
        except OSError:
            shutil.copyfile(path, copy_dir / path.name)
    manifest = json.loads((entry / "manifest.json").read_text())
    change(manifest)
    write_json(copy_dir / "manifest.json", manifest)
    return copy_dir


def run_refusals(config, out, grader, tool, entry, plan, search):
    name = "refusals"
    case_dir = out / "refusals"
    case_dir.mkdir()
    gain = plan["gain"]["value"]
    other_grant = config["allocation"]["changed_grant"]

    def rescaled(item, factor):
        item["gain"]["value"] = gain * factor
        for source in item["sources"]:
            source["entry"]["grid_gain"] = gain * factor

    def with_gain(value):
        def change(item):
            item["gain"]["value"] = value
        return change

    # Each rule with the words its refusal must carry: in the plan check, then in the native tool.
    gain_rules = [
        ("a missing gain", lambda item: item.pop("gain"), "names no grid gain", "names no grid gain"),
        ("a zero gain", with_gain(0.0), "must be positive", "must be positive"),
        ("a negative gain", with_gain(-gain), "must be positive", "must be positive"),
        # JSON has no infinity: the plan check meets the value, the native tool a file it cannot parse.
        ("a gain that is not finite", with_gain(math.inf), "not finite", "not valid JSON"),
        ("two gains in one run",
         lambda item: item["sources"][1]["entry"].update(grid_gain=1140.799, reference_power=1140.799 ** -2),
         "two gains in one run", "two gains in one run"),
        ("a gain 2e-6 off 1 / sqrt(reference_power)", lambda item: rescaled(item, 1.0 + 2e-6),
         "differs from 1 / sqrt(reference_power)", "differs from 1 / sqrt(reference_power)"),
        ("an override on an entry", lambda item: item["gain"].update(source="override", reason="replace the entry's gain"),
         "only for a fixture that has no entry", "only for a fixture that has no entry"),
    ]
    binding_rules = [
        ("a transmit grid whose hash is not the bound one",
         lambda item: item["grants"][1]["x"].update(sha256=item["grants"][0]["x"]["sha256"]),
         "record binds", "record binds"),
        ("a transmit grid bound by no hash", lambda item: item["grants"][1]["x"].update(sha256=None),
         "no bound transmit grid", "no bound transmit grid"),
        ("an opportunity without a bound grid", lambda item: item["opportunities"][5].update(grant_id=None),
         "has no bound grid", "has no bound grid"),
        ("an opportunity on a grant the run does not bind",
         lambda item: item["opportunities"][5].update(grant_id=other_grant), "does not bind", "does not bind"),
        ("a channel with another hash", lambda item: item["sources"][0]["h"].update(sha256="0" * 64),
         "record binds", "record binds"),
        ("a gain per opportunity", lambda item: item["opportunities"][0].update(gain=2.0),
         "requires exactly", "unknown field 'gain'"),
    ]
    measured = {}
    for label, rules in (("gain rules", gain_rules), ("binding rules", binding_rules)):
        refused, accepted = [], []
        for number, (what, change, checker_words, native_words) in enumerate(rules):
            mutated = copy.deepcopy(plan)
            mutated["run_id"] = f"s23-refusal-{label.split()[0]}-{number}"
            change(mutated)
            try:
                replay.check_files(mutated)
                checker = None
            except GridApplyError as error:
                checker = str(error)
            directory = case_dir / f"{label.split()[0]}-{number}"
            directory.mkdir()
            # The plan is written as it is, checks bypassed, so that the native tool meets it.
            (directory / "plan.json").write_text(json.dumps(mutated, indent=2, sort_keys=True) + "\n")
            status, _ = run_native(tool, directory / "plan.json", directory / "native", directory / "native.log")
            said = last_line(directory / "native.log")
            untouched = not any((directory / "native").iterdir())
            measured[what] = {"plan_check": checker, "native_status": status, "native_message": said,
                              "no_file_written": untouched}
            named = bool(checker) and checker_words in checker and native_words in said
            (refused if named and status == 2 and untouched else accepted).append(what)
        grader.check(name, label, not accepted and len(refused) == len(rules),
                     "refused by the plan check and by the native tool, each naming the rule, with status 2 before a "
                     "file is written: " + ", ".join(refused)
                     + ("" if not accepted else "; NOT REFUSED AS REQUIRED: " + ", ".join(accepted)))

    # The entry itself decides which grid is bound: the plan builder refuses an entry that binds
    # another hash than the store holds, and one that binds no grid.
    grant_id = plan["grants"][1]["grant_id"]

    def another_hash(manifest):
        manifest["grants"][grant_id]["x"]["x_sha256"] = "0" * 64

    def no_grid(manifest):
        manifest["grants"][grant_id]["x"] = None

    built = {}
    for what, change in (("an entry that binds another hash than the stored grid", another_hash),
                         ("an entry that binds no grid to a grant", no_grid)):
        scratch = case_dir / ("entry-" + what.split()[4])
        try:
            replay.entry_plan("s23-refusal-entry", [damaged_entry(entry, scratch, change)], search)
            built[what] = None
        except GridApplyError as error:
            built[what] = str(error)
        shutil.rmtree(scratch)
    measured.update({what: {"plan_builder": reason} for what, reason in built.items()})
    grader.check(name, "entry bindings", all(built.values()),
                 "the plan builder refuses " + " and ".join(what for what, reason in built.items() if reason)
                 + ("" if all(built.values()) else "; ACCEPTED: " + ", ".join(w for w, r in built.items() if not r))
                 + f": {next(iter(built.values()))}")
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
    parser.add_argument("--s22-run", required=True, type=Path,
                        help="run directory of the accepted Sprint 2.2 gate: it holds the pair and the slot grants")
    parser.add_argument("--grant-store", action="append", type=Path, default=[],
                        help="further directory that holds <grant_id>/store (default: the paths the entry recorded)")
    parser.add_argument("--adapter-build", type=Path, help="adapter build directory (default: from the configuration)")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.exit(2, f"grid_apply: {args.output_dir} already exists. Evidence is never overwritten.\n")
    try:
        config = load_config(args.config)
        receipt = native_adapter.verified_receipt(args.adapter_build or ROOT / config["adapter_build"])
        if "grid_apply" not in receipt["executables"]:
            raise ConfigError("the adapter build holds no grid_apply. Rebuild it: python scripts/native_adapter.py build")
        import numpy as np
    except (ConfigError, native_adapter.AdapterError, ImportError) as error:
        parser.exit(2, f"grid_apply: {error}\n")
    tool = Path(receipt["executables"]["grid_apply"]["path"])
    s22_run = args.s22_run.resolve()
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = args.output_dir.resolve()
    out.mkdir(parents=True)
    write_json(out / "adapter-receipt.json", receipt)

    grader, measured = Grade(), {}

    def guarded(case, action):
        before = len(grader.results)
        try:
            return action()
        except Exception as error:  # an unexpected fault fails its case and keeps the remaining cases
            (out / f"driver-error-{case}.txt").write_text(traceback.format_exc())
            grader.check(case, "driver", False, f"{type(error).__name__}: {error} (driver-error-{case}.txt)")
            return {}
        finally:
            if len(grader.results) == before:
                grader.check(case, "driver", False, "the case produced no check")

    measured["unit-vectors"] = guarded("unit-vectors", lambda: run_unit_vectors(out, grader, tool))
    # The stimulus of the accepted Sprint 2.2 run. Without it the three cases below cannot run.
    located = None
    try:
        located = locate_stores(config, s22_run, [path.resolve() for path in args.grant_store])
    except (GridApplyError, EpisodeError, GrantError, OSError) as error:
        problem = f"the stimulus of the Sprint 2.2 run is unusable: {error}"
    if located:
        entry, episode, plan, stores, search = located
        # The recorded store of the base grant lies in the Sprint 2.1 run, beside the other shipped grants.
        search = search + [Path(item["x"]["path"]).parents[2] for item in plan["grants"]]
        carrier = tuple(plan["carrier"][key] for key in replay.CARRIER_FIELDS)
        measured["unity"] = guarded("unity", lambda: run_unity(out, grader, np, tool, plan["grants"], stores, carrier))
        measured["reference-pair"] = guarded("reference-pair", lambda: run_pair(config, out, grader, np, tool, entry,
                                                                                 episode, plan, stores))
        measured["allocation"] = guarded("allocation", lambda: run_allocation(config, out, grader, np, tool, plan,
                                                                             stores, search))
        measured["refusals"] = guarded("refusals", lambda: run_refusals(config, out, grader, tool, entry, plan, search))
    else:
        for case in CASES[1:]:
            grader.check(case, "stimulus", False, problem)

    failed = [item for item in grader.results if not item["passed"]]
    passed_cases = [case for case in CASES if any(item["case"] == case for item in grader.results)
                    and all(item["passed"] for item in grader.results if item["case"] == case)]
    grade = {
        "schema": GRADE_SCHEMA, "selector": "grid_apply", "run_id": out.name, "cases": len(CASES),
        "cases_passed": len(passed_cases), "cases_failed": len(CASES) - len(passed_cases),
        "checks": len(grader.results), "checks_passed": len(grader.results) - len(failed), "checks_failed": len(failed),
        "judgment": "pass" if not failed and len(passed_cases) == len(CASES) else "fail",
        "reason": "every check of every case passed" if not failed else
                  f"{len(failed)} check(s) failed, first: {failed[0]['case']} {failed[0]['check']}",
        "tolerance": config["tolerance"], "results": grader.results}
    write_json(out / "grade.json", grade)
    write_json(out / "comparison.json", measured)
    lock = {item["name"]: item["commit"] for item in json.loads((ROOT / "ext/lock.json").read_text())["dependencies"]}
    write_json(out / "manifest.json", {
        "run_id": out.name, "selector": "grid_apply",
        "purpose": "Sprint 2.3 integration: native channel application with the grid gain into the pinned resource "
                   "grid, against a complex128 reference, on the stimulus of the accepted Sprint 2.2 run",
        "command": shlex.join([sys.executable, *sys.argv]), "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": platform.node(), "machine": platform.machine(), "python": platform.python_version(),
        "numpy": np.__version__, "parent_commit": git_commit(), "locked_sources": lock,
        "inputs": {str(args.config): digest(args.config)}, "driver_sha256": digest(__file__),
        "reference_sha256": digest(ROOT / "src/ocudu_cuda_isac/episodes/grid_apply.py"),
        "s22_run": str(s22_run), "grant_stores": [str(path.resolve()) for path in args.grant_store],
        "adapter": {"executable": str(tool), "sha256": receipt["executables"]["grid_apply"]["sha256"],
                    "receipt": "adapter-receipt.json", "ocudu_build": receipt["ocudu_build"]},
        "environment": {key: os.environ[key] for key in ("VIRTUAL_ENV", "CUDA_VISIBLE_DEVICES") if key in os.environ},
        "judgment": grade["judgment"], "reason": grade["reason"]})
    summary = (f"grid_apply: {grade['cases_passed']} of {grade['cases']} cases passed, "
               f"{grade['checks_passed']} of {grade['checks']} checks")
    (out / "grade.txt").write_text("\n".join(grader.lines + [summary]) + "\n", encoding="ascii")
    print(summary)
    print(f"Run artifacts: {out}")
    return 0 if grade["judgment"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
