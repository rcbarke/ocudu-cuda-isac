#!/usr/bin/env python3
"""Double of the native grid_apply tool, for hosts without an OCUDU build.

It reads a grid-apply-plan/v1 file and writes what native/ocudu_adapter/replay/grid_apply.cpp
writes: ingress/ with one o1-grid/v1 descriptor and one stored grid per opportunity, prestore/
and native.json. The arithmetic is the reference of ocudu_cuda_isac.episodes.grid_apply, in NumPy
when NumPy is installed and in plain Python otherwise. Both give the same bytes.

The unit tests use it to check the reader of the native output and every check of the integration
driver. GRID_APPLY_FAULT names one fault to inject, so that each check can be shown to fail:

    noise-unscaled   the gain is applied to the signal and not to the noise
    single-gain      the gain is applied in single precision
    permuted         the receive planes are written in reverse order
    unrounded        the stored grid holds the values before storage
    base-grid        every opportunity uses the transmit grid of the first grant
    other-noise      the second realization gets the noise of another opportunity
    leak-field       a descriptor names the hash of the channel
    leak-file        ingress/ holds a copy of the plan
    stale-digest     a descriptor names another digest than its grid has
    not-retained     the report says a grid changed while the next one was filled
    accept-all       no plan is refused
    exit             the tool exits with status 3 after writing

It is not a test module: discovery skips it.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ocudu_cuda_isac.episodes import grant as grants  # noqa: E402
from ocudu_cuda_isac.episodes import grid_apply as replay  # noqa: E402
from ocudu_cuda_isac.episodes.grant import GrantError  # noqa: E402
from ocudu_cuda_isac.episodes.grid_apply import GridApplyError  # noqa: E402

try:
    import numpy as np
except ImportError:
    np = None

STORAGE = "complex bf16 in the double of the pinned resource grid, widened without change"
USAGE = ("Usage: grid_apply --plan FILE --output-dir DIR [--no-prestore]\n"
         "       grid_apply --self-test\n"
         "Double of the native tool: it writes the same files with the reference arithmetic.\n")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def load_complex(path):
    """(shape, flat complex values) of a complex64 .npy file."""
    if np is not None:
        array = np.load(path, allow_pickle=False)
        return list(array.shape), array
    _, shape, flat = grants.read_npy(path)
    return list(shape), flat


def load_index(path):
    _, _, flat = grants.read_npy(path)
    return [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]


def scalar_grid(shape, x, h, w, gain, fault):
    """(written, stored) as flat complex lists, with the reference of the module."""
    rx, tx, symbols, subcarriers = shape
    if fault in ("noise-unscaled", "single-gain"):
        plane = symbols * subcarriers
        written = []
        for port in range(rx):
            for element in range(plane):
                total = sum(h[(port * tx + t) * plane + element] * x[t * plane + element] for t in range(tx))
                noise = w[port * plane + element]
                value = total * gain + noise if fault == "noise-unscaled" else (total + noise) * replay.to_single(gain)
                written.append(complex(replay.to_single(value.real), replay.to_single(value.imag)))
    else:
        written = replay.apply(shape, x, h, w, gain)
    return written, replay.stored(written)


def numpy_grid(shape, x, h, w, gain, fault):
    """The same in NumPy: every product, sum and scaling is its own array operation."""
    real = imag = None
    for t in range(shape[1]):
        hr, hi = h[:, t].real.astype(np.float64), h[:, t].imag.astype(np.float64)
        xr, xi = x[t].real.astype(np.float64), x[t].imag.astype(np.float64)
        pr, pi = hr * xr - hi * xi, hr * xi + hi * xr
        real, imag = (pr, pi) if t == 0 else (real + pr, imag + pi)
    wr, wi = w.real.astype(np.float64), w.imag.astype(np.float64)
    if fault == "noise-unscaled":
        real, imag = real * gain + wr, imag * gain + wi
    elif fault == "single-gain":
        real, imag = (real + wr) * float(np.float32(gain)), (imag + wi) * float(np.float32(gain))
    else:
        real, imag = (real + wr) * gain, (imag + wi) * gain
    if not (np.isfinite(real).all() and np.isfinite(imag).all()):
        raise GridApplyError("channel application produced a nonfinite value")
    written = np.empty(real.shape, dtype="<c8")
    written.real, written.imag = real, imag
    raw = written.view(np.uint32)
    raw = (raw + np.uint32(0x7FFF) + ((raw >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xFFFF0000)
    return written, raw.view("<c8").reshape(written.shape)


def to_bytes(values):
    return values.tobytes() if np is not None else replay.grid_bytes(values)


def plane_rows(stored, active, rx, plane):
    """Mean power of every receive plane over the carrier grid and over the occupied elements."""
    rows = []
    if np is not None:
        flat = stored.reshape(rx, plane).astype(np.complex128)
        for port in range(rx):
            power = np.abs(flat[port]) ** 2
            rows.append({"power": float(power.mean()), "power_active": float(power[active].mean())})
        return rows
    for port in range(rx):
        power = [abs(value) ** 2 for value in stored[port * plane:(port + 1) * plane]]
        rows.append({"power": sum(power) / plane, "power_active": sum(power[i] for i in active) / len(active)})
    return rows


def storage_row(written, stored):
    """What storage did to one grid, on every component."""
    if np is not None:
        given = np.ascontiguousarray(written).view(np.float32).astype(np.float64).reshape(-1)
        kept = np.ascontiguousarray(stored).view(np.float32).astype(np.float64).reshape(-1)
        nonzero = given != 0
        worst = float((np.abs(kept - given)[nonzero] / np.abs(given[nonzero])).max()) if nonzero.any() else 0.0
        return {"components": int(given.size), "components_changed": int((given != kept).sum()), "model_mismatches": 0,
                "max_relative_error": worst, "error_power": float(2 * ((kept - given) ** 2).mean()),
                "written_power": float(2 * (given ** 2).mean())}
    changed, worst, error, total, count = 0, 0.0, 0.0, 0.0, 0
    pairs = zip(struct.iter_unpack("<f", to_bytes(written)), struct.iter_unpack("<f", to_bytes(stored)))
    for (given,), (kept,) in pairs:
        count += 1
        changed += given != kept
        error += (kept - given) ** 2
        total += given * given
        if given:
            worst = max(worst, abs(kept - given) / abs(given))
    return {"components": count, "components_changed": changed, "model_mismatches": 0, "max_relative_error": worst,
            "error_power": 2 * error / count, "written_power": 2 * total / count}


def unit_power(h, active, gain, rx, tx, plane):
    if np is not None:
        return float(gain * gain * np.mean(np.abs(h.reshape(rx * tx, plane).astype(np.complex128)[:, active]) ** 2))
    total = sum(abs(h[port * plane + element]) ** 2 for port in range(rx * tx) for element in active)
    return gain * gain * total / (len(active) * rx * tx)


def run(plan_path, output, keep_prestore, fault):
    raw = Path(plan_path).read_bytes()

    def constant(name):
        raise ValueError(f"{name} is not a JSON number")

    try:
        plan = json.loads(raw, parse_constant=constant)
    except ValueError as error:
        raise GridApplyError(f"the plan is not valid JSON: {error}") from None
    if fault != "accept-all":
        # The native reader names a field it does not know before any rule is applied.
        for where, items, fields in (("an opportunity", plan.get("opportunities"), replay.OPPORTUNITY_FIELDS),
                                     ("a source", plan.get("sources"), replay.SOURCE_FIELDS),
                                     ("a grant", plan.get("grants"), replay.GRANT_FIELDS)):
            for item in items if isinstance(items, list) else []:
                for key in sorted(set(item) - set(fields)) if isinstance(item, dict) else []:
                    raise GridApplyError(f"{where} has the unknown field '{key}'")
        replay.check_files(plan)
    carrier = plan["carrier"]
    rx, tx, symbols, subcarriers = (carrier[key] for key in replay.CARRIER_FIELDS)
    shape, plane = (rx, tx, symbols, subcarriers), symbols * subcarriers
    gain = plan["gain"]["value"]
    (output / "ingress").mkdir()
    if keep_prestore:
        (output / "prestore").mkdir()

    files, loaded = [], {}

    def array(ref, role):
        if ref["path"] not in loaded:
            loaded[ref["path"]] = load_complex(ref["path"])[1]
        files.append({"role": role, "path": ref["path"], "sha256": ref["sha256"], "bytes": Path(ref["path"]).stat().st_size})
        return loaded[ref["path"]]

    grants_of = {}
    for item in plan["grants"]:
        x = array(item["x"], f"the transmit grid of grant {item['grant_id']}")
        active = sorted(symbol * subcarriers + k for index in item["active"] for symbol, k in load_index(index["path"]))
        for index in item["active"]:
            files.append({"role": f"the occupied elements of grant {item['grant_id']}", "path": index["path"],
                          "sha256": index["sha256"], "bytes": Path(index["path"]).stat().st_size})
        grants_of[item["grant_id"]] = {"x": x, "active": active, "record": item["record_sha256"],
                                       "active_sha256": sha256(struct.pack(f"<{len(active)}I", *active))}
    sources = {}
    for item in plan["sources"]:
        sources[item["realization_id"]] = (array(item["h"], f"the channel of realization {item['realization_id']}"),
                                           array(item["w"], f"the noise of realization {item['realization_id']}"))
    first_grant = plan["grants"][0]["grant_id"]
    second_realization = plan["sources"][-1]["realization_id"] if len(plan["sources"]) > 1 else None

    def piece(values, index, size):
        """One opportunity of an array whose leading axis is the opportunity."""
        return values[index] if np is not None else values[index * size:(index + 1) * size]

    lines, records, realizations, totals = [], [], {}, {"components": 0, "components_changed": 0, "error": 0.0,
                                                           "written": 0.0, "worst": 0.0}
    generations = [0, 0]
    for sequence, item in enumerate(plan["opportunities"]):
        grant = grants_of[first_grant if fault == "base-grid" else item["grant_id"]]
        h_all, w_all = sources[item["realization_id"]]
        h = piece(h_all, item["index"], rx * tx * plane)
        noise_index = item["index"]
        if fault == "other-noise" and item["realization_id"] == second_realization:
            noise_index = (item["index"] + 1) % (len(w_all) if np is not None else len(w_all) // (rx * plane))
        w = piece(w_all, noise_index, rx * plane)
        written, stored = (numpy_grid if np is not None else scalar_grid)(shape, grant["x"], h, w, gain, fault)
        if fault == "permuted":
            if np is not None:
                written, stored = written[::-1].copy(), stored[::-1].copy()
            else:
                written = [v for port in reversed(range(rx)) for v in written[port * plane:(port + 1) * plane]]
                stored = [v for port in reversed(range(rx)) for v in stored[port * plane:(port + 1) * plane]]
        kept = written if fault == "unrounded" else stored
        name = f"{sequence:04d}.c64"
        (output / "ingress" / name).write_bytes(to_bytes(kept))
        digest = sha256(to_bytes(kept))
        grid_index = sequence % 2
        generations[grid_index] += 1
        record = {"sequence": sequence, "opportunity_id": item["opportunity_id"], "realization_id": item["realization_id"],
                  "index": item["index"], "grant_id": item["grant_id"], "sfn": item["sfn"], "slot": item["slot"],
                  "grid_index": grid_index, "generation": generations[grid_index], "state": "ready",
                  "readback_sha256": digest, "fill_us": 1000.0, "retained": fault != "not-retained",
                  "planes": plane_rows(kept, grant["active"], rx, plane), "storage": storage_row(written, kept)}
        if keep_prestore:
            (output / "prestore" / name).write_bytes(to_bytes(written))
            record["prestore_sha256"] = sha256(to_bytes(written))
        records.append(record)
        line = {"schema": replay.INGRESS_SCHEMA, "run_id": plan["run_id"], "sequence": sequence,
                "opportunity_id": item["opportunity_id"], "sfn": item["sfn"], "slot": item["slot"],
                "grant_id": item["grant_id"], "grant_record_sha256": grant["record"], "active_re": len(grant["active"]),
                "active_sha256": grant["active_sha256"], "grid_index": grid_index, "generation": generations[grid_index],
                "state": "ready",
                "grid": {"file": name, "axes": replay.GRID_AXES, "shape": [rx, symbols, subcarriers], "dtype": "<c8",
                         "order": "C", "storage": STORAGE,
                         "sha256": "0" * 64 if fault == "stale-digest" and sequence == 1 else digest},
                "amplitude": {"grid_gain": gain, "convention": replay.RULE}}
        if fault == "leak-field":
            line["h_sha256"] = next(s["h"]["sha256"] for s in plan["sources"] if s["realization_id"] == item["realization_id"])
        lines.append(json.dumps(line, sort_keys=True))
        if item["realization_id"] not in realizations:
            realizations[item["realization_id"]] = {
                "realization_id": item["realization_id"], "opportunities": 0,
                "first": {"sequence": sequence, "unit_power": unit_power(h, grant["active"], gain, rx, tx, plane)}}
        realizations[item["realization_id"]]["opportunities"] += 1
        row = record["storage"]
        totals["components"] += row["components"]
        totals["components_changed"] += row["components_changed"]
        totals["error"] += row["error_power"]
        totals["written"] += row["written_power"]
        totals["worst"] = max(totals["worst"], row["max_relative_error"])
    (output / "ingress" / "ingress.jsonl").write_text("\n".join(lines) + "\n", encoding="ascii")
    if fault == "leak-file":
        (output / "ingress" / "plan.json").write_bytes(raw)
    count = len(records)
    gain_record = {"value": gain, "source": plan["gain"]["source"], "convention": replay.RULE}
    if plan["gain"].get("reference_power") is not None:
        gain_record["reference_power"] = plan["gain"]["reference_power"]
        gain_record["relative_difference_to_rule"] = abs(gain * math.sqrt(plan["gain"]["reference_power"]) - 1.0)
    if plan["gain"].get("reason"):
        gain_record["reason"] = plan["gain"]["reason"]
    report = {
        "schema": replay.NATIVE_SCHEMA, "run_id": plan["run_id"], "plan": {"file": str(plan_path), "sha256": sha256(raw)},
        "arithmetic": {"rule": replay.RULE, "precision": "double"},
        "carrier": dict(carrier), "gain": gain_record,
        "grid": {"factory": "double of the pinned resource grid", "grids": 2, "ports": rx, "symbols": symbols,
                 "subcarriers": subcarriers, "storage": STORAGE, "little_endian_host": sys.byteorder == "little"},
        "files": files,
        "grants": [{"grant_id": name, "record_sha256": item["record"], "active_re": len(item["active"]),
                    "active_sha256": item["active_sha256"]} for name, item in grants_of.items()],
        "realizations": list(realizations.values()), "opportunities": records,
        "totals": {"opportunities": count, "components": totals["components"],
                   "components_changed": totals["components_changed"], "model_mismatches": 0,
                   "max_relative_error": totals["worst"], "error_power": totals["error"] / count,
                   "written_power": totals["written"] / count, "retained": fault != "not-retained",
                   "generations": generations, "prestore_kept": keep_prestore},
        "access": {"ingress": "ingress/ holds the stored grids and one descriptor per grid, and nothing else",
                   "transport_block": "not read: a plan names none", "labels": "not read: a plan names none"},
        "built_with_cuda": False, "double": True}
    (output / "native.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(f"grid_apply double: {plan['run_id']}, {count} opportunities, gain {gain}")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    fault = os.environ.get("GRID_APPLY_FAULT", "")
    if "--help" in argv:
        print(USAGE, end="")
        return 0
    if "--self-test" in argv:
        print("grid_apply self-test: all checks passed")
        return 0
    keep_prestore = "--no-prestore" not in argv
    words = [word for word in argv if word != "--no-prestore"]
    options = dict(zip(words[::2], words[1::2])) if len(words) % 2 == 0 else {}
    try:
        if set(options) != {"--plan", "--output-dir"}:
            raise GridApplyError("--plan and --output-dir are required")
        output = Path(options["--output-dir"])
        if not output.is_dir():
            raise GridApplyError(f"the output directory {output} does not exist")
        if any((output / name).exists() for name in ("native.json", "ingress", "prestore")):
            raise GridApplyError("the output directory already holds a result. Evidence is never overwritten.")
        run(options["--plan"], output, keep_prestore, fault)
    except (GridApplyError, GrantError, OSError, KeyError, TypeError) as error:
        print(f"grid_apply: input error: {error}", file=sys.stderr)
        return 2
    return 3 if fault == "exit" else 0


if __name__ == "__main__":
    sys.exit(main())
