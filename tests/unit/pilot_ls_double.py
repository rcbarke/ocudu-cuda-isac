#!/usr/bin/env python3
"""Double of the native pilot_ls tool, for hosts without an OCUDU build.

It reads a pilot-ls-plan/v1 file and writes what native/ocudu_adapter/observer/pilot_ls.cpp
writes. The receiver side is the double of Sprint 2.4 (tests/unit/replay_pusch_double.py) on the
replay plan that the plan names: ingress/, radio/ and the report. With the tap "pilot_ls" it adds
observer/: one obs-descriptor/v1 record, one payload and one validity mask per opportunity,
computed from the stored grid with the reference of ocudu_cuda_isac.sensing.pilot_ls.

The unit tests use it to check the reader of the native output and every check of the integration
driver. PILOT_LS_FAULT names one fault to inject, so that each check can be shown to fail:

    stored-pilot     the estimate divides by the stored pilot and not by the reference of the source
    conjugate        the estimate is Y X / |X|^2
    swap-planes      the receive planes of the payload are written in reverse order
    shifted          every pilot is read one subcarrier higher
    data-dependent   the estimate depends on a data resource element
    mask-clear       the first validity bit is cleared although its estimate is finite
    wrong-slot       the absolute slot is one too high
    other-grid       the second observation names another grid digest than its opportunity was filled with
    touch-grid       the tap changes the first grid before it is decoded
    changed-decode   with the tap on, the CUDA processor returns a wrong byte
    host-fallback    the CUDA processor executes the host path
    drop-observation the last observation record is missing
    leak             an observation record names the hash of a transport block
    tap-when-off     a run without a tap writes observations
    accept-all       no plan is refused
    exit             the tool exits with status 3 after writing

PILOT_LS_BUILD=host makes it the tool of a build without CUDA, as REPLAY_PUSCH_BUILD does for the
double of Sprint 2.4.

It is not a test module: discovery skips it.
"""

import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ocudu_cuda_isac.episodes import grant as grants  # noqa: E402
from ocudu_cuda_isac.episodes import pilot_ls as pl  # noqa: E402
from ocudu_cuda_isac.episodes.grant import GrantError  # noqa: E402
from ocudu_cuda_isac.episodes.grid_apply import GridApplyError  # noqa: E402
from ocudu_cuda_isac.episodes.pilot_ls import PilotLsPlanError  # noqa: E402
from ocudu_cuda_isac.episodes.replay_pusch import ReplayPuschError  # noqa: E402
from ocudu_cuda_isac.sensing import pilot_ls as reference  # noqa: E402
from ocudu_cuda_isac.sensing.pilot_ls import PilotLsError  # noqa: E402
import replay_pusch_double as receiver  # noqa: E402

USAGE = ("Usage: pilot_ls --plan FILE --output-dir DIR [--only OPPORTUNITY_ID]... [--no-grids]\n"
         "                [--phy-debug]\n"
         "       pilot_ls --self-test\n"
         "Double of the native tool: it writes the same files with a model receiver.\n")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def faulty_estimate(source, grid, fault):
    """The estimate of the reference, or one with the named fault."""
    source = copy.copy(source)
    if fault == "stored-pilot":
        source.reference = [complex(grants.bf16(value.real), grants.bf16(value.imag)) for value in source.reference]
    elif fault == "conjugate":
        source.reference = [value.conjugate() for value in source.reference]
    read = grid
    if fault == "shifted":
        def read(plane, symbol, subcarrier):
            return grid(plane, symbol, min(subcarrier + 1, source.grid_subcarriers - 1))
    elif fault == "data-dependent":
        def read(plane, symbol, subcarrier):
            return grid(plane, symbol, subcarrier) + reference.single(0.001) * grid(plane, symbol + 1, subcarrier)
    made = reference.estimate(source, read)
    if fault == "swap-planes":
        size = len(made.values) // source.rx_planes
        values = [value for plane in reversed(range(source.rx_planes)) for value in made.values[plane * size:(plane + 1) * size]]
        made = reference.Estimate(source, values, made.mask, made.valid)
    elif fault == "mask-clear":
        mask = bytearray(made.mask)
        mask[0] &= 0xFE
        made = reference.Estimate(source, made.values, mask, made.valid - 1)
    return made


def run(plan_path, output, only, keep_grids, fault):
    raw = Path(plan_path).read_bytes()
    try:
        plan = json.loads(raw)
    except ValueError as error:
        raise PilotLsPlanError(f"the plan is not valid JSON: {error}") from None
    if fault == "accept-all":
        replay_plan = json.loads(Path(plan["replay_plan"]["path"]).read_text())
        sources = {}
        for item in plan["sources"]:
            with contextlib.suppress(PilotLsPlanError, PilotLsError, KeyError):
                sources[item["record_sha256"]] = pl.load_source(item)
    else:
        replay_plan, _, sources = pl.check_files(plan)
    tap = plan["observer"]["tap"]
    tapped = tap == "pilot_ls" or fault == "tap-when-off"
    if fault == "tap-when-off" and not sources:
        # The sources of the control plan's replay, as a tapped plan would bind them.
        built = pl.build_plan("fault", plan["replay_plan"]["path"])
        sources = {item["record_sha256"]: pl.load_source(item) for item in built["sources"]}

    # The receiver side, by the double of Sprint 2.4 on the replay plan. The grids are always kept
    # here, because the estimate is computed from them.
    inner = "wrong-byte" if fault == "changed-decode" and tap == "pilot_ls" else \
        fault if fault in ("accept-all", "host-fallback") else ""
    if os.environ.get("PILOT_LS_BUILD", "") == "host":
        os.environ["REPLAY_PUSCH_BUILD"] = "host"
    with contextlib.redirect_stdout(io.StringIO()):
        status = receiver.run(plan["replay_plan"]["path"], output, only, True, inner)
    report = json.loads((output / "native.json").read_text())
    records = [json.loads(line) for line in (output / "radio" / "radio.jsonl").read_text().splitlines() if line.strip()]
    descriptors = [json.loads(line) for line in (output / "ingress" / "ingress.jsonl").read_text().splitlines()
                   if line.strip()]
    by_sequence = {descriptor["sequence"]: descriptor for descriptor in descriptors}
    record_of = {grant["grant_id"]: grants.load(grant["record"]["path"]) for grant in replay_plan["grants"]}

    lines, clock, valid, holes, size = [], reference.SlotClock(), 0, 0, 0
    if tapped:
        (output / pl.OBSERVER_DIR).mkdir()
    for entry in report["opportunities"]:
        entry["observer"] = None
        if not tapped:
            continue
        sequence, descriptor = entry["sequence"], by_sequence[entry["sequence"]]
        path = output / "ingress" / descriptor["grid"]["file"]
        data = path.read_bytes()
        if fault == "touch-grid" and not lines:
            # A tap that writes: the grid the processors read is no longer the one that was made ready.
            data = struct.pack("<2f", 0.5, -0.5) + data[8:]
            path.write_bytes(data)
            digest = sha256(data)
            descriptor["grid"]["sha256"] = entry["readback_sha256"] = digest
            for record in records:
                if record["opportunity_sequence"] == sequence and record["ingress"]:
                    record["ingress"]["readback_sha256"] = digest
        planes, symbols, subcarriers = descriptor["grid"]["shape"]
        source = sources[descriptor["grant_record_sha256"]]
        started = time.monotonic_ns()
        made = faulty_estimate(source, reference.stored_grid(data, planes, symbols, subcarriers), fault)
        grant = record_of[descriptor["grant_id"]]
        payload_file, mask_file = f"{sequence:04d}.o2.c64", f"{sequence:04d}.o2.mask"
        (output / pl.OBSERVER_DIR / payload_file).write_bytes(made.payload())
        (output / pl.OBSERVER_DIR / mask_file).write_bytes(made.mask)
        absolute = clock.absolute_slot(entry["realization_id"], source.scs_khz, entry["sfn"], entry["slot"])
        digest = descriptor["grid"]["sha256"]
        record = reference.describe(source, made, {
            "run_id": plan["run_id"], "epoch": plan["observer"]["epoch"], "sequence": len(lines),
            "opportunity_id": entry["opportunity_id"], "opportunity_sequence": sequence,
            "episode_id": entry["realization_id"], "grant_id": descriptor["grant_id"],
            "grant_record_sha256": descriptor["grant_record_sha256"], "cell": grant.n_cell_id, "ue": grant.rnti,
            "sfn": entry["sfn"], "slot": entry["slot"], "absolute_slot": absolute + (fault == "wrong-slot"),
            "ingress": {"grid_index": descriptor["grid_index"], "generation": descriptor["generation"],
                        "readback_sha256": "0" * 64 if fault == "other-grid" and len(lines) == 1 else digest},
            "payload_file": payload_file, "mask_file": mask_file, "t_ready_ns": time.monotonic_ns(),
            "service_us": (time.monotonic_ns() - started) / 1000.0, "host": "double of the staged harness"})
        if fault == "leak":
            record["host"]["tb_sha256"] = sha256(grant.tb)
        entry["observer"] = {"sequence": len(lines), "payload_sha256": made.payload_sha256,
                             "mask_sha256": made.mask_sha256, "ingress_sha256": record["host"]["ingress"]["readback_sha256"],
                             "valid": made.valid, "holes": made.holes, "service_us": record["host"]["service_us"]}
        lines.append(record)
        valid, holes, size = valid + made.valid, holes + made.holes, size + len(made.payload())
    if tapped:
        kept = lines[:-1] if fault == "drop-observation" else lines
        (output / pl.OBSERVER_DIR / pl.RECORDS_FILE).write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in kept), encoding="ascii")

    # The radio records and the report of this run: its own identifier and its tap.
    for record in records:
        record["run_id"], record["tap"] = plan["run_id"], tap
    (output / "radio" / "radio.jsonl").write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
                                                 encoding="ascii")
    if not keep_grids:
        for descriptor in descriptors:
            (output / "ingress" / descriptor["grid"]["file"]).unlink()
            descriptor["grid"]["file"] = None
        report["grid"]["stored_grids_kept"] = False
    (output / "ingress" / "ingress.jsonl").write_text(
        "".join(json.dumps(descriptor, sort_keys=True) + "\n" for descriptor in descriptors), encoding="ascii")
    on = tap == "pilot_ls"
    times = [record["host"]["service_us"] for record in lines[1:]]
    stats = {"count": len(times), "excluded": "the first observation of the run"}
    if times:
        stats.update(mean=sum(times) / len(times), min=min(times), max=max(times))
    report.update({
        "schema": pl.NATIVE_SCHEMA, "run_id": plan["run_id"], "plan": {"file": str(plan_path), "sha256": sha256(raw)},
        "replay_plan": {"file": plan["replay_plan"]["path"], "sha256": plan["replay_plan"]["sha256"],
                        "run_id": replay_plan["run_id"]},
        "observer": {"tap": tap, "stage": plan["observer"]["stage"], "source": plan["observer"]["source"],
                     "layout": plan["observer"]["layout"], "epoch": plan["observer"]["epoch"],
                     "producer": reference.PRODUCER if on else None, "rule": reference.RULE if on else None,
                     "precision": reference.PRECISION if on else None,
                     "memory": "buffer of the double" if on else None, "records": len(lines), "valid": valid,
                     "holes": holes, "payload_bytes": size, "service_us": stats,
                     "sources": [{"grant_id": grant["grant_id"], "record_sha256": grant["record_sha256"],
                                  "source": sources[grant["record_sha256"]].name,
                                  "port": sources[grant["record_sha256"]].port,
                                  "pilots": len(sources[grant["record_sha256"]].index),
                                  "symbols": sources[grant["record_sha256"]].symbols,
                                  "first_subcarrier": sources[grant["record_sha256"]].subcarriers[0],
                                  "subcarrier_step": sources[grant["record_sha256"]].subcarrier_step,
                                  "subcarrier_count": len(sources[grant["record_sha256"]].subcarriers),
                                  "holes": sources[grant["record_sha256"]].holes,
                                  "coordinates_sha256": sources[grant["record_sha256"]].coordinates_sha256,
                                  "amplitude": sources[grant["record_sha256"]].amplitude,
                                  "antenna_map": sources[grant["record_sha256"]].antenna_map}
                                 for grant in replay_plan["grants"] if on and grant["record_sha256"] in sources]}})
    report["receiver"]["tap"] = tap
    report["totals"]["observations"] = len(lines)
    (output / "native.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(f"pilot_ls double: {plan['run_id']}, tap {tap}, {len(lines)} observations, "
          f"{len(report['opportunities'])} opportunities")
    return status


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    fault = os.environ.get("PILOT_LS_FAULT", "")
    if "--help" in argv:
        print(USAGE, end="")
        return 0
    if "--self-test" in argv:
        print("pilot_ls self-test: all checks passed")
        return 0
    keep_grids = "--no-grids" not in argv
    words = [word for word in argv if word not in ("--no-grids", "--phy-debug")]
    only = [words[index + 1] for index in range(0, len(words) - 1, 2) if words[index] == "--only"]
    options = {words[index]: words[index + 1] for index in range(0, len(words) - 1, 2) if words[index] != "--only"}
    try:
        if len(words) % 2 or set(options) != {"--plan", "--output-dir"}:
            raise PilotLsPlanError("--plan and --output-dir are required")
        output = Path(options["--output-dir"])
        if not output.is_dir():
            raise PilotLsPlanError(f"the output directory {output} does not exist")
        if any((output / name).exists() for name in ("native.json", "ingress", "radio", pl.OBSERVER_DIR)):
            raise PilotLsPlanError("the output directory already holds a result. Evidence is never overwritten.")
        status = run(options["--plan"], output, only, keep_grids, fault)
    except (PilotLsPlanError, PilotLsError, ReplayPuschError, GridApplyError, GrantError, OSError, KeyError,
            TypeError) as error:
        print(f"pilot_ls: input error: {error}", file=sys.stderr)
        return 2
    return 3 if fault == "exit" else status


if __name__ == "__main__":
    sys.exit(main())
