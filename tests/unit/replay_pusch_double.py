#!/usr/bin/env python3
"""Doubles of the native replay_pusch tool and of the fixture adapter, for hosts without an OCUDU build.

The tool double reads a replay-pusch-plan/v1 file and writes what
native/ocudu_adapter/receiver/replay_pusch.cpp writes: ingress/ with one o1-grid/v1 descriptor and
one stored grid per opportunity, radio/ with one radio/v1 record per planned decode and the decoded
bytes of every CRC pass, and native.json. The grid is the reference of
ocudu_cuda_isac.episodes.grid_apply. The receiver is a model, not a decoder: it estimates the
channel on the pilots of the grant, equalizes the data elements and reports a CRC pass when they
agree with the transmit grid of the plan. Then it returns the transport block of the grant.

The unit tests use it to check the reader of the native output and every check of the integration
driver. REPLAY_PUSCH_FAULT names one fault to inject, so that each check can be shown to fail:

    wrong-byte       the CUDA processor reports a valid CRC with one wrong byte
    forged-match     the tool claims a payload match for those bytes
    stale-bytes      the CUDA processor returns the bytes of the opportunity before
    forced-ack       a CRC failure is reported as a CRC pass
    host-fallback    the CUDA processor executes the host path
    changed-grid     the receive planes are written in reverse order
    other-grid       a record names another grid digest than its opportunity was filled with
    drop-record      the last radio record is missing
    leak             a grid descriptor names the hash of a transport block
    leak-file        ingress/ holds a copy of the plan
    anomaly          the report lists a second completion
    timeout          the CUDA processor does not complete the second opportunity
    accept-all       no plan is refused
    exit             the tool exits with status 3 after writing

REPLAY_PUSCH_BUILD=host makes it the tool of a build without CUDA: every decode requested from the
CUDA processor executes the host path and says why.

fixture_main is the double of grant_fixture: the pilots of the grant and data symbols that depend
on its transport block, so that a block per opportunity gives a transmit grid per opportunity.
ADAPTER_FAULT names one fault of it: pilot (one pilot is not the grant's), other-pdu (the PDU it
echoes has another RNTI) or exit (status 3 after writing).

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
from ocudu_cuda_isac.episodes import replay_pusch as rp  # noqa: E402
from ocudu_cuda_isac.episodes.grant import GrantError  # noqa: E402
from ocudu_cuda_isac.episodes.grid_apply import GridApplyError  # noqa: E402
from ocudu_cuda_isac.episodes.replay_pusch import ReplayPuschError  # noqa: E402

STORAGE = "complex bf16 in the double of the pinned resource grid, widened without change"
LEVELS = (0.31640625, 0.94921875)  # the two stored amplitudes of a 16QAM component
# The reported SINR of the pinned host receiver on a noiseless grid: 96.99 dB on one port.
SINR_FLOOR_DB = 96.9897
EVM_LIMIT = 0.35
# The filled grid and the model's answer for one opportunity, by the hashes that the plan binds. The
# driver tests fill the same grids many times, and the reference arithmetic is the slow part.
_FILLED = {}
USAGE = ("Usage: replay_pusch --plan FILE --output-dir DIR [--only OPPORTUNITY_ID]... [--no-grids]\n"
         "                    [--phy-debug]\n"
         "       replay_pusch --self-test\n"
         "Double of the native tool: it writes the same files with a model receiver.\n")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def data_symbol(block, number):
    """The data symbol of resource element number for a transport block: one of sixteen stored points."""
    nibble = block[number % len(block)] >> (4 * (number // len(block) % 2)) & 15
    return complex(LEVELS[nibble & 1] * (1 if nibble & 2 else -1), LEVELS[nibble >> 2 & 1] * (1 if nibble & 8 else -1))


def transmit_grid(grant):
    """The transmit grid the fixture double writes: stored pilots and block-dependent data symbols."""
    width = grant.grid_subcarriers
    x = [0j] * (grants.SYMBOLS_PER_SLOT * width)
    for (symbol, k), value in zip(grant.pilot_index, grant.pilot_reference[0]):
        x[symbol * width + k] = complex(grants.bf16(value.real), grants.bf16(value.imag))
    for number, (symbol, k) in enumerate(grant.data_index):
        x[symbol * width + k] = data_symbol(grant.tb, number)
    return x


def fixture_main(argv=None):
    """Double of grant_fixture: x_native.c64 and native.json for one grant record."""
    argv = list(sys.argv[1:] if argv is None else argv)
    loopback = "--no-loopback" not in argv
    words = [word for word in argv if word != "--no-loopback"]
    args = dict(zip(words[::2], words[1::2]))
    grant = grants.load(args["--grant"])
    x = transmit_grid(grant)
    fault = os.environ.get("ADAPTER_FAULT", "")
    if fault == "pilot":
        # A pilot that is not the grant's: the reference check of the store refuses the grid.
        symbol, k = grant.pilot_index[0]
        x[symbol * grant.grid_subcarriers + k] = -x[symbol * grant.grid_subcarriers + k]
    out = Path(args["--output-dir"])
    (out / "x_native.c64").write_bytes(replay.grid_bytes(x))
    pdu = {"scs_khz": grant.scs_khz, "sfn": grant.sfn, "slot": grant.slot, "rnti": grant.rnti, "n_id": grant.n_id,
           "bwp_start_rb": grant.bwp_start_rb, "bwp_size_rb": grant.bwp_size_rb, "prb_start": grant.prb_start,
           "prb_count": grant.prb_count, "symbol_start": grant.symbol_start, "symbol_count": grant.symbol_count,
           "layers": grant.layers, "rv": 0, "dmrs_config_type": grant.dmrs["config_type"], "dmrs_symbols": grant.dmrs["symbols"],
           "dmrs_scrambling_id": grant.dmrs["scrambling_id"], "dmrs_n_scid": grant.dmrs["n_scid"],
           "cdm_groups_without_data": grant.dmrs["cdm_groups_without_data"], "rx_ports": grant.rx_ports}
    if fault == "other-pdu":
        pdu["rnti"] += 1
    report = {"schema": "grant-fixture-native/v1", "grant_id": grant.grant_id, "dmrs_reference": "crb0", "pdu": pdu,
              "grid": {"storage": STORAGE}, "built_with_cuda": False,
              "loopback": {"executed": True, "tb_crc_ok": True, "bytes_equal": True, "bytes_compared": len(grant.tb)}
              if loopback else {"executed": False}}
    (out / "native.json").write_text(json.dumps(report))
    return 3 if fault == "exit" else 0


# ---------------------------------------------------------------- model receiver
def model_receiver(grant, x, stored, rx):
    """What the model receiver reports for one stored grid: CRC flag, EPRE, SINR and iterations.

    Pilot LS averaged over the DM-RS symbols, the estimate of the nearest pilot subcarrier for the
    data elements, maximum-ratio combining over the receive planes. The CRC passes when the
    equalized data symbols agree with the transmit grid after the best common gain.
    """
    width, plane = grant.grid_subcarriers, grants.SYMBOLS_PER_SLOT * grant.grid_subcarriers
    estimate, power, pilots = {}, 0.0, len(grant.pilot_index)
    for (symbol, k), value in zip(grant.pilot_index, grant.pilot_reference[0]):
        for port in range(rx):
            received = stored[port * plane + symbol * width + k]
            total, count = estimate.get((port, k), (0j, 0))
            estimate[(port, k)] = (total + received / value, count + 1)
            power += abs(received) ** 2
    channel = {key: total / count for key, (total, count) in estimate.items()}
    epre = 10 * math.log10(power / (pilots * rx)) if power > 0 else None
    cross, energy, pairs = 0j, 0.0, []
    for symbol, k in grant.data_index:
        pilot = k - (k - grant.subcarrier_start) % 2
        weights = [channel[(port, pilot)] for port in range(rx)]
        gain = sum(abs(weight) ** 2 for weight in weights)
        if gain == 0.0:
            return {"crc_ok": False, "epre_db": epre, "sinr_db": None, "iterations": None}
        estimate_x = sum(weight.conjugate() * stored[port * plane + symbol * width + k]
                         for port, weight in enumerate(weights)) / gain
        sent = x[symbol * width + k]
        cross += estimate_x * sent.conjugate()
        energy += abs(sent) ** 2
        pairs.append((estimate_x, sent))
    common = cross / energy
    error = sum(abs(value - common * sent) ** 2 for value, sent in pairs)
    evm = math.sqrt(error / (abs(common) ** 2 * energy)) if abs(common) > 0 else math.inf
    floor = SINR_FLOOR_DB + 10 * math.log10(rx)
    sinr = floor if evm == 0 else min(floor, -20 * math.log10(evm)) if math.isfinite(evm) else None
    crc_ok = common.real > 0.5 and evm < EVM_LIMIT
    return {"crc_ok": crc_ok, "epre_db": epre, "sinr_db": sinr, "iterations": 1.0 if crc_ok else None}


def load_complex(path):
    _, _, flat = grants.read_npy(path)
    return flat


def load_index(path):
    _, _, flat = grants.read_npy(path)
    return [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]


def run(plan_path, output, only, keep_grids, fault):
    with_cuda = os.environ.get("REPLAY_PUSCH_BUILD", "") != "host"
    raw = Path(plan_path).read_bytes()
    try:
        plan = json.loads(raw)
    except ValueError as error:
        raise ReplayPuschError(f"the plan is not valid JSON: {error}") from None
    if fault == "accept-all":
        channel = json.loads(Path(plan["channel_plan"]["path"]).read_text())
    else:
        channel = rp.check_files(plan)
    carrier = channel["carrier"]
    rx, tx, symbols, subcarriers = (carrier[key] for key in replay.CARRIER_FIELDS)
    shape, plane = (rx, tx, symbols, subcarriers), symbols * subcarriers
    gain, backends = channel["gain"]["value"], plan["receiver"]["backends"]
    identifiers = [item["opportunity_id"] for item in channel["opportunities"]]
    for name in only:
        if name not in identifiers:
            raise ReplayPuschError(f"--only names the opportunity {name}, which the plan does not hold")
    (output / "ingress").mkdir()
    (output / "radio").mkdir()

    files, grants_of = [], {}
    listed = {item["grant_id"]: item for item in plan["grants"]}
    for item in channel["grants"]:
        entry = listed[item["grant_id"]]
        grant = grants.load(entry["record"]["path"])
        active = sorted(symbol * subcarriers + k for index in item["active"] for symbol, k in load_index(index["path"]))
        grants_of[item["grant_id"]] = {"grant": grant, "plan": item, "active": active,
                                       "active_sha256": sha256(struct.pack(f"<{len(active)}I", *active)),
                                       "block": Path(entry["tb"]["path"]).read_bytes()}
        for role, ref in (("the transmit grid", item["x"]), ("the record", entry["record"]),
                          ("the transport block", entry["tb"])):
            files.append({"role": f"{role} of grant {item['grant_id']}", "path": ref["path"], "sha256": ref["sha256"],
                          "bytes": Path(ref["path"]).stat().st_size})
    sources, loaded = {}, {}

    def array(ref):
        if ref["path"] not in loaded:
            loaded[ref["path"]] = load_complex(ref["path"])
        return loaded[ref["path"]]

    for item in channel["sources"]:
        sources[item["realization_id"]] = item
        for role, ref in (("the channel", item["h"]), ("the noise", item["w"])):
            files.append({"role": f"{role} of realization {item['realization_id']}", "path": ref["path"],
                          "sha256": ref["sha256"], "bytes": Path(ref["path"]).stat().st_size})

    opportunities, ingress_lines, records, anomalies = [], [], [], []
    generations, used, stopped, previous_block = [0, 0], set(), {}, None
    for sequence, item in enumerate(channel["opportunities"]):
        if only and item["opportunity_id"] not in only:
            continue
        bound = grants_of[item["grant_id"]]
        grant = bound["grant"]
        source = sources[item["realization_id"]]
        known = (shape, gain, bound["plan"]["x"]["sha256"], source["h"]["sha256"], source["w"]["sha256"], item["index"],
                 bound["plan"]["record_sha256"])
        if fault == "accept-all" or known not in _FILLED:
            x, h_all, w_all = array(bound["plan"]["x"]), array(source["h"]), array(source["w"])
            h = h_all[item["index"] * rx * tx * plane:(item["index"] + 1) * rx * tx * plane]
            w = w_all[item["index"] * rx * plane:(item["index"] + 1) * rx * plane]
            stored = replay.stored(replay.apply(shape, x, h, w, gain))
            _FILLED[known] = (replay.grid_bytes(stored), model_receiver(grant, x, stored, rx))
        grid_bytes, model = _FILLED[known]
        if fault == "changed-grid":
            grid_bytes = b"".join(grid_bytes[8 * plane * port:8 * plane * (port + 1)] for port in reversed(range(rx)))
        digest = sha256(grid_bytes)
        name = f"{sequence:04d}.c64"
        if keep_grids:
            (output / "ingress" / name).write_bytes(grid_bytes)
        grid_index = len(opportunities) % 2
        generations[grid_index] += 1
        descriptor = {"schema": replay.INGRESS_SCHEMA, "run_id": channel["run_id"], "sequence": sequence,
                      "opportunity_id": item["opportunity_id"], "sfn": item["sfn"], "slot": item["slot"],
                      "grant_id": item["grant_id"], "grant_record_sha256": bound["plan"]["record_sha256"],
                      "active_re": len(bound["active"]), "active_sha256": bound["active_sha256"],
                      "grid_index": grid_index, "generation": generations[grid_index], "state": "ready",
                      "grid": {"file": name if keep_grids else None, "axes": replay.GRID_AXES,
                               "shape": [rx, symbols, subcarriers], "dtype": "<c8", "order": "C", "storage": STORAGE,
                               "sha256": digest},
                      "amplitude": {"grid_gain": gain, "convention": replay.RULE}}
        if fault == "leak":
            descriptor["tb_sha256"] = sha256(bound["block"])
        ingress_lines.append(json.dumps(descriptor, sort_keys=True))
        entry = {"sequence": sequence, "opportunity_id": item["opportunity_id"], "realization_id": item["realization_id"],
                 "index": item["index"], "grant_id": item["grant_id"], "sfn": item["sfn"], "slot": item["slot"],
                 "grid_index": grid_index, "generation": generations[grid_index], "readback_sha256": digest,
                 "fill_us": 1000.0, "retained": True, "released": True, "decodes": []}
        for backend in backends:
            key = f"{item['opportunity_id']}/{backend}"
            first_use = backend not in used
            crc_ok, block, error, completed = model["crc_ok"], bound["block"], None, True
            resident = backend == "cuda" and with_cuda and fault != "host-fallback" and rx in rp.RESIDENT_PORTS
            if backend in stopped:
                outcome, crc_ok, completed, error = "not_dispatched", False, False, stopped[backend]
            else:
                used.add(backend)
                if fault == "timeout" and backend == "cuda" and len(opportunities) == 1:
                    completed, crc_ok, error = False, False, "no completion within the declared time"
                    stopped[backend] = f"the cuda processor did not complete {item['opportunity_id']} within 30000 ms"
                    entry["released"] = False
                if fault == "forced-ack" and not crc_ok and completed:
                    crc_ok, block = True, bytes(len(block))
                if fault in ("wrong-byte", "forged-match") and backend == "cuda" and crc_ok:
                    block = block[:-1] + bytes([block[-1] ^ 1])
                if fault == "stale-bytes" and backend == "cuda" and crc_ok and previous_block is not None \
                        and len(previous_block) == len(block):
                    block = previous_block
                equal = crc_ok and block == bound["block"]
                outcome = rp.classify(completed, False, crc_ok, equal or fault == "forged-match")
            match = outcome == "decoded"
            decoded_file = f"{sequence:04d}.{backend}.bin" if crc_ok else None
            if crc_ok:
                (output / "radio" / decoded_file).write_bytes(block)
            executed = None if not completed else "resident" if resident else "host"
            path = {"requested": backend, "executed": executed, "fallback": None}
            if executed == "resident":
                path.update(device_grid=False, owned_device_grid=False)
            elif executed == "host" and backend == "cuda":
                path["fallback"] = "the adapter was built without CUDA" if not with_cuda else \
                    "the double was asked to fall back" if rx in rp.RESIDENT_PORTS else \
                    "the resident path takes 1, 2, 4 or 8 receive ports"
            decoder = None
            if completed:
                decoder = {"codeblocks": 1}
                if model["iterations"] is not None:
                    decoder["ldpc_iterations_mean"] = model["iterations"]
                if model["epre_db"] is not None:
                    decoder["epre_db"] = model["epre_db"]
                if model["sinr_db"] is not None:
                    decoder["sinr_db"] = model["sinr_db"]
            service = 200.0 + 10.0 * len(records) if completed else None
            ingress = {"grid_index": grid_index, "generation": generations[grid_index],
                       "readback_sha256": "0" * 64 if fault == "other-grid" and len(records) == 1 else digest}
            if completed:
                ingress["intact"] = True
            records.append({
                "schema": rp.RADIO_SCHEMA, "run_id": plan["run_id"], "sequence": len(records), "key": key,
                "opportunity_id": item["opportunity_id"], "opportunity_sequence": sequence, "backend": backend,
                "sfn": item["sfn"], "slot": item["slot"], "grant_id": item["grant_id"],
                "grant_record_sha256": bound["plan"]["record_sha256"],
                "allocation": {"crb_start": grant.crb_start, "prb_count": grant.prb_count,
                               "symbol_start": grant.symbol_start, "symbol_count": grant.symbol_count,
                               "dmrs_symbols": list(grant.dmrs["symbols"]), "rx_ports": list(grant.rx_ports),
                               "layers": grant.layers, "active_re": len(bound["active"]),
                               "active_sha256": bound["active_sha256"]},
                "ingress": ingress,
                "tb": {"id": item["grant_id"], "bytes": len(bound["block"]), "bits": 8 * len(bound["block"])},
                "outcome": outcome, "crc_ok": crc_ok, "payload_match": match,
                "decoded_sha256": sha256(block) if crc_ok else None, "decoded_file": decoded_file, "error": error,
                "path": path, "decoder": decoder,
                "timing": {"service_us": service, "first_use": first_use, "timeout_ms": plan["receiver"]["timeout_ms"]},
                "retry": 0, "tap": plan["receiver"]["tap"]})
            entry["decodes"].append({
                "backend": backend, "key": key, "outcome": outcome, "crc_ok": crc_ok, "payload_match": match,
                "first_differing_byte": (-1 if match else len(block) - 1) if crc_ok else None,
                "bytes_compared": len(block) if crc_ok else 0, "decoded_sha256": sha256(block) if crc_ok else None,
                "original_sha256": sha256(bound["block"]), "guard_intact": True if completed else None,
                "input_intact": True if completed else None, "completions": 1 if completed else 0,
                "executed": executed, "service_us": service})
        previous_block = bound["block"]
        opportunities.append(entry)
    if fault == "anomaly":
        anomalies.append({"kind": "duplicate_completion", "key": records[0]["key"],
                          "detail": "a second completion arrived. The first one stands."})
    kept = records[:-1] if fault == "drop-record" else records
    if fault == "leak-file":
        (output / "ingress" / "plan.json").write_bytes(raw)
    (output / "ingress" / "ingress.jsonl").write_text("\n".join(ingress_lines) + "\n", encoding="ascii")
    (output / "radio" / "radio.jsonl").write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in kept),
                                                 encoding="ascii")
    backend_list = []
    for number, backend in enumerate(backends):
        mine = [record for record in records if record["backend"] == backend]
        times = [record["timing"]["service_us"] for record in mine
                 if record["timing"]["service_us"] is not None and not record["timing"]["first_use"]]
        stats = {"count": len(times), "excluded": "the first decode of the backend, which includes its set-up"}
        if times:
            stats.update(mean=sum(times) / len(times), min=min(times), max=max(times))
        backend_list.append({
            "name": backend, "factory": "gpu" if backend == "cuda" else "auto", "harq_id": number,
            "acceleration_requested": backend == "cuda",
            "acceleration_available": with_cuda if backend == "cuda" else None,
            "planned": len(mine), "outcomes": {name: sum(record["outcome"] == name for record in mine)
                                               for name in rp.OUTCOMES},
            "resident": sum(record["path"]["executed"] == "resident" for record in mine),
            "host": sum(record["path"]["executed"] == "host" for record in mine),
            "stopped": stopped.get(backend), "service_us": stats})
    gain_record = {"value": gain, "source": channel["gain"]["source"], "convention": replay.RULE}
    for key in ("reference_power", "reason"):
        if channel["gain"].get(key) is not None:
            gain_record[key] = channel["gain"][key]
    report = {
        "schema": rp.NATIVE_SCHEMA, "run_id": plan["run_id"], "plan": {"file": str(plan_path), "sha256": sha256(raw)},
        "channel_plan": {"file": plan["channel_plan"]["path"], "sha256": plan["channel_plan"]["sha256"],
                         "run_id": channel["run_id"]},
        "selected": sorted(only) if only else None, "payload": plan["payload"],
        "receiver": dict(plan["receiver"], time_interpolation="average", equalizer="mmse", worker_threads=1,
                         factories="double of the pinned PUSCH processors", dmrs_reference="CRB 0",
                         grid_memory="host resource grid"),
        "backends": backend_list, "carrier": dict(carrier), "gain": gain_record,
        "grid": {"factory": "double of the pinned resource grid", "grids": 2, "ports": rx, "symbols": symbols,
                 "subcarriers": subcarriers, "storage": STORAGE, "stored_grids_kept": keep_grids,
                 "little_endian_host": sys.byteorder == "little"},
        "files": files,
        "grants": [{"grant_id": name, "record_sha256": item["plan"]["record_sha256"], "slot": item["grant"].slot,
                    "tb_sha256": sha256(item["block"]), "tb_bytes": len(item["block"])} for name, item in grants_of.items()],
        "opportunities": opportunities, "anomalies": anomalies,
        "totals": {"opportunities": len(opportunities), "backends": len(backends),
                   "planned": len(opportunities) * len(backends), "records": len(records),
                   "outcomes": {name: sum(record["outcome"] == name for record in records) for name in rp.OUTCOMES},
                   "anomalies": len(anomalies), "retained": True,
                   "grids_not_released": sum(not entry["released"] for entry in opportunities),
                   "generations": generations},
        "environment": {}, "access": {"transport_block": "read by the evaluator side of the double"},
        "built_with_cuda": with_cuda, "double": True}
    (output / "native.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(f"replay_pusch double: {plan['run_id']}, {len(opportunities)} opportunities, payload {plan['payload']['rule']}")
    return 1 if stopped else 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    fault = os.environ.get("REPLAY_PUSCH_FAULT", "")
    if "--help" in argv:
        print(USAGE, end="")
        return 0
    if "--self-test" in argv:
        print("replay_pusch self-test: all checks passed")
        return 0
    keep_grids = "--no-grids" not in argv
    words = [word for word in argv if word not in ("--no-grids", "--phy-debug")]
    only = [words[index + 1] for index in range(0, len(words) - 1, 2) if words[index] == "--only"]
    options = {words[index]: words[index + 1] for index in range(0, len(words) - 1, 2) if words[index] != "--only"}
    try:
        if len(words) % 2 or set(options) != {"--plan", "--output-dir"}:
            raise ReplayPuschError("--plan and --output-dir are required")
        output = Path(options["--output-dir"])
        if not output.is_dir():
            raise ReplayPuschError(f"the output directory {output} does not exist")
        if any((output / name).exists() for name in ("native.json", "ingress", "radio")):
            raise ReplayPuschError("the output directory already holds a result. Evidence is never overwritten.")
        status = run(options["--plan"], output, only, keep_grids, fault)
    except (ReplayPuschError, GridApplyError, GrantError, OSError, KeyError, TypeError) as error:
        print(f"replay_pusch: input error: {error}", file=sys.stderr)
        return 2
    return 3 if fault == "exit" else status


if __name__ == "__main__":
    sys.exit(fixture_main() if Path(sys.argv[0]).name == "grant_fixture" else main())
