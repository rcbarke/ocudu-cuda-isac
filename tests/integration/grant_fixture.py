#!/usr/bin/env python3
"""Sprint 2.1 integration driver (selector grant_fixture). Hardware command, not part of unit discovery.

For every grant of the configuration it feeds the same known transport block to both transmit
references, the pinned OCUDU chain (through the native fixture adapter built against the accepted
Sprint 1.1 build) and Sionna NR, aligns them and compares: transport block size, coding, the exact
set of occupied resource elements, every pilot and every data symbol to the declared storage
tolerance, gain and phase. It then archives the grant, the bytes and both grids in a store entry
and reads a pilot, a data symbol and the block hash back from that entry.

    python tests/integration/grant_fixture.py --config configs/integration/grant_fixture.json \\
        --output-dir results/runs/s21-spark-001

The output directory must be new. The exit status is 0 only when every check of every case passed.
"""

import argparse
import cmath
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import struct
import subprocess
import sys
import traceback
import warnings

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from ocudu_cuda_isac.episodes import grant as grants  # noqa: E402
from ocudu_cuda_isac.episodes import sionna_pusch  # noqa: E402
from ocudu_cuda_isac.episodes.grant import GrantError  # noqa: E402
from ocudu_cuda_isac.replay import ContractError, load_json  # noqa: E402
import native_adapter  # noqa: E402

CONFIG_SCHEMA = "integration-config/grant_fixture/v1"
GRADE_SCHEMA = "integration-grade/grant_fixture/v1"
NATIVE_SCHEMA = "grant-fixture-native/v1"
ADAPTER_TIMEOUT_S = 600
PINNED_SETTINGS = (None, "differs", "identical")


class ConfigError(ValueError):
    """The configuration or the run setup is unusable. Nothing was graded."""


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii")


def load_config(path):
    try:
        config = load_json(Path(path))
    except (ContractError, OSError) as error:
        raise ConfigError(f"cannot read the configuration: {error}") from None
    if not isinstance(config, dict) or set(config) != {"schema", "selector", "adapter_build", "tolerance", "cases"} \
            or config["schema"] != CONFIG_SCHEMA or config["selector"] != "grant_fixture":
        raise ConfigError(f"the configuration must be a {CONFIG_SCHEMA} object for the selector grant_fixture")
    tolerance = config["tolerance"]
    needed = ("storage", "relative", "absolute", "alignment_gain", "alignment_phase_rad", "reference_absolute")
    if not isinstance(tolerance, dict) or set(tolerance) != set(needed):
        raise ConfigError(f"tolerance requires exactly {', '.join(needed)}")
    # The store checks every grid with the module's declared tolerance. A second value would
    # let the comparison and the archive disagree.
    declared = grants.STORAGE
    if (tolerance["storage"], tolerance["relative"], tolerance["absolute"]) != \
            (declared["name"], declared["relative"], declared["absolute"]):
        raise ConfigError(f"the storage tolerance must be the declared one: {declared}")
    cases = config["cases"]
    if not isinstance(cases, list) or not cases:
        raise ConfigError("the configuration names no case")
    for case in cases:
        if not isinstance(case, dict) or not {"grant", "expect"} <= set(case) \
                or set(case) - {"grant", "expect", "pinned_dmrs_reference", "rehearsal_sha256"} \
                or case.get("pinned_dmrs_reference") not in PINNED_SETTINGS \
                or not isinstance(case.get("rehearsal_sha256", {}), dict):
            raise ConfigError("a case holds grant, expect and optionally pinned_dmrs_reference (differs or identical) "
                              "and rehearsal_sha256")
        expect = case["expect"]
        if not isinstance(expect, dict) or set(expect) != {"tbs_bits", "tb_sha256", "pilot", "data"}:
            raise ConfigError("expect requires exactly tbs_bits, tb_sha256, pilot, data")
    return config


def read_grid(path, values):
    """Read the adapter's raw little-endian complex64 grid."""
    data = Path(path).read_bytes()
    if len(data) != 8 * values:
        raise GrantError(f"{Path(path).name} holds {len(data)} bytes, the grant's grid needs {8 * values}")
    parts = struct.unpack(f"<{2 * values}f", data)
    return [complex(parts[index], parts[index + 1]) for index in range(0, 2 * values, 2)]


def alignment(values, references):
    """Least-squares complex gain of values against references: magnitude and phase."""
    power = sum(abs(reference) ** 2 for reference in references)
    gain = sum(value * reference.conjugate() for value, reference in zip(values, references)) / power
    return {"gain": abs(gain), "phase_rad": cmath.phase(gain)}


class Grader:
    """Collects PASS, FAIL and NOTE lines. A case passes only when it has no FAIL line."""

    def __init__(self):
        self.lines, self.results = [], []

    def check(self, case, name, passed, detail):
        self.results.append({"case": case, "check": name, "passed": bool(passed), "detail": detail})
        self._line(("PASS" if passed else "FAIL") + f"  S2.1 {case} {name}: {detail}")
        return bool(passed)

    def note(self, case, text):
        self._line(f"NOTE  S2.1 {case} {text}")

    def _line(self, line):
        self.lines.append(line)
        print(line, flush=True)


def run_adapter(adapter, request, output, reference, log):
    """Run the native fixture adapter on one request and return (status, report or None)."""
    output.mkdir(parents=True)
    command = [str(adapter), "--grant", str(request / "grant.json"), "--tb", str(request / "tb.bin"),
               "--output-dir", str(output), "--dmrs-reference", reference]
    with log.open("w") as stream:
        stream.write(shlex.join(command) + "\n")
        stream.flush()
        try:
            status = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=ADAPTER_TIMEOUT_S,
                                    check=False).returncode
        except (OSError, subprocess.TimeoutExpired) as error:
            stream.write(f"{error}\n")
            return -1, None
    try:
        report = json.loads((output / "native.json").read_text())
    except (OSError, ValueError):
        report = None
    return status, report


def pdu_matches(grant, report):
    """The adapter echoes the values it put into the PDU. They must be the grant's."""
    wanted = {"scs_khz": grant.scs_khz, "sfn": grant.sfn, "slot": grant.slot, "rnti": grant.rnti, "n_id": grant.n_id,
              "bwp_start_rb": grant.bwp_start_rb, "bwp_size_rb": grant.bwp_size_rb, "prb_start": grant.prb_start,
              "prb_count": grant.prb_count, "symbol_start": grant.symbol_start, "symbol_count": grant.symbol_count,
              "layers": grant.layers, "rv": 0, "dmrs_config_type": grant.dmrs["config_type"],
              "dmrs_symbols": grant.dmrs["symbols"], "dmrs_scrambling_id": grant.dmrs["scrambling_id"],
              "dmrs_n_scid": grant.dmrs["n_scid"], "cdm_groups_without_data": grant.dmrs["cdm_groups_without_data"],
              "rx_ports": grant.rx_ports}
    return report.get("pdu") == wanted and report.get("grant_id") == grant.grant_id


def run_case(case, config, adapter, receipt, out, grader, discrepancies, device):
    tolerance = config["tolerance"]
    grant_path = ROOT / case["grant"]
    name = Path(case["grant"]).stem
    measured = {"grant_file": case["grant"], "grant_sha256": digest(grant_path) if grant_path.is_file() else None}

    # ---- The record, validated by the Python module, and the request the adapter reads. ----
    try:
        grant = grants.load(grant_path)
        name = grant.grant_id
        case_dir = out / name
        case_dir.mkdir()
        grants.export(grant, case_dir / "request")
        grants.open_store(case_dir / "request")
    except (GrantError, OSError) as error:
        grader.check(name, "grant record", False, str(error))
        return measured
    derived = grant.derived()
    width = grant.grid_subcarriers
    values = grant.antenna_ports * grants.SYMBOLS_PER_SLOT * width
    grader.check(name, "grant record", True,
                 f"valid {grants.SCHEMA}, {derived['tbs_bits']} bits, CRB {grant.crb_start} to "
                 f"{grant.crb_start + grant.prb_count - 1}, {derived['data_re']} data and {derived['pilot_re']} pilot "
                 f"resource elements (request/)")

    # ---- First transmit reference: the pinned OCUDU chain at the Sprint 1 native boundary. ----
    status, report = run_adapter(adapter, case_dir / "request", case_dir / "native", "crb0", case_dir / "native.log")
    native = None
    if status == 0 and report and report.get("schema") == NATIVE_SCHEMA:
        try:
            native = read_grid(case_dir / "native" / "x_native.c64", values)
        except (GrantError, OSError) as error:
            grader.check(name, "native transmit reference", False, str(error))
    if native is None:
        if status != 0 or not report:
            grader.check(name, "native transmit reference", False,
                         f"the adapter exited with status {status} (see {name}/native.log)")
        return measured
    measured["native"] = {"report": report, "x_native_sha256": digest(case_dir / "native" / "x_native.c64")}
    grader.check(name, "native transmit reference", pdu_matches(grant, report) and report["dmrs_reference"] == "crb0",
                 "the adapter built the grid with the pinned test's transmit chain and echoed the grant's PDU "
                 "(native/native.json)" if pdu_matches(grant, report) else "the adapter's PDU differs from the grant")

    # ---- Second transmit reference: Sionna NR, aligned to carrier coordinates. ----
    sionna, problem = None, None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            sionna = sionna_pusch.reference(grant, device=device)
        except GrantError as error:
            problem = str(error)
        except Exception as error:  # a reference that cannot run is a failed check, with its cause on file
            problem = f"{type(error).__name__}: {error}"
            (case_dir / "sionna-error.txt").write_text(traceback.format_exc())
    if sionna is None:
        grader.check(name, "Sionna transmit reference", False, problem)
        return measured
    x_sionna = sionna.pop("x")
    sionna["warnings"] = sorted({str(item.message) for item in caught})
    write_json(case_dir / "sionna.json", sionna)
    measured["sionna"] = sionna
    grader.check(name, "Sionna transmit reference", len(x_sionna) == values,
                 f"PUSCHTransmitter of Sionna {sionna['versions']['sionna']} on {device}, DM-RS rescaled by "
                 f"{sionna['dmrs_scale_applied']:.6f} to the grant's amplitude (sionna.json)")

    # ---- The comparisons of the card. ----
    native_derived = report["derived"]
    sizes = (derived["tbs_bits"], native_derived.get("tbs_bits"), sionna["tb_size"])
    grader.check(name, "transport block size", len(set(sizes)) == 1 and sizes[0] == case["expect"]["tbs_bits"],
                 f"{sizes[0]} bits in the record, {sizes[1]} in OCUDU, {sizes[2]} in Sionna, "
                 f"{case['expect']['tbs_bits']} expected")
    mine = (derived["modulation"], derived["ldpc_base_graph"], derived["codeblocks"], derived["target_code_rate_x1024"])
    theirs = (native_derived.get("modulation"), native_derived.get("ldpc_base_graph"), native_derived.get("codeblocks"),
              native_derived.get("target_code_rate_x1024"))
    grader.check(name, "coding", mine == theirs and derived["coded_bits"] == sionna["coded_bits"],
                 f"{mine[0]}, {mine[1]}, {mine[2]} code block(s), rate {mine[3]}/1024 in the record and in OCUDU; "
                 f"{derived['coded_bits']} coded bits in the record, {sionna['coded_bits']} in Sionna")

    wanted = grant.occupied()
    masks = {}
    for label, grid in (("OCUDU", native), ("Sionna", x_sionna)):
        found = {(index // width, index % width) for index, value in enumerate(grid) if value != 0}
        masks[label] = (len(found - wanted), len(wanted - found))
    outside = report["grid"].get("nonzero_outside_carrier")
    grader.check(name, "resource element mask", all(value == (0, 0) for value in masks.values()) and outside == 0,
                 f"{len(wanted)} occupied elements; unexpected/empty: OCUDU {masks['OCUDU'][0]}/{masks['OCUDU'][1]}, "
                 f"Sionna {masks['Sionna'][0]}/{masks['Sionna'][1]}; {outside} nonzero outside the carrier grid")

    pilots = list(zip(grant.pilot_index, grant.pilot_reference[0]))
    native_pilots = [native[symbol * width + subcarrier] for (symbol, subcarrier), _ in pilots]
    sionna_pilots = [x_sionna[symbol * width + subcarrier] for (symbol, subcarrier), _ in pilots]
    bad_native = sum(not grants.within_storage(value, reference) for value, (_, reference) in zip(native_pilots, pilots))
    bad_sionna = sum(abs(value - reference) > tolerance["reference_absolute"]
                     for value, (_, reference) in zip(sionna_pilots, pilots))
    grader.check(name, "pilots", bad_native == 0 and bad_sionna == 0,
                 f"{len(pilots)} reference symbols; outside tolerance: OCUDU {bad_native} (storage tolerance), "
                 f"Sionna {bad_sionna} (within {tolerance['reference_absolute']:g} of the record)")

    native_data = [native[symbol * width + subcarrier] for symbol, subcarrier in grant.data_index]
    sionna_data = [x_sionna[symbol * width + subcarrier] for symbol, subcarrier in grant.data_index]
    bad_data = sum(not grants.within_storage(value, reference) for value, reference in zip(native_data, sionna_data))
    rounded = sum(value == complex(grants.bf16(reference.real), grants.bf16(reference.imag))
                  for value, reference in zip(native_data, sionna_data))
    worst = max(abs(value - reference) for value, reference in zip(native_data, sionna_data))
    grader.check(name, "data symbols", bad_data == 0 and len(native_data) == derived["data_re"],
                 f"{len(native_data)} symbols of both references compared, {bad_data} outside the storage tolerance, "
                 f"largest difference {worst:.3g}")
    grader.note(name, f"data symbols equal to the Sionna value rounded to {tolerance['storage']}: "
                      f"{rounded} of {len(native_data)}")

    aligned = {"data": alignment(native_data, sionna_data), "pilots": alignment(native_pilots, sionna_pilots)}
    in_tolerance = all(abs(item["gain"] - 1) <= tolerance["alignment_gain"]
                       and abs(item["phase_rad"]) <= tolerance["alignment_phase_rad"] for item in aligned.values())
    grader.check(name, "scale and phase", in_tolerance,
                 f"OCUDU against Sionna: data gain {aligned['data']['gain']:.5f}, phase "
                 f"{aligned['data']['phase_rad']:+.2e} rad; pilot gain {aligned['pilots']['gain']:.5f}, phase "
                 f"{aligned['pilots']['phase_rad']:+.2e} rad")

    loop = report.get("loopback", {})
    decoded = loop.get("executed") and loop.get("tb_crc_ok") and loop.get("bytes_equal") \
        and loop.get("bytes_compared") == len(grant.tb)
    grader.check(name, "receiver check", bool(decoded),
                 f"host PUSCH processor on {len(grant.rx_ports)} receive port(s), identity channel without noise: CRC "
                 f"{'pass' if loop.get('tb_crc_ok') else 'FAIL'}, {loop.get('bytes_compared')} decoded bytes "
                 f"{'equal' if loop.get('bytes_equal') else 'DIFFER from'} the original block")
    measured.update({"masks": masks, "pilots_outside_tolerance": {"ocudu": bad_native, "sionna": bad_sionna},
                     "data": {"compared": len(native_data), "outside_tolerance": bad_data,
                              "equal_after_rounding": rounded, "largest_difference": worst},
                     "alignment": aligned})

    # ---- The pinned test's own DM-RS reference point, where the case asks for it. ----
    setting = case.get("pinned_dmrs_reference")
    if setting:
        status, pinned = run_adapter(adapter, case_dir / "request", case_dir / "native-prb0", "prb0",
                                     case_dir / "native-prb0.log")
        if status != 0 or not pinned:
            grader.check(name, "pinned DM-RS reference point", False,
                         f"the adapter exited with status {status} (see {name}/native-prb0.log)")
        else:
            other = read_grid(case_dir / "native-prb0" / "x_native.c64", values)
            differing = sum(not grants.within_storage(other[symbol * width + subcarrier], reference)
                            for (symbol, subcarrier), reference in pilots)
            crc = bool(pinned["loopback"].get("tb_crc_ok")) and bool(pinned["loopback"].get("bytes_equal"))
            measured["pinned_reference"] = {"pilots_differing": differing, "host_decode": crc,
                                            "grid_identical": other == native}
            if setting == "identical":
                grader.check(name, "pinned DM-RS reference point", other == native and crc,
                             "the pinned test's PRB0 setting gives the identical grid here, because the bandwidth "
                             "part starts at CRB 0" if other == native else
                             f"PRB0 was expected to equal CRB0 here and {differing} pilots differ")
            else:
                reproduced = differing > 0 and not crc
                grader.check(name, "pinned DM-RS reference point", reproduced,
                             f"the pinned test's PRB0 setting differs from true PUSCH on {differing} of {len(pilots)} "
                             f"pilots and the host decode {'fails' if not crc else 'passes'}; the adapter's CRB0 "
                             "mapping is in force and is recorded in discrepancies.json")
                discrepancies.append({
                    "grant_id": name, "status": "resolved in the fixture adapter" if reproduced else "open",
                    "subject": "DM-RS reference point of the pinned test fixture",
                    "fixture": "pusch_e2e_pipeline_test.cpp sets pdsch_pdu.ref_point = PRB0, the first block of the "
                               "bandwidth part",
                    "true_pusch": "subcarrier 0 of CRB 0 (TS 38.211 6.4.1.1.3; DMRS_REF_POINT_K_TO_POINT_A in the "
                                  "pinned dmrs_pusch_estimator_impl)",
                    "observed": {"bwp_start_rb": grant.bwp_start_rb, "pilots": len(pilots),
                                 "pilots_differing_with_prb0": differing, "host_decode_with_prb0": crc,
                                 "host_decode_with_crb0": bool(decoded)},
                    "resolution": "native/ocudu_adapter/fixtures/grant_fixture.cpp sets CRB0. The two settings "
                                  "coincide when the bandwidth part starts at CRB 0, as in the pinned test."})

    # ---- Archive the grant, the bytes and both grids, then read the entry back. ----
    try:
        manifest = grants.export(grant, case_dir / "store", {
            "x": {"values": native, "provenance": {
                "reference": "pinned OCUDU transmit chain at the Sprint 1 native boundary, read back from the "
                             "resource grid", "dmrs_reference": "crb0",
                "adapter_sha256": receipt["executables"]["grant_fixture"]["sha256"],
                "ocudu_build": receipt["ocudu_build"], "built_with_cuda": report.get("built_with_cuda"),
                "storage": report["grid"]["storage"]}},
            "x_sionna": {"values": x_sionna, "provenance": {
                "reference": "Sionna NR PUSCHTransmitter, aligned to carrier coordinates",
                "versions": sionna["versions"], "dmrs_scale_applied": sionna["dmrs_scale_applied"],
                "own_dmrs_is_crb0_referenced": sionna["own_dmrs_is_crb0_referenced"]}}})
        store = grants.open_store(case_dir / "store")
    except (GrantError, OSError) as error:
        grader.check(name, "archived record", False, str(error))
        return measured
    hashes = {stem: manifest["files"][stem + ".npy"]["sha256"] for stem in ("x", "x_sionna")}
    measured["store"] = {"grid_sha256": hashes, "tb_sha256": manifest["files"]["tb.bin"]["sha256"]}
    grader.check(name, "archived record", True,
                 f"grant, {len(grant.tb)} original bytes, sensing source and both grids, every file verified on "
                 f"reading (store/manifest.json)")
    grader.note(name, f"grid hashes: x.npy {hashes['x'][:16]}, x_sionna.npy {hashes['x_sionna'][:16]}; "
                      f"transport block {manifest['files']['tb.bin']['sha256'][:16]}")
    # Recorded, not graded: whether this host reproduces the files of the development-host rehearsal.
    earlier = case.get("rehearsal_sha256", {})
    if earlier:
        same = {stem: "identical" if earlier.get(stem + ".npy") == value else "different"
                for stem, value in hashes.items()}
        measured["store"]["against_rehearsal"] = same
        grader.note(name, f"against the x86_64 rehearsal (not graded): x.npy {same['x']}, x_sionna.npy "
                          f"{same['x_sionna']}")

    # ---- What a second developer recovers from the entry, against the answers on file. ----
    expect = case["expect"]
    problems = []
    try:
        pilot = store.pilot(expect["pilot"]["index"])
        if (pilot["symbol"], pilot["subcarrier"]) != (expect["pilot"]["symbol"], expect["pilot"]["subcarrier"]) \
                or abs(complex(*pilot["reference"]) - complex(*expect["pilot"]["reference"])) > 1e-9:
            problems.append(f"pilot {expect['pilot']['index']} is {pilot}")
        for wanted_symbol in expect["data"]:
            got = store.data(wanted_symbol["index"])
            if (got["symbol"], got["subcarrier"]) != (wanted_symbol["symbol"], wanted_symbol["subcarrier"]) \
                    or not grants.within_storage(complex(*got["value"]), complex(*wanted_symbol["value"])):
                problems.append(f"data symbol {wanted_symbol['index']} is {got}")
        if hashlib.sha256((case_dir / "store" / "tb.bin").read_bytes()).hexdigest() != expect["tb_sha256"]:
            problems.append("the transport block hash differs")
    except GrantError as error:
        problems.append(str(error))
    first = expect["data"][0]
    if not problems:
        stored = store.data(first["index"])["value"]
    grader.check(name, "recovery from the record", not problems,
                 "; ".join(problems) if problems else
                 f"pilot {expect['pilot']['index']} at symbol {expect['pilot']['symbol']}, subcarrier "
                 f"{expect['pilot']['subcarrier']}; data symbol {first['index']} at symbol {first['symbol']}, "
                 f"subcarrier {first['subcarrier']} = {stored[0]:+.6f}{stored[1]:+.6f}j, the stored form of the "
                 f"expected {first['value'][0]:+.6f}{first['value'][1]:+.6f}j; transport block SHA-256 "
                 f"{expect['tb_sha256'][:16]}")
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
    parser.add_argument("--adapter-build", type=Path, help="adapter build directory (default: from the configuration)")
    parser.add_argument("--device", default="cpu", help="device of the Sionna reference (default: cpu)")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.exit(2, f"grant_fixture: {args.output_dir} already exists. Evidence is never overwritten.\n")
    try:
        config = load_config(args.config)
        receipt = native_adapter.verified_receipt(args.adapter_build or ROOT / config["adapter_build"])
    except (ConfigError, native_adapter.AdapterError) as error:
        parser.exit(2, f"grant_fixture: {error}\n")
    adapter = Path(receipt["executables"]["grant_fixture"]["path"])
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = args.output_dir.resolve()
    out.mkdir(parents=True)
    write_json(out / "adapter-receipt.json", receipt)

    grader, discrepancies, measured = Grader(), [], {}
    for case in config["cases"]:
        before = len(grader.results)
        try:
            result = run_case(case, config, adapter, receipt, out, grader, discrepancies, args.device)
        except Exception as error:  # an unexpected fault fails its case and keeps the remaining cases
            (out / "driver-error.txt").write_text(traceback.format_exc())
            grader.check(Path(case["grant"]).stem, "driver", False, f"{type(error).__name__}: {error} "
                                                                      "(see driver-error.txt)")
            result = {}
        name = grader.results[before]["case"] if len(grader.results) > before else Path(case["grant"]).stem
        measured[name] = result
        for failure in (item for item in grader.results[before:] if not item["passed"]):
            if failure["check"] != "pinned DM-RS reference point":
                discrepancies.append({"grant_id": name, "status": "open", "subject": failure["check"],
                                      "observed": failure["detail"],
                                      "resolution": "fix the grant or the mapper before S2.4"})

    names = list(measured)
    passed_cases = [name for name in names
                    if any(item["case"] == name for item in grader.results)
                    and all(item["passed"] for item in grader.results if item["case"] == name)]
    failed = [item for item in grader.results if not item["passed"]]
    grade = {
        "schema": GRADE_SCHEMA, "selector": "grant_fixture", "run_id": out.name,
        "cases": len(config["cases"]), "cases_passed": len(passed_cases), "cases_failed": len(names) - len(passed_cases),
        "checks": len(grader.results), "checks_passed": len(grader.results) - len(failed), "checks_failed": len(failed),
        "judgment": "pass" if not failed and len(passed_cases) == len(config["cases"]) else "fail",
        "reason": "every check of every case passed" if not failed else
                  f"{len(failed)} check(s) failed, first: {failed[0]['case']} {failed[0]['check']}",
        "tolerance": config["tolerance"], "results": grader.results,
    }
    write_json(out / "grade.json", grade)
    write_json(out / "comparison.json", measured)
    write_json(out / "discrepancies.json", {
        "discrepancies": discrepancies,
        "alignments": [
            {"subject": "DM-RS sequence origin in Sionna",
             "fact": "Sionna places the sequence from the first subcarrier of its bandwidth part and fills that part",
             "alignment": "data from a bandwidth part equal to the allocation, DM-RS from a bandwidth part that "
                          "starts at CRB 0 and ends with the allocation"},
            {"subject": "DM-RS amplitude",
             "fact": "OCUDU scales by 10**(3/20) for two CDM groups without data (TS 38.214 Table 6.2.2-1 in dB), "
                     "Sionna by sqrt(2)",
             "alignment": "the record states the OCUDU amplitude, which the receiver assumes; the Sionna pilots are "
                          "rescaled by their ratio, 0.998815. The difference is below the storage resolution."},
            {"subject": "bit order", "fact": "OCUDU takes transport block bytes, Sionna a bit vector",
             "alignment": "byte 0 first, most significant bit first"},
            {"subject": "storage", "fact": "the pinned grid stores complex bf16, Sionna computes in double precision",
             "alignment": f"compared within {config['tolerance']['relative']} of each component's magnitude plus "
                          f"{config['tolerance']['absolute']}"}]})
    lock = {item["name"]: item["commit"] for item in json.loads((ROOT / "ext/lock.json").read_text())["dependencies"]}
    write_json(out / "manifest.json", {
        "run_id": out.name, "selector": "grant_fixture", "purpose": "Sprint 2.1 integration: one legal uplink grant "
        "and its transmit reference, compared across both transmit references",
        "command": shlex.join([sys.executable, *sys.argv]), "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": platform.node(), "machine": platform.machine(), "python": platform.python_version(),
        "parent_commit": git_commit(), "locked_sources": lock,
        "config": {"file": str(args.config), "sha256": digest(args.config)},
        "inputs": {case["grant"]: digest(ROOT / case["grant"]) for case in config["cases"]
                   if (ROOT / case["grant"]).is_file()},
        "driver_sha256": digest(__file__),
        "adapter": {"executable": str(adapter), "sha256": receipt["executables"]["grant_fixture"]["sha256"],
                    "receipt": "adapter-receipt.json", "ocudu_build": receipt["ocudu_build"],
                    "ocudu_receipt_sha256": receipt["ocudu_receipt_sha256"]},
        "sionna_device": args.device,
        "environment": {key: os.environ[key] for key in ("VIRTUAL_ENV", "CUDA_VISIBLE_DEVICES") if key in os.environ},
        "processors": "host software factories of the pinned test for the transmit chain and the receiver check; "
                      "no accelerated processor is requested",
        "judgment": grade["judgment"], "reason": grade["reason"]})
    summary = (f"grant_fixture: {grade['cases_passed']} of {grade['cases']} cases passed, "
               f"{grade['checks_passed']} of {grade['checks']} checks")
    (out / "grade.txt").write_text("\n".join(grader.lines + [summary]) + "\n", encoding="ascii")
    print(summary)
    print(f"Run artifacts: {out}")
    return 0 if grade["judgment"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
