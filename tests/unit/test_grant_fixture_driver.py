"""Check the Sprint 2.1 integration driver against doubles of both transmit references.

No OCUDU build and no Sionna are needed. A small script stands in for the native adapter and a
function for the Sionna reference. Both derive a grid from the grant and the transport block, so a
correct run agrees and every injected fault must surface as the failed check that names it.
"""

import contextlib
import copy
import importlib.util
import io
import json
from math import sqrt
import os
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from ocudu_cuda_isac.episodes import grant as grants

spec = importlib.util.spec_from_file_location("grant_fixture_driver", ROOT / "tests/integration/grant_fixture.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)

CONFIG = json.loads((ROOT / "configs/integration/grant_fixture.json").read_text())
# The adapter double. It takes the options of the real adapter and writes the same two files. The
# variable ADAPTER_FAULT selects one deviation from a correct transmit reference.
ADAPTER_DOUBLE = '''import json, os, struct, sys
sys.path.insert(0, {source!r})
from ocudu_cuda_isac.episodes import grant as grants

options = dict(zip(sys.argv[1::2], sys.argv[2::2]))
fault = os.environ.get("ADAPTER_FAULT", "")
if fault == "exit":
    sys.exit("grant_fixture: input error: the grant has no field 'tb'")
grant = grants.load(options["--grant"])
assert open(options["--tb"], "rb").read() == grant.tb
width = grant.grid_subcarriers
x = [0j] * (14 * width)
pinned = options["--dmrs-reference"] == "prb0" and grant.bwp_start_rb != 0 and fault != "prb0-agrees"
for number, ((symbol, subcarrier), value) in enumerate(zip(grant.pilot_index, grant.pilot_reference[0])):
    if (pinned and number % 4) or (fault == "pilot" and number == 5):
        value = -value
    x[symbol * width + subcarrier] = complex(grants.bf16(value.real), grants.bf16(value.imag))
bits = grant.tb_bits()
for number, (symbol, subcarrier) in enumerate(grant.data_index):
    value = complex(1 - 2 * bits[2 * number % len(bits)], 1 - 2 * bits[(2 * number + 1) % len(bits)]) / 2 ** 0.5
    if fault == "data" and number == 7:
        value = -value
    x[symbol * width + subcarrier + (1 if fault == "shift" else 0)] = complex(grants.bf16(value.real), grants.bf16(value.imag))
derived = grant.derived()
decoded = not pinned and fault != "crc"
report = {{
    "schema": "grant-fixture-native/v1", "grant_id": grant.grant_id, "dmrs_reference": options["--dmrs-reference"],
    "built_with_cuda": True,
    "pdu": {{"scs_khz": grant.scs_khz, "sfn": grant.sfn, "slot": grant.slot, "rnti": grant.rnti, "n_id": grant.n_id,
            "bwp_start_rb": grant.bwp_start_rb, "bwp_size_rb": grant.bwp_size_rb, "prb_start": grant.prb_start,
            "prb_count": grant.prb_count + (1 if fault == "pdu" else 0), "symbol_start": grant.symbol_start,
            "symbol_count": grant.symbol_count, "layers": 1, "rv": 0, "dmrs_config_type": 1,
            "dmrs_symbols": grant.dmrs["symbols"], "dmrs_scrambling_id": grant.dmrs["scrambling_id"],
            "dmrs_n_scid": grant.dmrs["n_scid"], "cdm_groups_without_data": 2, "rx_ports": grant.rx_ports}},
    "derived": {{"tbs_bits": derived["tbs_bits"] + (8 if fault == "tbs" else 0), "modulation": derived["modulation"],
                "ldpc_base_graph": "BG1" if fault == "coding" else derived["ldpc_base_graph"],
                "codeblocks": derived["codeblocks"], "target_code_rate_x1024": derived["target_code_rate_x1024"]}},
    "grid": {{"storage": "complex bf16", "nonzero_outside_carrier": 3 if fault == "outside" else 0}},
    "loopback": {{"executed": True, "tb_crc_ok": decoded, "bytes_equal": decoded, "bytes_compared": len(grant.tb)}},
}}
with open(os.path.join(options["--output-dir"], "x_native.c64"), "wb") as stream:
    stream.write(struct.pack("<%df" % (2 * len(x)), *[part for value in x for part in (value.real, value.imag)]))
with open(os.path.join(options["--output-dir"], "native.json"), "w") as stream:
    json.dump(report, stream)
'''


def reference_double(grant, device="cpu"):
    """Stands in for sionna_pusch.reference: the same data rule as the adapter double, in double precision."""
    width = grant.grid_subcarriers
    x = [0j] * (14 * width)
    for (symbol, subcarrier), value in zip(grant.pilot_index, grant.pilot_reference[0]):
        x[symbol * width + subcarrier] = value
    bits = grant.tb_bits()
    for number, (symbol, subcarrier) in enumerate(grant.data_index):
        x[symbol * width + subcarrier] = complex(1 - 2 * bits[2 * number % len(bits)],
                                                 1 - 2 * bits[(2 * number + 1) % len(bits)]) / sqrt(2)
    return {"x": x, "tb_size": grant.tbs_bits, "coded_bits": grant.derived()["coded_bits"], "dmrs_symbols": [2, 11],
            "dmrs_scale_applied": 0.998815, "sionna_dmrs_beta": sqrt(2), "own_dmrs_is_crb0_referenced": False,
            "versions": {"sionna": "double", "torch": "none", "numpy": "none"}, "device": device}


class DriverTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        # A shell launcher, so that an interpreter path with a space cannot break the first line.
        (self.root / "adapter_double.py").write_text(ADAPTER_DOUBLE.format(source=str(ROOT / "src")))
        self.adapter = self.root / "grant_fixture"
        self.adapter.write_text(f"#!/bin/sh\nexec {shlex.quote(sys.executable)} "
                                f"{shlex.quote(str(self.root / 'adapter_double.py'))} \"$@\"\n")
        self.adapter.chmod(0o755)
        receipt = {"executables": {"grant_fixture": {"path": str(self.adapter), "sha256": "ab" * 32}},
                   "ocudu_build": "/build/ocudu-spark", "ocudu_receipt_sha256": "cd" * 32}
        for target, name, value in ((driver.native_adapter, "verified_receipt", lambda build: receipt),
                                    (driver.sionna_pusch, "reference", reference_double)):
            patcher = mock.patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        # Expectations of the doubles: the first data symbol follows the first two block bits.
        grant = grants.load(ROOT / "configs/grants/two_prb_offset.json")
        bits = grant.tb_bits()
        self.config = copy.deepcopy(CONFIG)
        self.config["cases"] = [self.config["cases"][0]]
        self.config["cases"][0]["expect"]["data"] = [
            {"index": 0, "symbol": 0, "subcarrier": 180,
             "value": [(1 - 2 * bits[0]) / sqrt(2), (1 - 2 * bits[1]) / sqrt(2)]}]
        self.runs = 0

    def run_driver(self, fault="", config=None, output=None):
        self.runs += 1
        path = self.root / f"config{self.runs}.json"
        path.write_text(json.dumps(config or self.config))
        out = output or self.root / f"run{self.runs}"
        stdout = io.StringIO()
        with mock.patch.dict(os.environ, {"ADAPTER_FAULT": fault}), contextlib.redirect_stdout(stdout), \
                contextlib.redirect_stderr(io.StringIO()) as stderr:
            try:
                status = driver.main(["--config", str(path), "--output-dir", str(out)])
            except SystemExit as stop:
                status = stop.code
        return status, stdout.getvalue(), stderr.getvalue(), out

    def test_agreeing_references_pass_every_check_and_leave_the_record(self):
        status, shown, _, out = self.run_driver()
        self.assertEqual(status, 0, shown)
        lines = shown.splitlines()
        self.assertFalse([line for line in lines if line.startswith("FAIL")], shown)
        checks = [line.split(" two-prb-offset ")[1].split(":")[0] for line in lines if line.startswith("PASS")]
        self.assertEqual(checks, ["grant record", "native transmit reference", "Sionna transmit reference",
                                  "transport block size", "coding", "resource element mask", "pilots", "data symbols",
                                  "scale and phase", "receiver check", "pinned DM-RS reference point",
                                  "archived record", "recovery from the record"])
        self.assertEqual(lines[-2], "grant_fixture: 1 of 1 cases passed, 13 of 13 checks")
        # The doubles are not the rehearsal's references. That comparison is a note and no check.
        self.assertIn("NOTE  S2.1 two-prb-offset against the x86_64 rehearsal (not graded): x.npy different, "
                      "x_sionna.npy different", lines)
        grade = json.loads((out / "grade.json").read_text())
        self.assertEqual((grade["judgment"], grade["cases"], grade["cases_passed"], grade["checks"], grade["checks_failed"]),
                         ("pass", 1, 1, 13, 0))
        self.assertEqual(grade["tolerance"]["relative"], 1 / 256)
        self.assertEqual((out / "grade.txt").read_text().splitlines(), lines[:-1])
        case = out / "two-prb-offset"
        for name in ("request/grant.json", "request/tb.bin", "native/x_native.c64", "native/native.json", "native.log",
                     "native-prb0/x_native.c64", "sionna.json", "store/x.npy", "store/x_sionna.npy",
                     "store/manifest.json"):
            self.assertTrue((case / name).is_file(), name)
        # The archived entry verifies on its own and names both references and their hashes.
        store = grants.open_store(case / "store")
        self.assertEqual(sorted(store.grids), ["x", "x_sionna"])
        references = store.manifest["transmit_references"]
        self.assertEqual(references["x.npy"]["provenance"]["adapter_sha256"], "ab" * 32)
        self.assertEqual(references["x_sionna.npy"]["provenance"]["versions"]["sionna"], "double")
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual(manifest["judgment"], "pass")
        self.assertEqual(manifest["adapter"]["ocudu_receipt_sha256"], "cd" * 32)
        self.assertEqual(list(manifest["inputs"]), ["configs/grants/two_prb_offset.json"])
        self.assertEqual(set(manifest["locked_sources"]), {"ocudu", "sionna", "sionna-rt"})
        # The pinned test's reference point is on file as a resolved discrepancy, with the alignments.
        record = json.loads((out / "discrepancies.json").read_text())
        self.assertEqual([(item["status"], item["observed"]["pilots_differing_with_prb0"], item["observed"]
                           ["host_decode_with_prb0"]) for item in record["discrepancies"]],
                         [("resolved in the fixture adapter", 18, False)])
        self.assertEqual(len(record["alignments"]), 4)
        for path in out.rglob("*.json"):
            self.assertTrue(path.read_text().isascii(), path)

    def test_each_fault_fails_the_check_that_names_it(self):
        cases = (
            ("exit", ["native transmit reference"], "the adapter exited with status 1"),
            ("pdu", ["native transmit reference"], "the adapter's PDU differs from the grant"),
            ("tbs", ["transport block size"], "384 bits in the record, 392 in OCUDU, 384 in Sionna"),
            ("coding", ["coding"], "16QAM, BG2, 1 code block"),
            # Data written one subcarrier too high: one element per symbol leaves the allocation.
            ("shift", ["resource element mask", "data symbols", "scale and phase", "archived record"],
             "unexpected/empty: OCUDU 12/12, Sionna 0/0"),
            ("outside", ["resource element mask"], "3 nonzero outside the carrier grid"),
            # One inverted symbol also moves the least-squares gain beyond its tolerance.
            ("pilot", ["pilots", "scale and phase", "archived record"],
             "outside tolerance: OCUDU 1 (storage tolerance), Sionna 0"),
            ("data", ["data symbols", "scale and phase"],
             "288 symbols of both references compared, 1 outside the storage tolerance"),
            ("crc", ["receiver check"], "CRC FAIL, 48 decoded bytes DIFFER from the original block"),
            ("prb0-agrees", ["pinned DM-RS reference point"], "differs from true PUSCH on 0 of 24 pilots and the "
                                                              "host decode passes"),
        )
        for fault, failed_checks, message in cases:
            status, shown, _, out = self.run_driver(fault)
            with self.subTest(fault=fault):
                self.assertEqual(status, 1, shown)
                failed = [line.split(" two-prb-offset ")[1].split(":")[0] for line in shown.splitlines()
                          if line.startswith("FAIL")]
                self.assertEqual(failed, failed_checks)
                self.assertIn(message, shown)
                grade = json.loads((out / "grade.json").read_text())
                self.assertEqual((grade["judgment"], grade["cases_passed"], grade["checks_failed"]),
                                 ("fail", 0, len(failed_checks)))
                self.assertIn(failed_checks[0], grade["reason"])
                # Every failed check is persisted as an open discrepancy for the next ticket.
                found = json.loads((out / "discrepancies.json").read_text())["discrepancies"]
                self.assertEqual(sorted(item["subject"] for item in found if item["status"] == "open"),
                                 sorted("DM-RS reference point of the pinned test fixture"
                                        if check == "pinned DM-RS reference point" else check
                                        for check in failed_checks))
                self.assertEqual(json.loads((out / "manifest.json").read_text())["judgment"], "fail")

    def test_a_reference_that_cannot_run_is_a_failure_and_never_a_skip(self):
        def missing(grant, device="cpu"):
            raise grants.GrantError("the Sionna reference needs the full environment: No module named 'sionna'")

        def broken(grant, device="cpu"):
            raise RuntimeError("device mismatch")

        for double, message in ((missing, "the Sionna reference needs the full environment"),
                                (broken, "RuntimeError: device mismatch")):
            with mock.patch.object(driver.sionna_pusch, "reference", double):
                status, shown, _, out = self.run_driver()
            self.assertEqual(status, 1)
            self.assertIn(f"FAIL  S2.1 two-prb-offset Sionna transmit reference: {message}", shown)
            self.assertNotIn("SKIP", shown)
            self.assertFalse((out / "two-prb-offset/store").exists())  # nothing is archived as agreed
            self.assertEqual(json.loads((out / "grade.json").read_text())["cases_passed"], 0)

    def test_answers_on_file_must_match_what_the_record_gives(self):
        config = copy.deepcopy(self.config)
        config["cases"][0]["expect"]["data"][0]["value"][0] *= -1
        status, shown, _, _ = self.run_driver(config=config)
        self.assertEqual(status, 1)
        self.assertIn("FAIL  S2.1 two-prb-offset recovery from the record: data symbol 0 is", shown)
        config = copy.deepcopy(self.config)
        config["cases"][0]["expect"].update(tbs_bits=392, tb_sha256="0" * 64)
        config["cases"][0]["expect"]["pilot"]["subcarrier"] = 182
        status, shown, _, _ = self.run_driver(config=config)
        self.assertEqual(status, 1)
        self.assertIn("384 in Sionna, 392 expected", shown)
        self.assertIn("recovery from the record: pilot 0 is", shown)
        self.assertIn("the transport block hash differs", shown)

    def test_a_bandwidth_part_at_crb_zero_reproduces_the_pinned_setting(self):
        config = copy.deepcopy(CONFIG)
        config["cases"] = [config["cases"][2]]
        grant = grants.load(ROOT / config["cases"][0]["grant"])
        bits = grant.tb_bits()
        config["cases"][0]["expect"]["data"] = [{"index": 0, "symbol": 0, "subcarrier": 0,
                                                 "value": [(1 - 2 * bits[0]) / sqrt(2), (1 - 2 * bits[1]) / sqrt(2)]}]
        status, shown, _, out = self.run_driver(config=config)
        self.assertEqual(status, 0, shown)
        self.assertIn("PASS  S2.1 table2-51prb-1rx pinned DM-RS reference point: the pinned test's PRB0 setting gives "
                      "the identical grid here", shown)
        self.assertEqual(json.loads((out / "discrepancies.json").read_text())["discrepancies"], [])

    def test_setup_errors_stop_before_anything_is_written(self):
        existing = self.root / "existing"
        existing.mkdir()
        status, _, error, _ = self.run_driver(output=existing)
        self.assertEqual(status, 2)
        self.assertIn("already exists. Evidence is never overwritten.", error)
        self.assertEqual(list(existing.iterdir()), [])
        changes = (
            (lambda c: c["tolerance"].update(relative=0.01), "the storage tolerance must be the declared one"),
            (lambda c: c.update(cases=[]), "names no case"),
            (lambda c: c.update(schema="other"), "must be a integration-config/grant_fixture/v1 object"),
            (lambda c: c["cases"][0].update(pinned_dmrs_reference="maybe"), "a case holds grant, expect"),
            (lambda c: c["cases"][0]["expect"].pop("pilot"), "expect requires exactly"),
        )
        for change, message in changes:
            config = copy.deepcopy(self.config)
            change(config)
            status, _, error, out = self.run_driver(config=config)
            self.assertEqual(status, 2)
            self.assertIn(message, error)
            self.assertFalse(out.exists())
        with mock.patch.object(driver.native_adapter, "verified_receipt",
                               side_effect=driver.native_adapter.AdapterError("no adapter receipt in build")):
            status, _, error, out = self.run_driver()
        self.assertEqual(status, 2)
        self.assertIn("no adapter receipt", error)
        self.assertFalse(out.exists())

    def test_shipped_configuration_is_valid_and_names_the_three_grants(self):
        config = driver.load_config(ROOT / "configs/integration/grant_fixture.json")
        self.assertEqual([case["grant"] for case in config["cases"]],
                         ["configs/grants/two_prb_offset.json", "configs/grants/demo_51prb_4rx.json",
                          "configs/grants/table2_51prb_1rx.json"])
        self.assertEqual([case.get("pinned_dmrs_reference") for case in config["cases"]], ["differs", None, "identical"])
        for case in config["cases"]:
            grant = grants.load(ROOT / case["grant"])
            derived = grant.derived()
            self.assertEqual((case["expect"]["tbs_bits"], case["expect"]["tb_sha256"]),
                             (derived["tbs_bits"], derived["tb_sha256"]))
            pilot = case["expect"]["pilot"]
            self.assertEqual(grant.pilot_index[pilot["index"]], (pilot["symbol"], pilot["subcarrier"]))
            self.assertAlmostEqual(abs(grant.pilot_reference[0][pilot["index"]] - complex(*pilot["reference"])), 0)
            for item in case["expect"]["data"]:
                self.assertEqual(grant.data_index[item["index"]], (item["symbol"], item["subcarrier"]))
        self.assertTrue((ROOT / "configs/integration/grant_fixture.json").read_text().isascii())
        self.assertTrue((ROOT / "tests/integration/grant_fixture.py").read_bytes().isascii())


if __name__ == "__main__":
    unittest.main()
