"""Check the Sprint 2.2 integration driver against doubles of the ray tracer and of the native adapter.

The doubles compute the analytic two-path model, so every check of the driver has a known answer
and each injected fault must fail the check that names it. NumPy is needed, as for the driver.
"""

import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    np = None

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ocudu_cuda_isac.episodes import episode as episodes  # noqa: E402
from ocudu_cuda_isac.episodes import episode_set as sets  # noqa: E402
from ocudu_cuda_isac.episodes import grant as grants  # noqa: E402
from ocudu_cuda_isac.episodes.episode import TC_PER_SECOND  # noqa: E402

DATA = complex(0.31640625, 0.31640625)  # a stored 16QAM point
ADAPTER = '''#!{python}
"""Double of the native fixture adapter: pilots of the grant, one fixed data symbol, a scripted receiver check."""
import json, os, struct, sys
from pathlib import Path
sys.path.insert(0, {source!r})
from ocudu_cuda_isac.episodes import grant as grants
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
grant = grants.load(args["--grant"])
fault = os.environ.get("ADAPTER_FAULT", "")
x = [0j] * (14 * grant.grid_subcarriers)
slot = dict(grant.record()["slot"], slot=7)
pilots = grants.build(dict(grant.record(), slot=slot)) if fault == "pilots" else grant
for (symbol, k), value in zip(grant.pilot_index, pilots.pilot_reference[0]):
    x[symbol * grant.grid_subcarriers + k] = complex(grants.bf16(value.real), grants.bf16(value.imag))
for symbol, k in grant.data_index:
    x[symbol * grant.grid_subcarriers + k] = complex({data!r})
out = Path(args["--output-dir"])
(out / "x_native.c64").write_bytes(struct.pack("<%df" % (2 * len(x)), *[p for v in x for p in (v.real, v.imag)]))
pdu = {{"scs_khz": grant.scs_khz, "sfn": grant.sfn, "slot": grant.slot, "rnti": grant.rnti, "n_id": grant.n_id,
       "bwp_start_rb": grant.bwp_start_rb, "bwp_size_rb": grant.bwp_size_rb, "prb_start": grant.prb_start,
       "prb_count": grant.prb_count, "symbol_start": grant.symbol_start, "symbol_count": grant.symbol_count,
       "layers": grant.layers, "rv": 0, "dmrs_config_type": 1, "dmrs_symbols": grant.dmrs["symbols"],
       "dmrs_scrambling_id": grant.dmrs["scrambling_id"], "dmrs_n_scid": grant.dmrs["n_scid"],
       "cdm_groups_without_data": grant.dmrs["cdm_groups_without_data"], "rx_ports": grant.rx_ports}}
report = {{"schema": "grant-fixture-native/v1", "grant_id": grant.grant_id, "dmrs_reference": "crb0", "pdu": pdu,
          "grid": {{"storage": "double"}}, "built_with_cuda": False,
          "loopback": {{"executed": True, "tb_crc_ok": fault != "crc", "bytes_equal": fault != "crc",
                       "bytes_compared": len(grant.tb)}}}}
(out / "native.json").write_text(json.dumps(report))
sys.exit(3 if fault == "exit" else 0)
'''


def load_driver():
    spec = importlib.util.spec_from_file_location("episode_pair_driver", ROOT / "tests/integration/episode_pair.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def analytic_export(fault=None):
    """A double of rt_channel.export_pair that stores the analytic model, optionally with one fault."""
    def export_pair(config_path, output_dir, variant, waveforms=None):
        if fault == "raise":
            raise RuntimeError("the ray tracer is not available")
        config, base, schedule = sets.load_pair_config(config_path)
        wavelength = episodes.SPEED_OF_LIGHT_M_S / base.carrier_hz
        geometry = {"ue_tx_m": [float(v) for v in config["endpoints"]["ue_tx_m"]],
                    "gnb_rx_m": [float(v) for v in config["endpoints"]["gnb_rx_m"]],
                    "rx_element_offsets_m": [[0.0, y * wavelength, 0.0] for y in (-0.75, -0.25, 0.25, 0.75)]}
        target = json.loads(json.dumps(config["target"]))
        traced = json.loads(json.dumps(target))
        if fault == "approaching":  # the stored channel moves the other way than the record says
            traced["motion"]["velocity_m_s"] = [-v for v in traced["motion"]["velocity_m_s"]]
        offsets = np.array([base.subcarrier_offset_hz(k) for k in range(base.grid_subcarriers)], dtype=np.float64)
        h_shape, w_shape = episodes.shapes(schedule)
        h_a, h_b = np.zeros(h_shape, dtype="<c8"), np.zeros(h_shape, dtype="<c8")
        truth = []
        for item in schedule.opportunities:
            for symbol in range(14):
                paths = episodes.free_space_paths(geometry, traced, item.symbol_time_tc(symbol) / TC_PER_SECOND,
                                                  base.carrier_hz, True)
                tones = [amplitude * np.exp(1j * np.array(phases))[:, None]
                         * np.exp(-2j * np.pi * (base.carrier_hz + offsets) * delay)[None, :]
                         for amplitude, delay, phases in paths]
                h_a[item.index, :, 0, symbol, :] = tones[0]
                h_b[item.index, :, 0, symbol, :] = tones[0] + (tones[1][::-1] if fault == "rx_order" else tones[1])
            position, velocity = episodes.target_state(target["motion"], item.slot_start_tc / TC_PER_SECOND)
            truth.append({"opportunity": item.index, "position_m": position, "velocity_m_s": velocity,
                          "bistatic": episodes.bistatic(geometry, position, velocity, base.carrier_hz)})
        power = float(np.mean(np.abs(h_a) ** 2))
        sigma = math.sqrt(power * 10 ** (-config["noise"]["snr_db"] / 10))
        generator = np.random.Generator(np.random.PCG64(config["noise"]["seed"]))
        w = ((generator.standard_normal(w_shape) + 1j * generator.standard_normal(w_shape)) * sigma / math.sqrt(2))
        ratio = 20 * math.log10(float(np.abs(h_b - h_a).max() / np.abs(h_a).max()))
        measured = {
            "a": {"snapshots": 1, "paths": 1, "solver_cfr_max_relative_error": 3e-5,
                  "direct_path": {"delay_ns": 66.9, "amplitude": [3.4e-4] * 4, "phase_rad": [0.0] * 4}},
            "b": {"snapshots": len(truth), "background_paths": [1, 1], "scattered_paths": [1, 1],
                  "solver_cfr_max_relative_error": 3e-5 if fault != "solver" else 0.2,
                  "uniform_grid_check": {"steps": 507, "max_relative_error": 3e-5},
                  "scattered_to_direct_db": {"median": ratio, "median_margin_to_rounding_rms_db": ratio + 55.6,
                                             "median_margin_to_rounding_bound_db": ratio + 48.2},
                  "solve_seconds": {"mean": 0.01, "max": 0.02, "total": 0.3},
                  "path_validity": {"gaps_ms": [0.5, 4.0], "max_relative_error": 1e-3}}}
        description = {"geometry": geometry, "array": {}, "scene": config["scene"], "target": target,
                       "solver": {"mitsuba_variant": variant}, "time_evolution": {},
                       "noise": {"snr_db": float(config["noise"]["snr_db"]), "sigma": sigma, "seed": config["noise"]["seed"]},
                       "calibration": {"reference_power": power}, "software": {}, "provenance": {},
                       "analytic_model": episodes.ANALYTIC_MODEL}
        return episodes.export(output_dir, config["episode_id"], schedule, description,
                               {"a": h_a.tobytes(), "b": h_b.tobytes(), "w": w.astype("<c8").tobytes()}, truth,
                               measured, waveforms)
    return export_pair


@unittest.skipIf(np is None, "the integration driver needs NumPy")
class DriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.driver = load_driver()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.adapter = self.root / "grant_fixture"
        self.adapter.write_text(ADAPTER.format(python=sys.executable, source=str(ROOT / "src"), data=DATA))
        self.adapter.chmod(0o755)
        # The record Sprint 2.1 archived for the base grant, with the double's transmit grid.
        base = grants.load(ROOT / "configs/grants/demo_51prb_4rx.json")
        x = [0j] * (14 * base.grid_subcarriers)
        for (symbol, k), value in zip(base.pilot_index, base.pilot_reference[0]):
            x[symbol * base.grid_subcarriers + k] = complex(grants.bf16(value.real), grants.bf16(value.imag))
        for symbol, k in base.data_index:
            x[symbol * base.grid_subcarriers + k] = DATA
        self.store = self.root / "s21" / "grant_fixture"
        grants.export(base, self.store / base.grant_id / "store", {"x": {"values": x, "provenance": {"double": True}}})
        receipt = {"executables": {"grant_fixture": {"path": str(self.adapter),
                                                     "sha256": hashlib.sha256(self.adapter.read_bytes()).hexdigest()}},
                   "ocudu_build": "double"}
        driver = self.driver
        self.saved = (driver.native_adapter.verified_receipt, driver.rt_channel.export_pair, driver.run_diagnostic,
                      driver.run_set, os.environ.get("ADAPTER_FAULT"))
        driver.native_adapter.verified_receipt = lambda build_dir: receipt
        driver.run_diagnostic = lambda config, out, grader, numpy: self.stub(grader, "diagnostic")
        driver.run_set = lambda config, out, grader, numpy: self.stub(grader, "episode-set")
        self.addCleanup(self.restore)

    def restore(self):
        driver = self.driver
        (driver.native_adapter.verified_receipt, driver.rt_channel.export_pair, driver.run_diagnostic, driver.run_set,
         fault) = self.saved
        os.environ.pop("ADAPTER_FAULT", None)
        if fault is not None:
            os.environ["ADAPTER_FAULT"] = fault

    @staticmethod
    def stub(grader, case):
        grader.check(case, "stub", True, "covered by the lab gate")
        return {}

    def run_driver(self, name, pair_fault=None, adapter_fault=None, store=None):
        self.driver.rt_channel.export_pair = analytic_export(pair_fault)
        os.environ.pop("ADAPTER_FAULT", None)
        if adapter_fault:
            os.environ["ADAPTER_FAULT"] = adapter_fault
        out = self.root / name
        with contextlib.redirect_stdout(io.StringIO()):
            status = self.driver.main(["--config", str(ROOT / "configs/integration/episode_pair.json"),
                                       "--grant-store", str(store or self.store), "--output-dir", str(out)])
        grade = json.loads((out / "grade.json").read_text())
        failed = sorted(f"{item['case']} {item['check']}" for item in grade["results"] if not item["passed"])
        return status, grade, failed, out

    def test_the_analytic_pair_passes_every_check(self):
        status, grade, failed, out = self.run_driver("pass")
        self.assertEqual((status, failed), (0, []))
        self.assertEqual((grade["judgment"], grade["cases"], grade["cases_passed"], grade["checks"]), ("pass", 4, 4, 20))
        lines = (out / "grade.txt").read_text()
        self.assertIn("episode_pair: 4 of 4 cases passed, 20 of 20 checks", lines)
        self.assertIn("PASS  S2.2 grant-fixture slot 17 grant: demo-51prb-4rx.s17", lines)
        self.assertIn("nu = -43.33 Hz is negative for the receding target", lines)
        self.assertIn("peak at 163.4 ns and -50.0 Hz, within one bin of 179.8 ns and -43.3 Hz", lines)
        self.assertIn("resolution 54.5 ns over 18.36 MHz and 25.0 Hz over 8 slots", lines)
        self.assertIn("the loader refuses one changed byte, a missing array, a partial write, a swapped axis", lines)
        manifest = json.loads((out / "reference_pair/store/manifest.json").read_text())
        self.assertEqual(sorted(manifest["grants"]),
                         sorted([p.name for p in (out / "grant_fixture").iterdir()] + ["demo-51prb-4rx"]))
        self.assertTrue(all(entry["x"]["x_sha256"] for entry in manifest["grants"].values()))
        stored = grants.open_store(out / "grant_fixture/demo-51prb-4rx.s19/store")
        self.assertEqual(stored.grant.slot, 19)
        comparison = json.loads((out / "comparison.json").read_text())
        self.assertLess(comparison["reference-pair"]["echo_relative_error"], 1e-4)
        self.assertEqual(json.loads((out / "manifest.json").read_text())["judgment"], "pass")
        with self.assertRaises(SystemExit) as refused, contextlib.redirect_stderr(io.StringIO()):
            self.driver.main(["--config", str(ROOT / "configs/integration/episode_pair.json"),  # never overwritten
                              "--grant-store", str(self.store), "--output-dir", str(out)])
        self.assertEqual(refused.exception.code, 2)

    def test_a_channel_that_moves_the_wrong_way_fails_the_sign_checks(self):
        status, grade, failed, _ = self.run_driver("approaching", pair_fault="approaching")
        self.assertEqual(status, 1)
        self.assertEqual(failed, ["reference-pair delay-Doppler on the pilot mask",
                                  "reference-pair phase at unequal intervals", "reference-pair scattered path",
                                  "reference-pair second reader"])
        self.assertEqual((grade["judgment"], grade["cases_passed"]), ("fail", 3))

    def test_a_reversed_receive_order_and_a_solver_disagreement_fail_their_checks(self):
        status, _, failed, _ = self.run_driver("order", pair_fault="rx_order")
        self.assertEqual((status, failed), (1, ["reference-pair scattered path", "reference-pair second reader"]))
        status, _, failed, _ = self.run_driver("solver", pair_fault="solver")
        self.assertEqual((status, failed), (1, ["reference-pair solver agreement"]))

    def test_a_generator_that_cannot_run_is_a_failed_check(self):
        status, grade, failed, out = self.run_driver("raise", pair_fault="raise")
        self.assertEqual((status, failed), (1, ["reference-pair export"]))
        self.assertIn("the ray tracer is not available", (out / "reference_pair/export-error.txt").read_text())
        self.assertEqual(grade["checks_failed"], 1)

    def test_grant_fixture_faults(self):
        slots = [f"grant-fixture slot {slot} grant" for slot in (17, 18, 19, 8, 9)]
        for fault in ("crc", "pilots", "exit"):
            status, _, failed, out = self.run_driver(f"adapter-{fault}", adapter_fault=fault)
            with self.subTest(fault=fault):
                self.assertEqual(status, 1)
                self.assertEqual([name for name in failed if name.startswith("grant-fixture")], sorted(slots))
        self.assertIn("host receiver CRC FAIL", (self.root / "adapter-crc/grade.txt").read_text())
        self.assertIn("the pilot at symbol 2", (self.root / "adapter-pilots/grade.txt").read_text())
        self.assertIn("exited with status 3", (self.root / "adapter-exit/grade.txt").read_text())
        status, _, failed, out = self.run_driver("no-record", store=self.root / "empty")
        self.assertEqual(status, 1)
        self.assertIn("grant-fixture base grant record", failed)
        self.assertIn("reference-pair stored pair", failed)  # one grant of the pair has no archived transmit grid
        self.assertIn("no verified Sprint 2.1 record of demo-51prb-4rx", (out / "grade.txt").read_text())

    def test_an_unusable_configuration_grades_nothing(self):
        config = json.loads((ROOT / "configs/integration/episode_pair.json").read_text())
        for change in (lambda c: c.update(selector="grant_fixture"), lambda c: c["tolerance"].pop("phase_rad"),
                       lambda c: c.update(variant="scalar_rgb"), lambda c: c["pair"].pop("expect")):
            edited = json.loads(json.dumps(config))
            change(edited)
            path = self.root / "config.json"
            path.write_text(json.dumps(edited))
            with self.assertRaises(SystemExit) as refused, contextlib.redirect_stderr(io.StringIO()):
                self.driver.main(["--config", str(path), "--grant-store", str(self.store),
                                  "--output-dir", str(self.root / "never")])
            self.assertEqual(refused.exception.code, 2)
        self.assertFalse((self.root / "never").exists())


if __name__ == "__main__":
    unittest.main()
