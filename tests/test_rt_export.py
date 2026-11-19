"""Offline checks of the Sionna RT output contract. No ray tracing is run."""

import hashlib
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_rt_episode.py"
spec = importlib.util.spec_from_file_location("rt_export", SCRIPT)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


@unittest.skipIf(np is None, "optional NumPy runtime required for RT array checks")
class RTExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        antenna, time, frequency = np.indices((4, exporter.TIME_COUNT, len(exporter.OFFSETS_HZ)))
        expected = (100 * antenna + 10 * time + frequency) + 1j * (100 * time - frequency)
        self.reference = expected[:, None, :, :]
        self.rt_shape = self.reference[None, :, None, :, :, :]

    def test_manifest_and_physical_axis_order(self):
        descriptor = exporter.store_cfr(self.root, "h.npy", self.rt_shape)
        target = self.root / descriptor["file"]
        data = np.load(target, allow_pickle=False)
        np.testing.assert_array_equal(data, self.reference)
        self.assertEqual(data[3, 0, 2, 11], 331 + 189j)
        self.assertEqual(descriptor["axes"], ["rx_element", "tx_element", "time", "frequency_offset"])
        self.assertEqual(descriptor["shape"], [4, 1, 4, 12])
        self.assertEqual(descriptor["dtype"], "<c8")
        self.assertEqual(descriptor["sha256"], hashlib.sha256(target.read_bytes()).hexdigest())

    def test_paired_manifest_and_atomic_failure(self):
        destination = self.root / "pair"
        software = {"fixture": "known arrays"}
        with self.assertRaisesRegex(ValueError, "distinct target-present"):
            exporter.write_pair(destination, self.rt_shape, self.rt_shape, software, "fixture")
        self.assertFalse(destination.exists())
        exporter.write_pair(destination, self.rt_shape, self.rt_shape + 0.125j, software, "fixture")
        manifest = json.loads((destination / "manifest.json").read_text())
        self.assertEqual(manifest["grid"]["episode_times_ns"], [0, 500000, 1000000, 1500000])
        self.assertEqual(manifest["grid"]["frequency_offsets_hz"], list(range(-180000, 180000, 30000)))
        self.assertFalse(manifest["grid"]["normalize_gain"])
        self.assertFalse(manifest["grid"]["normalize_delays"])
        self.assertEqual([c["target_present"] for c in manifest["cases"]], [False, True])
        for case in manifest["cases"]:
            channel = case["h"]
            raw = (destination / channel["file"]).read_bytes()
            self.assertEqual(channel["sha256"], hashlib.sha256(raw).hexdigest())
        with self.assertRaises(FileExistsError):
            exporter.write_pair(destination, self.rt_shape, self.rt_shape, software, "fixture")

    def test_rejects_ambiguous_or_invalid_cfr(self):
        corrupt = self.rt_shape.copy()
        corrupt.flat[0] = complex(float("nan"), 0)
        for data in (self.rt_shape[0], self.rt_shape.swapaxes(1, 2), self.rt_shape.real, corrupt):
            with self.subTest(shape=data.shape, dtype=data.dtype):
                with self.assertRaises(ValueError):
                    exporter.store_cfr(self.root, "h.npy", data)
                self.assertFalse((self.root / "h.npy").exists())


class DirectPathReferenceTests(unittest.TestCase):
    def test_free_space_reference_matches_the_hand_calculation(self):
        # 20 m ground range and 1.5 m height difference at 3.5 GHz, isotropic antennas.
        reference = exporter.direct_path_reference()
        self.assertAlmostEqual(reference["amplitude"], 3.398558642879076e-4, places=15)
        self.assertAlmostEqual(reference["delay_ns"], 66.90018572825998, places=9)

    @unittest.skipIf(np is None, "optional NumPy runtime required for RT array checks")
    def test_direct_path_check_accepts_free_space_and_rejects_wrong_physics(self):
        reference = exporter.direct_path_reference()
        offsets = np.array(exporter.OFFSETS_HZ, dtype=np.float64)

        def channel(amplitude=reference["amplitude"], delay_ns=reference["delay_ns"]):
            tones = amplitude * np.exp(-2j * math.pi * offsets * delay_ns * 1e-9)
            shape = (1, 4, 1, 1, exporter.TIME_COUNT, len(offsets))
            return np.broadcast_to(tones, shape).astype(np.complex64)

        measured = exporter.check_direct_path(channel())
        self.assertLess(measured["amplitude_relative_error"], 1e-6)
        self.assertLess(measured["delay_ns_max_error"], 0.01)
        self.assertAlmostEqual(measured["delay_ns_observed"], reference["delay_ns"], places=2)
        nonfinite = channel().copy()
        nonfinite.flat[0] = complex(float("inf"), 0)
        cases = ((channel(amplitude=1.001 * reference["amplitude"]), "gain differs"),
                 (channel(amplitude=0.0), "gain differs"),
                 (np.conj(channel()), "delay differs"),
                 (channel(delay_ns=reference["delay_ns"] + 1.0), "delay differs"),
                 (channel(delay_ns=0.0), "delay differs"),
                 (nonfinite, "finite complex"),
                 (channel()[0], "finite complex"),
                 (channel().real, "finite complex"))
        for data, message in cases:
            with self.subTest(message=message, shape=data.shape), self.assertRaisesRegex(ValueError, message):
                exporter.check_direct_path(data)


if __name__ == "__main__":
    unittest.main()
