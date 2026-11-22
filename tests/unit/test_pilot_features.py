#!/usr/bin/env python3
"""Unit gate of Sprint 3.3 (selector pilot_features): python tests/unit/test_pilot_features.py -v

The reference of the radar cube, of the delay-Doppler view and of the compact features is checked
on records whose answer can be worked out by hand: one return at an interior cell of a complete
lattice, its conjugate, a masked element behind a sentinel, records of the future and of an
earlier epoch, seven of eight observations. The reference pair of Sprint 2.2 is then synthesized
from its analytic model on the lattice of the demonstration and must peak where the card says.
The evaluator side cuts the same windows offline and computes the same view by explicit sums.

NumPy is needed. The last class attaches the reference sensor to a native producer of the ring,
and needs cmake and a C++17 compiler for that. The import rule of the sensing package has its own
module, tests/unit/test_core_imports.py.
"""

import contextlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    np = None

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ocudu_cuda_isac.sensing import obs  # noqa: E402
from ocudu_cuda_isac.sensing import pilot_ls as estimator  # noqa: E402
from ocudu_cuda_isac.sensing.sources import pusch_dmrs  # noqa: E402

if np is not None:
    from ocudu_cuda_isac.episodes import episode as episodes  # noqa: E402
    from ocudu_cuda_isac.episodes import pilot_features as evaluator  # noqa: E402
    from ocudu_cuda_isac.sensing import features  # noqa: E402
    from ocudu_cuda_isac.sensing.features import FeaturesError  # noqa: E402

SCS = 30
SYMBOLS = (2, 11)
EPISODE = 0x0123456789ABCDEF
NEEDS = "the reference of the features is NumPy on the CPU"


def make_record(sequence, slot, values, bits=None, epoch=1, episode=EPISODE, antenna_map=None, host=b"", **changes):
    """An obs/v1 record of the layout ocudu_o2_comb_c64 with the header a producer of Sprint 3.2 writes."""
    rx, rows, pilots = values.shape
    bits = np.ones(values.shape, dtype=bool) if bits is None else bits
    mask = np.packbits(bits.reshape(-1), bitorder="little").tobytes()
    payload = np.ascontiguousarray(values, dtype="<c8").tobytes()
    header = obs.new_header(
        epoch=epoch, sequence=sequence, episode=episode, opportunity=0x1000 + sequence, grant=5, absolute_slot=slot,
        cell=1, ue=0x4601, stage=2, source=1, layout=2, dtype=1, producer_steps=1, scale=0.998815, rank=3,
        shape=[rx, rows, pilots], strides=[rows * pilots, pilots, 1], first_subcarrier=0, subcarrier_step=2,
        subcarrier_count=pilots, dmrs_symbols=sum(1 << symbol for symbol in SYMBOLS[:rows]), rx_count=rx, layer_count=1,
        antenna_map=antenna_map or list(range(rx)), mask_bytes=len(mask), payload_bytes=len(payload), host_bytes=len(host),
        time_unit=1, digest_kind=0, evidence=2, port=0,
        t_first_sample=features.symbol_time_tc(SCS, slot, SYMBOLS[0]),
        t_last_sample=features.symbol_time_tc(SCS, slot, SYMBOLS[rows - 1]) + estimator.useful_tc(SCS),
        t_ready_ns=1000 + sequence, t_publish_ns=2000 + sequence, valid=int(bits.sum()))
    header.update(changes)
    return obs.Record(obs.unpack_header(obs.pack_header(header)), host, mask, payload)


def small_spec(**view):
    """The specification of the demonstration on a lattice of 16 pilots: eight periods, slot 7, symbol row 0."""
    spec = features.demo_spec(delay_bins=8)
    spec["view"].update(view)
    return spec


def lattice(delay_bin=5, doppler_bin=2, pilots=16, periods=8, rx=2, static=3.0, conjugate=False, slots=(7, 8, 9)):
    """Records of one return at an interior cell of a complete lattice, on a static path at delay bin 1.

    Every slot of the burst and both reference symbols carry the return with the phase of their
    own time, so a transform over any subset sees one Doppler shift.
    """
    k = np.arange(pilots)
    interval = features.period_tc(SCS, 10) / features.TC_PER_SECOND
    nu = doppler_bin / (periods * interval) * (-1 if conjugate else 1)
    origin = features.symbol_time_tc(SCS, slots[0], SYMBOLS[0])
    records = []
    for period in range(periods):
        for slot in slots:
            absolute = 10 * period + slot
            values = np.zeros((rx, 2, pilots), dtype=np.complex128)
            for row, symbol in enumerate(SYMBOLS):
                seconds = (features.symbol_time_tc(SCS, absolute, symbol) - origin) / features.TC_PER_SECOND
                for plane in range(rx):
                    values[plane, row] = static * np.exp(-2j * np.pi * k / pilots) \
                        + np.exp(-2j * np.pi * k * delay_bin / pilots) * np.exp(2j * np.pi * nu * seconds + 0.7j * plane)
            records.append(make_record(len(records), absolute, values))
    return records


def complete(lines):
    return [line for line in lines if line["status"] == "ok"]


def vector(line):
    return dict(zip(line["features"]["names"], line["features"]["values"]))


@unittest.skipIf(np is None, NEEDS)
class SymbolTimeTests(unittest.TestCase):
    """Slow time is the start of the useful part of a symbol, by the rule the sensing source records."""

    def test_the_time_of_a_symbol_is_that_of_the_source_record(self):
        for scs in (15, 30, 60):
            for slot in (0, 1, 7, 17, 19, 20 * 1024 + 7):
                times = pusch_dmrs.symbol_times(scs, slot)
                for entry in times:
                    self.assertEqual(features.symbol_time_tc(scs, slot, entry["symbol"]),
                                     estimator.slot_start_tc(scs, slot) + entry["useful_start_tc"], (scs, slot, entry))
        # The numbers of the episode interface: the two DM-RS symbols of a slot, and one TDD period.
        self.assertEqual(features.symbol_time_tc(30, 7, 11) - features.symbol_time_tc(30, 7, 2), 631296)
        self.assertEqual(features.symbol_time_tc(30, 17, 2) - features.symbol_time_tc(30, 7, 2), features.period_tc(30, 10))
        self.assertEqual(features.period_tc(30, 10) / features.TC_PER_SECOND, 0.005)
        with self.assertRaisesRegex(FeaturesError, "from 0 to 13"):
            features.symbol_time_tc(30, 7, 14)


@unittest.skipIf(np is None, NEEDS)
class SpecTests(unittest.TestCase):
    def test_the_demonstration_spec_and_its_registry_entries(self):
        spec = features.demo_spec()
        self.assertEqual(spec["sequence"], {"slots_in_period": [7], "reference_symbols": [0]})
        self.assertEqual((spec["window"]["periods"], spec["window"]["hop_periods"]), (8, 8))
        self.assertEqual(features.check_spec(json.loads(json.dumps(spec))), spec)
        self.assertEqual(len(features.spec_sha256(spec)), 64)
        self.assertNotEqual(features.spec_sha256(spec), features.spec_sha256(small_spec()))
        steps = features.resolve(spec)
        self.assertEqual({stage: entry["entry"] for stage, entry in steps.items()},
                         {"cube": "comb_cube", "static": "window_mean", "delay": "ifft", "doppler": "fft",
                          "features": "dd_compact"})
        # Every entry of the registry states where it runs, and all of this ticket run in NumPy on the CPU.
        for stage, entries in features.REGISTRY.items():
            for name, entry in entries.items():
                self.assertEqual((entry["device"], entry["library"], entry["runtime"]), ("cpu", "numpy", "reference"),
                                 (stage, name))
                self.assertTrue(entry["does"])

    def test_refusals(self):
        def changed(path, value):
            spec = features.demo_spec()
            target = spec
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            return spec

        ndft = changed(("steps", "doppler"), "ndft")
        cases = (
            (changed(("schema",), "pilot-features-spec/v2"), "unsupported specification schema"),
            (changed(("spec_id",), ""), "spec_id"),
            (changed(("input", "stage"), "O9"), "unknown receiver stage"),
            (changed(("input", "layout"), "ocudu_o3_fp16"), "no cube is defined here for the layout"),
            (changed(("radio", "scs_khz"), 45), "subcarrier spacing"),
            (changed(("radio", "period_slots"), 5), "whole number of subframes"),
            (changed(("sequence", "slots_in_period"), [10]), "from 0 to 9"),
            (changed(("sequence", "slots_in_period"), [8, 7]), "rising list"),
            (changed(("sequence", "reference_symbols"), []), "rising list"),
            (changed(("window", "periods"), 0), "window.periods"),
            (changed(("window", "hop_periods"), 0), "window.hop_periods"),
            (changed(("window", "min_valid_fraction"), 0), "min_valid_fraction"),
            (changed(("steps", "static"), "mti_2"), "unknown static step 'mti_2' (registered: none, window_mean)"),
            (changed(("steps", "doppler"), "music"), "unknown doppler step"),
            (changed(("view", "delay_bins"), 0), "view.delay_bins"),
            (changed(("view", "floor_db"), 5), "view.floor_db"),
            # The fft step needs one sample per period.
            (changed(("sequence", "slots_in_period"), [7, 8, 9]), "homogeneous sequence"),
            (changed(("sequence", "reference_symbols"), [0, 1]), "homogeneous sequence"),
            (changed(("view", "doppler_grid"), {"bins": 9, "step_hz": 25.0}), "must be null"),
            (changed(("window", "periods"), 1), "needs two samples"),
            (ndft, "view.doppler_grid of the ndft step"),
            (dict(ndft, view=dict(ndft["view"], doppler_grid={"bins": 1, "step_hz": 25.0})), "doppler_grid.bins"),
            (dict(ndft, view=dict(ndft["view"], doppler_grid={"bins": 9, "step_hz": 0})), "step_hz must be positive"),
            ({key: value for key, value in features.demo_spec().items() if key != "view"}, "requires exactly"),
        )
        for spec, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(FeaturesError, message.replace("(", r"\(")
                                                                       .replace(")", r"\)")):
                features.check_spec(spec)
        with self.assertRaisesRegex(FeaturesError, "homogeneous sequence"):
            features.Stream(changed(("sequence", "slots_in_period"), [7, 8, 9]))


@unittest.skipIf(np is None, NEEDS)
class KnownReturnTests(unittest.TestCase):
    """One return at an interior cell of a complete lattice: the peak, the signs and the baked vector."""

    def test_the_peak_lies_at_the_declared_cell_and_the_vector_is_the_baked_one(self):
        lines, views, stream = features.run_records(lattice(), small_spec())
        self.assertEqual({key: value for key, value in stream.counts.items() if value},
                         {"records": 24, "passed": 16, "ok": 1, "insufficient_history": 7})
        line = complete(lines)[0]
        axes, view = line["axes"], line["view"]
        # 16 pilots 60 kHz apart: 1,041.67 ns per bin, 16.67 us unambiguous. Eight samples 5 ms apart:
        # 25 Hz per bin, from -100 to +75 Hz.
        self.assertAlmostEqual(axes["delay"]["resolution_ns"], 1e9 / (16 * 60e3), places=6)
        self.assertAlmostEqual(axes["delay"]["unambiguous_ns"], 1e9 / 60e3, places=6)
        self.assertEqual((axes["doppler"]["bins"], axes["doppler"]["resolution_hz"], axes["doppler"]["unambiguous_hz"]),
                         ([-4, -3, -2, -1, 0, 1, 2, 3], 25.0, [-100.0, 75.0]))
        self.assertEqual(axes["doppler"]["sample_interval_s"], 0.005)
        self.assertEqual((axes["order"], axes["rx"]["elements"]), (["frequency", "slow_time", "rx"], [0, 1]))
        self.assertEqual((view["peak"]["delay_bin"], view["peak"]["doppler_bin"]), (5, 2))
        self.assertEqual((view["static_peak"]["delay_bin"], view["shape"], view["axes"]), (1, [8, 8], ["delay", "doppler"]))
        self.assertAlmostEqual(view["excess_delay_ns"], 4 * 1e9 / (16 * 60e3), places=6)
        # The return has amplitude 1 on two elements: the peak holds |1|^2 + |1|^2, and nothing else moves.
        self.assertAlmostEqual(view["peak"]["power"], 2.0, places=5)
        self.assertAlmostEqual(view["leakage"]["peak_fraction"], 1.0, places=9)
        self.assertEqual(views[0][1].shape, (8, 8))
        # The baked vector. Moving over static power is 1 / 9. A view without noise reads the floor.
        baked = {"peak_to_median_db": 120.0, "moving_to_static_db": 10 * math.log10(1 / 9),
                 "peak_delay_ns": 5 * 1e9 / (16 * 60e3), "peak_doppler_hz": 50.0, "doppler_centroid_hz": 50.0,
                 "doppler_spread_hz": 0.0, "profile_db@-100Hz": 0.0, "profile_db@-75Hz": 0.0, "profile_db@-50Hz": 0.0,
                 "profile_db@-25Hz": 0.0, "profile_db@+25Hz": 0.0, "profile_db@+50Hz": 120.0, "profile_db@+75Hz": 0.0}
        self.assertEqual(line["features"]["names"], list(baked))
        for name, value in vector(line).items():
            self.assertAlmostEqual(value, baked[name], places=4, msg=name)
        self.assertAlmostEqual(vector(line)["moving_to_static_db"], -9.542425, places=5)
        # What the window is made of.
        self.assertEqual([item["sequence"] for item in line["contributing"]], [0, 3, 6, 9, 12, 15, 18, 21])
        self.assertEqual(line["window"], {"id": f"1/{EPISODE:016x}/7", "epoch": 1, "episode": f"{EPISODE:016x}",
                                          "first_period": 0, "last_period": 7})
        self.assertEqual(line["validity"], {"elements": 256, "valid": 256, "holes": 0, "fill": 1.0, "lines_with_holes": 0,
                                            "interpolation": "none: a hole is zero in both transforms"})
        first = features.symbol_time_tc(SCS, 7, 2)
        self.assertEqual((line["time"]["t_first_sample"], line["time"]["t_last_sample"]),
                         (first, first + 7 * features.period_tc(SCS, 10) + estimator.useful_tc(SCS)))
        self.assertAlmostEqual(line["time"]["acquisition_us"], 35000 + 1e6 * 65536 / features.TC_PER_SECOND, places=6)
        self.assertEqual((line["runtime"], line["input"]["stage"], line["input"]["evidence"]), ("reference", "O2", ["timed"]))

    def test_conjugating_the_slow_time_phase_flips_the_doppler_sign(self):
        flipped = complete(features.run_records(lattice(conjugate=True), small_spec())[0])[0]
        self.assertEqual((flipped["view"]["peak"]["delay_bin"], flipped["view"]["peak"]["doppler_bin"]), (5, -2))
        self.assertAlmostEqual(vector(flipped)["peak_doppler_hz"], -50.0)
        self.assertAlmostEqual(vector(flipped)["doppler_centroid_hz"], -50.0, places=6)
        self.assertAlmostEqual(vector(flipped)["profile_db@-50Hz"], 120.0, places=4)
        # The delay does not move with the Doppler sign, and a later arrival lies at a higher bin.
        for delay_bin in (2, 6):
            line = complete(features.run_records(lattice(delay_bin=delay_bin), small_spec())[0])[0]
            self.assertEqual(line["view"]["peak"]["delay_bin"], delay_bin)

    def test_every_interior_cell_is_found_within_one_bin(self):
        for delay_bin in (2, 3, 6):
            for doppler_bin in (-3, -1, 1, 3):
                line = complete(features.run_records(lattice(delay_bin, doppler_bin), small_spec())[0])[0]
                self.assertEqual((line["view"]["peak"]["delay_bin"], line["view"]["peak"]["doppler_bin"]),
                                 (delay_bin, doppler_bin))

    def test_the_cube_is_frequency_by_slow_time_by_array_element(self):
        records = lattice()
        stream = features.Stream(small_spec())
        for record in records:
            stream.push(record)
        cube, valid = stream.last["cube"], stream.last["valid"]
        self.assertEqual((cube.shape, valid.shape, cube.dtype, bool(valid.all())), ((16, 8, 2), (16, 8, 2), np.complex128, True))
        values, _ = features.planes(records[3])
        np.testing.assert_array_equal(cube[:, 1, :], values[:, 0, :].T.astype(np.complex128))
        # A reversed antenna map: plane 0 holds element 1. The cube lists the elements in rising order.
        swapped = []
        for record in records:
            values, _ = features.planes(record)
            swapped.append(make_record(record.sequence, record.header["absolute_slot"], values[::-1], antenna_map=[1, 0]))
        other = features.Stream(small_spec())
        for record in swapped:
            other.push(record)
        np.testing.assert_array_equal(other.last["cube"], cube)
        self.assertEqual(other.last["axes"]["rx"]["elements"], [0, 1])

    def test_actual_times_of_every_slot_and_symbol_find_the_same_return(self):
        spec = small_spec(doppler_grid={"bins": 17, "step_hz": 12.5})
        spec["sequence"] = {"slots_in_period": [7, 8, 9], "reference_symbols": [0, 1]}
        spec["steps"]["doppler"] = "ndft"
        for conjugate, expected in ((False, 50.0), (True, -50.0)):
            lines, _, stream = features.run_records(lattice(conjugate=conjugate), spec)
            self.assertEqual((stream.counts["ok"], stream.counts["insufficient_history"], stream.counts["passed"]),
                             (1, 23, 0))
            line = complete(lines)[0]
            self.assertEqual((line["view"]["peak"]["delay_bin"], line["view"]["peak"]["doppler_hz"]), (5, expected))
            self.assertEqual((line["history"], len(line["contributing"]), line["axes"]["slow_time"]["samples"]),
                             ({"have": 48, "need": 48}, 24, 48))
            # 48 samples over 36.0 ms: the two symbols of a slot are 321 us apart, the slots 500 us.
            seconds = line["axes"]["slow_time"]["s"]
            self.assertAlmostEqual(seconds[1], 631296 / features.TC_PER_SECOND)
            self.assertAlmostEqual(seconds[2], 0.0005)
            self.assertIsNone(line["axes"]["doppler"]["sample_interval_s"])


@unittest.skipIf(np is None, NEEDS)
class CausalityTests(unittest.TestCase):
    """What may and what may not change the features of a window."""

    def setUp(self):
        self.records = lattice()
        self.spec = small_spec()
        self.reference = features.run_records(self.records, self.spec)[0]

    def changed(self, record, values=None, bits=None, **changes):
        base, mask = features.planes(record)
        header = dict(record.header)
        made = make_record(header["sequence"], changes.pop("slot", header["absolute_slot"]),
                           base if values is None else values, mask if bits is None else bits,
                           epoch=changes.pop("epoch", header["epoch"]), episode=changes.pop("episode", header["episode"]),
                           host=changes.pop("host", record.host), **changes)
        return made

    def test_a_masked_element_behind_a_sentinel_changes_nothing(self):
        outcomes = []
        for sentinel in (0.0, 1e30, complex(float("nan"), float("inf"))):
            records = list(self.records)
            values, valid = (array.copy() for array in features.planes(records[9]))
            values = values.astype(np.complex128)
            values[1, 0, 4], valid[1, 0, 4] = sentinel, False
            with np.errstate(over="ignore", invalid="ignore"):
                records[9] = self.changed(records[9], values, valid)
            lines, views, _ = features.run_records(records, self.spec)
            outcomes.append(([features.comparable(line) for line in lines], views[0][1].copy()))
        for lines, power in outcomes[1:]:
            self.assertEqual(lines, outcomes[0][0])
            np.testing.assert_array_equal(power, outcomes[0][1])
        line = complete(outcomes[0][0])[0]
        self.assertEqual(line["validity"], {"elements": 256, "valid": 255, "holes": 1, "fill": 255 / 256,
                                            "lines_with_holes": 1,
                                            "interpolation": "none: a hole is zero in both transforms"})
        # The hole is seen: the return stays in its cell and leaks a little, and the static path, three
        # times as strong, does not leak at all, because the mean of a line is that of its valid samples.
        self.assertEqual((line["view"]["peak"]["delay_bin"], line["view"]["peak"]["doppler_bin"]), (5, 2))
        self.assertLess(line["view"]["leakage"]["peak_fraction"], 0.999)
        self.assertGreater(line["view"]["leakage"]["peak_fraction"], 0.98)
        self.assertAlmostEqual(vector(line)["moving_to_static_db"], 10 * math.log10(1 / 9), places=1)
        stream = features.Stream(self.spec)
        for record in self.records[:9] + [self.changed(self.records[9], bits=np.where(
                np.arange(64).reshape(2, 2, 16) == 36, False, True))] + self.records[10:]:
            stream.push(record)
        cube, valid = stream.last["cube"], stream.last["valid"]
        self.assertEqual((cube[4, 3, 1], bool(valid[4, 3, 1]), int((~valid).sum())), (0j, False, 1))
        # Too many holes: the window is invalid and carries no vector.
        records = list(self.records)
        values, valid = features.planes(records[0])
        records[0] = self.changed(records[0], bits=np.zeros(valid.shape, dtype=bool))
        records[3] = self.changed(records[3], bits=np.zeros(valid.shape, dtype=bool))
        lines = features.run_records(records, self.spec)[0]
        self.assertEqual((lines[-1]["status"], "features" in lines[-1]), ("invalid", False))
        self.assertIn("192 of 256 elements of the cube are valid, below the share 0.9", lines[-1]["reason"])

    def test_future_samples_cannot_change_a_window(self):
        later = [make_record(24 + index, 80 + 10 * (index // 3) + 7 + index % 3,
                             np.full((2, 2, 16), 1e6 * (index + 1) * (1 + 1j))) for index in range(24)]
        lines = features.run_records(self.records + later, self.spec)[0]
        self.assertEqual([features.comparable(line) for line in lines[:8]],
                         [features.comparable(line) for line in self.reference])
        self.assertEqual([line["status"] for line in lines[8:]], ["insufficient_history"] * 7 + ["ok"])
        self.assertEqual(lines[-1]["window"]["first_period"], 8)
        # A stream answers at the record that completes the window, and not later.
        stream = features.Stream(self.spec)
        answers = [stream.push(record) for record in self.records]
        self.assertEqual([len(answer) for answer in answers], [1, 0, 0] * 8)
        self.assertEqual(answers[21][0]["status"], "ok")
        self.assertEqual(answers[21][0]["trigger"], {"epoch": 1, "sequence": 21, "opportunity": f"{0x1000 + 21:016x}",
                                                     "absolute_slot": 77})
        # A contributing sample does change it.
        records = list(self.records)
        records[21] = self.changed(records[21], values=features.planes(records[21])[0] * 1.5)
        other = complete(features.run_records(records, self.spec)[0])[0]
        self.assertNotEqual(other["features"]["values"], complete(self.reference)[0]["features"]["values"])
        # A record of the burst that the sequence does not select changes nothing either.
        records = list(self.records)
        records[4] = self.changed(records[4], values=np.full((2, 2, 16), 1e9 + 0j))
        self.assertEqual([features.comparable(line) for line in features.run_records(records, self.spec)[0]],
                         [features.comparable(line) for line in self.reference])

    def test_fields_outside_the_input_list_cannot_change_a_feature(self):
        # A host block, a digest, a grant key, a cell, the producer's steps and its counts: none is read.
        records = [self.changed(record, host=b"truth: target present at 10, 25, 1.5", grant=99, cell=7, ue=3,
                                producer_steps=15, digest_kind=1, digest=b"\x55" * 32, valid=0, reserved=77)
                   for record in self.records]
        lines = features.run_records(records, self.spec)[0]
        self.assertEqual([features.comparable(line) for line in lines], [features.comparable(line) for line in self.reference])
        self.assertTrue(set(features.INPUT_FIELDS) <= {name for name, _, _ in obs.FIELDS})
        for hidden in ("grant", "cell", "ue", "digest", "digest_kind", "valid", "host_bytes", "producer_steps"):
            self.assertNotIn(hidden, features.INPUT_FIELDS)
        # An attribute that a careless producer attaches is not looked at.
        for record in records:
            record.truth = {"present": True}
        self.assertEqual([features.comparable(line) for line in features.run_records(records, self.spec)[0]],
                         [features.comparable(line) for line in self.reference])
        # The opportunity key is a join key: it names the contributing records and enters no value.
        renamed = [self.changed(record, opportunity=0x9000 + record.sequence) for record in self.records]
        line = complete(features.run_records(renamed, self.spec)[0])[0]
        self.assertEqual(line["features"], complete(self.reference)[0]["features"])
        self.assertEqual(line["contributing"][0]["opportunity"], f"{0x9000:016x}")

    def test_an_earlier_epoch_and_another_episode_cannot_reach_a_window(self):
        # Seven observations of epoch 1, then all eight of epoch 2 on later slots.
        old = [self.changed(record, values=np.full((2, 2, 16), 50 + 50j)) for record in self.records[:21]]
        new = [self.changed(record, epoch=2, slot=record.header["absolute_slot"] + 80) for record in self.records]
        for index, record in enumerate(new):
            record.header["sequence"] = index
        lines, _, stream = features.run_records(old + new, self.spec)
        alone = features.run_records(new, self.spec)[0]
        self.assertEqual(stream.counts["resets"], 1)
        self.assertEqual([line["status"] for line in lines], ["insufficient_history"] * 14 + ["ok"])
        self.assertEqual([line["history"]["have"] for line in lines[7:]], [1, 2, 3, 4, 5, 6, 7, 8])
        for mine, other in zip(lines[7:], alone):
            mine, other = features.comparable(mine), features.comparable(other)
            self.assertEqual({key: value for key, value in mine.items() if key != "line"},
                             {key: value for key, value in other.items() if key != "line"})
        self.assertEqual({item["sequence"] for item in lines[-1]["contributing"]}, {0, 3, 6, 9, 12, 15, 18, 21})
        self.assertEqual(lines[-1]["window"]["epoch"], 2)
        # Another episode key starts a history too, and so does a slot that goes back.
        for change in (dict(episode=EPISODE + 1), dict(slot=7)):
            records = self.records[:12] + [self.changed(self.records[12], **change)] + self.records[13:]
            lines, _, stream = features.run_records(records, self.spec)
            self.assertEqual((stream.counts["ok"], stream.counts["resets"]), (0, 2 if "episode" in change else 1), change)

    def test_seven_of_eight_observations_are_a_warm_up(self):
        stream = features.Stream(self.spec)
        lines = [line for record in self.records[:21] for line in stream.push(record)]
        self.assertEqual([line["status"] for line in lines], ["insufficient_history"] * 7)
        self.assertEqual([line["history"] for line in lines], [{"have": have, "need": 8} for have in range(1, 8)])
        for line in lines:
            self.assertEqual(line["reason"], "the window is filling")
            self.assertEqual(line["window"]["last_period"], 7)
            self.assertFalse({"features", "view", "axes"} & set(line))
        self.assertEqual((stream.open_samples(), stream.views, stream.last), ((7, 8), [], None))
        self.assertEqual(stream.push(self.records[21])[0]["status"], "ok")
        self.assertEqual(stream.open_samples(), (0, 8))

    def test_a_missing_observation_is_a_gap_and_the_next_window_is_whole(self):
        second = [make_record(24 + index, 80 + 10 * (index // 3) + 7 + index % 3, features.planes(record)[0])
                  for index, record in enumerate(self.records)]
        # Observation 9, slot 37, was dropped at the ring.
        lines = features.run_records(self.records[:9] + self.records[10:] + second, self.spec)[0]
        self.assertEqual([line["status"] for line in lines], ["insufficient_history"] * 6 + ["gap"]
                         + ["insufficient_history"] * 7 + ["ok"])
        gap = lines[6]
        self.assertEqual((gap["missing"], gap["history"], gap["trigger"]["sequence"]),
                         ([{"period": 3, "slot_in_period": 7, "reference_symbol": 0}], {"have": 7, "need": 8}, 21))
        self.assertIn("1 of 8 samples of the window never arrived", gap["reason"])
        self.assertNotIn("features", gap)
        self.assertEqual(lines[-1]["window"]["first_period"], 8)
        # The last observation of a window was dropped: the first later record of the sequence closes it.
        lines = features.run_records(self.records[:21] + second, self.spec)[0]
        self.assertEqual([line["status"] for line in lines][6:9], ["insufficient_history", "gap", "insufficient_history"])
        self.assertEqual((lines[7]["trigger"], lines[7]["missing"][0]["period"], lines[7]["window"]["last_period"]),
                         (None, 7, 7))
        self.assertEqual(lines[-1]["status"], "ok")

    def test_an_answer_is_stamped_when_it_is_complete_and_a_sensor_warms_up_first(self):
        ticks = []

        def clock():
            ticks.append(len(ticks))
            return ticks[-1]

        real = features.view_of
        seen = []

        def view_of(*arguments):
            seen.append(len(ticks))  # how often the clock was read before this view was computed
            return real(*arguments)

        features.view_of = view_of
        try:
            lines = features.run_records(self.records, self.spec, clock)[0]
        finally:
            features.view_of = real
        # Eight answers, eight readings. The reading of the complete window follows its view.
        self.assertEqual([line["t_features_ns"] for line in lines], list(range(8)))
        self.assertEqual(seen, [7])
        self.assertTrue(all(line["t_features_ns"] is None for line in self.reference))
        # The warm-up runs every step of the specification once and keeps nothing.
        self.assertGreater(features.warm_up(self.spec), 0.0)
        ndft = small_spec(doppler_grid={"bins": 17, "step_hz": 12.5})
        ndft["sequence"] = {"slots_in_period": [7, 8, 9], "reference_symbols": [0, 1]}
        ndft["steps"]["doppler"] = "ndft"
        self.assertGreater(features.warm_up(ndft), 0.0)
        with self.assertRaisesRegex(FeaturesError, "unsupported specification schema"):
            features.warm_up(dict(self.spec, schema="other"))
        self.assertEqual([features.comparable(line) for line in features.run_records(self.records, self.spec)[0]],
                         [features.comparable(line) for line in self.reference])

    def test_sliding_windows_hop_by_periods(self):
        spec = small_spec()
        spec["window"].update(periods=4, hop_periods=2)
        lines = features.run_records(self.records, spec)[0]
        self.assertEqual([line["status"] for line in lines], ["insufficient_history"] * 3 + ["ok", "insufficient_history"] * 2
                         + ["ok"])
        self.assertEqual([line["window"]["first_period"] for line in complete(lines)], [0, 2, 4])
        self.assertEqual(complete(lines)[1]["axes"]["doppler"]["bins"], [-2, -1, 0, 1])

    def test_records_that_are_no_input_are_rejected_and_counted(self):
        good = self.records[0]
        values, valid = features.planes(good)
        cases = (
            (self.changed(good, stage=3), "the record is O3 of pusch_dmrs"),
            (self.changed(good, layout=1), "in the layout ocudu_o3_fp16"),
            (self.changed(good, dtype=2), "as <cfp16"),
            (self.changed(good, rank=1), "the shape is not"),
            (self.changed(good, layer_count=2), "the shape is not"),
            (self.changed(good, antenna_map=[1, 1]), "one array element per plane"),
            (self.changed(good, subcarrier_step=0), "not a regular comb"),
            (self.changed(good, t_first_sample=good.header["t_first_sample"] + 1), "sample times of the record"),
            (self.changed(good, time_unit=2), "sample times of the record"),
            (self.changed(good, slot=8, t_first_sample=good.header["t_first_sample"],
                          t_last_sample=good.header["t_last_sample"]), "sample times of the record"),
            (make_record(0, 7, values[:, :1, :], valid[:, :1, :]), None),
        )
        for record, message in cases:
            stream = features.Stream(self.spec)
            lines = stream.push(record)
            with self.subTest(message=message):
                if message is None:
                    self.assertEqual(lines[0]["status"], "insufficient_history")  # one reference symbol suffices for row 0
                    continue
                self.assertEqual((lines[0]["status"], lines[0]["window"], stream.counts["rejected"]), ("rejected", None, 1))
                self.assertIn(message, lines[0]["reason"])
        spec = small_spec()
        spec["sequence"]["reference_symbols"] = [1]
        line = features.Stream(spec).push(make_record(0, 7, values[:, :1, :], valid[:, :1, :]))[0]
        self.assertIn("holds 1 reference symbols, and the specification selects row 1", line["reason"])
        # A slot outside the sequence may carry another allocation: it does not disturb the history.
        # Inside the sequence another allocation starts a new one, because its samples lie elsewhere.
        other = [self.changed(record, first_subcarrier=1) if record.header["absolute_slot"] % 10 != 7 else record
                 for record in self.records]
        lines, _, stream = features.run_records(other, self.spec)
        self.assertEqual((stream.counts["resets"], complete(lines)[0]["features"]),
                         (0, complete(self.reference)[0]["features"]))
        moved = self.records[:12] + [self.changed(record, first_subcarrier=1) for record in self.records[12:]]
        lines, _, stream = features.run_records(moved, self.spec)
        self.assertEqual((stream.counts["resets"], stream.counts["ok"], [line["history"]["have"] for line in lines]),
                         (1, 0, [1, 2, 3, 4, 1, 2, 3, 4]))
        # A rejected record does not disturb the history it lies in.
        odd = self.changed(self.records[10], stage=3)
        lines = features.run_records(self.records[:10] + [odd] + self.records[11:], self.spec)[0]
        self.assertEqual([line["status"] for line in lines].count("rejected"), 1)
        self.assertEqual(complete(lines)[0]["features"], complete(self.reference)[0]["features"])


@unittest.skipIf(np is None, NEEDS)
class ReferencePairTests(unittest.TestCase):
    """The reference pair of Sprint 2.2 by its analytic model, on the lattice of the demonstration."""

    @classmethod
    def setUpClass(cls):
        carrier = 3.5e9
        wavelength = episodes.SPEED_OF_LIGHT_M_S / carrier
        cls.geometry = {"ue_tx_m": [0.0, 0.0, 1.5], "gnb_rx_m": [20.0, 0.0, 3.0],
                        "rx_element_offsets_m": [[0.0, 0.5 * wavelength * (n - 1.5), 0.0] for n in range(4)]}
        cls.target = {"sigma_m2": 1.0, "motion": {"kind": "constant_velocity", "position_m": [10.0, 25.0, 1.5],
                                                  "velocity_m_s": [0.0, 2.0, 0.0]}}
        offsets = (np.arange(0, 612, 2) - 306) * 30e3
        generator = np.random.Generator(np.random.PCG64(20261121))
        direct = wavelength / (4 * math.pi * math.dist(cls.geometry["ue_tx_m"], cls.geometry["gnb_rx_m"]))
        cls.records = {}
        for present in (False, True):
            made = []
            for period in range(8):
                for slot in (7, 8, 9):
                    absolute = 10 * period + slot
                    values = np.zeros((4, 2, 306), dtype=np.complex128)
                    for row, symbol in enumerate(SYMBOLS):
                        seconds = features.symbol_time_tc(SCS, absolute, symbol) / features.TC_PER_SECOND
                        for amplitude, delay, phases in episodes.free_space_paths(cls.geometry, cls.target, seconds,
                                                                                  carrier, present):
                            values[:, row, :] += amplitude * np.exp(1j * (np.array(phases)[:, None]
                                                                          - 2 * np.pi * (carrier + offsets[None, :]) * delay))
                    # 30 dB per resource element, and a pilot of twice the data power.
                    noise = (generator.standard_normal(values.shape) + 1j * generator.standard_normal(values.shape)) \
                        * 10 ** (-30 / 20) / 2
                    made.append(make_record(len(made), absolute, values / direct + noise, episode=EPISODE + present))
            cls.records[present] = made
        position, velocity = episodes.target_state(cls.target["motion"], 0.0)
        cls.truth = episodes.bistatic(cls.geometry, position, velocity, carrier)

    def test_the_return_peaks_in_delay_bin_3_and_doppler_bin_minus_2(self):
        self.assertAlmostEqual(self.truth["delay_ns"], 179.77, places=2)
        self.assertAlmostEqual(self.truth["doppler_hz"], -43.33, places=2)
        lines, views, _ = features.run_records(self.records[True], features.demo_spec())
        line = complete(lines)[0]
        axes, peak = line["axes"], line["view"]["peak"]
        self.assertAlmostEqual(axes["delay"]["resolution_ns"], 54.466, places=3)
        self.assertAlmostEqual(axes["delay"]["unambiguous_ns"], 16666.667, places=3)
        self.assertEqual((axes["doppler"]["resolution_hz"], axes["doppler"]["unambiguous_hz"]), (25.0, [-100.0, 75.0]))
        self.assertEqual((peak["delay_bin"], peak["doppler_bin"]), (3, -2))
        self.assertAlmostEqual(peak["delay_ns"], 163.399, places=3)
        self.assertEqual(peak["doppler_hz"], -50.0)
        # The direct path lies at 66.9 ns, delay bin 1, and is removed with the static part.
        self.assertEqual(line["view"]["static_peak"]["delay_bin"], 1)
        passed, facts = evaluator.resolved(line, self.truth)
        self.assertTrue(passed, facts)
        self.assertFalse(facts["aliased"])
        self.assertLess(abs(facts["delay_error_ns"]), 54.47)
        self.assertLess(abs(facts["doppler_error_hz"]), 25.0)
        named = vector(line)
        self.assertGreater(named["peak_to_median_db"], 18.0)
        self.assertLess(named["doppler_centroid_hz"], 0.0)
        self.assertEqual(max((value, name) for name, value in named.items() if name.startswith("profile_db"))[1],
                         "profile_db@-50Hz")
        # A transposed Doppler axis would put the return at +50 Hz: that cell is far weaker.
        power = views[0][1]
        bins = axes["doppler"]["bins"]
        self.assertGreater(10 * math.log10(power[3, bins.index(-2)] / power[3, bins.index(2)]), 15.0)
        # Without the target the same lattice holds noise only.
        absent = complete(features.run_records(self.records[False], features.demo_spec())[0])[0]
        self.assertLess(vector(absent)["peak_to_median_db"], 10.0)
        self.assertGreater(named["peak_to_median_db"] - vector(absent)["peak_to_median_db"], 10.0)
        self.assertAlmostEqual(vector(absent)["moving_to_static_db"], -33.0, delta=1.0)

    def test_the_doppler_phase_of_the_band_edges_is_within_the_bin(self):
        # A path whose delay changes advances by 2 pi nu (1 + f / f_c) dt: 0.26 percent at the band edge.
        self.assertAlmostEqual(9.18e6 / 3.5e9, 0.0026, places=4)
        self.assertLess(abs(self.truth["doppler_hz"]) * 0.0026, 25.0 / 100)

    def test_the_evaluator_cuts_the_same_windows_and_computes_the_same_view(self):
        spec = features.demo_spec()
        for present in (False, True):
            lines, views, _ = features.run_records(self.records[present], spec)
            found = []
            for record in self.records[present]:
                header = record.header
                values, valid = features.planes(record)
                found.append({"epoch": header["epoch"], "sequence": header["sequence"],
                              "episode": f"{header['episode']:016x}", "opportunity": f"{header['opportunity']:016x}",
                              "opportunity_id": str(header["sequence"]), "absolute_slot": header["absolute_slot"],
                              "first_subcarrier": 0, "subcarrier_step": 2, "antenna_map": [0, 1, 2, 3],
                              "symbols": list(SYMBOLS), "t_first_sample": header["t_first_sample"],
                              "t_last_sample": header["t_last_sample"], "values": values.astype(np.complex128),
                              "valid": valid})
            windows = evaluator.offline_windows(list(reversed(found)), spec)  # the order of the files does not matter
            self.assertEqual([(window["id"], window["eligible"]) for window in windows],
                             [(complete(lines)[0]["window"]["id"], True)])
            self.assertEqual(windows[0]["contributing"], [item["opportunity"] for item in complete(lines)[0]["contributing"]])
            made = evaluator.oracle(windows[0], spec)
            self.assertEqual(evaluator.differences(complete(lines)[0], views[0][1], made), [])
            self.assertLess(float(np.abs(views[0][1] - made["power"]).max() / made["power"].max()), 1e-12)
            # A stream that read a sample of another time, or a transposed Doppler axis, is found.
            other = windows[0]["cube"].copy()
            other[:, 7, :] = other[:, 0, :]
            leaked = dict(windows[0], cube=other)
            self.assertTrue(evaluator.differences(complete(lines)[0], views[0][1], evaluator.oracle(leaked, spec)))
            mirrored = evaluator.differences(complete(lines)[0], views[0][1][:, ::-1], made)
            self.assertTrue(mirrored and "differs from the oracle" in mirrored[0])
            # A window with a missing observation is not eligible offline either.
            holed = evaluator.offline_windows([item for item in found if item["absolute_slot"] != 37], spec)
            self.assertEqual([(window["eligible"], window["missing"]) for window in holed], [(False, 1)])

    def test_the_expected_return_folds_into_the_unambiguous_interval(self):
        self.assertAlmostEqual(evaluator.fold(-43.3, -112.5, 200.0), -43.3)
        self.assertAlmostEqual(evaluator.fold(130.0, -112.5, 200.0), -70.0)
        line = complete(features.run_records(self.records[True], features.demo_spec())[0])[0]
        passed, facts = evaluator.resolved(line, dict(self.truth, doppler_hz=self.truth["doppler_hz"] + 200.0))
        self.assertTrue(passed and facts["aliased"])
        for wrong in (dict(delay_ns=300.0), dict(doppler_hz=10.0), dict(delay_ns=5000.0)):
            self.assertFalse(evaluator.resolved(line, dict(self.truth, **wrong))[0], wrong)


@unittest.skipIf(np is None, NEEDS)
class OutputTests(unittest.TestCase):
    """What a sensor leaves, the second reader, and the command line."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.records = lattice()
        self.spec = small_spec()
        # The archive of a consumer of Sprint 3.2 that kept these records.
        self.archive = self.root / "archive"
        self.archive.mkdir()
        rows = []
        for record in self.records:
            (self.archive / obs.record_name(1, record.sequence)).write_bytes(record.to_bytes())
            rows.append({"epoch": 1, "sequence": record.sequence, "opportunity": f"{record.header['opportunity']:016x}",
                         "depth": 0, "bytes": len(record.to_bytes()), "t_publish_ns": record.header["t_publish_ns"],
                         "t_read_ns": 1, "delay_ns": 1, "intact": True, "file": True})
        (self.archive / "consumer.json").write_text(json.dumps({
            "schema": obs.CONSUMER_SCHEMA, "reader": "native", "definition": f"{obs.DEFINITION:016x}", "wait": "spin",
            "epoch": 1, "records": rows}))
        (self.root / "spec.json").write_text(json.dumps(self.spec))

    def made(self, name="out"):
        directory = self.root / name
        directory.mkdir()
        return directory, features.replay(self.archive, self.spec, directory)

    def test_an_output_verifies_and_a_second_reader_recovers_the_window(self):
        directory, report = self.made()
        self.assertEqual((report["counts"]["ok"], report["lines"], report["views"], report["open_at_end"]),
                         (1, 8, 1, {"have": 0, "need": 8}))
        self.assertEqual(report["source"], {"kind": "archive", "directory": str(self.archive), "reader": "native",
                                            "records": 24})
        self.assertEqual((report["runtime"]["entry"], report["runtime"]["device"], report["runtime"]["numpy"]),
                         ("reference", "cpu", np.__version__))
        self.assertEqual(report["steps"]["doppler"]["entry"], "fft")
        output = features.open_output(directory)
        line = output.ok[0]
        self.assertEqual(sorted(path.name for path in directory.iterdir()), ["features.jsonl", "sensor.json", "views"])
        self.assertEqual(line["view"]["file"], "views/000007.dd.f64")
        np.testing.assert_array_equal(output.view(line), features.run_records(self.records, self.spec)[1][0][1])
        shown = features.summary(output)
        self.assertEqual((shown["complete_windows"], shown["window"]["peak"]["delay_bin"], shown["window"]["contributing"]),
                         (1, 5, 8))
        self.assertEqual(shown["steps"]["static"], "window_mean (numpy on the cpu)")
        self.assertAlmostEqual(shown["window"]["features"]["peak_doppler_hz"], 50.0)
        with self.assertRaisesRegex(FeaturesError, "window 1 was asked for"):
            features.summary(output, 1)
        with self.assertRaisesRegex(FeaturesError, "holds a result already"):
            features.replay(self.archive, self.spec, directory)

    def test_a_changed_output_is_refused(self):
        def lines_of(directory):
            return [json.loads(text) for text in (directory / "features.jsonl").read_text().splitlines()]

        def rewrite(directory, lines):
            (directory / "features.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))

        def flip_view(directory):
            path = directory / "views/000007.dd.f64"
            path.write_bytes(bytes([path.read_bytes()[0] ^ 1]) + path.read_bytes()[1:])

        def other_peak(directory):
            lines = lines_of(directory)
            lines[7]["view"]["peak"]["doppler_bin"] = -2
            rewrite(directory, lines)

        def short_vector(directory):
            lines = lines_of(directory)
            lines[7]["features"]["values"].pop()
            rewrite(directory, lines)

        def nan_feature(directory):
            text = (directory / "features.jsonl").read_text().replace('"values": [120.0', '"values": [NaN', 1)
            (directory / "features.jsonl").write_text(text)

        def lost_line(directory):
            rewrite(directory, lines_of(directory)[:-1])

        def vector_on_a_warm_up(directory):
            lines = lines_of(directory)
            lines[0]["features"] = lines[7]["features"]
            rewrite(directory, lines)

        def other_spec(directory):
            report = json.loads((directory / "sensor.json").read_text())
            report["spec"]["window"]["periods"] = 4
            (directory / "sensor.json").write_text(json.dumps(report))

        def other_runtime(directory):
            report = json.loads((directory / "sensor.json").read_text())
            report["runtime"]["entry"] = "native"
            (directory / "sensor.json").write_text(json.dumps(report))

        cases = ((flip_view, "do not agree"), (other_peak, "do not agree"), (short_vector, "do not agree"),
                 (nan_feature, "do not agree"), (lost_line, "counts of the report"),
                 (vector_on_a_warm_up, "carries features"), (other_spec, "does not name its specification"),
                 (other_runtime, "the reference runtime"),
                 (lambda directory: (directory / "truth.json").write_text("{}"), "holds truth.json"),
                 (lambda directory: (directory / "views/extra.f64").write_text(""), "holds extra.f64"),
                 (lambda directory: (directory / "sensor.json").unlink(), "not a complete sensor output"))
        for number, (change, message) in enumerate(cases):
            directory, _ = self.made(f"out-{number}")
            change(directory)
            with self.subTest(message=message), self.assertRaisesRegex(FeaturesError, message):
                features.open_output(directory)

    def test_command_line(self):
        def run(*arguments):
            with contextlib.redirect_stdout(io.StringIO()) as said, contextlib.redirect_stderr(io.StringIO()) as errors:
                try:
                    status = features.main(list(arguments))
                except SystemExit as stop:
                    status = stop.code
            return status, said.getvalue(), errors.getvalue()

        status, said, _ = run("spec", "--output", str(self.root / "demo.json"))
        self.assertEqual(status, 0)
        self.assertEqual(features.load_spec(self.root / "demo.json"), features.demo_spec())
        self.assertEqual(json.loads(said)["sha256"], features.spec_sha256(features.demo_spec()))
        status, _, errors = run("spec", "--output", str(self.root / "demo.json"))
        self.assertEqual((status, "never overwritten" in errors), (2, True))
        (self.root / "out").mkdir()
        status, said, _ = run("replay", "--archive", str(self.archive), "--spec", str(self.root / "spec.json"),
                              "--output-dir", str(self.root / "out"))
        self.assertEqual(status, 0)
        self.assertIn("features: 24 records, 1 complete windows, 7 lines of a filling window, 0 gaps, 0 invalid, "
                      "0 rejected, 16 outside the sequence", said)
        status, said, _ = run("show", str(self.root / "out"))
        self.assertEqual((status, json.loads(said)["window"]["peak"]["doppler_bin"]), (0, 2))
        # Refusals leave with status 2 and one line.
        (self.root / "bad.json").write_text(json.dumps(dict(self.spec, schema="other")))
        (self.root / "empty").mkdir()
        for arguments, message in (
                (("replay", "--archive", str(self.archive), "--spec", str(self.root / "bad.json"), "--output-dir",
                  str(self.root / "empty")), "unsupported specification schema"),
                (("replay", "--archive", str(self.archive), "--spec", str(self.root / "missing.json"), "--output-dir",
                  str(self.root / "empty")), "cannot read the specification"),
                (("replay", "--archive", str(self.root / "empty"), "--spec", str(self.root / "spec.json"),
                  "--output-dir", str(self.root / "empty")), "holds no readable consumer.json"),
                (("replay", "--archive", str(self.archive), "--spec", str(self.root / "spec.json"), "--output-dir",
                  str(self.root / "out")), "holds a result already"),
                (("follow", "--socket", str(self.root / "no.sock"), "--spec", str(self.root / "spec.json"),
                  "--output-dir", str(self.root / "out")), "is not empty"),
                (("show", str(self.root / "empty")), "not a complete sensor output")):
            status, _, errors = run(*arguments)
            with self.subTest(message=message):
                self.assertEqual(status, 2)
                self.assertIn(message, errors)
                self.assertEqual(errors.count("\n"), 1)
        self.assertEqual(list((self.root / "empty").iterdir()), [])


@unittest.skipIf(np is None, NEEDS)
class LiveSensorTests(unittest.TestCase):
    """The reference sensor in a second process, on a native producer of the ring."""

    @classmethod
    def setUpClass(cls):
        cls.missing = [tool for tool in ("cmake",) if not shutil.which(tool)]
        if not any(shutil.which(compiler) for compiler in ("c++", "g++", "clang++")):
            cls.missing.append("a C++17 compiler")
        if cls.missing:
            return
        cls.temporary = tempfile.TemporaryDirectory()
        cls.build = Path(cls.temporary.name) / "build"
        cls.log, cls.status = "", 0
        for step in (["cmake", "-S", str(ROOT / "native/obs_ring"), "-B", str(cls.build)],
                     ["cmake", "--build", str(cls.build), "--target", "obs_feed", "-j2"]):
            result = subprocess.run(step, capture_output=True, text=True, timeout=1800, check=False)
            cls.log += result.stdout + result.stderr
            cls.status = cls.status or result.returncode

    @classmethod
    def tearDownClass(cls):
        if not cls.missing:
            cls.temporary.cleanup()

    def setUp(self):
        if self.missing:
            self.skipTest("the native producer needs " + ", ".join(self.missing))
        if self.status:
            self.fail("the native ring does not build:\n" + self.log[-3000:])
        scratch = tempfile.TemporaryDirectory(dir=self.temporary.name)
        self.addCleanup(scratch.cleanup)
        self.scratch = Path(scratch.name)

    def test_the_sensor_follows_a_native_producer_and_keeps_what_it_read(self):
        # obs_feed publishes one record per 20 slots, slot 8 of a period of 20, with the shape of the
        # demonstration and sample times of its own, which are not those of a slot.
        spec = features.demo_spec()
        spec["radio"]["period_slots"] = 20
        spec["sequence"]["slots_in_period"] = [8]
        (self.scratch / "spec.json").write_text(json.dumps(spec))
        out = self.scratch / "sensor"
        out.mkdir()
        socket = self.scratch / "ring.sock"
        environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        sensor = subprocess.Popen([sys.executable, "-m", "ocudu_cuda_isac.sensing.features", "follow", "--socket",
                                   str(socket), "--spec", str(self.scratch / "spec.json"), "--output-dir", str(out),
                                   "--wait", "wait"], env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True)
        # One record per 5 ms, the period of the demonstration. A sensor that falls behind is not a
        # fault of the ring: the producer drops and counts, so the counts below are the producer's.
        feed = subprocess.run([str(self.build / "obs_feed"), "--socket", str(socket), "--records", "24",
                               "--period-us", "5000"], capture_output=True, text=True, timeout=120, check=False)
        said, errors = sensor.communicate(timeout=120)
        self.assertEqual((feed.returncode, sensor.returncode), (0, 0), feed.stderr + errors)
        fed = json.loads(feed.stdout)
        self.assertEqual((fed["published"] + fed["dropped"], fed["released"], fed["pending"]), (24, fed["published"], 0))
        self.assertEqual(fed["dropped"], 0, "the sensor did not keep the pace of one record per 5 ms")
        self.assertIn("features: 24 records, 0 complete windows", said)
        output = features.open_output(out)
        report = output.report
        self.assertEqual((report["counts"]["records"], report["counts"]["rejected"], report["source"]["kind"],
                          report["source"]["reader"], report["source"]["records"], report["source"]["ended"]),
                         (24, 24, "ring", "reference", 24, "closed"))
        self.assertTrue(all("sample times of the record" in line["reason"] for line in output.lines))
        self.assertTrue(all(type(line["t_features_ns"]) is int for line in output.lines))
        # The records it read are kept as the reader of Sprint 3.2 keeps them, and replay to the same lines.
        archive = obs.open_archive(out / "records")
        self.assertEqual([record.sequence for record in archive.records], list(range(24)))
        # It warmed up before it attached, and every answer carries the time at which it was complete.
        self.assertGreater(report["source"]["warm_up_ms"], 0.0)
        self.assertEqual(sorted(line["t_features_ns"] for line in output.lines),
                         [line["t_features_ns"] for line in output.lines])
        again = [features.comparable(line) for line in features.run_records(archive.records, spec)[0]]
        self.assertEqual(again, [features.comparable(line) for line in output.lines])
        # With sample times of a slot the same records fill a window: the stream is fed here, by hand.
        fixed = []
        for record in archive.records:
            slot = record.header["absolute_slot"]
            header = dict(record.header, t_first_sample=features.symbol_time_tc(30, slot, 2),
                          t_last_sample=features.symbol_time_tc(30, slot, 11) + estimator.useful_tc(30))
            fixed.append(obs.Record(header, record.host, record.mask, record.payload))
        lines = features.run_records(fixed, spec)[0]
        self.assertEqual([line["status"] for line in lines], (["insufficient_history"] * 7 + ["ok"]) * 3)
        self.assertEqual(complete(lines)[0]["axes"]["doppler"]["resolution_hz"], 12.5)


class TreeTests(unittest.TestCase):
    FILES = ("src/ocudu_cuda_isac/sensing/features.py", "src/ocudu_cuda_isac/sensing/obs.py",
             "src/ocudu_cuda_isac/episodes/pilot_features.py", "tests/unit/test_pilot_features.py",
             "tests/unit/test_core_imports.py", "tests/unit/pilot_features_double.py",
             "tests/unit/test_pilot_features_driver.py", "tests/integration/pilot_features.py",
             "configs/integration/pilot_features.json", "scripts/accept_s3_3.sh", "tests/test_accept_s3_3.py",
             "docs/interfaces/PILOT_FEATURES.md", "Makefile")

    def test_files_are_ascii_and_do_not_name_the_project(self):
        # The working name lives on line 1 of the top-level README and nowhere else.
        name = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()[0].lstrip("# ").split(":")[0].strip()
        for relative in self.FILES:
            data = (ROOT / relative).read_bytes()
            self.assertTrue(data.isascii(), relative)
            if name and name.lower() != "ocudu-cuda-isac":
                self.assertNotIn(name.lower(), data.decode("ascii").lower(), relative)

    def test_the_gate_and_the_configuration_name_the_demonstration(self):
        config = json.loads((ROOT / "configs/integration/pilot_features.json").read_text())
        self.assertEqual((config["schema"], config["selector"], config["backends"]),
                         ("integration-config/pilot_features/v1", "pilot_features", ["cpu", "cuda"]))
        # The ring is that of Sprint 3.2, and the pair is the one that gate published.
        ring = json.loads((ROOT / "configs/integration/snapshot_ring.json").read_text())
        for name in ("stream", "window_records", "burst_records", "slots", "attach_timeout_ms", "drain_ms"):
            self.assertEqual(config["ring"][name], ring["ring"][name], name)
        self.assertEqual(config["ring"]["slots"], obs.slots_for(config["ring"]["window_records"],
                                                                config["ring"]["burst_records"]))
        self.assertEqual(config["pair"]["expect"]["opportunities"], ring["pair"]["expect"]["opportunities"])
        self.assertEqual(config["control"], {"driver": "snapshot_ring", "audit_run": "audit", "timed_run": "timed"})
        # Eight periods of three uplink slots per member, one of them in the sequence: 2 windows, 14
        # answers of a filling window and 32 records outside the sequence.
        spec, expect = config["spec"], config["pair"]["expect"]
        members, periods = 2, spec["window"]["periods"]
        self.assertEqual((spec["sequence"]["slots_in_period"], spec["sequence"]["reference_symbols"], periods), ([7], [0], 8))
        self.assertEqual(expect, {"opportunities": members * periods * 3, "complete_windows": members,
                                  "filling": members * (periods - 1), "outside_sequence": members * periods * 2})
        # The cell of the card: 179.8 ns over a bin of 54.47 ns, -43.3 Hz over a bin of 25 Hz.
        self.assertEqual((config["known_return"]["delay_bin"], config["known_return"]["doppler_bin"]),
                         (round(179.8 / (1e9 / (306 * 60e3))), round(-43.3 / 25.0)))
        self.assertIn("accept-s3.3:", (ROOT / "Makefile").read_text())
        runner = (ROOT / "scripts/accept_s3_3.sh").read_text()
        for needed in ("tests/unit/test_pilot_features.py", "tests/unit/test_core_imports.py",
                       "tests/integration/pilot_features.py --config configs/integration/pilot_features.json",
                       "-m ocudu_cuda_isac.sensing.features show", "sha256sum"):
            self.assertIn(needed, runner)
        if np is not None:
            self.assertEqual(spec, features.demo_spec())

    def test_this_sprint_adds_no_native_source_and_the_reader_keeps_its_default(self):
        for directory in ("native/obs_ring", "native/ocudu_adapter", "native/sense_core"):
            for path in (ROOT / directory).rglob("*"):
                self.assertNotIn("pilot_features", path.name, path)
        reader = (ROOT / "src/ocudu_cuda_isac/sensing/obs.py").read_text()
        self.assertIn("hold_us=0, window=0,\n         on_record=None):", reader)
        self.assertIn("            if on_record is not None:\n                on_record(record)\n", reader)
        # The reference of the features names its runtime and its device in every registry entry.
        source = (ROOT / "src/ocudu_cuda_isac/sensing/features.py").read_text()
        self.assertIn('_CPU = {"device": "cpu", "library": "numpy", "runtime": RUNTIME}', source)
        self.assertIn('RUNTIME = "reference"', source)


if __name__ == "__main__":
    unittest.main()
