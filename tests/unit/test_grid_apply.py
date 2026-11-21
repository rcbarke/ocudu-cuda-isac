#!/usr/bin/env python3
"""Sprint 2.3 unit gate (selector grid_apply): python tests/unit/test_grid_apply.py -v

Two parts. The Python part needs the standard library only: the scalar reference of the channel
application, the gain rules, the plan and its bindings, and the reader of the native output. The
native part builds native/ocudu_adapter/replay on its own, with a C++17 compiler and CMake and no
OCUDU source or build, runs its CTest cases and compares the native kernel with the Python
reference bit for bit. No Sionna, no NumPy, no OCUDU build and no GPU.
"""

import cmath
import contextlib
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ocudu_cuda_isac import replay as fixture_replay  # noqa: E402
from ocudu_cuda_isac.episodes import episode as episodes  # noqa: E402
from ocudu_cuda_isac.episodes import grant as grants  # noqa: E402
from ocudu_cuda_isac.episodes import grid_apply as replay  # noqa: E402
from ocudu_cuda_isac.episodes import schedule as schedules  # noqa: E402
from ocudu_cuda_isac.episodes.episode import TC_PER_SECOND  # noqa: E402
from ocudu_cuda_isac.episodes.grant import bf16  # noqa: E402
from ocudu_cuda_isac.episodes.grid_apply import GridApplyError  # noqa: E402
import grid_apply_double as double  # noqa: E402

REPLAY = ROOT / "native/ocudu_adapter/replay"
GRANTS = ROOT / "configs/grants"
DATA = complex(0.31640625, 0.31640625)  # a stored 16QAM point
GAIN = 2942.4243526773034               # calibration.grid_gain of the accepted reference pair
POWER = 1.1550196887810671e-07          # its reference power
CASES = ("sha256", "storage", "s03_vector", "unity", "zero_channel", "cancellation", "permutation", "unit_scale",
         "gain_reason", "gain_rules", "grid_states", "session", "changed_grant", "refusals")


def generated(seed, shape):
    """The fixture of 'replay_core_test --digest': X, then H, then W from one sequence, W scaled by 2^-6."""
    state = seed

    def value(scale=1.0):
        nonlocal state
        parts = []
        for _ in range(2):
            state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
            parts.append(((state >> 8) / 8388608.0 - 1.0) * scale)
        return complex(*parts)

    rx, tx, symbols, subcarriers = shape
    plane = symbols * subcarriers
    return ([value() for _ in range(tx * plane)], [value() for _ in range(rx * tx * plane)],
            [value(0.015625) for _ in range(rx * plane)])


def digest_of(values):
    return hashlib.sha256(replay.grid_bytes(values)).hexdigest()


class ReferenceTests(unittest.TestCase):
    """The arithmetic: Y = g (sum over tx of H X + W), once, in double precision, then the storage format."""

    def test_sprint_0_3_vector_gives_its_four_answers_and_exactly_twice_them(self):
        shape = (2, 2, 2, 1)
        x = [complex(v) for v in (1, 2, 3, 4)]
        h = [complex(v) for v in (1, 1, 1j, -1j, 0.5, 1, 2, 3)]
        answers = [1 + 3j, 2 - 4j, 6.5 + 0j, 14 + 0j]
        self.assertEqual(replay.apply(shape, x, h, [0j] * 4, 1.0), answers)
        self.assertEqual(replay.apply(shape, x, h, [0j] * 4, 2.0), [2 * value for value in answers])
        # Every answer has eight significant bits or fewer, so the grid keeps it.
        self.assertEqual(replay.stored(replay.apply(shape, x, h, [0j] * 4, 2.0)), [2 * value for value in answers])

    def test_gain_one_is_the_sprint_0_3_reference_on_its_checked_in_fixture(self):
        episode = json.loads((ROOT / "configs/smoke_replay.json").read_text())
        shape = tuple(episode["shape"])
        for record in episode["records"]:
            x, h, w = ([complex(*pair) for pair in record[key]] for key in ("x", "h", "w"))
            earlier = [complex(*pair) for pair in fixture_replay.received_grid(shape, x, h, w)]
            self.assertEqual(replay.apply(shape, x, h, w, 1.0), earlier, record["case"])
            self.assertEqual(replay.apply(shape, x, h, w, 2.0), [2 * value for value in earlier], record["case"])

    def test_unity_with_gain_one_returns_x_bit_for_bit(self):
        x = [0j if n % 5 == 4 else complex(bf16(math.sin(n)), bf16(math.cos(3 * n))) for n in range(24)]
        written = replay.apply((1, 1, 3, 8), x, [1 + 0j] * 24, [0j] * 24, 1.0)
        self.assertEqual(replay.grid_bytes(written), replay.grid_bytes(x))
        self.assertEqual(replay.grid_bytes(replay.stored(written)), replay.grid_bytes(x))
        # A square identity channel: receive plane i holds transmit port i.
        two = x + [complex(bf16(math.cos(n)), bf16(math.sin(2 * n))) for n in range(24)]
        identity = [1 + 0j] * 24 + [0j] * 48 + [1 + 0j] * 24
        self.assertEqual(replay.grid_bytes(replay.apply((2, 2, 3, 8), two, identity, [0j] * 48, 1.0)),
                         replay.grid_bytes(two))

    def test_zero_channel_gives_the_scaled_noise_for_any_x(self):
        w = replay.single([complex(1.0747184e-05 * (n - 7.5), 3.3986e-04 * math.sin(n + 1.0)) for n in range(16)])
        for x in ([1 + 1j] * 8, [complex(n, -n) for n in range(8)]):
            for gain in (1.0, 2.0, GAIN):
                written = replay.apply((2, 1, 2, 4), x, [0j] * 16, w, gain)
                self.assertEqual(written, [complex(replay.to_single(gain * v.real), replay.to_single(gain * v.imag))
                                           for v in w])

    def test_opposite_transmit_terms_cancel(self):
        x = [complex(n + 1, 2 - n) for n in range(4)] * 2
        h = [complex(0.5 * n, 1 - n) for n in range(4)]
        w = replay.single([complex(0.01 * n, -0.02 * n) for n in range(4)])
        self.assertEqual(replay.apply((1, 2, 2, 2), x, h + [-value for value in h], w, 1.0), w)

    def test_a_permutation_of_the_receive_planes_is_caught(self):
        x, h, w = generated(5, (4, 1, 2, 6))
        ordered = replay.stored(replay.apply((4, 1, 2, 6), x, h, w, 1.0))
        swapped = h[24:36] + h[12:24] + h[:12] + h[36:]
        permuted = replay.stored(replay.apply((4, 1, 2, 6), x, swapped, w[24:36] + w[12:24] + w[:12] + w[36:], 1.0))
        self.assertNotEqual(digest_of(ordered), digest_of(permuted))
        self.assertEqual(permuted[:12], ordered[24:36])  # plane 0 now holds plane 2
        self.assertEqual(permuted[12:24], ordered[12:24])

    def test_unit_scale_tolerance_applies_after_the_gain(self):
        x, h, w = generated(11, (4, 2, 3, 16))
        h = replay.single([value * 3.4e-4 for value in h])
        w = replay.single([value * 6.9e-4 for value in w])
        written = replay.apply((4, 2, 3, 16), x, h, w, GAIN)
        plane = 48
        for rx in range(4):
            for element in range(plane):
                exact = GAIN * (sum(h[(rx * 2 + tx) * plane + element] * x[tx * plane + element] for tx in range(2))
                                + w[rx * plane + element])
                value = written[rx * plane + element]
                self.assertTrue(replay.within_unit_scale(value, exact))
                self.assertLess(abs(value - exact), 1e-6 * abs(exact))
        # Before the gain the absolute term is a third of the return of the reference pair: a return
        # that is 30 percent wrong would pass. After the gain it does not.
        echo = complex(2.6468e-06, 0)
        self.assertTrue(replay.within_unit_scale(0.7 * echo, echo))
        self.assertFalse(replay.within_unit_scale(GAIN * 0.7 * echo, GAIN * echo))

    def test_storage_rounds_to_nearest_with_ties_to_even(self):
        step = 1.0 / 128.0
        for value, kept in ((1.0, 1.0), (0.31640625, 0.31640625), (1.0 + 0.5 * step, 1.0),
                            (1.0 + 1.5 * step, 1.0 + 2 * step), (-(1.0 + 1.5 * step), -(1.0 + 2 * step)),
                            (2.0 - 0.25 * step, 2.0)):
            self.assertEqual(replay.stored([complex(value, -value)]), [complex(kept, -kept)])
        _, h, _ = generated(3, (4, 1, 2, 50))
        for scale in (1.0, 3.4e-4, 2.6e-6, 7.8e-3, GAIN):
            for value in h:
                written = replay.to_single(scale * value.real)
                self.assertLessEqual(abs(bf16(written) - written), abs(written) / 256.0)

    def test_the_reason_for_the_gain(self):
        """The pinned CUDA receiver keeps its estimates in 16-bit floats, whose normal range ends at 6.1e-05.

        The stored channel is physical. In the reference pair the return has an amplitude of
        2.6468e-06 and the noise a sigma of 1.0747e-05, both below that range, where the step is
        fixed at 2^-24: the return keeps about five significant bits. With g = 1 / sqrt(reference
        power) the return is 7.8e-03 and the noise sigma 0.0316, and both keep ten or more.
        """
        echo, sigma, smallest = 2.6468e-06, 1.0747184230211497e-05, 2.0 ** -14
        self.assertAlmostEqual(smallest, 6.1035e-05, delta=1e-9)
        self.assertLess(echo, smallest)
        self.assertLess(sigma, smallest)
        kept = replay.half_precision(echo)
        self.assertEqual(kept, round(echo * 2 ** 24) / 2 ** 24)                # a multiple of the fixed step
        self.assertGreater(abs(kept - echo) / echo, 0.005)                    # most of a percent lost
        self.assertLess(math.log2(echo * 2 ** 24), 5.6)                       # about five bits
        scaled = GAIN * echo
        self.assertGreater(scaled, smallest)
        self.assertGreater(GAIN * sigma, smallest)
        self.assertLessEqual(abs(replay.half_precision(scaled) - scaled) / scaled, 2.0 ** -11)  # eleven bits
        self.assertAlmostEqual(GAIN * sigma, 10 ** (-30 / 20), delta=1e-6)    # 30 dB below unit signal power
        # The grid's own format has the exponent range of a single: the factor does not change its rounding.
        for value in (echo, scaled):
            written = replay.to_single(value)
            self.assertLessEqual(abs(bf16(written) - written) / written, 1 / 256)

    def test_bad_sizes_and_nonfinite_results_are_refused(self):
        with self.assertRaisesRegex(GridApplyError, "sizes of the carrier grid"):
            replay.apply((1, 1, 1, 2), [1j], [1j, 1j], [0j, 0j], 1.0)
        with self.assertRaisesRegex(GridApplyError, "single precision"):
            replay.apply((1, 1, 1, 1), [1 + 0j], [3e38 + 0j], [0j], 2.0)
        with self.assertRaisesRegex(GridApplyError, "nonfinite"):
            replay.apply((1, 1, 1, 1), [1 + 0j], [complex(math.inf, 0)], [0j], 1.0)


class GainRuleTests(unittest.TestCase):
    def test_a_gain_is_a_finite_positive_number(self):
        self.assertEqual(replay.check_gain(GAIN), GAIN)
        self.assertEqual(replay.check_gain(2), 2.0)
        for bad, text in ((None, "names no grid gain"), (0.0, "must be positive"), (-GAIN, "must be positive"),
                          (-0.0, "must be positive"), (math.nan, "not finite"), (math.inf, "not finite"),
                          (-math.inf, "not finite"), ("2942.4", "real number"), (True, "real number"),
                          (complex(1, 0), "real number")):
            with self.subTest(gain=bad), self.assertRaisesRegex(GridApplyError, text):
                replay.check_gain(bad)
        for shape in ((1, 1, 1, 1),):
            for bad in (0.0, -1.0, math.nan, None):
                with self.assertRaises(GridApplyError):
                    replay.apply(shape, [1j], [1j], [0j], bad)

    def test_the_gain_of_an_entry_follows_its_reference_power_within_1e_6(self):
        self.assertAlmostEqual(GAIN, 1 / math.sqrt(POWER), delta=1e-9)
        self.assertEqual(replay.entry_calibration({"calibration": {"grid_gain": GAIN, "reference_power": POWER}}),
                         (GAIN, POWER))
        self.assertTrue(replay.follows_rule(GAIN * (1 + 5e-7), POWER))
        self.assertFalse(replay.follows_rule(GAIN * (1 + 2e-6), POWER))
        self.assertFalse(replay.follows_rule(GAIN * (1 - 2e-6), POWER))
        for bad, text in (({}, "records no calibration.grid_gain"), ({"calibration": {"reference_power": POWER}}, "records no"),
                          ({"calibration": {"grid_gain": 0.0, "reference_power": POWER}}, "must be positive"),
                          ({"calibration": {"grid_gain": GAIN}}, "differs from 1 / sqrt"),
                          ({"calibration": {"grid_gain": GAIN, "reference_power": 0.0}}, "differs from 1 / sqrt"),
                          ({"calibration": {"grid_gain": GAIN * (1 + 2e-6), "reference_power": POWER}}, "by more than 1e-06"),
                          ({"calibration": {"grid_gain": 2 * GAIN, "reference_power": POWER}}, "differs from 1 / sqrt")):
            with self.subTest(manifest=bad), self.assertRaisesRegex(GridApplyError, text):
                replay.entry_calibration(bad)


def small_plan():
    """A plan of two realizations of one entry on two grants. No file of it exists: check_plan reads none."""
    ref = {"path": "/stimulus/file.npy", "sha256": "a" * 64}
    entry = {"episode_id": "ep-0123456789ab", "grid_gain": GAIN, "reference_power": POWER}
    return {"schema": replay.PLAN_SCHEMA, "run_id": "unit", "access": "offline_stimulus",
            "carrier": {"rx_ports": 4, "tx_ports": 1, "symbols": 14, "subcarriers": 612},
            "gain": {"value": GAIN, "source": "entry", "reference_power": POWER, "rule": replay.RULE},
            "grants": [{"grant_id": name, "record_sha256": "b" * 64, "slot": slot, "x": dict(ref), "active": [dict(ref)]}
                       for name, slot in (("demo", 7), ("demo.s08", 8))],
            "sources": [{"realization_id": name, "h": dict(ref), "w": dict(ref), "entry": dict(entry)}
                        for name in ("rz-aaaaaaaaaaaa", "rz-bbbbbbbbbbbb")],
            "opportunities": [{"opportunity_id": f"{name}.{index:04d}", "realization_id": name, "index": index,
                               "grant_id": grant, "sfn": 0, "slot": slot}
                              for name in ("rz-aaaaaaaaaaaa", "rz-bbbbbbbbbbbb")
                              for index, (grant, slot) in enumerate((("demo", 7), ("demo.s08", 8)))]}


class PlanRuleTests(unittest.TestCase):
    def refused(self, change, text):
        plan = small_plan()
        change(plan)
        with self.assertRaisesRegex(GridApplyError, text):
            replay.check_plan(plan)

    def test_a_complete_plan_passes_and_names_no_evaluator_field(self):
        plan = replay.check_plan(small_plan())
        text = json.dumps(plan)
        for word in ("truth", "labels", "target_present", "position_m", "tb"):
            self.assertNotIn(f'"{word}', text)
        self.assertEqual(plan["access"], "offline_stimulus")

    def test_every_gain_rule_of_the_ticket(self):
        self.refused(lambda plan: plan.pop("gain"), "names no grid gain")
        self.refused(lambda plan: plan.update(gain=None), "names no grid gain")
        self.refused(lambda plan: plan["gain"].update(value=0.0), "must be positive")
        self.refused(lambda plan: plan["gain"].update(value=-GAIN), "must be positive")
        self.refused(lambda plan: plan["gain"].update(value=math.inf), "not finite")
        self.refused(lambda plan: plan["gain"].update(value=math.nan), "not finite")
        self.refused(lambda plan: plan["gain"].update(value="2942.4"), "real number")

        def rescaled(plan, factor):
            plan["gain"]["value"] = GAIN * factor
            for source in plan["sources"]:
                source["entry"]["grid_gain"] = GAIN * factor

        self.refused(lambda plan: rescaled(plan, 1 + 2e-6), "differs from 1 / sqrt")
        self.refused(lambda plan: plan["gain"].pop("reference_power"), "no reference power")
        accepted = small_plan()
        rescaled(accepted, 1 + 5e-7)
        for source in accepted["sources"]:
            source["entry"]["reference_power"] = POWER
        replay.check_plan(accepted)
        # Two gains in one run, however close the second one is.
        self.refused(lambda plan: plan["sources"][1]["entry"].update(grid_gain=1140.799), "two gains in one run")
        self.refused(lambda plan: plan["sources"][1]["entry"].update(grid_gain=math.nextafter(GAIN, math.inf)),
                     "two gains in one run")
        # A gain per opportunity or per realization has no place in a plan.
        self.refused(lambda plan: plan["opportunities"][0].update(gain=2.0), "requires exactly")
        self.refused(lambda plan: plan["sources"][0].update(gain=2.0), "requires exactly")

    def test_an_override_is_for_a_fixture_without_an_entry_and_is_recorded(self):
        self.refused(lambda plan: plan["gain"].update(source="override", reason="replace the entry's gain"),
                     "only for a fixture that has no entry")
        fixture = small_plan()
        for source in fixture["sources"]:
            source["entry"] = None
        fixture["gain"] = {"value": 1.0, "source": "override"}
        with self.assertRaisesRegex(GridApplyError, "records its reason"):
            replay.check_plan(fixture)
        fixture["gain"]["reason"] = "  "
        with self.assertRaisesRegex(GridApplyError, "records its reason"):
            replay.check_plan(fixture)
        fixture["gain"]["reason"] = "unity check"
        self.assertEqual(replay.check_plan(fixture)["gain"]["reason"], "unity check")
        # The gain of an entry on a fixture, a mixed run and an unknown source.
        fixture["gain"] = {"value": GAIN, "source": "entry", "reference_power": POWER}
        with self.assertRaisesRegex(GridApplyError, "not both"):
            replay.check_plan(fixture)
        self.refused(lambda plan: plan["sources"][0].update(entry=None), "not both")
        self.refused(lambda plan: plan["gain"].update(source="measured"), '"entry" or "override"')

    def test_every_binding_rule_of_the_ticket(self):
        self.refused(lambda plan: plan["grants"][1]["x"].update(sha256=None), "no bound transmit grid")
        self.refused(lambda plan: plan["grants"][1]["x"].pop("sha256"), "no bound transmit grid")
        self.refused(lambda plan: plan["grants"][1]["x"].update(sha256="xyz"), "64 hexadecimal digits")
        self.refused(lambda plan: plan["opportunities"][1].update(grant_id=None), "has no bound grid")
        self.refused(lambda plan: plan["opportunities"][1].update(grant_id="two-prb-offset"), "does not bind")
        self.refused(lambda plan: plan["opportunities"][1].update(realization_id="rz-cccccccccccc"),
                     "realization that the run does not hold")
        self.refused(lambda plan: plan["opportunities"].append(plan["opportunities"][0]), "listed twice")
        self.refused(lambda plan: plan["grants"].append(plan["grants"][0]), "listed twice")
        self.refused(lambda plan: plan["sources"].append(plan["sources"][0]), "listed twice")
        self.refused(lambda plan: plan["grants"][0].update(active=[]), "lists no occupied")
        self.refused(lambda plan: plan["sources"][0]["h"].update(sha256=None), "bound by no hash")
        self.refused(lambda plan: plan.update(opportunities=[]), "at least one")
        self.refused(lambda plan: plan.update(schema="grid-apply-plan/v0"), "unsupported plan schema")
        self.refused(lambda plan: plan.update(access="runtime"), "offline_stimulus")
        self.refused(lambda plan: plan["carrier"].update(rx_ports=0), "integer from 1")
        self.refused(lambda plan: plan.update(labels={}), "requires exactly")


def small_grant(rx_ports=2):
    """The two-block unit grant of Sprint 2.1 with a second receive port."""
    record = json.loads((GRANTS / "two_prb_offset.json").read_text())
    record["rx_ports"] = list(range(rx_ports))
    return grants.build(record)


def transmit_grid(grant):
    """A grid the grant accepts: its pilots, rounded as the receiver grid stores them, and one data point."""
    width = grant.grid_subcarriers
    x = [0j] * (14 * width)
    for (symbol, k), value in zip(grant.pilot_index, grant.pilot_reference[0]):
        x[symbol * width + k] = complex(bf16(value.real), bf16(value.imag))
    for symbol, k in grant.data_index:
        x[symbol * width + k] = DATA
    return x


def make_stimulus(root, scale=3.0e-4, name="episode", binding=None):
    """A small episode entry with its grant stores: one period, three opportunities, two receive ports."""
    base = small_grant()
    schedule = schedules.build({"schema": schedules.SCHEMA, "schedule_id": "unit",
                                "tdd": {"pattern": "DDDDDDSUUU", "ul_slots": [7, 8, 9]}, "start_sfn": 0, "periods": 1,
                                "payload": "fixed"}, base)
    waveforms = {}
    for grant_id, grant in schedule.grants.items():
        directory = root / "stores" / grant_id / "store"
        if not directory.exists():
            grants.export(grant, directory, {"x": {"values": transmit_grid(grant), "provenance": {"reference": "unit"}}})
        manifest = json.loads((directory / "manifest.json").read_text())
        waveforms[grant_id] = {"x_sha256": manifest["files"]["x.npy"]["sha256"], "store": str(directory)}
    if binding:
        binding(waveforms)
    h_shape, w_shape = episodes.shapes(schedule)
    count = math.prod(w_shape)
    index = [(o, r, m, k) for o in range(h_shape[0]) for r in range(h_shape[1]) for m in range(14)
             for k in range(h_shape[4])]
    direct = [scale * cmath.exp(1j * (0.4 * r - 0.011 * k)) for o, r, m, k in index]
    echo = [0.01 * scale * cmath.exp(1j * (0.9 * r - 0.03 * k - 0.3 * m - 1.1 * o)) for o, r, m, k in index]
    state, noise = 12345, []
    for _ in range(count):
        parts = []
        for _ in range(2):
            state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
            parts.append(((state >> 8) / 8388608.0 - 1.0) * scale * 0.03)
        noise.append(complex(*parts))
    # reference_power as the exporter defines it: the stored target-absent channel, in single
    # precision, over the receive ports and the occupied elements of opportunity 0.
    occupied = sorted(schedule.opportunities[0].grant.occupied())
    stored = replay.single(direct[:len(base.rx_ports) * 14 * base.grid_subcarriers])
    power = sum(abs(stored[(r * 14 + m) * base.grid_subcarriers + k]) ** 2
                for r in range(len(base.rx_ports)) for m, k in occupied) / (len(base.rx_ports) * len(occupied))
    geometry = {"ue_tx_m": [0.0, 0.0, 1.5], "gnb_rx_m": [20.0, 0.0, 3.0],
                "rx_element_offsets_m": [[0.0, 0.02 * n, 0.0] for n in range(len(base.rx_ports))]}
    motion = {"kind": "constant_velocity", "position_m": [10.0, 25.0, 1.5], "velocity_m_s": [0.0, 2.0, 0.0]}
    description = {"geometry": geometry, "array": {}, "scene": {"kind": "free_space"},
                   "target": {"model": "constant_rcs_cuboid", "sigma_m2": 1.0, "motion": motion}, "solver": {},
                   "time_evolution": {}, "noise": {"snr_db": 30.0, "sigma": scale * 0.03, "seed": 12345},
                   "calibration": {"reference_power": power, "grid_gain": 1.0 / math.sqrt(power)},
                   "software": {}, "provenance": {}}
    truth = []
    for item in schedule.opportunities:
        position, velocity = episodes.target_state(motion, item.slot_start_tc / TC_PER_SECOND)
        truth.append({"opportunity": item.index, "position_m": position, "velocity_m_s": velocity})
    episodes.export(root / name, name, schedule, description,
                    {"a": direct, "b": [a + b for a, b in zip(direct, echo)], "w": noise}, truth, None, waveforms)
    return root / name


class StimulusTests(unittest.TestCase):
    """The plan builder on a small entry, the double of the native tool and the reader of its output."""

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.entry = make_stimulus(cls.root)
        cls.plan = replay.entry_plan("unit-pair", [cls.entry])
        replay.write_plan(cls.plan, cls.root / "plan.json")
        cls.output = cls.root / "native"
        cls.output.mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            cls.status = double.main(["--plan", str(cls.root / "plan.json"), "--output-dir", str(cls.output)])

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def scratch(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return Path(temporary.name)

    def test_entry_plan_binds_every_grid_by_the_hash_the_entry_names(self):
        plan, manifest = self.plan, json.loads((self.entry / "manifest.json").read_text())
        self.assertEqual(plan["gain"], {"value": manifest["calibration"]["grid_gain"], "source": "entry",
                                        "reference_power": manifest["calibration"]["reference_power"],
                                        "rule": plan["gain"]["rule"]})
        self.assertEqual(plan["carrier"], {"rx_ports": 2, "tx_ports": 1, "symbols": 14, "subcarriers": 612})
        self.assertEqual([item["grant_id"] for item in plan["grants"]],
                         ["two-prb-offset.s07", "two-prb-offset.s08", "two-prb-offset.s09"])
        for item in plan["grants"]:
            recorded = manifest["grants"][item["grant_id"]]
            self.assertEqual(item["x"]["sha256"], recorded["x"]["x_sha256"])
            self.assertEqual(item["record_sha256"], recorded["record_sha256"])
            self.assertEqual(replay.sha256_file(item["x"]["path"]), item["x"]["sha256"])
            self.assertEqual([Path(index["path"]).name for index in item["active"]], ["pilot_index.npy", "data_index.npy"])
        self.assertEqual(len(plan["sources"]), 2)
        self.assertEqual(plan["sources"][0]["w"], plan["sources"][1]["w"])  # one noise for both cases
        self.assertNotEqual(plan["sources"][0]["h"]["sha256"], plan["sources"][1]["h"]["sha256"])
        self.assertEqual([item["grant_id"] for item in plan["opportunities"][:3]],
                         ["two-prb-offset.s07", "two-prb-offset.s08", "two-prb-offset.s09"])
        self.assertEqual([(item["sfn"], item["slot"], item["index"]) for item in plan["opportunities"][3:]],
                         [(0, 7, 0), (0, 8, 1), (0, 9, 2)])
        # Identifiers are opaque: they name a realization, never a case or a label.
        for item in plan["opportunities"]:
            self.assertRegex(item["opportunity_id"], r"^rz-[0-9a-f]{12}\.\d{4}$")
        text = (self.root / "plan.json").read_text()
        for word in ("truth", "labels", "target_present", "position_m", "tb.bin", '"case"'):
            self.assertNotIn(word, text)
        replay.check_files(plan)
        self.assertEqual(replay.load_plan(self.root / "plan.json"), plan)
        with self.assertRaisesRegex(GridApplyError, "never overwritten"):
            replay.write_plan(plan, self.root / "plan.json")

    def test_the_builder_refuses_an_unbound_grid_another_hash_and_a_second_gain(self):
        def unbound(waveforms):
            del waveforms["two-prb-offset.s08"]

        def other(waveforms):
            waveforms["two-prb-offset.s08"]["x_sha256"] = "0" * 64

        for binding, text in ((unbound, "two-prb-offset.s08 has no bound grid"),
                              (other, "and the entry binds 0000")):
            with self.subTest(binding=binding.__name__), self.assertRaisesRegex(GridApplyError, text):
                replay.entry_plan("unit", [make_stimulus(self.root, name=f"episode-{binding.__name__}", binding=binding)])
        # A store that moved is found under a named root. A store that is gone is reported.
        def moved(waveforms):
            for item in waveforms.values():
                item["store"] = "/nonexistent/store"

        entry = make_stimulus(self.root, name="episode-moved", binding=moved)
        with self.assertRaisesRegex(GridApplyError, "no store entry of grant"):
            replay.entry_plan("unit", [entry])
        self.assertEqual(replay.entry_plan("unit", [entry], [self.root / "stores"])["grants"], self.plan["grants"])
        # Two entries with different channels have different gains: one run cannot hold both.
        stronger = make_stimulus(self.root, scale=6.0e-4, name="episode-stronger")
        with self.assertRaisesRegex(GridApplyError, "two gains in one run"):
            replay.entry_plan("unit", [self.entry, stronger])
        same = replay.entry_plan("unit", [self.entry, make_stimulus(self.root, name="episode-again")])
        self.assertEqual(len(same["sources"]), 4)
        self.assertEqual(len(same["opportunities"]), 12)
        self.assertEqual(same["grants"], self.plan["grants"])
        # A damaged entry, a damaged store and a calibration without a gain.
        scratch = self.scratch()
        shutil.copytree(self.entry, scratch / "entry")
        data = bytearray((scratch / "entry/b_h.npy").read_bytes())
        data[-1] ^= 1
        (scratch / "entry/b_h.npy").write_bytes(bytes(data))
        with self.assertRaisesRegex(GridApplyError, "not a usable episode entry"):
            replay.entry_plan("unit", [scratch / "entry"])
        shutil.copytree(self.entry, scratch / "no-gain")
        manifest = json.loads((scratch / "no-gain/manifest.json").read_text())
        del manifest["calibration"]["grid_gain"]
        (scratch / "no-gain/manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(GridApplyError, "records no calibration.grid_gain"):
            replay.entry_plan("unit", [scratch / "no-gain"])
        with self.assertRaisesRegex(GridApplyError, "at least one episode entry"):
            replay.entry_plan("unit", [])

    def test_check_files_refuses_another_hash_another_carrier_and_a_short_array(self):
        scratch = self.scratch()
        x = transmit_grid(small_grant())
        short = replay.write_complex(scratch / "x_short.npy", (1, 14, 600), [v for n, v in enumerate(x) if n % 612 < 600])
        cases = ((lambda plan: plan["grants"][0]["x"].update(sha256=plan["grants"][1]["x"]["sha256"]),
                  "does not have the hash its record binds"),
                 (lambda plan: plan["grants"][0].update(x=short),
                  r"changed allocation: .* is <c8 \[1, 14, 600\], expected <c8 \[1, 14, 612\]"),
                 (lambda plan: plan["carrier"].update(rx_ports=4), "not that of the carrier"),
                 (lambda plan: plan["opportunities"][0].update(index=3), "reads index 3"),
                 (lambda plan: plan["sources"][0]["h"].update(path=str(scratch / "absent.npy")), "cannot read"))
        for change, text in cases:
            plan = copy.deepcopy(self.plan)
            change(plan)
            with self.subTest(text=text), self.assertRaisesRegex(GridApplyError, text):
                replay.check_files(plan)
        with self.assertRaisesRegex(GridApplyError, "never overwritten"):
            replay.write_complex(scratch / "x_short.npy", (1, 14, 600), [0j] * 8400)
        with self.assertRaisesRegex(GridApplyError, "do not fill"):
            replay.write_complex(scratch / "wrong.npy", (2, 2), [0j] * 3)
        self.assertEqual(replay.array_header(scratch / "x_short.npy")[:2], ("<c8", [1, 14, 600]))

    def test_the_double_applies_the_reference_and_the_reader_verifies_it(self):
        self.assertEqual(self.status, 0)
        output = replay.open_output(self.output)
        self.assertEqual(len(output.descriptors), 6)
        self.assertEqual(output.report["gain"]["source"], "entry")
        base = small_grant()
        plane = 14 * base.grid_subcarriers
        gain = self.plan["gain"]["value"]
        # Opportunity 1 of the second realization against the scalar reference, every element.
        item, sequence = self.plan["opportunities"][4], 4
        grant = next(g for g in self.plan["grants"] if g["grant_id"] == item["grant_id"])
        source = next(s for s in self.plan["sources"] if s["realization_id"] == item["realization_id"])
        x = grants.read_npy(grant["x"]["path"])[2]
        h = grants.read_npy(source["h"]["path"])[2][item["index"] * 2 * plane:(item["index"] + 1) * 2 * plane]
        w = grants.read_npy(source["w"]["path"])[2][item["index"] * 2 * plane:(item["index"] + 1) * 2 * plane]
        expected = replay.apply((2, 1, 14, base.grid_subcarriers), x, h, w, gain)
        self.assertEqual((self.output / "prestore/0004.c64").read_bytes(), replay.grid_bytes(expected))
        self.assertEqual(output.grid_path(sequence).read_bytes(), replay.grid_bytes(replay.stored(expected)))
        self.assertEqual(output.descriptors[sequence]["grid"]["sha256"], digest_of(replay.stored(expected)))
        self.assertEqual(output.element(sequence, 1, 2, 180), replay.stored(expected)[plane + 2 * 612 + 180])
        with self.assertRaisesRegex(GridApplyError, "outside a grid"):
            output.element(sequence, 2, 0, 0)
        # The unit power the gain is defined by: one on the target-absent case within 1e-6.
        first = output.report["realizations"][0]["first"]
        self.assertEqual(first["sequence"], 0)
        self.assertAlmostEqual(first["unit_power"], 1.0, delta=1e-6)
        # A second reader and the access rule of the receiver side.
        shown = replay.summary(output, sequence, 1, 2, 180)
        self.assertEqual(shown["element"]["value"], [output.element(sequence, 1, 2, 180).real,
                                                     output.element(sequence, 1, 2, 180).imag])
        self.assertEqual((shown["verified_grids"], shown["grant_id"], shown["state"]), (6, "two-prb-offset.s08", "ready"))
        with self.assertRaisesRegex(GridApplyError, "opportunities 0 to 5"):
            replay.summary(output, 6)
        secrets = [entry["sha256"] for entry in json.loads((self.entry / "manifest.json").read_text())["files"].values()]
        self.assertEqual(replay.ingress_violations(self.output, secrets + ["truth", "target_present", "tb.bin"]), [])
        self.assertEqual(set(output.descriptors[0]), set(replay.INGRESS_FIELDS))

    def test_the_reader_refuses_a_damaged_output_and_names_what_the_receiver_side_could_read(self):
        def damaged(change):
            copy_dir = self.scratch() / "native"
            shutil.copytree(self.output, copy_dir)
            change(copy_dir)
            return copy_dir

        def flip(directory):
            data = bytearray((directory / "ingress/0002.c64").read_bytes())
            data[100] ^= 0x10
            (directory / "ingress/0002.c64").write_bytes(bytes(data))

        def edit(directory, change):
            lines = (directory / "ingress/ingress.jsonl").read_text().splitlines()
            record = json.loads(lines[1])
            change(record)
            lines[1] = json.dumps(record)
            (directory / "ingress/ingress.jsonl").write_text("\n".join(lines) + "\n")

        refusals = (
            (flip, "differs from its readback digest"),
            (lambda d: (d / "ingress/0003.c64").unlink(), "cannot read the grid of opportunity 3"),
            (lambda d: (d / "ingress/0003.c64").write_bytes(b"\0" * 8), "differs from its readback digest"),
            (lambda d: (d / "native.json").unlink(), "not a complete native output"),
            (lambda d: edit(d, lambda record: record.update(state="filling")), "does not describe a ready grid"),
            (lambda d: edit(d, lambda record: record.update(sequence=5)), "does not belong to opportunity 1"),
            (lambda d: edit(d, lambda record: record["grid"].update(shape=[2, 612, 14])), "does not describe a ready grid"),
            (lambda d: edit(d, lambda record: record["grid"].update(file="../prestore/0001.c64")), "outside ingress"),
            (lambda d: edit(d, lambda record: record.update(h_sha256="f" * 64)), "is not an o1-grid/v1 record"),
            (lambda d: (d / "ingress/ingress.jsonl").write_text(""), "different numbers of opportunities"))
        for change, text in refusals:
            with self.subTest(text=text), self.assertRaisesRegex(GridApplyError, text):
                replay.open_output(damaged(change))
        # The access check names every leak: a field, a nested field, a file and a forbidden string.
        leaks = damaged(lambda d: edit(d, lambda record: (record.update(h_sha256="f" * 64),
                                                          record["grid"].update(noise="w.npy"),
                                                          record["amplitude"].update(sigma=1e-5))))
        (leaks / "ingress/plan.json").write_text("{}")
        (leaks / "ingress/labels").mkdir()
        found = replay.ingress_violations(leaks, ["f" * 64, "w.npy", "never-present"])
        self.assertEqual(found, ["descriptor field 'h_sha256'", "descriptor field grid.noise",
                                 "descriptor field amplitude.sigma", "file ingress/labels", "file ingress/plan.json",
                                 f"the descriptors hold {'f' * 64!r}", "the descriptors hold 'w.npy'"])
        self.assertIn("unreadable", replay.ingress_violations(self.scratch())[0])

    def test_the_double_refuses_what_the_native_tool_refuses(self):
        for change in (lambda plan: plan.pop("gain"), lambda plan: plan["gain"].update(value=0.0),
                       lambda plan: plan["grants"][1]["x"].update(sha256=None),
                       lambda plan: plan["opportunities"][0].update(grant_id=None)):
            plan = copy.deepcopy(self.plan)
            change(plan)
            scratch = self.scratch()
            (scratch / "plan.json").write_text(json.dumps(plan))
            (scratch / "out").mkdir()
            with contextlib.redirect_stderr(io.StringIO()) as said:
                status = double.main(["--plan", str(scratch / "plan.json"), "--output-dir", str(scratch / "out")])
            self.assertEqual(status, 2)
            self.assertIn("grid_apply: input error:", said.getvalue())
            self.assertEqual(list((scratch / "out").iterdir()), [])
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(double.main(["--plan", str(self.root / "plan.json"), "--output-dir", str(self.output)]), 2)

    def test_command_line_builds_a_plan_and_reads_an_output_back(self):
        scratch = self.scratch()
        environment = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"}
        command = [sys.executable, "-m", "ocudu_cuda_isac.episodes.grid_apply"]
        made = subprocess.run(command + ["plan", "--episode", str(self.entry), "--run-id", "unit-cli", "--output",
                                         str(scratch / "plan.json")], capture_output=True, text=True, env=environment)
        self.assertEqual(made.returncode, 0, made.stderr)
        self.assertEqual(json.loads(made.stdout)["opportunities"], 6)
        self.assertEqual(json.loads((scratch / "plan.json").read_text())["grants"], self.plan["grants"])
        again = subprocess.run(command + ["plan", "--episode", str(self.entry), "--run-id", "unit-cli", "--output",
                                          str(scratch / "plan.json")], capture_output=True, text=True, env=environment)
        self.assertEqual(again.returncode, 2)
        self.assertIn("never overwritten", again.stderr)
        shown = subprocess.run(command + ["show", str(self.output), "--sequence", "3", "--rx", "1", "--symbol", "2",
                                          "--subcarrier", "180"], capture_output=True, text=True, env=environment)
        self.assertEqual(shown.returncode, 0, shown.stderr)
        record = json.loads(shown.stdout)
        self.assertEqual(record["readback_sha256"], hashlib.sha256((self.output / "ingress/0003.c64").read_bytes()).hexdigest())
        self.assertEqual(record["gain"]["value"], self.plan["gain"]["value"])
        missing = subprocess.run(command + ["show", str(scratch)], capture_output=True, text=True, env=environment)
        self.assertEqual(missing.returncode, 2)
        self.assertIn("not a complete native output", missing.stderr)


class FixturePlanTests(unittest.TestCase):
    def test_a_fixture_plan_records_its_override_and_checks_its_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            x = replay.write_complex(root / "x.npy", (2, 2, 1), [complex(v) for v in (1, 2, 3, 4)])
            grant = {"grant_id": "fixture", "record_sha256": "fixture:" + x["sha256"], "x": x,
                     "active": [replay.write_index(root / "active.npy", [(0, 0), (1, 0)])]}
            source = {"realization_id": "fx-s03",
                      "h": replay.write_complex(root / "h.npy", (1, 2, 2, 2, 1),
                                                [complex(v) for v in (1, 1, 1j, -1j, 0.5, 1, 2, 3)]),
                      "w": replay.write_complex(root / "w.npy", (1, 2, 2, 1), [0j] * 4)}
            opportunity = {"opportunity_id": "fx-s03.0000", "realization_id": "fx-s03", "index": 0, "grant_id": "fixture",
                           "sfn": 0, "slot": 0}
            plan = replay.fixture_plan("unit-s03", (2, 2, 2, 1), 2.0, "unit vector without an entry", [grant], [source],
                                       [opportunity])
            self.assertEqual(plan["gain"], {"value": 2.0, "source": "override", "reason": "unit vector without an entry",
                                            "rule": replay.RULE})
            self.assertIsNone(plan["sources"][0]["entry"])
            for bad, text in ((None, "names no grid gain"), (0.0, "must be positive"), (math.nan, "not finite")):
                with self.assertRaisesRegex(GridApplyError, text):
                    replay.fixture_plan("unit-s03", (2, 2, 2, 1), bad, "reason", [grant], [source], [opportunity])
            with self.assertRaisesRegex(GridApplyError, "records its reason"):
                replay.fixture_plan("unit-s03", (2, 2, 2, 1), 2.0, "", [grant], [source], [opportunity])
            with self.assertRaisesRegex(GridApplyError, r"changed allocation: .* expected <c8 \[1, 2, 1\]"):
                replay.fixture_plan("unit-s03", (2, 1, 2, 1), 2.0, "reason", [grant], [source], [opportunity])
            # The double runs it: the four answers, twice.
            (root / "out").mkdir()
            replay.write_plan(plan, root / "plan.json")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(double.main(["--plan", str(root / "plan.json"), "--output-dir", str(root / "out")]), 0)
            stored = struct.unpack("<8f", (root / "out/ingress/0000.c64").read_bytes())
            self.assertEqual(stored, (2.0, 6.0, 4.0, -8.0, 13.0, 0.0, 28.0, 0.0))
            report = json.loads((root / "out/native.json").read_text())
            self.assertEqual((report["gain"]["source"], report["gain"]["reason"]), ("override", "unit vector without an entry"))


class NativeCoreTests(unittest.TestCase):
    """The host-free core, built on its own: its CTest cases and its agreement with the Python reference."""

    @classmethod
    def setUpClass(cls):
        cls.missing = [tool for tool in ("cmake", "ctest") if not shutil.which(tool)]
        if not any(shutil.which(compiler) for compiler in ("c++", "g++", "clang++")):
            cls.missing.append("a C++17 compiler")
        if cls.missing:
            return
        cls.temporary = tempfile.TemporaryDirectory()
        cls.build = Path(cls.temporary.name) / "build"
        steps = (["cmake", "-S", str(REPLAY), "-B", str(cls.build)], ["cmake", "--build", str(cls.build)])
        cls.log, cls.status = "", 0
        for step in steps:
            result = subprocess.run(step, capture_output=True, text=True, timeout=900, check=False)
            cls.log += result.stdout + result.stderr
            cls.status = cls.status or result.returncode
        cls.test = cls.build / "replay_core_test"

    @classmethod
    def tearDownClass(cls):
        if not cls.missing:
            cls.temporary.cleanup()

    def setUp(self):
        if self.missing:
            self.skipTest("the native core needs " + ", ".join(self.missing))
        if self.status:
            self.fail("the native core does not build:\n" + self.log[-3000:])

    def test_core_builds_without_a_warning_and_links_only_its_module(self):
        self.assertNotRegex(self.log, r"warning:")
        link = (self.build / "CMakeFiles/replay_core_test.dir/link.txt").read_text()
        self.assertIn("libreplay_core.a", link)
        self.assertNotIn("ocudu", link.lower().replace(str(ROOT).lower(), ""))
        self.assertEqual(re.findall(r"\S+\.(?:a|so)\b", link), ["libreplay_core.a"])
        # The core and its unit target include nothing of OCUDU and nothing but the standard library.
        for name in ("replay_core.h", "replay_core.cpp", "tests/replay_core_test.cpp"):
            for include in re.findall(r'#include [<"]([^>"]+)[>"]', (REPLAY / name).read_text()):
                self.assertTrue(include == "replay_core.h" or "/" not in include or include.startswith("sys/"), include)
        # The OCUDU-bound files are only built where the pinned build is imported.
        cmake = (REPLAY / "CMakeLists.txt").read_text()
        self.assertIn("if(TARGET ocudu_pin_fixture)", cmake)
        self.assertNotIn("grid_apply", "".join(path.name for path in self.build.iterdir()))

    def test_every_native_unit_case_passes(self):
        listed = subprocess.run([str(self.test), "--list"], capture_output=True, text=True, check=True).stdout.split()
        self.assertEqual(tuple(listed), CASES)
        result = subprocess.run(["ctest", "--output-on-failure"], cwd=self.build, capture_output=True, text=True,
                                timeout=600, check=False)
        self.assertEqual(result.returncode, 0, result.stdout[-3000:])
        self.assertIn(f"100% tests passed, 0 tests failed out of {len(CASES)}", result.stdout)
        for name in CASES:
            self.assertRegex(result.stdout, rf"replay_core_{name} \.+\s+Passed")
        unknown = subprocess.run([str(self.test), "no_such_case"], capture_output=True, text=True, check=False)
        self.assertEqual(unknown.returncode, 2)

    def test_native_kernel_and_python_reference_give_the_same_bits(self):
        for seed, shape, gain in ((7, (4, 2, 3, 40), GAIN), (20261120, (2, 1, 14, 24), 1.0), (99, (1, 4, 2, 16), 2.0),
                                  (3, (4, 1, 14, 612), GAIN)):
            with self.subTest(shape=shape, gain=gain):
                result = subprocess.run([str(self.test), "--digest", str(seed), *map(str, shape), repr(gain)],
                                        capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 0, result.stderr)
                native = dict(line.split() for line in result.stdout.splitlines())
                x, h, w = generated(seed, shape)
                written = replay.apply(shape, x, h, w, gain)
                self.assertEqual(native["written"], digest_of(written))
                self.assertEqual(native["stored"], digest_of(replay.stored(written)))
        refused = subprocess.run([str(self.test), "--digest", "1", "1", "1", "1", "1", "0"], capture_output=True,
                                 text=True, check=False)
        self.assertEqual(refused.returncode, 2)
        self.assertIn("finite and positive", refused.stderr)


class TreeTests(unittest.TestCase):
    FILES = ("native/ocudu_adapter/replay/CMakeLists.txt", "native/ocudu_adapter/replay/replay_core.h",
             "native/ocudu_adapter/replay/replay_core.cpp", "native/ocudu_adapter/replay/replay_plan.h",
             "native/ocudu_adapter/replay/pinned_grid.h", "native/ocudu_adapter/replay/grid_apply.cpp",
             "native/ocudu_adapter/replay/tests/replay_core_test.cpp", "src/ocudu_cuda_isac/episodes/grid_apply.py",
             "tests/unit/test_grid_apply.py", "tests/unit/grid_apply_double.py", "tests/integration/grid_apply.py",
             "configs/integration/grid_apply.json", "scripts/accept_s2_3.sh")

    def test_files_are_ascii_and_do_not_name_the_project(self):
        # The working name lives on line 1 of the top-level README and nowhere else.
        name = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()[0].lstrip("# ").split(":")[0].strip()
        for relative in self.FILES:
            data = (ROOT / relative).read_bytes()
            self.assertTrue(data.isascii(), relative)
            if name and name.lower() != "ocudu-cuda-isac":
                self.assertNotIn(name.lower(), data.decode("ascii").lower(), relative)

    def test_the_tool_uses_the_pinned_writer_and_reader_and_keeps_the_two_sides_apart(self):
        grid = (REPLAY / "pinned_grid.h").read_text()
        self.assertIn("grid->get_writer().put(plane, symbol, 0, ocudu::span<const ocudu::cf_t>(row, count))", grid)
        self.assertIn("grid->get_reader().get(ocudu::span<ocudu::cf_t>(row, count), plane, symbol, 0)", grid)
        tool = (REPLAY / "grid_apply.cpp").read_text()
        self.assertIn("ocudu::create_resource_grid_factory()", tool)
        self.assertIn("ocudu::MAX_NSYMB_PER_SLOT, ocudu::MAX_NOF_SUBCARRIERS", tool)  # sized like the pinned test's rx_grid
        self.assertIn("const session     replay(spec);", tool)
        self.assertLess(tool.index("const session     replay(spec);"), tool.index("make_directory(ingress_path)"))
        for word in ("tb.bin", "truth", "target_present"):
            self.assertNotIn(word, tool)
        # The descriptor type of the core has no field for the channel, the noise or the block.
        header = (REPLAY / "replay_core.h").read_text()
        descriptor = header[header.index("struct o1_descriptor {"):header.index("/// Names one fill of one grid.")]
        self.assertEqual(re.findall(r"\b(\w+)\s*(?:=[^;]*)?;", descriptor),
                         ["run_id", "opportunity_id", "grant_id", "grant_record_sha256", "active_sha256", "active_re",
                          "sfn", "slot", "shape", "grid_gain", "storage", "readback_sha256"])
        self.assertEqual(tuple(json.loads((ROOT / "configs/integration/grid_apply.json").read_text())["tolerance"]),
                         ("unit_scale_absolute", "unit_scale_relative", "gain_rule_relative", "storage_relative",
                          "unit_power", "noise_db", "power_relative"))


if __name__ == "__main__":
    unittest.main()
