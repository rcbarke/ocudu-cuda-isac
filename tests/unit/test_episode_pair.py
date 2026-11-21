#!/usr/bin/env python3
"""Sprint 2.2 unit gate (selector episode_pair): python tests/unit/test_episode_pair.py -v

Standard library only. No Sionna, no NumPy, no OCUDU build and no GPU: the grant schedule, the
episode entry, its loader, the analytic two-path model and the episode set are checked on small
fixtures with known answers.
"""

import cmath
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ocudu_cuda_isac.episodes import episode as episodes  # noqa: E402
from ocudu_cuda_isac.episodes import episode_set as sets  # noqa: E402
from ocudu_cuda_isac.episodes import grant as grants  # noqa: E402
from ocudu_cuda_isac.episodes import schedule as schedules  # noqa: E402
from ocudu_cuda_isac.episodes.episode import EpisodeError, TC_PER_SECOND  # noqa: E402
from ocudu_cuda_isac.episodes.schedule import ScheduleError  # noqa: E402

GRANTS = ROOT / "configs/grants"
EPISODES = ROOT / "configs/episodes"
TDD = {"pattern": "DDDDDDSUUU", "ul_slots": [7, 8, 9]}
GEOMETRY = {"ue_tx_m": [0.0, 0.0, 1.5], "gnb_rx_m": [20.0, 0.0, 3.0]}
WAVELENGTH = episodes.SPEED_OF_LIGHT_M_S / 3.5e9
OFFSETS = [[0.0, y * WAVELENGTH, 0.0] for y in (-0.75, -0.25, 0.25, 0.75)]
TARGET = {"model": "constant_rcs_cuboid", "sigma_m2": 1.0, "size_m": [0.3, 0.3, 0.3],
          "motion": {"kind": "constant_velocity", "position_m": [10.0, 25.0, 1.5], "velocity_m_s": [0.0, 2.0, 0.0]}}


def two_port_grant():
    """The two-block unit grant of Sprint 2.1 with a second receive port, to test the receive order."""
    record = json.loads((GRANTS / "two_prb_offset.json").read_text())
    record["rx_ports"] = [0, 1]
    return grants.build(record)


def schedule_of(base, periods=2, **changes):
    record = {"schema": schedules.SCHEMA, "schedule_id": "unit", "tdd": dict(TDD), "start_sfn": 0,
              "periods": periods, "payload": "fixed"}
    record.update(changes)
    return schedules.build(record, base)


def description(base, analytic=False, labels=None, target=None):
    geometry = dict(GEOMETRY, rx_element_offsets_m=OFFSETS[:len(base.rx_ports)])
    result = {"geometry": geometry, "array": {}, "scene": {"kind": "free_space"}, "target": target or TARGET,
              "solver": {}, "time_evolution": {}, "noise": {"snr_db": 30.0, "sigma": 1e-5, "seed": 1},
              "calibration": {}, "software": {}, "provenance": {}}
    if analytic:
        result["analytic_model"] = episodes.ANALYTIC_MODEL
    if labels is not None:
        result["labels"] = labels
    return result


def truth_of(schedule, target=TARGET, carrier=3.5e9):
    rows = []
    for item in schedule.opportunities:
        position, velocity = episodes.target_state(target["motion"], item.slot_start_tc / TC_PER_SECOND)
        rows.append({"opportunity": item.index, "position_m": position, "velocity_m_s": velocity,
                     "bistatic": episodes.bistatic(GEOMETRY, position, velocity, carrier)})
    return rows


def grid(schedule, function, with_tx=True):
    """Flat C-order values of function(opportunity, rx, symbol, subcarrier) over the axes of H or W."""
    h_shape, _ = episodes.shapes(schedule)
    return [function(o, r, m, k) for o in range(h_shape[0]) for r in range(h_shape[1])
            for _ in range(h_shape[2] if with_tx else 1) for m in range(h_shape[3]) for k in range(h_shape[4])]


def encoded(o, r, m, k):
    """Indices in the value: opportunity, receive port and symbol in the real part, subcarrier in the imaginary."""
    return complex(1000 * o + 100 * r + m + 1, k + 1)


def model_pair(schedule, target=TARGET, sigma=None):
    """Both cases of the analytic free-space model on a schedule, as flat lists. Pure Python."""
    base = schedule.base
    geometry = dict(GEOMETRY, rx_element_offsets_m=OFFSETS[:len(base.rx_ports)])
    model = dict(target, sigma_m2=sigma if sigma is not None else target.get("sigma_m2", 1.0))
    absent, present = [], []
    for item in schedule.opportunities:
        cache = {}
        for symbol in range(grants.SYMBOLS_PER_SLOT):
            cache[symbol] = episodes.free_space_paths(geometry, model, item.symbol_time_tc(symbol) / TC_PER_SECOND,
                                                      base.carrier_hz, True)
        for rx in range(len(base.rx_ports)):
            for symbol in range(grants.SYMBOLS_PER_SLOT):
                for k in range(base.grid_subcarriers):
                    offset = base.subcarrier_offset_hz(k)
                    direct = episodes.path_response(cache[symbol][:1], base.carrier_hz, offset, rx)
                    absent.append(direct)
                    present.append(direct + episodes.path_response(cache[symbol][1:], base.carrier_hz, offset, rx))
    return absent, present


class Fixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def stored(self, schedule=None, name="entry", **arrays):
        schedule = schedule or schedule_of(two_port_grant())
        values = {"a": grid(schedule, encoded), "b": grid(schedule, lambda o, r, m, k: encoded(o, r, m, k) + 0.5),
                  "w": grid(schedule, lambda o, r, m, k: complex(k, -m), with_tx=False)}
        values.update(arrays)
        episodes.export(self.root / name, "ep-unit", schedule, description(schedule.base), values, truth_of(schedule))
        return self.root / name, schedule


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.base = grants.load(GRANTS / "demo_51prb_4rx.json")

    def test_opportunities_follow_the_tdd_pattern_and_the_times_of_the_record(self):
        schedule = schedule_of(self.base, periods=2)
        items = schedule.opportunities
        self.assertEqual([(item.period, item.sfn, item.slot, item.absolute_slot) for item in items],
                         [(0, 0, 7, 7), (0, 0, 8, 8), (0, 0, 9, 9), (1, 0, 17, 17), (1, 0, 18, 18), (1, 0, 19, 19)])
        # 30 kHz: every slot is 0.5 ms = 983,040 T_c. Symbol 2 starts 145,920 T_c and symbol 11 777,216 T_c later.
        self.assertEqual([item.slot_start_tc for item in items], [983040 * slot for slot in (7, 8, 9, 17, 18, 19)])
        self.assertEqual(items[0].symbol_time_tc(2), 7 * 983040 + 145920)
        self.assertEqual(items[0].symbol_time_tc(11) - items[0].symbol_time_tc(2), 631296)
        self.assertEqual(items[1].symbol_time_tc(2) - items[0].symbol_time_tc(11), 351744)
        self.assertAlmostEqual(1e6 * 631296 / TC_PER_SECOND, 321.09, places=2)
        self.assertAlmostEqual(1e6 * 351744 / TC_PER_SECOND, 178.91, places=2)
        self.assertEqual(items[3].symbol_time_tc(2) - items[0].symbol_time_tc(2), TC_PER_SECOND // 200)  # 5 ms
        self.assertEqual(schedule.duration_tc(), 20 * 983040)
        self.assertEqual([item.symbol_time_tc(s) for item in items[:1] for s in range(14)],
                         [7 * 983040 + entry["useful_start_tc"] for entry in self.base.symbol_times()])

    def test_every_slot_index_has_one_legal_grant_with_the_same_data_and_its_own_pilots(self):
        schedule = schedule_of(self.base, periods=3)
        self.assertEqual(sorted(schedule.grants), ["demo-51prb-4rx", "demo-51prb-4rx.s08", "demo-51prb-4rx.s09",
                                                   "demo-51prb-4rx.s17", "demo-51prb-4rx.s18", "demo-51prb-4rx.s19"])
        self.assertIs(schedule.opportunities[0].grant, self.base)
        self.assertIs(schedule.opportunities[6].grant, self.base)  # slot 7 of the next frame: the same waveform
        other = schedule.grants["demo-51prb-4rx.s17"]
        self.assertEqual((other.slot, other.tb, other.data_index, other.pilot_index),
                         (17, self.base.tb, self.base.data_index, self.base.pilot_index))
        self.assertNotEqual(other.pilot_reference, self.base.pilot_reference)
        self.assertEqual(grants.build(other.record()).derived(), other.derived())
        described = schedule.describe_grants()
        self.assertTrue(described["demo-51prb-4rx"]["is_base_grant"])
        self.assertEqual(described["demo-51prb-4rx.s17"]["symbol_useful_start_tc"][2], 145920)
        self.assertEqual(described["demo-51prb-4rx.s17"]["pilots"], 612)

    def test_frame_number_wraps_and_a_schedule_is_reproducible(self):
        schedule = schedule_of(self.base, periods=4, start_sfn=1023)
        self.assertEqual([item.sfn for item in schedule.opportunities], [1023] * 6 + [0] * 6)
        again = schedule_of(self.base, periods=4, start_sfn=1023)
        self.assertEqual([item.describe() for item in schedule.opportunities],
                         [item.describe() for item in again.opportunities])
        self.assertEqual(schedule.record(), again.record())

    def test_unequal_slots_add_up_to_one_frame(self):
        # 60 kHz: the longer first prefix falls on every second slot.
        self.assertEqual((schedules.slot_duration_tc(60, 0), schedules.slot_duration_tc(60, 1)), (492032, 491008))
        for scs in (15, 30, 60, 120):
            total = sum(schedules.slot_duration_tc(scs, slot) for slot in range(schedules.slots_per_frame(scs)))
            self.assertEqual(total, TC_PER_SECOND // 100)

    def test_illegal_schedules_are_refused(self):
        cases = (({"tdd": {"pattern": "DDDDDDSUUU", "ul_slots": [6]}}, "not an uplink slot"),
                 ({"tdd": {"pattern": "DDDSUUU", "ul_slots": [4]}}, "does not repeat in a frame"),
                 ({"tdd": {"pattern": "DDDDDDSUUU", "ul_slots": [8, 9]}}, "base grant's slot 7"),
                 ({"tdd": {"pattern": "DDDDDDSUUU", "ul_slots": [9, 7]}}, "increase without repetition"),
                 ({"tdd": {"pattern": "DDXDDDSUUU", "ul_slots": [7]}}, "D, S and U"),
                 ({"payload": "per_opportunity"}, 'payload must be "fixed"'), ({"periods": 0}, "periods"),
                 ({"start_sfn": 1024}, "start_sfn"), ({"schema": "pusch-grant-schedule/v2"}, "unsupported"))
        for changes, message in cases:
            with self.subTest(changes=changes), self.assertRaisesRegex(ScheduleError, message):
                schedule_of(self.base, **changes)
        with self.assertRaisesRegex(ScheduleError, "validated grant"):
            schedules.build(schedule_of(self.base).record(), {"grant_id": "x"})
        with self.assertRaisesRegex(ScheduleError, "requires exactly"):
            schedules.build({"schema": schedules.SCHEMA}, self.base)


class StoreTests(Fixture):
    def test_encoded_indices_survive_storage_with_axes_order_times_and_hashes(self):
        directory, schedule = self.stored()
        episode = episodes.open_episode(directory)
        self.assertEqual(episode.h_shape, [6, 2, 1, 14, 612])  # the singleton transmit axis is kept
        self.assertEqual(episode.w_shape, [6, 2, 14, 612])
        for o, r, m, k in ((0, 0, 0, 0), (5, 1, 13, 611), (3, 1, 2, 180), (2, 0, 11, 203)):
            self.assertEqual(episode.h("a", o, r, m, k), encoded(o, r, m, k))
            self.assertEqual(episode.h("b", o, r, m, k), encoded(o, r, m, k) + 0.5)
            self.assertEqual(episode.w(o, r, m, k), complex(k, -m))
            self.assertEqual(episode.echo(o, r, m, k), 0.5)
        # Little-endian complex64 in C order: the last axis is fastest, then symbol, transmit port, receive port.
        raw = (directory / "a_h.npy").read_bytes()
        header = 10 + struct.unpack("<H", raw[8:10])[0]
        self.assertIn(b"'descr': '<c8', 'fortran_order': False, 'shape': (6, 2, 1, 14, 612)", raw[:header])
        flat = ((3 * 2 + 1) * 14 + 2) * 612 + 180
        self.assertEqual(raw[header + 8 * flat:header + 8 * flat + 8], struct.pack("<2f", 3103.0, 181.0))
        self.assertEqual(len(raw), header + 8 * 6 * 2 * 14 * 612)
        # The exact requested symbol times and the grant's frequency offsets.
        self.assertEqual(episode.time_tc(3, 11), 17 * 983040 + 777216)
        self.assertEqual(episode.time_s(0, 2), (7 * 983040 + 145920) / TC_PER_SECOND)
        self.assertEqual((episode.offset_hz(0), episode.offset_hz(180), episode.offset_hz(611)),
                         (-9180000, -3780000, 9150000))
        manifest = episode.manifest
        self.assertEqual(manifest["axes"]["h"], ["opportunity", "rx_port", "tx_port", "symbol", "subcarrier"])
        self.assertEqual(manifest["axes"]["subcarrier_offset_hz"]["step"], 30000)
        self.assertIn("absolute delay and amplitude", manifest["axes"]["units"]["h"])
        for name, entry in manifest["files"].items():
            self.assertEqual(entry["sha256"], hashlib.sha256((directory / name).read_bytes()).hexdigest())
        self.assertEqual([case["target_present"] for case in manifest["cases"]], [False, True])
        self.assertEqual(len({case["realization_id"] for case in manifest["cases"]}), 2)
        self.assertEqual(manifest["grants"]["two-prb-offset"]["x"], None)
        coefficient = episode.coefficient("b", 3, 1, 2, 180)
        self.assertTrue(coefficient["is_pilot"])
        self.assertEqual((coefficient["sfn"], coefficient["slot"], coefficient["offset_hz"]), (0, 17, -3780000))

    def test_bytes_and_sequences_store_the_same_file(self):
        schedule = schedule_of(two_port_grant(), periods=1)
        values = grid(schedule, encoded)
        packed = struct.pack(f"<{2 * len(values)}f", *[part for value in values for part in (value.real, value.imag)])
        first, _ = self.stored(schedule, "from-list")
        second, _ = self.stored(schedule, "from-bytes", a=packed)
        self.assertEqual((first / "a_h.npy").read_bytes(), (second / "a_h.npy").read_bytes())

    def test_a_transposed_array_is_caught_by_the_encoded_indices(self):
        schedule = schedule_of(two_port_grant(), periods=1)  # three opportunities, two ports
        swapped = [encoded(o, r, m, k) for r in range(2) for o in range(3) for m in range(14) for k in range(612)]
        swapped = swapped[:len(grid(schedule, encoded))]
        directory, _ = self.stored(schedule, a=swapped)
        episode = episodes.open_episode(directory)  # a valid file: only its contents tell
        self.assertEqual(episode.h("a", 0, 0, 5, 7), encoded(0, 0, 5, 7))
        self.assertNotEqual(episode.h("a", 1, 0, 5, 7), encoded(1, 0, 5, 7))

    def test_the_loader_refuses_a_damaged_entry(self):
        source, _ = self.stored()

        def flipped(copy):
            data = bytearray((copy / "b_h.npy").read_bytes())
            data[-5] ^= 0x40
            (copy / "b_h.npy").write_bytes(bytes(data))

        def swapped_axes(copy):
            raw = (copy / "a_h.npy").read_bytes()
            body = raw[10 + struct.unpack("<H", raw[8:10])[0]:]
            header = "{'descr': '<c8', 'fortran_order': False, 'shape': (6, 14, 1, 2, 612), }"
            header += " " * (-(10 + len(header) + 1) % 64) + "\n"
            blob = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode("latin1") + body
            (copy / "a_h.npy").write_bytes(blob)
            rewrite(copy, lambda m: m["files"]["a_h.npy"].update(sha256=hashlib.sha256(blob).hexdigest(), bytes=len(blob),
                                                                 shape=[6, 14, 1, 2, 612]))

        def rewrite(copy, change):
            manifest = json.loads((copy / "manifest.json").read_text())
            change(manifest)
            (copy / "manifest.json").write_text(json.dumps(manifest))

        def partial(copy):
            data = (copy / "w.npy").read_bytes()
            (copy / "w.npy").write_bytes(data[:len(data) // 2])

        def fortran(copy):
            raw = (copy / "a_h.npy").read_bytes().replace(b"'fortran_order': False", b"'fortran_order': True ")
            (copy / "a_h.npy").write_bytes(raw)
            rewrite(copy, lambda m: m["files"]["a_h.npy"].update(sha256=hashlib.sha256(raw).hexdigest()))

        def same_channel(copy):
            shutil.copyfile(copy / "a_h.npy", copy / "b_h.npy")
            rewrite(copy, lambda m: m["files"].update({"b_h.npy": m["files"]["a_h.npy"]}))

        cases = (
            (flipped, "differs from the size or hash"),
            (swapped_axes, "not a C-ordered complex64 array of shape"),
            (lambda copy: (copy / "w.npy").unlink(), "cannot read w.npy"),
            (partial, "differs from the size or hash"),
            (lambda copy: (copy / "manifest.json").unlink(), "not a complete episode entry"),
            (fortran, "not a C-ordered complex64 array"),
            (lambda copy: rewrite(copy, lambda m: m["opportunities"][2].update(slot_start_tc=1)), "opportunities differ"),
            (lambda copy: rewrite(copy, lambda m: m["grant"]["slot"].update(slot=18)), "opportunities differ"),
            (lambda copy: rewrite(copy, lambda m: m["grant"]["allocation"].update(prb_start=40)), "grant schedule is invalid"),
            (lambda copy: rewrite(copy, lambda m: m["axes"]["subcarrier_offset_hz"].update(first=0)), "axes differ"),
            (lambda copy: rewrite(copy, lambda m: m["grants"]["two-prb-offset"].update(slot=3)), "grants differ"),
            (lambda copy: rewrite(copy, lambda m: m["cases"].reverse()), "cases must be"),
            (lambda copy: rewrite(copy, lambda m: m["cases"][1]["truth"].pop()), "one truth entry per opportunity"),
            (lambda copy: rewrite(copy, lambda m: m.update(schema="channel-episode/v2")), "not a channel-episode/v1"),
            (same_channel, "equals the target-absent"),
        )
        for number, (damage, message) in enumerate(cases):
            copy = self.root / f"copy-{number}"
            shutil.copytree(source, copy)
            damage(copy)
            with self.subTest(message=message), self.assertRaisesRegex(EpisodeError, message):
                episodes.open_episode(copy)
        episodes.open_episode(source)

    def test_export_refuses_bad_inputs_and_writes_nothing(self):
        schedule = schedule_of(two_port_grant(), periods=1)
        good = {"a": grid(schedule, encoded), "b": grid(schedule, lambda o, r, m, k: encoded(o, r, m, k) + 1),
                "w": grid(schedule, lambda o, r, m, k: 0j, with_tx=False)}
        base, truth = schedule.base, truth_of(schedule)
        nan = list(good["a"])
        nan[7] = complex(float("nan"), 0)
        one_port = description(base)
        one_port["geometry"]["rx_element_offsets_m"] = OFFSETS[:1]
        cases = (
            (dict(arrays=dict(good, a=good["a"][:-1])), "the grant schedule needs"),
            (dict(arrays=dict(good, a=nan)), "nonfinite"),
            (dict(arrays=dict(good, a=[0j] * len(good["a"]))), "all zero"),
            (dict(arrays=dict(good, b=good["a"])), "equals the target-absent"),
            (dict(arrays={"a": good["a"], "b": good["b"]}), 'exactly "a", "b" and "w"'),
            (dict(truth=truth[:-1]), "one truth entry per opportunity"),
            (dict(description=one_port), "one offset per receive port"),
            (dict(description={"geometry": {}}), "requires geometry"),
            (dict(description=dict(description(base), analytic_model="other/v1")), "unknown analytic model"),
            (dict(episode_id="a/b"), "ASCII identifier"),
            (dict(schedule=schedule.record()), "validated grant schedule"),
            (dict(waveforms={"other-grant": {"x_sha256": "0"}}), "does not use"),
        )
        for number, (changes, message) in enumerate(cases):
            arguments = dict(episode_id="ep-unit", schedule=schedule, description=description(base), arrays=good,
                             truth=truth)
            arguments.update(changes)
            target = self.root / f"refused-{number}"
            with self.subTest(message=message), self.assertRaisesRegex(EpisodeError, message):
                episodes.export(target, **arguments)
            self.assertFalse(target.exists())
        # One period of this schedule uses slots 7 to 9. The base grant's own slot 17 comes in the next period.
        self.assertEqual(sorted(schedule.grants), ["two-prb-offset.s07", "two-prb-offset.s08", "two-prb-offset.s09"])
        episodes.export(self.root / "kept", "ep-unit", schedule, description(base), good, truth,
                        waveforms={"two-prb-offset.s07": {"x_sha256": "ab" * 32, "store": "s22"}})
        with self.assertRaisesRegex(EpisodeError, "never overwritten"):
            episodes.export(self.root / "kept", "ep-unit", schedule, description(base), good, truth)
        kept = episodes.open_episode(self.root / "kept").manifest["grants"]
        self.assertEqual((kept["two-prb-offset.s07"]["x"]["store"], kept["two-prb-offset.s08"]["x"]), ("s22", None))
        self.assertEqual(list(self.root.glob("tmp*")), [])  # no partial directory is left behind


class PhysicsTests(Fixture):
    def test_free_space_model_gives_the_known_reference_case(self):
        # 20 m link at 3.5 GHz, a 1 m2 reflector 25 m off its midpoint: the values Sionna RT 2.2.0 traced.
        geometry = dict(GEOMETRY, rx_element_offsets_m=OFFSETS)
        direct, echo = episodes.free_space_paths(geometry, TARGET, 0.0, 3.5e9, True)
        self.assertAlmostEqual(direct[0], 3.398558642879076e-4, places=15)
        self.assertAlmostEqual(1e9 * direct[1], 66.90018572825998, places=9)
        self.assertEqual(direct[2], [0.0] * 4)  # the link is broadside to the array
        self.assertAlmostEqual(echo[0], 2.6480549582849525e-06, places=17)
        self.assertAlmostEqual(1e9 * echo[1], 179.76902259494506, places=9)
        for phase, traced in zip(echo[2], (1.9146188, -1.4561889, 1.4561889, -1.9146188)):
            self.assertAlmostEqual(math.remainder(phase - traced, 2 * math.pi), 0.0, places=5)
        self.assertEqual(len(episodes.free_space_paths(geometry, TARGET, 0.0, 3.5e9, False)), 1)
        reference = episodes.bistatic(GEOMETRY, [10.0, 25.0, 1.5], [0.0, 2.0, 0.0], 3.5e9)
        self.assertAlmostEqual(reference["excess_delay_ns"], 112.86883686668509, places=9)
        self.assertAlmostEqual(reference["doppler_hz"], -43.32534579012217, places=9)
        # The diagnostic geometry: about 5 ns and 9 Hz, an axis test only.
        diagnostic = episodes.bistatic(GEOMETRY, [9.0, 4.0, 1.5], [0.0, 1.0, 0.0], 3.5e9)
        self.assertAlmostEqual(diagnostic["excess_delay_ns"], 5.314, places=3)
        self.assertAlmostEqual(diagnostic["doppler_hz"], -8.699, places=3)
        approaching = episodes.bistatic(GEOMETRY, [10.0, 25.0, 1.5], [0.0, -2.0, 0.0], 3.5e9)
        self.assertAlmostEqual(approaching["doppler_hz"], 43.32534579012217, places=9)

    def test_a_single_path_keeps_its_phase_differences_at_two_unequal_intervals(self):
        base = two_port_grant()
        schedule = schedule_of(base, periods=1)
        amplitude, delay, doppler = 2.5e-6, 179.8e-9, -43.3

        def echo(o, r, m, k):
            time_s = schedule.opportunities[o].symbol_time_tc(m) / TC_PER_SECOND
            return amplitude * cmath.exp(-2j * math.pi * (base.carrier_hz + base.subcarrier_offset_hz(k)) * delay) \
                * cmath.exp(2j * math.pi * doppler * time_s)

        def direct(o, r, m, k):
            return 3.4e-4 * cmath.exp(-2j * math.pi * (base.carrier_hz + base.subcarrier_offset_hz(k)) * 66.9e-9)

        directory, _ = self.stored(schedule, a=grid(schedule, direct),
                                   b=grid(schedule, lambda o, r, m, k: direct(o, r, m, k) + echo(o, r, m, k)))
        episode = episodes.open_episode(directory)
        bound = episodes.c64_bound(3.4e-4, 3.4e-4) / amplitude  # phase error of the echo recovered from b - a
        self.assertLess(bound, 1e-4)
        within = episode.phase_advance((0, 2), (0, 11), 1, 180)
        between = episode.phase_advance((0, 11), (1, 2), 1, 180)
        self.assertAlmostEqual(within["interval_s"], 631296 / TC_PER_SECOND, places=15)
        self.assertAlmostEqual(between["interval_s"], 351744 / TC_PER_SECOND, places=15)
        self.assertAlmostEqual(within["measured_rad"], 2 * math.pi * doppler * within["interval_s"], delta=2 * bound)
        self.assertAlmostEqual(between["measured_rad"], 2 * math.pi * doppler * between["interval_s"], delta=2 * bound)
        self.assertAlmostEqual(within["measured_rad"], -0.08736, places=4)
        self.assertAlmostEqual(between["measured_rad"], -0.04867, places=4)
        # Along frequency the phase falls by 2 pi tau per hertz: 24 subcarriers of 30 kHz.
        step = cmath.phase(episode.echo(0, 0, 2, 204) * episode.echo(0, 0, 2, 180).conjugate())
        self.assertAlmostEqual(step, -2 * math.pi * 24 * 30e3 * delay, delta=2 * bound)
        # A conjugated time axis would flip the sign of both steps.
        self.assertLess(within["measured_rad"], 0)

    def test_a_weak_echo_needs_a_scale_aware_tolerance(self):
        schedule = schedule_of(two_port_grant(), periods=1)
        direct, weak = 3.4e-4, 3.4e-4 * 10 ** (-80 / 20)  # an echo 80 dB below the direct path
        directory, _ = self.stored(schedule, a=grid(schedule, lambda o, r, m, k: complex(direct, 0)),
                                   b=grid(schedule, lambda o, r, m, k: complex(direct, weak)))
        episode = episodes.open_episode(directory)
        recovered = episode.echo(2, 1, 13, 611)
        bound = episodes.c64_bound(direct, direct)
        self.assertLessEqual(abs(recovered - complex(0, weak)), bound)
        self.assertLess(bound, 0.01 * weak)  # the echo keeps two significant digits 80 dB down
        # The absolute term of the grid tolerance is far above this echo and would accept its absence.
        self.assertGreater(grants.STORAGE["absolute"], 10 * weak)
        self.assertGreater(abs(0j - complex(0, weak)), bound)
        self.assertEqual(episodes.c64_bound(1.0), 2 * math.sqrt(2) * 2.0 ** -24)

    def test_the_analytic_model_reads_back_from_a_stored_pair(self):
        schedule = schedule_of(grants.load(GRANTS / "two_prb_offset.json"), periods=1)
        absent, present = model_pair(schedule)
        episodes.export(self.root / "model", "ep-model", schedule, description(schedule.base, analytic=True),
                        {"a": absent, "b": present, "w": [0j] * len(absent)}, truth_of(schedule))
        episode = episodes.open_episode(self.root / "model")
        shown = episodes.summary(episode)
        self.assertEqual(shown["coefficient"]["symbol"], 2)
        self.assertLess(shown["coefficient"]["analytic_relative_error"], 1e-6)
        self.assertTrue(all(step["error_rad"] < 1e-3 for step in shown["phase_advance"]))
        self.assertLess(shown["phase_advance"][0]["measured_rad"], 0)  # the target recedes
        self.assertAlmostEqual(abs(episode.analytic_echo(0, 0, 2, 180)), 2.648e-6, delta=2e-9)
        with self.assertRaisesRegex(EpisodeError, "outside"):
            episode.h("b", 0, 1, 2, 180)

    def test_target_motion(self):
        position, velocity = episodes.target_state(TARGET["motion"], 0.5)
        self.assertEqual((position, velocity), ([10.0, 26.0, 1.5], [0.0, 2.0, 0.0]))
        walk = {"kind": "waypoints", "points": [[0, 0, 1], [3, 0, 1], [3, 4, 1]], "speed_m_s": 2.0, "start_s": 1.0}
        self.assertEqual(episodes.target_state(walk, 0.0), ([2.0, 0.0, 1.0], [2.0, 0.0, 0.0]))
        self.assertEqual(episodes.target_state(walk, 1.0), ([3.0, 1.0, 1.0], [0.0, 2.0, 0.0]))
        with self.assertRaisesRegex(EpisodeError, "past the end"):
            episodes.target_state(walk, 3.0)
        with self.assertRaisesRegex(EpisodeError, "constant_velocity or waypoints"):
            episodes.target_state({"kind": "orbit"}, 0.0)


class SetTests(Fixture):
    def config(self, **changes):
        config = json.loads((EPISODES / "smoke_set.json").read_text())
        config["grant"] = str(GRANTS / "two_prb_offset.json")
        config.update(changes)
        path = self.root / f"set-{len(list(self.root.glob('set-*.json')))}.json"
        path.write_text(json.dumps(config))
        return path

    def build_set(self, name="set"):
        """Write the smoke set with the analytic model in place of the ray tracer."""
        path = self.config(target={"model": "constant_rcs_cuboid", "center_height_m": 0.875, "sigma_m2": 1.0,
                                   "size_m": [0.3, 0.3, 0.3]})
        config, base = sets.load_set_config(path)
        plans = sets.plan(config, base)
        directory, manifests = self.root / name, {}
        (directory / "episodes").mkdir(parents=True)
        for item in plans:
            target = dict(config["target"], motion=item["motion"])
            absent, present = model_pair(item["schedule"], target)
            truth = truth_of(item["schedule"], target)
            sets.label(config, item, truth)
            manifests[item["episode_id"]] = episodes.export(
                directory / "episodes" / item["episode_id"], item["episode_id"], item["schedule"],
                description(base, labels=item["labels"], target=target),
                {"a": absent, "b": present, "w": [0j] * len(absent)}, truth)
        sets.write_manifest(directory, config, hashlib.sha256(path.read_bytes()).hexdigest(), plans, manifests)
        return directory

    def test_shipped_configurations_are_valid(self):
        config, base, schedule = sets.load_pair_config(EPISODES / "reference_pair.json")
        self.assertEqual((base.grant_id, len(schedule.opportunities), config["analytic_model"]),
                         ("demo-51prb-4rx", 24, episodes.ANALYTIC_MODEL))
        for name, wanted in (("smoke_set.json", {"training": 3, "validation": 1, "held_out": 1}),
                             ("demo_set.json", {"training": 300, "validation": 40, "held_out": 10})):
            config, base = sets.load_set_config(EPISODES / name)
            plans = sets.plan(config, base)
            self.assertEqual({group: sum(item["group"] == group for item in plans) for group in sets.GROUPS}, wanted)
            self.assertEqual(config["target"]["model"], "tr38901_human")
            for item in plans:  # the whole episode stays on its trajectory and inside the area
                last = item["schedule"].duration_tc() / TC_PER_SECOND
                position, _ = episodes.target_state(item["motion"], last)
                sets.cell(config, position[0], position[1])
        # The area of interest has the size of the cuSense campaign.
        self.assertAlmostEqual(config["area"]["x_m"][1] - config["area"]["x_m"][0], 6.78)
        self.assertAlmostEqual(config["area"]["y_m"][1] - config["area"]["y_m"][0], 10.06)

    def test_groups_come_from_trajectories_and_identifiers_are_opaque(self):
        config, base = sets.load_set_config(EPISODES / "smoke_set.json")
        plans = sets.plan(config, base)
        by_trajectory = {}
        for item in plans:
            by_trajectory.setdefault(item["trajectory"], set()).add(item["group"])
            self.assertRegex(item["episode_id"], r"^ep-[0-9a-f]{12}$")
            for word in ("train", "valid", "held", "present", "absent", item["trajectory"]):
                self.assertNotIn(word, item["episode_id"])
        self.assertTrue(all(len(groups) == 1 for groups in by_trajectory.values()))
        self.assertEqual([item["episode_id"] for item in plans], [item["episode_id"] for item in sets.plan(config, base)])
        self.assertEqual(len({item["noise"]["seed"] for item in plans}), len(plans))
        first = sets.assignment_sha256(config["trajectories"])
        config["trajectories"][0]["group"] = "validation"
        self.assertNotEqual(first, sets.assignment_sha256(config["trajectories"]))

    def test_grid_cells_and_trajectory_shapes(self):
        config, _ = sets.load_set_config(EPISODES / "smoke_set.json")
        self.assertEqual(sets.cell(config, 0.61, 1.47), [0, 0])
        self.assertEqual(sets.cell(config, 7.39, 11.53), [49, 33])  # the upper edge belongs to the last cell
        self.assertEqual(sets.cell(config, 4.0, 6.5), [25, 17])
        with self.assertRaisesRegex(EpisodeError, "outside the area"):
            sets.cell(config, 0.5, 5.0)
        record = sets.grid_record(config)
        self.assertEqual((record["rows"], record["cols"]), (50, 34))
        self.assertAlmostEqual(record["cell_m"][0], 0.2012)
        area = config["area"]
        lanes = sets.polyline({"id": "t", "kind": "lawnmower", "axis": "y", "lanes": 3, "margin_m": 0.5}, area, 0.9)
        self.assertEqual([round(x, 3) for x, _, _ in lanes], [1.11, 1.11, 4.0, 4.0, 6.89, 6.89])
        self.assertEqual([round(y, 3) for _, y, _ in lanes], [1.97, 11.03, 11.03, 1.97, 1.97, 11.03])
        spiral = sets.polyline({"id": "t", "kind": "spiral", "turns": 2, "margin_m": 0.5}, area, 0.9)
        self.assertEqual(len(spiral), 10)
        first = sets.polyline({"id": "t", "kind": "random", "seed": 11, "waypoints": 5, "margin_m": 0.3}, area, 0.9)
        second = sets.polyline({"id": "t", "kind": "random", "seed": 12, "waypoints": 5, "margin_m": 0.3}, area, 0.9)
        self.assertNotEqual(first, second)
        self.assertEqual(first, sets.polyline({"id": "t", "kind": "random", "seed": 11, "waypoints": 5,
                                               "margin_m": 0.3}, area, 0.9))
        self.assertTrue(all(z == 0.9 for _, _, z in first + spiral + lanes))

    def test_configuration_refusals(self):
        good = json.loads((EPISODES / "smoke_set.json").read_text())

        def edited(change):
            config = json.loads(json.dumps(good))
            change(config)
            return config

        cases = (
            (lambda c: c["trajectories"].append(dict(c["trajectories"][0], group="held_out")), "its own ASCII id"),
            (lambda c: c["trajectories"].pop(1), "no trajectory is assigned to the group validation"),
            (lambda c: c["trajectories"][0].update(group="test"), "group must be one of"),
            (lambda c: c["trajectories"][1].update(to_m=[9.0, 9.0]), "leaves the area of interest"),
            (lambda c: c["trajectories"][0].update(episodes=40), "runs past its end"),
            (lambda c: c["area"].update(x_m=[0.61, 8.5]), "leaves the room"),
            (lambda c: c["solver"]["background"].update(diffraction=True), "solver.background requires exactly"),
            (lambda c: c["scene"]["materials"].update(walls="paper"), "room material"),
            (lambda c: c["endpoints"].update(ue_tx_m=[5.2, 14.0, 1.0]), "outside the room"),
            (lambda c: c["position_grid"].update(rows=0), "position_grid.rows"),
            (lambda c: c.update(status="final"), "proposed or frozen"),
            (lambda c: c["schedule"]["tdd"].update(ul_slots=[5]), "schedule of the configuration is invalid"),
            (lambda c: c.update(grant="missing.json"), "grant of the configuration is invalid"),
        )
        for change, message in cases:
            path = self.root / "edited.json"
            config = edited(change)
            if config["grant"].startswith("../"):
                config["grant"] = str(GRANTS / "demo_51prb_4rx.json")
            path.write_text(json.dumps(config))
            with self.subTest(message=message), self.assertRaisesRegex(EpisodeError, message):
                loaded, base = sets.load_set_config(path)
                sets.plan(loaded, base)
        pair = json.loads((EPISODES / "reference_pair.json").read_text())
        pair["grant"] = str(GRANTS / "demo_51prb_4rx.json")
        for change, message in ((lambda c: c["array"].update(pattern="tr38901"), "needs free space"),
                                (lambda c: c["target"]["motion"].update(kind="waypoints"), "target motion|waypoint"),
                                (lambda c: c["target"].update(sigma_m2=0), "sigma_m2"),
                                (lambda c: c["schedule"].pop("periods"), "number of periods"),
                                (lambda c: c.update(schema="episode-set-config/v1"), "not a episode-pair-config/v1")):
            config = json.loads(json.dumps(pair))
            change(config)
            (self.root / "pair.json").write_text(json.dumps(config))
            with self.subTest(message=message), self.assertRaisesRegex(EpisodeError, message):
                sets.load_pair_config(self.root / "pair.json")

    def test_a_set_reads_back_and_refuses_a_changed_assignment(self):
        directory = self.build_set()
        stored = sets.open_set(directory)
        manifest = stored.manifest
        self.assertEqual(manifest["counts"]["by_group"], {"training": 3, "validation": 1, "held_out": 1})
        self.assertEqual(manifest["counts"]["target_present_runs"], manifest["counts"]["target_absent_runs"])
        self.assertEqual(len(stored.episodes("training")), 3)
        self.assertEqual(len(stored.episodes()), 5)
        training = stored.episodes("training")[0]
        truth = training.truth(0)
        self.assertEqual(truth["cell"], sets.cell(manifest["config"], truth["position_m"][0], truth["position_m"][1]))
        self.assertEqual(training.manifest["labels"]["group"], "training")
        self.assertEqual(manifest["coverage"]["by_group"]["training"], 3)
        self.assertIn("case a", manifest["calibration_episodes"])

        def rewrite(copy, change, relative="manifest.json"):
            manifest = json.loads((copy / relative).read_text())
            change(manifest)
            (copy / relative).write_text(json.dumps(manifest))

        first = manifest["episodes"][0]["directory"]
        cases = (
            (lambda copy: rewrite(copy, lambda m: m["episodes"][0].update(group="held_out")), "not in the group"),
            (lambda copy: rewrite(copy, lambda m: m["groups"]["held_out"].append(m["groups"]["training"][0])),
             "not a partition"),
            (lambda copy: rewrite(copy, lambda m: m["config"]["trajectories"][0].update(group="validation")),
             "not a partition"),
            (lambda copy: rewrite(copy, lambda m: m.update(assignment_sha256="0" * 64)), "assignment changed"),
            (lambda copy: rewrite(copy, lambda m: m["labels"].update(group="held_out"), first + "/manifest.json"),
             "differs from the set manifest"),
            (lambda copy: rewrite(copy, lambda m: m["counts"].update(target_absent_runs=4)), "counts"),
            (lambda copy: (copy / first / "w.npy").unlink(), "cannot read w.npy"),
            (lambda copy: (copy / "manifest.json").unlink(), "not a complete episode set"),
        )
        for number, (damage, message) in enumerate(cases):
            copy = self.root / f"set-copy-{number}"
            shutil.copytree(directory, copy)
            damage(copy)
            with self.subTest(message=message), self.assertRaisesRegex(EpisodeError, message):
                sets.open_set(copy)


class CommandTests(Fixture):
    def run_python(self, *arguments):
        return subprocess.run([sys.executable, *arguments], capture_output=True, text=True, timeout=120, check=False,
                              env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})

    def test_a_second_reader_recovers_a_coefficient_from_the_command_line(self):
        directory, _ = self.stored()
        result = self.run_python("-m", "ocudu_cuda_isac.episodes.episode", "show", str(directory), "--case", "a",
                                 "--opportunity", "3", "--rx", "1", "--symbol", "11", "--subcarrier", "203")
        self.assertEqual(result.returncode, 0, result.stderr)
        shown = json.loads(result.stdout)
        self.assertEqual(shown["coefficient"]["value"], [3112.0, 204.0])
        self.assertEqual(shown["coefficient"]["time_tc"], 17 * 983040 + 777216)
        self.assertEqual(shown["verified_files"], ["a_h.npy", "b_h.npy", "w.npy"])
        self.assertEqual(shown["sha256"]["b_h.npy"], hashlib.sha256((directory / "b_h.npy").read_bytes()).hexdigest())
        (directory / "w.npy").write_bytes(b"")
        refused = self.run_python("-m", "ocudu_cuda_isac.episodes.episode", "show", str(directory))
        self.assertEqual(refused.returncode, 2)
        self.assertIn("episode: w.npy differs", refused.stderr)

    def test_the_record_modules_need_no_array_or_simulation_package(self):
        code = ("import sys\n"
                "import ocudu_cuda_isac.episodes.episode, ocudu_cuda_isac.episodes.episode_set\n"
                "import ocudu_cuda_isac.episodes.schedule, ocudu_cuda_isac.episodes.rt_channel\n"
                "found = sorted(set(sys.modules) & {'numpy', 'torch', 'sionna', 'mitsuba', 'drjit'})\n"
                "sys.exit('imported: ' + ', '.join(found) if found else 0)\n")
        result = self.run_python("-c", code)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_the_exporter_takes_one_mode(self):
        script = str(ROOT / "scripts/export_rt_episode.py")
        result = self.run_python(script, "--output-dir", str(self.root / "x"), "--pair", "a.json", "--set", "b.json")
        self.assertEqual(result.returncode, 2)
        self.assertIn("not allowed with argument", result.stderr)
        self.assertEqual(self.run_python(script, "--help").returncode, 0)
        self.assertFalse((self.root / "x").exists())


if __name__ == "__main__":
    unittest.main()
