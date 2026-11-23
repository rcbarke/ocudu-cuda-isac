#!/usr/bin/env python3
"""Sprint 3.1 unit gate (selector pilot_ls): python tests/unit/test_pilot_ls.py -v

Three parts. The reference part needs the standard library only: the estimate of the card's
vectors, the layout, the validity bits, the antenna map, the time of an observation and its
record. The plan part runs the tapped replay on a small episode entry with a double of the native
tool: the plan and every rule that refuses one, the reader of the native output, the reference
recomputed from the stored grids, and the radio records with the tap on against the control. The
native part builds native/ocudu_adapter/observer on its own, with a C++17 compiler and CMake and
no OCUDU source or build, runs its CTest cases and compares the native estimate with the
reference bit for bit. No Sionna, no NumPy, no OCUDU build and no GPU.
"""

import cmath
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import random
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
from ocudu_cuda_isac.episodes import grant as grants  # noqa: E402
from ocudu_cuda_isac.episodes import grid_apply as replay  # noqa: E402
from ocudu_cuda_isac.episodes import pilot_ls as pl  # noqa: E402
from ocudu_cuda_isac.episodes import replay_pusch as rp  # noqa: E402
from ocudu_cuda_isac.episodes.pilot_ls import PilotLsPlanError  # noqa: E402
from ocudu_cuda_isac.sensing import pilot_ls as reference  # noqa: E402
from ocudu_cuda_isac.sensing import sources  # noqa: E402
from ocudu_cuda_isac.sensing.pilot_ls import PilotLsError  # noqa: E402
import pilot_ls_double as double  # noqa: E402
import replay_pusch_double as receiver_double  # noqa: E402
import test_grid_apply as apply_tests  # noqa: E402

OBSERVER = ROOT / "native/ocudu_adapter/observer"
CASES = ("kernel", "amplitude_scale", "data_invariance", "pilot_hole", "refusals", "antenna_map", "coordinates",
         "record", "ownership", "timed", "binding", "slot_times")
# The reference amplitude of two CDM groups without data, per component, in single precision.
COMPONENT = reference.single(10 ** (3 / 20) / 2 ** 0.5)
UNITY = reference.single(1.0 / COMPONENT)
# The native sources that Sprint 2 accepted. This ticket adds a tool beside them and edits none:
# replay_pusch stays the control, with the executable the accepted runs name. Sprint 3.2 amended
# the two host-free cores of that list, replay_core and dispatch_core, by the requirement of
# 2026-11-21 (the hash counts its own work, a grid can be owned by the ticket of its fill). Their
# new hashes are pinned in tests/unit/test_snapshot_ring.py, and the seven files below are as
# Sprint 2 accepted them.
ACCEPTED_SOURCES = {
    "fixtures/grant_fixture.cpp": "48e36372e08973b9331446292f723941bf8a64864419360266d575623a6537d8",
    "replay/replay_plan.h": "6b6ead72396b9ba5d1639b055eee395966355f7e3f53890064b07f3c20b8a981",
    "replay/pinned_grid.h": "465d265f46ca40c9613e77c0e3c9eec192a2f16b837a39c601e74bc65814ecc0",
    "replay/grid_apply.cpp": "62d76c8e3b105bb92e9874981889508a22ebd37bedd932ee26abcab300754f17",
    "receiver/grant_pdu.h": "f75d4ac1b0fe5ff17ba225b1b8b487dbf70bb6064482ce9c9b74d6f5f1f181aa",
    "receiver/receiver_plan.h": "b49f5a90f1070cf2a8966d45828817e9a265abf4eef3e72fe9ff6319ffdd0fca",
    "receiver/replay_pusch.cpp": "d6719fe8081415587a819f0c6b2c83b27e3933053ba68bb8a48303168dc86ec2",
}


def small_record(index=None, values=None, ports=(0,)):
    """A source on symbols 2 and 11 with a comb of step two from subcarrier 4: eight pilots of (+-0.5, +-0.5)."""
    index = [[symbol, k] for symbol in (2, 11) for k in (4, 6, 8, 10)] if index is None else index
    if values is None:
        values = [[[-0.5 if n % 2 else 0.5, 0.5 if n % 3 else -0.5] for n in range(len(index))] for _ in ports]
    symbols = sorted({symbol for symbol, _ in index})
    starts = {2: 145920, 11: 777216}
    return {"schema": sources.SCHEMA, "source": "pusch_dmrs", "channel": "PUSCH", "link_direction": "uplink",
            "reference_signal": "unit comb", "ports": list(ports), "processor": {"name": "pusch", "backend": None},
            "slot": {"scs_khz": 30, "sfn": 0, "slot": 7, "symbols": symbols, "tc_per_second": reference.TC_PER_SECOND,
                     "symbol_useful_start_tc": [starts[symbol] for symbol in symbols]},
            "grid": {"coordinates": "carrier", "symbols": 14, "subcarriers": 24},
            "mask": {"axes": ["pilot", ["symbol", "subcarrier"]], "index": index},
            "reference": {"axes": ["port", "pilot", ["real", "imaginary"]], "amplitude": 0.5 * 2 ** 0.5, "values": values}}


def grid_of(source, channel, data=1.0, scale=1.0):
    """A grid that carries channel(plane, symbol, subcarrier) on the pilots and a data value everywhere else."""
    pilots = {coordinate: source.reference[number] for number, coordinate in enumerate(source.index)}

    def grid(plane, symbol, subcarrier):
        if (symbol, subcarrier) in pilots:
            return channel(plane, symbol, subcarrier) * pilots[(symbol, subcarrier)] * scale
        return complex(data + 0.125 * subcarrier, data - 0.25 * symbol)
    return grid


class KernelTests(unittest.TestCase):
    """The estimate of one resource element and of one source: the vectors of the sprint card."""

    def test_a_pilot_of_one_plus_j_and_y_of_minus_one_plus_three_j_give_one_plus_two_j(self):
        self.assertEqual(reference.ls(-1 + 3j, 1 + 1j), (1 + 2j, True))
        for channel in (1 + 0j, -1j, -0.75 + 0.5j, 0j):
            for pilot in (0.5 + 0.5j, -0.5 + 0.5j, 0.5 - 0.5j):
                self.assertEqual(reference.ls(channel * pilot, pilot), (channel, True))

    def test_the_stored_pilot_against_the_reference_reads_a_unity_channel_as_1_00119(self):
        value, ok = reference.ls(complex(-1.0, 1.0), complex(-COMPONENT, COMPONENT))
        self.assertTrue(ok)
        self.assertEqual((value.real, value.imag), (UNITY, 0.0))
        self.assertAlmostEqual(UNITY, 1.00119, places=5)
        self.assertAlmostEqual(COMPONENT, 0.998815, places=6)

    def test_a_halved_pilot_amplitude_with_a_scaled_y_gives_the_same_estimate(self):
        full = reference.Source(small_record(), [0, 1])
        record = small_record()
        record["reference"]["values"] = [[[0.5 * part for part in pair] for pair in row]
                                         for row in record["reference"]["values"]]
        record["reference"]["amplitude"] *= 0.5
        half = reference.Source(record, [0, 1])

        def channel(plane, symbol, k):
            return complex(0.25 * k - 1.0, 0.5 * plane + 0.125 * symbol)
        one = reference.estimate(full, grid_of(full, channel))
        two = reference.estimate(half, grid_of(half, channel))
        self.assertEqual(one.payload(), two.payload())
        self.assertEqual(half.amplitude, 0.5 * full.amplitude)
        for plane in range(2):
            for row in range(2):
                for column in range(4):
                    self.assertEqual(one.values[full.element(plane, row, column)],
                                     channel(plane, *full.coordinate(row, column)))

    def test_altering_data_only_elements_leaves_the_estimate_as_it_was(self):
        source = reference.Source(small_record(), [0, 1])

        def channel(plane, symbol, k):
            return complex(1.0 - 0.5 * plane, 0.125 * k)
        one = reference.estimate(source, grid_of(source, channel, data=1.0))
        two = reference.estimate(source, grid_of(source, channel, data=-2.5))
        self.assertEqual((one.payload(), one.mask), (two.payload(), two.mask))
        other = reference.estimate(source, grid_of(source, lambda *_: 0.5 + 0j, data=1.0))
        self.assertNotEqual(one.payload_sha256, other.payload_sha256)

    def test_a_pilot_hole_clears_its_bit_and_a_measured_zero_sets_it(self):
        index = [[symbol, k] for symbol in (2, 11) for k in (4, 6, 8, 10) if (symbol, k) != (11, 8)]
        source = reference.Source(small_record(index), [0, 1])
        self.assertEqual((source.holes, source.cells[1]), (1, [4, 5, None, 6]))

        def channel(plane, symbol, k):
            return 0j if (plane, symbol, k) == (1, 2, 6) else 0.5 - 0.25j
        made = reference.estimate(source, grid_of(source, channel, data=2.0))
        self.assertEqual((len(made.values), made.valid, made.holes), (16, 14, 2))
        for plane in range(2):
            hole = source.element(plane, 1, 2)
            self.assertFalse(made.is_valid(hole))
            self.assertEqual(made.values[hole], 0j)
        zero = source.element(1, 0, 1)
        self.assertTrue(made.is_valid(zero))
        self.assertEqual(made.values[zero], 0j)
        # Bit n is bit (n mod 8) of byte (n // 8): the holes are elements 6 and 14.
        self.assertEqual(made.mask, bytes([0xbf, 0xbf]))
        self.assertEqual(made.payload_sha256, hashlib.sha256(made.payload()).hexdigest())

    def test_a_configured_pilot_with_zero_reference_power_is_rejected(self):
        record = small_record()
        record["reference"]["values"][0][3] = [0.0, 0.0]
        with self.assertRaisesRegex(PilotLsError, "zero reference power"):
            reference.Source(record, [0, 1])
        with self.assertRaisesRegex(PilotLsError, "zero reference power"):
            reference.ls(1 + 1j, 0j)

    def test_every_other_source_rule(self):
        def edited(change):
            record = small_record()
            change(record)
            return record
        two_ports = small_record(ports=(0, 1))
        rules = [
            (two_ports, [0, 1], "estimates one port"),
            (edited(lambda r: r["mask"]["index"].__setitem__(1, r["mask"]["index"][0])), [0, 1], "duplicate pilot"),
            (edited(lambda r: r["mask"]["index"].reverse()), [0, 1], "ordered by symbol, then by subcarrier"),
            (edited(lambda r: r["mask"]["index"][7].__setitem__(1, 24)), [0, 1], "pilot subcarrier"),
            (edited(lambda r: r["reference"]["values"][0].pop()), [0, 1], "one value per pilot"),
            (edited(lambda r: r["reference"]["values"][0][0].__setitem__(0, float("nan"))), [0, 1], "finite"),
            (edited(lambda r: r["slot"].pop("symbol_useful_start_tc")), [0, 1], "start time"),
            (edited(lambda r: r["slot"].update(scs_khz=20)), [0, 1], "subcarrier spacing"),
            (small_record(), [1, 1], "names an array element twice"),
            (small_record(), [], "antenna map"),
            (small_record(), [0, 4096], "antenna map"),
        ]
        for record, antenna_map, words in rules:
            with self.assertRaisesRegex(PilotLsError, words):
                reference.Source(record, antenna_map)

    def test_two_antenna_phases_and_a_deliberately_reversed_map(self):
        def channel(plane, symbol, k):
            return 1j if plane == 0 else -1 + 0j
        straight = reference.Source(small_record(), [0, 1])
        reversed_map = reference.Source(small_record(), [1, 0])
        one = reference.estimate(straight, grid_of(straight, channel))
        two = reference.estimate(reversed_map, grid_of(reversed_map, channel))
        # The producer never reorders: both payloads hold the planes in the order of the grid.
        self.assertEqual(one.payload(), two.payload())
        self.assertEqual((one.values[0], one.values[8]), (1j, -1 + 0j))
        # The map travels with the data: the phase of array element 0 is read from the plane it names.
        self.assertEqual((straight.plane_of(0), straight.plane_of(1)), (0, 1))
        self.assertEqual((reversed_map.plane_of(0), reversed_map.plane_of(1)), (1, 0))
        self.assertEqual(one.values[straight.element(straight.plane_of(0), 0, 0)], 1j)
        self.assertEqual(two.values[reversed_map.element(reversed_map.plane_of(0), 0, 0)], -1 + 0j)
        with self.assertRaisesRegex(PilotLsError, "names no receive plane"):
            straight.plane_of(2)

    def test_sparse_coordinates_of_the_payload(self):
        source = reference.Source(small_record(), [0, 1])
        self.assertEqual((source.symbols, source.subcarriers, source.subcarrier_step), ([2, 11], [4, 6, 8, 10], 2))
        self.assertEqual((source.shape, source.strides), ([2, 2, 4], [8, 4, 1]))
        self.assertEqual(source.coordinates_sha256, hashlib.sha256(struct.pack("<6I", 2, 11, 4, 6, 8, 10)).hexdigest())
        made = reference.estimate(source, grid_of(source, lambda plane, symbol, k: complex(symbol + 0.125 * plane, k)))
        for plane in range(2):
            for row in range(2):
                for column in range(4):
                    symbol, k = source.coordinate(row, column)
                    self.assertEqual(made.values[source.element(plane, row, column)], complex(symbol + 0.125 * plane, k))
        # The demonstration grant: 2 symbols of 306 pilots on four receive planes, 19,584 bytes.
        grant = grants.load(ROOT / "configs/grants/demo_51prb_4rx.json")
        demo = reference.Source(grant.source, grant.rx_ports)
        self.assertEqual((demo.shape, demo.subcarriers[0], demo.subcarrier_step, demo.holes), ([4, 2, 306], 0, 2, 0))
        self.assertEqual(8 * 4 * 2 * 306, 19584)
        # The offset grant: pilots on subcarriers 180 to 202 of CRB 15 and 16.
        offset = grants.load(ROOT / "configs/grants/two_prb_offset.json")
        small = reference.Source(offset.source, offset.rx_ports)
        self.assertEqual((small.shape, small.subcarriers[0], small.subcarriers[-1], small.symbols),
                         ([1, 2, 12], 180, 202, [2, 11]))

    def test_the_transmit_grid_of_a_grant_reads_as_unity_on_every_pilot(self):
        grant = grants.load(ROOT / "configs/grants/demo_51prb_4rx.json")
        source = reference.Source(grant.source, grant.rx_ports)
        stored = {coordinate: complex(grants.bf16(value.real), grants.bf16(value.imag))
                  for coordinate, value in zip(grant.pilot_index, grant.pilot_reference[0])}
        made = reference.estimate(source, lambda plane, symbol, k: stored[(symbol, k)])
        self.assertEqual((made.valid, made.holes), (2448, 0))
        self.assertEqual({(value.real, value.imag) for value in made.values}, {(UNITY, 0.0)})
        self.assertEqual(source.amplitude, 10 ** (3 / 20))

    def test_slot_times_and_the_slot_count_that_does_not_wrap(self):
        self.assertEqual([reference.slots_per_frame(scs) for scs in (15, 30, 60, 120)], [10, 20, 40, 80])
        self.assertEqual((reference.useful_tc(15), reference.useful_tc(30)), (131072, 65536))
        for scs in (15, 30, 60, 120):
            per = reference.slots_per_frame(scs) // 10
            self.assertEqual((reference.slot_start_tc(scs, 0), reference.slot_start_tc(scs, per)), (0, 1966080))
            self.assertEqual(reference.slot_start_tc(scs, reference.slots_per_frame(scs)), 19660800)
        self.assertEqual(reference.slot_start_tc(30, 7), 7 * 983040)
        self.assertEqual(reference.slot_start_tc(60, 1), 14 * (32768 + 2304) + 1024)
        # The slot start agrees with the symbol times of the source module.
        source = reference.Source(small_record(), [0, 1])
        self.assertEqual(reference.sample_times(source, 7), (7 * 983040 + 145920, 7 * 983040 + 777216 + 65536))
        clock = reference.SlotClock()
        self.assertEqual(clock.absolute_slot("a", 30, 1023, 7), 1023 * 20 + 7)
        self.assertEqual(clock.absolute_slot("a", 30, 0, 7), 1024 * 20 + 7)      # the frame number wrapped
        self.assertEqual(clock.absolute_slot("b", 30, 0, 7), 7)                  # another episode
        self.assertEqual(reference.opaque_key("abc"), "ba7816bf8f01cfea")
        with self.assertRaisesRegex(PilotLsError, "subcarrier spacing"):
            reference.slot_start_tc(45, 0)

    def test_the_record_identifies_an_observer_derived_estimate_and_holds_no_truth(self):
        source = reference.Source(small_record(), [0, 1])
        made = reference.estimate(source, grid_of(source, lambda *_: 1 + 0j))
        context = {"run_id": "unit", "epoch": 3, "sequence": 0, "opportunity_id": "op.0000", "opportunity_sequence": 0,
                   "episode_id": "rz-unit", "grant_id": "grant-a", "grant_record_sha256": "a" * 64, "cell": 1,
                   "ue": 4660, "sfn": 0, "slot": 7, "absolute_slot": 7,
                   "ingress": {"grid_index": 0, "generation": 1, "readback_sha256": "b" * 64},
                   "payload_file": "0000.o2.c64", "mask_file": "0000.o2.mask", "t_ready_ns": 1, "service_us": 2.0}
        record = reference.describe(source, made, context)
        self.assertEqual(reference.record_violations(record), [])
        self.assertEqual((record["what"]["stage"], record["what"]["producer"], record["what"]["producer_steps"],
                          record["what"]["source"], record["what"]["layout"], record["what"]["dtype"]),
                         ("O2", "observer", ["pilot_ls"], "pusch_dmrs", "ocudu_o2_comb_c64", "<c8"))
        self.assertEqual(record["what"]["scale"], source.amplitude)
        self.assertEqual((record["time"]["t_first_sample"], record["time"]["t_publish_ns"]), (7 * 983040 + 145920, None))
        self.assertEqual(record["payload"]["bytes"], 128)
        text = json.dumps(record)
        for word in ("truth", "label", "target", "tb_", "h.npy", "w.npy", "position"):
            self.assertNotIn(word, text)
        changes = [
            (lambda r: r.update(truth=1), "exactly the groups"),
            (lambda r: r["host"].update(tb_sha256="0" * 64), "host does not hold exactly"),
            (lambda r: r["what"].update(stage="O3"), "observer-derived O2"),
            (lambda r: r["what"].update(producer="pusch_processor"), "observer-derived O2"),
            (lambda r: r["what"].update(precision="float16"), "another precision"),
            (lambda r: r["shape"].update(strides=[4, 4, 1]), "contiguous"),
            (lambda r: r["allocation"].update(antenna_map=[0, 0]), "antenna map"),
            (lambda r: r["validity"].update(valid=15), "validity bits"),
            (lambda r: r["time"].update(t_publish_ns=5), "no publication"),
            (lambda r: r["payload"].update(bytes=64), "payload"),
            (lambda r: r["keys"].update(epoch=0), "join keys"),
            (lambda r: r["keys"].update(opportunity="0" * 16), "join key is not the key"),
        ]
        for change, words in changes:
            damaged = copy.deepcopy(record)
            change(damaged)
            self.assertTrue(any(words in problem for problem in reference.record_violations(damaged)), words)

    def test_the_sensing_module_imports_no_host_no_simulator_and_no_episode(self):
        code = ("import sys; import ocudu_cuda_isac.sensing.pilot_ls; "
                "print(sorted(name for name in sys.modules if name.startswith('ocudu_cuda_isac')))")
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                                env=dict(os.environ, PYTHONPATH=str(ROOT / "src")))
        loaded = json.loads(result.stdout.replace("'", '"'))
        self.assertEqual(loaded, ["ocudu_cuda_isac", "ocudu_cuda_isac.sensing", "ocudu_cuda_isac.sensing.pilot_ls",
                                  "ocudu_cuda_isac.sensing.sources", "ocudu_cuda_isac.sensing.sources.pusch_dmrs"])


class StimulusTests(unittest.TestCase):
    """The plan on a small entry, the double of the native tool, the reader and the joins."""

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.entry = apply_tests.make_stimulus(cls.root)
        cls.channel = replay.entry_plan("unit-run", [cls.entry])
        replay.write_plan(cls.channel, cls.root / "channel-plan.json")
        cls.replay_plan = rp.build_plan("unit-run", cls.root / "channel-plan.json")
        rp.write_plan(cls.replay_plan, cls.root / "plan.json")
        # The control: the double of the accepted tool of Sprint 2.4 on the replay plan.
        (cls.root / "control").mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            assert receiver_double.main(["--plan", str(cls.root / "plan.json"), "--output-dir",
                                         str(cls.root / "control")]) == 0
        cls.control = rp.open_output(cls.root / "control")
        cls.plan = pl.build_plan("unit-tap", cls.root / "plan.json")
        pl.write_plan(cls.plan, cls.root / "tap.json")
        cls.output = cls.root / "out"
        cls.status = cls.tool(cls.root / "tap.json", cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @staticmethod
    def tool(plan, output, fault="", extra=()):
        output.mkdir()
        os.environ["PILOT_LS_FAULT"] = fault
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                return double.main(["--plan", str(plan), "--output-dir", str(output), *extra])
        finally:
            os.environ.pop("PILOT_LS_FAULT")

    def scratch(self):
        directory = tempfile.TemporaryDirectory(dir=self.root)
        self.addCleanup(directory.cleanup)
        return Path(directory.name)

    def test_a_plan_binds_the_replay_plan_and_one_source_per_grant(self):
        plan = self.plan
        self.assertEqual((plan["schema"], plan["access"], plan["run_id"]),
                         ("pilot-ls-plan/v1", "offline_stimulus_and_evaluator", "unit-tap"))
        self.assertEqual(plan["replay_plan"], replay.file_ref(self.root / "plan.json"))
        self.assertEqual(plan["observer"], {"tap": "pilot_ls", "stage": "O2", "source": "pusch_dmrs",
                                            "layout": "ocudu_o2_comb_c64", "epoch": 1})
        self.assertEqual([source["grant_id"] for source in plan["sources"]],
                         [grant["grant_id"] for grant in self.replay_plan["grants"]])
        for source, grant in zip(plan["sources"], self.replay_plan["grants"]):
            store = Path(grant["record"]["path"]).parent
            self.assertEqual(source["record_sha256"], grant["record_sha256"])
            self.assertEqual(source["source"], replay.file_ref(store / "source.json"))
            self.assertEqual(source["pilot_index"], replay.file_ref(store / "pilot_index.npy"))
            self.assertEqual(source["pilot_ref"], replay.file_ref(store / "pilot_ref.npy"))
            self.assertEqual(source["antenna_map"], [0, 1])
        replay_plan, channel, compiled = pl.check_files(plan)
        self.assertEqual((replay_plan["receiver"]["tap"], len(compiled)), ("none", len(channel["grants"])))
        self.assertEqual(pl.load_plan(self.root / "tap.json"), plan)
        with self.assertRaisesRegex(PilotLsPlanError, "never overwritten"):
            pl.write_plan(plan, self.root / "tap.json")
        # The control plan of this tool names no source.
        control = pl.build_plan("unit-off", self.root / "plan.json", "none")
        self.assertEqual((control["observer"], control["sources"]),
                         ({"tap": "none", "stage": None, "source": None, "layout": None, "epoch": 1}, []))

    def test_every_plan_rule_of_the_ticket(self):
        scratch = self.scratch()
        first = self.plan["sources"][0]

        def edited_array(key, change):
            def apply(plan):
                _, shape, flat = grants.read_npy(first[key]["path"])
                change(flat)
                descr = "<i4" if key == "pilot_index" else "<c8"
                path = scratch / f"{key}-{len(list(scratch.iterdir()))}.npy"
                path.write_bytes(grants._npy_bytes(descr, shape, flat))
                record = json.loads(Path(first["source"]["path"]).read_text())
                name = "index" if key == "pilot_index" else "values"
                group = "mask" if key == "pilot_index" else "reference"
                record[group][name]["sha256"] = replay.sha256_file(path)
                source = path.with_suffix(".json")
                source.write_text(json.dumps(record))
                plan["sources"][0][key] = replay.file_ref(path)
                plan["sources"][0]["source"] = replay.file_ref(source)
            return apply

        def tapped_replay(plan):
            other = copy.deepcopy(self.replay_plan)
            other["receiver"]["tap"] = "o2"
            (scratch / "tapped-replay.json").write_text(json.dumps(other))
            plan["replay_plan"] = replay.file_ref(scratch / "tapped-replay.json")

        rules = [
            (lambda plan: plan["observer"].update(tap="o3"), "unknown tap"),
            (lambda plan: plan["observer"].update(stage="O3"), "exports the stage O2"),
            (lambda plan: plan["observer"].update(layout="ocudu_o3_fp16"), "exports the stage O2"),
            (lambda plan: plan["observer"].update(epoch=0), "epoch is at least 1"),
            (lambda plan: plan["observer"].update(source="srs"), "the plan taps srs"),
            (lambda plan: plan.update(access="receiver"), "access must be"),
            (lambda plan: plan.update(truth={}), "requires exactly"),
            (lambda plan: plan["replay_plan"].update(sha256="0" * 64), "replay plan does not have the hash"),
            (lambda plan: plan["replay_plan"].update(sha256=None), "bound by no hash"),
            (tapped_replay, "the replay plan is refused"),
            (lambda plan: plan["sources"].pop(1), "binds no sensing source for the grant"),
            (lambda plan: plan["sources"].append(copy.deepcopy(plan["sources"][0])), "listed twice"),
            (lambda plan: plan["sources"][0].update(record_sha256="0" * 64), "changed allocation"),
            (lambda plan: plan["sources"][0]["source"].update(sha256="0" * 64), "record binds"),
            (lambda plan: plan["sources"][0]["pilot_ref"].update(sha256="0" * 64), "record binds"),
            (lambda plan: plan["sources"][0].update(pilot_ref=plan["sources"][1]["pilot_ref"]),
             "names other arrays than the plan binds"),
            (lambda plan: plan["sources"][0].update(antenna_map=[0]), "receive planes"),
            (lambda plan: plan["sources"][0].update(antenna_map=[0, 0]), "names an array element twice"),
            (lambda plan: plan["sources"][0].update(antenna_map=[]), "antenna map"),
            (edited_array("pilot_ref", lambda flat: flat.__setitem__(3, 0j)), "zero reference power"),
            (edited_array("pilot_index", lambda flat: flat.__setitem__(3, flat[1])), "duplicate pilot"),
            (edited_array("pilot_index", lambda flat: flat.__setitem__(1, 0)), "outside the allocation"),
            (edited_array("pilot_index", lambda flat: flat.__setitem__(slice(0, 24, 2), [3] * 12)), "reference symbols"),
        ]
        for number, (change, words) in enumerate(rules):
            plan = copy.deepcopy(self.plan)
            change(plan)
            with self.assertRaisesRegex(PilotLsPlanError, words, msg=f"rule {number}: {words}"):
                pl.check_files(plan)
        # A run without a tap names nothing of the observer.
        off = pl.build_plan("unit-off", self.root / "plan.json", "none")
        for change in (lambda plan: plan["observer"].update(stage="O2"),
                       lambda plan: plan["sources"].append(copy.deepcopy(first))):
            plan = copy.deepcopy(off)
            change(plan)
            with self.assertRaisesRegex(PilotLsPlanError, "without a tap"):
                pl.check_plan(plan)

    def test_the_double_taps_every_grid_and_the_reader_verifies_it(self):
        self.assertEqual(self.status, 0)
        output = pl.open_output(self.output)
        self.assertEqual((output.run_id, output.tap, len(output.observations), len(output.records)),
                         ("unit-tap", "pilot_ls", 6, 12))
        self.assertEqual(output.report["schema"], "pilot-ls-native/v1")
        self.assertEqual(sorted(path.name for path in (self.output / "observer").iterdir()),
                         sorted(f"{n:04d}.o2.{kind}" for n in range(6) for kind in ("c64", "mask")) + ["o2.jsonl"])
        for position, record in enumerate(output.observations):
            self.assertEqual(reference.record_violations(record), [])
            self.assertEqual((record["keys"]["sequence"], record["keys"]["epoch"], record["shape"]["shape"]),
                             (position, 1, [2, 2, 12]))
            self.assertEqual(record["validity"], dict(record["validity"], valid=48, holes=0, mask_bytes=6))
            # The observation names the digest of the grid of its opportunity, which is the control's.
            descriptor = output.descriptors[position]
            self.assertEqual(record["host"]["ingress"]["readback_sha256"], descriptor["grid"]["sha256"])
            self.assertEqual(descriptor["grid"]["sha256"], self.control.descriptors[position]["grid"]["sha256"])
        self.assertEqual(pl.observer_violations(self.output, ["tb.bin", "h.npy", "w.npy", "target"]), [])

    def test_the_reference_recomputes_every_observation_bit_for_bit(self):
        output = pl.open_output(self.output)
        _, _, compiled = pl.check_files(self.plan)
        rows = pl.oracle(output, compiled)
        self.assertEqual(len(rows), 6)
        for row in rows:
            self.assertTrue(row["payload_equal"] and row["mask_equal"] and row["coordinates_equal"]
                            and row["identity_equal"], row)
        self.assertEqual(pl.expected_times(output, compiled), [])
        # Known H: the grid holds g H X + noise, so the estimate is near g H times the pilot ratio.
        gain = self.channel["gain"]["value"]
        record = output.observations[0]
        values = output.values(record)
        source = compiled[record["host"]["grant_record_sha256"]]
        for plane in range(2):
            for row in range(2):
                for column in range(12):
                    _, k = source.coordinate(row, column)
                    wanted = gain * 3.0e-4 * cmath.exp(1j * (0.4 * plane - 0.011 * k)) * UNITY
                    self.assertLess(abs(values[source.element(plane, row, column)] - wanted), 0.08 * abs(wanted))

    def test_the_tap_leaves_every_radio_record_of_the_control_as_it_was(self):
        output = pl.open_output(self.output)
        self.assertEqual(pl.radio_differences(output.records, self.control.records), [])
        self.assertEqual({record["tap"] for record in output.records}, {"pilot_ls"})
        self.assertEqual({record["tap"] for record in self.control.records}, {"none"})
        for record, control in zip(output.records, self.control.records):
            self.assertEqual(output.radio.decoded(record), self.control.decoded(control))
        # The same executable with the tap off: no observation, the same records.
        scratch = self.scratch()
        pl.write_plan(pl.build_plan("unit-off", self.root / "plan.json", "none"), scratch / "off.json")
        self.assertEqual(self.tool(scratch / "off.json", scratch / "off"), 0)
        off = pl.open_output(scratch / "off")
        self.assertEqual((off.tap, off.observations, (scratch / "off" / "observer").exists()), ("none", [], False))
        self.assertEqual(pl.radio_differences(off.records, self.control.records), [])
        self.assertEqual(pl.radio_differences(output.records, off.records), [])
        # A tap that changes a decode or a grid is found by the join.
        for fault, words in (("changed-decode", "decoded_sha256"), ("touch-grid", "ingress.readback_sha256")):
            target = self.scratch() / "out"
            self.assertEqual(self.tool(self.root / "tap.json", target, fault), 0)
            found = pl.radio_differences(pl.open_output(target).records, self.control.records)
            self.assertTrue(any(words in item for item in found), (fault, found[:2]))
        missing = pl.radio_differences(output.records[:-1], self.control.records)
        self.assertTrue(any("exists in one run only" in item for item in missing))

    def test_the_reader_refuses_a_damaged_output(self):
        def copied():
            target = self.scratch() / "out"
            shutil.copytree(self.output, target)
            return target

        def edit_record(position, change):
            def apply(target):
                path = target / "observer" / "o2.jsonl"
                records = [json.loads(line) for line in path.read_text().splitlines()]
                change(records[position])
                path.write_text("".join(json.dumps(record) + "\n" for record in records))
            return apply

        def flip_payload(target):
            path = target / "observer" / "0002.o2.c64"
            data = bytearray(path.read_bytes())
            data[5] ^= 1
            path.write_bytes(data)

        damages = [
            (flip_payload, "payload of observation record 2 differs"),
            (lambda target: (target / "observer" / "0001.o2.mask").write_bytes(bytes(6)), "validity bits"),
            (lambda target: (target / "observer" / "notes.txt").write_text("x"), "which no record names"),
            (lambda target: (target / "observer" / "0005.o2.c64").unlink(), "observation record 5"),
            (edit_record(1, lambda r: r["host"]["ingress"].update(readback_sha256="0" * 64)),
             "does not name the grid of its opportunity"),
            (edit_record(2, lambda r: r["keys"].update(sequence=7)), "out of sequence"),
            (edit_record(0, lambda r: r["what"].update(producer="pusch_processor")), "observer-derived O2"),
            (edit_record(3, lambda r: r["host"].update(tb_sha256="0" * 64)), "host does not hold exactly"),
            (edit_record(4, lambda r: r["payload"].update(file="0003.o2.c64")), "another record names"),
            (edit_record(4, lambda r: r["radio"].update(slot=r["radio"]["slot"] + 1)), "grid of its opportunity"),
        ]
        for damage, words in damages:
            target = copied()
            damage(target)
            with self.assertRaisesRegex(PilotLsPlanError, words):
                pl.open_output(target)
        # Faults of the tool that the reader finds on its own.
        for fault, words in (("drop-observation", "observation records for 6 filled"),
                             ("other-grid", "does not name the grid"), ("leak", "host does not hold exactly"),
                             ("tap-when-off", "holds an observation")):
            target = self.scratch() / "out"
            plan = self.root / "tap.json"
            if fault == "tap-when-off":
                plan = self.scratch() / "off.json"
                pl.write_plan(pl.build_plan("unit-off", self.root / "plan.json", "none"), plan)
            self.assertEqual(self.tool(plan, target, fault), 0)
            with self.assertRaisesRegex(PilotLsPlanError, words):
                pl.open_output(target)
        # A tapped output is not a control output of Sprint 2.4, and the reverse.
        with self.assertRaises(rp.ReplayPuschError):
            rp.open_output(self.output)
        with self.assertRaises(PilotLsPlanError):
            pl.open_output(self.root / "control")

    def test_faults_of_the_estimate_fail_the_reference(self):
        _, _, compiled = pl.check_files(self.plan)
        for fault in ("stored-pilot", "conjugate", "swap-planes", "shifted", "data-dependent"):
            target = self.scratch() / "out"
            self.assertEqual(self.tool(self.root / "tap.json", target, fault), 0)
            rows = pl.oracle(pl.open_output(target), compiled)
            self.assertFalse(any(row["payload_equal"] for row in rows), fault)
        target = self.scratch() / "out"
        self.assertEqual(self.tool(self.root / "tap.json", target, "mask-clear"), 0)
        self.assertFalse(any(row["mask_equal"] for row in pl.oracle(pl.open_output(target), compiled)))
        target = self.scratch() / "out"
        self.assertEqual(self.tool(self.root / "tap.json", target, "wrong-slot"), 0)
        self.assertEqual(len(pl.expected_times(pl.open_output(target), compiled)), 6)

    def test_the_double_refuses_what_the_native_tool_refuses(self):
        scratch = self.scratch()
        plan = copy.deepcopy(self.plan)
        plan["sources"][0]["record_sha256"] = "0" * 64
        (scratch / "plan.json").write_text(json.dumps(plan))
        self.assertEqual(self.tool(scratch / "plan.json", scratch / "out"), 2)
        self.assertEqual(list((scratch / "out").iterdir()), [])
        # An output directory that holds a result is refused.
        with contextlib.redirect_stderr(io.StringIO()) as said:
            self.assertEqual(double.main(["--plan", str(self.root / "tap.json"), "--output-dir", str(self.output)]), 2)
        self.assertIn("already holds a result", said.getvalue())

    def test_one_opportunity_is_repeated_exactly_and_grids_may_be_left_out(self):
        output = pl.open_output(self.output)
        chosen = output.observations[4]
        scratch = self.scratch()
        self.assertEqual(self.tool(self.root / "tap.json", scratch / "only",
                                   extra=("--only", chosen["host"]["opportunity_id"])), 0)
        again = pl.open_output(scratch / "only")
        self.assertEqual(len(again.observations), 1)
        record = again.observations[0]
        self.assertEqual((record["payload"]["digest"], record["host"]["ingress"]["readback_sha256"],
                          record["host"]["opportunity_sequence"], record["keys"]["sequence"]),
                         (chosen["payload"]["digest"], chosen["host"]["ingress"]["readback_sha256"], 4, 0))
        self.assertEqual(self.tool(self.root / "tap.json", scratch / "lean", extra=("--no-grids",)), 0)
        lean = pl.open_output(scratch / "lean")
        self.assertEqual([item["payload"]["digest"] for item in lean.observations],
                         [item["payload"]["digest"] for item in output.observations])
        self.assertIsNone(lean.grid(lean.observations[0]))

    def test_command_line_builds_a_plan_and_reads_an_output_back(self):
        scratch = self.scratch()
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            status = pl.main(["plan", "--replay-plan", str(self.root / "plan.json"), "--run-id", "by-hand",
                              "--output", str(scratch / "tap.json")])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(printed.getvalue())["observer"]["tap"], "pilot_ls")
        self.assertEqual(pl.load_plan(scratch / "tap.json")["sources"], self.plan["sources"])
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            status = pl.main(["show", str(self.output), "--sequence", "1", "--rx", "1", "--row", "1", "--pilot", "3"])
        self.assertEqual(status, 0)
        shown = json.loads(printed.getvalue())
        self.assertEqual((shown["verified_observations"], shown["stage"], shown["producer"], shown["source"]),
                         (6, "O2", "observer", "pusch_dmrs"))
        self.assertEqual((shown["element"]["symbol"], shown["element"]["subcarrier"], shown["element"]["valid"]),
                         (11, 186, True))
        with self.assertRaises(SystemExit) as stopped, contextlib.redirect_stderr(io.StringIO()):
            pl.main(["show", str(self.root / "control")])
        self.assertEqual(stopped.exception.code, 2)


class NativeCoreTests(unittest.TestCase):
    """The host-free observer core, built on its own: its CTest cases and its agreement with the reference."""

    @classmethod
    def setUpClass(cls):
        cls.missing = [tool for tool in ("cmake", "ctest") if not shutil.which(tool)]
        if not any(shutil.which(compiler) for compiler in ("c++", "g++", "clang++")):
            cls.missing.append("a C++17 compiler")
        if cls.missing:
            return
        cls.temporary = tempfile.TemporaryDirectory()
        cls.build = Path(cls.temporary.name) / "build"
        steps = (["cmake", "-S", str(OBSERVER), "-B", str(cls.build)], ["cmake", "--build", str(cls.build)])
        cls.log, cls.status = "", 0
        for step in steps:
            result = subprocess.run(step, capture_output=True, text=True, timeout=900, check=False)
            cls.log += result.stdout + result.stderr
            cls.status = cls.status or result.returncode
        cls.test = cls.build / "observer_core_test"

    @classmethod
    def tearDownClass(cls):
        if not cls.missing:
            cls.temporary.cleanup()

    def setUp(self):
        if self.missing:
            self.skipTest("the native core needs " + ", ".join(self.missing))
        if self.status:
            self.fail("the native core does not build:\n" + self.log[-3000:])

    def test_core_builds_without_a_warning_and_links_only_its_modules(self):
        self.assertNotRegex(self.log, r"warning:")
        link = (self.build / "CMakeFiles/observer_core_test.dir/link.txt").read_text()
        self.assertEqual(re.findall(r"\S+\.(?:a|so)\b", link), ["libobserver_core.a", "libreplay_core.a"])
        self.assertNotIn("ocudu", link.lower().replace(str(ROOT).lower(), ""))
        # The core and its unit target include the channel application core and the standard library.
        for name in ("observer_core.h", "observer_core.cpp", "tests/observer_core_test.cpp"):
            for include in re.findall(r'#include [<"]([^>"]+)[>"]', (OBSERVER / name).read_text()):
                self.assertTrue(include in ("observer_core.h", "replay_core.h") or "/" not in include, include)
        # The OCUDU-bound files are only built where the pinned build is imported.
        self.assertIn("if(TARGET ocudu_pin_fixture)", (OBSERVER / "CMakeLists.txt").read_text())
        self.assertNotIn("pilot_ls", "".join(path.name for path in self.build.iterdir()))

    def test_every_native_unit_case_passes(self):
        listed = subprocess.run([str(self.test), "--list"], capture_output=True, text=True, check=True).stdout.split()
        self.assertEqual(tuple(listed), CASES)
        result = subprocess.run(["ctest", "--output-on-failure"], cwd=self.build, capture_output=True, text=True,
                                timeout=600, check=False)
        self.assertEqual(result.returncode, 0, result.stdout[-3000:])
        self.assertIn(f"100% tests passed, 0 tests failed out of {len(CASES)}", result.stdout)
        for name in CASES:
            self.assertRegex(result.stdout, rf"observer_core_{name} \.+\s+Passed")
        unknown = subprocess.run([str(self.test), "no_such_case"], capture_output=True, text=True, check=False)
        self.assertEqual(unknown.returncode, 2)

    def test_native_estimate_and_reference_agree_bit_for_bit(self):
        generator = random.Random(20261121)

        def pattern(stored):
            # A finite single-precision value of moderate size. stored: as the receiver grid keeps it.
            raw = (generator.getrandbits(1) << 31) | (generator.randint(100, 150) << 23) | generator.getrandbits(23)
            value = struct.unpack("<f", struct.pack("<I", raw))[0]
            return grants.bf16(value) if stored else value

        vectors = [(-1.0, 3.0, 1.0, 1.0), (-1.0, 1.0, -COMPONENT, COMPONENT), (0.0, 0.0, COMPONENT, -COMPONENT),
                   (3.0e38, 0.0, 1.0e-3, 0.0)]
        vectors += [(pattern(n % 2), pattern(n % 2), pattern(False), pattern(False)) for n in range(4000)]
        # The reference symbols of the demonstration grant against stored values of every size.
        grant = grants.load(ROOT / "configs/grants/demo_51prb_4rx.json")
        vectors += [(pattern(True), pattern(True), reference.single(value.real), reference.single(value.imag))
                    for value in grant.pilot_reference[0][:612]]
        text = "".join("%08x %08x %08x %08x\n" % struct.unpack("<4I", struct.pack("<4f", *vector)) for vector in vectors)
        result = subprocess.run([str(self.test), "--vectors"], input=text, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.split("\n")[:-1]
        self.assertEqual(len(lines), len(vectors))
        finite = 0
        for vector, line in zip(vectors, lines):
            value, ok = reference.ls(complex(vector[0], vector[1]), complex(vector[2], vector[3]))
            real, imag, flag = line.split()
            self.assertEqual(flag == "1", ok, vector)
            if ok:
                finite += 1
                self.assertEqual(struct.pack("<2f", value.real, value.imag).hex(),
                                 struct.pack("<2I", int(real, 16), int(imag, 16)).hex(), vector)
        self.assertEqual(finite, len(vectors) - 1)
        self.assertEqual(lines[0], "3f800000 40000000 1")  # 1 + 2j

    def test_native_keys_and_slot_times_are_those_of_the_reference(self):
        for text in ("abc", "rz-22963ee9d985.0000", "a" * 64):
            result = subprocess.run([str(self.test), "--key", text], capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout.strip(), reference.opaque_key(text))
        for scs, slot in ((15, 0), (15, 9), (30, 7), (30, 20479), (60, 1), (60, 3), (120, 5), (120, 79), (30, 40967)):
            result = subprocess.run([str(self.test), "--slot-start", str(scs), str(slot)], capture_output=True, text=True,
                                    check=True)
            self.assertEqual(int(result.stdout), reference.slot_start_tc(scs, slot))
        refused = subprocess.run([str(self.test), "--slot-start", "45", "0"], capture_output=True, text=True, check=False)
        self.assertEqual(refused.returncode, 2)


class TreeTests(unittest.TestCase):
    FILES = ("native/ocudu_adapter/observer/CMakeLists.txt", "native/ocudu_adapter/observer/observer_core.h",
             "native/ocudu_adapter/observer/observer_core.cpp", "native/ocudu_adapter/observer/observer_plan.h",
             "native/ocudu_adapter/observer/pilot_ls.cpp", "native/ocudu_adapter/observer/tests/observer_core_test.cpp",
             "native/ocudu_adapter/CMakeLists.txt", "native/ocudu_adapter/README.md",
             "src/ocudu_cuda_isac/sensing/pilot_ls.py", "src/ocudu_cuda_isac/episodes/pilot_ls.py",
             "src/ocudu_cuda_isac/episodes/replay_pusch.py", "tests/unit/test_pilot_ls.py",
             "tests/unit/pilot_ls_double.py", "tests/unit/test_pilot_ls_driver.py", "tests/integration/pilot_ls.py",
             "configs/integration/pilot_ls.json", "scripts/accept_s3_1.sh", "scripts/native_adapter.py",
             "tests/test_accept_s3_1.py", "docs/interfaces/PILOT_LS.md", "Makefile")

    def test_files_are_ascii_and_do_not_name_the_project(self):
        # The working name lives on line 1 of the top-level README and nowhere else.
        name = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()[0].lstrip("# ").split(":")[0].strip()
        for relative in self.FILES:
            data = (ROOT / relative).read_bytes()
            self.assertTrue(data.isascii(), relative)
            if name and name.lower() != "ocudu-cuda-isac":
                self.assertNotIn(name.lower(), data.decode("ascii").lower(), relative)

    def test_the_accepted_native_sources_of_sprint_2_are_not_edited(self):
        adapter = ROOT / "native/ocudu_adapter"
        for relative, digest in ACCEPTED_SOURCES.items():
            self.assertEqual(hashlib.sha256((adapter / relative).read_bytes()).hexdigest(), digest, relative)
        self.assertEqual(len(ACCEPTED_SOURCES), 7)

    def test_the_tool_taps_the_ready_grid_before_the_processors_and_only_reads_it(self):
        tool = (OBSERVER / "pilot_ls.cpp").read_text()
        # The grid is filled and read exactly as the control does.
        self.assertIn("replay.deliver(sequence, pool, key)", tool)
        self.assertIn("const o1_view        view       = pool.view(key, bound.record_sha256);", tool)
        self.assertIn("grids.at(job.grid.grid)->ocudu_grid().get_reader()", tool)
        # The observer gets the read-only view of the ready grid, before the first decode of it.
        tap = tool.index("watcher.observe(view, where)")
        self.assertLess(tool.index("const o1_view        view       = pool.view(key, bound.record_sha256);"), tap)
        self.assertLess(tap, tool.index("const dispatched done = dispatch(book, *backend, name, key"))
        core = (OBSERVER / "observer_core.h").read_text() + (OBSERVER / "observer_core.cpp").read_text()
        self.assertIn("const isac_replay::o1_view& view", core)
        for word in ("writer(", ".put(", "commit(", "release(", "try_acquire("):
            self.assertNotIn(word, core)
        # The estimate belongs to the grid of its digest only when the grid still has it after the read.
        self.assertIn("if (view.digest() != descriptor.readback_sha256)", core)
        # Everything is checked before a processor exists or a file is written.
        self.assertLess(tool.index("watcher.bind(source);"), tool.index("std::make_unique<processor_backend>"))
        self.assertLess(tool.index("watcher.bind(source);"), tool.index("make_directory(ingress_path)"))
        # The processors are those of the control, with the settings Sprint 1 measured.
        control = (ROOT / "native/ocudu_adapter/receiver/replay_pusch.cpp").read_text()
        start = "/// One PUSCH processor of the pinned source, created as the pinned test creates it."
        end = "/// What the decoder itself said about one decode, for the radio record."
        mine, theirs = tool[tool.index(start):tool.index(end)], control[control.index(start):control.index(end)]
        # Sprint 4.1 changed one thing in the wrapper of this tool: the notifier of a decode may be arranged before
        # a paced run, so that its start allocates nothing. Without those lines the wrapper is that of the control.
        arrange = mine[mine.index("  /// Before a paced run: the notifier of a decode"):mine.index("  void start(const request& job")]
        self.assertIn("void arrange(std::uint64_t token, completion_sink& sink)", arrange)
        mine = mine.replace(arrange, "")
        for paced, plain in (
                ("    // A paced run arranged the notifier of this decode before its first fill. Every other run makes it "
                 "here.\n"
                 "    auto known = notifiers.find(job.token);\n"
                 "    if (known == notifiers.end()) {\n"
                 "      known = notifiers.emplace(job.token, std::make_unique<job_notifier>(job.token, sink)).first;\n"
                 "    }\n"
                 "    if (!known->second->claim()) {\n",
                 "    const auto inserted = notifiers.emplace(job.token, std::make_unique<job_notifier>(job.token, sink));\n"
                 "    if (!inserted.second) {\n"),
                ("                       *known->second,\n", "                       *inserted.first->second,\n")):
            self.assertEqual(mine.count(paced), 1, paced)
            mine = mine.replace(paced, plain)
        self.assertEqual(mine, theirs)
        self.assertIn('name == "cuda" ? "gpu" : "auto"', tool)
        self.assertEqual(tool.count("const joined_workers "), 3)
        # The tap is named in every radio record and the replay plan stays the control plan.
        self.assertIn('{"tap", tap.tap}};', tool)
        self.assertIn('if (plan.receiver.tap != "none") {', tool)
        # The estimate is compiled in the core, without contraction. The tool computes none itself
        # and is compiled like the control, with the flags of the pinned test and no other.
        cmake = (OBSERVER / "CMakeLists.txt").read_text()
        self.assertRegex(cmake, r"target_compile_options\(observer_core PRIVATE\s+-O2 -ffp-contract=off")
        self.assertNotIn("target_compile_options(pilot_ls", cmake)
        self.assertNotIn("isac_observer::estimate(", tool)
        self.assertNotIn("isac_observer::pilot_ls(", tool)
        for word in ("truth", "target_present", "position_m"):
            self.assertNotIn(word, tool)
            self.assertNotIn(word, core)

    def test_the_adapter_build_and_the_configuration_name_the_tool(self):
        self.assertIn("add_subdirectory(observer)", (ROOT / "native/ocudu_adapter/CMakeLists.txt").read_text())
        cmake = (OBSERVER / "CMakeLists.txt").read_text()
        self.assertEqual(tuple(re.search(r"set\(OBSERVER_CORE_CASES ([^)]*)\)", cmake)[1].split()), CASES)
        self.assertIn("add_test(NAME pilot_ls_host_tap COMMAND pilot_ls --self-test)", cmake)
        self.assertIn('"pilot_ls": Path("observer/pilot_ls")', (ROOT / "scripts/native_adapter.py").read_text())
        config = json.loads((ROOT / "configs/integration/pilot_ls.json").read_text())
        self.assertEqual((config["schema"], config["selector"], config["backends"]),
                         ("integration-config/pilot_ls/v1", "pilot_ls", ["cpu", "cuda"]))
        self.assertEqual(config["observer"], {"tap": "pilot_ls", "stage": "O2", "source": "pusch_dmrs", "epoch": 1})
        self.assertAlmostEqual(config["unity"]["expect"]["estimate"], UNITY, places=6)
        # The pair is tapped as a whole: both members, each with the opportunities the Sprint 2.4 gate expects of one.
        control = json.loads((ROOT / "configs/integration/replay_pusch.json").read_text())
        self.assertEqual(config["pair"]["expect"]["opportunities"], 2 * control["pair"]["expect"]["opportunities"])
        self.assertEqual(config["pair"]["expect"]["shape"], [4, 2, 306])
        # The strings of the record are the same in the core and in the reference.
        core = (OBSERVER / "observer_core.h").read_text()
        for text in (reference.RULE, reference.PRECISION, reference.LAYOUT):
            self.assertIn(f'"{text}"', core)
        self.assertIn(f'"{reference.DESCRIPTOR_SCHEMA}"', (OBSERVER / "pilot_ls.cpp").read_text())


if __name__ == "__main__":
    unittest.main()
