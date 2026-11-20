"""Sprint 2.1 unit gate (selector grant_fixture): the grant record, its indices and its store entry.

Standard library only. The expected pilot symbols below were produced by the two transmit
references, the pinned OCUDU chain and Sionna NR 2.2.0, which agree on every one of them. They are
an answer independent of the generator under test.
"""

import copy
import hashlib
import json
from math import sqrt
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ocudu_cuda_isac.episodes import grant as grants
from ocudu_cuda_isac.episodes import sionna_pusch
from ocudu_cuda_isac.episodes.grant import GrantError
from ocudu_cuda_isac.sensing import sources
from ocudu_cuda_isac.sensing.sources import pusch_dmrs

try:
    import numpy as np
except ImportError:
    np = None

BAKED = ROOT / "configs/grants/two_prb_offset.json"
# The baked grant: 51-block carrier, bandwidth part at CRB 10, two blocks at PRB 5 of it. The
# allocation is CRB 15 and 16, carrier subcarriers 180 to 203.
FIRST, LAST = 180, 203
DATA_SYMBOLS = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13]
# Signs of the real and imaginary part of each pilot, in the record's order: symbol 2 then
# symbol 11, subcarriers 180, 182, ... 202.
PILOT_SIGNS = [(-1, -1), (1, -1), (-1, -1), (1, -1), (-1, 1), (1, 1), (1, 1), (-1, 1), (-1, 1), (-1, -1), (-1, -1),
               (1, -1), (1, 1), (1, -1), (-1, 1), (1, -1), (1, -1), (1, -1), (-1, -1), (1, -1), (1, -1), (1, -1),
               (-1, 1), (-1, -1)]
PILOT_PART = 10 ** (3 / 20) / sqrt(2)  # 3 dB above the data, split over two components


def baked():
    return json.loads(BAKED.read_text())


class MockGrid:
    """Stands in for a resource grid: it records what a mapper allocates and writes."""

    def __init__(self):
        self.allocations, self.puts = [], []

    def make(self, ports, symbols, subcarriers):
        self.allocations.append((ports, symbols, subcarriers))
        return self

    def put(self, port, symbol, subcarrier, value):
        self.puts.append((port, symbol, subcarrier, value))


def transmit_grid(grant):
    """A grid with the grant's pilots and one distinct nonzero value on every data element."""
    x = [0j] * (grants.SYMBOLS_PER_SLOT * grant.grid_subcarriers)
    for (symbol, subcarrier), value in zip(grant.pilot_index, grant.pilot_reference[0]):
        x[symbol * grant.grid_subcarriers + subcarrier] = complex(grants.bf16(value.real), grants.bf16(value.imag))
    for number, (symbol, subcarrier) in enumerate(grant.data_index):
        x[symbol * grant.grid_subcarriers + subcarrier] = complex(1 + number, -1 - number) / 512
    return x


class GrantRecordTests(unittest.TestCase):
    def setUp(self):
        self.grant = grants.load(BAKED)

    def test_full_allocated_symbol_indexes_24_subcarriers_in_increasing_order(self):
        by_symbol = {}
        for symbol, subcarrier in self.grant.data_index:
            by_symbol.setdefault(symbol, []).append(subcarrier)
        self.assertEqual(list(by_symbol), DATA_SYMBOLS)  # symbols increase, the two DM-RS symbols hold no data
        for symbol in DATA_SYMBOLS:
            self.assertEqual(by_symbol[symbol], list(range(FIRST, LAST + 1)), symbol)
            self.assertEqual(len(by_symbol[symbol]), 24)
        self.assertEqual(len(self.grant.data_index), 12 * 24)
        self.assertEqual((self.grant.crb_start, self.grant.subcarrier_start, self.grant.subcarrier_count), (15, 180, 24))

    def test_pilot_locations_and_reference_symbols_match_both_transmit_references(self):
        expected = [(symbol, subcarrier) for symbol in (2, 11) for subcarrier in range(FIRST, LAST + 1, 2)]
        self.assertEqual(self.grant.pilot_index, expected)
        self.assertEqual(len(self.grant.pilot_reference), 1)  # one DM-RS port
        for value, (real, imag) in zip(self.grant.pilot_reference[0], PILOT_SIGNS, strict=True):
            self.assertAlmostEqual(value.real, real * PILOT_PART, places=12)
            self.assertAlmostEqual(value.imag, imag * PILOT_PART, places=12)
        self.assertFalse(set(self.grant.pilot_index) & set(self.grant.data_index))
        self.assertEqual(len(self.grant.occupied()), 288 + 24)

    def test_sequence_is_referenced_to_crb_zero_and_not_to_the_bandwidth_part(self):
        # Moving the bandwidth part and the allocation in opposite directions keeps the same CRBs,
        # so the pilots must not change. Moving only the bandwidth part must change them.
        same = baked()
        same["bwp"] = {"start_rb": 0, "size_rb": 51}
        same["allocation"]["prb_start"] = 15
        self.assertEqual(grants.build(same).pilot_reference, self.grant.pilot_reference)
        moved = baked()
        moved["bwp"]["start_rb"] = 11
        other = grants.build(moved)
        self.assertEqual(other.pilot_index[0], (2, FIRST + 12))
        self.assertNotEqual(other.pilot_reference, self.grant.pilot_reference)

    def test_slot_symbol_and_identity_each_change_the_pilots(self):
        for path, value in ((("slot", "slot"), 18), (("dmrs", "scrambling_id"), 204), (("dmrs", "n_scid"), 1)):
            record = baked()
            record[path[0]][path[1]] = value
            self.assertNotEqual(grants.build(record).pilot_reference, self.grant.pilot_reference, path)
        for path, value in (("n_id", 102), ("rnti", 4661)):  # data scrambling only
            record = baked()
            record[path] = value
            self.assertEqual(grants.build(record).pilot_reference, self.grant.pilot_reference, path)
        symbol_two, symbol_eleven = self.grant.pilot_reference[0][:12], self.grant.pilot_reference[0][12:]
        self.assertNotEqual(symbol_two, symbol_eleven)

    def test_transport_block_bytes_and_bit_order(self):
        self.assertEqual(self.grant.tb[:2], b"\x80\x01")
        bits = self.grant.tb_bits()
        self.assertEqual(len(bits), 384)
        self.assertEqual(bits[:16], [1] + [0] * 14 + [1])  # byte 0 first, most significant bit first
        derived = self.grant.derived()
        self.assertEqual(derived["tb_sha256"], hashlib.sha256(self.grant.tb).hexdigest())
        self.assertEqual(derived["tb_sha256"], "76dccd9e2ed11ce0a9d0eb52b0baaf1a51fa2b755f52df583593120bc72e2f8e")

    def test_transport_block_size_and_coding_follow_the_grant(self):
        derived = self.grant.derived()
        self.assertEqual((derived["tbs_bits"], derived["modulation"], derived["target_code_rate_x1024"]),
                         (384, "16QAM", 340))
        self.assertEqual((derived["ldpc_base_graph"], derived["tb_crc_bits"], derived["codeblocks"]), ("BG2", 16, 1))
        self.assertEqual(derived["coded_bits"], 288 * 4)
        # The two receiver workloads that Sprint 1.2 measured on the Spark.
        self.assertEqual(grants.transport_block_size(51, 14, 24, 4, 340), 9736)
        self.assertEqual(grants.coding(9736, 340), {"ldpc_base_graph": "BG1", "tb_crc_bits": 24, "codeblocks": 2})
        self.assertEqual(grants.transport_block_size(51, 14, 24, 8, 948), 54296)
        self.assertEqual(grants.coding(54296, 948), {"ldpc_base_graph": "BG1", "tb_crc_bits": 24, "codeblocks": 7})
        # Half-integer code rates of the 256QAM table and the smallest table entry.
        self.assertEqual(grants.transport_block_size(51, 14, 24, 8, 682.5), 38936)
        self.assertEqual(grants.transport_block_size(1, 14, 24, 2, 30), 24)

    def test_tx_scale_layers_and_port_mapping_are_stated(self):
        derived = self.grant.derived()
        self.assertEqual(derived["tx_scale"]["data_rms"], 1.0)
        self.assertAlmostEqual(derived["tx_scale"]["dmrs_amplitude"], 10 ** 0.15, places=12)
        self.assertEqual(derived["tx_scale"]["dmrs_epre_boost_db"], 3.0)
        self.assertEqual(derived["port_mapping"], {"layers_to_dmrs_ports": [0], "layers_to_antenna_ports": "identity",
                                                   "rx_ports": [0]})
        self.assertEqual(self.grant.source["source"], "pusch_dmrs")

    def test_frequency_offsets_and_symbol_times_are_physical(self):
        self.assertEqual(self.grant.subcarrier_offset_hz(FIRST), (180 - 306) * 30000)
        self.assertEqual(self.grant.subcarrier_offset_hz(306), 0)  # the carrier frequency
        self.assertEqual(self.grant.subcarrier_offset_hz(611) - self.grant.subcarrier_offset_hz(0), 611 * 30000)
        times = self.grant.symbol_times()
        # 30 kHz: 65536 T_c useful part, 4608 T_c prefix, 1024 T_c more on the first symbol of the slot.
        self.assertEqual((times[0]["cp_start_tc"], times[0]["useful_start_tc"]), (0, 5632))
        self.assertEqual(times[1]["cp_start_tc"], 5632 + 65536)
        self.assertEqual(times[2]["useful_start_tc"] - times[1]["useful_start_tc"], 4608 + 65536)
        self.assertEqual(times[13]["useful_start_tc"] + 65536, 983040)  # one 0.5 ms slot
        self.assertEqual(pusch_dmrs.TC_PER_SECOND // 2000, 983040)
        self.assertEqual(self.grant.source["slot"]["symbol_useful_start_tc"],
                         [times[2]["useful_start_tc"], times[11]["useful_start_tc"]])

    def test_one_cdm_group_without_data_leaves_the_odd_subcarriers_to_data(self):
        record = baked()
        record["dmrs"]["cdm_groups_without_data"] = 1
        record["tb"] = {"generator": grants.TB_GENERATOR, "seed": "one-group", "bytes": 51}
        grant = grants.build(record)
        on_pilot_symbol = [subcarrier for symbol, subcarrier in grant.data_index if symbol == 2]
        self.assertEqual(on_pilot_symbol, list(range(FIRST + 1, LAST + 1, 2)))
        self.assertEqual(len(grant.data_index), 288 + 24)
        self.assertEqual(grant.derived()["tx_scale"]["dmrs_amplitude"], 1.0)
        self.assertEqual(grant.tbs_bits, 408)  # 156 resource elements per block instead of 144

    def test_shipped_grants_load_with_their_known_sizes(self):
        expected = {"two_prb_offset.json": (384, [0], "76dccd9e2ed11ce0"),
                    "demo_51prb_4rx.json": (9736, [0, 1, 2, 3], "a2187d90d75129b1"),
                    "table2_51prb_1rx.json": (54296, [0], "810232b5e9915783")}
        found = {path.name: grants.load(path) for path in sorted((ROOT / "configs/grants").glob("*.json"))}
        self.assertEqual(set(found), set(expected))
        for name, (tbs, rx_ports, digest) in expected.items():
            grant = found[name]
            self.assertEqual((grant.tbs_bits, grant.rx_ports), (tbs, rx_ports), name)
            self.assertTrue(grant.derived()["tb_sha256"].startswith(digest), name)
            self.assertTrue((ROOT / "configs/grants" / name).read_text().isascii())


class RefusalTests(unittest.TestCase):
    """Each invalid record fails in build(), before a grid is allocated or written."""

    def refused(self, record, message, data=None):
        grid = MockGrid()
        with self.assertRaisesRegex(GrantError, message):
            grants.realize(record, grid.make, data)
        self.assertEqual((grid.allocations, grid.puts), ([], []))

    def test_invalid_port_fails_before_allocation(self):
        for ports, message in (([4], "invalid port 4"), ([1], "invalid port mapping"), ([0, 1], "invalid port mapping"),
                               ([0, 0], "duplicate pilot")):
            record = baked()
            record["dmrs"]["ports"] = ports
            self.refused(record, message)
        record = baked()
        record["rx_ports"] = [0, 0]
        self.refused(record, "invalid port: a receive port is listed twice")
        record["rx_ports"] = [16]
        self.refused(record, "rx_ports must be an integer from 0 to 15")
        # The pinned host demodulator takes the i-th listed port from grid port i.
        for ports in ([1], [0, 2], [1, 0], [1, 2, 3, 0]):
            record["rx_ports"] = ports
            self.refused(record, "invalid port list .* receive ports 0 to N-1 in order")
        record["rx_ports"] = [0, 1, 2, 3]
        self.assertEqual(grants.build(record).rx_ports, [0, 1, 2, 3])

    def test_out_of_bwp_prb_fails_before_allocation(self):
        record = baked()
        record["allocation"]["prb_start"] = 23  # PRB 23 and 24 of a 24-block bandwidth part
        self.refused(record, "out-of-BWP PRB")
        record = baked()
        record["bwp"] = {"start_rb": 40, "size_rb": 24}
        self.refused(record, "leaves the 51-block carrier grid")

    def test_duplicate_pilot_fails_before_allocation(self):
        record = baked()
        record["dmrs"]["symbols"] = [2, 2]
        self.refused(record, "duplicate pilot")
        record["dmrs"]["symbols"] = [2, 10]  # not a legal position for this allocation
        self.refused(record, "Table 6.4.1.1.3-3")

    def test_mismatched_payload_fails_before_allocation(self):
        record = baked()
        record["tb"] = {"hex": record["tb"]["hex"][:-2]}  # 47 bytes for a 48-byte block
        self.refused(record, "mismatched payload: the transport block holds 376 bits and the grant carries 384")
        record = baked()
        record["mcs"]["index"] = 11  # the same bytes under another MCS
        self.refused(record, "mismatched payload")
        self.refused(baked(), "mismatched payload: 287 data symbols for 288", data=[1j] * 287)

    def test_malformed_records_are_refused(self):
        cases = (
            (lambda r: r.pop("rnti"), "requires exactly"),
            (lambda r: r.update(extra=1), "requires exactly"),
            (lambda r: r.update(schema="pusch-grant/v0"), "unsupported grant schema"),
            (lambda r: r.update(grant_id="../x"), "grant_id"),
            (lambda r: r["carrier"].update(scs_khz=45), "scs_khz"),
            (lambda r: r["carrier"].update(cyclic_prefix="extended"), "normal cyclic prefix"),
            (lambda r: r["slot"].update(slot=20), "slot.slot must be an integer from 0 to 19"),
            (lambda r: r.update(rnti=0), "rnti"),
            (lambda r: r.update(rnti=True), "rnti"),
            (lambda r: r["mcs"].update(index=29), "mcs.index"),
            (lambda r: r["mcs"].update(table="qam1024"), "mcs.table"),
            (lambda r: r["codeword"].update(rv=2), "redundancy version 0"),
            (lambda r: r.update(layers=2), "one transmission layer"),
            (lambda r: r["tx"].update(precoding="codebook"), "identity precoding"),
            (lambda r: r["dmrs"].update(config_type=2), "configuration type 1"),
            (lambda r: r["dmrs"].update(length=2), "single-symbol"),
            (lambda r: r["allocation"].update(symbol_count=15), "symbol_count"),
            (lambda r: r["allocation"].update(mapping_type="C"), "mapping type must be A or B"),
            (lambda r: r.update(tb={"hex": "zz"}), "not hexadecimal"),
            (lambda r: r.update(tb={"generator": "other", "seed": "s", "bytes": 48}), "unknown transport block generator"),
            (lambda r: r.update(tb={"file": "../tb.bin", "bytes": 48, "sha256": "0" * 64}), "beside the grant record"),
        )
        for change, message in cases:
            record = baked()
            change(record)
            with self.assertRaisesRegex(GrantError, message):
                grants.build(record)

    def test_a_record_file_with_a_repeated_key_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "grant.json"
            path.write_text(BAKED.read_text().replace('"rnti": 4660,', '"rnti": 4660, "rnti": 4661,'))
            with self.assertRaisesRegex(GrantError, "repeated key"):
                grants.load(path)
            with self.assertRaisesRegex(GrantError, "cannot read"):
                grants.load(Path(temporary) / "absent.json")


class MockGridTests(unittest.TestCase):
    def test_pilots_then_data_reach_the_grid_in_record_order(self):
        grid = MockGrid()
        data = [complex(number, -number) for number in range(1, 289)]
        grant, returned = grants.realize(baked(), grid.make, data)
        self.assertIs(returned, grid)
        self.assertEqual(grid.allocations, [(1, 14, 612)])  # one antenna port, the whole carrier grid
        pilots, written = grid.puts[:24], grid.puts[24:]
        self.assertEqual([(symbol, subcarrier) for _, symbol, subcarrier, _ in pilots], grant.pilot_index)
        self.assertEqual([value for *_, value in pilots], grant.pilot_reference[0])
        self.assertEqual([value for *_, value in written], data)
        for position, symbol in enumerate(DATA_SYMBOLS):
            row = written[24 * position:24 * (position + 1)]
            self.assertEqual([(port, where, subcarrier) for port, where, subcarrier, _ in row],
                             [(0, symbol, subcarrier) for subcarrier in range(FIRST, LAST + 1)])
        self.assertEqual(len({(symbol, subcarrier) for _, symbol, subcarrier, _ in grid.puts}), 312)  # no RE twice

    def test_pilots_alone_are_written_without_data(self):
        grid = MockGrid()
        grants.realize(baked(), grid.make)
        self.assertEqual(len(grid.puts), 24)


class SourceRegistryTests(unittest.TestCase):
    def test_pusch_dmrs_is_the_registered_first_source(self):
        self.assertEqual(sources.names(), ["pusch_dmrs"])
        self.assertIs(sources.resolve("pusch_dmrs"), pusch_dmrs.build)
        with self.assertRaisesRegex(sources.SourceError, "unknown sensing source 'pdsch_dmrs'"):
            sources.resolve("pdsch_dmrs")
        with self.assertRaisesRegex(sources.SourceError, "already registered"):
            sources.register("pusch_dmrs", pusch_dmrs.build)

    def test_the_source_record_names_channel_direction_mask_reference_ports_processor_and_timing(self):
        record = grants.load(BAKED).source
        self.assertEqual(set(record), set(sources.FIELDS))
        self.assertEqual((record["channel"], record["link_direction"], record["ports"]), ("PUSCH", "uplink", [0]))
        self.assertEqual(record["processor"], {"name": "pusch", "backend": None})
        self.assertEqual(record["slot"]["symbols"], [2, 11])
        self.assertEqual(record["grid"], {"coordinates": "carrier", "symbols": 14, "subcarriers": 612})
        self.assertEqual(len(record["mask"]["index"]), 24)
        self.assertEqual(len(record["reference"]["values"][0]), 24)
        self.assertIs(sources.validate(record), record)
        self.assertTrue(json.dumps(record).isascii())

    def test_invalid_source_records_are_refused(self):
        good = grants.load(BAKED).source
        cases = (
            (lambda r: r["mask"]["index"].append([2, 180]), "duplicate pilot"),
            (lambda r: r["mask"]["index"].reverse(), "ordered by symbol"),
            (lambda r: r["reference"]["values"][0].__setitem__(3, [0.0, 0.0]), "zero reference power"),
            (lambda r: r["reference"]["values"][0].pop(), "one value per pilot"),
            (lambda r: r.update(link_direction="sidelink"), "uplink or downlink"),
            (lambda r: r["grid"].update(coordinates="bwp"), "carrier coordinates"),
            (lambda r: r["mask"]["index"].__setitem__(0, [2, 612]), "pilot subcarrier"),
            (lambda r: r.pop("processor"), "requires exactly"),
        )
        for change, message in cases:
            record = copy.deepcopy(good)
            change(record)
            with self.assertRaisesRegex(sources.SourceError, message):
                sources.validate(record)

    def test_second_port_of_the_cdm_group_alternates_its_sign(self):
        parameters = {"scs_khz": 30, "sfn": 0, "slot": 17, "grid_size_rb": 51, "crb_start": 15, "crb_count": 2,
                      "config_type": 1, "symbols": [2, 11], "ports": [0, 1], "scrambling_id": 203, "n_scid": 0,
                      "cdm_groups_without_data": 2}
        record = pusch_dmrs.build(parameters)
        first, second = record["reference"]["values"]
        for number, (a, b) in enumerate(zip(first, second)):
            sign = -1 if number % 2 else 1  # r(m) with odd m carries the weight -1 on port 1
            self.assertEqual(b, [sign * a[0], sign * a[1]])
        parameters["ports"] = [0, 2]
        with self.assertRaisesRegex(sources.SourceError, "different CDM groups"):
            pusch_dmrs.build(parameters)
        parameters["ports"] = [5]
        with self.assertRaisesRegex(sources.SourceError, "invalid port 5"):
            pusch_dmrs.build(parameters)

    def test_dmrs_positions_follow_the_table(self):
        self.assertEqual(pusch_dmrs.symbol_positions("A", 2, 1, 0, 14), [2, 11])
        self.assertEqual(pusch_dmrs.symbol_positions("A", 3, 0, 0, 14), [3])
        self.assertEqual(pusch_dmrs.symbol_positions("A", 2, 3, 0, 14), [2, 5, 8, 11])
        self.assertEqual(pusch_dmrs.symbol_positions("A", 2, 2, 0, 10), [2, 6, 9])
        self.assertEqual(pusch_dmrs.symbol_positions("B", 2, 1, 3, 7), [3, 7])
        self.assertEqual(pusch_dmrs.symbol_positions("B", 2, 3, 0, 14), [0, 3, 6, 9])
        for arguments in (("A", 2, 1, 1, 13), ("A", 2, 1, 0, 3), ("A", 3, 3, 0, 14), ("B", 2, 4, 0, 14)):
            with self.assertRaises(sources.SourceError):
                pusch_dmrs.symbol_positions(*arguments)

    def test_gold_sequence_and_seed(self):
        # c_init = 0 leaves the second register empty: the sequence is the first m-sequence, which
        # satisfies x(n + 31) = x(n + 3) + x(n) and has period 2**31 - 1.
        zero = pusch_dmrs.gold_sequence(0, 96)
        self.assertEqual([zero[n + 31] for n in range(65)], [zero[n + 3] ^ zero[n] for n in range(65)])
        self.assertEqual(pusch_dmrs.gold_sequence(12345, 40), pusch_dmrs.gold_sequence(12345, 80)[:40])
        self.assertNotEqual(pusch_dmrs.gold_sequence(12345, 64), pusch_dmrs.gold_sequence(12346, 64))
        self.assertEqual(pusch_dmrs.sequence_seed(0, 0, 0, 0), 2 ** 17)
        self.assertEqual(pusch_dmrs.sequence_seed(17, 2, 203, 0), ((14 * 17 + 3) * 407 * 2 ** 17 + 406) % 2 ** 31)
        self.assertEqual(pusch_dmrs.sequence_seed(1, 0, 0, 1), 15 * 2 ** 17 + 1)

    def test_the_sensing_core_imports_no_host_simulator_or_episode_module(self):
        code = ("import sys; import ocudu_cuda_isac.sensing.sources; "
                "print(sorted(m for m in sys.modules if m.split('.')[0] in ('sionna', 'torch', 'numpy') "
                "or m.startswith('ocudu_cuda_isac.episodes')))")
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
        self.assertEqual(result.stdout.strip(), "[]")


class StorageTests(unittest.TestCase):
    def test_bf16_keeps_eight_significant_bits_and_rounds_to_even(self):
        self.assertEqual(grants.bf16(1 / sqrt(10)), 81 / 256)      # a 16QAM inner level as the grid stores it
        self.assertEqual(grants.bf16(3 / sqrt(10)), 243 / 256)
        self.assertEqual(grants.bf16(PILOT_PART), 1.0)             # 0.99881 is nearer to 1 than to 255/256
        self.assertEqual(grants.bf16(-PILOT_PART), -1.0)
        self.assertEqual(grants.bf16(0.0), 0.0)
        self.assertEqual(grants.bf16(1 + 1 / 256), 1.0)            # a tie goes to the even neighbour
        self.assertEqual(grants.bf16(1 + 3 / 256), 1 + 4 / 256)
        self.assertEqual(grants.bf16(0.75), 0.75)

    def test_storage_tolerance_is_one_part_in_256_per_component(self):
        self.assertTrue(grants.within_storage(complex(81 / 256, -243 / 256), complex(1, -3) / sqrt(10)))
        self.assertTrue(grants.within_storage(1 + 1j, complex(PILOT_PART, PILOT_PART)))
        self.assertFalse(grants.within_storage(complex(0.32, 0), complex(1 / sqrt(10), 0)))
        self.assertFalse(grants.within_storage(complex(81 / 256, 81 / 256), complex(81 / 256, -81 / 256)))  # a sign
        self.assertEqual(grants.STORAGE["relative"], 1 / 256)


class StoreTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.grant = grants.load(BAKED)
        self.x = transmit_grid(self.grant)

    def entry(self, name="entry", grids=True):
        selected = {"x": {"values": self.x, "provenance": {"reference": "unit test"}}} if grids else None
        grants.export(self.grant, self.root / name, selected)
        return self.root / name

    def test_a_second_reader_recovers_pilot_data_symbol_and_block_hash(self):
        store = grants.open_store(self.entry())
        self.assertEqual(sorted(path.name for path in store.directory.iterdir()),
                         ["data_index.npy", "grant.json", "manifest.json", "pilot_index.npy", "pilot_ref.npy",
                          "source.json", "tb.bin", "x.npy"])
        pilot = store.pilot(13)
        self.assertEqual((pilot["symbol"], pilot["subcarrier"], pilot["offset_hz"]), (11, 182, (182 - 306) * 30000))
        self.assertAlmostEqual(pilot["reference"][0], PILOT_PART)
        self.assertAlmostEqual(pilot["reference"][1], -PILOT_PART)
        self.assertEqual(pilot["transmitted"], [1.0, -1.0])
        data = store.data(25)  # the second data symbol of the second allocated symbol
        self.assertEqual((data["symbol"], data["subcarrier"], data["value"]), (1, 181, [26 / 512, -26 / 512]))
        self.assertEqual((store.directory / "tb.bin").read_bytes(), self.grant.tb)
        self.assertEqual(store.manifest["files"]["tb.bin"]["sha256"], self.grant.derived()["tb_sha256"])
        self.assertEqual(store.manifest["derived"]["tbs_bits"], 384)
        self.assertEqual(store.manifest["representation"], grants.STORAGE)
        self.assertEqual(store.manifest["symbol_times"], self.grant.symbol_times())  # actual, not uniform, times
        self.assertEqual(store.manifest["tc_per_second"], 1966080000)
        self.assertEqual(store.manifest["files"]["x.npy"]["axes"], ["tx_port", "symbol", "subcarrier"])
        self.assertEqual(store.manifest["files"]["pilot_index.npy"]["columns"], ["symbol", "subcarrier"])
        self.assertEqual(store.manifest["transmit_references"]["x.npy"]["check"]["pilots_equal_after_rounding"], 24)
        self.assertEqual(store.grant.record(), self.grant.record())
        for path in store.directory.glob("*.json"):
            self.assertTrue(path.read_text().isascii(), path.name)
        with self.assertRaisesRegex(GrantError, "pilots 0 to 23"):
            store.pilot(24)
        with self.assertRaisesRegex(GrantError, "data symbols 0 to 287"):
            store.data(288)

    def test_an_entry_without_a_grid_still_gives_pilots_and_the_block(self):
        store = grants.open_store(self.entry(grids=False))
        self.assertIsNone(store.x)
        self.assertEqual(store.pilot(0)["subcarrier"], FIRST)
        self.assertNotIn("transmitted", store.pilot(0))
        with self.assertRaisesRegex(GrantError, "no transmit grid"):
            store.data(0)

    def test_an_entry_is_never_overwritten_and_appears_only_complete(self):
        directory = self.entry()
        with self.assertRaisesRegex(GrantError, "never overwritten"):
            grants.export(self.grant, directory)
        bad = list(self.x)
        bad[0] = 1j  # symbol 0, subcarrier 0 is outside the allocation
        with self.assertRaisesRegex(GrantError, "1 unexpected, 0 empty, first at symbol 0, subcarrier 0"):
            grants.export(self.grant, self.root / "bad", {"x": {"values": bad, "provenance": None}})
        self.assertFalse((self.root / "bad").exists())
        self.assertEqual(sorted(path.name for path in self.root.iterdir()), ["entry"])

    def test_a_grid_that_differs_from_the_grant_is_refused(self):
        width = self.grant.grid_subcarriers
        cases = (
            (lambda x: x.__setitem__(0 * width + FIRST, 0j), "0 unexpected, 1 empty"),             # a missing data symbol
            (lambda x: x.__setitem__(2 * width + FIRST + 1, 1j), "1 unexpected, 0 empty"),         # data on a DM-RS symbol
            (lambda x: x.__setitem__(2 * width + FIRST, -x[2 * width + FIRST]), "the pilot at symbol 2, subcarrier 180"),
            (lambda x: x.__setitem__(13 * width + LAST, complex(float("nan"), 0)), "nonfinite"),
            (lambda x: x.pop(), "the transmit grid holds 8567 values"),
        )
        for number, (change, message) in enumerate(cases):
            x = list(self.x)
            change(x)
            with self.assertRaisesRegex(GrantError, message):
                grants.export(self.grant, self.root / f"case{number}", {"x": {"values": x, "provenance": None}})
        with self.assertRaisesRegex(GrantError, 'under the name "x"'):
            grants.export(self.grant, self.root / "named", {"x_sionna": {"values": self.x, "provenance": None}})
        with self.assertRaisesRegex(GrantError, 'named "x" or "x_<name>"'):
            grants.export(self.grant, self.root / "named", {"x": {"values": self.x, "provenance": None},
                                                            "../y": {"values": self.x, "provenance": None}})
        huge = list(self.x)
        huge[FIRST] = complex(1e39, 0)  # finite, and beyond single precision
        with self.assertRaisesRegex(GrantError, "does not fit the array type"):
            grants.export(self.grant, self.root / "huge", {"x": {"values": huge, "provenance": None}})
        self.assertEqual(list(self.root.iterdir()), [])  # no refused entry leaves a directory

    def test_a_changed_file_is_detected_on_reading(self):
        def duplicate_pilot(path):
            data = bytearray(path.read_bytes())
            data[-8:] = data[-16:-8]  # the last pilot repeats the one before it
            path.write_bytes(bytes(data))

        cases = (
            ("tb.bin", lambda path: path.write_bytes(b"\x00" + path.read_bytes()[1:]), "tb.bin differs"),
            ("x.npy", lambda path: path.write_bytes(path.read_bytes()[:-1] + b"\x01"), "x.npy differs"),
            ("pilot_index.npy", duplicate_pilot, "pilot_index.npy differs"),
            ("manifest.json", lambda path: path.unlink(), "not a complete store entry"),
            ("data_index.npy", lambda path: path.unlink(), "cannot read data_index.npy"),
        )
        for number, (name, change, message) in enumerate(cases):
            directory = self.entry(f"entry{number}")
            change(directory / name)
            with self.assertRaisesRegex(GrantError, message):
                grants.open_store(directory)

    def test_a_consistent_forgery_is_still_refused_by_the_grant(self):
        # Hashes that agree with changed arrays do not help: the arrays must be the grant's own.
        def forge(directory, name, change):
            manifest = json.loads((directory / "manifest.json").read_text())
            data = bytearray((directory / name).read_bytes())
            change(data)
            (directory / name).write_bytes(bytes(data))
            manifest["files"][name]["sha256"] = hashlib.sha256(bytes(data)).hexdigest()
            (directory / "manifest.json").write_text(json.dumps(manifest))

        def repeat_last(data):
            data[-8:] = data[-16:-8]

        def flip_sign(data):
            data[-1] ^= 0x80  # the imaginary part of the last value

        def flip_first_pilot(data):
            start = len(data) - 8 * 14 * 612  # the array follows the header
            data[start + 8 * (2 * 612 + FIRST) + 3] ^= 0x80  # the real part at symbol 2, subcarrier 180

        cases = (("pilot_index.npy", repeat_last, "duplicate pilot in pilot_index.npy"),
                 ("data_index.npy", repeat_last, "data_index.npy differs from the data resource elements"),
                 ("pilot_ref.npy", flip_sign, "pilot_ref.npy differs from the reference symbols"),
                 ("x.npy", flip_first_pilot, "the pilot at symbol 2, subcarrier 180"))
        for number, (name, change, message) in enumerate(cases):
            directory = self.entry(f"forged{number}")
            forge(directory, name, change)
            with self.assertRaisesRegex(GrantError, message):
                grants.open_store(directory)

    def test_command_line_exports_and_shows_an_entry(self):
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        command = [sys.executable, "-m", "ocudu_cuda_isac.episodes.grant"]
        target = self.root / "cli"
        made = subprocess.run(command + ["export", "--grant", str(BAKED), "--output-dir", str(target)],
                              capture_output=True, text=True, env=env, check=False)
        self.assertEqual(made.returncode, 0, made.stderr)
        self.assertEqual(json.loads(made.stdout)["tbs_bits"], 384)
        again = subprocess.run(command + ["export", "--grant", str(BAKED), "--output-dir", str(target)],
                               capture_output=True, text=True, env=env, check=False)
        self.assertEqual(again.returncode, 2)
        self.assertIn("never overwritten", again.stderr)
        shown = subprocess.run(command + ["show", str(self.entry()), "--pilot", "1", "--data", "287"],
                               capture_output=True, text=True, env=env, check=False)
        self.assertEqual(shown.returncode, 0, shown.stderr)
        result = json.loads(shown.stdout)
        self.assertEqual(result["tb_sha256"], self.grant.derived()["tb_sha256"])
        self.assertEqual((result["pilot"]["symbol"], result["pilot"]["subcarrier"]), (2, 182))
        self.assertEqual((result["data"]["symbol"], result["data"]["subcarrier"]), (13, LAST))
        self.assertEqual(result["transmit_grids"], ["x"])
        self.assertTrue(shown.stdout.isascii())

    @unittest.skipIf(np is None, "optional NumPy runtime: the arrays must load with numpy.load")
    def test_numpy_reads_the_stored_arrays(self):
        directory = self.entry()
        self.assertEqual(np.load(directory / "pilot_index.npy").tolist(), [list(entry) for entry in self.grant.pilot_index])
        self.assertEqual(np.load(directory / "data_index.npy").shape, (288, 2))
        reference = np.load(directory / "pilot_ref.npy")
        self.assertEqual((reference.dtype.str, reference.shape), ("<c8", (1, 24)))
        x = np.load(directory / "x.npy")
        self.assertEqual((x.dtype.str, x.shape), ("<c8", (1, 14, 612)))
        self.assertEqual(x[0, 1, 181], complex(26, -26) / 512)
        self.assertEqual(int(np.count_nonzero(x)), 312)


class SionnaAlignmentTests(unittest.TestCase):
    def test_two_configurations_align_the_sionna_reference_without_importing_it(self):
        values = sionna_pusch.parameters(grants.load(BAKED))
        self.assertEqual(values["data_n_size_bwp"], 2)            # data: the bandwidth part is the allocation
        self.assertEqual(values["dmrs_n_size_bwp"], 17)           # DM-RS: CRB 0 to the end of the allocation
        self.assertEqual(values["dmrs_first_subcarrier"], FIRST)
        common = values["common"]
        self.assertEqual(common["carrier"]["slot_number"], 17)
        self.assertEqual(common["dmrs"]["n_id"], [203, 203])
        self.assertEqual((common["tb"]["mcs_table"], common["tb"]["mcs_index"], common["tb"]["n_id"]), (1, 10, 101))
        self.assertEqual(common["pusch"]["n_rnti"], 4660)
        self.assertNotIn("sionna", sys.modules)


if __name__ == "__main__":
    unittest.main()
