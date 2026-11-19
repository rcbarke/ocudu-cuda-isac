"""Check complex grid indexing, data boundaries, time joins, and CLI behavior."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ocudu_cuda_isac.replay import ContractError, join_timing, load_json, materialize

# Reference bytes of the checked-in fixture. Its arithmetic is exact in binary floating point,
# so every host and interpreter must reproduce these files.
REFERENCE_SHA256 = {
    "receiver_ingress.jsonl": "41080f80ddff93e3e44a3556952e0dec024f58d244cc9f850dbcc560d2371177",
    "offline_oracle.jsonl": "0502a57965110b0ff36e17a8f8416e039264d5b86ff970fd0a6e466f349813da",
    "timing_summary.json": "f2e67316c16c756b8abddaf7d2819070151497e6d8a4d10a135146ec8f6b7b5d",
}


def fixture(name):
    return json.loads((ROOT / "configs" / name).read_text())


class ReplayTest(unittest.TestCase):
    def setUp(self):
        self.episode = fixture("smoke_replay.json")
        self.trace = fixture("smoke_timing.json")

    def test_complex_grid_and_oracle_boundary(self):
        ingress, oracle = materialize(self.episode)
        self.assertEqual(ingress[1]["y"], [[1, 0], [0, 1], [0, 1], [-1, 0],
                                            [-1, 0], [0, -1], [0, -1], [1.125, -0.25]])
        self.assertEqual(oracle[1]["synthetic_case"], "per-receiver phase")
        self.assertEqual(set(ingress[1]), {"episode_id", "seq", "clock_domain", "symbol_times_ns", "ready_ns",
                                          "shape_rx_symbol_subcarrier", "subcarrier_offsets_hz", "pilot_mask", "y"})
        self.episode["records"][1]["case"] = "private evaluator note"
        self.assertEqual(materialize(self.episode)[0], ingress)

    def test_multi_transmit_multi_symbol(self):
        self.episode.update(shape=[2, 2, 2, 1], subcarrier_offsets_hz=[0])
        row = self.episode["records"][0]
        row.update(symbol_times_ns=[0,100], pilot_mask=[True, False], x=[[1,0],[2,0],[3,0],[4,0]],
                   h=[[1,0],[1,0],[0,1],[0,-1],[0.5,0],[1,0],[2,0],[3,0]], w=[[0,0]] * 4)
        self.episode["records"] = [row]
        self.assertEqual(materialize(self.episode)[0][0]["y"], [[1,3],[2,-4],[6.5,0],[14,0]])

    def test_join_uses_keys_and_preserves_failures(self):
        ingress, _ = materialize(self.episode)
        summary = join_timing(ingress, self.trace)
        self.assertEqual(summary["counts"], dict(on_time=1, late=1, dropped=1, missing=1))
        self.assertEqual(summary["updates"][0], {"seq": 0, "outcome": "on_time", "latency_ns": 100000})
        self.assertEqual(summary["updates"][1], {"seq": 1, "outcome": "late", "latency_ns": 100001})
        self.assertEqual(sum(summary["counts"].values()), len(ingress))
        self.trace["events"].reverse()
        self.assertEqual(join_timing(ingress, self.trace), summary)

    def test_malformed_grid_fails_before_emitting_records(self):
        for field, invalid in (("h", []), ("w", [[float("nan"), 0]] * 8),
                               ("pilot_mask", [1, 0]), ("symbol_times_ns", [20001]),
                               ("seq", True)):
            bad = copy.deepcopy(self.episode)
            bad["records"][0][field] = invalid
            with self.subTest(field=field), self.assertRaises(ContractError):
                materialize(bad)
        self.episode["records"][1]["seq"] = 0
        with self.assertRaisesRegex(ContractError, "increase"):
            materialize(self.episode)

    def test_rejects_unjoinable_timing(self):
        ingress, _ = materialize(self.episode)
        for patch in ({"clock_domain": "unsynchronized-clock"}, {"episode_id": "wrong"},
                      {"events": [{"seq": 4, "status": "published", "publish_ns": 90000}]},
                      {"events": [{"seq": 0, "status": "published", "publish_ns": 19999}]},
                      {"events": [self.trace["events"][0], self.trace["events"][0]]}):
            with self.subTest(patch=patch), self.assertRaises(ContractError):
                join_timing(ingress, {**self.trace, **patch})

    def test_shell_entry_point_is_reproducible(self):
        with tempfile.TemporaryDirectory() as temp:
            dest = Path(temp) / "scratch outputs"
            command = [str(ROOT / "scripts/run_smoke.sh"), "--output-dir", str(dest)]
            subprocess.run(command, cwd=temp, check=True, capture_output=True)
            first = {path.name: path.read_bytes() for path in dest.iterdir()}
            self.assertEqual(set(first), {"receiver_ingress.jsonl", "offline_oracle.jsonl", "timing_summary.json"})
            self.assertEqual(json.loads(first["timing_summary.json"])["scheduled_updates"], 4)
            subprocess.run(command, cwd=temp, check=True, capture_output=True)
            self.assertEqual(first, {path.name: path.read_bytes() for path in dest.iterdir()})
            # A repeat into another directory and the recorded reference agree byte for byte.
            other = Path(temp) / "second run"
            subprocess.run([command[0], "--output-dir", str(other)], cwd=temp, check=True, capture_output=True)
            self.assertEqual(first, {path.name: path.read_bytes() for path in other.iterdir()})
            self.assertEqual({name: hashlib.sha256(data).hexdigest() for name, data in first.items()},
                             REFERENCE_SHA256)

    def test_receiver_file_holds_no_evaluator_field(self):
        ingress, oracle = materialize(self.episode)
        private = {"x", "h", "w", "case", "synthetic_case", "shape_rx_tx_symbol_subcarrier"}
        for public, hidden in zip(ingress, oracle):
            self.assertFalse(private & set(public))
            self.assertEqual(set(hidden), {"episode_id", "seq", "synthetic_case",
                                           "shape_rx_tx_symbol_subcarrier", "h", "x", "w"})
            self.assertEqual((public["episode_id"], public["seq"]), (hidden["episode_id"], hidden["seq"]))

    def test_records_share_no_state_with_each_other_or_the_source(self):
        pristine = copy.deepcopy(self.episode)
        ingress, oracle = materialize(self.episode)
        expected = copy.deepcopy((ingress[1:], oracle[1:]))
        ingress[0]["subcarrier_offsets_hz"][0] = 12345
        ingress[0]["pilot_mask"][0] = False
        ingress[0]["symbol_times_ns"][0] = 7
        ingress[0]["y"][0][0] = 99.0
        oracle[0]["shape_rx_tx_symbol_subcarrier"][0] = 9
        for field in ("h", "x", "w"):
            oracle[0][field][0][0] = 99
        self.assertEqual((ingress[1:], oracle[1:]), expected)
        self.assertEqual(self.episode, pristine)

    def test_values_a_native_consumer_cannot_hold_are_rejected(self):
        too_large, huge = 2**63, 10**400  # beyond a signed 64-bit integer; beyond any double
        episode_cases = ((("records", 3, "ready_ns"), too_large), (("records", 3, "symbol_times_ns", 0), -1),
                         (("records", 0, "x", 0, 0), huge), (("records", 0, "h", 7, 1), -huge),
                         (("records", 0, "w", 0, 0), 1e309), (("subcarrier_offsets_hz",), [0, -30000]),
                         (("subcarrier_offsets_hz",), [0, 0]), (("subcarrier_offsets_hz",), [-huge, 0]),
                         (("subcarrier_offsets_hz",), [0.0, 30000]), (("shape",), [4, 1, 1, too_large]))
        for path, value in episode_cases:
            bad = copy.deepcopy(self.episode)
            node = bad
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = value
            with self.subTest(path=path, value=value), self.assertRaises(ContractError):
                materialize(bad)
        ingress, _ = materialize(self.episode)
        for patch in ({"deadline_ns": too_large}, {"deadline_ns": 0},
                      {"events": [{"seq": 0, "status": "published", "publish_ns": too_large}]},
                      {"events": [{"seq": too_large, "status": "dropped", "reason": "x"}]}):
            with self.subTest(patch=patch), self.assertRaises(ContractError):
                join_timing(ingress, {**self.trace, **patch})
        largest = copy.deepcopy(self.episode)
        largest["records"] = largest["records"][:1]
        largest["records"][0].update(symbol_times_ns=[2**63 - 2], ready_ns=2**63 - 1)
        self.assertEqual(materialize(largest)[0][0]["ready_ns"], 2**63 - 1)

    def test_ambiguous_documents_are_rejected_on_load(self):
        with tempfile.TemporaryDirectory() as temp:
            document = Path(temp) / "fixture.json"
            cases = (('{"seq": 0, "seq": 1}', "repeated key 'seq'"),
                     ('{"records": [{"h": [[1, 0]], "h": [[0, 1]]}]}', "repeated key 'h'"),
                     ('{"deadline_ns": NaN}', "nonfinite constant NaN"),
                     ('{"w": [[Infinity, 0]]}', "nonfinite constant Infinity"),
                     ('{"seq": 0,}', "fixture.json"))
            for text, message in cases:
                document.write_text(text)
                with self.subTest(text=text), self.assertRaisesRegex(ContractError, message):
                    load_json(document)
            document.write_text('{"seq": 0, "nested": {"seq": 1}}')
            self.assertEqual(load_json(document), {"seq": 0, "nested": {"seq": 1}})
        self.assertEqual(load_json(ROOT / "configs/smoke_replay.json"), self.episode)
        self.assertEqual(load_json(ROOT / "configs/smoke_timing.json"), self.trace)

    def test_rejected_input_writes_no_output(self):
        with tempfile.TemporaryDirectory() as temp:
            trace = Path(temp) / "trace.json"
            cases = ({**self.trace, "clock_domain": "unsynchronized-clock"},
                     {**self.trace, "events": self.trace["events"] + self.trace["events"][:1]})
            for index, document in enumerate(cases):
                trace.write_text(json.dumps(document))
                dest = Path(temp) / f"refused {index}"
                result = subprocess.run(
                    [sys.executable, "-m", "ocudu_cuda_isac", "--episode", str(ROOT / "configs/smoke_replay.json"),
                     "--trace", str(trace), "--output-dir", str(dest)],
                    cwd=ROOT / "src", capture_output=True, text=True, check=False)
                with self.subTest(index=index):
                    self.assertEqual(result.returncode, 2)
                    self.assertIn("Replay input error", result.stderr)
                    self.assertFalse(dest.exists())


if __name__ == "__main__":
    unittest.main()
