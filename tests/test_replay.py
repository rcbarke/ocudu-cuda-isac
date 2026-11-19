"""Check complex grid indexing, data boundaries, time joins, and CLI behavior."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ocudu_cuda_isac.replay import ContractError, join_timing, materialize


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


if __name__ == "__main__":
    unittest.main()
