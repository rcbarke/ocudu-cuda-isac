"""Unit gate of Sprint 3.2 (selector snapshot_ring): the observation record, the ring and its two readers.

Run directly, as the sprint card names it:

    python tests/unit/test_snapshot_ring.py

The record and the plan are checked with the standard library alone. The native part needs cmake,
ctest and a C++17 compiler and no OCUDU source or build: it builds the producer glue of
native/ocudu_adapter/transport/ on its own, which compiles the ring of native/obs_ring/ and the
two host-free cores it joins, runs every native unit case, and then lets the reference reader of
ocudu_cuda_isac.sensing.obs and the native consumer read the same records from a native producer.
"""

import copy
import hashlib
import json
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
from ocudu_cuda_isac.episodes import pilot_ls as pl  # noqa: E402
from ocudu_cuda_isac.episodes import snapshot_ring as sr  # noqa: E402
from ocudu_cuda_isac.episodes.snapshot_ring import SnapshotRingError  # noqa: E402
from ocudu_cuda_isac.sensing import obs  # noqa: E402
from ocudu_cuda_isac.sensing.obs import ObsError  # noqa: E402

RING = ROOT / "native/obs_ring"
ADAPTER = ROOT / "native/ocudu_adapter"
TRANSPORT = ADAPTER / "transport"
RING_CASES = ("definition", "capacity_two", "scratch_overwrite", "no_early_read", "stale_read", "double_release",
              "released_fault", "fifo", "spin_no_syscalls", "wait_flag", "wait_no_loss", "consumer_exit",
              "definition_refused", "read_only", "busy_and_drain", "long_path")
PUBLISHER_CASES = ("needs", "timed", "audit", "drop", "stale_grid", "refusals")
# The definition of obs/v1 and obs-ring/v1 that both readers were built from. A change of a field
# changes this value, and a consumer of the old value is refused when it attaches.
DEFINITION = "6f877a8710c433dd"
# The native sources that Sprint 2 accepted and that no sprint has edited since.
UNEDITED_SOURCES = {
    "fixtures/grant_fixture.cpp": "48e36372e08973b9331446292f723941bf8a64864419360266d575623a6537d8",
    "replay/replay_plan.h": "6b6ead72396b9ba5d1639b055eee395966355f7e3f53890064b07f3c20b8a981",
    "replay/pinned_grid.h": "465d265f46ca40c9613e77c0e3c9eec192a2f16b837a39c601e74bc65814ecc0",
    "replay/grid_apply.cpp": "62d76c8e3b105bb92e9874981889508a22ebd37bedd932ee26abcab300754f17",
    "receiver/grant_pdu.h": "f75d4ac1b0fe5ff17ba225b1b8b487dbf70bb6064482ce9c9b74d6f5f1f181aa",
    "receiver/receiver_plan.h": "b49f5a90f1070cf2a8966d45828817e9a265abf4eef3e72fe9ff6319ffdd0fca",
    "receiver/replay_pusch.cpp": "d6719fe8081415587a819f0c6b2c83b27e3933053ba68bb8a48303168dc86ec2",
}
# The two cores that this sprint amended, by the requirement of 2026-11-21: the hash counts its
# own work, and a grid and a decode can be owned by the ticket of a fill. Nothing else may change
# them: an edit of one of these files changes the executables that the accepted gates ran.
AMENDED_SOURCES = {
    "replay/replay_core.h": "dcd166b66e08d317f2275dcf5ffdf8dbdb53f936e98d653177703f310857e325",
    "replay/replay_core.cpp": "fa2f450b2f5ec35fd8dc02fb8e0b71ec6312034e7f890c76dbeefd5d7828f3c2",
    "receiver/dispatch_core.h": "604cb86e5abe1eea3902c0011a1f204c9ea5de648f429579d6ed66ba19af423b",
    "receiver/dispatch_core.cpp": "f51d74d8a2e5f510628bde8cfb3c23d4e39b88070e0f56057c74d884f5ababcd",
}


def small_header(**fields):
    base = obs.new_header(epoch=1, sequence=3, episode=0xE0, opportunity=0xA1, grant=7, absolute_slot=68, cell=1,
                          ue=0x4601, stage=2, source=1, layout=2, dtype=1, producer_steps=1, scale=1.4125375446227544,
                          rank=3, shape=[2, 2, 4], strides=[8, 4, 1], first_subcarrier=4, subcarrier_step=2,
                          subcarrier_count=4, dmrs_symbols=(1 << 2) | (1 << 11), rx_count=2, layer_count=1,
                          antenna_map=[5, 3], mask_bytes=2, payload_bytes=128, host_bytes=40, time_unit=1, digest_kind=0,
                          evidence=2, port=0, t_first_sample=145920, t_last_sample=842752, t_ready_ns=10, t_publish_ns=11,
                          valid=16)
    base.update(fields)
    return base


def small_plan(mode="timed", ring=True):
    """A plan of version 2 with one source. No file of it is read by check_plan."""
    bound = {"path": "/nowhere/file", "sha256": "a" * 64}
    return {"schema": sr.PLAN_SCHEMA, "run_id": "unit", "access": pl.ACCESS, "replay_plan": dict(bound),
            "observer": {"tap": "pilot_ls", "stage": "O2", "source": "pusch_dmrs", "layout": "ocudu_o2_comb_c64",
                         "epoch": 1},
            "sources": [{"grant_id": "g", "record_sha256": "b" * 64, "source": dict(bound), "pilot_index": dict(bound),
                         "pilot_ref": dict(bound), "antenna_map": [0, 1, 2, 3]}],
            "evidence": {"mode": mode}, "ring": sr.ring_spec(3, 3) if ring else None}


class RecordTests(unittest.TestCase):
    def test_the_definition_is_the_one_both_readers_were_built_from(self):
        self.assertEqual(f"{obs.DEFINITION:016x}", DEFINITION)
        self.assertEqual((obs.HEADER.size, obs.CONTROL.size, obs.STAGED_HOST.size), (256, 64, 40))
        self.assertTrue(obs.RECORD_DEFINITION.startswith("obs/v1 little-endian 256;schema:0:4;header_bytes:4:4;epoch:8:8;"))
        self.assertTrue(obs.RECORD_DEFINITION.endswith("digest:216:32;valid:248:4;reserved:252:4;"))
        self.assertIn("published:128:8;", obs.RING_DEFINITION)
        self.assertIn("cursor 4096;released:0:8;waiting:64:4;", obs.RING_DEFINITION)
        self.assertEqual((obs.fnv1a(""), obs.fnv1a("a")), (14695981039346656037, 0xAF63DC4C8601EC8C))
        # The byte-level header of the native library names the same fields at the same offsets.
        header = (RING / "include/obs/record.h").read_text()
        text = "".join(re.findall(r'^\s+"([^"]*)"\s*;?$', header[header.index("RECORD_DEFINITION[]"):], re.MULTILINE))
        self.assertEqual(text, obs.RECORD_DEFINITION)

    def test_a_header_packs_into_256_bytes_and_comes_back(self):
        header = small_header(digest=bytes(32))
        data = obs.pack_header(header)
        self.assertEqual(len(data), 256)
        self.assertEqual(struct.unpack_from("<IIQQ", data), (1, 256, 1, 3))
        self.assertEqual(struct.unpack_from("<Q", data, 48), (68,))
        self.assertEqual(struct.unpack_from("<4B", data, 64), (2, 1, 2, 1))
        self.assertEqual(struct.unpack_from("<d", data, 72), (1.4125375446227544,))
        self.assertEqual(struct.unpack_from("<5I", data, 80), (3, 2, 2, 4, 0))
        self.assertEqual(struct.unpack_from("<2H", data, 136), (5, 3))
        self.assertEqual(struct.unpack_from("<3I4B", data, 168), (2, 128, 40, 1, 0, 2, 0))
        self.assertEqual(struct.unpack_from("<qq", data, 200), (10, 11))
        self.assertEqual(struct.unpack_from("<II", data, 248), (16, 0))
        back = obs.unpack_header(data)
        self.assertEqual({key: back[key] for key in ("shape", "strides", "antenna_map")},
                         {"shape": [2, 2, 4, 0], "strides": [8, 4, 1, 0], "antenna_map": [5, 3] + [0] * 14})
        self.assertEqual(obs.pack_header(back), data)
        for bad in (dict(header, schema=2), dict(header, header_bytes=128)):
            with self.assertRaisesRegex(ObsError, "not an obs/v1 record"):
                obs.unpack_header(obs.pack_header(bad))
        with self.assertRaisesRegex(ObsError, "256 bytes"):
            obs.unpack_header(data[:200])

    def test_a_record_is_its_parts_without_padding(self):
        header = small_header()
        host = obs.STAGED_HOST.pack(1, 17, 1, 0, 4, 7, 0, 9)
        record = obs.Record(header, host, b"\xff\x7f", bytes(range(128)))
        data = record.to_bytes()
        self.assertEqual(len(data), 256 + 40 + 2 + 128)
        again = obs.read_record(data)
        self.assertEqual((again.header, again.host, again.mask, again.payload), (obs.unpack_header(data), host, b"\xff\x7f",
                                                                                 bytes(range(128))))
        self.assertEqual((again.shape, again.strides, again.sequence), ([2, 2, 4], [8, 4, 1], 3))
        self.assertTrue(again.is_valid(0) and again.is_valid(14) and not again.is_valid(15))
        self.assertEqual(obs.staged_host(again.host),
                         {"kind": 1, "opportunity_sequence": 17, "grid_index": 1, "sfn": 0, "grid_generation": 4, "slot": 7,
                          "reserved": 0, "t_read_ns": 9})
        described = again.described()
        self.assertEqual(described["what"], {"stage": "O2", "source": "pusch_dmrs", "port": 0, "layout": "ocudu_o2_comb_c64",
                                             "dtype": "<c8", "producer_steps": ["pilot_ls"], "scale": 1.4125375446227544,
                                             "evidence": "timed"})
        self.assertEqual(described["allocation"]["dmrs_symbols"], [2, 11])
        self.assertEqual((described["payload"]["digest_kind"], described["payload"]["digest"]), (None, None))
        # Nothing in a record is truth: no field for a channel, a noise, a transport block or a label.
        for word in ("truth", "label", "target", "tb_", "noise"):
            self.assertNotIn(word, obs.RECORD_DEFINITION)
        with self.assertRaisesRegex(ObsError, "its header names"):
            obs.read_record(data[:-1])
        with self.assertRaisesRegex(ObsError, "sizes its header names"):
            obs.Record(header, host, b"\xff", bytes(128))
        with self.assertRaisesRegex(ObsError, "not of the staged harness"):
            obs.staged_host(obs.STAGED_HOST.pack(2, 0, 0, 0, 0, 0, 0, 0))

    def test_element_order_of_the_layouts(self):
        # Element [1][3][2][5] of an ocudu_o3_fp16 payload of the demonstration.
        self.assertEqual(obs.element_coordinates("ocudu_o3_fp16", [2, 51, 4, 12], [1, 3, 2, 5]),
                         {"dmrs_symbol": 1, "rx_port": 2, "subcarrier": 41, "element": 1 * 2448 + 3 * 48 + 2 * 12 + 5})
        self.assertEqual(obs.element_coordinates("ocudu_o2_comb_c64", [4, 2, 306], [3, 1, 305]),
                         {"rx_plane": 3, "reference_symbol": 1, "pilot": 305, "element": 2447})
        self.assertEqual(obs.element_coordinates("ocudu_o1_cbf16", [4, 14, 612], [0, 2, 180])["element"], 2 * 612 + 180)
        with self.assertRaisesRegex(ObsError, "outside a payload"):
            obs.element_coordinates("ocudu_o3_fp16", [2, 51, 4, 12], [1, 3, 4, 5])
        with self.assertRaisesRegex(ObsError, "no element order"):
            obs.element_coordinates("aerial_hest_c64", [1, 2, 3], [0, 0, 0])

    def test_depth_follows_the_window(self):
        # The records a window holds in place, plus one uplink burst, plus one, as a power of two.
        self.assertEqual([obs.slots_for(*pair) for pair in ((3, 3), (1, 1), (0, 0), (3, 4), (4, 4), (10, 3))],
                         [8, 4, 1, 8, 16, 16])
        with self.assertRaisesRegex(ObsError, "non-negative"):
            obs.slots_for(-1, 3)
        with self.assertRaisesRegex(ObsError, "65536"):
            obs.slots_for(60000, 60000)
        self.assertEqual(sr.ring_spec(3, 3)["slots"], 8)
        self.assertEqual((obs.percentile([], 0.5), obs.percentile([3, 1, 2], 0.5), obs.percentile(range(1000), 0.999)),
                         (None, 2, 998))


class PlanTests(unittest.TestCase):
    def test_a_plan_of_version_two_names_its_mode_and_its_ring(self):
        plan = sr.check_plan(small_plan())
        self.assertEqual((plan["evidence"], plan["ring"]["slots"], plan["ring"]["wait"], plan["ring"]["attach"]),
                         ({"mode": "timed"}, 8, "spin", "required"))
        self.assertEqual(sr.check_plan(small_plan("audit", ring=False))["ring"], None)
        self.assertEqual(sr.check_plan(small_plan("audit"))["evidence"], {"mode": "audit"})
        self.assertEqual(str(sr.socket_path("/run/native", plan)), "/run/native/ring/o2.sock")

    def test_every_rule_of_the_plan_is_refused_with_its_reason(self):
        untapped = lambda plan: (plan["observer"].update(tap="none", stage=None, source=None, layout=None),  # noqa: E731
                                 plan.update(sources=[]))
        cases = (
            (lambda plan: plan.update(schema="pilot-ls-plan/v1"), "unsupported plan schema"),
            (lambda plan: plan.pop("evidence"), "requires exactly"),
            (lambda plan: plan["evidence"].update(mode="fast"), "evidence mode"),
            (lambda plan: plan["evidence"].update(extra=1), "evidence mode"),
            (lambda plan: plan.update(ring=None), "names a tap and no ring"),
            (untapped, "names a ring and no tap"),
            (lambda plan: plan["ring"].update(slots=16), "needs 8"),
            (lambda plan: plan["ring"].update(window_records=5), "needs 16"),
            (lambda plan: plan["ring"].update(window_records=-1), "integer from 0"),
            (lambda plan: plan["ring"].update(wait="sleep"), "ring.wait"),
            (lambda plan: plan["ring"].update(attach="maybe"), "ring.attach"),
            (lambda plan: plan["ring"].update(stream="../o2"), "ring.stream"),
            (lambda plan: plan["ring"].pop("drain_ms"), "ring requires exactly"),
            (lambda plan: plan["observer"].update(epoch=0), "epoch is at least 1"),
        )
        for edit, message in cases:
            plan = small_plan()
            edit(plan)
            with self.assertRaisesRegex(SnapshotRingError, message):
                sr.check_plan(plan)
        # A timed run without a tap needs no ring: it is the control of the timed harness.
        plan = small_plan(ring=False)
        untapped(plan)
        self.assertEqual(sr.check_plan(plan)["observer"]["tap"], "none")

    def test_an_account_line_is_checked_on_its_own(self):
        line = {"schema": sr.PUBLICATION_SCHEMA, "run_id": "r", "mode": "timed", "opportunity_id": "op.0",
                "opportunity_sequence": 0, "outcome": "published", "reason": None, "epoch": 1, "sequence": 0, "depth": 0,
                "bytes": 400, "keys": {"episode": "0" * 16, "opportunity": hashlib.sha256(b"op.0").hexdigest()[:16],
                                       "grant": "1" * 16},
                "absolute_slot": 8, "grid": {"grid_index": 0, "generation": 1}, "payload_bytes": 128, "valid": 16,
                "holes": 0, "t_read_ns": 10, "t_ready_ns": 20, "t_publish_ns": 30, "t_done_ns": 40, "service_us": 0.03,
                "estimate_us": 0.01, "publish_us": 0.02, "window_calls": 0}
        self.assertEqual(sr.publication_violations(line), [])
        dropped = dict(line, outcome="dropped", reason="no_credit", epoch=None, sequence=None, depth=None, bytes=None,
                       grid=None, valid=None, holes=None, t_ready_ns=None, t_publish_ns=None, estimate_us=None)
        self.assertEqual(sr.publication_violations(dropped), [])
        for edit, words in ((dict(line, t_publish_ns=15), "not in order"), (dict(line, reason="no_credit"), "names its epoch"),
                            (dict(dropped, reason=None), "names its reason"), (dict(dropped, sequence=4), "names its reason"),
                            (dict(line, outcome="lost"), "unknown outcome"), (dict(line, window_calls=-1), "not negative"),
                            (dict(line, keys=dict(line["keys"], opportunity="0" * 16)), "opportunity key"),
                            ({key: value for key, value in line.items() if key != "depth"}, "exactly the fields")):
            self.assertIn(words, " ".join(sr.publication_violations(edit)), words)


class NativeRingTests(unittest.TestCase):
    """The ring, the producer glue and the cores it joins, built without OCUDU, and both readers on one producer."""

    @classmethod
    def setUpClass(cls):
        cls.missing = [tool for tool in ("cmake", "ctest") if not shutil.which(tool)]
        if not any(shutil.which(compiler) for compiler in ("c++", "g++", "clang++")):
            cls.missing.append("a C++17 compiler")
        if cls.missing:
            return
        cls.temporary = tempfile.TemporaryDirectory()
        cls.build = Path(cls.temporary.name) / "build"
        steps = (["cmake", "-S", str(TRANSPORT), "-B", str(cls.build)], ["cmake", "--build", str(cls.build), "-j2"])
        cls.log, cls.status = "", 0
        for step in steps:
            result = subprocess.run(step, capture_output=True, text=True, timeout=1800, check=False)
            cls.log += result.stdout + result.stderr
            cls.status = cls.status or result.returncode
        cls.ring = cls.build / "obs_ring"
        cls.feed, cls.tail = cls.ring / "obs_feed", cls.ring / "obs_tail"

    @classmethod
    def tearDownClass(cls):
        if not cls.missing:
            cls.temporary.cleanup()

    def setUp(self):
        if self.missing:
            self.skipTest("the native ring needs " + ", ".join(self.missing))
        if self.status:
            self.fail("the native ring does not build:\n" + self.log[-3000:])
        scratch = tempfile.TemporaryDirectory(dir=self.temporary.name)
        self.addCleanup(scratch.cleanup)
        self.scratch = Path(scratch.name)

    def start_feed(self, socket, *options):
        return subprocess.Popen([str(self.feed), "--socket", str(socket), *options], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)

    def finish(self, process):
        said, errors = process.communicate(timeout=120)
        return process.returncode, (json.loads(said) if said.strip() else None), errors

    def test_ring_builds_without_a_warning_and_links_no_receiver(self):
        self.assertNotRegex(self.log, r"warning:")
        link = (self.ring / "CMakeFiles/obs_ring_test.dir/link.txt").read_text()
        self.assertEqual(re.findall(r"\S+\.(?:a|so)\b", link), ["libobs_ring.a"])
        for name in ("obs_tail", "obs_feed"):
            tool = (self.ring / f"CMakeFiles/{name}.dir/link.txt").read_text()
            self.assertEqual(re.findall(r"\S+\.(?:a|so)\b", tool), ["libobs_ring.a"], name)
        glue = (self.build / "CMakeFiles/obs_publisher_test.dir/link.txt").read_text()
        self.assertEqual([Path(name).name for name in re.findall(r"\S+\.(?:a|so)\b", glue)],
                         ["libobs_publisher.a", "libobserver_core.a", "libreplay_core.a", "libobs_ring.a"])
        self.assertNotIn("ocudu", (link + glue).lower().replace(str(ROOT).lower(), "").replace("ocudu_adapter", ""))
        # The ring includes its own two headers, the standard library and POSIX. Nothing of a receiver or a GPU.
        allowed = ("obs/record.h", "obs/ring.h", "sys/eventfd.h", "sys/mman.h", "sys/socket.h", "sys/stat.h", "sys/un.h")
        for path in sorted(RING.rglob("*")):
            if path.suffix in (".h", ".cpp"):
                for include in re.findall(r'#include [<"]([^>"]+)[>"]', path.read_text()):
                    self.assertTrue(include in allowed or "/" not in include, f"{path.name} includes {include}")
        for name in ("obs_publisher.h", "obs_publisher.cpp"):
            for include in re.findall(r'#include [<"]([^>"]+)[>"]', (TRANSPORT / name).read_text()):
                self.assertIn(include, ("obs/ring.h", "obs_publisher.h", "observer_core.h", "replay_core.h", "cstddef",
                                        "cstdint", "cstring", "string", "vector"))

    def test_every_native_unit_case_passes(self):
        for tool, cases in ((self.ring / "obs_ring_test", RING_CASES), (self.build / "obs_publisher_test", PUBLISHER_CASES)):
            listed = subprocess.run([str(tool), "--list"], capture_output=True, text=True, check=True).stdout.split()
            self.assertEqual(tuple(listed), cases)
        result = subprocess.run(["ctest", "--output-on-failure"], cwd=self.build, capture_output=True, text=True,
                                timeout=900, check=False)
        self.assertEqual(result.returncode, 0, result.stdout[-3000:])
        total = len(RING_CASES) + len(PUBLISHER_CASES) + 2
        self.assertIn(f"100% tests passed, 0 tests failed out of {total}", result.stdout)
        for name in RING_CASES:
            self.assertRegex(result.stdout, rf"obs_ring_{name} \.+\s+Passed")
        for name in PUBLISHER_CASES:
            self.assertRegex(result.stdout, rf"obs_publisher_{name} \.+\s+Passed")
        unknown = subprocess.run([str(self.ring / "obs_ring_test"), "no_such_case"], capture_output=True, text=True,
                                 check=False)
        self.assertEqual(unknown.returncode, 2)

    def test_both_readers_hash_the_same_definition(self):
        native = subprocess.run([str(self.ring / "obs_ring_test"), "--definition"], capture_output=True, text=True,
                                check=True).stdout.splitlines()
        self.assertEqual(native, [f"{obs.DEFINITION:016x}", obs.RECORD_DEFINITION, obs.RING_DEFINITION])
        self.assertEqual(native[0], DEFINITION)

    def test_the_reference_reader_reads_the_known_records_in_order(self):
        socket = self.scratch / "o2.sock"
        feed = self.start_feed(socket, "--records", "24", "--period-us", "1000")
        seen = []
        with obs.Consumer(socket, "wait", connect_timeout=30) as reader:
            self.assertEqual((reader.epoch, reader.ring["slot_count"], reader.ring["payload_capacity"]), (1, 8, 19584))
            self.assertEqual((reader.ring["payload_offset"] % 64, reader.ring["slot_bytes"] % 64), (0, 0))
            # The mapping is read-only: this process cannot alter what the producer published.
            with self.assertRaises(TypeError):
                reader._segment[0:1] = b"x"
            while True:
                try:
                    record = reader.next(2.0)
                except obs.RingClosed:
                    break
                self.assertIsNotNone(record)
                seen.append(record)
                reader.release()
            counts = reader.counts()
        status, report, errors = self.finish(feed)
        self.assertEqual((status, errors), (0, ""))
        self.assertEqual([record.sequence for record in seen], list(range(24)))
        self.assertEqual(counts, {"read": 24, "stale": 0, "released": 24, "skipped": 0, "sleeps": counts["sleeps"]})
        for number, record in enumerate(seen):
            header = record.header
            self.assertEqual((header["epoch"], header["opportunity"], header["episode"], header["grant"], header["cell"],
                              header["ue"], header["absolute_slot"]),
                             (1, number + 1, 0x0B5E0B5E0B5E0B5E, 7, 1, 0x4601, 20 * number + 8))
            self.assertEqual((record.shape, record.strides, header["valid"], header["evidence"], header["digest_kind"]),
                             ([4, 2, 306], [612, 306, 1], 2448, 2, 0))
            self.assertEqual(record.payload, bytes((31 * number + 7 * i) & 255 for i in range(19584)))
            self.assertEqual((record.mask, record.host), (b"\xff" * 306, struct.pack("<Q", number)))
            self.assertGreaterEqual(header["t_publish_ns"], header["t_ready_ns"])
        # Every record is accounted for by the producer, which made no system call in a publication.
        self.assertEqual({key: report[key] for key in ("admitted", "published", "dropped", "released", "reclaimed",
                                                       "pending", "faults", "epochs", "window_calls")},
                         {"admitted": 24, "published": 24, "dropped": 0, "released": 24, "reclaimed": 0, "pending": 0,
                          "faults": 0, "epochs": 1, "window_calls": 0})
        self.assertGreaterEqual(report["wakes"], 1)
        self.assertEqual(report["definition"], DEFINITION)

    def test_both_readers_keep_the_same_records(self):
        archives = {}
        for name in ("reference", "native"):
            socket, kept = self.scratch / f"{name}.sock", self.scratch / name
            kept.mkdir()
            feed = self.start_feed(socket, "--records", "12", "--period-us", "1000", "--payload-bytes", "1024")
            if name == "native":
                done = subprocess.run([str(self.tail), "--socket", str(socket), "--output-dir", str(kept), "--wait", "spin",
                                       "--capacity", "64"], capture_output=True, text=True, timeout=120, check=False)
                self.assertEqual(done.returncode, 0, done.stderr)
                self.assertIn("12 records read, 0 stale, 12 released, ended closed", done.stdout)
            else:
                report = obs.tail(socket, kept, "spin", connect_timeout=30)
                self.assertEqual((report["counts"]["read"], report["ended"], report["reader"]), (12, "closed", "reference"))
            self.assertEqual(self.finish(feed)[0], 0)
            archives[name] = obs.open_archive(kept)
        varying = ("t_ready_ns", "t_publish_ns")
        for mine, theirs in zip(archives["reference"].records, archives["native"].records):
            self.assertEqual({key: value for key, value in mine.header.items() if key not in varying},
                             {key: value for key, value in theirs.header.items() if key not in varying})
            self.assertEqual((mine.payload, mine.mask, mine.host), (theirs.payload, theirs.mask, theirs.host))
        self.assertEqual(len(archives["native"].records), 12)
        self.assertEqual(archives["native"].report["reader"], "native")
        # An archive is verified against its report: a changed record and a stray file are found.
        kept = self.scratch / "native"
        first = kept / obs.record_name(1, 0)
        data = first.read_bytes()
        first.write_bytes(data[:16] + struct.pack("<Q", 5) + data[24:])
        with self.assertRaisesRegex(ObsError, "differs from its line"):
            obs.open_archive(kept)
        first.write_bytes(data)
        (kept / "stray.bin").write_bytes(b"x")
        with self.assertRaisesRegex(ObsError, "does not list"):
            obs.open_archive(kept)

    def test_a_reader_of_another_definition_is_refused_with_both_hashes(self):
        socket = self.scratch / "o2.sock"
        feed = self.start_feed(socket, "--records", "2", "--attach-ms", "3000")
        with self.assertRaisesRegex(ObsError, f"{obs.DEFINITION ^ 1:016x}, and the producer from {DEFINITION}"):
            obs.Consumer(socket, "wait", connect_timeout=20, definition=obs.DEFINITION ^ 1)
        status, _, errors = self.finish(feed)
        self.assertEqual(status, 1)
        self.assertIn("no consumer attached within 3000 ms", errors)
        with self.assertRaisesRegex(ObsError, "no producer"):
            obs.Consumer(self.scratch / "nobody.sock", "spin", connect_timeout=0.05)

    def test_a_killed_consumer_is_replaced_in_a_new_epoch_and_every_record_is_accounted_for(self):
        socket, kept = self.scratch / "o2.sock", self.scratch / "second"
        kept.mkdir()
        first = subprocess.Popen([str(self.tail), "--socket", str(socket), "--exit-after", "4"], stderr=subprocess.PIPE,
                                 text=True)
        second = subprocess.Popen([str(self.tail), "--socket", str(socket), "--output-dir", str(kept), "--after-pid",
                                   str(first.pid), "--capacity", "256", "--record-bytes", "2048"],
                                  stdout=subprocess.PIPE, text=True)
        feed = self.start_feed(socket, "--records", "150", "--period-us", "1000", "--payload-bytes", "1024")
        self.assertEqual(first.wait(timeout=120), 9)  # ended without BYE, holding one credit
        self.assertIn("ending without BYE after 4 records", first.stderr.read())
        first.stderr.close()
        status, report, _ = self.finish(feed)
        self.assertEqual((status, second.wait(timeout=120)), (0, 0))
        second.stdout.close()
        self.assertEqual((report["epochs"], report["faults"], report["pending"], report["window_calls"]), (2, 0, 0, 0))
        self.assertGreaterEqual(report["reclaimed"], 1)
        self.assertEqual(report["published"], report["released"] + report["reclaimed"] + report["pending"])
        self.assertEqual(report["published"] + report["dropped"], 150)
        archive = obs.open_archive(kept)
        self.assertEqual(archive.report["epoch"], 2)
        self.assertEqual([record.sequence for record in archive.records], list(range(len(archive.records))))
        self.assertEqual(len(archive.records), report["released"] - 3)

    def test_a_slow_consumer_makes_the_producer_drop_and_never_wait(self):
        socket = self.scratch / "o2.sock"
        slow = subprocess.Popen([str(self.tail), "--socket", str(socket), "--hold-us", "4000"], stdout=subprocess.PIPE,
                                text=True)
        feed = self.start_feed(socket, "--records", "120", "--period-us", "500", "--payload-bytes", "1024", "--slots", "4")
        status, report, _ = self.finish(feed)
        self.assertEqual((status, slow.wait(timeout=120)), (0, 0))
        slow.stdout.close()
        self.assertGreater(report["dropped"], 0)
        self.assertEqual((report["max_depth"], report["faults"], report["window_calls"], report["pending"]), (3, 0, 0, 0))
        self.assertEqual(report["published"] + report["dropped"], 120)
        self.assertEqual(report["published"], report["released"])
        # 120 records at 500 us: the producer kept its pace although the consumer needs 4 ms per record.
        self.assertLess(report["fill_and_publish_ns"]["max"], 2_000_000)


class TreeTests(unittest.TestCase):
    FILES = ("native/obs_ring/CMakeLists.txt", "native/obs_ring/README.md", "native/obs_ring/include/obs/record.h",
             "native/obs_ring/include/obs/ring.h", "native/obs_ring/src/ring.cpp", "native/obs_ring/tests/obs_ring_test.cpp",
             "native/obs_ring/tools/obs_tail.cpp", "native/obs_ring/tools/obs_feed.cpp",
             "native/ocudu_adapter/transport/CMakeLists.txt", "native/ocudu_adapter/transport/obs_publisher.h",
             "native/ocudu_adapter/transport/obs_publisher.cpp",
             "native/ocudu_adapter/transport/tests/obs_publisher_test.cpp",
             "native/ocudu_adapter/transport/tests/alloc_count.cpp", "native/ocudu_adapter/observer/pilot_ls.cpp",
             "native/ocudu_adapter/observer/observer_core.h", "native/ocudu_adapter/observer/observer_core.cpp",
             "native/ocudu_adapter/observer/observer_plan.h", "native/ocudu_adapter/replay/replay_core.h",
             "native/ocudu_adapter/replay/replay_core.cpp", "native/ocudu_adapter/receiver/dispatch_core.h",
             "native/ocudu_adapter/receiver/dispatch_core.cpp", "native/ocudu_adapter/README.md",
             "src/ocudu_cuda_isac/sensing/obs.py", "src/ocudu_cuda_isac/episodes/snapshot_ring.py",
             "tests/unit/test_snapshot_ring.py", "tests/unit/snapshot_ring_double.py",
             "tests/unit/test_snapshot_ring_driver.py", "tests/integration/snapshot_ring.py",
             "configs/integration/snapshot_ring.json", "scripts/accept_s3_2.sh", "tests/test_accept_s3_2.py",
             "docs/interfaces/OBSERVATION.md", "docs/interfaces/SNAPSHOT_RING.md", "Makefile")

    def test_files_are_ascii_and_do_not_name_the_project(self):
        # The working name lives on line 1 of the top-level README and nowhere else.
        name = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()[0].lstrip("# ").split(":")[0].strip()
        for relative in self.FILES:
            data = (ROOT / relative).read_bytes()
            self.assertTrue(data.isascii(), relative)
            if name and name.lower() != "ocudu-cuda-isac":
                self.assertNotIn(name.lower(), data.decode("ascii").lower(), relative)

    def test_the_accepted_native_sources_are_those_of_sprint_2_but_for_the_two_amended_cores(self):
        for relative, digest in {**UNEDITED_SOURCES, **AMENDED_SOURCES}.items():
            self.assertEqual(hashlib.sha256((ADAPTER / relative).read_bytes()).hexdigest(), digest, relative)
        # The amendment keeps every default: a caller that names no mode runs the accepted code.
        core = (ADAPTER / "replay/replay_core.h").read_text()
        self.assertIn("void commit(const ticket& key, o1_descriptor descriptor, evidence mode = evidence::audit);", core)
        self.assertIn("hash_count sha256_count();", core)
        ledger = (ADAPTER / "receiver/dispatch_core.h").read_text()
        self.assertIn("isac_replay::evidence mode_ = isac_replay::evidence::audit);", ledger)
        for name in ("receiver/replay_pusch.cpp", "replay/grid_apply.cpp"):
            self.assertNotIn("evidence::", (ADAPTER / name).read_text(), name)

    def test_the_timed_loop_of_the_tool_takes_no_digest_writes_no_file_and_builds_no_record(self):
        tool = (ADAPTER / "observer/pilot_ls.cpp").read_text()
        start = tool.index("    for (std::size_t n = 0; n != chosen.size(); ++n) {\n      const std::size_t       sequence = chosen[n];")
        loop = tool[start:tool.index("  // ---- The drain:")]
        for word in ("sha256_hex", "sha256(", "sha256_count", "digest()", "write_bytes", "json", ".dump(", "read_back(",
                     "std::cout", "make_directory", "watcher.observe("):
            self.assertNotIn(word, loop, word)
        # A grid becomes ready without a readback digest, the tap is the publication, and the grid
        # is read in full only after the last decoder callback on it.
        commit = loop.index("pool.commit(key, replay.describe(sequence, bound, sink), isac_replay::evidence::timed);")
        publish = loop.index("out->publish(handle_of[sequence], ready);")
        decode = loop.index("dispatch(book, backend, names[n * backends.size() + b], key")
        copy_out = loop.index("if (keep_grids) {")
        self.assertLess(commit, publish)
        self.assertLess(publish, decode)
        self.assertLess(decode, copy_out)
        self.assertLess(copy_out, loop.index("ring->service();"))
        # The hash count is taken before the first fill and again after the drain, and the audited
        # loop, which Sprint 3.1 accepted, does not run in a timed run.
        self.assertLess(tool.index("hash_bound = isac_replay::sha256_count();"), tool.index("replay.deliver(sequence, pool, key)"))
        self.assertLess(tool.index("ring->drain("), tool.index("hash_drained = isac_replay::sha256_count();"))
        self.assertIn("for (std::size_t sequence = 0; !timed && sequence != replay.opportunities(); ++sequence) {", tool)
        self.assertIn("ledger      book(pool, mode);", tool)
        # The producer never names a system call or a hash between its admission and its publication.
        ring = (RING / "src/ring.cpp").read_text()
        window = ring[ring.index("bool producer::try_acquire(slot& out) noexcept"):ring.index("total.window_calls += ")]
        for word in ("sys::", "throw", "new ", "std::string", "mutex", "lock"):
            self.assertNotIn(word, window.replace("sys::calls()", ""), word)
        glue = (TRANSPORT / "obs_publisher.cpp").read_text()
        timed = glue[glue.index("const published_record& publisher::publish(std::size_t handle, const isac_replay::o1_view& view)"):
                     glue.index("namespace {\n\nstd::uint8_t nibble")]
        for word in ("sha256", "digest", "std::string", "std::vector", "new ", "to_string"):
            self.assertNotIn(word, timed, word)
        core = (ADAPTER / "observer/observer_core.cpp").read_text()
        timed = core[core.index("tap_result observer::observe_into("):core.index("const observation& observer::observe(")]
        for word in ("sha256_hex", "sha256(", "digest()", "opaque_key(", "std::vector", "push_back"):
            self.assertNotIn(word, timed, word)

    def test_the_build_the_gate_and_the_configuration_name_the_ring(self):
        adapter = (ADAPTER / "CMakeLists.txt").read_text()
        self.assertIn('add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../obs_ring" obs_ring)', adapter)
        self.assertLess(adapter.index("add_subdirectory(observer)"), adapter.index("add_subdirectory(transport)"))
        self.assertIn("target_link_libraries(pilot_ls PRIVATE obs_publisher observer_core dispatch_core replay_core "
                      "ocudu_pin_fixture)", (ADAPTER / "observer/CMakeLists.txt").read_text())
        ring = (RING / "CMakeLists.txt").read_text()
        self.assertEqual(tuple(re.search(r"set\(OBS_RING_CASES ([^)]*)\)", ring)[1].split()), RING_CASES)
        glue = (TRANSPORT / "CMakeLists.txt").read_text()
        self.assertEqual(tuple(re.search(r"set\(OBS_PUBLISHER_CASES ([^)]*)\)", glue)[1].split()), PUBLISHER_CASES)
        builder = (ROOT / "scripts/native_adapter.py").read_text()
        self.assertIn('"obs_tail": Path("obs_ring/obs_tail")', builder)
        self.assertIn("for tree in (ADAPTER, RING)", builder)
        config = json.loads((ROOT / "configs/integration/snapshot_ring.json").read_text())
        self.assertEqual((config["schema"], config["selector"], config["backends"]),
                         ("integration-config/snapshot_ring/v1", "snapshot_ring", ["cpu", "cuda"]))
        # The ring of the demonstration: three records in a window, three in a burst, eight slots.
        ring = config["ring"]
        self.assertEqual((ring["window_records"], ring["burst_records"], ring["slots"]), (3, 3, 8))
        self.assertEqual(ring["slots"], obs.slots_for(ring["window_records"], ring["burst_records"]))
        self.assertEqual(config["tap"], {"ceiling_us": sr.TAP_CEILING_US, "explain_us": sr.TAP_EXPLAIN_US,
                                         "expected_us": 10.0})
        # The pair is the one the Sprint 3.1 gate tapped, with the same count.
        tapped = json.loads((ROOT / "configs/integration/pilot_ls.json").read_text())
        self.assertEqual(config["pair"]["expect"]["opportunities"], tapped["pair"]["expect"]["opportunities"])
        self.assertEqual(config["pair"]["expect"]["payload_bytes"], tapped["pair"]["expect"]["payload_bytes"])
        self.assertEqual(config["observer"], tapped["observer"])
        self.assertIn("accept-s3.2:", (ROOT / "Makefile").read_text())
        # The strings of a plan and of a report are the same in the tool and in the reader.
        tool = (ADAPTER / "observer/pilot_ls.cpp").read_text() + (ADAPTER / "observer/observer_plan.h").read_text()
        for text in (sr.PLAN_SCHEMA, sr.NATIVE_SCHEMA, sr.PUBLICATION_SCHEMA, sr.RING_SCHEMA):
            self.assertIn(f'"{text}"', tool)
        self.assertEqual(copy.deepcopy(sr.PUBLISHED_RING_US)["graded"], False)


if __name__ == "__main__":
    unittest.main()
