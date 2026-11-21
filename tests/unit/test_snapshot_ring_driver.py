"""Check the Sprint 3.2 integration driver against a double of the native tool, its ring and its consumers.

The control is a Sprint 3.1 run that the driver of that sprint grades on the doubles of its own
driver tests, on top of a Sprint 2.4 run that is graded the same way: the small pair of two TDD
periods, six opportunities per member, four receive ports. The double of this sprint
(tests/unit/snapshot_ring_double.py) decodes that pair again with the model receiver, computes
each observation with the reference and models the ring and the consumers, so every check of the
driver has a known answer, and each injected fault must fail the check that names it. NumPy is
needed, as for the drivers this one is stacked on.
"""

import contextlib
import hashlib
import importlib.util
import io
import json
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import snapshot_ring_double as double  # noqa: E402
import test_pilot_ls_driver as control_tests  # noqa: E402

CHECKS = 40
CASES = 10
ENVIRONMENT = ("SNAPSHOT_RING_FAULT", "PILOT_LS_FAULT", "PILOT_LS_BUILD", "REPLAY_PUSCH_FAULT", "REPLAY_PUSCH_BUILD")


def load_driver():
    spec = importlib.util.spec_from_file_location("snapshot_ring_driver", ROOT / "tests/integration/snapshot_ring.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(np is None, "the integration driver needs NumPy")
class DriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.driver = load_driver()
        # The accepted Sprint 3.1 run: the driver of that sprint on the doubles of its tools.
        cls.stage = control_tests.DriverTests
        cls.stage.setUpClass()
        cls.addClassCleanup(cls.stage.tearDownClass)
        assert cls.stage.reference["status"] == 0, cls.stage.reference["failed"]
        cls.control = cls.stage.reference["out"]
        cls.root = cls.stage.root / "s32"
        cls.root.mkdir()
        config = json.loads((ROOT / "configs/integration/snapshot_ring.json").read_text())
        # The small pair: 12 opportunities, on a ring of four slots, so that a slow consumer overloads it.
        config["pair"]["expect"]["opportunities"] = 12
        config["ring"].update(window_records=1, burst_records=1, slots=4)
        cls.config = cls.root / "config.json"
        cls.config.write_text(json.dumps(config))
        cls.tool = cls.root / "tool"
        cls.tool.write_text("a double\n")
        cls.reference = cls.run_once(cls.root / "reference")

    @classmethod
    def receipt_of(cls):
        return {"executables": {name: {"path": str(cls.tool), "sha256": hashlib.sha256(cls.tool.read_bytes()).hexdigest()}
                                for name in ("pilot_ls", "obs_tail", "obs_feed", "replay_pusch")},
                "ocudu_build": "double", "pin": {"cuda": True}}

    @classmethod
    def run_once(cls, out, fault=None, config=None, control=None, extra=(), receipt=None):
        driver = cls.driver
        saved = (driver.native_adapter.verified_receipt, driver.run_scenario)
        driver.native_adapter.verified_receipt = lambda build_dir: receipt or cls.receipt_of()
        driver.run_scenario = double.scenario
        kept = {key: os.environ.get(key) for key in ENVIRONMENT}
        for key in kept:
            os.environ.pop(key, None)
        if fault:
            os.environ["SNAPSHOT_RING_FAULT"] = fault
        arguments = ["--config", str(config or cls.config), "--s31-run", str(control or cls.control),
                     "--output-dir", str(out), *extra]
        try:
            with contextlib.redirect_stdout(io.StringIO()) as said:
                status = driver.main(arguments)
        finally:
            driver.native_adapter.verified_receipt, driver.run_scenario = saved
            for key, value in kept.items():
                os.environ.pop(key, None)
                if value is not None:
                    os.environ[key] = value
        grade = json.loads((out / "grade.json").read_text())
        failed = sorted(f"{item['case']} {item['check']}" for item in grade["results"] if not item["passed"])
        return {"status": status, "grade": grade, "failed": failed, "out": out, "said": said.getvalue(),
                "lines": (out / "grade.txt").read_text()}

    def setUp(self):
        scratch = tempfile.TemporaryDirectory(dir=self.root)
        self.addCleanup(scratch.cleanup)
        self.out = Path(scratch.name)

    def faulty(self, fault, **options):
        return self.run_once(self.out / fault, fault=fault, **options)

    # ---------------------------------------------------------------- the reference
    def test_the_reference_passes_every_check(self):
        run = self.reference
        grade, lines, out = run["grade"], run["lines"], run["out"]
        self.assertEqual((run["status"], run["failed"]), (0, []))
        self.assertEqual((grade["judgment"], grade["hardware_gate"], grade["cases"], grade["cases_passed"], grade["checks"]),
                         ("pass", "closed", CASES, CASES, CHECKS))
        self.assertIn(f"snapshot_ring: {CASES} of {CASES} cases passed, {CHECKS} of {CHECKS} checks\n", lines)
        for line in (
                "PASS  S3.2 control accepted run: the Sprint 3.1 driver graded reference as pass: 9 of 9 cases",
                "12 accepted observations; the replay plan it tapped has the hash it binds, and the Sprint 2.4 run beside "
                "that plan holds 24 radio records",
                # The corrected observer computes what the accepted one computed.
                "PASS  S3.2 audit accepted estimates: 12 of 12 payloads and validity masks have the SHA-256 that the "
                "accepted Sprint 3.1 run recorded for the same opportunity",
                "PASS  S3.2 audit account: 12 eligible observations: 12 published and 0 dropped; of the published 12 "
                "released, 0 reclaimed and 0 pending at the cutoff",
                "12 of 12 records in the consumer's archive hold the bytes of the payload file and of the mask file",
                "NOTE  S3.2 audit parts of one audited observation, not graded",
                "digest of the grid 1356.4 us (expected 1356 us on the Spark)",
                # The timed path.
                "PASS  S3.2 timed no checksum: between the first fill and the drain of the ring the hash of the core "
                "computed 0 digests, took 0 calls with data and hashed 0 bytes",
                "PASS  S3.2 timed no system call: the producer made 0 system calls between an admission and its "
                "publication store over 12 publications; the consumer attached in the mode spin, and the producer wrote "
                "its eventfd 0 times",
                "PASS  S3.2 timed decodes: 24 radio records equal those the accepted replay_pusch wrote without an "
                "observer",
                "PASS  S3.2 timed records: the native consumer kept 12 records of epoch 1, sequences 0 to 11 in order",
                "12 of 12 payloads and validity masks, hashed here after the run, have the SHA-256 of the audited "
                "estimate of the same opportunity",
                "12 of 12 published records reached the consumer, 12 of them without a digest in the record",
                "PASS  S3.2 timed tap cost: service time of the tap from the read of the grid to the publication store, "
                "over 11 observations of 19584 bytes, the first one excluded: mean 7.4 us, largest 7.4 us, at or below "
                "the ceiling of 150 us; the estimate takes 7.0 us in the mean",
                "NOTE  S3.2 timed expected before the publication: about 10 us, from 9.7 us measured on the Spark",
                "NOTE  S3.2 timed delay of the spinning consumer from the publication store to its read, not graded: "
                "median 0.4 us, 99th percentile 0.4 us, 99.9th percentile 0.4 us, largest 0.4 us",
                "publishes 68000 bytes in 1.97 us at the median, 2.34 us quiet against 167 us with a cell on the air",
                # Both wait modes and both readers.
                "PASS  S3.2 wait no system call: the producer made 0 system calls between an admission and its "
                "publication store over 12 publications; the consumer attached in the mode wait, and the producer wrote "
                "its eventfd 12 times, each after a publication store that found the waiting flag set",
                "NOTE  S3.2 wait delay of the waiting consumer from the publication store to its read, not graded: "
                "median 60.0 us",
                "PASS  S3.2 reference-reader records: the reference consumer kept 12 records of epoch 1",
                "PASS  S3.2 reference-reader both readers: 12 of 12 records have the header fields, the host block, the "
                "validity bits and the payload bytes that the native consumer kept",
                # Overload and restart.
                "PASS  S3.2 slow-consumer overload: 4 observations found no credit and were dropped at once and counted",
                "the ring held at most 4 of its 4 slots: nothing was overwritten and the producer never waited",
                "PASS  S3.2 slow-consumer receiver continues: 24 radio records equal",
                "PASS  S3.2 slow-consumer account: 12 eligible observations: 8 published and 4 dropped; of the published "
                "8 released, 0 reclaimed and 0 pending at the cutoff",
                "PASS  S3.2 terminated-consumer new epoch: 2 consumers attached; the records were published in the epochs "
                "[1, 2], the second from sequence 0 again; 3 records that the first consumer held or never read were "
                "reclaimed",
                "PASS  S3.2 terminated-consumer account: 12 eligible observations: 12 published and 0 dropped; of the "
                "published 9 released, 3 reclaimed and 0 pending at the cutoff; 0 faults, 2 epochs; the second consumer "
                "kept all 6 records of its epoch, and the first, which was killed, kept none",
                "PASS  S3.2 tap-off native run: the timed harness without the tap and without a ring: no observation",
                "PASS  S3.2 refusals plan check: 7 of 7 faulty plans are refused by the plan check",
                "PASS  S3.2 refusals native tool: 7 of 7 faulty plans are refused by the native tool with status 2",
                "PASS  S3.2 backends decode count: 7 of 7 runs finished; 168 radio records for 168 planned decodes",
                "PASS  S3.2 backends executed path: 84 of 84 decodes requested from the CUDA processor ran its resident "
                "path"):
            self.assertIn(line, lines)
        self.assertNotIn("rehearsal", run["said"])
        # The evidence.
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual((manifest["selector"], manifest["axes"]["transport"], manifest["host_rehearsal"]),
                         ("snapshot_ring", "obs-ring/v1 in host shared memory", False))
        self.assertEqual([item["name"] for item in manifest["runs"]],
                         ["audit", "timed", "wait", "reference-reader", "slow-consumer", "terminated-consumer", "tap-off"])
        self.assertEqual([item["mode"] for item in manifest["runs"]], ["audit"] + ["timed"] * 6)
        self.assertEqual([len(item["consumers"]) for item in manifest["runs"]], [1, 1, 1, 1, 1, 2, 0])
        self.assertEqual(manifest["runs"][5]["consumer_statuses"], [9, 0])
        self.assertEqual(manifest["runs"][3]["consumers"][0]["reader"], "reference")
        self.assertEqual(manifest["control"]["judgment"], "pass")
        comparison = json.loads((out / "comparison.json").read_text())
        self.assertEqual(comparison["timed"]["published_ring_us"]["graded"], False)
        self.assertEqual(comparison["timed"]["hash_in_window"], {"digests": 0, "updates": 0, "bytes": 0})
        self.assertGreater(comparison["audit"]["hash_in_window"]["digests"], 0)
        for name in ("grade.json", "grade.txt", "comparison.json", "manifest.json", "adapter-receipt.json",
                     "timed/run/plan.json", "timed/run/native/ring/publications.jsonl", "timed/run/native/ring/ring.json",
                     "timed/run/consumer-0/consumer.json", "audit/run/native/observer/o2.jsonl"):
            self.assertTrue((out / name).is_file(), name)
        self.assertFalse((out / "timed/run/native/observer").exists())
        plan = json.loads((out / "timed/run/plan.json").read_text())
        self.assertEqual((plan["schema"], plan["evidence"], plan["ring"]["slots"], plan["ring"]["wait"],
                          plan["ring"]["attach"]), ("pilot-ls-plan/v2", {"mode": "timed"}, 4, "spin", "required"))

    def test_the_readers_of_this_sprint_verify_the_outputs_of_the_reference(self):
        from ocudu_cuda_isac.episodes import snapshot_ring as sr
        from ocudu_cuda_isac.episodes.snapshot_ring import SnapshotRingError
        from ocudu_cuda_isac.sensing import obs
        out = self.reference["out"]
        timed = sr.open_output(out / "timed/run/native")
        self.assertEqual((timed.mode, timed.tap, len(timed.publications), len(timed.published), timed.observer),
                         ("timed", "pilot_ls", 12, 12, None))
        self.assertTrue(sr.account(timed)["balanced"])
        archive = obs.open_archive(out / "timed/run/consumer-0")
        pairs, problems = sr.join_archive(timed, archive)
        self.assertEqual((len(pairs), problems), (12, []))
        audit = sr.open_output(out / "audit/run/native")
        self.assertEqual((audit.mode, len(audit.observer.observations)), ("audit", 12))
        # A timed record carries no digest: the evaluator takes it, and it is the audited one.
        digests = sr.payload_digests(pairs)
        for record in audit.observer.observations:
            self.assertEqual(digests[record["host"]["opportunity_id"]],
                             (record["payload"]["digest"], record["validity"]["sha256"]))
        self.assertEqual(sr.summary(timed)["hash_in_window"], {"digests": 0, "updates": 0, "bytes": 0})
        slow = sr.account(sr.open_output(out / "slow_consumer/run/native"))
        self.assertEqual((slow["published"], slow["dropped"], slow["balanced"]), (8, 4, True))
        ended = sr.account(sr.open_output(out / "terminated_consumer/run/native"))
        self.assertEqual((ended["released"], ended["reclaimed"], ended["epochs"], ended["balanced"]), (9, 3, 2, True))
        # The reader refuses an output that was changed after the run.
        native = self.out / "native"
        import shutil
        shutil.copytree(out / "timed/run/native", native)
        ring = native / "ring"
        lines = (ring / "publications.jsonl").read_text().splitlines()
        cases = (
            (lambda: (ring / "publications.jsonl").write_text("\n".join(lines[:-1]) + "\n"), "lines for 12 filled grids"),
            (lambda: (ring / "publications.jsonl").write_text("\n".join([lines[1], lines[0]] + lines[2:]) + "\n"),
             "does not belong to opportunity"),
            (lambda: (ring / "stray.sock").write_text(""), "the producer removes its socket"),
            (lambda: (native / "observer").mkdir(), "a timed run holds observer/"),
            (lambda: (ring / "ring.json").write_text("{}"), "is not the ring of the report"),
        )
        for change, message in cases:
            shutil.rmtree(native)
            shutil.copytree(out / "timed/run/native", native)
            change()
            with self.assertRaisesRegex(SnapshotRingError, message):
                sr.open_output(native)

    # ---------------------------------------------------------------- faults
    def test_each_fault_fails_the_check_that_names_it(self):
        cases = (
            ("timed-hash", ["timed no checksum", "wait no checksum", "tap-off native run", "slow-consumer account"]),
            ("window-call", ["timed no system call", "wait no system call"]),
            ("timed-payload", ["timed records", "wait records", "reference-reader records"]),
            ("timed-digest", ["timed records", "timed account"]),
            ("lost-record", ["timed account", "wait account"]),
            ("reorder", ["timed native run", "wait native run"]),
            ("unbalanced", ["audit account", "timed account", "slow-consumer account", "terminated-consumer account"]),
            ("slow-tap", ["timed tap cost"]),
            ("changed-decode", ["audit decodes", "timed decodes", "slow-consumer receiver continues",
                                "terminated-consumer receiver continues"]),
            ("no-drop", ["slow-consumer overload"]),
            ("blocking", ["slow-consumer overload"]),
            ("same-epoch", ["terminated-consumer new epoch", "terminated-consumer records"]),
            ("host-fallback", ["backends executed path"]),
            ("accept-all", ["refusals native tool"]),
            ("exit", ["audit native run", "timed native run", "tap-off native run"]),
        )
        for fault, expected in cases:
            run = self.faulty(fault)
            with self.subTest(fault=fault):
                self.assertEqual(run["status"], 1)
                self.assertEqual(run["grade"]["judgment"], "fail")
                for check in expected:
                    self.assertIn(check, run["failed"])

    def test_single_faults_fail_only_their_own_checks(self):
        self.assertEqual(self.faulty("slow-tap")["failed"], ["timed tap cost"])
        self.assertEqual(self.faulty("no-drop")["failed"], ["slow-consumer overload"])
        self.assertEqual(self.faulty("accept-all")["failed"], ["refusals native tool"])
        run = self.faulty("blocking")
        self.assertEqual(run["failed"], ["slow-consumer overload"])
        self.assertIn("the longest in 5000.00 us", run["lines"])

    # ---------------------------------------------------------------- setup
    def test_a_rehearsal_needs_a_rehearsed_control_and_an_output_is_never_overwritten(self):
        run = self.run_once(self.out / "rehearsal", extra=["--host-rehearsal"])
        self.assertEqual(run["status"], 1)
        self.assertIn("control accepted run", run["failed"])
        self.assertIn("this gate needs the judgment rehearsal", run["lines"])
        self.assertEqual(run["grade"]["hardware_gate"], "open: the CUDA processor did not execute, this is a host rehearsal")
        with self.assertRaises(SystemExit) as stopped, contextlib.redirect_stderr(io.StringIO()) as said:
            self.run_once(self.reference["out"])
        self.assertEqual(stopped.exception.code, 2)
        self.assertIn("already exists. Evidence is never overwritten.", said.getvalue())

    def test_a_run_without_the_control_grades_nothing_else(self):
        run = self.run_once(self.out / "nothing", control=self.out)
        self.assertEqual(run["status"], 1)
        self.assertIn("control accepted run", run["failed"])
        self.assertIn("timed control", run["failed"])
        self.assertEqual(run["grade"]["cases_passed"], 0)

    def test_a_faulty_configuration_or_adapter_stops_before_anything_is_graded(self):
        good = json.loads(self.config.read_text())
        cases = (
            (lambda config: config.update(selector="pilot_ls"), "for the selector snapshot_ring"),
            (lambda config: config.update(backends=["cpu"]), "decodes every grid with both"),
            (lambda config: config["ring"].update(slots=8), "ring.slots is 8"),
            (lambda config: config["tap"].update(ceiling_us=0), "each positive"),
            (lambda config: config["slow_consumer"].update(hold_us=0), "hold_us, a positive integer"),
            (lambda config: config["terminated_consumer"].update(exit_after=0), "exit_after, a positive integer"),
            (lambda config: config["control"].update(driver="replay_pusch"), "the driver pilot_ls"),
            (lambda config: config["pair"].pop("expect"), "pair requires expect"),
        )
        for number, (edit, message) in enumerate(cases):
            config = json.loads(json.dumps(good))
            edit(config)
            path = self.out / f"config-{number}.json"
            path.write_text(json.dumps(config))
            with self.assertRaises(SystemExit) as stopped, contextlib.redirect_stderr(io.StringIO()) as said:
                self.run_once(self.out / f"run-{number}", config=path)
            self.assertEqual(stopped.exception.code, 2)
            self.assertIn(message, said.getvalue())
            self.assertFalse((self.out / f"run-{number}").exists())
        receipt = self.receipt_of()
        del receipt["executables"]["obs_tail"]
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()) as said:
            self.run_once(self.out / "no-tail", receipt=receipt)
        self.assertIn("the adapter build holds no obs_tail", said.getvalue())

    def test_the_commands_of_the_consumers(self):
        driver = self.driver
        tools = {"obs_tail": Path("/build/obs_ring/obs_tail"), "pilot_ls": Path("/build/observer/pilot_ls")}
        native = driver.consumer_command(tools, {"reader": "native", "wait": "spin", "archive": Path("/out/consumer-0"),
                                                 "hold_us": 5000, "exit_after": 4}, Path("/out/native/ring/o2.sock"), None)
        self.assertEqual(native[:7], ["/build/obs_ring/obs_tail", "--socket", "/out/native/ring/o2.sock", "--output-dir",
                                      "/out/consumer-0", "--wait", "spin"])
        self.assertEqual(native[-4:], ["--hold-us", "5000", "--exit-after", "4"])
        after = driver.consumer_command(tools, {"reader": "native", "wait": "wait", "archive": Path("/out/consumer-1")},
                                        Path("/s"), 4711)
        self.assertEqual(after[-2:], ["--after-pid", "4711"])
        reference = driver.consumer_command(tools, {"reader": "reference", "wait": "wait", "archive": Path("/a")},
                                            Path("/s"), None)
        self.assertEqual(reference[:4], [sys.executable, "-m", "ocudu_cuda_isac.sensing.obs", "tail"])
        self.assertIn("--wait", reference)


if __name__ == "__main__":
    unittest.main()
