"""Check the Sprint 2 chain audit on four runs made with doubles of the native tools.

The Sprint 2.4 evidence is written by the integration driver against the doubles of
tests/unit/test_replay_pusch_driver.py, so its layout is the driver's. The Sprint 2.3 grids come
from the double of the channel application, and the stores and the pair from the Python modules.
The verdicts, receipts, manifests and grades of the three earlier gates hold the fields the audit
reads. NumPy is needed, as for the driver.
"""

import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    np = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests/unit"))
from scripts import verify_sprint2 as audit  # noqa: E402

CHECKS = 17
HOST = "spark-unit"
LOCK = {item["name"]: item["commit"] for item in json.loads((ROOT / "ext/lock.json").read_text())["dependencies"]}
TOOLS = {"grant_fixture": {"sha256": "0123456789abcdef" * 4}, "grid_apply": {"sha256": "fedcba9876543210" * 4},
         "replay_pusch": {"sha256": "0f1e2d3c4b5a6978" * 4}}
GATES = (("2.1", "s21", "grant_fixture", ("grant_fixture",), "2026-01-05T10:00:00Z", "2026-01-05T10:01:00Z", (3, 38)),
         ("2.2", "s22", "episode_pair", ("grant_fixture",), "2026-01-05T11:00:00Z", "2026-01-05T11:30:00Z", (4, 26)),
         ("2.3", "s23", "grid_apply", ("grant_fixture", "grid_apply"), "2026-01-05T12:00:00Z", "2026-01-05T12:01:00Z",
          (5, 25)))


def receipt(names):
    return {"ocudu_build": "/home/lab/build/ocudu-spark", "ocudu_receipt_sha256": "ab" * 32, "compiler_warnings": 0,
            "pin": {"cuda": True, "link_txt_sha256": "cd" * 32, "flags_make_sha256": "ef" * 32},
            "executables": {name: TOOLS[name] for name in names}}


def summary(sprint, profile="DGX Spark", host=HOST):
    return (f"Sprint {sprint} acceptance on {host}, 2026-01-05T10:00:00Z\ncommit 0123456789abcdef0123456789abcdef01234567\n"
            f"PASS  committed tree\nSprint {sprint}: ACCEPTED on this {profile}. Evidence is archived.\n")


@unittest.skipIf(np is None, "the Sprint 2.4 evidence is written by the integration driver, which needs NumPy")
class ChainAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_replay_pusch_driver as driver_tests
        cls.fixture = driver_tests.DriverTests
        cls.fixture.setUpClass()  # the stimulus, the grids of Sprint 2.3 and one driver run without a fault
        cls.addClassCleanup(cls.fixture.tearDownClass)
        cls.root = cls.fixture.root
        made = cls.fixture.made
        cls.runs = {"2.1": cls.root / "s21", "2.2": made["s22"], "2.3": made["s23"], "2.4": cls.root / "s24"}
        for sprint, name, selector, tools, started, completed, (cases, checks) in GATES:
            run = cls.runs[sprint]
            (run / "summary.txt").write_text(summary(sprint))
            (run / "adapter-receipt.json").write_text(json.dumps(receipt(tools)))
            manifest = {"run_id": selector, "selector": selector, "started_utc": started, "completed_utc": completed,
                        "locked_sources": LOCK}
            if sprint == "2.2":
                manifest["grant_store"] = str(cls.runs["2.1"] / "grant_fixture")
            if sprint == "2.3":
                manifest["s22_run"] = str(cls.runs["2.2"])
            (run / selector / "manifest.json").write_text(json.dumps(manifest))
            (run / selector / "grade.json").write_text(json.dumps({
                "judgment": "pass", "cases": cases, "cases_passed": cases, "checks": checks, "checks_passed": checks,
                "checks_failed": 0}))
        cls.gate_run(cls.runs["2.4"], cls.fixture.reference["out"])
        cls.reference = cls.audit_once(cls.root / "audit-reference")

    @classmethod
    def gate_run(cls, run, driver_output, move=True):
        """A Sprint 2.4 run directory around the output of the integration driver."""
        run.mkdir()
        (shutil.move if move else shutil.copytree)(str(driver_output), str(run / "replay_pusch"))
        (run / "summary.txt").write_text(summary("2.4"))
        (run / "adapter-receipt.json").write_text(json.dumps(receipt(TOOLS)))
        return run

    @classmethod
    def audit_once(cls, out, *extra, s24=None):
        said = io.StringIO()
        with contextlib.redirect_stdout(said):
            status = audit.main(["--s24-run", str(s24 or cls.runs["2.4"]), "--output-dir", str(out), *extra])
        grade = json.loads((out / "grade.json").read_text())
        failed = sorted(f"{item['stage']} {item['check']}" for item in grade["results"] if not item["passed"])
        return {"status": status, "grade": grade, "failed": failed, "out": out, "lines": (out / "grade.txt").read_text(),
                "said": said.getvalue()}

    def setUp(self):
        scratch = tempfile.TemporaryDirectory(dir=self.root)
        self.addCleanup(scratch.cleanup)
        self.out = Path(scratch.name)
        self.number = 0

    def audited(self, *extra, **options):
        self.number += 1
        return self.audit_once(self.out / f"audit-{self.number}", *extra, **options)

    @contextlib.contextmanager
    def changed(self, path, change):
        """One evidence file with another content for the length of a check."""
        saved = path.read_bytes()
        try:
            content = change(saved.decode("utf-8"))
            path.write_text(content if isinstance(content, str) else json.dumps(content))
            yield
        finally:
            path.write_bytes(saved)

    @staticmethod
    def edited(change):
        def apply(text):
            value = json.loads(text)
            change(value)
            return value
        return apply

    # ---------------------------------------------------------------- the accepted chain
    def test_the_chain_of_four_accepted_runs_passes_every_check(self):
        run = self.reference
        grade, lines, out = run["grade"], run["lines"], run["out"]
        self.assertEqual((run["status"], run["failed"]), (0, []))
        self.assertEqual((grade["judgment"], grade["stages"], grade["stages_passed"], grade["checks"], grade["profile"]),
                         ("pass", 9, 9, CHECKS, "DGX Spark"))
        self.assertEqual(grade["runs"], {"s21": "s21", "s22": "s22", "s23": "s23", "s24": "s24"})
        self.assertEqual(lines, run["said"].split("Audit artifacts:")[0])
        for text in (
                "PASS  S2 verdicts four gates: S2.1 s21: ACCEPTED on this DGX Spark; S2.2 s22: ACCEPTED on this DGX "
                "Spark; S2.3 s23: ACCEPTED on this DGX Spark; S2.4 s24: ACCEPTED on this DGX Spark; host spark-unit",
                "S2.4 8 of 8 cases and 36 of 36 checks, judgment pass; the hardware gate of S2.4 is closed\n",
                "PASS  S2 links each gate consumed its predecessor: S2.2 built its pair on the grant store of s21; S2.3 "
                "applied the stimulus of s22; S2.4 decoded the stimulus of s22; S2.4 compared its grids with those of s23",
                "no gate started before the run it consumes completed",
                "PASS  S2 build one receiver build: the four adapters were compiled and linked against one OCUDU build "
                "(/home/lab/build/ocudu-spark)",
                "grant_fixture 0123456789ab is the same executable in the four runs and grid_apply fedcba987654 in S2.3 "
                "and S2.4, beside replay_pusch 0f1e2d3c4b5a",
                "PASS  S2 waveform grant record: the store entry of demo-51prb-4rx in the Sprint 2.1 run verifies",
                "the block is 1217 bytes, SHA-256 a2187d90d75129b1",
                "PASS  S2 waveform transmit grids and pair: the entry unit-pair verifies against its manifest and its "
                "grant schedule: 6 opportunities per case on 6 grants (slots 7, 8, 9, 17, 18, 19)",
                "PASS  S2 channel plan of Sprint 2.3: the plan that grid_apply read",
                "PASS  S2 channel stored grids: 12 stored grids of Sprint 2.3 verify against their readback digests",
                "12 of 12 grids it decoded have the readback digest of the grid that Sprint 2.3 stored for the same "
                "opportunity, and all 24 radio records name the digest of their own opportunity",
                "PASS  S2 receiver original bytes: cpu 12 of 12; cuda 12 of 12 decodes of the pair end as decoded with "
                "the decoder's CRC pass, and the bytes in the files of the run are the 1217 bytes of tb.bin in the "
                "Sprint 2.1 store of demo-51prb-4rx, compared here byte for byte; 24 records under 24 unique keys",
                "PASS  S2 receiver executed path: in the 7 native runs of the gate, cpu: 50 on the host path; cuda: 50 "
                "on the resident path",
                "PASS  S2 receiver every decode accounted for: 100 decodes planned in 7 runs and 100 records",
                "76 decoded, 0 payload_mismatch, 24 crc_fail, 0 decoder_error, 0 timeout, 0 not_dispatched",
                "PASS  S2 payload a block per opportunity: the payload rule per_opportunity with the generator "
                "sha256-counter/v1 and the seed 's24-opportunity-block' is recorded in the plan; 24 of 24 decodes",
                "cpu returned 6 distinct blocks for 6 opportunities; cuda returned 6 distinct blocks for 6 opportunities",
                "PASS  S2 bad-input kept in the denominator: 2 runs on deliberately bad input (bad-input/noise_only, "
                "bad-input/phase_break): 24 of 24 planned decodes end as crc_fail",
                "PASS  S2 sources one pin: the four runs and this checkout name the same locked sources: ocudu 27e84cb3",
                "NOTE  S2 sources commits of the four gates, not graded: S2.1 0123456789ab",
                f"sprint2: 9 of 9 stages passed, {CHECKS} of {CHECKS} checks"):
            self.assertIn(text, lines)
        # One row per opportunity: the record, the grids and what each backend returned.
        chain = json.loads((out / "chain.json").read_text())["opportunities"]
        self.assertEqual(len(chain), 12)
        stored = json.loads((self.runs["2.3"] / "grid_apply/reference_pair/native/native.json").read_text())
        for item in stored["opportunities"]:
            row = chain[item["opportunity_id"]]
            self.assertEqual(row["y_sha256_s23"], item["readback_sha256"])
            self.assertEqual((row["decoded_sha256_cpu"], row["decoded_sha256_cuda"]), (row["tb_sha256"],) * 2)
            self.assertEqual((row["outcome_cpu"], row["executed_cpu"], row["outcome_cuda"], row["executed_cuda"]),
                             ("decoded", "host", "decoded", "resident"))
            self.assertEqual(len({row[key] for key in ("x_sha256", "h_sha256", "w_sha256", "grant_record_sha256")}), 4)
        self.assertEqual(len({row["x_sha256"] for row in chain.values()}), 6)
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual((manifest["judgment"], manifest["selector"]), ("pass", "sprint2_chain"))
        self.assertEqual(manifest["tools"]["replay_pusch"], TOOLS["replay_pusch"]["sha256"])
        # The audit writes into its own directory only, and never twice.
        with self.assertRaises(SystemExit) as refused, contextlib.redirect_stderr(io.StringIO()) as said:
            audit.main(["--s24-run", str(self.runs["2.4"]), "--output-dir", str(out)])
        self.assertEqual(refused.exception.code, 2)
        self.assertIn("already exists. Evidence is never overwritten.", said.getvalue())

    def test_a_named_run_must_be_the_recorded_one(self):
        # The same runs under their names: nothing changes.
        run = self.audited("--s21-run", str(self.runs["2.1"]), "--s22-run", str(self.runs["2.2"]),
                           "--s23-run", str(self.runs["2.3"]))
        self.assertEqual((run["status"], run["failed"]), (0, []))
        # Another Sprint 2.3 run, with the same content under another name.
        other = self.out / "s23-other"
        shutil.copytree(self.runs["2.3"], other)
        run = self.audited("--s23-run", str(other))
        self.assertEqual((run["status"], run["failed"]), (1, ["links each gate consumed its predecessor"]))
        self.assertIn("S2.4 compared its grids with those of s23, and the audit reads s23-other", run["lines"])
        # A Sprint 2.3 run that applied another stimulus.
        with self.changed(self.runs["2.3"] / "grid_apply/manifest.json",
                          self.edited(lambda value: value.update(s22_run="/runs/s22-another"))):
            run = self.audited()
        self.assertEqual(run["failed"], ["links each gate consumed its predecessor"])
        self.assertIn("S2.3 applied the stimulus of s22-another, and the audit reads s22", run["lines"])
        # A consumer that started before its producer completed.
        with self.changed(self.runs["2.3"] / "grid_apply/manifest.json",
                          self.edited(lambda value: value.update(started_utc="2026-01-05T11:10:00Z"))):
            run = self.audited()
        self.assertEqual(run["failed"], ["links order"])
        self.assertIn("S2.3 started before S2.2 completed", run["lines"])

    def test_verdicts_that_are_not_four_acceptances_of_one_host(self):
        cases = (
            ("2.4", lambda text: text.replace("Sprint 2.4: ACCEPTED on this DGX Spark.",
                                              "Sprint 2.4: REHEARSED on this host, NOT ACCEPTED."),
             "S2.4 s24: no ACCEPTED verdict without a failed line"),
            ("2.2", lambda text: text.replace("PASS  committed tree", "FAIL  committed tree"),
             "S2.2 s22: no ACCEPTED verdict without a failed line"),
            ("2.3", lambda text: text.replace("ACCEPTED on this DGX Spark", "ACCEPTED on this host"),
             "the four runs are not of one host and profile"),
            ("2.1", lambda text: text.replace(HOST, "another-host"), "host another-host, spark-unit: the four runs are not"),
            ("2.1", lambda text: "", "S2.1 s21: no ACCEPTED verdict without a failed line"),
        )
        for sprint, change, text in cases:
            with self.subTest(text=text), self.changed(self.runs[sprint] / "summary.txt", change):
                run = self.audited()
                self.assertEqual((run["status"], run["failed"]), (1, ["verdicts four gates"]))
                self.assertIn(text, run["lines"])
        # A grade of the driver that is not a pass with a closed hardware gate.
        grade = self.runs["2.4"] / "replay_pusch/grade.json"
        for change, text in (
                (lambda value: value.update(judgment="rehearsal", hardware_gate="open: this is a host rehearsal"),
                 "judgment rehearsal; the hardware gate of S2.4 is open: a rehearsal or a fallback is not an acceptance"),
                (lambda value: value.update(hardware_gate="open"), "judgment pass; the hardware gate of S2.4 is open: a "),
                (lambda value: value.update(checks_failed=1, checks_passed=35), "S2.4 8 of 8 cases and 35 of 36 checks")):
            with self.subTest(text=text), self.changed(grade, self.edited(change)):
                run = self.audited()
                self.assertEqual((run["status"], run["failed"]), (1, ["verdicts grades"]))
                self.assertIn(text, run["lines"])
        with self.changed(self.runs["2.2"] / "episode_pair/grade.json", lambda text: "{}"):
            run = self.audited()
        self.assertEqual(run["failed"], ["verdicts grades"])
        self.assertIn("S2.2 grade unreadable", run["lines"])

    def test_another_build_or_a_changed_tool_breaks_the_chain(self):
        cases = (
            ("2.2", lambda value: value.update(ocudu_build="/home/lab/build/ocudu-other"), ["build one receiver build"],
             "the adapters of the four runs were not built against one CUDA build"),
            ("2.3", lambda value: value["pin"].update(link_txt_sha256="00" * 32), ["build one receiver build"],
             "S2.3 /home/lab/build/ocudu-spark receipt abababababab link 000000000000 cuda True"),
            ("2.1", lambda value: value["executables"]["grant_fixture"].update(sha256="77" * 32), ["build unchanged tools"],
             "a tool changed between the runs, so an earlier grid was not made by the tool that S2.4 used"),
            ("2.4", lambda value: value["executables"]["grid_apply"].update(sha256="77" * 32), ["build unchanged tools"],
             "Run the four gates again in order: make accept-s2 RERUN=1"),
            ("2.4", lambda value: value["executables"].pop("replay_pusch"), ["build unchanged tools"], "S2.4 grant_fixture"),
            ("2.3", lambda value: value.update(compiler_warnings=2), ["build unchanged tools"], "2 warnings"),
        )
        for sprint, change, failed, text in cases:
            with self.subTest(text=text), self.changed(self.runs[sprint] / "adapter-receipt.json", self.edited(change)):
                run = self.audited()
                self.assertEqual((run["status"], run["failed"]), (1, failed))
                self.assertIn(text, run["lines"])
        # Four adapters without CUDA are one build, and not the build of this gate.
        saved = {sprint: (self.runs[sprint] / "adapter-receipt.json").read_bytes() for sprint in audit.SPRINTS}
        try:
            for sprint, data in saved.items():
                value = json.loads(data)
                value["pin"]["cuda"] = False
                (self.runs[sprint] / "adapter-receipt.json").write_text(json.dumps(value))
            run = self.audited()
        finally:
            for sprint, data in saved.items():
                (self.runs[sprint] / "adapter-receipt.json").write_bytes(data)
        self.assertEqual(run["failed"], ["build one receiver build"])
        (self.runs["2.2"] / "adapter-receipt.json").rename(self.out / "held.json")
        try:
            run = self.audited()
        finally:
            (self.out / "held.json").rename(self.runs["2.2"] / "adapter-receipt.json")
        self.assertEqual(run["failed"], ["build one receiver build", "build unchanged tools"])

    def test_a_changed_record_grid_or_plan_is_found(self):
        store = self.runs["2.1"] / "grant_fixture/demo-51prb-4rx/store"
        # One byte of the archived transport block.
        block = store / "tb.bin"
        saved = block.read_bytes()
        block.write_bytes(saved[:-1] + bytes([saved[-1] ^ 1]))
        try:
            run = self.audited()
        finally:
            block.write_bytes(saved)
        self.assertEqual(run["status"], 1)
        self.assertEqual(run["failed"], ["channel waveform", "payload waveform", "receiver waveform",
                                         "waveform grant record"])
        self.assertIn("nothing follows from it", run["lines"])
        # The plan that Sprint 2.3 archived, with another gain: its report names another hash, and the gain is not the entry's.
        plan = self.runs["2.3"] / "grid_apply/reference_pair/plan.json"
        with self.changed(plan, self.edited(lambda value: value["gain"].update(value=value["gain"]["value"] * 2))):
            run = self.audited()
        self.assertEqual(run["failed"], ["channel plan of Sprint 2.3"])
        self.assertIn("the archived plan of Sprint 2.3 differs from the entry in: gain", run["lines"])
        with self.changed(plan, self.edited(lambda value: value["sources"][1]["h"].update(sha256="0" * 64))):
            run = self.audited()
        self.assertEqual(run["failed"], ["channel plan of Sprint 2.3"])
        self.assertIn("differs from the entry in: realization", run["lines"])
        # A stored grid of Sprint 2.3 with one changed byte: the reader refuses the run, and nothing is compared with it.
        grid = self.runs["2.3"] / "grid_apply/reference_pair/native/ingress/0004.c64"
        saved = grid.read_bytes()
        grid.write_bytes(bytes([saved[0] ^ 1]) + saved[1:])
        try:
            run = self.audited()
        finally:
            grid.write_bytes(saved)
        self.assertEqual(run["failed"], ["channel plan of Sprint 2.3", "receiver same grids"])
        self.assertIn("differs from its readback digest", run["lines"])
        # A decoded byte of Sprint 2.4: the reader of the radio records refuses the run.
        decoded = self.runs["2.4"] / "replay_pusch/reference_pair/run/native/radio/0007.cuda.bin"
        saved = decoded.read_bytes()
        decoded.write_bytes(saved[:100] + bytes([saved[100] ^ 1]) + saved[101:])
        try:
            run = self.audited()
        finally:
            decoded.write_bytes(saved)
        self.assertEqual(run["status"], 1)
        for name in ("receiver same grids", "receiver original bytes", "receiver executed path",
                     "receiver every decode accounted for"):
            self.assertIn(name, run["failed"])
        self.assertIn("the decoded bytes of radio record 15 differ from their hash", run["lines"])
        # Another pin than the one of this checkout.
        with self.changed(self.runs["2.2"] / "episode_pair/manifest.json",
                          self.edited(lambda value: value["locked_sources"].update(ocudu="1" * 40))):
            run = self.audited()
        self.assertEqual(run["failed"], ["sources one pin"])
        self.assertIn("S2.2 ocudu 11111111", run["lines"])

    def test_runs_of_a_receiver_that_misbehaved_are_not_a_chain(self):
        expected = {
            # A fallback to the host path: the bytes are right, and the path is not the one of the gate.
            "host-fallback": (["receiver executed path", "verdicts grades"],
                              "cuda: 50 on the host path; 50 decodes did not execute the requested path"),
            # Other grids than those of Sprint 2.3.
            "changed-grid": (["receiver same grids", "verdicts grades"],
                             "0 of 12 decoded grids equal those of Sprint 2.3"),
            # A valid CRC with a wrong byte.
            "wrong-byte": (["payload a block per opportunity", "receiver every decode accounted for",
                            "receiver original bytes", "verdicts grades"], "cpu 12 of 12; cuda 0 of 12 decodes"),
            # Bytes of the opportunity before: only a block per opportunity shows it.
            "stale-bytes": (["payload a block per opportunity", "receiver every decode accounted for", "verdicts grades"],
                            "13 of 24 decodes return the block"),
            # A forced acknowledgement of bad input.
            "forced-ack": (["bad-input kept in the denominator", "receiver every decode accounted for", "verdicts grades"],
                           "0 of 24 planned decodes end as crc_fail"),
        }
        for fault, (failed, text) in expected.items():
            produced = self.fixture.run_once(self.out / f"driver-{fault}", fault=fault)
            self.assertEqual(produced["status"], 1)
            run = self.audited(s24=self.gate_run(self.out / f"s24-{fault}", produced["out"]))
            with self.subTest(fault=fault):
                self.assertEqual((run["status"], run["failed"]), (1, failed))
                self.assertIn(text, run["lines"])
        # A rehearsal on a build without CUDA, with every check of its driver passed.
        produced = self.fixture.run_once(self.out / "driver-rehearsal", environment={"REPLAY_PUSCH_BUILD": "host"},
                                         extra=("--host-rehearsal",))
        self.assertEqual(produced["status"], 0)
        run = self.audited(s24=self.gate_run(self.out / "s24-rehearsal", produced["out"]))
        self.assertEqual((run["status"], run["failed"]), (1, ["receiver executed path", "verdicts grades"]))
        self.assertIn("built without CUDA: bad-input/noise_only", run["lines"])

    def test_unusable_arguments_grade_nothing(self):
        empty = self.out / "s24-empty"
        empty.mkdir()
        for arguments, text in (
                (["--s24-run", str(empty)], "holds no replay_pusch/manifest.json: it is not a Sprint 2.4 run"),
                (["--s24-run", str(self.runs["2.3"])], "it is not a Sprint 2.4 run"),
                (["--s24-run", str(self.runs["2.4"]), "--s22-run", str(self.out / "no-such-run")],
                 "does not name its grant store")):
            with self.subTest(text=text), self.assertRaises(SystemExit) as refused, \
                    contextlib.redirect_stderr(io.StringIO()) as said:
                audit.main([*arguments, "--output-dir", str(self.out / "never")])
            self.assertEqual(refused.exception.code, 2)
            self.assertIn(text, said.getvalue())
            self.assertFalse((self.out / "never").exists())


class TreeTests(unittest.TestCase):
    def test_the_audit_is_ascii_needs_the_standard_library_only_and_starts_no_tool(self):
        source = (ROOT / "scripts/verify_sprint2.py").read_text(encoding="ascii")
        for word in ("numpy", "subprocess", "native_adapter"):
            self.assertNotIn(word, source)
        self.assertEqual(audit.SPRINTS, ("2.1", "2.2", "2.3", "2.4"))
        self.assertEqual(len(audit.STAGES), 9)


if __name__ == "__main__":
    unittest.main()
