#!/usr/bin/env python3
"""The Sprint 4.1 integration driver on a stand-in control: python tests/unit/test_mps_pacing_driver.py -v

The driver, tests/integration/mps_pacing.py, is a hardware command. Here it runs as a rehearsal.
The control is a Sprint 3.5 run that the driver of that sprint grades on the doubles of its own
driver tests: a small episode set written with the free-space model, a model of a few channels
trained on eight windows. The launcher and the processes of an arm are the double of this sprint
(tests/unit/mps_pacing_double.py): the receiver is the double of Sprint 5.2 with the offered
schedule of a paced run added by the release rule of the reference, and the sensor is the real
sense_host of the host twin of the core, on the records the double kept. The readers, the
reducers and every check of the driver are the real ones. The rehearsal grades every case,
writes mps_pacing.json and leaves its hardware gate open, and each injected fault must fail
the check that names it. The double of the receiver runs the lead-in of a plan on its own clock,
with a first decode of 1.3 ms, so that the checks of the cold start are graded here as well. The
double can rewrite one score of a sensor: the same input with another logit and another decode
with a logit far away fail the check of the scores, and a flip at the threshold between two
decodes is listed and fails nothing.

It needs cmake, a C++17 compiler, NumPy, PyTorch and pybind11, and no CUDA and no MPS.
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
    import torch  # noqa: F401
except ImportError:
    np = None

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sense_core_build  # noqa: E402

CASES = ("build", "control", "mps", "schedule", "dsp_only", "capture_only", "ai_only", "concurrent", "contention",
         "runtime", "faults", "records")
CHECKS = 42
BLOCKED = ["contention control", "runtime control", "faults control", "records control"]
BUILD = PROBLEM = STAGE = DRIVER = DOUBLE = ROOT_DIR = None


def setUpModule():
    global BUILD, PROBLEM, STAGE, DRIVER, DOUBLE, ROOT_DIR
    if np is None:
        PROBLEM = "the rehearsal needs NumPy and PyTorch in this interpreter"
        return
    BUILD, PROBLEM = sense_core_build.host_twin()
    if PROBLEM is not None:
        return
    # The stand-in for the accepted Sprint 3.5 run: the driver of that sprint on the doubles of its tools.
    import test_cusense_frontend_driver as control_tests
    import mps_pacing_double
    STAGE = control_tests.DriverTests
    STAGE.setUpClass()
    assert STAGE.reference["status"] == 0, STAGE.reference["failed"]
    spec = importlib.util.spec_from_file_location("mps_pacing_driver", ROOT / "tests/integration/mps_pacing.py")
    DRIVER = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(DRIVER)
    DOUBLE = mps_pacing_double
    DOUBLE.SENSE_HOST = BUILD / "sense_host"
    ROOT_DIR = STAGE.root / "s41"
    ROOT_DIR.mkdir()


def tearDownModule():
    stage = STAGE
    while stage is not None:
        stage.doClassCleanups()
        stage.tearDownClass()
        stage = getattr(stage, "stage", None)


class DriverTests(unittest.TestCase):
    def setUp(self):
        if PROBLEM is not None:
            self.skipTest(PROBLEM)
        scratch = tempfile.TemporaryDirectory(dir=ROOT_DIR)
        self.addCleanup(scratch.cleanup)
        self.scratch = Path(scratch.name)
        self.tool = self.scratch / "pilot_ls"
        self.tool.write_text('#!/bin/sh\necho "pilot_ls pacing: pilot-ls-plan/v4, absolute releases on the host '
                             'monotonic clock, paced book without an allocation per decode, lead-in ahead of the '
                             'schedule"\n')
        self.tool.chmod(0o755)
        settings = json.loads((ROOT / "configs/integration/mps_pacing.json").read_text())
        # The small set: one validation pair of two periods, so twelve releases and four windows per arm.
        settings["episodes"]["pairs"] = 1
        settings.update(repetitions=2, warmup=1, core_build=str(BUILD))
        settings["cpus"] = {"receiver": "1", "sensor": "0", "o2": "0", "tools": "0"}
        settings["contention"]["rule"].update(quantile="p50", min_samples=2)
        settings["contention"]["control_repetitions"] = 1
        settings["expect"].update(min_windows=2, max_skipped_releases=0)
        self.settings = settings

    def receipt(self):
        series = DRIVER.receiver_patch.identity(DRIVER.receiver_patch.load_series())
        names = ("pilot_ls", "obs_tail", "obs_feed", "replay_pusch")
        return {"executables": {name: {"path": str(self.tool), "sha256": hashlib.sha256(self.tool.read_bytes()).hexdigest()}
                                for name in names},
                "adapter_build": "double-o3", "ocudu_build": "double-o3", "pin": {"cuda": True},
                "patch": dict(series, tree_kind="patched-worktree"),
                "o3_route": {"compiled": True, "tap_interface_version": 1}, "tests_run": 95, "compiler_warnings": 0,
                "sources": {f"native/ocudu_adapter/pacing/{name}": "0" * 64
                            for name in ("CMakeLists.txt", "pacing_core.h", "pacing_core.cpp", "tests/pacing_core_test.cpp")}}

    def drive(self, name="out", fault=None, settings=None, extra=(), control=True, receipt=None):
        out = self.scratch / name
        config = self.scratch / f"{name}.json"
        config.write_text(json.dumps(settings or self.settings))
        saved = (DRIVER.native_adapter.verified_receipt, DRIVER.run_m1.launch, DRIVER.run_m1.Mps)
        DRIVER.native_adapter.verified_receipt = lambda build: receipt or self.receipt()
        DRIVER.run_m1.launch, DRIVER.run_m1.Mps = DOUBLE.launch, DOUBLE.Mps
        kept = os.environ.pop("MPS_PACING_FAULT", None)
        if fault:
            os.environ["MPS_PACING_FAULT"] = fault
        affinity = os.sched_getaffinity(0)
        arguments = ["--config", str(config), "--output-dir", str(out), "--rehearsal", "--core-build", str(BUILD), *extra]
        if control:
            arguments += ["--s35-run", str(STAGE.reference["out"])]
        said, errors, status = io.StringIO(), io.StringIO(), None
        try:
            with contextlib.redirect_stdout(said), contextlib.redirect_stderr(errors):
                try:
                    status = DRIVER.main(arguments)
                except SystemExit as stop:
                    status = stop.code
        finally:
            DRIVER.native_adapter.verified_receipt, DRIVER.run_m1.launch, DRIVER.run_m1.Mps = saved
            os.environ.pop("MPS_PACING_FAULT", None)
            if kept is not None:
                os.environ["MPS_PACING_FAULT"] = kept
            os.sched_setaffinity(0, affinity)
        made = {"status": status, "out": out, "said": said.getvalue(), "errors": errors.getvalue(), "grade": None,
                "failed": None}
        if (out / "grade.json").is_file():
            made["grade"] = json.loads((out / "grade.json").read_text())
            made["failed"] = [f"{item['case']} {item['check']}" for item in made["grade"]["results"] if not item["passed"]]
        return made

    def small(self):
        """One repetition, no warm-up and no time slicing: enough for a fault to reach its check."""
        settings = json.loads(json.dumps(self.settings))
        settings.update(repetitions=1, warmup=0, time_slicing_control=False)
        return settings

    def test_a_rehearsal_grades_every_case_and_leaves_the_hardware_gate_open(self):
        run = self.drive()
        self.assertEqual(run["status"], 0, run["said"][-3000:] + run["errors"][-2000:])
        grade, out = run["grade"], run["out"]
        self.assertEqual((grade["judgment"], grade["cases"], grade["cases_passed"], grade["checks"], grade["checks_failed"],
                          grade["rehearsal"], grade["control_in_effect"]), ("pass", 12, 12, CHECKS, 0, True, "mps_default"))
        self.assertTrue(grade["hardware_gate"].startswith("open: a rehearsal runs no receiver"))
        self.assertEqual(tuple(dict.fromkeys(item["case"] for item in grade["results"])), CASES)
        lines = (out / "grade.txt").read_text()
        for text in ("12 opportunities, one release each, over 17.0 ms of offered schedule with distances of 500, 4000, "
                     "5000 us", "24 of 24 released grids gave the original bytes", "0 of 24 offered releases were skipped",
                     "8 windows were scored beside the receiver and, from the buffered records of the capture-only arm, "
                     "by the sensor alone. 8 of them were given the same input bytes in both arms, and the sensor gave 8 "
                     "of them the same logit bit for bit and 8 the same map; their logits are 0.0e+00 apart at most, "
                     "1e-06 allowed. No window was scored from two decodes that differ. 8 decide alike, 0 flip(s)",
                     "the control in effect is mps_default",
                     "sense_host is killed 700 ms after the start of the offered schedule: the receiver was never signalled",
                     "carries the label time_slicing and never MPS", "is no client of the server, and the label is refused",
                     "1 held-out pairs of the set are not read by this gate",
                     "a lead-in of 6 decodes, the first releases once more, 10000 us ahead of their own release times, "
                     "from 10.0 to 4.0 ms before the schedule starts, proposed",
                     "lead-in ahead of the schedule",
                     "in every repetition the 6 decodes of the lead-in gave the original bytes of their block and were "
                     "done 3813.5 us or more before the first release; no release is the first use of the processor, "
                     "and none of the first 6 releases of a schedule was skipped",
                     "decode 1 1310.0 to 1310.0 us; decode 2 236.0 to 236.0 us"):
            self.assertIn(text, lines, text)
        self.assertEqual([item["check"] for item in grade["results"]].count("lead-in"), 3)
        # ---- The record of the card: every arm with its label, its counts and its tails, and nothing graded.
        record = json.loads((out / "mps_pacing.json").read_text())
        self.assertEqual((record["schema"], record["graded"], record["rehearsal"]), ("mps-pacing/v1", False, True))
        self.assertEqual((record["load"]["scripted"], record["load"]["core_network"]),
                         (True, "none: no UE, no 5GC and no PRACH procedure is attached"))
        self.assertEqual(set(record["arms"]), {"dsp_only", "capture_only", "ai_only", "concurrent", "concurrent_reference"})
        timed = [item for arm in ("dsp_only", "capture_only", "ai_only", "concurrent") for item in record["arms"][arm]
                 if item["control"] == "mps_default" and item["fault"] is None and not item["warmup"]
                 and item["sharing"]["requested"] == "mps"]
        self.assertEqual(len(timed), 8)
        for item in timed:
            self.assertEqual((item["sharing"]["label"], item["sharing"]["verified"], item["problems"]), ("mps", True, []))
        both = [item for item in timed if item["arm"] == "concurrent"]
        self.assertEqual({tuple(sorted(item["sharing"]["proof"]["roles"])) for item in both}, {("producer", "sensor")})
        for item in both + record["arms"]["dsp_only"]:
            pacing = item["radio"]["pacing"]
            self.assertEqual((pacing["offered"], pacing["counts"]["on_time"], pacing["counts"]["skipped"],
                              item["radio"]["decoded"]), (12, 12, 0, 12))
        self.assertEqual(record["frozen"]["rings"]["o3"]["slots"], 8)
        self.assertEqual([order[0] for order in record["frozen"]["order"]], ["dsp_only", "concurrent"])
        contention = record["contention"]
        self.assertEqual((contention["control_in_effect"], contention["shifts"]["mps_default"]["exceeded"],
                          contention["rule"]["status"]), ("mps_default", False, "proposed"))
        self.assertEqual({arm: counts for arm, counts in contention["offered"].items()},
                         {"dsp_only": [12, 12], "capture_only": [12, 12], "concurrent": [12, 12]})
        self.assertEqual(set(record["faults"]), {"consumer_exit", "consumer_restart"})
        # ---- The scores beside the receiver against the sensor alone, by repetition, with nothing to list.
        scores = record["decisions"]
        self.assertEqual((scores["joined"], scores["agree"], scores["same_input"], scores["same_input_same_logit"],
                          scores["other_input"], scores["flips"], scores["problems"]), (8, 8, 8, 8, 0, [], []))
        self.assertEqual((scores["logit_tolerance"], scores["same_input_tolerance"],
                          [entry["repetition"] for entry in scores["repetitions"]]), (0.05, 1e-06, ["r0", "r1"]))
        self.assertEqual(scores, json.loads((out / "comparison.json").read_text())["concurrent"]["agreement"])
        # ---- The cold start is apart: the decodes of the lead-in of every run with a receiver, and no release.
        cold = record["cold_start"]
        self.assertEqual((cold["lead_in"], cold["runs"], cold["first_use_in_schedule"], cold["skipped_first_releases"]),
                         ({"releases": 6, "lead_us": 10000, "status": "proposed"}, 6, 0, 0))
        self.assertIn("no sample of a service-time statistic", cold["isolated"])
        self.assertEqual([entry["p50_us"] for entry in cold["by_decode_us"]], [1310.0, 236.0, 200.5, 185.25, 181.0, 180.5])
        self.assertEqual({arm: len(runs) for arm, runs in cold["arms"].items()},
                         {"dsp_only": 2, "capture_only": 2, "concurrent": 2})
        self.assertEqual((cold["measured_without_lead_in"]["run"], cold["measured_without_lead_in"]["skipped_releases_per_run"]),
                         ("s41-spark-ecf8-20261123T140213Z", 2))
        # No sample of a service statistic is cold: every release that ran is counted, and none took the 1.3 ms.
        for arm in ("dsp_only", "capture_only", "concurrent"):
            self.assertEqual(contention["service"][arm]["count"], 24)
            self.assertLess(contention["service"][arm]["max_us"], 1000.0)
        for launch in ("r0-dsp_only", "r0-concurrent"):
            folder = out / "arms" / launch / "native"
            ahead = [json.loads(line) for line in (folder / "pacing/lead_in.jsonl").read_text().splitlines()]
            radio = [json.loads(line) for line in (folder / "radio/radio.jsonl").read_text().splitlines()]
            releases = [json.loads(line) for line in (folder / "pacing/releases.jsonl").read_text().splitlines()]
            self.assertEqual((len(ahead), len(radio), len(releases)), (6, 12, 12))
            self.assertFalse(any(record_["timing"]["first_use"] for record_ in radio))
            self.assertLess(max(line["t_done_ns"] for line in ahead), releases[0]["t_release_ns"])
            self.assertEqual([line["opportunity_id"] for line in ahead], [line["opportunity_id"] for line in releases[:6]])
        self.assertGreater(record["faults"]["consumer_exit"]["account"]["dropped"], 0)
        self.assertEqual(record["time_slicing"]["label"], "time_slicing")
        # ---- What the launches left: one directory each, with its specification and its record.
        manifest = json.loads((out / "manifest.json").read_text())
        labels = [launch["label"] for launch in manifest["launches"]]
        self.assertEqual(labels[:5], ["warmup-0", "r0-dsp_only", "r0-capture_only", "r0-ai_only", "r0-concurrent"])
        self.assertEqual(labels[5:9], ["r1-concurrent", "r1-capture_only", "r1-ai_only", "r1-dsp_only"])
        self.assertEqual(labels[9:], ["reference-0", "consumer_exit-0", "consumer_restart-0", "time_slicing-0"])
        for launch in manifest["launches"]:
            where = out / launch["directory"]
            spec = json.loads((where / "spec.json").read_text())
            self.assertEqual((spec["schema"], json.loads((where / "launch.json").read_text())["schema"]),
                             ("m1-arm/v1", "m1-launch/v1"))
            # No arm reconfigures the receiver, and every timed arm keeps the ring of the configuration.
            self.assertEqual(spec["producer"]["env"], {})
        plan = json.loads((out / "plans/plan-tap.json").read_text())
        self.assertEqual((plan["schema"], plan["evidence"]["mode"], plan["ring"]["slots"], plan["estimate"]["ring"]["slots"],
                          plan["ring"]["wait"], len(plan["pacing"]["release_us"]), plan["pacing"]["lead_in"]),
                         ("pilot-ls-plan/v4", "timed", 8, 8, "spin", 12, {"releases": 6, "lead_us": 10000}))
        replay_plan = json.loads(Path(plan["replay_plan"]["path"]).read_text())
        self.assertEqual(replay_plan["receiver"]["backends"], ["cuda"])
        self.assertEqual(json.loads((out / "plans/plan-dsp.json").read_text())["pacing"], plan["pacing"])

    def test_each_injected_fault_fails_the_check_that_names_it(self):
        arms = ["dsp_only launch", "capture_only launch", "ai_only launch", "concurrent launch"]
        for fault, wanted in (
                ("skipped-release", ["dsp_only schedule kept"]),
                ("early-start", arms),
                ("shifted-schedule", ["dsp_only launch", "capture_only launch", "concurrent launch"]),
                ("other-service", arms),
                ("no-client", ["ai_only launch", "concurrent launch"]),
                ("late-context", ["concurrent start order"]),
                ("drop", ["capture_only account", "concurrent account"]),
                ("sensor-exit", ["ai_only launch", "concurrent launch"]),
                ("receiver-exit", ["dsp_only launch", "capture_only launch", "concurrent launch"]),
                # The lead-in: a tool that runs none, a release that is cold beside one, and one that ends too late.
                ("no-lead-in", ["dsp_only launch", "capture_only launch", "concurrent launch"]),
                ("cold-schedule", arms),
                ("late-lead-in", ["dsp_only schedule kept", "dsp_only lead-in"]),
                # The scores: another logit for the same input bytes, and another decode with a logit far away.
                ("other-score", ["concurrent same decisions"]),
                ("far-score", ["concurrent same decisions"])):
            with self.subTest(fault=fault):
                run = self.drive(fault, fault, self.small())
                self.assertEqual(run["status"], 1, fault)
                self.assertEqual(run["failed"], wanted + BLOCKED, fault)
                self.assertEqual((run["grade"]["judgment"], run["grade"]["hardware_gate"].split(":")[0]), ("fail", "open"))
        # A refused label carries the client list, and a schedule that moved is named as such.
        lines = (self.scratch / "no-client" / "grade.txt").read_text()
        self.assertRegex(lines, r"the sensor process \d+ is not a client of the MPS server 900: it shares the device by "
                                r"time slicing\. Client list of the server 900: \d+")
        self.assertIn("the offsets of the releases are not those of the plan",
                      (self.scratch / "shifted-schedule" / "grade.txt").read_text())
        self.assertIn("a job started before its offered release time", (self.scratch / "early-start" / "grade.txt").read_text())
        self.assertIn("1 skipped (at most 0 per repetition)", (self.scratch / "skipped-release" / "grade.txt").read_text())
        # A lead-in that the tool did not run, a release that is still the first use, and a lead-in that reaches
        # into its schedule: each is named.
        self.assertIn("the lead-in of the report is not the lead-in of the plan",
                      (self.scratch / "no-lead-in" / "grade.txt").read_text())
        self.assertIn("names the first use of its backend, and the lead-in decoded on that backend before the schedule",
                      (self.scratch / "cold-schedule" / "grade.txt").read_text())
        late = (self.scratch / "late-lead-in" / "grade.txt").read_text()
        self.assertIn("a lead-in was not done before the first release of its schedule: it ended 1369.2 us after it; 3 "
                      "of the first 6 releases of a schedule were skipped: the cold start still costs the schedule", late)
        # The same input bytes with another logit, and two decodes further apart than was declared: each is named.
        self.assertIn("1 window(s) break the rule, first: r0: window ", (self.scratch / "other-score" / "grade.txt").read_text())
        self.assertRegex((self.scratch / "other-score" / "grade.txt").read_text(),
                         r"the same input bytes gave the logits -?\d+\.\d{9} and -?\d+\.\d{9}, 1\.000e-03 apart "
                         r"\(at most 1\.0e-06\)")
        self.assertRegex((self.scratch / "far-score" / "grade.txt").read_text(),
                         r"The remaining 1 window\(s\) were scored from two decodes of one grid that differ, as the "
                         r"receiver's estimate is not bit-reproducible: their logits are 2\.00e-01 apart at most, 0\.05 "
                         r"allowed\. 4 decide alike, 0 flip\(s\)\. 1 window\(s\) break the rule, first: r0: window \S+: two "
                         r"decodes that differ gave the logits")
        # A client of the server before the first arm: no arm is launched beside it.
        run = self.drive("foreign", "foreign-client", self.small())
        self.assertEqual(run["failed"][0], "mps daemon")
        self.assertFalse((run["out"] / "arms").exists())

    def test_a_flip_at_the_threshold_between_two_decodes_is_listed_and_fails_nothing(self):
        # One window is scored from two decodes, 1e-3 above the threshold beside the receiver and 1e-3 below it
        # alone: what the Spark measured in the run s41-spark-ecf8-20261123T161419Z, where a gate that asked for
        # equal decisions of two decodes failed on one window in 960.
        run = self.drive("flip", "threshold-flip", self.small())
        self.assertEqual((run["status"], run["failed"]), (0, []), run["said"][-3000:] + run["errors"][-2000:])
        self.assertEqual((run["grade"]["judgment"], run["grade"]["cases_passed"]), ("pass", 12))
        lines = (run["out"] / "grade.txt").read_text()
        self.assertRegex(lines, r"PASS  S4\.1 concurrent same decisions: 4 windows .* 3 of them were given the same input "
                                r"bytes in both arms, .* The remaining 1 window\(s\) were scored from two decodes of one "
                                r"grid that differ, .* their logits are 2\.00e-03 apart at most, 0\.05 allowed\. 3 decide "
                                r"alike, 1 flip\(s\), 1 of them at the threshold -?\d+\.\d{6}, each listed in "
                                r"comparison\.json: r0 \S+ with -?\d+\.\d{6} and -?\d+\.\d{6}, \+1\.00e-03 and -1\.00e-03 "
                                r"from it, two decodes\n")
        scores = json.loads((run["out"] / "mps_pacing.json").read_text())["decisions"]
        self.assertEqual(scores, json.loads((run["out"] / "comparison.json").read_text())["concurrent"]["agreement"])
        self.assertEqual((scores["joined"], scores["agree"], scores["same_input"], scores["other_input"],
                          scores["flips_at_threshold"], scores["problems"]), (4, 3, 3, 1, 1, []))
        self.assertEqual((scores["largest_logit_difference_same_input"], scores["largest_logit_difference"]),
                         (0.0, scores["largest_logit_difference_other_input"]))
        flip, = scores["flips"]
        self.assertEqual((flip["repetition"], flip["same_input"], flip["at_threshold"]), ("r0", False, True))
        self.assertEqual(flip["threshold_logit"], scores["threshold_logit"])
        self.assertAlmostEqual(flip["margins"][0], 1e-3, places=9)
        self.assertAlmostEqual(flip["margins"][1], -1e-3, places=9)
        # Both outputs still verify with the reader of the reference: the decisions follow from their logits.
        from ocudu_cuda_isac.sensing import cusense
        for arm, present in (("r0-concurrent", True), ("r0-ai_only", False)):
            output = cusense.open_output(run["out"] / "arms" / arm / "sensor")
            self.assertEqual(output.scored[1]["decision"]["present"], present, arm)

    def test_the_rule_and_the_order_of_the_controls_decide_what_is_applied(self):
        # A rule that needs more samples than the arms hold gives no verdict, and says so.
        settings = self.small()
        settings["contention"]["rule"].update(quantile="p99", min_samples=1000)
        run = self.drive("few", settings=settings)
        self.assertEqual(run["failed"], ["contention tail", "contention control in effect"])
        # Twelve samples of twelve releases: with a lead-in no release is left out as the first use.
        self.assertIn("the rule gives no verdict: 12 samples, and the rule needs 1000 for the quantile p99",
                      (run["out"] / "grade.txt").read_text())
        # Without a lead-in the plan is the plan of the accepted run: the first release is the first use of the
        # processor, it is left out of the statistics, and the gate says so and grades no lead-in.
        settings = self.small()
        settings["pacing"]["lead_in"] = None
        run = self.drive("cold", settings=settings)
        self.assertEqual(run["status"], 0, run["said"][-2000:])
        self.assertNotIn("lead-in", [item["check"] for item in run["grade"]["results"]])
        lines = (run["out"] / "grade.txt").read_text()
        self.assertIn("no lead-in: the first decode of a run is the first use of its processor", lines)
        self.assertEqual(lines.count("no lead-in: in each of 1 repetition(s) the first release is the first use of the "
                                     "processor"), 3)
        record = json.loads((run["out"] / "mps_pacing.json").read_text())
        self.assertEqual((record["cold_start"]["lead_in"], record["cold_start"]["first_use_in_schedule"]), (None, 3))
        self.assertEqual(record["contention"]["service"]["dsp_only"]["count"], 11)
        plan = json.loads((run["out"] / "plans/plan-tap.json").read_text())
        self.assertNotIn("lead_in", plan["pacing"])
        self.assertFalse((run["out"] / "arms/r0-dsp_only/native/pacing/lead_in.jsonl").exists())
        # A rule that allows nothing is exceeded by nothing here: the tails of the double are equal.
        settings = self.small()
        settings["contention"]["rule"].update(allowed_shift_us=0.0, allowed_shift_fraction=0.0)
        run = self.drive("strict", settings=settings)
        self.assertEqual((run["status"], run["grade"]["control_in_effect"]), (0, "mps_default"))
        record = json.loads((run["out"] / "mps_pacing.json").read_text())
        self.assertEqual(list(record["contention"]["shifts"]), ["mps_default"])
        self.assertNotIn("mps_priority", [item["control"] for arm in record["arms"].values() for item in arm])
        # The same configuration with its lead-in grades one check more per arm with a receiver.
        self.assertEqual([item["check"] for item in run["grade"]["results"]].count("lead-in"), 3)

    def test_setup_errors_leave_no_evidence(self):
        def refused(message, **options):
            run = self.drive("refused", **options)
            self.assertEqual(run["status"], 2)
            self.assertIn(message, run["errors"])
            self.assertFalse(run["out"].exists())

        held = self.small()
        held["episodes"]["group"] = "held_out"
        refused("this gate reads no held-out episode before Sprint 4.6 has frozen the profile", settings=held)
        wait = self.small()
        wait["rings"]["o3"]["wait"] = "wait"
        refused("the wait mode spin of a timed run", settings=wait)
        # The scores of two decodes are compared by tolerance. A configuration of the accepted run, which asked
        # for a share of equal decisions, names no tolerance for the same input and is refused.
        share = self.small()
        del share["expect"]["same_input_logit_tolerance"]
        share["expect"]["min_decision_agreement"] = 1.0
        refused("the configuration is incomplete or unusable: KeyError('same_input_logit_tolerance')", settings=share)
        loose = self.small()
        loose["expect"]["same_input_logit_tolerance"] = 0.1
        refused("a setting of the configuration is outside its range", settings=loose)
        lead = self.small()
        lead["pacing"]["lead_in"] = {"releases": 6}
        refused("pacing.lead_in is null, or names how many of the first releases are decoded ahead", settings=lead)
        deeper = self.small()
        deeper["rings"]["o3"]["slots"] = 64
        refused("no arm enlarges a ring", settings=deeper)
        host = self.small()
        host["receiver"]["backends"] = ["cpu", "cuda"]
        refused("a timed arm decodes on the CUDA processor alone", settings=host)
        shared = self.small()
        shared["cpus"]["sensor"] = "1"
        refused("outside the CPU set of the receiver", settings=shared)
        order = self.small()
        order["contention"]["order"] = ["release_after_radio", "mps_default", "mps_priority", "mps_thread_pct"]
        refused("contention.order is the order of the controls", settings=order)
        refused("the gate needs the accepted Sprint 3.6 run: --s36-run", control=False)
        missing = self.receipt()
        del missing["executables"]["pilot_ls"]
        refused("a build holds no pilot_ls", receipt=missing)
        (self.scratch / "refused").mkdir()
        run = self.drive("refused")
        self.assertEqual(run["status"], 2)
        self.assertIn("Evidence is never overwritten", run["errors"])


if __name__ == "__main__":
    unittest.main()
