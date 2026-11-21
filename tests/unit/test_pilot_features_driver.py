"""Check the Sprint 3.3 integration driver against a double of the native tool and its ring.

The control is a Sprint 3.2 run that the driver of that sprint grades on the doubles of its own
driver tests: the small pair of two TDD periods, six opportunities per member, four receive
ports, with the records its consumer kept and the files its audited observer wrote. The sensing
side under the driver is the real one. The double of this sprint (tests/unit/pilot_features_double.py)
only stands in for the native tool and the ring of the live case. The window of the small pair
holds two observations 5 ms apart, so its Doppler axis has one moving cell: the sign of that axis
and a gap inside a window are graded by the unit gate, on eight periods. Every check of the
driver has a known answer here, and each injected fault must fail the check that names it. NumPy
is needed, as for the driver.
"""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
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
import test_snapshot_ring_driver as control_tests  # noqa: E402

if np is not None:
    import pilot_features_double as double  # noqa: E402
    from ocudu_cuda_isac.sensing import features  # noqa: E402

CHECKS = 28
CASES = 8
ENVIRONMENT = ("PILOT_FEATURES_FAULT", "SNAPSHOT_RING_FAULT", "PILOT_LS_FAULT", "PILOT_LS_BUILD", "REPLAY_PUSCH_FAULT",
               "REPLAY_PUSCH_BUILD")


def load_driver():
    spec = importlib.util.spec_from_file_location("pilot_features_driver", ROOT / "tests/integration/pilot_features.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(np is None, "the integration driver needs NumPy")
class DriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.driver = load_driver()
        # The accepted Sprint 3.2 run: the driver of that sprint on the doubles of its tools.
        cls.stage = control_tests.DriverTests
        cls.stage.setUpClass()
        cls.addClassCleanup(cls.release)
        assert cls.stage.reference["status"] == 0, cls.stage.reference["failed"]
        cls.control = cls.stage.reference["out"]
        cls.root = cls.stage.root / "s33"
        cls.root.mkdir()
        cls.settings = json.loads((ROOT / "configs/integration/pilot_features.json").read_text())
        # The small pair: 12 opportunities, two periods per member, a ring of four slots.
        cls.settings["spec"]["window"].update(periods=2, hop_periods=2)
        cls.settings["ring"].update(window_records=1, burst_records=1, slots=4)
        cls.settings["pair"]["expect"] = {"opportunities": 12, "complete_windows": 2, "filling": 2, "outside_sequence": 8}
        cls.settings["known_return"].update(doppler_bin=-1)
        cls.config = cls.write_config(cls.root / "config.json", cls.settings)
        cls.tool = cls.root / "tool"
        cls.tool.write_text("a double\n")
        cls.reference = cls.run_once(cls.root / "reference")

    @classmethod
    def release(cls):
        """Clean up every stage under this one: each holds what the stage below it made."""
        stage = cls.stage
        while stage is not None:
            stage.doClassCleanups()
            stage.tearDownClass()
            stage = getattr(stage, "stage", None)

    @staticmethod
    def write_config(path, settings):
        path.write_text(json.dumps(settings))
        return path

    @classmethod
    def receipt_of(cls):
        return {"executables": {name: {"path": str(cls.tool), "sha256": hashlib.sha256(cls.tool.read_bytes()).hexdigest()}
                                for name in ("pilot_ls", "obs_tail", "obs_feed", "replay_pusch")},
                "ocudu_build": "double", "pin": {"cuda": True}}

    @classmethod
    def run_once(cls, out, fault=None, config=None, control=None, extra=(), receipt=None, patch=None):
        driver = cls.driver
        saved = (driver.native_adapter.verified_receipt, driver.run_live)
        driver.native_adapter.verified_receipt = lambda build_dir: receipt or cls.receipt_of()
        driver.run_live = double.live
        kept = {key: os.environ.get(key) for key in ENVIRONMENT}
        for key in kept:
            os.environ.pop(key, None)
        if fault:
            os.environ["PILOT_FEATURES_FAULT"] = fault
        arguments = ["--config", str(config or cls.config), "--s32-run", str(control or cls.control),
                     "--output-dir", str(out), *extra]
        undo = patch() if patch else None
        try:
            with contextlib.redirect_stdout(io.StringIO()) as said:
                status = driver.main(arguments)
        finally:
            if undo:
                undo()
            driver.native_adapter.verified_receipt, driver.run_live = saved
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

    def changed_config(self, edit, name="config.json"):
        settings = json.loads(json.dumps(self.settings))
        edit(settings)
        return self.write_config(self.out / name, settings)

    def changed_control(self, edit):
        """A copy of the control run with one thing changed. Its plans name the Sprint 2.4 run by absolute path."""
        copy = self.out / "control"
        shutil.copytree(self.control, copy)
        edit(copy)
        return copy

    # ---------------------------------------------------------------- the reference
    def test_the_reference_passes_every_check(self):
        run = self.reference
        grade, lines, out = run["grade"], run["lines"], run["out"]
        self.assertEqual((run["status"], run["failed"]), (0, []), lines)
        self.assertEqual((grade["judgment"], grade["hardware_gate"], grade["cases"], grade["cases_passed"], grade["checks"]),
                         ("pass", "closed", CASES, CASES, CHECKS))
        self.assertIn(f"pilot_features: {CASES} of {CASES} cases passed, {CHECKS} of {CHECKS} checks\n", lines)
        for line in (
                "PASS  S3.3 control accepted run: the Sprint 3.2 driver graded reference as pass: 10 of 10 cases",
                "the native consumer of the timed run kept 12 records, each the record of its account line, and 12 of 12 "
                "payloads and validity masks, hashed here, have the SHA-256 of the file that the observer of the audited "
                "run wrote for the same opportunity",
                "PASS  S3.3 control episode entry: the channel plan names the cases a and b of the entry unit-pair",
                # The causal stream on the recorded ring.
                "PASS  S3.3 stream output: 12 records of the ring in sequence order: 2 complete windows, 2 answers of a "
                "filling window, 8 records outside the sequence, 0 gaps, 0 invalid, 0 rejected",
                "PASS  S3.3 stream windows: each of the 2 complete windows was answered at the record that completed it, "
                "from 2 samples of 2 observations with every element valid",
                "PASS  S3.3 stream registry: the steps of the specification are registry entries, each with its "
                "execution device: cube comb_cube, static window_mean, delay ifft, doppler fft, features dd_compact, all "
                "in NumPy",
                "NOTE  S3.3 stream axes of a window, signed: delay 32 bins of 54.47 ns, unambiguous to 16.67 us",
                # The same O2 from files, with every record in hand.
                "PASS  S3.3 offline eligible windows: 12 observations read from the files of the audited run and cut by "
                "slot arithmetic give 2 eligible windows of 2",
                "PASS  S3.3 offline feature values: the view and the vector of each window against explicit Fourier "
                "sums in double precision",
                # The return, against the geometry of the entry.
                "PASS  S3.3 known-return declared cell: the strongest moving cell of the target-present member lies at "
                "delay bin 3 and Doppler bin -1, 163.4 ns and -100.0 Hz",
                "PASS  S3.3 known-return one resolvable bin: the geometry of the entry gives a return at 179.8 ns and "
                "-43.3 Hz at opportunity 0, 112.9 ns behind the direct path; the peak is -16.4 ns and -56.7 Hz from it, "
                "within one bin of 54.5 ns for 18.36 MHz and of 100.0 Hz for 2 samples over 5 ms",
                "PASS  S3.3 known-return present against absent: the peak of the present member stands",
                "PASS  S3.3 known-return signed axis: the peak lies at -100.0 Hz, on the side of the -43.3 Hz that the "
                "geometry gives, and the axis of this window has no cell at bin +1 to compare with",
                "NOTE  S3.3 known-return recorded, not graded: the static part peaks at delay bin 1, where the direct "
                "path of 66.9 ns lies",
                # Causality and the mask.
                "PASS  S3.3 causality insufficient history: the records of the present member before the one that "
                "completes its window are answered insufficient_history 1 times, with 1 of 2 samples, and no answer "
                "carries a view or a vector",
                "PASS  S3.3 causality future samples: with the payload of the 2 records after the completing one "
                "replaced by 1e+30, the window is answered with the same line",
                "PASS  S3.3 causality input fields: with the host block, the grant key, the cell, the UE, the producer "
                "steps, the digest and the valid count of every record changed, the window is the same: a stream reads "
                "29 fields of the header",
                "PASS  S3.3 causality earlier epoch: 3 records of epoch 0 with a payload of 1e+30 fill 1 of 2 "
                "observations of a window; the first record of epoch 1 starts a new history",
                "PASS  S3.3 causality sensitivity: with the payload of the completing record scaled by 1.5 the vector "
                "changes",
                "PASS  S3.3 mask sentinel: 4 elements of observation 6 of the present window lose their validity bit; "
                "with 1e+30 stored behind them the line, the view and the vector are those with zero stored, bit for bit",
                "the window counts 4 holes on 4 lines of the cube and stays complete at a fill of 0.9984",
                "PASS  S3.3 mask valid support: on the valid support the return stays in its cell",
                "PASS  S3.3 mask missing observation: without observation 6 the history of that member starts one "
                "period later",
                # The live ring.
                "PASS  S3.3 live native run: both members of the pair in the mode timed on a ring of 4 slots, with the "
                "reference sensor attached in the mode wait: 12 observations, cpu 12 of 12 decoded, cuda 12 of 12 decoded",
                "PASS  S3.3 live decodes: 24 radio records equal those the accepted replay_pusch wrote without an "
                "observer",
                "PASS  S3.3 live executed path: 12 of 12 decodes requested from the CUDA processor ran its resident path "
                "with the sensor attached",
                "PASS  S3.3 live sensor: python -m ocudu_cuda_isac.sensing.features follow, a second process on the "
                "standard-library reader, read 12 records of epoch 1",
                "PASS  S3.3 live account: 12 eligible observations: 12 published and 0 dropped; of the published 12 "
                "released, 0 reclaimed and 0 pending at the cutoff",
                "PASS  S3.3 live same features: the 4 answers of the live sensor are those of the recorded stream of the "
                "accepted Sprint 3.2 run, line for line, host times aside",
                "2 views with the same SHA-256 and 2 vectors equal bit for bit",
                "NOTE  S3.3 live from the publication store of the completing record to the complete vector, not graded, "
                "runtime reference",
                "PASS  S3.3 refusals specification check: 15 of 15 faulty specifications are refused by the check",
                "PASS  S3.3 refusals command line: 15 of 15 are refused by python -m ocudu_cuda_isac.sensing.features "
                "replay with status 2 and one line, and the output directory stays empty"):
            self.assertIn(line, lines)
        self.assertNotIn("rehearsal", run["said"])
        # The evidence.
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual((manifest["selector"], manifest["axes"]["runtime"], manifest["axes"]["receiver_stage"],
                          manifest["host_rehearsal"]), ("pilot_features", "reference", "o2_pilot_ls", False))
        self.assertEqual({stage: entry["device"] for stage, entry in manifest["steps"].items()},
                         dict.fromkeys(("cube", "static", "delay", "doppler", "features"), "cpu"))
        self.assertEqual([(item["name"], item["mode"], item["consumer_statuses"]) for item in manifest["runs"]],
                         [("live", "timed", [0])])
        self.assertEqual(manifest["runs"][0]["consumers"][0]["reader"], "features")
        self.assertEqual(manifest["control"]["judgment"], "pass")
        self.assertEqual((grade["runtime"], grade["spec_sha256"]), ("reference", features.spec_sha256(self.settings["spec"])))
        for name in ("grade.json", "grade.txt", "comparison.json", "manifest.json", "adapter-receipt.json", "spec.json",
                     "known_return.json", "recover.json", "stream/sensor/features.jsonl", "stream/sensor/sensor.json",
                     "live/run/plan.json", "live/run/spec.json", "live/run/sensor/features.jsonl",
                     "live/run/sensor/records/consumer.json", "live/run/native/ring/publications.jsonl"):
            self.assertTrue((out / name).is_file(), name)
        plan = json.loads((out / "live/run/plan.json").read_text())
        self.assertEqual((plan["schema"], plan["evidence"], plan["ring"]["slots"], plan["ring"]["wait"],
                          plan["ring"]["attach"]), ("pilot-ls-plan/v2", {"mode": "timed"}, 4, "wait", "required"))
        comparison = json.loads((out / "comparison.json").read_text())
        self.assertLess(comparison["offline"]["largest_cell_difference"], 1e-12)
        self.assertEqual(comparison["known-return"]["present"]["peak"]["delay_bin"], 3)
        self.assertEqual(comparison["known-return"]["resolved"]["aliased"], False)

    def test_a_second_reader_recovers_the_present_window_of_the_live_sensor(self):
        out = self.reference["out"]
        recover = json.loads((out / "recover.json").read_text())
        self.assertEqual((recover["sensor"], recover["window"]), ("live/run/sensor", 1))
        live, stream = features.open_output(out / recover["sensor"]), features.open_output(out / "stream/sensor")
        shown = features.summary(live, recover["window"])
        self.assertEqual((shown["source"], shown["window"]["peak"]["delay_bin"], shown["window"]["contributing"]),
                         ("ring", 3, 2))
        self.assertEqual((out / recover["sensor"] / shown["window"]["view_file"]).read_bytes(),
                         (out / recover["stream_view"]).read_bytes())
        self.assertEqual(shown["window"]["view_sha256"], stream.ok[1]["view"]["sha256"])
        # The truth of the entry is written by the evaluator, apart from what the sensor wrote.
        for path in (out / "live/run/sensor").rglob("*"):
            if path.suffix in (".json", ".jsonl"):
                for word in ("position_m", "velocity_m_s", "expected_delay", "target_present", "realization"):
                    self.assertNotIn(word, path.read_text(), path.name)
        self.assertIn("expected_first", json.loads((out / "known_return.json").read_text()))

    # ---------------------------------------------------------------- faults of the live case
    def test_each_fault_of_the_live_case_fails_the_check_that_names_it(self):
        cases = (("future-leak", ["live same features"]),
                 ("slow-sensor", ["live account", "live same features"]),
                 ("sensor-exit", ["live sensor"]),
                 ("no-output", ["live sensor"]),
                 ("changed-decode", ["live decodes"]),
                 ("host-fallback", ["live decodes", "live executed path"]),
                 ("exit", ["live native run"]))
        for fault, expected in cases:
            run = self.faulty(fault)
            with self.subTest(fault=fault):
                self.assertEqual((run["status"], run["grade"]["judgment"], run["failed"]), (1, "fail", expected))
                self.assertEqual(run["grade"]["cases_passed"], CASES - 1)
        # Two samples that are the same hold no moving part: the leaked window is not even complete.
        self.assertIn("the live sensor and the recorded stream differ: line 1 differs in axes, features, reason, status",
                      self.run_once(self.out / "leak-again", fault="future-leak")["lines"])

    # ---------------------------------------------------------------- faults of the sensing side
    def test_a_wrong_sign_is_found_by_the_oracle_and_by_the_geometry(self):
        def patch():
            real = features.view_of
            features.view_of = lambda spec, cube, valid, axes: real(spec, cube.conj(), valid, axes)
            return lambda: setattr(features, "view_of", real)

        run = self.run_once(self.out / "sign", patch=patch)
        self.assertEqual(run["status"], 1)
        for check in ("offline feature values", "known-return declared cell", "known-return one resolvable bin"):
            self.assertIn(check, run["failed"])
        # The stream agrees with itself, so the cases that compare it with itself still pass.
        for check in ("stream output", "live same features", "causality future samples"):
            self.assertNotIn(check, run["failed"])

    def test_a_stream_that_reads_a_masked_value_or_a_hidden_field_is_found(self):
        def unmasked():
            real = features.planes

            def planes(record):
                values, valid = real(record)
                return values, np.ones(valid.shape, dtype=bool)
            features.planes = planes
            return lambda: setattr(features, "planes", real)

        run = self.run_once(self.out / "unmasked", patch=unmasked)
        self.assertEqual((run["status"], run["failed"]), (1, ["mask sentinel"]))

        def leaky():
            real = features.planes

            def planes(record):
                values, valid = real(record)
                return values * (1.0 + 0.01 * (record.header["cell"] % 7)), valid
            features.planes = planes
            return lambda: setattr(features, "planes", real)

        run = self.run_once(self.out / "leaky", patch=leaky)
        self.assertEqual(run["status"], 1)
        self.assertIn("causality input fields", run["failed"])

    # ---------------------------------------------------------------- the configuration grades
    def test_the_expectations_of_the_configuration_are_graded(self):
        cases = ((lambda settings: settings["known_return"].update(delay_bin=4), ["known-return declared cell"]),
                 (lambda settings: settings["known_return"].update(min_peak_to_median_db=40.0),
                  ["known-return present against absent"]),
                 (lambda settings: settings["known_return"].update(max_absent_peak_to_median_db=1.0),
                  ["known-return present against absent"]),
                 (lambda settings: settings["tolerance"].update(power_relative=1e-18), ["offline feature values"]),
                 (lambda settings: settings["pair"]["expect"].update(complete_windows=3), ["stream output"]))
        for number, (edit, expected) in enumerate(cases):
            run = self.run_once(self.out / f"run-{number}", config=self.changed_config(edit, f"config-{number}.json"))
            with self.subTest(expected=expected):
                self.assertEqual((run["status"], run["failed"]), (1, expected))

    # ---------------------------------------------------------------- the control
    def test_a_changed_control_grades_nothing_else(self):
        def not_accepted(copy):
            grade = json.loads((copy / "grade.json").read_text())
            grade.update(judgment="fail", checks_failed=1)
            (copy / "grade.json").write_text(json.dumps(grade))

        run = self.run_once(self.out / "failed", control=self.changed_control(not_accepted))
        self.assertEqual(run["status"], 1)
        self.assertEqual(run["failed"], ["control accepted run"])
        self.assertIn("this gate needs the judgment pass with the hardware gate closed", run["lines"])
        shutil.rmtree(self.out / "control")

        def other_estimate(copy):
            path = sorted((copy / "timed/run/consumer-0").glob("*.obs"))[5]
            data = bytearray(path.read_bytes())
            data[-1] ^= 1
            path.write_bytes(bytes(data))

        run = self.run_once(self.out / "estimate", control=self.changed_control(other_estimate))
        self.assertEqual(run["status"], 1)
        self.assertIn("control records", run["failed"])
        self.assertIn("11 of 12 payloads and validity masks", run["lines"])
        for case in ("stream", "offline", "known-return", "causality", "mask", "live", "refusals"):
            self.assertIn(f"{case} control", run["failed"])
        self.assertFalse((self.out / "estimate/live").exists())
        run = self.run_once(self.out / "nothing", control=self.out)
        self.assertEqual((run["status"], run["grade"]["cases_passed"]), (1, 0))

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

    def test_a_faulty_configuration_or_adapter_stops_before_anything_is_graded(self):
        cases = (
            (lambda settings: settings.update(selector="snapshot_ring"), "for the selector pilot_features"),
            (lambda settings: settings.update(backends=["cpu"]), "decodes every grid with both"),
            (lambda settings: settings["ring"].update(slots=8), "ring.slots is 8"),
            (lambda settings: settings["ring"].update(wait="poll"), "ring requires stream"),
            (lambda settings: settings["control"].update(driver="pilot_ls"), "the driver snapshot_ring"),
            (lambda settings: settings["spec"]["steps"].update(doppler="music"), "unknown doppler step"),
            (lambda settings: settings["tolerance"].update(power_relative=0), "each a positive number"),
            (lambda settings: settings["known_return"].pop("min_mirror_db"), "known_return requires"),
            (lambda settings: settings["mask"].update(elements=0), "mask requires elements"),
            (lambda settings: settings["pair"].pop("expect"), "pair requires expect"),
        )
        for number, (edit, message) in enumerate(cases):
            path = self.changed_config(edit, f"config-{number}.json")
            with self.assertRaises(SystemExit) as stopped, contextlib.redirect_stderr(io.StringIO()) as said:
                self.run_once(self.out / f"run-{number}", config=path)
            self.assertEqual(stopped.exception.code, 2)
            self.assertIn(message, said.getvalue())
            self.assertFalse((self.out / f"run-{number}").exists())
        receipt = self.receipt_of()
        del receipt["executables"]["pilot_ls"]
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()) as said:
            self.run_once(self.out / "no-tool", receipt=receipt)
        self.assertIn("the adapter build holds no pilot_ls", said.getvalue())

    def test_the_shipped_configuration_is_the_demonstration(self):
        shipped = self.driver.load_config(ROOT / "configs/integration/pilot_features.json", features)
        self.assertEqual(shipped["spec"], features.demo_spec())
        self.assertEqual((shipped["known_return"]["delay_bin"], shipped["known_return"]["doppler_bin"]), (3, -2))
        self.assertEqual(shipped["pair"]["expect"], {"opportunities": 48, "complete_windows": 2, "filling": 14,
                                                     "outside_sequence": 32})
        self.assertEqual((shipped["ring"]["slots"], shipped["ring"]["wait"]), (8, "wait"))


if __name__ == "__main__":
    unittest.main()
