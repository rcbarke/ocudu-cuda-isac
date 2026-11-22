#!/usr/bin/env python3
"""Sprint 3.6 unit gate (selector sense_core): python tests/unit/test_sense_core.py -v

The native sensing core beside its reference. The gate builds the host twin of the core into a
scratch directory with scripts/native_core.py, which runs the unit cases of the core in CTest,
and then runs the vectors of the sprint card and whole record sequences through the binding and
through sense_host, each against ocudu_cuda_isac.sensing.cusense:

  the vectors of the card through the units of the core, within the declared FP32 tolerance
  element [1][3][2][5] of an ocudu_o3_fp16 payload, and the same values as aerial_hest_c64
  a slot at PRB offset 2, a gap, a rejected record, an invalid window and the optional stage
  the window mean bit for bit, the input within its bound, the scores within tolerance
  the output of sense_host from an archive and from a live ring, verified by cusense.open_output
  refusals: an altered artifact, another artifact than the freeze names, an engine of another host
  the ONNX export and the TorchScript module of the artifact against the reference model

It needs cmake, a C++17 compiler, NumPy, PyTorch and pybind11, and no CUDA: the kernels run as
their host twin, which is compiled from the same units. The cases of the kernels on a device are
those of the build with CUDA (sense_core_device_vectors) and of the integration driver.
"""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

try:
    import numpy as np
    import torch
except ImportError:  # the reference of the branch is NumPy and PyTorch
    np = torch = None

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import native_core  # noqa: E402
import sense_core_build  # noqa: E402

if np is not None:
    from ocudu_cuda_isac.episodes import sense_core as evaluator  # noqa: E402
    from ocudu_cuda_isac.sensing import cusense, export, obs, runtime  # noqa: E402
    from ocudu_cuda_isac.sensing.cusense import CuSenseError  # noqa: E402
    import sense_core_double as double  # noqa: E402
    import test_cusense_frontend as fixtures  # noqa: E402

BUILD = CORE = PROBLEM = None
SMOOTHING = {"entry": "map_mean_kalman", "maps": 3, "process_noise_m2_s3": 1.0, "measurement_noise_m2": 0.25,
             "initial_speed_m_s": 1.5}


def setUpModule():
    global BUILD, CORE, PROBLEM
    if np is None:
        PROBLEM = "the reference of the cuSense-like branch is NumPy and PyTorch"
        return
    BUILD, PROBLEM = sense_core_build.host_twin()
    if PROBLEM is not None:
        return
    try:
        CORE = runtime.load(BUILD)
    except runtime.RuntimeUnavailable as error:
        PROBLEM = str(error)


class CoreCase(unittest.TestCase):
    def setUp(self):
        if PROBLEM is not None:
            self.skipTest(PROBLEM)
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.scratch = Path(scratch.name)
        self.tolerance = runtime.tolerance(CORE)["fp32"]

    def plan(self, artifact, **options):
        return CORE.Plan(str(artifact.directory), device="host", engine="host_reference", **options)

    def host(self, *arguments):
        return subprocess.run([str(BUILD / "sense_host")] + [str(item) for item in arguments], capture_output=True,
                              text=True, timeout=300)


class BuildTests(CoreCase):
    def test_the_build_ran_the_unit_cases_of_the_core_and_names_its_sources(self):
        receipt = native_core.verified_receipt(BUILD)
        self.assertGreaterEqual(receipt["tests"]["ran"], 40)
        self.assertEqual((receipt["tests"]["failed"], receipt["tests"]["skipped"], receipt["device"]), (0, [], "host"))
        said = runtime.build(CORE)
        self.assertEqual(said["sources_sha256"], native_core.sources_sha256())
        self.assertEqual(said["sources_sha256"], json.loads(self.host("--build").stdout)["sources_sha256"])
        self.assertEqual((said["cuda"], said["tensorrt"], said["api_version"]), (False, None, CORE.API_VERSION))

    def test_the_registry_names_a_unit_for_every_entry_of_the_reference(self):
        units = {(item["stage"], item["entry"]): item for item in runtime.registry(CORE)}
        for stage, entries in cusense.REGISTRY.items():
            for entry in entries:
                self.assertIn((stage, entry), units)
                self.assertEqual((units[(stage, entry)]["runtime"], len(units[(stage, entry)]["unit_sha256"])), ("native", 64))
        self.assertIn(("ingest", "aerial_hest_c64"), units)
        # The tolerances of the engine are those of the Sprint 3.5 parity, and FP16 is declared apart.
        declared = runtime.tolerance(CORE)
        for name in ("logit_absolute", "logit_relative", "probability_absolute", "map_absolute"):
            self.assertEqual(declared["fp32"][name], evaluator.cusense_frontend.TOLERANCE[name])
        self.assertEqual(declared["fp16"]["payload_relative"], 2.0 ** -11)


class VectorTests(CoreCase):
    """The vectors of the sprint card through the units of the core."""

    def test_the_window_of_the_card_gives_0_5_plus_0_5j_then_0_2071_then_0_071(self):
        spec = fixtures.small_spec(subcarriers=24, rx=1)
        shape = cusense.shape_of(spec)
        template = {"values": np.full(shape, 0.5, dtype="<c8"), "count": np.full(shape, 4, dtype="<u4")}
        artifact = fixtures.make_artifact(self.scratch / "artifact", spec=spec, template=template, mean=0.2, scale=0.1)
        # 1+0j and 0+1j, and a third slot whose samples are not valid: zero by its magnitude.
        records = [fixtures.o3_record(np.full(shape, value), slot, index)
                   for index, (slot, value) in enumerate(((7, 1 + 0j), (8, 1j), (9, 0j)))]
        plan = self.plan(artifact)
        lines, _, _ = runtime.score(plan, records)
        decision = lines[-1][1]
        self.assertEqual([line[0] for line in lines], ["answer", "answer", "decision"])
        self.assertTrue(np.all(decision["mean"] == np.complex64(0.5 + 0.5j)) and np.all(decision["count"] == 2))
        relative = self.tolerance["vector_relative"]
        self.assertTrue(np.allclose(np.abs(decision["mean"]), 0.7071067811865476, rtol=relative, atol=0))
        self.assertTrue(np.allclose(decision["difference"], 0.2071067811865476, rtol=relative, atol=0))
        wanted = (0.2071067811865476 - 0.2) / (0.1 + 1e-8)
        bound = self.tolerance["input_ulps"] * self.tolerance["spacing"] * 0.7071067811865476 / (0.1 + 1e-8)
        self.assertAlmostEqual(wanted, 0.0710678, places=6)
        self.assertLessEqual(float(np.abs(decision["input"] - wanted).max()), bound)
        # The reference on the same records: the same mean, and the input within the declared bound.
        _, stream = cusense.run_records(records, spec)
        mean, count = stream.last
        self.assertTrue(np.array_equal(decision["mean"], mean) and np.array_equal(decision["count"], count))
        x = cusense.Model(artifact).prepare(mean, count)[0]
        allowed = evaluator.input_bound(x, template["values"], template["count"], artifact.manifest["normalizer"],
                                        self.tolerance)
        self.assertTrue(np.all(np.abs(decision["input"].astype(np.float64) - x) <= allowed))

    def test_the_mean_of_two_valid_samples_and_an_invalid_zero_is_2(self):
        spec = fixtures.small_spec(subcarriers=24, rx=1)
        artifact = fixtures.make_artifact(self.scratch / "artifact", spec=spec)
        records = [fixtures.o3_record(np.full((1, 24, 2), value), slot, index)
                   for index, (slot, value) in enumerate(((7, 1 + 1j), (8, 3 - 1j), (9, 0j)))]
        decision = runtime.score(self.plan(artifact), records)[0][-1][1]
        self.assertTrue(np.all(decision["mean"] == 2 + 0j) and np.all(decision["count"] == 2))
        self.assertEqual((decision["records_min"], decision["records_max"], decision["valid"]), (2, 2, 48))

    def test_element_1_3_2_5_of_an_o3_payload_is_symbol_1_port_2_subcarrier_41(self):
        spec = fixtures.small_spec(subcarriers=48, rx=4)
        artifact = fixtures.make_artifact(self.scratch / "artifact", spec=spec)
        record = fixtures.o3_record(np.zeros((4, 48, 2)), 7, 0, valid=np.zeros((4, 48, 2), dtype=bool))
        payload, mask = bytearray(record.payload), bytearray(record.mask)
        element = ((1 * 4 + 3) * 4 + 2) * 12 + 5   # the C order of [2][4][4][12]
        payload[4 * element:4 * element + 4] = np.array([0.75, -1.25], dtype="<f2").tobytes()
        mask[element // 8] |= 1 << (element % 8)
        marked = obs.Record(record.header, b"", bytes(mask), bytes(payload))
        cube, valid, first = self.plan(artifact).ingest(marked.to_bytes())
        self.assertEqual((cube.shape, first, int(valid.sum())), ((4, 48, 2), 0, 1))
        self.assertEqual((int(valid[2, 41, 1]), complex(cube[2, 41, 1])), (1, 0.75 - 1.25j))
        # The reference reader places the same element at the same position.
        values, good = cusense.planes(marked, spec)
        self.assertTrue(good[2, 41, 1] and good.sum() == 1 and complex(values[2, 41, 1]) == 0.75 - 1.25j)

    def test_the_same_values_in_the_three_layouts_give_one_cube(self):
        spec = fixtures.small_spec(subcarriers=48, rx=4)
        artifact = fixtures.make_artifact(self.scratch / "artifact", spec=spec)
        generator = np.random.Generator(np.random.PCG64(11))
        values = (generator.integers(-900, 900, (4, 48, 2)) + 1j * generator.integers(-900, 900, (4, 48, 2))) / 1024.0
        valid = generator.random((4, 48, 2)) > 0.2
        plan = self.plan(artifact)
        o3 = fixtures.o3_record(values, 7, 0, valid=valid)
        cube, bits, _ = plan.ingest(o3.to_bytes())
        self.assertTrue(np.array_equal(bits.astype(bool), valid) and np.array_equal(cube[valid], values.astype("<c8")[valid]))
        self.assertFalse(cube[~valid].any())
        # aerial_hest_c64 is [dmrs][subcarrier][rx_antenna][layer] of one layer, complex64.
        ordered, flags = values.transpose(2, 1, 0), valid.transpose(2, 1, 0).reshape(-1)
        payload, mask = ordered.astype("<c8").tobytes(), np.packbits(flags, bitorder="little").tobytes()
        header = fixtures.header_of(7, 0, [2, 48, 4, 1], [48 * 4, 4, 1, 1], 3, 4, 1, 7, 1, 48, payload, mask,
                                    int(flags.sum()))
        header["rx_count"] = 4
        aerial, aerial_bits, _ = plan.ingest(obs.Record(header, b"", mask, payload).to_bytes())
        self.assertTrue(np.array_equal(aerial, cube) and np.array_equal(aerial_bits, bits))
        comb, comb_bits, _ = plan.ingest(fixtures.o2_record(values, 7, 0, valid=valid).to_bytes())
        self.assertTrue(np.array_equal(comb, cube) and np.array_equal(comb_bits, bits))

    def test_a_slot_at_prb_offset_2_lands_on_subcarrier_24(self):
        spec = fixtures.small_spec(subcarriers=48, rx=2)
        artifact = fixtures.make_artifact(self.scratch / "artifact", spec=spec)
        records = [fixtures.o3_record(np.full((2, 48, 2), 0.25), 7, 0),
                   fixtures.o3_record(np.full((2, 12, 2), 1.0), 8, 1, first_subcarrier=24),
                   fixtures.o3_record(np.full((2, 48, 2), 0.25), 9, 2)]
        lines, _, _ = runtime.score(self.plan(artifact), records)
        decision = lines[-1][1]
        self.assertEqual([(item["first_position"], item["positions"]) for item in decision["contributing"]],
                         [(0, 48), (24, 12), (0, 48)])
        inside = np.zeros(48, dtype=bool)
        inside[24:36] = True
        self.assertTrue(np.all(decision["count"][:, inside, :] == 3) and np.all(decision["count"][:, ~inside, :] == 2))
        self.assertTrue(np.all(decision["mean"][:, inside, :] == 0.5) and np.all(decision["mean"][:, ~inside, :] == 0.25))
        _, stream = cusense.run_records(records, spec)
        self.assertTrue(np.array_equal(decision["mean"], stream.last[0]) and np.array_equal(decision["count"], stream.last[1]))


class ParityTests(CoreCase):
    """Whole record sequences through the core and through the reference."""

    def sequence(self):
        generator = np.random.Generator(np.random.PCG64(7))
        records = fixtures.periods_of(5, generator, cell=(1, 2))
        del records[7]                                                  # slot 8 of period 2 never arrives: a gap
        records.append(fixtures.o2_record(np.ones((2, 24, 2)), 57, 100))  # another stage: rejected
        few = np.zeros((2, 48, 2), dtype=bool)
        few[:, :10, :] = True
        records += [fixtures.o3_record(fixtures.channel(generator), slot, 200 + slot, valid=few) for slot in (67, 68, 69)]
        records += fixtures.periods_of(2, generator, episode=0x77, first_sequence=400)  # another episode: a new history
        return records

    def test_the_core_answers_as_the_reference_and_scores_within_tolerance(self):
        artifact = fixtures.make_artifact(self.scratch / "artifact", mean=0.1, scale=0.3, smoothing=SMOOTHING)
        records = self.sequence()
        lines, stream, decisions, inputs, maps = cusense.run_model(records, cusense.Model(artifact))
        plan = self.plan(artifact)
        native, native_inputs, native_maps = runtime.score(plan, records)
        self.assertEqual([line[1].get("status") or line[1]["kind"] for line in native], [line["status"] for line in decisions])
        self.assertEqual({line["status"] for line in decisions}, {"ok", "insufficient_history", "gap", "rejected", "invalid"})
        scored = [line for line in decisions if line["status"] == "ok"]
        mine = [line for line in native if line[0] == "decision" and line[1]["status"] == "ok"]
        self.assertEqual(len(mine), len(scored))
        manifest, tolerance = artifact.manifest, self.tolerance
        for index, (wanted, (_, decision, smoothed)) in enumerate(zip(scored, mine)):
            _, mean, count = stream.tensors[index]
            self.assertTrue(np.array_equal(decision["mean"], mean) and np.array_equal(decision["count"], count))
            bound = evaluator.input_bound(inputs[index], artifact.template["values"], artifact.template["count"],
                                          manifest["normalizer"], tolerance)
            self.assertTrue(np.all(np.abs(native_inputs[index].astype(np.float64) - inputs[index]) <= bound))
            self.assertLessEqual(abs(decision["logit"] - wanted["score"]["logit"]),
                                 tolerance["logit_absolute"] + tolerance["logit_relative"] * abs(wanted["score"]["logit"]))
            self.assertLessEqual(abs(decision["probability"] - wanted["score"]["probability"]), tolerance["probability_absolute"])
            self.assertLessEqual(float(np.abs(native_maps[index] - maps[index]).max()), tolerance["map_absolute"])
            self.assertEqual((decision["present"], list(decision["cell"]), list(decision["position_m"])),
                             (wanted["decision"]["present"], wanted["location"]["cell"], wanted["location"]["position_m"]))
            self.assertEqual((smoothed["maps"], list(smoothed["cell"])), (wanted["smoothed"]["maps"], wanted["smoothed"]["cell"]))
            for mine_value, reference_value in zip(smoothed["kalman_position_m"], wanted["smoothed"]["kalman_position_m"]):
                self.assertAlmostEqual(mine_value, reference_value, delta=1e-6)
        counters = plan.counters()
        self.assertEqual((counters["device_calls_open"], counters["views_held"], counters["submissions"]),
                         (0, 0, len(scored) + 1))  # the window with too few samples ran and was not scored
        self.assertEqual(stream.counts["resets"], counters["resets"])

    def test_the_pilot_comb_is_a_second_input_of_the_same_units(self):
        spec = fixtures.small_spec("o2_comb", subcarriers=24, rx=2)
        artifact = fixtures.make_artifact(self.scratch / "artifact", spec=spec, steps=("pilot_ls",))
        generator = np.random.Generator(np.random.PCG64(3))
        records = [fixtures.o2_record(fixtures.channel(generator, positions=24), 10 * period + slot, 3 * period + index)
                   for period in range(3) for index, slot in enumerate((7, 8, 9))]
        lines, stream, decisions, inputs, maps = cusense.run_model(records, cusense.Model(artifact))
        native, native_inputs, native_maps = runtime.score(self.plan(artifact), records)
        self.assertEqual(len(native_inputs), 3)
        for index in range(3):
            # Three complex64 samples do not add exactly in FP32, as FP16 samples do: the mean of the
            # comb agrees within the declared number of FP32 spacings at its magnitude.
            mean, wanted = native[3 * index + 2][1]["mean"].astype(np.complex128), stream.tensors[index][1]
            allowed = self.tolerance["mean_ulps"] * self.tolerance["spacing"] * np.abs(wanted)
            self.assertTrue(np.all(np.abs(mean.real - wanted.real) <= allowed) and np.all(np.abs(mean.imag - wanted.imag) <= allowed))
            self.assertTrue(np.array_equal(native[3 * index + 2][1]["count"], stream.tensors[index][2]))
            self.assertLessEqual(float(np.abs(native_maps[index] - maps[index]).max()), self.tolerance["map_absolute"])

    def test_three_records_are_one_submission_and_no_device_call_before_the_third(self):
        artifact = fixtures.make_artifact(self.scratch / "artifact")
        plan = self.plan(artifact)
        plan.warm_up(1)
        records = fixtures.periods_of(1, np.random.Generator(np.random.PCG64(1)))
        before = plan.counters()
        for index, record in enumerate(records):
            answers = plan.offer(record.to_bytes())
            now = plan.counters()
            self.assertEqual((now["device_calls"], now["submissions"], now["views_held"]),
                             (before["device_calls"], 0, index + 1))
            self.assertEqual(answers[0]["kind"], "ready" if index == 2 else "insufficient_history")
        self.assertTrue(plan.ready())
        self.assertEqual(plan.run()["status"], "ok")
        after = plan.counters()
        self.assertEqual((after["submissions"], after["device_calls_open"], after["views_held"],
                          after["device_allocations"]), (1, 0, 0, before["device_allocations"]))
        self.assertGreater(after["device_calls"], before["device_calls"])


class SensorTests(CoreCase):
    """sense_host, the sensor process, on an archive and on a ring."""

    def setUp(self):
        super().setUp()
        self.artifact = fixtures.make_artifact(self.scratch / "artifact", mean=0.1, scale=0.3, smoothing=SMOOTHING)
        self.records = ParityTests.sequence(self)
        self.archive = double.write_archive(self.scratch / "archive", self.records)

    def test_an_archive_gives_the_lines_of_the_reference_and_verifies(self):
        out = self.scratch / "native"
        out.mkdir()
        done = self.host("--archive", self.archive, "--artifact", self.artifact.directory, "--output-dir", out, "--device",
                         "host", "--engine", "host_reference", "--evidence", "audit", "--expect-sha256", self.artifact.sha256)
        self.assertEqual(done.returncode, 0, done.stderr)
        output = cusense.open_output(out)
        self.assertEqual((output.report["runtime"]["entry"], output.report["runtime"]["timing_valid"], output.report["loads"]),
                         ("native", False, 1))
        lines, stream, decisions, inputs, maps = cusense.run_model(self.records, cusense.Model(self.artifact))
        # Every window line is that of the reference but for its runtime entry: the digest of the mean as well.
        for mine, wanted in zip(output.windows.lines, lines):
            self.assertEqual(dict(cusense.comparable_window(mine), runtime="reference"), cusense.comparable_window(wanted))
        self.assertEqual(len(output.decisions), len(decisions))
        for mine, wanted in zip(output.decisions, decisions):
            for key in ("line", "status", "reason", "window", "trigger", "model", "smoothed"):
                if key != "smoothed" or wanted["smoothed"] is None:
                    self.assertEqual(mine[key], wanted[key], key)
            if wanted["status"] == "ok":
                self.assertEqual((mine["decision"], mine["location"]["cell"], mine["contributing"], mine["time"]),
                                 (wanted["decision"], wanted["location"]["cell"], wanted["contributing"], wanted["time"]))
        # The evaluator of the gate on the two outputs: no problem, every decision agrees.
        reference = self.scratch / "reference"
        reference.mkdir()
        (reference / "windows").mkdir()
        source = {"kind": "archive", "directory": str(self.archive), "reader": "reference", "records": len(self.records)}
        model = cusense.Model(self.artifact)
        lines, stream, decisions, inputs, maps = cusense.run_model(self.records, model)
        cusense.write_windows(reference / "windows", model.spec, lines, stream, source)
        cusense.write_output(reference, model, decisions, inputs, maps, source)
        problems, facts = evaluator.parity(output, cusense.open_output(reference), self.artifact, self.tolerance)
        self.assertEqual(problems, [])
        self.assertEqual((facts["decisions"]["agree"], facts["decisions"]["flips"], facts["joined"]),
                         (len(output.scored), 0, len(output.scored)))
        self.assertLessEqual(facts["largest_input_difference_of_its_bound"], 1.0)
        # A score that moved is found.
        text = (reference / "decisions.jsonl").read_text().splitlines()
        moved = json.loads(text[2])
        moved["score"]["logit"] += 0.01
        moved["score"]["probability"] = 1.0 / (1.0 + np.exp(-moved["score"]["logit"]))
        moved["decision"]["present"] = moved["score"]["logit"] > moved["decision"]["threshold_logit"]
        text[2] = json.dumps(moved, sort_keys=True)
        (reference / "decisions.jsonl").write_text("\n".join(text) + "\n")
        problems, _ = evaluator.parity(output, cusense.open_output(reference), self.artifact, self.tolerance)
        self.assertTrue(any("the logit differs" in problem for problem in problems), problems)

    def test_timed_evidence_names_no_digest_of_the_mean_and_only_the_native_runtime_may(self):
        out = self.scratch / "native"
        out.mkdir()
        done = self.host("--archive", self.archive, "--artifact", self.artifact.directory, "--output-dir", out, "--device",
                         "host", "--engine", "host_reference", "--evidence", "timed")
        self.assertEqual(done.returncode, 0, done.stderr)
        output = cusense.open_output(out)
        self.assertTrue(all(line["mean"]["sha256"] is None for line in output.windows.ok))
        self.assertTrue(all(len(line["input"]["sha256"]) == 64 for line in output.scored))
        # The same lines under the reference entry are refused: the reference always names its digests.
        path = out / "windows" / "windows.json"
        report = json.loads(path.read_text())
        report["source"]["evidence"] = "audit"
        path.write_text(json.dumps(report))
        with self.assertRaises(CuSenseError):
            cusense.open_output(out)
        # A decision that was altered after the run does not verify either.
        report["source"]["evidence"] = "timed"
        path.write_text(json.dumps(report))
        text = (out / "decisions.jsonl").read_text().splitlines()
        index = next(number for number, line in enumerate(text) if '"status": "ok"' in line)
        altered = json.loads(text[index])
        altered["decision"]["present"] = not altered["decision"]["present"]
        text[index] = json.dumps(altered)
        (out / "decisions.jsonl").write_text("\n".join(text) + "\n")
        with self.assertRaises(CuSenseError):
            cusense.open_output(out)

    def live(self, host_options, feed_options):
        out, socket, ready = self.scratch / "live", self.scratch / "obs.sock", self.scratch / "ready.json"
        out.mkdir()
        sensor = subprocess.Popen([str(BUILD / "sense_host"), "--socket", str(socket), "--artifact",
                                   str(self.artifact.directory), "--output-dir", str(out), "--device", "host", "--engine",
                                   "host_reference", "--ready-file", str(ready)] + [str(item) for item in host_options],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        deadline = time.monotonic() + 60
        while not ready.exists() and sensor.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(ready.exists(), "the sensor was not warm in time")
        feed = subprocess.run([str(BUILD / "sense_feed"), "--socket", str(socket), "--archive", str(self.archive)]
                              + [str(item) for item in feed_options], capture_output=True, text=True, timeout=120)
        said = sensor.communicate(timeout=120)
        self.assertEqual((feed.returncode, sensor.returncode), (0, 0), feed.stderr + said[1])
        return cusense.open_output(out), json.loads(feed.stdout.strip().splitlines()[-1]), json.loads(ready.read_text())

    def test_a_live_ring_gives_the_decisions_of_the_archive_with_their_marks(self):
        output, fed, ready = self.live(["--evidence", "timed", "--keep-records", "--budget-us", "0,0,0,200000"], [])
        self.assertEqual((fed["dropped"], fed["pending_at_cutoff"], fed["published"]), (0, 0, len(self.records)))
        self.assertEqual((ready["device"], ready["engine_backend"], ready["artifact_sha256"]),
                         ("host", "host_reference", self.artifact.sha256))
        lines, stream, decisions, inputs, maps = cusense.run_model(self.records, cusense.Model(self.artifact))
        self.assertEqual([line["status"] for line in output.decisions], [line["status"] for line in decisions])
        for line in output.scored:
            marks = line["marks"]
            self.assertTrue(line["t_ready_ns"] <= marks["t_notify_ns"] <= marks["t_submit_ns"] <= marks["t_ingest_ns"]
                            <= marks["t_kernels_ns"] <= marks["t_engine_ns"] <= line["t_decision_ns"])
            self.assertEqual(line["deadline"]["outcome"], "on_time")
        counts = output.report["counts"]
        self.assertEqual((counts["device_calls_in_an_open_window"], counts["stale_skipped"], counts["late"]), (0, 0, 0))
        # The records it read are kept as a consumer archive that the reference reader opens.
        # The carrier sets the epoch, the sequence and the two host times of a record, and nothing else.
        archive = obs.open_archive(output.directory / "windows" / "records")
        def content(record):
            header = dict(record.header, epoch=0, sequence=0, t_ready_ns=0, t_publish_ns=0)
            return obs.Record(header, b"", record.mask, record.payload).to_bytes()

        self.assertEqual(archive.report["reader"], "native")
        self.assertEqual([content(record) for record in archive.records], [content(record) for record in self.records])
        latency = evaluator.latency(output, clock_map="ready_is_last_sample")
        self.assertEqual((latency["cusense_boundary"]["count"], latency["post_window"]["count"], latency["timing_valid"]),
                         (len(output.scored), len(output.scored), False))
        self.assertGreater(latency["cusense_boundary"]["p50_us"], 0)
        self.assertGreater(latency["end_to_end"]["p50_us"], latency["post_window"]["p50_us"])

    def test_a_producer_that_is_faster_than_the_sensor_drops_and_never_waits(self):
        output, fed, _ = self.live(["--evidence", "timed", "--hold-us", "3000", "--budget-us", "0,0,0,1"],
                                   ["--pace", "none", "--slots", "4", "--repeat", "3"])
        self.assertGreater(fed["dropped"], 0)
        self.assertEqual((fed["admitted"] + fed["dropped"], fed["pending_at_cutoff"], fed["window_calls"], fed["faults"]),
                         (3 * len(self.records), 0, 0, 0))
        self.assertEqual(fed["released"], fed["published"])
        self.assertLess(fed["duration_ns"], 1e9)  # an unpaced producer: it offered every record at once
        counts = output.report["counts"]
        self.assertEqual(counts["late"], counts["scored"])
        self.assertTrue(all(line["deadline"]["outcome"] == "late" and line["deadline"]["overrun"] == ["window"]
                            for line in output.scored))

    def test_the_sensor_refuses_what_is_not_the_frozen_artifact(self):
        out = self.scratch / "out"
        out.mkdir()
        common = ["--archive", self.archive, "--output-dir", out, "--device", "host", "--engine", "host_reference"]
        done = self.host(*common, "--artifact", self.artifact.directory, "--expect-sha256", "0" * 64)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("was asked for", done.stderr)
        altered = self.scratch / "altered"
        shutil.copytree(self.artifact.directory, altered)
        data = bytearray((altered / "weights.f32").read_bytes())
        data[100] ^= 1
        (altered / "weights.f32").write_bytes(bytes(data))
        done = self.host(*common, "--artifact", altered)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("the checkpoint was altered", done.stderr)
        with self.assertRaises(CORE.SenseError):
            CORE.Plan(str(altered), device="host", engine="host_reference")
        # An engine of a GPU is not run by the host twin, and a build without a backend says so.
        with self.assertRaises(CORE.SenseError):
            CORE.Plan(str(self.artifact.directory), device="host", engine="tensorrt", engine_dir=str(self.scratch))
        with self.assertRaises(CORE.SenseError):
            CORE.Plan(str(self.artifact.directory), device="cuda", engine="host_reference")
        self.assertFalse(any(out.iterdir()))


class ExportTests(CoreCase):
    def test_the_onnx_export_is_the_model_of_the_artifact(self):
        try:
            import onnx  # noqa: F401
        except ImportError:
            self.skipTest("the export needs the onnx package")
        artifact = fixtures.make_artifact(self.scratch / "artifact")
        (self.scratch / "onnx").mkdir()
        record = export.write_onnx(artifact, self.scratch / "onnx")
        self.assertEqual((record["artifact_sha256"], record["input"]["shape"], record["outputs"]),
                         (artifact.sha256, [1, 2, 48], {"map": [1, 12], "presence_logit": [1]}))
        self.assertEqual(export.check_onnx(self.scratch / "onnx", artifact)["sha256"], record["sha256"])
        other = fixtures.make_artifact(self.scratch / "other", seed=9)
        with self.assertRaises(export.ExportError):
            export.check_onnx(self.scratch / "onnx", other)
        with self.assertRaises(export.ExportError):
            export.write_onnx(artifact, self.scratch / "onnx")  # an export is never overwritten
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            self.skipTest("ONNX Runtime is not installed: the graph is checked on the device by the integration driver")
        model = cusense.Model(artifact)
        generator = np.random.Generator(np.random.PCG64(5))
        for _ in range(4):
            x = generator.standard_normal((1, 2, 48)).astype("<f4")
            drawn, logit, _, _ = model.predict(x)
            mine, mine_logit = export.run_onnx(self.scratch / "onnx" / record["file"], x)
            self.assertLessEqual(float(np.abs(mine.reshape(drawn.shape) - drawn).max()), self.tolerance["map_absolute"])
            self.assertLessEqual(abs(float(mine_logit[0]) - logit), self.tolerance["logit_absolute"])

    def test_the_module_of_the_declared_fallback_names_its_artifact(self):
        artifact = fixtures.make_artifact(self.scratch / "artifact")
        (self.scratch / "module").mkdir()
        made = export.write_module(artifact, self.scratch / "module")
        self.assertEqual((made["backend"], made["artifact_sha256"], made["library"]["version"], made["device"]["name"]),
                         ("libtorch", artifact.sha256, torch.__version__.split("+")[0], "any"))
        loaded = torch.jit.load(str(self.scratch / "module" / made["file"]))
        x = torch.zeros((1, 2, 48))
        drawn, logit = loaded(x)
        wanted, wanted_logit, _, _ = cusense.Model(artifact).predict(x.numpy())
        self.assertTrue(np.allclose(drawn.detach().numpy().reshape(wanted.shape), wanted, atol=1e-6))
        self.assertAlmostEqual(float(logit.detach()[0]), wanted_logit, places=5)
        # This build holds no LibTorch backend: the plan says so and loads nothing.
        with self.assertRaises(CORE.SenseError):
            CORE.Plan(str(artifact.directory), device="host", engine="libtorch", engine_dir=str(self.scratch / "module"))


if __name__ == "__main__":
    unittest.main()
