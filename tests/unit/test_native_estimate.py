"""Sprint 5.2 unit gate (selector native_estimate): python tests/unit/test_native_estimate.py

The receiver's own channel estimate, the stage O3. This gate checks, without an OCUDU build and
without a CUDA device:

  * the reference of the stage: FP16, the layout, the matched estimator and the parity rules;
  * the plan of version 3 and the rules of an estimate;
  * the host-free capture core of native/ocudu_adapter/estimate/, built on its own with a fake
    asynchronous producer: only a completed copy of the expected estimate is published;
  * that the native core and the reference convert FP16 alike and index the layout alike;
  * the tree: the receiver patch, the device calls of the route and the strings both sides share.
"""

import copy
import json
import math
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
sys.path.insert(0, str(ROOT / "scripts"))
from ocudu_cuda_isac.episodes import native_estimate as ne  # noqa: E402
from ocudu_cuda_isac.episodes import snapshot_ring as sr  # noqa: E402
from ocudu_cuda_isac.episodes.native_estimate import NativeEstimatePlanError  # noqa: E402
from ocudu_cuda_isac.sensing import native_estimate as reference  # noqa: E402
from ocudu_cuda_isac.sensing.native_estimate import NativeEstimateError  # noqa: E402
import receiver_patch  # noqa: E402

ADAPTER = ROOT / "native/ocudu_adapter"
ESTIMATE = ADAPTER / "estimate"
CASES = ("half", "layout", "zero_and_absent", "refusals", "async_completion", "generation", "timed", "audit", "drop",
         "absent", "threads")
UNITY = 1.001186490058899  # a unity channel as the stage O2 reads it
DEMO = [2, 51, 4, 12]


def comb(planes, symbols, blocks, value):
    """A comb payload of the stage O2: value(plane, symbol, pilot) for six pilots per block."""
    return [value(plane, symbol, pilot) for plane in range(planes) for symbol in range(symbols)
            for pilot in range(6 * blocks)], [planes, symbols, 6 * blocks]


def plan_of(mode="audit", ring=None, estimate=None):
    """A plan of version 3 that binds no file: enough for every rule that needs none."""
    digest = "a" * 64
    bound = {"path": "/nowhere/file", "sha256": digest}
    return {"schema": ne.PLAN_SCHEMA, "run_id": "unit", "access": "offline_stimulus_and_evaluator", "replay_plan": bound,
            "observer": {"tap": "pilot_ls", "stage": "O2", "source": "pusch_dmrs", "layout": "ocudu_o2_comb_c64",
                         "epoch": 1},
            "sources": [{"grant_id": "g", "record_sha256": digest, "source": bound, "pilot_index": bound,
                         "pilot_ref": bound, "antenna_map": [0, 1]}],
            "evidence": {"mode": mode}, "ring": ring, "estimate": estimate}


class ReferenceTests(unittest.TestCase):
    def test_an_fp16_pair_converts_within_half_precision_rounding(self):
        self.assertEqual((reference.half(1.0), reference.half(0.5), reference.half(-2.0)), (1.0, 0.5, -2.0))
        # A unity channel reads 1.00119 in complex64 and 1 + 2^-10 in the FP16 of the receiver.
        self.assertEqual((reference.half(UNITY), reference.half_bits(UNITY)), (1.0 + 2.0 ** -10, 0x3C01))
        self.assertEqual(reference.half_bits(1.0 + 2.0 ** -11), 0x3C00)       # halfway: the even one
        self.assertEqual(reference.half_bits(1.0 + 3 * 2.0 ** -11), 0x3C02)
        self.assertEqual((reference.half_bits(65519.0), reference.half_bits(65520.0), reference.half_bits(-1e9)),
                         (0x7BFF, 0x7C00, 0xFC00))
        generator = random.Random(52)
        for _ in range(2000):
            value = complex(generator.uniform(-4, 4), generator.uniform(-4, 4))
            packed = reference.payload_values(reference.payload_bytes([value]))[0]
            for exact, stored in ((value.real, packed.real), (value.imag, packed.imag)):
                self.assertLessEqual(abs(stored - exact), reference.ulp(abs(exact)) / 2)
        self.assertEqual(reference.payload_values(struct.pack("<2e", 1.0, 0.5)), [complex(1.0, 0.5)])
        self.assertEqual((reference.ulp(1.0), reference.ulp(1.9), reference.ulp(0.5)), (2.0 ** -10, 2.0 ** -10, 2.0 ** -11))
        with self.assertRaisesRegex(NativeEstimateError, "no whole number of complex FP16 values"):
            reference.payload_values(b"\0" * 6)

    def test_element_1_3_2_5_is_dmrs_symbol_1_port_2_and_subcarrier_41(self):
        position = reference.element(DEMO, 1, 3, 2, 5)
        self.assertEqual(position, 2621)
        self.assertEqual(reference.coordinate(DEMO, position), {"dmrs_symbol": 1, "prb": 3, "rx_port": 2,
                                                                 "subcarrier_in_prb": 5, "subcarrier": 41})
        self.assertEqual((reference.elements(DEMO), reference.elements(DEMO) * reference.ELEMENT_BYTES), (4896, 19584))
        self.assertEqual([reference.element(DEMO, 0, 0, 0, 1), reference.element(DEMO, 0, 0, 1, 0),
                          reference.element(DEMO, 0, 1, 0, 0), reference.element(DEMO, 1, 0, 0, 0)], [1, 12, 48, 2448])
        for index in ((2, 0, 0, 0), (0, 51, 0, 0), (0, 0, 4, 0), (0, 0, 0, 12)):
            with self.assertRaisesRegex(NativeEstimateError, "outside a payload"):
                reference.element(DEMO, *index)
        with self.assertRaisesRegex(NativeEstimateError, "outside a payload"):
            reference.coordinate(DEMO, 4896)
        # The same rule as the reader of the ring.
        from ocudu_cuda_isac.sensing import obs
        self.assertEqual(obs.element_coordinates("ocudu_o3_fp16", DEMO, (1, 3, 2, 5)),
                         {"dmrs_symbol": 1, "rx_port": 2, "subcarrier": 41, "element": position})

    def test_a_zero_filled_buffer_fails_a_nonzero_reference_and_a_true_zero_stays_valid(self):
        values, shape = comb(1, 2, 4, lambda *_: complex(UNITY, UNITY / 2))  # the flat channel 1 + 0.5j
        matched, out_shape = reference.matched(values, shape)
        self.assertEqual(set(matched), {complex(1.0009765625, 0.50048828125)})
        tolerance = {"abs_ulps": 1.0, "relative": 0.002, "nmse": 1e-6}
        exact = reference.parity(matched, matched, out_shape)
        self.assertTrue(reference.within(exact, tolerance))
        self.assertEqual((exact["samples"], exact["numerator"], exact["nmse"]), (48, 0.0, 0.0))
        zeros = reference.parity([0j] * len(matched), matched, out_shape)
        self.assertFalse(reference.within(zeros, tolerance))
        # Even a tolerance that bounds nothing does not accept the stub of a host for a channel.
        self.assertFalse(reference.within(zeros, {"abs_ulps": None, "relative": None, "nmse": None}))
        self.assertEqual((zeros["zero_values"], zeros["nonzero_reference"], zeros["nmse"]), (48, 48, 1.0))
        # A reference of zeros is met by zeros, and a measured zero has a set validity bit.
        nothing = reference.parity([0j] * 96, [0j] * 96, out_shape, "all")
        self.assertTrue(reference.within(nothing, tolerance))
        payload = reference.payload_bytes([0j, complex(1, 0.5), complex(-0.0, 0.0)])
        self.assertEqual(reference.scan(payload), (b"\x07", 3, 0, 1))
        holed = payload + struct.pack("<2H", 0x3C00, 0x7C00) + struct.pack("<2H", 0x7E00, 0)
        self.assertEqual(reference.scan(holed), (b"\x07", 3, 2, 1))

    def test_the_matched_estimator_interpolates_in_fp16_and_smooths_over_the_allocation(self):
        # One block is not smoothed: a pilot keeps its value, an odd subcarrier is half of each, the last keeps pilot 5.
        pilots = [complex(n + 1, -n) for n in range(6)]
        stored, shape = reference.matched(pilots, [1, 1, 6])
        self.assertEqual(shape, [1, 1, 1, 12])
        self.assertEqual(stored[0:12:2], pilots)
        self.assertEqual(stored[1], complex(1.5, -0.5))
        self.assertEqual(stored[11], pilots[5])
        # A constant is filtered to itself, at the edges of the allocation as well.
        values, shape = comb(2, 2, 51, lambda *_: complex(UNITY, 0))
        self.assertEqual(set(reference.matched(values, shape)[0]), {complex(1.0009765625, 0)})
        # A single outlier is spread by the fifteen taps and renormalized where taps are left out.
        spike = [0j] * 60
        spike[30] = 1 + 0j
        smoothed = reference.smooth(spike)
        centre = [block[2 * pilot] for block in smoothed for pilot in range(6)]
        self.assertAlmostEqual(centre[30].real, 0.1950058248, places=3)
        self.assertEqual([round(value.real, 3) for value in centre[23:38]],
                         [round(tap, 3) for tap in reference.RC_TAPS][::-1])
        self.assertEqual(centre[22], 0j)
        edge = [0j] * 60
        edge[0] = 1 + 0j
        total = sum(reference.RC_TAPS[7:])
        self.assertAlmostEqual(reference.smooth(edge)[0][0].real, reference.RC_TAPS[7] / total, places=3)
        self.assertAlmostEqual(sum(reference.RC_TAPS), 1.0, places=6)
        with self.assertRaisesRegex(NativeEstimateError, "six pilots per resource block"):
            reference.matched([0j] * 7, [1, 1, 7])

    def test_the_in_place_orders_differ_from_the_matched_order_by_less_than_smoothing_does(self):
        generator = random.Random(5)
        row = [reference.chalf(complex(UNITY * math.cos(0.07 * n) + generator.gauss(0, 0.03),
                                       UNITY * math.sin(0.07 * n) + generator.gauss(0, 0.03))) for n in range(306)]
        matched, shape = reference.matched(row, [1, 1, 306])
        raw = reference.matched(row, [1, 1, 306], smoothing=False)[0]
        effect = reference.parity(raw, matched, shape)["nmse"]
        self.assertGreater(effect, 1e-4)
        for order in ("ascending", "descending"):
            other = reference.parity(reference.matched(row, [1, 1, 306], order=order)[0], matched, shape)
            self.assertGreater(other["max_fp16_ulps"], 1.0)        # the order of the reads matters
            self.assertLess(other["nmse"], 0.25 * effect)          # and stays below the declared fraction
        waves = [[block for block in range(51) if block % 3 == wave] for wave in range(3)]
        flat = [value for block in reference.smooth(row, waves=waves) for value in block]
        self.assertLess(reference.parity(flat, matched, shape)["nmse"], 0.25 * effect)
        with self.assertRaisesRegex(NativeEstimateError, "every resource block once"):
            reference.smooth(row, waves=[[0, 1]])
        # Without a fused multiply-add the result is the same within one FP16 spacing.
        plain = reference.parity(reference.matched(row, [1, 1, 306], fused=False)[0], matched, shape)
        self.assertLessEqual(plain["max_fp16_ulps"], 1.0)

    def test_compare_declares_its_bound_and_names_its_counts(self):
        generator = random.Random(9)
        values, shape = comb(2, 2, 12, lambda *_: complex(UNITY + generator.gauss(0, 0.02), generator.gauss(0, 0.02)))
        matched, out_shape = reference.matched(values, shape)
        tolerance = {"abs_ulps": None, "relative": None, "nmse": 1e-6, "smoothing_fraction": 0.25}
        good = ne.compare(matched, out_shape, values, shape, tolerance)
        self.assertTrue(good["passed"])
        self.assertEqual(good["pilot_support"]["samples"], 2 * 2 * 12 * 6)
        self.assertEqual(good["all_elements"]["samples"], 2 * 2 * 12 * 12)
        self.assertEqual(good["nmse_bound"], 0.25 * good["smoothing_effect_nmse"])
        self.assertEqual(sorted(good["in_place_model"]), ["ascending", "descending"])
        raw = reference.matched(values, shape, smoothing=False)[0]
        bad = ne.compare(raw, out_shape, values, shape, tolerance)
        self.assertFalse(bad["passed"])                      # an estimate that was never smoothed
        self.assertEqual(bad["unsmoothed_o2"]["nmse"], 0.0)  # and that is what it is
        self.assertFalse(ne.compare([0j] * len(matched), out_shape, values, shape, tolerance)["passed"])
        with self.assertRaisesRegex(NativeEstimatePlanError, "do not describe the same allocation"):
            ne.compare(matched, [2, 12, 4, 12], values, shape, tolerance)
        with self.assertRaisesRegex(NativeEstimatePlanError, "nmse is a number"):
            ne.check_tolerance(dict(tolerance, nmse=None))


class PlanTests(unittest.TestCase):
    def test_a_plan_of_version_3_names_an_estimate_or_null(self):
        self.assertIs(ne.check_plan(plan_of())["estimate"], None)
        spec = ne.estimate_spec()
        self.assertEqual(spec, {"tap": "native_estimate", "stage": "O3", "source": "pusch_dmrs", "layout": "ocudu_o3_fp16",
                                "backend": "cuda", "copy_stream": "producer", "landing": "slot", "timeout_us": 2000,
                                "ring": None})
        ne.check_plan(plan_of(estimate=spec))
        rings = plan_of("timed", sr.ring_spec(3, 3), ne.estimate_spec(ring=sr.ring_spec(3, 3, stream="o3")))
        ne.check_plan(rings)
        self.assertEqual(str(ne.socket_path("/run/native", rings)), "/run/native/estimate/o3.sock")

    def test_every_rule_of_an_estimate_is_refused(self):
        def broken(change, mode="audit", ring=None):
            plan = plan_of(mode, ring, ne.estimate_spec())
            change(plan)
            return plan

        cases = (
            (lambda plan: plan["estimate"].update(tap="pilot_ls"), "the tap native_estimate"),
            (lambda plan: plan["estimate"].update(layout="ocudu_o2_comb_c64"), "the layout ocudu_o3_fp16"),
            (lambda plan: plan["estimate"].update(backend="cpu"), 'estimate.backend is "cuda"'),
            (lambda plan: plan["estimate"].update(copy_stream="any"), 'estimate.copy_stream is "producer" or "side"'),
            (lambda plan: plan["estimate"].update(landing="pageable"), 'estimate.landing is "slot" or "staging"'),
            (lambda plan: plan["estimate"].update(timeout_us=-1), "estimate.timeout_us"),
            (lambda plan: plan["estimate"].pop("ring"), "estimate requires exactly"),
            (lambda plan: plan["estimate"].update(source="srs"), "taps the stage O2 of the same sensing source"),
            (lambda plan: plan["estimate"].update(ring=dict(sr.ring_spec(3, 3, stream="o3"), slots=4)),
             "estimate.ring.slots is 4"),
            (lambda plan: plan.pop("estimate"), "a plan requires exactly"),
            (lambda plan: plan.update(schema=sr.PLAN_SCHEMA), "unsupported plan schema"))
        for change, words in cases:
            with self.assertRaisesRegex(NativeEstimatePlanError, re.escape(words)):
                ne.check_plan(broken(change))
        with self.assertRaisesRegex(NativeEstimatePlanError, "a timed estimate exists in a slot of its ring only"):
            ne.check_plan(plan_of("timed", sr.ring_spec(3, 3), ne.estimate_spec()))
        with self.assertRaisesRegex(NativeEstimatePlanError, "different names"):
            ne.check_plan(plan_of("audit", sr.ring_spec(3, 3), ne.estimate_spec(ring=sr.ring_spec(3, 3))))
        untapped = plan_of(estimate=ne.estimate_spec())
        untapped["observer"].update(tap="none", stage=None, source=None, layout=None)
        untapped["sources"] = []
        with self.assertRaisesRegex(NativeEstimatePlanError, "taps the stage O2"):
            ne.check_plan(untapped)

    def test_a_capture_line_is_consistent_on_its_own(self):
        producer = dict.fromkeys(reference.PRODUCER_FIELDS, 1)
        line = {"schema": ne.CAPTURE_SCHEMA, "run_id": "r", "mode": "timed", "stage": "O3", "backend": "cuda",
                "opportunity_id": "op", "opportunity_sequence": 0, "outcome": "published", "reason": None,
                "admitted": True, "seen": True, "landed": True, "producer": producer,
                "copy": {"stream": "producer", "landing": "slot", "completion": "cuda_event", "polls": 1, "submit_us": 2.0},
                "ring": {"epoch": 1, "sequence": 0, "depth": 0, "bytes": 1, "window_calls": 0},
                "keys": {"episode": "0", "opportunity": ne.o2_reference.opaque_key("op"), "grant": "0"},
                "absolute_slot": 1, "grid": {"grid_index": 0, "generation": 1}, "payload_bytes": 48, "valid": 12,
                "holes": 0, "nonzero": 12, "t_expect_ns": 1, "t_tap_ns": 2, "t_submit_ns": 3, "t_resolve_ns": 4,
                "t_complete_ns": 5, "t_publish_ns": 6, "t_done_ns": 7, "resolve_us": 1.0}
        self.assertEqual(ne.capture_violations(line), [])
        # A record that was published before its copy completed.
        self.assertIn("not in order", ne.capture_violations(dict(line, t_publish_ns=4))[0])
        self.assertIn("names its place on the ring", ne.capture_violations(dict(line, ring=None))[0])
        self.assertIn("does not have the reason", ne.capture_violations(dict(line, reason="no_credit"))[0])
        absent = dict(line, outcome="absent", reason="no_estimate", seen=False, landed=False, producer=None, copy=None,
                      ring=None, valid=None, holes=None, nonzero=None, t_tap_ns=None, t_submit_ns=None,
                      t_complete_ns=None, t_publish_ns=None)
        self.assertEqual(ne.capture_violations(absent), [])
        self.assertIn("never reported", ne.capture_violations(dict(absent, seen=True, producer=producer))[-1])
        dropped = dict(absent, outcome="dropped", reason="no_credit", seen=True, producer=producer, admitted=False)
        self.assertEqual(ne.capture_violations(dropped), [])
        for outcome, reasons in ne.REASONS.items():
            self.assertIn(outcome, ne.OUTCOMES)
            self.assertTrue(reasons)


class NativeCoreTests(unittest.TestCase):
    """The capture core, built on its own: no OCUDU source, no OCUDU build and no CUDA."""

    @classmethod
    def setUpClass(cls):
        cls.missing = [tool for tool in ("cmake", "ctest", "c++") if shutil.which(tool) is None]
        if cls.missing:
            return
        cls.temporary = tempfile.TemporaryDirectory()
        cls.build = Path(cls.temporary.name) / "build"
        steps = (["cmake", "-S", str(ESTIMATE), "-B", str(cls.build)], ["cmake", "--build", str(cls.build), "-j2"])
        cls.log, cls.status = "", 0
        for step in steps:
            result = subprocess.run(step, capture_output=True, text=True, timeout=1200, check=False)
            cls.log += result.stdout + result.stderr
            cls.status = cls.status or result.returncode
        cls.test = cls.build / "estimate_core_test"

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
        link = (self.build / "CMakeFiles/estimate_core_test.dir/link.txt").read_text()
        self.assertEqual(re.findall(r"\S*?(lib\w+\.(?:a|so))\b", link),
                         ["libestimate_core.a", "libobs_publisher.a", "libobserver_core.a", "libreplay_core.a",
                          "libobs_ring.a"])
        for word in ("cuda", "ocudu_phy"):
            self.assertNotIn(word, link.lower().replace(str(ROOT).lower(), ""))
        # The core and its unit target include the cores they join and the standard library.
        allowed = {"estimate_core.h", "obs/ring.h", "obs_publisher.h", "observer_core.h", "replay_core.h"}
        for name in ("estimate_core.h", "estimate_core.cpp", "tests/estimate_core_test.cpp"):
            for include in re.findall(r'#include [<"]([^>"]+)[>"]', (ESTIMATE / name).read_text()):
                self.assertTrue(include in allowed or "/" not in include or include.startswith("sys/"), include)

    def test_every_native_unit_case_passes(self):
        listed = subprocess.run([str(self.test), "--list"], capture_output=True, text=True, check=True).stdout.split()
        self.assertEqual(tuple(listed), CASES)
        result = subprocess.run(["ctest", "--output-on-failure", "-R", "estimate_core_"], cwd=self.build,
                                capture_output=True, text=True, timeout=600, check=False)
        self.assertEqual(result.returncode, 0, result.stdout[-3000:])
        self.assertIn(f"100% tests passed, 0 tests failed out of {len(CASES)}", result.stdout)
        for name in CASES:
            self.assertRegex(result.stdout, rf"estimate_core_{name} \.+\s+Passed")
        unknown = subprocess.run([str(self.test), "no_such_case"], capture_output=True, text=True, check=False)
        self.assertEqual(unknown.returncode, 2)

    def test_native_core_and_reference_convert_and_index_alike(self):
        generator = random.Random(20261122)
        patterns = [0x3F8026E1, 0x3F800000, 0x00000000, 0x80000000, 0x7F800000, 0x477FE000, 0x477FF000, 0x33000000,
                    0x38800000, 0x387FC000]
        patterns += [(generator.getrandbits(1) << 31) | (generator.randint(95, 145) << 23) | generator.getrandbits(23)
                     for _ in range(6000)]
        text = "".join(f"{pattern:08x}\n" for pattern in patterns)
        result = subprocess.run([str(self.test), "--halves"], input=text, capture_output=True, text=True, check=True)
        lines = result.stdout.split()
        self.assertEqual(len(lines), 2 * len(patterns))
        for pattern, half, back in zip(patterns, lines[0::2], lines[1::2]):
            value = struct.unpack("<f", struct.pack("<I", pattern))[0]
            self.assertEqual(int(half, 16), reference.half_bits(value), f"{pattern:08x}")
            self.assertEqual(struct.unpack("<f", struct.pack("<I", int(back, 16)))[0], reference.half(value))
        self.assertEqual(lines[0], "3c01")  # the unity channel
        for index in ((1, 3, 2, 5), (0, 0, 0, 0), (1, 50, 3, 11), (0, 7, 1, 10)):
            said = subprocess.run([str(self.test), "--element", *map(str, index), "2", "51", "4"], capture_output=True,
                                  text=True, check=True).stdout.split()
            position = reference.element(DEMO, *index)
            self.assertEqual([int(said[0]), int(said[1])], [position, reference.coordinate(DEMO, position)["subcarrier"]])
        refused = subprocess.run([str(self.test), "--element", "2", "0", "0", "0", "2", "51", "4"], capture_output=True,
                                 text=True, check=False)
        self.assertEqual(refused.returncode, 2)


class TreeTests(unittest.TestCase):
    FILES = ("native/ocudu_adapter/estimate/CMakeLists.txt", "native/ocudu_adapter/estimate/estimate_core.h",
             "native/ocudu_adapter/estimate/estimate_core.cpp", "native/ocudu_adapter/estimate/tests/estimate_core_test.cpp",
             "native/ocudu_adapter/observer/estimate_tap.h", "native/ocudu_adapter/observer/pilot_ls.cpp",
             "native/ocudu_adapter/observer/observer_plan.h", "native/ocudu_adapter/observer/CMakeLists.txt",
             "native/ocudu_adapter/CMakeLists.txt", "native/ocudu_adapter/README.md",
             "patches/ocudu/0001-pusch-resident-estimate-tap.patch", "patches/ocudu/series.json", "patches/ocudu/README.md",
             "scripts/receiver_patch.py", "scripts/native_ocudu.py", "scripts/native_adapter.py", "scripts/accept_s5_2.sh",
             "src/ocudu_cuda_isac/sensing/native_estimate.py", "src/ocudu_cuda_isac/episodes/native_estimate.py",
             "tests/unit/test_native_estimate.py", "tests/unit/native_estimate_double.py",
             "tests/unit/test_native_estimate_driver.py", "tests/integration/native_estimate.py",
             "tests/test_accept_s5_2.py", "tests/test_receiver_patch.py", "configs/integration/native_estimate.json",
             "docs/interfaces/NATIVE_ESTIMATE.md", "Makefile")

    def test_files_are_ascii_and_do_not_name_the_project(self):
        # The working name lives on line 1 of the top-level README and nowhere else.
        name = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()[0].lstrip("# ").split(":")[0].strip()
        for relative in self.FILES:
            data = (ROOT / relative).read_bytes()
            self.assertTrue(data.isascii(), relative)
            if name and name.lower() != "ocudu-cuda-isac":
                self.assertNotIn(name.lower(), data.decode("ascii").lower(), relative)

    def test_the_patch_is_small_adds_no_kernel_and_never_waits_for_the_device(self):
        series = receiver_patch.load_series()
        self.assertEqual([item["file"] for item in series["patches"]], ["0001-pusch-resident-estimate-tap.patch"])
        patch = (ROOT / "patches/ocudu/0001-pusch-resident-estimate-tap.patch").read_text()
        touched = re.findall(r"^diff --git a/(\S+) b/", patch, re.MULTILINE)
        self.assertEqual(sorted(touched), sorted(series["files"]))
        self.assertEqual(len(touched), 2)
        self.assertFalse([path for path in touched if path.endswith((".cu", ".cuh"))])
        added = [line[1:] for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++")]
        removed = [line for line in patch.splitlines() if line.startswith("-") and not line.startswith("---")]
        self.assertEqual((len(added), removed), (137, []))
        for word in ("Synchronize", "cudaMemcpy", "cudaEvent", "malloc", "new ", "std::mutex", "throw"):
            self.assertFalse([line for line in added if word in line and not line.lstrip().startswith(("//", "///"))], word)
        code = "\n".join(added)
        self.assertIn("pusch_e2e_get_estimates_fp16(e2e_handle_)", code)
        self.assertIn("resident_estimate_tap.load(std::memory_order_acquire)", code)
        self.assertIn("constexpr unsigned PUSCH_RESIDENT_ESTIMATE_TAP_VERSION = 1;", code)
        self.assertIn("virtual void on_estimate(const pusch_resident_estimate& estimate) noexcept = 0;", code)
        manifest = series["manifest"]["patches"][0]
        self.assertEqual((manifest["kernels_changed"], manifest["interface"]["version"]), (0, 1))

    def test_the_route_orders_its_copy_and_only_asks_for_its_completion(self):
        route = (ADAPTER / "observer/estimate_tap.h").read_text()
        self.assertIn('#if !defined(ISAC_O3_ROUTE) || !defined(ENABLE_CUDA)', route)
        self.assertIn("static_assert(ocudu::PUSCH_RESIDENT_ESTIMATE_TAP_VERSION == 1,", route)
        submit = route[route.index("bool submit(const producer_view& view, void* destination) noexcept override"):
                       route.index("copy_state poll() noexcept override")]
        poll = route[route.index("copy_state poll() noexcept override"):route.index("copy_state drain() noexcept override")]
        # The copy is asynchronous, behind the producer, and one event is recorded behind the copy.
        self.assertLess(submit.index("cudaStreamWaitEvent(own, produced, 0)"), submit.index("cudaMemcpyAsync("))
        self.assertLess(submit.index("cudaMemcpyAsync("), submit.index("cudaEventRecord(done, stream)"))
        self.assertIn("cudaEventQuery(done)", poll)
        for text in (submit, poll):
            for word in ("Synchronize", "cudaMemcpy(", "sleep", "wait("):
                self.assertNotIn(word, text, word)
        # Exactly one call waits for the device, in drain(), which only settle() of the core calls.
        self.assertEqual(len(re.findall(r"cuda\w*Synchronize\(", route)), 1)
        self.assertIn("cudaEventSynchronize(done)", route[route.index("copy_state drain() noexcept override"):])
        core = (ESTIMATE / "estimate_core.cpp").read_text()
        self.assertEqual(core.count("device.drain()"), 1)
        self.assertIn("device.drain()", core[core.index("void capture::settle()"):])
        tap = core[core.index("void capture::on_estimate("):core.index("void capture::finish(")]
        for word in ("drain", "poll(", "throw", "push_back", "emplace", "std::string", "sha256"):
            self.assertNotIn(word, tap, word)
        # Nothing is copied asynchronously into memory that is not page-locked.
        self.assertIn("cudaHostAlloc(&memory, bytes, cudaHostAllocDefault)", route)
        self.assertIn("cudaHostRegister(base, bytes, cudaHostRegisterDefault)", route)

    def test_the_tool_expects_before_each_decode_and_resolves_after_it(self):
        tool = (ADAPTER / "observer/pilot_ls.cpp").read_text()
        audit = tool[tool.index("  // ---- The audited run:"):tool.index("  // ---- The timed run.")]
        # The timed run ends where the next section begins. Sprint 4.1 put the paced run between it and the drain,
        # and tests/unit/test_mps_pacing.py holds the paced loop to the same rule.
        timed = tool[tool.index("  // ---- The timed run."):tool.index("  // ---- The paced run (Sprint 4.1).")]
        self.assertLess(tool.index("  // ---- The paced run (Sprint 4.1)."), tool.index("  // ---- The drain:"))
        for loop, start in ((audit, "const dispatched done = dispatch(book, *backend, name, key"),
                            (timed, "dispatch(book, backend, names[n * backends.size() + b], key")):
            self.assertLess(loop.index("taker->expect(estimate_handle[sequence], key);"), loop.index(start))
            self.assertLess(loop.index(start), loop.index("taker->resolve()"))
            self.assertNotIn("settle", loop)
        # The timed loop hashes nothing, writes nothing and builds no record for the estimate either.
        for word in ("sha256_hex", "sha256(", "sha256_count", "write_bytes", "json", "estimate_entry", "std::cout"):
            self.assertNotIn(word, timed[timed.index("for (std::size_t n = 0; n != chosen.size(); ++n) {\n      const std::size_t "):],
                             word)
        drain = tool[tool.index("  // ---- The drain:"):]
        self.assertLess(drain.index("route->close();"), drain.index("taker->settle();"))
        self.assertLess(drain.index("taker->settle();"), drain.index("estimate_ring->drain("))
        self.assertLess(drain.index("estimate_ring->drain("), drain.index("hash_drained = isac_replay::sha256_count();"))
        # A build without the route refuses the plan before a processor exists.
        self.assertLess(tool.index("this build has no route to it"), tool.index("std::make_unique<processor_backend>"))
        self.assertIn("#ifndef ISAC_O3_ROUTE\n    throw input_error(", tool)
        self.assertIn('#ifdef ISAC_O3_ROUTE\n// The device side of the capture.', tool)

    def test_the_builds_the_gate_and_both_sides_share_their_strings(self):
        adapter = (ADAPTER / "CMakeLists.txt").read_text()
        self.assertIn('option(ISAC_O3_ROUTE "Compile the route to the estimate of the patched receiver into pilot_ls" OFF)',
                      adapter)
        self.assertLess(adapter.index("add_subdirectory(transport)"), adapter.index("add_subdirectory(estimate)"))
        observer = (ADAPTER / "observer/CMakeLists.txt").read_text()
        self.assertIn("target_link_libraries(pilot_ls PRIVATE estimate_core)", observer)
        self.assertIn("target_compile_definitions(pilot_ls PRIVATE ISAC_O3_ROUTE)", observer)
        self.assertNotIn("target_compile_options(pilot_ls", observer)
        cmake = (ESTIMATE / "CMakeLists.txt").read_text()
        self.assertEqual(tuple(re.search(r"set\(ESTIMATE_CORE_CASES ([^)]*)\)", cmake)[1].split()), CASES)
        builder = (ROOT / "scripts/native_adapter.py").read_text()
        self.assertIn("""f"-DISAC_O3_ROUTE={'ON' if o3 else 'OFF'}\"""", builder)
        self.assertIn('DEFAULT_BUILD_O3 = ROOT / "build/ocudu-adapter-o3"', builder)
        self.assertIn("accept-s5.2:", (ROOT / "Makefile").read_text())
        # The strings of a plan, of a report and of a record are the same in the tool and in the reader.
        tool = (ADAPTER / "observer/pilot_ls.cpp").read_text() + (ADAPTER / "observer/observer_plan.h").read_text()
        core = (ESTIMATE / "estimate_core.h").read_text()
        for text in (ne.PLAN_SCHEMA, ne.NATIVE_SCHEMA, ne.CAPTURE_SCHEMA, ne.ESTIMATE_DIR):
            self.assertIn(f'"{text}"', tool)
        for text in (reference.LAYOUT, reference.DTYPE, reference.PRECISION, reference.PRODUCER, ne.TAP, reference.STAGE):
            self.assertIn(f'"{text}"', core)
        self.assertIn(reference.RULE, core.replace('"\n    "', ""))
        for name in reference.PRODUCER_FIELDS + ne.CAPTURE_FIELDS:
            self.assertIn(f'{{"{name}",', tool, name)
        for reasons in ne.REASONS.values():
            for reason in filter(None, reasons):
                self.assertIn(f'"{reason}"', (ESTIMATE / "estimate_core.cpp").read_text() + core, reason)
        config = json.loads((ROOT / "configs/integration/native_estimate.json").read_text())
        self.assertEqual((config["schema"], config["selector"], config["backends"]),
                         ("integration-config/native_estimate/v1", "native_estimate", ["cpu", "cuda"]))
        self.assertEqual(config["expect"]["o3_unity"], reference.half(config["expect"]["o2_unity"]))
        self.assertEqual(config["expect"]["element"]["position"],
                         reference.element(config["expect"]["shape"]["demo-51prb-4rx"], *config["expect"]["element"]["index"]))
        self.assertEqual(reference.elements(config["expect"]["shape"]["demo-51prb-4rx"]) * 4, config["expect"]["payload_bytes"])
        for kind in ("flat", "varying", "noisy"):
            ne.check_tolerance(copy.deepcopy(config["tolerance"][kind]))
        # The declared fraction lies above what the study saw of the in-place orders and below one.
        self.assertLess(config["study"]["in_place"]["max_fraction_of_smoothing_effect"],
                        config["tolerance"]["noisy"]["smoothing_fraction"])
        self.assertLess(config["tolerance"]["noisy"]["smoothing_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
