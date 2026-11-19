"""Host-only fixtures for native build identity, workload selection and grading."""

from argparse import Namespace
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import native_ocudu
from scripts.verify_pusch_log import compare_table_ii

# Synthetic transcript in the pinned test's format. Times intentionally differ from the paper.
PAPER_LOG = """Config: 51 PRB, 30.0 dB SINR, MCS 27 (table=qam256), 1 layer(s), 1 RX port(s), DMRS Type1 [2,11], transform_precoding=off, 200 iterations, 5 warmup, rx_device_grid=direct-managed, gpu_output=pinned
TBS: 54296 bits (6787 bytes), Base Graph: BG1, Codeblocks: 7, Modulation: 256QAM
=== Results (200 iterations, 5 warmup skipped) ===
CPU: 200/200 pass (0.0% BLER)
GPU: 200/200 pass (0.0% BLER)  [resident GPU timing: 200/200]
GPU producer-owned grid: 200/200 measured iterations
GPU immutable owned snapshot: 200/200 measured iterations
Disagreements: 0
Byte mismatches: 0
Output guard failures: 0
CPU total: 150.0 200.0 280.0
GPU total: 200.0 250.0 400.0
Speedup:                  0.80x
PASS: Both paths decoded all 200 iterations, 0 byte mismatches
"""


class PaperProfileTest(unittest.TestCase):
    def test_slower_gpu_still_records_correctness_and_actual_latency(self):
        result = compare_table_ii(PAPER_LOG)
        self.assertTrue(result["correctness_matches_table_row"])
        self.assertFalse(result["gpu_faster_on_this_run"])
        self.assertEqual(result["observed"]["gpu_mean_us"], 250)
        self.assertEqual(result["reference"]["gpu_mean_us"], 183.9)

    def test_wrong_workload_damaged_timing_and_truncated_decode_fail(self):
        for old, new in [("qam256", "qam64"), ("30.0 dB", "25.0 dB"), ("30.0 dB", "nan dB"),
                         ("1 RX port(s)", "4 RX port(s)"), ("54296 bits", "54240 bits"),
                         ("gpu_output=pinned", "gpu_output=vector"), ("5 warmup skipped", "0 warmup skipped"),
                         ("GPU total: 200.0 250.0 400.0", "GPU total: 250 0 300"),
                         ("Speedup:                  0.80x", "Speedup: 1.82x"),
                         ("GPU immutable owned snapshot: 200", "GPU immutable owned snapshot: 199")]:
            with self.subTest(old=old, new=new), self.assertRaises(ValueError):
                compare_table_ii(PAPER_LOG.replace(old, new))
        with self.assertRaises(ValueError):
            compare_table_ii(PAPER_LOG + PAPER_LOG)


class BuildContractTest(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.source = self.root / "ext/ocudu"
        self.build = self.root / "build"
        self.run = self.root / "results"
        for path in [self.source, self.build, self.run, self.root / "isac/bin"]:
            path.mkdir(parents=True)
        self.stack.enter_context(patch.object(native_ocudu, "ROOT", self.root))
        self.stack.enter_context(patch.object(native_ocudu, "SOURCE", self.source))
        self.stack.enter_context(patch.object(native_ocudu.subprocess, "check_output", return_value="sm_121\n"))
        self.stack.enter_context(patch.object(native_ocudu, "record_command", side_effect=self.record_fake_command))
        self.commands = []
        self.cuda_found = "TRUE"
        self.gpu = {"compute_capability": "12.1"}

    def record_fake_command(self, command, run, filename, env=None):
        self.commands.append([str(word) for word in command])
        if filename in ("configure.log", "build.log"):
            self.assertIn(str(self.source.parent), env["GIT_CEILING_DIRECTORIES"].split(":"))
        if filename == "configure.log":
            (self.build / "CMakeCache.txt").write_text(
                f"CMAKE_HOME_DIRECTORY:INTERNAL={self.source}\nCMAKE_BUILD_TYPE:STRING=Release\n"
                "BUILD_TESTING:BOOL=ON\nENABLE_CUDA:BOOL=ON\nCMAKE_CUDA_ARCHITECTURES:STRING=121\n"
                f"CUDA_ACCEL_FOUND:BOOL={self.cuda_found}\nMCPU:STRING=neoverse-v2\n"
            )
        if filename == "build.log":
            binary = self.build / native_ocudu.TEST
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"synthetic parser fixture")
            gnb = self.build / "apps/gnb/gnb"
            gnb.parent.mkdir(parents=True)
            gnb.touch(mode=0o755)
        if filename == "gnb-version.txt":
            (run / filename).write_text("OCUDU 5G gNB version 26.04.0 ()")
        if filename == "native-help.txt":
            (run / filename).write_text("--require-gpu-resident")

    def build_full(self):
        args = Namespace(jobs=8, machine="spark", mcpu=None)
        native_ocudu.native_build(args, self.build, self.run, self.gpu, "pin", "all", {})

    def test_full_spark_recipe_and_stale_binary_rejection(self):
        self.build_full()
        for flag in ["-DENABLE_CUDA=ON", "-DMCPU=neoverse-v2", "-DCMAKE_CUDA_ARCHITECTURES=121",
                     "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_TESTING=ON"]:
            self.assertIn(flag, self.commands[0])
        self.assertEqual(self.commands[1], ["cmake", "--build", str(self.build), "-j8"])
        venv_bin = self.root / "isac/bin"
        self.assertEqual((venv_bin / "gnb").resolve(), self.build / "apps/gnb/gnb")
        self.assertEqual((venv_bin / native_ocudu.TEST.name).resolve(), self.build / native_ocudu.TEST)
        self.assertIn([str(self.build / "apps/gnb/gnb"), "--version"], self.commands)
        native_ocudu.checked_build(self.build, self.run, self.gpu, "pin")
        (self.build / native_ocudu.TEST).write_bytes(b"changed")
        with self.assertRaises(ValueError):
            native_ocudu.checked_build(self.build, self.run, self.gpu, "pin")

    def test_existing_executable_is_preserved_before_either_link_changes(self):
        venv_bin = self.root / "isac/bin"
        (venv_bin / "gnb").write_text("user-supplied executable")
        old_target = self.root / "older-build" / native_ocudu.TEST
        (venv_bin / native_ocudu.TEST.name).symlink_to(old_target)
        with self.assertRaisesRegex(ValueError, "existing non-symlink"):
            self.build_full()
        self.assertEqual((venv_bin / "gnb").read_text(), "user-supplied executable")
        self.assertEqual((venv_bin / native_ocudu.TEST.name).readlink(), old_target)

    def test_silent_cuda_fallback_blocks_build(self):
        self.cuda_found = "FALSE"
        with self.assertRaisesRegex(ValueError, "CUDA_ACCEL_FOUND"):
            self.build_full()
        self.assertEqual(len(self.commands), 1)
        self.assertFalse((self.build / "isac-native.json").exists())


if __name__ == "__main__":
    unittest.main()
