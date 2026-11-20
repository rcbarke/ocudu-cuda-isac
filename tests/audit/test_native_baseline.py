"""Host-only fixtures for native build identity, reference comparison, evidence grading and workload selection."""

from argparse import Namespace
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import shlex
import struct
import sys
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
# What CMake printed on the Spark when libgtest-dev was absent. The shell forced terminal colors.
CONFIGURE_WITHOUT_GTEST = """-- Found FFTW3F: /usr/lib/aarch64-linux-gnu/libfftw3f.so
-- Could NOT find ARMPL (missing: ARMPL_LIBRARIES ARMPL_INCLUDE_DIRS)
\x1b[31mCMake Error at /usr/share/cmake-3.28/Modules/FindPackageHandleStandardArgs.cmake:230 (message):
  Could NOT find GTest (missing: GTEST_LIBRARY GTEST_INCLUDE_DIR
  GTEST_MAIN_LIBRARY)
Call Stack (most recent call first):
  /usr/share/cmake-3.28/Modules/FindPackageHandleStandardArgs.cmake:600 (_FPHSA_FAILURE_MESSAGE)
  /usr/share/cmake-3.28/Modules/FindGTest.cmake:270 (FIND_PACKAGE_HANDLE_STANDARD_ARGS)
  CMakeLists.txt:468 (find_package)

\x1b[0m
-- Configuring incomplete, errors occurred!
"""
PACKAGES = ("libgtest-dev", "libyaml-cpp-dev", "libsctp-dev", "libmbedtls-dev")
SPARK_GPU = {"uuid": "GPU-spark", "name": "NVIDIA GB10", "compute_capability": "12.1", "driver": "580.178.04"}
LOCK_ENTRY = {"name": "ocudu", "commit": "27e84cb3da65205b55dbd56ace6fac93a035331e", "source_ref": "nvcuda_accel_02"}
KERNELS = ("pusch_e2e", "ldpc_decoder_flexible")


def fake_elf(sections, machine=183):
    """Smallest ELF64 file with a readable section name table. Machine 183 is AArch64."""
    names = [*sections, ".shstrtab"]
    table = b"\0" + b"".join(name.encode() + b"\0" for name in names)
    offsets = [1 + sum(len(name) + 1 for name in names[:index]) for index in range(len(names))]
    count = len(names) + 1
    header = b"\x7fELF\x02\x01\x01\0" + bytes(8) + struct.pack(
        "<HHIQQQIHHHHHH", 2, machine, 1, 0, 0, 64 + len(table), 0, 64, 0, 0, 64, count, count - 1)
    entries = bytes(64)
    for offset in offsets[:-1]:
        entries += struct.pack("<IIQQQQIIQQ", offset, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    entries += struct.pack("<IIQQQQIIQQ", offsets[-1], 3, 0, 0, 64, len(table), 0, 0, 0, 0)
    return header + table + entries


def spark_facts(**changes):
    """Observations shaped like host_facts() on the accepted Spark, with the newer toolkit and driver."""
    large = [5, 6, 7, 8, 9, 15, 16, 17, 18, 19]
    facts = {
        "machine": "aarch64", "kernel": "7.0.0-1025-nvidia", "node": "spark-ecf8",
        "os": {"id": "ubuntu", "version_id": "24.04", "pretty_name": "Ubuntu 24.04.5 LTS"},
        "cpus": {str(cpu): {"model": "Cortex-X925" if cpu in large else "Cortex-A725"} for cpu in range(20)},
        "cpu_models": ["Cortex-A725", "Cortex-X925"],
        "high_capacity_cpus": "5-9,15-19", "high_capacity_basis": "cpu_capacity", "allowed_cpus": "0-19",
        "governors": {f"policy{cpu}": "schedutil" for cpu in (0, 5, 10, 15)},
        "nvcc": {"version": "13.5.37", "codes": ["sm_120", "sm_121"]},
        "cxx": {"command": "c++", "banner": "c++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0", "family": "GCC",
                "version": "13.3.0", "accepts_mcpu": True},
        "cmake": "3.28.3", "gpu": dict(SPARK_GPU),
        "host_packages": [{"package": package, "found": True} for package in PACKAGES],
    }
    facts.update(changes)
    return facts


def reference_report(profile, facts, mcpu):
    """Compare synthetic observations as a 20-CPU Spark process would see them."""
    with patch.object(native_ocudu.os, "sched_getaffinity", return_value=set(range(20))):
        return native_ocudu.compare_reference(profile, facts, mcpu, LOCK_ENTRY)


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

    def test_paper_run_pins_the_reference_cpus_and_requires_the_resident_path(self):
        commands = []

        def record(command, run, filename, env=None):
            commands.append([str(word) for word in command])
            (run / filename).write_text("graded\n")

        with tempfile.TemporaryDirectory() as temp, patch.object(native_ocudu, "record_command", side_effect=record), \
                patch.object(native_ocudu.os, "sched_getaffinity", return_value=set(range(20))), \
                redirect_stdout(io.StringIO()):
            native_ocudu.benchmark("paper", Path(temp) / "build", Path(temp), SPARK_GPU, {})
            self.assertEqual(commands[0][:6], ["timeout", "--kill-after=10s", "5m", "taskset", "-c", "5-9,15-19"])
            line = " ".join(commands[0])
            for expected in ("CUDA_VISIBLE_DEVICES=GPU-spark", "--prb 51 --sinr 30 --mcs 27 --mcs-table qam256",
                             "--layers 1 --ports 1", "--iterations 200 --warmup 5", "--rx-device-grid direct-managed",
                             "--require-gpu-resident --gpu-output-buffer pinned"):
                self.assertIn(expected, line)
            self.assertIn("--paper-json", commands[1])
            with patch.object(native_ocudu.os, "sched_getaffinity", return_value={0, 1, 2, 3}), \
                    self.assertRaisesRegex(ValueError, "5-9,15-19"):
                native_ocudu.benchmark("paper", Path(temp) / "build", Path(temp), SPARK_GPU, {})


class LockedSourceTest(unittest.TestCase):
    def test_native_source_directory_and_commit_follow_the_lock(self):
        lock = {"dependencies": [
            {"name": "sionna", "path": "ext/sionna", "commit": "a" * 40},
            {"name": "ocudu", "path": "ext/ocudu-deepsig-cuda-inline", "commit": "c" * 40}]}
        with tempfile.TemporaryDirectory() as temp, patch.object(native_ocudu, "ROOT", Path(temp)):
            (Path(temp) / "ext").mkdir()
            (Path(temp) / "ext/lock.json").write_text(json.dumps(lock))
            entry = native_ocudu.locked_source("ocudu")
            self.assertEqual((entry["path"], entry["commit"]), ("ext/ocudu-deepsig-cuda-inline", "c" * 40))
            with self.assertRaisesRegex(ValueError, "no dependency named 'gnb'"):
                native_ocudu.locked_source("gnb")
        self.assertEqual(native_ocudu.SOURCE,
                         native_ocudu.ROOT / native_ocudu.locked_source("ocudu")["path"])


class FakeBuildCase(unittest.TestCase):
    """A scratch checkout whose configure and build steps are doubles that write CMake's files."""

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        self.source = self.root / "ext/ocudu"
        self.build = self.root / "build"
        self.run = self.root / "results"
        for path in [self.source, self.build, self.run, self.root / "isac/bin"]:
            path.mkdir(parents=True)
        self.stack.enter_context(patch.object(native_ocudu, "ROOT", self.root))
        self.stack.enter_context(patch.object(native_ocudu, "SOURCE", self.source))
        self.codes = "sm_120\nsm_121\n"
        self.stack.enter_context(patch.object(native_ocudu.subprocess, "check_output",
                                              side_effect=lambda *args, **options: self.codes))
        self.stack.enter_context(patch.object(native_ocudu, "record_command", side_effect=self.record_fake_command))
        self.stack.enter_context(patch.dict(native_ocudu.os.environ, {"CLICOLOR_FORCE": "1"}))
        self.machine = "aarch64"
        self.stack.enter_context(patch.object(native_ocudu.platform, "machine", side_effect=lambda: self.machine))
        self.commands = []
        self.gpu = {"compute_capability": "12.1"}
        # What the doubles write. A test changes one entry to model one defect.
        self.cache = {"CMAKE_BUILD_TYPE:STRING": "Release", "BUILD_TESTING:BOOL": "ON", "ENABLE_CUDA:BOOL": "ON",
                      "CMAKE_CUDA_ARCHITECTURES:UNINITIALIZED": "121", "CUDA_ACCEL_FOUND:BOOL": "TRUE",
                      "MCPU:STRING": "neoverse-v2", "CMAKE_GENERATOR:INTERNAL": "Unix Makefiles",
                      "CMAKE_HOME_DIRECTORY:INTERNAL": str(self.source)}
        self.cuda_compiler = ("NVIDIA", "13.5.37")
        self.cuda_code = "sm_121"
        self.kernels = KERNELS
        self.host_flags = "-DENABLE_CUDA -Wall -mcpu=neoverse-v2 -O3 -DNDEBUG -fno-trapping-math -fno-math-errno"
        self.built_kernels = KERNELS
        self.sections = [".text", ".nv_fatbin", ".data"]
        self.elf_machine = 183
        self.banner = "OCUDU 5G gNB version 26.04.0 (27e84cb3da)"
        self.rebuilt_cache = None  # entries a build step changes when it reruns CMake

    def write_configuration(self):
        (self.build / "CMakeCache.txt").write_text(
            "# synthetic cache\n" + "".join(f"{key}={value}\n" for key, value in self.cache.items()))
        records = self.build / "CMakeFiles/3.28.3"
        records.mkdir(parents=True, exist_ok=True)
        (records / "CMakeCXXCompiler.cmake").write_text(
            'set(CMAKE_CXX_COMPILER "/usr/bin/c++")\nset(CMAKE_CXX_COMPILER_ID "GNU")\n'
            'set(CMAKE_CXX_COMPILER_VERSION "13.3.0")\n')
        (records / "CMakeCUDACompiler.cmake").write_text(
            'set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")\n'
            f'set(CMAKE_CUDA_COMPILER_ID "{self.cuda_compiler[0]}")\n'
            f'set(CMAKE_CUDA_COMPILER_VERSION "{self.cuda_compiler[1]}")\n')
        code = self.cuda_code.replace("sm_", "compute_")
        nvcc = f'/usr/local/cuda/bin/nvcc -O3 "--generate-code=arch={code},code=[{code},{self.cuda_code}]" -x cu -c '
        entries = [{"directory": str(self.build / "lib/phy/cuda"), "file": f"{self.source}/lib/phy/cuda/src/{name}.cu",
                    "command": nvcc + f"{self.source}/lib/phy/cuda/src/{name}.cu",
                    "output": f"lib/phy/cuda/CMakeFiles/ocudu_phy_cuda.dir/src/{name}.cu.o"} for name in self.kernels]
        entries.append({"directory": str(self.build / "lib/phy/upper"), "file": f"{self.source}/lib/phy/upper/copy.cu",
                        "command": nvcc + f"{self.source}/lib/phy/upper/copy.cu",
                        "output": "lib/phy/upper/CMakeFiles/ocudu_upper_phy.dir/copy.cu.o"})
        test = f"{self.source}/{native_ocudu.TEST}.cpp"
        entries.append({"directory": str(self.build), "file": test, "output": "test.o",
                        "command": f"/usr/bin/c++ {self.host_flags} -std=gnu++17 -o test.o -c {test}"})
        (self.build / "compile_commands.json").write_text(json.dumps(entries))

    def record_fake_command(self, command, run, filename, env=None, live=False):
        self.commands.append([str(word) for word in command])
        # CMake's output is shown while it runs. Short identity commands only go to their log.
        self.assertEqual(live, filename in ("configure.log", "build.log"), filename)
        if filename in ("configure.log", "build.log"):
            self.assertIn(str(self.source.parent), env["GIT_CEILING_DIRECTORIES"].split(":"))
            self.assertNotIn("CLICOLOR_FORCE", env)
        if filename == "configure.log":
            self.write_configuration()
        if filename == "build.log":
            if self.rebuilt_cache:
                self.cache.update(self.rebuilt_cache)
                self.write_configuration()
            for path in (self.build / native_ocudu.TEST, self.build / native_ocudu.GNB):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(fake_elf(self.sections, self.elf_machine) + path.name.encode())
                path.chmod(0o755)
            for name in self.built_kernels:
                kernel = self.build / f"lib/phy/cuda/CMakeFiles/ocudu_phy_cuda.dir/src/{name}.cu.o"
                kernel.parent.mkdir(parents=True, exist_ok=True)
                kernel.write_bytes(b"object")
            (self.build / native_ocudu.KERNEL_LIBRARY).write_bytes(b"archive")
        if filename == "gnb-version.txt":
            (run / filename).write_text(f"\n--== OCUDU gNB (commit 27e84cb3da) ==--\n\n{self.banner}\n")
        if filename == "native-help.txt":
            # The pinned help text holds one non-ASCII character (an em dash).
            (run / filename).write_text(f"  --rx-device-grid MODE  (default: auto with CUDA {chr(0x2014)} managed)\n"
                                        "  --require-gpu-resident  Fail unless resident\n", encoding="utf-8")

    def build_full(self, machine="spark", mcpu=None, target="all", pin="pin"):
        args = Namespace(jobs=8, machine=machine, mcpu=mcpu)
        native_ocudu.native_build(args, self.build, self.run, self.gpu, pin, target, {})

    def published(self):
        return sorted(path.name for path in (self.root / "isac/bin").iterdir() if path.is_symlink())


class BuildContractTest(FakeBuildCase):
    def test_full_spark_recipe_and_stale_binary_rejection(self):
        self.build_full()
        for flag in ["-DENABLE_CUDA=ON", "-DMCPU=neoverse-v2", "-DCMAKE_CUDA_ARCHITECTURES=121",
                     "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_TESTING=ON", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"]:
            self.assertIn(flag, self.commands[0])
        self.assertEqual(self.commands[0][:5], ["cmake", "-S", str(self.source), "-B", str(self.build)])
        self.assertEqual(self.commands[1], ["cmake", "--build", str(self.build), "-j8"])
        venv_bin = self.root / "isac/bin"
        self.assertEqual((venv_bin / "gnb").resolve(), self.build / "apps/gnb/gnb")
        self.assertEqual((venv_bin / native_ocudu.TEST.name).resolve(), self.build / native_ocudu.TEST)
        self.assertIn([str(self.build / "apps/gnb/gnb"), "--version"], self.commands)
        native_ocudu.checked_build(self.build, self.run, self.gpu, "pin")
        (self.build / native_ocudu.TEST).write_bytes(b"changed")
        with self.assertRaises(ValueError):
            native_ocudu.checked_build(self.build, self.run, self.gpu, "pin")

    def test_receipt_records_compilers_cuda_units_targets_and_hashes(self):
        self.build_full()
        receipt = json.loads((self.build / "isac-native.json").read_text())
        self.assertEqual(receipt, json.loads((self.run / "native-build.json").read_text()))
        self.assertEqual(receipt["toolchain"]["cuda"],
                         {"path": "/usr/local/cuda/bin/nvcc", "id": "NVIDIA", "version": "13.5.37"})
        self.assertEqual(receipt["toolchain"]["cxx"]["version"], "13.3.0")
        self.assertEqual(receipt["cpu_target"], {"MCPU": "neoverse-v2"})
        self.assertEqual((receipt["compile"]["cuda_units"], receipt["compile"]["cuda_code"]), (3, "sm_121"))
        self.assertEqual(len(receipt["compile"]["kernel_objects"]), 2)
        self.assertIn("-mcpu=neoverse-v2", receipt["compile"]["host_flags"])
        targets = receipt["targets"]
        self.assertEqual(sorted(targets), ["gnb", "libocudu_phy_cuda.a", "pusch_e2e_pipeline_test"])
        self.assertEqual(targets["gnb"]["sha256"], native_ocudu.digest(self.build / native_ocudu.GNB))
        self.assertEqual(targets["gnb"]["sha256"], receipt["gnb_sha256"])
        self.assertEqual(targets["pusch_e2e_pipeline_test"]["sha256"], receipt["binary_sha256"])
        self.assertTrue(targets["gnb"]["cuda_device_code"] and targets["pusch_e2e_pipeline_test"]["cuda_device_code"])
        self.assertNotIn("cuda_device_code", targets["libocudu_phy_cuda.a"])
        self.assertEqual(receipt["source_kind"], "verified-archive")

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
        self.cache["CUDA_ACCEL_FOUND:BOOL"] = "FALSE"
        with self.assertRaisesRegex(ValueError, "CUDA_ACCEL_FOUND"):
            self.build_full()
        self.assertEqual(len(self.commands), 1)
        self.assertFalse((self.build / "isac-native.json").exists())

    def test_each_wrong_cache_value_stops_before_the_build(self):
        for key, value in [("CMAKE_BUILD_TYPE:STRING", "Debug"), ("ENABLE_CUDA:BOOL", "OFF"),
                           ("BUILD_TESTING:BOOL", "OFF"), ("CMAKE_CUDA_ARCHITECTURES:UNINITIALIZED", "120"),
                           ("MCPU:STRING", "native"), ("CMAKE_HOME_DIRECTORY:INTERNAL", "/another/source")]:
            original = self.cache[key]
            self.cache[key] = value
            self.commands.clear()
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key.split(":")[0] + "="):
                self.build_full()
            self.assertEqual(len(self.commands), 1)
            self.cache[key] = original

    def test_wrong_toolchain_stops_before_or_right_after_configuration(self):
        self.codes = "sm_90\nsm_120\n"
        with self.assertRaisesRegex(ValueError, "nvcc cannot emit sm_121"):
            self.build_full()
        self.assertEqual(self.commands, [])
        self.codes = "sm_121\n"
        defects = [("cuda_compiler", ("Clang", "18.1.3"), "expected NVIDIA nvcc"),
                   ("cuda_code", "sm_120", "not compiled for sm_121: copy.cu, ldpc_decoder_flexible.cu, pusch_e2e.cu"),
                   ("kernels", (), "no compile command for the in-tree CUDA kernels"),
                   ("host_flags", "-DENABLE_CUDA -mcpu=native -O3 -DNDEBUG -fno-trapping-math -fno-math-errno",
                    "not compiled with -mcpu=neoverse-v2"),
                   ("host_flags", "-mcpu=neoverse-v2 -O3 -DNDEBUG -fno-trapping-math -fno-math-errno",
                    "not compiled with -DENABLE_CUDA"),
                   ("host_flags", "-DENABLE_CUDA -mcpu=neoverse-v2 -O2 -g", "-O3 -DNDEBUG -fno-trapping-math")]
        for field, value, message in defects:
            original = getattr(self, field)
            setattr(self, field, value)
            self.commands.clear()
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, message):
                self.build_full()
            self.assertEqual(len(self.commands), 1, "the build must not start")
            setattr(self, field, original)
        (self.build / "CMakeFiles/3.99.0").mkdir()
        (self.build / "CMakeFiles/3.99.0/CMakeCXXCompiler.cmake").write_text("")
        with self.assertRaisesRegex(ValueError, "new build directory"):
            self.build_full()

    def test_build_without_compiled_kernels_or_device_code_publishes_nothing(self):
        defects = [("built_kernels", KERNELS[:1], "1 CUDA kernel object"),
                   ("sections", [".text", ".data"], "contains no CUDA device code"),
                   ("elf_machine", 62, "built for ELF machine 62")]
        for field, value, message in defects:
            original = getattr(self, field)
            setattr(self, field, value)
            for stale in self.build.glob("lib/phy/cuda/CMakeFiles/ocudu_phy_cuda.dir/src/*.o"):
                stale.unlink()
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                self.build_full()
            self.assertFalse((self.build / "isac-native.json").exists())
            self.assertEqual(self.published(), [])
            setattr(self, field, original)
        (self.build / native_ocudu.TEST).write_bytes(b"#!/bin/sh\n")
        with self.assertRaisesRegex(ValueError, "not a 64-bit little-endian ELF"):
            native_ocudu.elf_sections(self.build / native_ocudu.TEST)

    def test_cache_regenerated_by_the_build_is_archived_and_checked_again(self):
        self.rebuilt_cache = {"CMAKE_VERBOSE_MAKEFILE:BOOL": "FALSE"}
        self.build_full()
        receipt = json.loads((self.build / "isac-native.json").read_text())
        self.assertIn("CMAKE_VERBOSE_MAKEFILE:BOOL=FALSE", (self.run / "CMakeCache.txt").read_text())
        self.assertEqual(receipt["cache_sha256"], native_ocudu.digest(self.run / "CMakeCache.txt"))
        self.assertEqual(receipt["cache_sha256"], native_ocudu.digest(self.build / "CMakeCache.txt"))
        self.rebuilt_cache = {"CMAKE_BUILD_TYPE:STRING": "Debug"}
        with self.assertRaisesRegex(ValueError, "CMAKE_BUILD_TYPE=Release; found 'Debug'"):
            self.build_full()
        self.assertFalse((self.build / "isac-native.json").exists())

    def test_test_target_build_is_marked_and_checks_the_same_cuda_evidence(self):
        self.build_full(target="baseline")
        self.assertEqual(self.commands[1][-2:], ["--target", "pusch_e2e_pipeline_test"])
        receipt = json.loads((self.build / "isac-native.json").read_text())
        self.assertEqual((receipt["target"], sorted(receipt["targets"])),
                         ("baseline", ["libocudu_phy_cuda.a", "pusch_e2e_pipeline_test"]))
        self.assertEqual(self.published(), ["pusch_e2e_pipeline_test"])

    def test_local_profiles_name_the_gpu_architecture_and_cpu_target(self):
        self.gpu = {"compute_capability": "8.9"}
        self.machine = "x86_64"
        self.elf_machine = 62
        self.cuda_code = "sm_89"
        self.codes = "sm_89\n"
        del self.cache["MCPU:STRING"]
        self.cache.update({"CMAKE_CUDA_ARCHITECTURES:UNINITIALIZED": "89", "MARCH:STRING": "native",
                           "MTUNE:STRING": "generic"})
        self.host_flags = "-DENABLE_CUDA -march=native -mtune=generic -O3 -DNDEBUG"
        self.build_full(machine="local")
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=89", self.commands[0])
        self.assertFalse([word for word in self.commands[0] if word.startswith("-DMCPU")])
        receipt = json.loads((self.build / "isac-native.json").read_text())
        self.assertEqual((receipt["machine"], receipt["cuda_arch"], receipt["cpu_target"]),
                         ("local", "89", {"MARCH": "native", "MTUNE": "generic"}))
        with self.assertRaisesRegex(ValueError, "AArch64 compiler setting"):
            self.build_full(machine="local", mcpu="neoverse-v2")
        # A Spark receipt is required by the measured run. A local build cannot stand in for it.
        with self.assertRaisesRegex(ValueError, "Build identity changed"):
            native_ocudu.checked_build(self.build, self.run, self.gpu, "pin")
        self.machine = "aarch64"
        self.commands.clear()
        with self.assertRaisesRegex(ValueError, "--mcpu"):
            self.build_full(machine="local")
        self.assertEqual(self.commands, [])
        self.elf_machine = 183
        self.cache["MCPU:STRING"] = "neoverse-v2"
        self.host_flags = "-DENABLE_CUDA -mcpu=neoverse-v2 -O3 -DNDEBUG"
        self.build_full(machine="local", mcpu="neoverse-v2")
        self.assertIn("-DMCPU=neoverse-v2", self.commands[0])


class ToolOutputTest(unittest.TestCase):
    """What a person sees while a tool runs and after it fails."""

    def run_tool(self, script, live):
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(output):
            run = Path(temp)
            try:
                native_ocudu.record_command([sys.executable, "-c", script], run, "tool.log", live=live)
                error = None
            except native_ocudu.CommandFailed as failure:
                error = failure
            return error, (run / "tool.log").read_text(), output.getvalue(), (run / "commands.sh").read_text()

    def test_live_command_prints_while_it_runs_and_both_modes_keep_the_log(self):
        # The stand-in tool flushes each line, as CMake does. A tool that buffers one stream decides
        # its own line order, and an interpreter setting such as PYTHONUNBUFFERED must not change it.
        script = ("import sys; print('-- configuring', flush=True); "
                  "sys.stderr.write('CMake Error at top\\n'); sys.stderr.flush(); sys.exit(3)")
        error, log, shown, history = self.run_tool(script, live=True)
        self.assertEqual(log, "-- configuring\nCMake Error at top\n")
        self.assertIn("-- configuring\nCMake Error at top\n", shown)
        self.assertEqual((error.status, error.live, error.log.name), (3, True, "tool.log"))
        self.assertEqual(str(error), f"{Path(sys.executable).name} exited with status 3 (see tool.log)")
        self.assertIn("-c", history)
        error, log, shown, _ = self.run_tool(script, live=False)
        self.assertEqual((log, error.live), ("-- configuring\nCMake Error at top\n", False))
        self.assertNotIn("-- configuring\nCMake Error at top\n", shown)
        error, log, shown, _ = self.run_tool("print('done', flush=True)", live=True)
        self.assertEqual((error, log), (None, "done\n"))
        self.assertIn("done\n", shown)
        error, _, _, _ = self.run_tool("import os, signal; os.kill(os.getpid(), signal.SIGKILL)", live=True)
        self.assertIn("was stopped by signal 9", str(error))

    def test_output_the_terminal_cannot_encode_does_not_stop_the_build(self):
        class Terminal(io.StringIO):
            encoding = "ascii"

        terminal = Terminal()
        with redirect_stdout(terminal):
            native_ocudu.show(f"  --rx-device-grid MODE  (auto {chr(0x2014)} managed)\n")
        self.assertEqual(terminal.getvalue(), "  --rx-device-grid MODE  (auto ? managed)\n")

    def test_excerpt_quotes_cmake_errors_then_build_errors_then_the_end_of_the_log(self):
        excerpt = native_ocudu.failure_excerpt(CONFIGURE_WITHOUT_GTEST)
        self.assertEqual(excerpt[0], "CMake Error at /usr/share/cmake-3.28/Modules/"
                                     "FindPackageHandleStandardArgs.cmake:230 (message):")
        self.assertEqual(excerpt[1:3], ["  Could NOT find GTest (missing: GTEST_LIBRARY GTEST_INCLUDE_DIR",
                                        "  GTEST_MAIN_LIBRARY)"])
        self.assertEqual(excerpt[-1], "  CMakeLists.txt:468 (find_package)")
        self.assertFalse([line for line in excerpt if "\x1b" in line or "ARMPL" in line or "Configuring" in line])
        two = "CMake Error at a.cmake:1 (message):\n  first\n\nCMake Error at b.cmake:2 (message):\n  second\n-- done\n"
        self.assertEqual(native_ocudu.failure_excerpt(two),
                         ["CMake Error at a.cmake:1 (message):", "  first",
                          "CMake Error at b.cmake:2 (message):", "  second"])
        build = ("[ 10%] Building CXX object lib/support/CMakeFiles/ocudu_support.dir/error.cpp.o\n"
                 "/src/lib/phy/x.cpp:12:5: error: unused variable 'n' [-Werror=unused-variable]\n"
                 "gmake[2]: *** [lib/phy/CMakeFiles/x.dir/build.make:76: x.cpp.o] Error 1\n"
                 "[ 11%] Building CXX object lib/phy/y.cpp.o\n"
                 "/usr/bin/ld: libocudu_ldpc_cuda.a(e.cpp.o): undefined reference to `cuCtxCreate'\n")
        self.assertEqual(native_ocudu.failure_excerpt(build), [build.splitlines()[index] for index in (1, 2, 4)])
        plain = "".join(f"line {number}\n" for number in range(50))
        self.assertEqual(native_ocudu.failure_excerpt(plain, limit=3), ["line 47", "line 48", "line 49"])
        many = "".join(f"x.cpp:{number}: error: e\n" for number in range(40))
        self.assertEqual(native_ocudu.failure_excerpt(many, limit=5)[-2:], ["x.cpp:4: error: e", "... (35 more lines)"])
        self.assertEqual(native_ocudu.failure_excerpt(""), [])

    def test_missing_package_failure_names_every_absent_package_and_the_install_command(self):
        report = {"facts": {"host_packages": [{"package": "libgtest-dev", "found": False},
                                              {"package": "libyaml-cpp-dev", "found": False},
                                              {"package": "libsctp-dev", "found": True},
                                              {"package": "libmbedtls-dev", "found": None}]}}
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / "configure.log"
            log.write_text(CONFIGURE_WITHOUT_GTEST)
            error = native_ocudu.CommandFailed(["cmake", "-S", "source"], 1, log, live=True)
            text = native_ocudu.explain(error, report).splitlines()
            self.assertEqual(text[0], "cmake exited with status 1 (see configure.log)")
            self.assertEqual(text[1], "  CMake Error at /usr/share/cmake-3.28/Modules/"
                                      "FindPackageHandleStandardArgs.cmake:230 (message):")
            self.assertEqual(text[-4:], [
                "CMake could not find a required development package: libgtest-dev.",
                "The same configure requires these next, and the compiler cannot find them either: libyaml-cpp-dev.",
                "Install with: sudo apt-get install libgtest-dev libyaml-cpp-dev",
                "Then run the same command again. Nothing needs to be cleaned first."])
            # On the screen a live command has already printed its error: only the remedy is added.
            self.assertEqual(native_ocudu.explain(error, report, quote=False).splitlines(), [text[0], *text[-4:]])
            # Without the probe the package CMake named is still the remedy.
            self.assertEqual(native_ocudu.explain(error).splitlines()[-2],
                             "Install with: sudo apt-get install libgtest-dev")
            for words, package in (("yaml-cpp is required to build ocudu", "libyaml-cpp-dev"),
                                   ("Could NOT find SCTP (missing: SCTP_LIBRARIES)", "libsctp-dev"),
                                   ("Could NOT find MbedTLS (missing: MBEDTLS_LIBRARIES)", "libmbedtls-dev")):
                log.write_text(f"CMake Error at CMakeLists.txt:1 (message):\n  {words}\n")
                self.assertIn(f"sudo apt-get install {package}", native_ocudu.explain(error, {"facts": {}}))
            # Any other failure is quoted and gets no package advice, whatever the probe saw.
            log.write_text("-- Could NOT find libdw (missing: LIBDW_LIBRARY)\n"
                           "CMake Error at x:1 (message):\n  No CUDA\n")
            self.assertEqual(native_ocudu.explain(error, report).splitlines(),
                             [text[0], "  CMake Error at x:1 (message):", "    No CUDA"])
            log.unlink()
            self.assertEqual(native_ocudu.explain(error, report), text[0])
        self.assertEqual(native_ocudu.explain(ValueError("Missing host tool: nvcc."), report), "Missing host tool: nvcc.")

    def test_package_probe_asks_the_compiler_and_reports_unknown_without_one(self):
        present = {"gtest/gtest.h", "netinet/sctp.h", "mbedtls/md.h"}
        libraries = {"libgtest.a", "libsctp.so"}

        def compiler(command):
            if "-include" in command:
                return "" if command[command.index("-include") + 1] in present else None
            name = command[-1].split("=", 1)[1] if command[-1].startswith("-print-file-name=") else None
            if name:
                return f"/usr/lib/aarch64-linux-gnu/{name}\n" if name in libraries else name + "\n"
            return ""

        with patch.object(native_ocudu, "tool_output", side_effect=compiler):
            found = {item["package"]: item["found"] for item in native_ocudu.host_packages(["c++"])}
        # A header without its library, or neither, counts as absent.
        self.assertEqual(found, {"libgtest-dev": True, "libyaml-cpp-dev": False, "libsctp-dev": True,
                                 "libmbedtls-dev": False})
        with patch.object(native_ocudu, "tool_output", return_value=None):
            self.assertEqual([item["found"] for item in native_ocudu.host_packages(["c++"])], [None] * 4)
        self.assertEqual([entry[0] for entry in native_ocudu.HOST_PACKAGES], list(PACKAGES))


class ReferenceEnvironmentTest(unittest.TestCase):
    def test_spark_core_layout_is_read_from_the_kernel(self):
        large = {5, 6, 7, 8, 9, 15, 16, 17, 18, 19}
        capacities = {**{cpu: 718 + cpu for cpu in range(20) if cpu not in large},
                      **{cpu: 997 + cpu for cpu in large}, 19: 1024}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "cpuinfo").write_text("\n\n".join(
                f"processor\t: {cpu}\nBogoMIPS\t: 2000.00\nCPU implementer\t: 0x41\nCPU architecture: 8\n"
                f"CPU variant\t: 0x0\nCPU part\t: {'0xd85' if cpu in large else '0xd87'}\nCPU revision\t: 1"
                for cpu in range(20)) + "\n")
            for cpu in range(20):
                (root / f"cpu{cpu}").mkdir()
                (root / f"cpu{cpu}/cpu_capacity").write_text(f"{capacities[cpu]}\n")
            (root / "cpufreq").mkdir()
            topology = native_ocudu.cpu_topology(root / "cpuinfo", root)
            self.assertEqual(topology[5], {"model": "Cortex-X925", "capacity": 1002})
            self.assertEqual(topology[14], {"model": "Cortex-A725", "capacity": 732})
            self.assertEqual(native_ocudu.high_capacity_cpus(topology), (sorted(large), "cpu_capacity"))
            for cpu in range(20):
                (root / f"cpu{cpu}/cpu_capacity").unlink()
            topology = native_ocudu.cpu_topology(root / "cpuinfo", root)
            self.assertEqual(native_ocudu.high_capacity_cpus(topology), (sorted(large), "core model"))
            self.assertEqual(native_ocudu.cpu_list(native_ocudu.high_capacity_cpus(topology)[0]), "5-9,15-19")
            self.assertEqual(native_ocudu.cpu_topology(root / "absent", root / "absent"), {})
        uniform = {cpu: {"capacity": 1024} for cpu in range(4)}
        self.assertEqual(native_ocudu.high_capacity_cpus(uniform), ([], None))
        self.assertEqual(native_ocudu.high_capacity_cpus({}), ([], None))
        self.assertEqual((native_ocudu.cpu_list([3, 1, 2, 7]), native_ocudu.cpu_list([])), ("1-3,7", ""))

    def test_host_observation_parses_tool_banners_and_never_raises_on_a_missing_tool(self):
        answers = {"nvcc --version": "nvcc: NVIDIA (R) Cuda compiler driver\nCuda compilation tools, release 13.5, V13.5.37\n",
                   "nvcc --list-gpu-code": "sm_120\nsm_121\n",
                   "c++ --version": "c++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0\nCopyright (C) 2023 Free Software Foundation, Inc.\n",
                   "cmake --version": "cmake version 3.28.3\n",
                   f"c++ -mcpu=neoverse-v2 -x c++ -E {os.devnull} -o {os.devnull}": ""}
        with patch.object(native_ocudu, "tool_output", side_effect=lambda command: answers.get(shlex.join(command))), \
                patch.object(native_ocudu.platform, "machine", return_value="aarch64"), \
                patch.dict(native_ocudu.os.environ, {}, clear=False) as environment:
            environment.pop("CXX", None)
            facts = native_ocudu.host_facts(SPARK_GPU, "neoverse-v2")
            self.assertEqual(facts["nvcc"], {"version": "13.5.37", "codes": ["sm_120", "sm_121"]})
            self.assertEqual((facts["cxx"]["family"], facts["cxx"]["version"], facts["cxx"]["accepts_mcpu"]),
                             ("GCC", "13.3.0", True))
            self.assertEqual(facts["cmake"], "3.28.3")
            answers.clear()
            facts = native_ocudu.host_facts(SPARK_GPU, "neoverse-v2")
            self.assertEqual((facts["nvcc"], facts["cmake"]), ({"version": None, "codes": []}, None))
            self.assertEqual((facts["cxx"]["family"], facts["cxx"]["version"], facts["cxx"]["accepts_mcpu"]),
                             (None, None, False))

    def compare(self, profile="spark", mcpu="neoverse-v2", **changes):
        return reference_report(profile, spark_facts(**changes), mcpu)

    def test_accepted_spark_matches_the_platform_and_records_newer_software(self):
        report = self.compare()
        rows = {row["item"]: row for row in report["reference"]}
        self.assertEqual(list(rows), list(native_ocudu.REFERENCE))
        for item in ("machine", "gpu", "compute_capability"):
            self.assertEqual((rows[item]["verdict"], rows[item]["required"]), ("match", True))
        for item in ("cpu_cores", "benchmark_cpus", "host_compiler", "operating_system"):
            self.assertEqual((rows[item]["verdict"], rows[item]["required"]), ("match", False), item)
        self.assertEqual((rows["cuda_toolkit"]["observed"], rows["cuda_toolkit"]["reference"],
                          rows["cuda_toolkit"]["verdict"]), ("13.5.37", "13.0.88", "differs"))
        self.assertEqual((rows["gpu_driver"]["verdict"], rows["cpu_governor"]["verdict"]), ("differs", "differs"))
        self.assertEqual((rows["source_revision"]["observed"], rows["source_revision"]["verdict"]),
                         ("nvcuda_accel_02 at 27e84cb3", "differs"))
        self.assertEqual([need["met"] for need in report["requirements"]], [True, True, True])
        native_ocudu.require_profile(report)
        self.assertEqual(self.compare(governors={})["reference"][5]["verdict"], "unknown")

    def test_wrong_profile_or_toolchain_is_refused_before_configuring(self):
        rtx = {**SPARK_GPU, "name": "NVIDIA RTX 5000 Ada Generation", "compute_capability": "8.9"}
        with self.assertRaisesRegex(ValueError, "Spark profile needs.*gpu: expected NVIDIA GB10.*--machine local"):
            native_ocudu.require_profile(self.compare(gpu=rtx, nvcc={"version": "13.5.37", "codes": ["sm_89"]}))
        with self.assertRaisesRegex(ValueError, "machine: expected aarch64, found x86_64"):
            native_ocudu.require_profile(self.compare(machine="x86_64"))
        local = self.compare("local", None, machine="x86_64", gpu=rtx, nvcc={"version": "12.8.93", "codes": ["sm_89"]})
        self.assertFalse(any(row["required"] for row in local["reference"]))
        self.assertEqual([need["item"] for need in local["requirements"]], ["cuda_code_generation", "cmake"])
        native_ocudu.require_profile(local)
        refusals = [({"nvcc": {"version": "12.4.1", "codes": ["sm_90", "sm_100"]}}, "nvcc emits sm_121"),
                    ({"nvcc": {"version": None, "codes": []}}, "nvcc emits sm_121"),
                    ({"cmake": "3.16.3"}, "CMake 3.18 or newer"), ({"cmake": None}, "CMake 3.18 or newer"),
                    ({"cxx": {**spark_facts()["cxx"], "accepts_mcpu": False}}, "accepts -mcpu=neoverse-v2"),
                    ({"cxx": {**spark_facts()["cxx"], "accepts_mcpu": None}}, "accepts -mcpu=neoverse-v2")]
        for changes, message in refusals:
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "toolchain cannot build.*" + message):
                native_ocudu.require_profile(self.compare(**changes))
        self.assertTrue(self.compare()["benchmark_cpus_allowed"])
        with patch.object(native_ocudu.os, "sched_getaffinity", return_value={0, 1, 2, 3}):
            self.assertFalse(native_ocudu.compare_reference("spark", spark_facts(), "neoverse-v2", LOCK_ENTRY)
                             ["benchmark_cpus_allowed"])

    def test_one_physical_gpu_must_be_selected(self):
        one = "GPU-aaaa, NVIDIA GB10, 12.1, 580.178.04\n"
        two = one + "GPU-bbbb, NVIDIA RTX 5000 Ada Generation, 8.9, 580.178.04\n"
        with patch.object(native_ocudu.subprocess, "check_output") as query, \
                patch.dict(native_ocudu.os.environ, {}, clear=False) as environment:
            environment.pop("CUDA_VISIBLE_DEVICES", None)
            query.return_value = one
            self.assertEqual(native_ocudu.selected_gpu(), {"uuid": "GPU-aaaa", "name": "NVIDIA GB10",
                                                           "compute_capability": "12.1", "driver": "580.178.04"})
            query.return_value = two
            with self.assertRaisesRegex(ValueError, "Select one physical GPU"):
                native_ocudu.selected_gpu()
            environment["CUDA_VISIBLE_DEVICES"] = "GPU-bbbb"
            self.assertEqual(native_ocudu.selected_gpu()["compute_capability"], "8.9")
            environment["CUDA_VISIBLE_DEVICES"] = "0"
            with self.assertRaisesRegex(ValueError, "Select one physical GPU"):
                native_ocudu.selected_gpu()
            query.return_value = "GPU-aaaa, NVIDIA GB10, [N/A], 580.178.04\n"
            environment.pop("CUDA_VISIBLE_DEVICES")
            with self.assertRaisesRegex(ValueError, "invalid compute capability"):
                native_ocudu.selected_gpu()


class EvidenceGradeTest(FakeBuildCase):
    """Grade the files a finished build leaves behind, then damage them one at a time."""

    PIN = LOCK_ENTRY["commit"]

    def setUp(self):
        super().setUp()
        self.build_full(pin=self.PIN)
        self.receipt = json.loads((self.run / "native-build.json").read_text())
        self.report = reference_report("spark", spark_facts(), "neoverse-v2")
        self.platform = {"overrides": {"CUDA_VISIBLE_DEVICES": "0"}}
        self.status = "stage=completed\nexit_code=0\n"

    def grade(self):
        native_ocudu.write_json(self.run / "native-build.json", self.receipt)
        native_ocudu.write_json(self.run / "reference-environment.json", self.report)
        native_ocudu.write_json(self.run / "platform.json", self.platform)
        (self.run / "status.txt").write_text(self.status)
        return native_ocudu.grade(self.run, pin=self.PIN)

    def test_complete_spark_evidence_passes_and_lists_differences_from_the_reference(self):
        failed, lines = self.grade()
        text = "\n".join(lines)
        self.assertFalse(failed, text)
        self.assertNotIn("FAIL", text)
        for expected in (
                "PASS  S1.1 full Release build completed (target all)",
                f"PASS  S1.1 source identity: {self.PIN} (verified-archive)",
                "PASS  S1.1 gNB version banner: OCUDU 5G gNB version 26.04.0 (27e84cb3da)",
                "PASS  S1.1 reference platform (machine, GPU, compute capability): aarch64, NVIDIA GB10, 12.1",
                "PASS  S1.1 toolchain requirements: nvcc emits sm_121; CMake 3.18 or newer; c++ accepts -mcpu=neoverse-v2",
                "PASS  S1.1 build configuration: CMAKE_BUILD_TYPE=Release, ENABLE_CUDA=ON, CUDA_ACCEL_FOUND=TRUE, "
                "BUILD_TESTING=ON, CMAKE_CUDA_ARCHITECTURES=121, MCPU=neoverse-v2",
                "PASS  S1.1 CUDA compilation: 2 kernel sources of 3 CUDA sources for sm_121 by nvcc 13.5.37, "
                "host compiler GNU 13.3.0, device code linked into pusch_e2e_pipeline_test and gnb",
                "PASS  S1.1 compiled targets and hashes: gnb ",
                "NOTE  matches the reference: CPU cores (Cortex-A725, Cortex-X925), high-capacity CPU set "
                "(5-9,15-19), host compiler (GCC 13.3.0), operating system (Ubuntu 24.04)",
                "NOTE  recorded difference, CUDA toolkit: 13.5.37 here; 13.0.88 in the branch notes",
                "NOTE  recorded difference, GPU driver: 580.178.04 here; 580.95.05 in the branch notes",
                "NOTE  recorded difference, OCUDU revision: nvcuda_accel_02 at 27e84cb3 here; "
                "wg/nvcuda_accel_01 at 3c89cdee in the paper",
                "NOTE  S1.2 measurement condition, CPU frequency governor: schedutil here; performance in the paper"):
            self.assertIn(expected, text)
        self.assertEqual([line[:4] for line in lines], ["PASS"] * 8 + ["NOTE"] * 5)
        self.assertNotIn("allowed CPUs", text)
        self.assertNotIn("build environment override", text)

    def test_notes_cover_unobservable_measurement_conditions_and_build_overrides(self):
        self.report = reference_report(
            "spark", spark_facts(high_capacity_cpus="0-9", governors={}, allowed_cpus="0-3"), "neoverse-v2")
        self.report["benchmark_cpus_allowed"] = False
        self.platform = {"overrides": {"CXXFLAGS": "-O2", "CUDA_VISIBLE_DEVICES": "0"}}
        failed, lines = self.grade()
        text = "\n".join(lines)
        self.assertFalse(failed, text)
        for expected in ("NOTE  S1.2 measurement condition, high-capacity CPU set: 0-9 here; 5-9,15-19 in the paper",
                         "NOTE  S1.2 measurement condition, CPU frequency governor: not observable here; "
                         "performance in the paper",
                         "NOTE  S1.2 measurement condition, allowed CPUs: 0-3 here; the measured run needs 5-9,15-19",
                         "NOTE  build environment override: CXXFLAGS=-O2"):
            self.assertIn(expected, text)

    def test_each_damaged_record_fails_its_own_gate(self):
        def receipt(**changes):
            return lambda: self.receipt.update(changes)

        def row(item, **changes):
            return lambda: next(r for r in self.report["reference"] if r["item"] == item).update(changes)

        def target(name, **changes):
            return lambda: self.receipt["targets"][name].update(changes)

        damages = [
            (lambda: setattr(self, "status", "stage=build\nexit_code=1\n"), "full Release build completed"),
            (receipt(target="baseline"), "full Release build completed"),
            (receipt(commit="0" * 40), "source identity"),
            (lambda: (self.run / "gnb-version.txt").write_text("OCUDU 5G gNB version 26.04.0 (deadbeef)\n"),
             "gNB version banner"),
            (lambda: (self.run / "gnb-version.txt").write_text("no banner\n"), "gNB version banner"),
            (row("gpu", observed="NVIDIA RTX 5000 Ada Generation", verdict="differs"), "reference platform"),
            (row("machine", observed="x86_64", verdict="differs"), "reference platform"),
            (lambda: self.report["requirements"][0].update(met=False), "toolchain requirements"),
            (lambda: (self.run / "CMakeCache.txt").write_text(
                (self.run / "CMakeCache.txt").read_text().replace("ENABLE_CUDA:BOOL=ON", "ENABLE_CUDA:BOOL=OFF")),
             "build configuration"),
            (lambda: (self.run / "CMakeCache.txt").write_text((self.run / "CMakeCache.txt").read_text() + "# edited\n"),
             "build configuration"),
            (receipt(cuda_arch="120"), "build configuration"),
            (target("gnb", cuda_device_code=False), "CUDA compilation"),
            (lambda: self.receipt["compile"].update(kernel_objects=[]), "CUDA compilation"),
            (lambda: self.receipt["compile"].update(cuda_code="sm_120"), "CUDA compilation"),
            (lambda: self.receipt["toolchain"]["cuda"].update(id="Clang"), "CUDA compilation"),
            (target("pusch_e2e_pipeline_test", sha256="0" * 64), "compiled targets and hashes"),
        ]
        originals = (copy.deepcopy(self.receipt), copy.deepcopy(self.report), self.status,
                     (self.run / "CMakeCache.txt").read_text(), (self.run / "gnb-version.txt").read_text())
        for index, (damage, gate) in enumerate(damages):
            damage()
            failed, lines = self.grade()
            with self.subTest(index=index, gate=gate):
                self.assertTrue(failed)
                self.assertEqual([line for line in lines if line.startswith("FAIL")][0][:len(gate) + 11],
                                 f"FAIL  S1.1 {gate}")
            self.receipt, self.report, self.status = copy.deepcopy(originals[0]), copy.deepcopy(originals[1]), originals[2]
            (self.run / "CMakeCache.txt").write_text(originals[3])
            (self.run / "gnb-version.txt").write_text(originals[4])
        self.assertFalse(self.grade()[0])

    def test_git_build_must_name_its_revision_and_missing_records_fail_closed(self):
        self.receipt["source_kind"] = "git"
        (self.run / "gnb-version.txt").write_text("OCUDU 5G gNB version 26.04.0 ()\n")
        self.assertTrue(self.grade()[0])
        (self.run / "gnb-version.txt").write_text("OCUDU 5G gNB version 26.04.0 (27e84cb3da)\n")
        self.assertFalse(self.grade()[0])
        for name in ("status.txt", "native-build.json", "reference-environment.json", "CMakeCache.txt",
                     "platform.json", "gnb-version.txt"):
            self.grade()
            (self.run / name).rename(self.run / "moved")
            with self.subTest(missing=name):
                failed, lines = native_ocudu.grade(self.run, pin=self.PIN)
                self.assertTrue(failed)
                self.assertEqual(len(lines), 1)
                self.assertIn("FAIL  S1.1 build evidence is incomplete", lines[0])
            (self.run / "moved").rename(self.run / name)
        self.assertEqual(native_ocudu.grade(self.root / "absent", pin=self.PIN)[0], True)

    def test_local_profile_is_graded_without_the_spark_platform_requirement(self):
        rtx = {**SPARK_GPU, "name": "NVIDIA RTX 5000 Ada Generation", "compute_capability": "8.9"}
        self.receipt.update(machine="local", cuda_arch="89")
        self.receipt["compile"]["cuda_code"] = "sm_89"
        self.report = reference_report(
            "local", spark_facts(gpu=rtx, nvcc={"version": "13.5.37", "codes": ["sm_89"]}), "neoverse-v2")
        cache = (self.run / "CMakeCache.txt").read_text().replace("ARCHITECTURES:UNINITIALIZED=121",
                                                                  "ARCHITECTURES:UNINITIALIZED=89")
        (self.run / "CMakeCache.txt").write_text(cache)
        self.receipt["cache_sha256"] = native_ocudu.digest(self.run / "CMakeCache.txt")
        failed, lines = self.grade()
        text = "\n".join(lines)
        self.assertFalse(failed, text)
        self.assertIn("NOTE  local profile on aarch64, NVIDIA RTX 5000 Ada Generation, 8.9. The DeepSig reference "
                      "platform is aarch64, NVIDIA GB10, 12.1", text)
        self.assertNotIn("reference platform (machine", text)
        self.assertIn("CMAKE_CUDA_ARCHITECTURES=89, MCPU=neoverse-v2", text)


class CommandLineTest(unittest.TestCase):
    def run_main(self, *arguments):
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            try:
                status = native_ocudu.main(list(arguments))
            except SystemExit as stop:
                status = stop.code
        return status, output.getvalue(), errors.getvalue()

    def test_failed_preflight_keeps_its_evidence_and_named_run_directories_are_never_reused(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(native_ocudu, "ROOT", Path(temp).resolve()):
            root = Path(temp).resolve()
            status, output, errors = self.run_main("build", "--run-dir", "evidence/first")
            self.assertEqual(status, 1)
            self.assertIn(f"Run artifacts: {root / 'evidence/first'}", output)
            self.assertIn("preflight: Activate the project environment", errors)
            self.assertEqual((root / "evidence/first/status.txt").read_text(), "stage=preflight\nexit_code=1\n")
            self.assertIn("Activate", (root / "evidence/first/error.txt").read_text())
            status, _, _ = self.run_main("build", "--run-dir", "evidence/first")
            self.assertIn("already exists. Evidence is never overwritten.", str(status))
            self.assertEqual(sorted(path.name for path in (root / "evidence/first").iterdir()),
                             ["error.txt", "status.txt"])
            for arguments in (("build", "--build-dir", "ext/build"), ("build", "--build-dir", "isac"),
                              ("build", "--jobs", "0"), ("build", "--mcpu", "native"), ("run", "--machine", "local"),
                              ("grade",)):
                with self.subTest(arguments=arguments):
                    self.assertEqual(self.run_main(*arguments)[0], 2)
            self.assertFalse((root / "results").exists())

    def test_preflight_records_the_reference_comparison_before_it_refuses_or_builds(self):
        rtx = {**SPARK_GPU, "name": "NVIDIA RTX 5000 Ada Generation", "compute_capability": "8.9"}
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp).resolve()
            (root / "isac").mkdir()
            (root / "ext").mkdir()
            (root / "ext/lock.json").write_text(json.dumps({"dependencies": [dict(LOCK_ENTRY, path="ext/ocudu")]}))
            for target, name, value in ((native_ocudu, "ROOT", root), (native_ocudu.sys, "prefix", str(root / "isac"))):
                stack.enter_context(patch.object(target, name, value))
            which = stack.enter_context(patch.object(native_ocudu.shutil, "which", return_value="/usr/bin/tool"))
            gpu = stack.enter_context(patch.object(native_ocudu, "selected_gpu", return_value=dict(SPARK_GPU)))
            stack.enter_context(patch.object(native_ocudu, "host_facts",
                                             side_effect=lambda device, mcpu: spark_facts(gpu=device)))
            stack.enter_context(patch.object(native_ocudu.platform, "machine", return_value="aarch64"))
            recorded = stack.enter_context(patch.object(native_ocudu, "record_command"))
            build = stack.enter_context(patch.object(native_ocudu, "native_build"))
            checked = stack.enter_context(patch.object(native_ocudu, "checked_build"))
            measured = stack.enter_context(patch.object(native_ocudu, "benchmark"))

            status, _, errors = self.run_main("build", "--run-dir", "evidence/spark", "--jobs", "4")
            self.assertEqual(status, 0, errors)
            options, directory, run, device, pin, target, _ = build.call_args.args
            self.assertEqual((directory, run, device, pin, target, options.jobs, options.machine),
                             (root / "build/ocudu-spark", root / "evidence/spark", SPARK_GPU, LOCK_ENTRY["commit"],
                              "all", 4, "spark"))
            report = json.loads((run / "reference-environment.json").read_text())
            self.assertEqual((report["profile"], report["facts"]["gpu"]), ("spark", SPARK_GPU))
            self.assertIn("cpu_affinity", json.loads((run / "platform.json").read_text()))
            self.assertEqual((run / "status.txt").read_text(), "stage=completed\nexit_code=0\n")
            self.assertIn("pins.txt", [call.args[2] for call in recorded.call_args_list])

            gpu.return_value = rtx
            status, _, errors = self.run_main("build", "--run-dir", "evidence/other-gpu")
            self.assertEqual(status, 1)
            self.assertIn("preflight: The Spark profile needs the DeepSig reference platform", errors)
            rows = json.loads((root / "evidence/other-gpu/reference-environment.json").read_text())["reference"]
            self.assertEqual([row["verdict"] for row in rows[:3]], ["match", "differs", "differs"])
            self.assertEqual((root / "evidence/other-gpu/status.txt").read_text(), "stage=preflight\nexit_code=1\n")
            status, _, errors = self.run_main("build", "--machine", "local", "--run-dir", "evidence/no-cpu-target")
            self.assertEqual((status, build.call_count), (1, 1))
            self.assertIn("preflight: Name the CPU target of an AArch64 local build with --mcpu", errors)
            which.side_effect = lambda tool: None if tool == "nvcc" else "/usr/bin/tool"
            status, _, errors = self.run_main("build", "--run-dir", "evidence/no-nvcc")
            self.assertEqual((status, build.call_count), (1, 1))
            self.assertIn("Missing host tool: nvcc. Put the CUDA toolkit's bin directory on PATH.", errors)

            # The measured run reuses a verified build and never rebuilds when told to skip.
            which.side_effect = None
            gpu.return_value = dict(SPARK_GPU)
            status, _, errors = self.run_main("run", "--profile", "paper", "--skip-build", "--run-dir", "evidence/paper")
            self.assertEqual((status, build.call_count, checked.call_count), (0, 1, 1), errors)
            self.assertEqual(measured.call_args.args[:2], ("paper", root / "build/ocudu-spark"))
            self.assertTrue((root / "evidence/paper/reference-environment.json").is_file())

    def test_failed_configure_leaves_the_cmake_error_and_the_remedy_in_the_evidence(self):
        absent = [{"package": package, "found": package not in ("libgtest-dev", "libyaml-cpp-dev")}
                  for package in PACKAGES]

        def configure(args, build, run, gpu, pin, target, stage):
            stage["stage"] = "configure"
            (run / "configure.log").write_text(CONFIGURE_WITHOUT_GTEST)
            raise native_ocudu.CommandFailed(["cmake", "-S", "source"], 1, run / "configure.log", live=True)

        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp).resolve()
            (root / "isac").mkdir()
            (root / "ext").mkdir()
            (root / "ext/lock.json").write_text(json.dumps({"dependencies": [dict(LOCK_ENTRY, path="ext/ocudu")]}))
            for target, name, value in ((native_ocudu, "ROOT", root), (native_ocudu.sys, "prefix", str(root / "isac"))):
                stack.enter_context(patch.object(target, name, value))
            stack.enter_context(patch.object(native_ocudu.shutil, "which", return_value="/usr/bin/tool"))
            stack.enter_context(patch.object(native_ocudu, "selected_gpu", return_value=dict(SPARK_GPU)))
            stack.enter_context(patch.object(native_ocudu, "host_facts", side_effect=lambda device, mcpu: spark_facts(
                gpu=device, host_packages=absent)))
            stack.enter_context(patch.object(native_ocudu.platform, "machine", return_value="aarch64"))
            stack.enter_context(patch.object(native_ocudu, "record_command"))
            stack.enter_context(patch.object(native_ocudu, "native_build", side_effect=configure))
            status, output, errors = self.run_main("build", "--run-dir", "evidence/no-gtest")
            run = root / "evidence/no-gtest"
            self.assertEqual(status, 1)
            # Advice before CMake runs, without stopping it: CMake searches more places than the compiler.
            self.assertIn("The compiler cannot find these packages, which the configure requires: libgtest-dev, "
                          "libyaml-cpp-dev. CMake decides. If it stops, run: sudo apt-get install libgtest-dev "
                          "libyaml-cpp-dev", output)
            self.assertEqual((run / "status.txt").read_text(), "stage=configure\nexit_code=1\n")
            record = (run / "error.txt").read_text().splitlines()
            self.assertEqual(record[0], "cmake exited with status 1 (see configure.log)")
            self.assertIn("    Could NOT find GTest (missing: GTEST_LIBRARY GTEST_INCLUDE_DIR", record)
            self.assertEqual(record[-2], "Install with: sudo apt-get install libgtest-dev libyaml-cpp-dev")
            # The screen already showed CMake's own error while it ran. Only the verdict and remedy follow.
            self.assertEqual(errors.splitlines()[:2], [
                "configure: cmake exited with status 1 (see configure.log)",
                "CMake could not find a required development package: libgtest-dev."])
            self.assertNotIn("Could NOT find GTest", errors)
            self.assertEqual(errors.splitlines()[-1], f"See {run}")
            self.assertEqual(json.loads((run / "reference-environment.json").read_text())["facts"]["host_packages"],
                             absent)

    def test_grade_command_prints_the_lines_and_fails_on_missing_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            status, output, _ = self.run_main("grade", temp)
            self.assertEqual(status, 1)
            self.assertIn("FAIL  S1.1 build evidence is incomplete", output)
        with patch.object(native_ocudu, "grade", return_value=(False, ["PASS  S1.1 example", "NOTE  example"])):
            self.assertEqual(self.run_main("grade", "anywhere"), (0, "PASS  S1.1 example\nNOTE  example\n", ""))


if __name__ == "__main__":
    unittest.main()
