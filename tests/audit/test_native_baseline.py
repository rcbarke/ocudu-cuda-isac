"""Host-only fixtures for native build identity, reference comparison, evidence grading and measured runs."""

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

# Transcripts in the pinned test's format, described in test_pusch_log.py. Parser fixtures, no measurement.
TRANSCRIPTS = Path(__file__).resolve().parent / "transcripts"
PAPER_LOG = (TRANSCRIPTS / "paper.log").read_text()
SMOKE_LOG = (TRANSCRIPTS / "smoke.log").read_text()
# Listing in the layout `nvidia-smi -q` prints on the Spark with driver 580.178.04, shortened, with invented
# values: two desktop clients and one compute process.
NVIDIA_SMI = """==============NVSMI LOG==============

Timestamp                                              : Thu Nov 19 20:10:24 2026
Driver Version                                         : 580.178.04
CUDA Version                                           : 13.0

Attached GPUs                                          : 1
GPU 0000000F:01:00.0
    Product Name                                       : NVIDIA GB10
    GPU UUID                                           : GPU-spark
    Performance State                                  : P8
    Utilization
        GPU                                            : 3 %
        Memory                                         : 0 %
        Encoder                                        : 0 %
    Temperature
        GPU Current Temp                               : 35 C
    GPU Power Readings
        Average Power Draw                             : 4.31 W
    Processes
        GPU instance ID                   : N/A
        Compute instance ID               : N/A
        Process ID                        : 3120
            Type                          : G
            Name                          : /usr/lib/xorg/Xorg
            Used GPU Memory               : 155 MiB
        GPU instance ID                   : N/A
        Compute instance ID               : N/A
        Process ID                        : 3351
            Type                          : G
            Name                          : /usr/bin/gnome-shell
            Used GPU Memory               : 120 MiB
        GPU instance ID                   : N/A
        Compute instance ID               : N/A
        Process ID                        : 41877
            Type                          : C
            Name                          : python3
            Used GPU Memory               : 2048 MiB
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


class MeasuredRunTest(unittest.TestCase):
    """The command of each profile, what is recorded around it and what a failed run leaves behind."""

    def measure(self, profile, fail=()):
        """Run benchmark() against a recorder. Return (commands, stage, error, files written)."""
        commands, stage = [], {}

        def record(command, run, filename, env=None):
            commands.append([str(word) for word in command])
            (run / filename).write_text("PASS  graded\n" if filename == "verification.txt" else "transcript\n")
            if filename in fail:
                raise native_ocudu.CommandFailed(command, 2, run / filename)

        error = None
        with tempfile.TemporaryDirectory() as temp, patch.object(native_ocudu, "record_command", side_effect=record), \
                patch.object(native_ocudu.os, "sched_getaffinity", return_value=set(range(20))), \
                redirect_stdout(io.StringIO()) as shown:
            run = Path(temp)
            (run / "nvidia-smi.txt").write_text(NVIDIA_SMI)
            try:
                native_ocudu.benchmark(profile, run / "build", run, SPARK_GPU, stage)
            except native_ocudu.CommandFailed as failure:
                error = failure
            conditions = json.loads((run / "measurement-conditions.json").read_text())
            return commands, stage, error, conditions, shown.getvalue()

    def test_paper_run_pins_the_reference_cpus_and_requires_the_resident_path(self):
        commands, stage, error, conditions, shown = self.measure("paper")
        self.assertIsNone(error)
        self.assertEqual(commands[0][:6], ["timeout", "--kill-after=10s", "5m", "taskset", "-c", "5-9,15-19"])
        line = " ".join(commands[0])
        for expected in ("CUDA_VISIBLE_DEVICES=GPU-spark", "OCUDU_PUSCH_ACCELERATION_TIMING=1",
                         "--prb 51 --sinr 30 --mcs 27 --mcs-table qam256", "--layers 1 --ports 1",
                         "--dmrs-type type1 --dmrs-symbols 2,11", "--iterations 200 --warmup 5 --cpu-every 1",
                         "--rx-device-grid direct-managed --require-gpu-resident --gpu-output-buffer pinned --quiet"):
            self.assertIn(expected, line)
        # The transcript validator is the second command. It grades the same log for the paper profile.
        self.assertEqual(commands[1][1:], [str(native_ocudu.ROOT / "scripts/verify_pusch_log.py"),
                                           commands[1][2], "--profile", "paper", "--json", commands[1][6]])
        self.assertEqual((Path(commands[1][2]).name, Path(commands[1][6]).name), ("pusch.log", "table-ii.json"))
        self.assertEqual((stage, shown.strip()), ({"stage": "verify"}, "PASS  graded"))
        # The conditions name the exact command and what shared the GPU before it started.
        self.assertEqual((conditions["profile"], conditions["command"]), ("paper", commands[0]))
        self.assertEqual([client["name"] for client in conditions["gpu_clients"]],
                         ["/usr/lib/xorg/Xorg", "/usr/bin/gnome-shell", "python3"])
        self.assertEqual(conditions["gpu_utilization_percent"], 3)
        # A CPU allocation without the paper's cores is refused in the run stage, before anything starts.
        stage = {}
        with patch.object(native_ocudu.os, "sched_getaffinity", return_value={0, 1, 2, 3}), \
                self.assertRaisesRegex(ValueError, "5-9,15-19"):
            native_ocudu.benchmark("paper", Path("build"), Path("absent"), SPARK_GPU, stage)
        self.assertEqual(stage, {"stage": "run"})

    def test_smoke_run_uses_four_rx_ports_on_any_allowed_cpu(self):
        commands, _, error, conditions, _ = self.measure("smoke")
        self.assertIsNone(error)
        self.assertEqual(commands[0][:4], ["timeout", "--kill-after=10s", "5m", "env"])
        line = " ".join(commands[0])
        for expected in ("--mcs 10 --mcs-table qam64", "--ports 4", "--iterations 20 --warmup 5",
                         "--rx-device-grid direct-managed --require-gpu-resident --gpu-output-buffer vector"):
            self.assertIn(expected, line)
        self.assertNotIn("taskset", line)
        self.assertEqual(commands[1][-4:-1], ["--profile", "smoke", "--json"])
        self.assertEqual((Path(commands[1][-1]).name, conditions["profile"]), ("smoke.json", "smoke"))

    def test_failed_native_test_is_still_graded_and_reported_as_the_failure(self):
        commands, stage, error, _, shown = self.measure("paper", fail=("pusch.log", "verification.txt"))
        self.assertEqual(len(commands), 2, "the transcript of a failed run is graded too")
        self.assertEqual((stage, error.log.name, error.status), ({"stage": "run"}, "pusch.log", 2))
        self.assertIn("PASS  graded", shown)
        # A test that exits zero with a transcript the validator refuses stops in the verify stage.
        commands, stage, error, _, _ = self.measure("paper", fail=("verification.txt",))
        self.assertEqual((len(commands), stage, error.log.name), (2, {"stage": "verify"}, "verification.txt"))

    def test_conditions_are_read_from_the_archived_listing_and_the_kernel(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run, sysfs, proc = root / "run", root / "sys", root / "proc"
            run.mkdir()
            (run / "nvidia-smi.txt").write_text(NVIDIA_SMI)
            for cpu in (5, 19):
                policy = sysfs / f"devices/system/cpu/cpu{cpu}/cpufreq"
                policy.mkdir(parents=True)
                (policy / "scaling_cur_freq").write_text("3900000\n" if cpu == 5 else "2808000\n")
                (policy / "cpuinfo_max_freq").write_text("3900000\n")
            (sysfs / "devices/system/clocksource/clocksource0").mkdir(parents=True)
            (sysfs / "devices/system/clocksource/clocksource0/current_clocksource").write_text("arch_sys_counter\n")
            (sysfs / "module/drm_kms_helper/parameters").mkdir(parents=True)
            (sysfs / "module/drm_kms_helper/parameters/poll").write_text("Y\n")
            for pid, name in (("812", "gnb"), ("77", "odu_low"), ("900", "bash"), ("self", "python3")):
                (proc / pid).mkdir(parents=True)
                (proc / pid / "comm").write_text(name + "\n")
            (proc / "4242").mkdir()  # a process that ended while it was read
            conditions = native_ocudu.measurement_conditions(run, sysfs, proc)
            self.assertEqual(conditions["cpu_frequency"], {"5": {"current_khz": 3900000, "maximum_khz": 3900000},
                                                           "19": {"current_khz": 2808000, "maximum_khz": 3900000}})
            self.assertEqual((conditions["clocksource"], conditions["drm_kms_polling"]), ("arch_sys_counter", "Y"))
            self.assertEqual(conditions["modem_processes"], [{"pid": "77", "name": "odu_low"},
                                                             {"pid": "812", "name": "gnb"}])
            self.assertEqual(conditions["gpu_clients"][0], {"pid": "3120", "type": "G", "name": "/usr/lib/xorg/Xorg",
                                                            "used_gpu_memory": "155 MiB"})
            self.assertEqual((len(conditions["gpu_clients"]), conditions["gpu_utilization_percent"]), (3, 3))
            notes = native_ocudu.condition_notes(conditions)
            self.assertEqual(notes, [
                "NOTE  S1.2 measurement condition, GPU clients at the start: Xorg (G, 155 MiB), gnome-shell "
                "(G, 120 MiB), python3 (C, 2048 MiB); GPU utilization 3 percent",
                "NOTE  S1.2 measurement condition, compute workload on the GPU at the start: python3 (pid 41877); "
                "the entry condition is an idle GPU",
                "NOTE  S1.2 measurement condition, modem process running at the start: odu_low (pid 77), gnb "
                "(pid 812); Table II was measured with the live modem off",
                "NOTE  S1.2 measurement condition, DRM KMS polling: on here; the upstream tuning script "
                "scripts/ocudu_performance turns it off before a latency capture",
                "NOTE  S1.2 measurement condition, CPU clocks: CPUs 5,19 at 2.81 to 3.90 GHz of 3.90 GHz maximum, "
                "clocksource arch_sys_counter"])
            # A host that exposes none of this records unknowns and one line, and never raises.
            empty = native_ocudu.measurement_conditions(root / "absent", root / "absent", root / "absent")
            self.assertEqual((empty["cpu_frequency"], empty["gpu_clients"], empty["modem_processes"]), ({}, [], []))
            self.assertEqual((empty["clocksource"], empty["drm_kms_polling"], empty["gpu_utilization_percent"]),
                             (None, None, None))
            self.assertEqual(native_ocudu.condition_notes(empty),
                             ["NOTE  S1.2 measurement condition, GPU clients at the start: none listed"])
        self.assertEqual(native_ocudu.gpu_processes("    Processes                             : None\n"), [])
        # Older drivers spell the utilization row "Gpu". A row that reports no number stays unknown.
        for row, expected in (("        Gpu                               : 41 %", 41),
                              ("        GPU                               : N/A", None),
                              ("    GPU Current Temp                      : 35 C", None)):
            with tempfile.TemporaryDirectory() as temp:
                Path(temp, "nvidia-smi.txt").write_text("    Utilization\n" + row + "\n")
                found = native_ocudu.measurement_conditions(Path(temp), Path(temp, "sys"), Path(temp, "proc"))
                self.assertEqual(found["gpu_utilization_percent"], expected, row)


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


class PatchedBuildTest(FakeBuildCase):
    """Sprint 5.2: the receiver with the patch series is built from its own tree and never published."""

    PIN = LOCK_ENTRY["commit"]
    IDENTITY = {"schema": "receiver-patch/v1", "series": "patches/ocudu/series.json", "series_sha256": "ab" * 32,
                "base_commit": LOCK_ENTRY["commit"], "patches": [{"file": "0001-tap.patch", "sha256": "cd" * 32}],
                "files": {"include/tap.h": "ef" * 32, "lib/demod.cpp": "01" * 32}, "interfaces": []}

    def setUp(self):
        super().setUp()
        self.tree = self.root / "ext/ocudu-o3"  # beside the locked source, as the double of the commands expects
        self.tree.mkdir()
        self.cache["CMAKE_HOME_DIRECTORY:INTERNAL"] = str(self.tree)
        self.identity = dict(self.IDENTITY, tree_kind="patched-copy", tree=str(self.tree))
        patched = native_ocudu.receiver_patch
        self.stack.enter_context(patch.object(patched, "load_series", side_effect=lambda: {"sha256": "ab" * 32}))
        self.stack.enter_context(patch.object(patched, "verify_tree", side_effect=lambda tree, series: self.identity))
        self.stack.enter_context(patch.object(patched, "identity", side_effect=lambda series: dict(self.IDENTITY)))

    def build_patched(self):
        args = Namespace(jobs=8, machine="spark", mcpu=None)
        native_ocudu.native_build(args, self.build, self.run, self.gpu, self.PIN, "all", {}, source=self.tree,
                                  patch=dict(self.identity))

    def test_the_receipt_carries_the_series_and_nothing_is_published(self):
        self.build_patched()
        receipt = json.loads((self.build / "isac-native.json").read_text())
        self.assertEqual(receipt, json.loads((self.run / "native-build.json").read_text()))
        self.assertEqual((receipt["source"], receipt["locked_source"], receipt["source_kind"]),
                         (str(self.tree), str(self.source), "patched-copy"))
        self.assertEqual(receipt["patch"]["series_sha256"], "ab" * 32)
        self.assertEqual(receipt["patch"]["patches"], [{"file": "0001-tap.patch", "sha256": "cd" * 32}])
        self.assertIn("gnb_sha256", receipt)
        # The links in isac/bin belong to the unpatched build: the control.
        self.assertEqual(self.published(), [])
        configure = next(command for command in self.commands if command[:2] == ["cmake", "-S"])
        self.assertEqual(configure[2], str(self.tree))
        # A build of the locked source says nothing of a patch and is published as before.
        self.cache["CMAKE_HOME_DIRECTORY:INTERNAL"] = str(self.source)
        self.build_full(pin=self.PIN)
        self.assertNotIn("patch", json.loads((self.build / "isac-native.json").read_text()))
        self.assertEqual(self.published(), ["gnb", "pusch_e2e_pipeline_test"])

    def test_a_series_that_changed_during_the_build_leaves_no_receipt(self):
        self.identity = dict(self.identity, series_sha256="00" * 32)
        with self.assertRaisesRegex(ValueError, "The patch series changed while the patched receiver was built"):
            native_ocudu.native_build(Namespace(jobs=8, machine="spark", mcpu=None), self.build, self.run, self.gpu,
                                      self.PIN, "all", {}, source=self.tree, patch=dict(self.IDENTITY))
        self.assertFalse((self.build / "isac-native.json").exists())

    def test_the_patched_build_is_graded_under_its_own_label_against_the_series(self):
        self.banner = "OCUDU 5G gNB version 26.04.0 ()"  # a copy has no revision of its own
        self.build_patched()
        native_ocudu.write_json(self.run / "reference-environment.json", reference_report("spark", spark_facts(), "neoverse-v2"))
        native_ocudu.write_json(self.run / "platform.json", {"overrides": {}})
        (self.run / "status.txt").write_text("stage=completed\nexit_code=0\n")
        failed, lines = native_ocudu.grade(self.run, pin=self.PIN)
        self.assertFalse(failed, lines)
        self.assertIn(f"PASS  S5.2 patched receiver patch series {'ab' * 6} on {self.PIN[:12]}: 0001-tap.patch "
                      f"{'cd' * 6}, 2 files touched, built from a patched-copy", lines)
        self.assertTrue(all(line.startswith(("PASS  S5.2 patched receiver ", "NOTE  ")) for line in lines), lines)
        # The series of the tree moved on: the build is no longer the patched receiver of this tree.
        self.IDENTITY = dict(self.IDENTITY, series_sha256="11" * 32)
        failed, lines = native_ocudu.grade(self.run, pin=self.PIN)
        self.assertTrue(failed)
        self.assertIn("the series of patches/ocudu/ differs from it", "\n".join(lines))

    def test_the_patched_tree_and_its_build_get_their_own_directories(self):
        for arguments, words in ((["build", "--patched-source", "build/x"], "--patched-source can only be used with"),
                                 (["build", "--patched", "--patched-source", "ext/x"], "outside ext/ and isac/"),
                                 (["build", "--patched", "--build-dir", "build/a", "--patched-source", "build/a/src"],
                                  "must be two separate directories")):
            said = io.StringIO()
            with self.assertRaises(SystemExit), redirect_stderr(said):
                native_ocudu.main(arguments)
            self.assertIn(words, said.getvalue())


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


class RunEvidenceGradeTest(FakeBuildCase):
    """Grade the files a measured run leaves behind, then damage them one at a time."""

    PIN = LOCK_ENTRY["commit"]

    def setUp(self):
        super().setUp()
        self.build_full(pin=self.PIN)
        self.receipt = json.loads((self.run / "native-build.json").read_text())
        self.report = reference_report("spark", spark_facts(), "neoverse-v2")
        self.platform = {"profile": "paper", "gpu": dict(SPARK_GPU)}
        self.status = "stage=completed\nexit_code=0\n"
        self.transcript = PAPER_LOG
        self.evidence = self.root / "paper"
        self.evidence.mkdir()
        self.conditions = {"command": self.command("paper"), "gpu_clients": [], "modem_processes": []}

    def command(self, profile):
        """The command benchmark() records for a profile."""
        with patch.object(native_ocudu.os, "sched_getaffinity", return_value=set(range(20))), \
                redirect_stdout(io.StringIO()):
            native_ocudu.benchmark(profile, self.build, self.root, SPARK_GPU, {})
        return json.loads((self.root / "measurement-conditions.json").read_text())["command"]

    def record_fake_command(self, command, run, filename, env=None, live=False):
        if filename in ("pusch.log", "verification.txt"):
            (run / filename).write_text("")
        else:
            super().record_fake_command(command, run, filename, env=env, live=live)

    def grade(self):
        native_ocudu.write_json(self.evidence / "native-build.json", self.receipt)
        native_ocudu.write_json(self.evidence / "reference-environment.json", self.report)
        native_ocudu.write_json(self.evidence / "platform.json", self.platform)
        native_ocudu.write_json(self.evidence / "measurement-conditions.json", self.conditions)
        (self.evidence / "status.txt").write_text(self.status)
        (self.evidence / "pusch.log").write_text(self.transcript)
        return native_ocudu.grade_run(self.evidence, pin=self.PIN)

    def test_complete_paper_evidence_passes_and_lists_conditions_and_differences(self):
        failed, lines = self.grade()
        text = "\n".join(lines)
        self.assertFalse(failed, text)
        self.assertEqual([line[:4] for line in lines], ["PASS"] * 10 + ["NOTE"] * 12)
        binary = self.receipt["binary_sha256"][:12]
        for expected in (
                "PASS  S1.2 paper run completed: the native test and the transcript validator exited zero",
                f"PASS  S1.2 paper measured the full Spark Release build: pusch_e2e_pipeline_test {binary} of OCUDU "
                f"{self.PIN} (verified-archive)",
                "PASS  S1.2 paper ran on the intended device: aarch64, NVIDIA GB10, 12.1, GPU-spark, driver "
                "580.178.04, pinned to CPUs 5-9,15-19",
                "PASS  S1.2 paper CPU decodes: 200/200 CRC passes",
                "PASS  S1.2 paper CUDA work completed on the device: resident pipeline active after warmup",
                "NOTE  S1.2 paper performance, GPU mean: 264.9 us here; 183.9 us in the paper (+44.0 percent)",
                "NOTE  S1.2 measurement condition, GPU clients at the start: none listed",
                "NOTE  matches the reference: CPU cores (Cortex-A725, Cortex-X925), high-capacity CPU set "
                "(5-9,15-19), host compiler (GCC 13.3.0), operating system (Ubuntu 24.04)",
                "NOTE  S1.2 measurement condition, CPU frequency governor: schedutil here; performance in the paper",
                "NOTE  recorded difference, OCUDU revision: nvcuda_accel_02 at 27e84cb3 here; "
                "wg/nvcuda_accel_01 at 3c89cdee in the paper"):
            self.assertIn(expected, text)

    def test_smoke_evidence_needs_no_cpu_pinning_and_carries_no_paper_notes(self):
        self.platform["profile"] = "smoke"
        self.transcript = SMOKE_LOG
        self.conditions = {"command": self.command("smoke")}
        failed, lines = self.grade()
        self.assertFalse(failed, "\n".join(lines))
        self.assertEqual([line[:4] for line in lines], ["PASS"] * 10)
        self.assertIn("PASS  S1.2 smoke ran on the intended device: aarch64, NVIDIA GB10, 12.1, GPU-spark, driver "
                      "580.178.04", lines)
        self.assertIn("PASS  S1.2 smoke GPU decodes: 20/20 CRC passes", lines)
        # A paper run without the paper's CPU set is refused. The smoke command has no taskset.
        self.platform["profile"] = "paper"
        self.transcript = PAPER_LOG
        failed, lines = self.grade()
        self.assertTrue(failed)
        self.assertEqual([line.split(":")[0] for line in lines if line.startswith("FAIL")],
                         ["FAIL  S1.2 paper ran on the intended device"])

    def test_each_damaged_record_fails_its_own_gate(self):
        def receipt(**changes):
            return lambda: self.receipt.update(changes)

        def row(item, **changes):
            return lambda: next(r for r in self.report["reference"] if r["item"] == item).update(changes)

        def words(old, new):
            return lambda: self.conditions.update(command=[new if word == old else word
                                                           for word in self.conditions["command"]])

        def log(old, new):
            return lambda: setattr(self, "transcript", PAPER_LOG.replace(old, new))

        damages = [
            (lambda: setattr(self, "status", "stage=verify\nexit_code=1\n"), "run completed"),
            (lambda: setattr(self, "status", "stage=run\nexit_code=1\n"), "run completed"),
            (receipt(target="baseline"), "measured the full Spark Release build"),
            (receipt(machine="local"), "measured the full Spark Release build"),
            (receipt(commit="0" * 40), "measured the full Spark Release build"),
            (receipt(binary_sha256="0" * 64), "measured the full Spark Release build"),
            (lambda: self.receipt["targets"]["pusch_e2e_pipeline_test"].update(cuda_device_code=False),
             "measured the full Spark Release build"),
            (words("CUDA_VISIBLE_DEVICES=GPU-spark", "CUDA_VISIBLE_DEVICES=GPU-other"), "ran on the intended device"),
            (words("5-9,15-19", "0-4"), "ran on the intended device"),
            (row("gpu", observed="NVIDIA RTX 5000 Ada Generation", verdict="differs"), "ran on the intended device"),
            (log("54296 bits (6787 bytes)", "54240 bits (6780 bytes)"), "workload is the Table II"),
            (log("CPU: 200/200 pass (0.0% BLER)", "CPU: 199/200 pass (0.5% BLER)"), "CPU decodes"),
            (log("GPU: 200/200 pass (0.0% BLER)", "GPU: 0/200 pass (100.0% BLER)"), "GPU decodes"),
            (log("Byte mismatches: 0", "Byte mismatches: 1"), "decoded bytes agree"),
            (log("Output guard failures: 0", "Output guard failures: 1"), "decoded bytes agree"),
            (log("resident GPU timing: 200/200", "resident GPU timing: 0/200"), "CUDA work completed on the device"),
            (log("GPU total:          256.0    264.9    274.0", "GPU total:          256.0      nan    274.0"),
             "service times measured"),
            (log("PASS: Both paths decoded all 200 iterations", "PASS: Both paths agree on BLER"),
             "native test verdict"),
        ]
        originals = (copy.deepcopy(self.receipt), copy.deepcopy(self.report), self.status,
                     copy.deepcopy(self.conditions))
        for index, (damage, gate) in enumerate(damages):
            damage()
            failed, lines = self.grade()
            with self.subTest(index=index, gate=gate):
                self.assertTrue(failed)
                failures = [line for line in lines if line.startswith("FAIL")]
                self.assertEqual(len(failures), 1, failures)
                self.assertTrue(failures[0].startswith(f"FAIL  S1.2 paper {gate}"), failures[0])
                self.assertIn("NOTE  S1.2 paper comparison with Table II withheld: a gate above failed", lines)
            self.receipt, self.report, self.status = (copy.deepcopy(originals[0]), copy.deepcopy(originals[1]),
                                                      originals[2])
            self.conditions, self.transcript = copy.deepcopy(originals[3]), PAPER_LOG
        self.assertFalse(self.grade()[0])

    def test_missing_records_and_build_evidence_fail_closed(self):
        for name in ("status.txt", "native-build.json", "reference-environment.json", "platform.json",
                     "measurement-conditions.json", "pusch.log"):
            self.grade()
            (self.evidence / name).rename(self.evidence / "moved")
            with self.subTest(missing=name):
                failed, lines = native_ocudu.grade_run(self.evidence, pin=self.PIN)
                self.assertTrue(failed)
                self.assertEqual(len(lines), 1)
                self.assertRegex(lines[0], r"^FAIL  S1\.2( paper)? run evidence is incomplete")
            (self.evidence / "moved").rename(self.evidence / name)
        self.assertFalse(self.grade()[0])
        # The evidence of a build is not the evidence of a run.
        self.platform["profile"] = "spark"
        self.assertEqual(self.grade()[1], [f"FAIL  S1.2 run evidence is incomplete in {self.evidence}: KeyError('spark')"])
        self.assertTrue(native_ocudu.grade_run(self.root / "absent", pin=self.PIN)[0])

    def test_recorded_conditions_become_notes(self):
        self.conditions.update(gpu_clients=native_ocudu.gpu_processes(NVIDIA_SMI), gpu_utilization_percent=3,
                               modem_processes=[{"pid": "812", "name": "gnb"}], drm_kms_polling="Y")
        self.report = reference_report("spark", spark_facts(
            governors={f"policy{cpu}": "performance" for cpu in (0, 5, 10, 15)}), "neoverse-v2")
        failed, lines = self.grade()
        text = "\n".join(lines)
        self.assertFalse(failed, text)
        for expected in ("GPU clients at the start: Xorg (G, 155 MiB), gnome-shell (G, 120 MiB), python3 (C, 2048 MiB); "
                         "GPU utilization 3 percent",
                         "compute workload on the GPU at the start: python3 (pid 41877)",
                         "modem process running at the start: gnb (pid 812)",
                         "DRM KMS polling: on here",
                         "NOTE  matches the reference: CPU cores (Cortex-A725, Cortex-X925), high-capacity CPU set "
                         "(5-9,15-19), CPU frequency governor (performance)"):
            self.assertIn(expected, text)
        self.assertNotIn("CPU frequency governor: ", text)


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
            logs = [call.args[2] for call in recorded.call_args_list]
            self.assertIn("pins.txt", logs)
            # A build first installs the missing development libraries, shown while it runs, and only
            # then observes the host, so the recorded package probe describes the host that is built on.
            installer = recorded.call_args_list[logs.index("native-deps.log")]
            self.assertEqual(installer.args[0], ["bash", root / "scripts/install_native_deps.sh"])
            self.assertEqual(installer.kwargs, {"live": True})
            self.assertLess(logs.index("c++.txt"), logs.index("native-deps.log"))
            self.assertLess(logs.index("native-deps.log"), logs.index("pins.txt"))
            self.assertTrue((Path(native_ocudu.__file__).parent / "install_native_deps.sh").is_file())

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
            recorded.reset_mock()
            status, _, errors = self.run_main("run", "--profile", "paper", "--skip-build", "--run-dir", "evidence/paper")
            self.assertEqual((status, build.call_count, checked.call_count), (0, 1, 1), errors)
            self.assertNotIn("native-deps.log", [call.args[2] for call in recorded.call_args_list])

            # A failed package installation stops the build in its own stage, before the host is probed.
            def refuse(command, run_dir, log_name, **options):
                if log_name == "native-deps.log":
                    (run_dir / log_name).write_text("E: Unable to locate package libgtest-dev\n")
                    raise native_ocudu.CommandFailed(command, 1, run_dir / log_name, live=True)

            recorded.side_effect = refuse
            status, _, errors = self.run_main("build", "--run-dir", "evidence/no-apt")
            self.assertEqual((status, build.call_count), (1, 1))
            self.assertEqual((root / "evidence/no-apt/status.txt").read_text(), "stage=prerequisites\nexit_code=1\n")
            self.assertEqual((root / "evidence/no-apt/error.txt").read_text().splitlines(), [
                "bash exited with status 1 (see native-deps.log)", "  E: Unable to locate package libgtest-dev"])
            self.assertIn("prerequisites: bash exited with status 1 (see native-deps.log)", errors)
            self.assertFalse((root / "evidence/no-apt/reference-environment.json").exists())
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
            status, output, _ = self.run_main("grade-run", temp)
            self.assertEqual(status, 1)
            self.assertIn("FAIL  S1.2 run evidence is incomplete", output)
        with patch.object(native_ocudu, "grade", return_value=(False, ["PASS  S1.1 example", "NOTE  example"])):
            self.assertEqual(self.run_main("grade", "anywhere"), (0, "PASS  S1.1 example\nNOTE  example\n", ""))
        with patch.object(native_ocudu, "grade_run", return_value=(True, ["FAIL  S1.2 paper example"])):
            self.assertEqual(self.run_main("grade-run", "anywhere"), (1, "FAIL  S1.2 paper example\n", ""))
        self.assertEqual(self.run_main("grade-run")[0], 2)


if __name__ == "__main__":
    unittest.main()
