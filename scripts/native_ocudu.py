"""Build the locked OCUDU Release tree, run one isolated Spark PUSCH profile and grade the evidence of either.

With --patched the build takes its source from a separate tree of the pinned commit with the patch
series of patches/ocudu/ applied (Sprint 5.2), into a build directory of its own. The locked
checkout is never edited, and the links in isac/bin keep naming the unpatched build.
"""

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile

try:  # imported as scripts.native_ocudu by the tests, run as a file otherwise
    from scripts import receiver_patch, verify_pusch_log
except ImportError:
    import receiver_patch
    import verify_pusch_log

ROOT = Path(__file__).resolve().parents[1]


def locked_source(name):
    """Return the lock entry for a logical dependency name. ext/lock.json owns its path."""
    for item in json.loads((ROOT / "ext/lock.json").read_text())["dependencies"]:
        if item.get("name") == name:
            return item
    raise ValueError(f"ext/lock.json has no dependency named {name!r}")


SOURCE = ROOT / locked_source("ocudu")["path"]
TEST = Path("tests/integrationtests/phy/upper/channel_processors/pusch_e2e_pipeline_test")
GNB = Path("apps/gnb/gnb")
KERNEL_LIBRARY = Path("lib/phy/cuda/libocudu_phy_cuda.a")
KERNEL_OBJECTS = "ocudu_phy_cuda.dir"
MINIMUM_CMAKE = (3, 18)  # lib/phy/cuda/CMakeLists.txt in the pinned source
ELF_MACHINES = {"aarch64": 183, "x86_64": 62}
CPU_MODELS = {(0x41, 0xD85): "Cortex-X925", (0x41, 0xD87): "Cortex-A725"}

# The Spark profile is the build recipe of the pinned branch notes (docs/phy_cuda_acceleration.md).
SPARK = {"compute_capability": "12.1", "mcpu": "neoverse-v2", "cpus": "5-9,15-19"}
AFFINITY = {5, 6, 7, 8, 9, 15, 16, 17, 18, 19}
# Compiler flags the branch notes report for the host objects of the measured Release build.
REFERENCE_FLAGS = ("-O3", "-DNDEBUG", "-fno-trapping-math", "-fno-math-errno")
# Reference environment. "paper" is Pennybacker et al., arXiv:2608.04338v1, Section VI-A and
# Appendix A. "branch notes" is the measurement metadata in docs/phy_cuda_acceleration.md of the
# pinned source. Only the platform rows are requirements, and only for the Spark profile.
REFERENCE = {
    "machine": ("aarch64", "paper"),
    "gpu": ("NVIDIA GB10", "paper"),
    "compute_capability": (SPARK["compute_capability"], "branch notes"),
    "cpu_cores": ("Cortex-A725, Cortex-X925", "paper"),
    "benchmark_cpus": (SPARK["cpus"], "paper"),
    "cpu_governor": ("performance", "paper"),
    "host_compiler": ("GCC 13.3", "branch notes"),
    "operating_system": ("Ubuntu 24.04", "branch notes"),
    "cuda_toolkit": ("13.0.88", "branch notes"),
    "gpu_driver": ("580.95.05", "branch notes"),
    "source_revision": ("wg/nvcuda_accel_01 at 3c89cdee", "paper"),
}
PLATFORM_ROWS = ("machine", "gpu", "compute_capability")
MEASUREMENT_ROWS = ("benchmark_cpus", "cpu_governor")  # conditions of the S1.2 timing run
BUILD_ENVIRONMENT = ("CC", "CXX", "CUDACXX", "CUDAHOSTCXX", "CFLAGS", "CXXFLAGS", "CUDAFLAGS", "LDFLAGS")
# Development packages the pinned configure requires for this recipe: the Ubuntu package, the text
# CMake prints when it is absent, and the header and library CMake looks for. GTest and yaml-cpp are
# required by the top CMakeLists.txt (GTest because BUILD_TESTING is on), SCTP by lib/gateways and
# mbedTLS by lib/security.
HOST_PACKAGES = (
    ("libgtest-dev", r"Could NOT find GTest\b", "gtest/gtest.h", "gtest"),
    ("libyaml-cpp-dev", r"yaml-cpp is required|Could NOT find (YAMLCPP|yaml-cpp)\b", "yaml-cpp/yaml.h", "yaml-cpp"),
    ("libsctp-dev", r"Could NOT find SCTP\b", "netinet/sctp.h", "sctp"),
    ("libmbedtls-dev", r"Could NOT find MbedTLS\b", "mbedtls/md.h", "mbedcrypto"),
)
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")  # terminal colors that a tool forced into its output
BUILD_ERROR = re.compile(r"error:|undefined reference|nvcc fatal|\*\*\* |^FAILED: ")
# OCUDU applications that would share the GPU and the pinned cores with a measured run.
MODEM_PROCESSES = ("gnb", "odu", "odu_low")


class CommandFailed(ValueError):
    """A recorded host command ended with a failure status. Its log holds the tool's own message."""

    def __init__(self, args, status, log, live=False):
        self.status, self.log, self.live = status, log, live
        ending = f"was stopped by signal {-status}" if status < 0 else f"exited with status {status}"
        super().__init__(f"{Path(str(args[0])).name} {ending} (see {log.name})")


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def read_log(path):
    """Native tools print UTF-8 text. A stray byte must not hide the rest of a log."""
    return path.read_text(encoding="utf-8", errors="replace")


def show(text):
    """Print tool output. A character the terminal cannot encode must not stop a build."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    sys.stdout.write(text.encode(encoding, "replace").decode(encoding))
    sys.stdout.flush()


def record_command(args, run_dir, log_name, *, env=None, live=False):
    """Run one host command with its output kept in a log. A live command also prints while it runs."""
    command = shlex.join([str(arg) for arg in args])
    with (run_dir / "commands.sh").open("a") as history:
        history.write(command + "\n")
    print(f"{command}\n  log: {run_dir / log_name}", flush=True)
    path = run_dir / log_name
    with path.open("wb") as log:
        if live:
            process = subprocess.Popen(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=ROOT)
            with process.stdout:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    show(line.decode("utf-8", "replace"))
            status = process.wait()
        else:
            status = subprocess.run(args, env=env, stdout=log, stderr=subprocess.STDOUT, check=False,
                                    cwd=ROOT).returncode
    if status:
        raise CommandFailed(args, status, path, live)


def failure_excerpt(text, limit=30):
    """Lines of a failed tool log that name the failure: CMake errors, else compile and link errors, else its end."""
    lines = [ANSI_ESCAPE.sub("", line).rstrip() for line in text.splitlines()]
    chosen = []
    for start, first in enumerate(lines):
        if not first.startswith("CMake Error"):
            continue
        block = [first]
        for line in lines[start + 1:]:
            if line.startswith(("-- ", "CMake Error", "CMake Warning")):
                break
            block.append(line)
        while block and not block[-1]:
            block.pop()
        chosen += block
    if not chosen:
        chosen = [line for line in lines if BUILD_ERROR.search(line)] or [line for line in lines if line][-limit:]
    return chosen[:limit] + ([f"... ({len(chosen) - limit} more lines)"] if len(chosen) > limit else [])


def explain(error, report=None, quote=True):
    """What failed, the failing tool's own words (quote) and, for a missing package, the remedy."""
    lines = [str(error)]
    if not isinstance(error, CommandFailed):
        return lines[0]
    excerpt = failure_excerpt(read_log(error.log)) if error.log.is_file() else []
    if quote:
        lines += ["  " + line for line in excerpt]
    words = "\n".join(excerpt)
    named = [package for package, pattern, _, _ in HOST_PACKAGES if re.search(pattern, words)]
    if named:
        probed = (report or {}).get("facts", {}).get("host_packages") or []
        later = [item["package"] for item in probed if item["found"] is False and item["package"] not in named]
        lines.append("CMake could not find a required development package: " + ", ".join(named) + ".")
        if later:
            lines.append("The same configure requires these next, and the compiler cannot find them either: "
                         + ", ".join(later) + ".")
        lines.append("Install with: sudo apt-get install " + " ".join(named + later))
        lines.append("Then run the same command again. Nothing needs to be cleaned first.")
    return "\n".join(lines)


def tool_output(command):
    """Text printed by a short host command, or None when it is absent or fails."""
    try:
        done = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=60,
                              stdin=subprocess.DEVNULL, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout + done.stderr if done.returncode == 0 else None


def cpu_list(cpus):
    """Format CPU numbers the way taskset and the paper write them: 5-9,15-19."""
    ranges = []
    for cpu in sorted(set(cpus)):
        if ranges and cpu == ranges[-1][1] + 1:
            ranges[-1][1] = cpu
        else:
            ranges.append([cpu, cpu])
    return ",".join(str(first) if first == last else f"{first}-{last}" for first, last in ranges)


def selected_gpu():
    """Identify the one physical GPU this process will build for and measure."""
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,name,compute_cap,driver_version", "--format=csv,noheader"],
        text=True,
    )
    devices = [[v.strip() for v in row] for row in csv.reader(rows.splitlines())]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        devices = [row for row in devices if row[0] == visible or (visible == "0" and len(devices) == 1)]
    if len(devices) != 1 or len(devices[0]) != 4:
        raise ValueError("Select one physical GPU with CUDA_VISIBLE_DEVICES set to its full NVIDIA GPU UUID.")
    uuid, name, cc, driver = devices[0]
    if not re.fullmatch(r"\d+\.\d+", cc):
        raise ValueError(f"nvidia-smi reported an invalid compute capability: {cc!r}")
    return dict(uuid=uuid, name=name, compute_capability=cc, driver=driver)


def cpu_target(machine, mcpu):
    """Return the explicit AArch64 -mcpu value of this build, or None on other architectures."""
    if machine == "spark":
        return SPARK["mcpu"]
    if platform.machine() == "aarch64":
        if not mcpu:
            raise ValueError("Name the CPU target of an AArch64 local build with --mcpu "
                             "(for example --mcpu native).")
        return mcpu
    if mcpu:
        raise ValueError("--mcpu is an AArch64 compiler setting.")
    return None


def cache_values(path):
    entries = {}
    for line in read_log(path).splitlines():
        match = re.fullmatch(r"([A-Za-z_][A-Za-z_0-9]*):[^=]+=(.*)", line)
        if match:
            entries[match[1]] = match[2]
    return entries


def archived_cache(build, run, expected):
    """Copy the CMake cache into the evidence and require the recipe values in that copy."""
    copy = run / "CMakeCache.txt"
    shutil.copyfile(build / "CMakeCache.txt", copy)
    values = cache_values(copy)
    for key, expected_value in expected.items():
        if values.get(key) != expected_value:
            raise ValueError(f"Native configuration requires {key}={expected_value}; found {values.get(key)!r}")
    return values


def cpu_topology(cpuinfo=Path("/proc/cpuinfo"), sysfs=Path("/sys/devices/system/cpu")):
    """Core model and scheduler capacity of each logical CPU. Empty fields where the kernel has none."""
    cpus = {}
    try:
        blocks = cpuinfo.read_text(errors="replace").split("\n\n")
    except OSError:
        blocks = []
    for block in blocks:
        fields = dict((key.strip(), value.strip()) for key, _, value in
                      (line.partition(":") for line in block.splitlines()))
        try:
            identity = (int(fields["CPU implementer"], 16), int(fields["CPU part"], 16))
            cpus[int(fields["processor"])] = {"model": CPU_MODELS.get(identity, "0x%02x:0x%03x" % identity)}
        except (KeyError, ValueError):
            continue
    for directory in sysfs.glob("cpu[0-9]*"):
        try:
            capacity = int((directory / "cpu_capacity").read_text())
            cpus.setdefault(int(directory.name[3:]), {})["capacity"] = capacity
        except (OSError, ValueError):
            continue
    return dict(sorted(cpus.items()))


def high_capacity_cpus(topology):
    """CPUs the kernel rates near its largest capacity, else the Cortex-X925 cores. Empty if unknown."""
    capacities = {cpu: item["capacity"] for cpu, item in topology.items() if item.get("capacity")}
    if len(capacities) == len(topology) and len(set(capacities.values())) > 1:
        # The Spark's large cores report slightly different capacities (about 997 to 1024).
        limit = 0.85 * max(capacities.values())
        return sorted(cpu for cpu, capacity in capacities.items() if capacity >= limit), "cpu_capacity"
    large = sorted(cpu for cpu, item in topology.items() if item.get("model") == "Cortex-X925")
    if large and len(large) < len(topology):
        return large, "core model"
    return [], None


def host_packages(cxx):
    """Ask the compiler whether it can see each development package the configure requires.

    The answer is advice for a failed configure and never stops a build: CMake also searches
    prefixes the compiler does not. None means the compiler could not be asked.
    """
    usable = tool_output([*cxx, "-x", "c++", "-E", os.devnull, "-o", os.devnull]) is not None
    packages = []
    for package, _, header, library in HOST_PACKAGES:
        found = None
        if usable:
            seen = tool_output([*cxx, "-x", "c++", "-E", "-include", header, os.devnull, "-o", os.devnull])
            names = [(tool_output([*cxx, f"-print-file-name=lib{library}{suffix}"]) or "").strip()
                     for suffix in (".so", ".a")]
            found = seen is not None and any(os.path.isabs(name) for name in names)
        packages.append({"package": package, "header": header, "library": library, "found": found})
    return packages


def host_facts(gpu, mcpu):
    """Observe what the reference comparison and the build need. A missing fact is None, not an error."""
    topology = cpu_topology()
    large, basis = high_capacity_cpus(topology)
    governors = {}
    for policy in Path("/sys/devices/system/cpu/cpufreq").glob("policy*"):
        try:
            governors[policy.name] = (policy / "scaling_governor").read_text().strip()
        except OSError:
            continue
    try:
        release = platform.freedesktop_os_release()
    except (AttributeError, OSError):
        release = {}
    nvcc = tool_output(["nvcc", "--version"]) or ""
    toolkit = re.search(r"release \d+\.\d+, V(\d+(?:\.\d+)+)", nvcc)
    cxx = shlex.split(os.environ.get("CXX", "")) or ["c++"]
    banner = tool_output([*cxx, "--version"]) or ""
    first = banner.splitlines()[0] if banner else ""
    versions = re.findall(r"\d+\.\d+\.\d+", first)
    family = "GCC" if "Free Software Foundation" in banner else "clang" if "clang" in banner else None
    accepts = None
    if mcpu and platform.machine() == "aarch64":
        accepts = tool_output([*cxx, f"-mcpu={mcpu}", "-x", "c++", "-E", os.devnull, "-o", os.devnull]) is not None
    cmake = re.search(r"cmake version (\d+(?:\.\d+)+)", tool_output(["cmake", "--version"]) or "")
    return {
        "machine": platform.machine(), "kernel": platform.release(), "node": platform.node(),
        "os": {key.lower(): release.get(key) for key in ("ID", "VERSION_ID", "PRETTY_NAME")},
        "cpus": {str(cpu): item for cpu, item in topology.items()},
        "cpu_models": sorted({item["model"] for item in topology.values() if item.get("model")}),
        "high_capacity_cpus": cpu_list(large) if large else None, "high_capacity_basis": basis,
        "allowed_cpus": cpu_list(os.sched_getaffinity(0)),
        "governors": dict(sorted(governors.items())),
        "nvcc": {"version": toolkit[1] if toolkit else None,
                 "codes": (tool_output(["nvcc", "--list-gpu-code"]) or "").split()},
        "cxx": {"command": shlex.join(cxx), "banner": first, "family": family,
                "version": versions[-1] if versions else None, "accepts_mcpu": accepts},
        "cmake": cmake[1] if cmake else None,
        "host_packages": host_packages(cxx),
        "gpu": gpu,
    }


def compare_reference(machine, facts, mcpu, source):
    """Compare this host with the DeepSig reference and list what this build needs from its toolchain."""
    gpu, system, cxx = facts["gpu"], facts["os"], facts["cxx"]
    arch = gpu["compute_capability"].replace(".", "")
    governors = sorted(set(facts["governors"].values()))
    compiler = f"{cxx['family']} {cxx['version']}" if cxx["family"] and cxx["version"] else None
    distribution = (f"{(system['id'] or '').capitalize()} {system['version_id']}"
                    if system.get("id") and system.get("version_id") else None)
    observed = {
        "machine": (facts["machine"], facts["machine"] == "aarch64"),
        "gpu": (gpu["name"], "GB10" in gpu["name"]),
        "compute_capability": (gpu["compute_capability"], gpu["compute_capability"] == SPARK["compute_capability"]),
        "cpu_cores": (", ".join(facts["cpu_models"]) or None,
                      facts["cpu_models"] == ["Cortex-A725", "Cortex-X925"]),
        "benchmark_cpus": (facts["high_capacity_cpus"], facts["high_capacity_cpus"] == SPARK["cpus"]),
        "cpu_governor": (", ".join(governors) or None, governors == ["performance"]),
        "host_compiler": (compiler, cxx["family"] == "GCC" and (cxx["version"] or "").startswith("13.3.")),
        "operating_system": (distribution, distribution == "Ubuntu 24.04"),
        "cuda_toolkit": (facts["nvcc"]["version"], facts["nvcc"]["version"] == "13.0.88"),
        "gpu_driver": (gpu["driver"], gpu["driver"] == "580.95.05"),
        "source_revision": (f"{source.get('source_ref', 'locked')} at {source['commit'][:8]}",
                            source["commit"].startswith("3c89cdee")),
    }
    rows = []
    for item, (reference, origin) in REFERENCE.items():
        value, same = observed[item]
        rows.append({"item": item, "reference": reference, "source": origin, "observed": value,
                     "verdict": "unknown" if value is None else "match" if same else "differs",
                     "required": machine == "spark" and item in PLATFORM_ROWS})
    version = tuple(int(part) for part in (facts["cmake"] or "0").split(".")[:2])
    needs = [{"item": "cuda_code_generation", "needed": f"nvcc emits sm_{arch}",
              "observed": " ".join(facts["nvcc"]["codes"]) or None,
              "met": f"sm_{arch}" in facts["nvcc"]["codes"]},
             {"item": "cmake", "needed": "CMake %d.%d or newer" % MINIMUM_CMAKE, "observed": facts["cmake"],
              "met": version >= MINIMUM_CMAKE}]
    if mcpu:
        needs.append({"item": "cpu_target", "needed": f"{cxx['command']} accepts -mcpu={mcpu}",
                      "observed": cxx["banner"] or None, "met": cxx["accepts_mcpu"] is True})
    return {"schema": "isac-reference-environment/v1", "profile": machine, "reference": rows,
            "requirements": needs, "facts": facts,
            "benchmark_cpus_allowed": AFFINITY.issubset(os.sched_getaffinity(0))}


def require_profile(report):
    """Stop before configuring when the host cannot satisfy the selected profile."""
    wrong = [f"{row['item']}: expected {row['reference']}, found {row['observed']}"
             for row in report["reference"] if row["required"] and row["verdict"] != "match"]
    if wrong:
        raise ValueError("The Spark profile needs the DeepSig reference platform (" + "; ".join(wrong)
                         + "). Use --machine local on another host.")
    unmet = [f"{need['needed']} (found: {need['observed']})" for need in report["requirements"]
             if not need["met"]]
    if unmet:
        raise ValueError("The host toolchain cannot build this profile: " + "; ".join(unmet))


def cmake_compilers(build):
    """Return the compilers CMake selected. PATH and the cache alone do not identify them."""
    found = {}
    for language in ("CXX", "CUDA"):
        records = sorted((build / "CMakeFiles").glob(f"*/CMake{language}Compiler.cmake"))
        if len(records) != 1:
            raise ValueError(f"Expected one CMake {language} compiler record under {build / 'CMakeFiles'}; "
                             f"found {len(records)}. Use a new build directory after changing the toolchain.")
        text = read_log(records[0])
        fields = {}
        for field, suffix in (("path", ""), ("id", "_ID"), ("version", "_VERSION")):
            match = re.search(rf'^set\(CMAKE_{language}_COMPILER{suffix} "([^"]*)"\)', text, re.MULTILINE)
            fields[field] = match[1] if match else None
        if not all(fields.values()):
            raise ValueError(f"{records[0]} does not identify the {language} compiler")
        found[language.lower()] = fields
    if found["cuda"]["id"] != "NVIDIA":
        raise ValueError(f"CMake selected {found['cuda']['id']} for CUDA sources; expected NVIDIA nvcc")
    return found


def compile_plan(build, arch, mcpu, reference_flags):
    """Read the compile commands CMake generated and require CUDA code for this GPU architecture."""
    path = build / "compile_commands.json"
    if not path.is_file():
        raise ValueError(f"CMake wrote no compile commands to {path}")
    entries = json.loads(path.read_text(encoding="utf-8"))

    def words(entry):
        return entry.get("arguments") or shlex.split(entry.get("command", ""))

    cuda = [entry for entry in entries if entry["file"].endswith(".cu")]
    kernels = sorted(entry["output"] for entry in cuda if KERNEL_OBJECTS in entry.get("output", ""))
    if not kernels:
        raise ValueError("CMake generated no compile command for the in-tree CUDA kernels.")
    code = re.compile(rf"\bsm_{arch}\b")
    wrong = sorted(Path(entry["file"]).name for entry in cuda
                   if not any(code.search(word) for word in words(entry)))
    if wrong:
        raise ValueError(f"CUDA sources are not compiled for sm_{arch}: " + ", ".join(wrong))
    test = [entry for entry in entries if Path(entry["file"]).name == TEST.name + ".cpp"]
    if len(test) != 1:
        raise ValueError(f"Expected one compile command for {TEST.name}.cpp; found {len(test)}")
    flags = words(test[0])
    needed = ["-DENABLE_CUDA"] + ([f"-mcpu={mcpu}"] if mcpu else [])
    needed += list(REFERENCE_FLAGS) if reference_flags else ["-O3", "-DNDEBUG"]
    missing = [flag for flag in needed if flag not in flags]
    if missing:
        raise ValueError(f"{TEST.name} is not compiled with " + " ".join(missing))
    return {"compile_commands_sha256": digest(path), "units": len(entries), "cuda_units": len(cuda),
            "cuda_code": f"sm_{arch}", "kernel_objects": kernels,
            "host_flags": [flag for flag in flags if flag.startswith(("-m", "-O", "-f", "-D", "-std"))]}


def elf_sections(path):
    """Return the machine code and section names of a 64-bit little-endian ELF file."""
    with path.open("rb") as stream:
        header = stream.read(64)
        if len(header) < 64 or header[:4] != b"\x7fELF" or header[4:6] != b"\x02\x01":
            raise ValueError(f"{path} is not a 64-bit little-endian ELF file")
        machine = struct.unpack_from("<H", header, 18)[0]
        table_offset = struct.unpack_from("<Q", header, 40)[0]
        entry_size, count, names_index = struct.unpack_from("<HHH", header, 58)
        if entry_size < 64 or not 0 < names_index < count:
            raise ValueError(f"{path} has no readable section table")
        stream.seek(table_offset)
        table = stream.read(entry_size * count)
        if len(table) != entry_size * count:
            raise ValueError(f"{path} has a truncated section table")
        fields = [struct.unpack_from("<IIQQQQ", table, index * entry_size) for index in range(count)]
        stream.seek(fields[names_index][4])
        names = stream.read(fields[names_index][5])
    sections = []
    for name_offset, *_ in fields:
        end = names.find(b"\0", name_offset)
        sections.append(names[name_offset:end if end >= 0 else None].decode("ascii", "replace"))
    return machine, sections


def built_targets(build, plan, executables):
    """Hash the build products and require compiled CUDA kernels inside each executable."""
    missing = [name for name in plan["kernel_objects"] if not (build / name).is_file()]
    if missing:
        raise ValueError(f"The build left {len(missing)} CUDA kernel object(s) uncompiled, first: {missing[0]}")
    if not (build / KERNEL_LIBRARY).is_file():
        raise ValueError(f"The build did not produce the CUDA kernel library {build / KERNEL_LIBRARY}")
    targets = {}
    for path in [*executables, build / KERNEL_LIBRARY]:
        targets[path.name] = {"path": str(path.relative_to(build)), "bytes": path.stat().st_size,
                              "sha256": digest(path)}
    expected = ELF_MACHINES.get(platform.machine())
    for path in executables:
        machine, sections = elf_sections(path)
        if expected is not None and machine != expected:
            raise ValueError(f"{path} was built for ELF machine {machine}, not for {platform.machine()}")
        if ".nv_fatbin" not in sections:
            raise ValueError(f"{path.name} contains no CUDA device code (.nv_fatbin section). "
                             "The executable was linked without the CUDA kernels.")
        targets[path.name]["cuda_device_code"] = True
    return targets


def native_build(args, build, run, gpu, pin, selected_target, stage, source=None, patch=None):
    """Configure, build and inspect one tree. source and patch name a patched tree and its series.

    Without them the tree is the locked checkout, and its executables are published into isac/bin.
    A patched build is never published there: the accepted unpatched build stays the control.
    """
    source = Path(source) if source is not None else SOURCE
    arch = gpu["compute_capability"].replace(".", "")
    mcpu = cpu_target(args.machine, args.mcpu)
    stage["stage"] = "configure"
    # Invalidate old proof even if this rebuild fails partway.
    (build / "isac-native.json").unlink(missing_ok=True)
    native_targets = subprocess.check_output(["nvcc", "--list-gpu-code"], text=True).split()
    if f"sm_{arch}" not in native_targets:
        raise ValueError(f"nvcc cannot emit sm_{arch}. Available targets: {' '.join(native_targets)}")

    options = ["-DCMAKE_BUILD_TYPE=Release", "-DENABLE_CUDA=ON",
               f"-DCMAKE_CUDA_ARCHITECTURES={arch}", "-DBUILD_TESTING=ON",
               "-DDISABLE_MBEDTLS=OFF", "-DDISABLE_SCTP=OFF", "-DENABLE_UHD=OFF", "-DENABLE_ZEROMQ=OFF",
               "-DENABLE_PLUGINS=OFF", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"]
    if mcpu:
        options.append(f"-DMCPU={mcpu}")
    # An archive has no upstream .git. Keep its version script from reporting
    # the enclosing research repository's HEAD as OCUDU's revision.
    cmake_env = os.environ.copy()
    cmake_env.pop("CLICOLOR_FORCE", None)  # keep terminal color codes out of the archived logs
    if not (source / ".git").exists():
        ceilings = cmake_env.get("GIT_CEILING_DIRECTORIES", "").split(os.pathsep)
        cmake_env["GIT_CEILING_DIRECTORIES"] = os.pathsep.join(filter(None, [str(source.parent), *ceilings]))
    record_command(["cmake", "-S", source, "-B", build, *options], run, "configure.log", env=cmake_env, live=True)
    expected = {"CMAKE_HOME_DIRECTORY": str(source), "CMAKE_BUILD_TYPE": "Release",
                "BUILD_TESTING": "ON", "ENABLE_CUDA": "ON", "CMAKE_CUDA_ARCHITECTURES": arch,
                "CUDA_ACCEL_FOUND": "TRUE"}
    if mcpu:
        expected["MCPU"] = mcpu
    values = archived_cache(build, run, expected)
    # Evidence of real CUDA compilation: the compilers CMake selected and the commands it will run.
    toolchain = cmake_compilers(build)
    toolchain["generator"] = values.get("CMAKE_GENERATOR")
    toolchain["compiler_launcher"] = values.get("CMAKE_CXX_COMPILER_LAUNCHER") or None
    plan = compile_plan(build, arch, mcpu, reference_flags=args.machine == "spark")

    stage["stage"] = "build"
    command = ["cmake", "--build", build, f"-j{args.jobs}"]
    if selected_target == "baseline":
        command += ["--target", TEST.name]
    record_command(command, run, "build.log", env=cmake_env, live=True)
    stage["stage"] = "inspect"
    # The build may regenerate its build system. The receipt names the cache as the build left it.
    values = archived_cache(build, run, expected)
    binary = build / TEST
    record_command([binary, "--help"], run, "native-help.txt")
    if "--require-gpu-resident" not in read_log(run / "native-help.txt"):
        raise ValueError("Built program lacks the expected resident PUSCH test interface.")
    cpu = {"MCPU": mcpu} if mcpu else {key: values.get(key) for key in ("MARCH", "MTUNE")}
    stamp = dict(schema="isac-native-build/v2", commit=pin, binary_sha256=digest(binary),
                 cache_sha256=digest(run / "CMakeCache.txt"), cuda_arch=arch, machine=args.machine,
                 target=selected_target, build_evidence=str(run), source=str(source), build_directory=str(build),
                 source_kind="git" if (source / ".git").exists() else "verified-archive",
                 options=options, cpu_target=cpu, toolchain=toolchain, compile=plan, jobs=args.jobs,
                 host=platform.node(), completed_utc=datetime.now(timezone.utc).isoformat())
    if patch is not None:
        # The receipt of a patched receiver names its series, and the tree is checked once more
        # after the build: the objects were compiled from exactly the pin with the series applied.
        stamp["patch"] = receiver_patch.verify_tree(source, receiver_patch.load_series())
        if not receiver_patch.same_series(stamp["patch"], patch):
            raise ValueError("The patch series changed while the patched receiver was built. Build again.")
        stamp["source_kind"] = stamp["patch"]["tree_kind"]
        stamp["locked_source"] = str(SOURCE)
    binaries = [binary]
    if selected_target == "all":
        gnb = build / GNB
        if not os.access(gnb, os.X_OK):
            raise ValueError(f"Full build did not produce the gNB: {gnb}")
        record_command([gnb, "--version"], run, "gnb-version.txt")
        if "OCUDU 5G gNB version" not in read_log(run / "gnb-version.txt"):
            raise ValueError("Full build did not report the expected gNB version banner.")
        stamp["gnb_sha256"] = digest(gnb)
        binaries.append(gnb)
    stamp["targets"] = built_targets(build, plan, binaries)
    if patch is not None:
        write_json(run / "native-build.json", stamp)
        write_json(build / "isac-native.json", stamp)
        return

    # Validate all destinations before replacing links. Builds in different build
    # directories can finish concurrently and publish into this same environment.
    venv_bin = ROOT / "isac/bin"
    with (venv_bin / ".isac-native.lock").open("a") as link_lock:
        fcntl.flock(link_lock, fcntl.LOCK_EX)
        with tempfile.TemporaryDirectory(dir=venv_bin) as temp:
            for output in binaries:
                link = venv_bin / output.name
                if link.exists() and not link.is_symlink():
                    raise ValueError(f"Cannot replace an existing non-symlink at {link}")
                (Path(temp) / output.name).symlink_to(output)
            for output in binaries:
                (Path(temp) / output.name).replace(venv_bin / output.name)
    write_json(run / "native-build.json", stamp)
    write_json(build / "isac-native.json", stamp)


def checked_build(build, run, gpu, pin):
    stamp = json.loads((build / "isac-native.json").read_text())
    cache = build / "CMakeCache.txt"
    if (stamp["commit"] != pin or stamp["machine"] != "spark" or
        stamp["cuda_arch"] != gpu["compute_capability"].replace(".", "") or
        stamp["source"] != str(SOURCE) or stamp["binary_sha256"] != digest(build / TEST) or
        stamp["cache_sha256"] != digest(cache)):
        raise ValueError("Build identity changed. Rebuild using scripts/build_ocudu.sh.")
    shutil.copyfile(cache, run / cache.name)
    write_json(run / "native-build.json", stamp)


def read_value(path):
    """One-line kernel or driver setting, or None where the host does not expose it."""
    try:
        return path.read_text(errors="replace").strip() or None
    except OSError:
        return None


def gpu_processes(listing):
    """Clients in the Processes section of `nvidia-smi -q`: pid, type (C, G or C+G), name, memory."""
    clients = []
    for line in listing.splitlines():
        key, _, value = (part.strip() for part in line.partition(":"))
        if key == "Process ID":
            clients.append({"pid": value})
        elif clients and key in ("Type", "Name", "Used GPU Memory"):
            clients[-1][key.lower().replace(" ", "_")] = value
    return clients


def measurement_conditions(run, sysfs=Path("/sys"), proc=Path("/proc")):
    """Observe what a timing run shares the machine with. A fact the host does not expose is None.

    The governors and the allowed CPUs are in platform.json. The GPU clients come from the
    nvidia-smi listing the preflight has just archived in this run directory.
    """
    frequency = {}
    for cpu in sorted(AFFINITY):
        policy = sysfs / f"devices/system/cpu/cpu{cpu}/cpufreq"
        current, maximum = read_value(policy / "scaling_cur_freq"), read_value(policy / "cpuinfo_max_freq")
        if current and current.isdigit():
            frequency[str(cpu)] = {"current_khz": int(current),
                                   "maximum_khz": int(maximum) if maximum and maximum.isdigit() else None}
    modem = []
    for entry in proc.glob("[0-9]*"):
        name = read_value(entry / "comm")
        if name in MODEM_PROCESSES:
            modem.append({"pid": entry.name, "name": name})
    try:
        listing = read_log(run / "nvidia-smi.txt")
    except OSError:
        listing = ""
    # The Utilization section prints "GPU : 0 %" with driver 580 on the Spark and "Gpu" with older drivers.
    utilization = re.search(r"^\s*GPU\s*:\s*(\d+) %", listing, re.MULTILINE | re.IGNORECASE)
    try:
        load = list(os.getloadavg())
    except OSError:
        load = None
    return {"observed_utc": datetime.now(timezone.utc).isoformat(), "cpu_frequency": frequency,
            "clocksource": read_value(sysfs / "devices/system/clocksource/clocksource0/current_clocksource"),
            "drm_kms_polling": read_value(sysfs / "module/drm_kms_helper/parameters/poll"),
            "load_average": load, "gpu_utilization_percent": int(utilization[1]) if utilization else None,
            "gpu_clients": gpu_processes(listing), "modem_processes": sorted(modem, key=lambda item: int(item["pid"]))}


def benchmark(profile, build, run, gpu, stage):
    stage["stage"] = "run"
    paper = profile == "paper"
    iterations = verify_pusch_log.PROFILES[profile]["iterations"]
    if paper and not AFFINITY.issubset(os.sched_getaffinity(0)):
        raise ValueError("The paper profile requires cores 5-9,15-19 in this process's allowed CPU set.")
    command = ["timeout", "--kill-after=10s", "5m"]
    if paper:
        command += ["taskset", "-c", SPARK["cpus"]]
    overrides = {"CUDA_VISIBLE_DEVICES": gpu["uuid"], "OCUDU_PUSCH_ACCELERATION_TIMING": "1"}
    command += ["env", *(f"{key}={value}" for key, value in overrides.items()), build / TEST,
                "--prb", "51", "--sinr", "30", "--mcs", "27" if paper else "10",
                "--mcs-table", "qam256" if paper else "qam64", "--layers", "1",
                "--ports", "1" if paper else "4", "--dmrs-type", "type1", "--dmrs-symbols", "2,11",
                "--iterations", str(iterations), "--warmup", "5", "--cpu-every", "1",
                "--rx-device-grid", "direct-managed", "--require-gpu-resident",
                "--gpu-output-buffer", "pinned" if paper else "vector", "--quiet"]
    # What the run starts next to: GPU clients, a running modem, CPU clocks. Recorded, never refused.
    conditions = measurement_conditions(run)
    conditions.update(profile=profile, command=[str(word) for word in command])
    write_json(run / "measurement-conditions.json", conditions)
    failure = None
    try:
        record_command(command, run, "pusch.log")
    except CommandFailed as error:
        failure = error
    else:
        stage["stage"] = "verify"
    # The transcript of a failed run is graded too: its FAIL lines name what the test reported.
    verify = [sys.executable, ROOT / "scripts/verify_pusch_log.py", run / "pusch.log", "--profile", profile,
              "--json", run / verify_pusch_log.PROFILES[profile]["json"]]
    try:
        record_command(verify, run, "verification.txt")
    except CommandFailed as error:
        failure = failure or error
    if (run / "verification.txt").is_file():
        print(read_log(run / "verification.txt").strip(), flush=True)
    if failure:
        raise failure


def grade(evidence, pin=None):
    """Grade one build evidence directory. Return (failed, lines) with PASS, FAIL and NOTE lines."""
    lines = []
    failed = False

    label = "S1.1"

    def check(passed, text, detail=""):
        nonlocal failed
        failed = failed or not passed
        lines.append(f"{'PASS' if passed else 'FAIL'}  {label} {text}{'' if passed else detail}")

    try:
        status = dict(line.split("=", 1) for line in (evidence / "status.txt").read_text().split())
        receipt = json.loads((evidence / "native-build.json").read_text())
        # The build of the patched receiver of Sprint 5.2 is graded by the same rules, under its own label.
        label = "S5.2 patched receiver" if receipt.get("patch") else "S1.1"
        report = json.loads((evidence / "reference-environment.json").read_text())
        cache = cache_values(evidence / "CMakeCache.txt")
        overrides = json.loads((evidence / "platform.json").read_text()).get("overrides", {})
        banner = read_log(evidence / "gnb-version.txt") if receipt.get("target") == "all" else ""
        toolchain, plan, targets = receipt["toolchain"], receipt["compile"], receipt["targets"]
        arch, cpu = receipt["cuda_arch"], receipt["cpu_target"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        return True, [f"FAIL  {label} build evidence is incomplete in {evidence}: {error!r}"]

    if receipt.get("patch"):
        # The series the receipt names must be the series of this tree, patch by patch and file by file.
        patch = receipt["patch"]
        try:
            current = receiver_patch.identity(receiver_patch.load_series())
            same, why = receiver_patch.same_series(patch, current), "the series of patches/ocudu/ differs from it"
        except receiver_patch.PatchError as error:
            same, why = False, str(error)
        names = ", ".join(f"{item['file']} {item['sha256'][:12]}" for item in patch.get("patches", []))
        check(same and receipt["source_kind"] in receiver_patch.KINDS and patch.get("base_commit") == receipt["commit"],
              f"patch series {str(patch.get('series_sha256'))[:12]} on {receipt['commit'][:12]}: {names}, "
              f"{len(patch.get('files', {}))} files touched, built from a {receipt['source_kind']}", f": {why}")
    check(status.get("stage") == "completed" and status.get("exit_code") == "0"
          and receipt["target"] == "all", "full Release build completed (target all)",
          f": stage={status.get('stage')}, exit_code={status.get('exit_code')}, target={receipt['target']}")
    pin = pin or locked_source("ocudu")["commit"]
    check(receipt["commit"] == pin, f"source identity: {receipt['commit']} ({receipt['source_kind']})",
          f": ext/lock.json pins {pin}")
    revision = re.search(r"^OCUDU 5G gNB version \S+ \(([0-9a-f]*)\)$", banner, re.MULTILINE)
    named = bool(revision) and (pin.startswith(revision[1]) if revision[1]
                                else receipt["source_kind"] in ("verified-archive", receiver_patch.KIND_COPY))
    check(named, f"gNB version banner: {revision[0] if revision else 'absent'}",
          ": the banner must name the locked revision")

    platform_rows = [row for row in report["reference"] if row["item"] in PLATFORM_ROWS]
    summary = ", ".join(f"{row['observed']}" for row in platform_rows)
    if receipt["machine"] == "spark":
        check(all(row["verdict"] == "match" for row in platform_rows),
              f"reference platform (machine, GPU, compute capability): {summary}",
              "; expected " + ", ".join(row["reference"] for row in platform_rows))
    else:
        lines.append(f"NOTE  local profile on {summary}. The DeepSig reference platform is "
                     + ", ".join(row["reference"] for row in platform_rows))
    unmet = [need["needed"] for need in report["requirements"] if not need["met"]]
    check(not unmet, "toolchain requirements: " + "; ".join(need["needed"] for need in report["requirements"]),
          "; unmet: " + "; ".join(unmet))

    wanted = {"CMAKE_BUILD_TYPE": "Release", "ENABLE_CUDA": "ON", "CUDA_ACCEL_FOUND": "TRUE",
              "BUILD_TESTING": "ON", "CMAKE_CUDA_ARCHITECTURES": arch, **cpu}
    differing = [f"{key}={cache.get(key)!r}" for key, value in wanted.items() if cache.get(key) != value]
    if receipt["machine"] == "spark":
        recipe = {"CMAKE_CUDA_ARCHITECTURES": SPARK["compute_capability"].replace(".", ""), "MCPU": SPARK["mcpu"]}
        differing += [f"a receipt with {key}={wanted.get(key)!r}" for key, value in recipe.items()
                      if wanted.get(key) != value]
    same_cache = digest(evidence / "CMakeCache.txt") == receipt["cache_sha256"]
    check(not differing and same_cache,
          "build configuration: " + ", ".join(f"{key}={value}" for key, value in wanted.items()),
          "; found " + ", ".join(differing or ["a cache that differs from the receipt"]))

    linked = [name for name in (TEST.name, GNB.name) if targets.get(name, {}).get("cuda_device_code")]
    compiled = (toolchain["cuda"]["id"] == "NVIDIA" and plan["kernel_objects"]
                and plan["cuda_code"] == f"sm_{arch}" and linked == [TEST.name, GNB.name]
                and KERNEL_LIBRARY.name in targets)
    check(compiled, f"CUDA compilation: {len(plan['kernel_objects'])} kernel sources of {plan['cuda_units']} "
          f"CUDA sources for {plan['cuda_code']} by nvcc {toolchain['cuda']['version']}, host compiler "
          f"{toolchain['cxx']['id']} {toolchain['cxx']['version']}, device code linked into "
          + " and ".join(linked or ["no executable"]))
    hashes = ", ".join(f"{name} {item['sha256'][:12]}" for name, item in sorted(targets.items()))
    check(targets.get(TEST.name, {}).get("sha256") == receipt["binary_sha256"]
          and targets.get(GNB.name, {}).get("sha256") == receipt.get("gnb_sha256"),
          f"compiled targets and hashes: {hashes}", ": the target list disagrees with the receipt")

    lines += reference_notes(report, receipt["machine"] == "spark")
    for name in sorted(overrides):
        if name in BUILD_ENVIRONMENT:
            lines.append(f"NOTE  build environment override: {name}={overrides[name]}")
    return failed, lines


def reference_notes(report, spark):
    """NOTE lines for the reference rows that are recorded and not required: what matches, what differs."""
    labels = {"benchmark_cpus": "high-capacity CPU set", "cpu_governor": "CPU frequency governor",
              "cpu_cores": "CPU cores", "host_compiler": "host compiler", "operating_system": "operating system",
              "cuda_toolkit": "CUDA toolkit", "gpu_driver": "GPU driver", "source_revision": "OCUDU revision"}
    lines = []
    others = [row for row in report["reference"] if row["item"] not in PLATFORM_ROWS]
    same = [f"{labels[row['item']]} ({row['observed']})" for row in others if row["verdict"] == "match"]
    if same:
        lines.append("NOTE  matches the reference: " + ", ".join(same))
    for row in others:
        if row["verdict"] == "match":
            continue
        scope = "S1.2 measurement condition" if row["item"] in MEASUREMENT_ROWS else "recorded difference"
        found = "not observable" if row["verdict"] == "unknown" else row["observed"]
        lines.append(f"NOTE  {scope}, {labels[row['item']]}: {found} here; "
                     f"{row['reference']} in the {row['source']}")
    if spark and not report.get("benchmark_cpus_allowed"):
        allowed = report.get("facts", {}).get("allowed_cpus")
        lines.append(f"NOTE  S1.2 measurement condition, allowed CPUs: {allowed} here; "
                     f"the measured run needs {SPARK['cpus']}")
    return lines


def condition_notes(conditions):
    """NOTE lines for what a timing run shared the machine with. The entry condition is an idle GPU."""
    scope = "NOTE  S1.2 measurement condition"
    clients = conditions.get("gpu_clients") or []
    named = [f"{Path(item.get('name', '?')).name} ({item.get('type', '?')}, {item.get('used_gpu_memory', '?')})"
             for item in clients]
    usage = conditions.get("gpu_utilization_percent")
    lines = [f"{scope}, GPU clients at the start: {', '.join(named) or 'none listed'}"
             + (f"; GPU utilization {usage} percent" if usage is not None else "")]
    compute = [f"{Path(item.get('name', '?')).name} (pid {item['pid']})" for item in clients
               if "C" in item.get("type", "")]
    if compute:
        lines.append(f"{scope}, compute workload on the GPU at the start: {', '.join(compute)}; "
                     "the entry condition is an idle GPU")
    modem = [f"{item['name']} (pid {item['pid']})" for item in conditions.get("modem_processes") or []]
    if modem:
        lines.append(f"{scope}, modem process running at the start: {', '.join(modem)}; "
                     "Table II was measured with the live modem off")
    if conditions.get("drm_kms_polling") == "Y":
        lines.append(f"{scope}, DRM KMS polling: on here; the upstream tuning script scripts/ocudu_performance "
                     "turns it off before a latency capture")
    frequency = conditions.get("cpu_frequency") or {}
    if frequency:
        current = [item["current_khz"] / 1e6 for item in frequency.values()]
        limits = [item["maximum_khz"] / 1e6 for item in frequency.values() if item.get("maximum_khz")]
        lines.append(f"{scope}, CPU clocks: CPUs {cpu_list(int(cpu) for cpu in frequency)} at "
                     f"{min(current):.2f} to {max(current):.2f} GHz"
                     + (f" of {max(limits):.2f} GHz maximum" if limits else "")
                     + f", clocksource {conditions.get('clocksource') or 'not observable'}")
    return lines


def grade_run(evidence, pin=None):
    """Grade the evidence directory of one measured run. Return (failed, lines) like grade()."""
    lines = []
    failed = False
    label = "S1.2"

    def check(passed, text, detail=""):
        nonlocal failed
        failed = failed or not passed
        lines.append(f"{'PASS' if passed else 'FAIL'}  {label} {text}{'' if passed else detail}")

    try:
        record = json.loads((evidence / "platform.json").read_text())
        profile, gpu = record["profile"], record["gpu"]
        label = verify_pusch_log.PROFILES[profile]["label"]
        status = dict(line.split("=", 1) for line in (evidence / "status.txt").read_text().split())
        receipt = json.loads((evidence / "native-build.json").read_text())
        report = json.loads((evidence / "reference-environment.json").read_text())
        conditions = json.loads((evidence / "measurement-conditions.json").read_text())
        command = [str(word) for word in conditions["command"]]
        transcript = read_log(evidence / "pusch.log")
        binary = receipt["targets"][TEST.name]
        platform_rows = {row["item"]: row for row in report["reference"] if row["item"] in PLATFORM_ROWS}
        device = ", ".join(str(platform_rows[item]["observed"]) for item in PLATFORM_ROWS)
        uuid, driver = gpu["uuid"], gpu["driver"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        return True, [f"FAIL  {label} run evidence is incomplete in {evidence}: {error!r}"]

    check(status.get("stage") == "completed" and status.get("exit_code") == "0",
          "run completed: the native test and the transcript validator exited zero",
          f": stage={status.get('stage')}, exit_code={status.get('exit_code')}")
    pin = pin or locked_source("ocudu")["commit"]
    check(receipt.get("target") == "all" and receipt.get("machine") == "spark" and receipt.get("commit") == pin
          and binary.get("sha256") == receipt.get("binary_sha256") and binary.get("cuda_device_code") is True,
          f"measured the full Spark Release build: {TEST.name} {str(binary.get('sha256'))[:12]} "
          f"of OCUDU {receipt.get('commit')} ({receipt.get('source_kind')})",
          f"; found target={receipt.get('target')}, machine={receipt.get('machine')}; ext/lock.json pins {pin}")
    # The process could see only the selected GPU, and the paper row ran on the paper's CPU set.
    visible = f"CUDA_VISIBLE_DEVICES={uuid}" in command
    reference = all(row["verdict"] == "match" for row in platform_rows.values())
    pinned = any(command[index:index + 3] == ["taskset", "-c", SPARK["cpus"]] for index in range(len(command)))
    where = f"ran on the intended device: {device}, {uuid}, driver {driver}"
    if profile == "paper":
        where += f", pinned to CPUs {SPARK['cpus']}"
    check(visible and reference and (pinned or profile != "paper"), where,
          "; the recorded command is " + shlex.join(command))

    graded_failed, graded = verify_pusch_log.grade_transcript(transcript, profile)[:2]
    if failed and not graded_failed and profile == "paper":
        # A run that was not the accepted build on the paper's device and CPU set compares nothing.
        graded = [line for line in graded if not line.startswith("NOTE")]
        graded.append(f"NOTE  {label} {verify_pusch_log.WITHHELD}")
    failed = failed or graded_failed
    lines += graded
    if profile == "paper":
        lines += condition_notes(conditions) + reference_notes(report, True)
    return failed, lines


def new_run_directory(args, prefix):
    """Create the evidence directory. A named one must be new: evidence is never overwritten."""
    if args.run_dir is not None:
        run = (ROOT / args.run_dir).resolve()
        if run.exists():
            raise SystemExit(f"{run} already exists. Evidence is never overwritten.")
        run.mkdir(parents=True)
        return run
    out = (ROOT / args.output_root).resolve()
    out.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix + ".", dir=out))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True, dest="action")
    build_cmd = sub.add_parser("build", help="Build locked OCUDU from source")
    run_cmd = sub.add_parser("run", help="Run one Spark baseline")
    grade_cmd = sub.add_parser("grade", help="Grade the evidence directory of a finished build")
    grade_cmd.add_argument("evidence", type=Path, help="directory printed by the build as 'Run artifacts'")
    grade_run_cmd = sub.add_parser("grade-run", help="Grade the evidence directory of a measured run")
    grade_run_cmd.add_argument("evidence", type=Path, help="directory printed by the run as 'Run artifacts'")
    for cmd in [build_cmd, run_cmd]:
        cmd.add_argument("--build-dir", type=Path, help="out-of-source build directory")
        cmd.add_argument("--jobs", type=int, default=8)
        cmd.add_argument("--output-root", type=Path, default=Path("results/runs"))
        cmd.add_argument("--run-dir", type=Path, help="new evidence directory (default: a new one under --output-root)")
    build_cmd.add_argument("--machine", choices=("spark", "local"), default="spark")
    build_cmd.add_argument("--mcpu", help="AArch64 CPU target on a local profile")
    build_cmd.add_argument("--target", choices=("all", "baseline"), default="all")
    build_cmd.add_argument("--patched", action="store_true",
                           help="build the pinned source with the series of patches/ocudu/ applied, from a tree of its own")
    build_cmd.add_argument("--patched-source", type=Path,
                           help="where the patched tree lies (default: build/ocudu-src-o3)")
    run_cmd.add_argument("--profile", choices=("smoke", "paper"), default="smoke")
    run_cmd.add_argument("--skip-build", action="store_true", help="verify the saved build identity and reuse it")
    run_cmd.set_defaults(machine="spark", mcpu=None)
    args = parser.parse_args(argv)
    if args.action in ("grade", "grade-run"):
        failed, lines = (grade if args.action == "grade" else grade_run)(args.evidence.resolve())
        print("\n".join(lines))
        return int(failed)
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.mcpu and args.machine != "local":
        parser.error("--mcpu can only be used with --machine local")
    patched = bool(getattr(args, "patched", False))
    if getattr(args, "patched_source", None) is not None and not patched:
        parser.error("--patched-source can only be used with --patched")
    build = (ROOT / (args.build_dir or Path(f"build/ocudu-{args.machine}{'-o3' if patched else ''}"))).resolve()
    patched_source = (ROOT / (args.patched_source or Path("build/ocudu-src-o3"))).resolve() if patched else None
    for protected in [ROOT / "ext", ROOT / "isac"]:
        for place in filter(None, [build, patched_source]):
            if place == ROOT or place == protected or protected in place.parents:
                parser.error("Choose dedicated directories outside ext/ and isac/ for the build and for the patched tree.")
    if patched and (patched_source == build or build in patched_source.parents or patched_source in build.parents):
        parser.error("The patched tree and its build directory must be two separate directories.")
    run = new_run_directory(args, "ocudu-build" if args.action == "build" else "spark-pusch")
    print(f"Run artifacts: {run}", flush=True)
    stage = {"stage": "preflight"}
    status = 1
    report = None
    try:
        if Path(sys.prefix).resolve() != ROOT / "isac" or sys.prefix == sys.base_prefix:
            raise ValueError("Activate the project environment with 'source isac/bin/activate'.")
        required = ["nvidia-smi", "nvcc", "cmake", "c++"]
        if args.action == "run":
            required += ["timeout"]
            if args.profile == "paper":
                required.append("taskset")
        for tool in required:
            if not shutil.which(tool):
                hint = " Put the CUDA toolkit's bin directory on PATH." if tool == "nvcc" else ""
                raise ValueError(f"Missing host tool: {tool}.{hint}")
        gpu = selected_gpu()
        record_command(["nvidia-smi", "-q", "-i", gpu["uuid"]], run, "nvidia-smi.txt")
        for cmd in ["nvcc", "cmake", "c++"]:
            record_command([cmd, "--version"], run, cmd + ".txt")
        if args.action == "build" or not args.skip_build:
            # Idempotent: installs only the development libraries that dpkg reports missing.
            stage["stage"] = "prerequisites"
            record_command(["bash", ROOT / "scripts/install_native_deps.sh"], run, "native-deps.log", live=True)
            stage["stage"] = "preflight"
        mcpu = cpu_target(args.machine, args.mcpu)
        report = compare_reference(args.machine, host_facts(gpu, mcpu), mcpu, locked_source("ocudu"))
        write_json(run / "reference-environment.json", report)
        parent_revision = None
        if (ROOT / ".git").exists():
            parent_revision = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
        write_json(run / "platform.json", dict(gpu=gpu, platform=platform.platform(), python=sys.executable,
                   parent_revision=parent_revision, cpu_governors=report["facts"]["governors"],
                   cpu_affinity=sorted(os.sched_getaffinity(0)), started_utc=datetime.now(timezone.utc).isoformat(),
                   profile=getattr(args, "profile", args.machine),
                   overrides={k: v for k, v in os.environ.items()
                              if k in BUILD_ENVIRONMENT or k.startswith(("OCUDU_", "CUDA_", "CMAKE_"))}))
        require_profile(report)
        absent = [item["package"] for item in report["facts"].get("host_packages") or [] if item["found"] is False]
        if absent:
            print("The compiler cannot find these packages, which the configure requires: " + ", ".join(absent)
                  + ". CMake decides. If it stops, run: sudo apt-get install " + " ".join(absent), flush=True)
        stage["stage"] = "source"
        record_command([sys.executable, ROOT / "scripts/check_pins.py", "--worktrees", "--only", "ocudu"], run, "pins.txt")
        pin = locked_source("ocudu")["commit"]
        build.mkdir(parents=True, exist_ok=True)
        with (build / ".isac.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if args.action == "build" and patched:
                # The locked checkout was just verified as pristine. The patched tree is made beside it.
                stage["stage"] = "patch"
                series = receiver_patch.load_series()
                patch = receiver_patch.prepare_tree(SOURCE, patched_source, series)
                write_json(run / "receiver-patch.json", patch)
                print(f"Patched receiver source: {patched_source} ({patch['tree_kind']}, series "
                      f"{patch['series_sha256'][:12]})", flush=True)
                native_build(args, build, run, gpu, pin, args.target, stage, source=patched_source, patch=patch)
            elif args.action == "build":
                native_build(args, build, run, gpu, pin, args.target, stage)
            else:
                if not args.skip_build and not (build / "isac-native.json").exists():
                    native_build(args, build, run, gpu, pin, "baseline", stage)
                stage["stage"] = "build-identity"
                checked_build(build, run, gpu, pin)
                benchmark(args.profile, build, run, gpu, stage)
        stage["stage"] = "completed"
        status = 0
        return 0
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        (run / "error.txt").write_text(explain(error, report) + "\n")
        # A live command has just printed its own error. Do not repeat it on the screen.
        message = explain(error, report, quote=not getattr(error, "live", False))
        print(f"{stage['stage']}: {message}\nSee {run}", file=sys.stderr)
        return 1
    finally:
        (run / "status.txt").write_text(f"stage={stage['stage']}\nexit_code={status}\n")


if __name__ == "__main__":
    sys.exit(main())
