"""Build the locked OCUDU Release tree and collect one isolated Spark PUSCH run."""

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
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def locked_source(name):
    """Return the lock entry for a logical dependency name. ext/lock.json owns its path."""
    for item in json.loads((ROOT / "ext/lock.json").read_text())["dependencies"]:
        if item.get("name") == name:
            return item
    raise ValueError(f"ext/lock.json has no dependency named {name!r}")


SOURCE = ROOT / locked_source("ocudu")["path"]
TEST = Path("tests/integrationtests/phy/upper/channel_processors/pusch_e2e_pipeline_test")
AFFINITY = {5, 6, 7, 8, 9, 15, 16, 17, 18, 19}


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def record_command(args, run_dir, log_name, *, env=None):
    command = shlex.join([str(arg) for arg in args])
    with (run_dir / "commands.sh").open("a") as history:
        history.write(command + "\n")
    print(f"{command}\n  log: {run_dir / log_name}", flush=True)
    with (run_dir / log_name).open("w") as log:
        subprocess.run(args, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, cwd=ROOT)


def selected_gpu(machine):
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
    if machine == "spark" and (platform.machine() != "aarch64" or "GB10" not in name or cc != "12.1"):
        raise ValueError(f"The Spark profile needs aarch64 + GB10/12.1; found {platform.machine()}, {name}, {cc}")
    return dict(uuid=uuid, name=name, compute_capability=cc, driver=driver)


def cache_values(path):
    entries = {}
    for line in path.read_text().splitlines():
        match = re.fullmatch(r"([A-Za-z_][A-Za-z_0-9]*):[^=]+=(.*)", line)
        if match:
            entries[match[1]] = match[2]
    return entries


def native_build(args, build, run, gpu, pin, selected_target, stage):
    arch = gpu["compute_capability"].replace(".", "")
    stage["stage"] = "configure"
    # Invalidate old proof even if this rebuild fails partway.
    (build / "isac-native.json").unlink(missing_ok=True)
    native_targets = subprocess.check_output(["nvcc", "--list-gpu-code"], text=True).split()
    if f"sm_{arch}" not in native_targets:
        raise ValueError(f"nvcc cannot emit sm_{arch}. Available targets: {' '.join(native_targets)}")

    options = ["-DCMAKE_BUILD_TYPE=Release", "-DENABLE_CUDA=ON",
               f"-DCMAKE_CUDA_ARCHITECTURES={arch}", "-DBUILD_TESTING=ON",
               "-DDISABLE_MBEDTLS=OFF", "-DDISABLE_SCTP=OFF", "-DENABLE_UHD=OFF", "-DENABLE_ZEROMQ=OFF",
               "-DENABLE_PLUGINS=OFF"]
    if args.machine == "spark":
        options.append("-DMCPU=neoverse-v2")
    elif args.mcpu is not None:
        if platform.machine() != "aarch64":
            raise ValueError("--mcpu is an AArch64 compiler setting.")
        options.append(f"-DMCPU={args.mcpu}")
    # An archive has no upstream .git. Keep its version script from reporting
    # the enclosing research repository's HEAD as OCUDU's revision.
    cmake_env = os.environ.copy()
    if not (SOURCE / ".git").exists():
        ceilings = cmake_env.get("GIT_CEILING_DIRECTORIES", "").split(os.pathsep)
        cmake_env["GIT_CEILING_DIRECTORIES"] = os.pathsep.join(filter(None, [str(SOURCE.parent), *ceilings]))
    record_command(["cmake", "-S", SOURCE, "-B", build, *options], run, "configure.log", env=cmake_env)
    cache = build / "CMakeCache.txt"
    shutil.copyfile(cache, run / cache.name)
    values = cache_values(cache)
    expected = {"CMAKE_HOME_DIRECTORY": str(SOURCE), "CMAKE_BUILD_TYPE": "Release",
                "BUILD_TESTING": "ON", "ENABLE_CUDA": "ON", "CMAKE_CUDA_ARCHITECTURES": arch,
                "CUDA_ACCEL_FOUND": "TRUE"}
    if args.machine == "spark":
        expected["MCPU"] = "neoverse-v2"
    for key, expected_value in expected.items():
        if values.get(key) != expected_value:
            raise ValueError(f"Native configuration requires {key}={expected_value}; found {values.get(key)!r}")

    stage["stage"] = "build"
    command = ["cmake", "--build", build, f"-j{args.jobs}"]
    if selected_target == "baseline":
        command += ["--target", TEST.name]
    record_command(command, run, "build.log", env=cmake_env)
    binary = build / TEST
    record_command([binary, "--help"], run, "native-help.txt")
    if "--require-gpu-resident" not in (run / "native-help.txt").read_text():
        raise ValueError("Built program lacks the expected resident PUSCH test interface.")
    stamp = dict(commit=pin, binary_sha256=digest(binary), cache_sha256=digest(cache),
                 cuda_arch=arch, machine=args.machine, target=selected_target,
                 build_evidence=str(run), source=str(SOURCE),
                 source_kind="git" if (SOURCE / ".git").exists() else "verified-archive")
    binaries = [binary]
    if selected_target == "all":
        gnb = build / "apps/gnb/gnb"
        if not os.access(gnb, os.X_OK):
            raise ValueError(f"Full build did not produce the gNB: {gnb}")
        record_command([gnb, "--version"], run, "gnb-version.txt")
        if "OCUDU 5G gNB version" not in (run / "gnb-version.txt").read_text():
            raise ValueError("Full build did not report the expected gNB version banner.")
        stamp["gnb_sha256"] = digest(gnb)
        binaries.append(gnb)

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


def benchmark(profile, build, run, gpu, stage):
    paper = profile == "paper"
    iterations = 200 if paper else 20
    if paper and not AFFINITY.issubset(os.sched_getaffinity(0)):
        raise ValueError("The paper profile requires cores 5-9,15-19 in this process's allowed CPU set.")
    command = ["timeout", "--kill-after=10s", "5m"]
    if paper:
        command += ["taskset", "-c", "5-9,15-19"]
    overrides = {"CUDA_VISIBLE_DEVICES": gpu["uuid"], "OCUDU_PUSCH_ACCELERATION_TIMING": "1"}
    command += ["env", *(f"{key}={value}" for key, value in overrides.items()), build / TEST,
                "--prb", "51", "--sinr", "30", "--mcs", "27" if paper else "10",
                "--mcs-table", "qam256" if paper else "qam64", "--layers", "1",
                "--ports", "1" if paper else "4", "--dmrs-type", "type1", "--dmrs-symbols", "2,11",
                "--iterations", str(iterations), "--warmup", "5", "--cpu-every", "1",
                "--rx-device-grid", "direct-managed", "--require-gpu-resident",
                "--gpu-output-buffer", "pinned" if paper else "vector", "--quiet"]
    stage["stage"] = "run"
    record_command(command, run, "pusch.log")
    stage["stage"] = "verify"
    verify = [sys.executable, ROOT / "scripts/verify_pusch_log.py", run / "pusch.log", "--iterations", str(iterations)]
    if paper:
        verify += ["--paper-json", run / "table-ii.json"]
    record_command(verify, run, "verification.txt")
    print((run / "verification.txt").read_text().strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True, dest="action")
    build_cmd = sub.add_parser("build", help="Build locked OCUDU from source")
    run_cmd = sub.add_parser("run", help="Run one Spark baseline")
    for cmd in [build_cmd, run_cmd]:
        cmd.add_argument("--build-dir", type=Path, help="out-of-source build directory")
        cmd.add_argument("--jobs", type=int, default=8)
        cmd.add_argument("--output-root", type=Path, default=Path("results/runs"))
    build_cmd.add_argument("--machine", choices=("spark", "local"), default="spark")
    build_cmd.add_argument("--mcpu", help="AArch64 CPU target on a local profile")
    build_cmd.add_argument("--target", choices=("all", "baseline"), default="all")
    run_cmd.add_argument("--profile", choices=("smoke", "paper"), default="smoke")
    run_cmd.add_argument("--skip-build", action="store_true", help="verify the saved build identity and reuse it")
    run_cmd.set_defaults(machine="spark", mcpu=None)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.mcpu and args.machine != "local":
        parser.error("--mcpu can only be used with --machine local")
    build = (ROOT / (args.build_dir or Path(f"build/ocudu-{args.machine}"))).resolve()
    for protected in [ROOT / "ext", ROOT / "isac"]:
        if build == ROOT or build == protected or protected in build.parents:
            parser.error("Choose a dedicated out-of-source build directory outside ext/ and isac/.")
    out = (ROOT / args.output_root).resolve()
    out.mkdir(parents=True, exist_ok=True)
    prefix = "ocudu-build" if args.action == "build" else "spark-pusch"
    run = Path(tempfile.mkdtemp(prefix=prefix + ".", dir=out))
    print(f"Run artifacts: {run}", flush=True)
    stage = {"stage": "preflight"}
    status = 1
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
                raise ValueError(f"Missing host tool: {tool}")
        gpu = selected_gpu(args.machine)
        record_command(["nvidia-smi", "-q", "-i", gpu["uuid"]], run, "nvidia-smi.txt")
        for cmd in ["nvcc", "cmake", "c++"]:
            record_command([cmd, "--version"], run, cmd + ".txt")
        cpu_policies = {}
        for policy in Path("/sys/devices/system/cpu/cpufreq").glob("policy*"):
            governor = policy / "scaling_governor"
            if governor.is_file():
                cpu_policies[policy.name] = governor.read_text().strip()
        parent_revision = None
        if (ROOT / ".git").exists():
            parent_revision = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
        write_json(run / "platform.json", dict(gpu=gpu, platform=platform.platform(), python=sys.executable,
                   parent_revision=parent_revision, cpu_governors=cpu_policies,
                   cpu_affinity=sorted(os.sched_getaffinity(0)), started_utc=datetime.now(timezone.utc).isoformat(),
                   profile=getattr(args, "profile", args.machine),
                   overrides={k: v for k, v in os.environ.items() if k.startswith(("OCUDU_", "CUDA_"))}))
        stage["stage"] = "source"
        record_command([sys.executable, ROOT / "scripts/check_pins.py", "--worktrees", "--only", "ocudu"], run, "pins.txt")
        pin = locked_source("ocudu")["commit"]
        build.mkdir(parents=True, exist_ok=True)
        with (build / ".isac.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if args.action == "build":
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
        (run / "error.txt").write_text(str(error) + "\n")
        print(f"{stage['stage']}: {error}. See {run}", file=sys.stderr)
        return 1
    finally:
        (run / "status.txt").write_text(f"stage={stage['stage']}\nexit_code={status}\n")


if __name__ == "__main__":
    sys.exit(main())
