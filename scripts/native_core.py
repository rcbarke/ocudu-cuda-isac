#!/usr/bin/env python3
"""Probe, build, test and verify the native sensing core (Sprint 3.6, selector sense_core).

The core in native/sense_core is a CMake project of its own: C++ and CUDA, the observation ring
of native/obs_ring from its sources, and nothing of OCUDU. This tool is the one way a build of it
is made and named.

  probe    checkpoint D0, before anything is built. It finds the CUDA toolkit and TensorRT with
           its headers on this host, compiles the one file tools/sense_probe.cpp against them and
           runs it: a one-layer engine is built on the GPU, executed on a stream and read back.
           It writes sense-probe.json and says which backend this host has: tensorrt, or the
           declared fallback libtorch where TensorRT fails and the PyTorch distribution ships its
           C++ headers, or none.
  headers  fetch the open-source headers of the TensorRT release whose libraries are installed,
           for a host that has the libraries from a Python wheel and no headers.
  build    configure, build and run the unit cases of the core, and write the receipt
           isac-core.json: the sources by hash, the toolchain, the CUDA architecture, the
           TensorRT version, the library, the executables and the binding by hash, the link
           lines, and the cases that ran.
  verify   check a build against its receipt and against the tree, and print the receipt.
  engine   export the frozen artifact to ONNX and build its TensorRT engine on this GPU, or
           write the TorchScript module of the declared fallback.

    python scripts/native_core.py probe
    python scripts/native_core.py build                  # build/sense-core: the CUDA device
    python scripts/native_core.py build --host-only      # build/sense-core-host: the twin only
    python scripts/native_core.py verify
    python scripts/native_core.py engine --artifact DIR --output-dir DIR

The standard library only: it runs before isac holds anything. ASCII only.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "native/sense_core"
RING = ROOT / "native/obs_ring"
DEFAULT_BUILD = ROOT / "build/sense-core"
DEFAULT_BUILD_HOST = ROOT / "build/sense-core-host"
HEADERS_DIR = ROOT / "build/tensorrt-headers"
RECEIPT = "isac-core.json"
RECEIPT_SCHEMA = "isac-core-build/v1"
PROBE_FILE = "sense-probe.json"
PROBE_SCHEMA = "isac-core-probe/v1"
CONFIG_FILE = "sense-core-config.json"
EXECUTABLES = ("sense_host", "sense_feed", "sense_engine", "sense_probe")
REQUIRED = ("sense_host", "sense_feed")
LIBRARY = "libsense_core.a"
BINDING_GLOB = "python/_sense_core*.so"
TENSORRT_HEADERS_URL = "https://github.com/NVIDIA/TensorRT.git"
# What may never appear on a link line of the core: a receiver, an Aerial library, an interpreter.
FORBIDDEN_LINK = ("ocudu", "srsran", "aerial", "cuphy", "cumac", "python", "nvipc")
GENERATOR = "Unix Makefiles"


class CoreError(ValueError):
    """The host, the tree or a build of the sensing core is not what this tool can name."""


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    """Every file of the core tree and the record definition of the ring, with its hash."""
    made = {str(path.relative_to(CORE)): digest(path)
            for path in sorted(CORE.rglob("*")) if path.is_file() and "__pycache__" not in path.parts}
    made["obs_ring/include/obs/record.h"] = digest(RING / "include/obs/record.h")
    return made


def sources_sha256(files=None):
    """The hash a binary of the core names for its sources: CMakeLists.txt computes the same value.

    The text "path:sha256" of each file of native/sense_core in the order of its path, one per
    line, then the record definition of the ring.
    """
    files = sources() if files is None else files
    record = "obs_ring/include/obs/record.h"
    lines = [f"{name}:{files[name]}\n" for name in sorted(name for name in files if name != record)]
    lines.append(f"{record}:{files[record]}\n")
    return hashlib.sha256("".join(lines).encode("ascii")).hexdigest()


def run(args, timeout=600):
    """One command, quietly: (status, output). A tool that is missing gives status 127."""
    try:
        done = subprocess.run([str(arg) for arg in args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    except FileNotFoundError:
        return 127, f"{args[0]}: not found"
    except subprocess.TimeoutExpired:
        return 124, f"{args[0]}: no answer within {timeout} s"
    return done.returncode, done.stdout.decode("utf-8", "replace")


def record(args, run_dir, log_name, cwd=None, env=None):
    """Run one tool with its output on the screen and in a log. Return its exit status."""
    command = shlex.join(str(arg) for arg in args)
    with (run_dir / "commands.sh").open("a") as history:
        history.write(command + "\n")
    print(f"{command}\n  log: {run_dir / log_name}", flush=True)
    environment = {key: value for key, value in (env or os.environ).items() if key != "CLICOLOR_FORCE"}
    with (run_dir / log_name).open("wb") as log:
        process = subprocess.Popen([str(arg) for arg in args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd,
                                   env=environment)
        with process.stdout:
            for line in process.stdout:
                log.write(line)
                log.flush()
                sys.stdout.write(line.decode("utf-8", "replace"))
                sys.stdout.flush()
        return process.wait()


def first_line(args):
    status, text = run(args, 60)
    return text.strip().splitlines()[0] if status == 0 and text.strip() else None


# ---------------------------------------------------------------- what the host has
def triplet():
    return sysconfig.get_config_var("MULTIARCH") or f"{platform.machine()}-linux-gnu"


def find_cuda():
    """The CUDA toolkit: its compiler, its headers and its runtime library, or the reason there is none."""
    nvcc = shutil.which("nvcc")
    roots = [Path(os.environ[name]) for name in ("CUDA_HOME", "CUDA_PATH", "CUDAToolkit_ROOT") if os.environ.get(name)]
    if nvcc:
        roots.append(Path(nvcc).resolve().parents[1])
    roots += sorted(Path("/usr/local").glob("cuda*"), reverse=True)
    for root in roots:
        compiler = root / "bin/nvcc"
        include = root / "include"
        if not compiler.is_file() or not (include / "cuda_runtime.h").is_file():
            continue
        for lib in (root / "lib64", root / "lib", root / f"targets/{platform.machine()}-linux/lib"):
            found = sorted(lib.glob("libcudart.so*")) if lib.is_dir() else []
            if found:
                said = run([compiler, "--version"], 60)[1].strip().splitlines()
                version = next((line for line in said if "release" in line), said[-1] if said else None)
                return {"root": str(root), "nvcc": str(compiler), "include": str(include), "library_dir": str(lib),
                        "cudart": str(found[0]), "nvcc_version": version}
    return {"problem": "no CUDA toolkit with nvcc, cuda_runtime.h and libcudart was found: put nvcc on PATH or set "
                       "CUDA_HOME"}


def gpu_facts():
    """What nvidia-smi says of the first GPU, or the reason it says nothing."""
    status, text = run(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,uuid", "--format=csv,noheader"], 60)
    if status != 0 or not text.strip():
        return {"problem": "nvidia-smi names no GPU: " + (text.strip().splitlines() or ["no output"])[-1]}
    name, capability, driver, uuid = (part.strip() for part in (text.strip().splitlines()[0].split(",") + [""] * 4)[:4])
    architecture = capability.replace(".", "") if re.fullmatch(r"\d+\.\d+", capability) else None
    return {"name": name, "capability": capability, "driver": driver, "uuid": uuid, "architecture": architecture}


def header_version(include):
    """The TensorRT version a header directory names, or None."""
    try:
        text = (Path(include) / "NvInferVersion.h").read_text(errors="replace")
    except OSError:
        return None
    parts = []
    for part in ("MAJOR", "MINOR", "PATCH", "BUILD"):
        found = re.search(rf"#define (?:TRT_{part}_ENTERPRISE|NV_TENSORRT_{part})\s+(\d+)", text)
        if not found:
            return None
        parts.append(found.group(1))
    return ".".join(parts)


def system_places():
    """Where a package installation keeps TensorRT on this host: (header directories, library directories)."""
    name = triplet()
    return ([Path("/usr/include") / name, Path("/usr/include"), Path("/usr/local/include")],
            [Path("/usr/lib") / name, Path("/usr/lib64"), Path("/usr/local/lib")])


def find_tensorrt(args):
    """Where TensorRT is on this host: its headers, its libraries and what installed them.

    The order is: what the command line or TENSORRT_ROOT names, the system directories of a
    package installation, and the libraries of a Python wheel in this interpreter with headers
    that `native_core.py headers` fetched. Returns a record with "problem" when a part is missing.
    """
    candidates = []
    root = getattr(args, "tensorrt_root", None) or os.environ.get("TENSORRT_ROOT")
    include = getattr(args, "tensorrt_include", None)
    library = getattr(args, "tensorrt_library", None)
    if include or library:
        candidates.append(("named on the command line", [Path(include)] if include else [],
                           [Path(library)] if library else []))
    if root:
        candidates.append((f"TENSORRT_ROOT {root}", [Path(root) / "include"], [Path(root) / "lib", Path(root) / "lib64"]))
    candidates.append(("system packages",) + system_places())
    wheel = None
    for module in ("tensorrt_libs", "tensorrt_cu13_libs", "tensorrt_cu12_libs"):
        try:
            spec = importlib.util.find_spec(module)
        except (ImportError, ValueError):
            spec = None
        if spec is not None and spec.submodule_search_locations:
            wheel = Path(list(spec.submodule_search_locations)[0])
            break
    fetched = sorted(HEADERS_DIR.glob("*/include")) if HEADERS_DIR.is_dir() else []
    if wheel is not None:
        candidates.append((f"the Python wheel in {wheel}", fetched, [wheel]))
    seen = []
    for origin, includes, libraries in candidates:
        headers = next((path for path in includes if (path / "NvInfer.h").is_file()), None)
        parser = next((path for path in includes if (path / "NvOnnxParser.h").is_file()), None)
        infer = onnx = None
        for path in libraries:
            if not path.is_dir():
                continue
            infer = infer or next(iter(sorted(path.glob("libnvinfer.so")) + sorted(path.glob("libnvinfer.so.[0-9]*"))), None)
            onnx = onnx or next(iter(sorted(path.glob("libnvonnxparser.so"))
                                     + sorted(path.glob("libnvonnxparser.so.[0-9]*"))), None)
        seen.append({"origin": origin, "headers": str(headers) if headers else None,
                     "parser_header": str(parser) if parser else None, "libnvinfer": str(infer) if infer else None,
                     "libnvonnxparser": str(onnx) if onnx else None})
        if headers and infer:
            return {"origin": origin, "include": str(headers), "parser_include": str(parser) if parser else None,
                    "library_dir": str(infer.parent), "libnvinfer": str(infer),
                    "libnvonnxparser": str(onnx) if onnx else None, "header_version": header_version(headers),
                    "searched": seen}
    partial = next((item for item in seen if item["libnvinfer"] or item["headers"]), None)
    if partial and partial["libnvinfer"] and not partial["headers"]:
        problem = (f"the TensorRT libraries of {partial['origin']} come without headers. Fetch the headers of that "
                   "release: python scripts/native_core.py headers")
    elif partial:
        problem = f"{partial['origin']} holds TensorRT headers and no libnvinfer"
    else:
        problem = ("no TensorRT was found: neither a package installation, nor TENSORRT_ROOT, nor the wheel "
                   "tensorrt_cu13 in this interpreter")
    return {"problem": problem, "searched": seen}


def wheel_version():
    """The version of the TensorRT wheel in this interpreter, or None."""
    try:
        from importlib import metadata
    except ImportError:
        return None
    for name in ("tensorrt_cu13_libs", "tensorrt-cu13-libs", "tensorrt_cu13", "tensorrt-cu13", "tensorrt_libs", "tensorrt"):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return None


def find_libtorch():
    """Whether the PyTorch distribution of this interpreter ships what the LibTorch backend needs."""
    try:
        spec = importlib.util.find_spec("torch")
    except (ImportError, ValueError):
        spec = None
    if spec is None or not spec.submodule_search_locations:
        return {"problem": "this interpreter holds no torch"}
    root = Path(list(spec.submodule_search_locations)[0])
    missing = [name for name in ("include/torch/script.h", "lib/libtorch.so", "lib/libtorch_cpu.so", "lib/libc10.so")
               if not (root / name).is_file()]
    if missing:
        return {"root": str(root), "problem": "the PyTorch distribution does not ship " + ", ".join(missing)}
    version = None
    try:
        text = (root / "version.py").read_text()
        found = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", text)
        version = found.group(1) if found else None
    except OSError:
        pass
    return {"root": str(root), "version": version, "cuda": (root / "lib/libtorch_cuda.so").is_file()}


def python_packages():
    """The versions of the packages the export and the binding need, as this interpreter has them."""
    from importlib import metadata
    made = {}
    for name in ("onnx", "pybind11", "numpy", "torch", "onnxruntime"):
        try:
            made[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            made[name] = None
    return made


def pybind11_dir():
    try:
        import pybind11
        return pybind11.get_cmake_dir()
    except Exception:  # absent, or too old to say where its CMake files are
        return None


# ---------------------------------------------------------------- probe
def probe(args):
    build_dir = Path(args.build_dir or DEFAULT_BUILD).resolve() / "probe"
    build_dir.mkdir(parents=True, exist_ok=True)
    cuda, gpu, tensorrt, libtorch = find_cuda(), gpu_facts(), find_tensorrt(args), find_libtorch()
    report = {"schema": PROBE_SCHEMA, "probed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "host": platform.node(), "machine": platform.machine(), "python": platform.python_version(),
              "cuda": cuda, "gpu": gpu, "tensorrt": dict(tensorrt, wheel_version=wheel_version()),
              "libtorch": libtorch, "packages": python_packages(),
              "compiler": first_line([os.environ.get("CXX", "c++"), "--version"]),
              "cmake": first_line(["cmake", "--version"]), "engine": None, "problems": []}
    problems = report["problems"]
    for name, item in (("CUDA toolkit", cuda), ("TensorRT", tensorrt)):
        if "problem" in item:
            problems.append(f"{name}: {item['problem']}")
    if "problem" in gpu:
        problems.append(f"GPU: {gpu['problem']}")
    if not problems:
        binary, source = build_dir / "sense_probe", CORE / "tools/sense_probe.cpp"
        onnx = tensorrt["parser_include"] and tensorrt["libnvonnxparser"]
        command = [os.environ.get("CXX", "c++"), "-std=c++17", "-O2", source, "-isystem", tensorrt["include"],
                   "-isystem", cuda["include"], "-o", binary, tensorrt["libnvinfer"], cuda["cudart"],
                   f"-Wl,-rpath,{tensorrt['library_dir']}", f"-Wl,-rpath,{cuda['library_dir']}"]
        if onnx:
            command += ["-DSENSE_PROBE_ONNX", "-isystem", tensorrt["parser_include"], tensorrt["libnvonnxparser"]]
        else:
            problems.append("TensorRT: no NvOnnxParser.h with libnvonnxparser: the engine of an artifact cannot be built")
        status, text = run(command, 600)
        (build_dir / "compile.log").write_text(shlex.join(str(part) for part in command) + "\n" + text)
        if status != 0:
            problems.append("the probe does not compile against these headers and libraries (probe/compile.log): "
                            + (text.strip().splitlines() or ["no output"])[-1][:300])
        else:
            status, text = run([binary, "--device", str(args.device)], 900)
            (build_dir / "run.log").write_text(text)
            line = next((item for item in reversed(text.splitlines()) if item.startswith("{")), None)
            try:
                report["engine"] = json.loads(line) if line else None
            except ValueError:
                report["engine"] = None
            if report["engine"] is None:
                problems.append("the probe printed no report (probe/run.log): "
                                + (text.strip().splitlines() or ["no output"])[-1][:300])
            else:
                problems += [f"engine: {item}" for item in report["engine"].get("problems", [])]
                if status != 0 and not report["engine"].get("problems"):
                    problems.append(f"the probe ended with status {status}")
    usable = "problem" not in libtorch
    report["backend"] = "tensorrt" if not problems else "libtorch" if usable else None
    report["ok"] = not problems
    report["fallback"] = {"declared": "libtorch", "usable": usable,
                          "reason": libtorch.get("problem", "the PyTorch distribution ships its C++ headers and libraries")}
    out = Path(args.output) if args.output else build_dir.parent / PROBE_FILE
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii")
    if report["ok"]:
        engine = report["engine"]
        print(f"native_core: D0 passed: TensorRT {engine['tensorrt']['library']} ({tensorrt['origin']}) built and ran "
              f"a one-layer engine on {engine['device']['name']}, capability {engine['device']['capability']}, CUDA "
              f"runtime {engine['device']['cuda_runtime']}; a captured graph "
              f"{'launches' if engine['graph']['captured'] else 'is refused: ' + str(engine['graph']['reason'])}; "
              f"one run takes {engine['engine']['run_us']['p50']:.1f} us in the median ({out})")
        return 0
    for problem in problems:
        print(f"native_core: {problem}", file=sys.stderr)
    print("native_core: D0 failed: this host has no TensorRT backend. " +
          ("The kernels still build, and the declared fallback is the LibTorch backend: python scripts/native_core.py "
           "build --tensorrt off --libtorch. A run on it records the native showing as blocked."
           if usable else "The declared fallback is not usable either: " + libtorch["problem"]) + f" ({out})",
          file=sys.stderr)
    return 1


def headers(args):
    """Fetch the open-source headers of the TensorRT release whose libraries this interpreter holds."""
    version = args.version or wheel_version()
    if not version:
        raise CoreError("no TensorRT wheel is installed in this interpreter: name the release with --version, as 10.14")
    major_minor = ".".join(version.split(".")[:2])
    destination = HEADERS_DIR / major_minor
    if (destination / "include/NvInfer.h").is_file():
        print(f"native_core: {destination / 'include'} holds the headers of TensorRT {header_version(destination / 'include')}")
        return 0
    HEADERS_DIR.mkdir(parents=True, exist_ok=True)
    status, text = run(["git", "clone", "--depth", "1", "--branch", f"v{major_minor}", TENSORRT_HEADERS_URL,
                        str(destination)], 900)
    if status != 0 or not (destination / "include/NvInfer.h").is_file():
        raise CoreError(f"cannot fetch the headers of TensorRT {major_minor} from {TENSORRT_HEADERS_URL}: "
                        + (text.strip().splitlines() or ["no output"])[-1])
    found = header_version(destination / "include")
    if found is None or found.split(".")[:3] != version.split(".")[:3]:
        raise CoreError(f"the headers are those of TensorRT {found}, and the libraries are {version}")
    print(f"native_core: the headers of TensorRT {found} are in {destination / 'include'} (Apache-2.0, "
          f"{TENSORRT_HEADERS_URL} at v{major_minor})")
    return 0


# ---------------------------------------------------------------- build
def link_lines(build_dir):
    """The link line of the sensor host and of the binding, and the archive line of the library."""
    made = {}
    for name, path in (("sense_host", "CMakeFiles/sense_host.dir/link.txt"),
                       ("sense_core", "CMakeFiles/sense_core.dir/link.txt"),
                       ("binding", "CMakeFiles/_sense_core.dir/link.txt")):
        try:
            made[name] = " ".join((build_dir / path).read_text().split())
        except OSError:
            made[name] = None
    return made


def forbidden_in(line):
    """Names on a link line that no build of the core may link: of a receiver, of Aerial, of an interpreter."""
    found = []
    for token in (line or "").split():
        name = Path(token).name.lower()
        if token.startswith(("-l", "/")) or name.endswith((".a", ".so")) or ".so." in name:
            found += [token for word in FORBIDDEN_LINK if word in name]
    return sorted(set(found))


def build(args):
    host_only = args.host_only
    build_dir = Path(args.build_dir or (DEFAULT_BUILD_HOST if host_only else DEFAULT_BUILD)).resolve()
    run_dir = Path(args.run_dir).resolve() if args.run_dir else build_dir / "evidence"
    build_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    cmake = shutil.which("cmake")
    if cmake is None:
        raise CoreError("cmake is not on PATH: scripts/install_native_deps.sh installs the native tools")
    configure = [cmake, "-S", CORE, "-B", build_dir, "-G", GENERATOR, "-DCMAKE_BUILD_TYPE=Release",
                 f"-DPython_EXECUTABLE={sys.executable}", f"-DSENSE_CORE_PYTHON={'OFF' if args.no_binding else 'AUTO'}"]
    where = pybind11_dir()
    if where and not args.no_binding:
        configure.append(f"-Dpybind11_DIR={where}")
    cuda = gpu = tensorrt = None
    if host_only:
        configure += ["-DSENSE_CORE_HOST_ONLY=ON", "-DSENSE_CORE_TENSORRT=OFF"]
    else:
        cuda, gpu = find_cuda(), gpu_facts()
        if "problem" in cuda:
            raise CoreError(cuda["problem"] + ". A host without a CUDA toolkit builds the twin: --host-only")
        architecture = args.arch or gpu.get("architecture") or "native"
        configure += [f"-DCMAKE_CUDA_COMPILER={cuda['nvcc']}", f"-DCUDAToolkit_ROOT={cuda['root']}",
                      f"-DCMAKE_CUDA_ARCHITECTURES={architecture}", f"-DSENSE_CORE_TENSORRT={args.tensorrt.upper()}"]
        if args.tensorrt != "off":
            tensorrt = find_tensorrt(args)
            if "problem" in tensorrt:
                if args.tensorrt == "on":
                    raise CoreError(tensorrt["problem"])
                print(f"native_core: {tensorrt['problem']}: the core is built without the TensorRT backend", flush=True)
                configure[-1] = "-DSENSE_CORE_TENSORRT=OFF"
                tensorrt = None
            elif not tensorrt["parser_include"] or not tensorrt["libnvonnxparser"]:
                raise CoreError("TensorRT was found without NvOnnxParser.h or libnvonnxparser, which the engine builder "
                                "needs. Run the probe: python scripts/native_core.py probe")
            else:
                configure += [f"-DTENSORRT_INCLUDE_DIR={tensorrt['include']}",
                              f"-DTENSORRT_LIBRARY_DIR={tensorrt['library_dir']}"]
    libtorch = None
    if args.libtorch:
        libtorch = find_libtorch()
        if "problem" in libtorch:
            raise CoreError("the LibTorch backend cannot be built: " + libtorch["problem"])
        configure += ["-DSENSE_CORE_LIBTORCH=ON", f"-DTORCH_ROOT={libtorch['root']}"]
    files = sources()
    wanted = sources_sha256(files)
    if record(configure, run_dir, "configure.log") != 0:
        raise CoreError(f"the configuration failed (see {run_dir / 'configure.log'})")
    if record([cmake, "--build", build_dir, "-j", str(args.jobs)], run_dir, "build.log") != 0:
        raise CoreError(f"the build failed (see {run_dir / 'build.log'})")
    warnings = sum(": warning:" in line for line in (run_dir / "build.log").read_text(errors="replace").splitlines())
    status = record(["ctest", "--output-on-failure", "-j", str(max(1, min(args.jobs, 4)))], run_dir, "ctest.log",
                    cwd=build_dir)
    text = (run_dir / "ctest.log").read_text(errors="replace")
    counted = re.search(r"(\d+)% tests passed, (\d+) tests failed out of (\d+)", text)
    skipped = re.findall(r"^\s*\d+ - (\S+) \(Skipped\)", text, flags=re.M)
    if status != 0 or not counted or int(counted.group(2)) != 0:
        raise CoreError(f"the unit cases of the core failed (see {run_dir / 'ctest.log'})")
    try:
        config = json.loads((build_dir / CONFIG_FILE).read_text())
    except (OSError, ValueError) as error:
        raise CoreError(f"the build wrote no {CONFIG_FILE}: {error}") from None
    status, text = run([build_dir / "sense_host", "--build"], 60)
    try:
        said = json.loads(text)
    except ValueError:
        raise CoreError("sense_host --build printed no JSON: " + text.strip()[:200]) from None
    if said.get("sources_sha256") != wanted or config.get("sources_sha256") != wanted:
        raise CoreError(f"the binary names the sources {said.get('sources_sha256')}, and the tree has {wanted}: the "
                        "hash rule of CMakeLists.txt and of this tool disagree, or a file changed during the build")
    lines = link_lines(build_dir)
    for name in ("sense_host", "sense_core"):
        found = forbidden_in(lines[name])
        if found:
            raise CoreError(f"the link line of {name} names {', '.join(found)}: the core links no receiver, no Aerial "
                            "library and no interpreter")
    executables = {}
    for name in EXECUTABLES:
        path = build_dir / name
        if path.is_file():
            executables[name] = {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}
    missing = [name for name in REQUIRED if name not in executables]
    if missing:
        raise CoreError("the build holds no " + ", ".join(missing))
    binding = next(iter(sorted(build_dir.glob(BINDING_GLOB))), None)
    registry = json.loads(run([build_dir / "sense_host", "--registry"], 60)[1])
    tolerance = json.loads(run([build_dir / "sense_host", "--tolerance"], 60)[1])
    receipt = {
        "schema": RECEIPT_SCHEMA, "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": platform.node(), "machine": platform.machine(), "build_dir": str(build_dir),
        "sources": files, "sources_sha256": wanted, "source_files": len(files),
        "device": config["device"], "cuda_architectures": config["cuda_architectures"] or None,
        "toolchain": {"cmake": first_line([cmake, "--version"]), "generator": GENERATOR,
                      "cxx_compiler": config["cxx_compiler"], "cxx_compiler_path": config["cxx_compiler_path"],
                      "cuda_compiler": config["cuda_compiler"] or None,
                      "nvcc": None if cuda is None else cuda["nvcc"],
                      "cuda_toolkit": None if cuda is None else cuda["root"],
                      "flags": "-O2 -ffp-contract=off for C++, --fmad=false for CUDA: the units run as written"},
        "gpu": gpu,
        "tensorrt": None if not config["tensorrt"] else {
            "version": config["tensorrt"], "include": config["tensorrt_include"], "library": config["tensorrt_library"],
            "origin": None if tensorrt is None else tensorrt["origin"]},
        "libtorch": None if libtorch is None else {"version": libtorch["version"], "root": libtorch["root"]},
        "core": said, "registry": registry, "tolerance": tolerance,
        "library": {"path": str(build_dir / LIBRARY), "sha256": digest(build_dir / LIBRARY),
                    "bytes": (build_dir / LIBRARY).stat().st_size},
        "executables": executables,
        "binding": None if binding is None else {"path": str(binding), "sha256": digest(binding),
                                                 "bytes": binding.stat().st_size, "python": platform.python_version(),
                                                 "pybind11": python_packages()["pybind11"]},
        "link": lines,
        "tests": {"ran": int(counted.group(3)), "failed": 0, "skipped": skipped,
                  "device_vectors": "host only: not built" if host_only else
                                    "skipped: no device was reachable" if "sense_core_device_vectors" in skipped else "ran"},
        "compiler_warnings": warnings, "packages": python_packages()}
    receipt["backends"] = {"tensorrt": bool(config["tensorrt"]), "libtorch": libtorch is not None, "host_reference": True}
    (build_dir / RECEIPT).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(f"native_core: built {build_dir} for the device {config['device']}"
          + (f" (CUDA architecture {config['cuda_architectures']})" if config["cuda_architectures"] else "")
          + f", TensorRT {config['tensorrt'] or 'not built'}, {receipt['tests']['ran']} unit cases, "
          + (f"{len(skipped)} skipped ({', '.join(skipped)}), " if skipped else "none skipped, ")
          + f"{warnings} compiler warnings, sources {wanted[:16]}, sense_host {executables['sense_host']['sha256'][:12]} "
          + f"({build_dir / RECEIPT})")
    return 0


def verified_receipt(build_dir):
    """The receipt of a build, after its files and the tree were checked against it."""
    build_dir = Path(build_dir).resolve()
    try:
        receipt = json.loads((build_dir / RECEIPT).read_text(encoding="ascii"))
    except (OSError, ValueError) as error:
        raise CoreError(f"{build_dir} holds no receipt of a core build: {error}. Build it: python scripts/native_core.py "
                        "build") from None
    if not isinstance(receipt, dict) or receipt.get("schema") != RECEIPT_SCHEMA:
        raise CoreError(f"{build_dir / RECEIPT} is not a {RECEIPT_SCHEMA} receipt")
    try:
        files = sources()
        if receipt["sources"] != files or receipt["sources_sha256"] != sources_sha256(files):
            changed = sorted(name for name in set(files) | set(receipt["sources"])
                             if files.get(name) != receipt["sources"].get(name))
            raise CoreError(f"the sources changed since the build, first: {changed[0] if changed else 'the hash'}. "
                            "Rebuild: python scripts/native_core.py build")
        parts = [receipt["library"]] + list(receipt["executables"].values()) \
            + ([receipt["binding"]] if receipt["binding"] else [])
        for part in parts:
            if not Path(part["path"]).is_file() or digest(part["path"]) != part["sha256"]:
                raise CoreError(f"{part['path']} is missing or does not have the hash of the receipt. Rebuild")
        for name in REQUIRED:
            if name not in receipt["executables"]:
                raise CoreError(f"the receipt names no {name}")
        status, text = run([receipt["executables"]["sense_host"]["path"], "--build"], 60)
        if status != 0 or json.loads(text).get("sources_sha256") != receipt["sources_sha256"]:
            raise CoreError("sense_host does not name the sources of the receipt")
        for name in ("sense_host", "sense_core"):
            if forbidden_in(receipt["link"].get(name)):
                raise CoreError(f"the link line of {name} names a library the core must not link")
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, CoreError):
            raise
        raise CoreError(f"the receipt of {build_dir} is incomplete: {error!r}") from None
    return receipt


# ---------------------------------------------------------------- engine
def engine(args):
    receipt = verified_receipt(args.build_dir or DEFAULT_BUILD)
    out = Path(args.output_dir).resolve()
    if not out.is_dir() or any(out.iterdir()):
        raise CoreError(f"the output directory {out} does not exist or is not empty")
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from ocudu_cuda_isac.sensing import export
    except ImportError as error:
        raise CoreError(f"the export needs NumPy and ONNX in this interpreter: {error}") from None
    try:
        if args.backend == "libtorch":
            made = export.write_module(args.artifact, out / "module")
            print(json.dumps(made, sort_keys=True))
            return 0
        if not receipt["tensorrt"] or "sense_engine" not in receipt["executables"]:
            raise CoreError("this build of the core holds no TensorRT backend and no sense_engine. Run the probe, then "
                            "build with --tensorrt on")
        (out / "onnx").mkdir()
        (out / "engine").mkdir()
        exported = export.write_onnx(args.artifact, out / "onnx")
    except export.ExportError as error:
        raise CoreError(str(error)) from None
    command = [receipt["executables"]["sense_engine"]["path"], "--onnx", out / "onnx" / exported["file"], "--artifact",
               args.artifact, "--output-dir", out / "engine", "--device", str(args.device), "--precision", args.precision]
    if record(command, out, "sense-engine.log") != 0:
        raise CoreError(f"the engine was not built (see {out / 'sense-engine.log'})")
    manifest = json.loads((out / "engine/engine.json").read_text())
    print(f"native_core: the engine of the artifact {manifest['artifact_sha256'][:16]} was built with TensorRT "
          f"{manifest['library']['version']} as {manifest['precision']} on {manifest['device']['name']}: "
          f"{manifest['bytes']} bytes, {manifest['sha256'][:16]}, in {manifest['build_ms']:.0f} ms ({out / 'engine'})")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    def tensorrt_options(command):
        command.add_argument("--tensorrt-root", help="a TensorRT installation: include/ and lib/")
        command.add_argument("--tensorrt-include", help="the directory that holds NvInfer.h and NvOnnxParser.h")
        command.add_argument("--tensorrt-library", help="the directory that holds libnvinfer and libnvonnxparser")

    look = commands.add_parser("probe", help="checkpoint D0: build and run a one-layer TensorRT engine on this GPU")
    look.add_argument("--build-dir", help="core build directory (default: build/sense-core)")
    look.add_argument("--device", type=int, default=0)
    look.add_argument("--output", help="where the report goes (default: <build-dir>/sense-probe.json)")
    tensorrt_options(look)
    fetch = commands.add_parser("headers", help="fetch the open-source headers of the installed TensorRT release")
    fetch.add_argument("--version", help="the release, as 10.14 (default: that of the wheel in this interpreter)")
    make = commands.add_parser("build", help="configure, build and test the core; write its receipt")
    make.add_argument("--build-dir", help="default: build/sense-core, or build/sense-core-host with --host-only")
    make.add_argument("--host-only", action="store_true", help="the host twin only: no CUDA compiler and no device")
    make.add_argument("--tensorrt", choices=("auto", "on", "off"), default="auto")
    make.add_argument("--libtorch", action="store_true", help="build the declared fallback backend as well")
    make.add_argument("--arch", help="CUDA architecture, as 121 (default: that of the GPU nvidia-smi names)")
    make.add_argument("--no-binding", action="store_true", help="do not build the pybind11 binding")
    make.add_argument("--run-dir", help="directory for the logs (default: <build-dir>/evidence)")
    make.add_argument("--jobs", type=int, default=4)
    tensorrt_options(make)
    check = commands.add_parser("verify", help="check a build against its receipt and print the receipt")
    check.add_argument("--build-dir", help="core build directory (default: build/sense-core)")
    export = commands.add_parser("engine", help="export an artifact and build its engine on this GPU")
    export.add_argument("--artifact", required=True, help="directory of the frozen cusense-artifact/v1 artifact")
    export.add_argument("--output-dir", required=True, help="existing, empty directory: onnx/ and engine/, or module/")
    export.add_argument("--build-dir", help="core build directory (default: build/sense-core)")
    export.add_argument("--backend", choices=("tensorrt", "libtorch"), default="tensorrt")
    export.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    export.add_argument("--device", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        if args.command == "probe":
            return probe(args)
        if args.command == "headers":
            return headers(args)
        if args.command == "build":
            return build(args)
        if args.command == "engine":
            return engine(args)
        print(json.dumps(verified_receipt(args.build_dir or DEFAULT_BUILD), indent=2, sort_keys=True))
        return 0
    except CoreError as error:
        print(f"native_core: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
