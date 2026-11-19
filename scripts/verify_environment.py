#!/usr/bin/env python3
"""Exercise the selected isac profile and emit a machine-readable environment report."""

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from urllib.parse import urlsplit
from urllib.request import url2pathname

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIRS = frozenset(("site-packages", "dist-packages"))
RECREATE = "Remove isac/ and run 'make env' on this host."


def created_here(command, venv):
    """Return whether the creation command stored in pyvenv.cfg targets this directory."""
    # venv writes the absolute target last and unquoted, so try each suffix to keep spaces intact.
    starts = (index + 1 for index, char in enumerate(command) if char == " ")
    targets = (Path(command[start:]) for start in starts)
    return any(target.is_absolute() and target.resolve() == venv for target in targets)


def check_isolation(venv):
    if sys.version_info < (3, 11):
        raise ValueError("Python 3.11+ required for the shared research environment")
    if sys.prefix == sys.base_prefix or Path(sys.prefix).resolve() != venv:
        message = "Use this checkout's isac/bin/python or source isac/bin/activate"
        active = os.environ.get("VIRTUAL_ENV")
        if active and Path(active).resolve() != venv:
            message += (f". VIRTUAL_ENV is {active}, expected {venv}. Deactivate the other "
                        f"environment, or if isac/ was created at a different path: {RECREATE}")
        raise ValueError(message)
    settings = dict(line.split("=", 1) for line in (venv / "pyvenv.cfg").read_text().splitlines()
                    if "=" in line)
    settings = {key.strip().lower(): value.strip() for key, value in settings.items()}
    if settings.get("include-system-site-packages", "").lower() != "false":
        raise ValueError("Create isac without --system-site-packages")
    command = settings.get("command")
    if command and not created_here(command, venv):
        raise ValueError(f"{venv} was created at another path and a virtual environment "
                         f"cannot be relocated. {RECREATE}")


def check_search_path(venv, root, entries=None, stdlib=None):
    """Reject import roots outside the standard library, this venv and this checkout."""
    entries = sys.path if entries is None else entries
    stdlib = Path(os.__file__).resolve().parent if stdlib is None else stdlib
    archive = stdlib.with_name(f"python{sys.version_info[0]}{sys.version_info[1]}.zip")
    foreign = []
    for entry in entries:
        path = Path(entry or ".").resolve()
        if not path.exists() or path == archive:
            continue
        if path.is_relative_to(venv) or path.is_relative_to(root):
            continue
        if path.is_relative_to(stdlib) and not PACKAGE_DIRS.intersection(path.relative_to(stdlib).parts):
            continue
        foreign.append(str(path))
    if foreign:
        hint = " Unset PYTHONPATH and retry." if os.environ.get("PYTHONPATH") else ""
        raise ValueError("Packages outside the isac environment are importable from: "
                         + ", ".join(foreign) + "." + hint)


def require_local_install(name, path, editable=False):
    """Read pip's PEP 610 receipt to distinguish the selected source from a registry wheel."""
    dist = metadata.distribution(name)
    origin = json.loads(dist.read_text("direct_url.json") or "{}")
    url = urlsplit(origin.get("url", ""))
    if url.scheme != "file" or url.netloc not in ("", "localhost") or "dir_info" not in origin:
        hint = " (PYTHONPATH is set and can shadow pip's record)" if os.environ.get("PYTHONPATH") else ""
        raise ValueError(f"{name} must be installed from the local source at {path}{hint}")
    actual = Path(url2pathname(url.path)).resolve()
    if actual != path.resolve():
        raise ValueError(f"{name} came from {actual}, expected {path}")
    record = {"version": dist.version, "source": str(actual)}
    if editable:
        details = origin["dir_info"]
        if not isinstance(details, dict) or details.get("editable") is not True:
            raise ValueError(f"{name} must be an editable install of {path}. Run 'make env'.")
        record["editable"] = True
    return record


def project_import(root):
    """Import the project and require the module that executes to be the editable source."""
    import ocudu_cuda_isac

    module = Path(ocudu_cuda_isac.__file__).resolve()
    if not module.is_relative_to(root / "src"):
        raise ValueError(f"ocudu_cuda_isac resolves to {module}, expected the source under {root / 'src'}")
    return str(module)


def installed_distributions():
    """Name and version of every distribution this interpreter resolves, first match per name."""
    found = {}
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            found.setdefault(re.sub(r"[-_.]+", "-", name).lower(), dist.version)
    return dict(sorted(found.items()))


def describe_host():
    """Host identity recorded with every report, including a failed one."""
    try:
        release = platform.freedesktop_os_release()
    except (AttributeError, OSError):
        release = {}
    return {"os": {key.lower(): release.get(key) for key in ("ID", "VERSION_ID", "PRETTY_NAME")},
            "node": platform.node(), "prefix": sys.prefix, "base_prefix": sys.base_prefix,
            "project_root": str(ROOT)}


def check_simulation(backend):
    import mitsuba as mi

    variant = {"cpu": "llvm_ad_mono_polarized", "cuda": "cuda_ad_mono_polarized"}[backend]
    mi.set_variant(variant)
    import numpy as np
    import torch
    from sionna.phy.channel import AWGN
    from export_rt_episode import trace_scene

    if backend == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot use CUDA in this venv")
    x = torch.tensor([[1 + 2j, -3j]], device=backend, dtype=torch.complex64)
    y = AWGN(device=str(x.device))(x, 0.0)
    if y.device != x.device or not torch.equal(x, y):
        raise ValueError("Zero-noise Sionna PHY reference lost the input symbol or device")
    # Exercise the same LOS/array/Paths.cfr path consumed by the real offline exporter.
    cfr = trace_scene(False)
    if (cfr.shape != (1, 4, 1, 1, 4, 12) or not np.iscomplexobj(cfr)
            or not np.isfinite(cfr).all() or not np.any(cfr)):
        raise ValueError(f"Unexpected Sionna RT direct-link response, shape {cfr.shape}")
    active_variant = mi.variant()
    if active_variant != variant:
        raise RuntimeError(f"Requested Mitsuba {variant}, but the path solver used {active_variant}")
    return {"phy_device": str(y.device), "awgn_zero_noise_equal": True,
            "gpu": torch.cuda.get_device_name() if backend == "cuda" else None,
            "torch_cuda_build": torch.version.cuda, "rt_variant": active_variant,
            "cfr_shape": list(cfr.shape), "cfr_peak": float(np.abs(cfr).max())}


def check_native(venv, expected_commit):
    binaries = {}
    for name in ("gnb", "pusch_e2e_pipeline_test"):
        entry = venv / "bin" / name
        if not entry.is_file() or not os.access(entry, os.X_OK):
            raise ValueError(f"Full OCUDU build required: missing executable {entry}")
        with entry.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        binaries[name] = {"path": str(entry), "target": str(entry.resolve()), "sha256": digest}

    gnb = Path(binaries["gnb"]["target"])
    build = gnb.parents[2]  # <build>/apps/gnb/gnb
    receipt = build / "isac-native.json"
    stamp = json.loads(receipt.read_text())
    pusch = binaries["pusch_e2e_pipeline_test"]
    if (stamp.get("target") != "all" or stamp.get("commit") != expected_commit
            or not Path(pusch["target"]).is_relative_to(build)
            or stamp.get("gnb_sha256") != binaries["gnb"]["sha256"]
            or stamp.get("binary_sha256") != pusch["sha256"]):
        raise ValueError(f"Native binaries differ from the locked full-build receipt at {receipt}")

    result = subprocess.run([str(gnb), "--version"], capture_output=True, text=True,
                            check=True, timeout=15)
    # The pinned gNB announces itself before CLI11 processes --version.
    banner = re.search(r"^OCUDU 5G gNB version 26\.04\.0 \(([a-f0-9]{4,40})?\)$",
                       result.stdout, flags=re.MULTILINE)
    if not banner:
        raise ValueError(f"Unexpected gNB --version output: {result.stdout.strip()}")
    abbreviation = banner.group(1)
    if abbreviation and not expected_commit.startswith(abbreviation):
        raise ValueError(f"gNB version reports {abbreviation}, expected {expected_commit}")
    if not abbreviation and stamp.get("source_kind") != "verified-archive":
        raise ValueError("Git build failed to report its revision")
    return {"executables": binaries, "gnb_version": banner.group(), "build_receipt": str(receipt),
            "commit": expected_commit}


def run_checks(args, report):
    venv = ROOT / "isac"
    check_isolation(venv)
    check_search_path(venv, ROOT)
    report["venv"] = str(venv)
    distributions = report["distributions"]
    distributions["ocudu-cuda-isac"] = require_local_install("ocudu-cuda-isac", ROOT, editable=True)
    report["project_import"] = project_import(ROOT)
    report["installed"] = installed_distributions()
    check = subprocess.run([sys.executable, "-m", "pip", "--disable-pip-version-check", "check"],
                           capture_output=True, text=True, check=False)
    report["pip_check"] = check.stdout.strip()
    if check.returncode:
        raise RuntimeError((check.stdout + check.stderr).strip() or "pip check failed")
    if args.profile == "simulation":
        for name in ("sionna", "sionna-rt"):
            distributions[name] = require_local_install(name, ROOT / "ext" / name)
        versions = {"sionna": "2.2.0", "sionna-rt": "2.2.0",
                    "mitsuba": "3.9.1", "drjit": "1.5.0"}
        for name in ("mitsuba", "drjit", "torch", "numpy", "scipy"):
            distributions[name] = {"version": metadata.version(name)}
        for name, expected in versions.items():
            if distributions[name]["version"] != expected:
                raise ValueError(f"{name}: expected {expected}, found {distributions[name]['version']}")
        report["simulation"] = check_simulation(args.backend)
    if args.with_native:
        pins = json.loads((ROOT / "ext/lock.json").read_text())["dependencies"]
        commit = next(pin["commit"] for pin in pins if pin["path"] == "ext/ocudu")
        report["native"] = check_native(venv, commit)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("core", "simulation"), default="core")
    parser.add_argument("--backend", choices=("cpu", "cuda"),
                        help="force both simulation backends (default for simulation: cpu)")
    parser.add_argument("--with-native", action="store_true", help="verify both full-build executables and run gnb --version")
    parser.add_argument("--report", type=Path, help="write the JSON report to a new file")
    args = parser.parse_args(argv)
    if args.backend and args.profile != "simulation":
        parser.error("--backend applies to --profile simulation")
    if args.profile == "simulation" and not args.backend:
        args.backend = "cpu"
    if args.report and args.report.exists():
        parser.exit(2, f"Report already exists: {args.report}\n")
    report = {"schema": "isac-environment/v1", "profile": args.profile,
              "backend_requested": args.backend, "native_requested": args.with_native,
              "interpreter": sys.executable, "python": platform.python_version(),
              "platform": platform.platform(), "machine": platform.machine(),
              **describe_host(), "distributions": {}}
    try:
        run_checks(args, report)
        report["status"] = "pass"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    print(payload, end="")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        try:
            with args.report.open("x", encoding="utf-8") as file:
                file.write(payload)
        except FileExistsError:
            parser.exit(2, f"Report already exists: {args.report}\n")
    return int(report["status"] != "pass")


if __name__ == "__main__":
    sys.exit(main())
