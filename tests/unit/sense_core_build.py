"""One build of the host twin of the native sensing core for the unit modules of one process.

Three unit modules need a build of native/sense_core: the unit gate, the link rule and the
rehearsal of the driver. Each asks here. The first call builds the host twin into a scratch
directory with scripts/native_core.py, which needs cmake and a C++17 compiler and no CUDA, and
the others get the same directory. The directory is removed when the interpreter ends.

A module that only checks what was linked takes a build that exists already: the one
ISAC_SENSE_CORE_BUILD names, or build/sense-core, or build/sense-core-host of the checkout.
"""

import atexit
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import native_core  # noqa: E402

_made = {}


def existing():
    """A verified build of the checkout or of the environment, or None."""
    named = [Path(os.environ["ISAC_SENSE_CORE_BUILD"])] if os.environ.get("ISAC_SENSE_CORE_BUILD") else []
    for candidate in named + [native_core.DEFAULT_BUILD, native_core.DEFAULT_BUILD_HOST]:
        try:
            native_core.verified_receipt(candidate)
            return Path(candidate).resolve()
        except native_core.CoreError:
            continue
    return None


def host_twin(binding=True):
    """The host twin, built once per process. Returns (build directory or None, problem or None)."""
    if "twin" in _made:
        return _made["twin"]
    if shutil.which("cmake") is None or (shutil.which("c++") is None and not os.environ.get("CXX")):
        return None, "the host twin of the core needs cmake and a C++17 compiler"
    if binding and native_core.pybind11_dir() is None:
        return None, "the binding of the core needs pybind11 in this interpreter"
    scratch = tempfile.mkdtemp(prefix="sense-core-")
    atexit.register(shutil.rmtree, scratch, True)
    build = Path(scratch) / "sense-core-host"
    arguments = ["build", "--host-only", "--build-dir", str(build)] + ([] if binding else ["--no-binding"])
    if native_core.main(arguments) != 0:
        made = (None, "the host twin of the core does not build on this host")
    else:
        made = (build, None)
    if binding or made[0] is None:
        _made["twin"] = made
    return made
