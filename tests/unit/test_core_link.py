#!/usr/bin/env python3
"""The native sensing core links no host: python tests/unit/test_core_link.py -v

native/sense_core is a library and a sensor process that take observation views and return
decisions. They may link the C and C++ runtime, the observation ring, the CUDA runtime and an
engine library, TensorRT or LibTorch. They may not link an OCUDU archive, an Aerial library or a
Python library: an adapter produces records for them, and the sensor must start on a host
without an interpreter. The rule holds from Sprint 3.6 on and is checked three times: on the
link line that CMake wrote for the library and for sense_host, on what the dynamic loader
resolves for sense_host, and in the receipt of the build.

The build is the one ISAC_SENSE_CORE_BUILD names, or build/sense-core, or build/sense-core-host.
Without one the host twin is built into a scratch directory, which needs cmake and a C++17
compiler and no CUDA.
"""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import native_core  # noqa: E402
import sense_core_build  # noqa: E402

BUILD = None
PROBLEM = None


def setUpModule():
    global BUILD, PROBLEM
    BUILD = sense_core_build.existing()
    if BUILD is None:
        # With pybind11 the twin is the one the unit gate builds, and without it the twin has no binding.
        BUILD, PROBLEM = sense_core_build.host_twin(binding=native_core.pybind11_dir() is not None)


class CoreLinkTests(unittest.TestCase):
    def setUp(self):
        if PROBLEM is not None:
            self.skipTest(PROBLEM)
        self.receipt = native_core.verified_receipt(BUILD)

    def test_the_link_line_of_the_library_and_of_the_sensor_names_no_host(self):
        for target in ("sense_core", "sense_host"):
            line = " ".join((BUILD / f"CMakeFiles/{target}.dir/link.txt").read_text().split())
            self.assertEqual(native_core.forbidden_in(line), [], f"the link line of {target}: {line}")
            self.assertEqual(self.receipt["link"][target], line)
        # The sensor links the core and the ring, and nothing else that this project builds.
        line = self.receipt["link"]["sense_host"]
        self.assertIn("libsense_core.a", line)
        self.assertIn("libobs_ring.a", line)
        self.assertNotIn("libocudu", line.lower())

    def test_the_rule_names_what_it_forbids(self):
        for line in ("c++ -o sense_host a.o /opt/ocudu/lib/libocudu_phy.a", "c++ -o sense_host a.o -lsrsran_phy",
                     "c++ -o x a.o /usr/lib/aarch64-linux-gnu/libpython3.12.so.1.0", "c++ -o x a.o -lcuphy",
                     "c++ -o x a.o /opt/nvidia/aerial/lib/libnvipc.so"):
            self.assertTrue(native_core.forbidden_in(line), line)
        for line in ("c++ -o sense_host a.o libsense_core.a obs_ring/libobs_ring.a -lrt",
                     "c++ -o sense_host a.o /usr/local/cuda/lib64/libcudart.so /usr/lib/libnvinfer.so.10 -lpthread",
                     "c++ -o sense_host a.o /opt/torch/lib/libtorch.so /opt/torch/lib/libc10.so"):
            self.assertEqual(native_core.forbidden_in(line), [], line)

    def test_the_loader_resolves_no_host_library_for_the_sensor(self):
        if shutil.which("ldd") is None:
            self.skipTest("no ldd on this host")
        for name in ("sense_host", "sense_feed"):
            listed = subprocess.run(["ldd", str(BUILD / name)], capture_output=True, text=True).stdout
            # The name of each library, without the directory it was found in: a wheel of the CUDA
            # runtime lives under a directory named after an interpreter.
            names = [Path(line.split()[0]).name.lower() for line in listed.splitlines() if line.split()]
            self.assertTrue(names, name)
            for word in native_core.FORBIDDEN_LINK:
                self.assertFalse([found for found in names if word in found], f"{name} resolves a library named {word}")

    def test_the_sensor_starts_without_an_interpreter(self):
        environment = {"PATH": "/nonexistent", "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", "")}
        said = subprocess.run([str(BUILD / "sense_host"), "--build"], capture_output=True, text=True, env=environment)
        self.assertEqual(said.returncode, 0, said.stderr)
        self.assertIn(self.receipt["sources_sha256"], said.stdout)


if __name__ == "__main__":
    unittest.main()
