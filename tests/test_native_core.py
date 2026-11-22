"""Check scripts/native_core.py without a compiler, a GPU or TensorRT: the hash rule, the link rule and what it finds."""

import argparse
import hashlib
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import native_core  # noqa: E402

VERSION_HEADER = """#define TRT_MAJOR_ENTERPRISE 10
#define TRT_MINOR_ENTERPRISE 14
#define TRT_PATCH_ENTERPRISE 1
#define TRT_BUILD_ENTERPRISE 48
#define NV_TENSORRT_MAJOR TRT_MAJOR_ENTERPRISE //!< TensorRT major version.
"""
OLD_HEADER = "#define NV_TENSORRT_MAJOR 8\n#define NV_TENSORRT_MINOR 6\n#define NV_TENSORRT_PATCH 1\n#define NV_TENSORRT_BUILD 6\n"
NEXT_HEADER = VERSION_HEADER.replace("MAJOR_ENTERPRISE 10", "MAJOR_ENTERPRISE 11").replace(
    "MINOR_ENTERPRISE 14", "MINOR_ENTERPRISE 5").replace("PATCH_ENTERPRISE 1", "PATCH_ENTERPRISE 0").replace(
    "BUILD_ENTERPRISE 48", "BUILD_ENTERPRISE 63")


class NativeCoreTests(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.scratch = Path(scratch.name)

    def test_the_sources_hash_is_the_rule_cmake_states(self):
        files = native_core.sources()
        self.assertIn("CMakeLists.txt", files)
        self.assertIn("src/kernels/units.h", files)
        self.assertIn("include/sense_core/sense_core.h", files)
        self.assertIn("obs_ring/include/obs/record.h", files)
        self.assertFalse([name for name in files if "__pycache__" in name])
        record = "obs_ring/include/obs/record.h"
        text = "".join(f"{name}:{files[name]}\n" for name in sorted(files) if name != record) + f"{record}:{files[record]}\n"
        self.assertEqual(native_core.sources_sha256(files), hashlib.sha256(text.encode("ascii")).hexdigest())
        # One byte of one file is another build.
        changed = dict(files, **{"src/kernels/units.h": "0" * 64})
        self.assertNotEqual(native_core.sources_sha256(changed), native_core.sources_sha256(files))
        # The same rule, as CMakeLists.txt writes it down.
        cmake = (native_core.CORE / "CMakeLists.txt").read_text(encoding="ascii")
        self.assertIn('string(APPEND SENSE_SOURCE_LIST "${file}:${digest}\\n")', cmake)
        self.assertIn('string(APPEND SENSE_SOURCE_LIST "obs_ring/include/obs/record.h:${digest}\\n")', cmake)

    def test_the_core_takes_nothing_from_the_receiver_build(self):
        cmake = (native_core.CORE / "CMakeLists.txt").read_text(encoding="ascii")
        self.assertIn("project(sense_core LANGUAGES C CXX CUDA)", cmake)
        for word in ("ocudu_adapter", "OCUDU_BUILD", "find_package(ocudu", "srsran", "aerial_sdk"):
            self.assertNotIn(word, cmake)
        for path in native_core.CORE.rglob("*"):
            if path.is_file() and path.suffix in (".h", ".cpp", ".cu", ".c", ".txt", ".in"):
                text = path.read_text(encoding="ascii")  # every source of the core is ASCII
                self.assertNotIn("#include \"ocudu/", text, path)
                self.assertNotIn("#include <Python.h>", text, path)

    def test_the_link_rule(self):
        self.assertEqual(native_core.forbidden_in("c++ -o sense_host a.o libsense_core.a obs_ring/libobs_ring.a -lrt"), [])
        self.assertEqual(native_core.forbidden_in("c++ a.o /home/u/ocudu-cuda-isac/isac/lib/python3.12/site-packages/"
                                                  "tensorrt_libs/libnvinfer.so.10"), [])
        self.assertEqual(native_core.forbidden_in("c++ a.o /opt/ocudu/libocudu_phy.a -lpython3.12"),
                         ["-lpython3.12", "/opt/ocudu/libocudu_phy.a"])
        self.assertEqual(native_core.forbidden_in(None), [])

    def test_tensorrt_is_found_with_its_headers_or_the_missing_part_is_named(self):
        self.assertIsNone(native_core.header_version(self.scratch))
        root = self.scratch / "tensorrt"
        (root / "include").mkdir(parents=True)
        (root / "lib").mkdir()
        (root / "include/NvInferVersion.h").write_text(VERSION_HEADER)
        self.assertEqual(native_core.header_version(root / "include"), "10.14.1.48")
        (self.scratch / "old").mkdir()
        (self.scratch / "old/NvInferVersion.h").write_text(OLD_HEADER)
        self.assertEqual(native_core.header_version(self.scratch / "old"), "8.6.1.6")
        arguments = argparse.Namespace(tensorrt_root=str(root), tensorrt_include=None, tensorrt_library=None)
        # The host itself may hold TensorRT as a package installation, as the Spark does, and it is
        # found after what is named: here no system directory of the host is looked at.
        with mock.patch.object(native_core.importlib.util, "find_spec", return_value=None), \
                mock.patch.object(native_core, "HEADERS_DIR", self.scratch / "none"), \
                mock.patch.object(native_core, "system_places", return_value=([], [])):
            found = native_core.find_tensorrt(arguments)
            self.assertIn("problem", found)
            (root / "include/NvInfer.h").write_text("")
            self.assertIn("no libnvinfer", native_core.find_tensorrt(arguments)["problem"])
            (root / "lib/libnvinfer.so.10").write_text("")
            found = native_core.find_tensorrt(arguments)
            self.assertEqual((found["include"], found["libnvinfer"], found["header_version"], found["parser_include"]),
                             (str(root / "include"), str(root / "lib/libnvinfer.so.10"), "10.14.1.48", None))
            (root / "include/NvOnnxParser.h").write_text("")
            (root / "lib/libnvonnxparser.so.10").write_text("")
            found = native_core.find_tensorrt(arguments)
            self.assertEqual((found["parser_include"], found["libnvonnxparser"]),
                             (str(root / "include"), str(root / "lib/libnvonnxparser.so.10")))
            # Libraries without headers: the wheel. The message says how to get the headers.
            (root / "include/NvInfer.h").unlink()
            self.assertIn("native_core.py headers", native_core.find_tensorrt(arguments)["problem"])
        # A package installation is found where nothing is named, and after what is named.
        system = self.scratch / "system"
        (system / "include").mkdir(parents=True)
        (system / "lib").mkdir()
        (system / "include/NvInferVersion.h").write_text(NEXT_HEADER)
        for name in ("include/NvInfer.h", "include/NvOnnxParser.h", "lib/libnvinfer.so", "lib/libnvonnxparser.so"):
            (system / name).write_text("")
        (root / "include/NvInfer.h").write_text("")
        nothing = argparse.Namespace(tensorrt_root=None, tensorrt_include=None, tensorrt_library=None)
        with mock.patch.object(native_core.importlib.util, "find_spec", return_value=None), \
                mock.patch.object(native_core, "HEADERS_DIR", self.scratch / "none"), \
                mock.patch.object(native_core, "system_places", return_value=([system / "include"], [system / "lib"])), \
                mock.patch.dict(native_core.os.environ):
            native_core.os.environ.pop("TENSORRT_ROOT", None)
            found = native_core.find_tensorrt(nothing)
            self.assertEqual((found["origin"], found["include"], found["libnvinfer"], found["header_version"]),
                             ("system packages", str(system / "include"), str(system / "lib/libnvinfer.so"), "11.5.0.63"))
            found = native_core.find_tensorrt(arguments)
            self.assertEqual((found["origin"], found["header_version"]), (f"TENSORRT_ROOT {root}", "10.14.1.48"))
            # An empty place that was named is passed over, and the record says where was looked.
            empty = argparse.Namespace(tensorrt_root=str(self.scratch / "nothing"), tensorrt_include=None, tensorrt_library=None)
            found = native_core.find_tensorrt(empty)
            self.assertEqual((found["origin"], [item["origin"] for item in found["searched"]]),
                             ("system packages", [f"TENSORRT_ROOT {self.scratch / 'nothing'}", "system packages"]))
        # The system directories of a host, as the build looks at them.
        includes, libraries = native_core.system_places()
        self.assertEqual((includes[0], libraries[0]),
                         (Path("/usr/include") / native_core.triplet(), Path("/usr/lib") / native_core.triplet()))

    def test_the_tensorrt_sources_hold_for_tensorrt_10_and_11(self):
        # TensorRT 11 removed the weakly typed network with BuilderFlag::kFP16, and its runtime
        # libraries are libnvinfer.so.11. The core was first compiled against TensorRT 10.14 alone
        # and did not build on a host with 11.5: a name of one generation stands behind a check
        # of the version, and the other generation is refused in words.
        tool = (native_core.CORE / "tools/sense_engine.cpp").read_text(encoding="ascii")
        self.assertIn("#if NV_TENSORRT_MAJOR < 11\n#define SENSE_ENGINE_FP16_FLAG 1\n#else\n"
                      "#define SENSE_ENGINE_FP16_FLAG 0\n#endif\n", tool)
        code = "\n".join(line for line in tool.splitlines() if not line.lstrip().startswith("//"))
        guarded = re.findall(r"^#if SENSE_ENGINE_FP16_FLAG$(.*?)^#endif$", code, flags=re.S | re.M)
        self.assertEqual(len(guarded), 1)
        self.assertIn("config->setFlag(nvinfer1::BuilderFlag::kFP16);", guarded[0])
        self.assertNotIn("kFP16", code.replace(guarded[0], ""))
        self.assertIn('precision == "fp16" && SENSE_ENGINE_FP16_FLAG == 0', code)
        self.assertIn("--precision fp16 is not built with TensorRT %d.%d", code)
        # No other source of the core names a builder flag of one generation.
        for path in native_core.CORE.rglob("*"):
            if path.is_file() and path.suffix in (".h", ".cpp", ".cu") and path.name != "sense_engine.cpp":
                self.assertNotIn("BuilderFlag::", path.read_text(encoding="ascii"), path)
        # The runtime libraries of a wheel carry the major of their release, which the headers name.
        cmake = (native_core.CORE / "CMakeLists.txt").read_text(encoding="ascii")
        self.assertIn('list(APPEND SENSE_TENSORRT_NVINFER_NAMES "libnvinfer.so.${major}")', cmake)
        self.assertIn('list(APPEND SENSE_TENSORRT_NVONNXPARSER_NAMES "libnvonnxparser.so.${major}")', cmake)
        self.assertNotIn("libnvinfer.so.10", cmake)
        self.assertNotIn("libnvonnxparser.so.10", cmake)
        root = self.scratch / "tensorrt-11"
        (root / "include").mkdir(parents=True)
        (root / "lib").mkdir()
        (root / "include/NvInferVersion.h").write_text(NEXT_HEADER)
        for name in ("include/NvInfer.h", "include/NvOnnxParser.h", "lib/libnvinfer.so.11", "lib/libnvonnxparser.so.11"):
            (root / name).write_text("")
        arguments = argparse.Namespace(tensorrt_root=str(root), tensorrt_include=None, tensorrt_library=None)
        with mock.patch.object(native_core.importlib.util, "find_spec", return_value=None), \
                mock.patch.object(native_core, "HEADERS_DIR", self.scratch / "none"), \
                mock.patch.object(native_core, "system_places", return_value=([], [])):
            found = native_core.find_tensorrt(arguments)
        self.assertEqual((found["header_version"], found["libnvinfer"], found["libnvonnxparser"]),
                         ("11.5.0.63", str(root / "lib/libnvinfer.so.11"), str(root / "lib/libnvonnxparser.so.11")))

    def test_a_directory_without_a_receipt_is_no_build_and_the_usage_names_the_commands(self):
        with self.assertRaisesRegex(native_core.CoreError, "holds no receipt"):
            native_core.verified_receipt(self.scratch)
        (self.scratch / native_core.RECEIPT).write_text('{"schema": "other"}')
        with self.assertRaisesRegex(native_core.CoreError, "is not a isac-core-build/v1 receipt"):
            native_core.verified_receipt(self.scratch)
        said = subprocess.run([sys.executable, str(ROOT / "scripts/native_core.py"), "--help"], capture_output=True,
                              text=True)
        self.assertEqual(said.returncode, 0)
        for command in ("probe", "headers", "build", "verify", "engine"):
            self.assertIn(command, said.stdout)
        said = subprocess.run([sys.executable, str(ROOT / "scripts/native_core.py"), "verify", "--build-dir",
                               str(self.scratch / "missing")], capture_output=True, text=True)
        self.assertEqual(said.returncode, 1)
        self.assertIn("native_core: ", said.stderr)


if __name__ == "__main__":
    unittest.main()
