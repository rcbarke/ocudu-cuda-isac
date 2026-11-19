"""Portable tests of the environment gate. No heavy module imports or host GPU."""

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_environment.py"
spec = importlib.util.spec_from_file_location("verify_environment", SCRIPT)
environment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(environment)


class EnvironmentGateTests(unittest.TestCase):
    def test_project_venv_rejects_global_python_and_system_site_packages(self):
        with tempfile.TemporaryDirectory() as temp:
            venv = Path(temp).resolve() / "isac"
            venv.mkdir()
            with mock.patch.object(sys, "base_prefix", temp), mock.patch.object(sys, "prefix", temp):
                with self.assertRaises(ValueError):
                    environment.check_isolation(venv)
            with mock.patch.object(sys, "prefix", str(venv)), mock.patch.object(sys, "version_info", (3, 12)):
                config = venv / "pyvenv.cfg"
                config.write_text("include-system-site-packages = true\n")
                with self.assertRaises(ValueError):
                    environment.check_isolation(venv)
                config.write_text("include-system-site-packages = false\n")
                environment.check_isolation(venv)

    def test_local_receipt_must_reference_selected_tree(self):
        dist = mock.Mock(version="2.2.0")
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(
                environment.metadata, "distribution", return_value=dist):
            expected = Path(temp) / "local checkout"
            valid = {"url": expected.as_uri(), "dir_info": {}}
            for receipt in (None, {}, {**valid, "url": (Path(temp) / "other").as_uri()},
                            {"url": expected.as_uri(), "archive_info": {}}):
                dist.read_text.return_value = None if receipt is None else json.dumps(receipt)
                with self.subTest(receipt=receipt), self.assertRaises(ValueError):
                    environment.require_local_install("sionna", expected)
            dist.read_text.return_value = json.dumps(valid)
            self.assertEqual(environment.require_local_install("sionna", expected),
                             {"version": "2.2.0", "source": str(expected)})

    def test_full_native_receipt_and_version_banner(self):
        commit = "27e84cb3da65205b55dbd56ace6fac93a035331e"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            venv, build = root / "isac", root / "build"
            (venv / "bin").mkdir(parents=True)
            gnb = build / "apps/gnb/gnb"
            pusch = build / "tests/pusch_e2e_pipeline_test"
            for path in (gnb, pusch):
                path.parent.mkdir(parents=True)
                path.write_text("#!/bin/sh\nexit 0\n")
                path.chmod(0o755)
                (venv / "bin" / path.name).symlink_to(path)
            cases = [(commit[:8], "26.04.0", "git", True), ("", "26.04.0", "verified-archive", True),
                     ("deadbeef", "26.04.0", "git", False), ("", "26.04.0", "git", False),
                     (commit[:8], "25.04.0", "git", False)]
            for rev, version, kind, valid in cases:
                banner = f"OCUDU 5G gNB version {version} ({rev})"
                gnb.write_text(f'#!/bin/sh\n[ "$1" = "--version" ] || exit 2\necho "{banner}"\n')
                stamp = {"target": "all", "commit": commit, "source_kind": kind,
                         "gnb_sha256": hashlib.sha256(gnb.read_bytes()).hexdigest(),
                         "binary_sha256": hashlib.sha256(pusch.read_bytes()).hexdigest()}
                (build / "isac-native.json").write_text(json.dumps(stamp))
                with self.subTest(rev=rev, version=version, kind=kind):
                    if valid:
                        report = environment.check_native(venv, commit)
                        self.assertEqual(report["gnb_version"], banner)
                        self.assertEqual(report["executables"][pusch.name]["sha256"], stamp["binary_sha256"])
                    else:
                        with self.assertRaises(ValueError):
                            environment.check_native(venv, commit)
            stamp["target"] = "baseline"
            (build / "isac-native.json").write_text(json.dumps(stamp))
            with self.assertRaisesRegex(ValueError, "full-build receipt"):
                environment.check_native(venv, commit)
            (venv / "bin" / "gnb").unlink()
            with self.assertRaisesRegex(ValueError, "missing executable"):
                environment.check_native(venv, commit)

    def test_unsatisfied_check_returns_failure_and_writes_diagnostic(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(
                environment, "run_checks", side_effect=RuntimeError("test missing backend")):
            report = Path(temp) / "run" / "environment.json"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(environment.main(["--profile", "simulation", "--backend", "cuda",
                                                   "--report", str(report)]), 1)
            data = json.loads(report.read_text())
            self.assertEqual(data["status"], "failed")
            self.assertEqual(data["backend_requested"], "cuda")
            self.assertIn("test missing backend", data["error"])


if __name__ == "__main__":
    unittest.main()
