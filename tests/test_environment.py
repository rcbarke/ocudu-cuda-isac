"""Portable tests of the environment gate. No heavy module imports or host GPU."""

import contextlib
import hashlib
import importlib.util
import io
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
import types
import unittest
from unittest import mock
import venv

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SCRIPT = SCRIPTS / "verify_environment.py"
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
            listing = ("--rx-device-grid MODE", "--gpu-output-buffer MODE", "--require-gpu-resident")
            for path in (gnb, pusch):
                path.parent.mkdir(parents=True)
                path.write_text('#!/bin/sh\n[ "$1" = "--help" ] || exit 2\n'
                                + "".join(f'echo "  {option}"\n' for option in listing))
                path.chmod(0o755)
                (venv / "bin" / path.name).symlink_to(path)
            cache = build / "CMakeCache.txt"
            cache.write_text("CMAKE_BUILD_TYPE:STRING=Release\n")
            cases = [(commit[:8], "26.04.0", "git", True), ("", "26.04.0", "verified-archive", True),
                     ("deadbeef", "26.04.0", "git", False), ("", "26.04.0", "git", False),
                     (commit[:8], "25.04.0", "git", False)]
            for rev, version, kind, valid in cases:
                banner = f"OCUDU 5G gNB version {version} ({rev})"
                gnb.write_text(f'#!/bin/sh\n[ "$1" = "--version" ] || exit 2\necho "{banner}"\n')
                stamp = {"target": "all", "commit": commit, "source_kind": kind, "machine": "spark",
                         "cuda_arch": "121", "cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
                         "gnb_sha256": hashlib.sha256(gnb.read_bytes()).hexdigest(),
                         "binary_sha256": hashlib.sha256(pusch.read_bytes()).hexdigest()}
                (build / "isac-native.json").write_text(json.dumps(stamp))
                with self.subTest(rev=rev, version=version, kind=kind):
                    if valid:
                        report = environment.check_native(venv, commit)
                        self.assertEqual(report["gnb_version"], banner)
                        self.assertEqual(report["executables"][pusch.name]["sha256"], stamp["binary_sha256"])
                        self.assertEqual(report["pusch_test_options"], list(environment.NATIVE_TEST_OPTIONS))
                        self.assertEqual(report["build"], {"machine": "spark", "cuda_arch": "121",
                                                           "cpu_target": None, "toolchain": None})
                    else:
                        with self.assertRaises(ValueError):
                            environment.check_native(venv, commit)
            # A valid receipt again, then one defect at a time.
            gnb.write_text(f'#!/bin/sh\necho "OCUDU 5G gNB version 26.04.0 ({commit[:10]})"\n')
            stamp.update(source_kind="git", gnb_sha256=hashlib.sha256(gnb.read_bytes()).hexdigest())
            (build / "isac-native.json").write_text(json.dumps(stamp))
            environment.check_native(venv, commit)
            cache.write_text("CMAKE_BUILD_TYPE:STRING=Debug\n")
            with self.assertRaisesRegex(ValueError, "reconfigured after its receipt"):
                environment.check_native(venv, commit)
            cache.write_text("CMAKE_BUILD_TYPE:STRING=Release\n")
            pusch.write_text('#!/bin/sh\necho "  --rx-device-grid MODE"\necho "  --require-gpu-resident"\n')
            stamp["binary_sha256"] = hashlib.sha256(pusch.read_bytes()).hexdigest()
            (build / "isac-native.json").write_text(json.dumps(stamp))
            with self.assertRaisesRegex(ValueError, "--help lacks --gpu-output-buffer"):
                environment.check_native(venv, commit)
            pusch.write_text("#!/bin/sh\nexit 127\n")
            stamp["binary_sha256"] = hashlib.sha256(pusch.read_bytes()).hexdigest()
            (build / "isac-native.json").write_text(json.dumps(stamp))
            with self.assertRaises(subprocess.CalledProcessError):
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

    def test_relocated_or_foreign_active_environment_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp).resolve()
            venv_dir = base / "checkout with space" / "isac"
            venv_dir.mkdir(parents=True)
            (base / "link").symlink_to(venv_dir.parent)
            config = venv_dir / "pyvenv.cfg"
            created = ('include-system-site-packages = false\n'
                       'command = /usr/bin/python3 -m venv --prompt="isac" ')
            with mock.patch.object(sys, "prefix", str(venv_dir)), \
                    mock.patch.object(sys, "version_info", (3, 12)):
                config.write_text(created + str(base / "old checkout" / "isac") + "\n")
                with self.assertRaisesRegex(ValueError, "cannot be relocated"):
                    environment.check_isolation(venv_dir)
                for target in (venv_dir, base / "link" / "isac"):
                    config.write_text(created + str(target) + "\n")
                    with self.subTest(target=target):
                        environment.check_isolation(venv_dir)
            with mock.patch.object(sys, "prefix", sys.base_prefix), \
                    mock.patch.dict(os.environ, {"VIRTUAL_ENV": str(base / "elsewhere" / "isac")}):
                with self.assertRaisesRegex(ValueError, "VIRTUAL_ENV is .*elsewhere"):
                    environment.check_isolation(venv_dir)

    def test_search_path_rejects_packages_outside_the_environment(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp).resolve()
            root, stdlib = base / "checkout", base / "usr/lib/python3.12"
            venv_dir = root / "isac"
            site = venv_dir / "lib/python3.12/site-packages"
            archive = base / "usr/lib/python312.zip"
            leaks = (base / "usr/lib/python3/dist-packages", stdlib / "site-packages",
                     base / "other checkout/src")
            for path in (root / "scripts", root / "src", stdlib / "lib-dynload", site, *leaks):
                path.mkdir(parents=True)
            archive.write_bytes(b"")
            clean = [str(root / "scripts"), str(archive), str(stdlib), str(stdlib / "lib-dynload"),
                     str(site), str(root / "src"), str(base / "absent")]
            with mock.patch.object(sys, "version_info", (3, 12)):
                environment.check_search_path(venv_dir, root, clean, stdlib)
                for leak in leaks:
                    with self.subTest(leak=leak), \
                            self.assertRaisesRegex(ValueError, re.escape(str(leak))):
                        environment.check_search_path(venv_dir, root, [*clean, str(leak)], stdlib)
                with mock.patch.dict(os.environ, {"PYTHONPATH": str(leaks[0])}), \
                        self.assertRaisesRegex(ValueError, "Unset PYTHONPATH"):
                    environment.check_search_path(venv_dir, root, [str(leaks[0]), *clean], stdlib)

    def test_project_must_be_editable_and_import_from_its_source_tree(self):
        dist = mock.Mock(version="0.1.0")
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(
                environment.metadata, "distribution", return_value=dist):
            root = Path(temp).resolve() / "checkout"
            source = root / "src/ocudu_cuda_isac/__init__.py"
            source.parent.mkdir(parents=True)
            for details in ({}, {"editable": False}, {"editable": "true"}, None):
                dist.read_text.return_value = json.dumps({"url": root.as_uri(), "dir_info": details})
                with self.subTest(details=details), self.assertRaisesRegex(ValueError, "editable"):
                    environment.require_local_install("ocudu-cuda-isac", root, editable=True)
            dist.read_text.return_value = json.dumps(
                {"url": root.as_uri(), "dir_info": {"editable": True}})
            self.assertEqual(environment.require_local_install("ocudu-cuda-isac", root, editable=True),
                             {"version": "0.1.0", "source": str(root), "editable": True})
            stale = root / "isac/lib/python3.12/site-packages/ocudu_cuda_isac/__init__.py"
            for module in (stale, root.parent / "other/src/ocudu_cuda_isac/__init__.py"):
                double = types.SimpleNamespace(__file__=str(module))
                with self.subTest(module=module), \
                        mock.patch.dict(sys.modules, {"ocudu_cuda_isac": double}), \
                        self.assertRaisesRegex(ValueError, "expected the source under"):
                    environment.project_import(root)
            double = types.SimpleNamespace(__file__=str(source))
            with mock.patch.dict(sys.modules, {"ocudu_cuda_isac": double}):
                self.assertEqual(environment.project_import(root), str(source))

    def test_inventory_lists_each_resolved_distribution_once(self):
        def double(name, version):
            return mock.Mock(metadata={"Name": name}, version=version)

        found = [double("pip", "24.0"), double("ocudu_cuda_isac", "0.1.0"),
                 double("Ocudu.Cuda-ISAC", "9.9"), double(None, "1.0")]
        with mock.patch.object(environment.metadata, "distributions", return_value=found):
            self.assertEqual(list(environment.installed_distributions().items()),
                             [("ocudu-cuda-isac", "0.1.0"), ("pip", "24.0")])

    def test_core_profile_records_inventory_and_fails_on_inconsistent_pip(self):
        inventory = {"ocudu-cuda-isac": "0.1.0", "pip": "24.0"}
        receipt = {"version": "0.1.0", "source": str(environment.ROOT), "editable": True}
        doubles = {"check_isolation": None, "check_search_path": None,
                   "require_local_install": receipt, "installed_distributions": inventory,
                   "project_import": "/checkout/src/ocudu_cuda_isac/__init__.py"}
        with contextlib.ExitStack() as stack:
            for name, value in doubles.items():
                stack.enter_context(mock.patch.object(environment, name, return_value=value))
            run = stack.enter_context(mock.patch.object(environment.subprocess, "run"))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            temp = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            run.return_value = mock.Mock(returncode=0, stdout="No broken requirements found.\n",
                                         stderr="")
            self.assertEqual(environment.main(["--report", str(temp / "pass.json")]), 0)
            data = json.loads((temp / "pass.json").read_text())
            self.assertEqual((data["status"], data["installed"], data["pip_check"]),
                             ("pass", inventory, "No broken requirements found."))
            self.assertEqual(data["venv"], str(environment.ROOT / "isac"))
            self.assertEqual(data["distributions"], {"ocudu-cuda-isac": receipt})
            environment.require_local_install.assert_called_once_with(
                "ocudu-cuda-isac", environment.ROOT, editable=True)
            run.return_value = mock.Mock(
                returncode=1, stdout="torch 2.9.1 requires numpy, which is not installed.\n", stderr="")
            self.assertEqual(environment.main(["--report", str(temp / "broken.json")]), 1)
            data = json.loads((temp / "broken.json").read_text())
            self.assertEqual(data["status"], "failed")
            self.assertIn("requires numpy", data["error"])

    def test_simulation_report_binds_installs_to_the_verified_pins(self):
        pins = {"ocudu": {"name": "ocudu", "path": "ext/ocudu-deepsig-cuda-inline", "commit": "c" * 40,
                          "kind": "Git worktree"},
                "sionna": {"name": "sionna", "path": "ext/phy", "commit": "a" * 40, "kind": "Git worktree"},
                "sionna-rt": {"name": "sionna-rt", "path": "ext/rt", "commit": "b" * 40,
                              "kind": "Git worktree"}}
        versions = {"mitsuba": "3.9.1", "drjit": "1.5.0", "torch": "2.15.0", "numpy": "2.5.5",
                    "scipy": "1.18.1"}
        with contextlib.ExitStack() as stack:
            root = Path(stack.enter_context(tempfile.TemporaryDirectory())).resolve()
            (root / "ext").mkdir()
            (root / "ext/lock.json").write_text(json.dumps({"dependencies": list(pins.values())}))
            stack.enter_context(mock.patch.object(environment, "ROOT", root))
            for name in ("check_isolation", "check_search_path", "project_import",
                         "installed_distributions", "check_simulation"):
                stack.enter_context(mock.patch.object(environment, name, return_value={}))
            stack.enter_context(mock.patch.object(environment, "pinned_sources", return_value=pins))
            local = stack.enter_context(mock.patch.object(
                environment, "require_local_install", side_effect=lambda *a, **k: {"version": "2.2.0"}))
            match = stack.enter_context(mock.patch.object(
                environment, "matches_pinned_source",
                side_effect=[{"sionna/__init__.py", "sionna/phy.py"}, {"sionna/rt/paths.py"}]))
            native = stack.enter_context(mock.patch.object(environment, "check_native", return_value={}))
            stack.enter_context(mock.patch.object(environment.metadata, "version",
                                                  side_effect=versions.__getitem__))
            stack.enter_context(mock.patch.object(
                environment.subprocess, "run", return_value=mock.Mock(returncode=0, stdout="", stderr="")))
            output = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            self.assertEqual(environment.main(["--profile", "simulation", "--with-native"]), 0)
            report = json.loads(output.getvalue())
            self.assertEqual(report["sources"], pins)
            self.assertEqual([call.args for call in local.call_args_list[1:]],
                             [("sionna", root / "ext/phy"), ("sionna-rt", root / "ext/rt")])
            # Sionna is checked first; the files it owns are not re-attributed to Sionna RT.
            self.assertEqual([call.args for call in match.call_args_list],
                             [("sionna", root / "ext/phy", set()),
                              ("sionna-rt", root / "ext/rt", {"sionna/__init__.py", "sionna/phy.py"})])
            self.assertEqual(report["distributions"]["sionna-rt"],
                             {"version": "2.2.0", "commit": "b" * 40, "files_matching_source": 1})
            native.assert_called_once_with(root / "isac", "c" * 40)
            (root / "ext/lock.json").write_text(json.dumps({"dependencies": [{"path": "ext/ocudu"}]}))
            output.seek(0)
            output.truncate()
            self.assertEqual(environment.main(["--profile", "core", "--with-native"]), 1)
            self.assertIn("must name every dependency", json.loads(output.getvalue())["error"])

    def test_installed_files_must_equal_the_pinned_source_tree(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp).resolve()
            source, site = base / "checkout", base / "site-packages"
            files = {"pkg/__init__.py": "shared = 1\n", "pkg/solver.py": "paths = 2\n",
                     "pkg/data/scene.xml": "<scene/>\n"}
            for relative, text in files.items():
                for tree in (source / "src", site):
                    (tree / relative).parent.mkdir(parents=True, exist_ok=True)
                    (tree / relative).write_text(text)
            (base / "elsewhere").mkdir()
            (base / "elsewhere/linked.py").write_text("not part of this package\n")
            (source / "src/pkg/rt").symlink_to(base / "elsewhere")
            recorded = [*files, "pkg-1.0.dist-info/RECORD", "pkg/__pycache__/solver.cpython-312.pyc",
                        "../../bin/tool"]
            dist = mock.Mock(files=[environment.metadata.PackagePath(name) for name in recorded],
                             locate_file=lambda entry: site / entry)
            with mock.patch.object(environment.metadata, "distribution", return_value=dist):
                self.assertEqual(environment.matches_pinned_source("pkg", source), set(files))
                installed = site / "pkg/solver.py"
                installed.write_text("paths = 3  # built from another commit with the same version\n")
                with self.assertRaisesRegex(ValueError, "installed pkg/solver.py differs"):
                    environment.matches_pinned_source("pkg", source)
                self.assertEqual(environment.matches_pinned_source("pkg", source, {"pkg/solver.py"}),
                                 set(files) - {"pkg/solver.py"})
                installed.write_text(files["pkg/solver.py"])
                (source / "src/pkg/added.py").write_text("new upstream module\n")
                with self.assertRaisesRegex(ValueError, "pinned source file pkg/added.py is not installed"):
                    environment.matches_pinned_source("pkg", source)
                (source / "src/pkg/added.py").unlink()
                (source / "src/pkg/data/scene.xml").unlink()
                with self.assertRaisesRegex(ValueError, "installed pkg/data/scene.xml differs"):
                    environment.matches_pinned_source("pkg", source)
                dist.files = [environment.metadata.PackagePath("__editable__.pkg-1.0.pth")]
                with self.assertRaisesRegex(ValueError, "make env-full"):
                    environment.matches_pinned_source("pkg", source)
                dist.files = None
                with self.assertRaisesRegex(ValueError, "no installed source files"):
                    environment.matches_pinned_source("pkg", source)

    def test_loaded_backend_libraries_are_listed_once(self):
        maps = ("ffff80000000-ffff80200000 r-xp 00000000 103:02 11  /usr/lib/aarch64-linux-gnu/libc.so.6\n"
                "ffff81000000-ffff82000000 r-xp 00000000 103:02 12  /usr/lib/aarch64-linux-gnu/libLLVM.so.20.1\n"
                "ffff82000000-ffff83000000 r--p 00100000 103:02 12  /usr/lib/aarch64-linux-gnu/libLLVM.so.20.1\n"
                "ffff84000000-ffff85000000 r-xp 00000000 103:02 13  /usr/lib/aarch64-linux-gnu/libnvoptix.so.580.178.04\n"
                "ffff86000000-ffff87000000 r-xp 00000000 103:02 14  /usr/lib/aarch64-linux-gnu/libcuda.so.580.178.04\n"
                "ffff88000000-ffff89000000 rw-p 00000000 00:00 0\n"
                "ffff8a000000-ffff8b000000 rw-p 00000000 00:00 0  [heap]\n")
        with mock.patch.object(environment.Path, "read_text", return_value=maps):
            self.assertEqual(environment.loaded_libraries(),
                             ["/usr/lib/aarch64-linux-gnu/libLLVM.so.20.1",
                              "/usr/lib/aarch64-linux-gnu/libcuda.so.580.178.04",
                              "/usr/lib/aarch64-linux-gnu/libnvoptix.so.580.178.04"])
        with mock.patch.object(environment.Path, "read_text", side_effect=OSError("no procfs")):
            self.assertEqual(environment.loaded_libraries(), [])

    def test_report_records_the_host_and_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp, \
                mock.patch.object(environment, "run_checks") as checks, \
                contextlib.redirect_stdout(io.StringIO()):
            report = Path(temp) / "core-env.json"
            self.assertEqual(environment.main(["--profile", "core", "--report", str(report)]), 0)
            first = report.read_bytes()
            data = json.loads(first)
            self.assertEqual((data["status"], data["machine"], data["python"]),
                             ("pass", platform.machine(), platform.python_version()))
            self.assertEqual((data["node"], data["prefix"], data["base_prefix"], data["project_root"]),
                             (platform.node(), sys.prefix, sys.base_prefix, str(environment.ROOT)))
            self.assertEqual(set(data["os"]), {"id", "version_id", "pretty_name"})
            with contextlib.redirect_stderr(io.StringIO()) as stderr, \
                    self.assertRaises(SystemExit) as stop:
                environment.main(["--profile", "core", "--report", str(report)])
            self.assertEqual(stop.exception.code, 2)
            self.assertIn("already exists", stderr.getvalue())
            self.assertEqual(report.read_bytes(), first)
            self.assertEqual(checks.call_count, 1)

    def test_unknown_profile_and_misplaced_backend_fail_clearly(self):
        cases = ((["--profile", "gpu"], "invalid choice: 'gpu'"),
                 (["--profile", "core", "--backend", "cuda"], "--backend applies to --profile simulation"))
        for argv, text in cases:
            with self.subTest(argv=argv), mock.patch.object(environment, "run_checks") as checks, \
                    contextlib.redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as stop:
                    environment.main(argv)
                self.assertEqual(stop.exception.code, 2)
                self.assertIn(text, stderr.getvalue())
                checks.assert_not_called()


@unittest.skipUnless(shutil.which("bash") and sys.version_info >= (3, 11),
                     "the bootstrap needs Bash and a Python 3.11+ test interpreter")
class BootstrapScriptTests(unittest.TestCase):
    """Run the real bootstrap in a scratch checkout. Every case stops before pip installs."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.scratch = Path(temporary.name).resolve()
        self.root = self.scratch / "fresh checkout"
        (self.root / "scripts").mkdir(parents=True)
        for name in ("bootstrap_env.sh", "verify_environment.py"):
            shutil.copy(SCRIPTS / name, self.root / "scripts" / name)

    def bootstrap(self, *arguments, python=None, pythonpath=None):
        env = {key: value for key, value in os.environ.items()
               if key not in ("PYTHON", "PYTHONPATH", "VIRTUAL_ENV")}
        if python:
            env["PYTHON"] = str(python)
        if pythonpath:
            env["PYTHONPATH"] = str(pythonpath)
        return subprocess.run(["bash", str(self.root / "scripts/bootstrap_env.sh"), *arguments],
                              env=env, capture_output=True, text=True, timeout=120, check=False)

    def interpreter(self, body):
        double = self.scratch / "python-double"
        double.write_text("#!/bin/sh\n" + body)
        double.chmod(0o755)
        return double

    def shimmed_interpreter(self, module, source):
        """The running interpreter with one module replaced ahead of the standard library."""
        shim = self.scratch / ("shim-" + Path(module).stem)
        shim.mkdir()
        (shim / module).write_text(source)
        return self.interpreter(f'PYTHONPATH={shlex.quote(str(shim))} '
                                f'exec {shlex.quote(sys.executable)} "$@"\n')

    def real_environment(self):
        """A standard-library venv whose pip double answers --version and refuses anything else."""
        venv.EnvBuilder(with_pip=False, prompt="isac").create(self.root / "isac")
        package = next((self.root / "isac/lib").glob("python*/site-packages")) / "pip"
        package.mkdir()
        (package / "__init__.py").write_text("")
        (package / "__main__.py").write_text(
            'import sys\n'
            'if sys.argv[1:] != ["--version"]:\n'
            '    sys.exit("pip double: unexpected " + " ".join(sys.argv[1:]))\n'
            'print("pip 0 (test double)")\n')

    def test_unknown_profile_is_refused_before_anything_is_created(self):
        result = self.bootstrap("gpu")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Usage", result.stderr)
        self.assertFalse((self.root / "isac").exists())

    def test_directory_that_is_not_an_environment_is_preserved(self):
        keep = self.root / "isac/notes.txt"
        keep.parent.mkdir()
        keep.write_text("local work")
        result = self.bootstrap("core")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Refusing to reuse", result.stderr)
        self.assertEqual(keep.read_text(), "local work")

    def test_old_interpreter_and_missing_ensurepip_are_explained(self):
        cases = (("sitecustomize.py", "import sys\nsys.version_info = (3, 10, 12)\n", "Python 3.11+"),
                 ("ensurepip.py", "raise ImportError('ensurepip double')\n", "python3-venv"),
                 (None, None, "not found"))
        for module, source, expected in cases:
            python = self.shimmed_interpreter(module, source) if module else self.scratch / "absent"
            result = self.bootstrap("core", python=python)
            with self.subTest(expected=expected):
                self.assertEqual(result.returncode, 1)
                self.assertIn(expected, result.stderr)
                self.assertFalse((self.root / "isac").exists())

    def test_creation_that_fails_half_way_leaves_no_partial_environment(self):
        python = self.interpreter(
            '[ "$1" = "-c" ] && exit 0\n'
            'for target; do :; done\n'
            'mkdir -p "$target/bin" && echo "home = /usr/bin" > "$target/pyvenv.cfg"\n'
            'echo "ensurepip is not available" >&2\n'
            'exit 1\n')
        result = self.bootstrap("core", python=python)
        self.assertEqual(result.returncode, 1)
        self.assertIn("ensurepip is not available", result.stderr)
        self.assertIn("partial directory was removed", result.stderr)
        self.assertFalse((self.root / "isac").exists())

    def test_incomplete_environment_is_reported_and_left_in_place(self):
        config = self.root / "isac/pyvenv.cfg"
        config.parent.mkdir()
        config.write_text("include-system-site-packages = false\n")
        result = self.bootstrap("core")
        self.assertEqual(result.returncode, 1)
        self.assertIn("incomplete", result.stderr)
        self.assertIn("rm -rf isac", result.stderr)
        self.assertTrue(config.exists())

    def test_healthy_environment_reaches_the_editable_install(self):
        self.real_environment()
        result = self.bootstrap("core")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("pip double: unexpected install -e .", result.stderr)

    def test_full_profile_installs_the_source_directories_named_by_the_lock(self):
        self.real_environment()
        double = ('import sys\n'
                  'paths = {"sionna": "/locked/custom-sionna", "sionna-rt": PATH}\n'
                  'if sys.argv[1:] == ["--worktrees"]:\n'
                  '    sys.exit(0)\n'
                  'if len(sys.argv) == 3 and sys.argv[1] == "--path" and paths.get(sys.argv[2]):\n'
                  '    print(paths[sys.argv[2]])\n'
                  '    sys.exit(0)\n'
                  'sys.exit("check_pins double: refused " + " ".join(sys.argv[1:]))\n')
        pins = self.root / "scripts/check_pins.py"
        pins.write_text(double.replace("PATH", '"/locked/custom-rt"'))
        result = self.bootstrap("full")
        self.assertIn("pip double: unexpected install -e . /locked/custom-sionna /locked/custom-rt",
                      result.stderr)
        pins.write_text(double.replace("PATH", "None"))
        result = self.bootstrap("full")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("check_pins double: refused --path sionna-rt", result.stderr)
        self.assertNotIn("pip double", result.stderr)

    def test_global_packages_on_the_search_path_stop_the_install(self):
        self.real_environment()
        leak = self.scratch / "usr/lib/python3/dist-packages"
        leak.mkdir(parents=True)
        result = self.bootstrap("core", pythonpath=leak)
        self.assertEqual(result.returncode, 1)
        self.assertIn(str(leak), result.stderr)
        self.assertIn("Unset PYTHONPATH", result.stderr)
        self.assertNotIn("pip double", result.stderr)

    def test_environment_moved_with_its_checkout_is_refused(self):
        self.real_environment()
        self.root = self.root.rename(self.scratch / "renamed checkout")
        result = self.bootstrap("core")
        self.assertEqual(result.returncode, 1)
        self.assertIn("cannot be relocated", result.stderr)
        self.assertNotIn("pip double", result.stderr)
        self.assertTrue((self.root / "isac/pyvenv.cfg").exists())


if __name__ == "__main__":
    unittest.main()
