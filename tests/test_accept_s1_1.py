"""Check the Sprint 1.1 gate runner against tool doubles. No compiler, source or GPU is needed."""

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/accept_s1_1.sh"
GRADE = """PASS  S1.1 full Release build completed (target all)
PASS  S1.1 CUDA compilation: 18 kernel sources of 30 CUDA sources for sm_121 by nvcc 13.5.37
NOTE  recorded difference, CUDA toolkit: 13.5.37 here; 13.0.88 in the branch notes"""
# The interpreter double answers each gate from FAKE_* variables and logs the call.
PYTHON_DOUBLE = """#!/bin/sh
[ "$1" = "-c" ] && exec {real} "$@"
echo "$*" >> "$FAKE_LOG"
case "$*" in
    "-m pip freeze --all") echo "example==1.0" ;;
    *"-m unittest discover"*) printf '%s\\n' "$FAKE_UNIT" >&2; exit "${{FAKE_UNIT_STATUS:-0}}" ;;
    *"native_ocudu.py build --run-dir "*)
        [ -z "${{FAKE_LEAVE:-}}" ] || echo "written by the build" > "$FAKE_SOURCE/$FAKE_LEAVE"
        echo "Run artifacts: $4"
        echo "-- The CXX compiler identification is GNU 13.3.0"
        if [ -z "${{FAKE_NO_EVIDENCE:-}}" ]; then
            mkdir -p "$4"
            if [ "${{FAKE_BUILD:-0}}" != 0 ]; then
                echo "CMake Error at CMakeLists.txt:468 (find_package):" >&2
                printf 'stage=configure\nexit_code=1\n' > "$4/status.txt"
                printf '%s\n' "cmake exited with status 1 (see configure.log)" \
                    "  CMake Error at CMakeLists.txt:468 (find_package):" "    Could NOT find GTest" \
                    "Install with: sudo apt-get install libgtest-dev" > "$4/error.txt"
            fi
        fi
        exit "${{FAKE_BUILD:-0}}" ;;
    *"native_ocudu.py grade "*) printf '%s\\n' "$FAKE_GRADE"; exit "${{FAKE_GRADE_STATUS:-0}}" ;;
    *"--profile core --with-native"*)
        [ "${{FAKE_NATIVE:-0}}" = 0 ] || echo "ValueError: Native binaries differ from the locked full-build receipt" >&2
        exit "${{FAKE_NATIVE:-0}}" ;;
    *"--backend cpu --with-native"*) exit "${{FAKE_CPU:-0}}" ;;
    *"--backend cuda --with-native"*) exit "${{FAKE_CUDA:-0}}" ;;
    *"check_pins.py --worktrees --only ocudu") echo "ext/ocudu 27e84cb3 (Git worktree)"; exit "${{FAKE_PINS:-0}}" ;;
    *"check_pins.py --path ocudu") echo "$FAKE_SOURCE"; exit "${{FAKE_PATH:-0}}" ;;
    *) echo "interpreter double: unexpected $*" >&2; exit 64 ;;
esac
"""
GNB_DOUBLE = """#!/bin/sh
[ "$1" = "--version" ] || exit 64
printf '\\n--== OCUDU gNB (commit 27e84cb3da) ==--\\n\\n%s\\n' "${FAKE_BANNER-OCUDU 5G gNB version 26.04.0 (27e84cb3da)}"
exit "${FAKE_GNB:-0}"
"""
PUSCH_DOUBLE = """#!/bin/sh
[ "$1" = "--help" ] || exit 64
echo "  --rx-device-grid MODE  CUDA-visible RX grid mode"
echo "  --gpu-output-buffer MODE  GPU output payload buffer"
[ -n "${FAKE_NO_RESIDENT:-}" ] || echo "  --require-gpu-resident  Fail unless resident"
exit "${FAKE_HELP:-0}"
"""


def host_path():
    """PATH without any directory that already provides a native tool.

    After a real build the checkout's isac/bin holds gnb and the PUSCH test, and an activated
    shell or the gate itself puts that directory on PATH. The fixtures must not find them there.
    """
    tools = ("gnb", "pusch_e2e_pipeline_test")
    entries = os.environ.get("PATH", os.defpath).split(os.pathsep)
    return os.pathsep.join(entry for entry in entries
                           if entry and not any((Path(entry) / tool).exists() for tool in tools))


@unittest.skipUnless(shutil.which("bash"), "the gate runner is a Bash script")
class SprintOneBuildGateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / "host checkout"
        self.bin = self.root / "isac/bin"
        (self.root / "scripts").mkdir(parents=True)
        self.bin.mkdir(parents=True)
        (self.root / "ext/ocudu").mkdir(parents=True)
        shutil.copy(SCRIPT, self.root / "scripts/accept_s1_1.sh")
        self.write(self.bin / "python", PYTHON_DOUBLE.format(real=shlex.quote(sys.executable)))
        for name, text in (("apps/gnb/gnb", GNB_DOUBLE), ("tests/pusch_e2e_pipeline_test", PUSCH_DOUBLE)):
            target = self.root / "build/ocudu-spark" / name
            target.parent.mkdir(parents=True)
            self.write(target, text)
            (self.bin / target.name).symlink_to(target)
        self.calls = self.root.parent / "calls.log"
        self.run_dir = self.root.parent / "evidence"

    @staticmethod
    def write(path, text):
        path.write_text(text)
        path.chmod(0o755)

    def gate(self, *arguments, run_dir=None, search_path=None, **settings):
        env = {key: value for key, value in os.environ.items() if not key.startswith("FAKE_")}
        env.update(FAKE_LOG=str(self.calls), FAKE_UNIT="Ran 96 tests in 1.000s\n\nOK", FAKE_GRADE=GRADE,
                   FAKE_SOURCE=str(self.root / "ext/ocudu"), PATH=search_path or host_path())
        env.update({f"FAKE_{key.upper()}": str(value) for key, value in settings.items()})
        command = [shutil.which("bash"), str(self.root / "scripts/accept_s1_1.sh"),
                   "--run-dir", str(run_dir or self.run_dir), *arguments]
        return subprocess.run(command, env=env, capture_output=True, text=True, timeout=120, check=False)

    def test_all_gates_pass_on_a_spark_and_leave_evidence(self):
        result = self.gate()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("FAIL", result.stdout)
        self.assertIn("Sprint 1.1: ACCEPTED on this DGX Spark. Next gate: make accept-s1.2 (S1.2).", result.stdout)
        summary = (self.run_dir / "summary.txt").read_text()
        shown = result.stdout.splitlines()
        self.assertEqual([line for line in shown if line in summary.splitlines()], summary.splitlines())
        # Everything the builder and CMake print is shown while it runs and kept in its own log.
        live = [f"Run artifacts: {self.run_dir / 'native-build'}", "-- The CXX compiler identification is GNU 13.3.0"]
        self.assertEqual([line for line in shown if line not in summary.splitlines()], live)
        self.assertEqual((self.run_dir / "native-build.log").read_text().splitlines(), live)
        self.assertLess(shown.index("Building the locked OCUDU Release. CMake output follows. A first build "
                                    "takes a while."), shown.index(live[0]))
        self.assertLess(shown.index(live[1]), shown.index(
            "PASS  S1.1 native build command (native-build.log, evidence in native-build/)"))
        for line in ("spark profile", "PASS  unit tests (96 ran, 0 skipped)",
                     "PASS  S1.1 native build command (native-build.log, evidence in native-build/)",
                     *GRADE.splitlines(),
                     "PASS  S1.1 gnb and pusch_e2e_pipeline_test resolve inside isac/bin (native-tools.txt)",
                     "PASS  S1.1 gnb --version from the environment: OCUDU 5G gNB version 26.04.0 (27e84cb3da)",
                     "PASS  S1.1 pusch_e2e_pipeline_test --help lists the resident, device-grid and output-buffer options",
                     "PASS  S1.1 native environment report (native-env.json)",
                     "PASS  S1.1 OCUDU source at the locked revision after the build (sources-after.txt)"):
            self.assertIn(line, result.stdout)
        for name in ("python-packages.txt", "unit-tests.log", "native-build.log", "native-grade.txt",
                     "native-tools.txt", "gnb-version.txt", "pusch-help.txt", "native-env.log", "sources-after.txt"):
            self.assertTrue((self.run_dir / name).is_file(), name)
        self.assertTrue((self.run_dir / "native-build").is_dir())
        tools = (self.run_dir / "native-tools.txt").read_text().splitlines()
        self.assertEqual(tools[0], f"gnb -> {self.bin / 'gnb'} -> {self.root / 'build/ocudu-spark/apps/gnb/gnb'}")
        calls = self.calls.read_text().splitlines()
        self.assertIn(f"scripts/native_ocudu.py build --run-dir {self.run_dir / 'native-build'} --machine spark", calls)
        self.assertIn(f"scripts/native_ocudu.py grade {self.run_dir / 'native-build'}", calls)
        self.assertIn(f"scripts/verify_environment.py --profile core --with-native --report "
                      f"{self.run_dir / 'native-env.json'}", calls)
        self.assertFalse([call for call in calls if "--profile simulation" in call])
        # The source is located first. The build precedes its grading, and the pin is checked again
        # only after everything ran against the source.
        position = [next(index for index, call in enumerate(calls) if step in call)
                    for step in ("check_pins.py --path ocudu", "native_ocudu.py build", "native_ocudu.py grade",
                                 "--with-native", "check_pins.py --worktrees --only ocudu")]
        self.assertEqual(position, sorted(position))
        # An archive checkout has no Git metadata: the pin check verifies its files instead.
        self.assertNotIn("OCUDU source is pristine", result.stdout)

    def test_build_options_backends_and_the_local_profile_are_passed_through(self):
        result = self.gate("--jobs", "4", "--build-dir", "build/second tree", "cpu", "cuda")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.calls.read_text()
        self.assertIn("--machine spark --jobs 4 --build-dir build/second tree\n", calls)
        for backend in ("cpu", "cuda"):
            self.assertIn(f"--profile simulation --backend {backend} --with-native --report "
                          f"{self.run_dir / f'collocated-{backend}.json'}", calls)
            self.assertIn(f"PASS  S1.1 simulation on {backend} and native from one interpreter", result.stdout)
        result = self.gate("--machine", "local", "--mcpu", "neoverse-v2", run_dir=self.root.parent / "local")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--machine local --mcpu neoverse-v2\n", self.calls.read_text())
        self.assertIn("local profile", result.stdout.splitlines()[0])
        self.assertIn("ACCEPTED on this host for the local profile. The DeepSig Table II row stays assigned to a GB10.",
                      result.stdout)
        self.assertNotIn("DGX Spark", result.stdout)

    def test_any_failed_gate_rejects_the_host_and_the_rest_still_run(self):
        failing_grade = GRADE.replace("PASS  S1.1 CUDA", "FAIL  S1.1 CUDA")
        cases = (
            ({"grade": failing_grade, "grade_status": 1}, "FAIL  S1.1 CUDA compilation: 18 kernel sources"),
            ({"grade": "Traceback (most recent call last):", "grade_status": 1},
             "FAIL  S1.1 build evidence could not be graded (see native-grade.txt)"),
            ({"grade": ""}, "FAIL  S1.1 build evidence produced no graded checks (see native-grade.txt)"),
            ({"unit_status": 1, "unit": "Ran 96 tests in 1.000s\n\nFAILED (failures=1)"}, "FAIL  unit tests (96 ran"),
            ({"unit": "Ran 0 tests in 0.000s\n\nOK"}, "FAIL  unit tests (none ran)"),
            ({"unit": "no tests were collected"}, "FAIL  unit tests"),
            ({"gnb": 1}, "FAIL  S1.1 gnb --version from the environment (see gnb-version.txt)"),
            ({"banner": "srsRAN 5G gNB version 24.10"}, "FAIL  S1.1 gnb --version from the environment"),
            ({"help": 1}, "FAIL  S1.1 pusch_e2e_pipeline_test --help (see pusch-help.txt)"),
            ({"no_resident": 1}, "FAIL  S1.1 pusch_e2e_pipeline_test --help (see pusch-help.txt)"),
            ({"native": 1}, "FAIL  S1.1 native environment report (native-env.json) (see native-env.log)\n"
                            "      ValueError: Native binaries differ from the locked full-build receipt\n"),
            ({"unit_status": 1, "unit": "FAIL: test_replay (tests.ReplayTest.test_replay)\nRan 96 tests in 1s\n\n"
                                        "FAILED (failures=1)"},
             "FAIL  unit tests (96 ran; see unit-tests.log)\n      FAIL: test_replay (tests.ReplayTest.test_replay)\n"),
            ({"grade": "Traceback (most recent call last):\nKeyError: 'toolchain'", "grade_status": 1},
             "FAIL  S1.1 build evidence could not be graded (see native-grade.txt)\n"
             "      Traceback (most recent call last):\n      KeyError: 'toolchain'\n"),
            ({"banner": "srsRAN 5G gNB version 24.10"},
             "FAIL  S1.1 gnb --version from the environment (see gnb-version.txt)\n"
             "      --== OCUDU gNB (commit 27e84cb3da) ==--\n      srsRAN 5G gNB version 24.10\n"),
            ({"cuda": 1}, "FAIL  S1.1 simulation on cuda and native from one interpreter (collocated-cuda.json)"),
            ({"pins": 1}, "FAIL  S1.1 OCUDU source at the locked revision after the build"),
            ({"path": 1, "source": ""}, "FAIL  S1.1 OCUDU source directory could not be resolved"),
        )
        for index, (settings, message) in enumerate(cases):
            self.calls.write_text("")
            result = self.gate("cuda", run_dir=self.root.parent / f"evidence {index}", **settings)
            with self.subTest(settings=settings):
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn(message, result.stdout)
                self.assertIn("Sprint 1.1: NOT ACCEPTED on this host", result.stdout)
                self.assertNotIn("Sprint 1.1: ACCEPTED", result.stdout)
                calls = self.calls.read_text()
                for later in ("native_ocudu.py grade", "--profile core --with-native", "check_pins.py --worktrees"):
                    self.assertIn(later, calls)

    def test_failed_build_shows_the_tool_error_and_skips_the_gates_that_need_a_build(self):
        result = self.gate("cuda", build=1)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        shown = result.stdout.splitlines()
        # The failing tool's own message is on the screen, where the builder printed it.
        self.assertIn("CMake Error at CMakeLists.txt:468 (find_package):", shown)
        failure = ("FAIL  S1.1 native build command stopped in stage configure: "
                   "cmake exited with status 1 (see configure.log)")
        skip = ("SKIP  S1.1 evidence grade, isac/bin links, gnb --version, pusch_e2e_pipeline_test --help "
                "and environment reports: no completed build")
        self.assertLess(shown.index("CMake Error at CMakeLists.txt:468 (find_package):"), shown.index(failure))
        self.assertLess(shown.index(failure), shown.index(skip))
        self.assertIn("Sprint 1.1: NOT ACCEPTED on this host", shown)
        # The summary holds the failure, the tool's words and the remedy, so it explains itself.
        summary = (self.run_dir / "summary.txt").read_text().splitlines()
        position = summary.index(failure)
        self.assertEqual(summary[position + 1:position + 4],
                         ["        CMake Error at CMakeLists.txt:468 (find_package):", "          Could NOT find GTest",
                          "      Install with: sudo apt-get install libgtest-dev"])
        self.assertIn("CMake Error at CMakeLists.txt:468", (self.run_dir / "native-build.log").read_text())
        # One failure is reported once: nothing that needs the build ran, the source checks still did.
        self.assertEqual([line for line in shown if line.startswith("FAIL")], [failure])
        calls = self.calls.read_text()
        for skipped in ("native_ocudu.py grade", "--with-native"):
            self.assertNotIn(skipped, calls)
        self.assertIn("check_pins.py --worktrees --only ocudu", calls)
        self.assertIn("PASS  S1.1 OCUDU source at the locked revision after the build (sources-after.txt)", shown)
        for name in ("native-grade.txt", "native-tools.txt", "gnb-version.txt", "pusch-help.txt", "native-env.log"):
            self.assertFalse((self.run_dir / name).exists(), name)
        # A builder that dies before it writes anything is still one clear failure.
        result = self.gate(run_dir=self.root.parent / "silent", build=2, no_evidence=1)
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S1.1 native build command stopped in stage unknown: no evidence written", result.stdout)
        self.assertIn(skip, result.stdout)

    def test_native_tools_must_be_the_links_published_inside_the_environment(self):
        gnb = self.bin / "gnb"
        target = gnb.resolve()
        gnb.unlink()
        result = self.gate(run_dir=self.root.parent / "absent")
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S1.1 gnb and pusch_e2e_pipeline_test resolve inside isac/bin", result.stdout)
        self.assertIn("FAIL  S1.1 gnb --version from the environment", result.stdout)
        # A system-wide gnb earlier on PATH or a copied file cannot stand in for the published link.
        elsewhere = self.root.parent / "system/bin"
        elsewhere.mkdir(parents=True)
        shutil.copy(target, elsewhere / "gnb")
        result = self.gate(run_dir=self.root.parent / "system-gnb",
                           search_path=f"{elsewhere}{os.pathsep}{host_path()}")
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S1.1 gnb and pusch_e2e_pipeline_test resolve inside isac/bin", result.stdout)
        self.assertIn(f"gnb -> {elsewhere / 'gnb'}", (self.root.parent / "system-gnb/native-tools.txt").read_text())
        shutil.copy(target, gnb)
        result = self.gate(run_dir=self.root.parent / "copied")
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  S1.1 gnb and pusch_e2e_pipeline_test resolve inside isac/bin", result.stdout)

    def test_arguments_and_existing_evidence_are_refused_before_any_gate_runs(self):
        self.run_dir.mkdir()
        keep = self.run_dir / "summary.txt"
        keep.write_text("earlier run\n")
        new = {"run_dir": self.root.parent / "new"}
        refusals = (((), {}, "already exists"), (("gpu",), new, "Usage"), (("--run-dir",), new, "Usage"),
                    (("--jobs",), new, "Usage"), (("--machine", "gb300"), new, "Usage"), (("--machine",), new, "Usage"),
                    (("--skip-build",), new, "Usage"))
        for arguments, options, message in refusals:
            result = self.gate(*arguments, **options)
            with self.subTest(arguments=arguments):
                self.assertEqual(result.returncode, 2)
                self.assertIn(message, result.stderr)
        self.assertEqual(keep.read_text(), "earlier run\n")
        self.assertFalse((self.root.parent / "new").exists())
        self.assertFalse(self.calls.exists())
        self.assertEqual(self.gate("--help").returncode, 0)
        (self.bin / "python").unlink()
        result = self.gate(run_dir=self.root.parent / "new")
        self.assertEqual(result.returncode, 2)
        self.assertIn("make env", result.stderr)

    @unittest.skipUnless(shutil.which("git"), "the committed-tree and source checks read Git metadata")
    def test_evidence_needs_a_committed_tree_and_an_untouched_source(self):
        identity = {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
                    "GIT_AUTHOR_NAME": "fixture", "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
                    "GIT_COMMITTER_NAME": "fixture", "GIT_COMMITTER_EMAIL": "fixture@example.invalid"}

        def git(*arguments, cwd):
            subprocess.run(["git", *arguments], cwd=cwd, env={**os.environ, **identity},
                           capture_output=True, text=True, check=True)

        source = self.root.parent / "upstream source"
        source.mkdir()
        (source / "CMakeLists.txt").write_text("project(ocudu)\n")
        for repository in (source, self.root):
            git("init", "-q", cwd=repository)
            git("add", "-A", cwd=repository)
            git("commit", "-q", "-m", "fixture checkout", cwd=repository)
        before = "S1.1 OCUDU source is pristine before the build"
        after = "S1.1 out-of-tree build left no file in the OCUDU source"
        result = self.gate(run_dir=self.root.parent / "clean", source=source)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS  committed tree", result.stdout)
        self.assertRegex(result.stdout, r"commit [0-9a-f]{40}")
        self.assertIn(f"PASS  {before}", result.stdout)
        self.assertIn(f"PASS  {after}", result.stdout)
        lines = result.stdout.splitlines()
        position = [next(index for index, line in enumerate(lines) if text in line)
                    for text in (before, "Building the locked OCUDU Release", after)]
        self.assertEqual(position, sorted(position))
        # A file the build writes into the source fails only the gate that follows the build.
        result = self.gate(run_dir=self.root.parent / "written", source=source, leave="hashes.h")
        self.assertEqual(result.returncode, 1)
        self.assertIn(f"PASS  {before}", result.stdout)
        self.assertIn(f"FAIL  {after} (compare source-before.txt with source-after.txt)", result.stdout)
        self.assertEqual((self.root.parent / "written/source-before.txt").read_text(), "")
        self.assertIn("hashes.h", (self.root.parent / "written/source-after.txt").read_text())
        (source / "hashes.h").unlink()
        # A build directory made by hand inside the source is reported before the long build starts.
        (source / "build").mkdir()
        (source / "build/CMakeCache.txt").write_text("in-tree build\n")
        result = self.gate(run_dir=self.root.parent / "in-tree", source=source)
        self.assertEqual(result.returncode, 1)
        self.assertIn(f"FAIL  {before} (see source-before.txt)", result.stdout)
        self.assertIn(f"build directory made by hand inside {source}", result.stdout)
        self.assertIn(f"PASS  {after}", result.stdout)
        self.assertIn("build/CMakeCache.txt", (self.root.parent / "in-tree/source-before.txt").read_text())
        shutil.rmtree(source / "build")
        (self.root / "scripts/accept_s1_1.sh").write_text(SCRIPT.read_text() + "# local edit\n")
        result = self.gate(run_dir=self.root.parent / "dirty", source=source)
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL  committed tree", result.stdout)
        self.assertIn("accept_s1_1.sh", (self.root.parent / "dirty/git-status.txt").read_text())


if __name__ == "__main__":
    unittest.main()
