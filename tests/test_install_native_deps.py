"""Check the native prerequisite installer against package-tool doubles. Nothing real is installed.

The script runs with a PATH that holds only the doubles, so it cannot reach the real sudo, apt-get
or dpkg of the host that runs this test.
"""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/install_native_deps.sh"
PACKAGES = ["libfftw3-dev", "libmbedtls-dev", "libsctp-dev", "libyaml-cpp-dev", "libgtest-dev"]
# dpkg-query -W -f='${Status}' PACKAGE: answer from the list of installed packages in FAKE_STATE.
DPKG_QUERY = """#!/bin/sh
for package in "$@"; do :; done
while read -r name; do
    if [ "$name" = "$package" ]; then printf 'install ok installed'; exit 0; fi
done < "$FAKE_STATE"
echo "dpkg-query: no packages found matching $package" >&2
exit 1
"""
# apt-get: log the call. 'install' adds the named packages to FAKE_STATE unless told to fail or skip.
APT_GET = """#!/bin/sh
echo "apt-get $* frontend=${DEBIAN_FRONTEND:-unset}" >> "$FAKE_LOG"
if [ "$1" = "${FAKE_FAIL:-none}" ]; then echo "E: Unable to locate package"; exit 100; fi
[ "$1" = install ] || exit 0
shift
for word in "$@"; do
    case "$word" in
        -*) ;;
        *) [ -n "${FAKE_SKIP:-}" ] || echo "$word" >> "$FAKE_STATE" ;;
    esac
done
"""
# sudo [VAR=value]... COMMAND: log the call and run the command with those variables.
SUDO = """#!/bin/sh
echo "sudo $*" >> "$FAKE_LOG"
while case "${1:-}" in *=*) true ;; *) false ;; esac; do export "$1"; shift; done
exec "$@"
"""
ID = """#!/bin/sh
echo "${FAKE_UID:-1000}"
"""


@unittest.skipUnless(shutil.which("bash"), "the installer is a Bash script")
class NativePrerequisiteInstallerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.bin = self.root / "bin"
        self.bin.mkdir()
        for name, text in (("dpkg-query", DPKG_QUERY), ("apt-get", APT_GET), ("sudo", SUDO), ("id", ID)):
            (self.bin / name).write_text(text)
            (self.bin / name).chmod(0o755)
        self.state = self.root / "installed.txt"
        self.log = self.root / "calls.log"
        self.state.write_text("libfftw3-dev\n")
        self.log.write_text("")

    def install(self, *arguments, **settings):
        env = {"PATH": str(self.bin), "FAKE_STATE": str(self.state), "FAKE_LOG": str(self.log)}
        env.update({f"FAKE_{key.upper()}": str(value) for key, value in settings.items()})
        return subprocess.run([shutil.which("bash"), str(SCRIPT), *arguments], env=env, capture_output=True,
                              text=True, timeout=60, check=False)

    def calls(self):
        return self.log.read_text().splitlines()

    def test_installs_only_what_is_missing_and_a_second_run_changes_nothing(self):
        result = self.install()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        missing = "libmbedtls-dev libsctp-dev libyaml-cpp-dev libgtest-dev"
        self.assertIn(f"native prerequisites: installing {missing} with apt-get", result.stdout)
        self.assertEqual(result.stdout.splitlines()[-1], "native prerequisites: installed")
        self.assertEqual(self.calls(), [
            "sudo DEBIAN_FRONTEND=noninteractive apt-get update",
            "apt-get update frontend=noninteractive",
            f"sudo DEBIAN_FRONTEND=noninteractive apt-get install -y {missing}",
            f"apt-get install -y {missing} frontend=noninteractive"])
        self.assertEqual(sorted(self.state.read_text().split()), sorted(PACKAGES))
        # Run it again: every package is present, so no package tool is called and sudo is not asked.
        result = self.install()
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, f"native prerequisites: all 5 packages are installed ({' '.join(PACKAGES)})\n")
        self.assertEqual(len(self.calls()), 4)

    def test_root_installs_without_sudo(self):
        result = self.install(uid=0)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual([call.split()[0] for call in self.calls()], ["apt-get", "apt-get"])
        self.assertIn("frontend=noninteractive", self.calls()[1])

    def test_check_reports_without_installing(self):
        result = self.install("--check")
        self.assertEqual(result.returncode, 1)
        self.assertIn("missing libmbedtls-dev libsctp-dev libyaml-cpp-dev libgtest-dev. Install with: "
                      "bash scripts/install_native_deps.sh", result.stdout)
        self.assertEqual(self.calls(), [])
        self.state.write_text("\n".join(PACKAGES) + "\n")
        self.assertEqual(self.install("--check").returncode, 0)
        self.assertEqual(self.install("--force").returncode, 2)
        self.assertEqual(self.calls(), [])

    def test_failed_or_incomplete_install_is_an_error_with_the_manual_command(self):
        manual = ("sudo apt-get update && sudo apt-get install libmbedtls-dev libsctp-dev libyaml-cpp-dev "
                  "libgtest-dev")
        for settings, message in (({"fail": "update"}, "apt-get failed"), ({"fail": "install"}, "apt-get failed"),
                                  ({"skip": 1}, "still missing after apt-get: libmbedtls-dev")):
            result = self.install(**settings)
            with self.subTest(settings=settings):
                self.assertEqual(result.returncode, 1)
                self.assertIn(message, result.stderr)
                self.assertIn(manual, result.stderr)
                self.assertEqual(self.state.read_text(), "libfftw3-dev\n")
        (self.bin / "sudo").unlink()
        result = self.install()
        self.assertEqual(result.returncode, 1)
        self.assertIn("sudo is missing. As root run: apt-get install libmbedtls-dev", result.stderr)

    def test_host_without_dpkg_is_left_to_cmake(self):
        (self.bin / "dpkg-query").unlink()
        result = self.install()
        self.assertEqual(result.returncode, 0)
        self.assertIn("no dpkg and apt-get on this host. Install the equivalents of: " + " ".join(PACKAGES),
                      result.stdout)
        self.assertEqual(self.calls(), [])


if __name__ == "__main__":
    unittest.main()
