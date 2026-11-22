"""Check the reader of the receiver patch series and the patched tree it prepares. No OCUDU source is needed.

Each test makes a miniature project: a lock, a locked source of three files and a series of one
patch, written with Git from that source. The locked source is a Git checkout in one half of the
tests and a plain directory, as a verified archive leaves it, in the other.
"""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import receiver_patch as rp  # noqa: E402

GIT = ["git", "-c", "user.name=Unit", "-c", "user.email=unit@example.org", "-c", "commit.gpgsign=false"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@unittest.skipUnless(shutil.which("git"), "the patched tree is made with Git")
class ReceiverPatchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / "project"
        self.source = self.root / "ext/receiver"
        (self.source / "lib").mkdir(parents=True)
        (self.source / "lib/demod.cpp").write_text("int demodulate()\n{\n  return 0;\n}\n")
        (self.source / "lib/other.cpp").write_text("int other() { return 1; }\n")
        (self.source / "README").write_text("pinned receiver\n")
        subprocess.run(["git", "init", "-q", str(self.source)], check=True)
        subprocess.run([*GIT, "-C", str(self.source), "add", "-A"], check=True)
        subprocess.run([*GIT, "-C", str(self.source), "commit", "-q", "-m", "pin"], check=True)
        self.pin = subprocess.check_output(["git", "-C", str(self.source), "rev-parse", "HEAD"], text=True).strip()
        before = sha(self.source / "lib/demod.cpp")
        # The patch: one changed file and one new header, written from a scratch copy of the pin.
        (self.source / "lib/demod.cpp").write_text("#include \"tap.h\"\nint demodulate()\n{\n  tap();\n  return 0;\n}\n")
        (self.source / "lib/tap.h").write_text("inline void tap() {}\n")
        subprocess.run([*GIT, "-C", str(self.source), "add", "-N", "lib/tap.h"], check=True)
        patch = subprocess.check_output(["git", "-C", str(self.source), "diff"], text=True)
        after = {"lib/demod.cpp": sha(self.source / "lib/demod.cpp"), "lib/tap.h": sha(self.source / "lib/tap.h")}
        subprocess.run(["git", "-C", str(self.source), "reset", "-q", "--hard"], check=True)
        subprocess.run(["git", "-C", str(self.source), "clean", "-fdq"], check=True)
        folder = self.root / "patches/ocudu"
        folder.mkdir(parents=True)
        (folder / "0001-tap.patch").write_text(patch)
        self.manifest = {"schema": rp.SCHEMA, "dependency": "ocudu", "base_commit": self.pin, "base_ref": "main",
                         "patches": [{"file": "0001-tap.patch", "sha256": sha(folder / "0001-tap.patch"), "sprint": "5.2",
                                      "selector": "native_estimate", "interface": {"header": "lib/tap.h", "version": 1},
                                      "files": [{"path": "lib/tap.h", "change": "added", "sha256_before": None,
                                                 "sha256_after": after["lib/tap.h"]},
                                                {"path": "lib/demod.cpp", "change": "modified", "sha256_before": before,
                                                 "sha256_after": after["lib/demod.cpp"]}]}]}
        self.write_series()
        (self.root / "ext/lock.json").write_text(json.dumps({"dependencies": [
            {"name": "ocudu", "path": "ext/receiver", "commit": self.pin}]}))
        self.tree = self.root / "build/src-o3"

    def write_series(self):
        (self.root / rp.SERIES).write_text(json.dumps(self.manifest, indent=2) + "\n")

    def as_archive(self):
        shutil.rmtree(self.source / ".git")

    def status(self):
        return subprocess.check_output(["git", "-C", str(self.source), "status", "--porcelain"], text=True)

    def test_the_series_of_this_project_is_valid_and_names_one_small_patch(self):
        series = rp.load_series()
        record = rp.identity(series)
        self.assertEqual(record["schema"], "receiver-patch/v1")
        self.assertEqual([item["file"] for item in record["patches"]], ["0001-pusch-resident-estimate-tap.patch"])
        self.assertEqual(sorted(record["files"]), [
            "include/ocudu/phy/upper/channel_processors/pusch/pusch_resident_estimate_tap.h",
            "lib/phy/upper/channel_processors/pusch/pusch_demodulator_gpu_impl.cpp"])
        self.assertEqual(record["base_commit"], rp.locked()["commit"])
        self.assertEqual(record["interfaces"], [{"header": "include/ocudu/phy/upper/channel_processors/pusch/"
                                                           "pusch_resident_estimate_tap.h", "version": 1}])

    def test_a_manifest_that_disagrees_with_its_files_or_the_lock_is_refused(self):
        series = rp.load_series(self.root)
        self.assertEqual(sorted(series["files"]), ["lib/demod.cpp", "lib/tap.h"])
        cases = ((lambda m: m.update(base_commit="0" * 40), "ext/lock.json pins"),
                 (lambda m: m["patches"][0].update(sha256="0" * 64), "and the series names"),
                 (lambda m: m["patches"][0].update(file="0002-tap.patch"), "must start with 0001-"),
                 (lambda m: m["patches"][0]["files"][0].update(path="../outside.h"), "not a file of the tree"),
                 (lambda m: m["patches"][0]["files"][0].update(sha256_before="1" * 64), "is added and names a hash"),
                 (lambda m: m["patches"][0]["files"][1].update(sha256_after=None), "its SHA-256 after the series"),
                 (lambda m: m.update(patches=[]), "lists no patch"),
                 (lambda m: m.update(schema="other/v1"), "is not an ocudu-patch-series/v1 manifest"))
        saved = json.dumps(self.manifest)
        for change, words in cases:
            self.manifest = json.loads(saved)
            change(self.manifest)
            self.write_series()
            with self.assertRaisesRegex(rp.PatchError, words):
                rp.load_series(self.root)

    def test_a_patched_worktree_is_made_beside_a_git_checkout_that_stays_pristine(self):
        series = rp.load_series(self.root)
        rp.check_apply(self.source, series, self.root)
        record = rp.prepare_tree(self.source, self.tree, series, self.root)
        self.assertEqual((record["tree_kind"], record["series_sha256"]), ("patched-worktree", series["sha256"]))
        self.assertEqual((rp.state_of(self.tree, series), rp.state_of(self.source, series)), ("patched", "pristine"))
        self.assertEqual(self.status(), "")
        self.assertEqual(subprocess.check_output(["git", "-C", str(self.tree), "rev-parse", "HEAD"], text=True).strip(),
                         self.pin)
        self.assertEqual(rp.verify_tree(self.tree, series)["tree_kind"], "patched-worktree")
        # A second preparation leaves every file as it is, so a rebuild compiles nothing.
        stamp = (self.tree / "lib/demod.cpp").stat().st_mtime_ns
        rp.prepare_tree(self.source, self.tree, series, self.root)
        self.assertEqual((self.tree / "lib/demod.cpp").stat().st_mtime_ns, stamp)
        # An edit outside the series, and an edit inside it, are found and repaired.
        (self.tree / "lib/other.cpp").write_text("int other() { return 2; }\n")
        with self.assertRaisesRegex(rp.PatchError, "differs from the pin outside the series: lib/other.cpp"):
            rp.verify_tree(self.tree, series)
        (self.tree / "lib/tap.h").write_text("inline void tap() { /* edited */ }\n")
        with self.assertRaisesRegex(rp.PatchError, "is not the patched receiver source"):
            rp.verify_tree(self.tree, series)
        rp.prepare_tree(self.source, self.tree, series, self.root)
        self.assertEqual(rp.verify_tree(self.tree, series)["tree"], str(self.tree))
        self.assertEqual(self.status(), "")
        self.assertTrue(rp.same_series(record, rp.identity(series)))
        self.assertFalse(rp.same_series(record, dict(rp.identity(series), series_sha256="0" * 64)))

    def test_a_patched_copy_is_made_of_a_source_without_git_metadata(self):
        self.as_archive()
        series = rp.load_series(self.root)
        # The project itself is a repository: without the ceiling Git would skip every path of the patch.
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        rp.check_apply(self.source, series, self.root)
        record = rp.prepare_tree(self.source, self.tree, series, self.root)
        self.assertEqual(record["tree_kind"], "patched-copy")
        self.assertEqual((rp.state_of(self.tree, series), rp.state_of(self.source, series)), ("patched", "pristine"))
        self.assertFalse((self.tree / ".git").exists())
        self.assertEqual(json.loads((self.tree / rp.MARKER).read_text())["series_sha256"], series["sha256"])
        (self.tree / "lib/demod.cpp").write_text("broken\n")
        rp.prepare_tree(self.source, self.tree, series, self.root)
        self.assertEqual(rp.verify_tree(self.tree, series)["tree_kind"], "patched-copy")

    def test_a_tree_inside_the_source_a_foreign_directory_and_a_dirty_source_are_refused(self):
        series = rp.load_series(self.root)
        with self.assertRaisesRegex(rp.PatchError, "must lie outside the locked source"):
            rp.prepare_tree(self.source, self.source / "patched", series, self.root)
        self.tree.mkdir(parents=True)
        (self.tree / "note.txt").write_text("someone's files\n")
        with self.assertRaisesRegex(rp.PatchError, "is not a worktree of the locked source"):
            rp.prepare_tree(self.source, self.tree, series, self.root)
        self.assertTrue((self.tree / "note.txt").is_file())
        (self.source / "lib/demod.cpp").write_text("edited in place\n")
        with self.assertRaisesRegex(rp.PatchError, "is not a clean tree of the pinned commit"):
            rp.check_apply(self.source, series, self.root)


if __name__ == "__main__":
    unittest.main()
