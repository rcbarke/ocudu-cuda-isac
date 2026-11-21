"""Check the source-lock gate against small local Git fixtures. No network or real upstream."""

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import check_pins as pins

# The OCUDU directory deliberately differs from its logical name: the lock owns the path.
PATHS = {"ocudu": "ext/ocudu-deepsig-cuda-inline", "sionna": "ext/sionna", "sionna-rt": "ext/sionna-rt",
         "aerial-cuda-accelerated-ran": "ext/aerial-cuda-accelerated-ran",
         "aerial-sample-apps": "ext/aerial-sample-apps"}
REFERENCE = "aerial-sample-apps"
URLS = {name: f"https://example.invalid/{name}.git" for name in PATHS}
GIT_ENV = {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
           "GIT_AUTHOR_NAME": "fixture", "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
           "GIT_COMMITTER_NAME": "fixture", "GIT_COMMITTER_EMAIL": "fixture@example.invalid"}


@unittest.skipUnless(shutil.which("git"), "the pin gate reads Git metadata")
class PinGateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / "checkout"
        (self.root / "ext").mkdir(parents=True)
        self.git(self.root, "init", "-q")
        self.commits = {}
        for name, path in PATHS.items():
            checkout = self.root / path
            checkout.mkdir()
            self.git(checkout, "init", "-q")
            (checkout / "source.txt").write_text(name + "\n")
            self.git(checkout, "add", "source.txt")
            self.git(checkout, "commit", "-q", "-m", "fixture source")
            self.commits[name] = self.git(checkout, "rev-parse", "HEAD")
        self.write_modules()
        self.write_lock()
        self.commit_parent()

    def git(self, cwd, *arguments):
        return subprocess.run(["git", "-c", "advice.addEmbeddedRepo=false", *arguments], cwd=cwd,
                              env={**os.environ, **GIT_ENV}, capture_output=True, text=True,
                              check=True).stdout.strip()

    def write_modules(self, paths=PATHS, urls=URLS, sections=None):
        sections = sections or paths
        text = "".join(f'[submodule "{sections[name]}"]\n\tpath = {paths[name]}\n\turl = {urls[name]}\n'
                       for name in paths)
        (self.root / ".gitmodules").write_text(text)

    def entries(self):
        return [{"name": name, "path": path, "url": URLS[name], "commit": self.commits[name],
                 "archive_sha256": "a" * 64, "snapshot_manifest_sha256": "b" * 64}
                for name, path in PATHS.items()]

    def write_lock(self, entries=None, lock_format=4):
        entries = self.entries() if entries is None else entries
        lock = {"format": lock_format,
                "dependencies": [entry for entry in entries if entry.get("name") not in pins.REFERENCES],
                "references": [entry for entry in entries if entry.get("name") in pins.REFERENCES]}
        (self.root / "ext/lock.json").write_text(json.dumps(lock, indent=2) + "\n")

    def commit_parent(self):
        self.git(self.root, "add", "-A")
        self.git(self.root, "commit", "-q", "-m", "pin fixture sources")

    def lines(self, **options):
        return list(pins.check(self.root, **options))

    def test_checkout_path_comes_from_the_lock_not_from_the_logical_name(self):
        expected = [f"{PATHS[name]} {self.commits[name]}" for name in pins.LOCKED]
        self.assertEqual(self.lines(), [line + " (parent gitlink)" for line in expected])
        # A worktree check covers the three dependencies. A reference source keeps its gitlink identity.
        self.assertEqual(self.lines(worktrees=True), [line + " (Git worktree)" for line in expected[:3]]
                         + [line + " (parent gitlink)" for line in expected[3:]])
        self.assertEqual(self.lines(worktrees=True, references=True),
                         [line + " (Git worktree)" for line in expected])
        self.assertEqual(self.lines(worktrees=True, only="ocudu"), [expected[0] + " (Git worktree)"])
        self.assertEqual(list(pins.verify_sources(self.root, worktrees=True, only="sionna-rt")),
                         [{"name": "sionna-rt", "path": "ext/sionna-rt", "url": URLS["sionna-rt"],
                           "commit": self.commits["sionna-rt"], "kind": "Git worktree"}])
        with contextlib.redirect_stdout(io.StringIO()) as output:
            pins.main(["--path", "ocudu"], root=self.root)
        self.assertEqual(output.getvalue().strip(), str(self.root / "ext/ocudu-deepsig-cuda-inline"))

    def test_malformed_lock_names_the_problem(self):
        def changed(index, **fields):
            entries = self.entries()
            entries[index] = {**entries[index], **fields}
            return entries

        without_name = [{key: value for key, value in entry.items() if key != "name"}
                        for entry in self.entries()]
        cases = ((self.entries(), 3, "expected format 4"),
                 (without_name, 4, "lacks name"),
                 (changed(0, name="gnb"), 4, "dependencies must name exactly ocudu, sionna, sionna-rt"),
                 (self.entries()[:2] + self.entries()[3:], 4, "dependencies must name exactly"),
                 (self.entries()[:4], 4, "references must name exactly aerial-cuda-accelerated-ran, "
                                         "aerial-sample-apps; found aerial-cuda-accelerated-ran"),
                 (changed(0, path="ext/ocudu/inline"), 4, "directly under ext/"),
                 (changed(0, path="native/ocudu"), 4, "directly under ext/"),
                 (changed(0, path="ext/.."), 4, "directly under ext/"),
                 (changed(0, path="ext/sionna"), 4, "share one path"),
                 (changed(4, path="ext/sionna"), 4, "share one path"),
                 (changed(1, commit="main"), 4, "invalid commit"),
                 (changed(3, commit="26.2.0"), 4, "invalid commit"),
                 (changed(2, archive_sha256="0" * 63), 4, "invalid archive_sha256"))
        for entries, lock_format, message in cases:
            self.write_lock(entries, lock_format)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.lines()
        # A format 3 lock raised to 4 by hand still lacks the list of reference sources.
        lock = {"format": 4, "dependencies": self.entries()[:3]}
        (self.root / "ext/lock.json").write_text(json.dumps(lock))
        with self.assertRaisesRegex(ValueError, "'references' must be a list"):
            self.lines()

    def test_lock_path_must_match_the_committed_and_declared_submodule(self):
        renamed = self.entries()
        renamed[0]["path"] = "ext/ocudu"
        self.write_lock(renamed)
        with self.assertRaisesRegex(ValueError, r"Committed submodules \(.*ext/ocudu-deepsig-cuda-inline, "
                                                r".*differ from the lock paths \(.*ext/ocudu, "):
            self.lines()
        self.write_lock()
        self.write_modules(urls={**URLS, "sionna": "https://example.invalid/fork.git"})
        with self.assertRaisesRegex(ValueError, "ext/sionna: .gitmodules url .*fork.git.* differs from lock"):
            self.lines()
        self.write_modules(paths={name: PATHS[name] for name in ("ocudu", "sionna")})
        with self.assertRaisesRegex(ValueError, "ext/sionna-rt: not declared in .gitmodules"):
            self.lines()
        (self.root / ".gitmodules").unlink()
        with self.assertRaisesRegex(ValueError, ".gitmodules is missing"):
            self.lines()
        # A submodule moved with git mv keeps its old section name; only path and url matter.
        self.write_modules(sections={**PATHS, "ocudu": "ext/ocudu"})
        self.assertEqual(len(self.lines()), 5)
        self.write_modules(paths={name: PATHS[name] for name in pins.NAMES})
        with self.assertRaisesRegex(ValueError, "ext/aerial-cuda-accelerated-ran: not declared in .gitmodules"):
            self.lines()

    def test_submodule_must_be_committed_at_the_locked_revision(self):
        self.git(self.root, "rm", "-q", "--cached", PATHS["sionna-rt"])
        self.git(self.root, "commit", "-q", "-m", "drop one gitlink")
        with self.assertRaisesRegex(ValueError, "Committed submodules .*differ from the lock paths"):
            self.lines()
        self.commit_parent()
        self.assertEqual(len(self.lines()), 5)
        self.git(self.root, "rm", "-q", "--cached", PATHS[REFERENCE])
        self.git(self.root, "commit", "-q", "-m", "drop the gitlink of a reference source")
        with self.assertRaisesRegex(ValueError, "Committed submodules .*differ from the lock paths"):
            self.lines()
        self.commit_parent()
        moved = self.entries()
        moved[1]["commit"] = "c" * 40
        self.write_lock(moved)
        with self.assertRaisesRegex(ValueError, "ext/sionna: committed gitlink differs from lock commit"):
            self.lines()

    def test_changed_tracked_file_or_revision_fails_worktree_verification(self):
        tracked = self.root / PATHS["sionna"] / "source.txt"
        tracked.write_text("edited\n")
        with self.assertRaisesRegex(ValueError, "ext/sionna: expected pristine checkout"):
            self.lines(worktrees=True)
        self.assertEqual(len(self.lines(worktrees=True, only="ocudu")), 1)
        self.assertEqual(len(self.lines()), 5)
        tracked.write_text("sionna\n")
        (self.root / PATHS["sionna"] / "build.log").write_text("untracked build product\n")
        self.assertEqual(len(self.lines(worktrees=True)), 5)
        checkout = self.root / PATHS["sionna-rt"]
        (checkout / "source.txt").write_text("newer upstream\n")
        self.git(checkout, "commit", "-q", "-a", "-m", "newer upstream revision")
        with self.assertRaisesRegex(ValueError, "ext/sionna-rt: expected pristine checkout"):
            self.lines(worktrees=True)

    def test_reference_source_tree_is_verified_only_on_request(self):
        tracked = self.root / PATHS[REFERENCE] / "source.txt"
        tracked.write_text("edited\n")
        # The check that the gates run covers the dependencies. An edited reference tree does not fail it.
        self.assertEqual(len(self.lines(worktrees=True)), 5)
        self.assertEqual(len(self.lines(worktrees=True, only="ocudu")), 1)
        for options in ({"references": True}, {"only": REFERENCE}):
            with self.subTest(options=options), \
                    self.assertRaisesRegex(ValueError, "ext/aerial-sample-apps: expected pristine checkout"):
                self.lines(worktrees=True, **options)
        tracked.write_text(REFERENCE + "\n")
        line = f"{PATHS[REFERENCE]} {self.commits[REFERENCE]}"
        self.assertEqual(self.lines(worktrees=True, only=REFERENCE), [line + " (Git worktree)"])
        # Not fetched: the gitlink identity still holds, and a requested tree check names the remedy.
        shutil.rmtree(self.root / PATHS[REFERENCE])
        (self.root / PATHS[REFERENCE]).mkdir()
        self.assertEqual(self.lines(worktrees=True)[-1], line + " (parent gitlink)")
        for options in ({"references": True}, {"only": REFERENCE}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "missing source. Run make hydrate"):
                self.lines(worktrees=True, **options)

    def test_source_export_without_git_metadata_uses_declarations_and_receipts(self):
        for metadata in (self.root, *(self.root / path for path in PATHS.values())):
            shutil.rmtree(metadata / ".git")
        expected = [f"{PATHS[name]} {self.commits[name]} (source-export declaration)" for name in pins.LOCKED]
        self.assertEqual(self.lines(), expected)
        with self.assertRaisesRegex(ValueError, "missing source"):
            self.lines(worktrees=True)

    def test_path_lookup_refuses_other_work_and_unknown_names(self):
        for argv in (["--path", "ocudu", "--worktrees"], ["--path", "gnb"], ["--references"],
                     ["--path", REFERENCE, "--references"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as stop:
                pins.main(argv, root=self.root)
            self.assertEqual(stop.exception.code, 2)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            pins.main(["--path", REFERENCE], root=self.root)
        self.assertEqual(output.getvalue().strip(), str(self.root / "ext/aerial-sample-apps"))
        with contextlib.redirect_stdout(io.StringIO()) as output:
            pins.main(["--worktrees", "--references"], root=self.root)
        self.assertEqual(output.getvalue().count("(Git worktree)"), 5)

    def test_repository_lock_and_gitmodules_agree(self):
        entries = pins.load_lock(pins.ROOT)
        declared = pins.declared_submodules(pins.ROOT)
        self.assertEqual(sorted(entries), sorted(pins.LOCKED))
        for item in entries.values():
            with self.subTest(path=item["path"]):
                self.assertEqual(declared.get(item["path"]), item["url"])


if __name__ == "__main__":
    unittest.main()
