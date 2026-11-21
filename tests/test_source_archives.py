"""Check offline dependency acquisition with small synthetic source archives."""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import source_archives as sources


class SourcesTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.archive = self.root / "upstream.zip"
        self.commit = "b" * 40
        self.files = {
            "source.cc": (b"source\n", stat.S_IFREG | 0o644, "file"),
            "tools/run": (b"#!/bin/sh\n", stat.S_IFREG | 0o755, "executable"),
            "tools/see.cc": (b"../source.cc", stat.S_IFLNK | 0o777, "link"),
        }
        with zipfile.ZipFile(self.archive, "w") as archive:
            archive.comment = self.commit.encode()
            for name, (data, mode, _) in self.files.items():
                info = zipfile.ZipInfo("tree/" + name)
                info.external_attr = mode << 16
                archive.writestr(info, data)
        manifest = {name: [kind, hashlib.sha256(data).hexdigest()]
                    for name, (data, _, kind) in self.files.items()}
        self.item = {"commit": self.commit, "path": "ext/ocudu",
                     "archive_sha256": sources.digest_file(self.archive),
                     "snapshot_manifest_sha256": hashlib.sha256(sources.receipt_bytes(manifest)).hexdigest()}
        self.target = self.root / self.item["path"]

    def test_idempotent_extract_preserves_bytes_mode_and_links(self):
        self.target.mkdir(parents=True)
        sources.import_one(self.root, self.item, self.archive)
        self.assertEqual((self.target / "tools/see.cc").read_bytes(), b"source\n")
        self.assertEqual(os.readlink(self.target / "tools/see.cc"), "../source.cc")
        self.assertTrue(os.access(self.target / "tools/run", os.X_OK))
        first = (self.target / "source.cc").stat().st_ino
        sources.import_one(self.root, self.item, self.archive)
        self.assertEqual(first, (self.target / "source.cc").stat().st_ino)
        sources.verify_snapshot(self.root, self.item)

    def test_corruption_and_changed_source_rejected(self):
        with self.assertRaisesRegex(ValueError, "ZIP hash differs"):
            sources.import_one(self.root, {**self.item, "archive_sha256": "0" * 64}, self.archive)
        with self.assertRaisesRegex(ValueError, "wrong revision"):
            sources.import_one(self.root, {**self.item, "commit": "1" * 40}, self.archive)
        self.assertFalse(self.target.exists())
        sources.import_one(self.root, self.item, self.archive)
        (self.target / "source.cc").write_text("modified")
        with self.assertRaisesRegex(ValueError, "source differs"):
            sources.import_one(self.root, self.item, self.archive)
        sources.receipt_path(self.root, self.item).write_text(json.dumps({}))
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            sources.verify_snapshot(self.root, self.item)

    def test_reference_archive_is_optional_and_a_dependency_archive_is_required(self):
        reference = {**self.item, "path": "ext/reference", "commit": "c" * 40}
        (self.root / "ext").mkdir()
        lock = self.root / "ext/lock.json"
        lock.write_text(json.dumps({"dependencies": [self.item], "references": [reference]}))
        with mock.patch.object(sources, "ROOT", self.root), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            sources.main([str(self.root)])
        self.assertEqual(output.getvalue().splitlines(),
                         [f"ext/ocudu {self.commit} (archive verified)",
                          f"ext/reference {'c' * 40} (reference source, no archive supplied)"])
        self.assertTrue((self.target / "source.cc").is_file())
        self.assertFalse((self.root / "ext/reference").exists())
        lock.write_text(json.dumps({"dependencies": [reference], "references": []}))
        with mock.patch.object(sources, "ROOT", self.root), \
                self.assertRaisesRegex(ValueError, "Missing exact ext/reference ZIP"):
            sources.main([str(self.root)])

    def test_nonempty_destination_preserved_on_failure(self):
        self.target.mkdir(parents=True)
        keep = self.target / "local.cc"
        keep.write_text("local work")
        with self.assertRaisesRegex(ValueError, "missing source"):
            sources.import_one(self.root, self.item, self.archive)
        self.assertEqual(keep.read_text(), "local work")


if __name__ == "__main__":
    unittest.main()
