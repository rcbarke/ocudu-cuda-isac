"""Install the exact development-source ZIPs into ext/ and verify their files."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def receipt_bytes(records):
    return json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")


def unpack(archive, directory, revision):
    records = {}
    with zipfile.ZipFile(archive) as source:
        if source.comment != revision.encode("ascii"):
            raise ValueError(f"{archive.name}: wrong revision in archive comment")
        for info in source.infolist():
            parts = PurePosixPath(info.filename).parts
            if not parts or parts[0].startswith("/") or ".." in parts or ".git" in parts:
                raise ValueError(f"Unexpected ZIP member {info.filename}")
            if len(parts) == 1:
                continue
            relative = Path(*parts[1:])
            path = directory / relative
            if info.is_dir():
                path.mkdir(parents=True, exist_ok=True)
                continue
            if relative.as_posix() in records or not path.parent.resolve().is_relative_to(directory):
                raise ValueError(f"Invalid source path {relative}")
            path.parent.mkdir(parents=True, exist_ok=True)
            data = source.read(info)
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                target = data.decode()
                if os.path.isabs(target) or not (path.parent / target).resolve().is_relative_to(directory):
                    raise ValueError(f"Invalid symlink at {relative}")
                path.symlink_to(target)
                kind = "link"
            else:
                path.write_bytes(data)
                path.chmod(0o755 if mode & 0o111 else 0o644)
                kind = "executable" if mode & 0o111 else "file"
            records[relative.as_posix()] = [kind, hashlib.sha256(data).hexdigest()]
    return receipt_bytes(records)


def receipt_path(root, item):
    return root / "build" / "source-receipts" / (Path(item["path"]).name + ".json")


def verify_snapshot(root, item):
    base = root / item["path"]
    receipt = receipt_path(root, item)
    if not receipt.is_file():
        raise ValueError(f"{base}: missing source. Run make hydrate or import the pinned source archives.")
    raw = receipt.read_bytes()
    if hashlib.sha256(raw).hexdigest() != item["snapshot_manifest_sha256"]:
        raise ValueError(f"{receipt}: manifest differs from lock")
    for name, (kind, expected) in json.loads(raw).items():
        path = base / name
        if path.is_symlink():
            actual_kind = "link"
            actual = hashlib.sha256(os.fsencode(os.readlink(path))).hexdigest()
        elif path.is_file():
            actual_kind = "executable" if path.stat().st_mode & 0o111 else "file"
            actual = digest_file(path)
        else:
            raise ValueError(f"{path}: missing source file")
        if (kind, expected) != (actual_kind, actual):
            raise ValueError(f"{path}: source differs from pinned archive")


def import_one(root, item, archive):
    target = root / item["path"]
    if digest_file(archive) != item["archive_sha256"]:
        raise ValueError(f"{archive}: original ZIP hash differs from lock")
    if target.exists() and any(target.iterdir()):
        verify_snapshot(root, item)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".import-") as temp:
        staged = Path(temp) / "source"
        staged.mkdir()
        raw = unpack(archive, staged, item["commit"])
        if hashlib.sha256(raw).hexdigest() != item["snapshot_manifest_sha256"]:
            raise ValueError(f"{archive}: extracted manifest differs from lock")
        receipt = receipt_path(root, item)
        receipt.parent.mkdir(parents=True, exist_ok=True)
        staged_receipt = Path(temp) / "receipt.json"
        staged_receipt.write_bytes(raw)
        staged_receipt.replace(receipt)
        staged.replace(target)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="directory containing the original dependency ZIPs")
    args = parser.parse_args(argv)
    lock = json.loads((ROOT / "ext/lock.json").read_text())
    # Every dependency needs its ZIP. A reference source is imported when its ZIP is in the directory.
    items = [(item, True) for item in lock["dependencies"]]
    items += [(item, False) for item in lock.get("references", [])]
    candidates = {}
    for path in sorted(args.directory.glob("*.zip")):
        with zipfile.ZipFile(path) as archive:
            candidates.setdefault(archive.comment, []).append(path)
    for item, required in items:
        key = item["commit"].encode("ascii")
        choices = candidates.get(key, [])
        source = next((path for path in choices if digest_file(path) == item["archive_sha256"]), None)
        if source is None and not required:
            print(f"{item['path']} {item['commit']} (reference source, no archive supplied)")
            continue
        if source is None:
            raise ValueError(f"Missing exact {item['path']} ZIP with SHA-256 {item['archive_sha256']}")
        target = ROOT / item["path"]
        if (target / ".git").exists():
            raise ValueError(f"{target} already contains a Git checkout. Use make hydrate in Git mode.")
        import_one(ROOT, item, source)
        print(f"{item['path']} {item['commit']} (archive verified)")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, zipfile.BadZipFile) as error:
        sys.exit(str(error))
