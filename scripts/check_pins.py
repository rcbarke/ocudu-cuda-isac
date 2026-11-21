"""Check the source lock and submodule declarations; optionally check installed sources."""
import argparse
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

from source_archives import verify_snapshot

ROOT = Path(__file__).resolve().parents[1]
# Logical dependency names used by the tools. ext/lock.json maps each one to its checkout path.
NAMES = ("ocudu", "sionna", "sionna-rt")
# Reference sources are pinned by the same lock, .gitmodules entry and gitlink. They are read, and
# nothing builds or installs them in the isac environment. A worktree check covers the dependencies;
# it covers a reference source when --references is given or --only names that source.
REFERENCES = ("aerial-cuda-accelerated-ran", "aerial-sample-apps")
LOCKED = NAMES + REFERENCES
FORMAT = 4
FIELDS = ("name", "path", "url", "commit", "archive_sha256", "snapshot_manifest_sha256")


def git(*args, cwd):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def load_lock(root):
    """Return the lock entries keyed by logical name after validating their shape."""
    lock = json.loads((root / "ext/lock.json").read_text())
    if lock.get("format") != FORMAT:
        raise ValueError(f"ext/lock.json: expected format {FORMAT}, which names every dependency and "
                         f"lists the reference sources; found format {lock.get('format')!r}")
    groups = (("dependencies", NAMES), ("references", REFERENCES))
    for key, _ in groups:
        if not isinstance(lock.get(key), list):
            raise ValueError(f"ext/lock.json: {key!r} must be a list")
    items = [item for key, _ in groups for item in lock[key]]
    for item in items:
        missing = [field for field in FIELDS if not isinstance(item.get(field), str)]
        if missing:
            raise ValueError(f"ext/lock.json: entry {item.get('path', item.get('name'))!r} lacks "
                             + ", ".join(missing))
    for key, expected in groups:
        names = sorted(item["name"] for item in lock[key])
        if names != sorted(expected):
            raise ValueError(f"ext/lock.json {key} must name exactly " + ", ".join(expected) + "; found "
                             + (", ".join(names) or "none"))
    paths = [item["path"] for item in items]
    for path in paths:
        parts = PurePosixPath(path).parts
        if len(parts) != 2 or parts[0] != "ext" or parts[1] in (".", "..") or path != "/".join(parts):
            raise ValueError(f"ext/lock.json: {path!r} must be one directory directly under ext/")
    if len(set(paths)) != len(paths):
        raise ValueError("ext/lock.json: two dependencies share one path")
    for item in items:
        if not re.fullmatch("[0-9a-f]{40}", item["commit"]):
            raise ValueError(f"{item['path']}: invalid commit")
        for key in ("archive_sha256", "snapshot_manifest_sha256"):
            if not re.fullmatch("[0-9a-f]{64}", item[key]):
                raise ValueError(f"{item['path']}: invalid {key}")
    return {item["name"]: item for item in items}


def declared_submodules(root):
    """Map every path declared in .gitmodules to its URL, whatever the submodule is called."""
    if not (root / ".gitmodules").is_file():
        raise ValueError(".gitmodules is missing. Declare the locked sources as submodules "
                         "(see ext/README.md).")
    listing = subprocess.run(
        ["git", "config", "--file", ".gitmodules", "--null", "--get-regexp", r"^submodule\..*\.(path|url)$"],
        cwd=root, capture_output=True, text=True, check=False).stdout
    settings = {}
    for record in filter(None, listing.split("\0")):
        key, _, value = record.partition("\n")
        name, field = key[len("submodule."):].rsplit(".", 1)
        settings.setdefault(name, {})[field] = value
    return {entry.get("path"): entry.get("url") for entry in settings.values()}


def verify_sources(root, worktrees=False, only=None, references=False):
    """Yield the verified identity of each locked source. Raise ValueError on the first mismatch.

    The lock, the submodule declaration and the gitlink are checked for every source. With
    ``worktrees`` the installed tree of each dependency is checked too, and that of a reference
    source when ``references`` is set or ``only`` names it.
    """
    entries = load_lock(root)
    paths = sorted(item["path"] for item in entries.values())
    declared = declared_submodules(root)
    metadata = (root / ".git").exists()
    if metadata:
        tree = git("ls-tree", "-r", "HEAD", "--", "ext", cwd=root).splitlines()
        gitlinks = sorted(entry.split("\t", 1)[1] for entry in tree if entry.startswith("160000 commit "))
        if gitlinks != paths:
            raise ValueError("Committed submodules (" + (", ".join(gitlinks) or "none")
                             + ") differ from the lock paths (" + ", ".join(paths)
                             + "). Commit the submodules or correct the path in ext/lock.json.")
    for name in LOCKED:
        item = entries[name]
        path, sha = item["path"], item["commit"]
        if path not in declared:
            raise ValueError(f"{path}: not declared in .gitmodules (declared: "
                             + (", ".join(sorted(filter(None, declared))) or "none") + ")")
        if declared[path] != item["url"]:
            raise ValueError(f"{path}: .gitmodules url {declared[path]!r} differs from lock {item['url']!r}")
        if metadata and git("ls-tree", "HEAD", "--", path, cwd=root) != f"160000 commit {sha}\t{path}":
            raise ValueError(f"{path}: committed gitlink differs from lock commit {sha}")
        if only and name != only:
            continue
        kind = "parent gitlink" if metadata else "source-export declaration"
        if worktrees and (name in NAMES or references or name == only):
            checkout = root / path
            if (checkout / ".git").exists():
                if (git("rev-parse", "HEAD", cwd=checkout) != sha or
                        git("status", "--porcelain", "--untracked-files=no", cwd=checkout)):
                    raise ValueError(f"{path}: expected pristine checkout at {sha}")
                kind = "Git worktree"
            else:
                verify_snapshot(root, item)
                kind = "verified archive"
        yield {"name": name, "path": path, "url": item["url"], "commit": sha, "kind": kind}


def check(root, worktrees=False, only=None, references=False):
    """Yield one summary line per verified source."""
    for source in verify_sources(root, worktrees, only, references):
        yield f"{source['path']} {source['commit']} ({source['kind']})"


def main(argv=None, root=ROOT):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktrees", action="store_true", help="verify the actual installed source trees")
    parser.add_argument("--references", action="store_true",
                        help="with --worktrees, also verify the trees of the reference sources")
    parser.add_argument("--only", choices=LOCKED, help="limit installed-source verification to one source")
    parser.add_argument("--path", choices=LOCKED, help="print the locked checkout directory of one source")
    options = parser.parse_args(argv)
    if options.path and (options.worktrees or options.only or options.references):
        parser.error("--path cannot be combined with --worktrees, --references or --only")
    if options.references and not options.worktrees:
        parser.error("--references needs --worktrees")
    if options.path:
        print(root / load_lock(root)[options.path]["path"])
        return
    for line in check(root, options.worktrees, options.only, options.references):
        print(line)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
