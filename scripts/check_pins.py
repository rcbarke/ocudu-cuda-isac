"""Check the source lock and submodule declarations; optionally check installed sources."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

from source_archives import verify_snapshot

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("ocudu", "sionna", "sionna-rt")


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktrees", action="store_true", help="verify the actual installed source trees")
    parser.add_argument("--only", choices=NAMES, help="limit installed-source verification to one dependency")
    options = parser.parse_args()
    items = json.loads((ROOT / "ext/lock.json").read_text())["dependencies"]
    paths = sorted(item["path"] for item in items)
    if paths != sorted(f"ext/{name}" for name in NAMES):
        raise ValueError("The source lock must identify exactly the three dependencies")
    metadata = (ROOT / ".git").exists()
    if metadata:
        entries = git("ls-tree", "-r", "HEAD", "--", "ext").splitlines()
        gitlinks = sorted(entry.split("\t", 1)[1] for entry in entries if entry.startswith("160000 commit "))
        if gitlinks != paths:
            raise ValueError("Committed gitlinks differ from the source lock")
    for item in items:
        path, sha = item["path"], item["commit"]
        if not re.fullmatch("[0-9a-f]{40}", sha):
            raise ValueError(f"{path}: invalid commit")
        for key in ("archive_sha256", "snapshot_manifest_sha256"):
            if not re.fullmatch("[0-9a-f]{64}", item[key]):
                raise ValueError(f"{path}: invalid {key}")
        for field in ("path", "url"):
            if git("config", "--file", ".gitmodules", "--get", f"submodule.{path}.{field}") != item[field]:
                raise ValueError(f"{path}: submodule {field} differs from lock")
        if metadata and git("ls-tree", "HEAD", "--", path) != f"160000 commit {sha}\t{path}":
            raise ValueError(f"{path}: gitlink differs from lock")
        if options.only and path != f"ext/{options.only}":
            continue
        kind = "parent gitlink" if metadata else "source-export declaration"
        if options.worktrees:
            checkout = ROOT / path
            if (checkout / ".git").exists():
                if (git("rev-parse", "HEAD", cwd=checkout) != sha or
                        git("status", "--porcelain", "--untracked-files=no", cwd=checkout)):
                    raise ValueError(f"{path}: expected pristine checkout at {sha}")
                kind = "Git worktree"
            else:
                verify_snapshot(ROOT, item)
                kind = "verified archive"
        print(f"{path} {sha} ({kind})")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
