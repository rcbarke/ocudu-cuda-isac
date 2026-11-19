"""Validate the three committed gitlinks, optionally including checked-out worktrees."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktrees", action="store_true")
    options = parser.parse_args()
    deps = json.loads((ROOT / "deps/lock.json").read_text())["dependencies"]
    locked_paths = {item["path"] for item in deps}
    entries = git("ls-tree", "-r", "HEAD", "deps").splitlines()
    gitlinks = {entry.split("\t", 1)[1] for entry in entries if entry.startswith("160000 commit ")}
    if locked_paths != gitlinks or len(locked_paths) != len(deps):
        raise ValueError("The committed dependency gitlinks differ from deps/lock.json")
    for item in deps:
        path, sha = item["path"], item["commit"]
        entry = git("ls-tree", "HEAD", "--", path)
        if entry != f"160000 commit {sha}\t{path}":
            raise ValueError(f"{path}: expected gitlink {sha}, found {entry}")
        url = git("config", "--file", ".gitmodules", "--get", f"submodule.{path}.url")
        if url != item["url"]:
            raise ValueError(f"{path}: URL differs from the dependency lock")
        if options.worktrees:
            directory = ROOT / path
            if not (directory / ".git").exists():
                raise ValueError(f"{path}: worktree is uninitialized")
            if git("rev-parse", "HEAD", cwd=directory) != sha:
                raise ValueError(f"{path}: worktree differs from committed gitlink")
        print(f"{path} {sha}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
