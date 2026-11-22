"""Read the receiver patch series, check it and prepare a patched tree of the pinned OCUDU source.

The series in patches/ocudu/series.json is the only definition of what a patched receiver is. It
names the base commit, every patch with its hash and every file the series touches with its hash
before and after. This module never writes into the locked source directory: a patched tree is a
separate Git worktree of the pinned commit, or a copy for a source that was installed from the
verified archive, in a directory of the caller.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SERIES = Path("patches/ocudu/series.json")
SCHEMA = "ocudu-patch-series/v1"
IDENTITY_SCHEMA = "receiver-patch/v1"
MARKER = ".isac-patched.json"  # written into a patched tree: what was applied to it
KIND_WORKTREE = "patched-worktree"
KIND_COPY = "patched-copy"
KINDS = (KIND_WORKTREE, KIND_COPY)


class PatchError(ValueError):
    """The series, a patch or a patched tree is not what the manifest says. The message says what to do."""


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def locked(root=ROOT):
    """The lock entry of the OCUDU source."""
    for item in json.loads((root / "ext/lock.json").read_text())["dependencies"]:
        if item.get("name") == "ocudu":
            return item
    raise PatchError("ext/lock.json has no dependency named 'ocudu'")


def load_series(root=ROOT):
    """Return the validated series: the manifest, its hash and the files it touches.

    The manifest must name the commit the lock pins, every patch file must exist with the hash the
    manifest names, and every touched path must stay inside the source tree.
    """
    path = root / SERIES
    try:
        raw = path.read_bytes()
        series = json.loads(raw)
    except OSError as error:
        raise PatchError(f"cannot read the patch series {path}: {error}") from None
    except ValueError as error:
        raise PatchError(f"{path} is not valid JSON: {error}") from None
    if not isinstance(series, dict) or series.get("schema") != SCHEMA:
        raise PatchError(f"{path} is not an {SCHEMA} manifest")
    pin = locked(root)["commit"]
    if series.get("dependency") != "ocudu" or series.get("base_commit") != pin:
        raise PatchError(f"{path} was written for the commit {series.get('base_commit')}, and ext/lock.json pins {pin}. "
                         "A series is rebased and its hashes are renewed when the pin moves.")
    patches = series.get("patches")
    if not isinstance(patches, list) or not patches:
        raise PatchError(f"{path} lists no patch")
    files, entries = {}, []
    for index, patch in enumerate(patches):
        name = patch.get("file") if isinstance(patch, dict) else None
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".patch"):
            raise PatchError(f"{path}: patch {index} names no patch file of this directory")
        if not name.startswith(f"{index + 1:04d}-"):
            raise PatchError(f"{path}: the patch {name} is entry {index + 1} of the series and must start with "
                             f"{index + 1:04d}-")
        file = path.parent / name
        if not file.is_file():
            raise PatchError(f"{file} is missing")
        found = digest(file)
        if found != patch.get("sha256"):
            raise PatchError(f"{file} has the SHA-256 {found}, and the series names {patch.get('sha256')}")
        if not file.read_bytes().isascii():
            raise PatchError(f"{file} holds a byte that is not ASCII")
        touched = patch.get("files")
        if not isinstance(touched, list) or not touched:
            raise PatchError(f"{path}: the patch {name} lists no file it touches")
        for item in touched:
            relative = item.get("path") if isinstance(item, dict) else None
            parts = Path(relative).parts if isinstance(relative, str) else ()
            if not parts or Path(relative).is_absolute() or ".." in parts or parts[0] == ".git":
                raise PatchError(f"{path}: the patch {name} names the path {relative!r}, which is not a file of the tree")
            after = item.get("sha256_after")
            if item.get("change") not in ("added", "modified") or not isinstance(after, str) or len(after) != 64:
                raise PatchError(f"{path}: {relative} of {name} needs a change (added or modified) and its SHA-256 after "
                                 "the series")
            before = files[relative]["after"] if relative in files else item.get("sha256_before")
            if (item["change"] == "added") != (before is None):
                raise PatchError(f"{path}: {relative} of {name} is {item['change']} and names "
                                 + ("a hash before the series" if before else "no hash before the series"))
            first = files[relative]["before"] if relative in files else before
            files[relative] = {"before": first, "after": after}
        entries.append({"file": name, "sha256": found, "sprint": patch.get("sprint"), "selector": patch.get("selector"),
                        "interface": patch.get("interface")})
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "base_commit": pin,
            "base_ref": series.get("base_ref"), "patches": entries, "files": files, "manifest": series}


def identity(series, kind=None, tree=None):
    """What a receipt carries of the series: enough to tell one patched receiver from another."""
    record = {"schema": IDENTITY_SCHEMA, "series": str(SERIES), "series_sha256": series["sha256"],
              "base_commit": series["base_commit"],
              "patches": [{"file": item["file"], "sha256": item["sha256"]} for item in series["patches"]],
              "files": {path: item["after"] for path, item in sorted(series["files"].items())},
              "interfaces": [item["interface"] for item in series["patches"] if item.get("interface")]}
    if kind is not None:
        record["tree_kind"] = kind
    if tree is not None:
        record["tree"] = str(tree)
    return record


def same_series(left, right):
    """Whether two identities name the same patched receiver. Where the tree lies does not matter."""
    keys = ("schema", "series_sha256", "base_commit", "patches", "files")
    return isinstance(left, dict) and isinstance(right, dict) and all(left.get(key) == right.get(key) for key in keys)


def state_of(tree, series):
    """Compare the touched files of a tree with the series: 'patched', 'pristine' or a text that says what differs."""
    tree = Path(tree)
    patched, pristine, odd = 0, 0, []
    for relative, hashes in series["files"].items():
        path = tree / relative
        found = digest(path) if path.is_file() else None
        if found == hashes["after"]:
            patched += 1
        elif found == hashes["before"]:
            pristine += 1
        else:
            odd.append(f"{relative} is {'missing' if found is None else found[:12]}")
    if odd:
        return "; ".join(odd)
    if patched == len(series["files"]):
        return "patched"
    if pristine == len(series["files"]):
        return "pristine"
    return f"{patched} of {len(series['files'])} touched files are patched and {pristine} are not"


def _git(args, cwd, check=True, env=None):
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False, env=env)
    if check and result.returncode:
        raise PatchError(f"git {' '.join(str(arg) for arg in args)} failed in {cwd}: "
                         f"{(result.stderr or result.stdout).strip()[-400:]}")
    return result


def _outside_any_repository(tree):
    """Environment in which Git does not look above a tree for a repository.

    A copy lies under build/ of this project. Without the ceiling, git apply takes the paths of a
    patch as paths of the enclosing repository and skips every one of them without an error.
    """
    env = os.environ.copy()
    ceilings = [str(Path(tree).resolve().parent)] + [item for item in env.get("GIT_CEILING_DIRECTORIES", "").split(os.pathsep)
                                                     if item]
    env["GIT_CEILING_DIRECTORIES"] = os.pathsep.join(ceilings)
    return env


def check_apply(tree, series, root=ROOT):
    """git apply --check of the whole series on a clean tree of the pin. Nothing is written."""
    tree = Path(tree)
    state = state_of(tree, series)
    if state != "pristine":
        raise PatchError(f"{tree} is not a clean tree of the pinned commit: {state}")
    env = None if (tree / ".git").exists() else _outside_any_repository(tree)
    names = [str((root / SERIES).parent / item["file"]) for item in series["patches"]]
    _git(["apply", "--check", "--whitespace=nowarn", *names], tree, env=env)


def _apply(tree, series, root):
    env = None if (tree / ".git").exists() else _outside_any_repository(tree)
    for item in series["patches"]:
        _git(["apply", "--whitespace=nowarn", str((root / SERIES).parent / item["file"])], tree, env=env)
    state = state_of(tree, series)
    if state != "patched":
        raise PatchError(f"the series did not leave {tree} as its manifest says: {state}")


def prepare_tree(source, destination, series, root=ROOT):
    """Make destination a tree of the pinned commit with the series applied. Return its identity.

    source is the locked checkout. With Git metadata the tree is a linked worktree of it, detached
    at the pinned commit. Without, it is a copy. A destination that is already the patched tree is
    left as it is, so that a rebuild compiles only what changed. One that differs is brought back
    to the pin and patched again. Nothing is written into source.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or source in destination.parents or destination in source.parents:
        raise PatchError(f"the patched tree {destination} must lie outside the locked source {source}")
    pin = series["base_commit"]
    if (source / ".git").exists():
        kind = KIND_WORKTREE
        if destination.exists() and not (destination / ".git").exists():
            raise PatchError(f"{destination} exists and is not a worktree of the locked source. Remove it or name "
                             "another directory.")
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            _git(["worktree", "prune"], source)
            _git(["worktree", "add", "--detach", str(destination), pin], source)
        head = _git(["rev-parse", "HEAD"], destination).stdout.strip()
        if head != pin or state_of(destination, series) != "patched" or _unexpected(destination, series):
            # Only this worktree is reset. It belongs to the build and holds nothing else.
            _git(["checkout", "--force", "--detach", pin], destination)
            _git(["clean", "-fdq"], destination)
            _apply(destination, series, root)
    else:
        kind = KIND_COPY
        if destination.exists() and state_of(destination, series) != "patched":
            if not (destination / MARKER).is_file():
                raise PatchError(f"{destination} exists and is not a patched copy that this tool made. Remove it or "
                                 "name another directory.")
            shutil.rmtree(destination)
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, destination, symlinks=True, ignore=shutil.ignore_patterns(".git"))
            _apply(destination, series, root)
    record = identity(series, kind, destination)
    (destination / MARKER).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def _unexpected(tree, series):
    """Paths of a worktree that differ from the pin and are not files of the series."""
    listing = _git(["status", "--porcelain", "--untracked-files=all"], tree).stdout.splitlines()
    changed = {line[3:].strip().strip('"') for line in listing if line.strip()}
    return sorted(changed - set(series["files"]) - {MARKER})


def verify_tree(tree, series):
    """Require that a tree is the pinned source with exactly the series applied."""
    tree = Path(tree)
    state = state_of(tree, series)
    if state != "patched":
        raise PatchError(f"{tree} is not the patched receiver source of the series: {state}")
    if (tree / ".git").exists():
        head = _git(["rev-parse", "HEAD"], tree).stdout.strip()
        if head != series["base_commit"]:
            raise PatchError(f"{tree} is at {head}, and the series is based on {series['base_commit']}")
        extra = _unexpected(tree, series)
        if extra:
            raise PatchError(f"{tree} differs from the pin outside the series: " + ", ".join(extra[:5]))
    return identity(series, KIND_WORKTREE if (tree / ".git").exists() else KIND_COPY, tree)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="check the manifest against the patch files and the lock")
    check.add_argument("--apply-to", type=Path, help="a clean tree of the pinned commit: also run git apply --check")
    prepare = commands.add_parser("prepare", help="make a patched tree of the locked source")
    prepare.add_argument("destination", type=Path)
    verify = commands.add_parser("verify", help="require that a tree is the pin with exactly the series applied")
    verify.add_argument("tree", type=Path)
    args = parser.parse_args(argv)
    try:
        series = load_series()
        if args.command == "check":
            if args.apply_to is not None:
                check_apply(args.apply_to, series)
            record = identity(series)
            record["apply_check"] = str(args.apply_to) if args.apply_to is not None else None
        elif args.command == "prepare":
            record = prepare_tree(ROOT / locked()["path"], args.destination, series)
        else:
            record = verify_tree(args.tree, series)
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0
    except PatchError as error:
        print(f"receiver_patch: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
