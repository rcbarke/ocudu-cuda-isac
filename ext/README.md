# External sources

The project builds on three upstream sources. Each one is a Git submodule under `ext/`, pinned by a gitlink in this repository and described in [lock.json](lock.json). No upstream code is stored in this repository.

| Name | Directory | Upstream | Pinned revision |
|---|---|---|---|
| `ocudu` | `ext/ocudu-deepsig-cuda-inline` | [WG1 CUDA-accelerated OCUDU](https://gitlab.com/ocudu/work_groups/wg1_hw_accel/cuda_accelerated_ocudu) | Branch `nvcuda_accel_02` at `27e84cb3da65205b55dbd56ace6fac93a035331e` |
| `sionna` | `ext/sionna` | [NVlabs/sionna](https://github.com/NVlabs/sionna) | Tag `v2.2.0` at `2f0a3820269b53fd022780a6f2f5f720f7ceb9b4` |
| `sionna-rt` | `ext/sionna-rt` | [NVlabs/sionna-rt](https://github.com/NVlabs/sionna-rt) | Tag `v2.2.0` at `15b5ee036917a4c2a9b6420e05570b77717e2fcc` |

Sprint 0 acceptance on the DGX Spark (2026-11-19) verified all three trees pristine at these commits. OCUDU supplies the C++ and CUDA source for the Sprint 1 native build. Sionna and Sionna RT are installed into the `isac` environment from these directories.

## Fetch and verify

Run these from the repository root.

| Situation | Command |
|---|---|
| Fresh Git clone with network access | `make hydrate` |
| Sources already present | `python scripts/check_pins.py --worktrees` |
| Lock and submodule declarations only, no source needed | `make pins` |
| No Git access to the upstreams | `python scripts/source_archives.py /path/to/zip-directory` |

`make hydrate` runs `make pins`, fetches every declared submodule at its gitlink commit and then runs the worktree check. `make pins` compares `lock.json` with `.gitmodules` and with the gitlinks committed in this repository. The worktree check also requires each checkout to sit at its locked commit with no edited tracked file. Add `--only ocudu`, `--only sionna` or `--only sionna-rt` to check one source.

**Do not use `--recursive`** when cloning or updating. Sionna declares its own nested copy of Sionna RT, and that nested directory must stay empty. This project installs Sionna RT from `ext/sionna-rt` in the same pip transaction as Sionna. The simulation gate then compares every installed Sionna and Sionna RT file with these two trees.

**If a fetch is interrupted**, a source can be left cloned with no checked-out files while Git already treats it as current. This happens when the upstream default branch sits at the pinned commit, as Sionna's does. Rerunning `make hydrate` does not repair it, and the worktree check reports `expected pristine checkout`. On a clone with no local edits in `ext/`, restore the pinned trees and check again:

```bash
git submodule update --init --checkout --force
python scripts/check_pins.py --worktrees
```

`--force` discards edits to tracked files inside the sources, so use it only when there are none to keep.

Installing the full profile leaves `build/` and `*.egg-info/` inside the two Sionna trees. Both upstreams ignore those names, so the trees still pass the worktree check.

## The lock owns the paths

`lock.json` (format 3) is the only place that names a checkout directory. Each entry carries a logical `name` (`ocudu`, `sionna` or `sionna-rt`) that the build, bootstrap and verification tools look up, and a `path` directly under `ext/` that can be any directory name. `python scripts/check_pins.py --path ocudu` prints the resolved directory.

To rename a checkout, move the submodule with `git mv`, set the same `path` in `lock.json`, commit both and rerun `make pins`.

## Move to a new upstream revision

Keep the pinned sources pristine. Do not place dependency code in this repository's index as ordinary files. Apply any later integration patch in a separate worktree or copy and keep its digest.

1. Check out the new commit inside the submodule directory.
2. Set the same `commit` and `source_ref` in `lock.json`. Update the two archive digests in the same change so the ZIP route keeps matching.
3. Stage the submodule directory and `lock.json`, commit them together and run `make pins`.
4. Rerun the gates that depend on the source. For Sionna or Sionna RT, run `make env-full` and then `make accept-s0` with the host's backends. If the new revision changes its dependency pins, review `environments/full-constraints.txt`. For OCUDU, rerun the native build and the Sprint 1 baseline.

A new Sionna RT commit can keep the version number 2.2.0. The simulation gate compares installed files with the pinned tree for that reason, and it fails until `make env-full` reinstalls from the new commit.

## ZIP route

A host without Git access to the upstreams can import the three original ZIP archives instead. `scripts/source_archives.py` checks each archive against the SHA-256 and revision comment in `lock.json`. It preserves symlinks and executable files and stores a file manifest under `build/source-receipts/`. Later worktree checks read every imported file and verify its type, executable bit and SHA-256. Keep the receipts with an imported tree. Generated build products are outside this check.

This route was not exercised during Sprint 0 acceptance. The archive for Sionna RT must be the `v2.2.0` snapshot named in the lock. A snapshot of `main` carries another commit and is refused. The route supplies source code only. Compiler packages, Python wheels and native test dependencies still need a package index or a local mirror.
