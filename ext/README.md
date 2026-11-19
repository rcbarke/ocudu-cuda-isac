# External sources

The three mode-160000 entries in the parent checkout pin upstream revisions. The matching URLs, commits, and original ZIP checksums are in [lock.json](lock.json). The supplied project ZIP omitted the original parent's Git database. This distribution includes a new local Git checkout with the three gitlinks restored under `ext/`. Its history starts at this handoff. Configure a remote only if you intend to publish this history.

Activate the repository's `isac` environment and choose one source-acquisition method in a fresh checkout.

| Source available | Command from the project root |
|---|---|
| Network and parent Git metadata | `make hydrate` |
| The three original upstream ZIPs in a directory | `python scripts/source_archives.py /path/to/zip-directory` |
| Sources already present | `python scripts/check_pins.py --worktrees` |

`make pins` verifies all lock entries and submodule declarations. In a Git checkout it also checks the committed gitlinks. A source export without the parent `.git` can use the ZIP route. Archive import checks each supplied ZIP against its locked SHA-256 and revision comment. It preserves symlinks and executable files and stores a file manifest under `build/source-receipts/`. Subsequent worktree checks read each original source file and verify its type, executable bit and SHA-256. Keep the receipts with an imported source tree. Extra generated build products are outside this integrity check. Git mode verifies the checked-out commit and tracked-file cleanliness. Use `--only ocudu`, `--only sionna`, or `--only sionna-rt` with `--worktrees` for one installed dependency.

The source trees are populated on the development host. Do not place dependency code in the parent Git index as ordinary files. For a new upstream revision update the gitlink, declared source, archive digest and extracted-manifest digest together, then rerun the affected native or simulation gates. Keep the pinned native control pristine. Apply any later integration patch in a separate worktree or copy, retaining its digest. The offline route supplies source code. Compiler packages, Python wheels and native test dependencies may still require a package index or local mirror.
