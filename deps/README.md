# Pinned source dependencies

The parent commit tracks three mode-160000 Git entries. Their pinned revisions and clone URLs are readable in [lock.json](lock.json). A source ZIP produced by GitHub may omit the parent Git metadata. For that case use a clone of the contribution repository. The initial distribution archive includes a working parent repository with Git metadata.

```sh
make pins                 # Validate parent gitlinks and URLs offline
make hydrate              # Fetch the exact three worktrees from the upstreams
git submodule status
```

WG1's archived CUDA branch provides the build baseline. Check the published commit before building. An inaccessible GitLab revision is a source-acquisition failure and must be reported with the SHA. A later branch tip is a separate dependency update. The pinned Sionna checkout also declares its own nested RT checkout. Install from the project's direct `deps/sionna-rt` checkout. The three top-level submodules suffice for the initial research layout. For a dependency update, change the parent gitlink and lock in the same commit, run the source and smoke checks, then repeat the relevant hardware or trace compatibility gate.
