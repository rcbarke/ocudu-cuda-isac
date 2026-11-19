# Contributing

Start with [DEVELOPMENT.md](docs/dev/DEVELOPMENT.md). Use one issue per sprint increment. Include the intended behavior, upstream symbol and pinned revision, test evidence and any observed failure in the pull request. Include a regression with a changed observation schema or PHY seam. Capture contract changes in [REPLAY.md](docs/interfaces/REPLAY.md).

Keep source dependencies at the committed revisions. Submit minimal integration changes as reviewable patches based on those revisions. Put application logic outside the dependency checkout. The hosted CI job verifies CPU contracts. Hardware evidence accompanies the relevant sprint gate. Cite external code and data in [the sources ledger](docs/research/REFERENCES.md) and retain its notices.
