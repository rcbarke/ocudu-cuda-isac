# Upstream integration patches

Keep each patch small, named for the upstream commit and seam it extends. Record its reason, touched symbols, test, ownership/lifetime contract, original revision and SHA-256. Check `git apply --check` against a clean worktree before applying in a dedicated build checkout. Preserve a zero-change upstream baseline. Use the resident estimator's real producer output and an explicit event when adding O3. The source audit is in [OCUDU.md](../../docs/integration/OCUDU.md).
