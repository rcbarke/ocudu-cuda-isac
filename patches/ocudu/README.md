# Upstream integration patches

Keep each patch small, named for the upstream commit and seam it extends. Record its reason, touched symbols, test, ownership/lifetime contract, original revision and SHA-256. Check `git apply --check` against a clean worktree before applying in a dedicated build checkout. Preserve a zero-change upstream baseline. Use the resident estimator's real producer output and an explicit event when adding O3. The source audit is in [OCUDU.md](../../docs/integration/OCUDU.md).

## The series

[`series.json`](series.json) is the manifest (`ocudu-patch-series/v1`). It names the base commit, which must be the commit that `ext/lock.json` pins for `ocudu`, and for every patch its file, its SHA-256, its reason, the symbols it touches, the ownership contract and the hash of every touched file before and after. `scripts/receiver_patch.py` reads it, and nothing else decides what a patched receiver is.

| Patch | Sprint | Touches | What it adds |
|---|---|---|---|
| [`0001-pusch-resident-estimate-tap.patch`](0001-pusch-resident-estimate-tap.patch) | S5.2 `native_estimate` | one new header (88 lines), 49 lines in `pusch_demodulator_gpu_impl.cpp`, no kernel | A read-only tap on the channel estimate that the resident PUSCH demodulator leaves on the device (`d_estimates_fp16`): after pilot LS, frequency interpolation and frequency smoothing, per DM-RS symbol, before time interpolation. Without a registered observer the demodulator makes one atomic load per PDU. |

## How a patch is used

The submodule is never patched. `python scripts/native_ocudu.py build --patched` makes a separate worktree of the pinned commit in `build/ocudu-src-o3` (a copy for a source that was installed from the verified archive), applies the series there, checks every touched file against the manifest and builds that tree into `build/ocudu-spark-o3`. The receipt of that build, `isac-native.json`, carries the patch identity under `patch`. The links in `isac/bin` keep pointing at the accepted Sprint 1.1 build, which is the hook-free control.

```bash
python scripts/receiver_patch.py check                  # the manifest against the files and the lock
python scripts/receiver_patch.py check --apply-to DIR   # also: git apply --check on a clean tree of the pin
python scripts/native_ocudu.py build --patched          # the patched receiver, in its own directories
python scripts/native_adapter.py build --o3             # the adapter against it, with the O3 route
```

## Adding or changing a patch

1. Make the change in a worktree of the pinned commit, never in `ext/`.
2. Write the patch with `git diff` from that worktree, with the new files added by `git add -N`.
3. Update `series.json`: the SHA-256 of the patch file and of every touched file after the series. `python scripts/receiver_patch.py check` names what disagrees.
4. A patch that changes the interface of an earlier one raises that interface's version, and the adapter refuses to build against another version (`static_assert` in `native/ocudu_adapter/observer/estimate_tap.h`).
5. The gate of the sprint that owns the patch runs the patched receiver against the unpatched build: same keys, same CRC flags and same decoded bytes with the observer off and on.
