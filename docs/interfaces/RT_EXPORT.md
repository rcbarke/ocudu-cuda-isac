# Offline Sionna RT channel export: the diagnostic pair

This page describes the diagnostic of `scripts/export_rt_episode.py`: four uniform times and twelve tones. It is an axis test and is kept as a regression. The channel of an NR receiver trial, on the subcarriers and symbol times of an uplink grant, comes from the same script with `--pair` or `--set` and is described in [Channel episodes](EPISODE.md).

The exporter computes a paired bistatic channel for a single transmitting UE and a stationary four-element gNB receive array. It writes the complex frequency response `H` needed by a downstream replay producer. The present and absent scenes share the same free-space endpoints. Only the present scene contains the off-axis, moving, constant-RCS cuboid. The exporter rebuilds the background in each scene so the target geometry participates in visibility tests.

## Run on a provisioned RT host

Activate this checkout’s `isac` environment, run the full profile and verify its simulation backend using [the environment runbook](../dev/ENVIRONMENT.md). The script uses procedural geometry and runs offline. Choose a fresh output directory.

```bash
python3 scripts/export_rt_episode.py --output-dir build/rt-episode
```

Output appears only after both cases finish. The offline `manifest.json` identifies `a_h.npy` (target absent) and `b_h.npy` (target present), hashes each complete NPY file with SHA-256, and records solver options, geometry, physical coordinates, units, software versions and the Mitsuba variant. The script requires a valid direct path in both scenes, a valid scattering path in the present scene, finite coefficients and distinguishable outputs. Both solvers use deterministic mode and seed 42. Archive the manifest and its original files together for byte-identical replay across hosts.

| Field | Interpretation |
|---|---|
| Input geometry | UE at (0, 0, 1.5) m and gNB at (20, 0, 3) m in Sionna world coordinates. The 0.3 m target cube starts at (9, 4, 1.5) m and moves at (0, 1, 0) m/s. Its center has a constant 1 m² bistatic RCS. |
| Propagation | `PathSolver` computes direct background propagation. In the present scene, `RCSSolver` allows one target interaction with unobstructed legs. `Paths.concat` combines both sets before CFR evaluation. |
| Carrier and array | 3.5 GHz, vertical isotropic elements and synthetic array phase. The four receiver positions along local `y` are −0.75, −0.25, +0.25 and +0.75 wavelengths in array index order. |
| Time and frequency | Four model times of 0, 0.5, 1 and 1.5 ms measured from the geometry snapshot. Twelve baseband bins span −180 to +150 kHz at 30 kHz spacing. Absolute propagation delay and physical gain are preserved. |
| Stored tensor | Little-endian complex64 in C order. Shape `[4, 1, 4, 12]` represents `[rx_element, tx_element, time, frequency_offset]`. The exporter indexes the singleton radio-device axes explicitly and retains the singleton transmit-element axis. |

These are four uniformly spaced channel snapshots. At 360 kHz and 1.5 ms the diagnostic resolves neither its 5 ns of excess delay nor its 9 Hz of Doppler. For an NR receiver trial, the grant-aligned pair evaluates the channel at the frequencies and symbol times of the executed grant and stores the noise beside it, for `Y=HX+W`. The model time axis already includes Doppler phase. Keep `H`, the case mapping and target geometry in offline stimulus generation and scoring. The live estimator should receive its receiver observations and causal grant metadata. [Replay interface](REPLAY.md)

## Validate the portable file contract

```bash
python3 -m unittest discover -s tests -p 'test_rt_export.py' -v
```

The tests use known complex values to check antenna, time and frequency indexing, the on-disk shape, digest and byte order. They verify both file references, the declared time and frequency grids, and that a failed export leaves no partial pair. They also check rejection of invalid ranks/order, real inputs and nonfinite coefficients. NumPy is optional for the rest of the scaffold and required for these tests.

**Execution status.** The NumPy contract tests run in the unit suite. The ray-traced export ran in the pinned runtime on a development host (x86_64, LLVM variant) during Sprint 2.2: the direct path matched the free-space gain within 1.7e-07 and the delay within 0.002 ns, and the return had the excess delay and Doppler shift of the geometry, 5.314 ns and -8.699 Hz. `make accept-s2.2` repeats this as its first case on the lab host. `--variant` selects the Mitsuba variant. The portable tests provide no evidence of native PUSCH decoding, detection accuracy or wall-clock latency.

The upstream tutorial combines target and background paths on a shared scene. The pinned API source specifies the six-axis CFR order and per-path time evolution. [RCS tutorial](https://nvlabs.github.io/sionna/v2.2.0/rt/tutorials/RCS.html) · [CFR source](https://nvlabs.github.io/sionna/v2.2.0/_modules/sionna/rt/path_solvers/paths.html) · [RCS solver](https://nvlabs.github.io/sionna/v2.2.0/rt/api/rcs.html) · [PathSolver](https://nvlabs.github.io/sionna/v2.2.0/rt/api/paths_solvers.html)
