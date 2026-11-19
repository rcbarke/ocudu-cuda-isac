# Receiver-ingress replay

The host fixture checks one boundary of the proposed experiment. It applies a supplied time-indexed channel to transmitted symbols, emits a received resource grid for each scheduled update, and reconciles a separate synthetic publication trace. The replay fixture uses the standard library. Run it inside the local `isac` environment described in [S0](../dev/ENVIRONMENT.md).

## Run the four-update fixture

From the repository root:

```bash
./scripts/run_smoke.sh
python3 -m unittest discover -s tests -p 'test_replay.py' -v
```

The wrapper writes to `build/smoke/`. Use `--output-dir PATH` to choose another directory. Repeating the command overwrites the three fixture outputs with the same bytes. The fixture arithmetic is exact in binary floating point, so every host produces identical files. `tests/test_replay.py` records their SHA-256 digests. To compare a repeat by hand, run `./scripts/run_smoke.sh --output-dir build/smoke-repeat` and `diff -r build/smoke build/smoke-repeat`. For different input files, the underlying entry point is:

```bash
PYTHONPATH=src python3 -m ocudu_cuda_isac \
  --episode configs/smoke_replay.json \
  --trace configs/smoke_timing.json \
  --output-dir build/smoke
```

| Output | Consumer | Contents |
|---|---|---|
| `receiver_ingress.jsonl` | Future receiver adapter | Ordered received grids, logical times, pilot mask, frequency offsets and dimensions |
| `offline_oracle.jsonl` | Offline test evaluator | Known transmitted values, time-indexed channel, noise and synthetic case label |
| `timing_summary.json` | Trace reviewer | Per-update outcomes, latency for published records, denominator and aggregate counts |

The expected counts are four scheduled updates, one on time, one late, one dropped, and one missing. These values come from the fixed trace in `configs/smoke_timing.json`. No wall-clock, GPU, OCUDU decoder, sensing accuracy, or backpressure performance is measured by this fixture.

## Grid contract

The fixture schema `fixture-grid/v1` uses an opaque `episode_id`, ordered integer `seq` keys, and one declared `clock_domain`. A record carries `symbol_times_ns`, its earliest completed-grid `ready_ns`, and a common synthetic `pilot_mask[m,k]`. Frequency offsets are strictly increasing integers in Hz relative to the carrier. Identifiers, times and offsets must fit a signed 64-bit integer. A file with a repeated key or a nonfinite constant is rejected when it is loaded. Every emitted record owns its arrays, so a consumer that edits one record cannot change another record or the source episode. Complex values are JSON pairs `[real, imaginary]`. Arrays are flat, with the last axis varying fastest.

| Field | Axes | Purpose |
|---|---|---|
| `shape` | `[N_rx, N_tx, N_symbols, N_subcarriers]` | Receiver channels, transmitted antenna streams, symbol times, frequency offsets |
| `x` | `[tx, symbol, subcarrier]` | Known transmitted values |
| `h` | `[rx, tx, symbol, subcarrier]` | Complex CFR at the stated symbol times and subcarrier offsets |
| `w` | `[rx, symbol, subcarrier]` | One recorded additive-noise realization |
| `y` | `[rx, symbol, subcarrier]` | Sum of `h * x` over transmit antennas, plus `w` once per receive element |
| `case` | Scalar text | Synthetic test condition, retained only in the offline output |

The checked-in case has four receive channels, one transmit stream, one symbol, and two resource elements per record. It checks complex arithmetic and antenna order, including a controlled fade, receiver-dependent phase and an exact nonzero noise term. Pilot positions are toy inputs. The host fixture does not synthesize NR allocations. The two-stream regression exercises an additional transmit axis.

Associate outputs across processes by `(episode_id, seq)`. The synthetic trace names its own episode, clock and deadline. Events may arrive out of sequence. A published event supplies `publish_ns`; a dropped event supplies a reason. The evaluator rejects unknown/duplicate identifiers, mismatched clock names and publication earlier than grid readiness. Every scheduled source record contributes to the denominator. A missing event remains visible as missing. An update at the exact deadline is on time. Elapsed time starts at the first symbol of its acquisition window. Matching clock *names* exercises a contract check. A hardware experiment also needs a measured clock mapping and uncertainty.

## Connecting a generated scene

A production episode should store a hashed manifest for scene/mesh assets, solver releases and settings, antenna positions/order/calibration, geometry timestamp and validity interval, carrier, delay reference, gain normalization, NR numerology, grant/pilot mapping, and payload/noise configuration. Store checksum-addressed binary tensors with explicit dtype, byte order, axes and units. The small JSON fixtures serve as readable reference vectors.

For Sionna RT 2.2.0, compute target returns with `RCSSolver`, background paths with `PathSolver`, and concatenate paths computed with matching endpoints, array mode and scene configuration. Remove the target geometry before recomputing a physically target-absent scene, since the mesh can also block background rays. When using an analytic reference, record absolute path delays and physical gains. A time-indexed CFR already contains the modeled phase evolution. Apply it once at the recorded symbol times. [Sionna RCS tutorial](https://nvlabs.github.io/sionna/rt/tutorials/RCS.html) · [Paths API](https://nvlabs.github.io/sionna/rt/api/paths.html)

Preload the channel and stimulus on the DU host for matched receiver replay. Frozen `Y` gives common receiver inputs for the same grant and stimulus. If the grant or transmitted symbols change, apply `H` again with the new `X` and declared noise realization. At the OCUDU adapter, implement readiness events, grid lifetime and ownership, and align allocation metadata to the executed receiver. A sensing agent downstream can consume causally available receiver estimates and their actual pilot and time metadata. Keep the offline oracle file independent of that service's inputs. Diagnostic capture of four receiver stages and the timing cost of production observation export require separate measurements.

For a wall-clock test, timestamp first and last acquired samples, receiver output readiness, queue acceptance, transfer completion, inference completion and result publication. Use matched run IDs, retain drops and stale records, and align host timebases when the agent runs on another node. Timing-only outcomes in the example have no detector or target labels and cannot be interpreted as timely detection probability.
