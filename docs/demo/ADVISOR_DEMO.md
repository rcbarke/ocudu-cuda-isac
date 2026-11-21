# The first advisor demonstration

**What the audience should see.** On one DGX Spark, a known uplink payload is decoded by the CUDA-OCUDU receiver, and the receiver's own channel estimate appears on the screen. An independent native sensing process, CUDA kernels and a TensorRT engine with no Python in it, runs a cuSense-like pipeline on that estimate, and its latency is on the screen. As a moving return enters the scene its presence score rises and a location map follows the target. A browser page shows the radio, sensing and system KPIs live. Switch to a matched target-absent replay, then slow the sensing process to show bounded drops and a continuing PHY.

**Profile.** One UE, one layer, four RX, 20 MHz/51 PRBs at 30 kHz SCS, one cell, MPS on the GB10. No UE, 5GC or PRACH procedure is involved: a Sionna RT radar CFR applied to a legal grant at the receiver stands in for a connected UE. [Frozen profile](../../experiments/demo/profile.json)

## Delivery checkpoints

| Checkpoint | Live action | Evidence to show |
|---|---|---|
| Available now, CPU | Run `make check`. Open the smoke ingress records and timing summary. | The fixture applies a known H to X, joins scheduled observations, and accounts for missing/late outputs. Times and inputs are synthetic. |
| Native receiver baseline, S1 (accepted) | Follow [SPARK.md](SPARK.md). Run the supplied PUSCH test and grade stdout. | Exact revision, GPU target, selected CUDA path, positive high-SINR decode, CPU/GPU payload agreement, transcript and counters. |
| D1 · Decode through the replayed channel, S2 with S4.4–S4.5 | Replay the moving-target episode into the real PUSCH processor. Open the page. | Original bytes recovered on CPU and GPU, true CRC, service times. The page shows radio KPIs from the run records. |
| D2 · Observation and radar view, S3.1–S3.4 | Start the observer and the sensor process. | The observer's pilot LS (O2), the delay–Doppler map with the return in the expected bistatic cell, a presence score against its frozen threshold. |
| D3 · The receiver's own estimate, S5.2 | Switch to the patched receiver build. | The CUDA receiver's estimate (O3) per RX across the band, its producer trace, its parity with O2 and unchanged CRC and bytes against the unpatched build. |
| D4 · cuSense-like sensing under MPS, S3.5, S3.6, S4.1, S4.6, S4.2–S4.3 | Run the same legal grant stream with the native core, `sense_host`, as the second MPS client. Replay held-out present and absent episodes. | Location map and presence computed from O3, the native latency with the TensorRT engine beside the Python reference, two MPS clients, the calibrated deadlines, DSP-only against concurrent service times, the scorecard. Specify whether grants come from the native harness or the CU/DU scheduler. |
| Operational DU, S5 | Repeat through the actual DU scheduling path, with nonblocking export and sustained load. | Scheduler allocations, backend/queue telemetry, counted overruns and a sustained matched baseline. |

The live showing needs D4. D1 to D3 can be shown as they land. [Critical path](../dev/DEVELOPMENT.md#demonstration-critical-path)

## The sensing pipeline on the screen

| Stage | What it does | Ticket |
|---|---|---|
| Receiver estimate | The CUDA receiver's own estimate per DMRS symbol, after pilot LS, frequency interpolation and smoothing: four RX × 612 subcarriers × two DMRS symbols | S5.2 |
| Observer pilot LS | Channel estimate on the scheduled DMRS of the exact grid passed to PUSCH. The parity reference for the receiver's estimate and the fallback input. | S3.1 |
| Background template | Complex average of target-absent calibration records, fitted on the training split | S3.5 |
| Causal window | Coherent average of the UL slots of one TDD period, then magnitude-domain background subtraction and a z-score | S3.5 |
| Location model | Plain-block 1D CNN to a location grid, with a jointly trained presence logit | S3.5 |
| Native core | The same front end as CUDA kernels and the model as a TensorRT engine in `sense_host`, checked against the Python reference on the same records | S3.6 |
| Radar view | Radar cube and delay–Doppler map from eight updates 5 ms apart, with a linear presence head as the baseline | S3.3, S3.4 |

The template, the window and the CNN follow cuSense's published design, and the input is the same receiver stage cuSense reads from Aerial. Two changes are stated on the page: statistics are fitted on training data only, and target-absent episodes are scored. Call the result cuSense-like. It runs on a different receiver, bandwidth and scene from the published system. [Villa et al.](https://arxiv.org/abs/2512.06493v3)

## Three-minute walkthrough after D4

1. Open the page from a second machine. Read the header: run, machine, MPS client IDs, radio profile, sensing runtime, model hash, threshold and the calibrated deadlines. Show the measured DSP-only service time and the calibrated processing allowance.
2. Show the receiver-estimate panel: the CUDA receiver's estimate across the band on each RX, with the observer's pilots overlaid and the parity figure beside it.
3. Replay a held-out target-absent episode. The radio panel shows the decoded block count, CRC and payload match. The sensing panel shows a flat delay–Doppler map and a score below its frozen threshold.
4. Replay the matched present case. The score rises, the return appears in the expected bistatic cell and the location map follows the target. Turn on the evaluator view only after the decisions are logged, to overlay truth and position error.
5. Switch between the DSP-only and concurrent arms at the same offered load. The system panel shows both MPS clients, and the radio panel shows the change in PUSCH service time. Show the first-sample-to-publication latency against the calibrated deadline, and the native latency by cuSense's boundary beside the reference runtime on the same records.
6. In a separate diagnostic, stall the sensor. The ring fills, the drop counter rises and the radio panel keeps advancing.

## Demonstration record

Use the one-layer, four-RX, one-UE profile supported by the pinned receiver at 20 MHz and 30 kHz SCS. Publish one cuSense-like update per 5 ms TDD period from the UL slots of that period, and one delay–Doppler decision from eight updates. Train on receiver-derived records from the training trajectories of the episode set and freeze the template, normalizer, weights and threshold before any held-out episode is read. Record all of them with the pilot schedule and acquisition history. Show two illustrative episodes live and include the independent test episodes in the archived evidence. The demo budget, sample counts and pass criteria are specified in [S4](../dev/DEVELOPMENT.md#phase-s4).

**Calibrated service levels.** The radio allowance and both sensing deadlines start as proposals. S4.6 measures the implementation on calibration episodes, derives the values by rules fixed in advance, verifies them on a second run and freezes the profile before the evidence run. Show the calibrated values, and the proposals they replaced, in the header.

**Comparable, not replicated.** The pass criteria grade presence. Position error is recorded beside cuSense's published 73 cm and is not graded, as S1.2 recorded Table II timing beside the paper's. The graded branch runs in the native core with a TensorRT engine (S3.6). Its latency is shown by cuSense's boundary and end to end, and is recorded beside the published 241 µs, which belongs to cuSense's native serving option, without being graded. The radar view runs the Python reference and is labeled so. The published position error belongs to one room, 100 MHz and an over-the-air system. At 18.36 MHz of occupied bandwidth the nominal delay resolution here is about 16 m of bistatic path, so room-scale position comes from structure the model learned in this scene. Say so when showing the map.

S1.2 timing was accepted as comparable to the paper with host tuning deferred. Show differences between arms measured in the same session and make no absolute latency claim. S5.6 tunes the host and recalibrates for the paper.

Submit screenshots alongside the recorded KPI stream, checksums and machine-readable events so reviewers can reproduce the presentation from the measurements. [KPI contract](../interfaces/KPI.md)
