# The first advisor demonstration

**What the audience should see.** On one DGX Spark, a known uplink payload is decoded by the CUDA-OCUDU receiver while an independent batch-one CUDA sensing process raises a presence score as a moving return enters the scene. Switch to a matched target-absent replay, then slow the sensing process to show bounded drops and a continuing PHY.

## Delivery checkpoints

| Checkpoint | Live action | Evidence to show |
|---|---|---|
| Available now, CPU | Run `make check`. Open the smoke ingress records and timing summary. | The fixture applies a known H to X, joins scheduled observations, and accounts for missing/late outputs. Times and inputs are synthetic. |
| Native receiver baseline, S1 | Follow [SPARK.md](SPARK.md). Run the supplied PUSCH test and grade stdout. | Exact revision, GPU target, selected CUDA path, positive high-SINR decode, CPU/GPU payload agreement, transcript and counters. |
| First integrated showing, S4 | Run the same legal grant stream through an upstream-backed receiver process. Replay two held-out episodes and start the MPS sensing client. | True decoded bytes/CRC, O2 provenance or validated O3 provenance, two MPS clients, scene-independent evidence join and live display of decisions. Specify whether grants come from the native harness or CU/DU scheduler. |
| Operational DU, S5 | Repeat through the actual DU scheduling path, with nonblocking export and sustained load. | Scheduler allocations, backend/queue telemetry, counted overruns and a sustained matched baseline. |

## Three-minute walkthrough after S4

1. Display the run manifest, machine, MPS client IDs and common radio profile. Show measured DSP-only service and the declared processing allowance.
2. Replay a held-out target-absent episode. Show actual decoded block and checksum/CRC, pilot-energy or delay–Doppler view, and the sensor score relative to its frozen threshold.
3. Replay the matched present case. Show a rising score and the coarse delay/Doppler return in the expected bistatic bin. Overlay truth only in the evaluator view after decisions have been logged.
4. Compare DSP-only and concurrent runs at the same offered load. Show decision counts and a compact first-sample-to-publication latency distribution. Induce one queue overflow during a separate diagnostic and show that the DU keeps advancing.

## Demonstration record

Use a one-layer, four-RX, one-UE profile supported by the pinned receiver. Start at 20 MHz and 30 kHz SCS if supported with the selected grant and pilot geometry. Use one observation opportunity every 5 ms where the UL pattern permits. The initial presence model is a small frozen PyTorch head trained on pilot-derived features from disjoint generated episodes. Record weights, preprocessing, pilot schedule, score threshold and acquisition history. Show two illustrative episodes live and include independent test episodes in the archived evidence. The demo budget, sample counts and pass criteria are specified in [S4](../dev/DEVELOPMENT.md#phase-s4).

The 5GC can reside on the Spark. Identify its state as idle or active in the manifest. Only a protocol-capable UE that registers and establishes a session can supply an end-to-end user-plane result. Submit screenshots alongside checksums and machine-readable events so reviewers can reproduce the presentation from the measurements.
