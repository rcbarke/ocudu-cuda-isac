# DGX Spark receiver bring-up

Start from a clean, locked `ext/ocudu-deepsig-cuda-inline` and the repository-local `isac` environment. The first hardware result is the 20 MHz, one-layer correctness row from DeepSig Table II. The advisor demonstration later uses four RX ports, matched target-present/absent episodes and a trained detector linked to the actual receiver. [Environment setup](../dev/ENVIRONMENT.md) · [Native sprints, grading and troubleshooting](../dev/NATIVE_BASELINE.md)

## 1. Build and verify the native environment

```bash
source isac/bin/activate
make accept-s1.1 BACKENDS="cpu cuda"
```

The gate builds, grades the evidence and ends with `Sprint 1.1: ACCEPTED on this DGX Spark.` The full Release build uses eight jobs by default, CUDA architecture 121 and the upstream Spark CPU target. Use `JOBS=2` if memory is limited. `bash scripts/build_ocudu.sh` runs the build alone. Use a separate build directory for other toolchains or GPUs. The emitted build record links the clean OCUDU source identity, native executable and CMake cache. [Build entry point](../../scripts/build_ocudu.sh)

## 2. Run the paper comparison and four-RX diagnostic

```bash
make accept-s1.2
# One profile alone, for diagnosis
bash scripts/run_spark_baseline.sh --profile paper --skip-build
bash scripts/run_spark_baseline.sh --profile smoke --skip-build
```

The gate measures both profiles on the build that step 1 accepted, grades the archived evidence and ends with `Sprint 1.2: ACCEPTED on this DGX Spark.` and the Sprint 1 verdict. Stop gNB, training and inference processes first.

The **paper** profile uses 51 PRBs at 30 kHz SCS, MCS 27 from the 256-QAM table, 1 layer/1 RX, a direct-managed grid and pinned output. It checks 200 measured CPU/GPU decodes after warmup and reports actual CPU/GPU mean latency alongside the paper’s 334.0/183.9 µs. The **smoke** profile uses MCS 10 from the 64-QAM table, four RX ports and 20 measured frames. It remains the default for `bash scripts/run_spark_baseline.sh`, which can build the test target on demand. Use the same `--build-dir DIR` in both wrappers when overriding the default. [DeepSig, Table II](https://arxiv.org/abs/2608.04338v1)

Each command prints a unique evidence directory. The gate writes `results/runs/s12-<host>-<UTC time>/` with `summary.txt`, `paper/` and `smoke/`. Require the successful stage/status, native log with real CRC successes, zero CPU/GPU byte mismatches, and all measured frames on the resident/owned-grid path. Read `table-ii.json` for the paper profile’s actual means and timing differences. The supplied branch and paper measurement commit differ. The stock test generates its own input and compares CPU/GPU decoded bytes. S2 establishes replay against retained original TX payloads before sensing results are attributed to a real PUSCH observation. [Acceptance and regrade commands](../dev/NATIVE_BASELINE.md)

## 3. Build the adapter and archive the grant record

```bash
make accept-s2.1
```

The gate builds the project's fixture adapter out of tree against the build that step 1 accepted, feeds one known transport block to the pinned OCUDU transmit chain and to Sionna NR for each grant in `configs/grants/`, compares the two grids and archives the grant, the bytes and both grids. It ends with `Sprint 2.1: ACCEPTED on this DGX Spark.` It runs on the CPU. [Grant record](../interfaces/GRANT.md)

## 4. Demonstrate receiver-linked sensing after the native adapter is implemented

| Step | Evidence to retain |
| --- | --- |
| Replay matched target-absent and target-present episodes into the staged upstream PUSCH processor. Use one fixed UE and identical allocation, waveform and noise conditions. | Episode and waveform IDs, receiver-ingress hash, grant, actual decoded CRC and payload agreement. Label the invocation `staged receiver` and 5GC activity `idle` or `absent` as applicable. |
| Fork the actual ingress grid with protected ownership. Compute single-layer O2 LS on scheduled DM-RS and retain the RX and sparse pilot axes. | Pilot sequence and mask parity, record sequence, readiness and timestamp. Log late samples, stale records and ring drops. O2 is receiver-ingress pilot LS. A future O3 internal estimate requires independent producer-side validation. |
| Apply a simple calibrated delay–Doppler transform and run one compact batch-one learned presence head on the same past/present window. Export the receiver's own estimate (O3, S5.2) from a separately built patched receiver and run the cuSense-like front end and location model (S3.5) on it. | Display the two episodes, estimate and score from their shared O2 input, a frozen threshold, model hash and measured observation age. Feed only causally available receiver data to the model. Use separate truth for scoring. |
| Compare DSP-only execution to DSP plus inference on the same Spark under MPS. | Record each CUDA process's MPS attachment, CPU affinity, GPU caps, completed work, decision latency, CRCs and deadline misses. Keep acquisition and post-window delay separate. |
| Start the KPI aggregator and the page server. View the page from a second machine through an SSH tunnel. | `kpi.jsonl`, its parity with the scorecard, and the page-off, remote and local arms. The aggregator and server hold no CUDA context. [KPI contract](../interfaces/KPI.md) |

The run becomes a scheduler-paced CU/DU demonstration once it executes timed grants through the intended stack and accounts for wall-clock overruns. Report the stage achieved on hardware in the run manifest. Keep the high-SINR baseline log, captured receiver evidence and full measurement run separate from the host smoke fixtures. [Measurement protocol](../research/EVALUATION.md) · [NVIDIA MPS resource controls](https://docs.nvidia.com/deploy/mps/latest/when-to-use-mps.html)
