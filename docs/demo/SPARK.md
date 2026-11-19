# DGX Spark receiver bring-up

The initial advisor checkpoint uses one Spark, one fixed UE, one PUSCH layer, four receive channels, 20 MHz nominal bandwidth, 51 PRBs and 30 kHz subcarrier spacing. The native baseline below calls the upstream PUSCH processor on a generated workload. The subsequent checkpoint adds project replay and a learned presence detector with a delay–Doppler display. Record them as separate runs. [Advisor walkthrough](ADVISOR_DEMO.md) · [OCUDU integration](../integration/OCUDU.md)

## 1. Establish the project and hardware baseline

At the root of a Git checkout, run `make smoke` and `make test`. These commands exercise the host replay and repository tests. For native CUDA work use a DGX Spark with the development toolkit (`nvcc` with `sm_121` support), working driver, CMake, a supported C++ compiler, GTest, FFTW3, mbedTLS, SCTP and yaml-cpp. Upstream lists Debian/Ubuntu build packages in its [dependency script](../../deps/ocudu/docker/scripts/install_dependencies.sh). No radio hardware or core-network session is needed for the staged test.

## 2. Build and run the high-SINR baseline

From the repository root:

```bash
bash scripts/run_spark_baseline.sh
```

The wrapper verifies the Git tree pins, initializes the OCUDU dependency when needed and checks for a clean upstream checkout. It records GB10 compute capability 12.1, the driver, compiler and CUDA toolkit. It configures a Release build for `sm_121` and `-DMCPU=neoverse-v2`, the upstream GCC workaround for this Spark. Set `--jobs 2` to reduce build concurrency, `--build-dir DIR` to choose another CMake build, or `--output-root DIR` to choose the parent of the per-run folders. All relative paths start at the repository root. [Native wrapper](../../scripts/run_spark_baseline.sh) · [Upstream build instructions](../../deps/ocudu/docs/phy_cuda_acceleration.md)

For the staged test the wrapper builds `pusch_e2e_pipeline_test` with `BUILD_TESTING=ON`, `ENABLE_CUDA=ON`, mbedTLS and SCTP enabled. It checks for `CUDA_ACCEL_FOUND:BOOL=TRUE` in the generated cache. This catches a configure that quietly disabled the requested backend after failing to find a CUDA compiler. A missing target calls for checking the test switches and GTest, mbedTLS and SCTP discovery in the configure log. [CUDA build logic](../../deps/ocudu/lib/CMakeLists.txt) · [Test registration](../../deps/ocudu/tests/integrationtests/phy/upper/channel_processors/CMakeLists.txt)

The native workload uses MCS 10 with the `qam64` table, high SINR of 30 dB, Type-1 DM-RS on symbols 2 and 11, both CPU and GPU every iteration, and five warmups plus 20 measured decodes. It runs `--rx-device-grid direct-managed --require-gpu-resident` with stage timing enabled. The native byte-mismatch counter compares CPU and GPU output. Checking each against the retained transmitted bytes is part of integrated replay. The wrapper limits the native run to five minutes and writes the exact command, platform, dependency pins, CMake cache, build log, run transcript, validator output and exit stage under the printed `results/runs/spark-pusch.*` directory. In preflight failures only the available artifacts and exit stage are recorded.

## 3. Check what the run establishes

The validator requires 20/20 CRC passes on both backends, 20/20 resident GPU decodes, direct-grid and immutable-snapshot use on every measured iteration, zero CRC/byte disagreements and zero output guard failures. Certain upstream failure paths print a failure and return exit code zero. The wrapper checks the native process status **and** the summary. To recheck an individual run, pass its log path:

```bash
python3 scripts/verify_pusch_log.py path/to/pusch.log --iterations 20
```

A passing run ends at `stage=completed` with `exit_code=0`, and its log passes the transcript gate. This strict criterion is specific to the high-SINR smoke profile. A noisy sensitivity sweep has a different expected BLER. Record the DGX OS, power settings, thermal state, active GPU applications, CPU affinity and kernel scheduling policy when measuring performance. [Upstream result counters and final verdict](../../deps/ocudu/tests/integrationtests/phy/upper/channel_processors/pusch_e2e_pipeline_test.cpp)

The stock test uses a single-tap channel and randomized transmit data. At this pin, the selected non-transform-precoded branch builds its transmit grid with a configured PDSCH processor and decodes it using PUSCH. Its passing shared-channel stimulus is specific to the harness. For S2, validate a legal NR uplink grant, payload scrambling and pilot map with independent PUSCH reference vectors before making uplink waveform claims. In direct-managed mode the harness copies generated samples from its host reference onto a producer stream and publishes the owned GPU snapshot. The resulting measurements cover CPU/GPU receiver agreement and that ownership path. They do not yet exercise a moving target, learned inference, a scheduling CU/DU, a UE registration or a per-slot deadline. Treat the test's instrumentation-enabled latency as diagnostic.

## 4. Demonstrate receiver-linked sensing after the native adapter is implemented

| Step | Evidence to retain |
| --- | --- |
| Replay matched target-absent and target-present episodes into the staged upstream PUSCH processor. Use one fixed UE and identical allocation, waveform and noise conditions. | Episode and waveform IDs, receiver-ingress hash, grant, actual decoded CRC and payload agreement. Label the invocation `staged receiver` and 5GC activity `idle` or `absent` as applicable. |
| Fork the actual ingress grid with protected ownership. Compute single-layer O2 LS on scheduled DM-RS and retain the RX and sparse pilot axes. | Pilot sequence and mask parity, record sequence, readiness and timestamp. Log late samples, stale records and ring drops. O2 is receiver-ingress pilot LS. A future O3 internal estimate requires independent producer-side validation. |
| Apply a simple calibrated delay–Doppler transform and run one compact batch-one learned presence head on the same past/present window. | Display the two episodes, estimate and score from their shared O2 input, a frozen threshold, model hash and measured observation age. Feed only causally available receiver data to the model. Use separate truth for scoring. |
| Compare DSP-only execution to DSP plus inference on the same Spark under MPS. | Record each CUDA process's MPS attachment, CPU affinity, GPU caps, completed work, decision latency, CRCs and deadline misses. Keep acquisition and post-window delay separate. |

The run becomes a scheduler-paced CU/DU demonstration once it executes timed grants through the intended stack and accounts for wall-clock overruns. Report the stage achieved on hardware in the run manifest. Keep the high-SINR baseline log, captured receiver evidence and full measurement run separate from the host smoke fixtures. [Measurement protocol](../research/EVALUATION.md) · [NVIDIA MPS resource controls](https://docs.nvidia.com/deploy/mps/latest/when-to-use-mps.html)
