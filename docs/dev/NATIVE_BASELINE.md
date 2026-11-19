# Phase S1 · Build OCUDU and reproduce one DeepSig receiver result

The deliverable is a locally measured, digest-identified CPU/CUDA comparison from the pinned OCUDU receiver. Use the **20 MHz, one-layer PUSCH row in Table II** of Pennybacker, Liu, Kharchenko and O’Shea. Their experiment reported 200 successful decodes on each backend and zero byte mismatches. The small row avoids coupling this bring-up to a radio, core, scene, observer or trained model. The first four-RX sensing demonstration has a different workload. [Paper, §VI-A, Table II and Appendix](https://arxiv.org/abs/2608.04338v1) · [Source audit](../integration/OCUDU.md)

| Sprint | Deliverable | Component check, runnable on a CPU developer machine | Integration and evidence owner |
|---|---|---|---|
| S1.1 | Full out-of-source native Release build, discoverable from the active `isac` environment | Build contract and identity tests exercise synthetic CMake outputs | Spark operator retains successful configure/build logs, cache, executable SHA and native help |
| S1.2 | Single-row paper comparison and original four-RX smoke profile | Transcript validator tests exercise complete, corrupted, truncated and mismatched records | Spark operator acquires native run, retains raw log and checks observed values against the published result |

S1 is a laboratory gate before interpreting results from new receiver-linked ISAC components. Its parser and build-wrapper tickets are independently testable before hardware is reserved. A missing Spark leaves native evidence pending. Preserve the failed run directory and stage whenever an acquisition fails.

## S1.1 · Build and identify the native executable

**Entry.** Complete S0 source materialization and the project `isac` venv. For the developer’s initial end-to-end installation, also exercise the pinned Sionna PHY and RT probes. The native CMake invocation itself consumes only the OCUDU source and host toolchain. Use the environment runbook for driver, toolkit, compiler, apt packages, PyTorch platform wheel and per-host setup. [Environment runbook](ENVIRONMENT.md)

**Public interface.** Run from the repository root. The unpatched source is `ext/ocudu`. Build products are local to the selected host and path. The wrapper uses separate directories for Spark and other machines, validates the selected device and `nvcc` code-generation target, and locks the build directory while writing or measuring.

```bash
source isac/bin/activate
python scripts/check_pins.py --worktrees --only ocudu
bash scripts/build_ocudu.sh
command -v gnb pusch_e2e_pipeline_test
gnb --version
pusch_e2e_pipeline_test --help
python scripts/verify_environment.py --profile simulation --backend cuda --with-native
```

The Spark profile requires AArch64 and GB10 compute capability 12.1. The proposal expresses its build from the native source root as follows.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 \
  -DMCPU=neoverse-v2
cmake --build build -j8
```

The wrapper sets `-S ext/ocudu -B build/ocudu-spark` using absolute paths, preserving the dependency tree. It adds `BUILD_TESTING=ON` for the upstream PUSCH target, keeps mbedTLS/SCTP available, and disables optional UHD, ZeroMQ and external plugin discovery for this radio-free upstream control. It builds the default **all** target using eight jobs. The Spark CPU flag is the upstream workaround for the host GCC’s target support. Local host drivers, headers and toolkit remain prerequisites for an activated venv. [Upstream recipe](../../ext/ocudu/docs/phy_cuda_acceleration.md) · [Integration target](../../ext/ocudu/tests/integrationtests/phy/upper/channel_processors/CMakeLists.txt)

**Other build profiles.** Use `bash scripts/build_ocudu.sh --machine local` for a different supported GPU/CPU. The wrapper discovers compute capability through `nvidia-smi`, verifies matching `sm_*` support in `nvcc --list-gpu-code` and leaves the host CPU setting at upstream defaults. On AArch64, supply `--mcpu TARGET` when the compiler requires a host-specific setting. For multiple GPUs, export `CUDA_VISIBLE_DEVICES` as the full UUID of the intended physical GPU. Use `--build-dir DIR` for separate builds per hardware/toolchain and check the cache. The physical-device build does not allocate a MIG instance. A build on another architecture supplies a portability result on that machine. The DeepSig row is assigned to GB10.

**Unit check U1-build.** Exercise injected CMake cache values in `tests/audit/test_native_baseline.py`. A correct cache issues the full build command and links the gNB and native test into `isac/bin`. A cache that sets `CUDA_ACCEL_FOUND=FALSE` must fail before invoking the build. Alter the built file after creating its identity record and require `--skip-build` validation to reject it.

```bash
python -m unittest discover -s tests/audit -p 'test_native_baseline.py' -v
```

**Acceptance I1-build.** On the named Spark, confirm the build status is completed, CMake reports `Release`, `ENABLE_CUDA=ON`, `CUDA_ACCEL_FOUND=TRUE`, architecture `121`, and `MCPU=neoverse-v2`. Confirm `command -v` resolves both tools inside `isac/bin/`. The `gnb` link must resolve to `build/ocudu-spark/apps/gnb/gnb` and the PUSCH link to the locked integration-test build. The full build must load the gNB and emit its version banner in `gnb-version.txt`. For an archive build the binary may have an empty Git hash. The wrapper prevents version lookup from inheriting the enclosing project’s commit and records the verified archive identity. The venv-resolved executable must load, print the resident-test option on `--help`, and match the SHA-256 in `native-build.json`. Hand the run directory, build directory and exact source identity to S1.2. A test-target-only build using `--target baseline` suffices for an isolated diagnostic and records `target=baseline`. Use the full build to close I1-build. [Build implementation](../../scripts/native_ocudu.py)

## S1.2 · Reproduce the 20 MHz PUSCH correctness row

**Entry.** Use the completed S1.1 Spark build and an idle test GPU. Stop simultaneous gNB, training or inference processes for this comparison. Record any remaining GPU clients, CPU governor and clocks. Use the allowed high-capacity CPU set `5-9,15-19` reported in the paper. The runner checks it is available before starting `taskset`. [Paper, Appendix](https://arxiv.org/abs/2608.04338v1)

| Property | Paper comparator | Operational choice in the pinned test |
|---|---|---|
| Native revision | `wg/nvcuda_accel_01` at `3c89cdee` | `nvcuda_accel_02` at `27e84cb3da65205b55dbd56ace6fac93a035331e`. The branch documentation’s `9fd4047b43` is a third measurement revision. |
| Radio grant | 20 MHz class, 30 kHz SCS, 1 layer, 256-QAM table, MCS 27, two Type-1 DM-RS symbols, TBS 54.3 kbit | 51 PRBs, 1 RX port, pilot symbols 2 and 11, no transform precoding. Native header must report 54,296 bits (6,787 bytes), 256QAM. |
| Receiver and sample count | Direct-managed input, pinned GPU output, CPU/GPU decode on matched frames, 200 measured | Generated single-tap signal, 30 dB SINR, five warmup iterations, every frame on both backends, 200 measured, resident stage timing enabled. The 30 dB operating point, warmup and RX-port choice are explicit local settings where Table II does not specify them. |
| Published row | 200 CRC successes on each backend, no CPU/GPU byte mismatches | Require this correctness result **and** 200 resident decodes/owned grids/immutable snapshots plus zero guard failures. |
| Latency | Mean completed CPU work 334.0 µs, GPU work 183.9 µs, ratio 1.82× | Extract min/mean/max from the native totals, report actual means, speedup and deviations. Compare direction separately. The reference values do not serve as fixed timing thresholds across revisions. |

The pinned harness uses a compatible PDSCH-side grid builder for this non-transform-precoded case and its random generator does not expose a seed switch. CPU and GPU receive the same generated realization within the experiment. The byte check compares the two decoded outputs. S2 adds the independent legal-PUSCH reference and checks each decoded output against retained TX bytes. [Pinned native fixture](../../ext/ocudu/tests/integrationtests/phy/upper/channel_processors/pusch_e2e_pipeline_test.cpp)

```bash
source isac/bin/activate
bash scripts/run_spark_baseline.sh --profile paper --skip-build
```

**Outputs.** The command prints a new `results/runs/spark-pusch.*` directory. `commands.sh` contains the exact launcher, flags and environment overrides. `platform.json`, `nvidia-smi.txt`, compiler/toolkit version files, `pins.txt`, `native-build.json` and `CMakeCache.txt` identify the host, source and build. `pusch.log` is raw combined output. `verification.txt` records correctness, and `table-ii.json` includes the observed and published means, percent differences and whether the observed GPU completed work faster. The associated build directory’s run record holds configure/build logs. `status.txt` records exit stage and status for both successful and failed runs.

**Unit check U1.1.** Run both audit modules without a GPU. Inject a wrong MCS, TBS, RX count, output-buffer mode or iteration count. Truncate or duplicate a result field, reduce residency, flip a CRC verdict, corrupt the guard count or use an inconsistent speedup. Each must reject the transcript. A slower GPU must still emit its measured comparison when the correctness checks pass. Fixtures supply parser evidence only.

```bash
python -m unittest discover -s tests/audit -p 'test_*.py' -v
```

**Acceptance I1.1, I1.2.** Require a zero native process status followed by a zero validator status on the same log, `stage=completed`, 200/200 on both CRC paths, zero disagreements and byte mismatches, and the full resident/direct-grid counters. The workload header, TBS and positive finite completed-work latencies must match the row definition. Publish observed means, their ratio and deviation even if performance regresses. A functional reproduction closes the environment gate. A speedup claim additionally requires local evidence for the claimed direction and magnitude. Without `--skip-build`, a run reuses a verified existing build or builds the PUSCH target if no build receipt exists. An invalid existing receipt requires an explicit rebuild. This keeps a successful full-build receipt intact when running the shorter smoke. The high-SINR smoke remains available as a short diagnostic. It uses four RX ports, MCS 10, 20 measured frames, and checks ownership counters separately from this paper row. [Log grader](../../scripts/verify_pusch_log.py)

```bash
bash scripts/run_spark_baseline.sh --profile smoke --skip-build
# Regrade an archived acquisition after substituting the path printed by its runner.
python scripts/verify_pusch_log.py PATH/TO/pusch.log --iterations 200 --paper-json PATH/TO/comparison-review.json
```

**Recovery.** For missing libraries or targets inspect `configure.log` and the host dependency list. For `no kernel image`, compare the selected GPU, NVCC supported codes, CMake target and binary identity. Use a new build directory after changing compiler or platform. If the process exits with success but suppresses the GPU factory or prints a failed assertion/summary, inspect `pusch.log` and retain the failing run. A source/cache/binary mismatch requires a new build and fresh identity, never editing the acquisition. A CPU-affinity failure requires running within a Spark CPU allocation that includes the measured cores.
