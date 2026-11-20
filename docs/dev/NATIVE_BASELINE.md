# Phase S1 · Build OCUDU and reproduce one DeepSig receiver result

The deliverable is a locally measured, digest-identified CPU/CUDA comparison from the pinned OCUDU receiver. Use the **20 MHz, one-layer PUSCH row in Table II** of Pennybacker, Liu, Kharchenko and O’Shea. Their experiment reported 200 successful decodes on each backend and zero byte mismatches. The small row avoids coupling this bring-up to a radio, core, scene, observer or trained model. The first four-RX sensing demonstration has a different workload. [Paper, §VI-A, Table II and Appendix](https://arxiv.org/abs/2608.04338v1) · [Source audit](../integration/OCUDU.md)

| Sprint | Deliverable | Component check, runnable on a CPU developer machine | Integration and evidence owner |
|---|---|---|---|
| S1.1 | Full out-of-source native Release build, discoverable from the active `isac` environment | Build contract and identity tests exercise synthetic CMake outputs | Spark operator retains successful configure/build logs, cache, executable SHA and native help |
| S1.2 | Single-row paper comparison and original four-RX smoke profile | Transcript validator tests exercise complete, corrupted, truncated and mismatched records | Spark operator acquires native run, retains raw log and checks observed values against the published result |

S1 is a laboratory gate before interpreting results from new receiver-linked ISAC components. Its parser and build-wrapper tickets are independently testable before hardware is reserved. A missing Spark leaves native evidence pending. Preserve the failed run directory and stage whenever an acquisition fails.

## S1.1 · Build and identify the native executable

**Entry.** Complete S0 source materialization and the project `isac` venv. For the developer’s initial end-to-end installation, also exercise the pinned Sionna PHY and RT probes. The native CMake invocation itself consumes only the OCUDU source and host toolchain. Use the environment runbook for driver, toolkit, compiler, apt packages, PyTorch platform wheel and per-host setup. [Environment runbook](ENVIRONMENT.md)

The pinned configure stops without four development packages: GTest (the test tree is on), yaml-cpp, SCTP and mbedTLS. FFTW is optional and used when present. Install them once per host before the first build.

```bash
sudo apt-get install build-essential cmake pkg-config libfftw3-dev libmbedtls-dev \
  libsctp-dev libyaml-cpp-dev libgtest-dev
```

**Public interface.** Run from the repository root. The unpatched source is the `ocudu` entry of `ext/lock.json`. Build products are local to the selected host and path. One command builds, grades and archives the sprint. The individual commands stay available for diagnosis. Both print everything CMake prints while the configure and the build run, and keep it in `configure.log` and `build.log`. The wrapper uses separate build directories for Spark and other machines, validates the selected device and the toolchain before CMake runs, and locks the build directory while writing or measuring.

```bash
source isac/bin/activate
make accept-s1.1 BACKENDS="cpu cuda"     # collocated Spark that passed both S0 simulation backends
make accept-s1.1                         # receiver-only Spark
# Step by step
python scripts/check_pins.py --worktrees --only ocudu
bash scripts/build_ocudu.sh              # prints "Run artifacts: results/runs/ocudu-build.XXXXXXXX"
python scripts/native_ocudu.py grade results/runs/ocudu-build.XXXXXXXX
command -v gnb pusch_e2e_pipeline_test
gnb --version
pusch_e2e_pipeline_test --help
python scripts/verify_environment.py --profile core --with-native --report build/native-env.json
```

The Spark profile requires AArch64 and GB10 compute capability 12.1. The proposal expresses its build from the native source root as follows.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 \
  -DMCPU=neoverse-v2
cmake --build build -j8
```

The wrapper sets `-S` to the locked OCUDU directory and `-B build/ocudu-spark` using absolute paths, preserving the dependency tree. It adds `BUILD_TESTING=ON` for the upstream PUSCH target, keeps mbedTLS/SCTP available, and disables optional UHD, ZeroMQ and external plugin discovery for this radio-free upstream control. It also sets `CMAKE_EXPORT_COMPILE_COMMANDS=ON`. That option changes no compiled object and records every compile command for the evidence below. The wrapper builds the default **all** target using eight jobs (`--jobs N`, or `make accept-s1.1 JOBS=N`). The Spark CPU flag is the upstream workaround for the host GCC's target support. Local host drivers, headers and toolkit remain prerequisites for an activated venv. [Upstream recipe](../../ext/ocudu-deepsig-cuda-inline/docs/phy_cuda_acceleration.md) and [integration target](../../ext/ocudu-deepsig-cuda-inline/tests/integrationtests/phy/upper/channel_processors/CMakeLists.txt)

**Reference environment.** Before it configures anything, the builder compares the host with the environment DeepSig measured and writes `reference-environment.json`. The paper states the platform, the pinned CPU set and the governor (Section VI-A and Appendix A). The measurement notes in the pinned source (`docs/phy_cuda_acceleration.md`, called branch notes below) add the build recipe and the software versions. Platform and toolchain rows are requirements. The other rows are recorded, and the grade prints each difference as a `NOTE`.

| Item | DeepSig reference | Source | Treatment in S1.1 |
|---|---|---|---|
| Machine, GPU, compute capability | aarch64, NVIDIA GB10, 12.1 | Paper, branch notes | Required by the Spark profile. Another platform stops the run before CMake starts. |
| CUDA code generation | `sm_121` | Branch notes | Required. `nvcc --list-gpu-code` must offer it and every CUDA compile command must request it. |
| CPU target | `-mcpu=neoverse-v2` | Branch notes | Required. The host compiler must accept it and the PUSCH test must be compiled with it. |
| Host compile flags | `-O3 -DNDEBUG -fno-trapping-math -fno-math-errno` | Branch notes | Required on the compile command of the PUSCH test. |
| CMake | 3.18 or newer | Pinned `lib/phy/cuda/CMakeLists.txt` | Required. |
| CPU cores | Cortex-X925 and Cortex-A725 | Paper | Recorded from `/proc/cpuinfo`. |
| High-capacity CPU set | 5-9,15-19 | Paper | Recorded. It is a measurement condition of S1.2. The builder reads the kernel's `cpu_capacity` and falls back to the core model. |
| CPU frequency governor | performance | Paper | Recorded. It is a measurement condition of S1.2. |
| Host compiler and OS | GCC 13.3 on Ubuntu 24.04 | Branch notes | Recorded. |
| CUDA toolkit and GPU driver | 13.0.88 and 580.95.05 | Branch notes | Recorded. The Spark accepted in Sprint 0 runs toolkit 13.5 and driver 580.178.04, so its grade lists both. |
| OCUDU revision | `wg/nvcuda_accel_01` at `3c89cdee` | Paper | Recorded. The lock pins `nvcuda_accel_02` at `27e84cb3`. |

**Build evidence.** Each build writes a new directory and prints it as `Run artifacts`. Inside an acceptance run it is `native-build/`. A failed build keeps the same files up to the stage it reached.

| File | Contents |
|---|---|
| `status.txt`, `error.txt` | Last stage (`preflight`, `source`, `configure`, `build`, `inspect`, `completed`) and exit code. `error.txt` holds the failure, the lines of the failing log that name it (CMake errors, else compile and link errors, else the end of the log) and, when CMake could not find a required package, the `apt-get` command that installs it |
| `commands.sh` | Every command the builder ran |
| `nvidia-smi.txt`, `nvcc.txt`, `cmake.txt`, `c++.txt` | The selected GPU and the tool versions on `PATH` |
| `platform.json` | GPU, parent commit, CPU governors and affinity, and the environment variables that can change a build |
| `reference-environment.json` | The reference rows above, the toolchain requirements and every observed host fact, including whether the compiler can see each of the four required development packages (`host_packages`) |
| `pins.txt` | Identity of the locked OCUDU source before configuration |
| `configure.log`, `CMakeCache.txt`, `build.log` | CMake output, a copy of the cache as the build left it and the complete build log |
| `native-help.txt`, `gnb-version.txt` | Output of the two built executables |
| `native-build.json` | The receipt: commit, CMake options, CPU target, the compilers CMake selected, the CUDA compile summary and the SHA-256 of `gnb`, `pusch_e2e_pipeline_test` and `libocudu_phy_cuda.a`. The same file is stored as `isac-native.json` in the build directory. |

The receipt answers whether CUDA was really compiled. It names the `nvcc` that CMake selected and its version. It counts the CUDA sources and requires each compile command to target this GPU architecture. After the build it requires every kernel object of `ocudu_phy_cuda` to exist and both executables to carry linked device code (an ELF `.nv_fatbin` section). The help text of the PUSCH test is the same with and without CUDA, so it cannot show this.

**Other build profiles.** Use `bash scripts/build_ocudu.sh --machine local` for a different supported GPU/CPU. The wrapper discovers compute capability through `nvidia-smi` and verifies matching `sm_*` support in `nvcc --list-gpu-code`. On AArch64 a local build must name its CPU target with `--mcpu TARGET` (`--mcpu native` selects the upstream default). On x86_64 the upstream `-march=native` default applies and the receipt records it. For multiple GPUs, export `CUDA_VISIBLE_DEVICES` as the full UUID of the intended physical GPU. Use `--build-dir DIR` for separate builds per hardware/toolchain and check the cache. The physical-device build does not allocate a MIG instance. A build on another architecture supplies a portability result on that machine. The DeepSig row is assigned to GB10. `make accept-s1.1` takes the same `--machine`, `--mcpu`, `--build-dir` and `--jobs` through `scripts/accept_s1_1.sh` and words its verdict for the local profile.

**Unit check U1-build.** `tests/audit/test_native_baseline.py` drives the builder with doubles that write the files CMake writes. A correct configuration issues the full build command, records compilers, CUDA units, targets and hashes, and links the gNB and native test into `isac/bin`. Each wrong cache value, including `CUDA_ACCEL_FOUND=FALSE`, stops the run before the build. So do an `nvcc` that cannot emit the GPU architecture, a CUDA compiler that is not `nvcc`, a CUDA source compiled for another architecture, a build tree without kernel sources, and a PUSCH test compiled without `ENABLE_CUDA`, the CPU target or the reference flags. After the build, a missing kernel object, an executable without device code or one for another CPU architecture publishes no link and no receipt. A changed binary fails `--skip-build` validation. The reference comparison is tested against a synthetic Spark core layout, a non-GB10 host, a rejected CPU target and an old CMake. The grader is tested by damaging one evidence record at a time. The failure text is tested with the configure output the Spark printed when GTest was absent: the excerpt, the package named by CMake, the packages the compiler probe adds and the install command. A live command must print while it runs and still keep its log. `tests/test_accept_s1_1.py` runs the gate script against tool doubles, and `tests/test_environment.py` covers the receipt checks of `--with-native`.

```bash
python -m unittest discover -s tests/audit -p 'test_native_baseline.py' -v
python tests/test_accept_s1_1.py -v
```

**Acceptance I1-build.** `make accept-s1.1` writes `results/runs/s11-<host>-<UTC time>/` and ends with `Sprint 1.1: ACCEPTED on this DGX Spark.` only when every gate below passes. The run directory holds `summary.txt` with one line per gate, the build evidence in `native-build/` and one log per gate. An existing run directory is never reused. A second run reuses the build directory, so it is quick.

| Gate in `summary.txt` | Passes when |
|---|---|
| `committed tree` | `git status --short` is empty, so the evidence names one commit |
| `unit tests` | The portable suite ran at least one test with no failure. It runs first and needs no GPU. |
| `S1.1 OCUDU source is pristine before the build` | `git status --porcelain --untracked-files=all` inside the source is empty. A build directory made by hand inside the source fails here, before the long build starts. Remove it and rerun. |
| `S1.1 native build command` | The builder exits zero. A failure names the stage, and the summary quotes the failing tool's own error and the remedy under the `FAIL` line. The gates that read a completed build are then listed once as `SKIP` instead of failing one by one. |
| `S1.1 full Release build completed (target all)` | `status.txt` reports `stage=completed`, `exit_code=0` and the receipt names the full target |
| `S1.1 source identity` | The receipt commit equals the `ocudu` commit in `ext/lock.json` |
| `S1.1 gNB version banner` | `gnb --version` printed `OCUDU 5G gNB version 26.04.0 (...)` with an abbreviation of the locked commit. An archive build may print empty parentheses. |
| `S1.1 reference platform` | Machine, GPU and compute capability equal aarch64, NVIDIA GB10 and 12.1 |
| `S1.1 toolchain requirements` | `nvcc` emits `sm_121`, CMake is 3.18 or newer and the host compiler accepts `-mcpu=neoverse-v2` |
| `S1.1 build configuration` | The cache copy holds `CMAKE_BUILD_TYPE=Release`, `ENABLE_CUDA=ON`, `CUDA_ACCEL_FOUND=TRUE`, `BUILD_TESTING=ON`, `CMAKE_CUDA_ARCHITECTURES=121`, `MCPU=neoverse-v2` and its digest equals the receipt |
| `S1.1 CUDA compilation` | CMake selected NVIDIA `nvcc`, the kernel sources were compiled for `sm_121` and device code is linked into `pusch_e2e_pipeline_test` and `gnb` |
| `S1.1 compiled targets and hashes` | The receipt lists `gnb`, `pusch_e2e_pipeline_test` and `libocudu_phy_cuda.a` with SHA-256 digests that agree with its identity fields |
| `S1.1 gnb and pusch_e2e_pipeline_test resolve inside isac/bin` | With `isac/bin` first on `PATH`, `command -v` returns the two links the build published |
| `S1.1 gnb --version from the environment` | The environment's `gnb` starts and prints the version banner |
| `S1.1 pusch_e2e_pipeline_test --help` | The environment's test starts and lists `--require-gpu-resident`, `--rx-device-grid` and `--gpu-output-buffer` |
| `S1.1 native environment report` | `verify_environment.py --profile core --with-native` passes: both links match the receipt hashes, the build was not reconfigured after its receipt and the banner names the locked commit |
| `S1.1 simulation on <backend> and native from one interpreter` | Only with `BACKENDS`. The S0 simulation probe and the native checks pass in one process. |
| `S1.1 OCUDU source at the locked revision after the build` | The pin check still passes |
| `S1.1 out-of-tree build left no file in the OCUDU source` | The same listing after the build equals the one taken before it (`source-before.txt`, `source-after.txt`) |

A failed gate prints the last lines of its log under its `FAIL` line, on the screen and in `summary.txt`. `NOTE` lines do not change the verdict. They list each recorded difference from the reference and each S1.2 measurement condition that is not met yet, such as a governor other than `performance`. Regrade an archived build with `python scripts/native_ocudu.py grade PATH/TO/native-build`. Hand the run directory, build directory and exact source identity to S1.2. A test-target-only build using `--target baseline` suffices for an isolated diagnostic and records `target=baseline`. Only the full build closes I1-build. [Build implementation](../../scripts/native_ocudu.py) and [gate runner](../../scripts/accept_s1_1.sh)

**Rehearsal off the Spark, 2026-11-19.** The gate was exercised once on an x86_64 Ubuntu 24.04 container without a GPU, using GCC 13.3.0, CMake 3.28.3, the real `nvcc` 13.5.37 and a stand-in `nvidia-smi` that reports compute capability 12.1. The pinned source configured with the options above. The 24 CUDA sources that the two executables link compiled for `sm_121`, 18 of them the kernel sources of `ocudu_phy_cuda`. `gnb` and `pusch_e2e_pipeline_test` linked with device code, `gnb --version` printed the banner and the test printed its help. `bash scripts/accept_s1_1.sh --machine local` then passed every gate on that build, also on a repeat run. The rehearsal built only those two executables and not the rest of the default target. It executed no CUDA kernel, compiled nothing for AArch64 and never took the Spark profile. It checks the tooling. The Spark run alone closes I1-build.

**First Spark run, 2026-11-20.** Run `s11-spark-ecf8-20261120T002657Z` passed the platform and toolchain preflight (aarch64, NVIDIA GB10, 12.1, `sm_121`, CMake 3.28.3, GCC 13.3.0 accepting `-mcpu=neoverse-v2`) and recorded the paper's CPU layout and governor: `cpu_capacity` marks CPUs 5-9 and 15-19 as the large cores and every policy runs `performance`. It stopped in the configure step with `Could NOT find GTest`, because `libgtest-dev` was not installed. The same failure was then reproduced off the Spark by hiding GTest and yaml-cpp, which produced the error text and install command described under Recovery, and the failed build directory configured cleanly once the packages were back.

## S1.2 · Reproduce the 20 MHz PUSCH correctness row

**Entry.** Use the completed S1.1 Spark build and an idle test GPU. Stop simultaneous gNB, training or inference processes for this comparison. Record any remaining GPU clients, CPU governor and clocks. Use the allowed high-capacity CPU set `5-9,15-19` reported in the paper. The runner checks it is available before starting `taskset`. [Paper, Appendix](https://arxiv.org/abs/2608.04338v1)

| Property | Paper comparator | Operational choice in the pinned test |
|---|---|---|
| Native revision | `wg/nvcuda_accel_01` at `3c89cdee` | `nvcuda_accel_02` at `27e84cb3da65205b55dbd56ace6fac93a035331e`. The branch documentation’s `9fd4047b43` is a third measurement revision. |
| Radio grant | 20 MHz class, 30 kHz SCS, 1 layer, 256-QAM table, MCS 27, two Type-1 DM-RS symbols, TBS 54.3 kbit | 51 PRBs, 1 RX port, pilot symbols 2 and 11, no transform precoding. Native header must report 54,296 bits (6,787 bytes), 256QAM. |
| Receiver and sample count | Direct-managed input, pinned GPU output, CPU/GPU decode on matched frames, 200 measured | Generated single-tap signal, 30 dB SINR, five warmup iterations, every frame on both backends, 200 measured, resident stage timing enabled. The 30 dB operating point, warmup and RX-port choice are explicit local settings where Table II does not specify them. |
| Published row | 200 CRC successes on each backend, no CPU/GPU byte mismatches | Require this correctness result **and** 200 resident decodes/owned grids/immutable snapshots plus zero guard failures. |
| Latency | Mean completed CPU work 334.0 µs, GPU work 183.9 µs, ratio 1.82× | Extract min/mean/max from the native totals, report actual means, speedup and deviations. Compare direction separately. The reference values do not serve as fixed timing thresholds across revisions. |

The pinned harness uses a compatible PDSCH-side grid builder for this non-transform-precoded case and its random generator does not expose a seed switch. CPU and GPU receive the same generated realization within the experiment. The byte check compares the two decoded outputs. S2 adds the independent legal-PUSCH reference and checks each decoded output against retained TX bytes. [Pinned native fixture](../../ext/ocudu-deepsig-cuda-inline/tests/integrationtests/phy/upper/channel_processors/pusch_e2e_pipeline_test.cpp)

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

**Recovery.** The screen and `summary.txt` quote the failing tool. `status.txt` and `error.txt` in `native-build/` hold the same text. A `configure` stop that says `Could NOT find GTest`, `SCTP` or `MbedTLS`, or `yaml-cpp is required`, means a development package from the list under Entry is absent. `error.txt` names it together with any other of the four that the compiler cannot find, and prints the install command. Install them and run the same command again. The failed build directory needs no cleaning. Before CMake runs, the builder also prints which of the four the compiler cannot see. That line is advice and never stops a build, because CMake searches more prefixes than the compiler. A `preflight` stop names a missing host tool, a platform that is not the selected profile, or a toolchain requirement. Put the CUDA toolkit's `bin` directory on `PATH` when `nvcc` is missing. A stop right after `configure` names the cache value or compile command that differs from the recipe. A failed `OCUDU source is pristine` gate lists the offending files in `source-before.txt`. Delete a build directory made by hand inside the source, or restore an edited file with `git -C ext/ocudu-deepsig-cuda-inline checkout -- PATH`, then rerun. A stop in `inspect` means the build finished without compiled kernels or linked device code. For missing libraries or targets inspect `configure.log` and the host dependency list. For `no kernel image`, compare the selected GPU, NVCC supported codes, CMake target and binary identity. Use a new build directory after changing compiler or platform. If the process exits with success but suppresses the GPU factory or prints a failed assertion/summary, inspect `pusch.log` and retain the failing run. A source/cache/binary mismatch requires a new build and fresh identity, never editing the acquisition. A CPU-affinity failure requires running within a Spark CPU allocation that includes the measured cores.
