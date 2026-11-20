# Phase S1 · Build OCUDU and reproduce one DeepSig receiver result

The deliverable is a locally measured, digest-identified CPU/CUDA comparison from the pinned OCUDU receiver. Use the **20 MHz, one-layer PUSCH row in Table II** of Pennybacker, Liu, Kharchenko and O’Shea. Their experiment reported 200 successful decodes on each backend and zero byte mismatches. The small row avoids coupling this bring-up to a radio, core, scene, observer or trained model. The first four-RX sensing demonstration has a different workload. [Paper, §VI-A, Table II and Appendix](https://arxiv.org/abs/2608.04338v1) · [Source audit](../integration/OCUDU.md)

| Sprint | Deliverable | Component check, runnable on a CPU developer machine | Integration and evidence owner |
|---|---|---|---|
| S1.1 | Full out-of-source native Release build, discoverable from the active `isac` environment | Build contract and identity tests exercise synthetic CMake outputs | Spark operator retains successful configure/build logs, cache, executable SHA and native help |
| S1.2 | Single-row paper comparison and original four-RX smoke profile | Transcript validator tests exercise complete, corrupted, truncated and mismatched records | Spark operator acquires native run, retains raw log and checks observed values against the published result |

S1 is a laboratory gate before interpreting results from new receiver-linked ISAC components. Its parser and build-wrapper tickets are independently testable before hardware is reserved. A missing Spark leaves native evidence pending. Preserve the failed run directory and stage whenever an acquisition fails.

## S1.1 · Build and identify the native executable

**Entry.** Complete S0 source materialization and the project `isac` venv. For the developer’s initial end-to-end installation, also exercise the pinned Sionna PHY and RT probes. The native CMake invocation itself consumes only the OCUDU source and host toolchain. Use the environment runbook for driver, toolkit, compiler, apt packages, PyTorch platform wheel and per-host setup. [Environment runbook](ENVIRONMENT.md)

The pinned configure stops without four development packages: GTest (the test tree is on), yaml-cpp, SCTP and mbedTLS. FFTW is optional and used when present. The builder installs them. Before it observes the host it runs `scripts/install_native_deps.sh`, which asks dpkg which of `libfftw3-dev libmbedtls-dev libsctp-dev libyaml-cpp-dev libgtest-dev` are missing and installs only those with `sudo apt-get`. That list is the library part of OCUDU's own `docker/scripts/install_dependencies.sh`. With nothing missing the script changes nothing and never calls sudo, so every build and every gate run can call it. On a host without dpkg it prints the list and leaves the decision to CMake. The compiler, CMake, pkg-config and the CUDA toolkit stay the host's and are checked by the preflight.

```bash
bash scripts/install_native_deps.sh --check   # report only: exit 1 names what is missing
make native-deps                              # install what is missing, safe to repeat
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
| `status.txt`, `error.txt` | Last stage (`preflight`, `prerequisites`, `source`, `configure`, `build`, `inspect`, `completed`) and exit code. `error.txt` holds the failure, the lines of the failing log that name it (CMake errors, else compile and link errors, else the end of the log) and, when CMake could not find a required package, the `apt-get` command that installs it |
| `commands.sh` | Every command the builder ran |
| `nvidia-smi.txt`, `nvcc.txt`, `cmake.txt`, `c++.txt` | The selected GPU and the tool versions on `PATH` |
| `platform.json` | GPU, parent commit, CPU governors and affinity, and the environment variables that can change a build |
| `reference-environment.json` | The reference rows above, the toolchain requirements and every observed host fact, including whether the compiler can see each of the four required development packages (`host_packages`) |
| `native-deps.log` | Output of the prerequisite installer: the packages it found, or the `apt-get` run that installed the missing ones |
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

**Accepted Spark run, 2026-11-20 (UTC).** Run `s11-spark-ecf8-20261120T011024Z` at commit `cb9451e` closed I1-build. `make accept-s1.1 BACKENDS="cpu cuda"` passed all 20 gates on `spark-ecf8` and ended with `Sprint 1.1: ACCEPTED on this DGX Spark.` The grade lines of that run read as follows.

| Gate | Line in `summary.txt` |
|---|---|
| Source identity | `27e84cb3da65205b55dbd56ace6fac93a035331e (git)` |
| gNB version banner | `OCUDU 5G gNB version 26.04.0 (27e84cb3da)` |
| Reference platform | `aarch64, NVIDIA GB10, 12.1` |
| Toolchain requirements | `nvcc emits sm_121; CMake 3.18 or newer; c++ accepts -mcpu=neoverse-v2` |
| Build configuration | `CMAKE_BUILD_TYPE=Release, ENABLE_CUDA=ON, CUDA_ACCEL_FOUND=TRUE, BUILD_TESTING=ON, CMAKE_CUDA_ARCHITECTURES=121, MCPU=neoverse-v2` |
| CUDA compilation | `18 kernel sources of 30 CUDA sources for sm_121 by nvcc 13.5.37, host compiler GNU 13.3.0, device code linked into pusch_e2e_pipeline_test and gnb` |
| Compiled targets and hashes | `gnb 0815f3a2dfdc, libocudu_phy_cuda.a 4b4a39e38a9c, pusch_e2e_pipeline_test e49b2b81e28a` |

The run also passed the committed tree, the unit suite, the source checks before and after the build, the `isac/bin` links, `gnb --version`, the PUSCH test help, the native environment report and the collocated simulation reports on `cpu` and `cuda`. Its `NOTE` lines record that CPU cores, the high-capacity CPU set 5-9,15-19, the `performance` governor, GCC 13.3.0 and Ubuntu 24.04 match the reference. They record three differences: CUDA toolkit 13.5.37 against 13.0.88, GPU driver 580.178.04 against 580.95.05, and OCUDU `nvcuda_accel_02` at `27e84cb3` against `wg/nvcuda_accel_01` at `3c89cdee`. No S1.2 measurement condition was flagged. The receipt records 1,978 compile commands, all 1,927 C++ units compiled with `-mcpu=neoverse-v2 -O3`, all 30 CUDA units for `sm_121`, and `ccache` as the C and C++ compiler launcher, which the upstream CMake enables when it is installed. The build log holds no error. Its only compiler warnings are 682 `-Warray-bounds` lines from the C++ standard library header `array`, a warning the upstream build keeps from being an error. The archived evidence regrades to the same eight `PASS` lines with `python scripts/native_ocudu.py grade`. Hand this run directory, `build/ocudu-spark` and the source identity to S1.2.

**Earlier attempts on the same host.** Runs `s11-spark-ecf8-20261120T002657Z` and `s11-spark-ecf8-20261120T004837Z` passed the platform and toolchain preflight (aarch64, NVIDIA GB10, 12.1, `sm_121`, CMake 3.28.3, GCC 13.3.0 accepting `-mcpu=neoverse-v2`) and recorded the paper's CPU layout and governor: `cpu_capacity` marks CPUs 5-9 and 15-19 as the large cores and every policy runs `performance`. Both stopped in the configure step with `Could NOT find GTest`. `libgtest-dev`, `libyaml-cpp-dev`, `libsctp-dev` and `libmbedtls-dev` were not installed. The same failure was reproduced off the Spark by hiding GTest and yaml-cpp, which produced the error text and install command described under Recovery, and the failed build directory configured cleanly once the packages were back. The builder then gained the prerequisite installer described under Entry, and the four libraries were installed before the accepted run.

## S1.2 · Reproduce the 20 MHz PUSCH correctness row

**Entry.** Use the Spark build that Sprint 1.1 accepted and an idle test GPU. Stop simultaneous gNB, training or inference processes for this comparison. The paper measured Table II with the live modem off and never ran its microbenchmarks and its gNB campaigns at the same time. The run records the GPU clients that remain, the CPU governor and the CPU clocks. Use the allowed high-capacity CPU set `5-9,15-19` reported in the paper. The runner checks it is available before starting `taskset`. The paper hosts ran the `performance` governor, and the pinned branch notes run `scripts/ocudu_performance` from the OCUDU source before a latency capture. That script sets the governor and turns DRM KMS polling off. The accepted Spark already runs `performance` on every policy. [Paper, Section VI and Appendix](https://arxiv.org/abs/2608.04338v1)

| Property | Paper comparator | Operational choice in the pinned test |
|---|---|---|
| Native revision | `wg/nvcuda_accel_01` at `3c89cdee` | `nvcuda_accel_02` at `27e84cb3da65205b55dbd56ace6fac93a035331e`. The branch documentation’s `9fd4047b43` is a third measurement revision. |
| Radio grant | 20 MHz class, 30 kHz SCS, 1 layer, 256-QAM table, MCS 27, two Type-1 DM-RS symbols, TBS 54.3 kbit | 51 PRBs, 1 RX port, pilot symbols 2 and 11, no transform precoding. Native header must report 54,296 bits (6,787 bytes), 256QAM. |
| Receiver and sample count | Direct-managed input, pinned GPU output, CPU/GPU decode on matched frames, 200 measured | Generated single-tap signal, 30 dB SINR, five warmup iterations, every frame on both backends, 200 measured, resident stage timing enabled. The 30 dB operating point, warmup and RX-port choice are explicit local settings where Table II does not specify them. |
| Published row | 200 CRC successes on each backend, no CPU/GPU byte mismatches | Require this correctness result **and** 200 resident decodes/owned grids/immutable snapshots plus zero guard failures. |
| Latency | Mean completed CPU work 334.0 µs, GPU work 183.9 µs, ratio 1.82× | Extract min/mean/max from the native totals, report actual means, speedup and deviations. Compare direction separately. The reference values do not serve as fixed timing thresholds across revisions. |

The pinned harness uses a compatible PDSCH-side grid builder for this non-transform-precoded case and its random generator does not expose a seed switch. CPU and GPU receive the same generated realization within the experiment. The byte check compares the two decoded outputs. S2 adds the independent legal-PUSCH reference and checks each decoded output against retained TX bytes. [Pinned native fixture](../../ext/ocudu-deepsig-cuda-inline/tests/integrationtests/phy/upper/channel_processors/pusch_e2e_pipeline_test.cpp)

**Public interface.** One command measures both profiles, grades the archived evidence and prints one verdict. The individual commands stay available for diagnosis.

```bash
source isac/bin/activate
make accept-s1.2                       # the Table II row, then the four-RX smoke, on the accepted S1.1 build
make accept-s1 BACKENDS="cpu cuda"     # the S1.1 gate, then the S1.2 gate
# Step by step
make paper-baseline                    # bash scripts/run_spark_baseline.sh --profile paper --skip-build
bash scripts/run_spark_baseline.sh --profile smoke --skip-build
python scripts/native_ocudu.py grade-run results/runs/spark-pusch.XXXXXXXX
python scripts/verify_pusch_log.py PATH/TO/pusch.log --profile paper --json PATH/TO/review.json
```

The gate never builds. It reads the build directory from the `isac/bin/pusch_e2e_pipeline_test` link, so it measures the executables that Sprint 1.1 published. It also requires the Sprint 1.1 verdict in the run whose evidence the build receipt names. A build made or rebuilt outside `make accept-s1.1` is refused until that gate has passed again. The paper row runs first. The smoke follows. It uses four RX ports, MCS 10 from the 64-QAM table and 20 measured frames on any allowed CPU, which is the receiver shape of the first sensing demonstration. It carries no paper comparison.

**Run evidence.** Each measured run writes a new directory and prints it as `Run artifacts`. Inside an acceptance run the two directories are `paper/` and `smoke/`. A failed run keeps the same files up to the stage it reached.

| File | Contents |
|---|---|
| `status.txt`, `error.txt` | Last stage (`preflight`, `source`, `build-identity`, `run`, `verify`, `completed`) and exit code. `error.txt` quotes the failing tool. |
| `commands.sh` | Every command the runner ran, including the launcher, flags and environment of the native test |
| `nvidia-smi.txt`, `nvcc.txt`, `cmake.txt`, `c++.txt` | The selected GPU with its client processes just before the run, and the tool versions on `PATH` |
| `platform.json`, `reference-environment.json` | Profile, GPU, parent commit, governors and allowed CPUs, and the comparison with the DeepSig reference environment: revision, toolkit, driver and the other rows of the S1.1 table |
| `pins.txt`, `native-build.json`, `CMakeCache.txt` | Identity of the source and of the measured build. The runner refuses a build whose executable or cache differs from its receipt. |
| `measurement-conditions.json` | The exact command, the GPU clients and utilization at the start, running OCUDU modem processes (`gnb`, `odu`, `odu_low`), the clocks of CPUs 5-9 and 15-19, the clocksource, DRM KMS polling and the load average |
| `pusch.log` | Raw combined output of the native test |
| `verification.txt` | The transcript grade: one `PASS` or `FAIL` line per gate, then the comparison as `NOTE` lines |
| `table-ii.json` (paper), `smoke.json` (smoke) | The machine-readable grade: gates, workload header with the transport-block size and grid path, decode and residency counters, CPU and GPU min/mean/max, the GPU stage breakdown, LDPC iterations and SINR parity. The paper file adds the published row and the measured differences from it. It is written also when a gate fails. |

**Correctness and performance are graded apart.** Seven gates read the transcript: the workload, the CPU decodes, the GPU decodes, the agreement of the decoded bytes, the completed CUDA work, the measured service times and the verdict the test printed. Six of them reproduce the correctness of the published row. The service-time gate requires measured, finite, ordered values and nothing about their size. The comparison with the published 334.0 and 183.9 microsecond means is recorded as `NOTE` lines and under `performance` in `table-ii.json` with `"graded": false`. A GPU path that is slower than the CPU path passes and is recorded as slower. When any gate fails, the comparison is withheld, because a time measured on a path that fell back or decoded wrongly compares nothing. A speedup claim additionally requires local evidence for the claimed direction and magnitude.

The completed-work gate needs more than the exit status. The pinned test still decodes, times both paths and prints agreement when its GPU path fell back to the host. The gate therefore requires the line `E2E GPU Pipeline: ACTIVE` after warmup, resident timing on every measured iteration, the producer-owned device grid and the immutable owned snapshot on every measured iteration, and a positive LDPC decode time in the stage breakdown. The decoder measures that stage with CUDA events on its device stream. The run sets `CUDA_VISIBLE_DEVICES` to the UUID of the selected GPU, so the process can reach no other device.

**Unit check U1.1.** Run both audit modules without a GPU. `tests/audit/transcripts/` holds four transcripts in the exact format of the pinned test: its own report code, compiled from `pusch_e2e_pipeline_test.cpp` at `27e84cb3` against the bundled fmt, printed them for invented counters. No decode produced them and their times differ from the paper on purpose. The tests inject a wrong MCS, SINR, layer count, RX count, grid mode, output-buffer mode, TBS, codeblock count or warmup count. They reduce a CRC count, raise the disagreement, byte-mismatch or guard count, reduce each residency counter, remove or zero the device stage, damage, reorder or remove a timing, use an inconsistent speedup and replace the verdict. Each defect must fail its own gate and no other. A run of zero iterations, a truncated transcript and a repeated transcript must fail. A slower GPU must still pass and emit its measured comparison. `tests/audit/test_native_baseline.py` grades a synthetic run directory and damages one record at a time, and `tests/test_accept_s1_2.py` runs the gate script against tool doubles. Fixtures supply parser evidence only and no empirical result.

```bash
python -m unittest discover -s tests/audit -p 'test_*.py' -v
python tests/test_accept_s1_2.py -v
```

**Acceptance I1.1, I1.2.** `make accept-s1.2` writes `results/runs/s12-<host>-<UTC time>/` and ends with `Sprint 1.2: ACCEPTED on this DGX Spark.` only when every gate below passes. On the Spark that is 27 `PASS` lines. The run directory holds `summary.txt` with one line per gate, the evidence of the two runs in `paper/` and `smoke/` and one log per gate. An existing run directory is never reused.

| Gate in `summary.txt` | Passes when |
|---|---|
| `committed tree` | `git status --short` is empty, so the evidence names one commit |
| `unit tests` | The portable suite ran at least one test with no failure. It holds the graders' own tests and needs no GPU. |
| `S1.2 the Sprint 1.1 build is intact` | `verify_environment.py --profile core --with-native` passes: both links in `isac/bin` match the hashes of the full-build receipt, the build was not reconfigured after it and the gNB banner names the locked commit |
| `S1.2 the measured build is the one Sprint 1.1 accepted` | The run directory that holds the build evidence named in the receipt has a `summary.txt` with the line `Sprint 1.1: ACCEPTED on this DGX Spark` and no `FAIL` line |
| `S1.2 paper run command` | The runner exits zero: a zero status of the native test, then a zero status of the validator on the same log |
| `S1.2 paper run completed` | `status.txt` in `paper/` reports `stage=completed` and `exit_code=0` |
| `S1.2 paper measured the full Spark Release build` | The receipt copied into the run names target `all`, the Spark profile and the commit in `ext/lock.json`, and lists the test executable with its hash and linked device code |
| `S1.2 paper ran on the intended device` | The recorded command set `CUDA_VISIBLE_DEVICES` to the UUID of the selected GPU, that GPU is the reference platform (aarch64, NVIDIA GB10, 12.1), and the command ran under `taskset -c 5-9,15-19` |
| `S1.2 paper workload is the Table II 20 MHz one-layer row` | The header is 51 PRB, 30.0 dB SINR, MCS 27 (table=qam256), 1 layer, 1 RX port, DMRS Type1 [2,11], no transform precoding, 200 iterations, 5 warmup, `rx_device_grid=direct-managed`, `gpu_output=pinned`, and the transport block is 54296 bits (6787 bytes), BG1, 7 codeblocks, 256QAM |
| `S1.2 paper CPU decodes` | `CPU: 200/200 pass` |
| `S1.2 paper GPU decodes` | `GPU: 200/200 pass` |
| `S1.2 paper decoded bytes agree` | 0 CRC disagreements, 0 byte mismatches and 0 output guard failures |
| `S1.2 paper CUDA work completed on the device` | The resident pipeline is active after warmup, resident timing, producer-owned grid and immutable snapshot each count 200/200, and the LDPC decode stage has a positive mean time |
| `S1.2 paper service times measured` | CPU and GPU min, mean and max are positive, finite and ordered, and the printed speedup agrees with the two means |
| `S1.2 paper native test verdict` | The test printed `PASS: Both paths decoded all 200 iterations, 0 byte mismatches` once and no `*** FAIL` line |
| `S1.2 smoke ...` (ten lines) | The same ten gates for the four-RX profile with 20 measured frames. It needs no CPU pinning, and its header must be MCS 10 (table=qam64), 4 RX ports and `gpu_output=vector`. |
| `S1.2 OCUDU source at the locked revision after the runs` | The pin check still passes |

`NOTE` lines do not change the verdict. For the paper run they give the correctness statement, the CPU mean, GPU mean and speedup next to the published values, the SINR parity, the settings that Table II does not state, the GPU clients at the start, any compute client or modem process that shared the machine, DRM KMS polling when it is on, the CPU clocks, and every row of the reference environment that matches or differs. A like-for-like timing comparison needs a summary without a compute client, a modem process, a governor other than `performance` or a CPU set other than 5-9,15-19 among its `S1.2 measurement condition` notes. A run that reached the native test is graded even when it failed, so the `FAIL` lines name what the test reported. A run that stopped earlier prints its stage and reason and one `SKIP` line. Regrade an archived run with `python scripts/native_ocudu.py grade-run PATH/TO/paper`. [Transcript grader](../../scripts/verify_pusch_log.py), [evidence grader](../../scripts/native_ocudu.py) and [gate runner](../../scripts/accept_s1_2.sh)

**Sprint 1 in total.** Sprint 1 is accepted on a host when three things hold. `make accept-s1.1` ended with `Sprint 1.1: ACCEPTED on this DGX Spark.` `make accept-s1.2` ended with `Sprint 1.2: ACCEPTED on this DGX Spark.` The second measured the build the first accepted, with nothing rebuilt in between. The third and fourth S1.2 gates check that link, so a passing S1.2 summary ends with `Sprint 1: ACCEPTED on this DGX Spark (S1.1 run <id>, S1.2 run <id>). Next: S2.1 grant_fixture.` `make accept-s1 BACKENDS="cpu cuda"` runs both gates in order and stops if the first fails. Use it on a new host or after a change to the source, toolchain or driver. It reuses the build directory, so it is quick when the build is current.

**Rehearsal off the Spark, 2026-11-19.** The S1.2 gate was exercised on an x86_64 Ubuntu 24.04 container without a GPU. The real gate first refused that host: both runs stopped in the preflight because the Spark profile needs aarch64. For the rehearsal a scratch checkout then held a stand-in OCUDU pin, the layout and receipt of an accepted S1.1 build, and stand-ins for `nvidia-smi`, `nvcc`, `taskset` and the two executables. The runner was told the host is aarch64 with 20 allowed CPUs. The stand-in for the native test printed the pinned test's own report text for invented counters. `make accept-s1.2` then ran the real gate script, Git check, unit suite, environment report, runner, transcript validator, evidence grader and pin check, and passed its 27 gates. The same setup produced each refusal: a GPU path that fell back (the test exits 2), byte mismatches behind a zero exit status, a failed CRC on both paths, an uncommitted tree, a receipt that names no accepted S1.1 run, an executable changed after its receipt and a CPU allocation without cores 5-9,15-19. Diagnostics on stderr ahead of the transcript did not disturb the grade. No decode ran and no CUDA kernel executed. The rehearsal checks the tooling and the parser against the upstream report format. The Spark run alone closes I1.1 and I1.2.

**Recovery.** The screen and `summary.txt` quote the failing tool. `status.txt` and `error.txt` in `native-build/` hold the same text. A `configure` stop that says `Could NOT find GTest`, `SCTP` or `MbedTLS`, or `yaml-cpp is required`, means a development package from the list under Entry is absent. `error.txt` names it together with any other of the four that the compiler cannot find, and prints the install command. The builder installs them before it configures, so this stop now means the installation itself failed or the host has no dpkg. A `prerequisites` stop quotes `apt-get` or `sudo`. Once the packages are present, run the same command again. The failed build directory needs no cleaning. Before CMake runs, the builder also prints which of the four the compiler cannot see. That line is advice and never stops a build, because CMake searches more prefixes than the compiler. A `preflight` stop names a missing host tool, a platform that is not the selected profile, or a toolchain requirement. Put the CUDA toolkit's `bin` directory on `PATH` when `nvcc` is missing. A stop right after `configure` names the cache value or compile command that differs from the recipe. A failed `OCUDU source is pristine` gate lists the offending files in `source-before.txt`. Delete a build directory made by hand inside the source, or restore an edited file with `git -C ext/ocudu-deepsig-cuda-inline checkout -- PATH`, then rerun. A stop in `inspect` means the build finished without compiled kernels or linked device code. For missing libraries or targets inspect `configure.log` and the host dependency list. For `no kernel image`, compare the selected GPU, NVCC supported codes, CMake target and binary identity. Use a new build directory after changing compiler or platform. If the process exits with success but suppresses the GPU factory or prints a failed assertion/summary, inspect `pusch.log` and retain the failing run. A source/cache/binary mismatch requires a new build and fresh identity, never editing the acquisition. A CPU-affinity failure requires running within a Spark CPU allocation that includes the measured cores. An S1.2 gate that says `needs the Sprint 1.1 build` or `no ACCEPTED verdict` means the executables in `isac/bin` were not published by an accepted Sprint 1.1 run: the build was made with `make native-build`, rebuilt later, or the run directory of the accepted gate was removed. Run `make accept-s1.1`, which is quick on a current build, then `make accept-s1.2`. A `build-identity` stop means the executable or the cache changed after the receipt. A `run` stop that names cores 5-9,15-19 is the CPU-affinity failure above. A `run` stop with `timeout exited with status 2` is the test's own refusal: its resident or device-grid requirement was not met, and the `FAIL` lines of the grade name the counter. Status 124 means the five-minute limit was reached. Status 134 is an abort: the end of `pusch.log` holds the exception, and `Failed to allocate direct managed PUSCH RX resource grid` means the process found no usable CUDA device. A `verify` stop means the test exited zero and the validator refused its transcript. A decode count below 200/200 at 30 dB is a receiver or environment defect. Keep the run, repeat the gate once and keep both runs.
