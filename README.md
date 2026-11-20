# Lightspeed: An Inline CUDA PHY Ladder for Receiver-Conditioned ISAC from Ray-Traced Channels to Over-the-Air 5G

**Index terms:** OCUDU, Sionna RT, DeepSig, Aerial. **Repository:** `ocudu-cuda-isac`.

Receiver-aware integrated sensing and communication on CUDA-accelerated OCUDU. The first lab result is a reproducible CPU/GPU PUSCH decode on DGX Spark. The first project demonstration adds a second MPS client that runs a cuSense-like pipeline on that receiver's own channel estimate, with a live KPI page. The publication study compares receiver stages, PHY channels, radar and AI estimators, functional splits, memory paths and three hardware deployments under matched communication load. The same sensing core is then run on Aerial Framework and, over the air, on Aerial-OCUDU.

**Start with [docs/dev/DEVELOPMENT.md](docs/dev/DEVELOPMENT.md).** It contains the phase order, every sprint's component contract, isolated test, integration prerequisites and acceptance evidence.

## CURRENT STATUS

As of 2026-11-20 (UTC), **Sprint 0 (install and verify) and Sprint 1 (native Release build and the DeepSig Table II row) are accepted on one DGX Spark. Sprint 2.1 (`grant_fixture`), the first project-owned component, is implemented and passes its unit gate on a development host. Its integration gate, `make accept-s2.1`, has not run on the Spark yet.** S2.2 to S11 are specified and not yet implemented. The plan was audited on 2026-11-20 against the evaluation plan of that date. [Build order](docs/dev/DEVELOPMENT.md#build-order) · [requirements table](docs/dev/DEVELOPMENT.md#requirements-and-where-they-are-built)

### Sprint 0 acceptance

`make accept-s0 BACKENDS="cpu cuda"` ran every Sprint 0 gate on the Spark at commit `49b6092` and ended with `Sprint 0: ACCEPTED on this host, including simulation on cpu cuda`. The run ID is `s0-spark-ecf8-20261119T222041Z`. Later commits to the documentation and the hosted workflow do not change gated code.

| Gate | Result | What it showed |
|---|---|---|
| Committed tree | Pass | The evidence came from a clean checkout |
| S0.1 environment | Pass | Isolated `isac` venv, project installed editable from this checkout, `pip check` clean |
| S0.2 sources | Pass | Three Git submodules pristine at the commits in `ext/lock.json` |
| S0.2 simulation on CPU | Pass | Sionna PHY on the CPU and Sionna RT through LLVM 20.1 |
| S0.2 simulation on CUDA | Pass | Sionna PHY on the GB10 and Sionna RT through CUDA with OptiX |
| S0.3 replay | Pass | Four scheduled updates: one on time, one late, one dropped, one missing. Three output files byte-identical on repeat |
| Unit suite | Pass | 64 tests ran, none skipped |

Two numerical results back the simulation rows. The probe traces a 20 m line-of-sight link at 3.5 GHz and compares it with the free-space answer. On both backends the gain agreed within a relative 1.7e-7 (limit 1e-4). The delay agreed within 0.0022 ns on the CPU and 0.0020 ns on CUDA (limit 0.05 ns). Every installed Sionna file equals the pinned source tree: 205 files for Sionna and 10,439 for Sionna RT.

The replay fixture has the same SHA-256 digests on the Spark (aarch64) and on an x86_64 host. `tests/test_replay.py` records them, so `make check` proves the match on any host.

### Sprint 1.1 acceptance

`make accept-s1.1 BACKENDS="cpu cuda"` ran every Sprint 1.1 gate on the same Spark at commit `cb9451e` and ended with `Sprint 1.1: ACCEPTED on this DGX Spark.` The run ID is `s11-spark-ecf8-20261120T011024Z`. All 20 gate lines passed. Later commits to the documentation do not change gated code.

| Gate | Result | What it showed |
|---|---|---|
| Committed tree and unit suite | Pass | The evidence came from a clean checkout. 103 tests ran, none skipped |
| OCUDU source | Pass | Pristine at `27e84cb3` before the build. Still at the pin, with no new file, after it |
| Native build | Pass | Full Release build of the default target, out of tree in `build/ocudu-spark` |
| Reference platform and toolchain | Pass | aarch64, NVIDIA GB10, compute capability 12.1. `nvcc` emits `sm_121` and the host compiler accepts `-mcpu=neoverse-v2` |
| Build configuration | Pass | `CMAKE_BUILD_TYPE=Release`, `ENABLE_CUDA=ON`, `CUDA_ACCEL_FOUND=TRUE`, `BUILD_TESTING=ON`, `CMAKE_CUDA_ARCHITECTURES=121`, `MCPU=neoverse-v2` |
| CUDA compilation | Pass | 18 kernel sources of 30 CUDA sources compiled for `sm_121` by `nvcc` 13.5.37 with GCC 13.3.0. Device code is linked into `pusch_e2e_pipeline_test` and `gnb` |
| Executables | Pass | Both resolve inside `isac/bin`. `gnb --version` prints `OCUDU 5G gNB version 26.04.0 (27e84cb3da)`. The PUSCH test lists the resident, device-grid and output-buffer options |
| Environment reports | Pass | The build receipt matches both executables, alone and together with the Sionna CPU and CUDA probes from one interpreter |

The receipt identifies the build products by SHA-256. Their first twelve digits are `0815f3a2dfdc` for `gnb`, `e49b2b81e28a` for `pusch_e2e_pipeline_test` and `4b4a39e38a9c` for `libocudu_phy_cuda.a`.

The gate also compares the host with the environment DeepSig measured. CPU cores, the high-capacity CPU set 5-9,15-19, the `performance` governor, GCC 13.3 and Ubuntu 24.04 match. Three differences are recorded and do not fail the gate: CUDA toolkit 13.5.37 against 13.0.88, GPU driver 580.178.04 against 580.95.05, and OCUDU `nvcuda_accel_02` at `27e84cb3` against `wg/nvcuda_accel_01` at `3c89cdee`.

Two earlier runs on the same host, `s11-spark-ecf8-20261120T002657Z` and `s11-spark-ecf8-20261120T004837Z`, stopped in the configure step because OCUDU's development libraries were not installed. The builder now installs the missing ones before it configures. [Record](docs/dev/NATIVE_BASELINE.md)

### Sprint 1.2 acceptance

`make accept-s1.2` ran every Sprint 1.2 gate on the same Spark at commit `e770a28` (recorded by the run as `ea7bb43` before a message-only history rewrite, same tree) and ended with `Sprint 1.2: ACCEPTED on this DGX Spark.` and `Sprint 1: ACCEPTED on this DGX Spark`. The run ID is `s12-spark-ecf8-20261120T144154Z`. All 27 gate lines passed. The gate measured the executables that Sprint 1.1 accepted, with nothing rebuilt in between.

| Gate | Result | What it showed |
|---|---|---|
| Committed tree and unit suite | Pass | The evidence came from a clean checkout. 121 tests ran, none skipped |
| Build identity | Pass | Both executables in `isac/bin` match the receipt of the accepted run `s11-spark-ecf8-20261120T011024Z`. The source is still at `27e84cb3` after the runs |
| Device and CPU set | Pass | The process could see only the GB10, selected by UUID, and ran under `taskset -c 5-9,15-19` |
| Workload | Pass | The Table II 20 MHz one-layer row: 51 PRB, MCS 27 (qam256), 1 layer, 1 RX port, TBS 54296 bits, 200 measured frames after 5 warmup, direct-managed grid, pinned output |
| CPU and GPU decodes | Pass | 200 of 200 CRC passes on each path |
| Decoded bytes | Pass | 0 CRC disagreements, 0 byte mismatches, 0 output guard failures |
| Completed CUDA work | Pass | Resident timing, producer-owned device grid and immutable snapshot on 200 of 200 frames. The LDPC decode stage took 120.5 us by CUDA events |
| Service times | Pass | Measured and finite: CPU 397.2 us mean (123.0 to 997.0), GPU 239.6 us mean (213.0 to 327.0) |
| Four-RX smoke | Pass | 20 of 20 on each path with no differing byte. CPU 240.5 us mean, GPU 167.9 us mean |

Correctness equals the published row: a 54.3 kb transport block, 200 of 200 decoded frames on each path and no differing byte. The SINR the two paths report differs by less than 0.01 dB, where the paper states 0.02 dB or less.

Timing is recorded and not graded. The paper reports 334.0 us on the CPU, 183.9 us on the GPU and 1.82x. This run measured 397.2 us (+19 percent), 239.6 us (+30 percent) and 1.66x. The GPU value repeats: an attempt 14 minutes earlier measured 237.2 us. The CPU value does not: that attempt measured 448.9 us, and single frames range from 123 to 1177 us. Most of the GPU difference follows the SINR, which the paper does not state. At the 30 dB used here the decoders need about two LDPC iterations per frame. A hand-run probe at 40 dB, not archived, needed one and measured 198.9 us on the GPU. The run also differs from the paper's in the OCUDU revision, the CUDA toolkit and the driver, and a desktop session with a browser was open.

The timing was accepted on 2026-11-20 as comparable to the paper for bring-up. System tuning is deferred until the project features are implemented. The values are a record of this host and not a reproduction of the published latencies.

An earlier attempt, `s12-spark-ecf8-20261120T142736Z`, measured the same files on an uncommitted tree to test the gate before the commit. Its 26 other gates passed and it was not accepted for that reason alone. [Record](docs/dev/NATIVE_BASELINE.md)

### Sprint 2.1: implemented, gate pending on the Spark

Sprint 2.1 exports one legal uplink grant and its transmit reference. `src/ocudu_cuda_isac/episodes/grant.py` validates a `pusch-grant/v1` record, derives the transport block size, the resource elements in mapping order and the DM-RS reference symbols, and keeps them with the transmit grid in a checksummed store entry. The grant's DM-RS is registered as the first sensing source, `pusch_dmrs`. A thin native adapter builds the transmit grid with the transmit chain of the pinned `pusch_e2e_pipeline_test`, out of tree and against the build that Sprint 1.1 accepted. [Record, coordinates, store and alignment](docs/interfaces/GRANT.md)

`make accept-s2.1` feeds the same known transport block to both transmit references, the pinned OCUDU chain and Sionna NR, and compares them on three grants: the two-block unit fixture inside a larger carrier, the demonstration profile (51 blocks, four receive ports) and the PDU of the Table II row.

| Gate | What it requires |
|---|---|
| Committed tree and unit suite | A clean checkout. The card's unit command also runs on its own |
| References | The pinned Sionna on the CPU and the Sprint 1.1 build, verified from one interpreter |
| Adapter | Built with the compile flags and the link line of the pinned test. Its receipt binds the executable to its sources and to the OCUDU build |
| Transport block and coding | The same size in the record, in OCUDU and in Sionna. The same modulation, base graph and code block count |
| Resource elements | The occupied elements of both grids are exactly the grant's data and pilot elements |
| Pilots and data symbols | Every pilot equals its reference symbol and every data symbol of the two references agrees, within the storage tolerance of the pinned grid |
| Scale and phase | Gain within 0.5 percent of one and phase within 1 mrad between the references |
| Receiver check | The host PUSCH processor decodes the grid, without noise, to the original bytes on every receive port of the grant |
| Record | The grant, the bytes and both grids are archived and verified on reading. A pilot coordinate, a data symbol and the block hash are recovered from the record, the hash also with `sha256sum` |

One disagreement between the pinned test fixture and a true PUSCH was found and fixed. The test places the DM-RS sequence from the first block of the bandwidth part. PUSCH DM-RS starts at CRB 0. They coincide only when the bandwidth part starts at CRB 0, which is the only case the pinned test runs. The adapter uses CRB 0. On the offset grant every run archives the difference: with the test's setting 19 of 24 pilots differ and the receiver's CRC fails.

The steps of the gate were rehearsed on a development host: x86_64, a CPU-only build of the pinned OCUDU source, the pinned Sionna 2.2.0. All 38 checks of the three grants passed there, and every data symbol of the OCUDU grid equalled the Sionna value after rounding to the grid's storage format. The environment check and the pin check of the gate were not part of that rehearsal. It is no lab evidence. Sprint 2.1 is integrated when `make accept-s2.1` prints its verdict on the Spark.

### Verified configuration

| Item | Value on the accepted host |
|---|---|
| Host | DGX Spark `spark-ecf8`, aarch64, Ubuntu 24.04.5 LTS, kernel 7.0.0-1025-nvidia |
| GPU | NVIDIA GB10, compute capability 12.1, driver 580.178.04, CUDA toolkit 13.5 |
| CPU | Cortex-X925 on CPUs 5-9 and 15-19, Cortex-A725 on CPUs 0-4 and 10-14, every governor `performance` |
| Native toolchain | GCC 13.3.0, CMake 3.28.3, `nvcc` 13.5.37 |
| OCUDU | 26.04.0 at `27e84cb3da` (`nvcuda_accel_02`), Release, CUDA architecture 121, `-mcpu=neoverse-v2` |
| Python | 3.12.3 in the repository-local `isac` venv |
| PyTorch | 2.15.0+cu132, the PyPI aarch64 wheel, compiled for sm_80 through sm_120 |
| Sionna and Sionna RT | 2.2.0 and 2.2.0, built from `ext/` |
| Mitsuba and Dr.Jit | 3.9.1 and 1.5.0 |
| NumPy and SciPy | 2.5.5 and 1.18.1 |

`environments/full-constraints.txt` records the complete third-party set. The unit suite, the replay fixture and the CPU simulation probe also pass on an x86_64 Ubuntu 24.04 host without a GPU. The core checks ran there on Python 3.11, 3.12 and 3.13.

### What the gates reject

The gates were tested for what they refuse, in the unit suite or in an end-to-end negative run.

- **S0.1** refuses a system interpreter, packages that leak in through `PYTHONPATH` or the system site-packages, a project install that is not editable from this checkout, and a moved or incomplete venv.
- **S0.2** refuses a source tree that is off its pinned commit or has an edited tracked file, an installed Sionna file that differs from the pinned tree, a missing LLVM or CUDA backend, and a traced channel that deviates from free space. The file comparison matters because two Sionna RT commits both report version 2.2.0.
- **S0.3** refuses malformed or ambiguous fixtures, unknown or duplicate sequence IDs, a mismatched clock, and outputs that differ on repeat.
- **S1.1** refuses another platform under the Spark profile, an `nvcc` that cannot emit the GPU architecture, a cache value that differs from the recipe (including `CUDA_ACCEL_FOUND=FALSE`), a CUDA source compiled for another architecture, a PUSCH test compiled without `ENABLE_CUDA` or the CPU target, an executable without linked device code, binaries that differ from the build receipt, and an OCUDU source that is not pristine. A missing development library stopped two real runs in the configure step.
- **S1.2** refuses, in the unit suite and in a rehearsal against a stand-in for the native test: a workload other than the Table II row, fewer than 200 CRC passes on either path, a CRC disagreement, a byte mismatch, an output guard failure, a GPU path without resident timing or the producer-owned grid on every measured frame, a missing or non-finite service time, a transcript that is truncated, repeated or of zero iterations, a run that was not pinned to CPUs 5-9,15-19 or could see another GPU, and a build that Sprint 1.1 did not accept. A GPU path that is slower than the CPU path is recorded and not refused. An uncommitted tree stopped one real run.
- **S2.1** refuses, in the unit suite: a grant with an invalid DM-RS or receive port, an allocation outside the bandwidth part, a repeated pilot, DM-RS symbols the standard does not give for the allocation, and a payload that does not match the transport block size, each before a grid is allocated. A store entry is refused when a file differs from its hash, when a stored index or reference symbol is not the grant's own, when a grid occupies an element the grant does not, and when a pilot differs from its reference symbol. Against doubles of both transmit references the integration driver fails the check that names the fault: a differing transport block size or coding, a shifted or incomplete grid, a wrong pilot or data symbol, a failed receiver CRC, and a reference that cannot run, which is a failure and never a skip. The adapter builder refuses an OCUDU build without a Sprint 1.1 receipt, a build from another source or generator, and an adapter whose sources or OCUDU build changed after its receipt.
- **The gate runner** refuses an uncommitted tree, an empty test run, a skipped test on a simulation host, and an evidence directory that already exists.

### Open items

- **Sprint 2.1 gate.** `make accept-s2.1` has not run on the Spark. It needs the accepted Sprint 1.1 build and the full environment. The AArch64 compile of the adapter and its link against the CUDA build are first exercised there.
- **Next sprints.** S2.2 `episode_pair` generates the matched target-present and target-absent channels on the grant's subcarriers and symbol times. S2.3 `grid_apply` applies them natively. Both start from the Sprint 2.1 record.
- **Timing.** The Sprint 1.2 latencies were taken with desktop applications open and are 19 to 30 percent above the paper's. Tuning is deferred until the project features are implemented. Before a latency or speedup claim, measure on a quiet host, repeat the run to bound the CPU spread and state the SINR.
- **Other hosts.** Sprints 0 and 1 are accepted on one Spark. The second Spark needs its own `make accept-s0` and `make accept-s1` runs. The GB300 needs `make accept-s0` and `bash scripts/accept_s1_1.sh --machine local` with its own CPU target. The Sprint 1.2 gate measures the Table II row, which is assigned to a GB10.
- **Evidence location.** The run directories are on the Spark under `results/runs/`, which Git ignores. They are not in this repository.
- **Hosted checks.** `.github/workflows/portable.yml` is in place. No hosted run is recorded here yet.
- **Fresh clone.** `make hydrate` has not been run end to end on a new clone. The Spark's sources were added in place at the same commits. [ext/README.md](ext/README.md) covers an interrupted fetch.
- **ZIP import.** The archive route for hosts without Git access was not exercised against the current lock.
- **Watch item.** The cuBLAS release that PyTorch 2.15.0 installs has a published issue on sm_120 and sm_121: batched matrix multiply with broadcast operands can read out of bounds. Results are unaffected and Sprint 0 does not use it. It matters for later GPU model work.

## Set up a host

Use Linux with Python 3.11 or newer, `venv`, Git, GNU Make and Bash. The first install needs a package index or a local wheelhouse. See the [environment runbook](docs/dev/ENVIRONMENT.md) for OS packages and per-role detail.

### Core host

A core host runs the component tests and the replay fixture. It needs no GPU and no upstream source.

```bash
git clone https://github.com/rcbarke/ocudu-cuda-isac.git
cd ocudu-cuda-isac
make env
source isac/bin/activate
python scripts/verify_environment.py --profile core
make check
```

Clone without `--recursive`. `make check` verifies the source pins, runs the unit suite and replays the four-update fixture. Five tests need NumPy and are skipped on a core host. Run `python -m pip install 'numpy>=2.2.6'` first to include them.

### Simulation host

A simulation host adds the three pinned sources, PyTorch, Sionna and Sionna RT. The install is roughly 6 GB. The constraints file needs Python 3.12 or newer.

```bash
make hydrate
export PIP_CONSTRAINT="$PWD/environments/full-constraints.txt"
make env-full
unset PIP_CONSTRAINT
make accept-s0 BACKENDS="cpu cuda"
```

`make hydrate` fetches the three submodules at their pinned commits and verifies them. `PIP_CONSTRAINT` reproduces the validated third-party set. Leave it unset to let pip resolve newer releases. `make accept-s0` runs every Sprint 0 gate, writes the evidence to a new directory under `results/runs/` and prints one verdict. Name only `cpu` on a host without an NVIDIA GPU. [Source details](ext/README.md)

### Native receiver (Sprint 1 accepted)

```bash
source isac/bin/activate
make accept-s1.1 BACKENDS="cpu cuda"
make accept-s1.2
```

`make accept-s1.1` is the Sprint 1.1 gate. It first installs the OCUDU development libraries that dpkg reports missing (`scripts/install_native_deps.sh`, safe to repeat, asks for sudo only when something is missing). It shows the CMake output while it runs, builds Release with CUDA architecture 121 and `neoverse-v2` on Spark, compares the host with the environment DeepSig measured, requires compiled CUDA kernels and linked device code, and starts `gnb` and `pusch_e2e_pipeline_test` from the environment. Name the simulation backends only on a host that passed them in Sprint 0. The evidence goes to a new directory under `results/runs/`. `make native-build` runs the build alone, with host-specific targets for other machines. `make accept-s1.2` is the Sprint 1.2 gate. It measures the build that Sprint 1.1 accepted and never builds. It runs the one-layer DeepSig Table II profile on CPUs 5-9,15-19, then the short four-RX smoke of the original advisor profile. It requires 200 CRC passes on the CPU and on the GPU path, agreement of the decoded bytes and resident GPU work on every measured frame, and it records the local timings next to the published ones without grading them. A passing run ends with the Sprint 1.2 and the Sprint 1 verdicts. `make accept-s1` runs both gates in order, and `make paper-baseline` runs the paper profile alone. The source, hardware and runtime dependencies for these commands are documented in the [native runbook](docs/dev/NATIVE_BASELINE.md). On split deployments run the full Sionna check on the scene host and the native build and receiver check on the DSP host.

### Grant fixture (Sprint 2.1)

```bash
source isac/bin/activate
python tests/unit/test_grant_fixture.py -v
make accept-s2.1
```

The first command is the unit gate and needs only the core environment. `make accept-s2.1` is the integration gate. It builds the native fixture adapter into `build/ocudu-adapter` with the compile flags and the link line of the pinned `pusch_e2e_pipeline_test`, compares the pinned OCUDU transmit chain with Sionna NR on the three grants in `configs/grants/`, archives each record under a new directory in `results/runs/` and prints one verdict. It uses the CPU only and never rebuilds OCUDU. `make adapter-build` runs the adapter build alone. [Grant record](docs/interfaces/GRANT.md)

## Repository map

| Directory | Purpose |
|---|---|
| [docs/dev](docs/dev/DEVELOPMENT.md) | Sprint cards, host setup, paper baseline and reproduction |
| [docs](docs/README.md) | System design, integration boundaries, interfaces, measurement, evidence and references |
| [ext](ext/README.md) | CUDA-OCUDU, Sionna and Sionna RT as pinned Git submodules, with the lock that owns their paths |
| [environments](environments/README.md) | Host environment notes and the validated third-party constraints |
| [src/ocudu_cuda_isac](src/ocudu_cuda_isac/) | Project-owned portable replay and accounting, the grant record (`episodes/`) and the sensing-source registry (`sensing/sources/`) |
| [native/ocudu_adapter](native/ocudu_adapter/README.md) and [patches/ocudu](patches/ocudu/README.md) | Native adapter built against the accepted OCUDU build (the fixture adapter of S2.1; replay and observer adapters arrive with their tickets) and narrow upstream patches |
| [configs](configs/) and [experiments](experiments/README.md) | Working smoke inputs, grant records, integration configurations, advisor profile and publication matrix |
| [data](data/README.md) and [results](results/README.md) | Episode provenance and run evidence |
| [scripts](scripts/), [tests](tests/) and [CI](.github/workflows/portable.yml) | Environment, pin and acceptance gates, portable regressions and the hosted CPU checks |

M1 shares one Spark under MPS, M2 uses two Sparks with a measured CX7 observation path, and M3 uses two GPU instances on one GB300 with a full-GPU control. See [deployments](docs/architecture/DEPLOYMENTS.md), the [advisor demonstration](docs/demo/ADVISOR_DEMO.md) and the [upstream reuse ledger](docs/integration/UPSTREAM.md).
