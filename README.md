# ocudu-cuda-isac

Receiver-aware integrated sensing and communication on CUDA-accelerated OCUDU. The first lab result is a reproducible CPU/GPU PUSCH decode on DGX Spark. The first project demonstration adds a second MPS client that detects a moving target from that receiver's uplink pilots. The publication study compares receiver stages, radar and AI estimators, and three hardware deployments under matched communication load.

**Start with [docs/dev/DEVELOPMENT.md](docs/dev/DEVELOPMENT.md).** It contains the phase order, every sprint's component contract, isolated test, integration prerequisites and acceptance evidence.

## CURRENT STATUS

As of 2026-11-20 (UTC), **Sprint 0 (install and verify) and Sprint 1.1 (native Release build) are accepted on one DGX Spark. Sprint 1.2 (the DeepSig Table II row) is the next open gate.** S2 to S9 are specified and not yet implemented.

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
- **The gate runner** refuses an uncommitted tree, an empty test run, a skipped test on a simulation host, and an evidence directory that already exists.

### Open items

- **Sprint 1.2.** `make paper-baseline` is the next gate. It uses the accepted S1.1 build and has produced no Spark evidence yet.
- **Other hosts.** Sprints 0 and 1.1 are accepted on one Spark. The second Spark and the GB300 each need their own `make accept-s0` and `make accept-s1.1` runs. The GB300 uses `bash scripts/accept_s1_1.sh --machine local` with its own CPU target.
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

Clone without `--recursive`. `make check` verifies the source pins, runs the unit suite and replays the four-update fixture. Four tests need NumPy and are skipped on a core host. Run `python -m pip install 'numpy>=2.2.6'` first to include them.

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

### Native receiver (Sprint 1.1 accepted, Sprint 1.2 next)

```bash
source isac/bin/activate
make accept-s1.1 BACKENDS="cpu cuda"
make paper-baseline
```

`make accept-s1.1` is the Sprint 1.1 gate. It first installs the OCUDU development libraries that dpkg reports missing (`scripts/install_native_deps.sh`, safe to repeat, asks for sudo only when something is missing). It shows the CMake output while it runs, builds Release with CUDA architecture 121 and `neoverse-v2` on Spark, compares the host with the environment DeepSig measured, requires compiled CUDA kernels and linked device code, and starts `gnb` and `pusch_e2e_pipeline_test` from the environment. Name the simulation backends only on a host that passed them in Sprint 0. The evidence goes to a new directory under `results/runs/`. `make native-build` runs the build alone, with host-specific targets for other machines. The final command runs the one-layer DeepSig Table II profile, checks actual resident GPU counters and CPU/GPU decoded-byte agreement, and archives the local timings. The separate short four-RX smoke uses the original advisor profile. The source, hardware and runtime dependencies for these commands are documented in the [native runbook](docs/dev/NATIVE_BASELINE.md). On split deployments run the full Sionna check on the scene host and the native build and receiver check on the DSP host.

## Repository map

| Directory | Purpose |
|---|---|
| [docs/dev](docs/dev/DEVELOPMENT.md) | Sprint cards, host setup, paper baseline and reproduction |
| [docs](docs/README.md) | System design, integration boundaries, interfaces, measurement, evidence and references |
| [ext](ext/README.md) | CUDA-OCUDU, Sionna and Sionna RT as pinned Git submodules, with the lock that owns their paths |
| [environments](environments/README.md) | Host environment notes and the validated third-party constraints |
| [src/ocudu_cuda_isac](src/ocudu_cuda_isac/) | Project-owned portable replay and accounting |
| [native/ocudu_adapter](native/ocudu_adapter/README.md) and [patches/ocudu](patches/ocudu/README.md) | Future native observer/replay adapters and narrow upstream patches |
| [configs](configs/) and [experiments](experiments/README.md) | Working smoke inputs, advisor profile and publication matrix |
| [data](data/README.md) and [results](results/README.md) | Episode provenance and run evidence |
| [scripts](scripts/), [tests](tests/) and [CI](.github/workflows/portable.yml) | Environment, pin and acceptance gates, portable regressions and the hosted CPU checks |

M1 shares one Spark under MPS, M2 uses two Sparks with a measured CX7 observation path, and M3 uses two GPU instances on one GB300 with a full-GPU control. See [deployments](docs/architecture/DEPLOYMENTS.md), the [advisor demonstration](docs/demo/ADVISOR_DEMO.md) and the [upstream reuse ledger](docs/integration/UPSTREAM.md).
