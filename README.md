# ocudu-cuda-isac

Receiver-aware integrated sensing and communication on CUDA-accelerated OCUDU. The first lab result is a reproducible CPU/GPU PUSCH decode on DGX Spark. The first project demonstration adds a second MPS client that detects a moving target from that receiver's uplink pilots. The publication study compares receiver stages, radar and AI estimators, and three hardware deployments under matched communication load.

**Start with [docs/dev/DEVELOPMENT.md](docs/dev/DEVELOPMENT.md).** It contains the phase order, every sprint's component contract, isolated test, integration prerequisites and acceptance evidence. S0 and S1 ship runnable setup and verification tools. S2–S9 specify the remaining implementation and lab work.

## First checkout

Extract the archive into a new directory, preserving executable permissions and the included parent Git metadata. From the repository root on Linux, use Python 3.11+ with venv, Git, Make and Bash. The first install requires access to Python build packages or a local wheelhouse. See the [environment runbook](docs/dev/ENVIRONMENT.md) for OS dependencies and per-role instructions.

```bash
make env
source isac/bin/activate
python scripts/verify_environment.py --profile core
make check
```

The smoke produces four scheduled opportunities, one each on time, late, dropped and missing. The ordinary unit suite needs no GPU. To include the NumPy serialization tests on a core-only installation, run `python -m pip install 'numpy>=2.2.6'` before `make check`.

## Initial collocated setup and first Spark measurement

Choose one acquisition method for the three pinned source trees. With the supplied ZIPs, point the import command to their parent directory. Git checkout users with upstream access can run `make hydrate` instead. [Source acquisition details](ext/README.md)

```bash
python scripts/source_archives.py /path/to/original/archives
python scripts/check_pins.py --worktrees
make env-full
python scripts/verify_environment.py --profile simulation --backend cuda
make native-build
python scripts/verify_environment.py --profile simulation --backend cuda --with-native
make paper-baseline
```

The native build uses Release, CUDA architecture 121 and `neoverse-v2` on Spark, with host-specific targets for other machines. A full build exposes both `gnb` and `pusch_e2e_pipeline_test` in the activated environment. The final command runs the one-layer DeepSig Table II profile, checks actual resident GPU counters and CPU/GPU decoded-byte agreement, and archives the local timings. The separate short four-RX smoke uses the original advisor profile. The source, hardware and runtime dependencies for these commands are documented in the [native runbook](docs/dev/NATIVE_BASELINE.md). On split deployments run the full Sionna check on the scene host and the native build and receiver check on the DSP host.

## Repository map

| Directory | Purpose |
|---|---|
| [docs/dev](docs/dev/DEVELOPMENT.md) | Sprint cards, host setup, paper baseline and reproduction |
| [docs](docs/README.md) | System design, integration boundaries, interfaces, measurement, evidence and references |
| [ext](ext/README.md) | CUDA-OCUDU, Sionna and Sionna RT source pins and acquisition |
| [src/ocudu_cuda_isac](src/ocudu_cuda_isac/) | Project-owned portable replay and accounting |
| [native/ocudu_adapter](native/ocudu_adapter/README.md) and [patches/ocudu](patches/ocudu/README.md) | Future native observer/replay adapters and narrow upstream patches |
| [configs](configs/) and [experiments](experiments/README.md) | Working smoke inputs, advisor profile and publication matrix |
| [data](data/README.md) and [results](results/README.md) | Episode provenance and run evidence |
| [scripts](scripts/), [tests](tests/) and [CI](.github/workflows/portable.yml) | Executable checks, portable regressions and opt-in hardware verification |

M1 shares one Spark under MPS, M2 uses two Sparks with a measured CX7 observation path, and M3 uses two GPU instances on one GB300 with a full-GPU control. [Deployments](docs/architecture/DEPLOYMENTS.md) · [advisor demonstration](docs/demo/ADVISOR_DEMO.md) · [upstream reuse ledger](docs/integration/UPSTREAM.md)
