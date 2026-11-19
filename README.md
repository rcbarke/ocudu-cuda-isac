# ocudu-cuda-isac

Receiver-aware integrated sensing and communication on CUDA-accelerated OCUDU. The first lab milestone is one DGX Spark decoding a known UE uplink while a second CUDA process uses the receiver's pilot estimates to detect a moving target. The conference study varies receiver stage, sensing model and deployment while replaying the same radio episodes.

**Start with [docs/dev/DEVELOPMENT.md](docs/dev/DEVELOPMENT.md).** It assigns each code increment, test and acceptance gate from the first upstream PUSCH run to the publication artifact. The [advisor demo](docs/demo/ADVISOR_DEMO.md) describes what the audience will see and what evidence counts.

## Run the delivered scaffold

Extract the entire archive, including its Git metadata. Use Git, Bash, Make and Python 3.10 or later. From the repository root:

```sh
make check             # offline dependency pins, Python tests, deterministic replay smoke
make hydrate           # fetch the three upstream worktrees for native/offline development
make spark-baseline    # on GB10: build, run, verify and archive the native PUSCH baseline
```

An optional [Sionna RT exporter](docs/interfaces/RT_EXPORT.md) produces checksummed channel tensors for a moving target and its physically absent control.

The host smoke exercises channel replay and observation accounting on a small analytic fixture. A successful host run validates the adapter contract. The Spark demonstration requires building the pinned CUDA PHY, connecting the observation seam and running the MPS workload on GB10. The [Spark runbook](docs/demo/SPARK.md) gives the native prerequisites and evidence checks. The baseline independently verifies printed CPU/GPU agreement counters. The integrated demo adds the original transmitted bytes as a separate correctness reference.

## Navigate

| Directory | Purpose |
|---|---|
| [docs](docs/README.md) | System design, runbooks, interfaces, development gates and references |
| [deps](deps/README.md) | Git submodules for CUDA-OCUDU, Sionna and Sionna RT, with a machine-readable lock |
| [src/ocudu_cuda_isac](src/ocudu_cuda_isac/) | Small project-owned replay and evidence tools |
| [native/ocudu_adapter](native/ocudu_adapter/README.md), [patches/ocudu](patches/ocudu/README.md) | Native adapter boundary and policy for required WG1 observation hooks |
| [configs](configs/), [experiments](experiments/README.md) | Executable smoke input and distinct demo/publication protocols |
| [data](data/README.md), [results](results/README.md) | Episode provenance and run evidence layout |
| [scripts](scripts/), [tests](tests/), [.github](.github/) | Pin checks, smoke, source transcript checks, CPU CI and hardware test specification |

The measured comparison will cover one Spark with MPS, two Sparks connected by a 100 Gb/s CX7 DAC, and a GB300 station with separate MIG GPU instances. The receiver and sensing architecture derives from the [CUDA-OCUDU paper](https://arxiv.org/abs/2608.04338v1) and [cuSense paper](https://arxiv.org/abs/2512.06493v3). The [reuse ledger](docs/integration/UPSTREAM.md) identifies code, APIs, datasets and integration work.
