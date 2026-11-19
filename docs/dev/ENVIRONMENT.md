# Provision and verify the `isac` environment

Start at the repository root. Use one local `isac/` virtual environment per checkout on each machine. Python packages and the native launchers belong under that environment. Linux drivers, development headers, the CUDA toolkit, and the compiler are host prerequisites. CMake writes platform-specific objects under `build/`. Recreate the venv and build products when moving to another architecture.

| Host role | S0 profile | Evidence required before using that host |
|---|---|---|
| Component developer or CPU CI | `core` | Isolated interpreter, editable project import, component tests. NumPy enables the tensor-contract tests. |
| Offline scene generator or trainer | `full` | Three verified source trees, both Python distributions from those trees, zero-noise PHY and a real LOS path solve on the selected backend. |
| M1 Spark or later CUDA receiver host | `core` for a native-only role, `full` if it also runs simulation | Host CUDA toolchain and backed native executable checked in S1. Record the native result separately on each receiver platform. |

For the initial collocated bring-up use the **full profile** and run S0.1, S0.2, S0.3 and S1. A split deployment can publish separate environment reports for its simulation and receiver hosts. A developer working on a pure component can close their unit task with its own profile and fixtures.

## S0.1 — Isolated interpreter and local package

**Entry and files.** A writable checkout, a POSIX shell on Linux, Git, GNU Make and Python 3.11 or later with venv/ensurepip. The two new entry points are `scripts/bootstrap_env.sh` and `scripts/verify_environment.py`. The project source remains editable in `src/`. Select a Python ABI with wheels for your target architecture. Python 3.12 is the distribution default on Ubuntu 24.04 and satisfies both simulation source packages' declared requirements.

**Host preparation.** On Ubuntu 24.04 the following commands install the common tools and the native development packages listed by the pinned OCUDU build-dependency script. The second install is for receiver/build machines. Install LLVM on a machine using RT's CPU backend. Use equivalent distribution packages or an approved environment module on other hosts.

```bash
sudo apt-get update
sudo apt-get install git make python3 python3-venv
# CUDA receiver build host
sudo apt-get install build-essential cmake pkg-config libfftw3-dev libmbedtls-dev \
  libsctp-dev libyaml-cpp-dev libgtest-dev
# Offline CPU ray-tracing host
sudo apt-get install llvm
```

For a CUDA host, use the installed DGX image/toolchain if it supports the device. If the toolkit is absent, enable the vendor repository for the distribution and CPU architecture, then install the toolkit package for the selected release (for example `sudo apt-get install cuda-toolkit-13-0` with the appropriate CUDA 13.0 repository configured). Load the toolkit module or place its `bin/` directory on `PATH`. Verify the actual host compiler and GPU architecture. The receiver runbook gives the target-specific CMake flags. [CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) · [Native build and baseline](NATIVE_BASELINE.md)

```bash
python3 --version
git --version
make --version
# Receiver host
cmake --version
c++ --version
nvcc --version
nvidia-smi --query-gpu=name,uuid,compute_cap,driver_version --format=csv
# On Spark the compiler must list sm_121. Use the identified SM on another GPU.
nvcc --list-gpu-code
```

**Implementation and first check.** Create the venv on this machine. The script chooses `PYTHON` (default `python3`) the first time and reuses an isolated `isac/` on retries. It installs into that venv even when run from an unactivated shell.

```bash
bash scripts/bootstrap_env.sh core
source isac/bin/activate
run_dir=$(mktemp -d "results/environment.XXXXXXXX")
python scripts/verify_environment.py --profile core --report "$run_dir/core.json"
python -m unittest discover -s tests -p 'test_environment.py' -v
ocudu-isac-replay --help
```

**Unit acceptance.** A local directory with spaces round-trips through pip's direct-install receipt. A global interpreter, system-site package leakage, a registry install in place of local sources, and a local package from a different path are rejected. A failed profile check writes a diagnostic report and returns nonzero. The environment unit tests use temporary directories, synthetic receipts and a local executable that responds to a version query. No CUDA, source download, or native test is required.

**Integration acceptance.** A fresh initialization and a retry reuse `isac/bin/python`. The report shows `status=pass`, the correct Python and machine, the local project receipt, and a successful import. The CLI help resolves from the activated environment. To exercise NumPy-dependent component tests on a core host, install `python -m pip install 'numpy>=2.2.6'` there. A standard-library-only run reports those tests as skipped. Advance to the portable fixture in S0.3 independently of any lab hardware.

## S0.2 — Install and exercise the pinned upstreams

**Entry and outputs.** Use the active S0.1 interpreter. This sprint materializes the three pinned source trees and installs both simulation distributions. `ext/ocudu` supplies C++/CUDA source to the subsequent S1 build. The full S1 build exposes `gnb` and `pusch_e2e_pipeline_test` in `isac/bin/`. The simulation check returns a JSON report and a failure exit code if the requested CPU or CUDA backend cannot run.

**Acquire the source.** Use one documented route, then verify actual file contents or checked-out commits. The direct `ext/sionna-rt` checkout supplies RT. Both 2.2.0 wheels were built from the supplied archives with the nested RT checkout empty. An isolated package-only install resolved `sionna.phy` from Sionna and `sionna.rt` from the direct RT wheel. The numerical runtime check below additionally exercises their binary dependencies. [Dependency acquisition and lock](../../ext/README.md)

```bash
# With access to the upstream Git remotes
make hydrate
# Alternatively, for a fresh checkout with the three supplied upstream ZIPs
# python scripts/source_archives.py /absolute/path/to/archives
python scripts/check_pins.py --worktrees > "$run_dir/sources.txt"
```

**Install.** The pinned Sionna 2.2.0 package declares Python ≥3.11, torch ≥2.9.1, NumPy ≥2.2.6, SciPy ≥1.15.3, h5py ≥3.15.1, matplotlib ≥3.10.8 and importlib-resources ≥6.5.2. The pinned RT 2.2.0 package requires Mitsuba 3.9.1 and Dr.Jit 1.5.0. Pip resolves the full declared dependency lists. Read each checkout's top-level `pyproject.toml` for the effective constraints. On a GPU host, provision a torch distribution supporting the actual GPU SM, Python ABI, CPU architecture and driver. On a CPU-only host select a CPU torch wheel if desired. Install a compatible locally staged torch wheel with `python -m pip install /path/to/compatible-torch.whl` before the full bootstrap, or use the vendor's pip command for that host. Record the wheel or index and its version. [PyTorch platform selector](https://pytorch.org/get-started/locally/)

```bash
bash scripts/bootstrap_env.sh full
python -m pip check
python -m pip freeze --all > "$run_dir/python-packages.txt"
python scripts/verify_environment.py --profile simulation --backend cpu \
  --report "$run_dir/simulation-cpu.json"
```

On a GPU simulation host run the same command with `--backend cuda` and a fresh report path. Select the desired device using `CUDA_VISIBLE_DEVICES` before launching the check. This probe explicitly selects PyTorch's visible device and Mitsuba's polarized CUDA variant. A requested CUDA probe cannot pass via fallback to LLVM. A split deployment can collect only the backend it will use.

The bootstrap invokes pip with **both local simulation paths in one install transaction** so the transitive `sionna-rt` requirement selects the direct pinned checkout. Project sources install in editable mode and both upstream distributions install as regular local builds. To initialize without a package index, set `PIP_NO_INDEX=1` and `PIP_FIND_LINKS=/absolute/path/to/wheelhouse` for the bootstrap. Supply build backends (including a setuptools release with PEP 660 support) and the complete transitive wheel closure for this host. A source archive supplies code only. Preserve the wheelhouse, or the wheel hashes and package-index provenance, with a published configuration. A pip freeze is an inventory of one host's realized environment.

**Unit acceptance.** The same portable receipt and negative-path tests in S0.1 cover registry substitution, the wrong local tree, a missing dependency, and failed backend status reporting. Source lock checks compare the installed worktree to `ext/lock.json`. These tests can run before provisioning either simulation backend.

**Integration acceptance.** Both installed source receipts resolve to their local pinned paths, pip reports consistent dependencies, and the selected backend executes two numerical checks. Sionna PHY's zero-noise AWGN produces the original complex symbols on the requested device. Sionna RT traces an unobstructed UE-to-four-element-gNB path and returns a finite, nonzero complex CFR of shape `[1,4,1,1,4,12]`. The report includes the actual package versions, Mitsuba variant, torch device, CFR peak and shape. The physical target-present/export check belongs to S2. Preserve the failure report if a GPU wheel, LLVM shared library, driver or toolkit is missing and retry under a fresh report filename after correcting that host.

**Native handoff.** From this same activated venv follow S1 to compile the receiver and run the measured paper workload. After the full build, include `--with-native` to match both executables and their SHA-256 hashes to the same S1 full-build receipt and locked commit. The gate executes `gnb --version` and requires the pinned `OCUDU 5G gNB version 26.04.0 (...)` banner. A Git build must print an abbreviation of the locked revision. Verified archive builds can print empty parentheses because upstream obtains its hash from Git. The environment report records both resolved executables and the complete version line. Preserve both the environment report and the graded native transcript. The first establishes gNB startup through option parsing and receiver executable availability. The second establishes a receiver result. [Native build and baseline](NATIVE_BASELINE.md)

```bash
bash scripts/build_ocudu.sh
python scripts/verify_environment.py --profile simulation --backend cuda --with-native \
  --report "$run_dir/collocated.json"
bash scripts/run_spark_baseline.sh --profile paper --skip-build
```
