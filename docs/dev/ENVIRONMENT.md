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

**Implementation and first check.** Create the venv on this machine. The script chooses `PYTHON` (default `python3`) the first time and reuses an isolated `isac/` on retries. It installs into that venv even when run from an unactivated shell. Before writing anything it requires a 3.11+ base interpreter that can import `venv` and `ensurepip` (on Ubuntu, `sudo apt-get install python3-venv`). A creation that fails removes the partial `isac/` it just made. The script never deletes a directory it did not create in the same invocation: an existing `isac/` that is incomplete, was created at another path, or can import packages from outside the standard library, the venv and this checkout (typically through `PYTHONPATH`) stops the run before pip is used and prints the repair. A virtual environment cannot be moved or copied to another path or host, so the repair is `rm -rf isac` followed by `make env`. The editable install needs a package index or wheelhouse that provides `setuptools>=64`.

```bash
bash scripts/bootstrap_env.sh core
source isac/bin/activate
run_dir=$(mktemp -d "results/environment.XXXXXXXX")
python scripts/verify_environment.py --profile core --report "$run_dir/core.json"
python -m unittest discover -s tests -p 'test_environment.py' -v
ocudu-isac-replay --help
```

**Unit acceptance.** A local directory with spaces round-trips through pip's direct-install receipt. A global interpreter, system-site package leakage, a registry install in place of local sources, and a local package from a different path are rejected. So are a venv recorded at another path, an activation that points at a different environment, an import root outside the standard library, the venv and the checkout, a project install that is not editable, a project import that resolves outside `src/`, an inconsistent `pip check`, an unknown profile name and a report path that already exists. The inventory lists each resolved distribution once. A failed profile check writes a diagnostic report and returns nonzero. The bootstrap script runs in a scratch checkout against interpreter doubles for an old Python, a missing `ensurepip`, a missing interpreter, a creation that fails half way, a directory that is not a venv, an incomplete venv, a leaked `PYTHONPATH` and a moved checkout. Each case stops before pip installs anything. The environment unit tests use temporary directories, synthetic receipts and a local executable that responds to a version query. No CUDA, source download, or native test is required.

**Integration acceptance.** A fresh initialization and a retry reuse `isac/bin/python`. The report shows `status=pass`, the correct Python and machine, the OS release, node name, `venv`, `prefix` and `base_prefix`, the local project receipt with `editable: true`, a `project_import` under `src/`, every resolved distribution under `installed`, and the `pip_check` output. An existing report path is refused with exit code 2 before any check runs, so each acquisition needs a new name. The CLI help resolves from the activated environment. To exercise NumPy-dependent component tests on a core host, install `python -m pip install 'numpy>=2.2.6'` there. A standard-library-only run reports those tests as skipped. Advance to the portable fixture in S0.3 independently of any lab hardware.

## S0.2 — Install and exercise the pinned upstreams

**Entry and outputs.** Use the active S0.1 interpreter. This sprint materializes the three pinned source trees and installs both simulation distributions. `ext/ocudu-deepsig-cuda-inline` supplies C++/CUDA source to the subsequent S1 build. The full S1 build exposes `gnb` and `pusch_e2e_pipeline_test` in `isac/bin/`. The simulation check returns a JSON report and a failure exit code if the requested CPU or CUDA backend cannot run.

**Acquire the source.** Use one documented route, then verify actual file contents or checked-out commits. The direct `ext/sionna-rt` checkout supplies RT. Both 2.2.0 wheels were built from the supplied archives with the nested RT checkout empty. An isolated package-only install resolved `sionna.phy` from Sionna and `sionna.rt` from the direct RT wheel. The numerical runtime check below additionally exercises their binary dependencies. [Dependency acquisition and lock](../../ext/README.md)

```bash
# With access to the upstream Git remotes
make hydrate
# Alternatively, for a fresh checkout with the three supplied upstream ZIPs
# python scripts/source_archives.py /absolute/path/to/archives
python scripts/check_pins.py --worktrees > "$run_dir/sources.txt"
```

**Install.** The pinned Sionna 2.2.0 package declares Python ≥3.11, torch ≥2.9.1, NumPy ≥2.2.6, SciPy ≥1.15.3, h5py ≥3.15.1, matplotlib ≥3.10.8 and importlib-resources ≥6.5.2. The pinned RT 2.2.0 package requires Mitsuba 3.9.1 and Dr.Jit 1.5.0. Pip resolves the full declared dependency lists. Read each checkout's top-level `pyproject.toml` for the effective constraints. On a GPU host, provision a torch distribution supporting the actual GPU SM, Python ABI, CPU architecture and driver. On a CPU-only host select a CPU torch wheel if desired. Install a compatible locally staged torch wheel with `python -m pip install /path/to/compatible-torch.whl` before the full bootstrap, or use the vendor's pip command for that host. Record the wheel or index and its version. [PyTorch platform selector](https://pytorch.org/get-started/locally/)

**Selections checked against the pinned 2.2.0 documentation.** The pinned Sionna documentation asks for Python 3.11+, PyTorch 2.9+ and recommends Ubuntu 24.04. It gives Sionna RT the requirements of Mitsuba 3 and requires LLVM for Dr.Jit on CPU. Sionna RT 2.2.0 itself selects `cuda_ad_mono_polarized` and falls back to `llvm_ad_mono_polarized`. The probe sets exactly those two variants for `--backend cuda` and `--backend cpu`.

| Choice | Selected | Why it fits the host |
|---|---|---|
| PyTorch | `torch` from PyPI, 2.15.0 at the time of validation | Its Linux aarch64 and x86_64 wheels are CUDA 13.2 builds compiled for sm_80 through sm_120. A GB10 reports compute capability 12.1 and runs sm_120 code. CUDA 13.x needs driver 580 or newer. |
| Mitsuba and Dr.Jit | 3.9.1 and 1.5.0, the exact pins of Sionna RT 2.2.0 | PyPI has Python 3.12 wheels for aarch64 and x86_64 that contain both polarized variants. |
| CPU ray tracing | System LLVM | Dr.Jit 1.5.0 needs LLVM 15 or newer and searches `/usr/lib/{x86_64,aarch64}-linux-gnu/libLLVM*.so.*`. Set `DRJIT_LIBLLVM_PATH` to choose a library. Ubuntu 24.04 with LLVM 17, 18 and 20 installed loaded 20.1. |
| GPU ray tracing | Driver CUDA and OptiX | Dr.Jit 1.5.0 needs compute capability 7.5 or newer, `libcuda.so` and `libnvoptix.so.1`, and rejects drivers 559 to 565. `DRJIT_LIBCUDA_PATH` and `DRJIT_LIBOPTIX_PATH` override the search. |

`environments/full-constraints.txt` records the third-party set that was installed and probed. Set `PIP_CONSTRAINT` to its absolute path to reproduce that set, or leave it unset to let pip resolve newer releases. Install a different torch build before the bootstrap only when the CUDA probe shows that the host needs one.

```bash
# Optional: reproduce the validated third-party set
# export PIP_CONSTRAINT="$PWD/environments/full-constraints.txt"
bash scripts/bootstrap_env.sh full
python -m pip check
python -m pip freeze --all > "$run_dir/python-packages.txt"
python scripts/verify_environment.py --profile simulation --backend cpu \
  --report "$run_dir/simulation-cpu.json"
```

On a GPU simulation host run the same command with `--backend cuda` and a fresh report path. Select the desired device using `CUDA_VISIBLE_DEVICES` before launching the check. This probe explicitly selects PyTorch's visible device and Mitsuba's polarized CUDA variant. A requested CUDA probe cannot pass via fallback to LLVM. A split deployment can collect only the backend it will use.

The bootstrap invokes pip with **both local simulation paths in one install transaction** so the transitive `sionna-rt` requirement selects the direct pinned checkout. Project sources install in editable mode and both upstream distributions install as regular local builds. To initialize without a package index, set `PIP_NO_INDEX=1` and `PIP_FIND_LINKS=/absolute/path/to/wheelhouse` for the bootstrap. Supply build backends (including a setuptools release with PEP 660 support) and the complete transitive wheel closure for this host. A source archive supplies code only. Preserve the wheelhouse, or the wheel hashes and package-index provenance, with a published configuration. A pip freeze is an inventory of one host's realized environment.

**Unit acceptance.** The same portable receipt and negative-path tests in S0.1 cover registry substitution, the wrong local tree, a missing dependency, and failed backend status reporting. Source lock checks compare the installed worktree to `ext/lock.json`. The pin gate runs against local Git fixtures: a renamed checkout, an undeclared or uncommitted submodule, a changed revision and an edited tracked file are rejected. An installed file that differs from the pinned tree, a pinned module that is not installed and a build from another commit with the same version number are rejected with doubles. The free-space reference has a baked expected value, and a wrong gain, a wrong delay and a conjugated channel fail the direct-path check. These tests can run before provisioning either simulation backend.

**Integration acceptance.** The report lists the three source identities under `sources`, each verified against `ext/lock.json`. Both installed source receipts resolve to their local pinned paths, every installed Sionna and Sionna RT file equals the pinned tree (`files_matching_source`), pip reports consistent dependencies, and the selected backend executes two numerical checks. Sionna PHY's zero-noise AWGN produces the original complex symbols on the requested device. Sionna RT traces an unobstructed UE-to-four-element-gNB path and returns a finite complex CFR of shape `[1,4,1,1,4,12]` whose gain and delay equal the free-space answer for the fixed geometry (relative gain error at most 1e-4, delay error at most 0.05 ns). The report includes the actual package versions, Mitsuba variant, torch device, CFR peak and shape. The physical target-present/export check belongs to S2. Preserve the failure report if a GPU wheel, LLVM shared library, driver or toolkit is missing and retry under a fresh report filename after correcting that host.

**Native handoff.** From this same activated venv follow S1 to compile the receiver and run the measured paper workload. After the full build, include `--with-native` to match both executables and their SHA-256 hashes to the same S1 full-build receipt and locked commit. The gate rejects a build directory that was reconfigured after its receipt. It starts `pusch_e2e_pipeline_test --help` from the environment and requires the three options the measured row passes. It executes `gnb --version` and requires the pinned `OCUDU 5G gNB version 26.04.0 (...)` banner. A Git build must print an abbreviation of the locked revision. Verified archive builds can print empty parentheses because upstream obtains its hash from Git. The environment report records both resolved executables and the complete version line. Preserve both the environment report and the graded native transcript. The first establishes gNB startup through option parsing and receiver executable availability. The second establishes a receiver result. [Native build and baseline](NATIVE_BASELINE.md)

```bash
make accept-s1.1 BACKENDS="cpu cuda"    # S1.1: build, grade, start both executables, archive
bash scripts/run_spark_baseline.sh --profile paper --skip-build
```

## Sprint 0 acceptance on one host

`scripts/accept_s0.sh` runs the S0.1, S0.2 and S0.3 gates in one pass and writes the evidence to a new directory under the ignored `results/runs/`. It only runs the commands above, saves their output and prints one verdict. Name the simulation backends the host must pass. With none, it accepts a development host on the core gates alone and says that the simulation check still needs a provisioned scene host.

```bash
make accept-s0 BACKENDS="cpu cuda"      # collocated Spark: core, sources, both simulation backends, replay
make accept-s0                           # component developer or CPU CI host
```

The run directory holds `summary.txt`, the commit and `git-status.txt`, `core-env.json`, `sources.txt`, `python-packages.txt`, one `simulation-<backend>.json` per backend, the two replay outputs with `smoke-sha256.txt`, and `unit-tests.log`. The verdict is `ACCEPTED` only when the tree is committed, every gate exits zero, the replay shows four scheduled updates with one on time, one late, one dropped and one missing, its three files repeat byte for byte, and the unit suite ran with no failure. On a simulation host a skipped test also fails the gate. An existing run directory is never reused.
