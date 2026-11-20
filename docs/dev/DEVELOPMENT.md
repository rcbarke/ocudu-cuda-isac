# Implementing receiver-aware ISAC on CUDA-OCUDU

**First hardware result.** On one DGX Spark, build the pinned WG1 receiver and reproduce the correctness comparison in the 20 MHz, one-layer row of DeepSig Table II. **First advisor demonstration.** Feed the working receiver a known uplink through a replayed moving-target channel. On the same Spark, a second MPS client makes a causal presence decision from the pilots in the exact received grid. Show the decoded payload, the decision and both timing distributions. **Publication.** Measure how receiver processing, model choice, load and placement affect timely sensing at a fixed false-alarm operating point while preserving a specified radio service level.

## Choose a sprint

Each numbered card is one reviewable implementation assignment. S0–S9 preserve the original milestone IDs. The first column below is the recommended acquisition order. Component developers may pull any unit ticket once they have the listed fixture and language toolchain. Reserve named machines and adjacent services for the card's integration step.

| Phase | Pick a sprint | Integration dependency and phase exit |
|---|---|---|
| [S0 · Install and verify](#phase-s0) | S0.1 local environment · S0.2 pinned sources and simulation · S0.3 host replay oracle | Development host produces an environment report, exact source identities and repeatable four-opportunity fixture. A separate provisioned scene host can perform the full simulation check. |
| [S1 · Prove the receiver](#phase-s1) | S1.1 native Release build · S1.2 Table II row | On a provisioned Spark, link the gNB and test executable into `isac/bin`, then grade real CPU/CUDA decodes. Close initial bring-up before integrating project features. |
| [S2 · Drive receiver ingress](#phase-s2) | S2.1 legal transmitter fixture · S2.2 matched bistatic episodes · S2.3 native grid arithmetic · S2.4 actual PUSCH dispatch | Combine S1 + three independently unit-tested adapters to recover the original payload on CPU and GPU. |
| [S3 · Produce a causal observation](#phase-s3) | S3.1 pilot LS · S3.2 ownership and backpressure · S3.3 features and known return · S3.4 frozen presence model | Join observations to S2 decodes and run a real second-process inference. |
| [S4 · Show M1 on one Spark](#phase-s4) | S4.1 paced clients · S4.2 evidence accounting · S4.3 demonstration | S1–S3 with two attached MPS clients, matched controls, held-out episodes and overload evidence. |
| [S5 · Operate the DU](#phase-s5) | S5.1 scheduler binding · S5.2 O3 producer · S5.3 sustained operation | S4 + live DU allocations, estimator identity and 30-minute evidence. |
| [S6 · Test the science](#phase-s6) | S6.1 propagation · .2 detection · .3 angle · .4 splits · .5 encoders · .6 angle heads · .7 hypotheses · .8 measured capture extension | Use S2/S3 fixtures for algorithm development. S5.2 enables genuine O3 comparisons. |
| [S7 · Split two Sparks](#phase-s7) | S7.1 socket framing · .2 host-staged RDMA and clocks · .3 placement/core controls | Use the M1 observation contract on both nodes for end-to-end validation. |
| [S8 · Partition GB300](#phase-s8) | S8.1 GI binding/bridge · .2 matched station controls | S7 and S8 can integrate independently after S4, with S5 receiver evidence for publication. |
| [S9 · Freeze and reproduce](#phase-s9) | S9.1 preregister comparisons · .2 reduce evidence · .3 independent reproduction | Freeze S9.1 **before held-out acquisition**. Use archived S5–S8 runs to close S9.2–.3. |

The original proposal also discusses Aerial, WG2/E3 dApps, independent OTA and packetized 7.2x. Here the executed receiver is WG1 CUDA-OCUDU. cuSense informs the pilot/feature and inference comparison. Independent measured captures have their own S6.8 ticket. Integration with another stack or an OTA radio follows the primary study and needs its own evidence. [System](../architecture/SYSTEM.md) · [source and comparator ledger](../integration/UPSTREAM.md)

## How to implement and sign off one ticket

1. Select the card, its production owner and fixtures. Open one patch against the named package or native target. Resolve the exact upstream symbol and commit first. Reuse OCUDU's grants, PUSCH and CRC, Sionna's NR generator/ISAC primitives, RT's solvers, library FFT/eigensolvers and PyTorch layers. Commit small fixtures or their creation recipe with an independent expected answer.
2. Implement the production interface and run its focused unit test inside this checkout's `isac`. Inject a test clock, fake grid/transport or small tensor as specified. A component can be **unit accepted** before any hardware integration. Upstream whole-project integration suites are separate from building an individual adapter.
3. When the integration prerequisites exist, run the real producer and consumer. Grade against the named observable. Archive a unique run ID, command, inputs, toolchain/commit/patch hashes, raw logs, counts, judgment and reason. Preserve failed attempts. Record a blocked host or missing optional system explicitly. A skip, zero executed cases, a dry run or a fallback cannot close the requested hardware gate.
4. Mark the sprint **integrated** only with its real evidence. Keep unit and integration status separately in the issue/PR. For studies, a correctly measured adverse result completes the experiment. Passing a functional gate never implies a published speedup or service level.

**Command contract for future tickets.** The S2–S9 card heading gives a selector. Each implementing patch provides a directly executable `tests/unit/test_<selector>.py` with a `unittest.main()` entry point and a matching `tests/integration/<selector>.py` plus `configs/integration/<selector>.json`. Create the specific files with the feature. The native adapter may use a small CMake/CTest target with a fake interface and a thin Python test wrapper. Production C++ should build against the locked headers, and CPU-only unit targets should link only their module and doubles. Each integration driver takes `--config FILE --output-dir DIR`, refuses to overwrite an existing run, and writes a grade with an explicit case count. The two commands below exercise S3.1 after implementation.

```sh
python tests/unit/test_pilot_ls.py -v
python tests/integration/pilot_ls.py --config configs/integration/pilot_ls.json \
    --output-dir results/runs/s31-spark-001
```

These are build contracts for future patches. Runnable commands in S0/S1 are identified below. Keep hardware drivers outside default unit discovery and make each hardware command opt-in. Use `tests/unit/__init__.py` for discovery when adding the first future unit module. Never count an absence of imported tests as a successful gate. [Evidence format](REPRODUCE.md)

## Interfaces shared across tickets

The [replay contract](../interfaces/REPLAY.md) defines the shipped `fixture-grid/v1` vectors. The following fields belong to the production interfaces introduced by S2–S5. Keep the fixture readable for regression and add a schema version for the production records.

| Crossing | Required record and invariant |
|---|---|
| S2.1/S2.2 → S2.3 | Keep the grant, original TB, X/H/W arrays, axis/units metadata and checksums in the stimulus store. Bind one realization to its grant. Map frequency by physical offset and time by actual symbol timestamp. |
| S2.3 → S2.4 | Pass an owned or retained Y grid, its generation, PDU and ready event. The decoder sees the matching legal PUSCH allocation. Save the readback digest and true decoder callback independently of observer state. |
| S3.1 or S5.2 → S3.2 | Define a versioned descriptor with run/epoch, opaque episode and grant/opportunity IDs, cell/UE, absolute slot, source stage (O2/O3), shape, dtype, strides, mask, antenna map, time unit/domain and completion handle. Use opaque IDs only for joins. |
| S3.2 → S3.3/S3.4 | Transfer immutable observations and release credits by generation. A decision identifies all contributing observations, clock domain, first/last sample, publication time, frozen model/threshold and validity status. |
| S4–S9 → offline evaluation | Use `radio.jsonl` for true CRC, byte counts and terminal radio outcomes, `events.jsonl` for releases/copies/drops and `decisions.jsonl` for publications. Join to evaluator-only labels and original TBs after the run. Missing outcomes remain in their original denominators. |

For a minimal fake producer, use one epoch, one UE and two grants with distinct opportunity IDs. Repeat a slot number after restart to test the epoch boundary. Treat an inactive pilot and a physically measured zero as different mask states. A score from an older generation cannot close a newer opportunity.

## Handoff status

**Sprint 0 is accepted on one DGX Spark.** `make accept-s0 BACKENDS="cpu cuda"` passed every S0 gate on host `spark-ecf8` (aarch64, Ubuntu 24.04.5, Python 3.12.3, NVIDIA GB10 with driver 580.178.04) on 2026-11-19 at commit `49b6092`, run `s0-spark-ecf8-20261119T222041Z`. The run directory under the ignored `results/runs/` holds the reports and stays on that host. Run the same command on every further host before it produces evidence. Without `BACKENDS` it accepts a core-only development host and says that the simulation check still needs a provisioned scene host. [Environment runbook](ENVIRONMENT.md)

The accepted state differs from the original delivery in five ways.

- **Sources.** The three sources are Git submodules at the commits in `ext/lock.json`. The lock is the only place that names a checkout directory, and OCUDU lives in `ext/ocudu-deepsig-cuda-inline`. The ZIP import remains for a host without Git access and needs the three original archives named in the lock.
- **Environment.** The full profile installs the PyPI CUDA 13.2 build of torch 2.15.0 with Mitsuba 3.9.1 and Dr.Jit 1.5.0. `environments/full-constraints.txt` records the complete third-party set.
- **Simulation.** The numerical probe has run on the Spark with both backends: Sionna PHY on the CPU and on the GB10, and Sionna RT through LLVM and through CUDA with OptiX. Each traced channel matches the free-space gain and delay of the probe geometry, and every installed Sionna file equals the pinned tree.
- **Replay.** The four-opportunity fixture writes byte-identical files on x86_64 and aarch64. The test suite records their digests.
- **Hosted checks.** `.github/workflows/portable.yml` runs the core environment gate, the pin check, the unit suite and the replay fixture on a CPU runner for Python 3.11, 3.12 and 3.13.

**Sprint 1.1 is accepted on the same Spark.** `make accept-s1.1 BACKENDS="cpu cuda"` passed all 20 gates on `spark-ecf8` on 2026-11-20 (UTC) at commit `cb9451e`, run `s11-spark-ecf8-20261120T011024Z`. The locked OCUDU source built as a full Release with `ENABLE_CUDA=ON`, CUDA architecture 121 and `MCPU=neoverse-v2` (GCC 13.3.0, CMake 3.28.3, `nvcc` 13.5.37). The 18 kernel sources compiled for `sm_121`, both executables carry linked device code, and `gnb` and `pusch_e2e_pipeline_test` start from `isac/bin`. S1.1 is unit accepted and integrated. Two earlier runs stopped in the configure step because OCUDU's development libraries were missing. The builder now installs them with `scripts/install_native_deps.sh`. [Acceptance record](NATIVE_BASELINE.md)

**Sprint 1.2 is accepted on the same Spark, which closes Sprint 1.** `make accept-s1.2` passed all 27 gates on `spark-ecf8` on 2026-11-20 (UTC) at commit `ea7bb43`, run `s12-spark-ecf8-20261120T144154Z`. It measured the executables that the S1.1 run accepted, with nothing rebuilt. The Table II 20 MHz one-layer row decoded 200 of 200 frames on the CPU path and on the GPU path with no differing byte, and every measured frame used the resident GPU path. The four-RX smoke decoded 20 of 20 on each path. S1.2 is unit accepted and integrated. The measured means are recorded and not graded: 397.2 us on the CPU and 239.6 us on the GPU, against 334.0 and 183.9 us in the paper. They were accepted as comparable for bring-up on 2026-11-20, with desktop applications running, and system tuning is deferred until the project features are implemented. [Acceptance record](NATIVE_BASELINE.md)

S2.1 `grant_fixture` is the next sprint and the first project-owned component. The native wrapper finds the OCUDU source through the lock. S2 to S9 name implementation deliverables and lab gates. The launchers and selectors in those cards become executable as their owning patches land.

<a id="phase-s0"></a>

## S0 · Install on a fresh host

### S0.1 Create the local environment

**Build and inputs.** Start with Python 3.11+, venv/ensurepip, Bash, Git, Make and the OS packages in the [environment runbook](ENVIRONMENT.md). Run `make env`, then `source isac/bin/activate` from the root on *each* host. The bootstrap installs this checkout as an editable project. Use a separate venv on each node and record OS, architecture, Python and resolved distributions. Native compilers, driver, toolkit and device access belong to the host. Native commands and Python processes are launched in the project's activated environment.

**Unit.** `python -m unittest discover -s tests -p 'test_environment.py' -v` exercises an incorrect interpreter, accidental global packages and a mismatched install receipt with doubles. A bad profile must fail clearly.

**Integrate and accept.** Run `python scripts/verify_environment.py --profile core --report build/core-env.json`. Accept when the command exits zero, identifies this exact venv and project, and pip reports a consistent installation. Use a new report name for a repeated acquisition. This is sufficient to develop host-side contracts.

### S0.2 Acquire the sources and exercise PHY and ray tracing

**Build and inputs.** Use `make pins`, then `make hydrate` or `python scripts/source_archives.py /path/to/original/archives`. Run `python scripts/check_pins.py --worktrees`. Install the hardware-appropriate PyTorch wheel if needed, then `make env-full` in the activated environment. Both Sionna projects come from the local pins. Follow the [per-host install and troubleshooting sequence](ENVIRONMENT.md) for wheels, LLVM, CUDA, Mitsuba and Dr.Jit.

**Unit.** Run `python -m unittest discover -s tests -p 'test_source_archives.py' -v` and the environment tests. Known file, symlink and permission fixtures must round-trip. A changed archive, receipt or tracked file must fail verification. Importing the same valid archive twice must preserve the installation.

**Integrate and accept.** Run `python scripts/verify_environment.py --profile simulation --backend cpu --report build/simulation-cpu.json` or select `--backend cuda` on the designated CUDA simulation host. Require the pinned distributions' local origins, a zero-noise PHY operation on the requested device, and a nonzero finite direct-path CFR on the requested RT backend. Capture the wheel set. S0.2 verifies OCUDU source availability. S1 verifies the native binary and its execution. On a receiver-only node a core install plus native prerequisites suffices to execute S1. For the overall bring-up retain a real simulation report from the designated scene-generation environment.

### S0.3 Verify the portable replay and clock fixtures

**Delivered interface.** `configs/smoke_replay.json` plus `configs/smoke_timing.json` feed `src/ocudu_cuda_isac/replay.py`. Its flat complex arrays use last-axis-contiguous order. The public receiver file contains O1 and the permitted mask/timestamps. The separate evaluator gets X/H/W and truth.

**Unit.** `python -m unittest discover -s tests -p 'test_replay.py' -v`. The two-transmit, two-symbol case yields `[1+3j, 2-4j, 6.5+0j, 14+0j]`. Unknown/duplicate sequence IDs and a mismatched clock must be rejected. Reordering valid completion events leaves the same outcome.

**Integrate and accept.** Run `make smoke` or `make check`. Expect exactly four scheduled, one on time, one late, one dropped and one missing in `build/smoke/timing_summary.json`. Repeat the same fixture and compare all three output files byte-for-byte. Preserve the distinction between synthetic timestamps and physical measurements. Install NumPy to execute the optional RT serialization tests as well. [Replay and axes](../interfaces/REPLAY.md)

<a id="phase-s1"></a>

## S1 · Build and prove the pinned OCUDU receiver

### S1.1 Build the Release and expose the native test inside isac

**Inputs and commands.** Use a pristine locked OCUDU source and the [native build and prerequisite runbook](NATIVE_BASELINE.md). `make native-build` launches the supplied builder. The proposal uses the following Spark build from the OCUDU source root.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 \
  -DMCPU=neoverse-v2
cmake --build build -j8
```

The wrapper uses equivalent source/build arguments and enables the upstream test target required in S1.2. Keep the reusable upstream control pristine and build out of tree. Select each host's supported architecture and CPU target explicitly. Record the cache and compiler evidence for real CUDA compilation. The numeric Spark architecture in this recipe is specific to GB10.

**Unit.** Exercise native argument generation and wrong profile/cache/toolchain conditions using the isolated tests named in the runbook. **Integrate and accept.** On Spark, keep the build log, source identity, compiler versions, compiled targets and hashes. From the activated environment run `gnb --version` and `pusch_e2e_pipeline_test --help`. Run `python scripts/verify_environment.py --profile core --with-native --report build/native-env.json`. On a combined simulation/receiver machine, `--profile simulation --backend cuda --with-native` verifies all three dependencies from one interpreter. `make accept-s1.1` runs the build and every check of this card in one pass and writes the evidence to a new run directory. Add `BACKENDS="cpu cuda"` on a combined machine. The gate list and the reference-environment comparison are in the runbook. The native CPU/GPU result comes next.

### S1.2 Reproduce one DeepSig receiver result

**Build.** Use `make paper-baseline` after S1.1. Keep the simpler `make spark-baseline` fixture for diagnosis. The comparison point is Table II, 20 MHz, one layer, QAM256 MCS 27. The paper reports 200/200 successful CPU/GPU decodes and zero differing bytes. Its CPU/GPU mean service times are 334.0/183.9 µs on its earlier revision. Record the local timings, actual transport-block size, selected grid path, revision and all profile differences. [DeepSig paper](https://arxiv.org/abs/2608.04338v1) · [commands and evidence grading](NATIVE_BASELINE.md)

**Unit.** Grade synthetic transcripts with known pass and fail counts. Zero iterations, missing GPU residency, guard corruption, mismatch, wrong workload or missing timing cannot pass the applicable gate. Parser regression fixtures supply no empirical result. **Integrate and accept.** Require a successful process exit, real completed CUDA work on the intended device, 200 measured native CPU and GPU CRC passes apiece, matching decoded bytes and measured, finite service times. Grade reproduction of correctness separately from the measured performance comparison. This native test compares the two decoded outputs. S2 additionally checks both against the original transmitted bytes. Retain raw logs and the machine-readable comparison. `make accept-s1.2` runs the paper row and the four-RX smoke on the build that S1.1 accepted, grades every check of this card and writes the evidence to a new run directory. `make accept-s1` runs the S1.1 gate first. The gate list is in the runbook. The project contribution begins with S2 after this baseline.

<a id="phase-s2"></a>

## S2 · Deliver a known waveform to the native receiver

### S2.1 Export one legal uplink grant and transmit reference · `grant_fixture`

- **Build.** Own `src/ocudu_cuda_isac/episodes/grant.py` and a thin native fixture adapter in `native/ocudu_adapter/fixtures/`. Persist original TB bytes, RNTI, cell/scrambling IDs, numerology, BWP start/size, slot, symbols, PRB/RE ordering, DMRS type/ports/indices and reference symbols, TX scale, layers and port mapping. Reuse the pin's `run_e2e_pipeline_test` fixture. Its ordinary non-transform-precoded path maps a compatible PDSCH test grid. Cross-check against Sionna NR `PUSCHConfig`, `PUSCHDMRSConfig` and `PUSCHTransmitter` before using it as a UE reference. Capture per-RE data and pilots in the same carrier coordinate system. [Source map](../integration/OCUDU.md)
- **Unit.** Use a baked one-layer, one-port grant with two allocated PRBs offset inside a larger carrier. Verify 24 indexed subcarriers per full allocated symbol, order, byte/bit order and expected pilot locations. Invalid port, out-of-BWP PRB, duplicate pilot or mismatched payload/TBS fails before allocation. A mock grid suffices.
- **Integrate.** At S1's native boundary feed fixed known bits to both transmit references. Align scramble, normalization and phase. Verify mask and data-bearing RE set exactly and complex symbols to the declared representation tolerance. Archive the grant, bytes and a hash for each grid.
- **Accept.** A second developer can recover a pilot coordinate, one known data symbol and original TB hash from the record. If the fixture and true PUSCH disagree, persist the discrepancy and fix the grant/mapper before S2.4.

### S2.2 Generate a matched target-present/absent pair · `episode_pair`

- **Build.** Extend `scripts/export_rt_episode.py` via an episode module accepting S2.1's carrier offsets, actual symbol times and antenna order. Retain its existing four-time/twelve-tone diagnostic as a regression. Produce direct-only and moving-target cases with independently recomputed background for the physically absent scene. Keep waveform/noise fixed. Generate physical H using `PathSolver`, `RCSSolver`, `Paths.concat` and `Paths.cfr`. Preserve absolute delay and amplitude. For a uniform solver grid, evaluate path phase at the real symbol times or justify resampling with a measured phase error. Retrace visibility for pronounced motion. Record path-validity interval and update cost separately if enabling online updates.
- **Unit.** Feed deterministic complex coefficients with indices `(r,t,m,k)` encoded in the real/imaginary parts. Preserve singleton TX and original RX order, little-endian complex64, exact requested symbol timestamps, units and hashes. Inject one corrupted byte, swapped axis, missing array or partial write. The loader must reject the pair. A single path with $H(f,t)=a e^{-j2\pi f\tau}e^{j2\pi \nu t}$ recovers the specified phase differences at two unequal time intervals. Store weaker echoes with a scale-aware tolerance.
- **Integrate.** In the full S0.2 environment run the diagnostic with `python scripts/export_rt_episode.py --output-dir build/rt-pair-001`, then the grant-aligned implementation. Save array/calibration, carrier, frequency and time axes, mask, seed, scene/material hash, solver flags, versions and checksummed NPY files in the offline manifest. Check finite nonzero direct paths for both and a valid additional scattering path for the present case.
- **Accept.** The same stored pair is readable on a second host, matches every grant coordinate and has an independent analytic phase check. The current diagnostic's nominal 360 kHz occupied width and 1.5 ms first-to-last time span establish an axis test only. Create a separate grant-aligned demo trajectory. At 51 PRBs and 30 kHz SCS the full occupied bandwidth is 18.36 MHz (nominal delay resolution about 54.5 ns). Eight vectors spaced 5 ms apart give nominal 25 Hz Doppler bins spanning −100 to +100 Hz modulo aliasing. The diagnostic geometry yields only about 5 ns of excess delay and 9 Hz of Doppler. For a stronger unit/reference case, keep the 20 m direct link and put a reflector near its midpoint, 25 m off-axis, moving outward at 2 m/s at 3.5 GHz. This gives about 113 ns of excess delay and a Doppler magnitude near 43 Hz. Validate sign, SNR, sidelobes and both resolutions on the actual pilot mask and symbol times before freezing the demo case.

### S2.3 Implement native channel application · `grid_apply`

- **Build.** In `native/ocudu_adapter/replay/`, implement an allocation-aware reader for preloaded H, X and W, writing owned O1 grids to a supplied writer. Use X[tx,sym,k], H[rx,tx,sym,k] and W/Y[rx,sym,k]. Compute `Y=sum_tx(H*X)+W` once. Time-indexed H already carries Doppler. Retain gain convention, sparse active indices, readback hash, grant ID, fill/ready/release state and actual dtype conversion. Use a CPU reference and upstream writers first.
- **Unit.** Use S0.3's two-transmit vector and exact four answers. A SISO unity channel or a square identity MIMO channel with zero W preserves X, zero H yields W for any X, opposite transmit terms cancel, a deterministic antenna permutation is caught. For pre-quantized unit-scale complex64 use `abs(error) <= 1e-6 + 1e-5*abs(reference)`. Compare compressed/BF16 readback to the reference rounded to the documented storage representation. Test a stale generation and changed allocation.
- **Integrate.** Apply the same immutable pair and noise to the identical grant in Python and native code. Read back the exact grid delivered to the pinned OCUDU writer. Prove grant changes remap H and X or return an error, and hash small native/reference vectors.
- **Accept.** All occupied REs agree within the chosen precision, the receiver buffer is ready before use, and the runtime observation path cannot read H, W, original TB or labels. This gate can integrate the writer before linking the decoder.

### S2.4 Execute paired decodes through the real receiver · `replay_pusch`

- **Build.** Add a narrow native dispatch harness under `native/ocudu_adapter/receiver/`. Replace the controlled channel output at the `rx_grid` ingress of the pinned fixture. Pass PDU, same Y grid, original TB identity, and notifier/completion into both the CPU and CUDA processor. Record runtime backend/fallback, effective allocation, Y digest, true CRC and byte counts keyed to the grant. Preserve an untapped control.
- **Unit.** A fake notifier returning a valid CRC with one wrong byte must fail the comparison. Two swapped completions, a decoder error, a duplicate delivery and a silent timeout must resolve to explicit outcomes. Owned input must survive until completion.
- **Integrate.** Requires S1.2 and S2.1–.3. Run unity/noiseless first, then high-SNR direct path, then both target-pair members. Compare CPU and GPU outputs against the retained original bytes. Validate at least one deliberately bad channel or corrupted input against actual decoder observations. Keep CRC failures in the denominator. Never force a MAC ACK.
- **Accept.** Every admitted grid and terminal result has a unique key. In the high-SNR identity/direct cases both backends recover all original bytes with CRC pass. For the physical pair retain its observed outcome, hashes and numerical tolerance. Preserve enough input to rerun a failed grant exactly.

<a id="phase-s3"></a>

## S3 · Turn the same receiver ingress into a causal decision

### S3.1 Estimate single-port DMRS on a sparse grid · `pilot_ls`

- **Build.** In `native/ocudu_adapter/observer/`, export O2 from the exact O1 grid passed to PUSCH. For each allocated pilot compute `Y*conj(X_dmrs)/abs(X_dmrs)^2`. Use the executed slot's reference pilot sequence, amplitude, indices, port and phase convention. Include epoch/grant/sequence, RX map, valid mask and times in the immutable record. Identify this estimator as observer-derived. The device-resident O3 requires a separate S5 producer.
- **Unit.** For a pilot of `1+j` and noiseless `Y=-1+3j`, the LS result is `1+2j`. A halved pilot amplitude with correspondingly scaled Y must give the same estimate. Altering data-only REs leaves O2 invariant. A pilot hole produces an invalid mask bit. Reject a configured pilot with zero reference power. Test two antenna phases and a deliberately reversed map.
- **Integrate.** Match O2's ingress digest to S2.4 and recover known pilot H over all four RX on a direct high-SNR fixture. Compare with a separate sparse LS oracle. Pair tap-off/on recordings and compare true CRC and decoded bytes.
- **Accept.** Nonzero estimates at the declared pilots, correct sparse coordinates, source identity and no changes to deterministic decoded TBs. Emit evidence at the estimator precision actually used.

### S3.2 Publish observations with bounded ownership · `snapshot_ring`

- **Build.** In `native/ocudu_adapter/transport/` implement a preallocated local host-memory ring with per-slot generation and ownership. A try-acquire either yields one credit or immediately counts a drop. A producer completion publishes the immutable descriptor, the consumer retains it until its work completes, then exactly one release restores credit. A restarted consumer gets a new epoch. Reclaim abandoned records only once the old owner/copy cannot access them. Stamp queue depth, bytes and both completion points.
- **Unit.** Capacity two, no consumer: publish A,B, drop C, release A, admit D. Expect three admits, one drop and at most two retained items. B must survive producer scratch overwrite. Delay ready, double-release a slot, replay a stale generation and kill a consumer holding one credit. Verify no early read, no extra credit, bounded memory and recovery without waiting in the radio callback.
- **Integrate.** Run a slow consumer and a terminated consumer against S2.4+S3.1. Log matched tap-off/capture-on control decodes. Confirm that receiver opportunities continue across overload and observer restart.
- **Accept.** Every eligible observation ends as published, explicitly dropped or pending at the declared cutoff. At drain time, zero observations remain pending and all acquired credits are accounted for. Account for outstanding producer/consumer references before freeing memory.

### S3.3 Form causal features and verify a known return · `pilot_features`

- **Build.** In `src/ocudu_cuda_isac/sensing/features.py`, convert completed O2 records into a finite, masked causal history, a diagnostic delay–Doppler view and compact input features. Record acquisition first/last times, missing observations, interpolation choice and eligible windows. Use existing array/FFT operators. Publish an explicit insufficient-history outcome. Choose homogeneous sequences for the first FFT or use actual nonuniform sample times. Keep signed frequency/velocity conventions.
- **Unit.** On a complete lattice synthesize one return at an interior non-aliased delay/Doppler bin. The peak is within one declared bin, and conjugating the slow-time phase flips Doppler sign. Mask an RE and replace its stored value by a large sentinel: output on valid support is unchanged. Future samples, truth fields and earlier epochs cannot influence current features. Pre-filling seven of eight observations yields a warm-up state.
- **Integrate.** Consume S3.2 records, reconstruct complete windows offline from the same O2, compare feature values and eligible IDs. A known physical target must lie within one *resolvable* bin for the selected bandwidth and aperture. Preserve absent spectra and leakage/pilot-hole diagnostics.
- **Accept.** A baked feature vector, signed axes, resolution/alias limits and test tolerance let another developer distinguish a weak return from a transposed axis or a future-sample leak.

### S3.4 Train and load one presence head · `presence_head`

- **Build.** In `src/ocudu_cuda_isac/sensing/presence.py` and `scripts/train_presence.py`, implement a small PyTorch linear/logistic head on S3.3 features, a frozen normalizer, on-validation threshold, shape/dtype contract and a batch-one prediction API. Split by paired scene/trajectory before window creation. Load the weights once per process. Ensure the deployment artifact has only the trained parameters and preprocessing state it needs.
- **Unit.** Fixed two-feature weights `[1,-1]`, zero bias and features `[2,1]` yield logit 1 and probability approximately 0.731059. Save/reload gives the same score within 1e-6 on CPU. Reject wrong shape/feature ordering, altered checkpoint hash, cross-split scene and validation/test samples used to fit normalization.
- **Integrate.** Train using receiver-derived records from the training split, calibrate on validation negatives, freeze the threshold, then run batch-one inference in a separate CUDA process on held-out S3.2 observations. Join by observation ID and input digest. Record model/weights/preprocessor hashes and device timing.
- **Accept.** One actual model artifact can score the same held-out feature offline and on the selected device within a declared tolerance. Training and validation manifests and the freeze record precede test access. The demo's independent quality and timing criteria belong to S4.

<a id="phase-s4"></a>

## S4 · First integrated advisor demo

### S4.1 Pace two independent CUDA clients under MPS · `mps_pacing`

- **Build.** In `scripts/run_m1.py`, launch the receiver and sensor as separate processes with independent queues and one supported MPS service. Provide `dsp-only`, `capture-only` (perform the real snapshot/export, release it without inference), `ai-only` (buffered real O2), `concurrent` and separate fault-injection modes. Use absolute release times with an explicit processing allowance for a scripted receiver. Count overruns and skipped releases while preserving the offered schedule.
- **Unit.** Use a fake clock, process inventory and a schedule released every 5 ms. A worker that takes 12 ms cannot shift future release times or delete missed jobs. Reject duplicate client IDs or an unverified server/device pair. Stop/restart the fake observer without stopping the producer.
- **Integrate.** With S2/S3 and one GB10, record both MPS client PIDs, server/device mapping, active-thread setting and trace of concurrent service over a common interval. Freeze affinity, clocks, thermal state, package hashes, queue limits and cache warm-up. Keep the input realization identical across arms. For AI-only, replay the corresponding buffered O2. Keep capture-only as a separate copy-cost control.
- **Accept.** Both clients execute real work during the concurrent run. Isolated timing and arrival counts survive the comparison. Record whether load is scripted and whether the 5GC is idle or carries valid traffic.

### S4.2 Reconcile radio and sensing outcomes · `demo_scorecard`

- **Build.** In `src/ocudu_cuda_isac/evaluation/demo.py` join releases, completions, decisions and offline-only truth. Measure from first contributing sample to publication for sensing. Use scheduled release plus the configured allowance for each radio job. Distinguish acquisition duration from queue/copy/inference time and avoid summing overlapping GPU events. Report true CRC, payload bytes checked, goodput on its stated time base, p50/p95/p99 with sample counts, dropped/late/missing decisions and PHY misses.
- **Unit.** Test a deadline boundary and one ns past it. For five eligible target-present windows with two timely positives, one late positive, one negative and one drop, timely presence is 2/5. For four absent windows with one positive and one missing, report 1/4 false alerts and one missing. A duplicate key or negative latency fails. When four radio TBs are scheduled and three finish within budget, report 3/4 even if the fourth never finishes.
- **Integrate.** Recompute the same counts from raw M1 records with the observer off, capture-only and concurrent. Pair CPU/GPU or tap-off/on hashes on deterministic data. A valid CRC with incorrect bytes must flag a correctness failure. Keep standalone inference replay distinguishable from the concurrent acquisition.
- **Accept.** Every declared opportunity reconciles once, units and clock domains are explicit, zero eligible cases are reported as undefined, and an independent raw-data reduction agrees exactly on integer counts.

### S4.3 Run and replay the advisor showing · `advisor_demo`

- **Build.** Use `experiments/demo/profile.json` and add a terminal/replay report in `scripts/show_demo.py` that reads the frozen profile and events. Candidate settings are one UE, one layer, four RX, 20 MHz/51 PRBs at 30 kHz SCS. Form one decision from a history of eight eligible observations nominally 5 ms apart. This consumes about 35 ms from first to last observation start. Check the actual last symbol plus transfer/service against a proposed 50 ms sensing deadline. Freeze the actual schedule, radio allowance and any profile adjustment before acquiring results.
- **Unit.** A synthetic bundle with complete absent decisions on the thresholds passes with 8/10 timely present positives, 2/10 absent alerts and 99/100 radio jobs within allowance. Flipping one of the limiting outcomes makes its gate fail. Reject a missing control, unknown model or inconsistent run IDs. Replaying a finished bundle reproduces the scorecard without hardware.
- **Integrate.** Reuse exactly matched seeds/grants/weights across DSP-only, AI-only and concurrent runs. Collect at least three paired 60-second repetitions after warm-up. Alternate or randomize the order. Score one prospectively selected window per held-out episode, ten present and ten absent. Grade each concurrent repetition separately on the same 20 held-out episodes and show the per-episode outcomes. The three repetitions revisit the same scenes. Report the number of valid absent decisions. If missing absent decisions could change the false-alarm verdict, mark that gate inconclusive. Show one case of each live. Run overload/consumer-restart as a separate diagnostic.
- **Accept.** Require true native decode and zero payload mismatches on passing known TBs, real O2 to a real frozen sensor, concurrent MPS evidence, and all three predeclared count/latency gates in each concurrent repetition. Demonstrate the expected signed spectral response and bounded observer loss while radio jobs advance. Retain raw events, manifests, grades and one offline-replayable view. A failed numerical target is recorded as such. [Advisor walkthrough](../demo/ADVISOR_DEMO.md) · [experiment profile](../../experiments/demo/profile.json)

<a id="phase-s5"></a>

## S5 · Operational scheduled receiver

### S5.1 Bind live allocations to received grids · `du_grants`

- **Build.** In `native/ocudu_adapter/du/`, accept the native UL scheduling descriptor, convert the SFN/slot into an absolute opportunity within an epoch, and bind cell, UE, BWP/PRB allocation, grant, HARQ identity, TB and grid generation. Key the completed receive result to that identity. Preserve per-grant DMRS locations. Give missing/late allocations a counted terminal state. Keep the no-observer control.
- **Unit.** Feed the final slot of SFN 1023 then SFN 0 at μ=1. The absolute slot increases by one. Two simultaneous UE grants resolve to their distinct payloads. Inject an out-of-order completion, canceled allocation, reused generation and restart. Old completions never attach to a newer allocation. No stale read is dispatched.
- **Integrate.** After S2/S4, execute alternating legal PRB allocations on the DU, with at least one high-SNR known payload and a separately identified CRC failure fixture. Compare admitted grants, observed masks, original TX bytes and true CRC with the tap disabled/enabled. Confirm from scheduler events that this is the DU path. Record any HARQ or host fallback separately.
- **Accept.** `grants.jsonl` and `radio.jsonl` account for every offered grant exactly once. Known successful payloads match bit-for-bit across the two arms. Reused/wrapped IDs produce no join collisions.

### S5.2 Export the executed estimator · `native_estimate`

- **Build.** Add the smallest versioned patch in `patches/ocudu/` and adapter code in `native/ocudu_adapter/observer/`. At the pinned `pusch_processor_impl.cpp`, distinguish the host estimator from the resident path's zero-valued stub. Trace the GPU producer through `pusch_demodulator_gpu_impl.cpp` / `lib/phy/cuda/src/pusch_e2e_api.cu`. Capture an **actual** produced estimate after its write and before its workspace can be reused. Log producing symbol, backend, interpolation/scale, availability event, legal support and generation. Preserve the S3 O2/LS fallback as a separately tagged observation.
- **Unit.** Use a fake asynchronous producer with two different complex buffers. Attempt capture before its completion, overwrite the source immediately after permitted release, and delay the consumer. Only the completed generation can publish. For a known flat-channel reference of `1+0.5j`, a zero-filled buffer fails the nonzero-reference test. A true absent/zero input remains a valid zero when the reference expects zero.
- **Integrate.** On the native receiver, run flat/noiseless then known frequency/phase-varying, then seeded noisy channels. Compare to the matched estimator configuration on common physical pilot support. Predeclare dtype-aware absolute/relative tolerances and an NMSE tolerance for the noisy case in the config using the CPU/model precision study on calibration cases. Show a hook-disabled decode control and provenance for every supported branch, and record unsupported branches explicitly.
- **Accept.** Retain the minimal patch hash, producer/completion trace, `o3_parity.json` with numerator/denominator/sample counts, and paired CRC/bytes. A host stub, reconstructed O2 or an unverified filter output cannot close an O3 comparison. See the [native source audit](../integration/OCUDU.md).

### S5.3 Sustain the paced DU and recover from observer loss · `du_soak`

- **Build.** Extend the S5.1 DU path with an opportunity ledger and the S3.2 fixed-capacity observer pool. Pace releases against the declared wall clock, including overdue jobs. Reconcile offered work even across consumer cancellation, telemetry failure and restart. Export pool high-water marks, allocator calls after warm-up, queue depth, fallback selections, throttling and stale/drop counters.
- **Unit.** With two observer credits and no releases, issue three grants. The first two publish, the third drops observer work and all three radio jobs dispatch. Kill the consumer, begin a new epoch, inject an old completion and release a new item. All credits reconcile at shutdown. Only a current-generation completion can release the new slot.
- **Integrate.** After S5.1/.2, run at least 30 minutes per matched supported configuration with observation disabled, capture only and capture plus inference. Replay the same offered trace and include a controlled consumer stall. Use actual CRCs, including failures. Label forced ACK/load-only runs separately.
- **Accept.** `soak.json` shows reconciled radio and observation counts, observed deadline misses, queue bound, stable preallocated pool size after warm-up, recovery and continued scheduling through the stall. Baseline and observed arms agree on the deterministic known-decode fixture. Archive the overload result even when it exceeds the declared service budget.

<a id="phase-s6"></a>

## S6 · Receiver and sensing experiments

### S6.1 Generate paired research episodes · `paired_episodes`

- **Build.** Extend the S2 exporter with parameterized scene, target trajectory, materials, array manifest, real allocation/symbol times and held-out session/allocation groups. Use the pinned `PathSolver`, `RCSSolver`, `Paths.concat` and `Paths.cfr`. Physically remove the target for absent scenes and recompute the background. Record target-free, weak/strong, moving/stationary interval, clutter change and sparse-pilot cases. Save immutable H plus matching X, W and grants separately. Generate serializable observation inputs and evaluator-only truth with opaque keys.
- **Unit.** Replace the solver with a known CFR proportional to `exp(-j2πfτ + j2πνt)`. For `τ=1 μs`, a `+250 kHz` increment rotates by `−π/2`. For `ν=50 Hz`, a `+5 ms` increment rotates by `+π/2`. Applying noise once gives `Y=H·X+W`. Changing X alters Y and preserves H. Reject out-of-grid interpolation, nonfinite samples, swapped RX axes, mismatched hashes and held-out groups reused in training.
- **Integrate.** After S2, export at least one matched background/present pair at actual occupied frequencies and unequal symbol/CP times. Validate a predeclared interpolation tolerance against a denser temporal reference and test a trajectory crossing a geometry-update boundary. Preserve absolute time/phase and record refresh cadence. Check direct-path survival, finite amplitude and scattering in the appropriate scene. Retain both the full H replay and frozen Y hashes. Optional online geometry updates are a distinct run with service time and staleness evidence.
- **Accept.** `episode_index.json` plus the paired binary arrays identify calibration, axes, world frame, time support, versions, dtypes, solver flags and checksum of every input. Paired comparisons reuse the same legal grant and noise realization. The four-time, twelve-tone example validates layout only. Refer to the [RT exchange contract](../interfaces/RT_EXPORT.md).

### S6.2 Build a mask-aware spectral detector · `spectral_detector`

- **Build.** In `src/ocudu_cuda_isac/sensing/spectral.py`, consume actual frequency offsets, timestamps, complex pilots, validity mask and background calibration. Produce power over a declared delay–Doppler grid plus detection score and valid-support flag. Reuse Sionna's standard transforms on compatible regular grids. For irregular sampling implement the observation mapping/weighted sum with existing tensor operators. Normalize by valid weights and document window coherent gain. Compare fixed background subtraction, two-pulse MTI, causal whitening and CA/OS-CFAR. Record CFAR guard/training cells, edge handling, rank and post-search validation threshold.
- **Unit.** On 16 fully observed tones (`Δf=30 kHz`) and 16 equally spaced slow-time instants (`Δt=1 ms`), synthesize the tone with delay bin +2 and Doppler bin −3 using the sign convention in S6.1. A rectangular-window reference peaks at (+2,−3). On irregularly masked inputs compare the spectrum with a direct complex sum within a frozen dtype tolerance. Filling an invalid RE with any large sentinel leaves it unchanged. Two identical pulses cancel in MTI. Constant unit reference powers give a CA reference mean and OS reference statistic of one. Missing CFAR support returns an explicit invalid flag.
- **Integrate.** Replay the paired analytic and RT cases. Freeze threshold(s) after the **whole** preprocessing, search and detection pipeline on common validation negatives. Compare test false alarms, detection and physical errors under pilot gaps, changed clutter and stop/start motion. Track false alarms per eligible search opportunity, including absent opportunities with missing decisions.
- **Accept.** `detector_calibration.json` records the search region, validation units/counts and operating point. `spectral_scores.jsonl` links each estimate to actual support. Report delay as bistatic path time or excess path length relative to the declared reference, and Doppler in signed Hz. Convert to distance/velocity only with specified geometry.

### S6.3 Implement calibrated Bartlett and MUSIC · `angle_estimators`

- **Build.** In `src/ocudu_cuda_isac/sensing/angle.py`, accept array positions in metres, carrier, phase/amplitude calibration, axis convention and snapshots for a detected delay/Doppler cell. Use pinned Sionna `steering_vectors` and `angular_delay_doppler_spectrum` when compatible. Build Hermitian sample covariance with a documented snapshot axis. Use library `eigh`, an explicit loading/shrinkage rule, a declared estimated model-order rule (e.g. MDL within its valid snapshot regime) and calibrated steering search. Record the chosen mode count, search sector and eligibility. A fixed order is an explicitly labeled test/control.
- **Unit.** Use four half-wavelength-spaced elements, a known broadside steering vector, repeated complex amplitudes and fixed single-mode order. Both spectra peak at broadside in the declared visible sector. Repeat with known ±30° plane waves and the analytic steering formula to catch conjugation or array-order reversal. Test a scalar multiple of the identity, deficient snapshots, zero signal, invalid loading and mode count ≥ antenna count. Return abstention/diagnostic status for unidentifiable cases. A ULA mirror pair scores by its declared ambiguity class.
- **Integrate.** After S6.1/.2, use the same frozen detections, calibrated array, bins and snapshot budget for Bartlett and MUSIC. Verify estimated order on high-SNR calibration cases under the actual sample count. Include insufficient-support cases and multipath. Compare against independent scene geometry in the evaluator.
- **Accept.** `angles.jsonl` retains eligibility, estimator, model order, peak/ambiguity class and conditional error for every candidate. The timely metric still counts required-angle abstentions or missed detections as failures.

### S6.4 Freeze causal windows and split definitions · `causal_windows`

- **Build.** In `src/ocudu_cuda_isac/sensing/windows.py`, group by scene/trajectory pair or capture session and split before windowing. Define a separate unseen-allocation evaluation split or an explicit cross-domain design. Keep matched present/absent pairs together. Derive chronological histories from O1/O2/O3 using actual pilot positions, gaps, ages and causal receiver metadata. O0 truth and future CRCs remain in the offline data/evaluator. Fit scalers/backgrounds on training only and apply frozen thresholds selected on validation.
- **Unit.** Construct six short sessions with overlap and paired IDs. No group or raw contributing sample crosses folds. Shuffling manifest order preserves membership and hashes. Changing future values or truth fields leaves current feature bytes unchanged. A missing pilot has a false validity bit. A valid physical zero has a true bit. Reset state on new epoch/session and reject normalization that declares a validation/test sample as a fit input.
- **Integrate.** After S2/S3 and S6.1, join clean-channel oracle views and sampled/noisy O1, observer O2 and verified O3 by identical grant/episode/time. Assert H/X/W hashes for paired rows and align common pilot support. Record exclusions when a stage cannot be produced. Write runtime features without truth-bearing paths, filenames, model order labels or scene metadata.
- **Accept.** `splits.json`, `fit_manifest.json`, `window_index.jsonl` and the runtime feature schema allow the same samples to be reconstructed. Overlap/leakage tests pass and the stage coverage table has an explicit denominator.

### S6.5 Compare causal learned encoders · `causal_encoders`

- **Build.** In `src/ocudu_cuda_isac/sensing/models/`, reuse PyTorch Conv/GRU/Transformer layers with common finite history, task heads and output units (presence logit, delay, signed Doppler, angle or angle encoding). Apply physical-parameter losses only for eligible present samples. Limit GRU state to the same history as the CNN and causal masked Transformer. Include a cuSense-inspired CNN with documented changes for the new task. The source v3 localization head and its reported latency are a reference for this adaptation. [Villa et al., sections 6.1–6.2](https://arxiv.org/abs/2512.06493v3)
- **Unit.** For each architecture on CPU, verify output shapes/units, finite gradients and identical current predictions under future-only/label changes with fixed weights and inference mode. A history run after an episode reset equals a fresh instance on that history. Check masked invalid values have zero influence. Save/reload one fixed checkpoint and compare outputs at the declared precision.
- **Integrate.** After S6.4, train on identical group splits and record the same tuning/training resources, seeds, early-stop rules, feature access, precision and per-method size sweeps. Calibrate each method on the same validation negatives. Run frozen checkpoints at batch one on the target endpoint, and record weights hash, parameters, peak allocator memory, recurrent/cache state, latency and energy including preprocessing/transfer. Differences in parameter count do not establish execution cost.
- **Accept.** `model_registry.json` names each architecture, representation, training domain, budget, checkpoint/config/threshold hash and inference path. A reproducible tiny CPU training fixture verifies the optimizer updates weights. Population accuracy and device timing require the gated trials.

### S6.6 Isolate angle-head structure · `covariance_heads`

- **Build.** In `src/ocudu_cuda_isac/sensing/models/angle_heads.py` use a common head interface for direct angular prediction, raw sample covariance, classical shrinkage, and learned covariance with metadata on/off. Parameterize a learned covariance as a Hermitian PSD matrix, e.g. `LLᴴ+εI`. Reuse the S6.3 eigensolver, search, calibration and model-order rule across covariance heads. Freeze upstream detection and delay/Doppler outputs during this comparison.
- **Unit.** A fixed complex factor yields a Hermitian matrix, eigenvalues at least `−tolerance`, and the same search result before and after serialization. A shrinkage coefficient of zero yields the raw covariance, one yields the declared isotropic target. Invalid dimensions, non-Hermitian inputs to the raw backend or a failed detector gate yield explicit rejects/abstentions. Assert pointwise identical frozen detection scores and bin IDs across angle heads.
- **Integrate.** Pair every head on identical snapshots and accepted candidate IDs. Retain missed candidates in the full endpoint. Compare conditional physical error and timely success versus measured cost, including learned correction, eigendecomposition, search and transfer.
- **Accept.** `angle_ablation.jsonl` contains common input IDs and hashes, order, head, metadata flag, eligibility, result and measured service. A changed detector or order-selection policy creates a different comparison.

### S6.7 Test paired receiver-domain hypotheses · `domain_matrix`

- **Build.** In `experiments/conference/` define configuration rows for H1 (receiver stage at fixed sampling/noise), H2 (ideal, observation-matched and receiver-executed training tested on a common executed stage) and H3 (mask/timing/available receiver metadata on/off under held-out allocations and matched RAN load). Reserve clean O0 for oracle training/reference rows and identify transformations making a simulator observation-matched proxy. Evaluate O1/O2 and measured O3 explicitly. Reuse S6.2–.6 without changing a second factor within a comparison.
- **Unit.** A tiny manifest with one changed noise seed or allocation in a purported pair fails validation. Distinct scene pairs with the same numeric slot never merge. Reject O0-derived fields in deployed predictors, a threshold hash chosen on test cases, or an O3 row missing its producing symbol/backend. A three-row toy comparison with two paired rows reports coverage 2/3.
- **Integrate.** Freeze S9.1 protocol first. Use the same H/X/W, held-out windows, cadence, physical tolerances, validation false-alarm target and comparable training budgets across each paired family. Keep clean O0 comparison offline. Collect effect estimates and paired episode/session-clustered intervals, plus per-stage sample counts, false alarms and failures. Archive the complete or explicitly incomplete matrix.
- **Accept.** `domain_matrix.json` maps each hypothesis to paired run IDs and factor differences. Report uncertainty and null/adverse results. A scientific claim about O3 requires S5.2 native evidence. An O1/O2 experiment can still run and be reported on those stages.

### S6.8 Evaluate measured channel-estimate data · `measured_capture` (extension)

- **Build.** In `src/ocudu_cuda_isac/episodes/measured.py`, write a read-only adapter for the pinned public PUSCH H-estimates, retaining original timestamps, DMRS cadence, array order and camera-derived label validity. Allow separate independent lab OTA captures with documented alignment/label provenance. Publish a capability table per capture describing which detection/position labels, calibrated angles, raw IQ, TX payloads and timing boundaries exist.
- **Unit.** A tiny fabricated two-session capture has one valid label, one missing label and one timestamp gap. Emit the eligible count of one, a gap mask and explicit unknown-state metadata. An unlabeled interval cannot become a negative example. Preserve session split membership and source coordinates/units through conversion.
- **Integrate.** Replay at least one disjoint held-out run where available, preserving session-level calibration and disclosure of target-free calibration runs. Score only label-supported tasks and separate capture-domain results from simulated results. Obtain independently labeled raw inputs and a matching native run before adding receiver timing or BLER claims.
- **Accept.** `capture_manifest.json` records source/version/license/hash, domain, unit map, eligible counts, external task and limitations. Access failures leave this extension open while primary simulator/receiver experiments remain independently selectable. [Dataset and reference implementation ledger](../integration/UPSTREAM.md)

<a id="phase-s7"></a>

## S7 · Measure the two-Spark observation link

### S7.1 Frame the socket stream and bound credits · `socket_bridge`

- **Build.** In `native/ocudu_adapter/transport/`, add a length-delimited, versioned framing layer with explicit endian, dtype, shape, layout, epoch/sequence, source stage/time, payload length and checksum. Bound queues, frame size and credit count. Release a source buffer after its last read and a destination buffer after its consumer completes. Peer reconnect starts a fresh epoch.
- **Unit.** Split one header and its payload into one-byte fragments over a loopback/fake stream. Exactly one full valid record emits. Truncated and corrupt records produce counters and no inference input. Exercise mixed host endianness, malformed dimensions and maximum permitted size before allocating. With zero credit radio submission continues. Duplicate/stale completion returns no credit.
- **Integrate.** After the S3 observation contract, run the same immutable binary payload through local M1 transfer and the two-host socket route. Capture one sender and receiver record per admitted ID and verify exact byte hashes. Record interface/physical port, DAC, actual negotiated rate, MTU, send/receive buffering, warmed application payload throughput, offered cadence and wire-byte overhead.
- **Accept.** `socket_transport.jsonl` plus source/destination counters distinguish offered, admitted, completed, dropped and GPU-readable updates. Four RX × 14 symbols × 3,276 tones × 8 bytes yields **1,467,648 bytes** before framing. At 2,000 updates/s this is **23.482368 Gb/s**. This is a unit sizing vector. Measured configured pilot payloads may be smaller.

### S7.2 Stage the verbs path and establish timing · `verbs_clock`

- **Build.** Extend `native/ocudu_adapter/transport/` with registered pinned host buffers (`cudaHostAlloc` + `ibv_reg_mr`), bounded posted receives/credits, completion queues and explicit GPU copy-complete events. Keep socket and verbs on the identical record schema. Map PTP/PHC samples to the host trace clocks, estimate offset and residual/drift across intervals, and retain synchronization diagnostics. Separate sender-ready, post, full-receive completion, GPU-readable and publication times. [Spark CUDA porting guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/latest/porting/cuda.html)
- **Unit.** Fake the verbs completion queue and delay a posted send or GPU copy. Payload contents and buffer credit remain unchanged before the recorded release. Inject an old epoch and out-of-order completion. A known clock offset of +100 μs on host B makes a 150 μs cross-clock delta equal 50 μs after conversion. An uncertainty bound of ±80 μs makes the latency ranking inconclusive.
- **Integrate.** On the CX7 pair, replay a transport-only load and a joint S5 receiver/S3 sensor load over sockets and host-buffer verbs at matched bytes, rate, pinned memory and CPU allocation. Preserve copy/queue/wire times and calibration residual over the run. If synchronization fails, produce same-host durations and a result-return RTT using a single clock. Bind conclusions to the recorded timestamp boundaries.
- **Accept.** `verbs_transport.jsonl` and `clock_map.json` contain byte parity, credit accounting, backend, registration lifetime, achieved payload, clock map and uncertainty or explicit RTT-only status. An integration can establish a functioning bridge even when one-way latency remains unresolved.

### S7.3 Run two-host deployment and core controls · `two_spark`

- **Build.** Add `scripts/run_m2.py` and M2 launch profiles in `experiments/conference/`. Place CU/DU, replay and DSP on Spark 2. Place sensor and CPU 5GC on Spark 1. F1 and virtual RU remain local to Spark 2. Isolate N2, N3 and sensing flow counters, and make their application/port identities explicit. Include a control with the core held fixed while AI moves.
- **Unit.** A synthetic three-flow capture counts N2 setup, N3 payload and observer bytes separately. Refuse a UPF-goodput row lacking verified payload delivery. From two energy readings of 100 J and 130 J for separate hosts over a common interval, report system energy 230 J. Retain the energy scope of the interconnect.
- **Integrate.** After S5/S7.1–.2, run M2 integrated with the same episode, weights, RX representation, offered traffic and AI cadence as M1. Run idle-core/active-core cases separately. An active case requires a protocol-capable UE, registration, session setup, UPF and end-host receipts plus traffic counters. Record wall or system power on both nodes over the same interval, temperature, clock state and losses. Test radio-only, sensor-only and transport-only arms for each interpretation.
- **Accept.** `m2_comparison.json` archives paired traces, both nodes' energy boundaries, copy/transport/compute distributions, clock gate and joint endpoint. Record native negotiated 100 Gb/s, shaped faster link or different actual speed as distinct configurations. An unavailable protocol endpoint leaves core-specific evidence pending.

<a id="phase-s8"></a>

## S8 · Measure GB300 isolation

### S8.1 Bind workloads to distinct GPU instances · `mig_bridge`

- **Build.** In `native/ocudu_adapter/transport/` and `scripts/run_m3.py`, consume a read-only inventory of GPU UUID, GI/CI IDs, MIG UUID, supported profile and placement. Require separate GIs on the same GB300 for DSP and AI. Start from the operator-provisioned layout. Record actual profiles and capacity. A candidate is 4g.126gb for DSP plus 3g.126gb for AI when supported locally. Reuse the owned pinned-host transfer interface. Chain producer completion → D2H → ready → H2D → consumer release. [NVIDIA MIG deployment guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/deployment-considerations.html)
- **Unit.** Feed saved inventories with two CIs on the same GI, reversed device roles, wrong parent GPU, duplicate UUID and insufficient profile memory. Each rejects before worker launch. A valid dual-GI fixture launches exactly two workers with disjoint bindings. A late GPU copy prevents buffer reuse. Host-staged payload bytes survive consumer lag and restart.
- **Integrate.** On the configured station, capture `nvidia-smi -L`, supported/active GI/CI and placement records, driver/toolkit and each worker's own device identity. Send a fixed pilot payload across the bridge, verify device-side content and generation, then run a slower consumer until credits exhaust. Preserve byte/latency/drop counters. Keep any optional peer/broker path a separately verified transfer experiment.
- **Accept.** `mig_inventory.json` and `mig_bridge.jsonl` prove the two distinct GIs and real D2H/H2D completion path, plus successful known-TB decoding under pressure. A numerical device index alone cannot establish process placement. CUDA IPC across GIs is unsupported in the cited deployment guide.

### S8.2 Compare isolation, capacity and shared-host effects · `gb300_controls`

- **Build.** Add within-GB300 configurations in `experiments/conference/` for full-GPU MPS/time sharing as supported, DSP alone on its assigned GI, AI alone on its assigned GI, and both GIs together. Hold replay, features, real-time cadence, weights, dtype, affinity, queue capacity, power boundary and declared core state fixed. Export thermal throttling, memory/host/NIC use and copy counters. Save any different GI layout as a distinct profile.
- **Unit.** The paired-arm checker rejects swapped workload hashes, a changed baseline payload, an isolated run on a different-sized GI, and use of single-Spark measurements as the same-platform control. A service-time fixture of 4 ms isolated and 6 ms concurrent yields a 1.5× slowdown at the stated measurement boundary.
- **Integrate.** After S5/S8.1, run isolated baselines at their actual capacities, same-host concurrent MPS controls on the full device, and both GIs together at rising load. Repeat with the same local transfer representation and record the realized path to separate copy cost from GI capacity. Measure host/thermal/power interference with fixed partitions. Use observed p95/p99 plus sample counts, per-profile slowdowns and throughput/energy at each joint SLO.
- **Accept.** `m3_comparison.json` contains isolated and concurrent pairs on the station, platform differences, observed feasible and failing cells, and the raw identifiers needed for a rerun. Report a supported tested load/grid point and uncertainty. Unmeasured capacity or a hypothetical GI combination stays out of the frontier.

<a id="phase-s9"></a>

## S9 · Freeze, reduce and reproduce

### S9.1 Freeze the study before held-out execution · `study_protocol`

- **Build.** Create a validated, versioned protocol under `experiments/conference/`. Include hypotheses, exact eligible opportunity (slot/window/episode), target-free fraction, common validation false-alarm target, hardware mix, train/validation/test split hashes, group keys, model/config hashes, training/tuning budgets, precision, seeds, excluded startup, power sampling, repetitions and randomized trial order. Predeclare numeric sensing deadline from first sample to publication, error tolerances/units, radio release/deadline convention, maximum radio miss probability, acceptable BLER/goodput degradation, minimum sensing success and decision rule using confidence bounds. For comparisons with external labels, declare available truth and evaluable endpoints. Use a pilot restricted to calibration data when budgeting tolerances.
- **Unit.** Reject deadlines shorter than the chosen minimum acquisition duration for a proposed feasible positive case. Allow explicitly labeled infeasibility controls. Reject zero/negative tolerances where inappropriate, incompatible distance/angle conventions, unbound plans, mixed units, duplicate IDs or a held-out sample in a threshold-calibration set. A changed threshold or split changes the protocol hash.
- **Integrate.** Before S5–S8 held-out runs, extend the M1 launcher with an ordinary time-sharing control using the same two clients and record whether each is attached to MPS. Instantiate cells for M1 ordinary time sharing/MPS, M2 transport and core controls, M3 isolation, plus radio-only, sensor-only, transport/capture-only and joint controls. Freeze an achievable starting load and sweep radio utilization, cells/bandwidth, AI model/rate/bursts, bytes/update and precision separately. At comparison points hold other factors fixed. Specify an ordered load grid or multidimensional partial order. Separate a larger-antenna or compressed payload experiment from topology comparison.
- **Accept.** `protocol.json`, `planned_cells.jsonl` and split manifests are archived by hash before viewing held-out results. Each row names its platform, treatment, baseline, inputs and thresholds. Unsupported hardware is a planned cell with a recorded status. A revised protocol earns a new version.

### S9.2 Reduce events to service and quality outcomes · `study_metrics`

- **Build.** In `src/ocudu_cuda_isac/evaluation/`, left-join observations/results to every scheduled radio job and eligible sensing opportunity. Save first/last samples, radio release, ready, copy/transport, queue, inference, publication and actual RX completion times with clock domains and uncertainty. Report acquisition, post-window time and end-to-end observation age. Use device events for individual services and wall-clock boundaries for the critical path. Classify missing, invalid, late, drop, abstain, wrong estimate, CRC failure and byte mismatch. Distinguish BLER per attempted TB, offered/unfinished counts, unique correct payload bits per wall second, and application goodput when measured through the UPF. Declare HARQ deduplication.
- **Unit.** Six eligible present windows yield one timely/accurate positive, one late positive, one drop, one wrong angle, one abstention and one missing result. Timely success is 1/6. Four eligible absent windows (one alert, two valid negatives, one missing) give 1/4 alerts per eligible absent opportunity, 1/3 among observed decisions, and 1/4 missing coverage. A job with release 10 ms and allowance 2 ms completing at 12 ms meets an inclusive deadline. An unfinished scheduled job misses. In a 1 s trace, two attempts carrying the same known 100-byte TB (first CRC fail, retry success) yield 50% attempt BLER and 800 bit/s unique receiver goodput. A 50 W, 2 s constant-power trace gives 100 J. With two timely successes this is 50 J/success. A matched 80 J baseline gives 10 incremental J/success. With zero successes, the ratio is undefined.
- **Integrate.** Recompute from real raw inputs for each frozen cell. Preserve completed-latency quantiles with sample counts and full eligible denominators. Report overall false alarms, detection, conditional errors, energy total and incremental over equal-duration RAN/core baseline, per update and per timely correct estimate. Retain signed incremental estimates and meter/cadence limits. Use paired cluster-level resampling over independent episodes/sessions/trials with frozen seed, confidence level and repetition count. Suppress unsupported extreme tails and mark overlapping latency comparisons inconclusive when uncertainty dominates.
- **Accept.** `outcomes.jsonl` and `metrics.json` reconcile all opportunities without duplicate keys. Zero denominators remain undefined. Feasibility follows all frozen comms **and** sensing criteria, including declared statistical rule. Report greatest feasible **tested** load on a declared scalar sweep or nondominated feasible cells on multiple axes. Retain failing and inconclusive cells.

### S9.3 Independently rebuild each claim · `release_evidence`

- **Build.** Add a release/evidence manifest, raw-to-metric-to-figure recipes and a `claim_map` connecting each numerical paper statement to run, protocol, artifact hashes and code entry points. Keep generator/version, dependency licenses, data access instructions, serializable model weights, eligibility and stage provenance with the release. Preserve limits for simulated participants, offline captures, operational DU and any full-core trials. Version fixes or reruns without overwriting prior evidence.
- **Unit.** A miniature release with one figure and two source runs verifies end-to-end. Changing one raw byte, removing a claimed result, swapping unit labels or mapping a hardware outcome to synthetic timestamps fails. Deterministic numeric tables regenerate identically. Declare tolerances for stochastic/hardware outputs and deal with nonsemantic plot metadata separately.
- **Integrate.** An independent developer sets up a clean `isac` environment, runs one complete raw-to-figure example per major comparison (stage/domain, architecture, deployment), then repeats a preselected representative hardware subset under the frozen protocol. Compare with predeclared tolerance/uncertainty and preserve observed discrepancies. Record actual success, failed reproduction or blocked prerequisites by claim.
- **Accept.** `release_manifest.json`, `claim_map.json`, regenerated tables/figures and `independent_rerun.json` identify reviewer, exact commands, environment, denominators and artifacts. An archive checksum verifies identity. Independent regeneration verifies the computation. Repeating a real experiment verifies reproducibility within the declared bounds. See the [measurement protocol](../research/EVALUATION.md), [conference matrix](../../experiments/conference/MATRIX.md) and [reproduction checklist](REPRODUCE.md).

## Trace earlier acceptance gates to their new tickets

| Original verification requirement | Owning sprint(s) |
|---|---|
| U0.1 complex axes, U0.2 joins, I0.1 host smoke, I0.2 source provenance | S0.1–.3 |
| U1.1 bad transcripts, I1.1 actual CPU/CUDA agreement, I1.2 measured timers | S1.1–.2 |
| U2.1 complex parity, U2.2 physical delay/Doppler, U2.3 invalid mapping | S2.1–.3 and S3.3 |
| I2.1 original bytes in direct-path case, I2.2 paired physical replay | S2.2 and S2.4 |
| U3.1 valid pilots, U3.2 snapshot lifetime, U3.3 causality, U3.4 frozen weights | S3.1–.4 |
| I3.1 exact ingress–observer join, I3.2 consumer loss and restart | S3.1, S3.2 and S4.2 |
| I4.1 actual MPS/decoder/decision, I4.2 held-out detection | S4.1 and S4.3 |
| I4.3 service and denominators, I4.4 live/replayed demonstration | S4.2 and S4.3 |
| U5.1 · Processed estimate | S5.2 |
| U5.2 · Ownership, wrap and restart | S5.1 and S5.3 |
| I5.1 · Scheduler and backend fallback | S5.1 and S5.2 |
| I5.2 · Sustained deployment comparison | S5.3 |
| U6.1 · Physical reference | S6.1–S6.3 |
| U6.2 · Split independence | S6.4 |
| U6.3 · Causality | S6.4 and S6.5 |
| I6.1 · Paired receiver stages and training domains | S6.5–S6.7 |
| I6.2 · External data | S6.8 when applicable |
| U7.1–U7.2 · Framing, loss and credits | S7.1 and S7.2 |
| I7.1–I7.3 · Transport, clock and core | S7.1–S7.3 |
| U8.1–U8.2 · Assignment and lifetime | S8.1 |
| I8.1–I8.2 · Isolation and interference | S8.2 |
| U9.1–U9.2 · Denominators, domains and units | S9.1 and S9.2 |
| I9.1–I9.2 · Regeneration and independent repeat | S9.3 |

## Where to start when a gate fails

| Observation | First discriminating check |
|---|---|
| CUDA was configured but CPU and GPU appear identical in the log | Check the build cache, CUDA targets, binary identity, actual residency counters and completed decodes on the measured row. |
| CRC passes but the captured estimate is zero | Inspect the active producer. Validate O2 against known pilots, then inspect O3's device producer, event ordering, precision and lifetime. |
| Presence fires on absent scenes | Inspect the physically absent background, held-out split, training-only normalizer, invalid-mask handling and runtime feature allowlist. |
| Radio tails grow under concurrent inference | Hold offered releases fixed. Compare tap-off, copy-only and full sensing. Use completion traces to locate the delay. |
| Apparent network or MIG gain changes with deployment | Compare timestamp uncertainty, matched payload and transport-only controls on M2. Compare per-profile isolated and full-GPU runs on the same station for M3. |

For each failed gate retain the fixture digest, command, failing expected/observed value, raw log and smallest subsequent rerun. Update the ticket with its separate unit and integration status.
