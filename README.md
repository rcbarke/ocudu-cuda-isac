# Lightspeed: An Inline CUDA PHY Ladder for Receiver-Conditioned ISAC from Ray-Traced Channels to Over-the-Air 5G

**Index terms:** OCUDU, Sionna RT, DeepSig, Aerial. **Repository:** `ocudu-cuda-isac`.

Receiver-aware integrated sensing and communication on CUDA-accelerated OCUDU. The first lab result is a reproducible CPU/GPU PUSCH decode on DGX Spark. The first project demonstration adds a second MPS client that runs a cuSense-like pipeline on that receiver's own channel estimate, with a live KPI page. The publication study compares receiver stages, PHY channels, radar and AI estimators, functional splits, memory paths and three hardware deployments under matched communication load. The same sensing core is then run on Aerial Framework and, over the air, on Aerial-OCUDU.

**Start with [docs/dev/DEVELOPMENT.md](docs/dev/DEVELOPMENT.md).** It contains the phase order, every sprint's component contract, isolated test, integration prerequisites and acceptance evidence.

## CURRENT STATUS

As of 2026-11-22 (UTC), **Sprint 0 (install and verify), Sprint 1 (native Release build and the DeepSig Table II row) and Sprint 2 (a known uplink waveform through a traced channel and the native receiver, decoded by the host and the CUDA PUSCH processor to the original transport block) are accepted on one DGX Spark.** **Sprint 3.1 (the first observer: pilot least squares, the stage O2, exported from the grid that both PUSCH processors decode, with every decode unchanged) is accepted on the same Spark.** **Sprint 3.2 (the observation ring between the receiver and a sensor process, and the timed path without a checksum) is accepted on the same Spark: the tap takes 11.0 us in the mean, against 1,472 us with every digest.** **Sprint 3.3 (the radar cube, the delay-Doppler view and the causal features on the records of the ring, with the known return of the reference pair) is accepted on the same Spark: the return lies at delay bin 3 and Doppler bin -2, within one resolvable bin of its geometry and 22.3 dB above the median of its view, and the features agree bit for bit from the recorded ring and from a live ring.** **Sprint 3.4 (one linear presence head on those features, trained on receiver-derived records of the demonstration set, frozen as an artifact of two files and scored in a second process) is accepted on the same Spark: 32,640 decodes of 340 traced pairs gave 680 windows, the frozen head scores the 20 held-out windows on the GB10 and offline within 9.5e-07 of a logit, and every target-present window of the set scores above every target-absent one.** **Sprint 5.2 (the receiver's own CUDA channel estimate, the stage O3, exported through one receiver patch of 137 lines and no kernel) is accepted on the same Spark: 448 decodes of the patched receiver equal those of the unpatched build, and 128 exported estimates agree with the matched estimator on the stage O2 at an NMSE of 2.2e-09.** **Sprint 3.5 (the cuSense-like branch on that estimate: a background template and a z-score fitted on the training trajectories, a location map with a presence logit, one frozen artifact of four files per input entry) is implemented and passes its unit gate and its driver on doubles. Its hardware gate, `make accept-s3.5`, has not run on the Spark yet.** The next ticket after it is S3.6. S3.6 and S4 to S11 are specified and not yet implemented. One requirement was added on 2026-11-21: no checksum runs in a timed path. The plan was audited on 2026-11-20 against the evaluation plan of that date, and again for the code level of every ticket: sensing that is timed runs in a native core of CUDA kernels and TensorRT engines, and the demonstration shows that latency. [Build order](docs/dev/DEVELOPMENT.md#build-order) · [requirements table](docs/dev/DEVELOPMENT.md#requirements-and-where-they-are-built) · [code level of every ticket](docs/dev/DEVELOPMENT.md#code-level-of-every-ticket)

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

### Sprint 2.1 acceptance

`make accept-s2.1` ran every Sprint 2.1 gate on the same Spark at commit `e092623` and ended with `Sprint 2.1: ACCEPTED on this DGX Spark.` The run ID is `s21-spark-ecf8-20261120T223232Z`. All 48 gate lines passed. The gate used the build that Sprint 1.1 accepted, with nothing of OCUDU rebuilt, and finished on the CPU in under half a minute.

Sprint 2.1 exports one legal uplink grant and its transmit reference. `src/ocudu_cuda_isac/episodes/grant.py` validates a `pusch-grant/v1` record, derives the transport block size, the resource elements in mapping order and the DM-RS reference symbols, and keeps them with the transmit grid in a checksummed store entry. The grant's DM-RS is registered as the first sensing source, `pusch_dmrs`. A thin native adapter builds the transmit grid with the transmit chain of the pinned `pusch_e2e_pipeline_test`, out of tree and against the build that Sprint 1.1 accepted. [Record, coordinates, store and alignment](docs/interfaces/GRANT.md)

`make accept-s2.1` feeds the same known transport block to both transmit references, the pinned OCUDU chain and Sionna NR, and compares them on three grants: the two-block unit fixture inside a larger carrier, the demonstration profile (51 blocks, four receive ports) and the PDU of the Table II row.

| Gate | Result | What it showed |
|---|---|---|
| Committed tree and unit suite | Pass | The evidence came from a clean checkout. 185 tests ran, none skipped. The card's unit command ran on its own: 36 tests |
| References | Pass | The pinned Sionna 2.2.0 on the CPU and the Sprint 1.1 build, verified from one interpreter. The OCUDU source is still at `27e84cb3` after the runs |
| Adapter | Pass | `grant_fixture` built out of tree with the compile flags and the link line of the accepted `pusch_e2e_pipeline_test`: GCC 13.3.0, `-mcpu=neoverse-v2`, `ENABLE_CUDA`, 50 OCUDU archives, no compiler warning. Its receipt binds the executable `7a0e1c8ec1a0` to its sources and to the OCUDU build |
| Transport block and coding | Pass | 384, 9,736 and 54,296 bits in the record, in OCUDU and in Sionna. The same modulation, base graph and code block count |
| Resource elements | Pass | 312, 7,956 and 7,956 occupied elements in both grids, exactly the grant's data and pilot elements |
| Pilots and data symbols | Pass | 1,248 pilots equal their reference symbols and 14,976 data symbols of the two references agree, within the storage tolerance of the pinned grid. Largest difference 0.000757 for 16QAM and 0.00284 for 256QAM |
| Scale and phase | Pass | Data gain 1.00056 (16QAM) and 0.99914 (256QAM), pilot gain 1.00119, phase below 3 microradians. The limits are 0.5 percent and 1 mrad |
| Receiver check | Pass | The host PUSCH processor decoded each grid, without noise, to the original 48, 1,217 and 6,787 bytes, on one, four and one receive ports |
| Record | Pass | Three records archived and verified on reading. From the first, a second reader recovered pilot 0 at symbol 2, subcarrier 180, data symbol 0 and the block hash `76dccd9e2ed1`, the hash also with `sha256sum`. The same two commands gave the same answers by hand after the run |

One disagreement between the pinned test fixture and a true PUSCH was found and fixed. The test places the DM-RS sequence from the first block of the bandwidth part. PUSCH DM-RS starts at CRB 0. They coincide only when the bandwidth part starts at CRB 0, which is the only case the pinned test runs. The adapter uses CRB 0. The accepted run archives the difference: with the test's setting 19 of 24 pilots of the offset grant differ and the receiver's CRC fails. On the Table II grant both settings give the identical grid, and that grid equals the Sionna PUSCH grid. The transmit chain behind the Sprint 1.2 row therefore builds a legal PUSCH grid for that PDU.

Three properties of the pinned receiver were established in this sprint and are now rules in the cards of S2.2 to S3.1, S5.2, S5.5 and S6.9. [Precision of the stored grid](docs/interfaces/GRANT.md#precision-of-the-stored-grid) | [findings in the pinned source](docs/integration/OCUDU.md#findings-of-sprint-21)

- **Storage.** The resource grid stores complex bf16 on the host and on the device: eight significant bits per component. Every one of the 14,976 data symbols of the OCUDU grids equals the Sionna value rounded to that format. The 256QAM grid has a signal-to-quantization ratio of 56.1 dB. The same rule applied to values of arbitrary amplitude gives 55.6 dB at any signal level, which is the ceiling of a noiseless fixture at the receiver ingress.
- **Pilot amplitude.** Both pinned estimators are configured with a DM-RS amplitude of `10**(3/20)`, 0.998815 per component. The grid stores the pilot rounded to magnitude one per component. A pilot LS against the reference therefore reads a unity channel as 1.00119, the same on every pilot.
- **Receive ports.** A PUSCH PDU must list its receive ports as 0 to N-1 in order, because the pinned host demodulator reads the grid by list position. The grant record refuses any other list.

The archived records are portable. A rehearsal on an x86_64 host, with a CPU-only build of the pinned OCUDU source, produced the same bytes as the Spark for all three grants: the grant, the transport block, the indices, the reference symbols and both grids. The gate prints that comparison and does not grade it.

### Sprint 2.2 acceptance

`make accept-s2.2` ran every Sprint 2.2 gate on the same Spark at commit `e1db1df` and ended with `Sprint 2.2: ACCEPTED on this DGX Spark.` The run ID is `s22-spark-ecf8-20261121T004433Z`. All 35 gate lines passed, none failed and none was skipped. The gate took the grant record from the accepted Sprint 2.1 run, used the build that Sprint 1.1 accepted with nothing of OCUDU rebuilt, and ran on the CPU.

Sprint 2.2 gives the receiver its channel. `src/ocudu_cuda_isac/episodes/schedule.py` is the uplink grant fixture of an episode: it places PUSCH opportunities on the uplink slots of the TDD pattern and gives every slot index one legal grant, the base grant for its own slot and the same record with another slot otherwise. `episode.py` stores a matched pair on that schedule: the channel without the target, the channel with it and one noise realization, on the subcarriers, symbol times and antenna order of the grant record. `rt_channel.py` traces both with the pinned Sionna RT, and `episode_set.py` cuts walking trajectories into the demonstration episode set. [Episodes, schedule, set and analytic check](docs/interfaces/EPISODE.md)

`make accept-s2.2` grades four cases: the diagnostic exporter, the grant fixture, the reference pair and a small episode set. The reference pair is a 1 m2 reflector 25 m off the midpoint of a 20 m link, moving outward at 2 m/s, on the demonstration grant over eight TDD periods.

| Gate | Result | What it showed |
|---|---|---|
| Committed tree and unit suite | Pass | The evidence came from a clean checkout. 223 tests ran, none skipped. The card's unit command ran on its own: 23 tests |
| References and adapter | Pass | The pinned Sionna and Sionna RT 2.2.0 on the CPU and the Sprint 1.1 build, verified from one interpreter. The fixture adapter was built again and is the executable `7a0e1c8ec1a0` that Sprint 2.1 accepted, with no compiler warning. All three sources are at their locked revisions after the runs |
| Diagnostic | Pass | The four-time, twelve-tone exporter wrote both cases. Its direct path has the free-space gain within 1.7e-07 and the delay within 0.002 ns of 66.900 ns. Its return has the excess delay and the Doppler shift of the geometry, 5.314 ns and -8.699 Hz. An axis test only |
| Grant schedule | Pass | 24 opportunities on the uplink slots 7 to 9 of `DDDDDDSUUU` over eight periods, six grants. From the record: subcarriers -9.18 to +9.15 MHz, DM-RS symbols 321.09 us apart, the next slot's first DM-RS 178.91 us later |
| Base grant | Pass | The grant of slot 7 equals the record that Sprint 2.1 archived, every file verified. Its transmit grid `a18b72ae9b9ea84f` is the waveform of slot 7 |
| Added slot grants | Pass | The adapter realized the grants of slots 8, 9, 17, 18 and 19. For each it echoed the PDU, the 7,956 occupied elements and the 612 pilots equal the record, the 7,344 data symbols equal those of slot 7 bit for bit, 442 to 461 pilots differ from slot 7, and the host PUSCH processor returned the original 1,217 bytes with a CRC pass. Five records archived |
| Stored pair | Pass | Read back with the standard library: complex64 `[24, 4, 1, 14, 612]`. The opportunities, symbol times and subcarrier offsets equal the grant schedule, and six of six grants are bound to an archived transmit grid |
| Solver agreement | Pass | Path phases evaluated at the real symbol times differ from `Paths.cfr` by 3.2e-05 at the snapshot and by 3.2e-05 on its uniform grid of 507 steps through the DM-RS symbol times |
| Direct path | Pass | Target absent: all 822,528 coefficients are within 1.7e-05 of the free-space path, gain 3.3986e-04 and delay 66.900 ns |
| Scattered path | Pass | Target present minus absent equals the bistatic radar return, computed from the geometry at each symbol time, within 9.3e-04 of its amplitude 2.6468e-06 at all 822,528 coordinates, receive order included |
| Phase at unequal intervals | Pass | The return advances by -0.08741 rad in 321.09 us, by -0.04853 rad in 178.91 us and by -1.36098 rad in 5 ms. Largest error 4.6e-04 rad over every port and pilot. The Doppler shift, -43.33 Hz, is negative for the receding target |
| Delay-Doppler on the pilot mask | Pass | Pilot LS of `Y = H X + W` on symbol 2 of eight slots 5 ms apart, static part removed: the peak is at 163.4 ns and -50.0 Hz, within one bin of 179.8 ns and -43.3 Hz, and 23.0 dB above the median of the map. Resolution 54.5 ns and 25.0 Hz |
| Power margins | Pass | Scattered to direct power per resource element -42.19 dB, radar equation -42.17 dB. That is 13.4 dB above the RMS rounding error of the receiver grid and 6.0 dB above its bound. Noise at -30 dB covers the rounding |
| Time evolution | Pass | The geometry is traced again at every slot start. Extrapolating the target's paths by their Doppler shift to the next opportunity, 0.5 and 4.0 ms ahead, differs from the paths traced there by at most 1.8e-03. 29.7 ms per snapshot |
| Refusals | Pass | The loader refused a copy of the accepted pair with one changed byte, with a missing array, with a partial write and with a swapped axis |
| Episode set | Pass | Five matched pairs in a room with the TR 38.901 human target, 30 opportunities, the set and every episode verified on reading. Whole trajectories were assigned before tracing: three training episodes, one validation and one held-out. 30 positions are labelled on the 50 by 34 grid. The shortest path of every target-absent case has the free-space delay, gain and per-element phase of the recorded array within 1.6e-07. Every opportunity of every target-present case has at least 13 scattered paths |
| Second reader | Pass | A separate process recovered H at opportunity 0, symbol 2, subcarrier 0, 1.7e-05 from the analytic value, and the two phase steps of the return. `sha256sum` gave the hash of `b_h.npy`, `744702180dddaea1`. Run by hand after the gate, the reader returned the coefficient of port 3 that the driver's reader had recovered, and `sha256sum` the same hash |

The receiver check of the five added grants was the open risk of this sprint, because the development host has no OCUDU build and rehearsed with a stand-in for the adapter. On the Spark the pinned transmit chain built each grid and the pinned host receiver decoded it. The five grids agree, on the first 16 digits of their SHA-256, with grids computed on the development host as the Sionna NR grid rounded to the storage format. The rounding rule of Sprint 2.1 therefore holds on 36,720 more data symbols and 3,060 more pilots.

The pair is portable. Its three arrays have the same SHA-256 on the Spark and on an x86_64 host with the same pinned packages: `a_h.npy` `8b7ef3971f4c7975`, `b_h.npy` `744702180dddaea1` and `w.npy` `2b5b46bb1051c23f`. The single-precision trace on the LLVM variant and the noise generator give the same bytes on both architectures. The gate prints that comparison as a NOTE and does not grade it. The room set was not compared byte for byte. Its agreement with `Paths.cfr` was 7.5e-05 on the Spark and 7.2e-05 on the x86_64 host, so some step of the room trace differs in its last bits between the two.

Three facts from this sprint are rules for the next tickets. [Details](docs/interfaces/EPISODE.md)

- **Slot grants.** Opportunities 5 ms apart alternate between slot 7 and slot 17 of the frame, and their DM-RS differ. One transmit grid per slot index is needed, and the accepted run archives all six.
- **Amplitude.** The stored channel is physical: the direct path of the reference pair has amplitude 3.4e-04 and the return 2.6e-06. The entry records the factor that gives the receiver grid unit signal power, 2,942.4 for the reference pair. Applying it is a requirement of S2.3 (decided 2026-11-20): `Y = g (sum(H X) + W)`. The pinned CUDA receiver keeps its estimates in 16-bit floats, whose normal range ends at 6.1e-05. Without the factor the return and the noise fall below that range and the return keeps about five significant bits. With it both keep ten or more.
- **Doppler across the band.** The return's phase advances by `2 pi nu (1 + f / f_c) dt`. Over 5 ms the band edges differ from the carrier by 3.6 mrad, which a check must allow for.

### Sprint 2.3 acceptance

`make accept-s2.3` ran every Sprint 2.3 gate on the same Spark at commit `d027aad` and ended with `Sprint 2.3: ACCEPTED on this DGX Spark.` The run ID is `s23-spark-ecf8-20261121T032401Z`. All 35 gate lines passed, none failed and none was skipped. The gate took the pair and the six grant grids from the accepted Sprint 2.2 run, used the build that Sprint 1.1 accepted with nothing of OCUDU rebuilt, and ran on the CPU in 52 s.

Sprint 2.3 puts the channel into the receiver's input. `native/ocudu_adapter/replay/` is the first native component of the project that touches the receiver grid: a host-free core with the arithmetic, the gain rules, the owned grids and the readback digest, and the tool `grid_apply`, which fills the pinned OCUDU resource grid through its writer and reads it back through its reader. For every opportunity it computes `Y = g (sum(H X) + W)` over the whole carrier grid in double precision, with the grid gain the entry records, 2,942.4 for the reference pair. `src/ocudu_cuda_isac/episodes/grid_apply.py` builds the plan the tool reads, binds every transmit grid by the hash the entry names and holds the reference of the arithmetic. [Channel application, plan, grid states and accepted run](docs/interfaces/GRID_APPLY.md)

`make accept-s2.3` grades five cases: the unit vectors, the six grant grids through a unity channel, the reference pair, a changed grant and the refusals.

| Gate | Result | What it showed |
|---|---|---|
| Committed tree and unit suite | Pass | The evidence came from a clean checkout. 274 tests ran, none skipped. The card's unit command ran on its own: 29 tests, which build the host-free core without OCUDU and compare its kernel with the Python reference by digest |
| References and adapter | Pass | The Sprint 1.1 build, verified from the project interpreter. The adapter was built again with `ENABLE_CUDA` and the flags of the pinned test: `grid_apply` `158644b5a177` beside `grant_fixture` `7a0e1c8ec1a0`, the executable Sprints 2.1 and 2.2 accepted. 19 adapter tests, no compiler warning. All three sources are at their locked revisions after the runs |
| Native unit cases | Pass | The 14 cases of the host-free core and the unit vectors through the pinned resource grid passed in the adapter build |
| Unit vectors | Pass | The two-transmit vector of Sprint 0.3 through the pinned grid gives `1+3j, 2-4j, 6.5, 14` with `g = 1` and exactly twice them with `g = 2`. A zero channel with the gain of the pair gives `g W`, rounded to the storage format |
| Unity | Pass | The six grant grids through a unity channel on four receive planes with zero noise and `g = 1`: every plane returns `x.npy` bit for bit, 8,568 values each, and storage changed none of 411,264 components. The gain is an override, recorded with its reason |
| Plan | Pass | Built from the verified entry: 48 opportunities of two realizations, six grants, each transmit grid bound by the hash the entry names. One gain, 1.1e-16 from `1 / sqrt(reference_power)` |
| Native run | Pass | 48 grids applied, written through the pinned writer, made ready, read back and released. Two owned grids, generations 1 to 24, and each grid kept its digest while the next one was filled. 22 input files verified against their hashes |
| Values before storage | Pass | Native against the complex128 reference in Python on all 1,645,056 elements of 48 opportunities: every one within `1e-6 + 1e-5 abs(reference)` after the gain, largest error 8.1e-08, and every one equal to the reference converted to single precision, bit for bit |
| Stored grids | Pass | Read back from the pinned grid: all 1,645,056 elements, the 1,527,552 occupied ones included, equal the reference rounded to the storage format. 48 of 48 readback digests equal the reference's. Storage moved a component by at most 0.00389 of its magnitude, against the bound `2**-8` |
| Gain | Pass | `g = 2,942.4244`. The mean power of `g H` of the target-absent case on opportunity 0, over four receive ports and 7,956 occupied elements, is `1 + 7.1e-08` in Python and in the native tool. Both cases carry this gain and one noise file. `g sigma = 0.031623` |
| Difference of the two cases | Pass | The stored grid of the target-present case minus that of the target-absent case equals `g (H_b - H_a) X` within the rounding bound on all 1,645,056 components, largest error 0.99 of the bound. On the occupied elements the return has an RMS of 8.06e-03 and the rounding of the two grids 2.35e-03. Before storage the two cases differ by the return alone |
| Noise after storage | Pass | The stored grid minus the signal part has a power of -29.989 dB relative to unit signal power, against -30.000 dB. The rounding of the grid lies 55.8 dB below the stored values, so the noise is held |
| Receive planes | Pass | Mean power on the occupied elements of planes 0 to 3: 1.07780, 1.07772, 1.07767 and 1.07740 with the target absent, 1.07793, 1.07774, 1.07774 and 1.07749 with it present |
| Grant of each opportunity | Pass | The transmit grid follows the grant of the slot. A reference that keeps the grid of slot 7 at every opportunity differs from the stored grid on 455, 461, 455, 460 and 442 pilots of slots 8, 9, 17, 18 and 19 and on no other element, the counts Sprint 2.2 recorded |
| Changed grant | Pass | On one channel that holds its own coordinates, `demo-51prb-4rx` occupies 7,956 elements on subcarriers 0 to 611 and `two-prb-offset` 312 elements on subcarriers 180 to 203. Each stored grid is nonzero on exactly those elements of all four planes and equals H at those coordinates times X of the grant. A transmit grid of 600 subcarriers on the 612-subcarrier carrier is refused |
| Refusals | Pass | Refused by the plan check and by the native tool, each naming its rule, with status 2 before a file is written: seven gain rules and six binding rules. The plan builder refused an entry that binds another hash than the stored grid and an entry that binds no grid to a grant |
| Access | Pass | `ingress/` holds the 48 stored grids and their 48 descriptors and nothing else. A descriptor has the 15 declared fields and opaque opportunity identifiers, and no hash or name of H, W or a transport block and no label |
| Second reader | Pass | A separate process verified the 48 stored grids against their digests and recovered Y at opportunity 0, symbol 2, subcarrier 0: `+0.750000-1.179688j` on port 3 and `+0.746094-1.171875j` on port 0. `sha256sum` gave the digest of the first grid, `6a0a3c28258f7c04` |

The Spark's own build was the open risk of this sprint, because the gate had been rehearsed on an x86_64 host against a CPU-only build of the pinned source. On the Spark the replay code compiled with `-mcpu=neoverse-v2` and no warning and linked against the 50 archives and the CUDA runtime of the accepted build.

The result is portable. The 48 stored grids and the 48 grids before storage have the same bytes on the Spark and on the x86_64 host, and the digest over the stored grids is `62f9e9d2dbe09615` on both. Every number the driver measured is the same on both hosts except the fill time. The order of the arithmetic is fixed (products in transmit-port order, then the noise, then the gain, then single precision, then storage), so the native kernel and the Python reference agree bit for bit. The gate prints the digest comparison as a NOTE and does not grade it.

Timing is recorded and not graded. Filling one grid of 4 by 14 by 612 elements took 1,436 us on average and 1,445 us at most on one core of the Spark, against 1,163 us on the x86_64 host. A slot lasts 0.5 ms.

Two facts from this sprint are rules for the next tickets. [Details](docs/interfaces/GRID_APPLY.md#for-sprint-24)

- **Owned grid and descriptor.** S2.4 takes each grid from `session::deliver` as a ticket on an owned pinned grid and hands the reader of that grid to the PUSCH processors. The `o1-grid/v1` descriptor carries the grant, the frame, the slot and the readback digest. It names no channel, no noise, no transport block and no label. The gain is applied once, before storage, and a stored grid is never rescaled.
- **Prepared ahead.** A grid takes longer to fill than a slot lasts. The channel application is stimulus and no claim is made about its speed, but a paced run (S4.1) has to prepare each grid ahead of its slot. The owned grids and their ready state are built for that.

### Sprint 2.4 and Sprint 2 acceptance

`make accept-s2.4` passed on the DGX Spark `spark-ecf8` on 2026-11-21 (UTC) at commit `542b72c`, run `s24-spark-ecf8-20261121T083424Z`: 48 gate lines, none failed and none skipped. `make accept-s2` then audited the four runs of the sprint as one chain, run `s2-spark-ecf8-20261121T083811Z`: 27 gate lines, 9 of 9 stages and 17 of 17 checks. [Accepted run](docs/interfaces/REPLAY_PUSCH.md#accepted-run) | [the audit](docs/interfaces/REPLAY_PUSCH.md#the-sprint-2-audit)

| Gate | Result | What it showed |
|---|---|---|
| Unit suite and unit gate | 342 tests and 21 tests, none skipped | The 12 cases of the host-free dispatch core, and the native outcome rule equal to the Python one on 64 of 64 inputs |
| Adapter | `replay_pusch` `d2bcc09539c3` beside the unchanged `grid_apply` and `grant_fixture`, 34 CTest cases, no warning | `receiver/` compiles on AArch64 with `ENABLE_CUDA` and links against the accepted build |
| Integration grade | 8 of 8 cases, 36 of 36 checks, judgment `pass`, hardware gate closed | 280 decodes planned in 7 runs, 280 records under unique keys: 256 `decoded` and the 24 decodes of bad input `crc_fail`, no other outcome and no anomaly |
| Executed path | 140 of 140 on the host path, 140 of 140 on the resident CUDA path | No decode requested from the CUDA processor fell back. Every resident decode read a host resource grid. |
| Pair | 48 of 48 grid digests equal the reference and the grids Sprint 2.3 stored | 96 of 96 decodes return the original 1,217 bytes in one LDPC iteration, both members and both processors alike. Reported SINR 36.50 dB on the host path and 36.44 dB on the resident path |
| Payload per opportunity | 24 distinct blocks | 96 of 96 decodes return the block of their own opportunity |
| Bad input | 24 of 24 `crc_fail` after 6 LDPC iterations | None acknowledged, no bytes kept. A reported SINR does not predict the CRC |
| Sprint 2 as one chain | 9 of 9 stages | From the grant record of Sprint 2.1 to 96 files of decoded bytes with the hash of its transport block, on one host and one build |

Service time on four receive ports, recorded and not graded: 660 us on the host path and 292 us on the resident path. Three findings go on: the resident path bounds the SINR it reports to -10 to 60 dB, the resident decodes read a host resource grid, and the loopback of the Sprint 2.1 fixture adapter destroys its processor before its workers are joined. [Findings](docs/interfaces/REPLAY_PUSCH.md#findings)

### Sprint 3.1 acceptance

`make accept-s3.1` passed on the DGX Spark `spark-ecf8` on 2026-11-21 (UTC) at commit `cf562d6`, run `s31-spark-ecf8-20261121T174511Z`: 35 gate lines, none failed and none skipped. The gate decodes the plans that the accepted Sprint 2.4 run archived once more, with the first observer attached. `native/ocudu_adapter/observer/` holds a host-free observer core and the tool `pilot_ls`. Before each decode the observer reads the ready grid and computes `H = Y conj(X) / |X|^2` on the pilots of the grant's sensing source, on every receive plane: the stage O2, 4 by 2 by 306 complex64 values at the demonstration profile, with an `obs-descriptor/v1` record that names the digest of the grid it was computed from. `replay_pusch` is not edited and stays the untapped control. [Accepted run](docs/interfaces/PILOT_LS.md#accepted-run) | [estimate, layout and record](docs/interfaces/PILOT_LS.md)

| Gate | Result | What it showed |
|---|---|---|
| Unit suite and unit gate | 404 tests and 31 tests, none skipped | The vectors of the card, the 11 cases of the host-free observer core, and the native estimate equal to the Python reference bit for bit on 4,616 of 4,616 vectors |
| Adapter | `pilot_ls` `c3a214c2869c`, 48 CTest cases, no warning | `observer/` compiles on AArch64 with `ENABLE_CUDA` and links against the accepted build. Its self-test taps the pinned grid and decodes it |
| Control executables | `replay_pusch` `d2bcc09539c3`, `grid_apply` `158644b5a177`, `grant_fixture` `7a0e1c8ec1a0` | This tree builds the three executables of Sprint 2 with the bytes of the accepted Sprint 2.4 run |
| Integration grade | 9 of 9 cases, 23 of 23 checks, judgment `pass`, hardware gate closed | 128 grids tapped in five runs, 352 decodes in six |
| Unity | 15,324 of 15,324 estimates are 1.0011865 | The stored pilot of magnitude one per component against the reference of 0.998815, on every pilot, port and symbol, at the coordinates of each grant |
| Direct path | 58,752 of 58,752 estimates within the rounding of the grid | The estimate is the archived channel times the gain and the pilot ratio on all four receive planes, largest relative error 0.0029 |
| Pair, ingress | 48 of 48 | Every observation names the grid digest that the accepted `replay_pusch` decoded: O2 is taken from the exact O1 grid passed to PUSCH |
| Pair, estimate | 48 of 48 payloads bit for bit | The native estimate on the pinned grid equals the reference, and a sparse oracle in double precision agrees within 5.8e-08. 117,504 estimates, all nonzero and valid. The plane order is shown against the oracle with reversed planes |
| Pair, decodes | 96 of 96 radio records and decoded files | With the observer attached every key, grid digest, CRC flag, outcome and decoded byte is the control's. The same holds for the same executable without the tap |
| Payload per opportunity | 48 of 48 grids differ, 48 of 48 estimates do not | Changing the data elements leaves O2 as it was, bit for bit |
| Executed path | 176 of 176 on the host path, 176 of 176 on the resident CUDA path | No decode fell back with the observer attached. All 352 ended as `decoded` |
| Refusals and rerun | 14 plan rules, one opportunity alone | Each rule is refused by the plan check and by the native tool before a file is written. A repeated opportunity gives the same grid, estimate and bytes |

Recorded and not graded: the estimates of the two members of the pair differ by -41.8 dB, where the episode states the return at -42.2 dB. One observation takes 1,471 us against a slot of 500 us. About nine tenths of that is a SHA-256 digest of the whole grid that the native observer takes after its read, and the harness of Sprints 2.3 to 3.1 hashes each grid seven more times per opportunity. These hashes prove bytes in the gates and are outside every PUSCH service time that was reported. They may not run in a timed path: S3.2 removes them from the observer and measures the tap against a ceiling of 150 us, S4.1 removes them from the paced harness. [Requirement and breakdown](docs/dev/DEVELOPMENT.md#added-by-the-first-observer-run-of-2026-11-21)

A first run of the gate, `s31-spark-ecf8-20261121T171416Z`, built and ran the tool and failed on two checks of the gate itself: the configuration expected the observations of one member of the pair, and a check required a reversed plane order to fail on a direct channel that is the same on all four receive planes. Both were corrected. [First run](docs/interfaces/PILOT_LS.md#first-run-on-the-spark)

### Sprint 3.2, accepted

Sprint 3.2 builds the path by which an estimate leaves the receiver process, and removes every checksum from that path. `native/obs_ring/` holds the record `obs/v1` and the ring `obs-ring/v1`: one producer, one consumer, first in, first out, in host shared memory. `native/ocudu_adapter/transport/` joins the observer to the ring, and `pilot_ls` reads a second plan version that names the evidence mode and the ring. [What was built and how it is accepted](docs/interfaces/SNAPSHOT_RING.md) | [record, ring and view](docs/interfaces/OBSERVATION.md)

| | Built |
|---|---|
| The queue | Record `sequence` n lies in slot n modulo the slot count. Admission is two loads and a compare, publication is one store. Without a credit an observation is dropped and counted. The producer never waits, takes no lock and makes no system call between an admission and its publication. Eight slots of 20,288 bytes for the demonstration, 166,400 bytes with the header. |
| Ownership | The producer is the only writer of the segment and the consumer of its cursor page. Each maps the other's memory through a read-only descriptor. A slot carries a generation counter. Credits return through a released count, not through messages. |
| A consumer that leaves | A new consumer starts a new epoch. What the old one held is reclaimed then, and counted. A released count that moves backwards or passes the published count closes the channel and is counted. |
| The timed path | In the mode `timed` the observer writes its estimate straight into the slot. No SHA-256, no allocation and no text from the read of the grid to the publication. The join keys are hashed when the plan is bound. A grid and a decode are owned by the ticket of the fill. The hash of the core counts its own calls, so a run shows a count of zero between its first fill and its drain. |
| The audited path | The mode `audit` is the run of Sprint 3.1 with every digest, and publishes each observation on the ring as well. The same plan in both modes must give the same payload bytes and the same decoded bytes. |
| Two readers | `obs_tail`, the native consumer that keeps what it reads, and `ocudu_cuda_isac.sensing.obs`, the standard-library reader. Both read the same records from one producer. |

`make accept-s3.2` passed on the DGX Spark `spark-ecf8` on 2026-11-21 (UTC) at commit `9b96578`, run `s32-spark-ecf8-20261121T204425Z`: 52 gate lines, none failed and none skipped, 10 of 10 cases and 40 of 40 checks, hardware gate closed. The adapter was built with `ENABLE_CUDA` against the Sprint 1.1 build, with 75 adapter tests and no compiler warning. [Accepted run](docs/interfaces/SNAPSHOT_RING.md#accepted-run)

| Measured on the Spark, reference pair, 47 observations of 19,584 bytes per run, the first one excluded | Value |
|---|---|
| Tap in the mode `timed`, from the read of the grid to the publication store | Mean 11.0 us, largest 18.7 us. Ceiling 150 us. |
| Of it, the estimate | Mean 10.5 us, largest 17.5 us |
| Of it, the admission, the header and the publication | Mean 0.6 us, largest 1.2 us |
| The same observation in the mode `audit` | 1,472.2 us: 1,360.5 us the digest of the grid, 97.5 us the digest of the payload, 9.8 us the estimate |
| SHA-256 between the first fill and the drain | `timed`: 0 digests over 0 bytes. `audit`: 720 digests over 106,359,888 bytes. |
| System calls of the producer between an admission and its publication | 0 over 48 publications, in both wait modes |
| Delay of a spinning consumer, publication store to its read | Median 0.4 us, largest 0.8 us |
| Delay of a consumer asleep on its eventfd | Median 8.2 us, 99th percentile 17.5 us, largest 119.4 us. The producer's eventfd write after the store makes the publication step 5.3 us at the median. |
| A drop without a credit, consumer of 5 ms per record | 32 of 48 dropped and counted, the longest in 0.24 us. At most 8 of 8 slots held, nothing overwritten. |
| A consumer killed while it holds a credit | Two epochs. 33 published, 15 dropped, 25 released, 8 reclaimed, none pending. |
| Decodes in all 7 runs | 672 of 672 with every byte equal to the original, 336 on the resident CUDA path and 336 on the host path |

What the run says. Removing the checksums from the path takes the tap from 1,472 us to 11.0 us, and the estimate is then 95 percent of it: the ring adds 0.6 us. The 48 payloads of the timed run, hashed after the run, are those of the audited run and of the accepted Sprint 3.1 run `s31-spark-ecf8-20261121T174511Z`, and all 96 radio records of the pair equal those the accepted `replay_pusch` wrote without an observer. A sleeping consumer costs the producer one system call after each store and has a tail of 119 us on an untuned host, which is why a timed sensor spins. These are 47 values per run on a host that is not tuned and not paced: they bound the cost of the tap, and are no tail statement. The paced runs of S4.1 and the calibration of S4.6 give those.

Two host-free cores that Sprint 2 accepted, `replay_core` and `dispatch_core`, were amended for the hash count and the timed mode, so `replay_pusch`, `grid_apply` and `pilot_ls` have new bytes from this sprint on. The gate shows the equivalence by behavior, against the runs that were accepted before the amendment. Sprint 2 was then accepted again on the amended tree: `make accept-s2 RERUN=1` passed on 2026-11-21 at commit `7bad0f4`, chain run `s2-spark-ecf8-20261121T202119Z` with the runs `s21-`, `s22-`, `s23-` and `s24-spark-ecf8-20261121T202119Z`, 9 of 9 stages and 17 of 17 checks. Before the Spark run a development host passed the native cases under the thread and address sanitizers. [Checked on a development host](docs/interfaces/SNAPSHOT_RING.md#checked-on-a-development-host)

### Sprint 3.3, accepted

Sprint 3.3 is the first sensing code: the reference of what a sensor makes of the records on the ring. `src/ocudu_cuda_isac/sensing/features.py` forms a masked radar cube indexed by pilot frequency, slow time and array element, a delay-Doppler view of it after static removal, and a compact feature vector, one record at a time. It changes no native source. [Cube, view, vector, stream and gate](docs/interfaces/PILOT_FEATURES.md)

| | Built |
|---|---|
| The cube | `X[frequency, slow time, rx]` from `obs/v1` records of the stage O2, with a validity bit per element. A masked element is zero and is never read. Slow time is the start of the useful part of each selected symbol, from the slot of the record. |
| The window | The demonstration takes symbol 2 of the first uplink slot of each period: eight samples 5 ms apart, the homogeneous sequence of the card, transformed by FFT. A second Doppler step sums at the actual times of every slot and symbol. |
| The stream | Each record of the sequence is answered at once: `ok` with a view and a vector, `insufficient_history` while a window fills, `gap` for a window with a missing observation, `invalid`, or `rejected`. A history never spans two epochs or two episode keys. |
| The view | 32 delay bins of 54.47 ns by 8 Doppler bins of 25 Hz, signed: a later arrival at a higher delay bin, a receding target at a negative Doppler bin. Unambiguous to 16.67 us and from -100 to +75 Hz. |
| The vector | 13 named values: peak over median, moving over static power, the peak cell, Doppler centroid and spread, and the strongest cell of each Doppler bin over the median. The presence head of S3.4 reads it. |
| The registry | Every step is a named entry with its execution device. All of this sprint run in NumPy on the CPU with `runtime: reference`. S6.12 adds the native units. |
| The evaluator | `src/ocudu_cuda_isac/episodes/pilot_features.py` cuts the same windows offline from the files of the audited observer and computes the same view by explicit Fourier sums, and joins a window to the geometry of its episode. |
| The import rule | `tests/unit/test_core_imports.py`: the sensing package imports no episode module, no host and no simulator. |

**Accepted on the DGX Spark `spark-ecf8` on 2026-11-22 (UTC)**: `make accept-s3.3 S32_RUN=results/runs/s32-spark-ecf8-20261121T204425Z` passed at commit `0c4ad18`, run `s33-spark-ecf8-20261122T004801Z`: 43 gate lines, 8 of 8 cases and 28 of 28 checks, hardware gate closed, 490 unit tests, and every executable with the bytes of the accepted Sprint 3.2 run. [Accepted run](docs/interfaces/PILOT_FEATURES.md#accepted-run)

| Measured on the Spark, on the traced reference pair | Value |
|---|---|
| Strongest moving cell with the target | Delay bin 3 and Doppler bin -2: 163.4 ns and -50.0 Hz |
| Return from the geometry of the episode | 179.8 ns and -43.3 Hz: the peak is -16.4 ns and -6.7 Hz from it, inside one bin of 54.5 ns and 25.0 Hz |
| Peak over the median of the view | 22.3 dB with the target, 6.2 dB without (graded: at least 15.0 and at most 10.0) |
| Peak over the mirrored Doppler cell | 21.9 dB (graded: at least 10.0) |
| Stream against explicit Fourier sums | Largest cell difference 3.9e-15 of the peak, largest feature difference 1.8e-14 |
| Live ring, sensor in a second process | 48 published, 0 dropped, ring depth at most 1 of 8 slots. 16 answers equal to the recorded stream, 2 view hashes and 2 vectors bit for bit. |
| Receiver with the sensor attached | 96 decodes equal to the control, 48 on the resident CUDA path |
| Reference sensor, not graded | Read 20.9 us (median) after the publication store. Complete vector 756 and 871 us after it. Warm-up 6.41 ms before the attach. Python and NumPy on the CPU: no latency of the sensing core. |

What the run found:

- **The return is where the geometry puts it, through the whole chain.** Traced channel, pinned grid, native observer, ring and reference agree on one cell, and the sample times of the native records are those of the reference rule: no record was rejected.
- **Moving over static power does not see this target.** It reads -33.0 dB with the target and -33.6 dB without: the return is 42 dB under the direct path and the noise decides the value. The peak over the median and the Doppler profile carry the evidence. S3.4 selects its columns accordingly.
- **A first use belongs before the first slot.** A first run (`s33-spark-ecf8-20261121T223346Z`, not accepted) lost 6 of 48 observations in the live case: the reference sensor loaded its FFT at its first complete window and held the ring for 8.4 ms. The receiver was not delayed, dropped and counted, and the affected window was answered as a gap and not with a vector. The sensor now warms up before it attaches. The native core does the same with its plans and its engine.
- **The time of an answer is read when the answer is complete.** The first run stamped it before the view was computed.

Before the Spark run, a development host without an OCUDU build had passed the unit gate, among it the reference pair synthesized from its analytic model with the same peak cell, and the integration driver against a double of the native tool, where each of 17 injected faults fails the check that names it.

### Sprint 3.4, accepted

Sprint 3.4 turns the feature vector into a decision. `src/ocudu_cuda_isac/sensing/presence.py` holds one linear head with a frozen normalizer and a frozen threshold, and `scripts/train_presence.py` fits it. It changes no native source. [Artifact, fit, decisions and gate](docs/interfaces/PRESENCE_HEAD.md)

| | Built |
|---|---|
| The artifact | `presence-head/v1`: `head.json` with the input contract, the normalizer, the threshold and the hashes, and `weights.f32` with one float32 per column and the bias. No sample, no label, no split. An altered byte is refused. |
| The head | Bound to the names of the vector: it reads `peak_to_median_db` and the seven `profile_db@` values of the 13. Float32, one vector of shape `[1, 8]` at a time, on the CPU or on a CUDA device selected when it is loaded. Weights `[1, -1]` on the vector `[2, 1]` give the logit 1 and the probability 0.731059. |
| The fit | The split is that of the episode set, by trajectory. The normalizer and the weights read the training windows, the threshold the target-absent validation windows, and `freeze.json` names the hashes of the artifact and of both manifests before a held-out entry is replayed. |
| The decision | `presence-decision/v1`, one per answer of the feature stream: a complete window carries the logit, the probability, the decision, the digest of its input and the hashes of the model. Any other answer carries its status and no score. |
| The sensor | `python -m ocudu_cuda_isac.sensing.presence follow`: the reference sensor with the head, a second process that loads the artifact once and is warm before it attaches |
| The evaluator | `src/ocudu_cuda_isac/episodes/presence_head.py`: the split table of a set, the plans of its replay with the transmit grids of the reference pair, and the comparison of two decision sets by window and input digest |
| The gate | `make accept-s3.4`: ten cases on the accepted Sprint 3.3 run and a traced episode set |

**Accepted on the DGX Spark `spark-ecf8` on 2026-11-22 (UTC)**: `make accept-s3.4` passed at commit `b812d12`, run `s34-spark-ecf8-20261122T024432Z`, on the first attempt: 49 gate lines, 10 of 10 cases and 35 of 35 checks, hardware gate closed, 540 unit tests, and every executable with the bytes of the accepted Sprint 3.3 run. The stimulus is the demonstration set, traced on the Spark for the first time (`demo-set-20261122T022210Z`, 350 pairs). [Accepted run](docs/interfaces/PRESENCE_HEAD.md#accepted-run)

| | Accepted run |
|---|---|
| Receiver replay of the training and validation entries | 14 runs, 16,320 opportunities: 32,640 of 32,640 decodes original, 16,320 on the resident CUDA path, no observation dropped, 121 s |
| Windows | 600 training and 80 validation windows, one per realization, no gap and no rejected record |
| Head | `dd-linear-v1`, artifact `4b1444fa8b0deeeb`: 8 columns, 36 bytes of weights. Loss from 0.693 to 0.045 in 20 evaluations. |
| Threshold | Logit -3.675: 2 of the 40 target-absent validation windows above it, as a false-alarm target of 0.05 allows |
| Freeze | 2026-11-22T02:53:45Z, before the first plan of a held-out entry. No held-out key in a manifest. |
| Held-out run | 480 opportunities, 960 decodes original. The sensor loaded the artifact once on `cuda:0` (NVIDIA GB10), was warm 5.4 s before the first record and lost none: 20 scored windows. |
| Device against offline | The same features and input digests. Largest difference 9.5e-07 on a logit and 1.9e-08 on a probability, against 1e-04 and 1e-05. The same 20 decisions. |
| Recorded, not graded | Held-out: 10 of 10 target-present episodes detected, 1 of 10 target-absent alarmed. One inference on the GB10: 372 us in the median, `runtime: reference`. |

What the run found:

- **The two classes do not overlap on this set.** The peak over the median is at least 17.1 dB with the person and at most 7.7 dB without, and the area under the curve is 1.000 on the training, the validation and the held-out windows.
- **The threshold lies inside the target-absent windows, because its rule puts it there.** Two of forty validation negatives may exceed it, so 12 of 300 training and 1 of 10 held-out target-absent windows do. A threshold between the logits -3.38 and -0.44 would have given no alarm and no miss. The rule was fixed before the held-out windows were read and stays: the operating point is set in S4.6.
- **The target-absent windows of a set differ by their noise only**, so a false-alarm rate on them is that of the noise at 30 dB.
- **A set replays through the receiver.** No entry of a set had been decoded before. With the transmit grids of the reference pair every block of the 350 pairs came back, in runs without stored grids, which had not been exercised on hardware either.
- **A second CUDA process beside the receiver lost no observation**, without MPS, with the sensor started first.

Before the Spark run, a development host without an OCUDU build and without a CUDA device had passed the unit gate, and the integration driver against doubles of the native tool and of the ray tracer, where each of 22 injected faults fails the check that names it.

### Sprint 5.2, accepted

Sprint 5.2 sits inside phase S3 and is the first sprint that changes the receiver. It exports the channel estimate that the resident CUDA path computes for its own equalizer, the stage O3: after pilot least squares, frequency interpolation and frequency smoothing, per DM-RS symbol and before time interpolation, in the FP16 of the device. That is the stage cuSense reads from Aerial. [Layout, patch, capture, plan and parity](docs/interfaces/NATIVE_ESTIMATE.md)

| Part | Where |
|---|---|
| The receiver patch | `patches/ocudu/0001-pusch-resident-estimate-tap.patch`: one new header and 49 lines in `pusch_demodulator_gpu_impl.cpp`, no kernel. A read-only tap that costs one atomic load per PDU without an observer. `patches/ocudu/series.json` names its hashes, and `scripts/receiver_patch.py` reads it |
| The patched build | `python scripts/native_ocudu.py build --patched`: a separate worktree of the pinned commit in `build/ocudu-src-o3`, built into `build/ocudu-spark-o3`. The submodule stays pristine and `isac/bin` keeps naming the accepted Sprint 1.1 build, the hook-free control |
| The capture | `native/ocudu_adapter/estimate/`, host-free: expect before the decode, one asynchronous copy from the tap, ordered after the producer, and the completion event asked after the decode. Six outcomes, one account line per decode. The device calls are in `native/ocudu_adapter/observer/estimate_tap.h`, compiled only into the adapter build against the patched receiver (`python scripts/native_adapter.py build --o3`) |
| The plan | `pilot-ls-plan/v3`: version 2 with an `estimate` object. A build without the route refuses it. O3 is written to `estimate/` and published on a ring of its own |
| The reference | `src/ocudu_cuda_isac/sensing/native_estimate.py`: FP16, the layout, the matched estimator on the stage O2 and the parity rules. `src/ocudu_cuda_isac/episodes/native_estimate.py`: the plan, the reader and `o3_parity.json` |
| The gate | `make accept-s5.2`: the series, the patched build, both adapters and ten cases on the plans of the accepted Sprint 3.1 run |

**Accepted on the DGX Spark `spark-ecf8` on 2026-11-22 (UTC)**: `make accept-s5.2` passed on `spark-ecf8` on 2026-11-22 (UTC) at commit `2238d8c`, run `s52-spark-ecf8-20261122T045318Z`, on the first attempt: 69 gate lines, 10 of 10 cases and 44 of 44 checks, hardware gate closed, 583 unit tests with none skipped. The patched receiver built from a worktree of the pinned commit (18 kernel sources for `sm_121`, nvcc 13.5.37), and the accepted Sprint 1.1 build kept its receipt and its links. Over six audited runs and one timed run the patched receiver decoded 448 times, and every radio record equals the one the unpatched build wrote: key, CRC flag, decoded bytes and executed path. Every stage O2 payload has the digest of the accepted Sprint 3.1 run. Each of the 176 decodes on the resident CUDA path with an observer left exactly one estimate, with none absent, refused, failed or outside a capture, and every completion event was found complete at the first query. The flat estimates are exactly 1.0009765625 on every element. Against the matched estimator on the pilot support the 128 audited estimates give a numerator of 6.81e-04 over a denominator of 3.10e+05 on 309,084 samples, an NMSE of 2.2e-09: 4.4e-10 on the varying channel against a bound of 1.9e-06, and at most 6.7e-09 per estimate on the noisy pair against a bound of 9.7e-05. In the timed run both stages went to their rings, 48 of 48 estimates were published and released, with no digest between the first fill and the drain and no system call in a publication.

| | Accepted run |
|---|---|
| Patched receiver | built from a worktree of `27e84cb3` with the series `23cf49ee6a4b`; the unpatched build keeps its receipt `644193b58ba8` and its links |
| Adapters | `pilot_ls` `2fbd347e997d` without the route, `9f9680d43d2b` with it, 87 adapter tests each, no compiler warning |
| Decodes against the unpatched build | 448 of 448 radio records and files of decoded bytes equal, hook off and on |
| Flat channel | every element of 8 estimates is 1.0009765625+0j |
| Parity on the pilot support | 128 estimates, 309,084 samples, NMSE 2.2e-09; noisy pair at most 6.7e-09 against a bound of 9.7e-05 |
| Cost of the tap, not graded | 4.2 us inside the decode, 5.3 us after it; CUDA service time 201.2 us with the hook off and 210.0 us with it on, audited |
| Landing | the timed copy lands in its slot: `cudaHostRegister` accepts the ring segment on the GB10 |

One finding at the pin was confirmed by the run: the smoothing kernel filters its buffer in place and does not fix the order in which a block reads its neighbours, so the exported estimate is not the same in every decode of one grid. 45 of 48 timed payloads have the bytes of the audited run and 3 do not, all within the declared tolerance. [Accepted run](docs/interfaces/NATIVE_ESTIMATE.md#accepted-run)

### Sprint 3.5, implemented

Sprint 3.5 runs the cuSense-like branch on the stage O3 that Sprint 5.2 exports, with the pilot comb of the stage O2 as a second input entry. `src/ocudu_cuda_isac/sensing/cusense.py` holds the three stages, the artifact and the sensor, `src/ocudu_cuda_isac/sensing/models/cusense_cnn.py` the model entry, and `scripts/train_cusense.py` fits them. It changes no native source. [Stages, artifact, training, decisions and gate](docs/interfaces/CUSENSE_FRONTEND.md)

| Part | Where |
|---|---|
| Stage 1, the background | `H_B[a, k, s]`: the complex mean of the target-absent calibration records over their valid samples, fitted on the training split only and stored with its counts |
| Stage 2, the window | The three uplink slots of one TDD period, each placed at the offset of its allocation and averaged coherently, the template subtracted in the magnitude domain, the mean over the DM-RS symbols and one z-score. A causal stream answers once per period |
| Stage 3, the model | `cusense_cnn`: the published plain-block 1D CNN from `[1, A, K]` to a softmax map over the position grid, with a presence logit trained beside it. 2,384,933 parameters on the grid of the demonstration set |
| The artifact | `cusense-artifact/v1`: `model.json`, `template.c64`, `template.count.u4` and `weights.f32`. The template, the normalizer, the weights, the threshold and the grid geometry with its units each carry a hash |
| The input entries | `o3_fp16`, 4 by 612 by 2 at the demonstration profile, and `o2_comb`, 4 by 306 by 2. The same pipeline and the same model entry |
| The sensor | `python -m ocudu_cuda_isac.sensing.cusense follow`: a second process on the O3 ring, the artifact loaded once, one `cusense-decision/v1` line per answer of the stream, with the map, the cell and the position in metres |
| The evaluator | `src/ocudu_cuda_isac/episodes/cusense_frontend.py`: plans of version 3 for a set, the positions of the target, the oracle, the parity and the two records `cusense_eval.json` and `cusense_reference.json` |
| The gate | `make accept-s3.5`: twelve cases on the accepted Sprint 3.4 run, the accepted Sprint 5.2 build and the episode set of the Sprint 3.4 run |

**Implemented on 2026-11-22 (UTC), hardware gate open.** The unit gate (`python tests/unit/test_cusense_frontend.py -v`, 58 tests), the import rule, the driver tests on doubles of the native tool (14 tests: twelve cases and 47 checks pass, and each of 16 injected faults fails the check that names it) and the gate-runner tests pass on a development host. `make accept-s3.5` has not run on the Spark yet. The gate grades the stages, the artifact, the freeze, the presence decision at the frozen threshold and the parity of the device against the CPU, and one declared floor on the validation split. The presence counts and the position error on the held-out episodes are recorded beside the published 73 cm with `"graded": false`, and the latency comparison with the published 241 us is made by Sprint 3.6 on the native core. A rehearsal of the fit on free-space windows at the scale of the demonstration set gave a median position error of 0.91 m on held-out trajectories, against 3.04 m in the mean for the centre of the area. It is no prediction of the traced set. [Checked on a development host](docs/interfaces/CUSENSE_FRONTEND.md#checked-on-a-development-host)

### Verified configuration

| Item | Value on the accepted host |
|---|---|
| Host | DGX Spark `spark-ecf8`, aarch64, Ubuntu 24.04.5 LTS, kernel 7.0.0-1025-nvidia |
| GPU | NVIDIA GB10, compute capability 12.1, driver 580.178.04, CUDA toolkit 13.5 |
| CPU | Cortex-X925 on CPUs 5-9 and 15-19, Cortex-A725 on CPUs 0-4 and 10-14, every governor `performance` |
| Native toolchain | GCC 13.3.0, CMake 3.28.3, `nvcc` 13.5.37 |
| OCUDU | 26.04.0 at `27e84cb3da` (`nvcuda_accel_02`), Release, CUDA architecture 121, `-mcpu=neoverse-v2` |
| Adapter | `grant_fixture` `7a0e1c8ec1a0`, `grid_apply` `158644b5a177`, `replay_pusch` `d2bcc09539c3` and `pilot_ls` `c3a214c2869c` in `build/ocudu-adapter`, compiled and linked like the accepted `pusch_e2e_pipeline_test` `e49b2b81e28a` |
| Python | 3.12.3 in the repository-local `isac` venv |
| PyTorch | 2.15.0+cu132, the PyPI aarch64 wheel, compiled for sm_80 through sm_120 |
| Sionna and Sionna RT | 2.2.0 and 2.2.0, built from `ext/` |
| Mitsuba and Dr.Jit | 3.9.1 and 1.5.0 |
| Episode trace | Sionna RT on the Mitsuba variant `llvm_ad_mono_polarized`, deterministic solvers, seed 42, on the CPU |
| NumPy and SciPy | 2.5.5 and 1.18.1 |

`environments/full-constraints.txt` records the complete third-party set. The unit suite, the replay fixture and the CPU simulation probe also pass on an x86_64 Ubuntu 24.04 host without a GPU. The core checks ran there on Python 3.11, 3.12 and 3.13. The Sprint 2.1 comparison ran there against a CPU-only build of the pinned OCUDU source and wrote the same record bytes as the Spark. The Sprint 2.2 driver ran there with a stand-in for the fixture adapter and wrote the same bytes for the reference pair as the Spark. The Sprint 2.3 driver ran there against a CPU-only build of the pinned OCUDU source: the Sprint 2.2 driver traced the pair again with the real adapter, the pair and the six grant grids have the hashes the Spark recorded, the native result equals the Python reference on all 1,645,056 elements, and the 48 stored grids have the bytes the Spark stored. [Accepted run](docs/interfaces/GRID_APPLY.md#accepted-run)

### What the gates reject

The gates were tested for what they refuse, in the unit suite or in an end-to-end negative run.

- **S0.1** refuses a system interpreter, packages that leak in through `PYTHONPATH` or the system site-packages, a project install that is not editable from this checkout, and a moved or incomplete venv.
- **S0.2** refuses a source tree that is off its pinned commit or has an edited tracked file, an installed Sionna file that differs from the pinned tree, a missing LLVM or CUDA backend, and a traced channel that deviates from free space. The file comparison matters because two Sionna RT commits both report version 2.2.0.
- **S0.3** refuses malformed or ambiguous fixtures, unknown or duplicate sequence IDs, a mismatched clock, and outputs that differ on repeat.
- **S1.1** refuses another platform under the Spark profile, an `nvcc` that cannot emit the GPU architecture, a cache value that differs from the recipe (including `CUDA_ACCEL_FOUND=FALSE`), a CUDA source compiled for another architecture, a PUSCH test compiled without `ENABLE_CUDA` or the CPU target, an executable without linked device code, binaries that differ from the build receipt, and an OCUDU source that is not pristine. A missing development library stopped two real runs in the configure step.
- **S1.2** refuses, in the unit suite and in a rehearsal against a stand-in for the native test: a workload other than the Table II row, fewer than 200 CRC passes on either path, a CRC disagreement, a byte mismatch, an output guard failure, a GPU path without resident timing or the producer-owned grid on every measured frame, a missing or non-finite service time, a transcript that is truncated, repeated or of zero iterations, a run that was not pinned to CPUs 5-9,15-19 or could see another GPU, and a build that Sprint 1.1 did not accept. A GPU path that is slower than the CPU path is recorded and not refused. An uncommitted tree stopped one real run.
- **S2.1** refuses, in the unit suite: a grant with an invalid DM-RS or receive port, an allocation outside the bandwidth part, a repeated pilot, DM-RS symbols the standard does not give for the allocation, and a payload that does not match the transport block size, each before a grid is allocated. A store entry is refused when a file differs from its hash, when a stored index or reference symbol is not the grant's own, when a grid occupies an element the grant does not, and when a pilot differs from its reference symbol. Against doubles of both transmit references the integration driver fails the check that names the fault: a differing transport block size or coding, a shifted or incomplete grid, a wrong pilot or data symbol, a failed receiver CRC, and a reference that cannot run, which is a failure and never a skip. The adapter builder refuses an OCUDU build without a Sprint 1.1 receipt, a build from another source or generator, and an adapter whose sources or OCUDU build changed after its receipt. The accepted run holds one real negative: the grid built with the pinned test's DM-RS reference point fails the receiver's CRC on the offset grant.
- **S2.2** refuses, in the unit suite: a schedule on a slot that is not an uplink slot, a pattern that does not repeat in a frame and a payload rule other than the fixed block. An episode entry is refused when a byte changed, when an array is missing, incomplete or has other axes, when its opportunities, grants or axes differ from what the embedded grant yields, when a value is not finite and when the present case equals the absent one. A set is refused when an episode is outside the group of its trajectory, when a trajectory is in two groups and when the assignment changed. Against doubles of the ray tracer and the adapter the integration driver fails the check that names the fault: a channel that moves the other way, a reversed receive order, a disagreement with the solver, a failed receiver CRC, pilots of another slot and a missing Sprint 2.1 record. On the Spark the loader refused four damaged copies of the accepted pair.
- **S2.3** refuses, in the plan check and in the native core, before a grid is touched: a run without a gain, a gain that is zero, negative or not finite, two gains in one run, a gain more than 1e-6 off `1 / sqrt(reference_power)`, an override on a run that reads an entry, a transmit grid whose hash is not the bound one, an opportunity without a bound grid, a grid on another carrier and a transmit grid that is nonzero off its grant's elements. An owned grid refuses a read before it is ready, a stale generation, a second release and a consumer of another allocation. The reader refuses a stored grid that differs from its digest and names any file or field the receiver side must not see. All 35 mutations of the native core fail a unit case. With the unit cases bypassed, eleven of twelve mutated tools fail the integration driver on the development host, and the twelfth fails its unit case. Against a double of the tool the driver fails the check that names the fault: a gain that misses the noise, a gain in single precision, reversed receive planes, an unrounded grid, the grid of another slot, another noise for the second case, a leaked file or field, a stale digest and a tool that refuses nothing. On the Spark the plan check and the native tool refused seven gain rules, six binding rules and a grid on another carrier, each with status 2 before a file was written.
- **S2.4** refuses, in the plan check and in the native tool, before a processor is created or a file is written: a transport block of another size than the record gives or with another hash, a grant record that was edited after its grid was bound, a grant without a record, an opportunity in another slot than its grant or listed twice, an unknown backend, an observer on the control run and a channel plan with another hash. The dispatch core refuses the same key twice, a grid that is not ready, a stale generation, another grant record and a grid whose values changed. The reader refuses a record with an unknown field, a repeated key, decoded bytes that differ from their hash and a record that names another grid than its opportunity. The driver fails a decode whose stated outcome its own comparison does not confirm, a fallback to the host path and a host rehearsal that claims a pass. These were shown in the unit suite and against doubles of the tools, and the refusals of the native tool in the rehearsal.
- **S3.1** refuses, in the plan check and in the native tool, before a processor is created or a file is written: a tap that does not exist, a replay plan that is not the control plan or has another hash, a grant without a sensing source or with two, a source bound to another grant record or with a pilot outside the allocation, a source record or an array with another hash, a configured pilot with zero reference power, a second port and an antenna map that does not fit the receive planes. The reader refuses an observation that names another grid than its opportunity was filled with, a payload or a validity mask that differs from its hash, a record with a field outside the observation contract and an observation in a run without a tap. The gate refuses a control that Sprint 2.4 did not accept, an accepted executable with other bytes, and a tapped CUDA decode that fell back to the host path.
- **S2** as a whole is refused by the audit when a verdict is missing or is a rehearsal, when a gate did not consume the run of the one before, when the adapters were not built against one CUDA build or a tool changed between the runs, when a plan binds other files than the entry, when a decoded grid is not the stored one, when a decoded byte differs from the archived block and when a decode did not run the path it was requested on.
- **The gate runner** refuses an uncommitted tree, an empty test run, a skipped test on a simulation host, and an evidence directory that already exists.

### Open items

- **Demonstration set.** `configs/episodes/demo_set.json` is the initial configuration: 350 matched pairs, about 7 GB, a grid of 50 by 34 cells over 6.78 by 10.06 m, nine trajectories. It was judged sufficient to start on 2026-11-20 and will be tuned once the remaining components are built, so it keeps the status `proposed`. `make episode-set` traced it on the Spark on 2026-11-22 (`data/episodes/demo-set-20261122T022210Z`), and Sprint 3.4 trained on it. The accepted Sprint 3.4 run names that directory by absolute path, so it must stay where it is. The target-absent runs of a set differ by their noise only.
- **Payload.** The stored episode entries carry the transport block of the base grant at every opportunity (`payload: fixed`). Sprint 2.4 builds a transport block per opportunity (`per_opportunity`) on the slot grants and records the rule of every run. It is accepted, and no stored episode set carries such blocks yet.
- **Fixture adapter teardown.** The loopback of `grant_fixture` destroys its processor before its workers are joined and can end in a segmentation fault on a loaded host. The fix is one moved line. It changes the hash of an accepted executable, so apply it together with `make accept-s2 RERUN=1`. Until then a Sprint 2.1 or 2.2 gate that reports `the fixture adapter exited with status -11` is run again on a quiet host.
- **Fill time.** One grid of 4 by 14 by 612 elements takes 1.44 ms to fill on one core of the Spark, and a slot lasts 0.5 ms. About nine tenths of that is the readback digest and not the arithmetic: on a development host the channel application takes 123 us and the digest 1,069 us. The value is recorded and not graded. S4.1 prepares each grid ahead of its slot and leaves the digest out.
- **Next step.** `make accept-s3.5` on the Spark: the cuSense-like branch on the O3 records of the demonstration set. Then S3.6, the native core, which needs the artifact that run freezes. [Cards](docs/dev/DEVELOPMENT.md#phase-s3)
- **Sprint 3.5 on hardware.** Open until its gate runs: whether the Python sensor with a model of 2.4 million parameters keeps the pace of the receiver on a ring of eight slots, the parity of a convolutional model between the GB10 and the CPU, and what the model learns on the traced set. A dropped estimate fails the gate. The position error is recorded and not graded, and the two input entries and the linear head are compared on ten held-out pairs, which resolves little: a larger held-out group is a decision for S4.6.
- **O3 is not bit-reproducible.** The smoothing kernel of the pinned receiver filters in place. In the accepted Sprint 5.2 run 3 of 48 payloads differ between two decodes of the same grid, within the declared tolerance. Compare O3 runs by tolerance and never by digest.
- **Operating point of the presence head.** The frozen threshold follows a false-alarm target of 0.05 on forty validation negatives and lies inside the target-absent windows, although the classes of the demonstration set do not overlap. S4.6 sets the operating point of the demonstration. Until then the head alarms on about one target-absent window in twenty.
- **Set entries bind no transmit grid.** The exporter binds a grid per grant for the reference pair only. Sprint 3.4 replays the entries of a set with the grids that the pair recorded, under the same grant record (`entry_plan(bindings=...)`). A set on another grant needs its own stores first.
- **Two generations of Sprint 2 runs.** Sprint 3.2 amended two cores of Sprint 2, so `replay_pusch` and `grid_apply` have other bytes than the first accepted runs name. The Sprint 2 gates were run again on the amended tree and passed (`s2-spark-ecf8-20261121T202119Z`). The accepted Sprint 3.1 run `s31-spark-ecf8-20261121T174511Z` lies on the first generation and is the reference of the Sprint 3.2 gate, so the runs `s24-spark-ecf8-20261121T083424Z` and `s31-spark-ecf8-20261121T174511Z` must stay in `results/runs/`. A new `make accept-s3.1` takes the newest Sprint 2.4 run as its control.
- **Allocations in the staged harness.** A timed run computes no checksum, but the ledger and the processor wrapper of the staged harness still allocate and take a mutex once per decode, between a ready grid and the decoder's callback. S4.1 removes them. The observer and the ring do not allocate.
- **Checksums in the loop.** Required to change, decided 2026-11-21. An observation takes 1,471 us on the Spark and a fill 1,444 us, and about nine tenths of each is a SHA-256 digest of the whole grid. `pilot_ls` takes eight such digests per opportunity. They are native code, they are what the gates of Sprints 2.3 to 3.1 prove bytes with, and they are outside the PUSCH service times of 618 us and 265 us. No latency of the observer or of a paced run may be reported before S3.2 and S4.1 run without them. Measured on the Spark on 2026-11-21 by a diagnostic on the two host-free cores: an observation without any hash takes 9.7 us, of which 6.7 us are the estimate, and 110.1 us with the digest of the payload alone. Sprint 3.2 built the path without a hash, and its gate measured the tap in the tool, on the pinned grid and with the publication: 11.0 us in the mean and 18.7 us at most over 47 observations, against the ceiling of 150 us (run `s32-spark-ecf8-20261121T204425Z`). The allocations of the staged harness remain until S4.1, so no latency of a paced run is reported yet. [Requirement](docs/dev/DEVELOPMENT.md#added-by-the-first-observer-run-of-2026-11-21)
- **Observer placement.** The tap of Sprint 3.1 is synchronous: the estimate is complete in a host buffer before the first decode of its grid starts. Whether the tap moves to its own thread is open until S4.1 measures a paced run.
- **Plane order on the direct path.** The transmitter of the reference pair lies at broadside of the receive array, so its target-absent channel is the same on all four planes and cannot show a reversed plane order or an angle. The gate grades the plane order on the pair, where the noise differs per plane. An angle check of S3.3 or S6.3 needs a stimulus off broadside.
- **Noiseless fixtures.** The grid's storage leaves about 56 dB between a resource element's signal and its rounding error. A weak-target result needs the episode's noise above that floor, or its margin to the floor stated. The accepted reference pair states it: noise at -30 dB, the return at -42.2 dB and 13.4 dB above the RMS rounding error.
- **Grid payload.** The observation payload `grid_c64` of the memory ablation is a widened copy of a grid that the receiver stores in four bytes per value. Whether S5.5 adds the stored form as its own entry is open.
- **Timing.** The Sprint 1.2 latencies were taken with desktop applications open and are 19 to 30 percent above the paper's. Tuning is deferred until the project features are implemented. Before a latency or speedup claim, measure on a quiet host, repeat the run to bound the CPU spread and state the SINR.
- **Other hosts.** Sprints 0, 1, 2 and 3.1 are accepted on one Spark. The second Spark needs its own `make accept-s0`, `make accept-s1`, `make accept-s2.1`, `make accept-s2.2`, `make accept-s2.3`, `make accept-s2.4`, `make accept-s2` and `make accept-s3.1` runs. The GB300 needs `make accept-s0` and `bash scripts/accept_s1_1.sh --machine local` with its own CPU target. The Sprint 1.2 gate measures the Table II row, which is assigned to a GB10.
- **Evidence location.** The run directories are on the Spark under `results/runs/`, which Git ignores. They are not in this repository.
- **Hosted checks.** `.github/workflows/portable.yml` is in place. No hosted run is recorded here yet.
- **Fresh clone.** `make hydrate` has not been run end to end on a new clone. The Spark's sources were added in place at the same commits. [ext/README.md](ext/README.md) covers an interrupted fetch.
- **ZIP import.** The archive route for hosts without Git access was not exercised against the current lock.
- **Watch item.** The cuBLAS release that PyTorch 2.15.0 installs has a published issue on sm_120 and sm_121: batched matrix multiply with broadcast operands can read out of bounds. Results are unaffected and Sprint 0 does not use it. It matters for later GPU model work. The model of Sprint 3.5 uses convolutions and plain linear layers and no such product, and its unit gate checks the source for one.

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

Clone without `--recursive`. `make check` verifies the source pins, runs the unit suite and replays the four-update fixture. The tests that need NumPy are skipped on a core host, among them the reference of the features and every integration driver test. Run `python -m pip install 'numpy>=2.2.6'` first to include them.

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

### Grant fixture (Sprint 2.1 accepted)

```bash
source isac/bin/activate
python tests/unit/test_grant_fixture.py -v
make accept-s2.1

# Read an archived record back
run=$(ls -d results/runs/s21-* | tail -1)
python -m ocudu_cuda_isac.episodes.grant show "$run/grant_fixture/two-prb-offset/store" --pilot 0 --data 0
sha256sum "$run/grant_fixture/two-prb-offset/store/tb.bin"
```

The first command is the unit gate and needs only the core environment. `make accept-s2.1` is the integration gate. It builds the native fixture adapter into `build/ocudu-adapter` with the compile flags and the link line of the pinned `pusch_e2e_pipeline_test`, compares the pinned OCUDU transmit chain with Sionna NR on the three grants in `configs/grants/`, archives each record under a new directory in `results/runs/` and prints one verdict. It uses the CPU only and never rebuilds OCUDU. `make adapter-build` runs the adapter build alone. [Grant record](docs/interfaces/GRANT.md)

### Episode pair (Sprint 2.2 accepted)

```bash
source isac/bin/activate
python tests/unit/test_episode_pair.py -v
make accept-s2.2

# Read the stored pair back
run=$(ls -d results/runs/s22-* | tail -1)
python -m ocudu_cuda_isac.episodes.episode show "$run/episode_pair/reference_pair/store" --case b --opportunity 0 --rx 3
sha256sum "$run/episode_pair/reference_pair/store/b_h.npy"
```

The first command is the unit gate and needs only the core environment. `make accept-s2.2` is the integration gate. It needs the accepted Sprint 2.1 run under `results/runs/`, rebuilds the fixture adapter the way that gate did, realizes the grants of the schedule, traces the pair and the smoke set on the CPU and prints one verdict. The Spark accepted it on 2026-11-21 (UTC) in run `s22-spark-ecf8-20261121T004433Z`. `make episode-pair` and `make episode-set` trace alone, into a new directory under `data/episodes/`. [Episodes](docs/interfaces/EPISODE.md)

### Channel application (Sprint 2.3 accepted)

```bash
source isac/bin/activate
python tests/unit/test_grid_apply.py -v
make accept-s2.3

# Read one stored grid back
run=$(ls -d results/runs/s23-* | tail -1)
python -m ocudu_cuda_isac.episodes.grid_apply show "$run/grid_apply/reference_pair/native" --sequence 0 --rx 3 --symbol 2 --subcarrier 0
sha256sum "$run/grid_apply/reference_pair/native/ingress/0000.c64"
```

The first command is the unit gate. It needs the core environment, CMake and a C++17 compiler: it checks the Python reference and builds the host-free native core on its own, without OCUDU. `make accept-s2.3` is the integration gate. It needs the accepted Sprint 2.2 run under `results/runs/`, rebuilds the adapter with `grid_apply`, applies `Y = g (sum(H X) + W)` with the grid gain of the entry to all opportunities of the pair, writes each grid through the pinned resource grid writer, reads it back, compares it with a complex128 reference in Python and prints one verdict. It uses the CPU only and links no decoder. The Spark accepted it on 2026-11-21 (UTC) in run `s23-spark-ecf8-20261121T032401Z`. [Channel application](docs/interfaces/GRID_APPLY.md)

### Receiver replay (Sprint 2.4 and Sprint 2 accepted)

```bash
source isac/bin/activate
python tests/unit/test_replay_pusch.py -v
make accept-s2.4
make accept-s2

# Read one decode back and compare its bytes with the archived block
run=$(ls -d results/runs/s24-* | tail -1)
python -m ocudu_cuda_isac.episodes.replay_pusch show "$run/replay_pusch/reference_pair/run/native" --sequence 0 --backend cuda
s21=$(ls -d results/runs/s21-* | tail -1)
cmp "$run/replay_pusch/reference_pair/run/native/radio/0000.cuda.bin" "$s21/grant_fixture/demo-51prb-4rx/store/tb.bin"
```

The first command is the unit gate. It needs the core environment, CMake and a C++17 compiler: it checks the outcome rule, the payload, the plan and the reader, and builds the host-free dispatch core on its own, without OCUDU. `make accept-s2.4` is the integration gate. It needs the accepted Sprint 2.3 run and the Sprint 2.2 run that one applied under `results/runs/`, rebuilds the adapter with `replay_pusch`, decodes every grid of the stimulus on the host and on the CUDA PUSCH processor, compares the decoded bytes with the original transport block and prints one verdict. A decode that falls back to the host path fails it. `make accept-s2` then reads the four accepted runs of Sprint 2 as one chain, from the grant record to the decoded bytes, and prints `Sprint 2: ACCEPTED on this DGX Spark.` On a host without a CUDA build, `make accept-s2.4 REHEARSAL=1` grades the host path and ends REHEARSED, never ACCEPTED. [Receiver replay](docs/interfaces/REPLAY_PUSCH.md)

### Pilot least squares (Sprint 3.1 accepted)

```bash
source isac/bin/activate
python tests/unit/test_pilot_ls.py -v
make accept-s3.1

# Read one observation back: the first estimate of receive plane 0 of the first opportunity
run=$(ls -d results/runs/s31-* | tail -1)
python -m ocudu_cuda_isac.episodes.pilot_ls show "$run/pilot_ls/reference_pair/run/native" --sequence 0 --rx 0 --row 0 --pilot 0
```

The first command is the unit gate. It needs the core environment, CMake and a C++17 compiler: it checks the reference estimate, the plan and the reader, and builds the host-free observer core on its own, without OCUDU. `make accept-s3.1` is the integration gate. It needs the accepted Sprint 2.4 run under `results/runs/`, with the runs its plans name still in place, rebuilds the adapter with `pilot_ls`, decodes the plans of that run again with the observer attached, compares every estimate with the reference and every radio record with that run, and ends with `Sprint 3.1: ACCEPTED on this DGX Spark.` [Pilot least squares](docs/interfaces/PILOT_LS.md)

### Observation ring (Sprint 3.2 accepted)

```bash
source isac/bin/activate
python tests/unit/test_snapshot_ring.py -v
make accept-s3.2

# Read one timed record back, and the account of the run it came from
run=$(ls -d results/runs/s32-* | tail -1)
python -m ocudu_cuda_isac.sensing.obs show "$run/snapshot_ring/timed/run/consumer-0" --sequence 0
python -m ocudu_cuda_isac.episodes.snapshot_ring show "$run/snapshot_ring/timed/run/native"

# The ring alone, without a receiver: a producer of known records and a consumer that keeps them
build/ocudu-adapter/obs_ring/obs_feed --socket /tmp/o2.sock --records 2000 --period-us 500 &
build/ocudu-adapter/obs_ring/obs_tail --socket /tmp/o2.sock --wait spin
```

The first command is the unit gate. It needs the core environment, CMake and a C++17 compiler: it checks the record and the plan, builds the ring, the producer glue and the cores it joins on their own, without OCUDU, runs their cases and lets both readers read one native producer. `make accept-s3.2` is the integration gate. It needs the accepted Sprint 3.1 run under `results/runs/`, with the Sprint 2.4 run its plan lies in still in place, rebuilds the adapter with the ring, publishes the reference pair to a consumer in a second process in the modes `audit` and `timed`, compares every decode with the accepted Sprint 2.4 run and every estimate with the accepted Sprint 3.1 run, and ends with `Sprint 3.2: ACCEPTED on this DGX Spark.` The last two commands run the ring without a receiver. `obs_feed` prints what the ring counted and what one publication cost. [Observations on the ring](docs/interfaces/SNAPSHOT_RING.md)

### Radar cube and features (Sprint 3.3 accepted)

```bash
source isac/bin/activate
python tests/unit/test_pilot_features.py -v
python tests/unit/test_core_imports.py -v
make accept-s3.3

# Read the target-present window of the live sensor back
run=$(ls -d results/runs/s33-* | tail -1)
python -m ocudu_cuda_isac.sensing.features show "$run/pilot_features/live/run/sensor" --window 1

# The features of any consumer archive, by hand
python -m ocudu_cuda_isac.sensing.features spec --output build/features-001/spec.json
mkdir build/features-001/out
s32=$(ls -d results/runs/s32-* | tail -1)
python -m ocudu_cuda_isac.sensing.features replay --archive "$s32/snapshot_ring/timed/run/consumer-0" \
    --spec build/features-001/spec.json --output-dir build/features-001/out
```

The first two commands are the unit gate and the import rule. They need NumPy, and CMake with a C++17 compiler for the case that attaches the reference sensor to a native producer of the ring. `make accept-s3.3` is the integration gate. It needs the accepted Sprint 3.2 run under `results/runs/`, with the runs its plans name still in place, rebuilds the adapter, which must have the bytes of that run, computes the features of the reference pair from the recorded ring, from the files of the audited observer and from a live ring with the reference sensor in a second process, compares the three, checks the return against the geometry of the episode, and ends with `Sprint 3.3: ACCEPTED on this DGX Spark.` `S32_RUN=` names another control, and `REHEARSAL=1` rehearses on a host without CUDA. [Cube, view, vector and gate](docs/interfaces/PILOT_FEATURES.md)

### Presence head (Sprint 3.4 accepted)

```bash
source isac/bin/activate
python tests/unit/test_presence_head.py -v
python tests/unit/test_core_imports.py -v
make episode-set            # once: the demonstration set, about 7 GB under data/episodes (done on spark-ecf8)
make accept-s3.4

# Read the frozen head and a held-out decision back
run=$(ls -d results/runs/s34-* | tail -1)/presence_head
python -m ocudu_cuda_isac.sensing.presence inspect "$run/train/artifact"
python -m ocudu_cuda_isac.sensing.presence show "$run/live/held-out-00/sensor" --window 0
sha256sum "$run/train/artifact/weights.f32"
```

The first two commands are the unit gate and the import rule. They need NumPy and PyTorch, which the full profile installs (`make env-full`), and CMake with a C++17 compiler for the case that attaches the sensor to a native producer of the ring. `make accept-s3.4` is the integration gate. It needs the accepted Sprint 3.3 run under `results/runs/`, with the runs its plans name still in place, and a traced episode set under `data/episodes/`. It rebuilds the adapter, which must have the bytes of that run, replays the training and the validation entries through the tapped receiver, fits and freezes the head, replays the held-out entries with the presence sensor in a second process on the CUDA device, scores the same records offline, and ends with `Sprint 3.4: ACCEPTED on this DGX Spark.` `S33_RUN=` and `EPISODE_SET=` name another control and another set. [Presence head](docs/interfaces/PRESENCE_HEAD.md)

### The receiver's estimate (Sprint 5.2 accepted)

```bash
python tests/unit/test_native_estimate.py -v
python scripts/receiver_patch.py check
make accept-s5.2

run=$(ls -d results/runs/s52-* | tail -1)
cat "$run/summary.txt"
python -m ocudu_cuda_isac.episodes.native_estimate show "$run/native_estimate/noisy/run/native"
```

The first command is the unit gate. It needs CMake and a C++17 compiler and no OCUDU build. The second checks the patch series against its files and the lock. The gate needs the accepted Sprint 1.1 build behind `isac/bin` and the accepted Sprint 3.1 run, and runs only on the DGX Spark: its first run builds the patched receiver, and the accepted run took 18 minutes with that build and the unit tests. [The receiver's own estimate](docs/interfaces/NATIVE_ESTIMATE.md)

### cuSense front end and location model (Sprint 3.5 implemented)

```bash
source isac/bin/activate
python tests/unit/test_cusense_frontend.py -v
python tests/unit/test_core_imports.py -v
make accept-s3.5

# Read the frozen model, a held-out decision and the two records back
run=$(ls -d results/runs/s35-* | tail -1)/cusense_frontend
python -m ocudu_cuda_isac.sensing.cusense inspect "$run/train/o3/artifact"
python -m ocudu_cuda_isac.sensing.cusense show "$run/live/held-out-00/sensor" --window 0
sha256sum "$run/train/o3/artifact/weights.f32" "$run/train/o3/artifact/template.c64"
python -m json.tool "$run/cusense_reference.json" | head -60
```

The first two commands are the unit gate and the import rule. They need NumPy and PyTorch (`make env-full`), and CMake with a C++17 compiler for the case that attaches the sensor to a native producer of the ring. `make accept-s3.5` is the integration gate and runs only on the DGX Spark. It needs the accepted Sprint 3.4 run and the accepted Sprint 5.2 run under `results/runs/`, the patched receiver in `build/ocudu-spark-o3`, and the episode set that the Sprint 3.4 run names, still where it was. It rebuilds the adapter with the O3 route, which must have the bytes of the Sprint 5.2 run, and never rebuilds the receiver or traces an episode. It replays the training and the validation entries with both stages on their rings, fits and freezes one artifact per input entry, and only then replays the held-out entries with the sensor on the O3 ring. A run keeps about 1.5 GB. [cuSense front end and location model](docs/interfaces/CUSENSE_FRONTEND.md)

## Repository map

| Directory | Purpose |
|---|---|
| [docs/dev](docs/dev/DEVELOPMENT.md) | Sprint cards, host setup, paper baseline and reproduction |
| [docs](docs/README.md) | System design, integration boundaries, interfaces, measurement, evidence and references |
| [ext](ext/README.md) | CUDA-OCUDU, Sionna and Sionna RT as pinned Git submodules, two NVIDIA Aerial repositories pinned as read-only reference sources, and the lock that owns their paths |
| [environments](environments/README.md) | Host environment notes and the validated third-party constraints |
| [src/ocudu_cuda_isac](src/ocudu_cuda_isac/) | Project-owned portable replay and accounting, the grant record, the grant schedule, the channel episodes, the episode set, the replay plan with the reference of the channel application, the receiver replay plan with the payload rules and the reader of the radio records, and the plan and the reader of the tapped replay (`episodes/`), the sensing-source registry (`sensing/sources/`), the reference of the pilot least squares (`sensing/pilot_ls.py`), the reference reader of the ring (`sensing/obs.py`), the reference of the radar cube, the delay-Doppler view and the causal features (`sensing/features.py`) with its evaluator side (`episodes/pilot_features.py`), the reference of the presence head (`sensing/presence.py`) with its evaluator side (`episodes/presence_head.py`), and the reference of the cuSense-like branch (`sensing/cusense.py`, `sensing/models/cusense_cnn.py`) with its evaluator side (`episodes/cusense_frontend.py`) |
| [native/ocudu_adapter](native/ocudu_adapter/README.md) and [patches/ocudu](patches/ocudu/README.md) | Native adapter built against the accepted OCUDU build (the fixture adapter of S2.1, the channel application of S2.3, the receiver dispatch of S2.4, the observer of S3.1 and the producer glue of the observation ring of S3.2; later adapters arrive with their tickets) and narrow upstream patches |
| [configs](configs/) and [experiments](experiments/README.md) | Working smoke inputs, grant records, integration configurations, advisor profile and publication matrix |
| [data](data/README.md) and [results](results/README.md) | Episode provenance and run evidence |
| [scripts](scripts/), [tests](tests/) and [CI](.github/workflows/portable.yml) | Environment, pin and acceptance gates, portable regressions and the hosted CPU checks |
| [native/obs_ring](native/obs_ring/README.md) | The observation record `obs/v1` and the ring `obs-ring/v1` between the receiver and a sensor on one host (S3.2): C++17 on the CPU with no OCUDU include, a producer, a consumer, the consumer tool `obs_tail` and the producer of known records `obs_feed`. [Observation contract](docs/interfaces/OBSERVATION.md) |
| [native/sense_core](native/sense_core/README.md) | Specified, not yet built: the native sensing core that every timed sensing stage runs in (S3.6, S6.12, S6.13). [Code levels](docs/architecture/RUNTIME.md) |

M1 shares one Spark under MPS, M2 uses two Sparks with a measured CX7 observation path, and M3 uses two GPU instances on one GB300 with a full-GPU control. See [deployments](docs/architecture/DEPLOYMENTS.md), the [advisor demonstration](docs/demo/ADVISOR_DEMO.md) and the [upstream reuse ledger](docs/integration/UPSTREAM.md).
