# Development plan for CUDA-OCUDU ISAC

**First advisor milestone — S4.** Run the real CUDA-OCUDU PUSCH receiver on one DGX Spark. A second MPS client consumes pilot estimates derived from the same received grid and publishes one learned moving-target presence decision per eligible observation. Show verified payload bytes, the return in delay/Doppler, and the effect of inference on receiver timing. A native harness with legal grants is sufficient for this first showing. S5 carries the same experiment into the actual DU scheduling path.

**Publication milestone — S9.** Deliver a reproducible comparison of receiver-aware sensing under matched communication load. It needs an observed internal receiver estimate, a calibrated false-alarm operating point, causal baselines, three deployments, and measured deadlines, drops, goodput and energy. Preserve the first advisor configuration as a regression throughout.

## How to use this file

A sprint closes when a second developer can run its acceptance test and locate the raw evidence. Save the parent commit, submodule SHAs, exact invocation, config/model/trace hashes and unedited logs with every gate. Each hardware gate below is prospective until the lab attaches a run. Quantitative budgets below are **proposed starting values** for an engineering demo. Freeze them before the run, record changes as new configurations and decide publication SLOs independently.

| Phase | Sprint | Owner role | Dependency | Deliverable and exit evidence |
|---|---|---|---|---|
| Bring-up | S0 | Integration lead | Fresh parent checkout | Gitlinks match lock, unit suite and four-opportunity smoke pass |
| Bring-up | S1 | PHY lead | S0 + Spark | Real high-SINR CPU/GPU decodes agree on CRC and output bytes |
| First vertical slice | S2 | Scene/replay lead + PHY lead | S0; native integration after S1 | Matched present/absent CFRs reach the upstream receiver; O1 bytes and actual decode verified |
| First vertical slice | S3 | PHY + sensing leads | S2; train/test split can start earlier | Owned O2 records drive a frozen batch-one presence head; overflow leaves DSP running |
| Advisor demo | S4 | Integration + measurement leads | S1–S3 | Two real MPS clients, paired radio/sensor outcomes, dashboard or terminal view, replayable evidence |
| Operational receiver | S5 | PHY lead | S4 | Same contract under DU-scheduled grants; validated O3 and sustained fail-safe operation |
| Sensing study | S6 | Signal processing + ML leads | S2, S3, S5 for O3 | Detector, angle, training-domain and architecture ablations on locked splits |
| Network study | S7 | Transport lead | S4; S5 for research runs | M2 over CX7 with instrumented host staging, clock uncertainty and independent core flows |
| Isolation study | S8 | Platform lead | S4; S5 for research runs | M3 distinct hardware GIs with matched same-GB300 baselines |
| Publication | S9 | Evaluation lead + independent reviewer | S5–S8 | Feasibility frontiers, confidence intervals, artifacts and independently reproduced tables |

S1 and offline scene preparation can start in parallel. S7 and S8 can proceed independently after the M1 contract is stable. Reserve a Spark, its build log and a small deterministic fixture for the critical path. M2, GB300 MIG provisioning, OTA, protocol registration and a large dataset are outside the S4 entry criteria.

## What is implemented in the starting tree

- `make pins` checks real parent gitlinks and clone URLs. `make hydrate` materializes the three pinned worktrees. Sionna RT and Sionna PHY are separate direct dependencies for offline work.
- `make test` runs the portable tests. `make smoke` writes known receiver ingress, offline truth and synthetic timing records. Expected outcome is four scheduled opportunities with one on time, one late, one dropped and one missing. Read [the replay contract](../interfaces/REPLAY.md) before modifying a field.
- `scripts/verify_pusch_log.py` grades a native test transcript. [SPARK.md](../demo/SPARK.md) gives the upstream build, workload and parser invocation. A parsed example in a unit test is a parser regression. Native performance evidence requires an actual hardware run.
- The optional [offline RT exporter](../interfaces/RT_EXPORT.md) builds a small bistatic pair from the pinned solvers. Its tensor order and checksum writer have portable NumPy tests. The four sample times and twelve frequency offsets form a diagnostic grid. S2 adds grant-aligned dimensions and a trace from a provisioned RT host.
- Production grid injection, estimator export, ring ownership across processes, trained sensor weights and the hardware runs are sprint deliverables. Proposed future module names below identify implementation owners, not existing launch commands.

## Reuse and ownership

| Reuse or build | Location and engineering rule |
|---|---|
| Reuse CU/DU, legal grants/PUSCH fixtures, DMRS generation, CPU/CUDA backends, CRC and HARQ behavior | `deps/ocudu`. Identify exact factory, test and producing symbol before adding a hook. Reuse allocators, callbacks and owned-grid interfaces. Keep an unmodified control build. |
| Reuse target/background solvers, materials, Doppler, channel/CFR evaluation | `deps/sionna-rt`. The project owns scene configuration, metadata capture and replay handoff. For known test cases let upstream compute the channel. |
| Reuse OFDM/NR/ISAC utilities, trusted array/tensor operators and model primitives | `deps/sionna` and declared third-party libraries. Use documented APIs and reference vectors. Use established FFT, linear algebra, eigensolver and PyTorch layers for the publication implementations. |
| Compare cuSense data and open Aerial dApp code | Consult [the reuse ledger](../integration/UPSTREAM.md). The public framework and PUSCH channel-estimate dataset are transferable references. Verify weight availability, license, input shape and task against the pinned upstream before using a model. The demo head is trained for this repository’s presence endpoint. |
| Build the missing seams | Host contract in `src/ocudu_cuda_isac/`. Hot-path replay, producer completion, snapshots and host transport in `native/ocudu_adapter/`. Small versioned patches in `patches/ocudu/`. Add sensing orchestration, evidence join and evaluation only where upstream lacks the experiment-specific interface. |

Every new source module should document the upstream candidate inspected, the missing behavior, its input/output ownership, and the test that demonstrates it. First try a configuration, adapter or existing primitive. A request for a new receiver, channel solver or full dApp platform requires evidence that the existing one cannot supply the needed boundary.

## S0 — Bootstrap and portable contracts

**Objective.** Give every developer a reliable first run and a small reference for native adapter debugging. Establish the distinction between simulator truth (O0), received grid (O1), pilot LS (O2) and receiver-processed estimates (O3).

**Code tasks.** Clone or extract with Git metadata. Run `make check` before downloading large dependencies. Inspect `deps/lock.json`, then run `make hydrate` when network access is available. Record the local Git/pip/CMake versions. Preserve the O1 axis order, units and complex representation from [REPLAY.md](../interfaces/REPLAY.md). The host package deliberately uses tiny inputs so that a developer can calculate one element by hand.

**Tests and acceptance.**

| ID | Test and failure it detects | Pass evidence |
|---|---|---|
| U0.1 | Known complex X, H, W with two transmit streams and multiple receive channels; an axis transposition changes Y | Every element agrees with the hand-computed fixture; malformed dimensions fail before egress |
| U0.2 | Reorder events, duplicate a key, omit a terminal event, or mark a drop | Join by the opaque opportunity key; reject invalid duplicates, keep late/dropped/missing denominators |
| I0.1 | Run from a clean checkout without hydrating source trees | Host suite passes, smoke produces the three files, repeated identical fixture outputs match |
| I0.2 | Inspect parent gitlinks and bootstrap source trees | Offline pin check passes. With network access, checked-out HEADs match all three locks and no dependent install silently selects a different RT revision |

A platform without the GPU can close I0.1. A failed dependency fetch leaves I0.2 explicitly open. Do not advance a different CUDA branch under the old pin.

## S1 — Reproduce the DeepSig receiver on Spark

**Objective.** Prove that the selected revision executes the CUDA PUSCH path and returns a known transport block. Build this first, before debugging ray tracing or an AI model. The source note and the paper describe different revisions. The paper's speedups are workload-specific references. [Source audit](../integration/OCUDU.md) · [Pennybacker et al.](https://arxiv.org/abs/2608.04338v1)

**Code tasks.** From [SPARK.md](../demo/SPARK.md), configure the pinned release build with CUDA architecture 121 and the branch's Spark CPU target after confirming local device/toolchain compatibility. Build the targeted receiver test and benchmark. Record CUDA/driver, selected CPU set, power policy, clock state and process arguments. Start with a supported narrow one-layer high-SNR workload. Capture a successful CPU/reference arm and a CUDA arm with the same bits and input realization. Keep forced-MAC-ACK or dummy-RU loads in separately named runs.

**Unit and integration.** U1.1 grades the transcript: missing counters, zero iterations, byte mismatches, guard corruption and explicit failure markers invalidate the run even when the upstream executable returns zero. I1.1 requires a positive count of launched/completed CUDA work or other backend-specific evidence plus successful true CRC and exact agreement between the decoded CPU and GPU payloads on the high-SNR case. The upstream counter compares these two outputs. S2 adds an independent check against the original TX bytes. I1.2 records measured completed-work timings for the declared PRB/layer/antenna case and an identical CPU reference. Exclude startup by a declared warm-up. Keep small cases that run faster on CPU in the record.

**Close with** raw stdout, parser result, CMake configuration, pin hash, actual command, CPU/GPU selectors, true CRC and payload-byte counts. Do not promote the paper's mean, 10% BLER sensitivity or dummy-sector load to a local result. A fallback backend, unbuilt CUDA kernels, or zero actual decode attempts blocks this gate.

## S2 — Produce a replay episode and execute its received grid

**Objective.** Replace a known-good receiver stimulus with a *known* channel realization while preserving legal grant, scattered-symbol ordering and decoder context. The most informative first failure is a broken noiseless identity case.

**Code tasks, in order.**

1. Extract the upstream transmitter/reference grid, pristine transport block, actual DMRS allocation and port mapping from the passing S1 harness. The stock non-transform-precoded test uses a PDSCH-side builder for its compatible grid. For claimed uplink replay, validate against an independent PUSCH transmit reference (e.g., Sionna NR), covering scrambling and pilot REs as well as original payload bits. Freeze numerology, cell/UE, slot/symbol times, TX/RX counts and dtype. Compare the CPU and CUDA paths using these identical inputs.
2. Generate two offline cases. Start with a direct path (or identity CFR for adapter testing) then a moving constant-RCS return plus the same scene background. Start with a fixed Tx and four calibrated RX channels. Remove the target and recompute the background for the physically absent case. Store absolute delay and gain, frequencies, geometry, seeds, software/scene hashes and array conventions. For motion across visibility changes, retrace geometry and log the update cadence.
3. Preload H and X. For each RX, symbol and occupied frequency, compute the sum over TX streams of H times X and add W once. Evaluate at the actual symbol times. The time-indexed CFR already contains Doppler. Start with an owned host grid and use the existing writer/receiver dispatch. Record the real copies and producer-completion point before exploring a faster device handoff.
4. Inject Y at the upstream receiver grid boundary and compare it against a frozen-Y readback. Freeze the fixture before introducing array acceleration, mixed precision or a different allocator. Keep the oracle scene and path decomposition in an offline process/store.

**Unit gates.** U2.1 requires elementwise parity on the smoke fixture and an analytic phase rotation across at least two RX channels and two times. Proposed native complex64 check is `abs(error) <= 1e-6 + 1e-5 * abs(reference)` for unit-scale fixtures. Use amplitude-aware relative or power error for physically weak echoes. For a physically resolvable spectral test use sufficient occupied bandwidth and slow-time duration; the tiny RT exporter fixture only checks array and phase contracts. U2.2 checks one known delay and Doppler against occupied bandwidth and actual coherent observation span (within one resolvable bin). U2.3 rejects a swapped antenna order, incompatible mask, bad checksum and incompatible grant. A changed X requires remapping H onto its allocation.

**Integration gates.** I2.1 requires exact decoded bytes and true high-SNR CRC through the executed CPU and GPU receiver for the identity/direct-path case. I2.2 feeds both physically matched episodes through the same input seam. Log assigned, complete and released grid IDs for every admitted opportunity. Rerunning with the same record and seed reproduces Y within the declared precision. Save a small baked vector, a binary episode manifest and the actual decode evidence. Compression, random path updates and learned sensing remain disabled during fault localization.

## S3 — Add the smallest useful ISAC observer

**Objective.** Give a second process a causal sensing observation and measurable output without holding up DSP. For the first cut select O2 pilot LS derived from the same Y seen by the receiver. A verified O3 can replace it after explicit provenance and parity checks. The upstream resident estimator's host-facing stub can return zeros. Do not use it as measured CE.

**Code tasks.** Use the applicable upstream pilot mapping/reference estimator on the supported single-port grant. Explicitly record whether the estimate is observer-derived or produced by the native receiver. Export an immutable observation with epoch/sequence, pilot indices, symbol times, shape/strides, dtype, receiver mode and calibration. Use bounded preallocated host storage first. Define ownership in order: producer writes, completion marks ready, consumer owns a retained record, consumer completion returns credit. Missing credit drops the observer item immediately. A restart starts a new epoch and clears stale state. Never expose writable receiver buffers to the sensor.

On the sensing side, use a simple receiver-derived feature view (for example demeaned pilot magnitudes and phase-consistent delay/Doppler power over a causal history). Account for pilot holes. Use library FFT/tensor operators and validate against a single-path case. Train one small presence head with existing PyTorch primitives on offline receiver-derived features. A linear head on a compact power feature vector is adequate for the first demo. Record actual architecture, train/validation/test scene split, normalizer, fixed threshold and artifact hash. Load weights once on the second CUDA process and run batch one. The live diagnostic spectrum and the learned presence score are different outputs.

**Unit gates.** U3.1 checks estimated known nonzero channel, pilot locations, missing pilots, calibration permutation and finite values. U3.2 fills the ring, delays the consumer, reuses the producer buffer and reads the retained record. Its bytes must remain valid until release. U3.3 rejects future samples and label/scene-name features and resets finite history at episode boundaries. U3.4 freezes training-only preprocessing and validates a saved model's output on a known feature vector after reload.

**Integration gates.** I3.1 joins O2 to the exact S2 input and CRC by episode, slot and absolute opportunity. Repeat with the observer disabled and confirm identical payloads for the selected deterministic decode fixtures. I3.2 deliberately stalls or kills the consumer: the radio loop continues, samples do not accumulate indefinitely, and every skipped observation is counted once. Close with nonzero pilot evidence, bounded ring test, split manifest, model checksum and both success and overload traces.

## S4 — First integrated advisor demo

**Entry.** S1 has real GB10 decode evidence; S2 has replay identity parity; S3 has a task-specific frozen head and verified observer ownership. Start the DSP and AI processes as separate clients of the same MPS server. Record MPS attachment for both and a time-correlated trace showing both serviced over the same interval. Mere device enumeration is insufficient. Use installed MPS controls on GB10 and record the active-thread setting. Spark has no MIG. [NVIDIA MPS guide](https://docs.nvidia.com/deploy/mps/latest/when-to-use-mps.html) · [cuSense v3](https://arxiv.org/abs/2512.06493v3)

**Reference profile.** The [machine-readable demo profile](../../experiments/demo/profile.json) records these proposed settings and gates. Candidate: one cell, one UE, one layer, four RX, 20 MHz/51 PRB at 30 kHz SCS. Use branch-supported pilot configurations. Let the first demo sample one eligible UL opportunity per 5 ms TDD period and use eight causal observations (about 35 ms from the first to the last observation start). Record actual sample boundaries. A proposed first sensing deadline is 50 ms from the first contributing sample. If the actual TDD/DMRS history cannot meet that budget, choose and freeze a physically achievable deadline in the manifest before benchmarking. The radio processing allowance is separate; derive it from the executed receiver's scheduling contract or explicitly configure the paced harness budget.

**Run order.** Warm the model and replay buffers. Acquire matched DSP-only, AI-only from buffered real receiver observations, and simultaneous MPS runs. Keep the same grants, precision, data and affinity. Store at least three paired 60-second repetitions for the pre-ablation demo; randomize or alternate order to expose thermals. Run the overload/restart diagnostic separately. Show two held-out scenes live, and retain the scorecard on the full predeclared demo set.

| Gate | Suggested minimum for the first showing | Evidence and failure response |
|---|---|---|
| I4.1 Actual integration | Both CUDA clients active, correct native PUSCH backend, repeatable received-grid hashes, at least one verified high-SNR TB and no payload mismatches among CRC-passing known blocks | Include process/client IDs, producer events, backend flags, decoded bytes, and a score from that same run. If a consumer uses saved observations from an earlier run, classify it as AI-only. |
| I4.2 Sensing function | On 10 held-out present and 10 held-out absent episodes, at least 8/10 present produce a timely positive in a prospectively chosen decision window and at most 2/10 absent trigger a false alert. Use one scored window per episode with identical selection rules and frozen threshold. | Report all counts plus each result; no claimed population accuracy from this small fixture. Inspect echoes, normalization and split leakage before tuning on new training data. |
| I4.3 Basic performance | Account for every scheduled radio job and sensing window. Provisional demo target: at least 99% of radio jobs complete inside the declared radio allowance in the concurrent run, alongside the sensing criterion above. | Print p50/p95 and observed p99 with counts, DSP-only comparison, miss/drop and copy counts, completed goodput and run time. A budget miss is a useful live diagnostic; classify the gate accordingly. |
| I4.4 Live explanation | Switch matched present/absent episodes and replay the same evidence bundle offline | Show actual CRC/bytes and score, coarse spectral view and queue depth. Show that an intentionally slow observer drops work and the receiver keeps advancing. |

**Close with** a single tagged manifest, all raw counters and a short replayable presentation in [ADVISOR_DEMO.md](../demo/ADVISOR_DEMO.md). State whether the scheduler is scripted and which 5GC services had live traffic. Clearly label this first scope as a receiver-plus-sensor integration on one Spark. A 5GC process that merely runs on the host contributes no evidence about registration or N3 goodput.

## S5 — Harden the DU path and establish O3

**Code tasks.** Connect the virtual-radio grid producer and reader to the actual DU scheduling path. Preserve SFN/slot wraparound by joining on absolute sequence and epoch. Bind each accepted grant to its buffer, receiver configuration and completion event. Extend the actual estimator producer, or a proven native output bearing its results, with a read-only bounded export before workspace reuse. Audit uplink submission order, asynchronous completion and fallback cases in [OCUDU.md](../integration/OCUDU.md). A generic PHY tap triggered after PUSCH submission is a notification point; it needs explicit ordering and retained ownership. Test both resident and fallback cases separately.

**Tests.** U5.1 checks nonzero O3, expected shape/scale, physical phase behavior and agreement with a matched configured reference under noiseless and noisy scenes. U5.2 tests grid reuse, invalid generations, reordered callbacks, SFN wrap, full queues, cancellation and restart. I5.1 runs scheduler-variable allocations with a trace of actual offered TBs. Check late allocation handling, legitimate failing CRC and HARQ/fallback evidence if exercised. I5.2 runs a sustained load for at least 30 minutes at a declared supported case; compare tap-disabled, capture-only and full observer modes. Verify bounded memory, counted stale/lost records, tail latency, drops, throttling, continued scheduling and unchanged known decodes. If synthetic testmode ACK is enabled, tag it as load-only and prohibit its use for true BLER/goodput.

**Exit.** Operational path and stage provenance independently verified, repeated initialization/shutdown without data contamination, all opportunities reconciled, and a regression trace for each tested backend. Keep I4's integrated harness as the quick regression. A failing cross-boundary timing or O3 test leaves the corresponding publication comparison open.

## S6 — Receiver and sensing ablations

**Code tasks.** First implement reusable preprocessing wrappers around existing Sionna and tensor-library operations. Add mask-aware delay–Doppler, calibrated detection, and angle estimation using the measured array manifold. Test background subtraction, two-pulse moving-target indication, covariance whitening, and CA/OS-CFAR as defined ablations. For MUSIC, declare snapshot count, covariance regularization and model order. For learning, share causal finite-history loaders and task heads between the cuSense-inspired CNN, GRU and compact Transformer. Equalize model selection budgets, compare at fixed precision and actual runtime batch, and track weights, compute, copies and energy. Compare optional receiver metadata and learned/shrinkage covariance against a fixed MUSIC backend independently of the detection head.

**Tests.** U6.1 uses single-target analytic scenes and target absence for signed delay/Doppler, orientation, ghost rejection and broadside/mirror ambiguity. U6.2 enforces scene/session splits before windowing, no overlapping test histories, and training-only normalization and false-alarm tuning. U6.3 alters future slots and oracle labels while keeping causal input fixed: the current prediction must be unchanged. I6.1 runs the same frozen H/X/W across supported O1/O2/O3 training and evaluation domains with O0 offline. Record detector operating point and angle eligibility for every stage. I6.2 replays optional public captures under their real masks/cadence and a task compatible with their labels. Keep synthetic and external evidence separately identifiable.

**Exit.** Exact train/validation/test manifests, total tuning trials, per-method runtimes, false alarms, misses, errors on detections, and timely detection probability with uncertainty are downloadable and regenerable. Include negative results, failed targets at the declared deadline and data gaps. CFAR or a Transformer alone is a baseline, not evidence of cross-receiver generalization.

## S7 — Two-Spark transport experiment

**Code tasks.** Put CU/DU, replay and CUDA DSP on Spark 2; place the ISAC agent and CPU 5GC on Spark 1. Use a verified CX7 DAC targeting 100 Gb/s and record physical negotiated speed, MTU, interface(s) and achieved one-way application payload rate. Implement the portable socket transfer before a RoCE verbs backend. Use registered pinned host buffers and explicit credits. Distinguish full-payload completion and GPU-readable input on receive. Spark's documented verbs fallback uses host memory registered with the NIC. [Spark porting guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/latest/porting/cuda.html)

**Tests.** U7.1 checks schema, endian/layout, length/checksum, partial frame and reconnect epoch. U7.2 injects missing, reordered and stale records and delayed completion, verifies bounded rings and identical semantics to M1. I7.1 replays the same trace over M1 local bridge, M2 sockets and M2 host-buffer RDMA at equal actual payload and offered cadence. Compare instrumented transport-only and integrated runs. I7.2 maps PTP/PHC to both host clocks, measures offset/drift residual during the run, and bounds one-way uncertainty. If clocks fail, measure and label result-return RTT on one clock. I7.3 tests N2 CU-CP–AMF, N3 CU-UP–UPF and independent ISAC exports as distinct flows. For meaningful core measurements require registration, session establishment, payload through the UPF and counters at both ends. Otherwise report core state/background load explicitly.

**Exit.** Joined payload hashes and completeness, measured copy/queue/transport time, line rate versus achieved payload, both nodes' energy, and joint deadline outcomes. Retain the diagnostic with core fixed on Spark 1 and AI moved between hosts to isolate the effect of AI placement. An emulated 100G bandwidth cap on a faster link is a distinct configuration.

## S8 — GB300 hardware isolation

**Code tasks.** Place the full architecture on the MSI XpertStation WS300. Use different GPU instances (GIs), each with its own compute instance. Verify the installed driver, local MIG profile and allowable placement. Candidate allocation: DSP on `4g.126gb`, AI on `3g.126gb`. Record both MIG identifiers and verify that each worker sees the intended instance. Use a host-buffer bridge across process/GI boundaries for the first run. Any optional peer-copy path needs its own supported process/ownership mechanism and measured transfers; ordinary CUDA IPC across GIs is unsupported. See [DEPLOYMENTS.md](../architecture/DEPLOYMENTS.md).

**Tests.** U8.1 verifies the launcher rejects accidental assignment of both jobs to one GI and mismatched profile or source IDs. U8.2 validates bridge lifetime and buffer pressure under unequal rates. I8.1 runs isolated DSP and isolated sensing on each actual profile, both together, and full-GPU MPS on the same GB300. Repeat M1/M2 with the same tensor representation before attributing changes to MIG. I8.2 measures host/NIC/power/thermal interference while compute partitions are fixed. Keep core placement and wall-power measurement boundaries consistent.

**Exit.** Comparing real measured workloads yields the supported load at each joint SLO, per-profile slowdown, tails and energy per timely correct decision. State which changes derive from partition size, transfer path and underlying hardware by referring to the matched controls. If device profiles or software prerequisites differ, retain the actual evidence and repeat with an explicitly versioned supported layout.

## S9 — Publication and independent reproduction

**Code tasks.** Freeze the primary hypotheses and split/threshold settings before collecting held-out results. Generate the outcome tables from machine-readable events. Construct paired comparisons over matched episodes, load traces and seeds; include an honest failed or infeasible region. Log acquisition latency separately from post-window compute, overlapping GPU work, and end-to-end publication. Keep both machines' energy for M2 and matched DSP/core baselines for every deployment. Restrict empirical statements to the executed radio, sensing and transport paths.

**Tests and acceptance.** U9.1 recomputes metric denominators with deliberate drops, failed CRC, absent targets and stale or late positives. U9.2 verifies units, percentile sample counts, clock domains, truth isolation and duplicate keys. I9.1 independently reconstructs one raw-to-figure outcome per major comparison and verifies hashes, tolerances and diagnostic counts. I9.2 repeats a predeclared subset on a clean software environment and compares within uncertainty. Require a hardware/software manifest, exact commands, machine-readable outputs, figures with denominators and confidence intervals, source/weight/data availability records, and a mapping from each claim to an archived run. Use [the measurement definitions](../research/EVALUATION.md), [experiment matrix](../../experiments/conference/MATRIX.md) and [reproduction checklist](REPRODUCE.md).

## Triage and handoff

| Observed symptom | First experiment | Decision |
|---|---|---|
| Native “pass” with bad bytes or zero checks | Read independent stdout grade, compare high-SNR known TB and CPU reference | Hold S1 until backend selection and payload correctness are both demonstrated |
| Sensing succeeds on absent scenes | Disable truth joins in runtime, remove target mesh and rerun background, retest on a disjoint session | Separate model leakage, occlusion cues and false-alarm calibration |
| Zero CE while CRC passes | Locate the producer, inspect host-stub use, retain/await the real buffer and inspect expected nonzero pilots | Keep the output labeled O2 until O3 equivalence is proven |
| DSP misses grow with AI | Hold payload/arrival trace fixed; compare observer disabled, copy-only and full inference; inspect SM and memory service | Choose whether queue, copy or model is responsible before changing deadlines |
| M2 apparent one-way gain is smaller than clock error | Recheck offsets and compare complete result-return RTT | Report a bound or indistinguishable ranking |
| GB300 loses on a small slice | Repeat the isolated workload on that profile, compare same-host full-GPU MPS and measure bridge | Quantify partition cost and supported load before scaling conclusions |

Each sprint handoff identifies the precise command another engineer can run, the expected counters or bytes, the principal failure mode, and the next unblocked test. The first patch should shorten the path from a valid native uplink to a visible sensing result. Subsequent patches should increase what a reviewer can infer from that result.
