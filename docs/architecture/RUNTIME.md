# Code levels and the native sensing core

This project measures real-time sensing beside a CUDA receiver on DGX systems. Every sensing stage whose latency, deadline, contention or energy is reported executes as a CUDA kernel or a TensorRT engine, in a process that holds no Python interpreter. Python and PyTorch define each stage, train the models and serve as the reference that the compiled code is checked against. This page fixes which code lives at which level, how one sensing core runs beside DeepSig-OCUDU and inside the Aerial ports, and which run supports which claim. Tickets: [S3.2, S3.6](../dev/DEVELOPMENT.md#phase-s3), [S6.12, S6.13](../dev/DEVELOPMENT.md#phase-s6), [S10.1](../dev/DEVELOPMENT.md#phase-s10) and [S11.2](../dev/DEVELOPMENT.md#phase-s11).

## Levels

| Level | Holds | Directory | Language | Runs | On an Aerial host |
|---|---|---|---|---|---|
| Receiver | CU/DU and CUDA PHY | `ext/ocudu-deepsig-cuda-inline` | Upstream C++ and CUDA | Every slot | Replaced by Aerial's L1 |
| Receiver patch | Read-only export of the receiver's estimate (S5.2) | `patches/ocudu/` | C++ with CUDA runtime calls. No new kernel. | Every slot | Not needed. The Data Lake path is its counterpart. |
| Host adapter | Fixtures, replay, receiver harness, observer, ring producer, DU binding, split ingress | `native/ocudu_adapter/` | C++17 on the CPU | Every slot | Rewritten per host in `adapters/` |
| Observation transport | Record, ring, control channel, frames | `native/obs_ring/` | C++17 on the CPU. No OCUDU include, no CUDA. | Every observation | Used on DeepSig-OCUDU hosts. On Aerial the E3 shared memory carries the estimate. |
| Native sensing core | Ingest, kernels, engines, plans, the sensor host, the Python binding | `native/sense_core/` | C++17, CUDA, TensorRT | Every observation | Moves unchanged. Rebuilt for the host. |
| Reference and authoring | Registries, reference implementations, models, training, export | `src/ocudu_cuda_isac/sensing/` | Python, NumPy, PyTorch | Offline, and as the labeled `reference` runtime | Moves unchanged |
| Offline | Episodes, evaluation, calibration | `src/ocudu_cuda_isac/episodes/`, `evaluation/` | Python, Sionna, Sionna RT | Never in a slot loop | Moves unchanged |
| Tooling | Gates, launchers, KPI stream, page | `scripts/`, `gui/`, `src/ocudu_cuda_isac/telemetry/` | Python, shell | Outside every timed path | Moves unchanged |

Of the 55 tickets in the plan, one changes receiver source (S5.2). Sensing is never written into the receiver's `lib/phy/cuda`: that directory does not exist on Aerial-OCUDU, so code placed there is the one part of the project that cannot be carried over. It would also put sensing inside the slot deadline and remove the two-client layout that M1 to M3 measure.

The reference system is built the same way. In the pinned Aerial tree (`ext/aerial-cuda-accelerated-ran`, tag 26.2.0) the L1 only exports the estimate:

| Location in the pinned Aerial tree | What it does |
|---|---|
| `cuPHY/src/cuphy_channels/pusch_rx.cpp`, the copy into `pChannelEstsHost` | Queues an asynchronous device-to-host copy of each UE group's estimate tensor on the PUSCH stream, only when the caller supplied a host buffer |
| `cuPHY-CP/cuphydriver/src/uplink/phypusch_aggr.cpp` | Allocates that pinned buffer only when the Data Lake is enabled, and calls `DataLake::notify` after the slot, which hands pointers to a worker thread |
| `cuPHY-CP/data_lake/data_lake.cpp`, `e3_agent.hpp` | The worker copies into ping-pong shared-memory rows. The E3 Agent notifies subscribed dApps. `e3::StreamType` and `SharedMemoryHeader` are the wire contract. |

The sensing application is a separate process. The tree holds no cuSense code, and the pinned sample applications (`ext/aerial-sample-apps`, v1.2.0) hold three PRB-power dApps and no cuSense dApp, checkpoint or native engine. S5.2 with S3.2 is the same export half on DeepSig-OCUDU. [Aerial ports](../integration/AERIAL.md)

## Timed stages are compiled

1. A stage that a run times is a CUDA kernel or a TensorRT engine inside the native core. The process that executes it loads no Python interpreter.
2. Every stage has a Python reference under `src/ocudu_cuda_isac/sensing/` with the same registry name. The reference is the definition. The native unit must reproduce the reference on the card's hand-checkable vectors and on recorded observations, within a tolerance declared per precision.
3. A registry entry without a native unit is an offline entry. A cell that uses it reports accuracy and no latency, deadline, contention or energy.
4. From S3.6 on, every Integrate step that reports sensing latency, a deadline, PHY tails under sensing load or energy runs `runtime: native`. S3.6 is built in the demonstration, so the demonstration is included (see [Runtime axis](#runtime-axis)).

## The native core

```text
native/sense_core/
  CMakeLists.txt        project(sense_core LANGUAGES CXX CUDA). No OCUDU and no Aerial include or library.
  include/sense_core/   The C API: observation view, plan, decision, registry names
  src/ingest/           One unit per payload layout. Gathers any registered layout into the canonical cube.
  src/kernels/          One unit per registry entry: front end, radar cube, clutter, CFAR, angle
  src/engine/           Engine backends: TensorRT, and LibTorch as the declared fallback
  src/plan/             A configured chain of stages with its workspaces, stream and events
  host/                 sense_host: the sensor process on DeepSig-OCUDU hosts
  python/               pybind11 binding, imported as ocudu_cuda_isac.sensing.runtime
  tests/                CTest cases with the hand-checkable vectors of the cards
```

| Contract | Rule | Checked by |
|---|---|---|
| No host | The library and `sense_host` compile and link without an OCUDU archive, an Aerial library or a simulator. It is the native twin of the import rule for the Python core. | A unit test reads the link line (S3.6) |
| One input | The core consumes an [observation view](../interfaces/OBSERVATION.md#the-observation-view): a pointer, the record header, a layout name, a dtype and a memory kind. It never sees a grid reader, an E3 message or a file. | The same fixture reaches the core through the OCUDU layout and the Aerial layout and gives the same cube (S3.6) |
| One registry | A native unit carries the name of its Python reference. A plan is a list of names. The manifest stores the names with the hash of each unit. | Plan creation refuses an unknown name |
| Parity | Each native unit reproduces its reference on the card's vectors and on recorded observations. FP32 and FP16 tolerances are declared separately. Decisions at the frozen threshold are compared, and every flip is listed with its score margin. | `runtime_parity.json` and the parity files of S6.12 and S6.13 |
| Bounded execution | A plan allocates at configure time and never afterwards. It runs on a stream that the caller supplies, signals completion with an event and never synchronizes the device. An overrun is counted and its decision is published as late. A backlog appears as held credits and counted drops at the ring, never as a delay in the receiver. | A counting allocator and a fake clock in the unit tests |
| Precision | Each unit states its working precision. An ingest unit converts the payload dtype once. | The plan manifest |
| Engines | An engine is built on the GPU that runs it, from the ONNX export of a frozen artifact. It carries the artifact hash, the TensorRT version and the GPU it was built for, and it is refused anywhere else. | Engine load in S3.6 and S10.1 |
| C boundary | The public API is C with opaque handles, as the PUSCH API of the pinned receiver and the cuPHY API of the pinned Aerial tree are. Sources stay within C++17 and CUDA C++17, the common subset of the two stacks. | The API header compiles as C and as C++20 |
| Binding | `ocudu_cuda_isac.sensing.runtime` exposes plans, views and decisions to Python for tests, parity checks and offline scoring. No timed run imports it. | The sensor host starts on a host without Python |
| Receipt | `scripts/native_core.py` builds the core out of tree and writes `isac-core.json` with the hashes of sources, library, host, binding and engines, the toolchain, the CUDA architecture and the TensorRT version. | `make core-build`, then `verify`, as the adapter does |

The sensor process on DeepSig-OCUDU hosts is `sense_host`. It attaches to the observation ring as its only consumer, turns each record into a view, runs the plan named by the run profile and writes `decisions.jsonl`. Its file writes happen after the publication time is taken.

## Stage registry

Names are the entries of the [ablation axes](ABLATIONS.md). The reference column is the definition, and the native column is what a timed run executes.

| Stage | Entries | Reference | Native unit | Tickets |
|---|---|---|---|---|
| Ingest | `ocudu_o3_fp16`, `ocudu_o2_comb_c64`, `ocudu_o1_cbf16`, `aerial_hest_c64`, `aerial_iq_fp16` | The record reader | `src/ingest/` | S3.6, S10.2, S11.2 |
| cuSense front end | Window average with PRB placement, `background_magnitude`, DMRS average, z-score | `cusense.py` | `src/kernels/` | S3.5, S3.6 |
| Radar cube and delay-Doppler | Cube formation, windows, regular and irregular transforms | `features.py`, `spectral.py` | `src/kernels/` with cuFFT | S3.3, S6.2, S6.12 |
| Clutter | `background_complex`, `background_recursive`, `mti_2`, `mti_3`, `whitening`, `projection`, `nlms`, `rls` | `spectral.py` | `src/kernels/` | S6.2, S6.12 |
| Detector | `frozen_threshold`, `cfar_ca`, `cfar_os`, `cfar_go`, `cfar_so` | `presence.py`, `spectral.py` | `src/kernels/` | S3.4, S6.2, S6.12 |
| Angle | Covariance, `bartlett`, `music`, model order | `angle.py` | `src/kernels/` with a batched Hermitian eigensolver | S6.3, S6.13 |
| Models | `cusense_cnn`, `linear`, `gru`, `transformer`, `joint`, learned covariance heads | `models/` | `src/engine/` | S3.4, S3.5, S6.5, S6.6, S6.11, exported by the route of S3.6 |

## Runtime axis

| Entry | What executes | What it supports |
|---|---|---|
| `native` | The native core: CUDA kernels and engines in `sense_host`, a dApp or a pipeline module | Every latency, deadline, contention and energy statement, and the ports |
| `reference` | PyTorch and NumPy in a Python process, reading the ring through the standard-library reader | Correctness, training and accuracy endpoints. Its latency is recorded with this label. It supports no deadline, contention, energy or comparison statement. |
| `fused_trt` | The front end and the model lowered into one TensorRT engine, run by the same host (S10.1) | The Aerial Framework lowering route, and the comparison of a compiled graph with hand-written kernels |

The engine backend is recorded beside the runtime: `tensorrt`, or `libtorch` where a host has no TensorRT build for its CUDA stack. A fallback is named in the manifest and is never silent.

The demonstration (milestone D) runs `native` for its graded cuSense-like branch, with the TensorRT backend, and shows that latency beside the same pipeline on `reference`. It requires the TensorRT backend, so the probe of S3.6 runs first, and a host without a TensorRT build for its CUDA stack blocks the native showing. The delay-Doppler branch of the demonstration stays on `reference`, on the CPU, until S6.12 gives it native units, so that the receiver and `sense_host` are the only two CUDA clients of the timed arms.

These entries mirror the serving options that the cuSense paper compares on one pipeline (section 6.2): a Python model with TorchScript or TensorRT, one fused TensorRT engine, and a native option with CUDA preprocessing and a TensorRT engine. On a GH200 with MIG the paper reports 2,138 us at the median for its original Triton deployment and 241 us for the native option. The 241 us figure is therefore compared only with a `native` run here. [Villa et al.](https://arxiv.org/abs/2512.06493v3)

## One core on every host

| Host | Who builds the view | Payload memory | Ticket |
|---|---|---|---|
| DeepSig-OCUDU on one Spark, and on the GB300 | `sense_host`, from the observation ring | Host shared memory | S3.2, S3.6, S8.1 |
| DeepSig-OCUDU on two Sparks | `sense_host`, from socket or verbs frames | Pinned host memory | S7.1, S7.2 |
| Standalone E3 Agent on the Spark (port rehearsal) | The dApp adapter, from E3 shared memory | Host shared memory | S11.2, built in M1 |
| Aerial Framework | The pipeline adapter, from the inner receiver's estimate | Device memory | S10.2 |
| Aerial-OCUDU over the air | The dApp adapter, from E3 shared memory | Host shared memory | S11.2 |

Rebuilt per host: the binary for the GPU's architecture, the engines, the background template, the normalization and the calibrated service levels. Unchanged: the source, the registry, the record schemas, the unit vectors and the parity tests.

The port is rehearsed before any Aerial hardware exists. The pinned Aerial tree ships `cuPHY-CP/e3agent-standalone/`, which compiles the production E3 Agent and feeds its shared memory from a recorded trace instead of a live L1. S11.2 writes that trace from DeepSig-OCUDU observation records and runs the dApp adapter against it on the Spark. The decisions must equal those of `sense_host` on the same records. [Aerial ports](../integration/AERIAL.md)

## Timing boundaries

| Mark | Clock | Taken |
|---|---|---|
| `t_first_sample`, `t_last_sample` | Radio | From the record: first and last contributing sample of the window |
| `t_ready` | Host monotonic | Producer: the completion event of the copy that filled the record |
| `t_notify` | Host monotonic | Consumer: the notification of a record arrived |
| `t_ingest`, `t_kernels`, `t_engine` | Host monotonic, from CUDA events | Core: the last ingest, kernel and engine unit of the plan completed |
| `t_decision` | Host monotonic | Core: the decision record is complete. This is the publication time. |

| Latency | Definition | Use |
|---|---|---|
| End to end | `t_decision` minus `t_first_sample`, through the recorded map between the two clocks | This project's endpoint. A late result is a failure. |
| Post-window | `t_decision` minus `t_ready` of the last record of the window | Calibration (S4.6) |
| cuSense boundary | `t_decision` minus `t_notify` of the last record of the window | The comparison with the published figure |
| Export | `t_notify` minus the receiver's kernel completion | The cost of the observation path (S5.5) |

## Claims and the evidence each needs

| Statement a paper makes | Needs | Produced by |
|---|---|---|
| The receiver's own estimate leaves the receiver without changing a decode or delaying the PHY | Parity of O3, paired CRC and bytes against the unpatched build, reconciled ring accounting, the soak | S5.2, S3.2, S5.3 |
| Sensing meets a deadline beside the CUDA PHY at a fixed false-alarm point | A `native` run, service levels calibrated on that runtime and host state, the reduction of every opportunity | S3.6, S5.6, S9.2 |
| The compiled core runs below the PyTorch reference | Both runtimes on identical records and arms, with identical decisions or listed flips | S3.6: `runtime_parity.json`, `runtime_latency.json` |
| Sensing changes the radio's tails by a stated amount under MPS, across two hosts or under MIG | `native` runs with the tap off, capture only and concurrent at the same offered trace | S4.1, S5.5, S7.3, S8.2 |
| A classical chain and a learned model are compared at one deadline | Native units for every entry in both chains | S6.12, S6.13, S6.5 |
| A receiver stage or a training domain changes detection | Paired records. Either runtime, when the parity file shows the same decisions. | S6.7 |
| One sensing core runs beside DeepSig-OCUDU and on the Aerial stacks | The link test, the same vectors on each host, equal decisions from the ring and from the standalone E3 Agent, then the live runs | S3.6, S11.2, S10.2, S10.3 |
| Performance is comparable with cuSense | A `native` run, both latency boundaries, every difference listed, published figures beside and not graded | S3.6, S11.4 |
| Energy per timely estimate | `native` runs with the power boundary of the placement | S9.2 |

Three rules follow. A latency, deadline, contention or energy statement cites a run whose `runtime` is `native` and whose every stage has a native unit. An accuracy statement may cite either runtime when the parity file of that plan shows decision agreement, and otherwise names the runtime that produced it. A portability statement cites the rehearsal and the live run of the host it names.

## What stays as it is

Sprint 2 is unchanged. The grant record and the episodes are offline Python by design. Channel application (S2.3) and the receiver harness (S2.4) are C++ on the CPU in the host adapter: they generate the stimulus, and no claim is made about their speed. The first tickets that this page changes are S3.2, which moves the ring into the host-free `native/obs_ring/`, and S3.3 to S3.5, which become reference entries with native twins.
