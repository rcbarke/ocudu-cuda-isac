# Aerial Framework and the Aerial-OCUDU swap

Every simulated result is developed on DeepSig-OCUDU on a DGX Spark. Two Aerial hosts follow. Aerial Framework simulates the same sensing core inside a real-time PUSCH pipeline. For over-the-air work, DeepSig-OCUDU is swapped for Aerial-OCUDU and the contribution code is carried across. This page states what moves, what each host supplies and what has to be verified on the lab systems. Tickets: [S10 and S11](../dev/DEVELOPMENT.md#phase-s10).

## Three computers

| Step | Earlier ladder | Ladder used here | What it adds |
|---|---|---|---|
| Design and train | Sionna link-level models | DeepSig-OCUDU with Sionna RT channels, on one DGX Spark | The full PHY executes. No PHY abstraction. |
| Simulate | Aerial Omniverse Digital Twin | Aerial Framework, PUSCH over emulated 7.2x fronthaul | Real-time pipeline and fronthaul timing without the Omniverse front end |
| Deploy | Aerial Testbed | Aerial-OCUDU with a WNC O-RU, over the air | The same OCUDU L2 as development, on Aerial's L1 |

The ladder used here needs DGX systems. The earlier ladder stays valid on COTS hardware. The three-computer cycle is described by [Cohen-Arazi et al.](https://arxiv.org/abs/2510.01533v1)

## What moves and what does not

| Moves unchanged | Is rebuilt or refitted per host |
|---|---|
| The native sensing core in `native/sense_core/` and its Python reference in `src/ocudu_cuda_isac/sensing/` | The adapter that builds the observation view |
| The observation and decision record schemas | The background template, which belongs to one environment |
| The evaluation endpoints, denominators and pair checkers | Normalization statistics and model weights, when the receiver or scene changes |
| The KPI stream and page | TensorRT engines, which belong to one GPU and TensorRT version |
| The calibration procedure (S4.6) | The calibrated allowance and deadlines, which belong to one placement and host state |
| The unit vectors and the parity tests | The core's binary, which is built for the host's GPU architecture |

Weights trained on simulated scenes are not expected to work in a lab room. The change in performance when the receiver or the scene changes is a result this project measures (S10.3, S11.4), and nothing assumes it away. Published figures from other systems are recorded beside the measured ones and are not targets. The aim on each host is comparable performance.

## Aerial Framework (S10)

The Framework compiles Python, JAX or PyTorch algorithms and C++/CUDA kernels into pipelines of TensorRT engines, and its runtime executes them with real-time scheduling and inline GPU networking. It ships an example PUSCH pipeline and tutorials for the reference PUSCH receiver, MLIR-TensorRT lowering, an AI channel filter, the full PUSCH pipeline, fronthaul testing with an RU emulator, and PHY integration with MAC and RU emulators. [Framework documentation](https://docs.nvidia.com/aerial/framework/latest/index.html)

| Fact | Consequence for S10 |
|---|---|
| Development needs a recent GPU. Runtime testing needs a GPU, a NIC and a real-time kernel with MAC and RU emulation. | S10.1 and a functional S10.2 run fit one machine. Timing-valid runs do not. |
| Extending the Framework toward the full cuPHY stack met source dependencies on the RU emulator's signaling endpoint or a real O-RU, and a jitter-free standalone run needs two nodes joined by CX7 NICs. | S10.2 states its emulator placement and marks single-node runs as functional only. |
| In a native 7.2x deployment the lower PHY belongs to the O-RU, so a channel cannot be applied to the downlink inside the DU. | S10 runs the uplink only. Downlink channels stay on DeepSig-OCUDU split 8 (S5.4, S6.9). |
| The Framework has its own container and toolchain. | Keep it apart from the `isac` environment. Exchange artifacts and records, not Python environments. |

The adapter in `adapters/aerial_framework/` maps the PUSCH inner receiver's channel estimate to an observation view on the device and runs the native core as one pipeline module. S10.1 also lowers the pipeline through MLIR-TensorRT into one engine, the `fused_trt` runtime entry, and compares it with the kernels. The Framework's channel-filter design path is the seam for injecting a learned estimate (S6.11).

## Aerial-OCUDU over the air (S11)

The Aerial-OCUDU FAPI integration was published recently in an OCUDU branch other than the WG1 branch pinned here. Going over the air swaps DeepSig-OCUDU for that branch. S11.1 adds it as a fourth dependency with its repository, branch and commit, and proves the host with no project code attached before anything is ported.

| Contribution code | On Aerial-OCUDU |
|---|---|
| Native sensing core and its Python reference, record schemas, scorecard, KPI stream, page, calibration | Moves unchanged. The core and its engines are rebuilt on the host. |
| Scheduler binding of grants to observations (S5.1) | Moves with OCUDU's L2 |
| Native PHY observer for O1, O2 and O3 (S3.1, S5.2) | Does not move. Aerial's L1 replaces OCUDU's PHY. |
| Observation of the receiver's estimate | Supplied by Aerial's L1 through the dApp data path (S11.2) |

Aerial CUDA-Accelerated RAN exposes PHY data to co-located dApps. An E3 Agent in DU-low notifies subscribed dApps once per uplink slot and passes references into CPU shared memory, where selected data has been copied from the GPU into ping-pong buffers. On GPUs that support MIG the dApp and the RAN run in separate instances. [dApp framework](https://docs.nvidia.com/aerial/cuda-accelerated-ran/latest/dapp_framework/index.html)

| E3 data stream | Observation-record field |
|---|---|
| PUSCH DMRS channel estimates (shared memory) | The complex cube, after frequency interpolation and before time interpolation. This is the stage cuSense consumes and the same stage as O3 on DeepSig-OCUDU. |
| SRS I/Q and estimates | The `srs` source |
| Fronthaul I/Q | O1 |
| SFN, slot, agent-side and TAI timestamps | Absolute opportunity and clock domain. TAI differs from UTC by the leap-second offset. |
| RB start, RB size, symbols, layers | Allocation and mask. Place each slot at its PRB offset. |
| TB CRC, code-block errors, HARQ process | The radio outcome joined to the observation |
| RX antennas, cell ID | Antenna map and cell |

The adapter in `adapters/aerial_ocudu/` is a dApp whose logic is the native core of S3.6, attached behind the `e3::InferenceEngine` interface of the pinned sample applications. It turns the shared-memory row that an indication references into an observation view. Its replay mode reads recorded data with original timestamps, and the public PUSCH H-estimate release is its first fixture.

| To establish on the lab system | Note |
|---|---|
| The exact Aerial-OCUDU repository, branch and commit | Pin it in `ext/lock.json` when the port starts. One public guide describes this pairing as Aerial's L1 exposing FAPI over shared memory, a bridge that translates it to OCUDU's FAPI, and OCUDU's DU-high on the same host, on a GB10 system with a CX7 and an O-RU under PTP timing. Confirm whether that is the branch intended. [Guide](https://tossi.org/docs/ran-integration/fapi-split/aerial-xfapi-ocudu/) |
| The E3 Agent's streams when the L2 is OCUDU | The inspected ACAR manual documents OAI as its example L2. The L1 streams should not depend on the L2. Verify. |
| WNC O-RU on CX7 fronthaul with the Fibrolan Falcon-RX as grandmaster (G.8275.1 with SyncE) | The phase starts when both are received. Record PTP state in every run. |
| Pixel 7 UEs and the core | An end-to-end claim needs a registered UE with verified traffic. OAI 5GC is the default, carried over from the software-UE phase. |
| cuSense on the same system | A published design. The pinned sample applications hold no cuSense dApp, checkpoint or native engine. Reproduce it from the paper until one is published and pinned, and label the arm (S11.4). |

## Pinned reference sources

Both NVIDIA repositories are pinned in `ext/lock.json` as reference sources and fetched as submodules. They are read. Nothing builds or installs them in the `isac` environment, and no gate of the DeepSig-OCUDU ladder depends on them. [Fetch and verify](../../ext/README.md#reference-sources)

| In `ext/aerial-cuda-accelerated-ran` (tag 26.2.0) | Content | Used by |
|---|---|---|
| `cuPHY-CP/data_lake/e3_agent.hpp` | `e3::StreamType`, the append-only wire IDs of every stream (`H_ESTIMATES` is telemetry ID 3), and `SharedMemoryHeader`, version 0x010100 | The S11.2 adapter |
| `cuPHY/src/cuphy_channels/pusch_rx.cpp`, `cuPHY-CP/cuphydriver/src/uplink/phypusch_aggr.cpp`, `cuPHY-CP/data_lake/data_lake.cpp` | The export path of the estimate: an asynchronous device-to-host copy, a pointer handoff after the slot, a worker's copy into ping-pong shared memory | The counterpart of S5.2 and S3.2, and the comparison for the memory paths of S5.5 |
| `cuPHY-CP/e3agent-standalone/` | The production E3 Agent compiled without an L1, fed by a synthetic generator or by a replayed trace (`docs/trace_format.md`) | The S11.2 rehearsal on the Spark |
| `pyaerial/` | pybind11 bindings over cuPHY, a Python package above them and `model_to_engine` with ONNX and TensorRT exporters | The layering that S3.6 follows for its binding and its export route |
| `testBenches/gpu3GPPChan/` | A TR 38.901 section 7.9 sensing channel with target and background components. Its notes list only monostatic scenarios as calibrated. | Not used. A possible later channel source beside Sionna RT. |

| In `ext/aerial-sample-apps` (v1.2.0) | Content | Used by |
|---|---|---|
| `dapps/common/e3_manager/` | The E3 Manager and `inference_engine.h`, the interface an inference backend implements | The seam where S11.2 attaches the native core |
| `dapps/docs/data_representation.md` | Element order and type of every shared-memory stream | The `aerial_hest_c64` and `aerial_iq_fp16` layouts of the [observation contract](../interfaces/OBSERVATION.md#payload-layouts) |
| `dapps/applications/` | Three PRB-power dApps: embedded Python, Triton C API and Triton gRPC | The dApp structure and its license notice. No cuSense dApp, checkpoint or native engine is in this release. |
| `dapps/applications/*/deps/` | The versions the dApps install, among them TensorRT 10.14.1.48.post1 for CUDA 13 and pybind11 2.13 or later | The starting point for the pins of S3.6, to be confirmed on the Spark |

The Aerial tree tracks about 400 MB of test vectors and engines with Git LFS. `make hydrate` leaves them as pointer files. The standalone agent is built in its own container from the pinned tree, never in `isac`. The sample applications' changelog notes that the 26.2 standalone agent exits in synthetic mode when its example configuration names a replay file that does not exist.

## The port is rehearsed on the Spark

The dApp path does not wait for the O-RU. S11.2 exports the observation records of a DeepSig-OCUDU run as a trace for the standalone E3 Agent, replays it, and runs the dApp adapter with the native core against it on the Spark. Its decisions must equal those of the sensor that read the same records from the observation ring. That rehearsal is built in M1. It proves the adapter, the layout conversion and the core's independence from its host. It says nothing about over-the-air propagation, the live L1's timing or a registered UE, which stay with S11. [Code levels](../architecture/RUNTIME.md#one-core-on-every-host)

## The software-UE phase stays on DeepSig-OCUDU

Before any radio is attached, S6.10 connects an OAI software UE with an OAI 5GC to DeepSig-OCUDU at the split-8 ZeroMQ boundary, with the Sionna RT channel applied between them. That checks the replayed-UE shortcut against a UE that performs PRACH, registration and scheduled transmission, on the development host and in lockstep time. OAI's ZeroMQ radio and the pin's ZeroMQ radio both exchange IQ over request/reply sockets. Their pairing is verified in S6.10, with srsUE as the fallback. [OAI ZeroMQ radio notes](https://git.yigid.dev/yigid/openairinterface5g/src/branch/develop/radio/zmq/README.md)

## Evidence each rung can give

| Rung | Can show | Cannot show |
|---|---|---|
| DeepSig-OCUDU with Sionna RT | Receiver-stage effects, all ablations, GPU contention, payload correctness against known bytes, a software UE in lockstep time, the native core's latency beside the CUDA PHY, the dApp adapter against the standalone E3 Agent | Over-the-air propagation, hardware impairments |
| Aerial Framework | The lowered pipeline in a real-time PUSCH receiver with fronthaul timing | Downlink channels, user traffic |
| Aerial-OCUDU over the air | Over-the-air sensing with commercial UEs, comparison with cuSense, user traffic | Paired noise-identical ablations: no two captures repeat a channel |

Keep each result with the rung that produced it. A simulated result is not restated as an over-the-air result, and an over-the-air capture is not used to claim a paired receiver-stage effect.
