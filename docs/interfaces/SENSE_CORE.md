# Native sensing core

The contract of `native/sense_core/` as Sprint 3.6 built it: the C API, the plan of the cuSense-like branch, the engine directory, the sensor process and the files it writes. The reference of every unit named here is `src/ocudu_cuda_isac/sensing/cusense.py`. [Code levels and the native core](../architecture/RUNTIME.md) | [observation contract](OBSERVATION.md) | [reference of the branch](CUSENSE_FRONTEND.md) | [sprint card](../dev/DEVELOPMENT.md#phase-s3)

## What is in the directory

| Path | Holds |
|---|---|
| `include/sense_core/sense_core.h` | The C API. Opaque handles, plain structures, one status per call. It compiles as C99 and as C++20 |
| `src/core/` | SHA-256, a JSON reader that keeps every scalar as the file writes it, the reader of a frozen artifact with every hash checked, the registry and the declared tolerances |
| `src/kernels/units.h` | Every unit of the ingest and of the front end, written once. `cuda_device.cu` compiles them into two kernels, `host_device.cpp` into loops on the CPU: the host twin |
| `src/engine/` | The engine host with three backends: `tensorrt`, `libtorch` and `host_reference`, and the check of an engine manifest |
| `src/plan/` | The plan: the causal window of the reference stream, the submission of a window, the decision at the frozen threshold, the JSON lines, and the optional stage `map_mean_kalman` |
| `host/` | `sense_host`, the sensor process, its loop and its record sources: the ring and a consumer archive |
| `tools/` | `sense_feed`, a ring producer that publishes an archive at the pace of its slots; `sense_engine`, the engine builder; `sense_probe`, the checkpoint D0 |
| `python/binding.cpp` | The pybind11 extension `_sense_core`, loaded by `ocudu_cuda_isac.sensing.runtime` |
| `tests/` | The unit cases of CTest |

The project takes nothing from the OCUDU build. It compiles `native/obs_ring` from its sources, for the record definition and for the consumer that `sense_host` links.

## Build and receipt

```bash
python scripts/native_core.py probe              # checkpoint D0: TensorRT builds and runs an engine on this GPU
python scripts/native_core.py build              # build/sense-core: CUDA device, TensorRT where it was found
python scripts/native_core.py build --host-only  # build/sense-core-host: the host twin, no CUDA compiler
python scripts/native_core.py verify             # the receipt, after the files and the tree were checked
python scripts/native_core.py engine --artifact DIR --output-dir DIR   # onnx/ and engine/ for this GPU
```

`build` configures out of tree, builds, runs CTest and writes `isac-core.json` (`isac-core-build/v1`): every source file with its hash, the hash a binary names for its sources, the device, the CUDA architecture, the compilers, the TensorRT version with its origin, the library, the executables and the binding with their hashes, the link lines, the unit cases that ran and those that were skipped, and the registry and the tolerances of the build. `sense_host --build` prints the same source hash, and `verify` refuses a build whose binary names another.

The probe finds TensorRT in this order: what `--tensorrt-root`, `--tensorrt-include` and `--tensorrt-library` or `TENSORRT_ROOT` name, a package installation in the system directories, and the wheel `tensorrt_cu13` in the interpreter with the headers that `python scripts/native_core.py headers` fetched into `build/tensorrt-headers/`. It compiles `tools/sense_probe.cpp` on its own, runs it, and writes `sense-probe.json` (`isac-core-probe/v1`).

## Units and working precision

A unit carries the name of its registry entry in the reference. `sense_host --registry` lists them with the hash of the file that holds each.

| Stage | Entry | What the unit does |
|---|---|---|
| ingest | `ocudu_o3_fp16` | `[dmrs_symbol][prb][rx_port][subcarrier_in_prb]`, complex FP16, widened once to FP32 without a change of value and without a scale |
| ingest | `ocudu_o2_comb_c64` | `[rx_plane][reference_symbol][pilot]`, complex64 |
| ingest | `aerial_hest_c64` | `[dmrs][subcarrier][rx_antenna][layer]` of one layer, complex64. Reachable through `sense_plan_ingest` and the binding; an artifact for it comes with S10 |
| window | `coherent_mean` | The slots of one period, each at the offset of its allocation, averaged in the complex domain over their valid samples |
| clutter | `background_magnitude` | `\|H_t\| - \|H_B\|` where the window and the template both hold a sample |
| symbols | `dmrs_mean` | The mean over the DM-RS symbols that hold one |
| normalizer | `zscore_global` | `(d - mean) / (scale + 1e-8)`; a position without a sample is 0 |
| model | `cusense_cnn` | The engine: `input [1, A, K]` to `map [1, rows * cols]`, the softmax of the grid logits, and `presence_logit [1]` |
| detector | `frozen_threshold` | Present when the logit lies above the frozen threshold. On the host, in double precision |
| smoothing | `none`, `map_mean_kalman` | The optional stage, on the host, after the publication time of its decision |

The working precision is FP32. The validity rule of the reference is evaluated on exact products in double precision: a sample is valid when its bit is set, both parts are finite and its magnitude lies above `tau`. The units are compiled without fused multiply-add on both sides (`--fmad=false`, `-ffp-contract=off`), so the host twin executes the operations of the kernels one by one.

`sense_host --tolerance` prints what the build declares against the reference:

| Quantity | Tolerance |
|---|---|
| Window mean | 4 spacings of FP32 on each part at the magnitude of the element. For FP16 payloads the three additions are exact, and the mean is bit-identical to the complex64 mean of the reference |
| Input of the model | 16 spacings of FP32 at the larger of the window magnitude and the template magnitude, divided by the divisor of the z-score |
| Engine, FP32 | A logit by 1e-3 plus 1e-4 of its magnitude, a probability by 2.5e-4, a cell of the map by 1e-4: the tolerances of the Sprint 3.5 parity |
| FP16, declared apart | A stored FP16 value holds each part within a relative 2^-11. No unit adds to it. An engine built with `--precision fp16` has its own set (5e-2 plus 1e-2, 1e-2, 5e-3) and is graded only where a gate builds one |

## A plan

A plan is one configured chain of registry entries on one device, built from one frozen artifact. `sense_plan_create` reads the four files, checks every hash the manifest names, refuses a stage name without a unit, allocates every workspace, and binds the engine.

| Call | What it does |
|---|---|
| `sense_plan_warm_up` | Runs the chain on a window that equals the template, and captures the engine into a CUDA graph where it can |
| `sense_plan_offer` | Takes the next view in the order of its carrier. It reads the header and nothing else, tracks the window and holds the view in place. No device call |
| `sense_plan_submit` | The window is complete: one upload of the staged records, the ingest kernel, the front-end kernel, the engine, one download, behind one completion event |
| `sense_plan_poll` | Asks that event for its state. Never blocks |
| `sense_plan_collect` | The decision. The publication time `t_decision` is taken here |
| `sense_plan_skip` | Gives up a complete window that a newer complete one overtook |
| `sense_plan_close_open` | Gives up a window that is filling, as a gap, when its carrier went silent |

An answer has the status of the reference stream: `insufficient_history`, `gap`, `invalid`, `rejected`, `ok`. Two more exist only in a live core: `stale`, a record of the window was rewritten while it was read, and `stale_skipped`, the window was overtaken before it ran. Neither carries a score.

Rules a plan keeps, each with a counter that a report prints:

- Nothing is allocated after the plan was created (`device_allocations`).
- No device call is made while a window is open (`device_calls_in_an_open_window`).
- A view is released when the plan is done with its memory, in the order it was offered.
- A stage that takes longer than its budget is counted, and its decision is published with the outcome `late`.
- A record whose memory changed during the ingest copy yields no score, and nothing of its window reaches the device.

## Engine directory

```text
engine/model.plan     the serialized TensorRT engine
engine/engine.json    sense-engine/v1
```

`engine.json` names the artifact and its weights by hash, the library with its version, the precision, whether TF32 was allowed, the GPU by name, compute capability and identifier, the tensors, the ONNX file by hash, and the hash of `model.plan`. A plan loads the directory only for the same artifact, with the same TensorRT version, on the same GPU. An engine is built on every GPU that runs it. The default is FP32 without TF32.

The ONNX file comes from `ocudu_cuda_isac.sensing.export.write_onnx`, which builds the graph node by node from `weights.f32` and the configuration of `model.json` at operator set 17. The same artifact gives the same bytes. The declared fallback is the backend `libtorch` with the TorchScript module of `export.write_module`: it blocks, it is named in every manifest, and a run on it records the native showing as blocked. `host_reference` is the model in plain C++ on the CPU, for the host twin and the unit gate.

## sense_host

```bash
sense_host --socket PATH --artifact DIR --output-dir DIR [options]     # a live ring
sense_host --archive DIR --artifact DIR --output-dir DIR [options]     # the records of a consumer archive
sense_host --build | --registry | --tolerance
```

The sensor loads one artifact, builds one plan, warms it up and only then attaches, as the only consumer of its ring, in `spin` mode. It holds the records of an open window in place. When the last record of a window arrives it submits once, asks the completion event for its state until it has occurred, takes the decision, and releases the window in order. Before it submits it reads every record that is already published: a complete window that a newer one overtook is skipped and counted as `stale_skipped`. It keeps no queue. With `--release after_radio` the submission waits until `--radio-allowance-us` have passed since `t_ready` of the last record. A window that is still filling when the ring was silent for `--open-ms` (default 20) is closed as a gap, so that a producer that drains at the end of a run gets every credit back.

Nothing is allocated, hashed or written to a file of the run while records arrive. After the producer said that no record follows, the sensor writes the output of a cuSense sensor, which `cusense.open_output` verifies:

```text
decisions.jsonl      one cusense-decision/v1 line per answer, runtime "native"
inputs.f32           the input [A, K] of every scored window
maps.f32             the map [rows, cols] of every scored window
cusense.json         the report: artifact, entries with their units, runtime, plan, counts
windows/windows.jsonl, windows/windows.json
windows/records/     with --keep-records: every record it read, as obs_tail keeps them
```

A native decision adds to the reference fields: `engine` (backend and hash), `plan_sha256`, `core_sources_sha256`, `marks` (`t_notify_ns`, `t_submit_ns`, `t_ingest_ns`, `t_kernels_ns`, `t_engine_ns`) and `deadline` (the outcome against the budgets, or null without one). With `--evidence timed` the window mean stays on the device and its digest is null. With `--evidence audit` it returns with every decision, and its digest is the one the reference names. A run on the host twin, or on another backend than `tensorrt`, says `"timing_valid": false`.

## Latency definitions

| Definition | From the marks of a decision |
|---|---|
| cuSense boundary | `t_decision_ns - marks.t_notify_ns` |
| Post-window | `t_decision_ns - t_ready_ns` |
| End to end | `t_decision_ns` minus `t_first_sample` of the window, through the clock map of the run |

`t_ingest`, `t_kernels` and `t_engine` are taken on the device between events on the stream and placed before the host time at which the completion event was seen.

## Gate

`make accept-s3.6` runs the probe, the unit gates, the build with the kernels on the device, and `tests/integration/sense_core.py` on the accepted Sprint 3.5 run in eight cases. It writes `isac-core.json`, `runtime_parity.json`, `runtime_latency.json` and `cusense_reference.json`. The latency beside the published figure is recorded and not graded. The arms with a receiver beside the sensor on one GPU are Sprint 4.1's.

A development host without a GPU rehearses the driver on the host twin. The rehearsal grades every case but the checks of a device and leaves its hardware gate open:

```bash
python scripts/native_core.py build --host-only
python tests/unit/test_sense_core_driver.py -v       # makes a stand-in control and runs the driver with --rehearsal
```
