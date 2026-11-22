# Native sensing core

The contract of `native/sense_core/` as Sprint 3.6 built it: the C API, the plan of the cuSense-like branch, the engine directory, the sensor process and the files it writes. The reference of every unit named here is `src/ocudu_cuda_isac/sensing/cusense.py`. [Code levels and the native core](../architecture/RUNTIME.md) | [observation contract](OBSERVATION.md) | [reference of the branch](CUSENSE_FRONTEND.md) | [sprint card](../dev/DEVELOPMENT.md#phase-s3)

**Status.** Accepted on the DGX Spark `spark-ecf8` on 2026-11-22 (UTC): run `s36-spark-ecf8-20261122T194723Z` at commit `58e29c7`, eight of eight cases, 34 of 34 checks, hardware gate closed. [What the Spark run closed](#what-the-spark-run-closed)

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
| FP16, declared apart | A stored FP16 value holds each part within a relative 2^-11. No unit adds to it. An engine built with `--precision fp16` has its own set (5e-2 plus 1e-2, 1e-2, 5e-3) and is graded only where a gate builds one. `sense_engine` builds it with TensorRT 10 only: [Engine directory](#engine-directory) |

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

The sources compile against the headers of TensorRT 10 and of TensorRT 11, which differ in how an engine gets its precision. Up to TensorRT 10 a network is weakly typed and `BuilderFlag::kFP16` allows half precision in it. TensorRT 11 removed that network and that flag: every network is strongly typed, and an engine has the precision of the tensors of its ONNX graph. The export writes a float32 graph, so the FP32 engine is float32 throughout on both. `--precision fp16` builds an FP16 engine with TensorRT 10 and is refused with TensorRT 11 in one line, with status 2, before the device is touched: an FP16 engine there needs a float16 export, which no ticket has built. The build looks for the runtime libraries of a wheel by the major the headers name, `libnvinfer.so.10` or `libnvinfer.so.11`.

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

## What the Spark run closed

Run `s36-spark-ecf8-20261122T194723Z`, `make accept-s3.6` at commit `58e29c7`: 48 gate lines and no failure, 699 unit tests in 593 s with none skipped, the unit gate 17, the link rule 4 and the import rule 3. The gate took 10.5 minutes, and the driver 18 s of them. The first build on the Spark stopped at `BuilderFlag::kFP16`, which TensorRT 11 no longer has, and one test of the discovery assumed a host without TensorRT: both were fixed before the gate ran, and the gate passed on its first run.

| Item | Measured |
|---|---|
| Checkpoint D0 | TensorRT 11.5.0.63 from the system packages, headers and library of one version; a one-layer engine built on the NVIDIA GB10 (capability 12.1, CUDA runtime 13.5, driver 580.178.04), executed on a stream with device addresses and launched as a captured graph: 2.3 us in the median |
| Build | 44 source files, `69d70fc2f9ef150b`; `sm_121` by `nvcc` 13.5.37; no compiler warning; 43 unit cases with `sense_core_device_vectors` on the device, none skipped; `sense_host` `04db81d45bec`, `sense_engine` `5b00db136c7e`; the loader resolves 9 libraries for `sense_host`, none of an interpreter, of a receiver or of Aerial |
| Control | The Sprint 3.5 run `s35-spark-ecf8-20261122T155753Z`: the artifact `cusense-o3-v1` `f79cf8e924990703`, input `o3_fp16` of shape [1, 4, 612], a grid of 50 by 34 cells, threshold logit -7.698940; 480 held-out records and 160 scored windows |
| Export | `model.onnx` of 9,559,754 bytes, `4125aa76b2daff60`, at operator set 17, the same bytes twice; in ONNX Runtime on the CPU within 1.29e-03 of the scores the Sprint 3.5 sensor wrote on its device |
| Engine | FP32 without TF32, 51 layers, 9,712,452 bytes, `fd836c2d5333b525`, built in 2,091 ms; launched as a captured CUDA graph; refused with status 2 before a record is read for another GPU, another TensorRT version and another artifact |
| Units on the device | 48 of 48 records through the ingest unit give the cube of the reference reader bit for bit, values and validity; 16 of 16 window means are bit-identical; the input within 0.064 of its declared bound; 16 submissions, no device call in an open window, 7 allocations, all at configuration |
| Parity | 160 windows by the native core on `cuda:0` and by the reference on the CPU: the input within 0.069 of its bound (1.5e-06), the logit within 3.81e-06, the probability within 7.9e-08, a cell of the map within 6.1e-09; 160 of 160 decisions agree, no flip, no cell moved |
| Live ring | 2,400 records at 500 us on a ring of 8 slots, read in place: 800 windows scored, none dropped, skipped or late against a window budget of 1,500 us; 800 submissions, each with its marks in order; 7 allocations before the first record; 19 files mapped and no Python library; the logits equal those of the archive replay exactly |
| Overload | One record per 100 us against a sensor held 3,000 us on a window, 4 slots: 160 of 960 admitted, 800 dropped and counted; the producer kept 99.9 us per record with a publication of 1.49 us at the 99th percentile and no system call; 160 of 160 credits back; 27 windows scored and 27 gaps, never a backlog worked through |

**Recorded, not graded.** `runtime_latency.json`, the sensor alone on the device, 800 windows of the native core and 798 of the reference:

| Latency, us | Native, median | Native, 99th | Native, most | Reference, median | Reference, 99th |
|---|---|---|---|---|---|
| cuSense boundary: last record notified to decision | 130.0 | 183.4 | 267.0 | 624.0 | 1,118.3 |
| Post-window: last record ready to decision | 131.1 | 184.1 | 267.7 | 630.2 | 1,897.9 |
| End to end: first sample to decision | 1,485.5 | 1,538.6 | 1,622.2 | 1,984.6 | 3,252.3 |

The native stages in the median: 1.0 us from the notification to the submission, 13.2 us to collect the window, 13.3 us for the ingest kernel, 6.0 us for the front-end kernel and 96.3 us for the engine (112.1 us at the 99th percentile). The engine is three quarters of the window. The reference runtime, PyTorch on `cuda:0`, spends 62.9 us in its front end and 367.7 us in its inference in the median, had one window of 24.8 ms, and its producer dropped 5 of 2,400 records. By the cuSense boundary the reference takes 4.8 times as long as the native core in the median and 6.1 times at the 99th percentile. The acquisition of a window is 1,354.4 us and the same for both, so end to end the ratio is 1.34.

What these figures are not. No receiver ran: the arms with the receiver beside the sensor on one GPU are S4.1's, and this ticket closes with checkpoint D4. The host ran without MPS, with the performance governor and a load average near 1.7, and the driver's own Python process held a context of 170 MiB on the GPU during the run. `cusense_reference.json` puts the 130.0 us beside the published 241 us with `"graded": false` and seven differences: 51 resource blocks on two DM-RS symbols against 273 on three, about a ninth of the samples; a GB10 against a GH200 with MIG; a replayed simulated scene against an over-the-air room; a window of three slots; a cuSense-like implementation, since cuSense is not open source; and a sensor in a process of its own against a dApp beside a live L1. The position error is that of the Sprint 3.5 artifact, 2.33 m in the mean on the held-out windows from the raw map and 2.50 m after the mean of the maps and the Kalman filter, against the published 0.73 m, and is not changed by the runtime.

Two findings for the next tickets. The TensorRT engine without TF32 on the GB10 lies within 3.81e-06 of the CPU reference, where the PyTorch sensor of Sprint 3.5 on the same GB10 lay within 1.70e-03: the larger difference belongs to the PyTorch path on the device, and the FP32 tolerance of this core has room. And the Spark's TensorRT is 11.5.0.63: `environments/sense-core-constraints.txt` pins it from this run on, and an FP16 engine needs a float16 export there ([Engine directory](#engine-directory)).
