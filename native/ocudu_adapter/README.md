# Native adapter for DeepSig-OCUDU

This directory holds the project's native code for the pinned OCUDU receiver and its build wiring. It is built out of tree against the OCUDU build that Sprint 1.1 accepted. Nothing here writes into the OCUDU source or its build directory. Sensing algorithms do not live here. They stay in the host-free core so that the Aerial adapters can reuse them. [Port contract](../../docs/architecture/ABLATIONS.md) | [sprint cards](../../docs/dev/DEVELOPMENT.md)

## Build

```bash
python scripts/native_adapter.py build      # or: make adapter-build
python scripts/native_adapter.py verify
```

The builder reads the compile flags (`flags.make`) and the link line (`link.txt`) of `pusch_e2e_pipeline_test` from the accepted build and writes them to `build/ocudu-adapter/ocudu-pin.cmake`. `CMakeLists.txt` includes that file and defines the interface target `ocudu_pin_fixture`. A target that links it compiles with the pinned test's flags, apart from `-Werror`, and links the same OCUDU archives, so it runs the objects the Sprint 1 gates accepted. The builder then runs the adapter's CTest cases and writes `isac-adapter.json`, a receipt with the hashes of the executables, of every file in this directory and of the OCUDU build's recipe. `verify` refuses a build whose executable, sources or OCUDU build changed after the receipt.

The adapter is built a second time against the patched receiver of Sprint 5.2, into `build/ocudu-adapter-o3`, with `python scripts/native_adapter.py build --o3` (or `make adapter-build-o3`), after `python scripts/native_ocudu.py build --patched` built that receiver into `build/ocudu-spark-o3`. Only that build defines `ISAC_O3_ROUTE`, compiles `observer/estimate_tap.h` into `pilot_ls` and carries the identity of the patch series in its receipt. `pilot_ls --route` prints which receiver a tool is linked against. The build against the accepted Sprint 1.1 build has no route and refuses a plan that names an estimate.

The import needs the `Unix Makefiles` generator of the accepted build, the default of `make native-build`. On a development host without a Sprint 1.1 receipt, pass `--ocudu-build-dir DIR --no-receipt`.

The Spark built the adapter inside the accepted Sprint 2.1 run `s21-spark-ecf8-20261120T223232Z`. The import carried 17 compile options, among them `-mcpu=neoverse-v2`, the definition `ENABLE_CUDA`, 50 OCUDU archives and the CUDA 13.5 runtime libraries of the pinned link line. A CPU-only build of the same source has 48 archives and no CUDA library. The executable therefore needs the CUDA runtime to start, although the fixture requests host processors only. GCC 13.3.0 compiled `grant_fixture` without a warning, both CTest cases passed, and the receipt names the executable `7a0e1c8ec1a0` and the pinned test `e49b2b81e28a`, the one Sprint 1.1 accepted. Configure, compile, link and test finished within 20 seconds of the gate's start. Later native targets reuse this route, so a new adapter never rebuilds OCUDU and the accepted Sprint 1 build stays the control.

## Contents

| Directory | Ticket | Contents |
|---|---|---|
| `fixtures/` | S2.1 | `grant_fixture`: reads a `pusch-grant/v1` record and its transport block, builds the transmit grid with the transmit chain of the pinned test and writes it in carrier coordinates. It can feed that grid, without noise, to the host PUSCH processor and compare the decoded bytes. [Grant record](../../docs/interfaces/GRANT.md) |
| `replay/` | S2.3 | `grid_apply`: reads a `grid-apply-plan/v1` file, applies `Y = g (sum(H X) + W)` with the grid gain of the episode into a pinned resource grid, reads each grid back and writes the stored grids with their descriptors. Its arithmetic, gain rules, owned grids and digest are in `replay_core`, which includes nothing of OCUDU. [Channel application](../../docs/interfaces/GRID_APPLY.md) |
| `receiver/` | S2.4 | `replay_pusch`: reads a `replay-pusch-plan/v1` file, fills each grid with the channel application of `replay/`, hands the same grid and the PDU of its grant to the host and to the CUDA PUSCH processor of the pinned source and compares the decoded bytes with the original transport block. Its outcome rule, its keys, its held grids and its owned output buffers are in `dispatch_core`, which includes `replay_core` and nothing of OCUDU. Accepted on the Spark, and the untapped control of the later sprints. [Receiver replay](../../docs/interfaces/REPLAY_PUSCH.md) |
| `observer/` | S3.1, S5.2 | `pilot_ls`: reads a `pilot-ls-plan/v1` file, or a `pilot-ls-plan/v2` file with the evidence mode and the ring of S3.2, decodes the receiver replay it names exactly as `replay_pusch` does, and before each decode hands a read-only view of the ready grid to the observer, which computes `H = Y conj(X) / |X|^2` on the pilots of the grant's sensing source: the stage O2. The estimate, the validated source, the validity bits and the observation record are in `observer_core`, which includes `replay_core` and nothing of OCUDU. Accepted on the Spark. The export of the receiver's own estimate follows in S5.2. [Pilot least squares](../../docs/interfaces/PILOT_LS.md) |
| `transport/` | S3.2, S7, S8 | `obs_publisher`: the producer glue of the observation ring. It sizes a ring for a plan, creates it with `native/obs_ring/`, lets the observer write a timed estimate straight into a slot and publishes it, or copies an audited observation in. It includes `observer_core` and `obs/ring.h` and nothing of OCUDU. `pilot_ls` links it. The transports of S7 and S8 arrive with their tickets. Accepted on the Spark, run `s32-spark-ecf8-20261121T204425Z`. [Observations on the ring](../../docs/interfaces/SNAPSHOT_RING.md) |
| `estimate/` | S5.2 | `estimate_core`: the capture of the receiver's own channel estimate, the stage O3, without the device. What a decode is expected to produce, which estimate is refused, where the copy lands, when a copy may be published and the account of every decode. It includes the cores above and nothing of OCUDU or CUDA, and its unit target uses a fake asynchronous producer. The device calls are in `observer/estimate_tap.h`, which is compiled only with `ISAC_O3_ROUTE`. [The receiver's own estimate](../../docs/interfaces/NATIVE_ESTIMATE.md) |
| `pacing/` | S4.1 | `pacing_core`: the pacer, which releases one job per scheduled uplink slot at an absolute time and never shifts a release, and the paced book, the job records of the ledger of `receiver/` with nothing allocated and no mutex taken per decode. It includes `dispatch_core` and `replay_core` and nothing of OCUDU. `pilot_ls` links it and runs it on a plan `pilot-ls-plan/v4` with a `pacing` object; `pilot_ls --pacing` prints that version. Accepted on the Spark on 2026-11-23 (UTC), run `s41-spark-ecf8-20261123T140213Z`. After acceptance: the rules of a lead-in and a first token of the paced book, so that `pilot_ls` decodes the first releases of a schedule once more ahead of it, in a book of its own, and the first use of the processor is no release; three more unit cases, eleven in all. The lead-in has not yet run on the Spark. [Paced clients under MPS](../../docs/interfaces/MPS_PACING.md) |
| `du/`, `ingress/` | S5.1, S5.4, S6.10 | Scheduler binding, split ingress adapters and the ZeroMQ channel broker |

`fixtures/`, `replay/`, `receiver/`, `observer/`, `transport/` and `estimate/` exist so far. The adapter also builds `native/obs_ring/`, which stands outside it because a sensor links the ring without the receiver build. A new subdirectory adds its targets with `add_subdirectory` in `CMakeLists.txt`. A target that needs OCUDU links `ocudu_pin_fixture`. A CPU-only unit target links only its module and its doubles: `replay/` builds that way on its own, with `cmake -S native/ocudu_adapter/replay`, and the unit gate `tests/unit/test_grid_apply.py` does so. `receiver/` builds its host-free core the same way, with `cmake -S native/ocudu_adapter/receiver`, and the unit gate `tests/unit/test_replay_pusch.py` does so. `observer/` does the same with `cmake -S native/ocudu_adapter/observer`, and the unit gate `tests/unit/test_pilot_ls.py` builds it that way. The receipt lists the four executables and every file of this directory, so a build from before Sprint 3.1 is refused and the gates rebuild. Sprint 3.1 adds files and edits none of `fixtures/`, `replay/` and `receiver/`: the three executables of Sprint 2 keep their bytes, and `make accept-s3.1` checks that against the receipt of the accepted Sprint 2.4 run.

## `pilot_ls`

```bash
build/ocudu-adapter/observer/pilot_ls --plan FILE --output-dir OUT [--only OPPORTUNITY_ID]... [--no-grids] [--phy-debug]
build/ocudu-adapter/observer/pilot_ls --self-test
```

`OUT` must exist and must not hold a result. The plan comes from `python -m ocudu_cuda_isac.episodes.pilot_ls plan` (version 1: mode `audit`, no ring) or from `python -m ocudu_cuda_isac.episodes.snapshot_ring plan` (version 2: the evidence mode `audit` or `timed` and the ring) and names a replay plan of `replay_pusch` by its hash. With a ring the tool creates the segment and its socket, waits for the consumer as the plan says, publishes one `obs/v1` record per observation and writes `OUT/ring/` with the account of the ring. In the mode `timed` it computes no checksum between its first fill and its drain and writes its files after the drain. The tool writes `OUT/ingress/` and `OUT/radio/` as `replay_pusch` does, `OUT/observer/` (one `obs-descriptor/v1` record, one payload and one validity mask per opportunity) and `OUT/native.json`. With the tap `none` it writes no observer directory: that run is the control of this executable. The exit statuses are those of `replay_pusch`. `--self-test` decodes a two-block grant on the host path with the tap off and on and starts no CUDA work. The adapter build runs it and the 11 cases of the host-free observer core as CTest cases, 48 with those of the other directories.

| File in `observer/` | Links OCUDU | Contents |
|---|---|---|
| `observer_core.h`, `observer_core.cpp` | No | The estimate and its order of arithmetic, the validated source and the rectangle of the payload, the validity bits, the slot times, the opaque keys, and the observer with its bound sources and its observations |
| `tests/observer_core_test.cpp` | No | Unit cases against the storage model of the grid |
| `observer_plan.h` | Headers only | Reader of the plan and of a sensing-source record with its arrays |
| `pilot_ls.cpp` | Yes | The dispatch loop and the pinned processors of `replay_pusch.cpp`, the tap, and with a plan of version 2 the timed loop and the ring |

| File in `transport/` | Links OCUDU | Contents |
|---|---|---|
| `obs_publisher.h`, `obs_publisher.cpp` | No | The ring a plan needs, the producer of one stream, the timed publication into a slot and the audited copy, the host block of the staged harness |
| `tests/obs_publisher_test.cpp`, `tests/alloc_count.cpp` | No | Six unit cases against a consumer in the same process, with a count of allocations and of hash calls in the timed path |

The observer only reads the grid, through the `o1_view` of a ready grid, and records an estimate only when the grid still has its readback digest afterwards. The estimate is compiled in the core with `-ffp-contract=off`, because its order of arithmetic is part of the interface. The tool computes none itself and is compiled like `replay_pusch`. `replay_pusch.cpp` is not edited: a later tap is added to `pilot_ls.cpp`. [Estimate, layout and record](../../docs/interfaces/PILOT_LS.md)

## `replay_pusch`

```bash
build/ocudu-adapter/receiver/replay_pusch --plan FILE --output-dir OUT [--only OPPORTUNITY_ID]... [--no-grids] [--phy-debug]
build/ocudu-adapter/receiver/replay_pusch --self-test
```

`OUT` must exist and must not hold a result. The plan comes from `python -m ocudu_cuda_isac.episodes.replay_pusch plan`. The tool writes `OUT/ingress/` (one `o1-grid/v1` descriptor and one stored grid per opportunity, as `grid_apply` does), `OUT/radio/` (one `radio/v1` record per planned decode and the decoded bytes of every CRC pass) and `OUT/native.json`. It exits with status 2 for a refused plan, before a processor is created or a file is written, and with status 1 when a decode did not complete or for any other error. A CRC failure is an outcome and not an error. `--only` decodes the named opportunities alone. `--self-test` decodes a two-block grant on the host path and starts no CUDA work. The adapter build runs it and the 12 cases of the host-free dispatch core as CTest cases, 34 with those of `fixtures/` and `replay/`.

| File in `receiver/` | Links OCUDU | Contents |
|---|---|---|
| `dispatch_core.h`, `dispatch_core.cpp` | No | The outcome rule, the byte comparison, the ledger of keys, held grids, owned output buffers and anomalies, and the dispatch of one grid to one decoder |
| `tests/dispatch_core_test.cpp` | No | Unit cases against doubles of a processor |
| `grant_pdu.h` | Headers only | Reader of a grant record, its hash, the coding of the pinned source and the PUSCH PDU |
| `receiver_plan.h` | Headers only | Reader of the plan |
| `replay_pusch.cpp` | Yes | The pinned processors behind the decoder interface, and the tool |

A processor is destroyed only after the workers are joined. A worker is still inside its task for a moment after the decoder's callback, and a processor that is destroyed in that moment is used after it is freed. `replay_pusch` declares a `joined_workers` guard after its processors for that reason. The loopback of `grant_fixture` does not, and a new target that owns a processor must. [Finding](../../docs/interfaces/REPLAY_PUSCH.md#findings)

## `grid_apply`

```bash
build/ocudu-adapter/replay/grid_apply --plan FILE --output-dir OUT [--no-prestore]
build/ocudu-adapter/replay/grid_apply --self-test
```

`OUT` must exist and must not hold a result. The plan comes from `python -m ocudu_cuda_isac.episodes.grid_apply plan`. The tool writes `OUT/ingress/` (one `o1-grid/v1` descriptor and one stored grid per opportunity), `OUT/prestore/` and `OUT/native.json`. It exits with status 2 for a refused plan, before a file is written, and with status 1 for any other error. `--self-test` runs the unit vectors through the pinned grid. The adapter build runs it and the 14 cases of the host-free core as CTest cases, 19 with those of `grant_fixture`.

| File in `replay/` | Links OCUDU | Contents |
|---|---|---|
| `replay_core.h`, `replay_core.cpp` | No | The kernel, the gain rules, the array reader, the owned grids and their states, the session, the readback digest |
| `tests/replay_core_test.cpp` | No | Unit cases against doubles |
| `pinned_grid.h` | Yes | The pinned resource grid behind the grid interface of the core |
| `replay_plan.h` | Headers only | Reader of the plan |
| `grid_apply.cpp` | Yes | The tool |

## `grant_fixture`

```bash
build/ocudu-adapter/fixtures/grant_fixture --grant DIR/grant.json --tb DIR/tb.bin --output-dir OUT \
    [--dmrs-reference crb0|prb0] [--no-loopback]
```

`OUT` must exist and must not hold a result. The adapter writes `x_native.c64`, the grid as little-endian complex64 `[tx_port, symbol, subcarrier]` over the whole carrier, and `native.json` with the PDU it used, the transport block size and coding that OCUDU derived, the grid statistics and the loopback result. It exits with status 2 for an invalid record and with status 1 for any other error. `--dmrs-reference prb0` reproduces the DM-RS reference point of the pinned test, which differs from PUSCH when the bandwidth part does not start at CRB 0. The integration driver `tests/integration/grant_fixture.py` runs the adapter and grades its output.
