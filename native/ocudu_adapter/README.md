# Native adapter for DeepSig-OCUDU

This directory holds the project's native code for the pinned OCUDU receiver and its build wiring. It is built out of tree against the OCUDU build that Sprint 1.1 accepted. Nothing here writes into the OCUDU source or its build directory. Sensing algorithms do not live here. They stay in the host-free core so that the Aerial adapters can reuse them. [Port contract](../../docs/architecture/ABLATIONS.md) | [sprint cards](../../docs/dev/DEVELOPMENT.md)

## Build

```bash
python scripts/native_adapter.py build      # or: make adapter-build
python scripts/native_adapter.py verify
```

The builder reads the compile flags (`flags.make`) and the link line (`link.txt`) of `pusch_e2e_pipeline_test` from the accepted build and writes them to `build/ocudu-adapter/ocudu-pin.cmake`. `CMakeLists.txt` includes that file and defines the interface target `ocudu_pin_fixture`. A target that links it compiles with the pinned test's flags, apart from `-Werror`, and links the same OCUDU archives, so it runs the objects the Sprint 1 gates accepted. The builder then runs the adapter's CTest cases and writes `isac-adapter.json`, a receipt with the hashes of the executables, of every file in this directory and of the OCUDU build's recipe. `verify` refuses a build whose executable, sources or OCUDU build changed after the receipt.

The import needs the `Unix Makefiles` generator of the accepted build, the default of `make native-build`. On a development host without a Sprint 1.1 receipt, pass `--ocudu-build-dir DIR --no-receipt`.

## Contents

| Directory | Ticket | Contents |
|---|---|---|
| `fixtures/` | S2.1 | `grant_fixture`: reads a `pusch-grant/v1` record and its transport block, builds the transmit grid with the transmit chain of the pinned test and writes it in carrier coordinates. It can feed that grid, without noise, to the host PUSCH processor and compare the decoded bytes. [Grant record](../../docs/interfaces/GRANT.md) |
| `replay/` | S2.3 | Allocation-aware channel application |
| `receiver/` | S2.4 | Dispatch of one grid to the CPU and the CUDA PUSCH processor |
| `observer/` | S3.1, S5.2 | Pilot LS and the export of the receiver's own estimate |
| `transport/` | S3.2, S7, S8 | Bounded observation ring and its transports |
| `du/`, `ingress/` | S5.1, S5.4, S6.10 | Scheduler binding, split ingress adapters and the ZeroMQ channel broker |

Only `fixtures/` exists so far. A new subdirectory adds its targets with `add_subdirectory` in `CMakeLists.txt`. A target that needs OCUDU links `ocudu_pin_fixture`. A CPU-only unit target links only its module and its doubles.

## `grant_fixture`

```bash
build/ocudu-adapter/fixtures/grant_fixture --grant DIR/grant.json --tb DIR/tb.bin --output-dir OUT \
    [--dmrs-reference crb0|prb0] [--no-loopback]
```

`OUT` must exist and must not hold a result. The adapter writes `x_native.c64`, the grid as little-endian complex64 `[tx_port, symbol, subcarrier]` over the whole carrier, and `native.json` with the PDU it used, the transport block size and coding that OCUDU derived, the grid statistics and the loopback result. It exits with status 2 for an invalid record and with status 1 for any other error. `--dmrs-reference prb0` reproduces the DM-RS reference point of the pinned test, which differs from PUSCH when the bandwidth part does not start at CRB 0. The integration driver `tests/integration/grant_fixture.py` runs the adapter and grades its output.
