# Observation ring

The observation record `obs/v1`, the shared-memory ring `obs-ring/v1` and its control channel: the path by which an estimate leaves the receiver process and reaches the sensor process on one host. Built in Sprint 3.2 and accepted on the DGX Spark on 2026-11-21, run `s32-spark-ecf8-20261121T204425Z`: a publication takes 0.6 us at the median there, and a drop 0.24 us at most. [Observation contract](../../docs/interfaces/OBSERVATION.md) | [what was built and how it is accepted](../../docs/interfaces/SNAPSHOT_RING.md)

It is C++17 on the CPU. It includes nothing from OCUDU and uses no CUDA, so two programs link it without sharing anything else:

| Side | Linked by | Role |
|---|---|---|
| Producer | The OCUDU adapter, `native/ocudu_adapter/transport/` | Creates the ring, writes slots in sequence order, publishes with one store and no system call, counts drops, reads the consumer's released count |
| Consumer | `obs_tail` here, and the sensor in `native/sense_core/host/` from S3.6 on | Maps the ring read-only, reads slots in sequence order between two generation checks, releases in order through its own cursor page |

## Contents

| File | Contents |
|---|---|
| `include/obs/record.h` | The byte-level definition of the record: a header of 256 bytes, its enumerations and the hash of the definition |
| `include/obs/ring.h` | The segment, the slot prefix, the cursor page, the control message, the producer and the consumer |
| `src/ring.cpp` | The implementation, with every system call behind one counted shim |
| `tests/obs_ring_test.cpp` | 16 unit cases, each a CTest case |
| `tools/obs_tail.cpp` | A consumer that keeps what it reads: one file per record and `consumer.json` |
| `tools/obs_feed.cpp` | A producer of known records, for the ring without a receiver. It prints the account and the cost of a publication |

The reference reader `ocudu_cuda_isac.sensing.obs`, written with the Python standard library, reads the same segment and passes the same definition check. The socket and verbs transports of S7 and the bridge of S8.1 carry the same record over other media.

## Build and test

```bash
cmake -S native/obs_ring -B build/obs-ring -DCMAKE_BUILD_TYPE=Release
cmake --build build/obs-ring -j
ctest --test-dir build/obs-ring --output-on-failure
```

The adapter build includes this directory, so `make accept-s3.2` builds it with the flags of the accepted OCUDU build, under `build/ocudu-adapter/obs_ring/`. The unit gate `tests/unit/test_snapshot_ring.py` builds it on its own.

## Rules of the producer

- **Admission.** `try_acquire` loads the released count, compares it with the published count and returns a slot or counts a drop. It never waits.
- **Publication.** One store of the published count. With the wait mode `spin` the producer makes no system call from an admission to its publication. With `wait` it writes an eventfd only when the consumer has set its waiting flag.
- **Control.** `service()` accepts a consumer, starts an epoch and reclaims what the last consumer held. The producer calls it between observations, never inside one.
- **Depth.** `slots_for(window, burst)` is window plus burst plus one, rounded up to a power of two.

## Run the ring alone

```bash
build/ocudu-adapter/obs_ring/obs_feed --socket /tmp/o2.sock --records 2000 --period-us 500 &
build/ocudu-adapter/obs_ring/obs_tail --socket /tmp/o2.sock --wait spin
```
