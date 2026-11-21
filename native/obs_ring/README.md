# Observation ring

Specified, not yet built. This directory will hold the observation record `obs/v1`, the shared-memory ring `obs-ring/v1` and its control channel: the path by which an estimate leaves the receiver process and reaches the sensor process on one host. [Observation contract](../../docs/interfaces/OBSERVATION.md) | [S3.2 `snapshot_ring`](../../docs/dev/DEVELOPMENT.md#phase-s3)

It is C++17 on the CPU. It includes nothing from OCUDU and uses no CUDA, so two programs link it without sharing anything else:

| Side | Linked by | Role |
|---|---|---|
| Producer | The OCUDU adapter, `native/ocudu_adapter/transport/` | Creates the segment, writes slots, publishes without waiting, counts drops, reclaims credits |
| Consumer | The sensor, `native/sense_core/host/` | Maps the segment read-only, reads a slot between two generation checks, returns its credit |

`include/obs/record.h` will be the byte-level definition of the record. The reference reader `ocudu_cuda_isac.sensing.obs`, written with the Python standard library, reads the same segment and must pass the same unit vectors. The socket and verbs transports of S7 and the bridge of S8.1 carry the same record.
