# Observation record, ring and view

An observation is one receiver-side estimate or grid with the metadata a sensing stage may see. It leaves the receiver through one record format, whatever carries it, and it reaches the sensing core as one view, whatever host produced it. This page fixes the record, the shared-memory ring that carries it between two processes on one host, and the view the native core consumes. Tickets: [S3.1, S3.2, S3.6](../dev/DEVELOPMENT.md#phase-s3), [S5.2, S5.5](../dev/DEVELOPMENT.md#phase-s5), [S7](../dev/DEVELOPMENT.md#phase-s7), [S8.1](../dev/DEVELOPMENT.md#phase-s8), [S10.2](../dev/DEVELOPMENT.md#phase-s10) and [S11.2](../dev/DEVELOPMENT.md#phase-s11). [Code levels](../architecture/RUNTIME.md)

## One record, several carriers

| Carrier | Between | Ticket |
|---|---|---|
| Ring `obs-ring/v1` in host shared memory | Receiver process and sensor process on one host (M1, and the host-staged bridge of M3) | S3.2, S8.1 |
| Socket frames | Two hosts (M2) | S7.1 |
| Registered host buffers over verbs | Two hosts (M2) | S7.2 |
| No carrier: the adapter builds the view in its own process | Aerial Framework module, E3 dApp | S10.2, S11.2 |

The record header and the payload bytes are identical on every carrier. A frame adds its own length and checksum around them. `native/obs_ring/` holds the record definition, the ring and the control channel. It is C++17 on the CPU with no OCUDU include and no CUDA, so the receiver adapter links it as the producer and the sensor links it as the consumer. `native/obs_ring/include/obs/record.h` is the byte-level definition once S3.2 lands. This page fixes the fields and their meaning.

## The record `obs/v1`

Little-endian, fixed-width fields, no pointers. The header has one fixed size per version.

| Group | Fields | Meaning |
|---|---|---|
| Version | `schema`, `header_bytes` | A reader refuses a version it does not implement |
| Join keys | `epoch`, `sequence`, `episode`, `opportunity`, `grant` | Opaque integers used only for joins. `sequence` rises by one per published record within an epoch. No key carries truth, a class or a scene name. |
| Radio identity | `cell`, `ue`, `absolute_slot` | The slot counts from the epoch start and does not wrap with the frame number |
| What it is | `stage`, `producer_steps`, `source`, `layout`, `dtype`, `scale` | `stage` is O1, O2 or O3. `producer_steps` is a bit set of the steps the producer applied: pilot LS, frequency interpolation, smoothing, time interpolation. `source` names the sensing source, `layout` the payload layout below, `dtype` the element type. `scale` is the reference amplitude the producer used. |
| Shape | `rank`, `shape[]`, `strides[]` | In elements. A reader may refuse strides other than the layout's contiguous ones. |
| Allocation | `first_subcarrier`, `subcarrier_count`, `dmrs_symbols`, `rx_count`, `layer_count`, `antenna_map[]` | Carrier coordinates as in the [grant record](GRANT.md). `dmrs_symbols` is a bit set of symbol indices. `antenna_map[i]` is the array element behind receive plane `i`. |
| Validity | `mask_bytes` | A bit per payload element, in payload order, follows the header when the length is not zero. An inactive pilot has a cleared bit. A measured zero has a set bit. |
| Time | `time_unit`, `t_first_sample`, `t_last_sample`, `t_ready_ns`, `t_publish_ns` | Radio time of the first and last contributing sample in the named unit (T_c on DeepSig-OCUDU, TAI nanoseconds on Aerial). Host monotonic time of the producer's completion event and of publication. |
| Payload | `payload_bytes`, `digest_kind`, `digest` | The digest is SHA-256 or absent. A timed run may leave it out and let the evaluator digest the archived payload. |
| Host | `host_bytes` | An opaque block for fields only one host has. The sensing core never reads it. |

Nothing in the record is truth. H, W, the original transport block and the labels stay with the evaluator.

## Payload layouts

A layout names the element order of a payload as its producer holds it, so that the producer never reorders data in the receive path. The native core has one ingest unit per layout, which gathers the payload into the core's cube on the device.

| Layout | Producer | Element order and type | At the demonstration profile |
|---|---|---|---|
| `ocudu_o3_fp16` | Pinned OCUDU CUDA receiver, `d_estimates_fp16` (S5.2) | `[dmrs_symbol][prb][rx_port][subcarrier_in_prb]`, complex FP16 pairs, with a layer axis before the subcarrier for more than one layer | 2 x 51 x 4 x 12 values, 19,584 bytes |
| `ocudu_o2_comb_c64` | Observer pilot LS (S3.1) | `[rx_port][dmrs_symbol][pilot]`, complex64 | 4 x 2 x 306 values, 19,584 bytes |
| `ocudu_o1_cbf16` | Pinned OCUDU resource grid | `[rx_port][symbol][subcarrier]`, complex bf16 | 4 x 14 x 612 values, 137,088 bytes |
| `aerial_hest_c64` | Aerial L1, PUSCH DMRS estimates in E3 shared memory | `[dmrs][subcarrier][rx_antenna][layer]` per UE group, complex float32 | Set by the Aerial cell |
| `aerial_iq_fp16` | Aerial L1, fronthaul I/Q in E3 shared memory | `[rx_antenna][symbol][prb][subcarrier_in_prb][2]`, FP16 stored as 16-bit integers | Set by the Aerial cell |

The two Aerial layouts are those of `dapps/docs/data_representation.md` in the pinned sample applications. A new producer adds a layout and its ingest unit, and changes no kernel.

## The ring `obs-ring/v1`

One producer and one consumer share one segment of POSIX shared memory. A second consumer gets its own segment, and the producer's extra copy is counted. Each observation stream has its own segment as well: in the demonstration, O3 for `sense_host` and O2 for the reference sensor of the delay-Doppler branch.

| Rule | Detail |
|---|---|
| Single writer | Only the producer writes the segment. The consumer maps it read-only. The observation path is read-only by construction, and a faulty consumer cannot alter what the receiver published. |
| Fixed size | The segment is created once with a header, a slot count and a slot size taken from the run profile. Nothing is allocated after start. The header carries a magic value, the layout version, the sizes, the producer's process, the epoch and a hash of the record definition. A consumer built from another definition is refused. |
| Slot | A generation counter, the record header, the validity bits and the payload. The producer raises the generation to an odd value before it writes a slot and to the next even value when the slot is complete. |
| Publication | The producer publishes a slot only after the completion event of the copy that filled it. It then sends `PUBLISH` with the slot and its generation. It never waits: a full ring, an absent consumer or a full control channel is a counted drop, and the radio job continues. |
| Read | The consumer checks the generation named by `PUBLISH`, reads the slot, and checks the generation again. A mismatch is a counted stale read, and no decision uses that record. |
| Credit | The consumer sends `RELEASE` with the slot and generation when its work on the record is complete. The producer reuses a slot only after its release. A release for an old generation or an old epoch restores nothing. |
| Epoch | A consumer that attaches or reattaches starts a new epoch. The producer reclaims the slots of the old epoch once the old consumer's channel is closed. A consumer that hangs keeps its credits, the ring fills, and drops are counted. |
| Control channel | A local sequenced-packet socket in the run directory, with fixed-size messages `HELLO`, `PUBLISH`, `RELEASE` and `BYE`. The producer sends without blocking. |
| Notification | The socket message is the default. A consumer that polls the published count in the segment header is a second mode, compared in S5.5. |
| Memory | The payload area is host memory. Where the host supports it, the producer registers it with CUDA so that its asynchronous copy lands in the slot, and the consumer registers its read-only mapping for ingest. The effective path is read back and recorded as the `memory_path` entry of the run. |

Every eligible observation ends in one of three counted states: published and released, dropped with its reason, or pending at the declared cutoff. S3.2 owns that accounting.

Two readers implement the consumer side and pass the same unit vectors. `ocudu_cuda_isac.sensing.obs` is the reference reader, written with the Python standard library, so that the `reference` runtime needs no native binding. The consumer in `native/obs_ring/` is the one `sense_host` links.

## The observation view

The native core takes a view and nothing else. An adapter fills it and keeps the memory valid until the core signals that ingest of that view is complete.

| Field | Content |
|---|---|
| `record` | The `obs/v1` header |
| `payload`, `mask` | Addresses of the payload and of the validity bits |
| `memory` | `host_shared`, `host_pinned` or `device`, with the device and the stream that owns a device payload |
| `release` | The adapter's callback, invoked once after ingest. On the ring it sends `RELEASE`. |

| Host | The view is built from | `memory` |
|---|---|---|
| DeepSig-OCUDU, one host | A ring slot | `host_shared` |
| DeepSig-OCUDU, two hosts | A received frame | `host_pinned` |
| Standalone E3 Agent, Aerial-OCUDU | The E3 shared-memory row that an indication references | `host_shared` |
| Aerial Framework | The inner receiver's estimate tensor | `device` |

A `device` view lets an adapter hand over a buffer with no copy. An adapter in the receiver's own process could use it for an in-process placement. That placement is not scheduled. It would be a new entry on the placement axis with its own ticket.

## Decisions

The core returns a decision record, which `sense_host` appends to `decisions.jsonl`. Beside the fields of the [shared interfaces](../dev/DEVELOPMENT.md#interfaces-shared-across-tickets) it names the `runtime` entry, the engine backend, the plan with the hash of each unit, the core build from `isac-core.json`, and the timing marks of [RUNTIME.md](../architecture/RUNTIME.md#timing-boundaries). The Python sensor of the `reference` runtime writes the same record.

## What the unit tests show

| Case | Expected |
|---|---|
| Capacity two, no release: publish A and B, then C | A and B admitted, C dropped and counted, the producer returned without waiting |
| The producer overwrites its scratch buffer after publishing B | B in the ring is unchanged |
| A read that straddles a rewrite of the slot | Counted as a stale read. No record is delivered. |
| `RELEASE` twice, or for an old generation | One credit restored, never two |
| The consumer exits while holding a credit | New epoch, slots reclaimed after the channel closes, no credit from the old epoch accepted |
| A consumer built with another record definition | Refused at attach, with both hashes in the message |
| The consumer tries to write to its mapping | The write is refused |
| The same slot read by the reference reader and by the native consumer | Equal header fields and equal payload bytes |
| Element `[1][3][2][5]` of an `ocudu_o3_fp16` payload | DMRS symbol 1, receive port 2, subcarrier 41 of the allocation |
