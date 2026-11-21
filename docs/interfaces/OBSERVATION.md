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

Sprint 3.1 writes these fields as one JSON record per observation, `obs-descriptor/v1`, with the groups `keys`, `radio`, `what`, `shape`, `allocation`, `validity`, `time`, `payload` and `host`, beside the payload and the validity bits as files. The binary header of Sprint 3.2 packs the same values. In the staged harness the join keys are the leading 64 bits of the SHA-256 of the opaque identifiers, and `time_unit` is T_c from slot 0 of the epoch. [Record and layout of O2](PILOT_LS.md#the-record-obs-descriptorv1)

## Payload layouts

A layout names the element order of a payload as its producer holds it, so that the producer never reorders data in the receive path. The native core has one ingest unit per layout, which gathers the payload into the core's cube on the device.

| Layout | Producer | Element order and type | At the demonstration profile |
|---|---|---|---|
| `ocudu_o3_fp16` | Pinned OCUDU CUDA receiver, `d_estimates_fp16` (S5.2) | `[dmrs_symbol][prb][rx_port][subcarrier_in_prb]`, complex FP16 pairs, with a layer axis before the subcarrier for more than one layer | 2 x 51 x 4 x 12 values, 19,584 bytes |
| `ocudu_o2_comb_c64` | Observer pilot LS (S3.1) | `[rx_port][dmrs_symbol][pilot]`, complex64. The pilot axis is the sorted set of pilot subcarriers of the source, and the record gives its first subcarrier and its step. [Layout](PILOT_LS.md#layout-and-validity) | 4 x 2 x 306 values, 19,584 bytes |
| `ocudu_o1_cbf16` | Pinned OCUDU resource grid | `[rx_port][symbol][subcarrier]`, complex bf16 | 4 x 14 x 612 values, 137,088 bytes |
| `aerial_hest_c64` | Aerial L1, PUSCH DMRS estimates in E3 shared memory | `[dmrs][subcarrier][rx_antenna][layer]` per UE group, complex float32 | Set by the Aerial cell |
| `aerial_iq_fp16` | Aerial L1, fronthaul I/Q in E3 shared memory | `[rx_antenna][symbol][prb][subcarrier_in_prb][2]`, FP16 stored as 16-bit integers | Set by the Aerial cell |

The two Aerial layouts are those of `dapps/docs/data_representation.md` in the pinned sample applications. A new producer adds a layout and its ingest unit, and changes no kernel.

## The ring `obs-ring/v1`

One producer and one consumer share one ring in POSIX shared memory. The ring is the only queue between the receiver and a sensor, and it is first in, first out. A second consumer gets its own ring, and the producer's extra copy is counted. Each observation stream has its own ring as well: in the demonstration, O3 for `sense_host` and O2 for the reference sensor of the delay-Doppler branch. The queue was reviewed on 2026-11-21 against the published queues of the two reference stacks. [Process layout](../architecture/RUNTIME.md#process-layout)

| Rule | Detail |
|---|---|
| One writer per mapping | The producer creates the ring segment and is its only writer. The consumer maps it read-only. The consumer creates a cursor page of two cache lines and is its only writer. The producer maps it read-only. The observation path is read-only by construction, and a faulty consumer cannot alter what the receiver published. |
| Fixed size | The segment is created once with a header, a slot count and a slot size taken from the run profile. The slot count is a power of two. Nothing is allocated after start. The header carries a magic value, the layout version, the sizes, the producer's process, the epoch and a hash of the record definition. A consumer built from another definition is refused. |
| Order | First in, first out. The record with `sequence` n lies in slot n modulo the slot count. The producer publishes in sequence order. The consumer reads in sequence order and releases in sequence order. The ring offers no other order. |
| Counts | `published`, in the ring header, is the number of records the producer has completed in this epoch. `released`, in the cursor page, is the number the consumer has finished with. Each is a 64-bit count on its own cache line, written by one side with a single aligned store. `published - released` records are held. A credit is a free slot: the slot count minus the records held. |
| Slot | A generation counter, the record header, the validity bits and the payload. The producer raises the generation to an odd value before it writes a slot and to the next even value when the slot is complete. |
| Admission | Before it writes, the producer loads `released` and compares. With no credit the new observation is dropped and counted, nothing in the ring is overwritten, and the radio job continues. Admission is two loads and a compare. |
| Publication | The producer fills the slot it acquired and, after the completion event of the copy that filled it, stores `published`. That store is the publication. Between the admission and that store the producer makes no system call, takes no lock and allocates nothing. It never waits: a full ring or an absent consumer is a counted drop. |
| Read | The consumer loads `published` and reads the slots up to it in order. For each it checks the generation, reads, and checks the generation again. A mismatch is a counted stale read, and no decision uses that record. |
| Release | The consumer stores `released = n` when its work on every record below n is complete. A window that is read in place holds its oldest record, so releases follow the window. The producer treats a `released` that moves backwards or passes `published` as a fault of the consumer: it closes the epoch and counts it. |
| Depth | The run profile sets the slot count: the records a plan holds in place, plus one uplink burst, plus one, rounded up to a power of two. For the O3 stream of the demonstration that is 3 + 3 + 1, so 8 slots. Depth bounds the age of what the consumer can still read. A deeper ring adds age and no safety, so a backlog is never a reason to enlarge it. |
| Wait mode | `spin`: the consumer polls `published` on its own core with a pause between loads, and the producer never enters the kernel. `wait`: the consumer sets a waiting flag in its cursor page, checks `published` once more and sleeps on an eventfd. The producer loads the flag after its publication store and writes the eventfd, without blocking, only while the flag is set. Timed runs use `spin`. Gates and tests that reserve no core use `wait`. The effective mode is recorded with the run. |
| Epoch | A consumer that attaches or reattaches starts a new epoch with a new cursor page. The producer reclaims the slots of the old epoch once the old consumer's channel is closed. A consumer that hangs stops releasing, the ring fills, and drops are counted. |
| Control channel | A local sequenced-packet socket in the run directory, with fixed-size messages `HELLO`, `EPOCH` and `BYE`. It passes the descriptors of the cursor page and of the eventfd, and it detects a consumer that has gone. It carries no message per observation. |
| Memory | The payload area is host memory. Where the host supports it, the producer registers it with CUDA so that its asynchronous copy lands in the slot, and the consumer registers its read-only mapping for ingest. The effective path is read back and recorded as the `memory_path` entry of the run. |

Every eligible observation ends in one of three counted states: published and released, dropped with its reason, or pending at the declared cutoff. S3.2 owns that accounting.

**What the producer pays.** In the receiver's path one publication is the admission, the write of a slot by a copy that had to be made in any case, and one store. The reason is measured. The supervised ring of the OCUDU dApp platform, also a shared-memory ring beside a CUDA PHY on a GB10, makes three system calls per publication. Its authors report 1.97 us at the median for a 68 kB publication on a quiet host and with a cell on the air, but 2.34 us against 167 us at the 99.9th percentile, and a wake of the sleeping consumer of 99 us against 1,620 us at the 99.9th percentile. The bytes cost the same in both conditions. The tails come from the kernel and the scheduler. This ring therefore keeps system calls out of the producer, and the consumer of a timed run does not sleep. S3.2 measures both wait modes on the Spark, and the published figures are recorded beside them and not graded. [O'Shea et al.](https://arxiv.org/abs/2609.07805)

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
| The same `released` count stored twice, or a count from an old epoch | One credit restored per record, never two |
| `released` moves backwards, or passes `published` | The epoch is closed and the fault is counted. No slot is reused on the strength of that count. |
| Three publications, then the consumer reads | Sequences n, n+1 and n+2, from consecutive slots, in that order |
| 1,000 publications to a spinning consumer | The producer made no system call between admission and publication, by the count of the library's own system-call shim |
| A waiting consumer, flag set and then cleared | One eventfd write for the publication that found the flag set, none afterwards |
| The consumer sets the flag while the producer publishes | The record is seen at the consumer's second check or by the eventfd. No publication is slept through. |
| The consumer exits while holding a credit | New epoch, slots reclaimed after the channel closes, no credit from the old epoch accepted |
| A consumer built with another record definition | Refused at attach, with both hashes in the message |
| The consumer tries to write to the ring, or the producer to the cursor page | The write is refused |
| The same slot read by the reference reader and by the native consumer | Equal header fields and equal payload bytes |
| Element `[1][3][2][5]` of an `ocudu_o3_fp16` payload | DMRS symbol 1, receive port 2, subcarrier 41 of the allocation |
