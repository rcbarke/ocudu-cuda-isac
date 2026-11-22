# Observation record, ring and view

An observation is one receiver-side estimate or grid with the metadata a sensing stage may see. It leaves the receiver through one record format, whatever carries it, and it reaches the sensing core as one view, whatever host produced it. This page fixes the record, the shared-memory ring that carries it between two processes on one host, and the view the native core consumes. Tickets: [S3.1, S3.2, S3.6](../dev/DEVELOPMENT.md#phase-s3), [S5.2, S5.5](../dev/DEVELOPMENT.md#phase-s5), [S7](../dev/DEVELOPMENT.md#phase-s7), [S8.1](../dev/DEVELOPMENT.md#phase-s8), [S10.2](../dev/DEVELOPMENT.md#phase-s10) and [S11.2](../dev/DEVELOPMENT.md#phase-s11). [Code levels](../architecture/RUNTIME.md)

## One record, several carriers

| Carrier | Between | Ticket |
|---|---|---|
| Ring `obs-ring/v1` in host shared memory | Receiver process and sensor process on one host (M1, and the host-staged bridge of M3) | S3.2, S8.1 |
| Socket frames | Two hosts (M2) | S7.1 |
| Registered host buffers over verbs | Two hosts (M2) | S7.2 |
| No carrier: the adapter builds the view in its own process | Aerial Framework module, E3 dApp | S10.2, S11.2 |

The record header and the payload bytes are identical on every carrier. A frame adds its own length and checksum around them. `native/obs_ring/` holds the record definition, the ring and the control channel. It is C++17 on the CPU with no OCUDU include and no CUDA, so the receiver adapter links it as the producer and the sensor links it as the consumer. `native/obs_ring/include/obs/record.h` is the byte-level definition, built in Sprint 3.2, and `ocudu_cuda_isac.sensing.obs` holds the same table for the reference reader. This page fixes the fields and their meaning. [What Sprint 3.2 built and how it is accepted](SNAPSHOT_RING.md)

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

Sprint 5.2 produces the stage O3 in the layout `ocudu_o3_fp16` from the receiver's own device buffer, with the steps pilot LS and frequency interpolation, and smoothing for more than one resource block. It publishes on a ring of its own, beside the O2 stream. [The receiver's own estimate](NATIVE_ESTIMATE.md)

Nothing in the record is truth. H, W, the original transport block and the labels stay with the evaluator.

Sprint 3.1 writes these fields as one JSON record per observation, `obs-descriptor/v1`, with the groups `keys`, `radio`, `what`, `shape`, `allocation`, `validity`, `time`, `payload` and `host`, beside the payload and the validity bits as files. The binary header of Sprint 3.2 packs the same values. In the staged harness the join keys are the leading 64 bits of the SHA-256 of the opaque identifiers, hashed when the plan is bound and not per slot, and `time_unit` is T_c from slot 0 of the epoch. [Record and layout of O2](PILOT_LS.md#the-record-obs-descriptorv1)

### Byte layout of the header

The header is 256 bytes, little-endian, with every field naturally aligned and no padding between fields. A record is the header, then `host_bytes` of the host block, `mask_bytes` of validity bits and `payload_bytes` of payload. A file and a frame carry the four parts without padding. A slot of the ring places each part at a fixed offset.

| Offset | Bytes | Field | Offset | Bytes | Field |
|---|---|---|---|---|---|
| 0 | 4 | `schema`, 1 for `obs/v1` | 128 | 4 | `dmrs_symbols`, bit n for symbol n |
| 4 | 4 | `header_bytes`, 256 | 132 | 2 | `rx_count` |
| 8 | 8 | `epoch`, set by the carrier | 134 | 2 | `layer_count` |
| 16 | 8 | `sequence`, set by the carrier | 136 | 32 | `antenna_map`, 16 values of 16 bits |
| 24 | 8 | `episode` | 168 | 4 | `mask_bytes` |
| 32 | 8 | `opportunity` | 172 | 4 | `payload_bytes` |
| 40 | 8 | `grant` | 176 | 4 | `host_bytes` |
| 48 | 8 | `absolute_slot` | 180 | 1 | `time_unit`: 1 T_c, 2 TAI nanoseconds |
| 56 | 4 | `cell` | 181 | 1 | `digest_kind`: 0 none, 1 SHA-256 |
| 60 | 4 | `ue` | 182 | 1 | `evidence`: 1 audit, 2 timed |
| 64 | 1 | `stage`: 1 O1, 2 O2, 3 O3 | 183 | 1 | `port` |
| 65 | 1 | `source`: 1 `pusch_dmrs` | 184 | 8 | `t_first_sample` |
| 66 | 1 | `layout`, in the order of the table below from 1 | 192 | 8 | `t_last_sample` |
| 67 | 1 | `dtype`: 1 complex64, 2 complex FP16, 3 complex bf16 | 200 | 8 | `t_ready_ns` |
| 68 | 4 | `producer_steps`: bit 0 pilot LS, 1 frequency interpolation, 2 smoothing, 3 time interpolation | 208 | 8 | `t_publish_ns`, set by the carrier |
| 72 | 8 | `scale`, a double | 216 | 32 | `digest` |
| 80 | 4 | `rank` | 248 | 4 | `valid`, set validity bits |
| 84 | 16 | `shape`, four values of 32 bits | 252 | 4 | reserved, zero |
| 100 | 16 | `strides`, four values of 32 bits | | | |
| 116 | 4 | `first_subcarrier` | | | |
| 120 | 4 | `subcarrier_step`, 0 when the axis is not regular | | | |
| 124 | 4 | `subcarrier_count` | | | |

Both readers hash one text that lists every field of the header, of the segment, of the slot prefix and of the cursor page as `name:offset:bytes`, with FNV-1a over 64 bits. The value names the definition a reader was built from: `6f877a8710c433dd` for this version. It is compared when a consumer attaches and is never computed per record. A record carries no checksum of its own.

**The host block of the staged harness** is 40 bytes: the kind of the host (1), the position of the opportunity in the plan, the index of the grid, the frame number, the generation of the grid, the slot, four reserved bytes and the host monotonic time before the observer read its first row. The index and the generation are the ticket of the fill, which ties a timed observation to its grid. A sensing core never reads the block.

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
| Epoch | A consumer that attaches or reattaches starts a new epoch with a new cursor page. The one exception is the first consumer of a ring that attaches before the first publication: it takes the epoch the ring was created with. The producer reclaims the slots of the old epoch once the old consumer's channel is closed and a new owner exists: what was published and never released is counted as reclaimed. Until then a ring without a consumer keeps the last released count that it accepted, fills, and drops. A consumer that hangs stops releasing, the ring fills, and drops are counted. |
| Control channel | A local sequenced-packet socket in the run directory, with fixed-size messages `HELLO`, `EPOCH` and `BYE` of 64 bytes. `HELLO` passes the descriptors of the cursor page and of the eventfd, `EPOCH` the descriptor of the segment. Each side passes a read-only descriptor of the memory it writes, so the kernel refuses the other side a writable mapping. The channel detects a consumer that has gone. It carries no message per observation. The producer reads it in `service()`, which its owner calls outside the window between an admission and its publication: the staged harness calls it once per opportunity, after the decodes. A run directory may be longer than a socket address holds, so both sides reach the socket through a descriptor of its directory. |
| Attach when ready | A producer may publish from the moment `EPOCH` is sent. A consumer therefore allocates and touches its memory first and attaches last. It knows the sizes from the plan of the run. |
| Counted system calls | Every system call of the library passes one shim that counts it. The producer adds up the calls between each admission and its publication store, and a monitor reads that count in the segment header. The monotonic clock is read in user space where the host has a vDSO clock and is not counted. The clock source is recorded with a run. |
| Memory | The payload area is host memory. Where the host supports it, the producer registers it with CUDA so that its asynchronous copy lands in the slot, and the consumer registers its read-only mapping for ingest. The effective path is read back and recorded as the `memory_path` entry of the run. |

Every eligible observation ends in one of three counted states: published and released, dropped with its reason, or pending at the declared cutoff. S3.2 owns that accounting.

**What the producer pays.** In the receiver's path one publication is the admission, the write of a slot by a copy that had to be made in any case, and one store. The reason is measured. The supervised ring of the OCUDU dApp platform, also a shared-memory ring beside a CUDA PHY on a GB10, makes three system calls per publication. Its authors report 1.97 us at the median for a 68 kB publication on a quiet host and with a cell on the air, but 2.34 us against 167 us at the 99.9th percentile, and a wake of the sleeping consumer of 99 us against 1,620 us at the 99.9th percentile. The bytes cost the same in both conditions. The tails come from the kernel and the scheduler. This ring therefore keeps system calls out of the producer, and the consumer of a timed run does not sleep. S3.2 measures both wait modes on the Spark, and the published figures are recorded beside them and not graded. [O'Shea et al.](https://arxiv.org/abs/2609.07805)

Two readers implement the consumer side and pass the same unit vectors. `ocudu_cuda_isac.sensing.obs` is the reference reader, written with the Python standard library, so that the `reference` runtime needs no native binding. The consumer in `native/obs_ring/` is the one `sense_host` links.

### Segment, slot and cursor page

| Part | Offset | Content |
|---|---|---|
| Segment header, 4,096 bytes | 0 | Magic `OBSRING1`, layout version, header size, definition, slot count, slot size, size of the record header, capacities of the host block, the mask and the payload, their offsets in a slot, the producer's process |
| | 64 | `epoch`, on its own cache line |
| | 128 | `published`, on its own cache line |
| | 192 | Counts for a monitor: admitted, dropped, reclaimed, faults, eventfd writes, system calls inside a publication window |
| Slot, a multiple of 64 bytes | 0 | Prefix of 64 bytes: `generation`, the sequence and the epoch of the record in the slot, the queue depth that record met, its size |
| | 64 | The record header, 256 bytes |
| | 320 | The host block, then the validity bits at the next multiple of 8, then the payload at the next multiple of 64 |
| Cursor page, 4,096 bytes | 0 | `released`, on its own cache line |
| | 64 | The waiting flag, the wait mode, magic `OBSCURS1`, the definition, the consumer's process, the layout version |

At the demonstration profile a slot holds 20,288 bytes: 64 of prefix, 256 of record header, 40 of host block and 306 of validity bits, padded to 704, and 19,584 of payload. Eight slots and the header are 166,400 bytes, as the accepted Sprint 3.2 run reports them. The payload of a slot is aligned to 64 bytes, so the observer writes its estimate where the consumer reads it.

### Tools and the archive of a consumer

`native/obs_ring/` builds two tools beside the library. `obs_tail` is a consumer that keeps what it reads: it attaches, reads every record in place, takes the time of each read and writes, after the producer said that no record follows, one file per record with the bytes of the record and `consumer.json` with its counts and the delay of every read from the publication store. It can work a given time on each record, hold a window of records, end without a word after a number of records, and attach only once another process has ended. `obs_feed` is a producer of known records without a receiver: it feeds a reader under development and measures the ring alone on a host. The reference reader offers `python -m ocudu_cuda_isac.sensing.obs tail` with the same archive, and `show` to read one record of an archive back.

## The observation view

The native core takes a view and nothing else. An adapter fills it and keeps the memory valid until the core signals that ingest of that view is complete.

| Field | Content |
|---|---|
| `record` | The `obs/v1` header |
| `payload`, `mask` | Addresses of the payload and of the validity bits |
| `memory` | `host_shared`, `host_pinned` or `device`, with the device and the stream that owns a device payload |
| `release` | The adapter's callback, invoked once after ingest. On the ring it advances the released count of the cursor page. No message is sent. |

| Host | The view is built from | `memory` |
|---|---|---|
| DeepSig-OCUDU, one host | A ring slot | `host_shared` |
| DeepSig-OCUDU, two hosts | A received frame | `host_pinned` |
| Standalone E3 Agent, Aerial-OCUDU | The E3 shared-memory row that an indication references | `host_shared` |
| Aerial Framework | The inner receiver's estimate tensor | `device` |
| DeepSig-OCUDU staged harness, in-process control cell (S5.5) | The receiver's `d_estimates_fp16` | `device` |

A `device` view lets an adapter hand over a buffer with no copy. The in-process control cell of S5.5 uses it: an adapter in the harness's own process hands the core the receiver's estimate buffer with its completion event, and no ring is involved. [Control cell](../architecture/RUNTIME.md#the-in-process-control-cell)

## Decisions

The core returns a decision record, which `sense_host` appends to `decisions.jsonl`. Beside the fields of the [shared interfaces](../dev/DEVELOPMENT.md#interfaces-shared-across-tickets) it names the `runtime` entry, the engine backend, the plan with the hash of each unit, the core build from `isac-core.json`, and the timing marks of [RUNTIME.md](../architecture/RUNTIME.md#timing-boundaries). The Python sensor of the `reference` runtime writes the same record.

## What the unit tests show

The cases below are built: 16 native cases of the ring (`native/obs_ring/tests/`), 6 of the producer glue (`native/ocudu_adapter/transport/tests/`), and the unit gate `tests/unit/test_snapshot_ring.py`, which also lets both readers read one native producer.

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
