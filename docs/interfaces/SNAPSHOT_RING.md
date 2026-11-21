# Observations on the ring, and the timed path

Sprint 3.2, selector `snapshot_ring`. This page states what the sprint built, what it corrected in the observer of Sprint 3.1, the plan and the files of a run, and how the sprint is accepted. The record, the ring and the view are fixed in the [observation contract](OBSERVATION.md). Tickets: [S3.2](../dev/DEVELOPMENT.md#phase-s3), [requirement of 2026-11-21](../dev/DEVELOPMENT.md#added-by-the-first-observer-run-of-2026-11-21), [process layout](../architecture/RUNTIME.md#process-layout).

**Status.** Delivered on 2026-11-21 and checked on a development host. Not yet run on the Spark: `make accept-s3.2` is the gate.

## What the sprint builds

| Part | Where | Level |
|---|---|---|
| The record `obs/v1` and the ring `obs-ring/v1`: producer, consumer, control channel, counted system calls | `native/obs_ring/` | C++17 on the CPU, the standard library and POSIX. No OCUDU, no CUDA. |
| `obs_tail`, a consumer that keeps what it reads, and `obs_feed`, a producer of known records | `native/obs_ring/tools/` | The same |
| Producer glue: the observer writes its estimate into the slot it acquired | `native/ocudu_adapter/transport/` | C++17, host-free |
| The timed path of the observer, with slots bound before the run | `native/ocudu_adapter/observer/observer_core` | C++17, host-free |
| The hash count and the evidence mode of the two accepted cores | `native/ocudu_adapter/replay/replay_core`, `receiver/dispatch_core` | Amended, see below |
| `pilot_ls` as the producer of a ring, in the modes `audit` and `timed` | `native/ocudu_adapter/observer/pilot_ls.cpp` | Adapter C++, links OCUDU |
| The reference reader | `src/ocudu_cuda_isac/sensing/obs.py` | Python standard library |
| The plan of version 2, the reader of a run, the account | `src/ocudu_cuda_isac/episodes/snapshot_ring.py` | Python standard library |

## What was corrected from Sprint 3.1

The accepted Sprint 3.1 run measured 1,471 us for one observation, 99.3 percent of it SHA-256. "Before" is the accepted Sprint 3.1 tree.

| | Before | Now, in the mode `timed` |
|---|---|---|
| Digest of the grid after the observer's read | 274,176 bytes hashed per observation | None. The observation is tied to its grid by the ticket of the fill: index and generation. The ticket is checked before the first row and after the last. |
| Digest of the payload and of the mask | Two per observation | None. `digest_kind` is 0, and the evaluator digests what the consumer kept, after the run. |
| Join keys | Three SHA-256 per observation | Hashed once, when the plan is bound |
| Where the estimate goes | A buffer the observer allocates, then files | Straight into the slot of the ring that was acquired for it. No copy, no allocation. |
| Readback digest when a grid becomes ready, digests at admission, after the decode and at retirement | Seven per opportunity in the harness | None. A grid and a decode are owned by the ticket of the fill. |
| Files and records | Written as each opportunity is done | Written after the last opportunity. The stored grids and the decoded bytes are digested there, on the evaluator side. |
| Hand-off to a second process | None | One `obs/v1` record on the ring, or a counted drop |

The mode `audit` is the run of Sprint 3.1 with every digest, and additionally publishes each audited observation on the ring. One arithmetic serves both modes: `observe` and `observe_into` call the same `estimate_into`.

**The two amended cores.** `replay_core` and `dispatch_core` were accepted in Sprint 2 and pinned by hash. The requirement of 2026-11-21 names both changes: the hash counts its own work (`isac_replay::sha256_count`), and a grid becomes ready, is admitted and is resolved without a digest (`evidence::timed`). Both are additive and every default is the accepted behavior. The consequence is that `replay_pusch`, `grid_apply` and `pilot_ls` have new bytes in this tree, while `grant_fixture`, which links neither core, keeps its own. The byte-for-byte check of the Sprint 3.1 gate therefore holds only against a Sprint 2.4 run of the same tree. The gate of this sprint proves the equivalence by behavior instead: in the mode `audit` every radio record and decoded file is that of the accepted Sprint 2.4 run, and every payload has the SHA-256 that the accepted Sprint 3.1 run recorded. The unit gate pins the new hashes of the four amended files and the old hashes of the seven untouched ones.

## The plan `pilot-ls-plan/v2`

A plan of version 2 is a plan of Sprint 3.1 with two more objects. `python -m ocudu_cuda_isac.episodes.snapshot_ring plan` writes it. A plan of version 1 still runs as Sprint 3.1 accepted it.

| Field | Content |
|---|---|
| `evidence.mode` | `audit` or `timed` |
| `ring` | `null`, or the ring the run publishes on |
| `ring.stream` | Names the stream. The control channel is `DIR/ring/STREAM.sock`. |
| `ring.window_records`, `ring.burst_records`, `ring.slots` | The records a window holds in place, the records of one uplink burst, and the slot count: window plus burst plus one, rounded up to a power of two. 3, 3 and 8 for the demonstration. A plan with another slot count is refused. |
| `ring.wait` | `spin` or `wait`: the mode the consumer is expected in. The mode it attached with is recorded beside it. |
| `ring.attach` | `required`: the run waits up to `attach_timeout_ms` for a consumer before its first fill and fails without one. `none`: it starts at once. |
| `ring.drain_ms` | The declared cutoff: how long the run waits at its end for the consumer to release what it holds |

Refused before a processor is created, a ring is opened or a file is written: an unknown mode, a ring without a tap, a timed tap without a ring, a slot count that is not the rule's, an unknown wait mode or attach rule, a stream name that is not at most 16 lower-case letters and digits.

## One observation in the mode `timed`

1. `try_acquire`: the released count, the published count, one compare. Without a credit the observation is dropped and counted, no estimate is computed, and the decodes start.
2. The 256 bytes of the header, built when the plan was bound, are copied into the slot.
3. The observer reads the reference rows of the grid and writes the least-squares values and the validity bits into the slot.
4. `publish`: the epoch, the sequence and the publication time of the record, the generation of the slot, and one store of `published`.

Nothing in these four steps hashes, allocates, formats text or enters the kernel. Only when the consumer's waiting flag is set does the producer write the consumer's eventfd, after the store. The service time of the tap runs from before step 1 to after step 4. The estimate is step 3, and the publication is the rest.

**What a timed run still does between a ready grid and the decoder's callback.** The ledger of the staged harness and the processor wrapper allocate per decode and take a mutex, as in Sprint 2.4. They compute no checksum. Removing them is the part of the requirement that is assigned to S4.1. The DU of S5 has no harness: there the path of an observation is the four steps above.

## What a run writes

| File | `audit` | `timed` |
|---|---|---|
| `ingress/`, `radio/`, `native.json` | As Sprint 3.1, written as each opportunity is done | The same records, written after the last opportunity. The digests in them were taken after the drain. |
| `observer/` | As Sprint 3.1 | Absent. The estimate exists in the slot only, and the consumer keeps it. |
| `ring/publications.jsonl` | One `obs-publication/v1` line per filled grid: `published` with its epoch, sequence, depth, size, grid ticket and times, or `dropped` with the reason `no_credit` | The same |
| `ring/ring.json` | The ring: definition, sizes, wait mode planned and attached, epochs, the counts, what was pending at the cutoff | The same |
| `native.json`, `evidence` | The mode, and the digests, hash calls and hashed bytes between the first fill and the drain | The same, and all three must be zero |
| Consumer directory | `consumer.json` and one file per record, written by `obs_tail` or by the reference reader | The same |

The report has the schema `pilot-ls-native/v2`. `snapshot_ring.open_output` verifies either mode with the readers of Sprints 2.4 and 3.1 and checks the account against its lines.

## The account

Every eligible observation, one per filled grid, ends in exactly one state.

| State | Counted by |
|---|---|
| Published and released | The consumer advanced its released count past it |
| Published and reclaimed | Its epoch closed before it was released: the consumer left or was killed, and a new one attached |
| Published and pending at the cutoff | Not released when the drain ended |
| Dropped | No credit at its admission |

`eligible = published + dropped` and `published = released + reclaimed + pending` must hold, with no cancelled slot. A released count that moves backwards or passes the published count is a fault: the last accepted count stands, nothing is reused on its strength, and the consumer's channel is closed at the next `service()`.

## The gate

`make accept-s3.2` needs the accepted Sprint 3.1 run under `results/runs/`. That run names the plan of the reference pair, and the Sprint 2.4 run that plan lies in holds the decodes without an observer. The driver `tests/integration/snapshot_ring.py` grades ten cases.

| Case | What must hold |
|---|---|
| `control` | The Sprint 3.1 run is accepted, its pair verifies, and the Sprint 2.4 records beside its plan were written without a tap |
| `audit` | 48 of 48 payloads and masks have the SHA-256 of the accepted Sprint 3.1 run, from grids with that run's readback digest. Every radio record and decoded file is the control's. The consumer's records hold the bytes of the observer's files. The account balances with no drop. |
| `timed` | Zero digests, hash calls and hashed bytes between the first fill and the drain. Zero system calls between an admission and its publication, a spinning consumer, no eventfd write. Every radio record and decoded file is the control's. The payloads the consumer kept, hashed after the run, are the audited ones, and no record carries a digest. The tap's service time is at or below 150 us in the mean and at the largest value. |
| `wait` | The same with a consumer that sleeps on its eventfd |
| `reference-reader` | The standard-library reader keeps the records the native consumer kept: header fields, host block, validity bits and payload |
| `slow-consumer` | A consumer that works 5 ms on each record: observations are dropped at once and counted, the ring is never overwritten, every decode is the control's, the account balances |
| `terminated-consumer` | A consumer that ends without a word after 4 records, holding a credit, and one that takes over: two epochs, the second from sequence 0, the abandoned records reclaimed, every decode the control's |
| `tap-off` | The timed harness without the tap: no observation, no hash, the control's decodes |
| `refusals` | Seven plan rules, refused by the plan check and by the tool |
| `backends` | Every decode requested from the CUDA processor ran its resident path |

Recorded and not graded: the parts of an audited observation beside the breakdown of 2026-11-21 (1,356 us for the digest of the grid and 97 us for the digest of the payload are expected), the estimate and the publication of a timed observation apart, the consumer's delay from the publication store to its read at the median, the 99th and the 99.9th percentile and the largest value in both wait modes, and the published figures of the supervised ring of the OCUDU dApp platform with `"graded": false`. The expected tap is about 10 us before the publication, from 9.7 us measured on the Spark without a hash. A mean above 20 us is named in the gate's output and needs an explanation before the number is used.

The pair holds 48 observations per run. A 99.9th percentile over 47 values is the largest value. The tail statement of a paper needs the paced runs of S4.1 and the calibration of S4.6.

The gate runner adds its own lines: the unit suite and the unit gate, the adapter build with its CTest cases, the bytes of the fixture adapter, and a second reader that recovers one timed record and compares the digest of its payload, by `sha256sum`, with the payload file of the audited run and of the accepted Sprint 3.1 run.

## Checked on a development host

Before the Spark run, on an x86_64 host (Ubuntu 24.04, GCC 13.3.0, CMake 3.28.3, Python 3.13) without an OCUDU build and with two cores.

| | Development host |
|---|---|
| Ring | 16 of 16 native cases. The same cases pass under the thread sanitizer and under the address and undefined-behavior sanitizers without a report. |
| Producer glue | 6 of 6 native cases: one timed observation and one publish leave the hash count, the allocation count and the system-call count unchanged, and the payload in the slot is the audited payload bit for bit |
| Amended cores | The 15 cases of the channel application core, the 13 of the dispatch core and the 12 of the observer core pass, one new case each |
| Every host-free target in the adapter's order | 64 CTest cases, no compiler warning |
| Both readers | The reference reader and `obs_tail` read the same records from `obs_feed`, 40 of 40 equal. The definition text and its hash are equal in both languages. |
| Real processes | `obs_feed` with `obs_tail` in both wait modes, with the reference reader, with a consumer of 5 ms per record (41 of 60 dropped, none pending) and with a killed consumer and its successor (two epochs, 8 reclaimed, 52 released): every account balances. The driver's own launcher ran these five patterns. |
| `pilot_ls.cpp` | Passes a syntax check against the headers of the pinned OCUDU source, with and without `ENABLE_CUDA`. Not linked, not run. |
| Integration driver | 10 of 10 cases and 40 of 40 checks against the double. Each of 15 injected faults fails the check that names it. |
| Gate runner | 9 tests against tool doubles |
| Ring alone | Admission and the two generation stores: 26 ns at the median. With a copy of 19,584 bytes: 122 ns. One clock read: 21 ns. These are figures of this host and are not projected to the Spark. |

The double computes each estimate with the Python reference and models the ring, so the driver run shows that the checks are sound and nothing about the native tool. Open until the Spark runs the gate: `pilot_ls` links and its self-test passes with the timed section, the timed loop and its deferred evidence write what the readers verify, and the tap's cost on the pinned grid.

## Accepted run

Not yet run. `make accept-s3.2` on the DGX Spark writes `results/runs/s32-*/summary.txt`, which ends with `Sprint 3.2: ACCEPTED on this DGX Spark.` when every gate passed.
