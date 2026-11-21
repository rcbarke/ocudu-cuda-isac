# Paired decodes through the real receiver

Sprint 2.4 (`replay_pusch`) hands the grid that Sprint 2.3 fills to the receiver. For every opportunity the same owned grid, the PDU of the opportunity's grant and the identity of the original transport block go to the host PUSCH processor and to the CUDA PUSCH processor of the pinned source. Each decode ends in one explicit outcome under a key that is unique in the run, and the decoded bytes are compared with the original ones. This closes Sprint 2: a known waveform reaches the native receiver and comes back as the bytes that were sent. [Sprint card](../dev/DEVELOPMENT.md#phase-s2) | [channel application](GRID_APPLY.md) | [grant record](GRANT.md) | [episode entry](EPISODE.md)

**Status.** Accepted on the DGX Spark `spark-ecf8` on 2026-11-21 (UTC), run `s24-spark-ecf8-20261121T083424Z`: 48 gate lines, 8 of 8 cases and 36 of 36 checks, hardware gate closed. All 140 decodes requested from the CUDA processor ran its resident path. `make accept-s2` then accepted Sprint 2, run `s2-spark-ecf8-20261121T083811Z`. [Accepted run](#accepted-run) | [the audit](#the-sprint-2-audit) | [rehearsal](#rehearsal-on-a-development-host)

## Run it

```bash
# Unit gate: the outcome rule, the payload, the plan and the reader with the standard library, and
# the host-free dispatch core built on its own with CMake and a C++17 compiler.
# No OCUDU build, no NumPy, no GPU
python tests/unit/test_replay_pusch.py -v

# Integration gate on a host that passed make accept-s1.2 and make accept-s2.1 to make accept-s2.3
make accept-s2.4

# The four accepted runs of Sprint 2 as one chain. This closes the sprint
make accept-s2

# The steps of the gate, one at a time
make adapter-build
s23=$(ls -d results/runs/s23-* | tail -1)
s22=$(ls -d results/runs/s22-* | tail -1)
python tests/integration/replay_pusch.py --config configs/integration/replay_pusch.json \
    --s22-run "$s22" --s23-run "$s23" --output-dir results/runs/s24-spark-001/replay_pusch

# By hand: a plan from the stored pair, the native tool, a second reader, and the bytes with cmp
python -m ocudu_cuda_isac.episodes.replay_pusch plan --episode "$s22/episode_pair/reference_pair/store" \
    --run-id pair-by-hand --output-dir build/replay-pusch-001
mkdir build/replay-pusch-001/out
build/ocudu-adapter/receiver/replay_pusch --plan build/replay-pusch-001/plan.json --output-dir build/replay-pusch-001/out
python -m ocudu_cuda_isac.episodes.replay_pusch show build/replay-pusch-001/out --sequence 0 --backend cuda
s21=$(ls -d results/runs/s21-* | tail -1)
cmp build/replay-pusch-001/out/radio/0000.cuda.bin "$s21/grant_fixture/demo-51prb-4rx/store/tb.bin"
```

`make accept-s2.4` writes `results/runs/s24-<host>-<UTC time>/` and ends with `Sprint 2.4: ACCEPTED on this DGX Spark.` It rebuilds the adapter and never rebuilds OCUDU. It takes the newest accepted Sprint 2.3 run and the Sprint 2.2 run whose stimulus that run applied, so the grids it decodes can be compared with the grids Sprint 2.3 stored. `--s23-run` and `--s22-run` name other runs, and the second must be the one the first applied. On a host whose OCUDU build has no CUDA, `make accept-s2.4 REHEARSAL=1` grades the host path and ends with `Sprint 2.4: REHEARSED on this host, NOT ACCEPTED.`

## One decode

The harness replaces the controlled channel output at the `rx_grid` ingress of the pinned `pusch_e2e_pipeline_test`. For one opportunity it

1. fills a pinned resource grid with `session::deliver` of Sprint 2.3, which leaves the grid ready with its readback digest,
2. admits one decode per backend: the key, the ticket of the grid, the hash of the grant record and the original block go to the ledger, which holds the grid,
3. starts the processor with the PDU, the reader of that grid, a buffer for the decoded bytes that the ledger owns, and a notifier,
4. waits for the notifier, or for the declared time,
5. gives the decode its outcome and writes its radio record.

Both processors read the same grid, one after the other. The grid of an opportunity is retained while the next opportunity is filled into the other grid and decoded. It is then checked against its digest once more and released.

**The processors.** Each backend is one `pusch_processor` from `create_sw_pusch_processor_factory`, the factory helper of the pinned test, with the settings that the Sprint 1 gates measured: six LDPC iterations with early stop, time interpolation `average`, the MMSE equalizer. The backend `cpu` asks for the PUSCH type `auto`, the backend `cuda` for `gpu`. Each has its own receive buffer pool and its own HARQ process.

**The PDU.** `grant_pdu.h` builds it from the grant record, with the frame and the slot of the opportunity. The DM-RS is the PUSCH DM-RS, referenced to CRB 0, and the PDU names no reference point of the transmit side. The transport block size and the coding come from the pinned `get_nr_sch_mcs_description` and are checked against the record. The receive ports are `0` to `N-1`, the only list the pinned receiver decodes.

**One hash binds grid, PDU and block.** The channel plan of Sprint 2.3 binds every transmit grid to the hash of its grant record with the transport block. The tool computes the same hash from the record and the block it was given, and refuses a plan in which the two differ. A PDU can therefore not be built from another record than the grid, and a block can not be compared that the grid does not carry.

## Outcomes

Every decode of a plan ends in exactly one of six outcomes. The rule is `classify` in `dispatch_core`, and `ocudu_cuda_isac.episodes.replay_pusch.classify` repeats it. The unit gate compares the two on all 64 inputs.

| Outcome | When | In the denominator |
|---|---|---|
| `decoded` | The decoder reported a CRC pass and every decoded byte equals the original | Yes |
| `payload_mismatch` | The decoder reported a CRC pass and a byte differs. The comparison fails. | Yes |
| `crc_fail` | The decoder reported a CRC failure | Yes |
| `decoder_error` | The decoder raised or reported an error, wrote past the transport block, or its input grid changed while it was read | Yes |
| `timeout` | No completion arrived within the declared time | Yes |
| `not_dispatched` | The run stopped before this decode was started | Yes |

The CRC flag is the decoder's own. Nothing sets it, and nothing forces an acknowledgement. A CRC failure is an observation and not an error of the run: the tool exits with status 0 and the decode keeps its record and its key.

**Key.** `<opportunity_id>/<backend>`. One opportunity is decoded once by each backend. A key that was admitted before is refused as `duplicate_delivery` and recorded as an anomaly. The first admission stands.

**Admission.** The ledger refuses, and records, a grid that is not ready or whose generation is stale (`stale_grid`), a grid whose descriptor names another grant record than the decode (`changed_allocation`), a grid whose values no longer have the digest it became ready with (`changed_grid`) and an empty original block (`empty_payload`). None of them becomes a job.

**Ownership.** A grid is held from its admission until the completion callback of every decode on it has arrived. Until then `release_grid` returns false and the replay cannot fill the grid again. After the callback the ledger takes the digest of the grid once more: an input that changed during the decode makes the outcome `decoder_error`. A decode that timed out keeps its grid held, because the processor may still read it. The backend is then stopped, its remaining decodes become `not_dispatched`, the tool writes its evidence and exits with status 1 without tearing a processor down.

**Decoded bytes.** They go to a buffer the ledger owns, filled with a known byte before the decode and followed by 64 guard bytes. A damaged guard makes the outcome `decoder_error`. The comparison is over the whole block: blocks of different lengths are never equal, and the first differing byte is recorded.

**Completions.** The callback may arrive on any thread. A second callback for a job, a callback for an unknown job and a callback after a timeout are recorded as anomalies and change nothing. The first callback of a job stands.

## The payload

Sprint 2.2 gives every opportunity the transport block of the base grant. A transport block per opportunity is built here, where the decoded bytes are compared (decided 2026-11-20). A run records which rule it used, in the plan, in the report and in the manifest of the gate.

| Rule | Block of opportunity `i` | Transmit grid |
|---|---|---|
| `fixed` | The block of the base grant | The grid of the grant of its slot, from the store entries of Sprints 2.1 and 2.2 |
| `per_opportunity` | `generated_block("<seed>/<i as six digits>", bytes)` with the generator `sha256-counter/v1` of the grant module | Realized by the fixture adapter for the grant `<slot grant>.pNNNNNN`: the grant of the opportunity's slot with that block |

Under `per_opportunity` both cases of a pair carry the same block at the same opportunity, so the waveform stays fixed across the pair. The grant of a block differs from its slot grant in the identifier and the block only: the allocation, the pilots and every coordinate are the slot grant's. `payload_set` binds each block to its transmit grid in a `pusch-payload-set/v1` record, and `per_opportunity_channel_plan` builds the channel plan on those grids with the channel, the noise and the gain of the entry.

One block for all opportunities cannot show a decoder that returns the bytes of another opportunity. With a block per opportunity such bytes fail the comparison. The driver test shows it: a tool double that returns the bytes of the opportunity before passes every check of the fixed pair and fails the payload case.

The fixture adapter makes each of these grids without its loopback. Both backends decode every grid through the channel of the pair, which is the stronger check. [The reason](#findings)

## The plan: `replay-pusch-plan/v1`

`build_plan(run_id, channel_plan_path, payload=None, backends=("cpu", "cuda"), receiver=None)` writes what the native tool reads. A plan is an offline stimulus and evaluator record: it names transport blocks, and the receiver side of a run reads none of it.

| Field | Contents |
|---|---|
| `schema`, `run_id`, `access` | `replay-pusch-plan/v1`, the run, `offline_stimulus_and_evaluator` |
| `channel_plan` | Path and SHA-256 of the `grid-apply-plan/v1` file of Sprint 2.3 |
| `payload` | `rule`, `generator`, `seed`, `blocks`: the number of distinct blocks |
| `receiver` | `backends`, `ldpc_iterations`, `early_stop`, `timeout_ms`, `tap` |
| `grants` | Per grant of the channel plan: `grant_id`, `record_sha256`, `record` and `tb` (path and SHA-256 each), `tb_bytes` |

`tap` is `none`. This run is the untapped control of the later sprints: no observer is attached, and a plan that names one is refused.

**Refusals.** The plan check in Python and the native tool refuse, with status 2 and before a processor is created or a file is written:

- a transport block of another size than the record gives (`mismatched payload`), and a block or a record whose file does not have the bound hash,
- a record that was edited, so that its hash with the block is not the one that binds the grid (`changed allocation`),
- a grant of the channel plan without a record, a grant listed twice, a grant the channel plan does not hold,
- an opportunity in another slot than its grant, because the DM-RS depends on the slot index, and an opportunity listed twice,
- a backend that is not `cpu` or `cuda`, a backend listed twice, no backend,
- an observer on the control run, a timeout or an iteration count that is not positive,
- a channel plan with another hash, and everything Sprint 2.3 refuses in a channel plan: no gain, a transmit grid with another hash.

The plan check alone refuses a `per_opportunity` plan whose blocks are not the ones the rule generates from its seed. The native tool generates no block.

## The native core

`native/ocudu_adapter/receiver/` holds the dispatch. `dispatch_core` includes the channel application core and nothing of OCUDU, so the unit target links it with doubles only.

| File in `receiver/` | Links OCUDU | Contents |
|---|---|---|
| `dispatch_core.h`, `dispatch_core.cpp` | No | The outcome rule, the byte comparison, the ledger with its keys, its admissions, its held grids, its owned output buffers and its anomalies, and `dispatch`: admit, start, wait, resolve |
| `tests/dispatch_core_test.cpp` | No | Twelve unit cases against doubles of a processor |
| `grant_pdu.h` | Headers only | Reader of a grant record, its hash, the coding of the pinned source and the PUSCH PDU |
| `receiver_plan.h` | Headers only | Reader of the plan |
| `replay_pusch.cpp` | Yes | The pinned processors behind the `decoder` interface, and the tool |

The interface a processor has to meet is small: `decoder::start(request, completion_sink)` starts one decode, and `completion_sink::complete(token, crc_ok, error)` reports it, during the call or later, on any thread.

The unit cases, each a CTest case `dispatch_core_<name>`:

| Case | What it shows |
|---|---|
| `compare` | Equal blocks, a differing byte at the start, in the middle and at the end, blocks of other lengths, an empty block |
| `decoded` | A valid CRC with the original bytes is `decoded`, and the grid is released after the callback |
| `valid_crc_wrong_byte` | A notifier double that returns a valid CRC with one wrong byte fails the comparison: `payload_mismatch` with the position of the byte |
| `crc_fail` | A CRC failure is `crc_fail` even when the bytes happen to be right. It is counted and never acknowledged. |
| `swapped_completions` | Two decodes whose completions arrive in the other order each keep their own result |
| `decoder_error` | A decoder that raises on start and one that reports an error both end as `decoder_error` |
| `duplicate_delivery` | The same key twice is refused and recorded. A second completion is an anomaly, and the first stands. |
| `silent_timeout` | A decoder that never calls back ends as `timeout`. Its grid stays held, and a late completion is an anomaly. |
| `owned_input` | A grid cannot be released or filled again while a completion is outstanding, and it still has its digest at the completion |
| `changed_input` | A grid that changed before admission is refused, and one that changed during the decode makes it a `decoder_error`. A stale generation and another grant record are refused. |
| `guard` | A decoder that writes past the transport block is found by the guard bytes |
| `accounting` | Every planned decode has one key and one terminal outcome, a decode that was never started included |

## The tool and what it writes

```bash
build/ocudu-adapter/receiver/replay_pusch --plan FILE --output-dir DIR [--only OPPORTUNITY_ID]... [--no-grids] [--phy-debug]
build/ocudu-adapter/receiver/replay_pusch --self-test
```

`DIR` must exist and must not hold a result. The tool reads and checks everything first: the plan rules, the hash of the channel plan, everything the channel application checks, the record and the block of every grant against their hashes, the hash that binds each record to its grid, the transport block size, the receive ports and the slot of every opportunity. Then it creates two pinned grids, the processors and the ledger, and walks the opportunities in plan order. `--only` decodes the named opportunities alone, from the same plan, which is how a failed grant is run again exactly. `--no-grids` leaves the stored grids out and keeps their digests. Exit status 0 means every planned decode has a record, 2 a refused plan, 1 a decode that did not complete or any other error. `--self-test` decodes a two-block grant in a bandwidth part that starts above CRB 0 on the host path: it is one of the adapter's CTest cases and starts no CUDA work.

| Output | For | Contents |
|---|---|---|
| `ingress/ingress.jsonl` | Receiver side | One `o1-grid/v1` descriptor per opportunity, as Sprint 2.3 writes it |
| `ingress/NNNN.c64` | Receiver side | The stored grid that both processors read. Its SHA-256 is the readback digest. |
| `radio/radio.jsonl` | Evaluator | One `radio/v1` record per planned decode |
| `radio/NNNN.<backend>.bin` | Evaluator | The bytes the decoder left, for every decode with a CRC pass, a mismatch included |
| `native.json` | Evaluator | `replay-pusch-native/v1`: the plan and the channel plan by hash, the payload rule, the receiver settings, per backend its factory, its availability, its outcomes, its path counts and its service times, the verified input files, per opportunity the grid, its digest and both decodes, the anomalies and the totals |

**Record `radio/v1`.** `schema`, `run_id`, `sequence`, `key`, `opportunity_id`, `opportunity_sequence`, `backend`, `sfn`, `slot`, `grant_id`, `grant_record_sha256`, `allocation` (`crb_start`, `prb_count`, `symbol_start`, `symbol_count`, `dmrs_symbols`, `rx_ports`, `layers`, `active_re`, `active_sha256`), `ingress` (`grid_index`, `generation`, `readback_sha256`, `intact`), `tb` (`id`, `bytes`, `bits`), `outcome`, `crc_ok`, `payload_match`, `decoded_sha256`, `decoded_file`, `error`, `path`, `decoder`, `timing` (`service_us`, `first_use`, `timeout_ms`), `retry` and `tap`. A record holds these fields and no other. `retry` is 0 and `tap` is `none`: every decode here is an untapped first transmission.

**The executed path.** `path` holds `requested`, `executed` and `fallback`. `executed` is `resident` when the pinned decoder set `acceleration_pipeline_timing.valid`, which it does only when the resident CUDA demodulation and decode produced the result, and `host` otherwise. A resident decode also records whether the device grid was used and the stage times the decoder measured. A decode that was requested from the CUDA processor and ran on the host names why in `fallback`: the adapter was built without CUDA, no usable CUDA device at run time, a number of receive ports the resident path does not take (it takes 1, 2, 4 or 8), or the processor chose the host path. The pinned `gpu` factory falls back without a message, so this field is the only place that shows it.

**The decoder's own values.** `decoder` holds what the processor reported: the code blocks, the mean LDPC iterations, SINR, EPRE, RSRP, EVM, CFO and time alignment. `timing.service_us` runs from the start of the decode to the decoder's callback, and `first_use` marks the first decode of a backend in a run, which includes its set-up.

**Access.** `ingress/` holds the stored grids and their descriptors and nothing else, exactly as after Sprint 2.3: no hash or name of H, W or a transport block and no label. `radio/` holds the records and the decoded bytes. A record names its grant, its grid digest and its outcome, and no channel, no noise and no label. The tool reads the transport blocks on its evaluator side for the comparison and never hands one to a processor.

`open_output(directory)` verifies a native output with the standard library: every record against the report and against its fields, every key once, the decoded bytes against their hashes, the stored grids against their digests, and each record against the grid of its opportunity. `evaluate(output, plan)` then joins every decode to its original block once more, from the files the plan binds, and marks a record whose stated outcome its own comparison does not confirm.

## Evidence of the integration gate

| File in the run directory | Contents |
|---|---|
| `summary.txt` | Every gate line and the verdict |
| `adapter-build/`, `adapter-receipt.json` | Logs and receipt of the adapter build, with the CTest log of the native unit cases |
| `replay_pusch/grade.json`, `grade.txt` | Case count, check count, every check with its result, the judgment and the state of the hardware gate |
| `replay_pusch/comparison.json` | The measured numbers of every case |
| `replay_pusch/manifest.json` | Command, commit, locked source commits, input hashes, the two earlier runs, every native run with its plan hash and payload rule, adapter identity, the resolved ablation axes |
| `replay_pusch/unity/`, `direct/`, `bad_input/` | Arrays, plans, native output and log of each fixture |
| `replay_pusch/reference_pair/run/` | `channel-plan.json`, `plan.json`, `native/` with `ingress/`, `radio/` and `native.json` |
| `replay_pusch/payload/` | `payload.json`, the store entry of every block's grid, and the run of the pair on them |
| `replay_pusch/rerun/` | The two opportunities that were run again alone |
| `replay_pusch/refusals/` | The refused plan and the native log of every rule |
| `recovered.json` | What the command-line reader recovers from the radio records |

The eight cases and their checks:

| Case | Checks |
|---|---|
| `unity` | The native runs, then one check per grant: the six slot grants on four receive planes and the two other shipped grants on one, through a unity channel with zero noise and `g = 1`. Every plane of the grid is `x.npy` bit for bit, both backends return the original bytes, and a decode on the host path has the values Sprint 2.1 recorded: one LDPC iteration, an EPRE of 3.01 dB, a SINR of 96.99 dB on one port and 103.01 dB on four. |
| `direct` | The target-absent channel with the entry's gain and no noise on all opportunities: the native run, the grid digests against a complex128 reference with the signal-to-noise ratio the grid holds after storage, and both backends return the original bytes |
| `reference-pair` | Plan, native run, the digests of the decoded grids against the reference and against the grids Sprint 2.3 stored, the decodes, both members alike, both backends alike, access, second reader |
| `payload` | The blocks and their grids, the plan, the native run, every decode against a block generated here from the seed alone, a distinct block per opportunity, both members alike |
| `bad-input` | Nothing received (a zero channel with the noise of the entry) and a channel that inverts the data symbols against the DM-RS symbols at full power: every decode ends as `crc_fail` by the decoder's own flag, keeps its record, and none is acknowledged |
| `rerun` | A failed and a decoded opportunity, each run again alone from the archived plan: the same grid digest, the same outcome and the same bytes on each backend |
| `refusals` | The plan rules and the channel rules, by the plan check and by the native tool |
| `backends` | The path every decode executed, and the count of every planned decode over all runs |

The gate prints two things as NOTE lines and does not grade them: whether the digest over the decoded grids of the pair is the one of the accepted Sprint 2.3 run, and the service time of each backend. A difference between the two members of the pair, or between the two backends, is a finding and fails its check.

**The hardware gate.** The check `backends executed path` passes only when the adapter is built with CUDA, the device is usable, every decode requested from the CUDA processor ran its resident path and every decode requested from the host processor the host path. A fallback cannot close it. With `--host-rehearsal` the same cases run on a build without CUDA, the judgment is `rehearsal` and the hardware gate is recorded as open. `make accept-s2.4` accepts only the judgment `pass` with the hardware gate closed.

## Accepted run

`make accept-s2.4` passed on the DGX Spark `spark-ecf8` on 2026-11-21 (UTC) at commit `542b72c`, run `s24-spark-ecf8-20261121T083424Z`. All 48 gate lines passed, none failed and none was skipped. The grids came from the accepted Sprint 2.3 run `s23-spark-ecf8-20261121T032401Z`, the stimulus from the accepted Sprint 2.2 run `s22-spark-ecf8-20261121T004433Z` that it applied, and the build from Sprint 1.1.

| | Accepted run |
|---|---|
| Unit suite and unit gate | 342 tests and 21 tests, none skipped |
| Adapter | `replay_pusch` `d2bcc09539c3` beside `grid_apply` `158644b5a177` and `grant_fixture` `7a0e1c8ec1a0`, the executables the earlier gates accepted. Compiled with the flags of the pinned test and `ENABLE_CUDA`, no compiler warning, 34 CTest cases, 18 source files bound by the receipt. |
| Grade | 8 of 8 cases, 36 of 36 checks, judgment `pass`, hardware gate closed |
| Executed path | 140 of 140 decodes requested from the host processor on the host path. 140 of 140 requested from the CUDA processor on the resident path, 138 on four receive ports and two on one, none with a fallback. |
| Decodes | 280 planned in 7 runs, 280 records, each under a unique key: 256 `decoded`, 24 `crc_fail`, no other outcome, no anomaly, no grid held |
| Unity | Eight grants returned their 1,217, 6,787 and 48 bytes in one LDPC iteration on both paths, EPRE 3.01 dB. SINR on the host path 103.01 dB on four ports and 96.99 dB on one, the values of the accepted Sprint 2.1 run. SINR on the resident path 60.00 dB, its upper bound. |
| Direct path | 24 of 24 grid digests equal the complex128 reference. The grid holds the signal 56.1 dB above its own rounding. 48 of 48 decodes return the original bytes. Reported SINR 59.60 dB on the host path and 54.18 dB on the resident path. |
| Pair | 48 of 48 grid digests equal the reference and the grids of the accepted Sprint 2.3 run. 96 of 96 decodes return the original 1,217 bytes, SHA-256 `a2187d90d75129b1`, in one LDPC iteration, both members and both processors alike. |
| Reported SINR of the pair | 36.50 dB on the host path (36.32 to 36.68) and 36.44 dB on the resident path (36.26 to 36.62). The two paths differ by 0.06 dB in the mean and by at most 0.08 dB on one grid. EVM 0.0162 and 0.0168. |
| Payload | 24 distinct blocks, 6,845 to 6,925 of 7,344 data symbols differ from the slot grant's grid, all pilots equal. 96 of 96 decodes return the block of their own opportunity, 24 distinct blocks on each backend. |
| Bad input | 24 of 24 decodes end as `crc_fail` after 6 LDPC iterations, none acknowledged, no bytes kept. Noise only: EPRE -29.99 dB, SINR -8.20 dB and -8.15 dB. Phase break: SINR 103.01 dB and 60.00 dB with the CRC failing. |
| Rerun | `fx-noise-only.0005` and `rz-a614d5c14ca6.0023`, each alone from the archived plan: the same grid (`6904786293f9eafa`, `405677ad6799c3d6`), outcome and bytes on each backend |
| Refusals | Nine plan rules and three channel rules, by the plan check and by the native tool, each with status 2 before a file was written |
| Service time on four receive ports, 132 decodes | Host path: mean 660 us, median 578 us, largest 1,702 us. Resident path: mean 292 us, median 269 us, largest 714 us. Recorded and not graded. |
| Digest over the 48 decoded grids, first grid | `62f9e9d2dbe09615`, `6a0a3c28258f7c04`, those of the accepted Sprint 2.3 run |

- **Spark build.** This was an open risk, because the gate had been rehearsed on an x86_64 host only. The receiver code compiled on AArch64 with `-mcpu=neoverse-v2` and `ENABLE_CUDA` and linked against the 50 archives and the CUDA runtime of the accepted build, OCUDU build receipt `644193b58ba8`. `grant_fixture` and `grid_apply` kept the hashes the earlier runs accepted.
- **Resident path.** This was the other open risk: no decode had run on the resident CUDA path of a project grid before. Every decode requested from the CUDA processor set the decoder's `acceleration_pipeline_timing.valid` flag, and none recorded a fallback. The resident path returned the bytes of the host path on every grid, 128 times as `decoded` and 12 times as `crc_fail`.
- **Host grid.** Every resident decode read a host resource grid: `device_grid` and `owned_device_grid` are false in all 140 records, and the log of every native run has the line `PUSCH GPU grid path selected: host grid copy.` of the pinned demodulator, which copies the grid to the device itself. The device grid and the producer-owned grid that Sprint 1.2 measured are first used on a project grid in S5.5.
- **Portability.** The 24 transmit grids of the payload case have the same bytes on the Spark and on the x86_64 development host. On the host path the unity values of the two hosts agree within 0.0001 dB.
- **Second reader.** A separate process verified the 96 radio records, the decoded bytes and the stored grids of the pair against their hashes. The command-line reader recovered the decode `rz-22963ee9d985.0000/cuda`, slot 7, from the grid `6a0a3c28258f7c04`, and `cmp` found its 1,217 bytes equal to the archived transport block.
- **Service time.** On the pair the host path takes 623 us and the resident path 308 us in the mean, of which 88 us are the resident demodulation and 212 us the resident decode. A noise-only grid, which runs all 6 LDPC iterations, takes 1,265 us and 675 us. The first decode of each run is excluded, the dispatch is sequential and the host is not tuned. These are records and no latency claim.

The evidence is on the Spark under `results/runs/`, which Git ignores.

## Rehearsal on a development host

Before the Spark run the gate was rehearsed on an x86_64 host (Ubuntu 24.04, GCC 13.3.0, CMake 3.28.3, Python 3.13, NumPy 2.5.3) against a CPU-only build of the pinned OCUDU source at `27e84cb3`. The stimulus was made on that host in the layout of the accepted runs: the transmit grids are those of the accepted records (`x.npy` of `demo-51prb-4rx` is `a18b72ae9b9ea84f` on both), and the channel and the noise are a two-path model with the geometry and the noise level of the reference pair, not the traced arrays of the Spark. The numbers below are therefore a rehearsal of the host path and no hardware result.

| | Rehearsal, host path only |
|---|---|
| Unit gate | 21 tests, none skipped: the 12 cases of the dispatch core, and the native rule equal to the Python rule on 64 of 64 inputs |
| Adapter | 34 CTest cases, no compiler warning, 18 source files bound by the receipt. `grant_fixture` and `grid_apply` have the bytes they had before `receiver/` was added. `replay_pusch.cpp` also passes a syntax check with `ENABLE_CUDA` defined against the pinned headers. |
| Grade | 8 of 8 cases, 36 of 36 checks, judgment `rehearsal`, hardware gate open |
| Decodes | 280 planned in 7 runs, 280 records: 256 `decoded`, 24 `crc_fail`, no other outcome, no anomaly, no grid held |
| Unity | Eight grants returned their 1,217, 6,787 and 48 bytes in one LDPC iteration, EPRE 3.01 dB, SINR 103.01 dB on four ports and 96.99 dB on one, the values of the accepted Sprint 2.1 run |
| Direct path | 24 of 24 grid digests equal the complex128 reference. The grid holds the signal 56.1 dB above its own rounding. 48 of 48 decodes return the original bytes, reported SINR 59.6 dB |
| Pair | 48 of 48 grid digests equal the reference and the grids of the rehearsed Sprint 2.3 run. 96 of 96 decodes return the original 1,217 bytes, both members and both processors alike |
| Payload | 24 distinct blocks, 6,845 to 6,925 of 7,344 data symbols differ from the slot grant's grid, all pilots equal. 96 of 96 decodes return the block of their own opportunity |
| Bad input | 24 of 24 decodes end as `crc_fail`, none acknowledged, no bytes kept |
| Rerun | The last opportunity of the noise-only run and of the pair, each alone: the same grid, outcome and bytes |
| Refusals | Nine plan rules and three channel rules, each with status 2 before a file was written |

Both processors ran the host path there, because the build has no CUDA: every decode requested from the CUDA processor recorded `the adapter was built without CUDA`. The chain audit of Sprint 2 was run on these runs too. It passed the stages that read grids, plans and bytes, and failed the verdicts, the build and the executed path, as it must for a rehearsal.

Faults were injected at two levels. In the native core, 42 mutations of `dispatch_core.cpp` each fail a unit case: a rule that ignores an error, a damaged guard or a changed input, a CRC pass that is always a decode, a comparison that skips the first or the last byte or the lengths, a duplicate key, a changed grid or an empty block that is admitted, a grid that is not held, a second completion that overrides the first, a forced CRC flag, a completion that reaches another job, a release while a completion is outstanding, an exception of the start that escapes. Two of them passed at first and led to added checks: the first byte of the comparison, and a completion that names a decode that was never started. Against the doubles of the tools, each fault of `tests/unit/replay_pusch_double.py` fails the check that names it: a valid CRC with a wrong byte, a forged match, the bytes of another opportunity, a forced acknowledgement, a fallback to the host path, other grids, a missing record, a leak into `ingress/`, an anomaly, a decode that never completes, a tool that refuses nothing. The real tool was also run under load on the development host: 40 self-tests and 12 runs of the pair, 1,152 decodes, all `decoded`. The mutation and load runs were made on the development host.

## The Sprint 2 audit

`make accept-s2` runs `scripts/verify_sprint2.py` on the newest accepted Sprint 2.4 run and on the runs it names. Each gate of Sprint 2 graded its own step. The audit checks what links them, from the archived files, with the standard library, and starts no native tool.

| Stage | Checks |
|---|---|
| `verdicts` | Four ACCEPTED verdicts without a failed line, on one host and one profile. Four passing grades, the hardware gate of Sprint 2.4 closed. |
| `links` | Sprint 2.2 built its pair on the grant store of the Sprint 2.1 run, Sprint 2.3 applied the stimulus of the Sprint 2.2 run, Sprint 2.4 decoded that stimulus and compared with that Sprint 2.3 run. No gate started before the run it consumes completed. |
| `build` | The four adapters were compiled and linked against one OCUDU build with CUDA. `grant_fixture` is the same executable in the four runs and `grid_apply` in the last two. No compiler warning. |
| `waveform` | The store entry of the base grant verifies and its record has the hash the pair binds. The entry and the store entry of every grant verify. |
| `channel` | The plan that `grid_apply` read binds exactly the files of the entry, with the entry's gain. Its stored grids verify. |
| `receiver` | The receiver read a plan that binds the same files. Every grid it decoded has the digest of the grid Sprint 2.3 stored. The decoded bytes in the files of the run are `tb.bin` of the Sprint 2.1 store, byte for byte, on both backends. Every decode ran the path it was requested on. Every planned decode of the gate has one record and one outcome. |
| `payload` | Every decode of the payload run returns the block the rule gives for its opportunity, generated again from the seed |
| `bad-input` | Every decode of bad input ends as `crc_fail` and none left bytes |
| `sources` | The four runs and the checkout name the same locked sources |

The audit writes `chain/grade.json`, `chain/grade.txt` and `chain/chain.json`, one row per opportunity of the pair with the hashes of its record, its transmit grid, its channel and noise, the grid stored for it and what each backend returned. The gate then takes the hash of `tb.bin` and of every file of decoded bytes of the pair once more with `sha256sum`. `make accept-s2 RERUN=1` first runs the four gates again, in order, each on the run of the one before, and audits the four new runs. A run that was moved is named with `--s24-run`, `--s23-run`, `--s22-run` or `--s21-run` and must have the name that was recorded.

`make accept-s2` ends with `Sprint 2: ACCEPTED on this DGX Spark.` A rehearsal of Sprint 2.4 cannot be audited into an acceptance: the audit needs the ACCEPTED verdict, the closed hardware gate and the resident path in the records.

**Accepted.** `make accept-s2` passed on `spark-ecf8` on 2026-11-21 (UTC) at commit `542b72c`, run `s2-spark-ecf8-20261121T083811Z`: 27 gate lines, 9 of 9 stages and 17 of 17 checks on the runs `s21-spark-ecf8-20261120T223232Z`, `s22-spark-ecf8-20261121T004433Z`, `s23-spark-ecf8-20261121T032401Z` and `s24-spark-ecf8-20261121T083424Z`. The unit suite ran 342 tests and the four unit gates 36, 23, 29 and 21. The four adapters were built against one OCUDU build, receipt `644193b58ba8`, with `grant_fixture` `7a0e1c8ec1a0` in all four runs. The record of the base grant has the hash `e262986ff8243db0` that the pair binds. `sha256sum` gave `a2187d90d75129b1` for `tb.bin` of the Sprint 2.1 store and for 96 of 96 files of decoded bytes of the pair. The four runs and the checkout name OCUDU `27e84cb3`, Sionna `2f0a3820` and Sionna RT `15b5ee03`. The four gates were accepted at the commits `e092623`, `e1db1df`, `d027aad` and `542b72c`.

## Findings

**The loopback of the fixture adapter can end in a segmentation fault on a loaded host.** `fixtures/grant_fixture.cpp`, accepted with Sprint 2.1, destroys its PUSCH processor when the loopback block ends and stops its worker pool after that. The pinned decoder decodes a transport block of more than one code block on the workers, and a worker is still inside its task for a moment after the processor's callback: it returns its codeblock decoder to the pool of the processor (`pusch_decoder_impl::fork_codeblock_task`). When the main thread destroys the processor in that moment, the worker writes into freed memory. On the development host two of 24 fixture runs of one rehearsal ended that way, with status -11 and no output, while the unit suite was running on both cores. The kernel log names the worker thread and an address inside that task, at an instruction that clears a bit through a pointer that no longer pointed at the pool. It is rare, and it was not reproduced on demand: 450 further runs under two kinds of synthetic load and every run on an idle host completed, and no run on the Spark has shown it, the accepted runs of the four gates included. The diagnosis rests on that log and on the source. The transmit grid is not affected: a run either writes a complete result or none.

- Sprint 2.4 does not depend on it. The payload case runs the adapter with `--no-loopback`, and `replay_pusch` joins its workers before a processor is destroyed on every way out (`joined_workers`).
- The accepted adapter is not changed here, because a change gives `grant_fixture` another hash than the one Sprints 2.1 to 2.3 accepted, and the audit requires one executable for all grids. The fix is one line: call `worker_pool->stop()` inside the loopback block, after `rx_notifier.wait()` has returned and before the block ends, so that the workers are joined while the processor still exists. The later call stays and does nothing. Apply it together with `make accept-s2 RERUN=1`, which accepts the four gates again on the new executable.
- Until then, a Sprint 2.1 or 2.2 gate that fails with `the fixture adapter exited with status -11` is this race. Run the gate again on a quiet host.

**The `gpu` factory falls back without a message.** When no usable device is found, the pinned `create_sw_pusch_processor_factory(..., "gpu", ...)` returns a processor that runs the host demodulator and decoder. The only trace is the result's `acceleration_pipeline_timing.valid`, which stays false. The tool records the path of every decode from that flag and the gate grades it. On the Spark the flag was set on 140 of 140 decodes requested from the CUDA processor.

**The resident path bounds the SINR it reports.** `pusch_demodulator_gpu_impl.cpp` limits the reported SINR to the range of -10 to 60 dB, and the EPRE and the RSRP to -140 to 60 dB. On a noiseless grid the host path reports 103.01 dB on four ports and the resident path 60.00 dB, on the same grid and with the same decoded bytes. Inside the range the two paths agree: 36.50 dB and 36.44 dB on the pair. Quote a reported SINR with the path that produced it, and do not read a value of 60.00 dB or -10.00 dB from the resident path as a measurement.

**A reported SINR does not predict the CRC.** With the data symbols inverted against the DM-RS symbols both paths report the SINR of a noiseless grid, 103.01 dB and 60.00 dB, and every decode fails its CRC after 6 LDPC iterations. The estimate comes from the pilots. The outcome of a decode is its CRC flag and its bytes, never its SINR.

**Near the rounding of the grid the two paths report a different SINR.** On the direct path without noise, where the grid holds the signal 56.1 dB above its own rounding, the host path reports 59.60 dB and the resident path 54.18 dB. On the pair at 30 dB the difference is 0.06 dB. The pinned CUDA receiver keeps its estimates in 16-bit floats, which is the likely reason. The gate did not isolate it. The decoded bytes are those of the host path on every grid of the gate.

## For Sprint 3

- `tap: none` marks this run as the control. An observer of Sprint 3 attaches between the ready grid and the processors and must leave every field of these records unchanged: the same keys, the same grid digests and the same decoded bytes with the observer on are the first check of S3.1.
- The observer gets the grid through `ledger::view(token)` or `o1_pool::view(ticket, grant_record_sha256)`. Both are read access to the grid the processors read, and the ledger holds that grid until the last callback.
- `radio/radio.jsonl` is the radio record of the [run manifest](../dev/REPRODUCE.md). Its denominator is every planned decode: `decoded`, `payload_mismatch`, `crc_fail`, `decoder_error`, `timeout` and `not_dispatched` all count.
- One grid takes about 1.4 ms to fill on the Spark and a slot lasts 0.5 ms. One decode of that grid takes 0.66 ms on the host path and 0.29 ms on the resident path in the mean. The dispatch here is sequential and unpaced. Pacing belongs to S4.1.
- Join the workers before a processor is destroyed, on every way out of a tool that owns one.
- Sprint 3.1 keeps this tool as it is and adds the tapped replay as a second tool, `observer/pilot_ls`, which reads a plan that names this plan by its hash. The reader `open_output` and `record_violations` take the report schema and the tap of such a run as arguments. Their defaults are the ones of this sprint. [Pilot least squares](PILOT_LS.md)
