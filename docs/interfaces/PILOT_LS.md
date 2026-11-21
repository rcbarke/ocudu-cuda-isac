# Pilot least squares on the receiver grid: the stage O2

Sprint 3.1 (`pilot_ls`) attaches the first observer to the receiver replay. Before a grid is decoded, the observer reads it through the read access of the channel application core and computes, for every pilot of the grant's sensing source and every receive plane, `H = Y conj(X) / |X|^2`. The result is the stage O2 of the [stage axis](../architecture/ABLATIONS.md): observer-derived, sparse, on the DM-RS comb. The host and the CUDA PUSCH processor then decode the same grid, and every radio record must be the one that the accepted Sprint 2.4 run holds for that decode. [Sprint card](../dev/DEVELOPMENT.md#phase-s3) | [receiver replay](REPLAY_PUSCH.md) | [observation record](OBSERVATION.md) | [grant record](GRANT.md)

**Status.** Delivered and unit accepted on 2026-11-21. Built and run on the DGX Spark `spark-ecf8` the same day, run `s31-spark-ecf8-20261121T171416Z`: the tool linked against the accepted build without a warning and every check of the observer and of the decodes passed. The run is not an acceptance: two checks of the gate itself were wrong and failed. Both are corrected, and the corrected driver grades the native outputs of that run with 9 of 9 cases and 23 of 23 checks. The hardware gate stays open until `make accept-s3.1` passes on the Spark. [First run on the Spark](#first-run-on-the-spark)

## Run it

```bash
# Unit gate: the reference, the plan and the reader with the standard library, and the host-free
# observer core built on its own with CMake and a C++17 compiler.
# No OCUDU build, no NumPy, no GPU
python tests/unit/test_pilot_ls.py -v

# Integration gate on a host that passed make accept-s2
make accept-s3.1

# The steps of the gate, one at a time
make adapter-build
s24=$(ls -d results/runs/s24-* | tail -1)
python tests/integration/pilot_ls.py --config configs/integration/pilot_ls.json \
    --s24-run "$s24" --output-dir results/runs/s31-spark-001/pilot_ls

# By hand: a plan on the archived replay plan of the pair, the native tool and a second reader
python -m ocudu_cuda_isac.episodes.pilot_ls plan --replay-plan "$s24/replay_pusch/reference_pair/run/plan.json" \
    --run-id pair-by-hand --output build/pilot-ls-001/plan.json
mkdir build/pilot-ls-001/out
build/ocudu-adapter/observer/pilot_ls --plan build/pilot-ls-001/plan.json --output-dir build/pilot-ls-001/out
python -m ocudu_cuda_isac.episodes.pilot_ls show build/pilot-ls-001/out --sequence 0 --rx 0 --row 0 --pilot 0
```

`make accept-s3.1` writes `results/runs/s31-<host>-<UTC time>/` and ends with `Sprint 3.1: ACCEPTED on this DGX Spark.` It rebuilds the adapter and never rebuilds OCUDU. It takes the newest accepted Sprint 2.4 run as its control, or the one named with `S24_RUN=`. The plans of that run name files of the Sprint 2.1 to 2.3 runs by absolute path, so those runs must still be where they were. On a host whose OCUDU build has no CUDA, `make accept-s3.1 REHEARSAL=1` grades the host path against a rehearsed Sprint 2.4 run and ends with `Sprint 3.1: REHEARSED on this host, NOT ACCEPTED.`

## Where the tap sits

`replay_pusch`, the tool that Sprint 2.4 accepted, refuses a plan that names an observer and stays the untapped control. Its source is not edited, and the gate checks that this tree still builds it, `grid_apply` and `grant_fixture` with the bytes of the accepted Sprint 2.4 run. The tapped replay is a second tool, `native/ocudu_adapter/observer/pilot_ls`. It fills, dispatches and records exactly as `replay_pusch` does, with the same processors, and adds one step. For one opportunity it

1. fills a pinned resource grid with `session::deliver` of Sprint 2.3, which leaves the grid ready with its readback digest,
2. hands a read-only `o1_view` of that ready grid to the observer, which estimates the source bound to the grant record of the grid into a buffer it owns,
3. admits one decode per backend, starts each processor on the reader of the same grid and waits for its callback,
4. writes the observation and the radio records, and releases the grid once every decode on it has called back.

The observer reads and never writes. Its core holds no call that acquires, writes, commits or releases a grid. After the read it takes the digest of the grid once more, and an estimate is recorded only when the grid still has the digest it became ready with. The record names that digest, so an observation is joined to the grid that both processors decoded.

The tool has two controls. The accepted Sprint 2.4 run was written by another executable without an observer. With `tap: none` the tool itself runs without the observer and writes no observer directory. A tapped run is compared with both.

## The estimate

For a pilot at symbol `l` and subcarrier `k` with the reference symbol `X` of the source record, and the stored value `Y` of receive plane `r`:

```
d  = xr * xr + xi * xi
nr = yr * xr + yi * xi          hr = nr / d
ni = yi * xr - yr * xi          hi = ni / d
```

`Y` and `X` are single-precision values, widened to double precision without change. Every product is exact in double precision, so each sum rounds once and the result does not depend on whether a compiler fuses a product into the sum. `hr` and `hi` are then rounded to single precision, to nearest with ties to even. **The estimator precision is complex64, computed through double precision**, and every record states it. The core is compiled with `-ffp-contract=off`. The tool computes no estimate itself and is compiled like `replay_pusch`, with the flags of the pinned test and no other. `ocudu_cuda_isac.sensing.pilot_ls.ls` repeats the rule with the standard library, and the unit gate compares the two bit for bit on 4,616 vectors.

**The reference is the source record's.** The mask and the reference symbols come from `source.json`, `pilot_index.npy` and `pilot_ref.npy` of the grant's store entry, the record of the registered sensing source `pusch_dmrs`. The reference is never replaced by the stored pilot. For two CDM groups without data the reference is 0.998815 per component and the grid stores the pilot with magnitude one per component, so a unity channel reads as 1.0011865 on every pilot, port and symbol. The factor is real and common. O2 stays comparable with the receiver's own estimate, which is configured with the same amplitude. [Precision of the stored grid](GRANT.md#precision-of-the-stored-grid)

**One port.** The observer estimates one reference-signal port. A source record with more than one port is refused, and S6.9 registers one record per port.

## Layout and validity

Payload layout `ocudu_o2_comb_c64`: `[rx_plane][reference_symbol][pilot]`, last axis fastest, little-endian complex64.

| Axis | Contents |
|---|---|
| `rx_plane` | The receive planes in the order of the grid. Nothing is reordered. `antenna_map[i]` names the array element behind plane `i` and travels in the record. |
| `reference_symbol` | The symbols of the slot that carry a pilot of the source, increasing |
| `pilot` | The sorted set of subcarriers that carry a pilot on any of those symbols, in carrier coordinates. For a regular comb the record gives `first_subcarrier` and `subcarrier_step`: column `c` lies on `first_subcarrier + c * subcarrier_step`. A step of 0 means the axis is not regular. `coordinates_sha256` is the SHA-256 over the symbols and then the subcarriers, each a little-endian uint32. |

| Grant | Shape | Bytes | Pilot subcarriers |
|---|---|---|---|
| `demo-51prb-4rx` | 4 x 2 x 306 | 19,584 | 0, 2, ..., 610 on symbols 2 and 11 |
| `table2-51prb-1rx` | 1 x 2 x 306 | 4,896 | 0, 2, ..., 610 |
| `two-prb-offset` | 1 x 2 x 12 | 192 | 180, 182, ..., 202 |

One validity bit per element, in payload order: bit `n mod 8` of byte `n // 8`.

| Bit | Value | Meaning |
|---|---|---|
| Set | The estimate | The element is a pilot of the source and its estimate is finite. A measured zero, `Y = 0` on a pilot, is a set bit with the value zero. |
| Cleared | Zero | A hole: the source lists no pilot at that symbol and subcarrier, or the estimate is not finite |

## The record: `obs-descriptor/v1`

One JSON record per observation in `observer/o2.jsonl`. Its groups are the field groups of the [`obs/v1` record](OBSERVATION.md#the-record-obsv1), whose binary header Sprint 3.2 packs from the same values. A record holds these fields and no other.

| Group | Fields |
|---|---|
| `schema`, `run_id` | `obs-descriptor/v1` and the run |
| `keys` | `epoch`, `sequence` and the opaque keys `episode`, `opportunity`, `grant`. `sequence` rises by one per observation of an epoch. Each opaque key is the leading 64 bits of the SHA-256 of the text it stands for, as 16 hexadecimal digits. |
| `radio` | `cell`, `ue`, `sfn`, `slot`, `absolute_slot`. The absolute slot counts from the epoch start and does not wrap with the frame number: a frame or a slot that goes back inside one episode is the next period of 1,024 frames. |
| `what` | `stage` `O2`, `producer` `observer`, `producer_steps` `["pilot_ls"]`, `source`, `port`, `layout`, `dtype` `<c8`, `scale` (the reference amplitude), `rule`, `precision` |
| `shape` | `rank`, `axes`, `shape`, `strides` in elements |
| `allocation` | `first_subcarrier`, `subcarrier_step`, `subcarrier_count`, `dmrs_symbols`, `rx_count`, `layer_count`, `antenna_map`, `coordinates_sha256` |
| `validity` | `mask_bytes`, `file`, `sha256`, `valid`, `holes` |
| `time` | `unit` `tc`, `tc_per_second`, `t_first_sample`, `t_last_sample`, `t_ready_ns`, `t_publish_ns`. The first sample is the start of the useful part of the first reference symbol and the last sample the end of the useful part of the last one, in T_c from slot 0 of the epoch. `t_ready_ns` is the host monotonic time at which the estimate was complete. `t_publish_ns` is null: publication belongs to Sprint 3.2. |
| `payload` | `bytes`, `file`, `digest_kind` `sha256`, `digest` |
| `host` | Fields only the staged harness has: `name`, `opportunity_id`, `opportunity_sequence`, `grant_id`, `grant_record_sha256`, `ingress` (`grid_index`, `generation`, `readback_sha256`), `completion` `host_synchronous`, `service_us`. A sensing core never reads this group. |

The record identifies the estimator as observer-derived (`producer`), so a consumer cannot take it for the estimate of the PUSCH processor. That one is the stage O3 and has its own producer in S5.2. Nothing in a record is truth: no channel, no noise, no transport block and no label. The episode key is made from the identifier of the realization, which names no case.

The slot of an observation is checked three ways before anything is read: the slot of the opportunity, the slot in the descriptor of the grid and the slot the reference symbols were generated for must be equal, because the DM-RS sequence depends on the slot index.

## The plan: `pilot-ls-plan/v1`

`build_plan(run_id, replay_plan_path, tap="pilot_ls", antenna_map=None, epoch=1)` in `ocudu_cuda_isac.episodes.pilot_ls` writes what the tool reads.

| Field | Contents |
|---|---|
| `schema`, `run_id`, `access` | `pilot-ls-plan/v1`, the run, `offline_stimulus_and_evaluator` |
| `replay_plan` | Path and SHA-256 of a `replay-pusch-plan/v1` file. It must be a control plan, with `receiver.tap` `none`. |
| `observer` | `tap` (`pilot_ls` or `none`), `stage`, `source`, `layout`, `epoch`. Without a tap the three names are null. |
| `sources` | Per grant of the channel plan: `grant_id`, `record_sha256`, `source`, `pilot_index` and `pilot_ref` (path and SHA-256 each), `antenna_map`. Empty without a tap. |

The antenna map defaults to the receive ports of the grant in order. A sensing source is bound to its grid by the hash that Sprint 2.3 binds the transmit grid with: the hash of the grant record with its transport block.

**Refusals.** The plan check in Python and the native tool refuse, with status 2 and before a processor is created or a file is written:

- a tap that is not `none` or `pilot_ls`, another stage or layout than the tap exports, an epoch below 1,
- a replay plan with another hash, and a replay plan that names a tap itself,
- a grant without a sensing source, a source listed twice, a source for a grant the channel plan does not bind,
- a source bound to another grant record than the grid (`changed allocation`), a source that names another slot, other reference symbols or another port than the record of the grant, and a pilot outside the allocation,
- a source record or an array whose file does not have the bound hash, and a record that names other arrays than the plan binds,
- a configured pilot with zero reference power, a duplicate pilot, pilots out of order, a reference of another size, more than one port,
- an antenna map for another number of receive planes than the grid has, and one that names an array element twice,
- a run without a tap that names a stage, a layout or a source,
- everything Sprint 2.4 refuses in a replay plan and Sprint 2.3 in a channel plan.

## The native core

`native/ocudu_adapter/observer/` holds the observer. `observer_core` includes the channel application core and nothing of OCUDU, so the unit target links it with doubles only.

| File in `observer/` | Links OCUDU | Contents |
|---|---|---|
| `observer_core.h`, `observer_core.cpp` | No | The estimate, the validated source and its rectangle, the validity bits, the slot times, the opaque keys, and `observer`: the bound sources and the observations of a run |
| `tests/observer_core_test.cpp` | No | Eleven unit cases against the storage model of the grid |
| `observer_plan.h` | Headers only | Reader of the plan and of a source record with its arrays |
| `pilot_ls.cpp` | Yes | The tapped replay: the dispatch loop and the pinned processors of `replay_pusch.cpp`, and the tap |

The unit cases, each a CTest case `observer_core_<name>`:

| Case | What it shows |
|---|---|
| `kernel` | A pilot of `1+j` and a noiseless `Y = -1+3j` give `1+2j`. `Y = H X` returns `H`. The stored pilot against the reference reads as 1.00119. A result that does not fit single precision is flagged. |
| `amplitude_scale` | A halved pilot amplitude with a correspondingly scaled Y gives the same estimate, bit for bit |
| `data_invariance` | Altering data-only resource elements changes the grid digest and leaves O2 as it was. Changing a pilot changes it. |
| `pilot_hole` | A pilot the record leaves out is a cleared bit with the value zero, on every plane. A measured zero is a set bit. The bit order of the mask. |
| `refusals` | A configured pilot with zero reference power, a duplicate pilot, pilots out of order or outside the grid, two ports, a reference of another size or not finite, no pilot, other reference symbols, no symbol times, a bad amplitude, antenna map, spacing or slot |
| `antenna_map` | Two antennas with two phases. The payload keeps the order of the grid, and with a deliberately reversed map array element 0 is read from the other plane. |
| `coordinates` | Symbols, subcarriers, step, shape, strides and the digest of the axes. A channel that encodes its coordinate comes back at the right element. An irregular axis reports no step. |
| `record` | Stage, producer, source, layout and precision. The keys. The ingress digest equals the digest of the view. The slot count across a frame wrap, another episode and a new epoch. The sample times. |
| `ownership` | The grid has its digest after the tap. An observation survives the release and the next fill of its grid. A view of an earlier fill is stale: nothing is read and nothing is recorded. |
| `binding` | A second source for a grant record, a grid of an unbound record, a grid of another slot and a grid with another number of planes are refused, and none leaves an observation |
| `slot_times` | Slots per frame and the start of a slot at 15, 30, 60 and 120 kHz, where the first symbol of every half subframe carries the longer prefix |

The adapter build adds three CTest cases of the tool. `pilot_ls --self-test` decodes a two-block grant in a bandwidth part that starts above CRB 0 on the host path and starts no CUDA work. It repeats the checks of `replay_pusch --self-test`, then taps the pinned grid: a unity channel reads as 1.00119 on all 48 pilots of both receive planes, the observation names the digest of the grid, the tapped grid decodes to the original bytes, and a receive plane turned by `j` reads as `j` times 1.00119 on that plane alone. The reference symbols of the self-test are read off the noiseless transmit grid. The gate takes them from the source record.

## The tool and what it writes

```bash
build/ocudu-adapter/observer/pilot_ls --plan FILE --output-dir DIR [--only OPPORTUNITY_ID]... [--no-grids] [--phy-debug]
build/ocudu-adapter/observer/pilot_ls --self-test
```

`DIR` must exist and must not hold a result. Everything is read and checked first, then the grids, the processors and the observer are created. The options and the exit statuses are those of `replay_pusch`.

| Output | For | Contents |
|---|---|---|
| `ingress/` | Receiver side | As `replay_pusch` writes it: one `o1-grid/v1` descriptor and one stored grid per opportunity |
| `observer/o2.jsonl` | Observer side | One `obs-descriptor/v1` record per opportunity. Only with the tap. |
| `observer/NNNN.o2.c64`, `NNNN.o2.mask` | Observer side | The payload and its validity bits |
| `radio/` | Evaluator | As `replay_pusch` writes it. `tap` is `pilot_ls` or `none`, and `run_id` is the run of this plan. |
| `native.json` | Evaluator | `pilot-ls-native/v1`: the report of `replay_pusch`, with the replay plan by hash, an `observer` section (tap, stage, source, layout, epoch, rule, precision, counts, service time, one entry per bound source) and per opportunity the digests of its observation |

`open_output(directory)` verifies a native output with the standard library. The radio records, the decoded bytes and the grids are verified by the reader of Sprint 2.4, with the tap of the run in every record. Every filled grid has exactly one observation, in sequence and consistent on its own, whose payload and validity bits have the hashes and sizes it names, whose set bits are the count it names, and which names the readback digest, the index and the generation of the grid of its opportunity. `observer/` holds nothing else. A run without a tap holds no observation.

Three functions join an output to what it is compared with.

| Function | Result |
|---|---|
| `oracle(output, sources)` | Recomputes every estimate from the stored grid and the source record with the reference. Per observation: payload and mask equal bit for bit, coordinates equal, identity equal. |
| `expected_times(output, sources)` | The observations whose slot count, sample times or episode key differ from the ones the reference derives |
| `radio_differences(tapped, control)` | Every field of a radio record that differs between two runs. Compared: the key, the opportunity, the backend, the slot, the grant, the allocation, the transport block entry, the outcome, the CRC flag, the payload match, the hash and the name of the decoded bytes, the error, the grid digest with its index and generation, and the requested and executed path. Not compared: the run, the tap, the times and the values the decoder measured. |

## Evidence of the integration gate

| File in the run directory | Contents |
|---|---|
| `summary.txt` | Every gate line and the verdict |
| `adapter-build/`, `adapter-receipt.json` | Logs and receipt of the adapter build, with the CTest log of the native unit cases |
| `pilot_ls/grade.json`, `grade.txt` | Case count, check count, every check with its result, the judgment, the state of the hardware gate and the estimator precision |
| `pilot_ls/comparison.json` | The measured numbers of every case |
| `pilot_ls/manifest.json` | Command, commit, locked source commits, input hashes, the Sprint 2.4 run, every native run with its plan hash, its tap and its control, adapter identity, the resolved ablation axes |
| `pilot_ls/unity/`, `direct/`, `reference_pair/`, `payload/`, `tap_off/` | The plan, the native output and the log of each run |
| `pilot_ls/rerun/`, `refusals/` | The opportunity that was run again alone, and the refused plan and the native log of every rule |
| `recovered.json` | What the command-line reader recovers from the observations |

The nine cases and their checks:

| Case | Checks |
|---|---|
| `control` | The Sprint 2.4 driver graded its run as a pass with the hardware gate closed. The plans this gate taps have the hashes its manifest names, and their radio records verify with the reader of that sprint and carry no tap. |
| `unity` | The native runs with every radio record equal to the control's. Then one check per run: every estimate of every grant is 1.0011865 with a zero imaginary part and a set validity bit, with the shape and the first subcarrier of its grant, and the reference recomputes each payload. |
| `direct` | The native run. Known channel: every estimate equals `g H` times the stored pilot over the reference, within `2**-8` of the magnitude plus `1e-6`, on all receive planes. Receive planes: the phase from each plane to the next agrees with the known channel. Where the known channel tells a plane from its mirror image at half of the elements or more, fewer than half of the estimates may stay inside the tolerance with the planes in reverse order. Where it does not, as for a source at broadside of the array, the reversal is not graded here and the line says so. |
| `reference-pair` | Native run with the shape and size of the demonstration. Ingress: every observation names the digest of the grid the control decoded, on both backends. Reference: bit for bit, with the source, the coordinates, the slot count and the sample times. Sparse oracle: a least-squares oracle in double precision agrees within `1e-6`, and every estimate at a declared pilot is nonzero. Receive planes: against the same oracle with the planes of the grid in reverse order fewer than one estimate in a hundred agrees, because the noise of the entry differs on every plane. Decodes: every radio record and every file of decoded bytes is the control's. Access: no name or hash of H, W or a transport block and no label in `observer/`. |
| `payload` | The native run with every radio record equal to the control's. Data invariance: every grid differs from the grid of the same opportunity with the fixed block, and every estimate is that grid's estimate bit for bit. |
| `tap-off` | The same executable without the tap writes no observer directory, its radio records equal the control's, and the records with the tap on equal the records with the tap off. |
| `rerun` | One opportunity alone from the archived plan with `--only`: the same grid, estimate, slot count, sample times, outcome and bytes |
| `refusals` | Fourteen plan rules, by the plan check and by the native tool |
| `backends` | The path every tapped decode executed, and the count of every planned decode and every observation |

The gate prints three things as NOTE lines and does not grade them: the difference between the estimates of the two members of the pair, the service time of the tap, and the PUSCH service time with the tap beside the control's.

**The hardware gate.** The check `backends executed path` passes only when the adapter is built with CUDA, the device is usable and, with the observer attached, every decode requested from the CUDA processor ran its resident path. The gate runner adds two lines of its own: this tree builds `replay_pusch`, `grid_apply` and `grant_fixture` with the bytes of the Sprint 2.4 run, and a second reader recovers one observation, whose payload hash and grid hash are taken once more by `sha256sum` and whose grid digest is read from the radio record of the Sprint 2.4 run.

## Checked on a development host

Before the Spark run, on an x86_64 host (Ubuntu 24.04, GCC 13.3.0, CMake 3.28.3, Python 3.13, NumPy 2.5.3) without an OCUDU build:

| | Development host |
|---|---|
| Unit suite | 403 tests, none failed: the 342 of the accepted tree and 61 of this sprint |
| Unit gate | 31 tests, none skipped: the 11 cases of the observer core, the native estimate equal to the reference on 4,616 of 4,616 vectors, the native keys and slot times equal to the reference |
| Observer core | Builds on its own without a warning and links `libobserver_core.a` and `libreplay_core.a` only |
| Observer core on stored grids | A one-off harness linked the observer core and the plan reader, without OCUDU, and ran them on the 38 stored grids of the driver run, 4 by 14 by 612 values each: every payload, validity mask, join key, slot count and sample time equals the record of the reference. The same with `-O3` and fused multiply-add enabled. The harness is not part of the tree. |
| `pilot_ls.cpp` | Passes a syntax check against the headers of the pinned OCUDU source at `27e84cb3`, with and without `ENABLE_CUDA`, with `-Wall -Wextra -Wshadow` and no warning. Not linked, not run. |
| Accepted sources | The eleven native source files of Sprint 2 have the hashes of the accepted tree. The unit gate checks it. |
| Integration driver | 9 of 9 cases and 23 of 23 checks against the doubles, on a Sprint 2.4 run that the driver of that sprint graded on its own doubles. As a rehearsal on a build without CUDA: judgment `rehearsal`, hardware gate open. |
| Faults | Each fault of `tests/unit/pilot_ls_double.py` fails the check that names it: an estimate against the stored pilot, a missing conjugate, reversed receive planes, shifted coordinates, an estimate that depends on data, a wrong validity bit, a wrong slot, an observation of another grid, a tap that writes the grid, a tap that changes a decode, a missing observation, a leak, observations without a tap, a fallback to the host path, a tool that refuses nothing, a tool that exits with an error |

The doubles compute the estimate with the Python reference, so the driver run shows that the checks are sound and nothing about the native tool. Three things were open before the Spark ran the gate, and its first run settled each: `pilot_ls` links against the accepted build and its self-test passes on the pinned grid, the native estimate on the pinned grid equals the reference bit for bit on AArch64, and the decodes with the observer attached are those of the accepted run on the resident CUDA path.

## First run on the Spark

`make accept-s3.1` ran on `spark-ecf8` on 2026-11-21 (UTC) at commit `b339c2a`, run `s31-spark-ecf8-20261121T171416Z`, and ended NOT ACCEPTED with 20 of 22 checks. What failed was the gate, in two places. The native outputs of that run pass the corrected driver.

| | First run on the Spark |
|---|---|
| Unit suite and unit gate | 403 tests and 31 tests, none skipped |
| Adapter | `pilot_ls` `c3a214c2869c` beside `replay_pusch` `d2bcc09539c3`, `grid_apply` `158644b5a177` and `grant_fixture` `7a0e1c8ec1a0`, the bytes of the accepted Sprint 2.4 run. `ENABLE_CUDA`, 48 CTest cases, no compiler warning. The tap self-test on the pinned grid passed. |
| Unity | 15,324 estimates on 8 grids, every one 1.0011865 |
| Direct path | 58,752 of 58,752 estimates equal `g H` times the pilot ratio within the rounding of the grid, largest relative error 0.0029 |
| Pair | 48 of 48 observations name the grid the accepted `replay_pusch` decoded, first `6a0a3c28258f7c04`. 48 of 48 payloads and masks equal the reference bit for bit, on AArch64. The oracle in double precision agrees within 5.8e-08. 117,504 of 117,504 estimates are nonzero. |
| Decodes | 96 of 96 radio records and files of decoded bytes of the pair are the control's. 352 decodes in 6 runs, all `decoded`, 176 on the host path and 176 on the resident CUDA path, 128 observations for 128 tapped grids |
| Payload | 48 of 48 grids differ and 48 of 48 estimates are the same bit for bit |
| Recorded, not graded | The estimates of the two members differ by -41.8 dB. The tap takes 1,470 us per grid. |

The two faults of the gate:

- **The expected number of observations was that of one member.** `configs/integration/pilot_ls.json` named 24, the count the Sprint 2.4 configuration holds per member. The run produced the 48 of both members, as it must, and the check `reference-pair native run` failed on the count while its text showed nothing wrong. The configuration now names 48, the line prints what it expected, and a unit test ties the value to the Sprint 2.4 configuration.
- **The direct path cannot show a reversed plane order.** The check `direct receive planes` required that the estimates leave the tolerance when the planes are reversed. The target-absent channel of the reference pair is the same on all four receive planes to the last bit: the transmitter lies at broadside of the array. With the planes reversed all 58,752 estimates stayed inside, which is correct. The unit pair of the driver tests has a phase step of 0.4 rad per plane, so the fault did not show there. The check now grades the reversal only where the known channel tells mirrored planes apart, and the pair has its own check: against the oracle with the planes of the grid reversed, 358 of 117,504 estimates agree, because the noise differs on every plane. Reversed payload planes injected into the outputs of this run fail that check, the reference and the oracle.

One measurement goes on. The tap takes 1.47 ms for 2,448 estimates, and about nine tenths of it is the digest that the observer takes of the whole grid after its read, 274 kB through the grid reader. The share was measured on a development host with the storage model of the grid. It is the time a fill takes for the same reason. It is outside the PUSCH service time and is not graded. A timed run leaves that digest to the evaluator, as the observation contract allows, which S3.2 and S4.1 have to do before a paced run.


## For Sprint 3.2

- The observation owns its payload in host memory and is complete before the first decode starts. Sprint 3.2 replaces that buffer by a slot of the ring and fills `t_publish_ns`. The fields of the record do not change.
- `observer::observe` returns a reference that stays valid until the next call. A producer that publishes copies the header fields it needs before it observes again.
- The tap is synchronous and sits in the dispatch thread. Its service time is recorded and is outside the PUSCH service time. Whether it moves to its own thread is decided where a paced run measures it, in S4.1.
- `pilot_ls` is the tapped harness. A later tap is added to it, and `replay_pusch` stays the control.
