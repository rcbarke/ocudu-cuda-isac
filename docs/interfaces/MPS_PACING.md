# Paced clients under MPS

The contract of Sprint 4.1 (`mps_pacing`): the paced plan of the staged harness, the release rule, the files a paced run writes, the launcher `scripts/run_m1.py` with its arms, the proof of the label MPS, the contention controls and the record `mps_pacing.json`. [Process layout and contention](../architecture/RUNTIME.md#process-layout) | [observation contract](OBSERVATION.md) | [ring](SNAPSHOT_RING.md) | [native sensing core](SENSE_CORE.md) | [sprint card](../dev/DEVELOPMENT.md#phase-s4)

**Status.** Accepted on the DGX Spark `spark-ecf8` on 2026-11-23 (UTC): `make accept-s4.1` passed at commit `ed0354d`, run `s41-spark-ecf8-20261123T222740Z`, with the lead-in and with the rule of the scores: 12 of 12 cases, 45 of 45 checks, hardware gate closed. That run is the reference of this sprint, and every figure quoted from Sprint 4.1 is its figure. [What the accepted run measured](#what-the-accepted-run-measured)

**How the sprint closed.** The gate ran five times on the Spark. The first two runs found defects of the tooling. The third passed at commit `3302329` without a lead-in and measured what a cold first decode costs a schedule. The fourth ran the lead-in and failed on a rule of the gate that asked two decodes of one grid for equal decisions. The fifth is the accepted one. Three choices of the architecture came out of those runs and are final for this sprint: [the lead-in](#the-lead-in), which keeps the cold start out of every schedule; [the scores of two decodes](#the-scores-of-two-decodes), compared by the input the sensor was given; and the clock of a fault, which is the announced start of the schedule. [The first accepted run](#what-the-first-accepted-run-measured-without-a-lead-in) | [the first run](#what-the-first-spark-run-showed) | [the second](#what-the-second-spark-run-showed) | [the run with the lead-in](#what-the-run-with-the-lead-in-showed)

## What the sprint adds

| Path | Holds |
|---|---|
| `native/ocudu_adapter/pacing/` | `pacing_core.h`, `pacing_core.cpp`: the pacer, the paced book and the rules of a lead-in. Host-free: it includes the dispatch core and the channel application core and nothing of OCUDU. `tests/pacing_core_test.cpp`: eleven cases in CTest |
| `native/ocudu_adapter/observer/pilot_ls.cpp`, `observer_plan.h` | The plan version 4, the paced loop of the staged harness and the lead-in ahead of it. The loops of the plan versions 1 to 3 are unchanged |
| `src/ocudu_cuda_isac/episodes/mps_pacing.py` | The plan, the reference of the release rule, the reader of a paced output, the inventory rules, the reducers. Standard library only |
| `scripts/run_m1.py` | The launcher: the MPS control daemon of this checkout and one arm from its specification |
| `tests/unit/test_mps_pacing.py` | The unit gate of the card |
| `tests/unit/test_mps_pacing_driver.py`, `tests/unit/mps_pacing_double.py` | The driver as a rehearsal on a stand-in control, with the real `sense_host` of the host twin |
| `tests/integration/mps_pacing.py`, `configs/integration/mps_pacing.json` | The driver of the gate and its configuration |
| `scripts/accept_s4_1.sh`, `tests/test_accept_s4_1.py` | The gate runner, `make accept-s4.1`, and the checks of what it decides before anything is built |

No source of `native/obs_ring`, of `native/sense_core`, of the receiver patch or of the ledger of Sprint 2.4 changes. Of the accepted native tools only `pilot_ls` has other bytes after this sprint. The driver requires that `obs_tail`, `obs_feed` and `replay_pusch` keep the bytes of the adapter that the Sprint 3.5 run verified.

## The paced plan

`pilot-ls-plan/v4` is the plan of Sprint 5.2 (`pilot-ls-plan/v3`) with one more object, `pacing`. With `pacing: null` the tool behaves as on version 3.

```json
"pacing": {
  "clock": "host_monotonic",
  "start_delay_us": 50000,
  "allowance_us": 500,
  "latest_start_us": 250,
  "release_us": [0, 500, 1000, 5000, 5500, 6000]
}
```

| Field | Meaning |
|---|---|
| `clock` | `host_monotonic`: the clock of the observation ring and of the sensor |
| `start_delay_us` | From the end of the preparation (processors created, consumers attached, first grids filled) to the start of the schedule |
| `allowance_us` | The processing allowance of one job. A job that is done later than its release plus the allowance is an overrun |
| `latest_start_us` | How long after its release a job may still start. Later, it is a skipped release |
| `release_us` | One offset per opportunity of the replay, in its order, strictly increasing, from the start of the schedule |

The object may name one more member, `lead_in`, which is null or the [lead-in](#the-lead-in) of the run:

```json
"lead_in": {"releases": 6, "lead_us": 10000}
```

| Field | Meaning |
|---|---|
| `releases` | How many of the first releases are decoded once more ahead of the schedule, 1 to 64 and no more than the schedule holds |
| `lead_us` | How long before its own release each of them is issued |

The member was added after the sprint was first accepted. A plan without it is the plan of the first accepted run, hash for hash, and runs without a lead-in. A lead-in is refused unless its last decode, done on its allowance, has ended when the first release is offered, `offset(releases - 1) - lead_us + allowance_us <= offset(0)`, and unless its first decode is issued after the schedule was fixed, `lead_us - offset(0) <= start_delay_us`.

`mps_pacing.release_offsets` derives the offsets from the frame and the slot of each opportunity: a slot lasts `slot_us`, consecutive entries are separated by `gap_slots` idle slots. A paced plan is refused unless its evidence mode is `timed` and it names as many releases as the replay has opportunities. The driver refuses a configuration whose rings do not wait in `spin` mode, whose receiver decodes on another processor than CUDA, or that enlarges a ring for an arm. `pilot_ls --pacing` prints the plan version the build reads and that it reads a lead-in.

## The release rule

```text
release(n) = start + offset(n)                          never shifted
the worker is free at t <= release(n) + latest_start    job n runs: at release(n), or, when its grid
                                                        was ready later, as soon as it is ready (late)
the worker is free at t >  release(n) + latest_start    job n is a skipped release
job n is done later than release(n) + allowance         an overrun
```

A job is the tap of one grid and its decodes. A skipped release keeps its key: it has a line in `pacing/releases.jsonl` and one `not_dispatched` radio record per backend with the reason `missed_release`, and it stays in every denominator. A worker that takes 12 ms on a schedule released every 5 ms skips the two releases it worked through and starts the next one on that release's own time. The rule is `isac_pacing::pacer` in C++ and `mps_pacing.Pacer` in Python, and the unit gate runs one trace through both.

The grid of a release is filled ahead of its slot, between the end of the job before it and its release time, without a digest. Its fill time is recorded per release.

## The paced book

The ledger of Sprint 2.4 allocates and takes a mutex once per decode, between a ready grid and the decoder's callback. A paced run uses `isac_pacing::paced_book` in its place. Everything of a decode is named and allocated before the first fill: the key, the original block, the output buffer with its guard, the request. In the loop an admission checks the state and the generation of the grid, a completion is a few stores and one atomic flag, and the dispatch thread spins on that flag. The book keeps the rules of the ledger in the mode `timed`, and its job records equal those of the ledger field by field (`book_decoded`, `book_outcomes`, `book_refusals`). `book_no_allocation` counts the calls of `operator new` over a run of the loop and requires zero. The processor wrapper of `pilot_ls` differs from that of `replay_pusch` in one thing, which `tests/unit/test_pilot_ls.py` now states: the notifier of a decode can be arranged before the first fill (`arrange`), so that its start allocates nothing. The notifier still takes its own uncontended mutex once in the decoder's callback, as in the control.

## What a paced run writes

Everything a plan of version 3 writes, with the report schema `pilot-ls-native/v4`, and:

| File | Holds |
|---|---|
| `pacing/releases.jsonl` | One `pace-release/v1` line per opportunity: `sequence`, `opportunity_id`, `sfn`, `slot`, `offset_us`, `outcome` (`on_time`, `late`, `skipped`), `overrun`, the times `t_release_ns`, `t_free_ns`, `t_ready_ns`, `t_start_ns`, `t_done_ns`, `fill_us`, and per decode `backend`, `key`, `outcome`, `skip`, `t_start_ns`, `t_callback_ns`, `service_us` |
| `native.json`, `pacing` | A `pace-report/v1` object: the offered schedule, the start of the schedule on the host clock, and the counts `offered`, `taken`, `on_time`, `late`, `skipped`, `overruns`, `max_lateness_ns`, `max_excess_ns`. With a lead-in, its member `lead_in`: the releases and the lead, how many decodes were issued and decoded, when the first started and the last was done, how long before the first release that was, and the service times by backend |
| `pacing/lead_in.jsonl` | Only with a lead-in. One `pace-lead-in/v1` line per decode that was issued ahead of the schedule: `sequence`, the opportunity it repeats with `opportunity_sequence`, `opportunity_id`, `sfn`, `slot`, `grant_id` and `offset_us`, `lead_us`, `issued`, the times `t_issue_ns`, `t_ready_ns`, `t_start_ns`, `t_done_ns`, `fill_us`, `released`, and per decode `backend`, `key`, `outcome`, `crc_ok`, `payload_match`, `t_start_ns`, `t_callback_ns`, `service_us` and `path` with the stage times of the receiver |

`mps_pacing.open_output` reads such a directory with the readers of the earlier sprints and then checks every release line against the rule, against the offered schedule of the plan (`offered_matches`) and against the radio records. A job that started before its release time, an offset that is not the plan's, a count that the lines do not give and a decode without its radio record are each refused by name. With a lead-in it reads `pacing/lead_in.jsonl` as well: every line against the opportunity it repeats and against its own times, the member of the report against the lines, and the radio records against both, so that a release that names the first use of a processor on which the lead-in decoded is refused. `pacing/` holds nothing else.

## The launcher

```bash
python scripts/run_m1.py mps start|status|stop [--pipe-dir DIR] [--log-dir DIR] [--visible DEVICE]
python scripts/run_m1.py arm --spec FILE --record FILE
```

The MPS control daemon of this checkout keeps its pipe and its log under `build/mps`, apart from any daemon of the system: both directories are set explicitly for the daemon and for every client. An arm specification is `m1-arm/v1`; the record of a launch is `m1-launch/v1`.

| Arm | Producer | Consumers | CUDA clients |
|---|---|---|---|
| `dsp_only` | `pilot_ls`, paced, tap off | none | receiver |
| `capture_only` | `pilot_ls`, paced, O2 and O3 on their rings | `obs_tail` on each ring: reads and releases every record, infers nothing | receiver |
| `ai_only` | `sense_feed`: the O3 records the capture-only arm kept, at the pace of their slots | `sense_host` | sensor |
| `concurrent` | `pilot_ls`, paced, O2 and O3 on their rings | `sense_host` on O3; the reference sensor of Sprint 3.4 on O2, on the CPU, outside the PHY CPU set | receiver, sensor |
| `concurrent_reference` | as `concurrent` | the cuSense-like branch on the reference sensor in the interpreter | receiver, reference sensor |

Order of a launch. Every consumer is started first and must say that it is ready: a sensor loads its artifact, creates its CUDA context and warms its engine before the receiver exists. Then the producer starts, waits for the consumers of its rings and runs its schedule. Nothing is ended before the producer has ended. The receiver runs on the CPUs 5-9 and 15-19, `sense_host` on one core outside them, and no process is started with a shell. The launcher places a process on any online CPU of the machine, whatever CPUs the launcher itself may run on, reads the affinity back from the kernel into the record, and names a process that does not run where it was asked as a problem of the arm.

**The label MPS.** While the processes run, the launcher reads the server list and the client list of the control interface. An arm is labeled `mps` only when one reading names every CUDA process of the arm as a client of the one server on the one device. A refusal carries the list: `the sensor process 4242 is not a client of the MPS server 900: it shares the device by time slicing. Client list of the server 900: 4100`. A second server, a client that is no process of the arm and a device other than the daemon's are refused as well. An arm outside the daemon carries the label `time_slicing`, and the driver runs one as its own labeled control.

**Faults.** One per arm at most: `consumer_exit` ends `sense_host` with SIGKILL at a given time after the start of the offered schedule; `consumer_restart` then starts `obs_tail` in its place. `pilot_ls` announces that start once, in its log, when its preparation is done and before the schedule exists (`pilot_ls: schedule start N ns on the host monotonic clock`), and the launcher reads it there. A producer without a schedule, `sense_feed`, is timed from its start. The record of the fault names its clock. The launcher never signals the producer. The gate requires that the receiver decodes every released grid and that what the ring lost is counted.

## Contention

No arm slows, queues or reconfigures the receiver, and no arm enlarges a ring. The controls act on the sensor client only, one at a time, in this order:

| Control | What changes | Read back |
|---|---|---|
| `mps_default` | Nothing. Release `immediate` | The default active thread percentage and the default client priority of the server |
| `mps_priority` | `CUDA_MPS_CLIENT_PRIORITY=1` in the environment of the sensor: below normal | What the launcher added to the environment of the sensor, in its record, and the default client priority of the server |
| `mps_thread_pct` | `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` in the environment of the sensor | `get_active_thread_percentage` of the server |
| `release_after_radio` | The sensor submits a window after the remaining service time of its last uplink slot | The allowance in the plan of the sensor |

The rule compares the PUSCH service-time tail of the capture-only arm with that of the concurrent arm on the same offered schedule: `tail-shift-rule/v1` with a quantile, a least sample count, and an allowed shift that is the larger of an absolute term and a fraction of the base tail. The rule in `configs/integration/mps_pacing.json` (99th percentile, 1,000 samples, 25 us or 10 percent) has the status `proposed`: S4.6 declares the rule of the profile. With fewer samples than the rule needs it gives no verdict, and the gate says so. The first control whose shift does not exceed the rule is the control in effect. A control that this host or its MPS release lacks is a planned cell with its reason.

## The scores of two decodes

The check `concurrent same decisions` compares what `sense_host` scored beside the receiver with what it scored alone, from the records that the capture-only arm of the same repetition kept. The two arms read two decodes of one grid, and the receiver's estimate is not bit-reproducible between two decodes: [Sprint 5.2 measured that](NATIVE_ESTIMATE.md). A window is therefore compared by what its sensor was given, which the digest of its input says.

| Window | Rule | Bound |
|---|---|---|
| Same input: both outputs name one digest | The sensor was given the same bytes and has to give the same score: the logits lie within the bound | `expect.same_input_logit_tolerance`, 1e-6, the bound of the live case of Sprint 3.6 for a record that is read twice |
| Other input: the two decodes differ | The logits lie within the bound | `expect.logit_tolerance`, 0.05, as declared before the first run |
| A decision that differs, inside the bound of its window | A flip at the threshold: the threshold lies between two logits that are at most the bound apart, so both lie within the bound of it. Listed with both logits and both margins, and no failure | The rule of the parity of Sprints 3.5 and 3.6 |
| A decision that differs otherwise | A failure: the logits lie further apart than the bound, the threshold does not lie between them, or the two outputs name two thresholds | |

The check passes when the two arms share a window and no window breaks a rule. Its line states the counts of both kinds of window, the largest difference of each, how many decide alike, and every flip. `mps_pacing.decision_agreement` is the rule, one call per repetition. `comparison.json` holds the result under `concurrent.agreement`, also when a later case fails, and the record holds it under `decisions`.

**Why no share of equal decisions is asked.** Until the run with the lead-in the check required that every window decides alike in both arms (`expect.min_decision_agreement: 1.0`), and it computed the logit differences without grading them. That rule cannot hold on the entries this gate replays. The threshold of the artifact is frozen by the rule of Sprint 3.5 as the logit of one validation window, the target-absent window of rank 17, and this gate replays validation entries. So a window of this gate lies at the threshold by the rule that set it. The window 2 of the episode `387b28319ca93c0d` does: in its usual decode `sense_host` scores it at -7.699032, 9.2e-05 below the threshold of -7.698940, and no other window of the gate lies within 2e-03 of it. Its decision is decided by the last bits of the receiver's estimate. The gate does not read the freeze of Sprint 3.5, so that this is the window of rank 17 follows from its distance and is not proven here. 31 of 34 scorings of that window by `sense_host` in the first four Spark runs had one input and that logit. Three had the input of another decode, with -7.699052, -7.699307 and -7.696203, and the last of them lies above the threshold. The three earlier runs passed the check because no two decodes of that window fell on two sides of the threshold, and not because the rule held. [What the run with the lead-in showed](#what-the-run-with-the-lead-in-showed)

**What the rule adds.** The exact half is new and is stricter than the old rule: wherever the sensor was given the same bytes it must give the same score, beside the receiver as alone. That is the statement about the sensor under MPS, and the sensor outputs of the five Spark runs hold it in 3,687 of 3,687 windows. What is left to the tolerance is the receiver's own irreproducibility, which Sprint 5.2 recorded and which no arm of this sprint changes. The tolerance of 0.05 is the declared one and was not fitted: the largest difference that two decodes have shown is 3.68e-03 in 1,104 windows.

**How it is checked.**

| Where | What |
|---|---|
| `test_mps_pacing.py`, `ScoreTests` | The rule on decision lines: the same input with the same score, inside and beyond its bound; two decodes inside and beyond the tolerance; the flip the Spark measured, with its logits, listed and no failure; the same two logits from one input, a flip further apart than the tolerance, a flip the logits do not explain and two thresholds, each a failure |
| `test_mps_pacing_driver.py` | The driver on a double that rewrites one score of a sensor so that its reader still accepts the output. Another logit for the same input bytes and another decode with a logit 0.2 away fail `concurrent same decisions`. A window 1e-03 above the threshold beside the receiver and 1e-03 below it alone, from two decodes, passes every case and is listed in the check and in the record |
| `test_accept_s4_1.py` | What the gate script runs on the record, taken from the script: the scores in its line, a window that breaks the rule, a flip away from the threshold, and a record that was not written |
| `make accept-s4.1` | The check `concurrent same decisions` on the three repetitions, and the same of the record |

## The record

`mps_pacing.json` (`mps-pacing/v1`, `"graded": false`) holds, per launch of every arm: the label with its proof, the processes with their commands, CPUs, environments and whether they map a Python library, the counts of the pacer, the service times (median, 99th percentile, maximum), the account of each ring, the decisions and latencies of the sensor; and once: the cold start of the receiver, apart (`cold_start`, [the lead-in](#the-lead-in)); the scores of the sensor beside the receiver against the sensor alone, with the counts by repetition and every flip (`decisions`, [the scores of two decodes](#the-scores-of-two-decodes)); the frozen conditions (CPU sets, ring depths, wait modes, order of the arms, package and tool hashes, thermal readings), the contention result, the faults, the time-slicing control and the load: `scripted: true`, and no UE, no 5GC and no PRACH procedure attached.

The paced receiver fixes the map between the radio clock and the host clock: the slot of an opportunity begins at its release time. Three latencies per window follow from it: from the release of its first slot to its decision, from the release of its last slot, and from the moment the receiver was done with its last slot. They are recorded beside the latency of the cuSense boundary (Sprint 3.6) and are not graded.

**A window closes with the last uplink slot of its period.** The sensor therefore submits after the burst it belongs to, into the slack before the next one. With the TDD pattern of the demonstration the number of decodes that ran on the device during a sensor submission can be zero by design. The driver records the overlap count and grades only that both clients did real work over a common interval of the arm.

## Gate

`make accept-s4.1` (`scripts/accept_s4_1.sh`) on the DGX Spark that passed `make accept-s3.6`. It needs the accepted Sprint 3.6 run and the Sprint 3.5 run it names with its episode set, the patched receiver in `build/ocudu-spark-o3`, the environment of `make env-full` and `make core-env`, TensorRT and `nvidia-cuda-mps-control`. It rebuilds the adapter (not OCUDU), verifies the sensing core, and runs the driver in twelve cases: `build`, `control`, `mps`, `schedule`, `dsp_only`, `capture_only`, `ai_only`, `concurrent`, `contention`, `runtime`, `faults`, `records`. With the lead-in of its configuration it grades 45 checks, three more than a cold run: one `lead-in` per arm with a receiver. The check `concurrent same decisions` grades [the scores of two decodes](#the-scores-of-two-decodes). The gate script requires of the record that no window breaks that rule and that every flip lies at the threshold, and it names a record that the driver did not write, which is the case whenever a case before the last one failed. It reads validation entries only. No held-out episode is read before S4.6 has frozen the profile.

## What the accepted run measured

`make accept-s4.1` passed on `spark-ecf8` on 2026-11-23 (UTC) at commit `ed0354d`, run `s41-spark-ecf8-20261123T222740Z`: 64 gate lines, 58 PASS and 6 NOTE, and no failure; 12 of 12 cases and 45 of 45 checks of the driver with its hardware gate closed; 743 unit tests with none skipped; 99 adapter tests without a compiler warning; `pilot_ls` `844aa4be0802`, and five other tools of the adapter with the bytes that the Sprint 3.5 run verified. The controls are the accepted runs `s36-spark-ecf8-20261122T194723Z` and `s35-spark-ecf8-20261122T155753Z`: 20 validation pairs of `demo-set-v1`, 960 opportunities and 320 windows. No held-out episode was read. Every plan carries a lead-in of six decodes, 10 ms ahead.

**Graded.**

| What | Result |
|---|---|
| Sharing | Every timed arm carries the label MPS on one reading of the client list of the server on the GB10: the receiver in `dsp_only` and `capture_only`, the sensor in `ai_only`, both process identifiers in `concurrent`. The driver itself is refused the label with the empty client list. Read back: default active thread percentage 100.0, default client priority 0 |
| Start order and placement | In every repetition `sense_host` had loaded its artifact, created its context and warmed its engine, and was a client of the server, before the receiver was started. It maps no Python library. It ran on CPU 4, the receiver on 5-9 and 15-19, each read back from the kernel |
| One realization | Both plans bind one replay plan and one offered schedule: 960 releases over 1,635 ms at distances of 500, 4,000 and 5,000 us. Every arm with a receiver was offered and released the 960 releases in each of three repetitions |
| Decodes | In each of `dsp_only`, `capture_only` and `concurrent`: 2,880 of 2,880 released grids gave the original bytes with a CRC pass on the resident CUDA path, without an anomaly. No release was skipped and none overran the allowance |
| Schedule | No job started before its release and no offset moved. A start lies 0.01 us after its release in the median and 0.03 us at the 99th percentile. Two of 8,640 jobs started late, by 17.9 us at most |
| Lead-in | In every repetition of the three arms with a receiver the six decodes of the lead-in gave the original bytes and were done 3,812 us or more before the first release. No release is the first use of its processor, and none of the first six releases was skipped |
| Export | `capture_only`: 2,880 records of each stage published, read and released, none dropped. `concurrent`: 2,880 estimates to `sense_host` and 2,880 O2 records to the delay-Doppler branch on the CPU, none dropped. 960 windows scored in `ai_only` and in `concurrent`, without a gap |
| Scores | 960 windows scored beside the receiver and, from the buffered records, by the sensor alone. 647 were given the same input bytes in both arms: 647 have the same logit and the same map bit for bit. 313 were scored from two decodes that differ: their logits lie within 2.24e-03, with 0.05 allowed. 960 decide alike, no flip |
| Contention | The 99th percentile of the PUSCH service time moved by +1.3 us from capture only to concurrent, with 25.0 us allowed by the proposed rule: the control in effect is `mps_default`, and no control was applied |
| Faults | `sense_host` killed 700 ms after the announced start of the schedule: the receiver was never signalled, ended by itself and decoded 958 of 958 released grids; 419 estimates published, 539 without a credit dropped and counted, 8 pending at the cutoff. With `obs_tail` taking over: 959 of 959 decoded; 865 published, 94 dropped, 8 reclaimed with the closed epoch, and the new consumer kept 440 records of a new epoch |
| Records | 43 processes of 17 launches with their command, CPUs, environment, times and status; `mps_pacing.json` with `"graded": false`, with `cold_start` and `decisions` |

**Recorded, not graded.** The host ran the performance governor at a load average near 1.9, the GPU at 38 to 42 C. Every figure is valid for that state. No sample is left out of a statistic.

| Quantity | Median | 99th percentile | Most | Samples |
|---|---|---|---|---|
| PUSCH service time, tap off | 183.0 us | 237.1 us | 343.6 us | 2,880 |
| PUSCH service time, capture only | 178.2 us | 229.7 us | 482.4 us | 2,880 |
| PUSCH service time, concurrent | 178.1 us | 231.0 us | 408.2 us | 2,880 |
| PUSCH service time beside the Python reference sensor | 178.0 us | 239.6 us | 410.3 us | 960 |
| PUSCH service time, time slicing | 177.3 us | 261.8 us | 433.6 us | 960 |
| cuSense boundary, `sense_host` beside the receiver under MPS | 142.7 us | 264.1 us | 398.8 us | 960 |
| cuSense boundary, `sense_host` alone under MPS | 142.7 us | 440.9 us | 952.1 us | 960 |
| cuSense boundary, `sense_host` beside the receiver, time slicing | 215.3 us | 277.6 us | 386.1 us | 320 |
| cuSense boundary, Python reference beside the receiver under MPS | 2,151.8 us | 4,334.1 us | 51,088 us | 312 |
| Release of the first slot of a window to its decision | 1,335.4 us | 1,471.4 us | 1,641.6 us | 960 |
| End of the radio job of the last slot to the decision | 142.8 us | | | 960 |

The sensor figures of the three repetitions are those of the slowest repetition, and the largest value is that of all three. By repetition the boundary beside the receiver is 141.8, 142.5 and 142.7 us in the median and 187.1, 204.1 and 264.1 us at the 99th percentile; alone it is 142.7, 139.8 and 140.0 us and 440.9, 165.8 and 186.6 us.

| The cold start, apart: decode of the lead-in, nine runs | Least | Median | Most |
|---|---|---|---|
| First | 1,226.0 us | 1,282.7 us | 1,302.4 us |
| Second | 220.8 us | 231.0 us | 234.6 us |
| Third | 180.3 us | 189.1 us | 205.2 us |
| Fourth | 179.7 us | 183.3 us | 197.5 us |
| Fifth | 175.9 us | 180.8 us | 184.4 us |
| Sixth | 172.4 us | 178.8 us | 186.7 us |
| First release of the schedule behind it | 181.0 us | 186.4 us | 199.0 us |

Of the first decode, 903 to 948 us are the decode stage and 305 to 337 us the demodulation. The lead-in was done 3,812 to 3,826 us before the first release.

**What the figures say.**

- **The native sensor beside the receiver costs the receiver nothing that this run resolves.** Capture only and concurrent agree within 0.1 us in the median and 1.3 us at the 99th percentile. The arm with the tap off lies 5 us above both in the median, which is the scatter between launches and no cost of the tap.
- **The cold start is out of the schedule.** Without a lead-in every run lost two releases, one overrun and its first window. With it 8,640 of 8,640 releases were released and decoded, and the first release takes what every other takes. The first decode of a receiver process costs 1.23 to 1.30 ms and is reported from `cold_start` only. [The cold start](#the-cold-start)
- **The sensor computes the same thing beside the receiver as alone.** The same input bytes gave the same logit and map in 647 of 647 windows. What differs between the arms is the receiver's estimate between two decodes, by 2.24e-03 in the logit at most in this run. [The scores of two decodes](#the-scores-of-two-decodes)
- **MPS costs the sensor about 10 us in the median, and its tail is not yet stable.** Sprint 3.6 measured 130.0 us without MPS and without a receiver; here it is 139.8 to 142.7 us alone and 141.8 to 142.7 us beside the receiver. The 99th percentile of a repetition of 320 windows lies between 166 and 441 us alone and between 187 and 264 us beside the receiver: three or four windows decide it. S4.6 needs more windows per statement of a tail. The published 241 us remain no like-for-like figure: [the differences](SENSE_CORE.md#what-the-spark-run-closed).
- **Time slicing costs the sensor 73 us in the median**, 215.3 us against 142.7 us, on the same arm outside the daemon.
- **The Python reference sensor is 15 times slower, and what it costs the receiver differs between runs.** Its latency is 2,151.8 us in the median and 51 ms at most. Beside it the receiver's 99th percentile was 239.6 us without an overrun in this run, and 491.4 us with 13 overruns in the first accepted run. It supports no deadline in either.
- **A kill of the sensor reaches the receiver's timing and not its decodes.** The signal was sent 700 ms into the schedule and the process was gone at 816 ms. Between 703 and 784 ms the receiver's service time rose to 2.1 ms at most, 2.3 ms in the arm with the takeover, with four overruns in each arm and two and one releases skipped. Every released grid decoded. The slow decodes at the start of the fault arms of the first accepted run did not recur.
- **The proposed tail rule decides inside the scatter between runs.** Its shift was -13.2, +27.1, -11.6, +2.7 and +1.3 us in five runs of the same arms. S4.6 declares the rule with the sample count that separates a sensor from that scatter.
- **No submission of the sensor overlapped a decode**, in 960 windows of each arm: a window closes with the last uplink slot of its period. These arms show two clients on one device over a common interval of 4.9 s and never at one instant.

| Run | Windows with the same input in both arms | Same logit and map, bit for bit | Windows with two decodes that differ | Largest logit difference | Flips |
|---|---|---|---|---|---|
| First, `013427Z` | 928 | 928 | 29 | 1.74e-03 | 0 |
| Second, `030415Z` | 890 | 890 | 67 | 3.68e-03 | 0 |
| First accepted, `140213Z` | 450 | 450 | 507 | 2.26e-03 | 0 |
| With the lead-in, `161419Z` | 772 | 772 | 188 | 3.49e-03 | 1, at the threshold |
| Accepted, `222740Z` | 647 | 647 | 313 | 2.24e-03 | 0 |
| All five | 3,687 | 3,687 | 1,104 | 3.68e-03 | 1 |

## What the first accepted run measured, without a lead-in

This run is kept for one reason: it is the measurement of a cold schedule, which [the lead-in](#the-lead-in) was built on. Its service statistics leave out the first decode of every run, its counts hold two skipped releases and a lost window per run, and its check of the decisions was the rule that [the scores of two decodes](#the-scores-of-two-decodes) replaced. None of its figures is pooled with those of the accepted run.

`make accept-s4.1` passed on `spark-ecf8` on 2026-11-23 (UTC) at commit `3302329`, run `s41-spark-ecf8-20261123T140213Z`: 61 gate lines, 55 PASS and 6 NOTE, and no failure; 12 of 12 cases and 42 of 42 checks of the driver with its hardware gate closed; 735 unit tests with none skipped; 96 adapter tests without a compiler warning. `pilot_ls` has the bytes `4eb0b20881de` of the second run, and five other tools of the adapter have the bytes of the adapter that the Sprint 3.5 run verified. The controls are the accepted runs `s36-spark-ecf8-20261122T194723Z` and `s35-spark-ecf8-20261122T155753Z`: 20 validation pairs of `demo-set-v1`, 960 opportunities and 320 windows. No held-out episode was read.

**Graded.**

| What | Result |
|---|---|
| Sharing | Every timed arm carries the label MPS on one reading of the client list of the server on the GB10: the receiver in `dsp_only` and `capture_only`, the sensor in `ai_only`, both process identifiers in `concurrent`. The driver itself is refused the label with the empty client list. Read back: default active thread percentage 100.0, default client priority 0 |
| Start order and placement | In every repetition `sense_host` had loaded its artifact, created its context and warmed its engine, and was a client of the server, before the receiver was started. It maps no Python library. It ran on CPU 4, the O2 reader on CPU 3, the receiver on 5-9 and 15-19, each read back from the kernel |
| One realization | Both plans bind one replay plan and one offered schedule: 960 releases over 1,635 ms at distances of 500, 4,000 and 5,000 us. Every arm with a receiver was offered the 960 releases in each of three repetitions |
| Decodes | In each of `dsp_only`, `capture_only` and `concurrent`: 2,874 of 2,874 released grids gave the original bytes with a CRC pass on the resident CUDA path, without an anomaly. 6 of 2,880 releases were skipped and stay in the denominator |
| Schedule | No job started before its release and no offset moved. A start lies 0.01 us after its release in the median and 0.03 us at the 99th percentile |
| Export | `capture_only`: 2,874 records of each stage published, read and released, none dropped. `concurrent`: 2,874 estimates to `sense_host` and 2,874 O2 records to the delay-Doppler branch on the CPU, none dropped |
| Decisions | 957 windows scored beside the receiver and, from the buffered records, by the sensor alone: 957 decide alike, and the largest logit difference is 2.26e-03. Graded by the rule of that run, equal decisions. By [the rule since](#the-scores-of-two-decodes) the same outputs give 450 windows with the same input and the same score bit for bit, and 507 on two decodes that differ |
| Faults | `sense_host` killed 700 ms into the schedule: the receiver was never signalled, ended by itself and decoded 956 of 956 released grids; 416 estimates published, 540 without a credit dropped and counted, 8 pending at the cutoff. With `obs_tail` taking over: 951 of 951 decoded; 850 published, 101 dropped, 8 reclaimed with the closed epoch, and the new consumer kept 434 records of a new epoch |
| Records | 43 processes of 17 launches with their command, CPUs, environment, times and status; `mps_pacing.json` with `"graded": false` |

**Recorded, not graded.** The host ran the performance governor at a load average near 1.9, the GPU at 39 to 43 C. Every figure is valid for that state.

| Quantity | Median | 99th percentile | Most | Samples |
|---|---|---|---|---|
| PUSCH service time, tap off | 179.2 us | 229.5 us | 446.2 us | 2,871 |
| PUSCH service time, capture only | 179.6 us | 247.4 us | 471.3 us | 2,871 |
| PUSCH service time, concurrent | 179.0 us | 235.8 us | 435.0 us | 2,871 |
| PUSCH service time beside the Python reference sensor | 182.8 us | 491.4 us | 802.2 us | 956 |
| PUSCH service time, time slicing | 177.0 us | 262.3 us | 374.0 us | 957 |
| cuSense boundary, `sense_host` beside the receiver under MPS | 142.9 us | 204.6 us | | 957 |
| cuSense boundary, `sense_host` alone under MPS | 139.8 us | 204.0 us | | 957 |
| cuSense boundary, `sense_host` beside the receiver, time slicing | 215.6 us | 236.4 us | | 319 |
| cuSense boundary, Python reference beside the receiver under MPS | 2,230.2 us | 4,689.8 us | 56,059 us | 311 |
| Release of the first slot of a window to its decision | 1,336.8 us | 1,446.6 us | | 957 |
| End of the radio job of the last slot to the decision | 143.0 us | | | 957 |

The sensor figures of the three repetitions are those of the slowest repetition. The contention rule compares the two 99th percentiles of capture only and concurrent: -11.6 us against 25.0 us allowed, so the control in effect is `mps_default` and no control was applied.

**What the figures say.**

- **The native sensor beside the receiver costs the receiver nothing that this run resolves.** The three arms with a receiver agree within 0.6 us in the median. Their 99th percentiles lie 18 us apart, with the concurrent arm between the two others.
- **MPS costs the sensor about 10 us and the receiver beside it about 3 us.** Sprint 3.6 measured 130.0 us in the median without MPS and without a receiver, this run 139.8 us alone under MPS and 142.9 us beside the receiver. The published 241 us remain no like-for-like figure: [the differences](SENSE_CORE.md#what-the-spark-run-closed).
- **Time slicing costs the sensor 73 us in the median**, 215.6 us against 142.9 us, on the same arm outside the daemon.
- **The Python reference sensor is no second client for a paced receiver.** Beside it the receiver's 99th percentile is 491.4 us against 235.8 us, 13 jobs overran the allowance of 500 us against one in a run of the concurrent arm, and 18 started late against two at most. Its own latency is 15.6 times that of the native sensor in the median.
- **A kill of the sensor reaches the receiver's timing and not its decodes.** The signal was sent 700 ms into the schedule and the process was gone at 814 ms. In that time the receiver's service time rose to 2.3 ms at most and one release was skipped, six in the arm with the takeover. Every released grid decoded. Both fault arms of this run also hold service times of 0.5 to 1.0 ms in the first 140 ms of their schedule, which the second run did not show; their cause is not known.
- **The first decode of a receiver process is cold, and on a schedule it costs the first window.** [The cold start](#the-cold-start)
- **The proposed tail rule decides inside the scatter between runs.** Its shift was -13.2 us, +27.1 us and -11.6 us in three runs of the same arms, and the tap-off arm lay above the capture-only arm in one of them. S4.6 declares the rule with the sample count that separates a sensor from that scatter.
- **No submission of the sensor overlapped a decode**, in 957 windows of each run: a window closes with the last uplink slot of its period. These arms show two clients on one device over a common interval of 4.9 s and never at one instant.

## The cold start

The first decode of a receiver process pays for what the processor sets up on its first use. The first accepted run measured it in every one of its twelve launches with a receiver and no fault.

| Decode of the process | Above the median of its run, mean | Least | Most |
|---|---|---|---|
| First | 1,185.1 us | 1,085.1 us | 1,330.1 us |
| Second | 56.2 us | 48.4 us | 60.0 us |
| Third | 20.6 us | 11.2 us | 32.0 us |
| Fourth | 5.5 us | -0.3 us | 14.2 us |

In the nine graded runs the first decode took 1,262.7 to 1,507.2 us, 1,345.8 us in the median, against 179 us for a decode of the same run. The receiver's own stage times name where: the decode stage 930 to 1,123 us against 139 us, the demodulation 313 to 418 us against 38 us. The first submission of the estimate tap costs 5 to 7 us against 2.2 us.

On an offered schedule that first decode is an overrun of the allowance, and the two releases behind it in the same burst, 500 and 1,000 us later, find the worker busy beyond their latest start and are skipped. The first window of the run then lacks two of its three slots and is a gap. **The loss is 2 of 960 releases, one overrun and 1 of 320 windows in every run of every arm**: all 6 skipped releases and all 3 overruns of each timed arm of the first accepted run, and its 3 gaps. The service-time statistics leave the first use out, as since Sprint 2.4, and they keep the second decode, which lies at the 99th percentile of its arm, and the third.

The counts are true and they are not about pacing: a receiver that serves a cell is warm. The remedy is [the lead-in](#the-lead-in), and the figures of this section stay the measurement of the loss without one: the configuration of the gate carries them as `pacing.cold_start_without_lead_in`, and the record of every later run repeats them beside its own cold start.

## The lead-in

**What it is.** A paced run may decode the first releases of its schedule once more, ahead of it:

```text
lead(k) = start + offset(k) - lead        for k below the number of lead-in releases
```

With the values of the gate, six releases and a lead of 10 ms, the lead-in is the first two bursts of the schedule two periods early: its decodes are issued 10.0, 9.5, 9.0, 5.0, 4.5 and 4.0 ms before the first release, with the grids, the PDUs and the transport blocks of the opportunities they repeat. A decode of the lead-in is issued at its time, or as soon as the one before it is done, and is never skipped: its first decode is the cold one and runs through the times of the two behind it.

**What it is not.** A decode of the lead-in is no release. It has no line in `pacing/releases.jsonl` and no radio record. Nothing of it is tapped: the stage O2 is not read, and the receiver has no observer registered while the lead-in runs, so that no estimate is captured, no credit of a ring is taken and no record reaches a consumer. It is kept in a paced book of its own whose tokens lie behind every token of the schedule (`isac_pacing::paced_book` with a first token), so that no count, no anomaly and no sample of the schedule's book can be one of its decodes. The offered schedule is fixed before the lead-in and is not moved by it: the start of the schedule is announced first, and the lead-in runs inside the start delay.

**Where the cold start goes.** With a lead-in the first use of every processor is one of its decodes. No release of the schedule carries `first_use`, the service statistics of the report leave nothing out, and the reader refuses an output in which a release names the first use of a processor that the lead-in decoded on. `pacing/lead_in.jsonl` holds what each decode of the lead-in took, with the stage times of the receiver, and the member `lead_in` of the report holds when the lead-in was done and how long before the first release. `mps_pacing.cold_start` reduces one run, and the record of the gate holds, under `cold_start`: the lead-in, the service time by decode of the lead-in over all runs with a receiver, the stage times of the first decode, the time between the lead-in and the first release, what the first release of the schedule took, and the loss that the accepted run measured without a lead-in. **A figure of the schedule is read from `radio/radio.jsonl` and `pacing/releases.jsonl` alone, and the cold start from `pacing/lead_in.jsonl` alone**: the two never share a sample.

**Why this and not something smaller.**

| Alternative | Why it was not taken |
|---|---|
| One decode before the schedule | It removes the 1.2 ms of the first use, the two skipped releases and the lost window. It leaves the second and the third decode of the process in the schedule, 56 and 21 us above the median, and the second lies at the 99th percentile of its arm in every run |
| Several decodes, one behind the other, before the start delay | The receiver would then idle for the 50 ms of the start delay before its first release, which no burst of the schedule follows. Issued at the offsets of the schedule, a whole number of periods ahead, the lead-in ends about 3.8 ms before the first release, as every burst ends about 3.8 ms before the next: the first burst of the schedule is preceded by what precedes every other |
| Leaving the first releases out of the statistics afterwards | The skipped releases and the lost window would stay in the counts, and samples would be removed by a rule that was chosen after they were seen |
| The warm-up launch of the gate | It warms the host and not the next process: after the warm-up launch of the first accepted run the first decode of every later process still took 1.26 to 1.51 ms |

Six decodes and 10 ms are proposed values, as the allowance is. The transient lasts four decodes, and six are two whole bursts. S4.6 declares them with the profile.

**What it does not change.** The first submission of the estimate tap, 5 to 7 us against 2.2 us, is still in the first release that is tapped, and the report already leaves the first copy out of its tap statistics. The first slot of a burst takes 3 to 11 us more in the median than the two behind it, after 4 ms of idle time: that is a property of the schedule and stays in every statistic. A plan without `lead_in`, or a configuration with `pacing.lead_in: null`, runs cold as the first accepted run did, and the gate then says so and grades no lead-in.

**How it is checked.**

| Where | What |
|---|---|
| `lead_in_rules` (CTest) and `test_mps_pacing.py` | The rules of a lead-in in both languages, with the same six times |
| `book_lead_in` (CTest) | Two books on one pool and one decoder: no decoder sees a token twice, a token of one book is no job of the other, counts, records and anomalies are apart, both grids are free for the schedule, and a lead-in decode that never calls back keeps its grid |
| `lead_in_takes_the_cold_start` (CTest) | On the real clock, a decoder whose first use takes 1.3 ms on the schedule of the demonstration. Without a lead-in the first release overruns and the two behind it are skipped. With one, the first use is a decode of the lead-in and no release pays for it |
| `test_mps_pacing.py`, tree and line cases | A line of the lead-in against the opportunity it repeats and its own times, sixteen refusals; the lead-in of `pilot_ls` in its order, without a tap, a publication, a release or a use of the schedule's book |
| `test_mps_pacing_driver.py` | The driver on a double whose lead-in has a first decode of 1.3 ms: the cold start in the record and in no statistic. Three faults fail their checks: a tool that runs no lead-in, a release that is still the first use, a lead-in that ends inside its schedule. The same configuration without a lead-in grades none |
| `make accept-s4.1` | Per arm with a receiver, the check `lead-in`: every decode of the lead-in gave the original bytes and was done before the first release, no release is the first use of its processor, and none of the first six releases of a schedule was skipped. The gate script requires the same of the record and counts the lines of `pacing/lead_in.jsonl` with `grep` |

`pilot_ls.cpp` compiles against the pinned headers with and without the route to the estimate, and the development host has no OCUDU build and no GPU. Three things were open before the lead-in ran on the Spark, and the run `s41-spark-ecf8-20261123T161419Z` settled each: a receiver that decoded six grids 4 to 10 ms ago decodes its first release in its steady time, closing and opening the observer of the receiver around the lead-in leaves the capture as it was, and the figures themselves. [What the run with the lead-in showed](#what-the-run-with-the-lead-in-showed)

## What the first Spark run showed

`make accept-s4.1` ran on `spark-ecf8` on 2026-11-23 (UTC), run `s41-spark-ecf8-20261123T013427Z` at commit `0799d7a`: 11 of 12 cases and 40 of 42 checks. The paced loop, MPS on the GB10, the engine beside the receiver, the takeover by `obs_tail` and the time-slicing control all ran. Three defects of this sprint's tooling showed, and none of the receiver, the ring or the sensor.

- **The fault was timed from the start of the producer.** `pilot_ls` prepares for about eight seconds before its schedule starts, so a sensor that was ended 700 ms after the receiver started had never attached. In `consumer_exit` the receiver then waited 120 s for a consumer and ended with status 1; in both arms the sensor was gone before the receiver had a CUDA context, so no reading could name both and the label was refused. The fault is now timed from the announced start of the schedule.
- **No process ran where it was asked.** The driver moves itself to the tool cores, and the launcher took the CPUs it may itself run on for the CPUs of the machine: every requested core was outside that set, no affinity was applied, and the receiver and the sensor inherited the tool cores 0-2 and 10-14. The record said so (`"applied": null`), and nothing graded it. The service times and latencies of that run are therefore those of the Cortex-A725 cores and are not the figures of this sprint. The launcher now places on any online CPU, reads the affinity back and fails an arm that is not placed.
- **The second reader counted nothing.** The gate runner looked for `"schema": "pace-release/v1"` with a space, and the native tool writes none. The files held 960 releases, 2 skipped and 958 decoded radio records.

## What the second Spark run showed

`make accept-s4.1` ran again on `spark-ecf8` on 2026-11-23 (UTC), run `s41-spark-ecf8-20261123T030415Z` at commit `aad5f35`, with the three corrections above. Every line of the hardware side passed: the driver graded 12 of 12 cases and 42 of 42 checks with its hardware gate closed, every process ran on the CPUs it was asked for, both faults came 700 ms into the schedule, and the second reader counted 960 releases, 2 skipped and 958 decoded radio records. The gate was still not accepted, on its line `unit tests`: 2 of 735 failed.

- **Two tree checks of earlier sprints read past their section.** `tests/unit/test_snapshot_ring.py` (Sprint 3.2) and `tests/unit/test_native_estimate.py` (Sprint 5.2) require that the timed loop of `pilot_ls` writes nothing, and both took the text of that loop from its first line to the section `The drain`. Sprint 4.1 put the paced run between the two, so both checks had come to read the paced section as well, and the line that announces the start of the schedule, which the first correction added there, is an output. Nothing is wrong in the tool: the line is written once, after the preparation and before the schedule exists. Both checks now end where the timed section ends, and `tests/unit/test_mps_pacing.py` holds the whole paced section to the same words, with that one line named. The correction changes no source of a tool and no script of the gate: `pilot_ls` keeps the bytes of this run.
- **Why it reached the Spark.** After the first correction the modules of this sprint and the tree check of `test_pilot_ls.py` were run again, and the whole suite was not. The text of `pilot_ls.cpp` is read by the tree checks of four sprints (3.1, 3.2, 5.2 and 4.1). The whole suite is run on the final tree before a patch of this sprint leaves the development host.

Three things of that run are findings and no defects of the gate. They are recorded here for S4.6, which calibrates on these arms.

- **The first decode of a receiver process is cold, and on a schedule it costs the first window.** In every run of every arm the first decode took 1.25 to 1.49 ms (2.54 ms in the warm-up arm) against 0.19 ms in the median. It is an overrun, and the two releases behind it in the same burst, 500 and 1,000 us later, are skipped: that is every one of the 6 skipped releases and 3 of the overruns of each timed arm, and the first window of every run is a gap for it, 319 windows scored of 320. The first accepted run measured the same. [The cold start](#the-cold-start)
- **The proposed tail rule decides inside its own noise.** The 99th percentile of the arm with the tap off, 246.1 us, lies 9.2 us above that of the capture-only arm, 236.9 us, which does more. The concurrent arm lies 27.1 us above capture only, against 25.0 us allowed, so `mps_priority` was applied and gave +11.3 us. The first run, on other cores, gave -13.2 us without a control. With 2,871 decodes per arm the rule of 25 us or 10 percent does not separate the sensor from the scatter between runs.
- **No submission of the sensor overlapped a decode**, in 957 windows, as the section on the record explains. The 27.1 us are therefore no evidence of two clients on the device at one time.

## What the run with the lead-in showed

`make accept-s4.1` ran on `spark-ecf8` on 2026-11-23 (UTC), run `s41-spark-ecf8-20261123T161419Z` at commit `c00b707`, with the lead-in: 737 unit tests with none skipped, 99 adapter tests without a compiler warning, `pilot_ls` `844aa4be0802`. The driver graded 7 of 12 cases and 33 of 38 checks. One check failed, `concurrent same decisions`. The four cases behind it, `contention`, `runtime`, `faults` and `records`, run only when every arm before them ran clean. They were not run, so the record was not written and the gate was not accepted.

**The lead-in did what it was built for.** Ten launches with a receiver, nine of them graded, each with a lead-in of six decodes.

| What | Without a lead-in (first accepted run) | With the lead-in (this run) |
|---|---|---|
| Releases skipped, per run of 960 | 2 | 0 |
| Overruns of the allowance, per run | 1 | 0 |
| Windows scored, per run of 320 | 319 | 320 |
| Released grids decoded to the original bytes, per arm | 2,874 of 2,874 | 2,880 of 2,880 |
| Releases that are the first use of the processor | one per run | none |
| Samples left out of the service statistics | the first decode of every run | none |

| Decode of the lead-in | Least | Median | Most |
|---|---|---|---|
| First | 1,268.3 us | 1,320.7 us | 1,578.4 us |
| Second | 233.7 us | 242.8 us | 260.0 us |
| Third | 194.9 us | 209.4 us | 211.1 us |
| Fourth | 180.6 us | 186.0 us | 203.7 us |
| Fifth | 180.2 us | 185.5 us | 198.0 us |
| Sixth | 178.5 us | 183.5 us | 193.5 us |

The lead-in was done 3,805 to 3,820 us before the first release. Behind it the first release of a schedule took 185.0 to 196.4 us, 189.6 us in the median, and the five behind it 190.7, 189.3, 194.6, 189.0 and 191.9 us in the median, against 189.9 us over all 8,640 releases: the transient of four decodes is no longer in the schedule. Of the first decode, 922 to 1,199 us are the decode stage and 327 to 434 us the demodulation. The capture is as it was: 2,880 records of each stage published, read and released in `capture_only`, 2,880 estimates to `sense_host` and 2,880 O2 records to the delay-Doppler branch in `concurrent`, none dropped, and 960 windows scored in each of the two arms with a sensor, without a gap. Five of 8,640 jobs started late, by 39.4 us at most, in the middle of their runs.

Recorded, not graded: the PUSCH service time over 2,880 decodes per arm, with nothing left out, is 187.2 us in the median, 219.5 us at the 99th percentile and 443.4 us at most with the tap off, 186.3, 256.6 and 451.6 us with both stages exported, and 191.2, 259.3 and 448.9 us beside the sensor. The medians lie 7 to 12 us above those of the first accepted run. That is a difference between runs of the gate that was there before the lead-in: the medians of single launches of these three arms were 180 to 192 us in the second run, 177 to 182 us in the first accepted run and 184 to 192 us in this one.

**The defect: a rule that asked two decodes for equal decisions.**

- **What failed.** 960 windows were scored beside the receiver and alone: 959 decide alike. In the third repetition the window 2 of the episode `387b28319ca93c0d` scored -7.696203 beside the receiver and -7.699032 alone, 2.7e-03 above and 9.2e-05 below the threshold of -7.698940. The two inputs have two digests: two decodes of one grid.
- **Why that window.** It lies at the threshold, and the threshold was set on a window of this set. The rule of Sprint 3.5 freezes the threshold as the logit of the validation target-absent window of rank 17, and this gate replays validation entries. Of the 320 windows it replays, one lies within 1e-03 of the threshold, two within 3.7e-03, the largest difference two decodes have shown, and four within 0.05.
- **Why it is no defect of the lead-in, of the sensor or of MPS.** Over the four Spark runs the two arms scored 3,831 windows. 3,040 were given the same input bytes in both arms, and all 3,040 have the same logit and the same map, bit for bit. 791 were given two decodes that differ, and their logits lie within 3.68e-03. One of them is the flip.
- **What was wrong in the gate.** The check required equal decisions in every window, and it did not grade the logit differences it computed, although the configuration declared a tolerance for them. The first rule cannot hold on these entries, and the second was missing. [The scores of two decodes](#the-scores-of-two-decodes) is the corrected rule. On the sensor outputs of all four runs it passes, with one listed flip.
- **Why it reached the Spark.** `decision_agreement` had no unit case, and both sensors of the double are given the same records, so no rehearsal had ever compared two decodes. Both are added: four unit cases with the logits the Spark measured, two faults of the double that fail the check, and one condition, a flip at the threshold between two decodes, that the driver must pass and list.
- **A record that was not written was reported with a traceback.** The gate script now says that the driver writes the record in its last case.

| Run | Windows with the same input in both arms | Same logit and map, bit for bit | Windows with two decodes that differ | Largest logit difference | Flips |
|---|---|---|---|---|---|
| First, `013427Z` | 928 | 928 | 29 | 1.74e-03 | 0 |
| Second, `030415Z` | 890 | 890 | 67 | 3.68e-03 | 0 |
| Accepted, `140213Z` | 450 | 450 | 507 | 2.26e-03 | 0 |
| With the lead-in, `161419Z` | 772 | 772 | 188 | 3.49e-03 | 1 |

Two things of this run are findings for S4.6.

- **How often two decodes differ depends on the launch.** Between the two arms of one repetition 7 to 224 of about 320 windows were given two inputs, 2 to 70 percent. A window is the mean of three records, and Sprint 5.2 found 3 of 48 records with other bytes.
- **A decision within 4e-03 of the threshold is not reproducible on this receiver**, and the validation entries hold a window at the threshold by the rule that set it. Agreement of decisions on validation entries is stated with a tolerance or not at all.

## What the Spark has shown, and what it has not

Nothing of this sprint could run where it was written, which has no GPU, no MPS and no OCUDU build. The runs above showed each of these on the GB10.

- **The paced loop of `pilot_ls`**, in 17 launches of the accepted run.
- **The lead-in**, in 17 launches of the accepted run and ten of the run before it: no release of a baseline arm skipped, none overran, and the first window of every run scored.
- **MPS on the GB10.** The release on the Spark answers `get_server_list`, `get_client_list`, `get_default_active_thread_percentage` and `get_default_client_priority`. The second run applied `mps_priority` to the sensor, with the variable in the environment of the sensor process; whether the server honours it was not separated from the scatter between runs.
- **A consumer that takes over**, and a sensor that is killed, 700 ms into the schedule.
- **Time slicing beside the daemon**: a labeled control, never labeled MPS.

Not shown: `mps_thread_pct` and `release_after_radio` were never applied, because an earlier control or none was in effect. No contention control has run with the lead-in, since none was called for. The proposed values, 500 us of allowance, 250 us of latest start, six decodes of lead-in 10 ms ahead, core 4 for the sensor and core 3 for the O2 reader, held in every run they were used in and are still proposed. S4.6 declares them.
