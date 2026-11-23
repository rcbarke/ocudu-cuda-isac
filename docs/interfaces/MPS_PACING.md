# Paced clients under MPS

The contract of Sprint 4.1 (`mps_pacing`): the paced plan of the staged harness, the release rule, the files a paced run writes, the launcher `scripts/run_m1.py` with its arms, the proof of the label MPS, the contention controls and the record `mps_pacing.json`. [Process layout and contention](../architecture/RUNTIME.md#process-layout) | [observation contract](OBSERVATION.md) | [ring](SNAPSHOT_RING.md) | [native sensing core](SENSE_CORE.md) | [sprint card](../dev/DEVELOPMENT.md#phase-s4)

**Status.** Built, and not yet accepted. The portable half is graded here: the release rule in two languages, the paced book against the ledger of Sprint 2.4, the launcher on a fake host, and the driver as a rehearsal on a stand-in control (12 of 12 cases, 39 of 39 checks, hardware gate open). `make accept-s4.1` on the DGX Spark is the gate. [What only the Spark can show](#what-only-the-spark-can-show)

## What the sprint adds

| Path | Holds |
|---|---|
| `native/ocudu_adapter/pacing/` | `pacing_core.h`, `pacing_core.cpp`: the pacer and the paced book. Host-free: it includes the dispatch core and the channel application core and nothing of OCUDU. `tests/pacing_core_test.cpp`: eight cases in CTest |
| `native/ocudu_adapter/observer/pilot_ls.cpp`, `observer_plan.h` | The plan version 4 and the paced loop of the staged harness. The loops of the plan versions 1 to 3 are unchanged |
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

`mps_pacing.release_offsets` derives the offsets from the frame and the slot of each opportunity: a slot lasts `slot_us`, consecutive entries are separated by `gap_slots` idle slots. A paced plan is refused unless its evidence mode is `timed` and it names as many releases as the replay has opportunities. The driver refuses a configuration whose rings do not wait in `spin` mode, whose receiver decodes on another processor than CUDA, or that enlarges a ring for an arm. `pilot_ls --pacing` prints the plan version the build reads.

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
| `native.json`, `pacing` | A `pace-report/v1` object: the offered schedule, the start of the schedule on the host clock, and the counts `offered`, `taken`, `on_time`, `late`, `skipped`, `overruns`, `max_lateness_ns`, `max_excess_ns` |

`mps_pacing.open_output` reads such a directory with the readers of the earlier sprints and then checks every release line against the rule, against the offered schedule of the plan (`offered_matches`) and against the radio records. A job that started before its release time, an offset that is not the plan's, a count that the lines do not give and a decode without its radio record are each refused by name.

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

Order of a launch. Every consumer is started first and must say that it is ready: a sensor loads its artifact, creates its CUDA context and warms its engine before the receiver exists. Then the producer starts, waits for the consumers of its rings and runs its schedule. Nothing is ended before the producer has ended. The receiver runs on the CPUs 5-9 and 15-19, `sense_host` on one core outside them, and no process is started with a shell.

**The label MPS.** While the processes run, the launcher reads the server list and the client list of the control interface. An arm is labeled `mps` only when one reading names every CUDA process of the arm as a client of the one server on the one device. A refusal carries the list: `the sensor process 4242 is not a client of the MPS server 900: it shares the device by time slicing. Client list of the server 900: 4100`. A second server, a client that is no process of the arm and a device other than the daemon's are refused as well. An arm outside the daemon carries the label `time_slicing`, and the driver runs one as its own labeled control.

**Faults.** One per arm at most: `consumer_exit` ends `sense_host` with SIGKILL at a given time after the producer started; `consumer_restart` then starts `obs_tail` in its place. The launcher never signals the producer. The gate requires that the receiver decodes every released grid and that what the ring lost is counted.

## Contention

No arm slows, queues or reconfigures the receiver, and no arm enlarges a ring. The controls act on the sensor client only, one at a time, in this order:

| Control | What changes | Read back |
|---|---|---|
| `mps_default` | Nothing. Release `immediate` | The default active thread percentage and the default client priority of the server |
| `mps_priority` | `CUDA_MPS_CLIENT_PRIORITY=1` in the environment of the sensor: below normal | What the launcher added to the environment of the sensor, in its record, and the default client priority of the server |
| `mps_thread_pct` | `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` in the environment of the sensor | `get_active_thread_percentage` of the server |
| `release_after_radio` | The sensor submits a window after the remaining service time of its last uplink slot | The allowance in the plan of the sensor |

The rule compares the PUSCH service-time tail of the capture-only arm with that of the concurrent arm on the same offered schedule: `tail-shift-rule/v1` with a quantile, a least sample count, and an allowed shift that is the larger of an absolute term and a fraction of the base tail. The rule in `configs/integration/mps_pacing.json` (99th percentile, 1,000 samples, 25 us or 10 percent) has the status `proposed`: S4.6 declares the rule of the profile. With fewer samples than the rule needs it gives no verdict, and the gate says so. The first control whose shift does not exceed the rule is the control in effect. A control that this host or its MPS release lacks is a planned cell with its reason.

## The record

`mps_pacing.json` (`mps-pacing/v1`, `"graded": false`) holds, per launch of every arm: the label with its proof, the processes with their commands, CPUs, environments and whether they map a Python library, the counts of the pacer, the service times (median, 99th percentile, maximum), the account of each ring, the decisions and latencies of the sensor; and once: the frozen conditions (CPU sets, ring depths, wait modes, order of the arms, package and tool hashes, thermal readings), the contention result, the faults, the time-slicing control and the load: `scripted: true`, and no UE, no 5GC and no PRACH procedure attached.

The paced receiver fixes the map between the radio clock and the host clock: the slot of an opportunity begins at its release time. Three latencies per window follow from it: from the release of its first slot to its decision, from the release of its last slot, and from the moment the receiver was done with its last slot. They are recorded beside the latency of the cuSense boundary (Sprint 3.6) and are not graded.

**A window closes with the last uplink slot of its period.** The sensor therefore submits after the burst it belongs to, into the slack before the next one. With the TDD pattern of the demonstration the number of decodes that ran on the device during a sensor submission can be zero by design. The driver records the overlap count and grades only that both clients did real work over a common interval of the arm.

## Gate

`make accept-s4.1` (`scripts/accept_s4_1.sh`) on the DGX Spark that passed `make accept-s3.6`. It needs the accepted Sprint 3.6 run and the Sprint 3.5 run it names with its episode set, the patched receiver in `build/ocudu-spark-o3`, the environment of `make env-full` and `make core-env`, TensorRT and `nvidia-cuda-mps-control`. It rebuilds the adapter (not OCUDU), verifies the sensing core, and runs the driver in twelve cases: `build`, `control`, `mps`, `schedule`, `dsp_only`, `capture_only`, `ai_only`, `concurrent`, `contention`, `runtime`, `faults`, `records`. It reads validation entries only. No held-out episode is read before S4.6 has frozen the profile.

## What only the Spark can show

Nothing below ran where this sprint was written, which has no GPU, no MPS and no OCUDU build.

- **The paced loop of `pilot_ls`.** It compiles against the pinned headers with and without the route to the estimate, and its two cores are tested without OCUDU. Its first execution is the gate's.
- **MPS on the GB10.** The commands `get_server_list`, `get_client_list`, `get_active_thread_percentage`, `get_default_active_thread_percentage` and `get_default_client_priority` follow the NVIDIA documentation. Whether the release on the Spark answers each of them, and whether it honours a client priority, is read on the Spark. A command that is refused makes its control a planned cell.
- **A consumer that takes over.** `obs_tail` attaching to a ring whose consumer was killed is exercised with the host twin for the kill only.
- **Time slicing beside the daemon.** A client outside the daemon while the daemon holds the device may be refused by the driver. The cell is then recorded as planned with the reason.
- **The proposed values.** 500 us of allowance, 250 us of latest start, core 4 for the sensor and core 3 for the O2 reader are first values. A run that skips more than ten releases per repetition fails `schedule kept` and names the count.
