# Radar cube, delay-Doppler view and causal features

Sprint 3.3, selector `pilot_features`. This page states what the sprint built, the cube and its axes, the specification a sensor is bound to, what a causal stream answers at each record, the view and the vector with a baked example, the files of a sensor, and how the sprint is accepted. The record a stream reads is fixed in the [observation contract](OBSERVATION.md), and the ring it arrives on in [Observations on the ring](SNAPSHOT_RING.md). Tickets: [S3.3](../dev/DEVELOPMENT.md#phase-s3), [stage registry](../architecture/RUNTIME.md#stage-registry), [episode with the known return](EPISODE.md#the-reference-pair-and-its-analytic-check).

**Status.** Delivered and unit accepted on a development host on 2026-11-21 (UTC). The integration gate `make accept-s3.3` is open until it has run on the DGX Spark. [What a development host checked](#checked-on-a-development-host) | [what the Spark run must show](#what-the-spark-run-must-show)

## Run it

```bash
# Unit gate: the reference on vectors with a known answer. NumPy, and cmake with a C++17 compiler
# for the case that attaches the sensor to a native producer of the ring. No OCUDU build, no GPU
python tests/unit/test_pilot_features.py -v
python tests/unit/test_core_imports.py -v

# Integration gate on a host that passed make accept-s3.2
make accept-s3.3

# The steps of the gate, one at a time
make adapter-build
s32=$(ls -d results/runs/s32-* | tail -1)
python tests/integration/pilot_features.py --config configs/integration/pilot_features.json \
    --s32-run "$s32" --output-dir results/runs/s33-spark-001/pilot_features

# By hand: the specification of the demonstration, the records a consumer kept, and a second reader
python -m ocudu_cuda_isac.sensing.features spec --output build/features-001/spec.json
mkdir build/features-001/out
python -m ocudu_cuda_isac.sensing.features replay --archive "$s32/snapshot_ring/timed/run/consumer-0" \
    --spec build/features-001/spec.json --output-dir build/features-001/out
python -m ocudu_cuda_isac.sensing.features show build/features-001/out --window 1

# Live: the reference sensor as the consumer of a ring, in its own process
python -m ocudu_cuda_isac.sensing.features follow --socket OUT/ring/o2.sock \
    --spec build/features-001/spec.json --output-dir build/features-001/live
```

`make accept-s3.3` writes `results/runs/s33-<host>-<UTC time>/` and ends with `Sprint 3.3: ACCEPTED on this DGX Spark.` It takes the newest accepted Sprint 3.2 run as its control, or the one named with `S32_RUN=`. The plans of that run name files of the Sprint 2 runs by absolute path, so those runs must still be where they were. It rebuilds the adapter and never rebuilds OCUDU. On a host whose OCUDU build has no CUDA, `make accept-s3.3 REHEARSAL=1` grades the host path against a rehearsed Sprint 3.2 run and ends with `Sprint 3.3: REHEARSED on this host, NOT ACCEPTED.`

## What the sprint builds

| Part | Where | Level |
|---|---|---|
| The cube, the delay-Doppler view, the compact vector, the causal stream, the specification, the registry of steps, the reference sensor `follow`, the reader of a sensor output | `src/ocudu_cuda_isac/sensing/features.py` | Python reference: NumPy on the CPU, `runtime: reference`. Imports the record reader and the slot times of the sensing package and nothing else of the project. |
| The evaluator side: the same O2 from files, windows cut with every record in hand, the view by explicit Fourier sums, the join to the truth of an episode entry | `src/ocudu_cuda_isac/episodes/pilot_features.py` | Offline Python with NumPy |
| The import rule of the sensing core | `tests/unit/test_core_imports.py` | Standard library |
| A callback in the reference reader | `ocudu_cuda_isac.sensing.obs.tail(on_record=...)` | One additive argument. Without it the reader of Sprint 3.2 is unchanged. |

No native source changes. The gate rebuilds the adapter and requires every executable to have the bytes of the accepted Sprint 3.2 run. Sprint 6.12 adds the native unit of every step under the same registry name and reuses the vectors of this sprint.

## From records to a cube

A stream reads `obs/v1` records of the layout `ocudu_o2_comb_c64`, `[rx_plane][reference_symbol][pilot]`, in the order of the ring. A cube is `X[frequency, slow time, rx]` in complex128, with one validity bit per element.

| Axis | Contents | Unit |
|---|---|---|
| `frequency` | The pilot comb of the record: column `c` lies on subcarrier `first_subcarrier + c * subcarrier_step` of the carrier grid | Hz from subcarrier 0: the spacing is `subcarrier_step` times the subcarrier spacing, 60 kHz for the demonstration |
| `slow_time` | One sample per selected reference symbol of each selected record, in the order of their times. The time of a sample is the start of the useful part of its symbol, by the rule of TS 38.211 5.3.1 that the sensing source records, counted from slot 0 of the epoch. | s from the first sample of the window |
| `rx` | The array elements of the antenna map in rising order. Plane `i` of a record holds element `antenna_map[i]`, so a record with a reversed map gives the same cube. | Array element |

**Validity.** An element without its bit is zero in the cube and is never read: a stored value behind a cleared bit, however large, changes nothing. A value that is not finite is treated as a hole. Nothing is interpolated. A hole is zero in both transforms, the mean of a line is taken over its valid samples, and every complete window records its holes, its fill and the lines of the cube that hold a hole.

**What a stream reads of a record.** The 29 header fields of `features.INPUT_FIELDS`, the validity bits and the payload. It does not read the host block, the grant key, the cell, the UE, the producer steps, the digest or the valid count. The episode key and the epoch group records into histories. The opportunity key names the contributing records of a window and enters no value.

## The specification: `pilot-features-spec/v1`

A sensor is bound to one specification before it reads a record. `python -m ocudu_cuda_isac.sensing.features spec` writes that of the delay-Doppler branch of the demonstration, and `configs/integration/pilot_features.json` carries the same object.

| Field | Content | Demonstration |
|---|---|---|
| `input` | Receiver stage, sensing source, layout and dtype of the records it reads | `O2`, `pusch_dmrs`, `ocudu_o2_comb_c64`, `<c8` |
| `radio` | `scs_khz` and `period_slots`, the TDD period in slots. A period is a whole number of subframes. | 30 kHz, 10 slots: 5 ms |
| `sequence` | `slots_in_period` and `reference_symbols`: which slots of a period and which reference-symbol rows of a record enter slow time | Slot 7, row 0: symbol 2 of the first uplink slot |
| `window` | `periods`, `hop_periods` and `min_valid_fraction` | 8 periods, a hop of 8, 0.9 |
| `steps` | One registry entry per stage: `cube`, `static`, `delay`, `doppler`, `features` | `comb_cube`, `window_mean`, `ifft`, `fft`, `dd_compact` |
| `view` | `delay_bins` kept, `doppler_grid` (null for `fft`, `bins` and `step_hz` for `ndft`) and `floor_db` | 32 bins, null, 120 dB |

**Homogeneous sequence or actual times.** The two DM-RS symbols of a slot are 321.09 us apart and the three uplink slots of a period 500 us, so the samples of a burst are not on the 5 ms grid. The step `fft` therefore accepts one slot of the period and one reference symbol only, and checks on the actual times of every window that they are equally spaced. The step `ndft` takes any selection and sums at the actual times on a declared Doppler grid. The demonstration uses `fft` on eight samples 5 ms apart, the lattice of the Sprint 2.2 check. A Doppler phase advances by `2 pi nu (1 + f / f_c) dt`: across the band it differs from the carrier value by 0.26 percent, 0.11 Hz for the reference return and far inside a bin of 25 Hz, so both steps use one Doppler shift for the band.

Refused before a record is read: another schema, an unknown stage or source, a layout without a cube, a subcarrier spacing that is not 15, 30, 60 or 120 kHz, a period that is no whole number of subframes, a slot outside the period, a list that does not rise, a window or a hop of no period, a valid share outside (0, 1], an unregistered step, more than one slot or symbol under `fft`, a Doppler grid under `fft` or none under `ndft`, fewer than two samples, no delay bin.

## Steps and where they run

Every step is an entry of `features.REGISTRY` and states its execution device. A sensor output and the gate's manifest list the resolved entries.

| Stage | Entry | What it does | Runs |
|---|---|---|---|
| `cube` | `comb_cube` | Pilot comb by slow time by array element, complex128, with validity bits | NumPy on the CPU |
| `static` | `window_mean` | Per frequency and element, subtracts the mean over the valid samples of the window: the zero-Doppler part of this window and of no other | NumPy on the CPU |
| `static` | `none` | Removes nothing | NumPy on the CPU |
| `delay` | `ifft` | Inverse FFT over the regular comb, rectangular taper | NumPy on the CPU |
| `doppler` | `fft` | FFT over a homogeneous sequence, rectangular taper | NumPy on the CPU |
| `doppler` | `ndft` | Fourier sum at the actual sample times on a declared grid, rectangular taper | NumPy on the CPU |
| `features` | `dd_compact` | The vector below | NumPy on the CPU |

All of them carry `runtime: reference`. `window_mean` is the static removal of this diagnostic view and belongs to no trained model: the background template of S3.5 and the clutter entries of S6.2 are separate entries of the clutter axis.

## What a stream answers

`Stream.push(record)` takes the next record of the ring and returns its answers at once, from what it has seen up to that record. A record of a slot outside the sequence causes no answer and is counted.

| Status | When | Carries |
|---|---|---|
| `ok` | The record completes a window: every sample of the sequence over all its periods is held | The axes, the view with its peak, the vector, the contributing records, the validity counts and the times |
| `insufficient_history` | A record of the sequence arrived and its window is still filling | `history.have` of `history.need` samples. No view, no vector. |
| `gap` | A window closed without one of its samples: at its last sample, or at the first later record of the sequence when the last sample itself never came | The missing samples by period, slot and symbol row. No vector. |
| `invalid` | A complete window whose valid share is below `min_valid_fraction`, whose view holds no moving power, or whose times are not equally spaced under `fft` | The reason. No vector. |
| `rejected` | A record that is no input of the specification: another stage, source, layout or dtype, another shape, an irregular comb, sample times that are not those of its slot | The reason. The history is not touched. |

**Histories.** The first record of the sequence starts a history, and its period is the first period of the first window. Later windows follow at the hop. Another epoch, another episode key or a slot that goes back starts a new history, and so does another allocation on a record of the sequence, so a window never spans two of them. A slot outside the sequence may carry another allocation and disturbs nothing. A stream keeps the samples of the open window and nothing older. `open_samples()` reports what was held when the records ended.

**Causality.** An answer is made when its record is pushed and is never revised. The unit gate and the driver replace every later record by `1e30`, change every field outside the input list, and put most of a window of an earlier epoch in front: the window must be the same line, bit for bit.

## The view and the vector

For a complete window, with `K` pilots, `N` samples and `M = min(delay_bins, K)`:

```
y[k, n, r] = x[k, n, r] - mean over the valid n of x[k, n, r]        (static: window_mean)
D[m, d, r] = (1 / (K N)) * sum over k, n of y[k, n, r] exp(+j 2 pi k m / K) exp(-j 2 pi n d / N)
P[m, d]    = sum over r of |D[m, d, r]|^2                             m = 0 .. M-1, d = -N/2 .. N/2-1
```

**Signs.** A path of delay `tau` and Doppler shift `nu` contributes `exp(-j 2 pi f tau) exp(+j 2 pi nu t)`. A later arrival lies at a higher delay bin. A phase that advances in slow time, a path that shortens, lies at a positive Doppler bin, and a receding target at a negative one. The delay axis starts at zero and is the absolute path delay: the direct path is not removed from it, and the static part reports its own strongest delay bin beside the peak.

| Quantity of the demonstration window | Value |
|---|---|
| Delay bin and unambiguous delay | 54.47 ns (306 pilots 60 kHz apart, 18.36 MHz) and 16.67 us |
| Delay kept in the view | 32 bins, 1,743 ns, 523 m of bistatic path |
| Doppler bin and unambiguous Doppler | 25.0 Hz (8 samples 5 ms apart) and -100 to +75 Hz |
| Acquisition | 35.03 ms from the first to the last contributing sample |
| Tapers | Rectangular on both axes: first sidelobes near -13 dB |

**The vector `dd_compact`.** The statistics exclude the zero-Doppler column when a static step runs, because that column holds what the step left. `median` is the median power of the remaining cells, floored at `floor_db` under the peak, so that a view without noise has finite features.

| Position | Name | Definition | Unit |
|---|---|---|---|
| 0 | `peak_to_median_db` | Strongest moving cell over the median | dB |
| 1 | `moving_to_static_db` | Power of the cube after static removal over the power of the removed part, over the whole comb | dB |
| 2 | `peak_delay_ns` | Delay of the strongest moving cell | ns |
| 3 | `peak_doppler_hz` | Doppler shift of that cell, signed | Hz |
| 4 | `doppler_centroid_hz` | Power-weighted mean Doppler shift of the moving cells, signed | Hz |
| 5 | `doppler_spread_hz` | Power-weighted standard deviation around the centroid | Hz |
| 6 on | `profile_db@<nu>Hz` | For each nonzero Doppler bin in rising order, its strongest cell over the median | dB |

The demonstration window gives 13 values. Every ratio is unchanged by a common real gain, so the factor 1.0012 between the stored pilot and its reference does not enter. The names travel with every vector, and a consumer binds to the list and not to the positions. Positions 2 and 3 locate the peak and carry no evidence of presence on their own. S3.4 selects the columns its head is trained on.

**A baked vector.** `tests/unit/test_pilot_features.py` builds a complete lattice of 16 pilots, two elements and eight periods: a static path of amplitude 3 at delay bin 1 and one return of amplitude 1 at delay bin 5 and Doppler bin +2, without noise.

| Name | Value | Why |
|---|---|---|
| `peak_to_median_db` | 120.0 | Nothing else moves, so the median is the floor |
| `moving_to_static_db` | -9.542 | `10 log10(1 / 9)` |
| `peak_delay_ns` | 5,208.33 | Bin 5 of 1,041.67 ns |
| `peak_doppler_hz` | +50.0 | Bin +2 of 25 Hz |
| `doppler_centroid_hz`, `doppler_spread_hz` | +50.0, 0.0 | One cell |
| `profile_db@+50Hz` | 120.0 | The peak |
| The six other profile values | 0.0 | At the floor |

The peak holds the power 2.0, `|1|^2` on each element. Conjugating the slow-time phase gives the same vector with -50.0 Hz and the 120.0 at `profile_db@-50Hz`. The tolerance of the unit gate is four decimal places, set by the single precision of the payload. A transposed axis moves the peak to another cell, a future sample changes the vector, and a weak return keeps its cell and lowers position 0: the three are told apart.

## What a sensor writes

`follow` and `replay` write one directory. `open_output` verifies it.

| File | Contents |
|---|---|
| `features.jsonl` | One `pilot-features/v1` line per answer, in order |
| `views/<line>.dd.f64` | The view of each complete window: little-endian float64 `[delay][doppler]`. Its SHA-256 is in its line, taken after the last record was read. |
| `sensor.json` | `pilot-features-report/v1`: the specification with its hash, the resolved steps with their devices, the runtime, the source (ring or archive), the counts by status and the samples left open |
| `records/` | `follow` only: the records it read, as the reader of Sprint 3.2 keeps them |

A line holds `status`, `reason`, `window` (identifier, epoch, episode key, first and last period), `trigger` (the record that caused the answer), `history`, `spec_sha256`, `runtime` and three host times: `t_ready_ns` and `t_publish_ns` of the trigger and `t_features_ns`, when the answer was complete. A complete window adds `axes` (the signed axes with resolution and alias limits), `view` (peak, static peak, excess delay in bins of the view, the share of the moving power in the peak cell, file and hash), `features` (names and values), `contributing` (sequence, opportunity key and slot of each record), `validity`, `time` (first and last sample in T_c and the acquisition time) and `input`. The window identifier is `<epoch>/<episode key>/<last period>`. Two runs on the same records agree on every field but the three host times. No line names a case, a position or a label.

## The evaluator side

`ocudu_cuda_isac.episodes.pilot_features` is what an evaluator holds against a stream after a run. It shares no window logic and no transform with the sensing side.

| Function | What it does |
|---|---|
| `observations(observer)` | Reads the estimates that an audited observer wrote as files: the same O2 that the ring carried |
| `offline_windows(found, spec)` | Sorts all observations and cuts windows by slot arithmetic, eligible or not |
| `oracle(window, spec)` | The view as explicit Fourier sums at the physical frequencies and at the sample times of the records, in double precision, and the vector from it |
| `differences(line, power, made)` | Where a window of the stream differs: cells within `1e-9` of the peak, features within `1e-6` in their unit, the same peak cell |
| `realizations(channel_plan)`, `expected_return(entry, index)` | The case behind each episode key, and the delay and Doppler shift of the first-order return from the geometry of the entry. This is truth, read after the features are written. |
| `resolved(line, expected)` | Whether the peak lies within one resolvable bin: the delay as it is, the Doppler shift folded into the unambiguous interval, with the fold reported |

## The gate

`make accept-s3.3` needs the accepted Sprint 3.2 run under `results/runs/`. The driver `tests/integration/pilot_features.py` grades eight cases.

| Case | What must hold |
|---|---|
| `control` | The Sprint 3.2 run is accepted. Its audited and its timed run of the pair verify and decoded one replay plan. The 48 payloads the native consumer kept from the ring have the SHA-256 of the files the audited observer wrote. The channel plan names the two cases of one episode entry. |
| `stream` | The records of the ring in sequence order give 2 complete windows, 14 answers of a filling window and 32 records outside the sequence, with no gap, no invalid window and no rejected record. Each window is answered at the record that completes it. The steps are registry entries on the CPU. |
| `offline` | Cut from the files of the audited run with every record in hand: the same eligible windows with the same contributing opportunities. By explicit Fourier sums: every cell within `1e-9` of the peak and every feature within `1e-6`. |
| `known-return` | On the target-present member the strongest moving cell is delay bin 3 and Doppler bin -2. It lies within one resolvable bin of the delay and the Doppler shift that the geometry of the entry gives, 179.8 ns and -43.3 Hz. It stands at least 15 dB above the median, and the strongest cell of the absent member at most 10 dB above its own. The cell at Doppler bin +2 of the same delay is at least 10 dB weaker. |
| `causality` | The records before the completing one give seven answers `insufficient_history` without a vector. With every later record replaced by `1e30`, with every field outside the input list changed, and with most of a window of an earlier epoch in front, the window is the same line. With the completing record scaled, it is not. |
| `mask` | Four elements of one observation lose their validity bit. With `1e30` stored behind them the line, the view and the vector are those with zero stored. The holes are counted and the return stays in its cell. Without one observation the window is a gap that names the missing sample. |
| `live` | `pilot_ls` publishes the pair again in the mode `timed` and `features follow` reads the ring in a second process. Every decode is the control's and every CUDA decode ran the resident path. The account balances with no drop. The sensor's records have the SHA-256 of the audited estimates. Its answers are those of the recorded stream line for line, with the same view hashes and the same vectors bit for bit. |
| `refusals` | Fifteen faulty specifications are refused by the check and by the command line with status 2, before a file is written |

Recorded and not graded: the axes with their resolution and alias limits, the delay bin of the static part beside the direct path of 66.9 ns, how far the return moves over the window, moving over static power with and without the target, the Doppler centroid, the share of the moving power in the peak cell, and the time from the publication of the completing record to the complete vector with `runtime: reference`. That time is Python behind a reader that sleeps. It is no latency of the sensing core, and S6.12 times the native units.

The gate runner adds its own lines: the unit suite, the unit gate and the import rule with no skipped test, the adapter build, the bytes of every executable against the accepted Sprint 3.2 run, and a second reader that prints the target-present window of the live sensor and compares the digest of its view, by `sha256sum`, with the digest its line names and with the view of the recorded stream.

**Why three sources.** The recorded ring is what a sensor saw. The files of the audited observer are the same O2 by another path, so a window logic or a transform that is wrong in the stream does not agree with them. The live ring shows that the stream does the same in a second process on records as they arrive, with the receiver decoding.

## Checked on a development host

On an x86_64 host (Ubuntu 24.04, GCC 13.3.0, CMake 3.28.3, Python 3.13, NumPy 2.5.3) without an OCUDU build and with two cores.

| | Development host |
|---|---|
| Unit gate | 27 of 27 cases, none skipped. The baked vector, both signs, the mask, causality, warm-up, gaps, sliding windows, refusals, the reader and the command line. |
| Reference pair by its analytic model | Synthesized on the lattice of the demonstration at 30 dB per resource element: peak at delay bin 3 and Doppler bin -2, 163.4 ns and -50.0 Hz, within one bin of 179.77 ns and -43.33 Hz. More than 18 dB above the median with the target and less than 10 dB without. The cell at +50 Hz is more than 15 dB weaker. |
| Evaluator against the stream | Largest cell difference below `1e-12` of the peak on both members |
| Reference sensor on a native producer | `features follow` in its own process read 24 records from `obs_feed` and kept them. `obs_feed` writes sample times of its own, so all 24 are rejected with that reason: the time rule is checked against a native producer. The kept records replay to the same lines. |
| Import rule | 3 of 3: no module of the sensing package imports an episode module, a host or a simulator, in its source and in a loaded interpreter |
| Integration driver | 8 of 8 cases and 28 of 28 checks against the double, on the small pair of the driver tests (two periods). Each of 17 injected faults fails the check that names it. |
| Driver at the demonstration profile | 8 of 8 cases and 28 of 28 checks with the shipped configuration, on a pair of eight periods built from the analytic model and passed through the doubles of Sprints 2.3 to 3.2: 48 records, 2 windows of 8 observations, peak at delay bin 3 and Doppler bin -2, 22.6 dB above the median against 4.5 dB without the target, 16.6 dB above the mirrored cell, largest cell difference to the oracle `7.1e-15`. A rehearsal, run once and not shipped as a test. |
| Gate runner | 8 tests against tool doubles |

The double of the live case models the ring and runs the real sensing code on its records. What this host leaves open is listed next.

## What the Spark run must show

| Open on the development host | Closed by `make accept-s3.3` |
|---|---|
| The return in the traced pair, through the pinned grid and the native observer | `known-return` on the records of the accepted Sprint 3.2 run |
| The reference sensor on the ring of `pilot_ls`, with the receiver decoding | `live`: no drop, the features of the recorded stream, the decodes of the control |
| The sample times of the native records against the rule of the reference | No record is rejected in `stream` and in `live` |
| The same executables as the accepted Sprint 3.2 run | The gate line on the bytes of every tool |

The driver's expectations for the Spark are those of the table of the gate. A drop in `live` fails the case: the reference sensor copies what it needs and releases at once, so the ring of eight slots is not expected to fill. If it does, the finding is the pace of a Python consumer and not the features, and the sensor's own account names it.

## Choices made here

- **One sample per period for the first transform.** The demonstration window is the homogeneous sequence of the card: symbol 2 of slot 7 of each period. The other five samples of a period are not used by it. `ndft` over all 48 samples of eight periods is registered and unit tested, and finds the same return.
- **Static removal inside the window.** The mean of the window is subtracted, so the view needs no calibration record and no state beyond the window. A target that does not move during the window is removed with the direct path.
- **No interpolation.** A hole stays a hole. With the masked mean the static part does not leak through a hole. Only the return loses the sample.
- **A window with a missing observation is not answered with a vector.** It is a `gap`. Tolerating a missing sample is a registry entry for S6.2 to add, with its own leakage statement.
- **One window per eight updates.** The hop of the demonstration is the window length, as the profile states. A hop of one period is a change of one field.
- **The delay axis is absolute.** The view reports the bin of the static part, and the excess delay in whole bins. A finer excess delay needs the interpolation of S6.2.
- **Reference runtime.** Every number here is NumPy on the CPU. The latency of these steps is measured by the native units of S6.12, which reuse the vectors of this page.
