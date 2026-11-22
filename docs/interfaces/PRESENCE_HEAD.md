# Presence head: one frozen linear model on the delay-Doppler features

Sprint 3.4, selector `presence_head`. This page states what the sprint built, the artifact and its arithmetic with a baked example, how the head binds to the names of a vector, the rules of the fit and of the threshold, the decision record, the files of a sensor and of a training run, the evaluator side, and how the sprint is accepted. The vector a head reads is fixed in [Radar cube and features](PILOT_FEATURES.md), and the record behind it in the [observation contract](OBSERVATION.md). Tickets: [S3.4](../dev/DEVELOPMENT.md#phase-s3), [stage registry](../architecture/RUNTIME.md#stage-registry), [episode set](EPISODE.md).

**Status.** Built and unit accepted on a development host on 2026-11-22 (UTC). The hardware gate `make accept-s3.4` is open until it has run on the DGX Spark: the receiver replay of an episode set, the fit on its records and the head on the CUDA device have been exercised against doubles only. [What a development host checked](#checked-on-a-development-host) | [what the Spark run must close](#open-until-the-spark-run)

## Run it

```bash
# Unit gate: the head on vectors with a known answer. NumPy and PyTorch, and cmake with a C++17
# compiler for the case that attaches the sensor to a native producer of the ring. No OCUDU build
python tests/unit/test_presence_head.py -v
python tests/unit/test_core_imports.py -v

# Integration gate on a host that passed make accept-s3.3, with the full profile in isac
make episode-set          # once: traces the demonstration set into data/episodes/demo-set-<UTC time>
make accept-s3.4

# The steps of the gate, one at a time
make adapter-build
s33=$(ls -d results/runs/s33-* | tail -1)
set=$(ls -d data/episodes/demo-set-* | tail -1)
python tests/integration/presence_head.py --config configs/integration/presence_head.json \
    --s33-run "$s33" --episode-set "$set" --output-dir results/runs/s34-spark-001/presence_head

# By hand, on the files of a gate run
run=results/runs/s34-spark-001/presence_head
python -m ocudu_cuda_isac.sensing.presence device
python -m ocudu_cuda_isac.sensing.presence inspect "$run/train/artifact"
python -m ocudu_cuda_isac.sensing.presence show "$run/live/held-out-00/sensor" --window 0
python scripts/train_presence.py --training "$run/samples/training.jsonl" --validation "$run/samples/validation.jsonl" \
    --spec "$run/spec.json" --config "$run/training.json" --output-dir build/presence-001
mkdir build/presence-001/offline
python -m ocudu_cuda_isac.sensing.presence replay --archive "$run/live/held-out-00/sensor/features/records" \
    --spec "$run/spec.json" --artifact build/presence-001/artifact --device cpu --output-dir build/presence-001/offline

# Live: the sensor with a head as the consumer of a ring, in its own process
python -m ocudu_cuda_isac.sensing.presence follow --socket OUT/ring/o2.sock --spec "$run/spec.json" \
    --artifact "$run/train/artifact" --device cuda --output-dir build/presence-001/live
```

`make accept-s3.4` writes `results/runs/s34-<host>-<UTC time>/` and ends with `Sprint 3.4: ACCEPTED on this DGX Spark.` It takes the newest accepted Sprint 3.3 run as its control, or the one named with `S33_RUN=`, and the newest `data/episodes/demo-set-*` as its stimulus, or the set named with `EPISODE_SET=`. It rebuilds the adapter, never rebuilds OCUDU and never traces an episode. On a host whose OCUDU build has no CUDA, `make accept-s3.4 REHEARSAL=1` grades the host path against a rehearsed Sprint 3.3 run, scores on the CPU and ends with `Sprint 3.4: REHEARSED on this host, NOT ACCEPTED.`

## What the sprint builds

| Part | Where | Level |
|---|---|---|
| The artifact `presence-head/v1`, the head with its batch-one interface, the decision record, the sensor `follow`, the offline commands `replay` and `score`, the readers, and the fit: normalizer, weights, threshold and the split rules | `src/ocudu_cuda_isac/sensing/presence.py` | Python reference: PyTorch, `runtime: reference`. Imports the feature reference and the record reader of the sensing package and nothing else of the project. |
| The training command: two sample files in, one artifact, two manifests and a freeze record out | `scripts/train_presence.py` | Offline Python with PyTorch. Imports the sensing package only. |
| The evaluator side: the split table of an episode set, the plans of its replay, the join of windows and decisions to the table, the parity of two decision sets | `src/ocudu_cuda_isac/episodes/presence_head.py` | Offline Python |
| One more argument in the plan of Sprint 2.3 | `ocudu_cuda_isac.episodes.grid_apply.entry_plan(bindings=...)` | Additive. Without it the plan is that of Sprint 2.3. |

No native source changes. The gate rebuilds the adapter and requires every executable to have the bytes of the accepted Sprint 3.3 run. The export route of S3.6 turns the artifact into an engine under the registry name `linear`.

## The artifact: `presence-head/v1`

One directory of two files. `open_artifact` verifies it, and refuses a directory that holds anything else.

| File | Contents |
|---|---|
| `head.json` | The input contract, the normalizer, the threshold, the hash of the weights and its own hash |
| `weights.f32` | One little-endian float32 per column, in the order of `input.columns`, then the bias |

| Field of `head.json` | Content | Demonstration |
|---|---|---|
| `entries` | The registry entry of each stage | `zscore`, `linear`, `frozen_threshold` |
| `input` | `features_entry`, `spec_id` and `spec_sha256` of the specification the vector comes from, `vector` (every name of that vector, in its order), `columns` (the names the head reads), `shape` and `dtype` of one input | `dd_compact`, `dd-uniform-8`, 13 names, 8 columns, `[1, 8]`, `<f4` |
| `normalizer` | `mean` and `scale` per column, and their SHA-256 | From the training windows |
| `weights` | File, layout, count, size and SHA-256 | 9 values, 36 bytes |
| `threshold` | `logit`, `probability`, the rule, the false-alarm target, the count of validation negatives, how many may lie above the threshold and how many do | Target 0.05: 2 of 40 allowed |
| `output` | The three formulas below | |
| `sha256` | SHA-256 over every other field in canonical JSON: the hash of the artifact | Named by every decision and by the freeze record |

It holds no sample, no label, no split, no scene and no path of a training run. A changed byte of the weights, an edited threshold or normalizer, a third file and an artifact other than the one a freeze record names are refused, each with its own message.

## The arithmetic

```
x        = the columns of the vector, by name, as float32            shape [1, F]
z        = (x - mean) / scale                                        normalizer: zscore
logit    = w . z + b                                                 model: linear
p        = 1 / (1 + exp(-logit))
present  = logit > threshold.logit                                   detector: frozen_threshold
```

All of it runs in float32 on the device the head was loaded on, one vector at a time. `Head.predict(x)` takes float32 of shape `[1, F]` as a NumPy array or a tensor and returns the logit, the probability and the time of the call in nanoseconds, from the vector in host memory to both values back in host memory. Another shape, another dtype and a value that is not finite are refused. `Head.score(names, values)` binds a named vector first and adds the decision and the SHA-256 of `x`, the input digest by which two runs are joined.

**A baked head.** `tests/unit/test_presence_head.py` writes an artifact with two columns `a` and `b`, the weights `[1, -1]`, no bias, mean 0 and scale 1.

| Vector | Logit | Probability | Why |
|---|---|---|---|
| `a = 2, b = 1` | 1.0 | 0.731059 | `2 - 1`, and `1 / (1 + e^-1)` |
| `a = 1, b = 1` | 0.0 | 0.5 | Not present at a threshold of 0: the bound is strict |
| `a = 5, b = 2` with mean `[1, 1]` and scale `[2, 0.5]` | 0.0 | 0.5 | `(5 - 1) / 2 - (2 - 1) / 0.5` |

The input digest of the first row is the SHA-256 of the eight bytes of `2.0` and `1.0` as little-endian float32. Read back from disk, a head gives the same logit bit for bit on the CPU, which is inside the `1e-6` of the card.

## Binding by name

A vector arrives with its names (`features.names` of a feature line), and a head is bound to them. The names must be those of `input.vector`, in that order: they are the contract of one specification, and a window of another specification has other names. The head then reads `input.columns` and nothing else of the vector.

| Refused | Message |
|---|---|
| The same names in another order | `the vector has its values in another order than the head was trained on: position 0 is 'b' and the head expects 'a'` |
| A missing or an extra name | `the vector is not that of the head: 1 names against 2, without 'b'` |
| A line of another specification | `the line was computed with the specification ..., and the head was trained on ...` |

The columns are selected by name when the head is trained. The training configuration lists names or patterns with `*`, and every pattern must select at least one name. The demonstration reads `peak_to_median_db` and `profile_db@*`: the peak over the median and the strongest cell of every nonzero Doppler bin. It does not read `moving_to_static_db`, which did not see the target of the reference pair, and it does not read `peak_delay_ns` and `peak_doppler_hz`, which locate a return and are those of a noise cell without one ([findings of Sprint 3.3](PILOT_FEATURES.md#accepted-run)). Changing a value the head does not read changes neither the logit nor the input digest.

## Training, and what it may read

`scripts/train_presence.py` takes two files of `presence-sample/v1` rows and no third one. A row is the vector of one complete window with what an evaluator joined to it: `window`, `episode` (the opaque key), `split`, `scene` (the trajectory), `pair` (the episode), `label`, `spec_sha256`, `names` and `values`.

| Step | Reads | Rule |
|---|---|---|
| Split check | Both files | Every row carries the split of its file. No window, episode key, pair or scene is in both. One specification, one list of names. A row of the held-out split is refused wherever it stands. |
| Manifests | Both files | `training_manifest.json` and `validation_manifest.json` are written before the fit: the hash of the sample file, the window identifiers, the episode keys, the scenes and what the rows are used for |
| Normalizer | Training rows | Mean and population deviation of every column, taken on the values as float32. A column that does not vary keeps the scale 1 and is listed. |
| Weights | Training rows | Mean binary cross-entropy of the logits plus `l2 / 2` times the squared norm of the weights, minimised from zero by L-BFGS with a strong Wolfe line search, in float64 on the CPU. The same rows give the same weights bit for bit. |
| Threshold | Target-absent validation rows | `allowed = floor(target_false_alarm * negatives)`. The threshold is the logit of the negative that ranks `allowed + 1` from the top, so at most `allowed` negatives lie above it. The logits are those of the deployed arithmetic: float32 on the CPU, one row at a time. A target-present row is never read. |
| Artifact | | Written, read back, and every row scored again: within `1e-6` of the fit |
| Freeze | | `freeze.json` names the hashes of the artifact, of its two files and of both manifests, the settings, the loss before and after, and the time on the wall clock and on the monotonic clock of the host |

The functions refuse by themselves what the command refuses: `fit_normalizer` and `fit_head` take rows of the training split only, `calibrate_threshold` rows of the validation split only.

**By hand.** Two training rows with the values 1 and 3 in a column give the mean 2 and the scale 1. Ten validation negatives with the logits 2.5, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0 and -3.0: a target of 0 puts the threshold at 2.5 with none above it, a target of 0.2 allows two and puts it at 1.0 with two above it, and a target of 0.05 allows none of ten.

**The split is made before a window exists.** The unit of the split is the trajectory of the episode set: an episode has the group of its trajectory, and the set records that assignment with its hash before an episode is traced ([episode set](EPISODE.md)). Both members of a matched pair share the episode and therefore the split. The table that maps an episode key to its split, its scene, its pair and its label is `split.json` of a gate run. It is the evaluator's. No record, no feature line, no decision and no artifact carries any of it.

## Decisions: `presence-decision/v1`

A head gives one decision per line of the feature stream, when that line is complete, and never revises it.

| Line of the stream | Decision |
|---|---|
| `ok` | `score` (logit and probability), `decision` (`present`, the threshold and the detector entry), `input` (the hash of the specification, the features entry, the column count and the SHA-256 of the input), `infer_ns`, and of its line `contributing`, `time` (first and last sample in T_c), `window` and `trigger` |
| `insufficient_history`, `gap`, `invalid`, `rejected` | The status and the reason of the line, with `score`, `decision` and `input` null |

Every decision names its `model` (head identifier, registry entry and the hashes of the artifact, of the weights and of the normalizer), the `runtime`, the `device`, and four host times: `t_ready_ns` and `t_publish_ns` of the trigger, `t_features_ns` and `t_decision_ns`. Two runs on the same records agree on every field but those times, `infer_ns` and the device. Of a line a head reads `status`, `spec_sha256`, `features.names` and `features.values`.

## What a sensor writes

`follow` and `replay` write one directory. `open_output` verifies it.

| File | Contents |
|---|---|
| `features/` | The feature output of Sprint 3.3: `features.jsonl`, `views/`, `sensor.json`, and with `follow` the records it read under `records/` |
| `decisions.jsonl` | One `presence-decision/v1` line per feature line, in order |
| `presence.json` | `presence-report/v1`: the manifest of the artifact, the resolved entries, the device with its name and capability, the PyTorch build, how often the process loaded the artifact, the source, and the counts |

`score` writes `decisions.jsonl` and `presence.json` for a feature output that exists already and names it by path.

**Loaded once, warm before the attach.** `follow` loads the artifact once, creates the context of its device, runs one window of noise through the feature steps and the mean of the training split through the head, and only then attaches. With `--ready-file` it writes that file at this point, with the device, the hash of the artifact and both warm-up times, and a launcher starts the receiver after it. The first inference of a process on a CUDA device takes seconds, and a first run of Sprint 3.3 lost six observations to a first use of 8 ms.

## Steps and where they run

| Stage | Entry | What it does | Runs |
|---|---|---|---|
| `normalizer` | `zscore` | Per column, `(x - mean) / scale` with the mean and the deviation of the training split | PyTorch, on the device selected when the head is loaded |
| `model` | `linear` | One weight per column and a bias, float32, batch one | PyTorch, on that device |
| `detector` | `frozen_threshold` | Present when the logit lies above a threshold calibrated on validation negatives and frozen | Python on the CPU |

All of them carry `runtime: reference`. The feature steps in front of them are those of Sprint 3.3, NumPy on the CPU.

## The evaluator side

`ocudu_cuda_isac.episodes.presence_head` is what an evaluator holds around the head. The sensing package does not import it.

| Function | What it does |
|---|---|
| `split_table(episode_set)` | The `presence-split/v1` table of a verified set: per episode its trajectory, its group, its grid gain and the opaque keys of its two realizations |
| `chunks(table, group, size)` | The entries of one group in runs of at most `size`, one grid gain per run |
| `bindings_of(entry)` | The transmit grid that an accepted entry recorded for each grant |
| `write_plans(run_id, rows, bindings, directory, ring)` | The channel plan, the replay plan and the plan of the tapped run for some entries of a set |
| `samples(output, table, role, expected)` | The complete windows of a feature output as sample rows, with an account of every realization without a window |
| `parity(device, offline, tolerance)` | Two decision sets joined by window identifier: the same input digest, the same contributing observations, logit and probability within the tolerance, and every decision that differs at the threshold with its margin |
| `evaluate(output, table, role, expected)` | The decisions by label at the frozen threshold, with a realization without a decision kept in its denominator as missing |

**Replaying a set.** The exporter of Sprint 2.2 writes the entries of a set without a transmit grid: only the reference pair of its gate binds one per grant. A set uses the same grant record and the same schedule, so its grants have the names and the record hashes of the pair's. `entry_plan(bindings=...)` binds an entry that names no grid to the grid that the pair recorded for the grant of that name, and the plan still requires the store to hold the grant record of the entry, by hash and field by field. The payload rule is `fixed`.

## The gate

`make accept-s3.4` needs the accepted Sprint 3.3 run under `results/runs/`, a traced episode set under `data/episodes/`, and NumPy and PyTorch in `isac`. The driver `tests/integration/presence_head.py` grades ten cases in this order.

| Case | What must hold |
|---|---|
| `control` | The Sprint 3.3 run is accepted and was graded with the specification of this configuration. The plan of its live run names the replay of the reference pair, whose entry verifies and binds the transmit grid of six grants, each in a store that verifies. |
| `split` | The set verifies with every episode and has the identifier and the counts the configuration expects. No trajectory is in two groups and both members of each pair are in one. Every entry lasts at least one window, and every grant has the record hash of the grant of that name in the pair. |
| `records` | `pilot_ls` replays the training and the validation entries in the mode `timed`, at most 25 entries per run, with `obs_tail` keeping every record. Every decode gives the original bytes on the host and on the CUDA processor, the CUDA decodes run the resident path, no observation is dropped, and every kept record is that of its account line. |
| `features` | The stream of Sprint 3.3 on the kept records gives one complete window per realization, with no gap, no invalid window and no rejected record. Each window is joined to the split table by its episode key, and no window, key, pair or trajectory is in both sample files. |
| `train` | `scripts/train_presence.py` exits with status 0. The artifact is `head.json` and `weights.f32`, with one weight per column and the bias, and names no evaluator field. The normalizer, computed again from the training windows alone, is that of the artifact. The threshold, computed again from the artifact read back and the validation negatives, is the same record. The loss falls from ln 2, and the artifact read back scores every row within `1e-6` of the fit. |
| `freeze` | The freeze record names the hashes of the artifact, of its files and of both manifests, hashed again here. The manifests list keys of their own split only and no held-out key or window. No plan, record or feature of a held-out entry exists in the run at this point. |
| `live` | The freeze record and the artifact are unchanged, and the freeze precedes the first held-out plan on the monotonic clock. `pilot_ls` replays the held-out entries with `presence follow` on the ring in a second process. The decodes are original and resident. The sensor loaded the frozen artifact once, ran on the selected device, was warm before the first record was published and read the ring to its end. No observation is dropped. There is one scored window per held-out realization and one decision per answer of the stream. |
| `parity` | `presence replay`, in a process of its own on the CPU, on the records the sensor kept: the same feature lines and view hashes, the same input digests and contributing observations, the logit within `1e-4` and the probability within `1e-5`, and the same decisions at the frozen threshold. |
| `causality` | A filling window has a decision without a score. With every later record replaced by `1e30`, the decision is the same line. With the peak, the validity counts, the axes and the host times of a line changed, and 1000 added to every value the head does not read, the logit and the input digest are the same, and they change with a column the head reads. No output of the sensor and no artifact names a trajectory, an episode or a label. |
| `refusals` | Eighteen refusals by the interface and eight by the command lines with status 2: shapes, dtype, ordering, another specification, an altered checkpoint, another artifact than the frozen one, samples of the wrong split and a scene in two splits |

Recorded and not graded: the detections and false alarms of the head on the samples of its fit and on the held-out episodes at the frozen threshold, with the area under the curve (`held_out_evaluation.json`), the time of one inference on the device, the time from the publication of the completing record to the decision, and the warm-up times. The quality and the timing of the demonstration are criteria of Sprint 4, at the operating point that S4.6 calibrates. The times here are PyTorch in a Python sensor behind a reader that sleeps, beside a receiver that shares the device without MPS: `runtime: reference`, and no latency of the sensing core.

The gate runner adds its own lines: the unit suite, the unit gate and the import rule with no skipped test, the adapter build, the bytes of every executable against the accepted Sprint 3.3 run, the CUDA device that PyTorch reaches in `isac`, and a second reader: `presence inspect` on the artifact with the hash of the freeze record, `presence show` on the output of the sensor, and `sha256sum` on the weights file.

**Order as evidence.** `manifest.json` of a run records the monotonic time of every receiver run, of the freeze and of the first plan of a held-out entry. Opening the set verifies the hashes of all its files, the held-out ones among them. That is a check of the stimulus. No estimate of a held-out grid exists before the freeze, because no held-out grid was filled.

## Files of a gate run

| File in `presence_head/` | Contents |
|---|---|
| `grade.json`, `grade.txt`, `comparison.json`, `manifest.json` | Cases, checks, measured values, command, commit, inputs, order of the run |
| `split.json` | The split table. Evaluator only. |
| `records/<role>-NN/` | The three plans, the native output and the archive of `obs_tail` of one receiver run |
| `features/<role>-NN/` | The feature output of that run |
| `samples/training.jsonl`, `samples/validation.jsonl` | The sample rows. Evaluator only. |
| `train/` | `artifact/`, `training_manifest.json`, `validation_manifest.json`, `fit.json`, `freeze.json` |
| `live/held-out-NN/` | The plans, the native output, `sensor-ready.json` and `sensor/`, the output of the presence sensor |
| `parity/held-out-NN/`, `parity/held-out-NN.json` | The offline output and the comparison |
| `held_out_evaluation.json` | Counts by label at the frozen threshold, `"graded": false` |
| `recover.json` | Where a second reader starts |

## Checked on a development host

On an x86_64 host (Ubuntu 24.04, GCC 13.3.0, CMake 3.28.3, Python 3.13, NumPy 2.5.3, PyTorch 2.15.0 without a CUDA device) without an OCUDU build and with two cores, on 2026-11-22 (UTC).

| | Development host |
|---|---|
| Unit gate | 30 of 30 cases, none skipped: the baked head, save and reload, shapes and ordering, 12 altered checkpoints, the normalizer and the threshold by hand, 13 split faults, the fit, the training command and its refusals, the stream, the output and its reader, the command line, the evaluator, and the sensor on a native producer |
| Sensor on a native producer | `presence follow` in its own process read 24 records from `obs_feed`, loaded the artifact once and wrote 24 decisions. `obs_feed` writes sample times of its own, so all 24 are rejected by the stream and carry no score. With sample times of a slot the kept records give three scored windows offline. |
| Import rule | 3 of 3, with `presence` among the modules of the sensing package |
| Integration driver | 10 of 10 cases and 35 of 35 checks against the doubles, on a set of four pairs of two periods written with the free-space model in place of the ray tracer. Each of 22 injected faults fails the check that names it: 4 in the training runs, 10 in the held-out run, 2 in the offline comparison, 5 in what was frozen and 1 in the head. |
| Gate runner | 8 tests against tool doubles |
| The fit at the size of the demonstration | A rehearsal, run once and not shipped as a test: the trajectories of `demo_set.json` with the free-space point-target model on the lattice of the demonstration, 600 training, 80 validation and 20 held-out windows of 13 values. The loss falls from 0.693 to 0.228 in 48 evaluations. At the threshold of two allowed validation negatives the head finds 34 of 40 validation and 10 of 10 held-out target-present windows, with 1 of 10 held-out false alarms. This says that the code runs at that size and that the columns carry a walking target in free space. It says nothing of the traced room. |

The doubles model the receiver and the ring and run the real sensing code, the real training command and the real evaluator. A window of two periods has one moving Doppler cell, so the head of the driver tests reads two columns, and what it learns from four windows means nothing.

## Open until the Spark run

| Open on the development host | Closed by `make accept-s3.4` |
|---|---|
| The receiver replay of set entries bound to the grids of the pair: 14 runs, 16,320 opportunities, never done before | `records`: every decode original, the account balanced with no drop |
| `pilot_ls --no-grids` in the mode `timed` | `records` and `live`: the native output verifies with its reader |
| One complete window per realization on traced channels | `features` |
| The fit on 600 traced windows | `train` |
| PyTorch on the GB10 in a second process beside the receiver, without MPS | `live`: the sensor ran on `cuda:0` and lost no observation |
| The same score on the device and on the CPU | `parity`: the largest difference of a logit against `1e-4` |
| The same executables as the accepted Sprint 3.3 run | The gate line on the bytes of every tool |

If `records` fails because the native output of a run without stored grids does not verify, set `receiver.keep_grids` to `true` in the configuration: the runs then store every grid as Sprint 3.2 did, about 4.6 GB for the demonstration set. A drop in `records` or `live` fails the case and is a finding about the pace of a consumer, as in Sprint 3.3. A flip in `parity` names a window whose logit lies within the tolerance of the threshold.

## Choices made here

- **A linear head on eight named columns.** The peak over the median and the Doppler profile. It is the baseline of S3.5 and stays in the demonstration as the scalar score on the delay-Doppler features.
- **One window per realization.** The hop of the demonstration is its window, and an entry of the demonstration set lasts one window. The set therefore gives 600 training, 80 validation and 20 held-out windows. More windows per episode need a shorter hop or longer entries: a change of the specification or of the set, each with a new hash.
- **The threshold is a logit of a validation negative.** No interpolation. With 40 negatives and a target of 0.05 two may lie above it. The false-alarm rate that follows on other episodes is measured in S4.
- **Training records come through a native consumer, held-out records through the sensor.** `obs_tail` keeps every record of the training and validation runs and the stream is replayed on its archive. The held-out runs are followed live by the sensor with the head. Both paths compute the same features: Sprint 3.3 proved the stream on a recorded ring and on a live one.
- **Two files, not a checkpoint of the framework.** Raw float32 and JSON load with any runtime, hash to one value and hold nothing that executes. The export route of S3.6 reads the same two files.
- **The receiver and the sensor share the device by its default time slicing here.** Two CUDA clients under MPS, with the sensor's context created first, are the subject of S4.1.
- **Reference runtime.** Every time of this page is PyTorch in a Python process. S3.6 builds the engine and times it in the native core.
