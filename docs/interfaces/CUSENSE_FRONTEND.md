# cuSense front end and location model: three stages on the receiver's own estimate

Sprint 3.5, selector `cusense_frontend`. This page states what the sprint built, the three stages with a baked example, the two input entries, the causal window stream, the model entry and its losses, the artifact, the rules of the fit and of the threshold, the decision record, the optional stage, the files of a sensor and of a training run, the evaluator side, and how the sprint is accepted. The input is the stage O3 of [the receiver's own estimate](NATIVE_ESTIMATE.md), with the stage O2 of [pilot least squares](PILOT_LS.md) as the second entry. The records, the ring and the reader are those of [the observation ring](SNAPSHOT_RING.md), and the split, the threshold rule and the baseline are those of [the presence head](PRESENCE_HEAD.md). The published system is cuSense: [Villa et al., arXiv:2512.06493v3](https://arxiv.org/abs/2512.06493v3), section 5.2.

**Status.** Implemented and checked on a development host: the unit gate, the import rule, the driver on doubles of the native tool and the gate runner on doubles pass. The hardware gate is open until `make accept-s3.5` passes on the DGX Spark. [Checked on a development host](#checked-on-a-development-host) and [what the Spark run has to close](#what-the-spark-run-has-to-close)

## Run it

```bash
# Unit gate: the three stages, the model and the artifact on values with a known answer. NumPy and
# PyTorch, and cmake with a C++17 compiler for the case that attaches the sensor to a native
# producer of the ring. No OCUDU build
python tests/unit/test_cusense_frontend.py -v
python tests/unit/test_core_imports.py -v

# Integration gate on the DGX Spark that passed make accept-s3.4 and make accept-s5.2
make accept-s3.5

# The steps of the gate, one at a time
make adapter-build-o3
s34=$(ls -d results/runs/s34-* | tail -1)
set=$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["episode_set"]["directory"])' \
    "$s34/presence_head/manifest.json")
python tests/integration/cusense_frontend.py --config configs/integration/cusense_frontend.json \
    --s34-run "$s34" --episode-set "$set" --output-dir results/runs/s35-spark-001/cusense_frontend

# By hand, on the files of a gate run
run=results/runs/s35-spark-001/cusense_frontend
python -m ocudu_cuda_isac.sensing.cusense device
python -m ocudu_cuda_isac.sensing.cusense inspect "$run/train/o3/artifact"
python -m ocudu_cuda_isac.sensing.cusense show "$run/live/held-out-00/sensor" --window 0
python -m ocudu_cuda_isac.sensing.cusense windows --archive "$run/records/training-00/consumer-o3" \
    --spec "$run/specs/o3.json" --output-dir /tmp/windows      # an existing, empty directory
python -m ocudu_cuda_isac.sensing.cusense replay --archive "$run/live/held-out-00/sensor/windows/records" \
    --artifact "$run/train/o3/artifact" --device cpu --output-dir /tmp/offline
python scripts/train_cusense.py --training "$run/samples/o3/training.jsonl" \
    --validation "$run/samples/o3/validation.jsonl" --spec "$run/specs/o3.json" --grid "$run/grid.json" \
    --config "$run/training-o3.json" --output-dir /tmp/train

# Live: the sensor as the consumer of the O3 ring, in its own process
python -m ocudu_cuda_isac.sensing.cusense follow --socket <native>/estimate/o3.sock \
    --artifact "$run/train/o3/artifact" --output-dir /tmp/sensor --ready-file /tmp/ready.json
```

## What the sprint builds

| Part | Where |
|---|---|
| The reference of the three stages, the artifact and the sensor | `src/ocudu_cuda_isac/sensing/cusense.py` |
| The model entry `cusense_cnn`, its losses and its flat weight file | `src/ocudu_cuda_isac/sensing/models/cusense_cnn.py` |
| The training script | `scripts/train_cusense.py` |
| The evaluator side: plans, positions, samples, oracle, parity, evaluation, reference record | `src/ocudu_cuda_isac/episodes/cusense_frontend.py` |
| Unit gate, driver, its tests and its double | `tests/unit/test_cusense_frontend.py`, `tests/integration/cusense_frontend.py`, `tests/unit/test_cusense_frontend_driver.py`, `tests/unit/cusense_frontend_double.py` |
| Configuration and gate | `configs/integration/cusense_frontend.json`, `scripts/accept_s3_5.sh`, `make accept-s3.5`, `tests/test_accept_s3_5.py` |

No native source changes. The adapter is the one with the O3 route that Sprint 5.2 accepted, and the gate requires its bytes. Every entry of this sprint carries `runtime: "reference"`: NumPy on the CPU for the front end and PyTorch on the selected device for the model. Sprint 3.6 runs the same stages as CUDA kernels and a TensorRT engine and checks them against the vectors and the held-out scores of this page.

## The three stages

A record of the stage O3 holds the estimate of one uplink slot: `[dmrs_symbol][prb][rx_port][12]` as complex FP16, per DM-RS symbol, after frequency interpolation and before time interpolation. The front end reads it as `H[a, k, s]`: array element `a` in rising order of the antenna map, subcarrier `k` on the support of the specification, DM-RS symbol `s`. At the demonstration profile that is 4 by 612 by 2. An element is valid when its validity bit is set, its value is finite and its magnitude lies above `tau` = 1e-10, the rule that leaves zero padding out of every average.

| Stage | Arithmetic | Baked example |
|---|---|---|
| 1, background | `H_B[a,k,s]` = the complex mean of the target-absent calibration records over their valid samples, with the count of those samples | `1+1j`, `3-1j` and one invalid zero give `2+0j` from two samples |
| 2, window | the uplink slots of one TDD period, each placed at the subcarrier offset of its allocation, averaged in the complex domain over the valid samples of each element | `1+0j` and `0+1j` give `0.5+0.5j`, magnitude 0.7071; a slot at resource block 2 lands on subcarrier 24 |
| 2, clutter | the magnitude of the window minus the magnitude of the template, where both hold a sample | a template magnitude of 0.5 leaves 0.2071 |
| 2, symbols | the mean over the DM-RS symbols that hold a sample | |
| 2, z-score | `(d - mean) / (scale + 1e-8)` with one mean and one deviation, as float32; an element without a sample is 0 | mean 0.2 and deviation 0.1 give 0.071 |
| 3, model | `X[1, A, K]` to a softmax map over the position grid and a presence logit | the map is `[H, W]` and sums to one for K = 612 and K = 306 |

The subtraction is of magnitudes, as published: a template of another phase leaves the same value. Nothing is interpolated: an element without a valid sample is left out of every average and is 0 at the input of the model.

## Input entries

| Entry | Stage | Layout | Support at the demonstration profile |
|---|---|---|---|
| `o3_fp16` | O3, the receiver's own estimate | `ocudu_o3_fp16`, FP16 widened once, no scale applied | 612 subcarriers, step 1 |
| `o2_comb` | O2, pilot least squares of the observer | `ocudu_o2_comb_c64` | 306 pilots, step 2 |

Both run the same stream, the same clutter step and the same model entry: the model pools over frequency, so it takes any width from 16 on. A specification `cusense-spec/v1` names the entry, the slots of the window (7, 8 and 9 of a period of ten at 30 kHz), the support, the steps and `tau`. A record of another stage, another shape, another chain of producer steps, more than one layer or sample times that are not those of its slot is `rejected` before its payload is read. An O3 record must carry pilot least squares and frequency interpolation and no time interpolation; smoothing is in the chain except for an allocation of one resource block. Slots of one window that were estimated by two chains are no window.

## The window stream

`WindowStream.push(record)` takes the records in the order of the ring and answers at each record of a window slot.

| Status | Meaning |
|---|---|
| `insufficient_history` | the window of this period is still filling |
| `ok` | the last slot of the period arrived: the window is complete, with its mean as complex64 and its count of records per element |
| `gap` | a slot of the period never arrived; the window carries no tensor |
| `invalid` | fewer than `min_valid_fraction` of the elements hold a sample, or the slots disagree on their steps |
| `rejected` | the record is no input of the specification |

A window identifier is `epoch/episode key/period`. A history never spans two episodes or two epochs. A window is complete when its last record is pushed, and nothing later is waited for. A stream reads 30 named fields of a header (`INPUT_FIELDS`), the validity bits and the payload, and not the cell, the UE, the grant, the digest or the host block. A record carries no label, no split and no position. No digest is taken while a stream reads or a model decides: `seal()` and `seal_decisions()` take the hashes after the last record, as [no checksum runs in a timed path](../dev/DEVELOPMENT.md#requirements-and-where-they-are-built).

## The model entry `cusense_cnn`

The published plain-block 1D CNN: a stem `Conv1d(A, 64, 7)` with batch normalization, ReLU and a max pool of 2, three blocks of two convolutions each to 128, 256 and 512 channels at stride 2, an adaptive average pool over frequency, and a trunk of `Linear(512, 512)`, ReLU, dropout 0.3, `Linear(512, 256)`, ReLU, dropout 0.15. Two linear heads read the trunk: `location` to `H * W` grid logits and `presence` to one logit. Both start at zero, so an untrained model answers with a uniform map and a logit of 0, and the first loss of a fit has a known value. On the grid of the demonstration set, 50 by 34 cells, the model holds 2,384,933 parameters in 43 tensors, 9,554,580 bytes. It uses convolutions and plain linear layers only, and no batched product with broadcast operands.

The target of the map is the cell of the target at the last opportunity of the window, smoothed by a Gaussian of `sigma_cells` and normalized on the grid. The losses are the registry `LOSSES`:

| Head | Loss | Eligible |
|---|---|---|
| `location_grid` | KL divergence of the target against the softmax of the grid logits, in nats | target-present samples |
| `presence` | binary cross-entropy of the logit against the label, in nats | every sample |

The KL divergence of a distribution with itself is 0, and a uniform map over four cells against one cell gives ln 4 = 1.386294. A batch of target-absent samples sends no gradient into the map.

**Two changes from the published system.** The template and the statistics of the z-score are fitted on the training split only, and name the windows they were fitted on by a hash. And a presence logit is trained with the map, so that a target-absent window is scored: the published system has no detection stage.

## The artifact: `cusense-artifact/v1`

A directory of exactly four files.

| File | Holds |
|---|---|
| `model.json` | the manifest: the entries of every stage, the input contract (specification, array elements, producer steps, shape `[1, A, K]` as `<f4`), the normalizer, the model configuration with its heads and losses, the layout of the weights, the grid geometry with its units, the threshold and the optional stage |
| `template.c64` | `H_B[a, k, s]` as complex64 |
| `template.count.u4` | the number of valid samples behind each element of the template |
| `weights.f32` | every tensor of the model as little-endian float32, one after another, in the order of `weights.tensors` |

Each part has its hash in the manifest: the template, its counts and the weights by the SHA-256 of their files, the normalizer, the threshold and the grid by the SHA-256 of their canonical JSON, and the whole by `sha256` over the manifest. `open_artifact` checks all of them and refuses a fifth file, statistics that name another split than training, and a hash other than the one a freeze record names. The grid states its units: a cell is `[row, column]` with rows along y, `cell_m` is `[height along y, width along x]`, and a position is `[x, y]` in metres in the frame of the scene, the centre of a cell for an estimate. The artifact holds no sample, no label and no file name of a training run.

## Training, and what it may read

`scripts/train_cusense.py` reads two sample sets, a specification, a grid and a training configuration `cusense-train-config/v1`, and nothing else. It imports no evaluator module and opens no episode. A sample set is three files: `<name>.jsonl` with one row per window (`cusense-sample/v1`: window, episode key, split, scene, pair, label, cell, position, the hash of its specification and of its tensors), `<name>.c64` with the window means and `<name>.u1` with their counts.

1. Check the split: a held-out row is refused, and so is a scene, a pair, an episode key or a window in both sets. Write `training_manifest.json` and `validation_manifest.json`.
2. Fit the template on the target-absent training windows, weighted by their counts, and the normalizer on every training window.
3. Fit the weights on the training windows by the joint loss with Adam, from the seed of the configuration. The validation windows are measured after every epoch and select nothing: the weights are those of the last epoch.
4. Score every window with the deployed arithmetic, float32 on the CPU, one window at a time.
5. Set the threshold with `presence.calibrate_threshold`, the rule of Sprint 3.4: with a false-alarm target of 0.05 and 320 target-absent validation windows, 16 may lie above it.
6. Write the artifact, read it back, require the same scores within 1e-5 and the same inputs bit for bit, and write `fit.json` and `freeze.json`.

`freeze.json` names the artifact, its files and its five parts by hash, both manifests and the time of the freeze on the monotonic clock. The defaults are the published ones: Adam at 1e-3, batches of 256, 100 epochs, `sigma_cells` 8.

## Decisions: `cusense-decision/v1`

A sensor writes one decision per answer of the stream. A complete window gives `score` (logit and probability), `decision` (present when the logit lies above the frozen threshold), `location` (the cell of the largest value of the map, the first one in row order, the centre of that cell in metres and the value there), `input` and `location.map` with their SHA-256, the contributing records, the first and the last sample of the window, the hashes of the model, the time of the front end and of the inference, and the host times. Any other answer gives a decision under its status with no score, no map and no location.

**The optional stage** `map_mean_kalman` is an output of its own, `smoothed`, and changes no raw field: the mean of the newest `maps` maps of one history (10 by default), the cell of its largest value, and a constant-velocity Kalman filter on that position. A history ends with its episode, its epoch or a period that does not follow. With the entry `none` the field is null.

## What a sensor writes

| File | Holds |
|---|---|
| `decisions.jsonl` | one `cusense-decision/v1` line per answer of the stream |
| `inputs.f32`, `maps.f32` | the input `[A, K]` and the map `[rows, cols]` of every scored window, in order |
| `cusense.json` | the report: the manifest of the artifact, the entries with their library, device and runtime, the device facts, the counts, the source and how often the artifact was loaded |
| `windows/` | `windows.jsonl`, `windows.json` and, for a live sensor, the records it read under `records/` |

`follow` loads the artifact once, runs its front end once on noise and its model once on the template, writes the ready file and only then attaches. A launcher starts the receiver after that file. `open_output` verifies every line against its tensors: the hashes, a map that sums to one with the location at its largest value, the probability that follows from the logit and the decision that the threshold gives.

## Steps and where they run

| Stage | Entries | Library and device |
|---|---|---|
| input | `o3_fp16`, `o2_comb` | NumPy, CPU |
| ingest | `ocudu_o3_fp16`, `ocudu_o2_comb_c64` | NumPy, CPU |
| window | `coherent_mean` | NumPy, CPU |
| clutter | `background_magnitude` | NumPy, CPU |
| symbols | `dmrs_mean` | NumPy, CPU |
| normalizer | `zscore_global` | NumPy, CPU |
| model | `cusense_cnn` | PyTorch, the device selected when the model is loaded |
| detector | `frozen_threshold` | Python, CPU |
| smoothing | `none`, `map_mean_kalman` | NumPy, CPU |

Every entry carries `runtime: "reference"`. [Stage registry](../architecture/RUNTIME.md)

## The evaluator side

`src/ocudu_cuda_isac/episodes/cusense_frontend.py` holds what an evaluator adds around the core, offline. `write_plans` writes the replay of some entries of a set as a `pilot-ls-plan/v3`: both stages, each on a ring of its own, in the mode timed, with the transmit grids that the control binds. `positions` reads where the target of a realization is at every opportunity: that is truth, read after the windows are written. `samples` joins the complete windows to the split by their episode key and to the position at the last opportunity of the window. `oracle_window` and `oracle_input` compute a window and an input again, value by value, with the reader of their stage and no code of the stream. `parity` joins the decisions of a device and of the CPU by window and by the digest of the input. `evaluate` counts presence by label with its denominators, by window and by one prospectively selected window per realization, the last one, and gives the position error in metres on the scored windows of target-present realizations: of the raw map, of the mean of the maps and of the Kalman filter. `reference_record` puts the measured figures beside the published ones.

## The gate

`make accept-s3.5` runs the portable suite, the unit gate and the import rule, checks the patch series, takes the accepted Sprint 3.4 run and the accepted Sprint 5.2 run as controls and the episode set of the Sprint 3.4 run as the stimulus, rebuilds the adapter with the O3 route against the patched receiver of Sprint 5.2 and requires the bytes of that run, runs the driver, and reads the frozen artifact, a held-out decision and both records back with other tools. The driver grades twelve cases and 47 checks.

| Case | What it grades |
|---|---|
| `control` | the accepted Sprint 3.4 run, its frozen linear head, the stores of the grids its plans bind, and the adapter with the route |
| `split` | the set is the one of the control and verifies, a trajectory is in one group, every entry gives the expected windows of one period, and the grid is the expected one |
| `records` | the replay of the training and validation entries on plans of version 3: every decode original, the CUDA processor on its resident path, and the account of both rings with every record kept |
| `windows` | one complete window per period for both entries, the same slots in both, the oracle, and the sample sets |
| `train` | the script on the O3 samples: the artifact and its hashes, the template and the statistics recomputed from the training set alone, the threshold recomputed by its rule, and the fit |
| `second-input` | the same five checks on the O2 samples |
| `freeze` | the hash chain of both freeze records and no held-out identifier in a manifest, with nothing of the held-out group replayed yet |
| `live` | the held-out replay with the sensor on the O3 ring in a second process on the device: loaded once, warm before the first record, every record read, one decision per answer |
| `parity` | the same records offline on the CPU: the same windows, the same input digests, the logit, the probability and the map within the tolerance, and the decisions |
| `evaluation` | the O2 arm and the linear head of Sprint 3.4 on the records of the same held-out run, `cusense_eval.json` with every denominator, and `cusense_reference.json` |
| `causality` | no score without a window, records of the future, fields outside the input, and no label in an output |
| `refusals` | 23 by the interface and 9 by the command lines, with status 2, one line and nothing written |

**What is graded and what is recorded.** The stages, the artifact, the freeze, the presence decision at the frozen threshold and the parity are graded. One figure of quality is graded, on the validation split and declared in the configuration before the run: the area under the curve of the presence logit must reach `expect.validation_auc_min`, 0.75, so that a model that learned nothing is not frozen. The presence counts and the position error on the held-out episodes are recorded in `cusense_eval.json` and in a note of the grade, and decide nothing. `cusense_reference.json` records the position error beside the published mean of 73.3 cm with `"graded": false` and lists every difference: a simulated scene, 20 MHz against 100 MHz, a GB10 against a GH200, two DM-RS symbols against three, exact positions against camera labels, the size of the sets, the split, the two changes above, and the runtime. The batch-one latency is recorded with its runtime entry and is not compared with the published 241 us: that figure belongs to the native serving option, and Sprint 3.6 makes the comparison on the native core.

Tolerances of the parity: a logit may differ by 1e-3 plus 1e-4 of its magnitude, a probability by 2.5e-4 and a cell of the map by 1e-4. A saturated presence logit is a large float32 value, so its tolerance has a relative term; near the threshold the absolute term decides. A decision that differs at the threshold is a failure when its margin is larger than the tolerance of its logit, and is listed otherwise. The threshold recomputed by the gate must have the counts of the frozen one and a logit within 1e-5 of it. There is no host rehearsal: the stage O3 exists only where the CUDA processor ran its resident path.

## Files of a gate run

| Path under `cusense_frontend/` | Holds |
|---|---|
| `grade.txt`, `grade.json`, `comparison.json`, `manifest.json`, `recover.json` | the grade, what was measured, the run and what a second reader starts from |
| `split.json`, `grid.json`, `specs/o3.json`, `specs/o2.json`, `training-o3.json`, `training-o2.json` | the evaluator's split table, the grid of the set and what each fit was given |
| `records/<run>/` | the three plans, `native/` with `ring/` and `estimate/`, and `consumer-o2/`, `consumer-o3/` |
| `windows/<arm>/<run>/` | `windows.jsonl` and `windows.json` |
| `samples/<arm>/` | `training` and `validation` as `.jsonl`, `.c64` and `.u1` |
| `train/<arm>/` | `artifact/`, both manifests, `fit.json` and `freeze.json` |
| `live/held-out-NN/` | the plans, `native/`, `sensor/`, `sensor-ready.json` and `consumer-o2/` |
| `parity/`, `evaluation/o2/`, `evaluation/linear/` | the offline outputs |
| `cusense_eval.json`, `cusense_reference.json` | the two records of the card |

At the demonstration set a run keeps 16,800 records of each stage, about 20 kB each, and the window tensors of 5,440 windows per entry: about 1.5 GB.

## Checked on a development host

The unit gate runs 58 tests, the driver tests 14 and the gate-runner tests 10. The driver passes its twelve cases and 47 checks on the doubles of Sprints 3.2 and 5.2 with the episode set of the Sprint 3.4 driver tests, and each of 16 injected faults fails the check that names it.

A rehearsal of the fit at the scale of the demonstration set, on free-space windows of the analytic target and not on receiver records: 4,800 training, 640 validation and 160 held-out windows of 4 by 612, the published widths and the default configuration, 100 epochs on two CPU cores. The first loss equals its expected value, 1.635268. The presence logit separates every split with an area under the curve of 1.000, and the threshold lies at 16 of 320 validation negatives. The position error on the 80 held-out target-present windows is 1.90 m in the mean and 0.91 m in the median, against 3.04 m for the centre of the area. The validation loss of the map is lowest near epoch 20 and rises afterwards, while the median error stays. The free-space rehearsal of Sprint 3.4 was no prediction of the traced set, and this one is none either.

## What the Spark run has to close

- The receiver replay of 340 pairs on plans of version 3 with both rings attached, and the account of 16,320 records per stage.
- A timed plan of version 3 without stored grids (`pilot_ls --no-grids`): Sprint 5.2 stored its grids, and Sprint 3.4 ran without them on plans of version 2. Should a native output not verify for that reason, `receiver.keep_grids` in the configuration stores them, at about 4.6 GB more.
- The fit on the GB10 and its time, the reload on the CPU within 1e-5, and the area under the curve on the validation windows against the declared floor.
- The sensor in a second CUDA process on O3 records of 20 kB: whether a Python reader with a model of 2.4 million parameters keeps the pace of the receiver on a ring of eight slots. A dropped estimate fails `live account`.
- The parity of the GB10 against the CPU for a convolutional model, within the declared tolerances.
- The position error and the presence counts on the ten held-out pairs for both input entries, beside the linear head.

## Choices made here

1. **The window is one TDD period**, its three uplink slots, and its hop is one period: eight windows per realization of the demonstration set. A window that misses a slot is a `gap`, as in Sprint 3.3.
2. **The template is a mean over records, weighted by their counts**, and is stored with those counts, so that a template of windows of unequal fill is the mean the card defines.
3. **One mean and one deviation** normalize every antenna and subcarrier, as published, fitted on the training windows.
4. **The weights are those of the last epoch.** The validation split sets the threshold and is not used to select a checkpoint as well.
5. **The threshold rule is that of Sprint 3.4**, on the logits of the deployed arithmetic, so the two models are compared at operating points set the same way.
6. **Presence is counted by window and by one window per episode**, the last one, selected before the run. The second count is the one comparable with the linear head, which decides once per episode.
7. **A floor on the validation split is graded and the held-out figures are not.** The floor is declared in the configuration.
8. **The O2 arm is scored offline** on the records a native consumer kept during the held-out run of the O3 sensor, so both arms and the linear head decide on the same slots of one receiver replay.
9. **Sample tensors are stored once**, in the sample sets. A window output holds lines with hashes.
10. **The front end stays in NumPy on the CPU.** Its native form is Sprint 3.6's, and a reference that moved it to the device would measure nothing the demonstration reports.
