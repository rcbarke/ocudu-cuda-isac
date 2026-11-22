# Channel episodes on the grant's coordinates

Sprint 2.2 (`episode_pair`) generates the channel that the receiver sees: a matched pair of the same scene without and with a moving target, evaluated on the subcarriers, symbol times and antenna order of an uplink grant schedule. It also exports the demonstration episode set from the same module. Every axis is read from the grant record that Sprint 2.1 accepted. [Sprint card](../dev/DEVELOPMENT.md#phase-s2) | [grant record](GRANT.md) | [diagnostic exporter](RT_EXPORT.md)

## Run it

```bash
# Unit gate: standard library only. No Sionna, no NumPy, no OCUDU build, no GPU
python tests/unit/test_episode_pair.py -v

# Integration gate on a host that passed make accept-s0 (cpu), make accept-s1.1 and make accept-s2.1
make accept-s2.2

# The steps of the gate, one at a time
python scripts/export_rt_episode.py --output-dir build/rt-pair-001
python scripts/export_rt_episode.py --pair configs/episodes/reference_pair.json --output-dir build/reference-pair-001
python scripts/export_rt_episode.py --set configs/episodes/smoke_set.json --output-dir build/smoke-set-001
s21=$(ls -d results/runs/s21-* | tail -1)
python tests/integration/episode_pair.py --config configs/integration/episode_pair.json \
    --grant-store "$s21/grant_fixture" --output-dir results/runs/s22-spark-001

# Read the stored pair of the newest gate run back, with the standard library only
run=$(ls -d results/runs/s22-* | tail -1)
python -m ocudu_cuda_isac.episodes.episode show "$run/episode_pair/reference_pair/store" --case b --opportunity 0 --rx 3
sha256sum "$run/episode_pair/reference_pair/store/b_h.npy"

# The proposed demonstration set (350 pairs, about 7 GB), when its configuration is frozen
make episode-set
```

`make accept-s2.2` writes `results/runs/s22-<host>-<UTC time>/` and ends with `Sprint 2.2: ACCEPTED on this DGX Spark.` It uses the CPU only (Mitsuba variant `llvm_ad_mono_polarized`) and never rebuilds OCUDU. On a development host the driver takes about half a minute. The Spark accepted it on 2026-11-21 (UTC) in run `s22-spark-ecf8-20261121T004433Z`. [Accepted run](#accepted-run)

## The uplink grant fixture: `pusch-grant-schedule/v1`

`src/ocudu_cuda_isac/episodes/schedule.py` places PUSCH opportunities on the uplink slots of a TDD pattern and gives each one a legal grant. Standard library only.

| Key | Contents |
|---|---|
| `schema`, `schedule_id` | `pusch-grant-schedule/v1` and an identifier |
| `tdd` | `pattern`, one letter per slot (`DDDDDDSUUU`), and `ul_slots`, the uplink slots of the pattern that carry PUSCH (`[7, 8, 9]`) |
| `start_sfn`, `periods` | Frame of the first period and the number of periods. The first period starts at slot 0 of that frame. |
| `payload` | `fixed`: every opportunity carries the transport block of the base grant |

Every opportunity has an index, its period, frame and slot, an absolute slot count, its slot start in T_c from the start of the schedule and the grant it executes. Symbol `l` lies at `slot_start_tc + Grant.symbol_times()[l]["useful_start_tc"]`, and subcarrier `k` at `Grant.subcarrier_offset_hz(k)`. Nothing is derived again.

**One grant per slot index.** The DM-RS sequence depends on the slot index in the frame and not on the frame number. The data symbols depend on neither. The schedule therefore keeps one grant record per slot index. For the base grant's own slot it is the base grant itself, so the transmit grid that Sprint 2.1 archived applies unchanged. Every other slot index gets the base record with that slot and the identifier `<grant_id>.sNN`. The demonstration grant in slot 7 with `DDDDDDSUUU` gives six grants: slot 7 and `.s08`, `.s09`, `.s17`, `.s18`, `.s19`.

The integration gate realizes the five new grants with the fixture adapter that Sprint 2.1 accepted. For each one the adapter must echo the grant's PDU, the occupied resource elements and the pilots must equal the record, the data symbols must equal those of slot 7 bit for bit, the pilots must differ from slot 7, and the host PUSCH processor must return the original bytes. Each grid is archived as a `pusch-grant-store/v1` entry under `episode_pair/grant_fixture/<grant_id>/store`, and the pair's manifest names the SHA-256 of the transmit grid of every grant.

At 30 kHz the shipped grants give these times, all from the record:

| Quantity | Value |
|---|---|
| Slot | 983,040 T_c, 0.5 ms |
| DM-RS symbols 2 and 11 | 631,296 T_c apart, 321.09 us |
| Symbol 11 to symbol 2 of the next slot | 351,744 T_c, 178.91 us |
| Symbol 2 of one period to symbol 2 of the next | 5 ms |
| Subcarriers | -9.18 MHz to +9.15 MHz, 612 on a 30 kHz grid |

## The entry: `channel-episode/v1`

`export(directory, ...)` in `src/ocudu_cuda_isac/episodes/episode.py` writes one directory. It appears only when complete and is never overwritten.

| File | Contents |
|---|---|
| `a_h.npy` | Target absent. complex64 `H[opportunity, rx_port, tx_port, symbol, subcarrier]` |
| `b_h.npy` | Target present. Same axes |
| `w.npy` | complex64 `W[opportunity, rx_port, symbol, subcarrier]`, one noise realization for both cases |
| `manifest.json` | Everything below, and the SHA-256, size, axes and shape of each file |

The arrays are little-endian NumPy `.npy` files in C order over the whole carrier grid: 14 symbols by `12 * grid_size_rb` subcarriers, in the carrier coordinates of the grant. At the demonstration profile one opportunity holds 4 by 1 by 14 by 612 values, 274,176 bytes. Receive plane `i` is antenna `i`. The singleton transmit axis is kept.

**Units and application.** `H` is the physical channel: linear complex amplitude gain with the absolute delay and amplitude of every path. `W` is in the units of `H` times unit-power data. The receiver observes

```
Y[rx, symbol, subcarrier] = sum over tx of H[rx, tx, symbol, subcarrier] * X[tx, symbol, subcarrier] + W[rx, symbol, subcarrier]
```

on the resource elements of the opportunity's grant, with `X` from that grant's store entry.

**Manifest.**

| Key | Contents |
|---|---|
| `grant`, `schedule` | The base grant record and the schedule record. The loader rebuilds both. |
| `opportunities`, `grants` | Slot, frame and start time of each opportunity. Per grant: slot, record hash, symbol times, pilot symbols and the bound transmit grid (`x`). |
| `axes`, `mask` | Axis names, shapes, the frequency rule, the time rule, units and the application rule |
| `geometry`, `array` | UE and gNB positions, gNB yaw, the offset of every receive element in world coordinates and in wavelengths, pattern, polarization |
| `scene`, `target` | Scene kind, room size and materials with the SHA-256 of the generated scene files. Target model and motion. |
| `solver`, `software` | Solver flags, seed, Mitsuba variant, package versions, machine |
| `time_evolution` | How path phases were evaluated, the measured extrapolation error and the trace cost |
| `noise`, `calibration` | SNR, sigma, seed and generator. Reference power and the grid gain. |
| `cases` | `a` and `b`, each with an opaque `realization_id` and what the generator measured. Case `b` carries the truth per opportunity. |
| `analytic_model` | Present when the scene has an analytic answer |
| `labels` | In a set: trajectory, group and grid cells |

The truth of an opportunity is the target position and velocity at its slot start, the delay, excess delay and Doppler shift of the first-order return from geometry, and the scattered-to-direct power.

**Reading.** `open_episode(directory)` verifies an entry with the standard library and reads single coefficients from the files, so an entry of any size opens quickly and no NumPy is needed on a second host. It refuses a file whose size or hash differs, a header that is not a C-ordered complex64 array of the shape the grant schedule gives, opportunities, grants or axes that differ from what the embedded grant yields, a nonfinite value, a channel without a path and a present case equal to the absent one. `numpy.load` reads the same files.

## How the channel is generated

`src/ocudu_cuda_isac/episodes/rt_channel.py` needs the full environment and runs offline.

| Step | Rule |
|---|---|
| Scenes | Two scenes with the same endpoints. Only one holds the target, so the background of the target-absent case is computed on its own and the target takes part in every visibility test of the other. |
| Solvers | `PathSolver` for the background, `RCSSolver` for the paths scattered by the target, `Paths.concat` to join them. Deterministic mode, fixed seed, synthetic array. |
| Frequency | The 612 offsets of `Grant.subcarrier_offset_hz`, around the carrier of the grant |
| Time | The geometry is traced again at the slot start of every opportunity. Inside the slot each path phase advances by its Doppler shift, evaluated at the useful start of every symbol. |
| Target absent | One geometry. Nothing moves, so the channel is the same at every opportunity. |

`Paths.cfr` evaluates a uniform time grid from the snapshot, and the symbols of a grant are not on one. The module therefore takes each path's coefficient, delay and Doppler shift from the concatenated `Paths` object and evaluates

```
H[rx, t, k] = sum over paths of a[rx] * exp(-j 2 pi (f_c + f_k) tau) * exp(j 2 pi nu (t - t_snapshot))
```

in double precision at the real symbol times. Two measurements justify it and are archived with every entry. Every snapshot is compared with `Paths.cfr` at the snapshot instant. Once per case the time evolution is compared with `Paths.cfr` on a uniform grid whose step divides the DM-RS symbol times: 1,536 T_c, 507 steps for symbols 2 and 11.

| Measured on a development host (x86_64, LLVM variant) | Reference pair | Room with the human target, five episodes |
|---|---|---|
| Difference to `Paths.cfr` at the snapshot, relative to the RMS of H | 3.2e-05 | 5.7e-05 to 7.2e-05 |
| Difference to `Paths.cfr` on its uniform grid through the DM-RS times | 3.2e-05 | 5.6e-05 to 6.8e-05 |
| Target paths extrapolated 0.5 ms ahead against the paths traced there | 7.6e-04 | 1.3e-04 to 7.8e-04 |
| The same 4.0 ms ahead | 1.8e-03 | 9.0e-04 to 6.3e-03 |
| Paths | 1 background, 1 scattered | 19 to 25 background, 13 scattered |
| Trace time per snapshot | 16 ms | about 0.1 s after the first |

The error at 0.5 ms is the single-precision resolution of the traced geometry, under a milliradian on the target's paths and about 60 dB under the return. It grows with the time ahead, most for a person walking close to the array. The geometry is therefore traced again at every slot and never carried over a period. No online update of paths is enabled.

**Scenes.** `free_space` is the analytic control. `open_room` is a procedural rectangular room with a concrete floor, a ceiling board and plasterboard walls (ITU-R P.2040 materials), written as a Mitsuba scene and hashed. **Targets.** `constant_rcs_cuboid` is `ConstantRCSSensingTarget`, the analytic control. `tr38901_human` is `TR38901SensingTarget(object_type="human")` with its random components off, facing its walking direction.

## The reference pair and its analytic check

`configs/episodes/reference_pair.json` is the grant-aligned demonstration trajectory of the card: the 20 m link of the diagnostic, UE at (0, 0, 1.5) m and gNB at (20, 0, 3) m, with a 1 m2 reflector at (10, 25, 1.5) m moving outward at 2 m/s, on the demonstration grant over eight TDD periods with all three uplink slots: 24 opportunities, 40 ms.

The analytic model `free-space-point-target/v1` is in `episode.py` and needs the standard library only. The direct path has the Friis amplitude `lambda / (4 pi d)` and the delay `d / c`. The return has the bistatic radar amplitude `lambda sqrt(sigma) / ((4 pi)^1.5 d1 d2)` and the delay `(d1 + d2) / c`. Element `n` of the receive array adds `2 pi / lambda` times the projection of its offset on the direction of arrival. The model is evaluated with the exact geometry of each symbol time, so it shares no code and no approximation with the generator.

| Quantity | Model | Traced, development host |
|---|---|---|
| Direct path | 3.3986e-04, 66.900 ns | within 1.7e-05 at all 822,528 coordinates |
| Return | 2.648e-06, 179.77 ns at t = 0, 112.87 ns after the direct path | within 9.3e-04 of its amplitude at all coordinates |
| Doppler shift | -43.33 Hz, negative for the receding target | phase steps below |
| Scattered to direct power per resource element | -42.17 dB | -42.19 dB |

A path whose delay changes at the rate `-nu / f_c` advances by `2 pi nu (1 + f / f_c) dt` at the subcarrier offset `f`. The gate checks the return at two unequal intervals and across one period: -0.0874 rad in 321.09 us, -0.0487 rad in 178.91 us and -1.361 rad in 5 ms at the carrier, with a largest error of 4.6e-04 rad over every port and pilot on the development host. The band edges differ from the carrier value by 0.26 percent, 3.6 mrad over 5 ms. A check that assumes one Doppler phase for the whole band fails at that level.

**On the pilot mask.** The gate forms the pilot LS of `Y = H X + W` on the first DM-RS symbol of the first uplink slot of each period, removes the static part and transforms: 306 pilots over 18.36 MHz and 8 slots 5 ms apart.

| Quantity | Value |
|---|---|
| Delay resolution and unambiguous delay | 54.5 ns and 16.67 us |
| Doppler resolution and unambiguous Doppler | 25.0 Hz and -100 to +100 Hz |
| Peak, development host | 163.4 ns and -50.0 Hz, within one bin of 179.8 ns and -43.3 Hz |
| Peak above the median of the map | 23.0 dB at 30 dB SNR per resource element |
| Sidelobes with rectangular windows | -14.9 dB in delay, -15.5 dB in Doppler |

The return is 113 ns behind the direct path, two delay bins, and is found by its Doppler shift after static removal. The two DM-RS symbols of a slot and the three uplink slots of a period are not on the 5 ms grid. S3.3 chooses between the uniform subset used here and the actual nonuniform times.

## Noise, gain and the receiver grid

`W` is circular complex Gaussian with `E|W|^2 = sigma^2 = reference_power / 10^(snr_db / 10)` on every resource element and receive port. `reference_power` is the mean `|H|^2` of the target-absent case over the receive ports and the occupied resource elements of opportunity 0. Both cases share the file, so a difference between their receiver outputs comes from the target. The generator is `numpy.random.Generator(PCG64(seed)).standard_normal`, real part first. The file and its hash are what a consumer binds to.

`calibration.grid_gain` is `1 / sqrt(reference_power)`: a real factor on `Y` that gives the receiver grid unit mean signal power. It is recorded and not applied. The physical amplitudes are small, 3.4e-04 for the direct path of the reference pair. The receiver grid stores a floating format, so the factor does not change its rounding. The pinned CUDA receiver keeps its estimates in 16-bit floats, whose normal range ends at 6.1e-05. Applying the gain is a requirement of S2.3 (decided 2026-11-20): `Y = g (sum over tx of H * X + W)` with `g = calibration.grid_gain`, 2,942.4 for the reference pair. Without it the return (2.6e-06) and the noise fall below the normal range and the return keeps about five significant bits. With it the return is 7.8e-03, the noise has sigma 0.0316 and both keep ten or more.

| Level relative to the direct path per resource element | Reference pair |
|---|---|
| Noise | -30 dB |
| Return | -42.2 dB |
| RMS rounding error of the receiver grid (Sprint 2.1) | -55.6 dB |
| Bound of that rounding, `2**-8` | -48.2 dB |

The return is 13.4 dB above the RMS rounding error and 6.0 dB above its bound, and the noise covers both. The manifest records the distribution of the scattered-to-direct power under `cases[1].measured.scattered_to_direct_db` and the per-opportunity mean in the truth.

**Precision of the files.** A complex64 component keeps 24 significant bits. An echo recovered as `b - a` has an error set by the direct path, `c64_bound(|direct|, |direct|)`, not by its own size. At 80 dB below the direct path it still keeps two significant digits. The absolute term of the grid tolerance (`1e-6`) is far above such an echo and must not be used on these files.

## The episode set: `channel-episode-set/v1`

`src/ocudu_cuda_isac/episodes/episode_set.py` reads an `episode-set-config/v1` file. A set is a directory with `manifest.json` and one episode entry per pair under `episodes/<episode_id>/`.

| Rule | How it is kept |
|---|---|
| Whole trajectories go to training, validation and held-out groups before any window is cut | The group is a field of the trajectory in the configuration. An episode has the group of its trajectory and nothing else decides it. The manifest stores the hash of the assignment, and `open_set` refuses an episode outside the group of its trajectory, a trajectory in two groups and a changed assignment. |
| Equal numbers of target-present and target-absent runs | Every episode is a matched pair |
| Calibration episodes for the background template | Case `a` of the episodes of the training group |
| A labeled position grid | The grid is in the configuration and in every episode. Each opportunity of a present case carries its cell. |
| Opaque identifiers | `ep-` and twelve hexadecimal digits of a hash. The identifier names neither the group nor the trajectory. Each case has its own `realization_id`. |

A trajectory is a walk at constant speed along a polyline inside the area of interest: `line`, `lawnmower`, `spiral` or `random` waypoints from a seed. It is cut into episodes: episode `i` starts at walk time `start_s + i * stride_s` and lasts `periods` TDD periods. The noise seed of an episode follows from the set's seed and the episode identifier.

Two configurations ship.

| | `smoke_set.json` | `demo_set.json` (status `proposed`) |
|---|---|---|
| Purpose | The set case of the integration gate | The demonstration set for S3.3 to S3.5 and S4 |
| Scene | Room of 8.0 by 13.0 by 3.2 m. UE at (5.2, 12.3, 1.0) m, gNB at (4.0, 0.6, 2.0) m facing the room. Reflections to depth 2 on both solvers. | The same |
| Area of interest and grid | 6.78 by 10.06 m, the size of the cuSense target area. 50 by 34 cells of 0.201 by 0.199 m | The same |
| Target | `tr38901_human`, 1.2 m/s | The same |
| Trajectories | 3: one per group | 9: five training, two validation, two held-out |
| Episodes | 5 pairs of 2 periods, 30 opportunities, 24 MB | 350 pairs of 8 periods: 300 training, 40 validation, 10 held-out. 8,400 opportunities, about 7 GB |
| Training coverage of the grid | 3 cells | 318 of 1,700 cells |

The ten held-out pairs give the ten present and ten absent episodes of the demonstration profile. `demo_set.json` was judged sufficient as the initial configuration on 2026-11-20 and will be tuned once the remaining components are built. It keeps the status `proposed` until then. Set `status` to `frozen` before a result is reported on it. A change after that is a new `set_id`.

## Evidence of the integration gate

| File in the run directory | Contents |
|---|---|
| `summary.txt` | Every gate line and the verdict |
| `adapter-build/`, `adapter-receipt.json` | Logs and receipt of the fixture adapter build |
| `episode_pair/grade.json`, `grade.txt` | Case count, check count, every check with its result, the tolerances |
| `episode_pair/comparison.json` | The measured numbers of every case |
| `episode_pair/manifest.json` | Command, commit, locked source commits, input hashes, adapter identity, Mitsuba variant |
| `episode_pair/diagnostic/` | The diagnostic pair and the exporter log |
| `episode_pair/grant_fixture/<grant_id>/` | `request/`, `native/` and `store/` of every grant the schedule adds |
| `episode_pair/reference_pair/store/` | The stored pair. `second-reader.json` beside it. |
| `episode_pair/episode_set/store/` | The smoke set |
| `recovered.json` | What the command-line reader recovers from the stored pair |

The gate prints two comparisons as NOTE lines and does not grade them: whether `x.npy` of the base grant has the hash of the accepted Sprint 2.1 run, and whether the three arrays of the pair have the hashes of the development-host rehearsal. A host whose single-precision trace differs in the last bit can still pass every graded check.

The four cases and their checks:

| Case | Checks |
|---|---|
| `diagnostic` | The exporter runs and its files verify. The direct path equals free space. The return has the delay and Doppler shift of the geometry, 5.31 ns and -8.70 Hz: an axis test only. |
| `grant-fixture` | The schedule's opportunities, grants and times. The base grant equals the archived Sprint 2.1 record. One check per added slot grant. |
| `reference-pair` | Export, stored pair, solver agreement, direct path, scattered path, phase at unequal intervals, delay-Doppler on the pilot mask, power margins, time evolution, refusals, second reader |
| `episode-set` | Export and reading, groups, position labels, direct path and array geometry, scattering |

## Accepted run

`make accept-s2.2` passed on the DGX Spark `spark-ecf8` on 2026-11-21 (UTC) at commit `e1db1df`, run `s22-spark-ecf8-20261121T004433Z`. All 35 gate lines passed, none failed and none was skipped. The integration grade is 4 of 4 cases and 26 of 26 checks. The grant record came from the accepted Sprint 2.1 run `s21-spark-ecf8-20261120T223232Z`.

| | Accepted run |
|---|---|
| Unit suite and unit gate | 223 tests and 23 tests, none skipped |
| Fixture adapter | `grant_fixture` `7a0e1c8ec1a0`, the executable Sprint 2.1 accepted, built again with no compiler warning |
| Base grant | `demo-51prb-4rx`, `x.npy` `a18b72ae9b9ea84f`, equal to the archived record |
| Added slot grants | `.s08` `4512e28d31e8229b`, `.s09` `707dafb060b08d95`, `.s17` `63b18dfd87bac54f`, `.s18` `95779c5e02065d28`, `.s19` `f58c031d21a3f4c0`. Each: 7,956 occupied elements, 612 pilots equal to the record, 7,344 data symbols equal to slot 7, 442 to 461 pilots different from slot 7, host receiver CRC pass with the original 1,217 bytes |
| Pair | `a_h.npy` `8b7ef3971f4c7975`, `b_h.npy` `744702180dddaea1`, `w.npy` `2b5b46bb1051c23f` |
| Direct path and return | Within 1.7e-05 and within 9.3e-04 of the analytic model at all 822,528 coordinates |
| `Paths.cfr` agreement | 3.2e-05 for the pair, 7.5e-05 for the room set |
| Phase steps of the return | -0.08741, -0.04853 and -1.36098 rad, largest error 4.6e-04 rad |
| Pilot mask | Peak at 163.4 ns and -50.0 Hz, 23.0 dB above the median |
| Trace cost | 0.71 s for 24 snapshots, 29.7 ms per snapshot |
| Set | 5 pairs, 30 opportunities, 3 training, 1 validation, 1 held-out, assignment `44e4fde62fa16374` |

- **Receiver check.** The pinned transmit chain built the grid of each added slot grant and the pinned host receiver decoded it. The five grids agree, on the first 16 digits of their SHA-256, with grids computed on a development host as the Sionna NR grid rounded to the storage format: the rounding rule of Sprint 2.1 holds on 36,720 more data symbols and 3,060 more pilots.
- **Portability.** The three arrays of the pair have the same SHA-256 as on an x86_64 host with the same pinned packages (Python 3.12.3, Sionna RT 2.2.0, Mitsuba 3.9.1, Dr.Jit 1.5.0, NumPy 2.5.5, LLVM variant). The gate prints the comparison as a NOTE. The room set was not compared byte for byte, and its `Paths.cfr` agreement was 7.2e-05 on that host against 7.5e-05 on the Spark.
- **Second reader.** `python -m ocudu_cuda_isac.episodes.episode show` on the stored pair returned H at opportunity 0, port 3, symbol 2, subcarrier 0 as -3.283090e-04+7.725997e-05j, 1.7e-05 from the analytic value, and the phase steps -0.08741 and -0.04815 rad. `sha256sum b_h.npy` gave the hash of the manifest.
- **Development host.** The driver had run there first with a stand-in for the fixture adapter, because that host has no OCUDU build. The tables of this document that name a development host are from that run. The measured values of the pair are the same on the Spark. The trace is about twice as fast there per snapshot.

Decided on 2026-11-20: S2.3 applies the grid gain as a requirement, `demo_set.json` is sufficient as the initial configuration and is tuned once the remaining components are built, and a transport block per opportunity is assigned to S2.4. Still open: the target-absent channel is stored once per opportunity although it does not change, and the target-absent runs of a set differ by their noise only.

**Replaying a set (Sprint 3.4).** An entry of a set binds no transmit grid: only the reference pair of the gate does. A set has the grant record and the schedule of the pair, so the plan of a set replay takes the grids that the pair recorded, grant by grant, and requires the same grant record. An episode has the group of its trajectory, and that assignment is the split of the presence head: training, validation and held-out windows never share a trajectory. An entry of the demonstration set lasts eight periods, one window of the delay-Doppler branch, so each realization gives one window. [Presence head](PRESENCE_HEAD.md)

The evidence is on the Spark under `results/runs/`, which Git ignores.
