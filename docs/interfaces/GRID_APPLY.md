# Channel application into the receiver grid

Sprint 2.3 (`grid_apply`) is the first native component that touches the receiver's input. It applies the stored channel and noise of an episode to the transmit grid of each grant, with the grid gain of the entry, writes the result through the writer of the pinned OCUDU resource grid and reads it back. Sprint 2.4 hands that grid to the PUSCH processors. [Sprint card](../dev/DEVELOPMENT.md#phase-s2) | [grant record](GRANT.md) | [episode entry](EPISODE.md) | [fixture contract](REPLAY.md)

## Run it

```bash
# Unit gate: the Python reference with the standard library, and the host-free native core built
# on its own with CMake and a C++17 compiler. No OCUDU build, no NumPy, no GPU
python tests/unit/test_grid_apply.py -v

# Integration gate on a host that passed make accept-s1.1, make accept-s2.1 and make accept-s2.2
make accept-s2.3

# The steps of the gate, one at a time
make adapter-build
s22=$(ls -d results/runs/s22-* | tail -1)
python tests/integration/grid_apply.py --config configs/integration/grid_apply.json \
    --s22-run "$s22" --output-dir results/runs/s23-spark-001/grid_apply

# By hand: a plan from the stored pair, the native tool, and a second reader on its output
mkdir -p build/grid-apply-001/out
python -m ocudu_cuda_isac.episodes.grid_apply plan --episode "$s22/episode_pair/reference_pair/store" \
    --run-id pair-by-hand --output build/grid-apply-001/plan.json
build/ocudu-adapter/replay/grid_apply --plan build/grid-apply-001/plan.json --output-dir build/grid-apply-001/out
python -m ocudu_cuda_isac.episodes.grid_apply show build/grid-apply-001/out --sequence 0 --rx 3 --symbol 2 --subcarrier 0
sha256sum build/grid-apply-001/out/ingress/0000.c64
```

`make accept-s2.3` writes `results/runs/s23-<host>-<UTC time>/` and ends with `Sprint 2.3: ACCEPTED on this DGX Spark.` It rebuilds the adapter, uses the CPU only, links no decoder and never rebuilds OCUDU. The driver takes a few seconds.

## The rule

For one opportunity, over the whole carrier grid:

```
Y[rx, symbol, subcarrier] = g * (sum over tx of H[rx, tx, symbol, subcarrier] * X[tx, symbol, subcarrier]
                                 + W[rx, symbol, subcarrier])
```

| Operand | Source | Axes |
|---|---|---|
| `X` | `x.npy` of the store entry of the opportunity's grant | complex64 `[tx_port, symbol, subcarrier]` |
| `H` | `a_h.npy` or `b_h.npy` of the episode entry, one opportunity | complex64 `[rx_port, tx_port, symbol, subcarrier]` |
| `W` | `w.npy` of the entry, the same opportunity | complex64 `[rx_port, symbol, subcarrier]` |
| `g` | `calibration.grid_gain` of the entry | real, positive, one value for the run |
| `Y` | the receiver grid | complex bf16 `[rx_port, symbol, subcarrier]`, receive plane `i` is antenna `i` |

It is computed once, in double precision. Time-indexed H already carries the Doppler shift. X depends on the slot index, so each opportunity takes the grid of its own grant. The order of the operations is fixed, so that the native kernel, the scalar Python reference and a NumPy reference give the same bits:

1. each product as `(hr xr - hi xi, hr xi + hi xr)`, in transmit-port order;
2. the products summed in that order, then W added;
3. each part of the sum multiplied by `g`;
4. each part converted to single precision (nearest, ties to even) and handed to the grid's writer, which rounds it to the storage format.

The operands of step 1 are single-precision values. Their products are exact in double precision, so the result does not depend on whether a compiler fuses a product into the sum. The core is still compiled with `-ffp-contract=off`. The unit gate compares the native kernel with the Python reference by digest on generated fixtures with one, two and four transmit ports.

**Tolerances.** Two comparisons are declared, and both apply after the gain.

| Stage | Rule |
|---|---|
| Values handed to the grid, complex64 | `abs(error) <= 1e-6 + 1e-5 * abs(reference)` against a complex128 reference. The absolute term is a third of the return of the reference pair before the gain (2.6e-06), so it is only meaningful at unit scale. |
| Stored grid, complex bf16 | Equal to the reference rounded by `grant.bf16`, element by element. The gate also records how many values equal the reference converted to single precision bit for bit. |

## The gain

`g` is `calibration.grid_gain` of the episode entry, `1 / sqrt(reference_power)`. It is real and positive, one value for a replay run, the same for both cases of a pair and for every episode of a set, and it is applied to the signal and to the noise. A stored grid is never rescaled.

**Why.** The stored channel is physical. The grid's own format has the exponent range of a single, so the factor does not change its rounding. The pinned CUDA receiver keeps its estimates in 16-bit floats, whose normal range ends at 6.1e-05.

| Quantity of the reference pair | Without the gain | With `g = 2,942.4` |
|---|---|---|
| Direct path | 3.3986e-04 | 1.000 |
| Return | 2.6468e-06, below the normal range, about five significant bits | 7.79e-03, eleven bits |
| Noise sigma | 1.0747e-05, below the normal range | 0.0316, 30 dB below unit signal power |

**Refusals.** Both the plan check in Python and the native core refuse, before a grid is touched:

| Refused | Words of the refusal |
|---|---|
| A run without a gain | `names no grid gain` |
| A gain that is zero or negative | `must be positive` |
| A gain that is not finite | `not finite`. JSON has no infinity, so the native tool refuses such a file as invalid. |
| Two gains in one run | `two gains in one run`. Every realization carries the gain its entry recorded, and each must equal the gain of the run exactly. |
| A gain that differs from `1 / sqrt(reference_power)` by more than 1e-6 | `differs from 1 / sqrt(reference_power)` |
| A transmit grid whose hash is not the bound one | `its record binds` |
| A transmit grid bound by no hash, an opportunity without a bound grid | `no bound transmit grid`, `has no bound grid` |
| An override on a run that reads an entry | `only for a fixture that has no entry` |

An override is allowed for a fixture that has no entry, such as the unit vectors and the unity check. It names its reason, and the plan, the native report and every grid descriptor record it.

## The plan: `grid-apply-plan/v1`

`src/ocudu_cuda_isac/episodes/grid_apply.py` writes the file the native tool reads. Standard library only.

| Key | Contents |
|---|---|
| `schema`, `run_id`, `access` | `grid-apply-plan/v1`, an identifier, `offline_stimulus` |
| `carrier` | `rx_ports`, `tx_ports`, `symbols`, `subcarriers` of the carrier grid |
| `gain` | `value`, `source` (`entry` or `override`), `reference_power` of the entry or the `reason` of the override |
| `grants` | Per grant: `grant_id`, `record_sha256`, `slot`, the transmit grid `x` as `{path, sha256}` and its occupied elements `active`, a list of int32 `[n, 2]` files of `(symbol, subcarrier)` |
| `sources` | Per channel realization: an opaque `realization_id`, `h` and `w` as `{path, sha256}`, and `entry` with `episode_id`, `grid_gain` and `reference_power`, or `null` for a fixture |
| `opportunities` | In replay order: an opaque `opportunity_id`, its `realization_id`, the `index` into H and W, the `grant_id`, `sfn` and `slot` |

`entry_plan(run_id, entries, stores)` builds a plan from one or more episode entries. It verifies each entry with `open_episode`, takes the gain from its calibration and requires one value across entries. For every grant it opens the store entry with `open_store` and requires the hash of `x.npy` to be the one the entry binds, looking first at the directory the entry recorded and then under `<root>/<grant_id>/store` of each named root. The occupied elements are `pilot_index.npy` and `data_index.npy` of that store. `fixture_plan(...)` builds a plan from arrays a test wrote, with an override.

**Who binds what.** The plan builder decides which grid an entry binds, because only it reads the entry. The native tool reads the plan and no manifest: it verifies every file against the hash the plan names, and it would have to read the truth of an entry to do more. A plan names no transport block and no label, and identifiers are opaque: `rz-` and twelve hexadecimal digits for a realization, that and the opportunity index for an opportunity.

## The native core

`native/ocudu_adapter/replay/` holds the code. The core includes nothing of OCUDU.

| File | Contents |
|---|---|
| `replay_core.h`, `replay_core.cpp` | Host-free core: the kernel `apply`, the gain rules, the reader of the arrays, the owned grids, the session, the readback digest, a SHA-256 and a model of the grid's storage |
| `tests/replay_core_test.cpp` | Unit target: the core and doubles only, 14 cases, one CTest case each |
| `pinned_grid.h` | The pinned resource grid as the grid of a replay: `put` through `resource_grid_writer`, `get` through `resource_grid_reader` |
| `replay_plan.h` | Reader of the plan, with the JSON library of the locked OCUDU source |
| `grid_apply.cpp` | The tool |

| Piece of the core | What it does |
|---|---|
| `array_file` | Maps a little-endian C-ordered `.npy` file of complex64 or int32 read-only and hashes it |
| `session` | The preloaded stimulus of a run. Its constructor applies every rule: the gain, each file against its hash, the shape of each array on the carrier grid, finite values, and a transmit grid that is nonzero on exactly the occupied elements of its grant. It throws before a grid is touched. |
| `apply` | The kernel, into any `grid_sink` |
| `grid_sink` | The supplied writer and its reader: `put`, `get`, `clear`. `pinned_grid` is the upstream one, `model_grid` stores by the documented rule with no OCUDU code. |
| `o1_pool`, `ticket`, `o1_view` | Grids the replay owns, with their state and generation |
| `measure` | What storage did to one grid, on every component, against the values handed to the writer |

**Allocation.** A fill takes X and the occupied elements from the grant of its opportunity. H and W cover the whole carrier grid in the grant's coordinates, so a grant that changes inside one carrier is a remap: `session::fill(opportunity, grant_id, ...)` reads H at the elements the new grant transmits on. A grant the run does not bind, and a grid on another carrier, are refused.

**Grid states.** A grid is `free`, `filling` or `ready`. `try_acquire` takes a free grid, raises its generation and returns a ticket, or returns false at once when no grid is free. `commit` reads the grid back, records the digest in its descriptor and makes it ready. `view` returns read access to a ready grid for the consumer that decodes this grant record. `release` frees it, exactly once.

| Misuse | Result |
|---|---|
| Reading a grid that is still filling | `state_error`: is filling, not ready |
| A ticket or a view of an earlier fill | `state_error`: stale generation |
| A second release | `state_error`: is free, not ready |
| A consumer that decodes another grant record | `state_error`: changed allocation |
| A fill that fails | The grid is cleared and returned to the pool |

The pool serves one producer thread. The ownership of an observation across threads is the ring of Sprint 3.2.

## The tool and what it writes

```bash
build/ocudu-adapter/replay/grid_apply --plan FILE --output-dir DIR [--no-prestore]
build/ocudu-adapter/replay/grid_apply --self-test
```

`DIR` must exist and must not hold a result. The tool reads and checks everything first, then creates two pinned grids sized as the pinned test sizes its receive grid (receive ports by 14 symbols by 3,300 subcarriers) and walks the opportunities in plan order. Each grid is filled, made ready, read back through a view, measured and written. It stays retained while the next opportunity is filled into the other grid, is checked against its digest once more and is released. Exit status 2 means a refused plan, 1 any other error. `--self-test` runs the unit vectors through the pinned grid and is one of the adapter's CTest cases.

| Output | For | Contents |
|---|---|---|
| `ingress/ingress.jsonl` | Receiver side | One `o1-grid/v1` descriptor per opportunity |
| `ingress/NNNN.c64` | Receiver side | The stored grid as read back: little-endian complex64 `[rx_port, symbol, subcarrier]`, each value a widened bf16. Its SHA-256 is the readback digest. |
| `prestore/NNNN.c64` | Evaluator | The complex64 values handed to the writer, before storage |
| `native.json` | Evaluator | `grid-apply-native/v1`: the gain, the verified input files, per opportunity the grid, generation, digest, fill time, plane powers and storage statistics, per realization the unit power of its first opportunity |

**Descriptor `o1-grid/v1`.** `schema`, `run_id`, `sequence`, `opportunity_id`, `sfn`, `slot`, `grant_id`, `grant_record_sha256`, `active_re`, `active_sha256`, `grid_index`, `generation`, `state`, `grid` (`file`, `axes`, `shape`, `dtype`, `order`, `storage`, `sha256`) and `amplitude` (`grid_gain`, `convention`). The digest is taken over the stored values, because the grid holds complex bf16.

**Access.** `ingress/` holds the stored grids and their descriptors and nothing else. A descriptor has the fields above and no other. It names no channel, no noise, no transport block and no label, and the tool reads no transport block. `ingress_violations()` lists any other file or field and any hash or name of H, W or a transport block that appears in the descriptors. In the core, `o1_view` and `o1_descriptor` hold no route to H, W or X.

`open_output(directory)` verifies a native output with the standard library: every descriptor against the report, every stored grid against its digest.

## Evidence of the integration gate

| File in the run directory | Contents |
|---|---|
| `summary.txt` | Every gate line and the verdict |
| `adapter-build/`, `adapter-receipt.json` | Logs and receipt of the adapter build, with the CTest log of the native unit cases |
| `grid_apply/grade.json`, `grade.txt` | Case count, check count, every check with its result, the tolerances |
| `grid_apply/comparison.json` | The measured numbers of every case |
| `grid_apply/manifest.json` | Command, commit, locked source commits, input hashes, adapter identity |
| `grid_apply/unit_vectors/`, `unity/`, `allocation/` | Arrays, plan, native output and log of each fixture |
| `grid_apply/reference_pair/` | `plan.json`, `native/` with `ingress/`, `prestore/` and `native.json`, `second-reader.json` |
| `grid_apply/refusals/` | The refused plan and the native log of every rule |
| `recovered.json` | What the command-line reader recovers from the stored grids |

The five cases and their checks:

| Case | Checks |
|---|---|
| `unit-vectors` | The two-transmit vector of the Sprint 0.3 fixture gives `1+3j, 2-4j, 6.5, 14` with `g = 1` and exactly twice them with `g = 2`. A zero channel gives `g W`. |
| `unity` | The run records its override. One check per archived grant grid: with a unity channel on four receive planes, zero noise and `g = 1` every plane returns `x.npy` bit for bit. |
| `reference-pair` | Plan, native run, values before storage, stored grids, gain, difference of the two cases, noise after storage, receive planes, grant of each opportunity, access, second reader |
| `allocation` | The grant changes on a channel that holds its own coordinates. A grid on another carrier is refused. |
| `refusals` | The gain rules, the binding rules, the bindings of the entry |

The gate prints two things as NOTE lines and does not grade them: whether the stored grids have the digests of the development-host rehearsal, and the fill time of one grid.

## Accepted run

`make accept-s2.3` passed on the DGX Spark `spark-ecf8` on 2026-11-21 (UTC) at commit `d027aad`, run `s23-spark-ecf8-20261121T032401Z`. All 35 gate lines passed, none failed and none was skipped. The stimulus came from the accepted Sprint 2.2 run `s22-spark-ecf8-20261121T004433Z`, and the slot 7 grant from the Sprint 2.1 run it names.

| | Accepted run |
|---|---|
| Unit suite and unit gate | 274 tests and 29 tests, none skipped |
| Adapter | `grid_apply` `158644b5a177` and `grant_fixture` `7a0e1c8ec1a0`, the executable Sprints 2.1 and 2.2 accepted, compiled with the flags of the pinned test and `ENABLE_CUDA`, no compiler warning, 19 CTest cases |
| Grade | 5 of 5 cases, 25 of 25 checks |
| Values before storage | 1,645,056 of 1,645,056 elements of 48 opportunities equal the complex128 reference converted to single precision bit for bit. Largest error 8.1e-08. |
| Stored grids | 1,645,056 of 1,645,056 elements, the 1,527,552 occupied ones included, equal the reference rounded to the storage format. 48 of 48 digests. |
| Storage rule | The pinned writer followed the documented rule on all 3,290,112 components. It moved a component by at most 0.00389 of its magnitude (bound `2**-8`). |
| Unity | Six grant grids, four planes each, returned bit for bit |
| Gain | `g = 2,942.4244`. The mean power of `g H` of the target-absent case on opportunity 0 is `1 + 7.1e-08`, in Python and in the native tool. `g sigma = 0.031623`. |
| Difference of the two cases | Equal to `g (H_b - H_a) X` within the rounding bound on all 1,645,056 components, largest error 0.99 of the bound. The return has an RMS of 8.06e-03, the rounding of the two grids 2.35e-03. |
| Noise after storage | -29.989 dB relative to unit signal power, against -30.000 dB. The rounding lies 55.8 dB below the stored values. |
| Receive planes, occupied elements | 1.0778, 1.0777, 1.0777, 1.0774 with the target absent |
| Grant of each opportunity | A reference with the grid of slot 7 everywhere differs on 455, 461, 455, 460 and 442 pilots of slots 8, 9, 17, 18 and 19 and on no other element, the counts Sprint 2.2 recorded |
| Fill time of one grid | 1,436 us mean and 1,445 us at most on one core for 4 by 14 by 612 elements |
| Digest over the 48 stored grids, first grid | `62f9e9d2dbe09615`, `6a0a3c28258f7c04` |

- **Spark build.** This was the open risk, because the gate had been rehearsed off the Spark only. The replay code compiled on AArch64 with `-mcpu=neoverse-v2` and linked against the 50 archives and the CUDA runtime of the accepted build. The adapter receipt binds its 11 source files by hash. The reference check of the gate ran against the real build.
- **Portability.** The gate had run first on an x86_64 development host (Ubuntu 24.04, GCC 13.3.0, Python 3.12.3, NumPy 2.5.5) against a CPU-only build of the pinned OCUDU source, on a stimulus with the hashes of the accepted Sprint 2.2 run. The 48 stored grids and the 48 grids before storage have the same bytes on both hosts. Every number of `comparison.json` is the same except the fill time, 1,163 us there. The gate prints the digest comparison as a NOTE.
- **Second reader.** A separate process verified the 48 stored grids against their digests. At opportunity 0, symbol 2, subcarrier 0 it recovered `+0.750000-1.179688j` on port 3 and `+0.746094-1.171875j` on port 0, the reference rounded to the grid. `sha256sum` gave the digest of `ingress/0000.c64`.
- **Faults.** Every fault that was injected failed a check. All 35 mutations of the native core fail a unit case. With the native unit cases bypassed, eleven of twelve mutated tools fail the integration driver. The twelfth, a grid that is not cleared before a fill, cannot be seen on a fresh grid and fails its unit case. Against the double of the tool, each fault of `tests/unit/grid_apply_double.py` fails the check that names it. The mutation runs were made on the development host.

**A finding for Sprint 4.1.** One grid takes 1.44 ms to fill on one core of the Spark, and a slot lasts 0.5 ms. The channel application is stimulus and no claim is made about its speed, but a paced run must prepare each grid ahead of its slot. The owned grids and their ready state are built for that.

The evidence is on the Spark under `results/runs/`, which Git ignores.

## For Sprint 2.4

- Link `replay_core` and include `pinned_grid.h` and `replay_plan.h`. `session::deliver(opportunity, pool, ticket)` fills a pinned grid and makes it ready. `pool.view(ticket, grant_record_sha256)` is the read access of the consumer, and `pinned_grid::ocudu_grid().get_reader()` is the reader the PUSCH processors take.
- The PDU is the grant's, with the DM-RS referenced to CRB 0. The descriptor carries the grant, the frame and the slot of the opportunity.
- Keep the readback digest with the decode result. It is taken over the stored values and does not depend on the observer.
- The transport block per opportunity is built there, where the decoded bytes are compared.
