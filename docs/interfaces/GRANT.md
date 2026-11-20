# Grant record and transmit reference

Sprint 2.1 (`grant_fixture`) fixes one legal uplink grant as a record that every later ticket reads: the original transport block, the identities, the allocation, the DM-RS and the transmit grid. `src/ocudu_cuda_isac/episodes/grant.py` validates the record and derives its indices with the standard library only. The transmit grid comes from two independent references, the pinned OCUDU transmit chain and Sionna NR, which must agree before the grid is archived. [Sprint card](../dev/DEVELOPMENT.md#phase-s2) | [OCUDU source map](../integration/OCUDU.md)

## Run it

```bash
# Unit gate: no OCUDU build, no Sionna, no NumPy needed
python tests/unit/test_grant_fixture.py -v

# Integration gate on a host that passed make accept-s0 (cpu) and make accept-s1.1
make accept-s2.1

# The steps of the gate, one at a time
python scripts/native_adapter.py build
python tests/integration/grant_fixture.py --config configs/integration/grant_fixture.json \
    --output-dir results/runs/s21-spark-001

# Read an archived record of the newest gate run back
run=$(ls -d results/runs/s21-* | tail -1)
python -m ocudu_cuda_isac.episodes.grant show "$run/grant_fixture/two-prb-offset/store" --pilot 0 --data 0
sha256sum "$run/grant_fixture/two-prb-offset/store/tb.bin"
```

`make accept-s2.1` writes `results/runs/s21-<host>-<UTC time>/` and ends with `Sprint 2.1: ACCEPTED on this DGX Spark.` It uses the CPU only and never rebuilds OCUDU. The Spark accepted it on 2026-11-20 in run `s21-spark-ecf8-20261120T223232Z`. [Accepted run](#accepted-run)

## The record: `pusch-grant/v1`

Three grants ship in `configs/grants/`.

| File | Purpose | Allocation | MCS | Transport block |
|---|---|---|---|---|
| `two_prb_offset.json` | Unit fixture | Two blocks at PRB 5 of a bandwidth part that starts at CRB 10 in a 51-block carrier: CRB 15 and 16, subcarriers 180 to 203 | 16QAM, table `qam64` index 10 | 384 bits, bytes in the file |
| `demo_51prb_4rx.json` | Demonstration profile | 51 blocks, four receive ports | 16QAM, table `qam64` index 10 | 9,736 bits, generated |
| `table2_51prb_1rx.json` | The PDU of the pinned test's Table II row | 51 blocks, one receive port | 256QAM, table `qam256` index 27 | 54,296 bits, generated |

Every key is required and no other key is accepted.

| Key | Contents |
|---|---|
| `schema`, `grant_id` | `pusch-grant/v1` and an opaque identifier used for joins and directory names |
| `carrier` | `scs_khz`, `cyclic_prefix` (`normal`), `grid_size_rb`, `carrier_hz`, `n_cell_id` |
| `bwp` | `start_rb` from CRB 0 and `size_rb` |
| `slot` | `sfn` and `slot`, the slot index in the frame |
| `rnti`, `n_id` | RNTI and the data scrambling identity |
| `allocation` | `mapping_type`, `prb_start` and `prb_count` inside the bandwidth part, `symbol_start`, `symbol_count` |
| `mcs`, `codeword` | Table (`qam64`, `qam256`, `qam64LowSe`) and index. Redundancy version 0, new data |
| `layers`, `tx` | One layer on one antenna port with identity precoding |
| `dmrs` | `config_type` 1, `length` 1, `type_a_position`, `additional_position`, the resulting `symbols`, `ports`, `scrambling_id`, `n_scid`, `cdm_groups_without_data` |
| `rx_ports` | Receive ports of the PUSCH PDU, in antenna order: `[0, 1, ..., N-1]` |
| `tb` | `{"hex": ...}`, or `{"generator": "sha256-counter/v1", "seed": ..., "bytes": ...}`, or in a store entry `{"file": "tb.bin", "bytes": ..., "sha256": ...}` |

The generator concatenates SHA-256 of the ASCII text `seed:0`, `seed:1` and so on, and cuts the result to the stated length. Any language reproduces it.

## What the module derives

| Value | Rule |
|---|---|
| Transport block size | TS 38.214 6.1.4.2 with exact rational arithmetic. The payload must have this size. |
| Coding | LDPC base graph, CRC length and code block count of TS 38.212 |
| DM-RS symbols | TS 38.211 Table 6.4.1.1.3-3 for the mapping type, duration and additional position. A record that lists other symbols is refused. |
| Pilots | Configuration type 1, the sequence of TS 38.211 6.4.1.1.1.1 referenced to subcarrier 0 of CRB 0, in the order symbol then subcarrier |
| Data elements | Increasing subcarrier inside a symbol, then the next symbol. On a DM-RS symbol the subcarriers of the CDM groups without data stay empty. |
| TX scale | Data symbols have unit mean power. The DM-RS amplitude is `10**(3/20)` for two CDM groups without data (TS 38.214 Table 6.2.2-1), which is what the pinned receiver assumes. |
| Bit order | Byte 0 first, most significant bit of each byte first |

The module's transport block size, modulation, base graph and code block count were compared with the pinned OCUDU calculator for all three MCS tables, every valid index, one and two CDM groups without data and 1 to 273 blocks: 46,956 combinations, no difference. The comparison ran on a development host and is not part of a gate. The integration gate compares the three shipped grants on the lab host, and the accepted run found the same size and coding in the record, in OCUDU and in Sionna for each of them.

**Refused before any grid is allocated:** an invalid DM-RS or receive port, a port mapping other than layer i on DM-RS port i, a receive port list other than 0 to N-1 in order, an allocation outside the bandwidth part, a bandwidth part outside the carrier, a repeated DM-RS symbol or port, DM-RS symbols that the table does not give, and a payload whose size differs from the transport block size. `realize(record, make_grid, data)` validates first and calls `make_grid` only afterwards.

## Coordinates and time

All coordinates are carrier coordinates. Subcarrier `k = 12 * CRB + resource element`, and CRB 0 is subcarrier 0 of the grid and the DM-RS reference point. This is the indexing of the pinned resource grid. Subcarrier `k` lies at `(k - 6 * grid_size_rb) * scs` from `carrier_hz`, so the 51-block grid spans -9.18 MHz to +9.15 MHz.

`Grant.symbol_times()` gives the start of the cyclic prefix and of the useful part of each symbol in units of T_c (1,966,080,000 per second), measured from the slot start. At 30 kHz the useful part is 65,536 T_c, the prefix is 4,608 T_c and the first symbol of the slot has 1,024 T_c more. The symbol times are therefore not uniform. S2.2 evaluates the channel at these times.

## Sensing source `pusch_dmrs`

The grant's DM-RS is registered as the first sensing source in `src/ocudu_cuda_isac/sensing/sources/`. A source record (`sensing-source/v1`) names the channel, the link direction, the pilot mask, the reference symbols, the ports, the executing processor and the slot timing. The observer of S3.1 takes its mask and its reference symbols from that record and from nothing else. S6.9 adds a source by adding one module and one `register` call. The registry and the `pusch_dmrs` module import no host, no simulator and no episode module, and a unit test checks that.

The record names the processor that executes the signal in the receiver (`pusch`). Its `backend` is `null` in the transmit reference. The run that executes the grant records it.

## Store entry: `pusch-grant-store/v1`

`export(grant, directory, grids)` writes one directory per grant. It appears only when complete and is never overwritten.

| File | Contents |
|---|---|
| `grant.json` | The record, with the transport block as a file reference |
| `tb.bin` | The original transport block bytes. `sha256sum tb.bin` gives the hash in the manifest. |
| `source.json` | The `pusch_dmrs` source record, with its two arrays by file reference |
| `pilot_index.npy` | int32 `[pilot, 2]`: symbol and subcarrier of each pilot |
| `pilot_ref.npy` | complex64 `[dmrs_port, pilot]`: the transmitted reference symbols |
| `data_index.npy` | int32 `[data_re, 2]`: symbol and subcarrier of each data element, in mapping order |
| `x.npy` | complex64 `[tx_port, symbol, subcarrier]`: the transmit grid of the pinned OCUDU chain as a receiver reads it from the resource grid |
| `x_sionna.npy` | The same grid from Sionna NR, aligned to carrier coordinates |
| `manifest.json` | SHA-256, size, axes, shape, dtype and units of every file, the derived values, the symbol times, the storage tolerance and the provenance of each grid |

The arrays are little-endian NumPy `.npy` files in C order. `numpy.load` reads them, and the module reads and writes them without NumPy. `open_store(directory)` verifies every hash, recomputes the indices and the reference symbols from the grant and compares them with the stored arrays, and checks every grid against the grant: the occupied elements must be exactly the data and pilot elements, and every pilot must equal its reference symbol. A file that was changed, also together with its hash, is refused.

## Aligning the two transmit references

The pinned test builds its transmit grid with the host PDSCH chain, which is legal for a PUSCH without transform precoding because scrambling, coding, modulation and mapping coincide for one codeword. The native adapter `native/ocudu_adapter/fixtures/grant_fixture.cpp` reuses that chain through the test's own factory helper. Sionna's `PUSCHTransmitter` is the independent uplink reference. Four differences are aligned and recorded in `discrepancies.json` of every run.

| Subject | Difference | Alignment |
|---|---|---|
| DM-RS reference point in the pinned test | The test sets `ref_point = PRB0`, the first block of the bandwidth part. PUSCH DM-RS is referenced to CRB 0, and the pinned PUSCH estimator generates it that way. The two coincide only when the bandwidth part starts at CRB 0, as it does in the pinned test. | The adapter sets `CRB0`. `--dmrs-reference prb0` reproduces the test's setting so that the difference is archived: on the offset grant the pilots differ and the host decode fails. |
| DM-RS sequence origin in Sionna | Sionna places the sequence from the first subcarrier of its bandwidth part, and its PUSCH fills that part | Data from a bandwidth part equal to the allocation. DM-RS from a bandwidth part that starts at CRB 0 and ends with the allocation. |
| DM-RS amplitude | OCUDU uses `10**(3/20)`, Sionna `sqrt(2)`. Their ratio is 0.998815. | The record states the OCUDU amplitude. The Sionna pilots are rescaled by the ratio. |
| Storage | The pinned grid stores complex bf16: eight significant bits per component, rounded to nearest even. Sionna computes in double precision. | Values are compared within `2**-8` of each component's magnitude plus `1e-6`. |

The driver also runs a noiseless loopback: the adapter feeds its grid, unchanged, to the host PUSCH processor on every receive port of the grant and compares the decoded bytes with the record's transport block. This checks the grant against the receiver. It is not a replay result, which S2.4 produces.

## Precision of the stored grid

The pinned resource grid stores complex bf16, on the host (`resource_grid_reader.h`, `resource_grid_writer.h`) and on the device (`resource_grid_cuda_visible_impl.h`). A value keeps the upper 16 bits of each float32 component, rounded to nearest with ties to even: eight significant bits per component and four bytes per complex value. `grant.bf16` implements the rule and `x.npy` holds the values after it. Every stage that reads the receiver ingress inherits this precision.

| Quantity | Value | Basis |
|---|---|---|
| Rounding rule | Every OCUDU data symbol equals the Sionna value after `grant.bf16` and every OCUDU pilot equals its reference symbol after it: 14,976 data symbols and 1,248 pilots, no exception | Accepted run. The data count is a gate line, the pilot count was recomputed from the archived grids |
| 16QAM data | Both amplitude levels round up by the same factor. The stored grid is the exact one times 1.00056 | Accepted run |
| 256QAM data | Signal-to-quantization ratio 56.1 dB, mean gain 0.99914 | Recomputed from the archived grids |
| Values of arbitrary amplitude | Signal-to-quantization ratio 55.6 dB at any signal level, because the format is floating point. Each component is within `2**-8` of its magnitude | Computed with the same rule on complex Gaussian values at three levels. No lab measurement |
| Pilots | Reference 0.998815 per component, 3.000 dB. Stored 1.0 per component, 3.010 dB, which is the EPRE the host receiver reports. LS of the stored pilot against the reference gives 1.00119 on every pilot | Accepted run |

Four rules follow for the tickets that read or write the grid.

- **A unity channel is exact.** Every value of `x.npy` is already in the storage format. S2.3 with a unity channel and no noise must return X bit for bit.
- **Any other grid is compared after rounding.** Round the reference with `grant.bf16` and compare for equality, or compare within `2**-8` of each component's magnitude plus `1e-6`, which is the tolerance `STORAGE` in `grant.py`.
- **A noiseless grid has a floor.** The rounding error is about 56 dB below the signal of the same resource element. It is a deterministic function of the grid: identical grids give identical errors, so it does not average out over repeated slots the way noise does. Noise above the floor hides it. Without noise, a scattered path that lies more than about 56 dB under the direct path per resource element competes with it, and only the coherent gain of the processing separates the two. State the margin in any weak-target result.
- **A flat channel reads as 1.00119.** Both pinned estimators are configured with the reference amplitude `10**(3/20)`: `ch_est_config.scaling` in `pusch_processor_impl.cpp` and `e2e_cfg.dmrs_scaling` in `pusch_demodulator_gpu_impl.cpp`. The observer of S3.1 uses the same reference from `source.json`. The factor is real and the same on every pilot, port and symbol, so it does not change a delay, a Doppler shift, an angle or a normalized feature. A check of a known channel includes it.

## Evidence of the integration gate

| File in the run directory | Contents |
|---|---|
| `summary.txt` | Every gate line and the verdict |
| `adapter-build/`, `adapter-receipt.json` | Configure, build and test logs of the adapter, the import file with the pinned compile and link recipe, and the receipt with the hashes of the executable, its sources and the OCUDU build |
| `grant_fixture/grade.json`, `grade.txt` | Case count, check count, every check with its result, the tolerance |
| `grant_fixture/comparison.json` | The measured numbers of every case |
| `grant_fixture/discrepancies.json` | The resolved reference-point discrepancy, the alignments, and one open entry per failed check |
| `grant_fixture/manifest.json` | Command, commit, locked source commits, input hashes, adapter and OCUDU build identity |
| `grant_fixture/<grant_id>/` | `request/` (the adapter's input), `native/` (grid and report of the pinned chain), `native-prb0/` where a case asks for it, `sionna.json`, and `store/`, the archived record |
| `recovered.json` | What the command-line reader recovers from the first case's record |

A failed check stays in the run directory with its numbers. Fix the grant or the mapper before S2.4, and run the gate again into a new directory.

## Accepted run

`make accept-s2.1` passed on the DGX Spark `spark-ecf8` on 2026-11-20 (UTC) at commit `e092623`, run `s21-spark-ecf8-20261120T223232Z`. All 48 gate lines passed, none failed and none was skipped. The integration grade is 3 of 3 cases and 38 of 38 checks. The run started at 22:32:32Z, the adapter was built and tested by 22:32:52Z and the comparison of the three grants took four seconds.

| | `two-prb-offset` | `demo-51prb-4rx` | `table2-51prb-1rx` |
|---|---|---|---|
| Transport block | 384 bits | 9,736 bits | 54,296 bits |
| Coding | 16QAM, BG2, 1 code block, rate 340/1024, 1,152 coded bits | 16QAM, BG1, 2 code blocks, rate 340/1024, 29,376 coded bits | 256QAM, BG1, 7 code blocks, rate 948/1024, 58,752 coded bits |
| Occupied elements | 288 data, 24 pilots | 7,344 data, 612 pilots | 7,344 data, 612 pilots |
| Elements outside the mask or the tolerance | 0 | 0 | 0 |
| Largest data difference between the references | 0.000757 | 0.000757 | 0.00284 |
| Data gain and pilot gain, OCUDU against Sionna | 1.00056 and 1.00119 | 1.00056 and 1.00119 | 0.99914 and 1.00119 |
| Receiver check without noise | CRC pass, 48 bytes equal, 1 port | CRC pass, 1,217 bytes equal, 4 ports | CRC pass, 6,787 bytes equal, 1 port |
| LDPC iterations and SINR the host receiver reports | 1.00 and 96.99 dB | 1.00 and 103.01 dB | 1.00 and 96.99 dB |
| The pinned test's `PRB0` setting | 19 of 24 pilots differ, CRC fails | Not requested | Identical grid |
| SHA-256 of `tb.bin` | `76dccd9e2ed11ce0` | `a2187d90d75129b1` | `810232b5e9915783` |
| SHA-256 of `x.npy` | `ba9258f0630ad918` | `a18b72ae9b9ea84f` | `47f62296ce657633` |
| SHA-256 of `x_sionna.npy` | `c8405ab131f0250d` | `4b049a21cbddee03` | `4f8539ca26f11da0` |

The hashes show their first 16 digits. The full values are in `configs/integration/grant_fixture.json`.

- **Adapter.** `grant_fixture` `7a0e1c8ec1a0`, compiled by GCC 13.3.0 with the flags and the link line of the accepted `pusch_e2e_pipeline_test` `e49b2b81e28a`: `-mcpu=neoverse-v2`, `ENABLE_CUDA`, 50 OCUDU archives, no compiler warning, both CTest cases passed.
- **References.** OCUDU 26.04.0 at `27e84cb3`, still at the locked revision after the runs. Sionna 2.2.0 with PyTorch 2.15.0+cu132 on the CPU.
- **Second reader.** `python -m ocudu_cuda_isac.episodes.grant show` on the first record returned pilot 0 at symbol 2, subcarrier 180, -3.78 MHz from the carrier, reference -0.998815 in each component and transmitted -1.0, and data symbol 0 at symbol 0, subcarrier 180 with the value 0.31640625 in each component, the stored form of 0.316228. `sha256sum tb.bin` gave the block hash of the manifest.
- **Portability.** A rehearsal on an x86_64 host with a CPU-only build of the pinned OCUDU source wrote the same bytes for all three grants: `grant.json`, `source.json`, `tb.bin`, the three index and reference arrays, `x.npy` and `x_sionna.npy`. The gate prints the grid comparison as a NOTE and does not grade it, so a host whose arithmetic differs in the last bit can still pass every graded check.

The evidence is on the Spark under `results/runs/`, which Git ignores.
