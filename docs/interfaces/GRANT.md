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

# Read an archived record back
python -m ocudu_cuda_isac.episodes.grant show results/runs/<run>/grant_fixture/two-prb-offset/store --pilot 0 --data 0
```

`make accept-s2.1` writes `results/runs/s21-<host>-<UTC time>/` and ends with `Sprint 2.1: ACCEPTED on this DGX Spark.` It uses the CPU only and never rebuilds OCUDU.

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

The module's transport block size, modulation, base graph and code block count were compared with the pinned OCUDU calculator for all three MCS tables, every valid index, one and two CDM groups without data and 1 to 273 blocks: 46,956 combinations, no difference. The comparison ran on a development host and is not part of a gate. The integration gate compares the three shipped grants on the lab host.

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
