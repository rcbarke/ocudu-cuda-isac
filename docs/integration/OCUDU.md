# Integrating the CUDA-accelerated OCUDU receiver

The physical-layer implementation lives in `ext/ocudu-deepsig-cuda-inline`. This repository owns the episode reader, integration adapter, observer transport and research evaluation. Start with one fixed transmitting UE, one layer, four receive channels, 51 PRBs at 30 kHz subcarrier spacing and normal cyclic prefix. Establish a working PUSCH receive baseline before enabling an observer. [Spark baseline recipe](../demo/SPARK.md)

## Source and reproducibility

The parent gitlink pins WG1's `nvcuda_accel_02` branch at `27e84cb3da65205b55dbd56ace6fac93a035331e`. The supplied source ZIP records the same commit in its archive comment. Upstream's Spark benchmark table identifies an earlier measurement commit, `9fd4047b43`. Use the current pin for the run record and retain its own measurements. [WG1 project](https://gitlab.com/ocudu/work_groups/wg1_hw_accel/cuda_accelerated_ocudu) · [Branch source](https://gitlab.com/ocudu/work_groups/wg1_hw_accel/cuda_accelerated_ocudu/-/tree/nvcuda_accel_02) · [Upstream configuration and benchmarks](../../ext/ocudu-deepsig-cuda-inline/docs/phy_cuda_acceleration.md)

From the superproject root in a Git checkout:

```bash
git submodule update --init -- ext/ocudu-deepsig-cuda-inline
git ls-tree HEAD -- ext/ocudu-deepsig-cuda-inline
git -C ext/ocudu-deepsig-cuda-inline rev-parse HEAD
git -C ext/ocudu-deepsig-cuda-inline status --short
```

The tree entry must have mode `160000`. Its SHA and the checked-out SHA must agree. Preserve the upstream worktree when adding research code. If a narrow observation hook requires an upstream patch, version that patch under `patches/`, record its hash and the gitlink in the run manifest, and validate the patched receiver against the unmodified baseline. Build in the superproject's `build/` directory so the dependency remains reviewable. Sionna and Sionna RT have separate dependency paths. A replay using cached channels can execute without invoking either package in the slot loop. [Git submodules](https://git-scm.com/docs/gitsubmodules) · [Sionna RCS example and its PHY dependencies](https://nvlabs.github.io/sionna/rt/tutorials/RCS.html)

## Receiver path in the pinned tree

| Source in `ext/ocudu-deepsig-cuda-inline` | Integration decision |
| --- | --- |
| [PUSCH pipeline test](../../ext/ocudu-deepsig-cuda-inline/tests/integrationtests/phy/upper/channel_processors/pusch_e2e_pipeline_test.cpp), function `run_e2e_pipeline_test` | Reuse the factory, PDU construction, buffer pool, grid writing, processor calls and result notifiers as a staged harness reference. The upstream fixture generates random payloads and uses a single-tap channel. A deterministic episode requires a project adapter supplying matching test data, received grid and PDU. |
| [PUSCH receiver](../../ext/ocudu-deepsig-cuda-inline/lib/phy/upper/channel_processors/pusch/pusch_processor_impl.cpp), `process`, `resident_stub_estimator_results` | The host branch runs its DM-RS estimator. The resident branch can bypass it. The placeholder's channel-estimate accessors fill the output with zeros. Record the selected processing branch for each observation. |
| [GPU demodulator](../../ext/ocudu-deepsig-cuda-inline/lib/phy/upper/channel_processors/pusch/pusch_demodulator_gpu_impl.cpp), [CUDA receive implementation](../../ext/ocudu-deepsig-cuda-inline/lib/phy/cuda/src/pusch_e2e_api.cu) | The GPU path computes its own estimates. Select the exact producing stage for the supported topology and copy its output under a producer event before workspace reuse. A fused kernel may need a small optional export. Define axes, selected pilots, interpolation and normalization explicitly. |
| [PHY tap interface](../../ext/ocudu-deepsig-cuda-inline/include/ocudu/phy/upper/phy_tap/phy_tap.h), [UL processor](../../ext/ocudu-deepsig-cuda-inline/lib/phy/upper/uplink_processor_impl.cpp), [factory](../../ext/ocudu-deepsig-cuda-inline/lib/phy/upper/upper_phy_factories.cpp) | The tap exposes incoming symbols and scheduled PDU metadata. The built-in factory returns null when `OCUDU_HAS_PHY_TAP` is absent. In `process_symbol_pdus`, PUSCH dispatch precedes host snapshot staging and `handle_ul_symbol`. PUSCH can execute asynchronously. Use a read-only observer and explicit completion events for the batch it consumes. |
| [Grid reader](../../ext/ocudu-deepsig-cuda-inline/include/ocudu/phy/support/resource_grid_reader.h), [CUDA-visible grid](../../ext/ocudu-deepsig-cuda-inline/lib/phy/upper/resource_grid_cuda_visible_impl.h) | An owned snapshot may be available in complex BF16, ordered by port, symbol and subcarrier. `stage_owned_snapshot_from_host()` can copy through a pinned host buffer. Check the Boolean results, readiness and lifetime at runtime. |
| [CUDA SRS estimator](../../ext/ocudu-deepsig-cuda-inline/lib/phy/upper/signal_processors/srs/srs_estimator_cuda_impl.cpp) | SRS has its own estimator and snapshot selection. Add it as a separate pilot experiment after the primary PUSCH DM-RS path works. |
| [Dummy RU](../../ext/ocudu-deepsig-cuda-inline/lib/ru/dummy/), [loopback radio](../../ext/ocudu-deepsig-cuda-inline/lib/radio/realtime_loopback/), [OFH compression](../../ext/ocudu-deepsig-cuda-inline/lib/ofh/compression/iq_compression_cuda.cpp), [DPDK Ethernet](../../ext/ocudu-deepsig-cuda-inline/lib/ofh/ethernet/dpdk/) | Dummy timing and sample loopback provide pacing tools. Supply the UE waveform, scattering channel and target labels in the project. CUDA IQ compression and CPU packet transport are distinct parts of the optional fronthaul path. |

## Where the resident estimate lives

S5.2 exports the receiver's own estimate in the demonstration. At the pin the pieces are these.

| Location | What it does |
|---|---|
| `lib/phy/upper/channel_processors/pusch/pusch_processor_impl.cpp` | When the resident plan can bypass the channel estimator, `process` passes `resident_stub_estimator_results`. Its accessors return zeros. |
| `lib/phy/cuda/src/pusch_e2e_api.cu`, one-layer path | Launches `kernel_fused_lse_freq_interp_noise_fp16` (pilot LS, frequency interpolation to all twelve subcarriers of each PRB, noise variance), then `kernel_fd_smoothing_fp16` on the same buffer. |
| `d_estimates_fp16` in the kernel handle | FP16 complex, laid out `[dmrs_symbol, prb, rx_port, 12]` for one layer, with a layer axis before the subcarrier axis for MIMO. Owned by the handle and reallocated when the allocation grows. |
| `lib/phy/cuda/include/pusch_e2e.h` | Declares `pusch_e2e_get_estimates_fp16` "for diagnostics". Upstream's `pusch_mimo_eq_parity_test.cpp` reads the buffer through it after a device synchronize. |
| `lib/phy/upper/channel_processors/pusch/pusch_demodulator_gpu_impl.h` | Holds the handle as the private member `e2e_handle_`. No route to the estimate leaves the demodulator. |

The estimate is per DMRS symbol, after frequency interpolation and smoothing and before time interpolation. That is the stage cuSense reads from Aerial. At the demonstration profile it is two DMRS symbols × 51 PRBs × four RX × 12 subcarriers, 19,584 bytes in FP16. The S5.2 patch gives the observer a read-only route from the demodulator, and the adapter copies on the producer's stream and publishes on the copy's completion event. A receive path must not synchronize the device the way the upstream test does.

## Functional splits and radios in the pinned tree

| Application target | Split | Built by default |
|---|---|---|
| `gnb` (CU and DU in one process), `odu` | Dynamic: chosen by the configured RU | Yes |
| `gnb_split_8`, `odu_split_8` | 8 | No (`EXCLUDE_FROM_ALL`) |
| `gnb_split_7_2`, `odu_split_7_2` | 7.2x | No |
| `gnb_split_6`, `odu_split_6` | 6 | No |
| `odu_low` | 6, the PHY side | Yes |
| `ocu`, `ocucp`, `ocuup` | CU, CU-CP, CU-UP | Yes |

Radio units are `lib/ru/dummy`, `lib/ru/sdr` and `lib/ru/ofh`. Radios are `lib/radio/zmq`, `lib/radio/realtime_loopback`, `lib/radio/uhd` and `lib/radio/sidekiq`. `apps/examples/ofh/ru_emulator` is the Open Fronthaul RU emulator, with a socket transceiver and an optional DPDK transceiver. The split-6 unit under `apps/units/flexible_o_du/split_6` ends in `split6_plugin.h`, and the open tree supplies `split6_plugin_dummy.cpp`, whose FAPI gateways do nothing. The accepted S1.1 build sets `ENABLE_UHD=OFF`, `ENABLE_ZEROMQ=OFF` and `ENABLE_PLUGINS=OFF`, and `ENABLE_DPDK` is off by default. S5.4 adds build profiles for the splits it runs. `configs/testmode.yml` enables a test UE with PUSCH and PDSCH active. It creates load and carries no channel. This project deploys the monolithic, split-8 and split-7.2x targets. Split 6 is listed because it is in the tree, and it is not deployed.

## Channels and reference signals in the pinned tree

| Channel or signal | Location under `lib/phy/upper/` | Acceleration named in the pin's notes | Sensing use |
|---|---|---|---|
| PUSCH and its DMRS | `channel_processors/pusch`, `signal_processors/pusch` | CUDA resident path | Demonstration source, O2 and O3 |
| PUCCH and its DMRS | `channel_processors/pucch`, `signal_processors/pucch` | None: CPU. The paper calls this a residency gap. | Narrowband uplink source, formats 1 to 4 (S6.9) |
| SRS | `signal_processors/srs` | CUDA estimator | Wideband uplink sounding source |
| PDSCH and its DMRS | `channel_processors/pdsch`, `signal_processors/pdsch` | CUDA block processor | Downlink source with a fully known grid, observed at the UE's position (S6.9) |
| PDCCH and its DMRS | `channel_processors/pdcch`, `signal_processors/pdcch` | None: CPU | Always-present downlink control source |
| SSB, CSI-RS, PRS, PT-RS | `channel_processors/ssb`, `signal_processors/ssb`, `nzp_csi_rs`, `prs`, `ptrs` | None named | Optional downlink sources |

The pinned Sionna NR module generates PUSCH only. Masks and sequences for the other channels come from these OCUDU generators and are checked against OCUDU's own test vectors.

## Selectors and coverage

Use the native test's own CLI for the staged baseline. For a later DU/gNB launch, configure the `expert_phy` block exposed by the chosen application's schema. Set `pusch_acceleration_mode: enabled` when testing CUDA, or `disabled` for the CPU reference. Set `ul_cuda_visible_grid_mode: managed` for the explicit Spark managed-grid experiment. The grid policy also accepts `auto` or `pinned`. `srs_acceleration_mode` applies when SRS is actually scheduled. Record the effective selectors, allocations, and selected path. A GPU-enabled executable can still process individual operations on the host. [Upstream selectors and support limits](../../ext/ocudu-deepsig-cuda-inline/docs/phy_cuda_acceleration.md)

The first baseline uses Type-1 DM-RS, non-transform-precoded new PUSCH data and no UCI. This fits the branch's documented resident SCH path. HARQ retransmission uses host LDPC. CSI Part 2 also takes a host path. Keep these cases separately labeled when expanding coverage. For a BLER sweep use `pusch_e2e_sensitivity_sweep`; its ordinary host grid serves a different test purpose from the `direct-managed` producer/consumer check. [Integration targets](../../ext/ocudu-deepsig-cuda-inline/tests/integrationtests/phy/upper/channel_processors/CMakeLists.txt)

The stock non-transform-precoded test uses a PDSCH-side builder for its compatible test grid. Its byte counter compares CPU and GPU decoded data. For a replayed UE, cross-check the transmitted PUSCH data and DMRS against a true uplink reference and compare both receiver outputs with the retained original transport block.

## Observation handoff to implement

1. The adapter binds the episode's input symbols, port map, slot/symbol indices, PUSCH allocation, DM-RS sequence and payload to the same upstream invocation. The received grid includes the direct UE–receiver path, target/clutter contributions and receiver noise. Keep geometric truth in the evaluator.
2. For the advisor checkpoint, form **O2** from an owned copy of the exact receiver-ingress grid passed to the real PUSCH. On scheduled nonzero single-layer pilot REs, compute `H_ls = Y * conj(X_dmrs) / |X_dmrs|²` per RX channel, preserving the sparse mask. Reuse or validate against upstream's [PUSCH DM-RS sequence generation and mapping](../../ext/ocudu-deepsig-cuda-inline/lib/phy/upper/signal_processors/pusch/dmrs_pusch_estimator_impl.cpp). Match slot, symbol, scrambling ID, n_SCID, allocation, DM-RS port and transmitted pilot amplitude. Check parity against a no-noise flat channel. Label this estimate **O2/pilot LS** with its ingress source and alongside the real transport-block result.
3. To obtain **O3** (S5.2, built in the demonstration), capture the actual complex estimate and mask from the selected OCUDU estimator before workspace reuse. Validate values on a known channel and show scheduling/CRC equivalence with observation enabled. A raw grid, a simulator coefficient or the resident host stub is insufficient evidence of this stage.
4. Allocate a bounded observer ring. Acquire a free record before submitting the observation copy. Order that copy after the producer, record completion on the copy stream and release the source hold once the copy is safe. Transfer ownership of the completed record to the consumer. Pair the grid reader's prepare call with its enqueue notification, or its release call if abandoned. If a record or required snapshot is unavailable, increment a drop counter and return to receiver processing.
5. Carry slot, symbol, sequence, tensor dimensions/strides, format, RX ordering, pilot mask, relevant receiver parameters and availability timestamp. Pass only information available by publication time to the model. The observer uses its own memory and must finish with the record before reuse.
6. Run the same episode with the observer disabled and enabled. Compare CRCs, decoded payloads, deadline misses, observer drops and output freshness. Record retries and stage/stream trace evidence. A successful producer-event test on the stock grid establishes a memory contract for that workload. The O2 advisor checkpoint additionally requires a target decision derived from actual receiver-ingress pilot observations. A comparison of internal estimator stages additionally requires verified O3 capture.

## Evidence levels

A host smoke run verifies repository plumbing. The stock CUDA test verifies CPU/GPU receive behavior and residency on its generated workload. A staged adapter using the real PUSCH processor can establish receiver-linked sensing before bringing up the full CU/DU scheduler. Record which of these produced each result, the achieved wall-clock cadence, and whether a UE/core session carried live traffic. A configured or idle 5GC contributes no end-to-end throughput measurement.
