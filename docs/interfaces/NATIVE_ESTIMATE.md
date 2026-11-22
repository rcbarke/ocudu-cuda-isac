# The receiver's own estimate: stage O3 (`native_estimate`)

Sprint 5.2 exports the channel estimate that the resident CUDA path of the pinned receiver computes for its own equalizer. It is the first sprint that changes the receiver, by one patch of 137 added lines and no kernel, and the first whose observation is produced on the device. The stage O2 of Sprint 3.1 is unchanged and stays a stream of its own: it is the parity reference and the fallback input.

Status: implemented and unit accepted on an x86_64 host without an OCUDU build and without a CUDA device. The hardware gate `make accept-s5.2` is open until it has run on the DGX Spark. Nothing in this page is a measurement of the Spark.

## What O3 is

At the pinned commit the one-layer resident path runs two kernels on one buffer of the kernel handle, `d_estimates_fp16`, and the later kernels only read it.

| Step | Kernel | Arithmetic |
|---|---|---|
| Pilot least squares and frequency interpolation | `kernel_fused_lse_freq_interp_noise_fp16` | `h = y conj(p) / |p|^2` in single precision with `|p|^2 = dmrs_scaling^2`, stored as FP16. Inside each resource block a pilot subcarrier keeps its pilot, a subcarrier between two pilots is half of each, in FP16 arithmetic, and subcarrier 11 keeps pilot 5. |
| Frequency smoothing, for more than one resource block | `kernel_fd_smoothing_fp16` | A FIR filter of 15 raised-cosine taps over the pilots of the whole allocation, in single precision. A tap beyond the allocation is left out and the sum is divided by the sum of the taps that were used. The twelve subcarriers of each block are written back by the same interpolation rule and stored as FP16. |

The estimate is per DM-RS symbol and before time interpolation, which happens later inside the equalization kernel and never writes this buffer. That is the stage cuSense reads from Aerial.

| Property | Value |
|---|---|
| Layout | `ocudu_o3_fp16`: `[dmrs_symbol][prb][rx_port][subcarrier_in_prb]`, last axis fastest |
| Element | complex FP16, `{real, imaginary}`, little-endian, 4 bytes. The bytes of a record are the bytes of the device buffer: nothing is reordered, scaled or converted in the receive path. |
| Demonstration profile | 2 DM-RS symbols x 51 PRBs x 4 RX x 12 subcarriers = 4,896 elements, 19,584 bytes, 612 mask bytes |
| Element `[1][3][2][5]` | position 2,621: DM-RS symbol 1, receive port 2, subcarrier 41 of the allocation |
| Validity | one bit per element in payload order. The estimate is interpolated, so every element is defined. A bit is cleared only where a part is not finite. A measured zero is a set bit with the value zero. |
| Scale | `dmrs_scaling = 10^(3/20)` as a single-precision value, 1.4125376, in the `scale` field of the record |
| Unity channel | 1.0011865 in the stage O2 and exactly 1.0009765625 (`0x3C01`) in O3: the ratio of the two reference amplitudes, rounded to FP16 |

## The receiver patch

[`patches/ocudu/0001-pusch-resident-estimate-tap.patch`](../../patches/ocudu/0001-pusch-resident-estimate-tap.patch) adds one header and 49 lines to `pusch_demodulator_gpu_impl.cpp`. The series manifest is [`patches/ocudu/series.json`](../../patches/ocudu/series.json).

- `ocudu::pusch_resident_estimate_tap` is an observer with one method, `on_estimate`. One per process is registered with `set_pusch_resident_estimate_tap`.
- `process_gpu_e2e` calls it once per PDU, on the thread that demodulates, after `pusch_e2e_process_full_gpu_optimized` returned success: every kernel that writes the estimate of that PDU is enqueued on the demodulator's stream by then.
- The call carries a const device pointer, the size, the stream, the device, a launch count, the identity of the PDU (RNTI, frame, slot), the allocation (first block, block count, whether it is contiguous), the DM-RS symbols, ports and layers, the DM-RS type, the scale and which steps ran.
- The tap must not write the buffer, wait for the device or throw. The buffer stays owned by the kernel handle: the next PDU of the same demodulator overwrites it and a larger allocation frees it.
- Without a registered observer the demodulator makes one atomic load per PDU.
- Not tapped: transform precoding, which keeps its estimate in another buffer. The host path and every fallback to it have no device buffer and never call the tap.

The submodule is never patched. `python scripts/native_ocudu.py build --patched` makes a separate worktree of the pinned commit in `build/ocudu-src-o3` (a copy for a source installed from the verified archive), applies the series, checks every touched file against the manifest and builds into `build/ocudu-spark-o3`. Its receipt carries the patch identity under `patch`. The links in `isac/bin` keep naming the accepted Sprint 1.1 build, which is the hook-free control.

## The capture

The rules are in the host-free core `native/ocudu_adapter/estimate/` (C++17, no OCUDU and no CUDA include). The device calls are in `native/ocudu_adapter/observer/estimate_tap.h`, which exists only in the adapter build against the patched receiver (`ISAC_O3_ROUTE`). One capture is open at a time, in three steps.

| Step | When | What it does |
|---|---|---|
| `expect` | before the decode starts | Opens the capture of one prepared decode. With a ring this is the admission: one credit or a counted drop. |
| `on_estimate` | from the receiver's tap, inside the decode | Compares what the receiver reports with the expected PDU, then enqueues one `cudaMemcpyAsync` of the buffer, ordered after the producer, and records one event behind it. It never waits and never throws. |
| `resolve` | after the decode returned | Asks the event for its state with `cudaEventQuery`, sets the validity bits and publishes. Only a completed copy of the expected estimate is published. |

Every decode of the producing backend ends as one of six outcomes, one line of `estimate/captures.jsonl` each.

| Outcome | Reasons | Meaning |
|---|---|---|
| `published` | | one `obs/v1` record on the ring of the O3 stream |
| `captured` | | audit without a ring: the payload is complete and archived |
| `dropped` | `no_credit` | the consumer left no credit. Nothing is copied in the mode timed. |
| `absent` | `no_estimate` | the receiver reported no estimate: the decode did not run the resident path |
| `refused` | `null_estimate`, `layers`, `not_contiguous`, `not_interpolated`, `other_pdu`, `other_allocation`, `size` | the reported estimate is not the expected one |
| `failed` | `submit`, `copy`, `timeout`, `overtaken`, `busy` | the copy was not enqueued, ended with an error, did not complete within `timeout_us`, was overtaken by a later launch, or found an earlier copy still in flight |

Two settings of the plan choose the path, and the effective path is recorded, never silent.

| Setting | Values | Meaning |
|---|---|---|
| `copy_stream` | `producer` (baseline) | The copy is enqueued on the demodulator's stream, so it runs before every later kernel of that demodulator and no later PDU can overwrite the buffer first. It lies between the kernels of the demodulation and those of the decode. |
| | `side` | An event is recorded on the demodulator's stream and the copy runs on a stream of the observer that waits for it. Valid only where a capture is resolved before the next decode starts, as on the staged harness. A copy that a later launch overtook is `failed`, never published. |
| `landing` | `slot` | A timed copy lands in the slot of the ring that was acquired for it. The ring segment is registered with `cudaHostRegister`. Where the runtime refuses that, the copy lands in the staging and the refusal is recorded in `native.json`. |
| | `staging` | The copy lands in page-locked memory from `cudaHostAlloc` and is copied into the slot when it is published. An audited copy always lands here. |

Nothing is copied asynchronously into memory that is not page-locked, because such a copy makes its caller wait for the device. The only call that waits for the device is `drain`, which `settle` calls after the last decode for a copy that ran out of time.

## Plan `pilot-ls-plan/v3`

A plan of version 3 is a plan of version 2 ([SNAPSHOT_RING.md](SNAPSHOT_RING.md)) with one more object, `estimate`, which is `null` or:

```json
{
  "tap": "native_estimate", "stage": "O3", "source": "pusch_dmrs", "layout": "ocudu_o3_fp16",
  "backend": "cuda", "copy_stream": "producer", "landing": "slot", "timeout_us": 2000,
  "ring": null
}
```

`ring` is `null` or a ring object with its own stream name: the O3 stream has its own ring in `DIR/estimate/STREAM.sock`. A timed estimate needs one. A plan written by `ocudu_cuda_isac.episodes.native_estimate` that names an estimate taps the stage O2 as well. Only the `pilot_ls` of the O3 adapter runs a plan that names an estimate. Every other build refuses it with exit status 2, before a processor exists. With `estimate` null a plan of version 3 runs what the same plan of version 2 runs, and that is the hook-disabled control of the patched build.

## Files of a run

| File | Mode | Contents |
|---|---|---|
| `native.json` | both | schema `pilot-ls-native/v3`: everything of version 2, `o3_route`, and `estimate` with the outcomes, the effective copy stream and landing, the device, and the times of the tap (`submit_us`, inside the decode) and of the completion and publication (`resolve_us`, after it) |
| `estimate/captures.jsonl` | both | one `o3-capture/v1` line per decode of the producing backend: the outcome, the trace of the producer (launch, backend, steps, scale, allocation, DM-RS symbols) and of the copy (stream, landing, completion event, polls, times) |
| `estimate/o3.jsonl` | audit | one `obs-descriptor/v1` record per captured estimate, with the trace of its producer in `host` |
| `estimate/NNNN.o3.fp16`, `NNNN.o3.mask` | audit | the payload and its validity bits, with their SHA-256 in the record |
| `estimate/ring.json` | with a ring | the O3 ring and its counts |

`observer/` and `ring/` are those of versions 1 and 2, so their readers keep working.

## Parity

The exported estimate is compared with the **matched estimator** on the stage O2 of the same opportunity: `ocudu_cuda_isac.sensing.native_estimate.matched` applies the three steps above to the comb payload of O2, with the receiver's precision at each step. The verdict is taken on the common physical pilot support, the six pilot subcarriers of each block. `o3_parity.json` (`o3-parity/v1`) holds, per estimate and in total, the numerator `sum |O3 - R|^2`, the denominator `sum |R|^2` and the sample count.

**Finding at the pin: the smoothing kernel filters in place.** One CUDA block per resource block and DM-RS symbol reads the pilots of up to two neighbouring blocks on each side and writes its own. Nothing in the kernel fixes whether a block reads a neighbour before or after that neighbour was written, so two decodes of one grid may differ. The matched estimator is the order in which every block reads unfiltered pilots. A flat channel is filtered to itself in every order.

The tolerances are declared in [`configs/integration/native_estimate.json`](../../configs/integration/native_estimate.json) from a CPU model study on calibration channels (2 to 6 paths, excess delays up to 800 ns, without noise and at 30, 20 and 10 dB, 12 seeds each), which also ran the in-place filter in ascending, descending and random block schedules.

| Class | Control run | Declared tolerance |
|---|---|---|
| flat | unity channel, no noise | every pilot within 1 FP16 spacing at the magnitude of the reference, relative error at most 0.002, NMSE at most 1e-6. Every element is exactly 1.0009765625. |
| varying | direct path of the reference pair, no noise | NMSE at most 1e-6, or a quarter of the NMSE that the smoothing step itself makes on the same pilots when that is larger |
| noisy | reference pair at 30 dB | the same rule |

In the study the in-place orders stay below 7.7 percent of the smoothing effect, and an estimate without smoothing is at 100 percent, so the declared quarter separates the two. Without the in-place effect, model variants differ by at most one FP16 spacing. An estimate of zeros never meets a reference that is not zero, whatever the bounds. The driver records, without grading it, how many estimates lie within one FP16 spacing of the matched order.

## Commands

```bash
python tests/unit/test_native_estimate.py -v          # unit gate: no OCUDU, no CUDA
python scripts/receiver_patch.py check                # the series against its files and the lock
make native-build-o3                                  # the patched receiver, build/ocudu-spark-o3
make adapter-build-o3                                 # the adapter with the O3 route, build/ocudu-adapter-o3
build/ocudu-adapter-o3/observer/pilot_ls --route      # which receiver the tool is linked against
make accept-s5.2                                      # every gate, on the DGX Spark
```

`make accept-s5.2` checks the series, builds the patched receiver and both adapters, and runs ten cases on the plans of the accepted Sprint 3.1 run: `control`, `patch`, `flat`, `varying`, `noisy`, `hook-off`, `side-stream`, `timed`, `refusals`, `backends`. It ends ACCEPTED only when every decode of the patched receiver is the one the unpatched build wrote (key, CRC flag, decoded bytes, executed path), the stage O2 is the accepted one, every decode on the resident path left one estimate, and every estimate meets its declared tolerance.

## Open risks carried by the gate

| Risk | Where it shows | What is prepared |
|---|---|---|
| The copy on the producer's stream sits between the demodulation and the decode kernels. The pin moved its own SINR readback off that stream because a device-to-host copy between two kernels costs a dispatch boundary on the GB10. | `hook-off` NOTE: CUDA service time unpatched, hook off and hook on | `copy_stream: side`, graded in the case `side-stream`. The timed distribution is an S5.5 cell. |
| `cudaHostRegister` of the shared ring segment may be refused on the GB10. | `timed` NOTE and `estimate.landing` in `native.json` | automatic landing in the staging, recorded |
| The in-place filter may make the exported estimate differ between two decodes of one grid. | `timed` NOTE: payloads identical to the audited run | the declared tolerance covers every block order of the model |

## What is not built here

Sensing on O3 (S3.5), the live DU allocations (S5.1 repeats this parity fixture), the timed memory-path matrix (S5.5) and the in-process control cell. More than one layer and transform precoding are recorded as unsupported.
