# KPI stream and live page

The demonstration shows its radio, sensing and system KPIs on one browser page. The page reads a derived stream. The run's raw records remain the evidence, and no gate grades from the page. Tickets: [S4.4 `kpi_stream` and S4.5 `live_gui`](../dev/DEVELOPMENT.md#phase-s4).

## Data path

| Step | Process | Reads | Writes |
|---|---|---|---|
| 1 | Receiver and sensor (the two MPS clients) | Episodes, grants | `radio.jsonl`, `events.jsonl`, `decisions.jsonl` |
| 2 | Aggregator, `python -m ocudu_cuda_isac.telemetry` | Those three files, host samplers | `kpi.jsonl` |
| 3 | Server, `scripts/serve_gui.py` | `kpi.jsonl` | Server-sent events on localhost |
| 4 | Page, `gui/index.html` | The event stream | The screen |

The receiver and sensor never write to the aggregator, the server or the page. They only append to their own files, so a stalled page cannot slow a radio job.

```sh
python -m ocudu_cuda_isac.telemetry --run-dir results/runs/<run-id>
python scripts/serve_gui.py --live results/runs/<run-id>      # during a run
python scripts/serve_gui.py --replay results/runs/<run-id>    # afterwards, recorded pace
python scripts/serve_gui.py --fixture                         # S0.3 smoke outputs, before S2 exists
```

These commands are build contracts of S4.4 and S4.5 and become runnable when those patches land.

## Rules that protect the measurement

1. The aggregator and the server hold no CUDA context. They are neither MPS clients nor GPU compute processes. GPU figures come from the driver's reporting interface.
2. Both run outside the PHY CPU set (5–9 and 15–19 on the Spark).
3. Both open the run directory read-only, apart from `kpi.jsonl`.
4. The page is one static file. It fetches nothing from outside the host, so it works on an isolated lab network.
5. The server binds to localhost. View it from a second machine through an SSH tunnel. A browser on the Spark adds graphics clients to the GB10, so the manifest records page attachment and viewer location, and S4.5 measures the page-off, remote and local arms.
6. Truth appears only in the evaluator view, which reads the offline join after the decisions are logged.
7. A sampler that is unavailable reports `null` with a reason. A producer that stops makes its panel `stale`. Neither is shown as zero.

## The `kpi/v1` record

One JSON object per line, one line per aggregation window (0.5 s by default).

| Field | Content |
|---|---|
| `schema`, `run_id`, `seq`, `t_host_ns`, `window_s` | Identity, order and the host-clock end of the window |
| `state` | `live`, `stale` or `replay` |
| `axes` | The run's resolved registry entries, sent once in the first record |
| `radio` | Window and cumulative values of the radio KPIs below |
| `sensing` | Window and cumulative values of the sensing KPIs below, the latest score and decision |
| `system` | Host and GPU samples, ring state |
| `maps` | Optional delay–Doppler map, location map and receiver-estimate magnitude, each downsampled to at most 64 × 64 values, at most twice per second |
| `evaluator` | Present only in replay with the evaluator view enabled: label and position error |

Quantiles use the nearest-rank rule and always carry their sample count. Cumulative counts must equal the S4.2 scorecard on the same bundle.

## KPI catalog

| Panel | KPI | Definition | Source |
|---|---|---|---|
| Header | Run, commit, profile, arm | Run ID, parent commit, radio profile, `dsp-only`, `ai-only` or `concurrent` | Manifest |
| Header | Model and threshold | Artifact hash and the frozen threshold | Manifest |
| Header | Service levels | The calibrated radio allowance and sensing deadlines, their calibration run and the proposals they replaced | `slo_calibration.json` |
| Radio | CRC pass | Transport blocks with a true CRC pass over scheduled blocks | `radio.jsonl` |
| Radio | Payload match | Decoded bytes equal to the known transmitted bytes, and the mismatch count | `radio.jsonl` |
| Radio | Receiver goodput | Unique correct payload bits per wall second, with the time base stated | `radio.jsonl` |
| Radio | PUSCH service time | p50, p95 and p99 with count, per backend | `radio.jsonl` |
| Radio | Allowance misses | Jobs that finished after scheduled release plus the configured allowance, unfinished jobs included | `radio.jsonl` |
| Radio | Resident fraction | Measured frames on the GPU-resident path | `radio.jsonl` |
| Receiver estimate | Estimate magnitude | The CUDA receiver's own estimate (O3) on each RX across the allocated subcarriers, latest slot, with the observer's pilots overlaid | `events.jsonl` |
| Receiver estimate | Parity with the observer | NMSE between O3 and the observer's pilot LS (O2) on the pilot support, per window | `events.jsonl` |
| Sensing | Presence score | Latest score against the frozen threshold, and the decision | `decisions.jsonl` |
| Sensing | Timely, late, dropped, missing | Counts over eligible opportunities | `decisions.jsonl`, `events.jsonl` |
| Sensing | Sensing latency | First contributing sample to publication: acquisition, post-window time and their sum, against the calibrated deadline (S4.6) | `decisions.jsonl` |
| Sensing | Delay–Doppler map | Power over the declared grid with the strongest cell in ns of excess bistatic delay and signed Hz | `decisions.jsonl` |
| Sensing | Location map | The cuSense-like grid, with its peak in metres | `decisions.jsonl` |
| Sensing | Detections and false alarms | Shown only in the evaluator view, with denominators | Offline join |
| System | MPS clients | Attached client count and PIDs | Host sampler |
| System | GPU | Utilization, memory in use, power and temperature | Host sampler |
| System | Observer ring | Depth, credits, drops and the copy time | `events.jsonl` |
| System | Inference time | p50 and p99 of the sensor's device time, with count | `decisions.jsonl` |
| System | PHY CPU load | Load on the PHY CPU set | Host sampler |
| Timeline | Score, service time and drops | The last 60 s, with arm changes and overload marked | `kpi.jsonl` |

Three tiles at the top summarize the page for a first-time viewer: the radio is decoding correctly, the sensing decision and its latency against the deadline, and the change in PUSCH service time between the DSP-only and the concurrent arm.

## What the page is not

It is a view of one run. It does not grade a gate, compute a confidence interval or replace the scorecard. A screenshot is evidence only together with the recorded stream that reproduces it and the raw records that reproduce the stream.
