# Reproduce a run

Start in the local `isac` environment. `make check`, `make smoke` and `make pins` are portable. S0 also ships the source/environment verifier; S1 ships the gates `make accept-s1.1` and `make accept-s1.2` for the enabled host, with `make native-build` and `make paper-baseline` as their single steps. S2.1 ships `make accept-s2.1`, with `make adapter-build` and `tests/integration/grant_fixture.py` as its single steps. S2.2 ships `make accept-s2.2`, with `scripts/export_rt_episode.py` and `tests/integration/episode_pair.py` as its single steps, and `make episode-set` for the demonstration set. See [environment setup](ENVIRONMENT.md) and [native acquisition](NATIVE_BASELINE.md). The native commands whose implementation is assigned to later sprints must be added to the run manifest and runbook when implemented. Store each acquisition as a new run under `results/runs/<run-id>/`; failed acquisitions are useful evidence.

| File in a hardware run | Required contents |
|---|---|
| `manifest.json` | Parent commit, upstream SHAs, patch digests, build/toolchain/backend, machine/driver/CUDA, MPS clients or MIG UUIDs/profiles, thermal and power policy, CPU affinity, clock domains, run purpose, scenario/model hashes, offered load and predeclared budgets, the resolved entry on every ablation axis, and whether the KPI page was attached and from where |
| `commands.sh` and `stdout.log` | Exact setup, execution, elapsed wall time, exit codes, independent transcript checks, start/stop markers |
| `events.jsonl` | Opaque episode, absolute opportunity/sequence, cell/UE, slot and symbol, queue state, producer/consumer events, timestamps with units and domains, explicit drops/expiry |
| `radio.jsonl` | Grant, legal RE mask, TB identifier and byte count, true CRC/error/retry, backend and PHY deadlines |
| `decisions.jsonl` | Frozen model/algorithm, supported output and score, input observation span, publication time, status |
| `kpi.jsonl` | Derived KPI windows for the live page. Never an input to a grade. Its counts must equal the offline reduction. |
| `slo_calibration.json` | For any run graded against a deadline or an allowance: rule hash, sample counts, derived values, verification run and host state |
| `offline-evaluation.json` | Post-run truth join, denominator reconciliation, errors, false alarms, timely counts, CIs and energy basis |
| `summary.md` | Claim, stage executed, run length, limits, failed gates, figure/table source hashes |

For a rerun, start with the [locked source acquisition and verification](../../ext/README.md). Reconstruct the environment recorded by the original run and validate the episode checksum. Use the same grant/stimulus/noise, mask, array mapping and model checkpoint. Compare payload bytes and numerical outputs with declared tolerances before interpreting timing. Timing varies with host conditions. Record paired intervals and offsets rather than requiring bit-identical timestamps. Preserve both runs if either fails.

Freeze long-running publication artifacts by tagging the parent Git revision, retaining trace IDs and immutable hashes in a public manifest, and archiving externally distributable episodes/results separately. Record access instructions for restricted datasets. Choose an appropriate license for original code before a public release.
