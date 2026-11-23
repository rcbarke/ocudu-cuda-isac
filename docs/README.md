# Project documentation

| Read this | Use it for |
|---|---|
| [Development plan](dev/DEVELOPMENT.md) | Sprint order, code ownership, unit/integration failures, gates and evidence |
| [Environment](dev/ENVIRONMENT.md) and [native baseline](dev/NATIVE_BASELINE.md) | Install project and pinned sources, verify each runtime and reproduce the one-layer paper result |
| [Advisor demonstration](demo/ADVISOR_DEMO.md) | Three-minute walkthrough, run order, decisions and exclusions from demo claims |
| [DGX Spark runbook](demo/SPARK.md) | Reproduce upstream CUDA-OCUDU with versioned commands |
| [Architecture](architecture/SYSTEM.md), [deployment modes](architecture/DEPLOYMENTS.md) and [ablation axes](architecture/ABLATIONS.md) | Radio path, ownership, GPU partitions, functional splits, link economics, core placement, and the registry that keeps every ablation and host modular |
| [Code levels and the native core](architecture/RUNTIME.md) | Which code runs at which level, the native sensing core that every timed stage runs in, the runtime axis, how the core ports to the Aerial stacks, and which run supports which claim |
| [OCUDU integration](integration/OCUDU.md) and [upstream inventory](integration/UPSTREAM.md) | Source locations, capabilities, reusable open-source code, and missing seams |
| [Aerial ports](integration/AERIAL.md) | Three-computer ladder, the Aerial Framework port, the swap to Aerial-OCUDU for over-the-air work, what each host supplies |
| [Replay contract](interfaces/REPLAY.md) | Runnable host fixture, tensor conventions and verification |
| [Grant record](interfaces/GRANT.md) | The uplink grant record, carrier coordinates, the sensing source, the stimulus store entry and how the two transmit references are aligned |
| [Channel episodes](interfaces/EPISODE.md) | The uplink grant schedule of an episode, the matched target-absent and target-present pair on the grant's subcarriers and symbol times, its analytic check, noise and gain, and the demonstration episode set |
| [Channel application](interfaces/GRID_APPLY.md) | The native step that fills the receiver grid: `Y = g (sum(H X) + W)` with the grid gain of the episode, the replay plan, the owned grids and their states, the stored-grid descriptor and what the receiver side may read |
| [Receiver replay](interfaces/REPLAY_PUSCH.md) | The native step that hands each filled grid to the host and the CUDA PUSCH processor: the six outcomes and their keys, the grid held until the decoder's callback, the transport block per opportunity, the replay plan, the `radio/v1` record, the executed path, and the audit that closes Sprint 2 |
| [Pilot least squares](interfaces/PILOT_LS.md) | The first observer on the receiver grid: the estimate `H = Y conj(X) / |X|^2` and its precision, the layout and the validity bits of the stage O2, the `obs-descriptor/v1` record, the plan of the tapped replay, and how a tapped run is compared with the untapped control |
| [Observation contract](interfaces/OBSERVATION.md) | The observation record, the shared-memory ring between the receiver and the sensor, payload layouts and the view the native core consumes |
| [Observations on the ring](interfaces/SNAPSHOT_RING.md) | Sprint 3.2: what the ring and the timed path are built from, the plan `pilot-ls-plan/v2`, the files and the account of a run, the gate and what a development host checked |
| [Radar cube and features](interfaces/PILOT_FEATURES.md) | Sprint 3.3: the cube and its axes, the specification of a sensor, what a causal stream answers at each record, the delay-Doppler view and the feature vector with a baked example, the registry of steps with their devices, the files of a sensor, the evaluator side and the gate |
| [Presence head](interfaces/PRESENCE_HEAD.md) | Sprint 3.4: the artifact of two files and its arithmetic with a baked example, how a head binds to the names of a vector, what the fit and the threshold may read, the decision record, the files of a sensor and of a training run, the split of an episode set and its replay, and the gate |
| [The receiver's own estimate](interfaces/NATIVE_ESTIMATE.md) | Sprint 5.2: the stage O3 and its layout, the receiver patch and its series, the capture and its six outcomes, the plan `pilot-ls-plan/v3`, the matched estimator, the declared parity tolerances, the gate and the accepted run |
| [Native sensing core](interfaces/SENSE_CORE.md) | Sprint 3.6: the C API and the plan of `native/sense_core/`, its units with their declared tolerances against the reference, the engine directory, the sensor process `sense_host` with the files it writes, the latency definitions and the gate |
| [Paced clients under MPS](interfaces/MPS_PACING.md) | Sprint 4.1: the paced plan `pilot-ls-plan/v4`, the release rule, the files of a paced run, the launcher `scripts/run_m1.py` with its arms, the proof of the label MPS, the contention controls and `mps_pacing.json`; what the accepted run measured, and the cold start of the receiver |
| [cuSense front end and location model](interfaces/CUSENSE_FRONTEND.md) | Sprint 3.5: the three stages with a baked example, the two input entries, the causal window stream, the model entry and its losses, the artifact of four files, what the fit and the threshold may read, the decision record and the optional stage, the evaluator side, what the gate grades and what it records |
| [KPI stream and live page](interfaces/KPI.md) | KPI record and catalog, page rules and commands for the demonstration |
| [Offline RT exporter](interfaces/RT_EXPORT.md) | The four-time, twelve-tone diagnostic pair, kept as a regression of the axis order |
| [Measurement protocol](research/EVALUATION.md) | Deadlines, denominators, splits, scientific controls and uncertainty |
| [References](research/REFERENCES.md) | Primary papers, repositories, datasets and product documentation |
| [Evidence and reproduction](dev/REPRODUCE.md) | Run manifest, artifacts, independent rerun and failure retention |
