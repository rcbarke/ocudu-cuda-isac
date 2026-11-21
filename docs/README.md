# Project documentation

| Read this | Use it for |
|---|---|
| [Development plan](dev/DEVELOPMENT.md) | Sprint order, code ownership, unit/integration failures, gates and evidence |
| [Environment](dev/ENVIRONMENT.md) and [native baseline](dev/NATIVE_BASELINE.md) | Install project and pinned sources, verify each runtime and reproduce the one-layer paper result |
| [Advisor demonstration](demo/ADVISOR_DEMO.md) | Three-minute walkthrough, run order, decisions and exclusions from demo claims |
| [DGX Spark runbook](demo/SPARK.md) | Reproduce upstream CUDA-OCUDU with versioned commands |
| [Architecture](architecture/SYSTEM.md), [deployment modes](architecture/DEPLOYMENTS.md) and [ablation axes](architecture/ABLATIONS.md) | Radio path, ownership, GPU partitions, functional splits, link economics, core placement, and the registry that keeps every ablation and host modular |
| [OCUDU integration](integration/OCUDU.md) and [upstream inventory](integration/UPSTREAM.md) | Source locations, capabilities, reusable open-source code, and missing seams |
| [Aerial ports](integration/AERIAL.md) | Three-computer ladder, the Aerial Framework port, the swap to Aerial-OCUDU for over-the-air work, what each host supplies |
| [Replay contract](interfaces/REPLAY.md) | Runnable host fixture, tensor conventions and verification |
| [Grant record](interfaces/GRANT.md) | The uplink grant record, carrier coordinates, the sensing source, the stimulus store entry and how the two transmit references are aligned |
| [Channel episodes](interfaces/EPISODE.md) | The uplink grant schedule of an episode, the matched target-absent and target-present pair on the grant's subcarriers and symbol times, its analytic check, noise and gain, and the demonstration episode set |
| [KPI stream and live page](interfaces/KPI.md) | KPI record and catalog, page rules and commands for the demonstration |
| [Offline RT exporter](interfaces/RT_EXPORT.md) | The four-time, twelve-tone diagnostic pair, kept as a regression of the axis order |
| [Measurement protocol](research/EVALUATION.md) | Deadlines, denominators, splits, scientific controls and uncertainty |
| [References](research/REFERENCES.md) | Primary papers, repositories, datasets and product documentation |
| [Evidence and reproduction](dev/REPRODUCE.md) | Run manifest, artifacts, independent rerun and failure retention |
