# Conference matrix

Each family freezes everything except one [ablation axis](../../docs/architecture/ABLATIONS.md). Protocol A holds every simulated result, all on DeepSig-OCUDU, and uses the first thirteen families. Protocol B (the Aerial evaluation) uses the last two.

| Family | Freeze | Vary | Answer | Ticket |
|---|---|---|---|---|
| Receiver stage | H/X/W, grant, pilot mask, scenes | Offline oracle, received grid, pilot LS, the receiver's executed estimate | Information loss and robustness | S5.2, S6.7 |
| Sensing source | Scene, target, radio load | PUSCH, PUCCH, PDSCH and PDCCH reference signals, SRS | Detection per channel at its own bandwidth and cadence | S6.9 |
| UE realization | Scene, target, source | Replayed Sionna RT radar CFR with no UE, OAI softUE with OAI 5GC on split-8 ZeroMQ | Whether the replayed UE stands in for a connected one | S6.10 |
| Clutter suppression | Histories, detector, threshold procedure | Background template, MTI, whitening, adaptive cancellation | Detection under clutter change and stop/start motion | S6.2 |
| Detector | Histories, clutter stage | Fixed threshold, CA, OS, GO and SO CFAR in one and two dimensions | False alarms and detection at clutter edges | S6.2 |
| Angle processing | Detection stage, array and snapshots | Bartlett, MUSIC, shrinkage, learned covariance | Conditional angle accuracy and eligibility | S6.3, S6.6 |
| Sensing method | Histories, splits, calibration budget | Classical chain, cuSense-like CNN, causal GRU, compact Transformer | Timely detection, physical and position error, cost | S6.5 |
| Joint objective | Splits, budgets, false-alarm target | Communication weight of the joint model | Sensing, estimate error, executed block error rate and cost | S6.11 |
| Functional split | Episodes, model, payload | Staged harness, monolithic, split 8, split 7.2x | Whether the observation and decisions survive the boundary | S5.4 |
| Memory path | Radio/AI tasks, offered trace | Grid memory, export path, payload, MPS controls | Copy cost, PHY tails, sensing latency | S5.5 |
| Deployment | Radio/AI tasks and payload | Single Spark MPS, two Spark hosts, GB300 MIG | Maximum load satisfying joint thresholds | S4, S7, S8 |
| Transfer and load | Interface and content per update, chosen radio+AI setting | Local ring, sockets or host-buffer RDMA, cross-GI host bridge. Cells, traffic, observation rate, AI bursts and precision. | Transfer time, feasible capacity and energy per timely estimate | S7, S8, S9.1 |
| Sensing runtime | Records, weights, plan, arms, offered trace | Python reference, native core, fused TensorRT engine | Latency below the reference, agreement of decisions, change in PHY tails | S3.6, S10.1 |
| Receiver and host | Episodes, frozen model | DeepSig-OCUDU receiver, Aerial Framework receiver | Estimate and decision shift between receivers | S10.3 |
| Over the air | Captures on Aerial-OCUDU, training sessions | cuSense as published, the cuSense-like pipeline, the best classical and learned chain | Position error, presence at fixed false alarms, latency by both definitions | S11.4 |

Deadlines and the radio allowance in every family are the calibrated values of that placement (S4.6, S5.6). Published figures from other systems are recorded beside measured ones and are not graded. A family that reports latency, a deadline, contention or energy runs the native core, with a native unit for every stage it times. [Code levels](../../docs/architecture/RUNTIME.md)

Begin with a small balanced factorial subset. Declare primary comparisons, folds, tolerances and targets before running the held-out tests. Use one-factor perturbations at knees of the curves. Record why any infeasible or unsupported cell was excluded. See [acceptance gates](../../docs/dev/DEVELOPMENT.md).
