# Conference matrix

| Family | Freeze | Vary | Answer |
|---|---|---|---|
| Receiver stage | H/X/W, grant, pilot mask, scenes | Offline oracle, received grid, pilot LS, processed CE | Information loss and robustness |
| Sensing method | Histories, splits, calibration budget | Classical spectrum/CFAR, CNN, causal GRU, compact Transformer | Timely detection, physical error, cost |
| Angle processing | Detection stage, array and snapshots | Bartlett, MUSIC, shrinkage, learned covariance | Conditional angle accuracy and eligibility |
| Deployment | Radio/AI tasks and payload | Single Spark MPS, two Spark hosts, GB300 MIG | Maximum load satisfying joint thresholds |
| Transfer | Interface and content per update | Local ring, sockets or host-buffer RDMA, cross-GI host bridge | Transfer time, CPU and bandwidth impact |
| Load | Chosen radio+AI setting | Cells, traffic, observation rate, AI bursts and precision | Feasible capacity and energy per timely estimate |

Begin with a small balanced factorial subset. Declare primary comparisons, folds, tolerances and targets before running the held-out tests. Use one-factor perturbations at knees of the curves. Record why any infeasible or unsupported cell was excluded. See [acceptance gates](../../docs/dev/DEVELOPMENT.md).
