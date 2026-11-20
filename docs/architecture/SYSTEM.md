# System architecture

One fixed UE illuminates an environment with zero or one moving target. A calibrated four-channel receiver at the DU observes the direct UE path, static multipath and the moving return. The initial experiment uses legal PUSCH data and DMRS from a scheduled grant. It measures how receiver processing and competing CUDA work affect an ISAC decision that has a deadline. No UE or core is attached in the simulated experiments: a Sionna RT radar CFR applied at the receiver stands in for the UE's transmission through the scene. Later experiments register further reference signals (PUCCH, PDSCH and PDCCH) and attach a software UE on the same path. [Ablation axes](ABLATIONS.md)

## Follow one observation

| Step | Producer → consumer | Ownership, representation and clock |
|---|---|---|
| 1. Generate an episode | Offline Sionna RT → offline store | Evaluate complex CFR at the occupied baseband frequencies and actual symbol times. Store axes, dtype, calibration, units, reference, solver settings and hashes. Keep scene truth with the evaluator. |
| 2. Replay | Grant/stimulus + episode → project adapter | Use the existing legal transmit grid and matching DMRS. Form each RX grid as the sum of per-port channel times signal plus receiver-specific noise. H already includes Doppler phase. Put Y in an available OCUDU grid and signal its readiness. |
| 3. Execute | OCUDU grid → pinned PUSCH processor | Preserve the real scheduler or harness grant, allocator, grid ownership, CPU/CUDA selectors and decoder. Record true CRC and compare recovered bytes against the transmitted block. |
| 4. Observe | Owned ingress or estimator output → bounded queue | Release only a completed snapshot. Attach the actual pilot positions, sequence ID, array order, clock domain and first/last sample times. A full ring drops observer work. Move raw Y or CE only on the declared observation route. |
| 5. Sense | Second CUDA process → publication | In M1 attach DSP and sensor as different MPS clients. Build one causal radar cube from the pilot history and publish a presence score, a location map where the model has one, and support metadata. Join each publication to its origin before assessing deadlines. |
| 6. Evaluate | Publications and receiver events → offline evaluation | Join truth on opaque IDs. Keep first contributing sample, output publication and receiver-result availability distinct. Compare target-present and target-absent episodes plus matched DSP-only runs. |

### Data the model can see

| Stage | Material | Legitimate use |
|---|---|---|
| O0 | Simulator paths/CFR, target pose, scattering component | Generate input and score output offline. An oracle gives an upper reference. |
| O1 | Noisy received grid Y from the receiver ingress | Estimator checks, raw-grid comparisons and loss/copy accounting |
| O2 | LS channel estimate on the actual DMRS/pilot REs plus validity mask | Observer-side reference and fallback sensing input. Record whether estimated by a reference observer or a receiver estimator. |
| O3 | Executed receiver's processed estimate with producer-completion evidence | The demonstration's cuSense-like input (S5.2 is built in the demonstration) and the publication comparison of filtering, interpolation and regularization. Decode outcomes are separate events. |

O2 can be produced by extracting pilots from the owned input grid and using the pinned/reference estimator for the supported port configuration. It is a receiver-input-derived estimate. O3 must come from the actual producing implementation. In the inspected WG1 CUDA path, the host-facing estimator stub can return zeros. The executed estimate stays on the device in `d_estimates_fp16` after pilot LS, frequency interpolation and smoothing, and S5.2 exports it from there. A successful CRC alone cannot validate an estimate export. See the [source audit](../integration/OCUDU.md).

## Two clocks and two geometries

The radio clock labels slots and symbol times. A monotonic host clock measures wall-clock arrival, queueing and publication. Each log records both where applicable. For two hosts, include a calibrated map and uncertainty. GPU events give local durations. Define sensing latency from the first contributing sample, including acquisition. A fixed 0.5 ms slot interval does not itself specify the receiver processing deadline.

Use a monostatic single-reflector scene to validate the solver against an analytic answer. The study geometry is bistatic. Target path length is Tx-to-target plus target-to-Rx distance. For a delay relative to the direct path, subtract direct Tx-to-Rx distance before dividing by light speed. For fixed endpoints the target-return Doppler is minus the derivative of the bistatic path length divided by wavelength. Distinguish target echo and background. Physically remove the target and recompute the background for target-absent trials because the target can cast a shadow.

## Sources, UE realization and hosts

The path above does not depend on which reference signal is observed, how the UE is realized or which stack runs the receiver. Three registries carry those choices.

| Choice | Demonstration | Later entries |
|---|---|---|
| Sensing source | PUSCH DMRS | PUCCH, PDSCH and PDCCH DMRS, SRS (S6.9) |
| UE realization | None attached. A Sionna RT radar CFR applied to a legal grant at the receiver replaces the 5GC, the UE and the PRACH connection. | An OAI software UE with an OAI 5GC on the split-8 ZeroMQ boundary (S6.10), then Pixel 7 UEs over the air (S11) |
| Host | DeepSig-OCUDU staged harness | DeepSig-OCUDU DU as monolithic, split 8 and split 7.2x (S5.4), Aerial Framework (S10), Aerial-OCUDU over the air (S11) |

The sensing core sees only the observation record. Each host has one adapter that produces that record. Simulators are used offline to generate episodes and never run in a slot loop. [Port contract](ABLATIONS.md) · [Aerial ports](../integration/AERIAL.md)

## KPI plane

A separate CPU-only aggregator reads the run's records and writes a derived KPI stream for a browser page. It holds no CUDA context, runs outside the PHY CPU set and cannot block a producer. The page is a view. The raw records remain the evidence. [KPI contract](../interfaces/KPI.md)

## Integration boundary

Keep channel generation offline for initial timed runs. The native replay code owns the pacing and grid handoff. For the first demonstration use a precomputed trace, one receiver, and one GPU agent on the Spark. Functional splits beyond the staged harness are launch profiles of the same adapter. Packetized RU, UE registration, core user traffic, and any closed-loop changes require separate endpoints and verification. Deployment-specific transfer paths are defined in [DEPLOYMENTS.md](DEPLOYMENTS.md). The source and native interfaces are linked in [UPSTREAM.md](../integration/UPSTREAM.md) and [OCUDU.md](../integration/OCUDU.md).
