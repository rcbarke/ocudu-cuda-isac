# Evaluation protocol

## Questions and controls

1. How does each receiver stage change sensing at a fixed pilot allocation? Match noisy realizations and antenna calibration, then compare O1, O2 and actual O3 with O0 as an offline reference. Measure false alarms and missed targets as well as error when detected.
2. Which model meets a sensing deadline under actual communication load? Use the same causal histories, features, target definitions, training splits and compute/tuning budgets for the adapted CNN, GRU and compact Transformer. Compare classical, shrinkage and learned covariance using a common calibrated angle estimator. Include mask-aware delay–Doppler, Bartlett and MUSIC, and a calibrated CFAR baseline.
3. Which deployment supports the greatest workload while sustaining PHY and sensing service? Start with a matched absolute workload. Then sweep load and compare feasible ranges. MPS on a full GB300 versus MIG on the same station identifies isolation effects independently of platform speed. DSP-only, AI-only and transport-only runs identify service and copy costs.

## Endpoints and denominators

Let W cover the interval from the first to the last contributing RF sample. Report acquisition, post-window latency and their sum L from the first sample to result publication. At a predeclared false-alarm target, record the fraction of all eligible target-present opportunities with a detection inside physical-error tolerances and L ≤ D. A late, missing, dropped, abstaining, or invalid answer cannot succeed. Count a target-absent false alarm whenever an eligible absent opportunity triggers a positive; report missing absent opportunities separately, since overload could conceal false alarms. State whether an opportunity is a window, a slot or an episode. Also report per-episode totals.

For communication, report scheduled TBs, true CRC outcomes, payload mismatches, offered payload per second, and correctly decoded data bits per unit wall time. Declare retransmission and unique-payload rules. PHY misses use the actual scheduled release plus the configured processing allowance, including late or unfinished jobs in the denominator. Distinguish receiver goodput from application goodput through the UPF.

For timing, save sample counts, p50/p95/p99 and, if statistically supported, p99.9. Counts of uncensored scheduled work accompany latency distributions for completed work. Use per-episode or session-clustered paired intervals. For rare-event claims show the event count and interval; with zero events over N independent opportunities a one-sided 95% upper bound is about 3/N. Adjacent slots can be correlated. Record temperature, throttling, clocks, warm-up, and run order. Report both nodes' total energy for M2 and calculate joules per timely correct estimate from system energy and the actual number of successes.

## Prevent shortcuts in training and evaluation

Split by geometry, trajectory or capture session before windowing. Keep paired present/absent realizations and all overlapping windows in the same split. Fit scalers, clutter templates and threshold on training/validation data only. Hide target-specific paths, geometry truth, future outcomes, file names carrying class, and truth IDs from features. Require the same false-alarm calibration procedure per method. Apply a fixed pilot mask and representation across deployment comparisons, then run payload/quantization sweeps as separate experiments. Validate DOA under stated covariance sample count, model order, steering convention, and calibration. An ambiguous angle, including mirror solutions of a linear array, should be scored using the declared observation geometry.

## External validity

The public PUSCH channel-estimate dataset can test front-end conventions and domain shift using its session splits and labels. Check carrier, array, axes, pilot cadence and target task before interpretation. A file of channel estimates lacks the original transmitted TBs and complete receiver timeline needed to measure CUDA-OCUDU BLER and latency. Treat independent OTA/packetized RU work as an additional evidence tier. See the [reference ledger](REFERENCES.md).
