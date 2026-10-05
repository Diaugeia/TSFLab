---
name: "pems_bay"
description: "PEMS-BAY: 5-minute traffic speed at 325 Caltrans PeMS sensors in the San Francisco Bay Area (first half of 2017) with a sensor graph, the DCRNN benchmark. Use for graph-based spatiotemporal short-term (12-in/12-out) forecasting; not for long-horizon LTSF or comparison with masked-metric papers without masking."
---

# pems_bay

## Overview

PEMS-BAY is a spatiotemporal traffic benchmark: 325 sensors recording traffic speed every five minutes over six months of 2017 (52,116 steps), with a sensor graph. The data come from Caltrans PeMS. In this repository it loads through the `cauair_st` node loader as a `(T, N, 3)` bundle (value, time-in-day, day-in-week) with an adjacency matrix, so graph and spatiotemporal models can use the road network.

## Protocol and pitfalls

- **Protocol.** DCRNN predicts 12 steps from 12 steps with a 70/10/20 split; the preset's `seq_lens`/`pred_lens` of 12 record that protocol. The DCRNN paper's date range conflicts with its step count (52,116 steps is about 181 days, i.e. January to June).
- **Bundle contract.** `input_dim = 3` keeps the value plus two calendar covariates (time-in-day, day-in-week) appended by `tsf data prepare --from traffic --add-time`; `scale = false` leaves values unscaled, so scale the value channel upstream or flip `scale` if the model expects z-scored input.
- **Window split.** `tsf data prepare --from traffic --splits` (default 0.7,0.1,0.2) splits window centres chronologically; the indices depend on the converter's `--seq-len` and `--pred-len` (default 12 each; rebuild the bundle to change either). The `mean`/`std` in `his.npz` are fitted on training rows only (before `train_end`), and `seq_len`, `pred_len`, `train_end` and window counts are recorded in `his.npz` and `split.json`, so `scale = true` uses no validation or test statistics. Rebuild bundles from older converters (96/96 defaults, whole-series statistics).
- **Adjacency.** `adj_mx.npy` comes from the converter's `--adj` input; check how it was built before comparing graph models across papers.
- **Metrics.** The repository evaluator has no masked-metric option; DCRNN-style papers mask zero (missing) targets, so unmasked numbers are not comparable.
- **`drop_last`.** Training drops its last partial batch; validation and test keep it, so test metrics cover every window.
