---
name: "pems07"
description: "PEMS07: 5-minute traffic flow at 883 Caltrans PeMS District 7 (Los Angeles area) sensors (2017-05-01 to 2017-08-31) with a sensor graph, the STSGCN/ASTGCN benchmark. Use for graph-based spatiotemporal short-term (12-in/12-out) flow forecasting; not for long-horizon LTSF or comparison with 6:2:2 results."
---

# pems07

## Overview

PEMS07 is a spatiotemporal traffic benchmark: 883 sensors recording traffic flow every five minutes over 2017-05-01 to 2017-08-31, with a sensor graph. It covers PeMS District 7 (Los Angeles area). In this repository it loads through the `cauair_st` node loader as a `(T, N, 3)` bundle (value, time-in-day, day-in-week) with an adjacency matrix, so graph and spatiotemporal models can use the road network.

## Protocol and pitfalls

- **Split.** TSFLab uses a chronological 7:1:2 split for this dataset. Results reported under other splits (6:2:2) are not directly comparable.
- **Bundle contract.** `input_dim = 3` keeps the value plus two calendar covariates (time-in-day, day-in-week) appended by `tsf data prepare --from traffic --add-time`; `scale = false` leaves values unscaled, so scale the value channel upstream or flip `scale` if the model expects z-scored input.
- **Window split.** `tsf data prepare --from traffic --splits` (default 0.7,0.1,0.2) splits window centres chronologically; the indices depend on the converter's `--seq-len` and `--pred-len` (default 12 each; rebuild the bundle to change either). The `mean`/`std` in `his.npz` are fitted on training rows only (before `train_end`), and `seq_len`, `pred_len`, `train_end` and window counts are recorded in `his.npz` and `split.json`, so `scale = true` uses no validation or test statistics. Rebuild bundles from older converters (96/96 defaults, whole-series statistics).
- **Adjacency.** `adj_mx.npy` comes from the converter's `--adj` input; check how it was built before comparing graph models across papers.
- **Metrics.** The repository evaluator has no masked-metric option; DCRNN-style papers mask zero (missing) targets, so unmasked numbers are not comparable.
- **`drop_last`.** Training drops its last partial batch; validation and test keep it, so test metrics cover every window.
