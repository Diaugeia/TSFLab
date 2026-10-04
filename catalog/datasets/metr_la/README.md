---
name: "metr_la"
description: "METR-LA: 5-minute traffic speed at 207 Los Angeles County loop detectors (March to June 2012) with a sensor graph, the DCRNN benchmark. Use for graph-based spatiotemporal short-term (12-in/12-out) forecasting; not for long-horizon LTSF or comparison with masked-metric papers without masking."
---

# metr_la

## Overview

METR-LA is a spatiotemporal traffic benchmark: 207 sensors recording traffic speed every five minutes over about four months in 2012 (34,272 steps), with a sensor graph. Zeros are commonly treated as missing readings (community practice, not verified here). In this repository it loads through the `cauair_st` node loader as a `(T, N, 3)` bundle (value, time-in-day, day-in-week) with an adjacency matrix, so graph and spatiotemporal models can use the road network.

## Protocol and pitfalls

- **Get the files (`script`).** TSFLab does not ship them. Run `uv run tsf data prepare --from dcrnn`: it downloads `metr-la.h5` from the DCRNN README's Google Drive folder and `adj_mx.pkl` from the DCRNN repository (commit `602afd9d`), builds the bundle as `tsf data prepare --from traffic --add-time --freq-min 5` does, checks every sha256, and writes `dataset/metr_la`. It needs `gdown` and `h5py` (`data` extra); if Google Drive refuses, pass a browser-downloaded file with `--h5`.
- **Protocol.** DCRNN predicts 12 steps from 12 steps (horizons 3, 6, 12 reported) with a 70/10/20 split; the preset's `seq_lens`/`pred_lens` of 12 record that protocol.
- **Bundle contract.** `input_dim = 3` keeps the value plus two calendar covariates (time-in-day, day-in-week) appended by `tsf data prepare --from traffic --add-time`; `scale = false` leaves values unscaled, so scale the value channel upstream or flip `scale` if the model expects z-scored input.
- **Window split.** `tsf data prepare --from traffic --splits` (default 0.7,0.1,0.2) splits window centres chronologically; the indices depend on the converter's `--seq-len` and `--pred-len` (default 12 each; rebuild the bundle to change either). The `mean`/`std` in `his.npz` are fitted on training rows only (before `train_end`), and `seq_len`, `pred_len`, `train_end` and window counts are recorded in `his.npz` and `split.json`, so `scale = true` uses no validation or test statistics. Rebuild bundles from older converters (96/96 defaults, whole-series statistics).
- **Adjacency.** `adj_mx.npy` comes from the converter's `--adj` input; check how it was built before comparing graph models across papers.
- **Metrics.** The repository evaluator has no masked-metric option; DCRNN-style papers mask zero (missing) targets, so unmasked numbers are not comparable.
- **`drop_last`.** Loaders keep the last partial batch for every split.
