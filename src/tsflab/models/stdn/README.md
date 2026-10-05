---
name: "STDN"
description: "Traffic model that gates node embeddings into trend and seasonal parts: GRU plus dynamic-graph diffusion for trend, history cross-attention for seasonality. Use for spatio-temporal node forecasting with a road graph and daily/weekly patterns; not for non-spatial multivariate data or probabilistic output."
---

# STDN

## Idea

- Traffic has complex spatial and temporal dynamics; STDN builds a dynamic graph, adds spatio-temporal embeddings, and disentangles trend-cyclical and seasonal components per node before an encoder-decoder.
- Adds a Laplacian-eigenvector spatial embedding (from the adjacency) to time-of-day and weekday embeddings, and splits projected values into trend and seasonal parts with a sigmoid gate (`trend_gate`).
- Trend branch: a GRU summary per node builds a top-k sparse softmax graph per batch, and `DynamicDiffusion` propagates the summary over it, repeated across the horizon.
- Seasonal branch: future time/horizon embeddings query multi-head attention over the seasonal history (`history_attention`); both futures are concatenated and mapped to one value per node.

## When to use

- Traffic flow or speed on a sensor network with a known adjacency (Laplacian positions; identity when none is given) and strong time-of-day and weekday cycles.
- Relations between nodes that change over time: the trend graph is rebuilt per batch.
- Point forecasts only; one value per node.

## Configure

- `enc_in` follows the node count; `num_nodes` and `adj_mx` come from the dataset graph.
- `time_slice_size` follows the sampling frequency: minutes per step (60 for hourly, 5 for 5-minute data); it sets `1440 // time_slice_size` time-of-day slots.
- Other hyperparameters: preset defaults in `configs/models/STDN.toml`; tune generically (`reference` is capped at the node count).

## Differences

Local rewrite after reviewing the paper and the pinned BasicTS implementation (`GestaltCogTeam/BasicTS@c218c07`, Apache-2.0).

- Spatial Laplacian positions and calendar embeddings gate the trend/seasonal decomposition; dynamic diffusion forecasts the trend and history-to-future attention the seasonal branch, as in the reference.
- Spatiotemporal batches give `[time_in_day, day_in_week]` node covariates; node 0's calendar is read for every node, as the official runner (`TE[:, :, 0, :]`). Raw six-column marks are converted to the same fractions.
- The preset uses `time_slice_size = 5` (288 slots), the official PEMS value; set 60 for hourly data.
- Without calendar marks, time-of-day slots fall back to step position modulo the slot count and weekday to zero.

Citation: Cao, Wang, Jiang, Yu, Dong, "Spatiotemporal-aware Trend-Seasonality Decomposition Network for Traffic Flow Forecasting", AAAI 2025, doi:10.1609/aaai.v39i11.33247 (arXiv:2502.12213).
