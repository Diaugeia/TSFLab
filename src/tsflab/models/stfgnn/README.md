---
name: "STFGNN"
description: "Spatial-temporal fusion graph network: GLU graph convolutions on a 4-step graph fusing the road graph with a DTW temporal graph from the training split, plus a gated dilated convolution per layer. Use for short-term traffic flow forecasting on a known road graph; not for data without a graph or very large node counts."
---

# STFGNN

## Idea

- Temporal graph `A_TG` (Alg. 1-2): before the first epoch, the training series is cut into whole days, each day z-scored, and banded DTW (`dtw_band = 12`) compares nodes; each node keeps its nearest `max(1, floor(N * dtw_sparsity))` nodes (`fit_temporal_graph`).
- Fusion graph (`fusion_adjacency`, 4N x 4N): `A_TG` on the outer diagonal blocks and corners, the road graph on the inner blocks, identity links between steps, self-loops; a learnable mask reweights it (as in STSGCN).
- Each layer cuts `T - 3` windows of 4 steps and runs one `SynchronousGraphModule` per window (3 GLU convolutions, max, crop to block 1), then adds a gated dilated convolution (`tanh * sigmoid`, kernel 2, dilation 3).
- 3 layers turn 12 input steps into 3; each horizon step has its own two-layer FC; Huber loss (`huber_delta`).

## When to use

- Traffic flow sensor networks with a road graph and a daily cycle, short horizons (the paper: 12 to 12 steps on PEMS03/04/07/08).
- Needs `adj_mx`; construction fails without it. Without an ordered training series the temporal graph stays the identity (a warning is raised).
- Dense `4N x 4N` matrices; DTW costs `O(N^2 * L * band * days)` once (seconds on a GPU for PEMS08). Calendar marks are not used. Point output only.

## Configure

- `enc_in`: number of nodes; `adj_mx`: injected road graph.
- `steps_per_day`: samples per day, the DTW period (288 for 5-minute data).
- `dtw_sparsity`: temporal neighbors per node as a fraction of `N`; `num_layers`: `seq_len` must exceed `3 * num_layers`.

Other hyperparameters: preset defaults in `configs/models/STFGNN.toml`; tune generically.

## Differences

Clean-room implementation from the paper; the unlicensed official MXNet code (`a02feee5`: model, utils, main, DTW scripts) was read to settle omissions and nothing was copied. The banded DTW reproduces `compute_dtw` exactly and the fusion graph `construct_adj_fusion` exactly (both checked on random inputs). Followed from the code: no residual inside modules, crop to block 1, links two steps apart, order-1 DTW on per-day z-scored training days, `T - 3` modules per layer, MXNet `Xavier(0.0003)` init.

- Leakage check: the official DTW scripts use only the first 60% of the data (the training split); here only the history span of the catalog training windows is used.
- Temporal neighbors exclude the node itself (the official top-k counts it, leaving PEMS08 almost without temporal edges).
- Huber loss on the runner's scaled targets (official: raw flow values); learning-rate schedule and early stopping are runner settings.
