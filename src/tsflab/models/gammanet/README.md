---
name: "GAMMANet"
description: "Interleaved road-graph GAT and Mamba scans, first along time then along nodes, on STAEformer-style calendar and adaptive embeddings. Use for sensor-network traffic forecasting with a known road graph and timestamps; not for data without a graph or calendar marks."
---

# GAMMANet

## Idea

- `Model.embed` (Sec. 3.2.1): `input_proj` maps the raw `[flow, time_in_day, day_of_week]` triple (Eq. 2); time-of-day and day-of-week tables and a `(T, N, d_a)` adaptive embedding are concatenated to width `d_h = 3 d_f + d_a`.
- `GraphAttentionConv` is a multi-head GAT on a fixed edge list (every non-zero off-diagonal adjacency entry plus self-loops), heads averaged; `GATSublayer` applies it at every time step with `LN(x + GAT(x))` (Eqs. 4-5).
- `AxisMamba` scans the shared `mamba` mixer (`d_state = 16`, `d_conv = 4`, `expand = 2`) along time (one sequence per node, Eq. 6) or along nodes (one sequence per step, Eq. 8), with `LN(x + Mamba(x))`.
- Eq. (3): `(GAT -> Mamba_T) x L -> (GAT -> Mamba_S) x L` (`gat = false` keeps only the scans, as the official PEMS-BAY/PEMS08 configs); `output_proj` (Eq. 9) maps each node's flattened `T x d_h` block to the horizon.

## When to use

- Designed for road-sensor traffic (METR-LA, PEMS-BAY, PEMS0x) with a physical road graph and strong time-of-day/day-of-week patterns.
- Needs the adjacency for spatial attention and calendar marks for the embeddings; without them the GAT is self-only and the calendar features are zero.
- Reported experiments map 12 input steps to at most 12 outputs despite the long-horizon framing; long horizons are untested.
- The adaptive embedding and output map are sized by `seq_len` and node count, so a trained model is tied to both.

## Configure

- `enc_in`: number of graph nodes (the runner injects `num_nodes`); sizes the `(T, N, d_a)` adaptive embedding.
- `adj_mx`: the dataset adjacency; every non-zero off-diagonal entry is a directed edge, weights ignored.
- `steps_per_day`: samples per day (288 for 5-minute data); time-of-day index is `round(time_in_day * steps_per_day)`.

Other hyperparameters: preset defaults in `configs/models/GAMMANet.toml`; tune generically.

## Differences

- Preset follows Sec. 4.2 and the METRLA entry of `model/GAMMANet.yaml` (`d_f = 24`, `d_a = 80`, `num_layers = 3`, 4 GAT heads, `steps_per_day = 288`, mixed projection).
- Scan layout: default `scan_layout = "axis"` follows the paper's per-axis scans; the pinned code's memory-order flattening (`"official"`) is available but is not the paper's method (see `issues`).
- GAT is a local edge-list implementation of the `torch_geometric` `GATConv` defaults the code uses (`concat=False`, self-loops, LeakyReLU 0.2, no attention dropout, Glorot init); Mamba uses the shared pure-PyTorch `mamba` mixer with the reference step-size init instead of `mamba_ssm` CUDA kernels (slower, not bit-identical); its scan is recomputed in backward (`checkpoint_scan`), because storing its `[tokens, d_inner, d_state]` tensors, which the fused kernel never stores, takes about 7.6 GB per layer at PEMS08 batch 64.
- The preset is used for every spatial dataset; the official PEMS08 entry instead sets `gat: False` and batch 16.
- Inputs: raw calendar marks go through `to_spatiotemporal`; the day-of-week index (0-6) is both the table index and the third raw feature of `input_proj`, matching the official layout.
- Not carried: the unused `spatial_embedding_dim` (always 0 in the configs) and the code's non-mixed projection branch. The loss is the run's criterion (none is released).
- Checked: the GAT against a dense masked-softmax derivation, edge construction, per-axis causality of both scans, the official memory-order layout, reference Mamba widths, embedding layout, Eq. (3) ordering and Eq. (9).
- Citation: He, D., Gao, Y., Yan, H., Jiang, B. "GAMMA-Net: Adaptive Long-Horizon Traffic Spatio-Temporal Forecasting Model based on Interleaved Graph Attention and Multi-Axis Mamba." arXiv:2604.16859, 2026.
