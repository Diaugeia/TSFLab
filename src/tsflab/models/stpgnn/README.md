---
name: "STPGNN"
description: "Traffic graph network that identifies pivotal (highly connected) nodes and gates a temporally windowed graph convolution by pivotal membership, in parallel with ordinary graph and temporal paths. Use for node-level traffic forecasting with a road graph; not for non-graph multivariate data or probabilistic output."
---

# STPGNN

## Idea

- `PivotalNodeIdentification` scores nodes by physical plus learned-affinity degree, exposes the top-k pivotal set and a smooth sigmoid membership, and returns a learned adaptive graph added to the physical one.
- `PivotalGraphConvolution` (Eq. 7) averages a sliding temporal window with learned weights, propagates over the graph, and is multiplied by pivotal membership.
- `ParallelSTLayer` runs the pivotal path, an ordinary graph path (Eq. 8) and a temporal conv in parallel and fuses them linearly with a residual LayerNorm; a flatten MLP reads out the horizon.
- Calendar features are concatenated to each node's value at input: (month, day, weekday, hour) fractions from raw stamps, or the dataset's per-node `[time_in_day, day_in_week]` covariates (zero-padded to four) in spatiotemporal mode.

## When to use

- Traffic-flow style node forecasting where a few hub nodes connect to many others and are harder to predict.
- A physical adjacency should be available; without one the physical graph is the identity and only the learned adaptive graph carries spatial links.
- Moderate node counts: the learned affinity is a dense `N x N` matrix.
- Not for non-graph multivariate data, or when probabilistic output is needed.

## Configure

- `num_nodes`: number of graph nodes (injected from the dataset, else `enc_in`); input must be `[B, seq_len, num_nodes]`.
- `adj_mx`: dataset adjacency `[num_nodes, num_nodes]`, injected by the runner; self-loops are added and rows normalized.
- `topk`: size of the inspectable pivotal set, clamped to `num_nodes`.

Other hyperparameters: preset defaults in `configs/models/STPGNN.toml`; tune generically.

## Differences

- Clean-room implementation from the paper's pivotal-node identification, Eq. 7 pivotal graph convolution, Eq. 8 ordinary diffusion and parallel temporal branch; no official source copied.
- Pivotal membership is a smooth sigmoid of centred scores (trainable for all nodes) rather than a hard top-k mask; the exact top-k set stays inspectable via `last_indices`.
- The pivotal temporal window is left-padded so each output uses the previous `kernel_size` steps and the sequence length is preserved.
- The official model (`model.py`, `util.py`) feeds only the value (`in_dim = 1`) and indexes a time-of-day slot embedding (`nodevec_p1[ind]`) to build a time-dependent adaptive graph; this rewrite uses calendar input features and a static learned affinity instead.
