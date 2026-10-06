---
name: "STSGCN"
description: "Spatial-temporal synchronous GCN: stacked GLU graph convolutions on a graph joining 3 adjacent steps, with a learnable edge mask, per-window modules and per-horizon heads. Use for short-term traffic flow forecasting on a known road graph; not for data without a graph, long inputs or very large node counts."
---

# STSGCN

## Idea

- Localized graph (`localized_adjacency`): the spatial 0/1 graph on 3 diagonal blocks, each node linked to itself in the adjacent step, self-loops; a learnable mask `W_mask` (initialized to the edge pattern) reweights it element-wise for all convolutions (Eq. 7).
- `STSGCL`: adds learnable temporal `[T, C]` and spatial `[N, C]` embeddings, cuts `T - 2` windows of 3 steps, and runs one `SynchronousGraphModule` (STSGCM) per window: 3 GLU graph convolutions, element-wise max, crop to the middle step.
- 4 layers turn 12 input steps into 4; each horizon step has its own two-layer FC on the flattened `[4 * C]` node features (Eq. 8).
- Trained with the Huber loss (`huber_delta`, official `rho = 1`).

## When to use

- Traffic flow or similar sensor networks with a given road graph and short horizons (the paper: 12 to 12 steps of 5 minutes on PEMS03/04/07/08).
- Needs `adj_mx`; construction fails without it. Positive weights count as edges, so weighted graphs lose their weights.
- Dense `3N x 3N` matrices: memory grows with `N^2`. Calendar marks are not used. Point output only.

## Configure

- `enc_in`: number of nodes.
- `adj_mx`: injected road graph.
- `num_layers`: `seq_len` must exceed `2 * num_layers`.

Other hyperparameters: preset defaults in `configs/models/STSGCN.toml`; tune generically.

## Differences

Clean-room implementation from the paper; the unlicensed official MXNet code (`3f825f64`: `models/stsgcn.py`, `utils.py`, `main.py`, configs) was read to settle omissions and nothing was copied. Followed from the code: the ReLU input layer, the 128-wide output heads, one mask shared by all layers, cropping after the max, and the MXNet `Xavier(magnitude = 0.0003)` initializer for weights and embeddings (initial outputs are near zero). The `sharing` and `relu` ablation variants are `individual = false` and `activation = "relu"`.

- Huber loss on the runner's scaled targets; the official code feeds standardized inputs but computes the loss on raw flow values, so `huber_delta = 1` covers a different range.
- Learning-rate warm-up and polynomial decay, early stopping and metrics are runner settings.
