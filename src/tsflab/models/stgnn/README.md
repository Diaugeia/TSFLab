---
name: "STGNN"
description: "Graph GRU whose input and state are filtered by a GCN over a learned positional relation masked to the road graph, followed by a Transformer layer over time. Use for traffic speed or flow on a known road graph at short horizons; not without a graph or for long inputs (sequential GRU)."
---

# STGNN

## Idea

- `PositionalGraphFilter` (Eq. 3-7): learned per-node positional vectors give `R = softmax(p_i^T p_j)`; entries outside `A + I > 0` are dropped, self-loops added, and the result is symmetrically normalized before `relu(R_hat X W)`.
- GRU (Eq. 8-9): the filter is applied to each step input and to the previous state; the gating is `graph_conv_gru` with linear maps, shared over nodes.
- Transformer (Eq. 10-15): sinusoidal position (`embed.PositionalEmbedding`) and one post-norm multi-head encoder layer over time per node capture global temporal dependence.
- Prediction (Sec. 3.3): a feed-forward network maps each node's encoded sequence to the horizon; trained with MAE (Eq. 16-17).

## When to use

- Traffic speed or flow on a road network (the paper uses METR-LA and PEMS-BAY, 12 -> 12 steps).
- Needs `adj_mx`; only its sparsity pattern is used. Construction fails without it.
- The GRU runs step by step, so cost grows with `seq_len`; the head size grows with `seq_len * hidden_dim`.

## Configure

- `enc_in`: number of nodes.
- `adj_mx`: `[enc_in, enc_in]` adjacency injected from the dataset graph.

Other hyperparameters: preset defaults in `configs/models/STGNN.toml` (paper: hidden 64, 4 heads, one GRU and one Transformer layer); tune generically.

## Differences

No official code exists; implemented from the paper only. The public `LMissher/STGNN` repository is third-party and was not used. Fidelity is `inferred` because the paper leaves these details open (see the card's issues):

- `phi` in Eq. 3 is identity and the score is `p_i^T p_j` (Eq. 4 as written); the `W1`, `W2` named in the text are not used.
- One positional relation is shared by the input and state filters, each with its own weight.
- Eq. 9 is read as the standard GRU; the Transformer uses layer normalization (post-norm) and a `4 * hidden_dim` feed-forward width.
- The prediction head flattens each node's sequence and applies Linear-ReLU-Linear.
- Only the target series is used; timestamps are ignored.
