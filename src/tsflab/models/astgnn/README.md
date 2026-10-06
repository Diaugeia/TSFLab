---
name: "ASTGNN"
description: "Graph Transformer encoder-decoder: trend-aware convolutional temporal attention, attention-reweighted dynamic graph convolution, node embeddings, autoregressive decoding. Use for traffic on a known road graph at short horizons; not without a graph, for very many nodes, or long horizons."
---

# ASTGNN

## Idea

- `TrendAwareAttention`: multi-head self-attention over time per node; queries and keys come from width-`kernel_size` convolutions over time (centered in the encoder, causal in the decoder), so attention compares local trends, not single points.
- `DynamicGraphConvolution`: per time step, `S = softmax(Z Z^T / sqrt(d)) / sqrt(d)` reweights the normalized adjacency `D^-1(A+I)` (`adj_norm.transition_matrix`), then `relu(((A_hat * S) Z) Theta)`; it is the position-wise feed-forward of every layer.
- `SpatialEmbedding`: a learned per-node vector (spatial heterogeneity), optionally smoothed by graph convolutions; added with the fixed sinusoidal position of `embed.PositionalEmbedding`.
- Encoder-decoder with pre-LayerNorm residual sublayers; the decoder starts from the last observation and predicts autoregressively.

## When to use

- Traffic flow or speed on a road network with a binary or weighted adjacency (PEMS03/04/07/08, METR-LA, PEMS-BAY).
- Needs `adj_mx`; construction fails without it.
- Cost: dense N x N spatial attention at every step and `pred_len` decoder passes at inference.
- Point output from the target series only; timestamps are not used.
- Feed scaled values (for PEMS presets `dataset.params.scale = true`): the output is a linear map of layer-normalized features, so raw flows in the hundreds train very slowly (the paper scales to [-1, 1]).

## Configure

- `enc_in`: number of nodes.
- `adj_mx`: `[enc_in, enc_in]` adjacency injected from the dataset graph.

Other hyperparameters: preset defaults in `configs/models/ASTGNN.toml` (official PEMS settings: `d_model = 64`, `n_heads = 8`, `num_layers = 4`, `kernel_size = 3`, MAE, lr 1e-3); tune generically.

## Differences

Independent rewrite checked against the authors' code (`guoshnBJTU/ASTGNN` at `9c2e19b9`, no license). The TKDE paper PDF was not reachable from the intake machine (paywalled); architecture facts come from the official code and the paper abstract.

- Recent-window ASTGNN only. The periodic ASTGNN(p) variant needs separately sampled weekly or daily segments that the runtime batch does not carry.
- Official-code bugs are fixed: the detached input embedding, the nonstandard sinusoid table, and shared smoothing layers (see the card's issues).
- Training: `teacher_forcing = true` reproduces the first (teacher-forced) stage; the autoregressive fine-tune stage is a separate run with `teacher_forcing = false`. Evaluation always decodes autoregressively.
- The official min-max scaling to [-1, 1] is replaced by the runner's scaler.
