---
name: "MSGNet"
description: "Graph-and-attention forecaster that learns a separate MixHop graph over series for each FFT-selected period scale. Use for multivariate data with several periods and scale-dependent inter-series correlation; not for hundreds of channels, univariate series, or probabilistic output."
---

# MSGNet

## Idea

- `dominant_periods` picks the top-k FFT periods; `MultiScaleGraphBlock` runs one `ScaleGraphBranch` per period and softmax-weights the branches by spectral strength.
- Within a branch the series is cut into period segments and `nn.MultiheadAttention` models intra-series dependence inside each segment.
- `AdaptiveMixHopGraph` learns a low-rank adjacency per scale and propagates with MixHop (`gcn_depth`, `propalpha`) to capture inter-series relations.
- A single `nn.Linear(seq_len, pred_len)` head follows the blocks; inputs are standardized with detached per-window statistics.

## When to use

- Multivariate series with several salient periods whose inter-series correlations differ by time scale (the paper's motivation); periods are re-detected per batch from the FFT.
- Correlated channels: each scale learns its own adjacency, so channel mixing carries the signal.
- Not for many channels: every branch holds `enc_in x enc_in` adjacency logits and normalizes over channels, so cost and parameters grow with the channel count.
- Not for univariate data (no graph to learn), exogenous or calendar inputs (marks are ignored), or quantile output (point forecast only).

## Configure

- `enc_in`: must equal the dataset's channel count (graph size and channel LayerNorm width).
- `top_k`: number of FFT periods; must be at most `seq_len // 2` (available non-DC bins), and `seq_len` should cover the longest period of interest.

Other hyperparameters: preset defaults in `configs/models/MSGNet.toml`; tune generically.

## Differences

- Independent implementation of FFT scale discovery, scale-specific adaptive MixHop graphs, intra-segment attention, and amplitude-weighted aggregation from the paper; the official repository has no license file and was used for reference only.
- `c_out` must equal `enc_in`; the convolution, skip, and embedding arguments of the official signature (`conv_channel`, `skip_channel`, `embed`, `freq`, `individual`) are accepted and unused.
- The per-scale attention runs in chunks of 32768 sequences (batch x segments x nodes);
  PyTorch's fused kernels reject more than 65535 sequences in one call. Same function.
