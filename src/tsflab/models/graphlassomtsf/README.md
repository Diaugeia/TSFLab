---
name: "GraphLassoMTSF"
description: "Graphical-lasso precision of the training series defines a sampled sparse graph for a diffusion graph-GRU seq2seq. Use for multivariate series with sparse conditional dependencies but no known graph; not for independent channels or long horizons."
---

# GraphLassoMTSF

## Idea

- Phase 1 (Eq. 1, `estimate_precision`): z-score the training series, add `cov_eps` to the covariance diagonal, solve the graphical lasso (off-diagonal l1, `lasso_lambda = 0.02`, ADMM) on the correlation matrix and rescale; `training_setup` runs it once on the training split and stores `Theta`.
- Phase 2 sampling (Eq. 4): each row of `Theta` is a categorical over target nodes; training draws one hard Gumbel-max edge per row per forward pass, evaluation uses the expectation `softmax(Theta)`.
- Graph convolution (Eq. 5): `[x, P x, ...] W + b` with `P = (D^{-1}(A + I))^T`; `GraphGRUCell` (Eq. 6) uses diffusion convolution for reset, update and candidate maps on the shared `graph_conv_gru` gating.
- Encoder-decoder: stacked cells encode the history, the decoder starts from a zero GO symbol and projects one value per node per step; training adds the official curriculum (label fed with probability `k / (k + exp(step / k))`).

## When to use

- Designed for multivariate series whose channels share sparse conditional dependencies but no graph is given; the graph is learned from the training split.
- Mixes channels through the learned graph, so it helps when cross-channel dependencies are real; channels that move independently gain nothing.
- Autoregressive GRU decoding suits short horizons; long horizons are slow and accumulate error.
- Point output only.

## Configure

- `enc_in`: the dataset's channel count (each channel is one node of the precision matrix).
- The precision graph is fitted on the training split automatically before the first epoch.

Other hyperparameters: preset defaults in `configs/models/GraphLassoMTSF.toml`; tune generically.

## Differences

- Independent rewrite of arXiv 2306.17090 v1 after reading the pinned official code (no license file, `NOASSERTION`); the graphical lasso is solved as the official `gglasso` does (connected-component screening, then ADMM per block), which always returns a positive-definite iterate on ill-conditioned wide data; cost is at most 1000 eigendecompositions of the largest block.
- Only the static GraphLASSO variant; the time-varying variant (Eqs. 2, 7) has no defined rule at inference and is omitted.
- Edge sampling follows the code (one Gumbel-max edge per row of raw `Theta`), not the paper's Bernoulli sampling; evaluation uses the expected graph so inference is deterministic.
- Precision fitted on the catalog training split (the official 80% cut leaks validation data); the dual-random-walk filter option is omitted.
- Teacher forcing is skipped for `MS` targets; training uses the configured loss and catalog optimizer.

Full detail: `reference.md`.
