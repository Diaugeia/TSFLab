---
name: "DUET"
description: "Distribution-routed mixture of trend/seasonal linear experts, then channel attention masked by a learned frequency-domain metric. Use for multivariate forecasting with heterogeneous temporal distributions and intertwined channel correlations; not for univariate or very many channels."
---

# DUET

## Idea

- Temporal clustering: a gating MLP (`topk_expert_router`) reads each raw channel series (channel-independent, as the official `CI=True`), and `topk_dense_mix` keeps the top-k experts per channel.
- Each `TemporalExpert` splits the input with its own moving-average kernel (`moving_avg`, `moving_avg - 2`, ...) and maps trend and seasonal parts with separate linear layers.
- Channel clustering: `FrequencyChannelMask` compares rFFT amplitudes of the raw channels with a learnable Mahalanobis metric `Q = A^T A` (Eqs. 15-18), turns inverse distances into probabilities, and samples a binary channel mask with a hard Gumbel-softmax; `ChannelAttention` applies it as masked attention (Eq. 20).
- A linear head maps channel tokens to the horizon; `revin` wraps the model.

## When to use

- Designed for multivariate data whose temporal patterns are heterogeneous because the distribution shifts over time, and whose channel correlations are complex (paper: 25 datasets, 10 domains).
- The mask keeps channels with similar spectra and drops unrelated ones; useful when groups of channels co-move.
- The mask is sampled at inference too (as the official code), so a forecast depends on the random seed.
- Pairwise channel attention costs grow quadratically with channel count.

## Configure

- `enc_in`: number of channels. With `enc_in = 1` the channel encoder is skipped.
- `seq_len` sizes the router input and the `(seq_len // 2 + 1)^2` metric `A`.

Other hyperparameters: preset defaults in `configs/models/DUET.toml`; tune generically.

## Differences

Clean-room implementation after reading `decisionintelligence/DUET` at `dcc6e678` (MIT; files in `card.toml`); no source code or checkpoint reused.

- Structure: distributional router, trend/seasonal experts, top-k mixture, frequency-domain channel mask, masked channel encoder, direct horizon head.
- Channel mask follows the official `Mahalanobis_mask` (diagonal scaled by 0.99, its Gumbel logits, finite `-log(1e10)` fill), not the paper's exact `P_ii = 1`, `Bernoulli(P)` and `-inf` (see `[[issues]]`). Distances use `|Ay_i|^2 + |Ay_j|^2 - 2 Ay_i.Ay_j`, equal to the official pairwise difference but without a `[B, N, N, F]` tensor.
- Router noise is one trainable scale per expert (shared `topk_expert_router`), not the official input-dependent noise encoder with `W_H`; the official load-balancing loss (`cv_squared`) is not added.
- Experts use kernels `moving_avg`, `moving_avg - 2`, ...; the official experts all use `moving_avg`. No attention dropout, final encoder LayerNorm or head dropout; `fc_dropout` drops expert output.
- The gating MLP and top-k sparsification (floor renormalization) come from the shared `topk_expert_router`.
- The per-expert moving average stays local: it accepts even kernels by asymmetric padding, which the shared odd-only `series_decomposition` (relied on by BiST) does not.

Citation: Qiu, X., Wu, X., Lin, Y., Guo, C., Hu, J., Yang, B. "DUET: Dual Clustering Enhanced Multivariate Time Series Forecasting." KDD 2025, pp. 1185-1196. doi:10.1145/3690624.3709325.
