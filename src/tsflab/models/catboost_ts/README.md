---
name: "CatBoostTS"
description: "Differentiable CatBoost-style baseline: a linear forecast refined by soft oblivious-tree stages over the flattened multichannel window. Use for small-channel multivariate data as a tree-boosting baseline; not for many-channel data, categorical features, or probabilistic output."
---

# CatBoostTS

## Idea

- `soft_tree` provides differentiable oblivious trees (`SoftObliviousTree`) of fixed depth, trained by gradient descent rather than greedy splits.
- A linear `base` forecast is refined in `num_estimators` stages, each adding `learning_rate` times a tree output.
- Each stage's tree input is the flattened series minus `tanh(context(forecast / stage))`, a differentiable stand-in for ordered-boosting prior context.
- Channels and time are flattened into one feature vector; `revin` normalizes the input.

## When to use

- A tree-boosting baseline for multivariate windows where cross-channel interactions matter: every stage sees all channels and lags jointly.
- Not for many-channel data: the base, tree, and context layers scale with `(seq_len*enc_in) x (pred_len*enc_in)`.
- Not a CatBoost replacement for categorical or tabular covariates; there is no categorical-feature processing and the output is a point forecast.

## Configure

- `enc_in`: the dataset's channel count; the model checks inputs are exactly `[B, seq_len, enc_in]`.
- `max_dense_parameters` (default 2,000,000,000): cap on the dense `(seq_len*enc_in) x (pred_len*enc_in)` weights (8 GB float32 on the host, about 32 GB on the device with gradients and Adam). A larger cell raises `ValueError` before any allocation; at the default, traffic, electricity, solar, covid19 and wike2000 are refused.

Other hyperparameters: preset defaults in `configs/models/CatBoostTS.toml`; tune generically.

## Differences

- Clean-room baseline with soft oblivious trees conditioned on prior forecast context.
- Not implemented: CatBoost's permutation-based ordered boosting, ordered target statistics, categorical-feature processing, and the external library API.
- No CatBoost source code was inspected or copied.
