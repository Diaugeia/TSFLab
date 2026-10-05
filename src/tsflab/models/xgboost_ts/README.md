---
name: "XGBoostTS"
description: "Gradient-trained ensemble of soft decision trees over the flattened lookback of all channels, with column masks, shrinkage, and backcast-corrected inputs, inspired by XGBoost. Use as a cheap tree-style baseline on few-channel data; not as a real XGBoost reproduction or for hundreds of channels."
---

# XGBoostTS

## Idea

- Each of `num_estimators` `SoftDecisionTree`s (`soft_tree`) sees the flattened lookback of all channels under its own fixed random column mask (`column_fraction`).
- The forecast is a linear base plus `learning_rate`-shrunken tree corrections; between trees the input is reduced by a tanh of a learned backcast of the correction, a differentiable stand-in for residual fitting.
- `aux_loss` adds L1/L2 penalties on leaf values; `revin` wraps the model. Trained end to end by gradient descent, not by boosting.

## When to use

- A tree-flavoured baseline that compares against neural models in the same training loop.
- Every tree reads all channels at all lookback steps, so it can use cross-channel information; the input grows as `seq_len * enc_in`.
- Small and fast with the default 16 depth-3 trees.
- Point output only.

## Configure

- `enc_in`: must equal the dataset channel count (the flattened input is `seq_len * enc_in`).
- `max_dense_parameters` (default 2,000,000,000): cap on the dense `(seq_len*enc_in) x (pred_len*enc_in)` weights (8 GB float32 on the host, about 32 GB on the device with gradients and Adam). A larger cell raises `ValueError` before any allocation; at the default, traffic, electricity, solar, covid19 and wike2000 are refused.

Other hyperparameters: preset defaults in `configs/models/XGBoostTS.toml`; tune generically.

## Differences

- Borrows only additive trees, shrinkage, feature subsampling, and regularization as high-level ideas; no official forecasting codebase exists.
- Not implemented: XGBoost's second-order objective, sparsity-aware hard split search, weighted quantile sketch, systems optimizations, or library API.
- No XGBoost source was inspected or copied.
