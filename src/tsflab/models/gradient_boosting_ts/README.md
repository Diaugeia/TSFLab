---
name: "GradientBoostingTS"
description: "Differentiable boosting baseline: a linear forecast plus additive soft-tree stages trained end to end, each updating a learned residual input state. Use as a small nonlinear baseline on few-channel data with short training windows; not for many channels, graphs or covariates."
---

# GradientBoostingTS

## Idea

- A linear `base` forecast is corrected by `num_estimators` soft decision trees (`SoftDecisionTree` from `soft_tree`), each scaled by `learning_rate`.
- After each stage a learned tanh backcast subtracts the tree's explanation from the input state, mimicking boosting on residuals.
- All stages are trained jointly by gradient descent rather than fitting frozen pseudo-residuals, so it is not scikit-learn gradient boosting.
- Works on the flattened `seq_len * enc_in` window inside `revin` (disable with `use_revin = false`).

## When to use

- A compact nonlinear baseline (12 depth-3 trees by default) for short training windows where larger networks overfit, and as a tree-ensemble reference point.
- Mixes channels through the flattened window; base, trees and backcasts scale with `seq_len * enc_in` by `pred_len * enc_in`, so cost grows quickly with many channels or long windows.
- Not for graph, calendar or covariate tasks, nor for probabilistic output.

## Configure

- `enc_in`: the dataset's channel count (all layers map `seq_len * enc_in` inputs to `pred_len * enc_in` outputs).
- `max_dense_parameters` (default 2,000,000,000): cap on the dense `(seq_len*enc_in) x (pred_len*enc_in)` weights (8 GB float32 on the host, about 32 GB on the device with gradients and Adam). A larger cell raises `ValueError` before any allocation; at the default, traffic, electricity, solar, covid19 and wike2000 are refused.

Other hyperparameters: preset defaults in `configs/models/GradientBoostingTS.toml`; tune generically.

## Differences

- Clean-room baseline; Friedman (2001) supplies only the stage-wise additive principle, and no external source code was inspected or copied.
- All soft-tree stages are applied end to end and the residual lives in input space through learned backcasts; trees are not fitted to frozen loss pseudo-residuals, and scikit-learn is not reproduced.
