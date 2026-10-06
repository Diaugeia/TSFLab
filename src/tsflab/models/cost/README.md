---
name: "CoST"
description: "Two-stage forecaster: a dilated-conv encoder learns disentangled trend (causal AR experts, MoCo loss) and seasonal (Fourier layer, amplitude/phase contrast) features; a closed-form ridge maps the last step's features to the horizon. Use for seasonal-trend data with cheap fitting; not for probabilistic output."
---

# CoST

## Idea

- Stage I (`pretrain`, `pretraining-stage`): `dilated_conv_encoder` over values plus standardized calendar features; trend = mean of causal convolutions with kernels `kernels`; season = per-frequency complex linear layer between rfft and irfft.
- Loss `L_time + alpha / 2 (L_amp + L_phase)`: MoCo InfoNCE (momentum key encoder, queue 256, temperature 0.07) on the projected trend of one random step; two-view instance contrast on the amplitude and phase of the seasonal features. Views come from scale, shift and jitter augmentations; SGD with cosine decay.
- Stage II: ridge regression from `[trend, season]` of the last lookback step to all `pred_len` values, closed form; the penalty is chosen from 13 values by RMSE + MAE on held-out training windows.
- All parameters are frozen afterwards, so the TSFLab trainer skips gradient training; `forward` is encoder plus ridge.

## When to use

- Series with clear seasonal and trend structure; calendar marks are used as covariates.
- When cheap, deterministic fitting is wanted: a few hundred contrastive iterations and one linear solve.
- Not for probabilistic output or for horizons whose label width `pred_len x channels` makes the `repr_dims x labels` normal equations too large.

## Configure

- `enc_in`: the channel count.
- `kernels`: AR expert kernel sizes; the paper uses `2^i` up to `seq_len / 2`.
- `channel_independent`: `true` encodes each channel separately (official Electricity setting).
- `training.loss` and the trainer optimizer are not used (no trainable parameters remain after `pretrain`).

Other hyperparameters: preset defaults in `configs/models/CoST.toml`; tune generically.

## Differences

Independent implementation; the encoder (trend and season outputs) and the seasonal contrastive loss were compared with salesforce/CoST `7bb5271c` under shared weights (max abs difference 0).

- Pre-training samples TSFLab training windows of length `seq_len` (official: random crops of a 201-step `max_train_length`); iterations follow the official 200/600 rule on the training-split size.
- The ridge penalty is selected on the last 25 % of training windows, because the pre-training stage has no access to the validation split (official: the validation split).
- Representations come from the `seq_len` lookback (official: a 201-step causal window per step).
- Calendar standardization uses training-window statistics; contrastive labels are device-agnostic.
