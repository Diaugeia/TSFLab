---
name: "TimeGrad"
description: "Autoregressive multivariate diffusion forecaster: an RNN over lags and calendar features conditions a DDPM that samples each step's vector of all channels; sample paths give quantiles. Use for joint probabilistic forecasts of correlated channels at short horizons; not for long horizons or tight budgets."
---

# TimeGrad

## Idea

- An LSTM (or GRU) reads, per step, the lags `x_{t-l} / scale` of all `D` channels, a learned variate embedding and Fourier calendar features; its state `h_{t-1}` is projected to a conditioning vector (Eq. 9).
- `ddpm_epsilon` models `p(x_t | h_{t-1})` jointly over the `D` channels; the noise predictor (Fig. 2) treats the variate axis as a circular 1-D signal: 8 gated residual blocks with dilations 1, 2, a sinusoidal step embedding and an upsampled condition, skip outputs summed.
- Training (`training_objective`, Algorithm 1): teacher-forced epsilon MSE with one uniform diffusion step per time step.
- Inference (Algorithm 2): `num_samples` paths are drawn step by step, each sample fed back as the next lag; `empirical_quantiles` returns the quantile grid (output type `quantile`).

## When to use

- Probabilistic forecasts where channels are correlated and the joint distribution matters (sums, coverage across series).
- Short horizons with seasonal lags inside the lookback.
- Not for long horizons or large `D` under a tight budget: inference costs `pred_len x diff_steps` denoiser passes per sample batch, and validation also samples.

## Configure

- `enc_in`: the channel count.
- `lags`: 1 plus the dominant periods; `max(lags) + context_length <= seq_len` (the preset uses `[1, 24]`).
- `time_features`: calendar fields that cycle at the data frequency.
- `context_length`: RNN unroll before the horizon; 0 = `seq_len - max(lags)`.
- Train with `training.loss = "quantile"` (validation and test score sampled quantiles).

Other hyperparameters: preset defaults in `configs/models/TimeGrad.toml`; tune generically.

## Differences

Independent implementation; the denoiser and one reverse diffusion step were compared with pytorch-ts `7860c969` under shared weights and seeds (max abs difference 0).

- TSFLab windows: lags are read inside `seq_len` (official history is `context + max(lag)` beyond the context) and the preset lags are `[1, 24]`, not `[1, 24, 168]`; training windows come from the TSFLab loader, not the official instance sampler.
- Loss over context and horizon follows the code (`context_loss`); the paper sums only the horizon.
- Calendar features use fixed periods (the official per-index period is a bug); no missing-value mask.
- Univariate inputs work (official code fails for one series).
