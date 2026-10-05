---
name: "DistDF"
description: "Training objective that aligns joint history-forecast and history-label Gaussian moments (Bures-Wasserstein) mixed with MSE, on a compact shared linear forecaster. Use when labels are strongly autocorrelated and MSE training is biased; not for changing model capacity or probabilistic output."
---

# DistDF

## Idea

- Changes the loss, not the architecture: the forecaster is a shared `channel_wise_linear` map wrapped by `revin`.
- `joint_distribution_discrepancy` forms joint samples `[history, target]` and `[history, forecast]` over batch-channel pairs, estimates Gaussian mean and covariance (with `covariance_eps` jitter), and computes the squared Bures-Wasserstein distance (Algorithm 1, Eqs. 5-6).
- `training_objective` uses `gamma * discrepancy + (1 - gamma) * MSE`; plain `forward` is point forecasting.

## When to use

- Series with strong autocorrelation across the horizon (structured, forecastable signals), where per-step MSE is a biased training target.
- Cheap carrier: one shared linear map; channels are forecast independently.
- Point output only; the Gaussian moments are a training device, not a predictive distribution.

## Configure

- `enc_in`: the dataset's channel count.

Other hyperparameters: preset defaults in `configs/models/DistDF.toml`; tune generically (`gamma`, `covariance_eps`).

## Differences

- Clean-room implementation of Algorithm 1 and Eqs. (5)-(6); `utils/fft_ot.py` of the pinned official code was read for the square root, nothing was copied.
- The released scripts train with a per-channel trace upper bound of the Bures term; this entry keeps the paper's exact term (see the card issues).
- Square roots run in float64 (official `sqrtm_svd_stable`); an `eigh` that does not converge is retried with a small trace-scaled diagonal jitter, then replaced by the SVD.
- The paper applies DistDF to several external backbones; this entry supplies a compact shared linear carrier.
- Batch-channel pairs form the empirical samples, and positive jitter stabilizes small covariances.
