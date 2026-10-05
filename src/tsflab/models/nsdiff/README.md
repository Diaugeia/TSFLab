---
name: "NsDiff"
description: "Diffusion model whose forward chain ends at a learned N(mean, variance) prior with an uncertainty-aware noise schedule; outputs sample quantiles. Use for probabilistic forecasting of non-stationary series with time-varying variance; not for cheap point forecasts or univariate-only training budgets."
---

# NsDiff

## Idea

- `StationaryPriorMean` is the Non-stationary Transformer prior `f(X)` (stationarized inputs, de-stationary attention with learned `tau`/`delta`, de-normalized output) on the shared `embed`, `self_attention_family`, `transformer_encdec` and `masking` components.
- `SlidingVarianceEstimator` is `g(X)`: an MLP from trailing-window input variances to softplus future variances; `target_variance` gives its sliding-window target `sigma_Y0` (Eq. 14).
- Uncertainty-aware forward process: per-step variance `beta_t^2 g + alpha_t beta_t sigma_Y0` (Eq. 6), closed form `(beta_bar_t - beta_tilde_t) g + beta_tilde_t sigma_Y0` (Eq. 7-8), endpoint `N(f(X), g(X))`.
- `NoiseVarianceDenoiser` predicts noise and reverse variance; training is the Eq. 13 KL plus MSE for `f` and a square-root-variance regression for `g`, end to end via the `training_objective` hook.
- Reverse steps follow Algorithm 2 (solve Eq. 15-18 for `sigma_Y0`, posterior Eq. 9-12); `forward` returns empirical quantiles of `num_samples` trajectories.

## When to use

- Probabilistic forecasting where both the mean and the uncertainty shift over time: the prior models a time-varying mean and variance, and the schedule adapts noise to them.
- Non-stationary multivariate series (the prior mean uses stationarization plus de-stationary attention).
- Not for cheap point forecasting: sampling costs `diffusion_steps` denoiser calls per trajectory times `num_samples`. Not when `seq_len` is too short for a sliding variance window.

## Configure

- `enc_in`: number of input channels; must equal the dataset's channel count (the diffusion runs over all channels; the runner selects the `MS` target).
- `rolling_length`: sliding-variance window; follows `seq_len` and must satisfy `0 < rolling_length < seq_len`. Official scripts use `seq_len = 168` with 96 (ETTh1, ETTh2, the default) or 24 (ETTm1, ETTm2, ExchangeRate); the preset uses 24, the only official value below the benchmark lookbacks 36 and 96.

Other hyperparameters: preset defaults in `configs/models/NsDiff.toml`; tune generically.

## Differences

- Independent rewrite from the paper (arXiv v3) and the unlicensed official repository (`NOASSERTION`); nothing copied.
- Output `[B, pred_len, enc_in, K]` empirical quantiles at `evaluation.quantile_levels`; point metrics use the median. `x_dec` is ignored.
- Training follows the official joint objective (`load_pretrain=False`), not the paper's separate pre-training of `f` and `g`; validation uses pinball loss on sampled quantiles instead of CRPS.
- Schedule endpoint `beta_end = 0.01` (official default; paper text says 0.02). `gamma_2` keeps the paper/official Eq. 12 form, which differs from exact Gaussian conditioning.
- The Eq. 15-18 quadratic is guarded (discriminant floored at `1e-20`, root at zero); the official code has none.
- `g` reads trailing-window variances as the official code does, not the raw window of App. C.2.2. The unused official `DataEmbedding` and VAE heads are omitted.
- No executable official reference comparison (external package, unlicensed repository). Full detail in reference.md.
