---
name: "ddpm_epsilon"
description: "Fixed-schedule Gaussian DDPM around a caller-owned noise predictor: closed-form noising, epsilon-MSE loss with uniform steps, and ancestral sampling with posterior variance. Use as the emission head of sample-based probabilistic forecasters; not for learned variances, fast samplers, or non-Gaussian noise."
---

# ddpm_epsilon

## What it does

`GaussianDDPM(steps=100, beta_start=1e-4, beta_end=0.1, schedule="linear")` builds
`beta_n` (`linear`, or `quad` = linear in `sqrt(beta)`), `alpha_n = 1 - beta_n`,
`abar_n = prod alpha`, and stores float32 tables (non-persistent buffers).

- `q_sample(x0, n, noise) = sqrt(abar_n) x0 + sqrt(1 - abar_n) noise`.
- `loss(denoiser, x0)`: one `n ~ U{0..steps-1}` per leading index, `eps ~ N(0, I)`,
  returns `mean((denoiser(q_sample(x0, n, eps), n) - eps)^2)`.
- `p_sample(denoiser, x, step)`: `x0_hat = x / sqrt(abar) - sqrt(1/abar - 1) eps`,
  posterior mean `c0 x0_hat + cn x`, plus `sqrt(beta_tilde) z` except at `step == 0`.
  Equal to `(x - beta / sqrt(1 - abar) eps) / sqrt(alpha) + sqrt(beta_tilde) z`.
- `sample(denoiser, shape)`: white noise through steps `steps-1 .. 0` under `no_grad`.

## When to use

Use when a forecaster emits a sample (one time step, a window, or a variate
vector) from a conditional diffusion whose network predicts the added noise; wrap
conditioning in the `denoiser` closure. Do not use for learned variances, DDIM or
other fast samplers, x0- or v-prediction, clipping of `x0_hat`, or schedules other
than linear/quad.

## Interface

- `GaussianDDPM(steps: int, beta_start: float, beta_end: float, schedule: "linear" | "quad")`;
  `beta_schedule(steps, beta_start, beta_end, kind) -> float64 [steps]`.
  `steps < 1`, not `0 < beta_start <= beta_end < 1`, or another schedule raise `ValueError`.
- `denoiser(x_n, n) -> eps` with the shape of `x_n`; `n` is a `[S]` long tensor
  (steps `0 .. steps-1`, paper `1 .. N`). Coefficients broadcast over trailing axes.
- No parameters; `steps` is a plain attribute; tables follow the module device.
- Checked equal (max abs difference 0 for coefficients and one reverse step under a
  shared seed) to `pts/modules/gaussian_diffusion.py` at pytorch-ts `7860c969`.
