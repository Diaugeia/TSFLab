---
name: "Koopa"
description: "Koopman forecaster: a Fourier filter splits time-invariant and time-variant dynamics, advanced by a global learned operator and a window-estimated local operator in stacked residual blocks. Use for non-stationary series under tight compute; not for weakly correlated many-channel data."
---

# Koopa

## Idea

- `FourierDynamicsSplit` keeps the top `alpha` fraction of rFFT bins by batch-average energy as the time-invariant part; the remainder is time-variant.
- `GlobalKoopmanPredictor` advances invariant latents with one learned shared transition; `LocalKoopmanPredictor` estimates a least-squares DMD operator (`estimate_operator`, `torch.linalg.lstsq` as in the official `KPLayer`) from the latest `seg_len` latent states.
- `MeasurementFunction` encodes the channel vector at each step to a latent and decodes after rolling the operator forward `pred_len` steps.
- `KoopmanBlock`s are stacked residually: each subtracts its reconstructions and adds its forecast contribution.

## When to use

- Non-stationary series whose dynamics drift: the local operator is re-estimated from each window.
- Tight training budgets: the paper reports large training-time and memory savings over Transformer forecasters.
- The measurement function mixes all channels at each step, so it assumes channels share dynamics.
- Dominant-frequency selection is per batch, so it works best when periodic components are stable.

## Configure

- `enc_in`: number of data channels (measurement-function input width).
- `seg_len`: latest latent states used for the local operator; defaults to `pred_len`, at least 2, clipped to the available states.

Other hyperparameters: preset defaults in `configs/models/Koopa.toml`; tune generically.

## Differences

- Clean-room implementation; reference-only source was not copied.
- Mapping: Fourier Filter -> `FourierDynamicsSplit`; measurement function -> `MeasurementFunction`; time-variant operator -> `LocalKoopmanPredictor`; invariant operator -> `GlobalKoopmanPredictor`; residual hierarchy -> `KoopmanBlock`.
- Dominant modes are chosen per batch, not from dataset-global masks.
- A non-finite local operator falls back to the identity for the whole batch, as upstream; the check also catches `inf`, upstream only `nan`.
- Rolling adaptation with incoming ground truth and numerical reference comparison are out of scope.
