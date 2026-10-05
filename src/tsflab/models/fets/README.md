---
name: "FeTS"
description: "Patch tokens scored by a Fourier-plus-polynomial basis into a binary feature mask that gates local feature aggregation, then local-conv and global fusion. Use for channel-independent long-term forecasting that benefits from selecting informative feature dimensions; not for cross-channel modelling."
---

# FeTS

## Idea

- `FourierPolyMask` scores each patch-token dimension with a cosine, sine and polynomial basis and thresholds at the mean to get a binary mask; the forward mask is exact and a sigmoid straight-through estimator carries gradients.
- `adaptive_features` (AdaFE) uses the mask to gate a learned local aggregation kernel over neighbouring feature dimensions, added residually to the patch tokens.
- A local `Conv1d` branch and a global mean branch (DSFFN) are concatenated and fused, then flattened and projected linearly to the horizon.
- PatchTST patching (end replication padding, as in the official code) with channels folded into the batch; `revin` wraps the model.

## When to use

- Designed to amplify informative feature dimensions and suppress irrelevant ones in long-term forecasting, with a compact model.
- Channel-independent with shared weights: no cross-channel interaction.

## Configure

- `enc_in`: number of channels.
- `patch_len`, `stride` (clamped to `seq_len` and `patch_len`): the input is end-padded by replication (by `stride` when patches overlap, else by the remainder), so the most recent steps are always covered.

Other hyperparameters: preset defaults in `configs/models/FeTS.toml`; tune generically.

## Differences

Written for TSFLab from Eqs. (2)-(14) of the AAAI paper; `models/FeTS.py` of `lllucky111/FeTS` at `d908e434` (no license file, recorded `NOASSERTION`) was inspected as reference only, nothing copied.

- The paper's binary threshold mask is non-differentiable and the gradient path is unstated; here the forward mask stays exact with a sigmoid straight-through gradient in training.
- Mask threshold follows paper Eqs. (7)-(8) (per-row mean, `>=`); the official code uses one batch-wide mean with `>`.
- DSFFN differs from both sources: a k=3 local conv, a 1x1 fusion conv, and GELU + dropout in place of the Eq. (14) output conv; tokens are added to the AdaFE output before LayerNorm, while the official code adds the embedding after DSFFN. Default AdaFE kernel is 3 (official 5).
- One compact AdaFE/DSFFN block; the paper's dataset-specific training schedule and hyperparameter sweep are not reproduced.

Citation: Wang, L., Chen, J., Liu, S. "FeTS: A Feature-Aware Framework for Time Series Forecasting." AAAI 2026, pp. 26328-26336. doi:10.1609/aaai.v40i31.39838.
