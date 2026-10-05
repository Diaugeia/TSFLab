---
name: "MICN"
description: "Multi-scale isometric convolution network: multi-kernel decomposition, linear trend regression, and per-scale downsampling plus full-length isometric convolution over the zero-padded seasonal part. Use for long-term forecasting with trend and local fluctuations at modest compute; not for calendar-driven targets."
---

# MICN

## Idea

- `MultiScaleDecomposition` averages several moving-average decompositions (`series_decomposition`) into one seasonal and one trend part.
- Trend: `mode = "regre"` maps the trend with a linear layer over time (weights start at `1 / pred_len`); `mode = "mean"` repeats the window mean.
- Seasonal: the seasonal part is extended with `pred_len` zeros and embedded (`embed` token convolution plus sinusoidal positions).
- Each `MICLayer` runs one `LocalGlobalBranch` per scale `s`: decompose again, downsample with a stride-`s` convolution (local features), apply an isometric convolution (left-pad the `S` downsampled steps with `S - 1` zeros, kernel `S`; global correlations), upsample with a transposed convolution. A `(scales, 1)` Conv2d merges the branches; a feed-forward with residual follows.
- A linear projection maps to channels; the last `pred_len` steps plus the trend forecast give the output.

## When to use

- Long-term forecasting where both local fluctuations and global trends matter, as a CNN alternative to quadratic attention.
- The explicit trend branch helps on trending series.
- Channels are mixed by the input embedding, so it is a weaker fit for many weakly correlated channels.
- No instance normalization and no calendar features.

## Configure

- `enc_in`: number of input channels.
- `conv_kernel`: distinct integers >= 2, smaller than `seq_len`; each is a moving-average window (even values +1) and a downsampling stride. The isometric kernel of each scale is derived from `seq_len + pred_len`.
- `mode`: `regre` (paper default) or `mean`.

Other hyperparameters: preset defaults in `configs/models/MICN.toml`; tune generically.

## Differences

- Clean-room implementation; the unlicensed repository (`models/model.py`, `models/local_global.py`, `run.py` at `370c69b8`) is reference-only, nothing copied.
- The reference adds a calendar embedding of `x_mark_dec`; it is omitted here, so the embedding is the token convolution plus positions only.
- The shared `embed` token convolution has no bias; the reference convolution has one.
- The MIC-layer dropout is fixed at 0.05 and `dropout` sets only the embedding dropout, as in the reference (see `[[issues]]`).
- `c_out` must equal `enc_in`.
