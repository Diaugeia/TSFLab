---
name: "Basisformer"
description: "Timestamp-generated learnable basis; cross-attention coefficients weight its future part; InfoNCE plus smoothness. Use for forecasting where calendar time determines recurring shapes and interpretable bases help; not for data without timestamps or probabilistic output."
---

# Basisformer

## Idea

- A four-layer bottleneck MLP maps the normalized timestamp of the first history step to `N` basis vectors over `seq_len + pred_len` steps, each of unit norm over time.
- `CoefModule` stacks bidirectional cross-attention blocks (series attend to basis and basis to series) and returns per-head coefficients `c [B, H, C, N]`.
- `forecast_from` projects the future basis into `H` heads, weights it by the coefficients (Eq. 5), fuses the heads with a bottleneck MLP, and undoes the per-window standardization.
- The training objective adds an InfoNCE alignment between historical and future-view coefficients (Eq. 6) and a basis curvature penalty (Eqs. 7-8).

## When to use

- Series whose shapes recur with calendar time (daily, weekly, seasonal), so a basis conditioned on the timestamp can be reused across windows; bases are inspectable.
- Needs raw calendar marks; without them every window shares one basis.
- Channels share the basis and are coupled through cross-attention; point output only.

## Configure

- `enc_in`: number of channels.
- `heads`: `pred_len >= heads` and `d_model >= heads`.
- `bottleneck`: `pred_len >= bottleneck`.
- `timestamp_origin_year`, `timestamp_span_years`: origin and unit of `tau`; cover the dataset's calendar range (defaults 2010 and 10 years).

Other hyperparameters: preset defaults in `configs/models/Basisformer.toml`; tune generically.

## Differences

Independent rewrite after inspecting the pinned official code (no license file); nothing copied.

- `tau` is the calendar time of the first history step from a fixed origin, shared across splits; the official loader uses the window index per split.
- Smoothness uses the official mean absolute second difference, not Eq. (7)'s squared norm.
- `c_y` uses all channels also for `MS`; the official `MS` mixing layers are not implemented.
- Unused official `MLP_x`/`MLP_sx` omitted; the modern `weight_norm` parametrization is used.
- The timestamp MLP's two one-input layers are plain linear maps: weight-normalized, their direction gets no gradient and underflows to NaN under weight decay.

Full detail in `reference.md`.
