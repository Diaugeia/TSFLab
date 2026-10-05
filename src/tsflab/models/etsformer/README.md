---
name: "ETSformer"
description: "Exponential-smoothing Transformer: level/growth/seasonality decomposition via exponential-smoothing attention and top-k Fourier frequency attention. Use for long-term forecasting of series with clear trend and seasonality; not for calendar-driven or irregular series."
---

# ETSformer

## Idea

- `FrequencyAttention` keeps the top-k amplitude Fourier bases (excluding DC) of the residual and extrapolates them over history and horizon as seasonality.
- Multi-head exponential smoothing (MH-ESA): a linear map, the first difference, `ExponentialSmoothing` with one learnable alpha per head (`n_heads`), and an output linear map extract growth.
- Each `ETSLayer` removes seasonality and growth from the residual in turn; growth is extrapolated with multi-head damping (one learnable damping factor per head). The decoder has one growth + seasonal stack per encoder layer (`e_layers`).
- A level recurrence with learnable alpha adds the final level to the projected seasonality and growth forecasts; channels are embedded jointly by a circular convolution.

## When to use

- Designed for long-term forecasting where the series decomposes into level, growth and seasonality, giving interpretable components (classical exponential-smoothing principle).
- Seasonality is a few dominant Fourier bases, so it suits stable periodic patterns rather than shifting periods.
- Calendar covariates are not used (paper Sec. 4.1.1).

## Configure

- `enc_in`: number of channels.

Other hyperparameters: preset defaults in `configs/models/ETSformer.toml`; tune generically.

## Differences

Paper-driven local implementation; `thuml/Time-Series-Library` at `230805fe` (MIT) is reference only, nothing copied.

- Implements the exponential-smoothing recurrence, top-amplitude Fourier seasonality, residual level/growth/seasonality stacks, growth damping and additive decoder.
- `n_heads` sets the heads of the exponential smoothing and of the growth damping (`d_model` must be divisible by it).
- No `d_layers`: the paper decoder has one stack per encoder layer and the reference asserts `e_layers == d_layers`.
- No `embed` / `freq`: the paper uses only a convolutional input embedding without time covariates (Sec. 4.1.1); the reference adds a `DataEmbedding` with calendar marks (`models/ETSformer.py` L25-26, L59). Timestamp marks are accepted and not consumed.
- The first growth difference uses the first step as its predecessor; the reference prepends a learnable `z0` (`layers/ETSformer_EncDec.py` L111, L125). Smoothing is a step recurrence, not the reference's FFT convolution, and no training-time input augmentation (`Transform`) is applied.
- No published-benchmark reproduction is claimed.

Citation: Woo, G., Liu, C., Sahoo, D., Kumar, A., Hoi, S. "ETSformer: Exponential Smoothing Transformers for Time-series Forecasting." arXiv:2202.01381 (2022).
