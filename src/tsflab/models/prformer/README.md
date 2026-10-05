---
name: "PRformer"
description: "Pyramidal recurrent embedding (strided convolutions over chains of period lengths, one GRU per scale) turns each variate into a token for a Transformer over variates. Use for long-lookback multivariate forecasting with several known periods and correlated channels; not for aperiodic or univariate data."
---

# PRformer

## Idea

- `pyramid_chains` groups the period lengths (`conv_windows`, e.g. 24, 48, 72, 144 hours for ETTh1) into multiplicative chains; every scale gets `d_model / #windows` GRU units.
- `PyramidChain.bottom_up` (Eq. 2): strided 1D convolutions with kernel = stride = the ratio of consecutive periods (the first level uses the first period); `top_down` (Eq. 3) upsamples level by level (nearest neighbour) and adds the bottom-up feature of the same scale.
- `PyramidalRNNEmbedding` (Eq. 4): one GRU per scale, last hidden states mixed by `softmax(alpha / T)`, concatenated and projected to one `d_model` token per variate; no positional encoding.
- `Model` (Eqs. 1, 5): RevIN, PRE tokens, post-norm Transformer encoder (`transformer_encdec`, `self_attention_family`) over variate tokens, linear projection to the horizon; trained with MAE.

## When to use

- Multivariate data with several nested periods (for example daily and multi-day cycles in hourly data): the pyramid aggregates the lookback at each period scale.
- Long lookbacks (the preset uses `seq_len = 720`): the RNN embedding compresses long histories into one token per variate.
- Correlated channels: attention runs across variate tokens. Not for univariate or aperiodic series, where the period pyramid has nothing to exploit.

## Configure

- `enc_in` follows the channel count: must equal the number of input channels.
- `conv_windows` follows the dataset periods: strictly increasing period lengths in steps; ratios of consecutive periods form the convolution strides, and `d_model` must be large enough for the number of scales. Periods longer than `seq_len` are dropped (preset at `seq_len` 96: 24, 48, 72; at 36: 24); at least one must fit.

Other hyperparameters: preset defaults in `configs/models/PRformer.toml`; tune generically.

## Differences

Independent rewrite of Eqs. 1-5 after reading the pinned official code (`usualheart/PRformer`, `df718ad`, Apache-2.0); nothing copied.

- Scale weights follow the paper (`softmax(alpha / T)` across scales); in the official code the softmax runs over a singleton axis, so every weight is 1 and `alpha` never trains.
- Attention is unmasked per Eq. (5); the official encoder applies a causal mask over variate order (`causal_variate_mask = true` reproduces it).
- Periods longer than `seq_len` are dropped; the official scripts use `seq_len` 660-720, where every period fits.
- Calendar marks are not added as extra tokens; upsampled maps are padded to any length mismatch (official: one step only); RevIN uses `sqrt(var + eps)`.
