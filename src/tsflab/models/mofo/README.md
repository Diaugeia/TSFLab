---
name: "MoFo"
description: "One token per phase of a known period (period-structured patches), attention modulated by a learnable relaxation of circular phase distance, flatten-linear head. Use for long-term forecasting of strongly periodic series; not for aperiodic data, cross-channel interaction, or probabilistic output."
---

# MoFo

## Idea

- `pad_to_periods` (Eq. 1) prepends `X[(T mod P):P]` so the window covers whole periods, ending at the last step; `period_structured_patches` (Eq. 2) makes `P = periodic` patches, patch `i` holding the period-aligned steps `x_i, x_{i+P}, ...`.
- A linear map embeds each patch (Eq. 3); optional learned offsets are added: a per-channel bias (`bias`) and a phase-identity embedding indexed by `(phase_last - i) mod P` (`cias`), where `phase_last` comes from the calendar marks.
- `MoFoLayer`: pre-norm RMSNorm, multi-head attention across the `P` patches with `log S(Gamma; alpha, beta)` added to the scores (Eqs. 4, 9, 12; `Gamma` is the circular phase distance), then a SwiGLU feed-forward.
- A flatten-linear head maps `P * d_model` to `pred_len` (Eq. 14); channels share weights; `revin` (affine) wraps the model.

## When to use

- Long-horizon forecasting of series with a stable, known period (for example daily cycles in hourly data).
- Tight memory or time budgets: attention cost depends on `periodic`, not on `seq_len`.
- Not for aperiodic or weakly seasonal data or a mis-specified period, when cross-channel interaction carries the signal, or when quantiles are needed.

## Configure

- `enc_in`: number of input channels; must equal the dataset's channel count.
- `periodic`: the dataset's dominant period in steps (preset 24 for hourly data with a daily cycle). `seq_len` must be at least `periodic`; any remainder is padded (Eq. 1).
- `cias = 1` needs raw marks `[batch, seq_len, 6]`; the phase is the absolute step count of the last mark modulo `periodic`.

Other hyperparameters: preset defaults in `configs/models/MoFo.toml`; tune generically.

## Differences

- Independent implementation of paper Eqs. (1)-(14) with the MIT official code (`patchs/MoFo.py` at `2d14b47e`) as reference; nothing copied.
- Rows hold period-aligned steps as in the paper; the official unflatten makes continuous patches (see `[[issues]]`).
- `alpha` and `beta` are scalar logits that train; the official product parameterization stays at its initial values (see `[[issues]]`).
- Layer order follows the code (attention, then feed-forward), not Appendix C Eq. (28).
- The phase for `cias` comes from raw marks for any `periodic`; the official timeF decoding supports only 24, 96, 144, 288 and is correct only for 24. The unused official weekday embedding (`ciasW`) is omitted.
- Training loss: the official adapter trains with a frequency-domain MAE rescaled per channel by the maximum channel loss (`adapters_for_MoFo.py` L577-579; paper Eq. 15 gives the channel weighting). TSFLab provides it as the `freq_weighted_mae` loss, which a run config must select; the default loss is MSE.
