---
name: "TFT"
description: "Temporal Fusion Transformer: static-context variable selection, LSTM encoder-decoder, static enrichment and causal interpretable attention give multi-horizon quantiles. Use for quantile forecasts with calendar effects and inspectable importances; not for point-only tasks or many channels on tight memory."
---

# TFT

## Idea

- Inputs are embedded per variable (Sec. 4.2): the target (observed input) by a linear map, five known calendar fields of the raw marks (month, day, weekday, hour, minute) and the entity identifier (static input) by embeddings.
- A static variable-selection network and four GRNs give context vectors `c_s, c_e, c_h, c_c` (Sec. 4.3) from `gated_residual_network`.
- Separate past and future `VariableSelectionNetwork`s, conditioned on `c_s`, weight the inputs per step (Eq. 6-8).
- An LSTM encoder (initial state `c_h, c_c`) and decoder give local features with a gated skip (Eq. 17); a GRN with `c_e` enriches them (Eq. 18).
- `interpretable_attention` (shared values, causal mask) mixes all positions (Eq. 19-20); a position-wise GRN and a gated skip over the block follow (Eq. 21-22).
- One linear map per quantile on the horizon positions (Eq. 23); `output_type = "quantile"`, `(B, pred_len, C, Q)`, trained with the pinball (`quantile`) loss. `fuse()` also returns the static, past and future selection weights and the attention maps.

## When to use

- Multi-horizon quantile forecasts where calendar effects and per-series identity carry signal (the paper's electricity, traffic, retail and volatility setups).
- When variable importance and temporal attention patterns should be inspected.
- `MS` mode: the other channels become observed inputs of the single target series.
- Not when only point forecasts are needed, for long horizons or many channels in M mode on tight memory (every channel is a separate entity with `(5 + 1)` embedded inputs per step), or when real-valued exogenous future inputs are needed (only calendar marks are known inputs).

## Configure

- `enc_in`: the dataset's channel count. In `M`/`S` mode each channel is one entity with its own static embedding; in `MS` mode all channels are observed inputs and the last one is the target.
- Marks must use the raw layout `[year, month, day, weekday, hour, minute]`; year is not used.
- `quantile_levels` defaults to `evaluation.quantile_levels` (the paper uses 0.1, 0.5, 0.9). Pair with `[training] loss = "quantile"`.

Other hyperparameters: preset defaults in `configs/models/TFT.toml` (the paper's electricity values: `d_model = 160`, 4 heads, dropout 0.1); tune generically. The paper also clips gradients (max norm 0.01 for electricity) and uses Adam at 1e-3.

## Differences

- Compared with paper Sec. 4-5 and the official TensorFlow code (google-research `tft/libs/tft_model.py` at `e49bbfe3`). Layer order, contexts, LSTM state wiring, shared-value attention, causal mask, dropout positions and Keras initialization follow the code; see `[[issues]]` for the seven paper and code problems found.
- Known inputs are the calendar fields of TSFLab marks as categorical embeddings; the official formatters also use dataset-specific known reals (for example `hours_from_start`), holidays and promotions, and observed covariates, which the standard forecaster call does not carry.
- The static input is the channel index (TSFLab has no static metadata); the official Electricity/Traffic setups use the same identifier as their only static input.
- Normalization is the dataset's per-channel train-split z-score, which equals the official per-entity `StandardScaler` for Electricity/Traffic. No RevIN.
- Quantiles are unconstrained linear outputs as in Eq. 23; they can cross.
