---
name: "SBULSTM"
description: "Stacked bidirectional and unidirectional LSTMs over the whole sensor vector; the first layer (BDLSTM-I) imputes missing inputs from its recurrent state. Use for correlated traffic-network series with missing values and short lookbacks; not for long windows, long horizons or thousands of channels."
---

# SBULSTM

## Idea

- Each step reads the full sensor vector `x_t` (width `enc_in`); all sensors share one recurrent state, so spatial correlation is learned by the gates.
- First layer: `ImputationLSTMLayer` (LSTM-I, or BDLSTM-I when bidirectional). An imputation unit infers `x~_t` from `C_{t-1}, h_{t-1}` (Eq. 10), missing entries take `x~_t` (Eq. 12), and the mask enters every gate (Eqs. 13-16).
- Bidirectional layers average their forward and backward outputs (Eq. 20); further plain LSTM/BDLSTM layers are stacked (`layers`).
- The last layer is `enc_in` wide; its last hidden state is the next-step forecast (Eq. 9). Longer horizons roll the forecast back into the window.
- Training adds `aux_loss = imputation_weight * mean |x - x~|` over observed inputs, averaged over the two directions (Eqs. 21-22).

## When to use

- Multivariate or traffic-network series with randomly or block-wise missing inputs (NaNs in `x_enc`), short lookbacks (paper: 10 steps of 5 minutes) and short horizons.
- Correlated channels whose joint state matters; no graph is needed.
- Not for long lookbacks or long horizons (per-step Python recurrence, recursive rollout), nor for very many channels (gate weights grow as `enc_in^2`).

## Configure

- `enc_in`: the channel or node count.
- `hidden_dim`: omit for the paper default (`enc_in`); scale with the channel count.
- `value_range`: `unit` only when inputs are scaled into [0, 1] (paper); TSFLab standardizes data, so the preset uses `unbounded`.

Other hyperparameters: preset defaults in `configs/models/SBULSTM.toml` (`layers`, `imputation`, `imputation_weight`).

## Differences

- Local implementation from the paper. The official repository (no license) holds only the 2018 plain LSTM/BDLSTM stacks; it and the authors' MIT Graph-Markov-Network baseline were read for layer averaging and alignment; nothing was copied. LSTM-I has no official code, so fidelity is `paper-only`.
- Only the forecast after the window is scored (Eq. 9), not every window position as in the official training loop (which leaks through the backward direction).
- Eq. 22 uses the mean, not the sum, of absolute imputation errors; `imputation_weight` (paper lambda, unreported) defaults to 1.0.
- `value_range = "unbounded"` (local) adds a linear `enc_in -> enc_in` readout and drops the sigmoid of the imputation unit, for standardized data.
- Multi-step forecasts are recursive (the paper forecasts one step); forecasts re-enter the window as observed values, and the regularizer uses only the first pass.
- Missing inputs are NaNs, not zeros; plain stacked layers use `torch.nn.LSTM` (same equations, two bias vectors).
