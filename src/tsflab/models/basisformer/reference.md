# Basisformer — reference

## Inputs and marks

`forward(x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None)` takes a history
`[batch, seq_len, enc_in]` and the raw six-column calendar marks
`[batch, seq_len, 6]` (`year, month, day, weekday, hour, minute`); only the
first history step's timestamp is read and `x_dec`/`x_mark_dec` are ignored.
Without marks the timestamp is 0, so every window shares one learned basis.

## Paper-to-code map

- Sec. 3.3 basis: `BottleneckMLP(1, N * (I + O), map_bottleneck)` on the
  timestamp, reshaped to `[B, I + O, N]` and L2-normalized over time (`Model.basis`).
- Sec. 3.1, Eqs. (1)-(4): `CrossAttentionBlock` (post-norm attention and a
  `4 * d_model` ReLU feed-forward) and `BidirectionalCrossAttentionBlock`, where
  both directions read the previous layer's representations with separate
  weights; `CoefModule` ends with per-head projections and scaled dot products.
- Sec. 3.2, Eq. (5): `forecast_from` (head split of the projected future basis,
  coefficient-weighted sum, head-fusion bottleneck MLP, inverse standardization
  with the per-window mean and unbiased standard deviation).
- Eqs. (6)-(9): `alignment_loss`, `smoothness_loss`, `training_terms` and the
  spec's `training_objective` (`criterion + loss_weight_infonce * L_align +
  loss_weight_smooth * L_smooth`, both weights 1 by default as in the paper).
  Validation and test use plain `forward`.
- Every linear map with more than one input feature is weight-normalized, as in the official code; the timestamp MLP's two one-input layers are plain (see Differences).
- Constraints: `pred_len >= heads` (the projected future basis is split into
  `heads` chunks of `pred_len // heads`), `pred_len >= bottleneck`,
  `d_model >= heads`. The official scripts train with learning rate `5e-4`,
  batch size 32 and patience 3; the preset keeps the paper's architecture defaults.

## Differences in detail

Inspected at `f2f647ec815baa6338111df0545002cc56eab1f0`: `model.py`, `utils.py`, `main.py`, `data_provider/data_loader.py`, `script/M.sh`.

- Timestamp input: the paper defines `tau = t / T` over the whole series; the official loader divides the window index by the number of windows in the current split. TSFLab passes calendar marks, so `tau` is the elapsed time of the first history step since January 1 of `timestamp_origin_year`, in units of `timestamp_span_years`. The basis MLP's first layer is affine in `tau`, so an affine rescaling spans the same function class; only the numeric range differs. Equal calendar times share a basis across splits.
- Smoothness: Eq. (7) writes a squared norm; the official trainer uses the mean absolute value of the second differences, which `smoothness_loss` follows.
- `c_y` is computed from the full future window (all channels), also for `MS` runs, where the forecast loss uses the trailing target channel; the official `MS` channel-mixing layers are not implemented.
- `MLP_x` and `MLP_sx`, constructed but unused in the official model, are omitted.
- One-input layers: in the official `map_MLP`, `linear1[0]` and `skip` take the scalar timestamp, so weight normalization gives `w = g * sign(v)` and `v` gets no loss gradient. Under weight decay (TSFLab protocol: Adam, weight decay `1e-4`; official: AdaBelief, none) `v` shrinks until `|v|` underflows and the weight is NaN. These two layers are plain `nn.Linear` maps, the same function class with the same initialization.
- `torch.nn.utils.parametrizations.weight_norm` replaces the deprecated `torch.nn.utils.weight_norm` (same reparameterization).
- Checked: timestamp mapping, unit-norm basis, bidirectional block wiring, coefficient inner products, Eq. (5) against an explicit loop, Eq. (6) against an explicit InfoNCE sum, the second-difference operator, and the weighted objective.

## Citation

```bibtex
@inproceedings{ni2023basisformer,
  title     = {BasisFormer: Attention-based Time Series Forecasting with Learnable and Interpretable Basis},
  author    = {Ni, Zelin and Yu, Hang and Liu, Shizhan and Li, Jianguo and Lin, Weiyao},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2023}
}
```
