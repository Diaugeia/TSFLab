# NsDiff — reference

## Differences in detail

Checked at the pinned revision: `src/models/NsDiff.py`, `src/layer/nsdiff_utils.py`, `src/layer/denoise.py`, `src/layer/mu_backbone.py`, `src/layer/g_backbone.py`, `src/utils/sigma.py`, `src/experiments/NsDiff.py`, `configs/nsdiff.yml`, `scripts/NSDiff/*.sh`, and the `torch_timeseries==0.1.10` layers it imports. Inputs are `[B, seq_len, enc_in]` with raw six-column marks; the decoder value input `x_dec` is ignored (the prior builds its own decoder input from the normalized history). The output is `[B, pred_len, enc_in, K]`, the linear-interpolated empirical quantiles of `num_samples` reverse-diffusion samples at the configured `evaluation.quantile_levels` (or `quantile_levels`); the evaluator's point metrics use the median.

- **Training.** The official default (`load_pretrain=False`, used by the released scripts) trains `f`, `g` and the denoiser jointly with one optimizer on the KL term of Eq. 13 plus `MSE(f, Y)` and `MSE(sqrt(g), sqrt(sigma_Y0))`; `training_objective` does this, replacing the configured loss during training. The paper's separate pre-training of `f` and `g` (Algorithm 1, Sec. 4.3; App. B.2 reports end-to-end training as comparable) is not provided. Timesteps are drawn antithetically and variances get the official `1e-7` floor during training. Validation and early stopping use the configured `quantile` (pinball) criterion on sampled quantiles instead of the official CRPS on samples; the official runs use 10 epochs, Adam at `1e-3` and batch 32, while the repository's standard trainer configuration applies here.
- **Schedule.** Linear `beta` from `1e-4` to `beta_end`, `T = 20`. The paper text states `beta_T = 0.02`; the official runnable default (`NsDiffParameters.beta_end`) is `0.01`, which the preset uses. Coefficients are computed in float64 by recurrences equivalent to Eq. 8 (official: float32 sums); `t - 1` quantities use the official value `1` at the first step.
- **Posterior.** `gamma_2` follows Eq. 12 and the official code (`sqrt(alpha_t)(alpha_t - 1)`), which differs from exact Gaussian conditioning on Eq. 6 (`sqrt(alpha_t)(sqrt(alpha_t) - 1)`); the paper form is kept. The quadratic's discriminant is floored at `1e-20` and its root at zero (identity whenever Eq. 17 holds; the official code has no guard).
- **Variance prior `g`.** As in the official code (and unlike the `Linear(seq_len, ...)` of App. C.2.2), the MLP input is the `seq_len - rolling_length` trailing-window variances of the history after dropping the first window; hidden size 512, softplus output. `rolling_length` must be smaller than `seq_len`. The official scripts (`scripts/NSDiff/*.sh`) all use `windows = 168` with `rolling_length` 96 (the `NsDiff.py` default, ETTh1/ETTh2) or 24 (ETTm1, ETTm2, ExchangeRate). The preset uses 24: 96 is invalid at the benchmark lookbacks 96 and 36, and 24 leaves 72 (or 12) variance inputs for `g`.
- **Prior mean `f`.** Non-stationary Transformer as in the official `mu_backbone.Model` (d_model 512, 8 heads, 2 encoder / 1 decoder layers, d_ff 1024, dropout 0.05, GELU, projector hidden `[64, 64]`). The decoder label window is `task.label_len` (official: `seq_len // 2`; use that value). The shared `embed` component uses TSFLab's raw six-column `timeF` marks and bias-free token and time-feature projections, while `torch_timeseries` uses frequency-dependent normalized time features with biased projections. The unused VAE heads of the official backbone are omitted.
- **Denoiser.** Three step-scaled linear layers (width 128, per-step scales uniform-initialised, `T + 1` rows) with softplus over the concatenated `[Y_t, f(X), g(X)]`; the noise head is linear and the variance head is `softplus(Linear(softplus(h)))`, as officially. The official model also computes a `DataEmbedding` of `(X, marks)` whose output the denoiser never uses; it is omitted (identical outputs, fewer parameters).
- **Sampling.** Algorithm 2 with `T` denoiser calls per trajectory, starting from `N(f(X), g(X))`; trajectories are drawn in chunks of `sample_batch_size` (the official code draws one at a time). The official evaluator slices the target channel after sampling for `MS`; here the diffusion is multivariate over all `enc_in` channels and the runner selects the target channel.
- **Checks.** Eq. 6-8 composition, Eq. 9-12 posterior, Eq. 15-18 inversion, the App. C.2.1 variance target, de-stationary attention, the Algorithm 2 final step, quantile output and the joint objective were checked against the paper. There is no executable official reference comparison (the official code depends on an external package and an unlicensed repository).

## Citation

```bibtex
@inproceedings{ye2025nsdiff,
  title     = {Non-stationary Diffusion For Probabilistic Time Series Forecasting},
  author    = {Weiwei Ye and Zhuopeng Xu and Ning Gui},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  year      = {2025},
  url       = {https://arxiv.org/abs/2505.04278}
}
```
