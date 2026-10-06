---
name: "LogTrans"
description: "Decoder-only probabilistic Transformer with causal-convolution queries/keys and LogSparse attention; one network shared by all channels emits a Gaussian per step autoregressively. Use for probabilistic forecasts of related series with long lookbacks; not for cross-channel dependence or fast long-horizon inference."
---

# LogTrans

## Idea

- Each channel is a univariate series; the input row is `[z_{t-1} / nu, calendar x_t, position + series-ID embedding]` (Sec. 3, App. A.2), projected to `d_model`.
- `logsparse_conv_attention`: queries and keys from a causal `Conv1d` of kernel `kernel_size` (Sec. 4.1); the LogSparse mask lets a cell see itself and cells at distances 1, 2, 4, ... (Sec. 4.2), optionally a dense `local_size` window and restarts every `restart_len`.
- `gaussian_parameter_head` gives `(mu, sigma)` (softplus), rescaled by the series scale `nu`.
- Training (`training_objective`): teacher-forced Gaussian NLL over the history and the horizon (App. A.2). Inference: autoregressive decoding with cached keys and values, feeding back the mean; output type `distribution`.

## When to use

- Probabilistic forecasts of many related, roughly independent series with daily and weekly cycles (calendar marks are used).
- Long lookbacks where full attention memory is the limit (`sparse = true`); `kernel_size > 1` helps when anomalies or level changes confuse point-wise matching.
- Not for cross-channel dependence (channels share weights only) or long horizons at tight inference cost: decoding is sequential.

## Configure

- `enc_in`: the dataset channel count.
- `e_layers`: full reach of the plain LogSparse pattern needs `floor(log2(seq_len + pred_len - 1)) + 1` layers.
- `restart_len`, `local_size`: 0 disables them; the paper's long-window setting uses `restart_len = (seq_len + pred_len) / 8` and `local_size = ceil(log2(restart_len))`.
- Train with `training.loss = "nll_gaussian"` (validation and test score the decoded distribution).

Other hyperparameters: preset defaults in `configs/models/LogTrans.toml`; tune generically.

## Differences

No official code exists (fidelity `paper-only`); equations and structure were checked against Secs. 3-4 and App. A.2/A.4.

- Inference feeds back the predicted mean instead of ancestral sampling, so the output is one Gaussian per step, not sample quantiles (as in TSFLab `DeepAR`).
- Model width, input projection, post-norm blocks and dropout are local choices (the paper gives none); embeddings are 20-wide as in App. A.2.
- Scale `nu = 1 + mean |z|` instead of DeepAR's `1 + mean z`, because TSFLab series are standardized.
- Covariates are time-of-day and day-of-week only (no year, month, minute or age).
- Local and restart windows are defined on query-relative distances; the A.4 dense-prefix allowance and BERTAdam warm-up are not reproduced; the window-weighted sampling of training windows is replaced by the TSFLab loader.
- Training windows cover `seq_len + pred_len` steps; the NLL starts at the second step (no lag for the first).
