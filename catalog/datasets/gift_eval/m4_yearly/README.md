---
name: "gift_eval/m4_yearly"
description: "GIFT-Eval M4 Yearly: 22,974 series of Econ/Fin data (univariate, mean length 37), horizon 6. Use for zero-shot evaluation on many very short yearly economic series with one test window each; not for multivariate, long-lookback, or long-horizon forecasting."
---

# gift_eval/m4_yearly

## Overview

M4 Yearly (`m4_yearly`) is the GIFT-Eval collection of yearly series from the M4 forecasting competition (Makridakis et al.), via the Monash repository. It belongs to the Econ/Fin domain of GIFT-Eval (Salesforce AI Research), a zero-shot-oriented benchmark for general forecasting models. Scale (source-reported): 22,974 series, one variate each, mean length 37 steps, one test window. Only the short-term preset (horizon 6) ships here; GIFT-Eval also defines medium and long terms for some datasets.

## Protocol and pitfalls

- **Get the data (`upstream`).** TSFLab does not re-host GIFT-Eval. Run `uv run tsf data prepare --from gift-eval --datasets m4_yearly`: it downloads `m4_yearly` from https://huggingface.co/datasets/Salesforce/GiftEval and links it at `dataset/gift_eval`.
- **Split (loader).** Per series the last `pred_len * windows` steps are test, the preceding `pred_len` steps validation, the rest train. `windows` is `min(max(1, ceil(0.1 * shortest_series_length / pred_len)), 20)` computed from the shortest series (M4 ids use 1), so it can differ from the per-dataset window counts in the benchmark paper.
- **Scaling.** One StandardScaler per channel is fitted on the concatenated training regions of all series, not per series. Series shorter than `seq_len + pred_len` are skipped in train/val; test windows with too little context are dropped.
- **Context and metrics.** Context length is not fixed by GIFT-Eval; pick `seq_len` per model and report it. Leaderboard metrics (MASE, CRPS) are scale-free and probabilistic; scores from this preset are comparable only when the same metric and windows are used.
- **Leakage.** The pretraining corpora of several foundation models overlap GIFT-Eval test data (source-reported). Use the GIFT-Eval pretrain split or a clean corpus when claiming zero-shot results.
