---
name: "gift_eval/temperature_rain_with_missing"
description: "GIFT-Eval Temperature and Rain, daily: 32,072 series of Nature data (univariate, mean length 725), horizon 30. Use for zero-shot evaluation on many daily weather series that contain missing values; not for models that cannot mask or impute NaN inputs."
---

# gift_eval/temperature_rain_with_missing

## Overview

Temperature and Rain (`temperature_rain_with_missing`) is the GIFT-Eval collection of daily temperature and rainfall series with missing values from Australian weather stations, from the Monash repository. It belongs to the Nature domain of GIFT-Eval (Salesforce AI Research), a zero-shot-oriented benchmark for general forecasting models. Scale (source-reported): 32,072 series, one variate each, mean length 725 steps, 3 short-term test windows. Only the short-term preset (horizon 30) ships here; GIFT-Eval also defines medium and long terms for some datasets.

## Protocol and pitfalls

- **Get the data (`upstream`).** TSFLab does not re-host GIFT-Eval. Run `uv run tsf data prepare --from gift-eval --datasets temperature_rain_with_missing`: it downloads `temperature_rain_with_missing` from https://huggingface.co/datasets/Salesforce/GiftEval and links it at `dataset/gift_eval`.
- **Split (loader).** Per series the last `pred_len * windows` steps are test, the preceding `pred_len` steps validation, the rest train. `windows` is `min(max(1, ceil(0.1 * shortest_series_length / pred_len)), 20)` computed from the shortest series (M4 ids use 1), so it can differ from the per-dataset window counts in the benchmark paper.
- **Scaling.** One StandardScaler per channel is fitted on the concatenated training regions of all series, not per series. Series shorter than `seq_len + pred_len` are skipped in train/val; test windows with too little context are dropped.
- **Context and metrics.** Context length is not fixed by GIFT-Eval; pick `seq_len` per model and report it. Leaderboard metrics (MASE, CRPS) are scale-free and probabilistic; scores from this preset are comparable only when the same metric and windows are used.
- **Leakage.** The pretraining corpora of several foundation models overlap GIFT-Eval test data (source-reported). Use the GIFT-Eval pretrain split or a clean corpus when claiming zero-shot results.
- **Missing values.** NaNs pass through the loader (scaler statistics ignore them), so models receive NaN inputs and targets unless you impute or mask them.
