---
name: "gift_eval/jena_weather_D"
description: "GIFT-Eval Jena Weather at daily frequency (id jena_weather/D): one series of 21 meteorological variates, short-term horizon 30. Use for zero-shot multivariate weather forecasting (one series, 21 variates); not for comparison with the LTSF `weather` preset (different protocol)."
---

# gift_eval/jena_weather_D

## Overview

Jena Weather (jena_weather/D) is the GIFT-Eval series collection built from 21 meteorological indicators from the Max Planck Institute for Biogeochemistry weather station in Jena, as prepared for Autoformer. It is part of the Nature domain of the GIFT-Eval benchmark, a zero-shot-oriented suite for general time series forecasting models (Salesforce AI Research). This preset forecasts the short-term horizon (30 steps); GIFT-Eval defines medium and long terms for some datasets, but this repository ships the short-term preset only.

## Protocol and pitfalls

- **Get the data (`upstream`).** TSFLab does not re-host GIFT-Eval. Run `uv run tsf data prepare --from gift-eval --datasets jena_weather/D`: it downloads `jena_weather/D` from https://huggingface.co/datasets/Salesforce/GiftEval and links it at `dataset/gift_eval`.
- Split (loader behavior): per series the last `pred_len * windows` steps are test, the preceding `pred_len` steps validation, the rest train. `windows` is `min(max(1, ceil(0.1 * shortest_series_length / pred_len)), 20)` computed from the shortest series (M4 ids use 1), so it can differ from the per-dataset window counts in the benchmark paper.
- Scaling: one StandardScaler per channel is fitted on the concatenated training regions of all series, not per series. Series shorter than `seq_len + pred_len` are skipped in train/val; test windows with too little context are dropped.
- Context length is not fixed by GIFT-Eval; pick `seq_len` per model and report it. Leaderboard metrics (MASE, CRPS) are scale-free and probabilistic; scores from this preset are comparable only when the same metric and windows are used.
- Leakage: the pretraining corpora of several foundation models overlap GIFT-Eval test data (source-reported). Use the GIFT-Eval pretrain split or a clean corpus when claiming zero-shot results.
- Multivariate: `features = "M"` forecasts all variates; `S` keeps the first variate only.
- Id: this preset selects `jena_weather/D`, one of three frequencies in `Salesforce/GiftEval` (checked against the Hugging Face repository tree, not a local copy); siblings are `gift_eval/jena_weather_10T` and `gift_eval/jena_weather_H`. The older unsuffixed id `jena_weather` is not a preset.
