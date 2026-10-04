---
name: "gift_eval/ett1_15T"
description: "GIFT-Eval ETT (set 1) at 15-minute frequency: 1 series, 7 variates each of Energy data, mean length 69,680 steps; short-term horizon 48. Use for zero-shot evaluation on the full resampled ETT series; not for comparison with the LTSF `etth*`/`ettm*` 12/4/4-month protocol."
---

# gift_eval/ett1_15T

## Overview

ETT (set 1) (ett1/15T) is the GIFT-Eval series collection built from oil temperature and six power-load features of the first Electricity Transformer Temperature station (ETTh1/ETTm1 lineage), resampled to this frequency. It is part of the Energy domain of the GIFT-Eval benchmark, a zero-shot-oriented suite for general time series forecasting models (Salesforce AI Research). This preset forecasts the short-term horizon (48 steps); GIFT-Eval defines medium and long terms for some datasets, but this repository ships the short-term preset only.

## Protocol and pitfalls

- **Get the data (`upstream`).** TSFLab does not re-host GIFT-Eval. Run `uv run tsf data prepare --from gift-eval --datasets ett1/15T`: it downloads `ett1/15T` from https://huggingface.co/datasets/Salesforce/GiftEval and links it at `dataset/gift_eval`.
- Split (loader behavior): per series the last `pred_len * windows` steps are test, the preceding `pred_len` steps validation, the rest train. `windows` is `min(max(1, ceil(0.1 * shortest_series_length / pred_len)), 20)` computed from the shortest series (M4 ids use 1), so it can differ from the per-dataset window counts in the benchmark paper.
- Scaling: one StandardScaler per channel is fitted on the concatenated training regions of all series, not per series. Series shorter than `seq_len + pred_len` are skipped in train/val; test windows with too little context are dropped.
- Context length is not fixed by GIFT-Eval; pick `seq_len` per model and report it. Leaderboard metrics (MASE, CRPS) are scale-free and probabilistic; scores from this preset are comparable only when the same metric and windows are used.
- Leakage: the pretraining corpora of several foundation models overlap GIFT-Eval test data (source-reported). Use the GIFT-Eval pretrain split or a clean corpus when claiming zero-shot results.
- Multivariate: `features = "M"` forecasts all variates; `S` keeps the first variate only.
- This is the full ETT series resampled by GIFT-Eval, not the 12/4/4-month LTSF protocol used by the `etth*`/`ettm*` presets; results are not comparable.
