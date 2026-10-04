---
name: "gift_eval/bitbrains_fast_storage_H"
description: "GIFT-Eval Bitbrains fastStorage at hourly frequency: 1,250 series, 2 variates each of Web/CloudOps data, mean length 721 steps; short-term horizon 48. Use for zero-shot forecasting of many two-variate VM resource-usage traces; not for long-horizon evaluation (short-term preset only)."
---

# gift_eval/bitbrains_fast_storage_H

## Overview

Bitbrains fastStorage (bitbrains_fast_storage/H) is the GIFT-Eval series collection built from resource-usage traces of virtual machines on the fastStorage tier of a managed-hosting provider (Bitbrains). It is part of the Web/CloudOps domain of the GIFT-Eval benchmark, a zero-shot-oriented suite for general time series forecasting models (Salesforce AI Research). This preset forecasts the short-term horizon (48 steps); GIFT-Eval defines medium and long terms for some datasets, but this repository ships the short-term preset only.

## Protocol and pitfalls

- **Get the data (`upstream`).** TSFLab does not re-host GIFT-Eval. Run `uv run tsf data prepare --from gift-eval --datasets bitbrains_fast_storage/H`: it downloads `bitbrains_fast_storage/H` from https://huggingface.co/datasets/Salesforce/GiftEval and links it at `dataset/gift_eval`.
- Split (loader behavior): per series the last `pred_len * windows` steps are test, the preceding `pred_len` steps validation, the rest train. `windows` is `min(max(1, ceil(0.1 * shortest_series_length / pred_len)), 20)` computed from the shortest series (M4 ids use 1), so it can differ from the per-dataset window counts in the benchmark paper.
- Scaling: one StandardScaler per channel is fitted on the concatenated training regions of all series, not per series. Series shorter than `seq_len + pred_len` are skipped in train/val; test windows with too little context are dropped.
- Context length is not fixed by GIFT-Eval; pick `seq_len` per model and report it. Leaderboard metrics (MASE, CRPS) are scale-free and probabilistic; scores from this preset are comparable only when the same metric and windows are used.
- Leakage: the pretraining corpora of several foundation models overlap GIFT-Eval test data (source-reported). Use the GIFT-Eval pretrain split or a clean corpus when claiming zero-shot results.
- Multivariate: `features = "M"` forecasts all variates; `S` keeps the first variate only.
