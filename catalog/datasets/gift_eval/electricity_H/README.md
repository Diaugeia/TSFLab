---
name: "gift_eval/electricity_H"
description: "GIFT-Eval Electricity at hourly frequency: 370 series of Energy data, mean length 35,064 steps; short-term horizon 48. Use for zero-shot forecasting of 370 client load series; not for comparison with the LTSF `electricity` preset (321 clients, different protocol)."
---

# gift_eval/electricity_H

## Overview

Electricity (electricity/H) is the GIFT-Eval series collection built from electricity consumption of 370 Portuguese clients, 2011-2014 (UCI ElectricityLoadDiagrams20112014). It is part of the Energy domain of the GIFT-Eval benchmark, a zero-shot-oriented suite for general time series forecasting models (Salesforce AI Research). This preset forecasts the short-term horizon (48 steps); GIFT-Eval defines medium and long terms for some datasets, but this repository ships the short-term preset only.

## Protocol and pitfalls

- **Get the data (`upstream`).** TSFLab does not re-host GIFT-Eval. Run `uv run tsf data prepare --from gift-eval --datasets electricity/H`: it downloads `electricity/H` from https://huggingface.co/datasets/Salesforce/GiftEval and links it at `dataset/gift_eval`.
- Split (loader behavior): per series the last `pred_len * windows` steps are test, the preceding `pred_len` steps validation, the rest train. `windows` is `min(max(1, ceil(0.1 * shortest_series_length / pred_len)), 20)` computed from the shortest series (M4 ids use 1), so it can differ from the per-dataset window counts in the benchmark paper.
- Scaling: one StandardScaler per channel is fitted on the concatenated training regions of all series, not per series. Series shorter than `seq_len + pred_len` are skipped in train/val; test windows with too little context are dropped.
- Context length is not fixed by GIFT-Eval; pick `seq_len` per model and report it. Leaderboard metrics (MASE, CRPS) are scale-free and probabilistic; scores from this preset are comparable only when the same metric and windows are used.
- Leakage: the pretraining corpora of several foundation models overlap GIFT-Eval test data (source-reported). Use the GIFT-Eval pretrain split or a clean corpus when claiming zero-shot results.
- 370 clients here versus the 321 clients of the LTSF `electricity` preset.
