---
name: "gift_eval/solar_H"
description: "GIFT-Eval Solar, hourly: 137 series of Energy data (univariate, mean length 8,760), horizon 48. Use for zero-shot evaluation on hourly, zero-inflated solar output with a daily cycle; not for MAPE-type metrics (night zeros) or cross-plant multivariate modelling."
---

# gift_eval/solar_H

## Overview

Solar (`solar/H`) is the GIFT-Eval collection of solar power production records of 137 photovoltaic plants in Alabama for 2006 (NREL data), as prepared for LSTNet. It belongs to the Energy domain of GIFT-Eval (Salesforce AI Research), a zero-shot-oriented benchmark for general forecasting models. Scale (source-reported): 137 series, one variate each, mean length 8,760 steps, 19 short-term test windows. Only the short-term preset (horizon 48) ships here; GIFT-Eval also defines medium and long terms for some datasets.

## Protocol and pitfalls

- **Get the data (`upstream`).** TSFLab does not re-host GIFT-Eval. Run `uv run tsf data prepare --from gift-eval --datasets solar/H`: it downloads `solar/H` from https://huggingface.co/datasets/Salesforce/GiftEval and links it at `dataset/gift_eval`.
- **Split (loader).** Per series the last `pred_len * windows` steps are test, the preceding `pred_len` steps validation, the rest train. `windows` is `min(max(1, ceil(0.1 * shortest_series_length / pred_len)), 20)` computed from the shortest series (M4 ids use 1), so it can differ from the per-dataset window counts in the benchmark paper.
- **Scaling.** One StandardScaler per channel is fitted on the concatenated training regions of all series, not per series. Series shorter than `seq_len + pred_len` are skipped in train/val; test windows with too little context are dropped.
- **Context and metrics.** Context length is not fixed by GIFT-Eval; pick `seq_len` per model and report it. Leaderboard metrics (MASE, CRPS) are scale-free and probabilistic; scores from this preset are comparable only when the same metric and windows are used.
- **Leakage.** The pretraining corpora of several foundation models overlap GIFT-Eval test data (source-reported). Use the GIFT-Eval pretrain split or a clean corpus when claiming zero-shot results.
- **Night zeros.** Solar output is zero at night (55% of values are zero in the local LTSF `solar` file), so relative-error metrics such as MAPE are undefined there.
