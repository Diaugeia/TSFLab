---
name: "gift_eval/saugeenday_D"
description: "GIFT-Eval Saugeen River Flow, daily: 1 series of Nature data (univariate, mean length 23,741), horizon 30. Use for zero-shot or single-series evaluation on one long daily river-flow record; not for cross-series, multivariate, or graph forecasting."
---

# gift_eval/saugeenday_D

## Overview

Saugeen River Flow (`saugeenday/D`) is the GIFT-Eval collection of mean flow of the Saugeen River at Walkerton (Ontario), from the Monash repository. It belongs to the Nature domain of GIFT-Eval (Salesforce AI Research), a zero-shot-oriented benchmark for general forecasting models. Scale (source-reported): 1 series, one variate each, mean length 23,741 steps, 20 short-term test windows. Only the short-term preset (horizon 30) ships here; GIFT-Eval also defines medium and long terms for some datasets.

## Protocol and pitfalls

- **Get the data (`upstream`).** TSFLab does not re-host GIFT-Eval. Run `uv run tsf data prepare --from gift-eval --datasets saugeenday/D`: it downloads `saugeenday/D` from https://huggingface.co/datasets/Salesforce/GiftEval and links it at `dataset/gift_eval`.
- **Split (loader).** Per series the last `pred_len * windows` steps are test, the preceding `pred_len` steps validation, the rest train. `windows` is `min(max(1, ceil(0.1 * shortest_series_length / pred_len)), 20)` computed from the shortest series (M4 ids use 1), so it can differ from the per-dataset window counts in the benchmark paper.
- **Scaling.** One StandardScaler per channel is fitted on the concatenated training regions of all series, not per series. Series shorter than `seq_len + pred_len` are skipped in train/val; test windows with too little context are dropped.
- **Context and metrics.** Context length is not fixed by GIFT-Eval; pick `seq_len` per model and report it. Leaderboard metrics (MASE, CRPS) are scale-free and probabilistic; scores from this preset are comparable only when the same metric and windows are used.
- **Leakage.** The pretraining corpora of several foundation models overlap GIFT-Eval test data (source-reported). Use the GIFT-Eval pretrain split or a clean corpus when claiming zero-shot results.
