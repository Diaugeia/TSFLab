---
name: "gift_eval"
description: "GIFT-Eval, Salesforce's zero-shot-oriented benchmark for general time series forecasting; TSFLab ships 53 of its dataset-frequency series as short-term presets. Use for broad multi-domain, multi-frequency evaluation of foundation and baseline models; not for the single long-horizon multivariate LTSF protocol."
---

# gift_eval

## Overview

GIFT-Eval (General Time Series Forecasting Model Evaluation) was built by Salesforce AI Research to compare time series foundation models and classical or deep baselines on one broad, zero-shot-oriented suite. The benchmark holds 23 datasets across 7 domains and 10 frequencies (144,000 series, 177 million observations; 97 dataset, frequency, and term configurations), mixing 15 univariate and 8 multivariate datasets (source-reported). TSFLab ships 53 of the dataset-frequency series as presets named `gift_eval/<id>`, each at the short-term horizon.

Use a family member when you need breadth (many domains, frequencies, series counts, and horizons) rather than the single long-horizon multivariate protocols of the LTSF presets (`etth1`, `electricity`, `weather`, ...). Each series card links back here.

## Protocol and pitfalls

- **Get the data (`upstream`).** TSFLab does not re-host GIFT-Eval. Run `uv run tsf data prepare --from gift-eval`: it downloads all 55 sets from https://huggingface.co/datasets/Salesforce/GiftEval and links them at `dataset/gift_eval` (`--datasets` picks a subset).
- Official protocol: the last 10% of each series is test, scored with non-overlapping rolling windows of length equal to the horizon (at most 20 windows); the window before the test region serves as validation. Context length is chosen by the model.
- The loader in this repository derives the number of test windows from the shortest series of the dataset (`min(max(1, ceil(0.1 * min_len / pred_len)), 20)`, 1 for M4), which can differ from the per-dataset counts in the paper. Inspect `tsf catalog show gift_eval/<id>` and the loader before comparing against leaderboard numbers.
- The scaler is fitted on the training regions of all series together; z-scored losses weigh series by their original variance unlike the scale-free official metrics (MASE, CRPS on the leaderboard).
- Leakage: the pretraining corpora of TimesFM, Chronos, and Moirai partly overlap GIFT-Eval test data (source-reported). Zero-shot claims need the GIFT-Eval pretrain split or a verified clean corpus.
- `*_with_missing` series contain NaNs that the loader passes through; handle them in the model or the task.
- Multi-frequency datasets (electricity, ETT, solar, ...) repeat the same underlying data at several resolutions; do not treat them as independent datasets when aggregating scores.
