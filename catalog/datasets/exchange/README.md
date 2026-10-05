---
name: "exchange"
description: "Daily exchange rates of eight currencies from the LSTNet collection (7,588 days, 8 channels). Use for long-horizon forecasting of non-seasonal, random-walk-like, drifting series checked against a last-value baseline; not for evaluating seasonal or periodic machinery."
---

# exchange

## Overview

Exchange-rate holds the daily exchange rates of eight countries' currencies, collected by Lai et al. for LSTNet and reused by Autoformer and Time-Series-Library. It is tiny (8 channels, about 7.6 thousand days), has no seasonality, and behaves like correlated random walks, so it is the standard check of whether a model beats a naive last-value forecast.

## Protocol and pitfalls

- **Get the file (`upstream`).** TSFLab does not re-host it. Run `uv run tsf data download exchange`: it fetches `exchange_rate.csv` byte-for-byte from the THUML Time-Series-Library Hugging Face repository, checks sha256 `48b4d9d3d508f5104162e85b9a6042e3557fde11aa9f2944eba8c0d0efc89842`, and writes `dataset/exchange_rate/exchange_rate.csv`.
- **Protocol.** Autoformer/TFB use 7:1:2, lookback 96, horizons 96/192/336/720 (LSTNet used 6:2:2 with horizons 3/6/12/24 days).
- **No seasonality.** Series are close to random walks, so a repeat-last-value forecast is hard to beat at long horizons; compare against it before claiming progress, and expect model rankings to differ from seasonal sets.
- **Distribution shift.** Levels drift across train and test, which z-scoring on the training split does not remove; instance normalization helps.
- **`drop_last`.** Training drops its last partial batch; validation and test keep it, so test metrics cover every window.
- **Target column.** `features = "S"` forecasts `OT`; `"MS"` forecasts the last column.
