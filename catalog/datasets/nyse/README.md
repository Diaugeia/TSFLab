---
name: "nyse"
description: "One NYSE stock (CHS) from the Relational Stock Ranking data, TFB packaging: five max-normalized daily features (four close moving averages and the close), 1,243 trading days. Use for short, smooth, highly correlated multivariate TFB runs; not for real OHLCV forecasting or calendar features."
---

# nyse

## Overview

NYSE is one stock from the NYSE universe of Feng et al.'s Relational Stock Ranking data, packaged by TFB (Qiu et al., 2024) as a 5-channel daily benchmark. TFB labels the channels Open, High, Low, Close and Volume, but the values are Feng et al.'s engineered features: 5-, 10-, 20- and 30-day moving averages of the close price and the close price itself, each divided by the stock's maximum close. The series are smooth, trending and almost perfectly correlated.

## Protocol and pitfalls

- **Get the file (`script`).** The price terms forbid re-hosting, so TSFLab does not ship it. Run `uv run tsf data prepare --from tfb --datasets nyse`: it downloads the TFB forecasting archive from Google Drive, pivots member `forecasting/NYSE.csv` to wide by position (dates from the first block, as TFB's `read_data` does; see reference.md), checks every sha256, and writes `dataset/NYSE/NYSE.csv`.
- **Protocol.** Horizons 24/36/48/60 trading days with lookback 36 or 104, as TFB does for its short datasets; 7:1:2 leaves about 249 test rows.
- **Split.** 7:1:2 chronological, as in TFB for this dataset (TFB's `rolling_forecast_config.json` reserves 6:2:2 for ETT, PEMS, AQShunyi/AQWan and Solar). Train-only z-scoring per channel.
- **Not OHLCV.** Despite the names, channels 1 to 4 are moving averages of the close and `Volume` is the close; there is no traded volume. Moving averages are nearly deterministic given recent closes, so all-channel averages look easier than forecasting a price.
- **Date column.** The `date` column holds row numbers; parsed as datetimes they become nanosecond offsets from 1970-01-01, so calendar time features (weekday, month) are meaningless. Use models that do not rely on timestamp marks, or treat them as constant.
- **Full-sample normalization.** Values are already divided by the maximum close over the whole series (a full-sample statistic chosen by the original authors); TSFLab's train-only z-scoring does not undo that.
- **Target column.** The preset sets `target = "Volume"` (the last channel, the normalized close); set `Close` for the 30-day moving average.
- **`drop_last`.** Training drops its last partial batch; validation and test keep it, so test metrics cover every window.
