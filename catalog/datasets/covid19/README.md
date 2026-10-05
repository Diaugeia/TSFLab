---
name: "covid19"
description: "Daily COVID-19 new and cumulative cases and deaths for 237 countries and territories (2020-2023, 948 channels, TFB packaging). Use for short-horizon (24-60 day) forecasting of many zero-inflated, scale-heterogeneous channels with a late regime shift; not for long-horizon LTSF or MAPE-type metrics."
---

# covid19

## Overview

Covid-19 is TFB's (Qiu et al., 2024) country-level pandemic panel: for each of 237 countries and territories it holds daily new cases, cumulative cases, new deaths and cumulative deaths, 948 channels over 1,392 days. It is the widest of TFB's short multivariate benchmarks and mixes monotone cumulative curves, spiky and often-zero daily counts, and series that differ by several orders of magnitude.

## Protocol and pitfalls

- **Protocol.** Horizons 24/36/48/60 days with lookback 36 or 104, as TFB does for its short datasets (the paper groups Covid-19 with ILI, NN5, FRED-MD, NASDAQ, NYSE and Wike2000).
- **Split.** 7:1:2 chronological, as in TFB for this dataset (TFB's `rolling_forecast_config.json` gives the 6:2:2 exception only to ETT, PEMS, AQShunyi/AQWan and Solar; every TFB script for this dataset uses that config). Train-only z-scoring per channel.
- **Regime shift.** The split boundaries fall at about 2022-09-03 (validation) and 2023-01-20 (test), so the test split covers the late, low-reporting phase of the pandemic; the share of zero `New_cases` values rises from 41% in the training split to 80% in the test split (measured), consistent with sparser reporting late in the pandemic.
- **Zeros and corrections.** All-zero channels have zero training variance (the scaler leaves them constant) and negative daily counts are reporting corrections, not data errors; relative metrics such as MAPE are undefined on them.
- **Scale.** Cumulative channels are monotone and span orders of magnitude across countries; raw-scale MSE is dominated by the largest countries.
- **Target column.** `features = "S"` and `"MS"` forecast `Zimbabwe;Cumulative_deaths`, the last column (TFB itself evaluates all channels jointly); the column names contain `;` and spaces, so quote them when overriding `target`.
- **`drop_last`.** Training drops its last partial batch; validation and test keep it, so test metrics cover every window.
