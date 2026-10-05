---
name: "czelan"
description: "Half-hourly sap flow of one plant plus ten environmental drivers at SAPFLUXNET site CZE_LAN (2016-2017, 11 channels, TFB packaging). Use for long-horizon forecasting of a diurnal biophysical response from covariates; not for learning yearly seasonality (13 months only) or MAPE-type metrics (night zeros)."
---

# czelan

## Overview

CzeLan is one SAPFLUXNET site (CZE_LAN) as packaged by TFB (Qiu et al., 2024): a 30-minute record of sap flow of one plant (`CZE_LAN_Cbe_Jt_16`) together with ten site environmental drivers. Sap flow is near zero at night and follows radiation and vapour-pressure deficit by day, so the set tests forecasting a biophysical response from strongly diurnal covariates.

## Protocol and pitfalls

- **Protocol.** Horizons 96/192/336/720 steps (2 to 15 days at 30 minutes) with lookback 96, 336 or 512, as TFB does for its long datasets.
- **Split.** 7:1:2 chronological, as in TFB for this dataset (TFB's `rolling_forecast_config.json` gives the 6:2:2 exception only to ETT, PEMS, AQShunyi/AQWan and Solar; every TFB script for this dataset uses that config). Train-only z-scoring per channel.
- **Season.** The record covers about 13 months and validation and test start at about 2017-02-20 and 2017-04-03, so each calendar season appears at most once in training and the test months' sap-flow regime is seen at most partially (or not at all) before evaluation.
- **Night zeros.** Sap flow and radiation are exactly zero at night; MAPE-type metrics are undefined there, and the repeated zeros inflate apparent accuracy.
- **Imputation.** The TFB file has no NaN although field sap-flow records normally have gaps; how gaps were filled is not documented, so some timestamps may be filled values.
- **Covariates.** `ext_rad` is a deterministic function of time and location, and `precip` is almost always zero; both are easy to forecast and lower all-channel averages.
- **Target column.** The preset sets `target = "CZE_LAN_Cbe_Jt_16"` (the sap-flow channel, the first column); `"MS"` moves it last and forecasts it from the ten drivers.
- **`drop_last`.** Training drops its last partial batch; validation and test keep it, so test metrics cover every window.
