---
name: "wind"
description: "15-minute power of one wind farm with five NWP forecast channels and measured wind speed, 2020-01-01 to 2021-05-22 (PaddleSpatial data, TFB packaging, 7 channels). Use for long-horizon energy forecasting with weather-forecast covariates; not for MAPE-type metrics (near-zero and negative power)."
---

# wind

## Overview

Wind is a single wind farm's power output at 15-minute resolution over about 17 months, with five numerical-weather-prediction forecast channels (wind speed, direction, temperature, pressure, humidity) and the measured wind speed. It was collected for D3VAE (Li et al., 2022) and published in PaddleSpatial; TFB (Qiu et al., 2024) uses it as its energy benchmark, on which simple linear models are competitive.

## Protocol and pitfalls

- **Protocol.** Horizons 96/192/336/720 steps (1 to 7.5 days) with lookback 96, 336 or 512, as TFB does for its long datasets.
- **Split.** 7:1:2 chronological, as in TFB for this dataset (TFB's `rolling_forecast_config.json` reserves 6:2:2 for ETT, PEMS, AQShunyi/AQWan and Solar). Train-only z-scoring per channel.
- **Season.** Validation starts about 2020-12-20 and test about 2021-02-09, so the model is tested on late winter to spring while trained mostly on 2020.
- **Forecast covariates.** The `pred_*` channels are, by their names, weather forecasts for each timestamp (the source does not document their issue time); in `"M"` mode they are forecast as if unknown, while in practice they would be known ahead, so this preset under-uses them. `pred_pressure` is unchanged between consecutive steps 91% of the time, so it is nearly trivial to forecast and lowers all-channel averages.
- **Negative power.** `target` is slightly negative in 12% of steps; the source does not document why (auxiliary consumption at standstill is a common cause). Relative metrics such as MAPE are unstable near zero.
- **Target column.** The preset sets `target = "target"` (power, the last channel), so `"S"` and `"MS"` forecast power.
- **`drop_last`.** Training drops its last partial batch; validation and test keep it, so test metrics cover every window.
