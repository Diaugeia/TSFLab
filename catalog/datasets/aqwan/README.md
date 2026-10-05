---
name: "aqwan"
description: "Hourly air pollutants and weather at the Wanshouxigong station of the UCI Beijing Multi-Site Air Quality data (2013-2017, 11 channels, TFB packaging). Use for multivariate long-horizon forecasting of spiky, heavy-tailed, weather-driven environmental series; not for spatial-graph air-quality tasks (one station only)."
---

# aqwan

## Overview

AQWan is the hourly record of one of 12 stations in the UCI Beijing Multi-Site Air Quality dataset, here the Wanshouxigong station: PM2.5, PM10, SO2, NO2, CO, O3, temperature, pressure, dew point, rain, and wind speed over four years. TFB (Qiu et al., 2024) packaged the two stations `AQShunyi` and `AQWan` as 11-channel multivariate benchmarks. Pollutant series are spiky and heavy-tailed, with strong dependence on weather and season.

## Protocol and pitfalls

- **Split.** TSFLab uses a chronological 7:1:2 split for this dataset. Results reported under other splits (6:2:2) are not directly comparable.
- **Imputation.** The TFB file's NaN-free values were filled; the filling method is not documented, so errors at formerly missing hours are not real measurements.
- **Heavy tails.** PM2.5 and PM10 spike during haze episodes; z-scoring on the training split leaves outliers, and MSE is dominated by a few events.
- **Target column.** The preset sets `target = "WSPM"` (wind speed, the last channel), not a pollutant, so `features = "S"` and `"MS"` forecast wind speed; set `target` to a pollutant such as `PM2.5` if that is the goal. TFB keeps the original UCI channel names, so there is no `OT` column (derived from the TFB reader, not a local copy); the loader raises if `target` is absent.
- **Layout.** TFB ships a long `date,data,cols` layout; the `custom` loader needs a wide CSV (`date` plus one column per channel), so pivot first.
- **`drop_last`.** Training drops its last partial batch; validation and test keep it, so test metrics cover every window.
