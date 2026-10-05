---
name: "rt/weather_openmeteo_temp"
description: "Hourly 2 m temperature at 82 US and EU cities from the Open-Meteo Historical Forecast API, frozen release of real-time track weather_openmeteo_temp; seq_len 168, horizon 24. Use for multi-site hourly temperature forecasting; not for station-measured noise or learning the annual cycle from one year."
---

# rt/weather_openmeteo_temp

## Overview

`rt_weather_openmeteo_temp` serves the Open-Meteo temperature real-time track as a static spatiotemporal dataset: hourly 2 m air temperature at 82 US and EU cities as nodes plus calendar covariates. Values are model analysis from operational weather models; history is a 365-day backfill (analysis hours from 2022) plus each weekly release. Use it to train and check models before submitting to the live track.

## Protocol and pitfalls

- **Model values, not station readings.** Values are model analysis at coordinates, so they are smoother than gauge data; the newest three hours are forecasts and are not stored.
- **Strong annual cycle.** A one-year history gives the training split one season and the test split another, so the seasonal shift is large.
- **Frozen snapshot, not the live track.** No `version` is pinned, so the preset reads whatever the local panel store holds; pin a release or set `revision` for a reproducible study. The live track's rolling weekly rounds (`docs/en/realtime.md`) remain the contamination-free evaluation; a static split says nothing about data that arrive later.
- **Gap filling.** Unobserved cells are forward-filled; only a series' leading gap is back-filled from its first reading. The loader fills the whole panel before splitting, so a channel that starts late carries a constant first value in training; fully empty channels become zero.
- **Scaling.** One scalar mean and standard deviation from the training rows is shared by all channels, as in the real-time export, so large channels dominate raw-unit MSE.
- **`drop_last`.** Training drops its last partial batch; validation and test keep it, so test metrics cover every window.
