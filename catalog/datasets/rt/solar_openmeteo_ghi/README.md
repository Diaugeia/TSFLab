---
name: "rt/solar_openmeteo_ghi"
description: "Hourly global horizontal irradiance at 55 PV-relevant sites from the Open-Meteo Historical Forecast API, frozen release of real-time track solar_openmeteo_ghi; seq_len 168, horizon 24. Use for multi-site irradiance forecasting with night zeros; not for station-measured noise or MAPE-type metrics."
---

# rt/solar_openmeteo_ghi

## Overview

`rt_solar_openmeteo_ghi` serves the Open-Meteo irradiance real-time track as a static spatiotemporal dataset: hourly global horizontal irradiance (watts per square metre) at 55 PV-relevant sites as nodes. Irradiance is the weather driver of PV output, and the source needs no API key. History is a 365-day backfill plus each weekly release. Use it to train and check models before submitting to the live track.

## Protocol and pitfalls

- **Model values, not station readings.** Values are model analysis at coordinates, so they are smoother than gauge data; the newest three hours are forecasts and are not stored.
- **Night zeros.** Irradiance is exactly zero at night, so MAPE-style metrics are undefined and MSE is driven by daytime peaks.
- **Frozen snapshot, not the live track.** No `version` is pinned, so the preset reads whatever the local panel store holds; pin a release or set `revision` for a reproducible study. The live track's rolling weekly rounds (`docs/en/realtime.md`) remain the contamination-free evaluation; a static split says nothing about data that arrive later.
- **Gap filling.** Unobserved cells are forward-filled; only a series' leading gap is back-filled from its first reading. The loader fills the whole panel before splitting, so a channel that starts late carries a constant first value in training; fully empty channels become zero.
- **Scaling.** One scalar mean and standard deviation from the training rows is shared by all channels, as in the real-time export, so large channels dominate raw-unit MSE.
- **`drop_last`.** Training drops its last partial batch; validation and test keep it, so test metrics cover every window.
