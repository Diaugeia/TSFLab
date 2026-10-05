---
name: "rt/air_airnow_us"
description: "Hourly PM2.5 at up to 200 US stations from EPA AirNow (preliminary data), frozen release of real-time track air_airnow_us; seq_len 168, horizon 24. Use for hourly multi-station air-quality forecasting on noisy, irregular data; not for seasonality beyond a few weeks or regulatory-grade values."
---

# rt/air_airnow_us

## Overview

`rt_air_airnow_us` serves the AirNow PM2.5 real-time track as a static spatiotemporal dataset: hourly PM2.5 (micrograms per cubic metre) at up to 200 US stations (sites reporting at least 90% of hours at bootstrap), from EPA AirNow's preliminary, unvalidated `HourlyData` files. It holds far less history than the other tracks: an 8-week (56-day) bootstrap plus each weekly release. Use it to train and check models before submitting to the live track.

## Protocol and pitfalls

- **Sensor sets are fixed at bootstrap.** The channel set is frozen so rounds stay comparable; stations that start reporting later are not added, and stations that stop remain as forward-filled constants.
- **Reported values are noisy.** PM2.5 readings can be sparse, spiky, and include instrument noise; the real-time store keeps them as reported.
- **Mixed time zones.** Stamps are naive UTC while stations span several time zones, so local daily cycles are phase-shifted between channels.
- **Short history.** Only the 56-day bootstrap plus later weeks exist, so seasonal structure beyond a few weeks is absent from the training split.
- **Negative values.** Negative PM2.5 readings are raw instrument noise and are kept.
- **Frozen snapshot, not the live track.** No `version` is pinned, so the preset reads whatever the local panel store holds; pin a release or set `revision` for a reproducible study. The live track's rolling weekly rounds (`docs/en/realtime.md`) remain the contamination-free evaluation; a static split says nothing about data that arrive later.
- **Gap filling.** Unobserved cells are forward-filled; only a series' leading gap is back-filled from its first reading. The loader fills the whole panel before splitting, so a channel that starts late carries a constant first value in training; fully empty channels become zero.
- **Scaling.** One scalar mean and standard deviation from the training rows is shared by all channels, as in the real-time export, so large channels dominate raw-unit MSE.
- **`drop_last`.** Training drops its last partial batch; validation and test keep it, so test metrics cover every window.
