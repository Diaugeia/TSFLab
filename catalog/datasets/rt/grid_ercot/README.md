---
name: "rt/grid_ercot"
description: "Hourly load of the eight ERCOT weather zones in Texas from 2019, in America/Chicago local time; frozen release of real-time track grid_ercot, seq_len 168, horizon 24. Use for small multivariate hourly load forecasting with several years of history; not for spatiotemporal models that need many nodes."
---

# rt/grid_ercot

## Overview

`rt_grid_ercot` serves the ERCOT load real-time track as a static dataset in the spatiotemporal layout: hourly load (megawatts) of the eight weather zones COAST, EAST, FWEST, NORTH, NCENT, SOUTH, SCENT and WEST as nodes plus calendar covariates, from ERCOT `Native_Load` archives since 2019-01-01 and weekly releases. Use it to train and check models before submitting to the live track.

## Protocol and pitfalls

- **Local clock time.** Hour-ending values are stored at the start of the hour they cover, in America/Chicago local time; the repeated hour of the autumn clock change is averaged. Stamps are naive local time, so daylight saving shifts the daily cycle against UTC by an hour for part of the year.
- **Only eight channels.** The panel is small for spatiotemporal models; it behaves like a multivariate load series.
- **Frozen snapshot, not the live track.** No `version` is pinned, so the preset reads whatever the local panel store holds; pin a release or set `revision` for a reproducible study. The live track's rolling weekly rounds (`docs/en/realtime.md`) remain the contamination-free evaluation; a static split says nothing about data that arrive later.
- **Gap filling.** Unobserved cells are forward-filled; only a series' leading gap is back-filled from its first reading. The loader fills the whole panel before splitting, so a channel that starts late carries a constant first value in training; fully empty channels become zero.
- **Scaling.** One scalar mean and standard deviation from the training rows is shared by all channels, as in the real-time export, so large channels dominate raw-unit MSE.
- **`drop_last`.** Training drops its last partial batch; validation and test keep it, so test metrics cover every window.
