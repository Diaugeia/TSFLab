---
name: "rt/traffic_pems_sac"
description: "Hourly flow of 801 Caltrans PeMS stations in District 3 (Sacramento), 2019-2023, as nodes with calendar covariates; frozen release of real-time track traffic_pems_sac. Use for large-panel hourly flow forecasting across a 2020 regime shift; not for graph models (no adjacency)."
---

# rt/traffic_pems_sac

## Overview

`rt_traffic_pems_sac` serves the hourly total vehicle flow of every station in Caltrans PeMS District 3 (Sacramento) as a static dataset: the panel store `dataset/realtime/traffic_pems_sac` read at release `2026.09.28-2003`, 801 stations as nodes plus time-of-day and day-of-week covariates, 43,824 hourly rows from 2019-01-01 to 2023-12-31, 23.83% of cells missing before filling (measured). It is the same panel a real-time round uses as history, so a model trained here can be compared with the live `traffic_pems_sac` track.

## Protocol and pitfalls

- **Coverage is not uniform.** The NaN share falls from 39.7% in 2019 to 0.0% in 2023, so the test split is the best-covered year while the training split carries most of the filled gaps.
- **2020 regime.** Mean flow drops from 2,275 vehicles per hour in 2019 to 1,911 in 2020 partly recovers to 2,041 in 2021, then falls to 1,664 in 2023, so the training split mixes pre-pandemic and pandemic regimes.
- **No adjacency.** The store carries no station coordinates, so graph models receive no adjacency.
- **Frozen snapshot, not the live track.** The preset reads the local panel store at the pinned release (`version`). The live track's rolling weekly rounds (`docs/en/realtime.md`) remain the contamination-free evaluation; a static split says nothing about data that arrive later.
- **Gap filling.** Unobserved cells are forward-filled; only a series' leading gap is back-filled from its first reading. The loader fills the whole panel before splitting, so a channel that starts late carries a constant first value in training; fully empty channels become zero.
- **Scaling.** One scalar mean and standard deviation from the training rows is shared by all channels, as in the real-time export, so large channels dominate raw-unit MSE.
- **`drop_last`.** Training drops its last partial batch; validation and test keep it, so test metrics cover every window.
