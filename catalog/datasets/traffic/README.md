---
name: "traffic"
description: "Hourly road occupancy rate of 862 San Francisco Bay Area freeway sensors, 2015-2016 (Caltrans PeMS via LSTNet/Autoformer). Use for long-horizon LTSF with many channels and daily and weekly rush-hour cycles; not for graph models (no adjacency) or comparison with PeMS flow benchmarks."
---

# traffic

## Overview

Traffic is the hourly road occupancy rate (the fraction of time a sensor is occupied, between 0 and 1) of 862 loop-detector sensors on San Francisco Bay Area freeways during 2015 and 2016. Caltrans collects it in PeMS; Lai et al. (LSTNet) packaged an hourly version that Informer, Autoformer, and Time-Series-Library adopted. It is the standard 800-plus-channel benchmark with sharp daily and weekly rush-hour patterns, and it has no sensor locations or adjacency in this packaging.

## Protocol and pitfalls

- **Split.** TSFLab uses a chronological 7:1:2 split for this dataset. Results reported under other splits (6:2:2) are not directly comparable.
- **Target column.** This copy names the sensors `1` to `862` and has no `OT` column, so the preset targets `862`, the last sensor, which the TSLib copy calls `OT`. `"S"` and `"MS"` forecast sensor `862`.
- **Memory.** 862 channels with a long lookback make attention-style models expensive; many papers cap batch size or channel sampling, which changes results.
- **Different dataset, same name.** The PeMS03/04/07/08 presets are 5-minute flow graphs and the `rt/traffic_pems_*` presets are hourly flow per station from 2019; none are comparable with this occupancy file.
- **`drop_last`.** Training drops its last partial batch; validation and test keep it, so test metrics cover every window.
