---
name: "solar"
description: "Simulated 10-minute power of 137 Alabama PV plants for 2006 (NREL data, LSTNet packaging), about 55% exact zeros at night. Use for long-horizon multivariate forecasting with a strong daily cycle and zero inflation; not for MAPE-type metrics or for real calendar features without checking the assumed start."
---

# solar

## Overview

Solar-Energy contains the simulated power output of 137 photovoltaic plants in Alabama for the year 2006. NREL publishes the underlying 5-minute simulations for solar integration studies; LSTNet packaged a 10-minute version that became a standard benchmark for periodic, zero-inflated multivariate series: output follows the day-night cycle and cloud-driven fluctuations and is exactly zero at night.

## Protocol and pitfalls

- **Zeros at night.** About 55% of all values are exactly zero (hours 0-4 and 19-23 are all zero, measured). Relative-error metrics such as MAPE are undefined there, z-scoring turns the night level into a large negative constant, and a large share of the error budget is the day-night mask rather than cloud-driven variation.
- **Split.** TSFLab uses a chronological 7:1:2 split for this dataset. Results reported under other splits (6:2:2) are not directly comparable.
- **Synthetic timestamps.** The file has no date column. The preset sets `start = "2006-01-01 00:00"` and `freq = "10min"` (LSTNet: calendar year 2006; 52,560 rows = 365 x 144), so row i is stamped `start + i * freq` and the loader emits real `(year, month, day, weekday, hour, minute)` marks for models that use them. This assumes row 0 is 00:00 on 2006-01-01; the measured night-hour pattern is consistent with it but does not prove it. Remove `start` to restore constant all-zero marks.
- **Target.** `target = "0"` selects the first plant for `features = "S"`; `"MS"` forecasts the last plant.
- **`drop_last`.** Training drops its last partial batch; validation and test keep it, so test metrics cover every window.
