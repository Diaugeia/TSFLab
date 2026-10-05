---
name: "weather"
description: "21 meteorological indicators every 10 minutes for 2020 at the MPI-BGC Jena roof station, the Autoformer LTSF set. Use for long-horizon multivariate forecasting with correlated channels and a daily cycle; not for MAPE-type metrics (zero-heavy rain channels) or as-is comparisons that keep the -9999 sentinels."
---

# weather

## Overview

Weather is the 2020 record of the roof weather station of the Max Planck Institute for Biogeochemistry in Jena, Germany: 21 indicators (air pressure, temperatures, humidity, vapor pressure, wind speed and direction, rain, radiation, and derived quantities) sampled every 10 minutes. Autoformer introduced this subset to the LTSF canon; it is the usual benchmark for many correlated channels with a smooth daily cycle and sparse intermittent channels such as rain.

## Protocol and pitfalls

- **Split.** 7:1:2 chronological, scaling on training rows only. The training split is January to mid-September 2020, validation runs to mid-October, and the test split is roughly 19 October to December (measured), so seasonal shift between train and test is large.
- **Sentinels.** `-9999` marks missing readings in `OT` (50 rows), `max. PAR` (30), and `wv` (1). The preset sets `missing_sentinels = [-9999]`, so they become NaN and are filled causally before scaling (forward fill, then back fill only for a leading gap); all affected rows fall in the training split (rows 7,244 to 31,773 of 52,696, measured). Left in place they inflate the training standard deviation of `OT` from 18.2 to 384.0, of `max. PAR` from 573.5 to 643.4, and of `wv` from 1.67 to 52.1, with z-scores down to -192 (measured on the training split). Results therefore differ from papers that use the file as-is and from earlier TSFLab weather runs. Remove the parameter to reproduce the as-is protocol.
- **Target.** `"S"` and `"MS"` both forecast `OT`: the loader moves the named target to the last channel in `MS` mode, so the alphabetical column order of this file does not change the target.
- **Encoding.** Column names containing the micro and superscript characters were stored with replacement characters; do not select columns by those names.
- **Intermittent channels.** `rain (mm)` and `raining (s)` are mostly zero, so MSE is dominated by smooth channels while MAPE-style metrics are undefined.
- **`drop_last`.** Training drops its last partial batch; validation and test keep it, so test metrics cover every window.
