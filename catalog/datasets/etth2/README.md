---
name: "etth2"
description: "ETTh2: hourly oil temperature (OT) and six load features of ETT-small transformer station 2, the standard Informer LTSF benchmark. Use for long-horizon (96-720) forecasting comparable with LTSF papers; not for high-dimensional channel modelling (7 channels) or shift-free evaluation (cooler, calmer test period)."
---

# etth2

## Overview

ETT-small (Electricity Transformer Temperature) records two years of data from two transformer stations in China. ETTh2 is the hourly series of station 2: six power-load features (HUFL/HULL high, MUFL/MULL middle, LUFL/LULL low useful and useless load) and the oil temperature `OT`, the quantity the dataset was built to forecast because oil temperature indicates transformer overload risk. It was introduced with Informer (AAAI 2021) and became the default long-sequence benchmark, so almost every LTSF paper reports it (ettm2 is the same station at the other resolution; etth1 is the sibling series).

## Protocol and pitfalls

- **Truncation.** The loader reads only the first 14,400 rows (20 months), so the last 3,020 rows of the 17,420-row CSV are never used (measured). The 20 months split chronologically as 12/4/4 months, which the TSFLab 6:2:2 ratio `[0.6, 0.2, 0.2]` reproduces exactly (results under 7:1:2 are not comparable): train rows 0-8,639, validation to row 11,520, test to row 14,400.
- **Window overlap.** Validation and test splits start `seq_len` rows early so the first target window has full context. This reuses earlier rows only as inputs; scaling statistics are fitted on the training rows alone (`scale = true`).
- **`drop_last`.** Training drops its last partial batch; validation and test keep it (`drop_last=False`), so test metrics cover every window. Reference code that drops the last test batch reports slightly different numbers, more so at large batch sizes.
- **Target.** `features = "S"` forecasts `OT`; `"MS"` forecasts the last column, which is also `OT` here; `"M"` forecasts all seven channels.
- **Frequency naming.** `ETTm*` is 15-minute data, not one-minute data, although the upstream README text says "every minute" (the CSV is verified 15-minute).
- **Distribution shift.** `OT` mean / std is 26.87 / 11.59 on the training rows and 14.41 / 7.13 on the test rows (measured, used rows), so the test period is much cooler and calmer than training; exact zeros are 9.8% of all values in the CSV (measured).
