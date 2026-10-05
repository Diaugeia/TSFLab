---
name: "nn5"
description: "Daily cash withdrawals of 111 ATMs from the NN5 competition (Monash/TFB packaging, 791 days, imputed). Use for short, weekly-seasonal, low-data multivariate forecasting; not for comparison with the competition's 56-day task or for studying raw missing values."
---

# nn5

## Overview

NN5 is the daily cash-withdrawal history of 111 ATMs from the NN5 forecasting competition. The Monash archive republished it and TFB (Qiu et al., 2024) packaged it as a 111-channel multivariate CSV. The series have a strong day-of-week pattern, occasional holidays and outages, and a short length of 791 days, so it tests small-data seasonal forecasting.

## Protocol and pitfalls

- **Protocol.** TFB multivariate: horizons 24/36/48/60, lookback 36 or 104, split 7:1:2. The competition's own task was a 56-day horizon; results are not comparable.
- **Imputation.** The file hides missing values by imputation, so models are scored on filled values at some timestamps.
- **Length.** 791 steps with a 7:1:2 split leaves about 160 test rows; with lookback 104 and horizon 60 only a handful of test windows exist.
- **`drop_last`.** Training drops its last partial batch; validation and test keep it, so test metrics cover every window.
- **Target column.** `features = "S"` forecasts `OT`; `"MS"` forecasts the last column.
