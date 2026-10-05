---
name: "wike2000"
description: "Daily page views of 2,000 Wikipedia pages over 792 days (GluonTS wiki2000_nips train split, TFB packaging). Use for short, high-dimensional, heavy-tailed count forecasting; not for comparison with GluonTS/GP-Copula results, or with the 2014-01-05 outage day left unmasked."
---

# wike2000

## Overview

Wike2000 is TFB's (Qiu et al., 2024) web-traffic benchmark: daily page-view counts of 2,000 Wikipedia pages for 792 days. It is exactly the training split of the GluonTS `wiki2000_nips` dataset used in the GP-Copula paper. Series are heavy-tailed integer counts with weekly patterns, bursts and trends, and the 2,000 channels make it the highest-dimensional short benchmark in TFB.

## Protocol and pitfalls

- **Get the file (`script`).** The GluonTS/TFB copy carries no data license, so TSFLab does not ship it. Run `uv run tsf data prepare --from tfb --datasets wike2000`: it downloads the TFB forecasting archive from Google Drive, pivots member `forecasting/Wike2000.csv` to wide (last channel renamed `OT`), checks every sha256, and writes `dataset/Wike2000/Wike2000.csv`.
- **Protocol.** Horizons 24/36/48/60 days with lookback 36 or 104, as TFB does for its short datasets; 7:1:2 leaves about 159 test rows, so the longest setting has few windows.
- **Split.** 7:1:2 chronological, as in TFB for this dataset (TFB's `rolling_forecast_config.json` reserves 6:2:2 for ETT, PEMS, AQShunyi/AQWan and Solar). Train-only z-scoring per channel.
- **Outage day.** 2014-01-05 is zero for every page and falls in the test split (which starts about 2013-09-25); it is an artifact, and errors on and around it dominate squared-error metrics. Report whether it was masked.
- **Heavy tails.** Page views span six orders of magnitude across pages and contain viral bursts; z-scoring per channel on the training split leaves extreme test values.
- **Different benchmark.** GluonTS/GP-Copula results use the 912-day test split with 30-day horizons; TFB results on this 792-day file are not comparable to them.
- **Target column.** `features = "S"` forecasts `OT`; `"MS"` forecasts the last column.
- **`drop_last`.** Training drops its last partial batch; validation and test keep it, so test metrics cover every window.
