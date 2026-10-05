---
name: "fred_md"
description: "FRED-MD monthly US macroeconomic indicators in the TFB packaging (107 series, 728 months, 1959-2019). Use for short-horizon (24-60 month) forecasting of trending, scale-heterogeneous series with a short history; not for long-lookback models or stationarity-transformed FRED-MD protocols."
---

# fred_md

## Overview

FRED-MD is a monthly database of US macroeconomic indicators (output, labor, housing, prices, money, interest rates, stock market) maintained at the St. Louis Fed from FRED data. TFB packaged 107 of the series for 728 months as a multivariate benchmark. The series are non-stationary, trend-dominated, and heterogeneous in scale, with only 728 observations.

## Protocol and pitfalls

- **Get the file (`script`).** FRED's terms forbid re-hosting, so TSFLab does not ship it. Run `uv run tsf data prepare --from tfb --datasets fred_md`: it downloads the TFB forecasting archive from Google Drive, pivots member `forecasting/FRED-MD.csv` to wide (channels in first-appearance order, last channel renamed `OT`), checks every sha256, and writes `dataset/FRED-MD/FRED-MD.csv`.
- **Protocol.** Horizons 24/36/48/60 months with lookback 36 or 104 and split 7:1:2; the test split starts around mid-2007 (row 582 of 728, assuming the file starts in 1959-01) and therefore covers the 2008 financial crisis, with only about 146 test rows.
- **Revisions.** FRED-MD is revised monthly; a file downloaded later contains different vintages and a different series count than the TFB snapshot.
- **Transformations.** The standard FRED-MD recommends stationarity transformations per series; the preset forecasts levels as given in the file, which are trending.
- **`drop_last`.** Training drops its last partial batch; validation and test keep it, so test metrics cover every window.
- **Target column.** `features = "S"` forecasts `OT`; `"MS"` forecasts the last column.
