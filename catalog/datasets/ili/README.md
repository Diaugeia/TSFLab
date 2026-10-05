---
name: "ili"
description: "Weekly US influenza-like-illness surveillance from CDC ILINet (7 channels, 966 steps), the Autoformer LTSF set. Use for short, strongly seasonal, low-data forecasting at 24-60 week horizons; not for stable model rankings (few test windows) or calendar covariates (dates unverified)."
---

# ili

## Overview

ILI (national_illness) is the weekly record of influenza-like-illness surveillance in the United States from the CDC's ILINet: weighted and unweighted ILI percentages, counts by age group, total ILI cases, and the number of reporting providers, with `OT` as the target. Autoformer popularized it for LTSF with horizons of 24 to 60 weeks. At 966 weekly steps it is the shortest standard LTSF set, so training windows are scarce and results are noisy.

## Protocol and pitfalls

- **Protocol.** Lookback 36 (TFB also searches 104), horizons 24/36/48/60, split 7:1:2 (about 193 weekly test rows), so only a few dozen test windows exist at the longest horizon.
- **Short test set.** Metrics have high variance and depend on exact window handling, including `drop_last`; report seeds and window counts.
- **Seasonality.** A strong annual flu-season peak means only a few seasons fall in the test split. If the file's dates are real, the test split (about late 2016 to mid-2020) includes the first COVID-19 weeks while the 2009 H1N1 outbreak sits in training; the date column is not verified.
- **Date column.** The standard file's dates are nominal; do not use them as a calendar for external covariates without checking.
- **`drop_last`.** Training drops its last partial batch; validation and test keep it, so test metrics cover every window.
- **Target column.** `features = "S"` forecasts `OT`; `"MS"` forecasts the last column.
