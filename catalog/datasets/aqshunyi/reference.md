# aqshunyi — reference

## Provenance and license

- Source: UCI Machine Learning Repository, Beijing Multi-Site Air Quality Data (https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data), doi 10.24432/C5RK5G. Air data come from the Beijing Municipal Environmental Monitoring Center and weather from the China Meteorological Administration; the donor is Song Chen.
- License: Creative Commons Attribution 4.0 ("sharing and adaptation of the datasets for any purpose, provided that the appropriate credit is given"). The TFB package adds no separate data license.
- Cite Cautionary tales on air-quality improvement in Beijing (Zhang et al., Proceedings of the Royal Society A, 2017) and TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods (Qiu et al., PVLDB 2024).
- `AQShunyi` is the Shunyi station (verified from the TFB file's source name `PRSA_Data_Shunyi_20130301-20170228.csv`).
- Redistribution: `hosted` with attribution (CC BY 4.0).

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Rows | 35,064 hourly steps | source-reported (TFB data file) |
| Channels | 11: PM2.5, PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM | source-reported |
| Span | 2013-03-01 00:00 to 2017-02-28 23:00 | source-reported |
| Missing values | UCI original: 8,523 NA cells at this station (all columns); TFB file: none | source-reported |

The repository neither ships nor pins this file (`dataset/` is local and the Hub has no published copy), so the numbers above are those of the standard public distribution and a different copy may differ. Run `tsf data inspect --config configs/datasets/aqshunyi.toml` on your copy before relying on them.

## Related datasets

- [`aqwan`](../aqwan/README.md): other Beijing station, same packaging
- [`weather`](../weather/README.md): other meteorological set
