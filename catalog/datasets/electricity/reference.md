# electricity — reference

## Provenance and license

- Raw data: Artur Trindade, UCI Machine Learning Repository, ElectricityLoadDiagrams20112014 (https://doi.org/10.24432/C58C86), license CC BY 4.0 ("sharing and adaptation of the datasets for any purpose, provided that the appropriate credit is given").
- Preprocessing: Lai et al. 2018 (LSTNet), https://github.com/laiguokun/multivariate-time-series-data. That repository carries no license file, so the preprocessed copy is derived from CC BY data and must credit Trindade.
- Cite LSTNet and the UCI dataset. UCI reports kW per 15 minutes while the LSTNet README says kWh, so the unit of this file is ambiguous; treat values as relative load.
- Redistribution: `hosted` with attribution (CC BY 4.0, UCI dataset 321); the LSTNet preprocessing adds no terms of its own. Credit Trindade and UCI.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Rows | 26,304 | measured |
| Channels | 321 (columns named 1 to 321; no `OT` column) | measured |
| Time span | 2012-01-01 00:00:01 to 2014-12-31 23:00:01, step 1 hour, no gaps or duplicates | measured |
| Missing values | 0 | measured |
| Value range | 0 to 764000, mean 2539, std 15028 | measured |
| Exact zeros | 1.09% of all values; clients mostly zero: 106, 108, 183, 299 | measured |
| Raw UCI data | 370 clients, 15-minute, 2011-2014 | source-reported (UCI) |

Measured on `dataset/electricity/electricity.csv` (read-only).

## Related datasets

- [`traffic`](../traffic/README.md): other 800+ channel hourly LTSF benchmark
- [`weather`](../weather/README.md): other standard 7:1:2 LTSF set
- [`solar`](../solar/README.md): other periodic energy benchmark
- [`gift_eval/electricity_H`](../gift_eval/electricity_H/README.md): GIFT-Eval electricity (370 clients), different protocol
