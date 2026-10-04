# covid19 — reference

## Provenance and license

- Packaging: TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods (Qiu et al., PVLDB 2024), https://arxiv.org/abs/2403.20150. TFB cites Transfer Graph Neural Networks for Pandemic Forecasting (Panagopoulos et al., AAAI 2021) for this dataset but does not name the raw data source.
- The channel names (`<Country>;New_cases`, `Cumulative_cases`, `New_deaths`, `Cumulative_deaths`) and the 2020-01-03 start match the WHO COVID-19 global data CSV schema (`Date_reported, Country_code, Country, WHO_region, New_cases, Cumulative_cases, New_deaths, Cumulative_deaths`, https://data.who.int/dashboards/covid19/data). This attribution is inferred from the schema; values were not cross-checked because WHO revises counts retrospectively.
- License: data.who.int states that its datasets are provided under Creative Commons Attribution 4.0 International unless indicated otherwise, with WHO's terms of use (no misrepresentation, no warranty). TFB's MIT license covers code, not data. `redistribution` is `hosted` with `conditions`: attribute WHO and follow the additional WHO terms (https://data.who.int/dashboards/covid19/more-resources).
- File: TFB's pre-processed forecasting archive (Google Drive id `1vgpOmAygokoUt235piWKUjfwao6KwLv7`, linked from the TFB README; archive sha256 `01794d0000ccc7e033523481cfa117f647c9740d6efbf0626f56228f1bcf785c`), member `forecasting/Covid-19.csv`, pivoted from TFB's long `date,data,cols` layout to a wide CSV in TFB's channel order with values and dates unchanged; channel names are TFB's. `dataset/Covid-19/SOURCE.txt` records the member and converted-file hashes.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Rows | 1,392 daily steps, no gaps | measured |
| Span | 2020-01-03 to 2023-10-25 | measured |
| Channels | 948 = 237 countries/territories x 4 metrics, ordered by metric (all New_cases, then Cumulative_cases, New_deaths, Cumulative_deaths), countries alphabetical within a metric | measured |
| Missing values | no NaN | measured |
| Exact zeros | 36.5% of all values; New_cases 51.0%, New_deaths 70.8%, Cumulative_cases 8.7%, Cumulative_deaths 15.6% | measured |
| All-zero channels | 20 (e.g. Holy See, Niue, Pitcairn Islands, Tokelau, Turkmenistan, Democratic People's Republic of Korea) | measured |
| Negative values | 99 cells (65 in New_cases, 34 in New_deaths); cumulative channels are never negative | measured |
| Target channel | Zimbabwe;Cumulative_deaths: 0 to 5,720, median 4,705 | measured |

Measured from the local file `dataset/Covid-19/Covid-19.csv` (read-only; converted from the TFB archive, sha256 in `SOURCE.txt`). The repository neither ships nor publishes this file; run `tsf data inspect --config configs/datasets/covid19.toml` on your copy before relying on these numbers.

## Related datasets

- [`ili`](../ili/README.md): other short epidemiological set
- [`nn5`](../nn5/README.md): other short daily TFB set
- [`wike2000`](../wike2000/README.md): other high-dimensional short daily TFB set
