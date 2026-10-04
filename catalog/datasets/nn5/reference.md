# nn5 — reference

## Provenance and license

- Competition data by Crone et al. (NN5 competition); republished by the Monash Time Series Forecasting Archive (https://zenodo.org/records/3889740, doi 10.5281/zenodo.3889740) under CC BY 4.0.
- TFB packaging: TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods (Qiu et al., PVLDB 2024), https://arxiv.org/abs/2403.20150; the TFB repository is MIT-licensed code with no separate data license.
- Redistribution: `hosted` under the Monash/Zenodo CC BY 4.0 copy; credit Crone et al. and the Monash archive.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Rows | 791 daily steps | source-reported (TFB data file) |
| Channels | 111 ATM series | source-reported |
| Dates | file dates 1996-03-18 to 1998-05-17 are nominal; the competition covers two years of daily data | source-reported |
| Missing values | none in the TFB file (imputed 'without missing values' version) | source-reported |

The repository neither ships nor pins this file (no published Hub copy), so these are the standard public distribution's numbers; a different copy may differ.

## Related datasets

- [`ili`](../ili/README.md): other short low-frequency set
- [`fred_md`](../fred_md/README.md): other short low-frequency set
- [`gift_eval/restaurant`](../gift_eval/restaurant/README.md): daily visitor counts at restaurants (GIFT-Eval)
