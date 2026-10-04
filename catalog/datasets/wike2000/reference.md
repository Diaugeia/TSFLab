# wike2000 — reference

## Provenance and license

- Packaging: TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods (Qiu et al., PVLDB 2024), https://arxiv.org/abs/2403.20150. TFB cites Probabilistic Forecasting with Spline Quantile Function RNNs (Gasthaus et al., AISTATS 2019) for this dataset.
- Identified source: the file equals, value for value and in the same series order, the `train` split of GluonTS `wiki2000_nips` (https://github.com/awslabs/gluonts/raw/b89f203595183340651411a41eeb0ee60570a4d9/datasets/wiki2000_nips.tar.gz; 2,000 series, start 2012-01-01, 792 days). GluonTS documents it as one of the datasets of the GP-Copula paper (Salinas et al., NeurIPS 2019, https://arxiv.org/abs/1910.03002); the GluonTS test split extends the same series to 912 days.
- License: the file comes from the TFB archive, which repackages GluonTS `wiki2000_nips` (the GP-Copula datasets of Salinas et al., NeurIPS 2019: 2,000 Wikipedia pages, daily, 2012-01-01 to 2014-03-02; the date span rules out the 2015-2017 Kaggle web-traffic data). TFB (MIT code), GluonTS (Apache-2.0 code) and the GP-Copula release state no data license, so TSFLab ships the fetch script (`script`) rather than the file. The Wikimedia page counts underneath are CC0-1.0; a copy rebuilt from the Wikimedia dumps could be hosted.
- File: TFB's pre-processed forecasting archive (Google Drive id `1vgpOmAygokoUt235piWKUjfwao6KwLv7`, linked from the TFB README; archive sha256 `01794d0000ccc7e033523481cfa117f647c9740d6efbf0626f56228f1bcf785c`), member `forecasting/Wike2000.csv`, pivoted from TFB's long `date,data,cols` layout to a wide CSV in TFB's channel order with values and dates unchanged; the anonymous channels keep TFB's names 0 to 1998 and the last one (1999) is renamed `OT`. `dataset/Wike2000/SOURCE.txt` records the member and converted-file hashes.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Rows | 792 daily steps, no gaps | measured |
| Span | 2012-01-01 to 2014-03-02 | measured |
| Channels | 2,000 page-view series (`0` to `1998`, then `OT`) | measured |
| Values | non-negative integers, 0 to 7,752,515, median 2,467 | measured |
| Missing values | no NaN | measured |
| Zeros | exactly 2,000 cells, all on 2014-01-05 (every series is 0 that day) | measured |
| Target `OT` | 0 to 112,463 | measured |

Measured from the local file `dataset/Wike2000/Wike2000.csv` (read-only; converted from the TFB archive, sha256 in `SOURCE.txt`).

## Related datasets

- [`covid19`](../covid19/README.md): other high-dimensional short daily TFB set
- [`nn5`](../nn5/README.md): other short daily TFB set
- [`traffic`](../traffic/README.md): other high-dimensional usage set
