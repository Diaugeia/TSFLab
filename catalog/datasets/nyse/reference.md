# nyse — reference

## Provenance and license

- Source: Temporal Relational Ranking for Stock Prediction (Feng et al., ACM TOIS 2019), repository https://github.com/fulifeng/Temporal_Relational_Stock_Ranking (inspected at commit cfbb01bdf194b81bc5893a1b37aff1c0d0d2a82d); raw end-of-day prices were collected from Google Finance. TFB cites Feng et al. for this dataset.
- Identified series: every channel equals rows 1 to 1243 of `data/2013-01-01/NYSE_CHS_1.csv`, columns 1 to 5, exactly (found by matching all NYSE files; TFB does not name the stock). Per the repository's `preprocess/eod.py`, those columns are the 5/10/20/30-day moving averages of the close and the close, divided by the maximum close; TFB's Open/High/Low/Close/Volume names are therefore misleading.
- License: the Feng et al. repository is AGPL-3.0 for its code; the Google Finance prices carry no redistribution right, so `redistribution` is `script`. TFB's MIT license covers code, not data.
- Packaging: TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods (Qiu et al., PVLDB 2024), https://arxiv.org/abs/2403.20150.
- File: TFB's pre-processed forecasting archive (Google Drive id `1vgpOmAygokoUt235piWKUjfwao6KwLv7`, linked from the TFB README; archive sha256 `01794d0000ccc7e033523481cfa117f647c9740d6efbf0626f56228f1bcf785c`), member `forecasting/NYSE.csv`, pivoted from TFB's long `date,data,cols` layout to a wide CSV in TFB's channel order with values and dates unchanged; TFB's long file carries per-channel date labels that are rotated by one step per block, and, like TFB's own reader, the conversion takes dates from the first block and values by position (verified against the Feng et al. file). `dataset/NYSE/SOURCE.txt` records the member and converted-file hashes.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Rows | 1,243 trading days | measured |
| Date column | row numbers 1.0 to 1243.0 (float), not calendar dates | measured |
| Channels | 5: Open, High, Low, Close, Volume (TFB names) = MA5, MA10, MA20, MA30 of close, and close, each / max close | measured (matched to the Feng et al. file) |
| Value range | 0.362 to 1.000 (Volume max 1.0 by construction); Open 0.371 to 0.994 | measured |
| Missing values | no NaN, no zeros | measured |

Measured from the local file `dataset/NYSE/NYSE.csv` (read-only; converted from the TFB archive, sha256 in `SOURCE.txt`).

## Related datasets

- [`nasdaq`](../nasdaq/README.md): the NASDAQ stock from the same source
- [`exchange`](../exchange/README.md): other daily finance set
- [`rt/stock_sp500`](../rt/stock_sp500/README.md): frozen real-time US stock panel
