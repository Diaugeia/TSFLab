# czelan — reference

## Provenance and license

- Source: SAPFLUXNET, a global database of sap flow measurements (Poyatos et al., Global transpiration data from sap flow measurements: the SAPFLUXNET database, Earth System Science Data 13, 2021, https://doi.org/10.5194/essd-13-2607-2021); data on Zenodo, https://doi.org/10.5281/zenodo.3971689. TFB cites Poyatos et al. for this dataset.
- Site `CZE_LAN` and plant `CZE_LAN_Cbe_Jt_16` are taken from the TFB channel name; the remaining channel names are SAPFLUXNET's environmental variable names. Which SAPFLUXNET release and timestamp convention TFB used is not documented, and the values were not cross-checked against SAPFLUXNET.
- License: the SAPFLUXNET Zenodo record is released under Creative Commons Attribution 4.0 International. TFB's MIT license covers code, not data.
- Packaging: TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods (Qiu et al., PVLDB 2024), https://arxiv.org/abs/2403.20150.
- File: TFB's pre-processed forecasting archive (Google Drive id `1vgpOmAygokoUt235piWKUjfwao6KwLv7`, linked from the TFB README; archive sha256 `01794d0000ccc7e033523481cfa117f647c9740d6efbf0626f56228f1bcf785c`), member `forecasting/CzeLan.csv`, pivoted from TFB's long `date,data,cols` layout to a wide CSV in TFB's channel order with values and dates unchanged; channel names are TFB's. `dataset/CzeLan/SOURCE.txt` records the member and converted-file hashes.
- Redistribution: `hosted` with attribution (CC BY 4.0).

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Rows | 19,934 half-hourly steps, no gaps | measured |
| Span | 2016-05-06 00:11:03 to 2017-06-25 06:41:03 | measured |
| Channels | 11: `CZE_LAN_Cbe_Jt_16` (sap flow of one plant) and `ta` (air temperature), `rh` (relative humidity), `sw_in` (incoming shortwave radiation), `ppfd_in` (photosynthetic photon flux density), `ws` (wind speed), `precip` (precipitation), `swc_shallow` and `swc_deep` (shallow and deep soil water content), `ext_rad` (extraterrestrial radiation), `vpd` (vapour pressure deficit) | measured |
| Timestamps | 30-minute steps at minute :11:03 / :41:03 (not aligned to the clock hour) | measured |
| Missing values | no NaN | measured |
| Sap-flow channel | 0 to 16,019.3, mean 3,383.9 | measured |
| Exact zeros | 18.5% of the sap-flow channel, 97.4% of `precip`; 22.3% of all values | measured |
| Negative values | none | measured |

Measured from the local file `dataset/CzeLan/CzeLan.csv` (read-only; converted from the TFB archive, sha256 in `SOURCE.txt`). The repository neither ships nor publishes this file; run `tsf data inspect --config configs/datasets/czelan.toml` on your copy before relying on these numbers.

## Related datasets

- [`zafnoo`](../zafnoo/README.md): the other SAPFLUXNET site in TFB
- [`weather`](../weather/README.md): meteorological drivers at 10-minute resolution
- [`aqshunyi`](../aqshunyi/README.md): other environmental TFB set with weather channels
