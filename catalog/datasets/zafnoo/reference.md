# zafnoo — reference

## Provenance and license

- Source: SAPFLUXNET, a global database of sap flow measurements (Poyatos et al., Global transpiration data from sap flow measurements: the SAPFLUXNET database, Earth System Science Data 13, 2021, https://doi.org/10.5194/essd-13-2607-2021); data on Zenodo, https://doi.org/10.5281/zenodo.3971689. TFB cites Poyatos et al. for this dataset.
- Site `ZAF_NOO` and plant `ZAF_NOO_E3_IRR_Mdo_Jt_2` are taken from the TFB channel name; the remaining channel names are SAPFLUXNET's environmental variable names. Which SAPFLUXNET release and timestamp convention TFB used is not documented, and the values were not cross-checked against SAPFLUXNET.
- License: the SAPFLUXNET Zenodo record is released under Creative Commons Attribution 4.0 International. TFB's MIT license covers code, not data.
- Packaging: TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods (Qiu et al., PVLDB 2024), https://arxiv.org/abs/2403.20150.
- File: TFB's pre-processed forecasting archive (Google Drive id `1vgpOmAygokoUt235piWKUjfwao6KwLv7`, linked from the TFB README; archive sha256 `01794d0000ccc7e033523481cfa117f647c9740d6efbf0626f56228f1bcf785c`), member `forecasting/ZafNoo.csv`, pivoted from TFB's long `date,data,cols` layout to a wide CSV in TFB's channel order with values and dates unchanged; channel names are TFB's. `dataset/ZafNoo/SOURCE.txt` records the member and converted-file hashes.
- Redistribution: `hosted` with attribution (CC BY 4.0).

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Rows | 19,225 half-hourly steps, no gaps | measured |
| Span | 2008-05-15 23:20:37 to 2009-06-20 11:20:37 | measured |
| Channels | 11: `ZAF_NOO_E3_IRR_Mdo_Jt_2` (sap flow of one plant) and `ta` (air temperature), `rh` (relative humidity), `sw_in` (incoming shortwave radiation), `ppfd_in` (photosynthetic photon flux density), `ws` (wind speed), `precip` (precipitation), `swc_shallow` and `swc_deep` (shallow and deep soil water content), `ext_rad` (extraterrestrial radiation), `vpd` (vapour pressure deficit) | measured |
| Timestamps | 30-minute steps at minute :20:37 / :50:37 (not aligned to the clock hour) | measured |
| Missing values | no NaN | measured |
| Sap-flow channel | 0 to 9,657.0, mean 448.0 | measured |
| Exact zeros | 34.7% of the sap-flow channel, 93.5% of `precip`; 23.4% of all values | measured |
| Negative values | 51 cells, all in `ta` (sub-zero air temperature, min -1.8) | measured |

Measured from the local file `dataset/ZafNoo/ZafNoo.csv` (read-only; converted from the TFB archive, sha256 in `SOURCE.txt`).

## Related datasets

- [`czelan`](../czelan/README.md): the other SAPFLUXNET site in TFB
- [`weather`](../weather/README.md): meteorological drivers at 10-minute resolution
- [`aqwan`](../aqwan/README.md): other environmental TFB set with weather channels
