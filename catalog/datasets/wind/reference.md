# wind — reference

## Provenance and license

- Source: PaddleSpatial `paddlespatial/datasets/WindPower/wind.csv` (https://github.com/PaddlePaddle/PaddleSpatial/tree/main/paddlespatial/datasets/WindPower, last changed at commit a8e3053936696ec71f8ae8cfc0de7fe6eb1693cb), described in the D3VAE README as "a collected WindPower dataset"; TFB cites Generative Time Series Forecasting with Diffusion, Denoise, and Disentanglement (Li et al., NeurIPS 2022) for it. The wind farm, its location and the NWP provider are not documented.
- Cross-check: the converted TFB file equals PaddleSpatial's `wind.csv` (same 48,673 rows, columns and timestamps; values identical up to float formatting, max difference 2.8e-14).
- License (checked 2026-10-03): the data are released under CC BY 4.0 as the SDWPF dataset (https://figshare.com/articles/dataset/SDWPF_dataset/24798654), so `redistribution` is `hosted` with attribution; the PaddleSpatial repository code is Apache-2.0. TFB's MIT license covers code, not data.
- Packaging: TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods (Qiu et al., PVLDB 2024), https://arxiv.org/abs/2403.20150.
- File: TFB's pre-processed forecasting archive (Google Drive id `1vgpOmAygokoUt235piWKUjfwao6KwLv7`, linked from the TFB README; archive sha256 `01794d0000ccc7e033523481cfa117f647c9740d6efbf0626f56228f1bcf785c`), member `forecasting/Wind.csv`, pivoted from TFB's long `date,data,cols` layout to a wide CSV in TFB's channel order with values and dates unchanged; channel names are TFB's (and PaddleSpatial's), including the original spelling `ture_w_speed`. `dataset/Wind/SOURCE.txt` records the member and converted-file hashes.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Rows | 48,673 steps at 15 minutes, no gaps | measured |
| Span | 2020-01-01 00:00 to 2021-05-22 00:00 (date strings like `2020/1/1 0:00`) | measured |
| Channels | 7: `pred_w_speed`, `pred_w_dir`, `pred_temp`, `pred_pressure`, `pred_humidity` (forecasts), `ture_w_speed` (measured wind speed), `target` (power) | measured |
| `target` | -0.95 to 98.62, mean 20.28; negative in 6,050 steps, exactly 0 in 228 | measured |
| Forecast channels | wind speed 0 to 20.1, direction 0 to 359, temperature -16.4 to 34.9, pressure 997 to 1,038, humidity 14 to 100 | measured |
| Missing values | no NaN | measured |

Measured from the local file `dataset/Wind/Wind.csv` (read-only; converted from the TFB archive, sha256 in `SOURCE.txt`).

## Related datasets

- [`solar`](../solar/README.md): other renewable-power set
- [`weather`](../weather/README.md): meteorological drivers
- [`etth1`](../etth1/README.md): other energy set at sub-daily resolution
