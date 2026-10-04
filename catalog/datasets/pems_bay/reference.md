# pems_bay — reference

## Provenance and license

- Producer: DCRNN authors packaged Caltrans PeMS (Bay Area) data; repository https://github.com/liyaguang/DCRNN (code MIT per GitHub API). The README gives only Google Drive / Baidu links for `pems-bay.h5` and states no data license. Caltrans PeMS Conditions of Use (https://pems.dot.ca.gov/?view=tou) say: "In general, information presented on this web site, unless otherwise indicated, is considered in the public domain", and that to use information "not owned or created by the State, you must seek permission directly from the owning (or holding) sources". The packagers add no terms of their own, so `redistribution` is `hosted` with credit to Caltrans PeMS.
- Cite: Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting (Li et al., ICLR 2018), https://arxiv.org/abs/1707.01926.
- Obtain `pems-bay.h5` and `adj_mx_bay.pkl` from the DCRNN README; TSFLab does not ship them.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Sensors | 325 | source-reported (DCRNN) |
| Steps | 52,116 at 5 minutes | source-reported (TFB Table 5; file not re-counted) |
| Span | paper: 2017-01-01 to 2017-05-31; step count implies to about 2017-06-30 | source-reported (conflicting) |
| Quantity | traffic speed | source-reported |

The preset reads a converted node bundle (`his.npz` with `data` shaped `(T, N, 3)`, `adj_mx.npy`, and `idx_train/val/test.npy`) produced by `tsf data prepare --from traffic`; the repository neither ships nor pins it, so the numbers above are those of the public distribution, and the converted bundle must be inspected before use.

## Related datasets

- [`metr_la`](../metr_la/README.md): other DCRNN speed benchmark
- [`pems04`](../pems04/README.md): flow benchmark for the same district (District 4) with the same loader
- [`rt/traffic_pems_ba`](../rt/traffic_pems_ba/README.md): hourly flow for the same Bay Area district (District 4), 2019-2023
