# metr_la — reference

## Provenance and license

- Producer: DCRNN authors (USC and Caltech) packaged Los Angeles County loop-detector data (Jagadish et al., 2014); repository https://github.com/liyaguang/DCRNN (code MIT per GitHub API). Checked: the DCRNN README (data only via Google Drive / Baidu Yun, no data terms) and the repository license. No terms from the original LA County / Caltrans provider were found for this extract (unlike PEMS-BAY, it is not documented as a PeMS download), so `redistribution` is `script` (the DCRNN MIT license covers code, not data).
- Cite: Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting (Li et al., ICLR 2018), https://arxiv.org/abs/1707.01926.
- Obtain `metr-la.h5` and `adj_mx.pkl` from the DCRNN README; TSFLab does not ship them (see `tsf data prepare --from traffic --help`).

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Sensors | 207 | source-reported (DCRNN) |
| Steps | 34,272 at 5 minutes (= 119 days) | source-reported (TFB data file) |
| Span | 2012-03-01 to 2012-06-27 23:55 (paper: to 06-30; conflicting) | source-reported |
| Quantity | traffic speed | source-reported |
| Missing values | zeros mark missing in the original h5; none as NaN in the TFB copy | source-reported |

The preset reads a converted node bundle (`his.npz` with `data` shaped `(T, N, 3)`, `adj_mx.npy`, and `idx_train/val/test.npy`) produced by `tsf data prepare --from traffic`; the repository neither ships nor pins it, so the numbers above are those of the public distribution, and the converted bundle must be inspected before use.

## Related datasets

- [`pems_bay`](../pems_bay/README.md): other DCRNN speed benchmark
- [`pems04`](../pems04/README.md): flow benchmark with the same loader
- [`traffic`](../traffic/README.md): hourly LTSF traffic set without a graph
