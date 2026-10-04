# pems07 — reference

## Provenance and license

- Producer: Caltrans PeMS District 7 data, processed by Song et al. (STSGCN, https://github.com/Davidham3/STSGCN) and Guo et al. (ASTGCN, https://github.com/Davidham3/ASTGCN, AAAI 2019). Neither repository carries a license (GitHub API reports none for STSGCN; ASTGCN redirects to a repository without one), so the packaged files have no stated license. Caltrans PeMS Conditions of Use (https://pems.dot.ca.gov/?view=tou) say: "In general, information presented on this web site, unless otherwise indicated, is considered in the public domain", and that to use information "not owned or created by the State, you must seek permission directly from the owning (or holding) sources". The packagers add no terms of their own, so `redistribution` is `hosted` with credit to Caltrans PeMS.
- Cite Spatial-Temporal Synchronous Graph Convolutional Networks (Song et al., AAAI 2020), https://ojs.aaai.org/index.php/AAAI/article/view/5438, and the ASTGCN paper (https://ojs.aaai.org/index.php/AAAI/article/view/3881).
- TSFLab does not ship the data (STSGCN distributes the files through Baidu Pan; the ASTGCN authors' `guoshnBJTU/ASTGNN` repository (commit 9c2e19b) hosts the same `PEMS0X.npz` and distance `.csv` files in git); convert with `tsf data prepare --from traffic`.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Sensors | 883 | source-reported (STSGCN Table 1) |
| Span | 2017-05-01 to 2017-08-31 | source-reported (STSGCN) |
| Steps | 28,224 (98 days; the stated span would imply 35,424) | measured (`PEMS07.npz`, guoshnBJTU/ASTGNN@9c2e19b) |
| Frequency | 5 minutes (12 per hour) | source-reported |
| Quantity | traffic flow (the ASTGCN files for PEMS04/08 also carry occupancy and speed; STSGCN and this preset use flow only) | source-reported |

The preset reads a converted node bundle (`his.npz` with `data` shaped `(T, N, 3)`, `adj_mx.npy`, and `idx_train/val/test.npy`) produced by `tsf data prepare --from traffic`; the repository neither ships nor pins it, so the numbers above are those of the public distribution, and the converted bundle must be inspected before use.

## Related datasets

- [`pems03`](../pems03/README.md): sibling STSGCN flow benchmark
- [`pems04`](../pems04/README.md): sibling STSGCN flow benchmark
- [`metr_la`](../metr_la/README.md): DCRNN speed benchmark with the same loader
