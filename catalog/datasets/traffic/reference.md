# traffic — reference

## Provenance and license

- Source system: Caltrans Performance Measurement System (PeMS), http://pems.dot.ca.gov. Caltrans PeMS Conditions of Use (https://pems.dot.ca.gov/?view=tou) say: "In general, information presented on this web site, unless otherwise indicated, is considered in the public domain", and that to use information "not owned or created by the State, you must seek permission directly from the owning (or holding) sources". The packagers add no terms of their own, so `redistribution` is `hosted` with credit to Caltrans PeMS.
- Packaging: Lai et al. 2018, https://github.com/laiguokun/multivariate-time-series-data, which has no license file (GitHub API: none) and no data terms in its README; the preprocessed copy therefore carries no license of its own. The PeMS public-domain policy applies, so the LSTNet copy is `hosted` with credit to Caltrans PeMS.
- Cite LSTNet (SIGIR 2018) and credit Caltrans PeMS.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Rows | 17,544 | measured |
| Channels | 862 sensors (columns named 1 to 862; no `OT` column) | measured |
| Time span | 2015-01-01 00:00:01 to 2016-12-31 23:00:01, step 1 hour, no gaps | measured |
| Missing values | 0 | measured |
| Value range | 0.0000 to 0.7240, mean 0.0567, std 0.0540 | measured |
| Exact zeros | 0.90% of all values | measured |
| Quantity | occupancy rate in [0, 1]; the 2015-2016 span is 731 days | source-reported (LSTNet) |

Measured on `dataset/traffic/traffic.csv` (read-only).

## Related datasets

- [`electricity`](../electricity/README.md): other hourly LTSF set with hundreds of channels
- [`pems04`](../pems04/README.md): 5-minute PeMS flow graph with adjacency
- [`rt/traffic_pems_ba`](../rt/traffic_pems_ba/README.md): hourly PeMS flow per station, District 4, 2019-2023
