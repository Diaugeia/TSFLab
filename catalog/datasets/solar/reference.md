# solar — reference

## Provenance and license

- Producer: NREL (the site now resolves under nlr.gov), Solar Power Data for Integration Studies, https://www.nlr.gov/grid/solar-power-data. The page states no license; it warns that the data are for specific years and are not representative of typical radiation levels or for site-specific project development.
- Packaging: Lai et al. 2018 (LSTNet), https://github.com/laiguokun/multivariate-time-series-data (no license file per GitHub API). Checked the NREL/NLR data page (no license, terms, or citation statement) and the NLR disclaimer (https://www.nlr.gov/disclaimer.html), which says only that NLR-authored documents are DOE-sponsored, that the Government keeps a license to reproduce and distribute them, and that use of documents "may be subject to U.S. and foreign Copyright Laws"; it grants no dataset license. The NREL data use disclaimer applies, so `redistribution` is `hosted` with `conditions`: credit DOE/NREL, include the disclaimer, and imply no endorsement.
- Contradiction to be aware of: NREL describes simulated 5-minute data, while the LSTNet README calls the series records sampled every 10 minutes.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Rows | 52,560 (= 365 days x 144 steps) | measured |
| Channels | 137 plants (headerless text file, no date column) | measured |
| Missing values | 0 | measured |
| Value range | 0.0 to 88.9, mean 6.353, std 10.151 | measured |
| Exact zeros | 55.1% of all values; no negative values | measured |
| Night hours (>= 99% zeros, hour of day from row index) | 0, 1, 2, 3, 4, 19, 20, 21, 22, 23 | measured |
| Zero fraction at hours 5 / 6 / 7 / 18 / 19 | 0.86 / 0.45 / 0.10 / 0.96 / 1.00 | measured |
| Span and origin | calendar year 2006, Alabama, simulated PV output | source-reported (LSTNet, NREL) |

Measured on `dataset/solar/solar.txt` (read-only); hour of day is derived assuming row 0 is 00:00.

## Related datasets

- [`electricity`](../electricity/README.md): other periodic energy set, 7:1:2
- [`gift_eval/solar_10T`](../gift_eval/solar_10T/README.md): GIFT-Eval packaging of the same plants, different protocol
