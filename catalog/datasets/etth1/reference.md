# etth1 — reference

## Provenance and license

- Producer: Haoyi Zhou et al. (Beihang University) with Beijing Guowang Fuda Science & Technology Development Co.; repository https://github.com/zhouhaoyi/ETDataset.
- Cite: Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting (Zhou et al., AAAI 2021), https://arxiv.org/abs/2012.07436. The upstream README asks users to cite Informer.
- License: Creative Commons Attribution-NoDerivatives 4.0 International (the repository's LICENSE file). Verbatim copies with attribution are allowed; a modified, re-split, or re-packaged copy is a derivative and is not covered, so publish only unmodified files. The Informer code repository is Apache-2.0 (separate from the data).
- Hub: `tsf data download --list` shows whether this preset is published; check the license before running `tsf data publish`.
- Redistribution: `hosted` under the `conditions` in `card.toml`: re-host only verbatim copies (a CSV-to-parquet format shift is allowed) with attribution; never cleaned, cropped, or re-split variants.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| CSV rows | 17,420 | measured |
| Rows used by the loader | 14,400 (to 2018-02-20 23:00:00) | measured (loader code) |
| Channels | 7: HUFL, HULL, MUFL, MULL, LUFL, LULL, OT | measured |
| Time span (CSV) | 2016-07-01 00:00:00 to 2018-06-26 19:00:00 | measured |
| Step | hourly (all 17,419 gaps equal; no duplicates) | measured |
| Missing values | 0 | measured |
| Exact zeros | 1.00% of all CSV values | measured |
| OT mean / std (used rows) | 14.363 / 8.969 | measured |
| OT range (CSV) | -4.08 to 46.01 | measured |
| Split rows (6:2:2) | train 8,640, validation 2,880, test 2,880 | measured (loader code) |
| Provenance statement | two stations in two regions of one Chinese province; roughly two years | source-reported (upstream README, Informer paper) |

Measured on `dataset/ETT-small/ETTh1.csv` (read-only).

## Related datasets

- [`ettm1`](../ettm1/README.md): same station at the other resolution
- [`etth2`](../etth2/README.md): other station, same resolution
- [`gift_eval/ett1_H`](../gift_eval/ett1_H/README.md): GIFT-Eval resampling of the full series, different protocol
