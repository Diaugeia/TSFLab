# fred_md — reference

## Provenance and license

- Source: FRED-MD by McCracken and Ng, https://www.stlouisfed.org/research/economists/mccracken/fred-databases; cite the 2016 JBES paper.
- License: checked the FRED-MD page (https://www.stlouisfed.org/research/economists/mccracken/fred-databases), which states no license, copyright or reuse terms for the CSV files, and FRED's legal terms (https://fred.stlouisfed.org/legal/), which say that before using series "owned by third parties for anything other than your own personal use, you must contact the data owner" and prohibit mirroring or scraping all of FRED; FRED-MD series are drawn from many original owners. FRED's terms forbid archiving or compiling FRED data and no FRED-MD-specific grant exists, so `redistribution` is `script`; do not publish copies.
- TFB packaging: https://arxiv.org/abs/2403.20150.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Rows | 728 monthly steps | source-reported (TFB data file) |
| Channels | 107 (of 134 in the paper) | source-reported |
| Span | 1959-01 to 2019-08 | source-reported |
| Missing values | none in the TFB file | source-reported |

The repository neither ships nor pins this file (`dataset/` is local and the Hub has no published copy), so the numbers above are those of the standard public distribution and a different copy may differ. Run `tsf data inspect --config configs/datasets/fred_md.toml` on your copy before relying on them.

## Related datasets

- [`ili`](../ili/README.md): other short low-frequency set
- [`nn5`](../nn5/README.md): other short set, daily
- [`exchange`](../exchange/README.md): other finance set
