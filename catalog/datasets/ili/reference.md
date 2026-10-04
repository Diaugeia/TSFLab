# ili — reference

## Provenance and license

- Source: US CDC FluView ILINet, https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html (the dashboard itself states no dataset license). CDC's Use of Agency Materials policy (https://www.cdc.gov/other/agencymaterials.html) says: "Most of the information on the CDC and ATSDR websites is not subject to copyright, is in the public domain, and may be freely used or reproduced without obtaining copyright permission", subject to attribution to CDC, a disclaimer that CDC does not endorse the use, and no change to the substantive content; some pages carry third-party or state/local material that may be copyrighted. Applying that to ILINet is an inference from CDC's general policy, not a dataset license.
- Packaging: Wu et al. 2021 (Autoformer), https://arxiv.org/abs/2106.13008; repository https://github.com/thuml/Autoformer is MIT (code), and the data files state no separate terms. `redistribution` is `hosted`: ILINet is a US Government work in the public domain; credit CDC FluView.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Rows (standard file) | 966 weekly steps | source-reported (TFB data file) |
| Channels | 7: % WEIGHTED ILI, %UNWEIGHTED ILI, AGE 0-4, AGE 5-24, ILITOTAL, NUM. OF PROVIDERS, OT | source-reported |
| Span | file dates 2002-01-01 to 2020-06-30; Autoformer text says 2002-2021 | source-reported (conflicting) |
| Missing values | none in the standard file | source-reported |

The repository neither ships nor pins this file (no published Hub copy), so these are the standard public distribution's numbers; a different copy may differ.

## Related datasets

- [`exchange`](../exchange/README.md): other small non-seasonal LTSF set
- [`fred_md`](../fred_md/README.md): other short low-frequency macro set
- [`nn5`](../nn5/README.md): short daily ATM set
