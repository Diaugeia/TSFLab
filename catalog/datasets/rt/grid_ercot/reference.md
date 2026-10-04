# rt/grid_ercot — reference

## Provenance and license

- Source: ERCOT public load data (`Native_Load` archives, refreshed monthly, and the MIS report NP6-345-CD). The ERCOT terms of use (https://www.ercot.com/help/terms) allow re-hosting unmodified data that credit ERCOT, so `redistribution` is `hosted` with `conditions`.
- Releases are mirrored to the Hugging Face dataset `Diaugeia/TSFLab-RealTime` (`grid_ercot/`).

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Channels | 8 weather zones | source-reported (config) |
| Bootstrap | `Native_Load` archives from 2019-01-01 | source-reported (config) |
| Track config | `configs/realtime/grid_ercot.toml`: freq h, seq_len 168, horizon 24, min_coverage 0.9, timezone America/Chicago | source-reported (config) |

No local copy of `dataset/realtime/grid_ercot` exists in a development checkout, so nothing is measured: row count, channel count, and missingness are unknown until a store is provided (or `dataset.params.revision` pulls one from the Hub) and the card re-measured. The preset is therefore not pinned to a release.

## Related datasets

- [`electricity`](../../electricity/README.md): static electricity-consumption set
