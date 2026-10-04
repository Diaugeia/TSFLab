# rt/stock_sp500 — reference

## Provenance and license

- Source: the constituent list comes from `datasets/s-and-p-500-companies` (ODC-PDDL per `docs/en/realtime.md`), prices from the Nasdaq historical API, with the Yahoo chart API as fallback (both split-adjusted, not dividend-adjusted).
- The Nasdaq and Yahoo terms forbid redistribution of the prices, so `redistribution` is `script`; only the constituent list (ODC-PDDL) is open.
- The weekly workflow mirrors releases to the **private** Hugging Face dataset `Diaugeia/TSFLab-RealTime` (`stock_sp500/`) for team use only. Vendor terms forbid public re-hosting, so that repository must stay private; outside the team, build the store locally (README, Protocol and pitfalls).

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Transform | log(close_t / close_{t-1}) of split-adjusted closes | source-reported (config) |
| Bootstrap | full history from 2019-01-02 for the current constituents | source-reported (config) |
| Track config | `configs/realtime/stock_sp500.toml`: freq B, seq_len 20, horizon 5, min_coverage 0.8, timezone America/New_York | source-reported (config) |

No local copy of `dataset/realtime/stock_sp500` exists in a development checkout, so nothing is measured: row count, channel count, and missingness are unknown until a store is provided (or `dataset.params.revision` pulls one from the Hub) and the card re-measured. The preset is therefore not pinned to a release.

## Related datasets

- [`rt/stock_nasdaq100`](../stock_nasdaq100/README.md): large-cap subset of the US universe
- [`rt/stock_hs300`](../stock_hs300/README.md): China equivalent
