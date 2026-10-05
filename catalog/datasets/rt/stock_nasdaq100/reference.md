# rt/stock_nasdaq100 — reference

## Provenance and license

- Source: the Nasdaq historical price API, with the Yahoo chart API and Sina as fallbacks (`docs/en/realtime.md`). Nasdaq and Yahoo return split-adjusted, not dividend-adjusted, closes; the Sina fallback is dividend-adjusted and can differ by the dividend yield on ex-dividend days.
- The Nasdaq and Yahoo terms forbid redistribution, so `redistribution` is `script`; only the Nasdaq-100 constituent list is open.
- Not redistributed: vendor terms forbid re-hosting, so TSFLab never uploads this panel (`tsf realtime update --push` skips it). Fetch it from the source with `tsf realtime update --bootstrap --track stock_nasdaq100` (README, Protocol and pitfalls).

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Transform | log(close_t / close_{t-1}) of split-adjusted closes | source-reported (config) |
| Bootstrap | full history from 2019-01-02 for the current constituents | source-reported (config) |
| Track config | `configs/realtime/stock_nasdaq100.toml`: freq B, seq_len 20, horizon 5, min_coverage 0.8, timezone America/New_York | source-reported (config) |

No local copy of `dataset/realtime/stock_nasdaq100` exists in a development checkout, so nothing is measured: row count, channel count, and missingness are unknown until a store is built (`tsf realtime update --bootstrap --track stock_nasdaq100`; the Hub holds no copy) and the card re-measured. The preset is therefore not pinned to a release.

## Related datasets

- [`rt/stock_sp500`](../stock_sp500/README.md): broader US universe
- [`rt/stock_hs300`](../stock_hs300/README.md): China equivalent
