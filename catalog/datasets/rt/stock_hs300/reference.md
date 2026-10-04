# rt/stock_hs300 — reference

## Provenance and license

- Source: the AKShare Python library (https://github.com/akfamily/akshare, MIT-licensed code) fetching daily prices from third-party vendors (Eastmoney, Sina Finance, and others); AKShare states its data are for academic research. The per-endpoint vendor was not traced.
- Data license: the Eastmoney and Sina terms forbid redistribution (MIT covers AKShare's code, not the market data), so `redistribution` is `script`; only the CSI 300 constituent list is open.
- The weekly workflow mirrors releases to the **private** Hugging Face dataset `Diaugeia/TSFLab-RealTime` (`stock_hs300/`) for team use only. Vendor terms forbid public re-hosting, so that repository must stay private; outside the team, build the store locally (README, Protocol and pitfalls).

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Transform | log(close_t / close_{t-1}) of forward-adjusted closes | source-reported (config) |
| Bootstrap | full history from 2019-01-02 for the current constituents | source-reported (config) |
| Track config | `configs/realtime/stock_hs300.toml`: freq B, seq_len 20, horizon 5, min_coverage 0.8, timezone Asia/Shanghai | source-reported (config) |

No local copy of `dataset/realtime/stock_hs300` exists in a development checkout, so nothing is measured: row count, channel count, and missingness are unknown until a store is provided (or `dataset.params.revision` pulls one from the Hub) and the card re-measured. The preset is therefore not pinned to a release.

## Related datasets

- [`exchange`](../../exchange/README.md): other finance set, static
- [`rt/stock_nasdaq100`](../stock_nasdaq100/README.md): US large-cap equivalent
