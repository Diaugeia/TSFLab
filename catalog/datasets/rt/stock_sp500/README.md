---
name: "rt/stock_sp500"
description: "Daily log returns of about 500 S&P 500 constituents from 2019-01-02, frozen release of real-time track stock_sp500; seq_len 20, horizon 5. Use for daily return forecasting on an equity panel (zero is a strong baseline); not for price levels or survivorship-free backtests."
---

# rt/stock_sp500

## Overview

`rt_stock_sp500` freezes the S&P 500 real-time track as a static dataset: one channel per constituent at bootstrap (about 500), each the daily log return log(close_t / close_(t-1)) from the Nasdaq historical price API (Yahoo fallback), split-adjusted closes, on a business-day grid from 2019-01-02. It is the widest stock panel in the repository. The panel store is read and split chronologically 7:1:2.

## Protocol and pitfalls

- **Get the data (`script`).** Vendor terms forbid re-hosting the prices. Build the store from the vendors with `uv run tsf realtime update --bootstrap --track stock_sp500`; it is written to `dataset/realtime/stock_sp500`, where this preset reads it.
- **Survivorship bias.** The panel is built from the constituents at bootstrap, so the history contains only stocks that were in the index then; static results look better than live ones.
- **Returns, not prices.** Values are daily log returns with a near-zero mean and heavy tails; a zero forecast is a strong baseline and MSE is dominated by volatile days.
- **Holidays and suspensions.** Exchange holidays and halts are unobserved cells on a business-day grid; the loader forward-fills them, which turns a missing return into a repeated one rather than a zero.
- **Frozen snapshot, not the live track.** No `version` is pinned, so the preset reads whatever the local panel store holds; pin a release or set `revision` for a reproducible study. The live track's rolling weekly rounds (`docs/en/realtime.md`) remain the contamination-free evaluation; a static split says nothing about data that arrive later.
- **Gap filling.** Unobserved cells are forward-filled; only a series' leading gap is back-filled from its first reading. The loader fills the whole panel before splitting, so a channel that starts late carries a constant first value in training; fully empty channels become zero.
- **Scaling.** One scalar mean and standard deviation from the training rows is shared by all channels, as in the real-time export, so large channels dominate raw-unit MSE.
- **`drop_last`.** Loaders keep the last partial batch for every split.
