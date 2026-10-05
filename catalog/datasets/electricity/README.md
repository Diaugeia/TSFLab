---
name: "electricity"
description: "Hourly consumption of 321 Portuguese clients, 2012-2014 (LSTNet/Autoformer preprocessing of the UCI load diagrams). Use for high-dimensional long-horizon forecasting with strong daily and weekly seasonality; not for few-channel studies or unscaled-error comparisons (client scales differ by orders of magnitude)."
---

# electricity

## Overview

Electricity (ECL) holds the electricity consumption of 321 clients of a Portuguese utility, aggregated to hourly values for 2012-2014. The raw UCI data (15-minute readings for 370 clients, 2011-2014) was reduced to 321 clients and hourly steps by LSTNet, and Informer, Autoformer, and Time-Series-Library carried that version forward. Each client is a channel, so it is the standard high-dimensional multivariate benchmark (321 channels) with strong daily and weekly seasonality and very different client scales.

## Protocol and pitfalls

- **Split.** TSFLab uses a chronological 7:1:2 split for this dataset. Results reported under other splits (6:2:2) are not directly comparable. Scaling statistics come from the training rows only.
- **Target column.** This copy names the clients `1` to `321` and has no `OT` column, so the preset targets `321`, the last client, which is the column the TSLib copy calls `OT`. `features = "M"` uses all 321 clients; `"S"` and `"MS"` forecast client `321`.
- **Scale heterogeneity.** Client magnitudes span several orders (maximum 764,000 versus mean 2,539), so per-channel z-scoring is essential and a few clients dominate unscaled errors.
- **Zero blocks.** Four clients are mostly zero and one client only starts after row 160 (measured); a zero is not always a measurement.
- **Timestamps** carry a one-second offset (`00:00:01`); calendar features are unaffected.
- **`drop_last`.** Training drops its last partial batch; validation and test keep it, so test metrics cover every window.
