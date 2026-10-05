---
name: "ASTGCN"
description: "Spatial and temporal attention modulating Chebyshev graph convolution plus gated temporal conv; recent branch only. Use for traffic-style sensor forecasting on a known road graph; not for data without a graph or for daily/weekly multi-branch setups."
---

# ASTGCN

## Idea

- `SpatialTemporalAttention` learns dense node-by-node and step-by-step attention matrices from each block's input.
- `AttentionChebyshevConvolution` multiplies each Chebyshev support (`graph_spectral`) elementwise by the spatial attention before propagating features.
- `ASTGCNBlock` adds a gated (tanh x sigmoid) temporal convolution, a residual projection and `LayerNorm`.
- Only the recent-history branch is implemented; a final convolution treats history steps as channels to emit the horizon.

## When to use

- Traffic flow or similar sensor networks with a given adjacency and dynamic spatial-temporal correlations.
- Needs `adj_mx` (PEMS and METR-LA presets ship it); construction fails without it.
- The paper's daily- and weekly-periodic branches are absent, so periodicity is only captured within the lookback.
- Dense node attention is quadratic in the node count; point output only.

## Configure

- `enc_in`: number of nodes.
- `cov_dim`: covariate features per node (calendar or node-structured marks).
- `adj_mx`: `[enc_in, enc_in]` adjacency, turned into Chebyshev supports of order `K`.

Other hyperparameters: preset defaults in `configs/models/ASTGCN.toml`; tune generically.

## Differences

Clean-room implementation from the paper's attention, Chebyshev-filter, temporal-convolution and projection equations; neither the unlicensed official code nor an earlier CauAir-derived implementation was used. Only the recent-history component is implemented because the runtime batch has no separate daily and weekly windows, so three-branch fusion, the paper's traffic preprocessing, and the masked training objective are omitted. Raw calendar stamps or node-structured marks go through the shared marks contract.
