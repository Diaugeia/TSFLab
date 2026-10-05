---
name: "AirFormer"
description: "Air-quality Transformer: causal windowed temporal attention, dartboard regional spatial attention, stochastic latents. Use for air-quality forecasting over many geolocated stations with covariates; not for non-spatial multivariate data or calibrated probabilistic output."
---

# AirFormer

## Idea

- Causal temporal attention (CT-MSA) uses a growing causal window per block (`min(seq_len, 2 ** (index + 1))`).
- Dartboard spatial attention (DS-MSA) lets each station attend to a few aggregated regions given by a `(node, region, node)` dartboard projection, instead of all stations.
- A top-down stochastic stage samples a Gaussian latent at every block during training and uses the mean at evaluation.
- A linear map over time (`temporal_head`) turns history length into horizon length before the output head.

## When to use

- Nationwide-scale air-quality forecasting over hundreds to thousands of stations, where full station attention is too expensive and nearby regions matter most.
- Needs a station graph; construction fails without it. Station geography gives the paper's dartboard projection; the runner injects only the dataset adjacency, which acts as one region, so the spatial prior is weaker than the paper's.
- Latents model data uncertainty during training, but evaluation returns point forecasts (latent means).

## Configure

- `enc_in`: number of stations.
- `cov_dim`: covariate features per station.
- `adj_mx`: passed as `dartboard_mx` `[enc_in, regions, enc_in]`; an ordinary `[N, N]` adjacency is treated as one region.

Other hyperparameters: preset defaults in `configs/models/AirFormer.toml`; tune generically. `d_model` must be divisible by `nhead`.

## Differences

Local implementation from the paper after inspecting the official `src/models/airformer.py` at the pinned revision; nothing copied (no license file). CT-MSA is causal; DS-MSA attends from every station to its regional aggregates; training samples each top-down latent and evaluation uses latent means. The official code loads a precomputed dartboard `assignment.npy` and `mask.npy` from station geography (`get_dartboard_info`); no TSFLab dataset ships one, so runs use the adjacency as a single region.
