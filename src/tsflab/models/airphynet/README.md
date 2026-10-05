---
name: "AirPhyNet"
description: "Graph neural ODE with gated diffusion-advection vector field, evolved from a sampled GRU initial state. Use for multi-station air-quality forecasting with distance and wind-flow graphs; not for data without a physical transport graph."
---

# AirPhyNet

## Idea

- A per-station GRU encodes pollutant and covariate history into a reparameterized Gaussian initial latent state (sampled only in training).
- `PhysicsVectorField` follows `dz/dt = -alpha k tanh(Lz) - (1-alpha) tanh(Mz)`, with `L` the distance-graph Laplacian and `M` a directed flow operator.
- The latent state is integrated one step per horizon with Euler or RK4 and a shared MLP decoder maps each state to concentration.

## When to use

- Air-pollutant forecasting across stations where particle diffusion and advection explain spatial spread, including sparse-data and sudden-change scenarios targeted by the paper.
- Needs a distance graph (`adj_mx`); construction fails without it.
- The advection graph is never supplied by TSFLab data: a directed ring placeholder is used and a warning is raised, so the advection term has no physical meaning.
- Uses meteorology covariates in the encoder; point output at evaluation (latent mean).

## Configure

- `enc_in`: number of stations.
- `cov_dim`: covariate features per station.
- `adj_mx`: `[enc_in, enc_in]` distance graph (diffusion Laplacian).
- `flow_mx`: `[enc_in, enc_in]` directed flow graph (advection); only for direct construction, the runner does not inject it.

Other hyperparameters: preset defaults in `configs/models/AirPhyNet.toml`; tune generically.

## Differences

Clean-room implementation of Eq. (9)-(12) from the paper; the reference code was not copied. Eq. (9) maps to the encoder and initial mean/scale, Eq. (10)-(11) to `PhysicsVectorField`, Eq. (12) to a local differentiable solver and decoder. The latent initial state is sampled in training and its mean is used in evaluation. The official code builds the advection graph per batch from the last wind variables through a learned `flow_net` along the station edges (`ode_func.py` lines 96-126); TSFLab passes no wind variables or edge list for it, so the placeholder replaces it.
