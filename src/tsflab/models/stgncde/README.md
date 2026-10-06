---
name: "STGNCDE"
description: "STG-NCDE: coupled temporal and spatial neural CDEs driven by a cubic-spline path of each node's window, with a learned node-adaptive graph, solved by RK4. Use for sensor-network traffic forecasting without a reliable adjacency; not for long windows, very many nodes, or tight compute."
---

# STGNCDE

## Idea

- Control path (Sec. 'Overall design'): per node, the natural cubic spline (`natural_cubic_spline`) of `[t, x_t]` on the unit grid `t = 0 .. T-1`.
- Temporal CDE (Eqs. 4-5, 8): `dh = f(h) dX`, with `f` = FC-ReLU stack and a tanh output reshaped to `hidden x channels`; each node is processed independently.
- Spatial CDE (Eqs. 6-7, 9-11): `dz = g(z) f(h) dX`, with `g` = FC-ReLU, a node-adaptive graph convolution over `softmax(relu(E E^T))` (`node_adaptive_graph_conv`, as in the official code), and a tanh output reshaped to `hidden x hidden`.
- Both CDEs form one augmented ODE (Eq. 12) solved by `torchdiffeq` (RK4 on the observation grid, adjoint backward pass by default); initial states are linear maps of `X(t_0)`.
- Output (Eq. 13): a linear map of each node's `z(T)` to the horizon.

## When to use

- Designed for 12 -> 12 traffic forecasting on PeMS sensor networks (flow or speed), where node behaviour differs and the graph is learned.
- Continuous-time path: tolerant of irregular input sampling in principle; this port uses the regular grid.
- Cost: `seq_len - 1` RK4 steps of four field evaluations, each with a dense `N x N` graph and per-node `hidden x hidden` matrices; slow for long windows or thousands of nodes.
- Not for data without node structure or for long horizons.

## Configure

- `enc_in`: the dataset's node count (sizes the node embeddings).
- `input_dim`: data channels in the path; `1` (value only) is the official setting, `1 + F` adds node covariates or the time-of-day/day-of-week marks.
- `adj_mx` is injected by the runner and ignored.

Other hyperparameters: preset defaults in `configs/models/STGNCDE.toml` (official `run.sh`); tune generically.

## Differences

- Independent rewrite of arXiv 2112.03558 (AAAI 2022) after reading `jeongwhanchoi/STG-NCDE@49480bdf` (MIT). With mapped weights the forward pass matches the official `NeuralGCDE` to 4e-6, and the adjoint gradients match to 3e-6 (float32, 8 nodes).
- Follows the code where it departs from the paper: node-adaptive Chebyshev filter instead of `(I + A) W`, `z(0)` from `X(t_0)`, time channel in the path, Xavier / U(0, 1) initialization.
- The spline is built inside `forward` (official: precomputed in the data loader); same values.
- Scaling is the runner's training-split scaling, not the official whole-series z-score (leakage); the loss is the run's loss on the scaled target (official: MAE on de-normalized values, `real_value = True`).
- `solver` and `adjoint` are parameters; `adjoint = false` gives exact gradients of the RK4 solve.
- Unused official parameters (`NeuralGCDE.node_embeddings`) are omitted; the missing-data experiments of the paper are not reproduced.
