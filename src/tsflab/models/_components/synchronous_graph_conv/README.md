---
name: "synchronous_graph_conv"
description: "STSGCN synchronous module: stacked GLU or ReLU graph convolutions on a graph joining adjacent steps, element-wise max over layers, cropped to one time block; plus the localized-graph builder. Use for sliding-window spatio-temporal graph models; not for learned or per-sample graphs or very large N."
---

# synchronous_graph_conv

## What it does

`localized_adjacency(adj_mx, steps)` binarizes and symmetrizes the spatial graph,
copies it onto `steps` diagonal blocks, links each node to itself in the next and
previous step, and sets every self-loop: a `[steps * N, steps * N]` float32 tensor.

`SynchronousGraphModule` (one STSGCM) maps a window `h_0 [steps * N, C]` through
`len(filters)` layers `h_l = act(A h_{l-1} W_l + b_l)`. GLU splits the projection into
value and gate halves (`value * sigmoid(gate)`); ReLU uses one projection. The output
is the element-wise max of all `h_l`, restricted to the rows of time block `crop`.

## When to use

Use for models that slide a short window over time and convolve each window on a
fixed joined spatio-temporal graph (STSGCN localized graph, STFGNN fusion graph). Do
not use for graphs that change per sample, for learned adjacency generators, or for
very large `steps * N` (the adjacency is dense).

## Interface

- `SynchronousGraphModule(num_nodes, steps, in_dim, filters, *, activation="GLU", num_modules=1, crop=1, init_magnitude=0.0003)`.
  `filters` must share one width (max aggregation); `crop` in `[0, steps)`.
- `forward(x, adjacency)`: `x [B, M, steps * N, C]`, `adjacency [steps * N, steps * N]`
  (pass a masked or weighted version as needed) -> `[B, M, N, filters[-1]]`.
  `num_modules` is `1` (shared by every window) or `M` (one module per window).
- Parameters `weights.{l}` `[num_modules, in, factor * out]` and `biases.{l}`
  `[num_modules, 1, factor * out]` (`factor` 2 for GLU); MXNet `Xavier(magnitude)`
  uniform init `U(+-sqrt(magnitude / ((in + out) / 2)))`, zero biases.
- `mxnet_xavier_uniform_(tensor, fan_in, fan_out, magnitude)`: the same rule for
  other layers of a consumer.
