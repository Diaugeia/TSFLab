---
name: "GMAN"
description: "Graph multi-attention encoder-decoder: gated spatial and temporal attention conditioned on a node2vec road-graph embedding and calendar one-hots, plus history-to-horizon transform attention. Use for multi-step traffic forecasting on a known road graph; not for data without a graph or very large node counts."
---

# GMAN

## Idea

- Spatio-temporal embedding: `node2vec_embedding` of the road graph (fixed, computed once from `adj_mx`) and one-hot day-of-week / time-of-day, each through a two-layer FC, summed per node and step for history and horizon.
- `STAttBlock`: spatial attention over all nodes and temporal attention over steps, both on `[hidden || STE]`, fused by `gated_fusion` (`z * H_S + (1 - z) * H_T`), plus a residual connection.
- `TransformAttention`: each horizon step attends to every history step, with queries from the horizon STE, keys from the history STE and values from the encoder output.
- Encoder and decoder each stack `num_blocks` blocks; a two-layer FC emits one value per node and step.

## When to use

- Traffic speed or flow sensors with a road adjacency and regular sampling; the paper targets long horizons (up to 1 hour of 5-minute steps).
- Needs `adj_mx` (PEMS, METR-LA and PEMS-BAY presets ship it); construction fails without it.
- Uses calendar marks of history and horizon; without horizon marks the horizon calendar is extrapolated one step at a time from the last history stamp.
- Spatial attention is dense in N; the paper's group attention for large graphs is not provided. Point output only.

## Configure

- `enc_in`: number of nodes.
- `adj_mx`: injected road graph; the node2vec embedding (`node2vec_*`, `se_dim`) is computed from it at construction and stored as a buffer.
- `steps_per_day`: samples per day (288 for 5-minute data); time-of-day indices come from the marks.

Other hyperparameters: preset defaults in `configs/models/GMAN.toml`; tune generically.

## Differences

Independent PyTorch implementation checked against the official TensorFlow code (Apache-2.0, `45ed232f`): METR/PeMS `model.py`, `tf_utils.py`, `utils.py`, `train.py` and `node2vec/`. Followed from the code: ReLU projections with batch norm (eps 1e-3) in every nonlinear FC, the two-layer FC after each attention and after the fusion, the gate (bias on the temporal term only; `gated_fusion` carries both biases, both zero-initialized), Glorot-uniform weights with zero biases, the causal temporal mask and output dropout 0.1 of the METR-LA script. Paper L = 3, K = 8, d = 8.

- Batch-norm momentum is fixed (0.01) instead of the official decaying schedule.
- node2vec runs in-house (`node2vec_embedding`) with the official walk settings; skip-gram uses Adam on sampled minibatches, so the vectors differ from the shipped `SE(*.txt)` files.
- Loss, scaling (official: one global mean and std of the training inputs), early stopping and learning-rate decay are runner settings.
- Only the value channel enters the model; calendar marks feed the temporal embedding.
