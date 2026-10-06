---
name: "PDFormer"
description: "Spatial-temporal Transformer whose heads split into temporal, hop-masked geographic (keys enriched with delay-aware traffic patterns) and DTW-masked semantic attention. Use for traffic flow on a road graph with calendar marks; not without a graph or for very many nodes."
---

# PDFormer

## Idea

- Data embedding (Eq. 2): value projection, Laplacian-eigenvector node embedding (`adj_norm`), minute-of-day and day-of-week embeddings (`marks` layout), sinusoidal position (`embed.PositionalEmbedding`).
- `SpatialTemporalAttention` (Eq. 3-15): one block with `t_num_heads` temporal heads, `geo_num_heads` geographic heads restricted to nodes within `far_mask_delta` hops, and `sem_num_heads` semantic heads restricted to the `dtw_delta` most DTW-similar nodes.
- Delay-aware feature transformation (Eq. 7-11): the last `s_attn_size` values of each node attend over a k-Shape pattern set; the mixed pattern vectors are added to the geographic keys.
- Output (Eq. 16): summed per-layer 1x1 skip projections, a history-to-horizon 1x1 conv and a feature 1x1 conv; direct multi-step output.
- `training_setup` fits the semantic mask and pattern set on the training split; the training objective is Huber with an optional horizon curriculum.

## When to use

- Traffic flow or speed on a road network with time-of-day/day-of-week marks (PEMS03/04/07/08 presets).
- Needs `adj_mx`; construction fails without it. Dense N x N attention per step.
- Training setup runs an exact DTW over all node pairs (about 1.5 min on CPU for 307 nodes; grows with N^2).

## Configure

- `enc_in`: number of nodes; `adj_mx`: injected dataset adjacency.
- `far_mask_delta`: hop threshold (official 7 on PeMS).
- `steps_per_day`: samples per day (288 for 5-minute data).
- `cand_key_days`, `dtw_delta`: training-split pattern span and semantic neighbour count.
- `curriculum_step`: batches per added horizon step (0 disables; official PeMS04 1274 at batch 16).

Other hyperparameters: preset defaults in `configs/models/PDFormer.toml` (official PeMS settings); tune generically.

## Differences

Independent implementation from the paper, checked against `BUAABIGSCity/PDFormer` at `f8c8f6ad` (MIT).

- Leakage fixed: DTW profiles use the training split only; the official code averages the whole series including test.
- Exact DTW replaces `fastdtw` (radius 6); local k-Shape follows tslearn 0.5.2 `KShape`, seeded, on training windows standardized with the training mean and std (as the official scaler does).
- Directed graphs are symmetrized for the Laplacian eigenvectors (eigh); missing eigenvectors on tiny graphs are zero-padded.
- Huber loss and curriculum act on the model's input scale (the official loss de-normalizes first); optimizer, warm-up and cosine schedule are run settings.
- `pre_norm = true` follows the released PeMS configs; the paper text implies post-norm.
