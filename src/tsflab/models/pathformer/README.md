---
name: "Pathformer"
description: "Multi-scale Transformer: per patch size, local attention inside patches and global attention across them, mixed by an input-dependent router. Use for series whose dynamics span several temporal scales and vary across inputs; not for cross-channel modelling, exogenous inputs, or probabilistic output."
---

# Pathformer

## Idea

- Multi-scale division: each expert cuts every channel into patches of one size; `DualScaleAttention` applies local attention inside each patch and global attention across patches.
- `AdaptivePathway` holds one expert per patch size and mixes them with a router fed by the input's mean, standard deviation, mean absolute difference, and spectral concentration.
- Routing is dense and differentiable; the top-k pathways are only recorded in `last_topk` for inspection.
- Pathway layers are residual with LayerNorm, followed by `nn.Linear(seq_len, pred_len)`; `revin` wraps the model.

## When to use

- Series with patterns at several temporal resolutions (for example short daily cycles and longer weekly ones), covered by the per-layer patch sizes.
- Inputs whose temporal dynamics vary across windows: the router re-weights scales per sample and channel from simple statistics.
- Not when cross-channel dependence carries the signal (channels are processed independently), when timestamps or covariates matter (marks are not used), or when quantiles are needed (point output); attention per expert makes it heavier than linear baselines.

## Configure

- `enc_in`: number of input channels; must equal the dataset's channel count (the forward pass checks it).
- `patch_size_list`: `layer_nums * num_experts` patch sizes, a flat list reshaped per layer; choose sizes that divide `seq_len` and match the data's periods or sub-periods (non-dividing sizes are zero-padded at the end).

Other hyperparameters: preset defaults in `configs/models/Pathformer.toml`; tune generically.

## Differences

- Clean-room implementation from the paper's multi-scale division, dual-attention, and adaptive-pathway descriptions; the reference repository is unlicensed and no source was copied.
- The local (intra-patch) attention over batch x channels x patches sequences runs in chunks of 32768: PyTorch's fused attention rejects more than 65535 sequences per call with dropout (220,672 on traffic at batch 16). Same function; only the dropout random stream is split.
- Dense differentiable routing replaces hard sparse top-k dispatch because the common forecast API has no auxiliary balance-loss channel; the top-k paths stay inspectable.
