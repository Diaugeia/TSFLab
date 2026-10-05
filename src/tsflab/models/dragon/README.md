---
name: "DRAGON"
description: "Multivariate de Bruijn graphs of discretized training-set tuples, encoded by PPR-diffused graph attention and fed to a TimesNet. Use for multivariate data with recurring symbolic patterns and periodicity, with the training split available for graph building; not for very many channels or pure univariate use."
---

# DRAGON

## Idea

- Builds multivariate de Bruijn graphs (MdBG) from the scaled training split: values are cut into equal-width bins per variable, nodes are `(variable, (k-1)-tuple)`, k-tuples add prefix -> suffix edges, and tuples of all variables at the same step are joined by hyper-tuple edges. One graph per alphabet size.
- PPR graph diffusion keeps each node's `gdc_topk` strongest sources; only that edge set is used.
- `DragonEncoder`: sampled raw tuples per node go through graph attention layers; outputs are masked by the window's nodes and pooled into `seq_len` rows by learned time queries.
- The softmax-weighted encoder outputs are concatenated with the TSLib data embedding (with calendar marks) and refined by a TimesNet (dominant-period folding, Inception 2D convolutions), then RevIN-denormalized.
- Before `training_setup` builds the graphs the graph branch contributes zeros; graph state lives in buffers so checkpoints carry it.

## When to use

- Aimed at series whose discretized values recur as symbolic patterns, and at joint patterns across variables (hyper-tuple edges); the TimesNet backbone targets periodic structure.
- The graph is fitted once on the training split; regimes absent from training map to nearest nodes only.
- Node count grows with variables, series length and alphabet sizes. Above 32768 nodes per graph (dense PPR inverse) or an edge x head x width budget of 2^30 per attention layer, `training_setup` raises a ValueError: wide or long datasets such as traffic, electricity, solar, covid19, NN5, wike2000 and PEMS are not applicable; ETT, weather, ILI and NASDAQ fit.
- Official experiments use a very short lookback (`seq_len = 12`).

## Configure

- `enc_in`: number of variables; the training series must be `[T, enc_in]`.
- The MdBGs are built automatically by `training_setup` from the whole scaled training split; no value is set by hand.
- `gdc_topk` larger than the node count keeps every node.

Other hyperparameters: preset defaults in `configs/models/DRAGON.toml`; tune generically.

## Differences

Independent rewrite of Section 2 and Appendices A-C after reading `KurbanIntelligenceLab/MultdBG-Time-Series-Library` at `6b9fa564` (MIT); NetworkX, scikit-learn binning and PyTorch Geometric (`GDC`, `GATConv`) are reimplemented from their documented semantics. The official code is followed where it departs from the paper.

- Diffusion inverse in float64; top-k ties go to the lower node index.
- Node tuples are sampled with the code's padded-slot distribution without materializing the padded table.
- The graph branch is zero until `training_setup` runs.
- Hyper-tuple edges are deduplicated in bounded chunks and the PPR matrices are built in place (same graph, bounded host memory); the scale limits above are TSFLab's, since the official code has none.
- Each window's graph encoding is activation-checkpointed in training (same values and gradients, about one extra graph forward per step): as upstream, every window runs the GAT over the whole MdBG, about 1.2 GB of stored activations per ETTh1 window, which exceeds a 48 GB GPU at batch 128.
- `--reverse` / `--undirected` graph options and non-TimesNet downstream models are not implemented. Details in `reference.md`.
