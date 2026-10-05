# DRAGON — reference

## Implementation mapping

- `build_mdbg` (Section 2.1, Algorithm 1): `uniform_bin_edges` fitted per
  variable; nodes in first-appearance order; each node keeps the raw
  `(k-1)`-tuples mapped to it.
- `ppr_topk_edges`: unit self-loops, symmetric degree normalization,
  `S = alpha (I - (1 - alpha) T)^-1`, top-`gdc_topk` sources per node.
- `DragonEncoder` (Fig. 2): `node_feat_size` raw tuples per node projected to
  `enc_in` channels; `GraphAttentionConv` layers with dropout before each, ELU
  between, heads concatenated, last layer averaged to `d_graph`. `node_mask`
  picks the node of each window tuple, else the first node of that variable
  within L1 distance 1, else the nearest.
- Downstream TimesNet (Fig. 3): cataloged `embed`, `revin`, `marks`,
  `dominant_periods` and `inception_block`; one shared LayerNorm.

## Differences in detail

- Sources read: `dBG/MultiDeBruijnGraph.py`, `dBG/dBGSampler.py`,
  `dBG/GraphEncoder.py`, `data_provider/data_loader.py` (`dBG_Dataset`),
  `exp/exp_basic.py`, `exp/exp_long_term_forecasting.py`, `models/TimesNet.py`,
  `layers/Conv_Blocks.py`, `layers/Embed.py`, `run.py`, `requirements.txt`,
  `runscripts/`.
- Resolved from the official code: one uniform discretizer per variable (bin
  `searchsorted(inner edges, x, right)`; a constant variable gets one bin);
  masks computed from the scaled window before instance normalization; the
  attention layers are one input layer `enc_in -> enc_in x heads`,
  `graph_layers - 2` hidden layers and an averaged output layer to `d_graph`,
  Glorot initialization, LeakyReLU 0.2 and self-loops; pooling scale
  `1 / sqrt(d_graph)`; encoder mixing weights start from `randn`; TimesNet uses
  the TSLib `timeF` hourly embedding (circular token convolution, sinusoidal
  positions, four calendar features), mean/std normalization, `predict_linear`,
  TimesBlocks with zero padding and amplitude-softmax aggregation, and
  Kaiming-initialized Inception kernels.
- Preset: `runscripts/runscripts/slurm_jobs/dBG_window/TimesNet_ETTh1_96.sh`
  (`seq_len = 12`, `label_len = 6`, horizon 96, `d_model = 16`, `d_ff = 32`,
  `top_k = 5`, `k = 4`, alphabets 20/25/30, `--use_gdc`, `d_graph = 16`, 2
  attention layers, 4 tuples per node, 16 heads, GDC top-k 32) with `run.py`
  defaults (MSE, learning rate `1e-4`, batch 32, dropout 0.1, encoder dropout
  0.4, PPR `alpha = 0.05`, 6 Inception kernels).
- PyTorch Geometric uses an unstable float32 sort for top-k and fails when
  `gdc_topk` exceeds the node count; here ties go to the lower index and all
  nodes are kept. scikit-learn's bin assignment is reproduced for its current
  `searchsorted` implementation.
- Checked: discretization against scikit-learn's uniform `KBinsDiscretizer`,
  the MdBG of a hand-worked series, PPR diffusion and top-k against the dense
  formula, graph attention against a per-node evaluation, the nearest-node rule,
  window masks, padded-slot sampling, the encoder and full forward against
  manual compositions, checkpoint reload of a fitted model. Reported benchmark
  numbers are not reproduction claims of this implementation.

## Citation

Cakiroglu, M. O., Altun, I. B., Dalkilic, M., Buxton, E., Kurban, H. "Multivariate de Bruijn Graphs: A Symbolic Graph Framework for Time Series Forecasting." ICML 2025 Workshop on Foundation Models for Structured Data.

## Memory

`GraphEncoder_Attn_new.forward` (`dBG/GraphEncoder.py`) loops over the batch
and, for every window, samples node inputs and runs every GAT layer over the
whole graph; `DragonEncoder` does the same. Stored activations per window and
layer are about `E x heads x width` (gathered source features) plus several
`E x heads` attention tensors, with `E = gdc_topk x N` edges. On the scaled
ETTh1 training split the preset graphs have about 4.1k, 6.0k and 8.1k nodes
(alphabets 20, 25, 30), so about 1.2 GB per window for the three encoders:
about 40 GB at the official batch 32 and about 160 GB at batch 128. Training
therefore checkpoints each window (`torch.utils.checkpoint`, RNG state
preserved): only the pooled `[seq_len, d_graph]` output is kept and the window
is recomputed in backward. Values and gradients are unchanged; the cost is one
extra graph forward per step.
