---
name: "DFDGCN"
description: "Traffic GNN: gated dilated temporal convolutions with graph propagation over static, adaptive and a per-sample frequency-domain graph built from FFT magnitudes and node identity. Use for sensor networks with a road graph and calendar marks; not for data without node structure."
---

# DFDGCN

## Idea

- `FrequencyGraph` builds a directed per-sample graph from the normalized FFT magnitude of each node's history (scaled by `a`), a node identity embedding and time-of-day / day-of-week embeddings, keeping the `subgraph` strongest neighbours per node; frequency magnitudes reduce time-shift sensitivity.
- `DynamicGraphMix` propagates features over four graphs (forward/reverse supports, a self-adaptive graph, and the frequency graph) and projects the stacked hops.
- The temporal backbone stacks `gated_dilated_conv` layers with skip connections from the gated output, residual graph mixing and BatchNorm; a conv head outputs all horizons.
- Input channels are the value and time of day; day of week enters only the frequency graph.

## When to use

- Traffic or sensor networks where node patterns are similar but shifted in time, so frequency magnitudes reveal relations that raw values hide.
- Needs a node graph (`adj_mx`) and benefits from daily/weekly calendar marks; point output only.

## Configure

- `enc_in`: number of graph nodes N.
- `adj_mx`: the dataset's `[N, N]` adjacency, injected by the runner. Required: construction fails without it.
- `steps_per_day`: calendar steps per day (288 for 5-minute data).
- `a`: scale of the normalized spectra; `subgraph`: neighbours kept per node (clamped to N).

Other hyperparameters: preset defaults in `configs/models/DFDGCN.toml`; tune generically.

## Differences

- Local rewrite of the dilated backbone, predefined/adaptive/dynamic graph mixture, frequency graph, embeddings and head after `dfdgcn_arch.py` of the MIT official code at `31050585`.
- Frequency graph follows the official code where it goes beyond the paper (normalization and `a`, top-k `subgraph` mask applied before the softmax, per-node projection); see `[[issues]]`.
- Each layer pads causally (`gated_dilated_conv`) instead of one left zero-pad of the input to the receptive field; the last-step output is the same when the receptive field fits in `seq_len`, but BatchNorm statistics include the padded steps.
- Calendar fractions are mapped to embedding rows by `round(fraction * steps)`; the official code reads raw day-of-week integers.
- The preset uses smaller widths and two blocks instead of the official four.
- Official preprocessing, masked-MAE training, and published numerical results are not included.
