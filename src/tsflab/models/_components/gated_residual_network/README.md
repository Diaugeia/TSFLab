---
name: "gated_residual_network"
description: "Gated residual network LayerNorm(skip(a) + GLU(W1 ELU(W2 a + W3 c))) with optional context, its GLU and gated add-norm, and a softmax variable-selection network (TFT). Use for gated, context-conditioned feature processing or per-instance input selection; not for ungated MLP blocks."
---

# gated_residual_network

## What it does

Four modules from Sec. 4.1-4.2 of the Temporal Fusion Transformer:

- `GatedLinearUnit(in_dim, out_dim, dropout)`: `sigmoid(W4 g + b4) * (W5 g + b5)`, with dropout on `g` first.
- `GateAddNorm(in_dim, out_dim, dropout)`: `LayerNorm(residual + GLU(x))`, the gated skip connection.
- `GatedResidualNetwork(in_dim, hidden_dim, out_dim=None, context_dim=None, dropout)`:
  `eta2 = ELU(W2 a + W3 c + b2)`, `eta1 = W1 eta2 + b1`, `out = LayerNorm(skip(a) + GLU(dropout(eta1)))`.
  `skip` is the identity when `out_dim == in_dim`, else a `Linear`. `W3` has no bias; without a context `c = 0`.
- `VariableSelectionNetwork(num_vars, dim, context_dim=None, dropout)`: weights
  `v = softmax(GRN_v(flatten(xi), c))` over variables, one GRN per variable
  (`dim -> dim`), output `sum_j v_j GRN_j(xi_j)`.

## When to use

Use when a block should apply nonlinear processing only where it helps (the GLU can
close and leave the skip path), when a static or global context must condition a
feature transform, or when several embedded input variables (covariates, calendar
fields, the target) must be weighted per instance and time step. Do not use for a
plain MLP or residual block without a gate (see `mixer_block`), for a gate over a
single stream without residual (`softmax_gate`), or for selection over time tokens.

## Interface

- `GatedResidualNetwork.forward(a, context=None) -> [..., out_dim]`; `a` must end in
  `in_dim` (`ValueError` otherwise). The context is projected to `hidden_dim` and
  added before the ELU, so it must broadcast, e.g. `[B, 1, d]` against `[B, T, d]`.
  Passing a context to a GRN built without `context_dim` raises `ValueError`.
- `VariableSelectionNetwork.forward(xi, context=None) -> (combined [..., dim], weights [..., num_vars])`;
  `xi` must end in `[num_vars, dim]`. With `num_vars = 1` the weight is always 1.
- State-dict keys: GRN `input_proj`, `context_proj` (when built with a context),
  `hidden_proj`, `skip` (when widths differ), `gate_norm.glu.value`, `gate_norm.glu.gate`,
  `gate_norm.norm`; VSN `weight_grn.*`, `variable_grns.<j>.*`.
- Default PyTorch initialization; consumers may re-initialize. Dropout is the only
  train/eval difference.
