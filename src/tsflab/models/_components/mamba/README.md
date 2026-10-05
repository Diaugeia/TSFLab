---
name: "mamba"
description: "Kernel-free pure-PyTorch Mamba selective-SSM mixer (optional causal depthwise conv, sequential scan), plus RMSNorm and a pre-norm residual block. Use for portable Mamba layers over time or variate tokens of modest length; not for very long sequences, fp16, or exact mamba_ssm kernel parity."
---

# mamba

## What it does

`MambaBlock` is a Mamba mixer that needs no `mamba_ssm` or CUDA kernel. For input
`x: [B, L, d_model]` it computes:

1. `in_proj` (no bias) to `d_inner` signal `x'` and `d_inner` gate `res`.
2. When `use_conv=True` (default), a causal depthwise `Conv1d` (kernel `d_conv`,
   `padding=d_conv-1`, output cropped to `L`) over time on `x'`, then SiLU;
   otherwise `x'` goes straight to the scan.
3. `ssm`: `A = -exp(A_log)` (`[d_inner, d_state]`); `x_proj` yields `delta_raw`
   (`dt_rank`), `B`, `C` (each `d_state`) per step, passed through `Dropout(x_dropout)`
   (identity when `x_dropout=0` or in eval mode); `delta = softplus(dt_proj(delta_raw))`;
   `selective_scan` runs `h_t = exp(delta_t A) * h_{t-1} + delta_t B_t u_t`,
   `y_t = C_t . h_t + D * u_t` with a Python loop over `t`.
4. `out_proj(y * SiLU(res))`.

`MambaResidualBlock` is `x + MambaBlock(RMSNorm(x))`. `RMSNorm` is
`x * rsqrt(mean(x^2, -1) + eps) * weight`.

## When to use

Use for a portable CPU/GPU selective SSM over a token axis (time tokens, or
variate tokens in inverted models such as `s_mamba`) when the sequence is of
modest length, as a recurrent alternative to attention. Do not use for very long
sequences where the Python-loop scan is too slow, when exact numerical match to
`mamba_ssm` kernels or fp16 training is required, or when a scalar-state variant
with an editable state is wanted (see `hyper_state_scan`).

## Interface

`MambaBlock(d_model, d_inner, dt_rank, d_conv, d_state, *, use_conv=True,
x_dropout=0.0, reference_dt_init=False, checkpoint_scan=False)`, the five widths are positive ints with no
defaults and are not validated (invalid values fail inside torch); `x_dropout` must lie in
`[0, 1)`, else `ValueError`. The keyword-only options are described under Variants. `d_inner` is the expanded width (callers use `expand * d_model`),
`dt_rank` the low-rank width of the step-size path, `d_conv` the conv kernel,
`d_state` the state size per channel. Methods: `forward(x)` for `[B, L, d_model]` to
`[B, L, d_model]`; `ssm(x)` for `[B, L, d_inner]` to `[B, L, d_inner]`;
static `selective_scan(u, delta, a, b, c, d)` with `u, delta: [B, L, d_inner]`,
`a: [d_inner, d_state]`, `b, c: [B, L, d_state]`, `d: [d_inner]` returning
`[B, L, d_inner]`.

- State-dict keys of `MambaBlock`: `A_log` `[d_inner, d_state]` (init log(1..d_state)),
  `D` `[d_inner]` (ones), `in_proj.weight`, `conv1d.weight`/`conv1d.bias`,
  `x_proj.weight`, `dt_proj.weight`/`dt_proj.bias`, `out_proj.weight`. No buffers.
  `use_conv=False` removes the two `conv1d.*` keys; the other options add no keys.
  `dt_proj` uses default `nn.Linear` init unless `reference_dt_init=True`.
- `RMSNorm(d_model, eps=1e-5)`: parameter `weight` `[d_model]`; no shape check.
- `MambaResidualBlock(d_model, d_inner, dt_rank, d_conv, d_state)`: keys `mixer.*`
  and `norm.weight`; output shape equals input, so `d_model` is preserved.
- Stateless (no cache across calls); no errors are raised by the component, shape
  mismatches surface as torch errors. The scan keeps its hidden state in a local
  zero tensor allocated on `delta`'s device with default float dtype, and computes
  `A` and `D` in float32, so half-precision inputs are not a supported contract.
  Cost is O(L) Python-loop steps; memory holds `[B, L, d_inner, d_state]` tensors.
  `checkpoint_scan=True` stores only the scan inputs and recomputes the scan in
  backward (same values and gradients, one extra scan per step); use it when
  `B x L` tokens are many, as for spatio-temporal grids.
