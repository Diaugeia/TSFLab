---
name: "logsparse_conv_attention"
description: "Causal multi-head self-attention with causal-convolution queries/keys and a LogSparse mask (exponential distances, optional local window and restarts), plus an exact cached one-step path. Use for causal temporal mixing under memory limits; not for bidirectional encoders or cross-attention."
---

# logsparse_conv_attention

## What it does

- `logsparse_mask(length, *, local=0, restart=0, sparse=True)`: bool `[L, L]`,
  `True` = blocked. Query `l` may see key `j <= l` when the distance `d = l - j`
  (taken modulo `restart` when `restart > 0`) is `0`, at most `local`, or
  `local + 2**m`, `m >= 0`. With `local = restart = 0` this is the paper's
  `I_l = {l - 2^floor(log2 l), ..., l - 2^0, l}`. `sparse=False` gives the plain
  causal mask.
- `ConvSelfAttention(d_model, n_heads, kernel_size=1, dropout=0.0)`:
  `Q = CausalConv_k(x)`, `K = CausalConv_k(x)` (`Conv1d` after `k - 1` left zeros),
  `V = Linear(x)`; per head `softmax(Q K^T / sqrt(d_head) + M) V`, heads
  concatenated and projected by `Linear`. Dropout acts on the attention weights.
  `kernel_size = 1` is canonical attention.

## When to use

Use inside a causal (decoder-only, autoregressive) temporal stack where anomalies
or change points make point-wise query-key matching unreliable (`kernel_size > 1`),
or where the attention matrix memory must stay near `O(L log L)` per layer
(`sparse`, with `log2 L + 1` layers for full reach). Do not use for
bidirectional encoders, cross-attention, variate-token mixing, or padding masks.
The mask is a dense `[L, L]` tensor: memory saving is in the attended set, not in
kernels (the paper's own implementation was a mask as well).

## Interface

- `forward(x [B, L, d_model], blocked [L, L] bool) -> [B, L, d_model]`.
- `project(x) -> (q, k, v)`, each `[B, heads, L, d_head]` (queries and keys of
  every position, used to seed a cache).
- `forward_last(x [B, L, d_model], blocked_row [L] bool, cache: dict) -> [B, 1, d_model]`:
  output of position `L - 1` only; `cache["k"]`, `cache["v"]` hold `[B, heads, L - 1, d_head]`
  of earlier positions and are extended in place. Equals the last row of `forward`
  because convolution and mask are causal.
- Errors: `d_model % n_heads != 0`, `kernel_size < 1`, `length < 1`, negative
  `local`/`restart`, or `0 < restart <= local + 1` raise `ValueError`.
- State-dict keys: `query_conv.*`, `key_conv.*`, `value.*`, `out.*`.
