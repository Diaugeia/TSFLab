---
name: "interpretable_attention"
description: "Multi-head attention whose heads share one value projection and are averaged before a bias-free output map (TFT), plus a causal mask helper. Use when head-averaged weights should read as position importance; not when heads need separate values or concatenation."
---

# interpretable_attention

## What it does

`InterpretableMultiHeadAttention(d_model, num_heads, dropout)` computes, per head `h`,
`A_h = softmax(Q W_Q^h (K W_K^h)^T / sqrt(d_head))` with `d_head = d_model // num_heads`,
applies every `A_h` to the same values `V W_V` (`W_V` shared by all heads), averages the
heads, and maps back with `W_H`: `out = (1/H sum_h A_h V W_V) W_H`. All projections are
bias-free. Dropout is applied to each head's output and to the projected output.

`causal_mask(length)` returns a lower-triangular boolean mask (query `i` sees keys `j <= i`,
the diagonal included).

## When to use

Use for temporal self-attention where the averaged attention map should be interpretable
as a single importance pattern over positions (TFT-style decoders), or as a cheaper
attention with one value projection. Do not use when heads must carry different value
subspaces or be concatenated (`self_attention_family`'s `AttentionLayer`), or when a
sparse, patch, or variate-token attention is needed.

## Interface

- `forward(q, k, v, mask=None) -> (output [B, Tq, d_model], attention [B, H, Tq, Tk])`.
- `mask`: boolean `[Tq, Tk]` or `[B, Tq, Tk]`; False entries get `-inf` before the softmax,
  so each query row must keep at least one True entry.
- `d_model` must be a positive multiple of `num_heads` (`ValueError` otherwise).
- State-dict keys: `query.weight [H*d_head, d_model]`, `key.weight [H*d_head, d_model]`,
  `value.weight [d_head, d_model]`, `out.weight [d_model, d_head]`.
- Dropout is the only train/eval difference; no state between calls.
