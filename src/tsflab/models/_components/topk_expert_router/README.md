---
name: "topk_expert_router"
description: "Two-layer GELU gating MLP with optional trainable noise giving softmax expert weights, plus top-k sparsification with a dense-weight floor (DUET, Dynamic TMoE). Use for mixture-of-experts forecasters that route inputs to a few experts; not for compute-skipping or exactly sparse routing."
---

# topk_expert_router

## What it does

`GatingMLP(in_features, experts, hidden, noisy)` computes
`logits = Linear(GELU(Linear(x)))`; in training mode and when `noisy`, it adds
`randn * softplus(noise_scale)` (per expert) to the logits; it returns
`softmax(logits, -1)`. `topk_dense_mix(w, k, floor)` keeps the `k` largest entries of
`w` (zeroing the rest), adds `floor * w` everywhere and renormalizes:
`(sparse + floor*w) / sum(sparse + floor*w)`. The floor keeps non-selected experts
gradient-alive.

## When to use

Use in mixture-of-experts forecasters where inputs (for example raw per-channel
windows, as in `duet`) should be routed
mostly to `k` experts while a small dense gradient path keeps the others
trained. Do not use if non-selected experts must be skipped to save compute, if
an exactly sparse weight vector is required with `floor > 0`, or when the gate
needs a different architecture (fixed at two layers with GELU).

## Interface

`GatingMLP(in_features, experts, hidden, noisy=True)`. Keys: `network.0.weight`,
`network.0.bias` (`[hidden, in_features]`, `[hidden]`), `network.2.weight`,
`network.2.bias` (`[experts, hidden]`, `[experts]`) and, only when `noisy`,
`noise_scale` `[experts]` (zeros, so initial noise std is softplus(0)=0.693). No
buffers, no constructor validation. `forward(features)` returns softmax over the
last axis, shape `[*, experts]`, in the dtype and device of the input; noise uses
the global torch RNG and only applies when `self.training` is true (eval is a
plain softmax).

`topk_dense_mix(weights, k, floor) -> Tensor`: `weights` is a non-negative dense
distribution on the last axis (typically softmax output); `1 <= k <= experts`
and `floor >= 0`, else `ValueError` (a `floor` of 0 gives exact
hard top-k renormalization). Differentiable through the kept values and the floor
term; the selection is not. Ties follow `torch.topk`. Stateless, same
dtype/device as input.
