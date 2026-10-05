# topk_expert_router — reference

## Origin and granularity

Extracted in commit `cd1b398d` ("topk_expert_router for DUET and DynamicTMoE"),
which says both routers were reproduced bit-for-bit. `GatingMLP` is DUET's
`DistributionalRouter` gate; `topk_dense_mix` is the same arithmetic DUET applied
after its router and DynamicTMoE applied in `routing_weights` (it passes a
configurable `routing_floor`; DUET passes 1e-3). Model-local: the gate input
features (DUET: each raw channel series of length `seq_len`), DynamicTMoE's drift, MMD and memory
logits (it uses only `topk_dense_mix`), expert networks, and MAGE's gate, DUET's
even-kernel moving average and STWave's Haar step, which the commit kept local
with documented reasons.

## Invariants and equivalence evidence

- Contract checks (at extraction): `GatingMLP` state-dict keys (with and without noise),
  zero-initialized `noise_scale`, softmax rows sum to 1 and are non-negative, eval
  output equals `softmax(network(x))`, training noise is seed-reproducible and
  differs from eval, finite gradients; `topk_dense_mix` rows sum to 1, `floor = 0`
  keeps exactly `k` non-zero renormalized entries, `floor > 0` makes all entries
  positive, dtype is preserved, gradients are finite, `k > experts` raises
  `RuntimeError`; a validation check covered the `ValueError` for
  `k` outside `[1, experts]` and negative `floor`. Seeded gate and mix outputs were
  pinned as regression values.
- Equivalence against pre-refactor values of both consumers (`duet` and
  `dynamic_tmoe`) was checked at extraction: state-dict keys, shapes and values,
  forward outputs and input gradients equal before and after extraction.
- The same extraction check proved `topk_dense_mix` equals both original inline formulas term
  for term, rows sum to one, the noisy gate in eval mode equals a plain softmax, and
  state-dict attribute names are preserved. These checks passed in the full suite
  run of 2026-10-03 before the test suite was consolidated.

## Variants and options

`noisy=False` removes `noise_scale` and the noise. `k` and `floor` set sparsity
and gradient leakage. Not covered: load-balancing or auxiliary losses, capacity
limits, token dispatch or exact sparse expert execution (DUET and DynamicTMoE run
every expert densely and mix with the weights), learned `k`.

## Related components

`weight_set_router` (input-independent, softmax-with-temperature routing over
weight sets; this component is input-conditioned), `sparse_connection_router`
(learned top-k binary connection matrix over positions, not mixture weights),
`topk_expert_attention` (its own `LocalExpertRouter` keeps the top-k keys per
query and softmaxes the kept logits, without a floor), `freq_band_moe` (dense
softmax gates over spectral bands), `soft_tree` (differentiable tree routing).
