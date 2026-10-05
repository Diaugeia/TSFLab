# mamba — reference

## Origin and granularity

The block equations are those of Mamba (Gu and Dao, 2023). The repository's first
copy was model-local in `mambasimple` (commit `43d54934`, "dependency-free Mamba,
manual selective scan, no mamba_ssm kernels", from TSLib's MambaSimple); it was
moved to this component and shared by `s_mamba` and `bimamba` (commit `3ac9e264`
then `33ea2050` colocated it under the components package). The boundary is the
mixer plus its normalization and residual wrapper. Model-local: tokenization or
inverted embedding (`s_mamba`), forward/backward fusion, the forget/new-feature
gate and FFN (`bimamba`'s `MambaPlus`), the horizon projection, and choosing
`d_inner`, `dt_rank`, `d_conv`, `d_state`. The `use_conv`, `x_dropout`, and
`reference_dt_init` options were added for `mambats`. Direct consumers: `mambasimple`,
`s_mamba`, `bimamba`, `mambats`, `mou`, `samba` (subclasses `MambaBlock` and overrides
`forward`), `timemachine`, `penguin` (`RMSNorm` only), and the `composed` slot adapters;
the generated block is the authoritative list.

## Invariants and equivalence evidence

- Contract checks (at extraction) pinned the interface: state-dict keys and
  default `A_log`/`D` init, `[B, L, d_model]` shape and dtype, causality (perturbing
  steps `>= 4` leaves outputs `< 4` unchanged), gradient flow, `RMSNorm` unit
  RMS, `selective_scan` against an explicit per-step loop of the recurrence, and a
  seeded numerical regression value.
- The same checks pinned the keyword options: identical state-dict keys with `x_dropout` and
  `reference_dt_init`, no `conv1d.*` keys and unchanged output shape for `use_conv=False`,
  `x_dropout=1.0` raising `ValueError`, `softplus(dt_proj.bias)` inside `[1e-4, 0.1]`,
  `dt_proj.weight` bounded by `dt_rank**-0.5`, and `x_dropout` acting only in training mode.
- Model-level checks (pre-consolidation suite) asserted `s_mamba` layers are instances
  of the shared `MambaBlock` and exercised `bimamba`'s `MambaPlus` wrapper around it.
- No comparison against the official `mamba_ssm` kernels.

## Variants and options

The constructor widths plus four keyword-only options, all defaulting to the
original block: `use_conv=False` skips the causal convolution and its SiLU (the
scan then sees the in-projection output directly, as in MambaTS);
`x_dropout=p` applies dropout to the joint step-size/B/C projection output, active
only in training mode (the "selective parameter dropout" of MambaTS); and
`reference_dt_init=True` draws `dt_proj.weight` uniformly in `+-dt_rank**-0.5` and sets
the bias to the inverse softplus of a log-uniform step in `[1e-3, 1e-1]` (floor
`1e-4`), as the reference Mamba does; `checkpoint_scan=True` wraps `selective_scan` in
`torch.utils.checkpoint` while gradients are enabled, so the `[B, L, d_inner, d_state]`
discretized tensors and states are recomputed in backward instead of stored (the scan
has no randomness, so values and gradients are unchanged). Not covered here: bidirectional use (instantiate two
blocks and flip the sequence, as `bimamba` and `s_mamba` do), a gated "Mamba+"
variant (model-local in `bimamba`), a scalar-state scan (see `hyper_state_scan`),
parallel-scan or fused CUDA kernels, and step/cached inference.

## Related components

`hyper_state_scan` (also a sequential-scan SSM, but with a scalar state and grid
mixing rather than a per-channel `d_state` selective scan), `revin` (typical input
normalization in front of SSM forecasters), `mixer_block` (MLP-style token/channel
mixing, no recurrence).
