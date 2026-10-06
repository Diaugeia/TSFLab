---
name: "dilated_conv_encoder"
description: "TS2Vec-style encoder: residual blocks of two same-padded dilated Conv1d layers (kernel 3, dilation 2^i) with pre-GELU, preserving sequence length. Use as a per-step representation backbone for contrastive or forecasting heads; not when strict causality is required."
---

# dilated_conv_encoder

## What it does

`DilatedConvEncoder(in_channels, channels, kernel_size=3)` stacks one
`DilatedConvBlock` per entry of `channels`, block `i` with dilation `2**i`:
`y = conv2(gelu(conv1(gelu(x)))) + r(x)`, where `r` is the identity, or a 1x1
`Conv1d` when the width changes and always in the last block.
`SamePadConv` pads `field // 2` on both sides with `field = (k - 1) d + 1` and
drops the last output step when `field` is even, so the length stays `T`.

## When to use

Use as a backbone that gives every time step a representation from a wide,
two-sided receptive field (about `2^(len(channels)+1)` steps), e.g. for
contrastive representation learning or per-step heads. Do not use when an output
step may not see later inputs (padding is symmetric, so it is not causal).

## Interface

- `DilatedConvEncoder(in_channels: int, channels: list[int], kernel_size: int = 3)`;
  empty `channels` or `kernel_size < 1` raise `ValueError`.
- `forward(x [B, in_channels, T]) -> [B, channels[-1], T]`.
- Also exported: `SamePadConv(in, out, kernel_size, dilation=1)` and
  `DilatedConvBlock(in, out, kernel_size, dilation, final=False)`.
- State-dict keys: `net.<i>.conv1.conv.*`, `net.<i>.conv2.conv.*`, `net.<i>.projector.*`
  (same layout as the TS2Vec/CoST `DilatedConvEncoder`); PyTorch default initialization.
