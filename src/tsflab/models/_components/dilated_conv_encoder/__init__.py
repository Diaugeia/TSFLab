"""TS2Vec-style dilated convolutional encoder (non-causal, length preserving).

Block ``i`` has dilation ``2**i``: ``y = conv2(gelu(conv1(gelu(x)))) + r(x)`` with
``r`` the identity or a 1x1 convolution (when widths differ, and always in the last
block). Each convolution pads symmetrically to keep the length; when the receptive
field ``(k - 1) * dilation + 1`` is even, the last output step is dropped.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SamePadConv", "DilatedConvBlock", "DilatedConvEncoder"]


class SamePadConv(nn.Module):
    """``Conv1d`` on ``[B, C, T]`` returning length ``T`` for any kernel and dilation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        field = (kernel_size - 1) * dilation + 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=field // 2, dilation=dilation)
        self.trim = 1 if field % 2 == 0 else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return out[..., : out.shape[-1] - self.trim]


class DilatedConvBlock(nn.Module):
    """Pre-activation residual block of two dilated ``SamePadConv`` layers."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, final: bool = False) -> None:
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.projector is None else self.projector(x)
        return self.conv2(F.gelu(self.conv1(F.gelu(x)))) + residual


class DilatedConvEncoder(nn.Module):
    """Stack of ``DilatedConvBlock`` with widths ``channels`` and dilations ``1, 2, 4, ...``.

    ``forward(x [B, in_channels, T]) -> [B, channels[-1], T]``.
    """

    def __init__(self, in_channels: int, channels: list[int], kernel_size: int = 3) -> None:
        super().__init__()
        if not channels or kernel_size < 1:
            raise ValueError("channels must be non-empty and kernel_size positive")
        widths = [in_channels, *channels]
        self.net = nn.Sequential(*[
            DilatedConvBlock(widths[i], widths[i + 1], kernel_size, 2**i, final=i == len(channels) - 1)
            for i in range(len(channels))
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
