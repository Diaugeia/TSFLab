"""Interpretable multi-head attention with one value projection shared by all heads.

Independent implementation of Sec. 4.4 of the Temporal Fusion Transformer
(Lim et al., IJF 2021, arXiv 1912.09363):

    A_h        = softmax(Q W_Q^h (K W_K^h)^T / sqrt(d_attn))           (Eq. 10)
    H~         = 1/m_H * sum_h A_h V W_V                               (Eq. 14-16)
    output     = H~ W_H                                                (Eq. 13)

with ``d_attn = d_V = d_model // num_heads`` and bias-free projections. Because
every head attends over the same values, the head-averaged attention matrix is
an interpretable importance pattern over positions.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def causal_mask(length: int, device: torch.device | None = None) -> torch.Tensor:
    """Boolean ``[length, length]`` mask, True where query ``i`` may see key ``j <= i``."""
    return torch.ones(length, length, dtype=torch.bool, device=device).tril()


class InterpretableMultiHeadAttention(nn.Module):
    """Shared-value multi-head attention on ``[B, T, d_model]``.

    ``forward(q, k, v, mask)`` returns ``(output [B, Tq, d_model], attention
    [B, heads, Tq, Tk])``. ``mask`` is a boolean ``[Tq, Tk]`` or ``[B, Tq, Tk]``
    tensor; False entries are excluded from the softmax. Dropout is applied to
    each head's output and to the projected output.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model < 1 or num_heads < 1 or d_model % num_heads:
            raise ValueError("d_model must be a positive multiple of num_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.query = nn.Linear(d_model, num_heads * self.d_head, bias=False)
        self.key = nn.Linear(d_model, num_heads * self.d_head, bias=False)
        self.value = nn.Linear(d_model, self.d_head, bias=False)
        self.out = nn.Linear(self.d_head, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, tq, _ = q.shape
        tk = k.shape[1]
        queries = self.query(q).view(batch, tq, self.num_heads, self.d_head).transpose(1, 2)
        keys = self.key(k).view(batch, tk, self.num_heads, self.d_head).transpose(1, 2)
        values = self.value(v)  # [B, Tk, d_head], shared by every head
        scores = queries @ keys.transpose(-1, -2) / math.sqrt(self.d_head)
        if mask is not None:
            if mask.ndim == 3:
                mask = mask[:, None]
            scores = scores.masked_fill(~mask, float("-inf"))
        attention = torch.softmax(scores, dim=-1)
        heads = self.dropout(attention @ values[:, None])  # [B, H, Tq, d_head]
        output = self.dropout(self.out(heads.mean(dim=1)))
        return output, attention


__all__ = ["InterpretableMultiHeadAttention", "causal_mask"]
