"""Convolutional causal self-attention with a LogSparse attention pattern.

Two paper-neutral pieces from Li et al., "Enhancing the Locality and Breaking the
Memory Bottleneck of Transformer on Time Series Forecasting" (NeurIPS 2019):

- ``logsparse_mask``: the causal index sets of Sec. 4.2. A cell attends to itself
  and to cells at exponential distances ``1, 2, 4, ...`` in the past; optionally a
  dense left window first (local attention) and the pattern restarted every
  ``restart`` steps of distance (restart attention).
- ``ConvSelfAttention``: multi-head masked self-attention whose queries and keys
  come from a causal 1-D convolution of kernel ``kernel_size`` (Sec. 4.1, Fig. 1d)
  and whose values come from a kernel-1 projection. ``kernel_size = 1`` is canonical
  attention. ``forward_last`` computes only the newest position from cached keys and
  values, which equals the last row of ``forward`` because every operation is causal.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["logsparse_mask", "ConvSelfAttention"]


def _allowed_distance(distance: torch.Tensor, local: int, restart: int) -> torch.Tensor:
    """Whether a key ``distance`` steps in the past (``>= 0``) is attended."""
    if restart > 0:
        distance = distance % restart
    # Dense part: the cell itself plus ``local`` left neighbours.
    dense = distance <= local
    # Beyond the dense window, exponential steps measured from its left edge.
    beyond = distance - local
    positive = beyond.clamp(min=1)
    power = (positive & (positive - 1)) == 0
    return dense | ((beyond > 0) & power)


def logsparse_mask(
    length: int,
    *,
    local: int = 0,
    restart: int = 0,
    sparse: bool = True,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Boolean ``[length, length]`` mask; ``True`` marks a blocked (``-inf``) pair.

    Row ``l`` (query) may attend column ``j`` (key) only when ``j <= l`` and the
    distance ``l - j`` is ``0``, at most ``local``, or ``local + 2**m`` (``m >= 0``),
    after reducing the distance modulo ``restart`` when ``restart > 0``. With
    ``sparse=False`` the mask is the plain causal (upper-triangular) mask.
    """
    if length < 1:
        raise ValueError("length must be positive")
    if local < 0 or restart < 0:
        raise ValueError("local and restart must be non-negative")
    if restart and restart <= local + 1:
        raise ValueError("restart must exceed local + 1")
    index = torch.arange(length, device=device)
    distance = index.view(-1, 1) - index.view(1, -1)
    causal = distance >= 0
    if not sparse:
        return ~causal
    allowed = causal & _allowed_distance(distance.clamp(min=0), local, restart)
    return ~allowed


class ConvSelfAttention(nn.Module):
    """Masked multi-head self-attention with causal-convolution queries and keys.

    ``forward(x [B, L, d_model], blocked [L, L] bool) -> [B, L, d_model]``.
    Queries and keys: ``Conv1d(d_model, d_model, kernel_size)`` over time after
    left zero padding of ``kernel_size - 1`` steps (causal); values: ``Linear``;
    output: ``Linear``; dropout on the attention weights.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        kernel_size: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if kernel_size < 1:
            raise ValueError("kernel_size must be positive")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.kernel_size = kernel_size
        self.query_conv = nn.Conv1d(d_model, d_model, kernel_size)
        self.key_conv = nn.Conv1d(d_model, d_model, kernel_size)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _causal_conv(self, conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        # [B, L, D] -> [B, L, D]; position t sees inputs t - k + 1 .. t only.
        padded = F.pad(x.transpose(1, 2), (self.kernel_size - 1, 0))
        return conv(padded).transpose(1, 2)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

    def _attend(self, q, k, v, blocked) -> torch.Tensor:
        # softmax(Q K^T / sqrt(d_k) + M) V, M = -inf on blocked pairs.
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(blocked, float("-inf"))
        weights = self.dropout(torch.softmax(scores, dim=-1))
        context = weights @ v
        batch, _, length, _ = context.shape
        return self.out(context.transpose(1, 2).reshape(batch, length, self.d_model))

    def project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Head-split ``(q, k, v)``, each ``[B, heads, L, head_dim]``."""
        q = self._heads(self._causal_conv(self.query_conv, x))
        k = self._heads(self._causal_conv(self.key_conv, x))
        v = self._heads(self.value(x))
        return q, k, v

    def forward(self, x: torch.Tensor, blocked: torch.Tensor) -> torch.Tensor:
        q, k, v = self.project(x)
        return self._attend(q, k, v, blocked)

    def forward_last(
        self,
        x: torch.Tensor,
        blocked_row: torch.Tensor,
        cache: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Output ``[B, 1, d_model]`` of the newest position of ``x [B, L, d_model]``.

        ``cache`` holds keys ``"k"``, ``"v"`` (``[B, heads, L - 1, head_dim]``) of the
        earlier positions and is extended in place; ``blocked_row`` is row ``L - 1``
        of the mask restricted to its first ``L`` columns. An empty cache starts it.
        """
        tail = x[:, -self.kernel_size:]
        q = self._heads(self._causal_conv(self.query_conv, tail)[:, -1:])
        k_new = self._heads(self._causal_conv(self.key_conv, tail)[:, -1:])
        v_new = self._heads(self.value(x[:, -1:]))
        if "k" in cache:
            k_new = torch.cat((cache["k"], k_new), dim=2)
            v_new = torch.cat((cache["v"], v_new), dim=2)
        cache["k"], cache["v"] = k_new, v_new
        return self._attend(q, k_new, v_new, blocked_row.view(1, 1, 1, -1))
