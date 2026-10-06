"""Gated residual network (GRN), its GLU gate, and a GRN variable-selection network.

Independent implementation of Sec. 4.1-4.2 of the Temporal Fusion Transformer
(Lim et al., IJF 2021, arXiv 1912.09363):

    GLU(g)    = sigmoid(W4 g + b4) * (W5 g + b5)                 (Eq. 5)
    eta2      = ELU(W2 a + W3 c + b2)                            (Eq. 4)
    eta1      = W1 eta2 + b1                                     (Eq. 3)
    GRN(a, c) = LayerNorm(skip(a) + GLU(dropout(eta1)))          (Eq. 2)

``skip`` is the identity when the output width equals the input width and a
linear projection otherwise. The variable-selection network (Eq. 6-8) weights
per-variable GRN outputs with ``softmax(GRN(flatten(xi), c))``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedLinearUnit(nn.Module):
    """``sigmoid(W4 g + b4) * (W5 g + b5)`` with optional dropout on ``g`` first."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        if in_dim < 1 or out_dim < 1:
            raise ValueError("in_dim and out_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.dropout = nn.Dropout(dropout)
        self.value = nn.Linear(in_dim, out_dim)
        self.gate = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        return torch.sigmoid(self.gate(x)) * self.value(x)


class GateAddNorm(nn.Module):
    """``LayerNorm(residual + GLU(x))``, the gated skip connection of TFT."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.glu = GatedLinearUnit(in_dim, out_dim, dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.norm(residual + self.glu(x))


class GatedResidualNetwork(nn.Module):
    """GRN(a, c) on ``[..., in_dim]`` with an optional context ``[..., context_dim]``.

    The context must broadcast against ``a`` after its last axis is projected,
    for example ``[B, 1, d]`` against ``[B, T, d]``.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int | None = None,
        context_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if in_dim < 1 or hidden_dim < 1:
            raise ValueError("in_dim and hidden_dim must be positive")
        out_dim = in_dim if out_dim is None else out_dim
        if out_dim < 1 or (context_dim is not None and context_dim < 1):
            raise ValueError("out_dim and context_dim must be positive")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.skip = nn.Identity() if out_dim == in_dim else nn.Linear(in_dim, out_dim)
        self.input_proj = nn.Linear(in_dim, hidden_dim)  # W2, b2
        self.context_proj = (
            nn.Linear(context_dim, hidden_dim, bias=False) if context_dim else None  # W3
        )
        self.hidden_proj = nn.Linear(hidden_dim, hidden_dim)  # W1, b1
        self.gate_norm = GateAddNorm(hidden_dim, out_dim, dropout)

    def forward(self, a: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        if a.shape[-1] != self.in_dim:
            raise ValueError(f"expected last axis {self.in_dim}, got {a.shape[-1]}")
        hidden = self.input_proj(a)
        if context is not None:
            if self.context_proj is None:
                raise ValueError("this GRN was built without a context input")
            hidden = hidden + self.context_proj(context)
        hidden = self.hidden_proj(F.elu(hidden))
        return self.gate_norm(hidden, self.skip(a))


class VariableSelectionNetwork(nn.Module):
    """Softmax-weighted sum of per-variable GRN outputs (TFT Eq. 6-8).

    ``forward(xi, context)`` takes ``xi`` shaped ``[..., num_vars, dim]`` and an
    optional context broadcastable to ``[..., context_dim]``; it returns the
    combined ``[..., dim]`` features and the selection weights ``[..., num_vars]``.
    """

    def __init__(
        self,
        num_vars: int,
        dim: int,
        context_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_vars < 1 or dim < 1:
            raise ValueError("num_vars and dim must be positive")
        self.num_vars = num_vars
        self.dim = dim
        self.weight_grn = GatedResidualNetwork(
            num_vars * dim, dim, out_dim=num_vars, context_dim=context_dim, dropout=dropout
        )
        self.variable_grns = nn.ModuleList(
            GatedResidualNetwork(dim, dim, dropout=dropout) for _ in range(num_vars)
        )

    def forward(
        self, xi: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if xi.shape[-2:] != (self.num_vars, self.dim):
            raise ValueError(
                f"expected [..., {self.num_vars}, {self.dim}], got {tuple(xi.shape)}"
            )
        weights = torch.softmax(self.weight_grn(xi.flatten(-2), context), dim=-1)
        processed = torch.stack(
            [grn(xi[..., j, :]) for j, grn in enumerate(self.variable_grns)], dim=-2
        )
        return (weights.unsqueeze(-1) * processed).sum(dim=-2), weights


__all__ = [
    "GateAddNorm",
    "GatedLinearUnit",
    "GatedResidualNetwork",
    "VariableSelectionNetwork",
]
