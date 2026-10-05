"""Clean-room Air-DualODE following the ICLR 2025 dual-dynamics design.

The explicit branch implements the boundary-aware diffusion-advection equation
(BA-DAE, paper Eq. 6), including its learnable open-system correction.  The
latent branch evolves a masked-attention graph state (Eqs. 9--10).  Their
time-aligned representations are concatenated and fused on the geographic
graph before decoding, as described after Decay-TCL in Eqs. 11--13.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.marks import to_spatiotemporal


def _normalize(matrix: torch.Tensor) -> torch.Tensor:
    return matrix / matrix.sum(-1, keepdim=True).clamp_min(1e-6)


def _placeholder_advection(nodes: int) -> torch.Tensor:
    """Directed ring used for advection when no flow graph is given (not the paper's wind graph)."""
    advection = torch.zeros(nodes, nodes)
    if nodes > 1:
        index = torch.arange(nodes)
        advection[index, (index + 1) % nodes] = 1
    return _normalize(advection)


class BoundaryAwareDynamics(nn.Module):
    """Paper Eq. (6): diffusion, advection, and source/sink correction."""

    def __init__(self, nodes: int, context_dim: int) -> None:
        super().__init__()
        self.coefficient_estimator = nn.GRU(context_dim, 3, batch_first=True)

    def coefficients(
        self, history: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # history: (B, N, T, C)
        batch, nodes, steps, channels = history.shape
        _, hidden = self.coefficient_estimator(history.reshape(batch * nodes, steps, channels))
        estimate = hidden[-1].reshape(batch, nodes, 3)
        gate = torch.sigmoid(estimate[..., :1])
        diffusion = F.softplus(estimate[..., 1:2])
        correction = F.softplus(estimate[..., 2:3]) - 1.0
        return gate, diffusion, correction

    def forward(
        self,
        state: torch.Tensor,
        diffusion_support: torch.Tensor,
        advection_support: torch.Tensor,
        gate: torch.Tensor,
        coefficient: torch.Tensor,
        correction: torch.Tensor,
    ) -> torch.Tensor:
        diagonal = torch.diag(diffusion_support.sum(-1))
        flow_diagonal = torch.diag(advection_support.sum(-1))
        laplacian = diagonal - diffusion_support
        diffusion = -torch.einsum("nm,bm->bn", laplacian, state)
        advection = torch.einsum("nm,bm->bn", advection_support - flow_diagonal, state)
        return gate.squeeze(-1) * coefficient.squeeze(-1) * diffusion + (
            1.0 - gate.squeeze(-1)
        ) * advection + correction.squeeze(-1) * state


class DataDrivenDynamics(nn.Module):
    """Masked-attention ODE field for dependencies omitted by BA-DAE."""

    def __init__(self, latent_dim: int, heads: int) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(latent_dim, heads, batch_first=True)
        self.local = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.SiLU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )

    def forward(self, state: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        blocked = support <= 0
        message, _ = self.attention(state, state, state, attn_mask=blocked)
        return torch.tanh(self.local(torch.cat([state, message], dim=-1)))


class Model(nn.Module):
    """Physics/data dual Neural ODE with graph fusion for air forecasting."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        adj_mx: np.ndarray | torch.Tensor | None = None,
        flow_mx: np.ndarray | torch.Tensor | None = None,
        cov_dim: int = 2,
        phy_latent_dim: int = 16,
        unk_latent_dim: int = 16,
        gcn_hidden_dim: int = 32,
        n_heads: int = 4,
        ode_method: Literal["euler", "rk4"] = "euler",
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len, enc_in, cov_dim, phy_latent_dim, unk_latent_dim, gcn_hidden_dim, n_heads) <= 0:
            raise ValueError("Air-DualODE dimensions must be positive")
        if unk_latent_dim % n_heads:
            raise ValueError("unk_latent_dim must be divisible by n_heads")
        if ode_method not in {"euler", "rk4"}:
            raise ValueError("ode_method must be 'euler' or 'rk4'")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.cov_dim = cov_dim
        self.ode_method = ode_method

        if adj_mx is None:
            raise ValueError(
                "Air-DualODE needs the station distance graph adj_mx [enc_in, enc_in] for diffusion, "
                "attention masking and graph fusion; use a graph dataset that ships adj_mx.npy"
            )
        diffusion = torch.as_tensor(adj_mx, dtype=torch.float32)
        if flow_mx is None:
            warnings.warn(
                "Air-DualODE: no flow_mx given; advection uses a directed ring placeholder, not the "
                "paper's wind-derived flow graph, so the advection term carries no physical meaning",
                stacklevel=2,
            )
            advection = _placeholder_advection(enc_in)
        else:
            advection = torch.as_tensor(flow_mx, dtype=torch.float32)
        if diffusion.shape != (enc_in, enc_in) or advection.shape != (enc_in, enc_in):
            raise ValueError("adj_mx and flow_mx must both have shape (enc_in, enc_in)")
        self.register_buffer("diffusion_support", _normalize(diffusion.clamp_min(0)))
        self.register_buffer("advection_support", _normalize(advection.clamp_min(0)))

        context_dim = 1 + cov_dim
        self.physics = BoundaryAwareDynamics(enc_in, context_dim)
        self.data_encoder = nn.GRU(context_dim, unk_latent_dim, batch_first=True)
        self.data_field = DataDrivenDynamics(unk_latent_dim, n_heads)
        self.physics_encoder = nn.Linear(1, phy_latent_dim)
        self.graph_fusion = nn.Sequential(
            nn.Linear(phy_latent_dim + unk_latent_dim, gcn_hidden_dim),
            nn.SiLU(),
            nn.Linear(gcn_hidden_dim * 2, gcn_hidden_dim),
            nn.SiLU(),
        )
        self.decoder = nn.Linear(gcn_hidden_dim, 1)

    def _integrate(self, state: torch.Tensor, field) -> torch.Tensor:
        if self.ode_method == "euler":
            return state + field(state)
        k1 = field(state)
        k2 = field(state + 0.5 * k1)
        k3 = field(state + 0.5 * k2)
        k4 = field(state + k3)
        return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_dec, x_mark_dec
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError("Air-DualODE expects (batch, configured seq_len, enc_in)")
        if x_mark_enc is None:
            x_mark_enc = x_enc.new_zeros((x_enc.shape[0], self.seq_len, 6))
        features = to_spatiotemporal(x_enc, x_mark_enc)
        if features.shape[-1] != 1 + self.cov_dim:
            raise ValueError(f"Air-DualODE expects exactly {self.cov_dim} covariates")
        history = features.permute(0, 2, 1, 3)
        gate, coefficient, correction = self.physics.coefficients(history)
        physical = x_enc[:, -1]
        sequence = history.reshape(
            x_enc.shape[0] * self.enc_in, self.seq_len, 1 + self.cov_dim
        )
        _, hidden = self.data_encoder(sequence)
        data = hidden[-1].reshape(x_enc.shape[0], self.enc_in, -1)

        outputs = []
        for _ in range(self.pred_len):
            physical = self._integrate(
                physical,
                lambda state: self.physics(
                    state,
                    self.diffusion_support,
                    self.advection_support,
                    gate,
                    coefficient,
                    correction,
                ),
            )
            data = self._integrate(
                data, lambda state: self.data_field(state, self.diffusion_support)
            )
            joint = torch.cat([self.physics_encoder(physical.unsqueeze(-1)), data], dim=-1)
            local = self.graph_fusion[1](self.graph_fusion[0](joint))
            neighbors = torch.einsum("nm,bmd->bnd", self.diffusion_support, local)
            fused = self.graph_fusion[3](
                self.graph_fusion[2](torch.cat([local, neighbors], dim=-1))
            )
            outputs.append(self.decoder(fused).squeeze(-1))
        return torch.stack(outputs, dim=1)
