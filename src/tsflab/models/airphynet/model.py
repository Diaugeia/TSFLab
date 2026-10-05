"""Clean-room AirPhyNet from the ICLR 2024 diffusion-advection equations.

The implementation follows paper equations (9)--(12): a per-station GRU
encodes the observed pollutant/covariate sequence, a reparameterized initial
latent state is evolved by a graph differential equation, and a shared decoder
maps every future latent state back to pollutant concentration.  The ODE vector
field keeps the paper's two physical paths separate::

    dz/dt = -alpha * k * tanh(L z) - (1-alpha) * tanh(M z)

``L`` is the distance-graph Laplacian and ``M`` is a directed flow operator.
No implementation source from the former CauAir-derived file is retained.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.marks import to_spatiotemporal


def _row_normalize(matrix: torch.Tensor) -> torch.Tensor:
    return matrix / matrix.sum(-1, keepdim=True).clamp_min(1e-6)


def _placeholder_flow(nodes: int) -> torch.Tensor:
    """Directed ring used for advection when no flow graph is given (not the paper's wind graph)."""
    flow = torch.zeros(nodes, nodes)
    if nodes > 1:
        index = torch.arange(nodes)
        flow[index, (index + 1) % nodes] = 1
    return _row_normalize(flow)


class PhysicsVectorField(nn.Module):
    """AirPhyNet Eq. (11), with learnable diffusion and gated aggregation."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.diffusion_raw = nn.Parameter(torch.zeros(latent_dim))
        self.gate_logits = nn.Parameter(torch.zeros(latent_dim))

    def forward(
        self,
        state: torch.Tensor,
        distance_support: torch.Tensor,
        flow_support: torch.Tensor,
    ) -> torch.Tensor:
        laplacian = torch.diag(distance_support.sum(-1)) - distance_support
        flow_operator = torch.diag(flow_support.sum(-1)) - flow_support
        diffusion = torch.einsum("nm,bmd->bnd", laplacian, state)
        advection = torch.einsum("nm,bmd->bnd", flow_operator, state)
        coefficient = F.softplus(self.diffusion_raw).view(1, 1, -1)
        gate = torch.sigmoid(self.gate_logits).view(1, 1, -1)
        return (
            -gate * coefficient * torch.tanh(diffusion)
            - (1.0 - gate) * torch.tanh(advection)
        )


class Model(nn.Module):
    """Physics-guided graph Neural ODE forecaster for node concentrations."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        adj_mx: np.ndarray | torch.Tensor | None = None,
        flow_mx: np.ndarray | torch.Tensor | None = None,
        cov_dim: int = 2,
        latent_dim: int = 8,
        rnn_units: int = 32,
        ode_method: Literal["euler", "rk4"] = "rk4",
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len, enc_in, cov_dim, latent_dim, rnn_units) <= 0:
            raise ValueError("AirPhyNet dimensions must be positive")
        if ode_method not in {"euler", "rk4"}:
            raise ValueError("ode_method must be 'euler' or 'rk4'")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.cov_dim = cov_dim
        self.ode_method = ode_method

        if adj_mx is None:
            raise ValueError(
                "AirPhyNet needs the station distance graph adj_mx [enc_in, enc_in] for its diffusion "
                "Laplacian; use a graph dataset that ships adj_mx.npy"
            )
        distance = torch.as_tensor(adj_mx, dtype=torch.float32)
        if flow_mx is None:
            warnings.warn(
                "AirPhyNet: no flow_mx given; advection uses a directed ring placeholder, not the "
                "paper's wind-derived flow graph, so the advection term carries no physical meaning",
                stacklevel=2,
            )
            flow = _placeholder_flow(enc_in)
        else:
            flow = torch.as_tensor(flow_mx, dtype=torch.float32)
        if distance.shape != (enc_in, enc_in) or flow.shape != (enc_in, enc_in):
            raise ValueError("adj_mx and flow_mx must both have shape (enc_in, enc_in)")
        self.register_buffer("distance_support", _row_normalize(distance.clamp_min(0)))
        self.register_buffer("flow_support", _row_normalize(flow.clamp_min(0)))

        self.encoder = nn.GRU(1 + cov_dim, rnn_units, batch_first=True)
        self.initial_mean = nn.Linear(rnn_units, latent_dim)
        self.initial_scale = nn.Linear(rnn_units, latent_dim)
        self.vector_field = PhysicsVectorField(latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, rnn_units),
            nn.Tanh(),
            nn.Linear(rnn_units, 1),
        )

    def _ode_step(self, state: torch.Tensor) -> torch.Tensor:
        field = lambda z: self.vector_field(z, self.distance_support, self.flow_support)
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
            raise ValueError("AirPhyNet expects (batch, configured seq_len, enc_in)")
        if x_mark_enc is None:
            x_mark_enc = x_enc.new_zeros((x_enc.shape[0], self.seq_len, 6))
        features = to_spatiotemporal(x_enc, x_mark_enc)
        if features.shape[-1] != 1 + self.cov_dim:
            raise ValueError(f"AirPhyNet expects exactly {self.cov_dim} covariate channels")
        sequence = features.permute(0, 2, 1, 3).reshape(
            x_enc.shape[0] * self.enc_in, self.seq_len, 1 + self.cov_dim
        )
        _, hidden = self.encoder(sequence)
        hidden = hidden[-1].reshape(x_enc.shape[0], self.enc_in, -1)
        mean = self.initial_mean(hidden)
        scale = F.softplus(self.initial_scale(hidden)) + 1e-4
        state = mean + torch.randn_like(scale) * scale if self.training else mean
        predictions = []
        for _ in range(self.pred_len):
            state = self._ode_step(state)
            predictions.append(self.decoder(state).squeeze(-1))
        return torch.stack(predictions, dim=1)
