"""Local STG-NCDE: coupled temporal and spatial neural controlled differential equations.

Independent rewrite of Choi et al., AAAI 2022 (arXiv 2112.03558), checked against
the pinned official code (jeongwhanchoi/STG-NCDE at 49480bdf, MIT). Equation
numbers refer to the paper.

Pipeline for one window ``x [B, T, N]``:

1. Control path: per node, the natural cubic spline of ``[t, x_t]`` over the
   unit time grid ``t = 0 .. T-1`` (official code prepends the time channel).
2. Initial values from the first observation ``X(t_0)``:
   ``h(0) = FC(X(t_0))`` and, as in the official code, ``z(0) = FC'(X(t_0))``.
3. Augmented ODE (Eq. 12), solved with ``torchdiffeq``::

       dh/dt = f(h) dX/dt
       dz/dt = g(z) f(h) dX/dt

   ``f`` (Eq. 8): FC, ReLU, ``num_layers - 1`` x (FC, ReLU), FC to
   ``hidden x C``, tanh. ``g`` (Eqs. 9-11 as in the official code): FC, ReLU,
   node-adaptive graph convolution over ``softmax(relu(E E^T))`` with
   ``cheb_k`` terms, FC to ``hidden x hidden``, tanh.
4. Output (Eq. 13): a linear map of ``z(T)`` to the ``pred_len`` steps of
   every node.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchdiffeq

from tsflab.models._components.marks import to_spatiotemporal
from tsflab.models._components.natural_cubic_spline import NaturalCubicSpline, natural_cubic_spline_coeffs
from tsflab.models._components.node_adaptive_graph_conv import NodeAdaptiveGraphConv

SOLVERS = ("rk4", "euler", "midpoint", "dopri5")


class TemporalField(nn.Module):
    """CDE function ``f`` (Eq. 8): ``[B, N, hidden] -> [B, N, hidden, channels]``."""

    def __init__(self, channels: int, hidden_dim: int, hidden_hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.channels, self.hidden_dim = channels, hidden_dim
        self.linear_in = nn.Linear(hidden_dim, hidden_hidden_dim)
        self.linears = nn.ModuleList(
            nn.Linear(hidden_hidden_dim, hidden_hidden_dim) for _ in range(num_layers - 1)
        )
        self.linear_out = nn.Linear(hidden_hidden_dim, hidden_dim * channels)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        out = self.linear_in(h).relu()
        for linear in self.linears:
            out = linear(out).relu()
        return self.linear_out(out).view(*h.shape[:-1], self.hidden_dim, self.channels).tanh()


class SpatialField(nn.Module):
    """CDE function ``g`` (Eqs. 9-11): ``[B, N, hidden] -> [B, N, hidden, hidden]``.

    The graph step mixes nodes with a node-adaptive filter over the learned graph
    ``softmax(relu(E E^T))``: ``sum_k T_k(A) B_0 W_{n,k} + b_n`` (official
    ``VectorField_g.agc``); the paper writes it as ``(I + A) B_0 W_spatial``.
    """

    def __init__(self, num_nodes: int, hidden_dim: int, hidden_hidden_dim: int, embed_dim: int, cheb_k: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear_in = nn.Linear(hidden_dim, hidden_hidden_dim)
        self.node_embeddings = nn.Parameter(torch.empty(num_nodes, embed_dim))
        self.graph_conv = NodeAdaptiveGraphConv(hidden_hidden_dim, hidden_hidden_dim, cheb_k, embed_dim)
        self.linear_out = nn.Linear(hidden_hidden_dim, hidden_dim * hidden_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.linear_in(z).relu()
        out = self.graph_conv(out, self.node_embeddings)
        return self.linear_out(out).view(*z.shape[:-1], self.hidden_dim, self.hidden_dim).tanh()


class CoupledVectorField(nn.Module):
    """Right-hand side of the augmented ODE (Eq. 12) for the state ``(h, z)``.

    Built per forward call around the shared ``f`` and ``g`` modules and that
    batch's control path, so the adjoint backward pass sees the same path.
    """

    def __init__(self, temporal: TemporalField, spatial: SpatialField, control: NaturalCubicSpline) -> None:
        super().__init__()
        self.temporal = temporal
        self.spatial = spatial
        self.control = control

    def forward(self, t: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        h, z = state
        control = self.control.derivative(t).unsqueeze(-1)  # [B, N, C, 1]
        dh = self.temporal(h) @ control  # f(h) dX/dt, [B, N, hidden, 1]
        dz = self.spatial(z) @ dh  # g(z) f(h) dX/dt
        return dh.squeeze(-1), dz.squeeze(-1)


class Model(nn.Module):
    """Spatio-temporal graph neural controlled differential equation forecaster."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        adj_mx: np.ndarray | None = None,
        input_dim: int = 1,
        hidden_dim: int = 64,
        hidden_hidden_dim: int = 64,
        num_layers: int = 2,
        embed_dim: int = 10,
        cheb_k: int = 2,
        solver: str = "rk4",
        adjoint: bool = True,
    ) -> None:
        super().__init__()
        del adj_mx  # the graph is learned from node embeddings
        if min(pred_len, num_nodes, input_dim, hidden_dim, hidden_hidden_dim, num_layers, embed_dim, cheb_k) < 1:
            raise ValueError("STGNCDE dimensions must be positive")
        if seq_len < 2:
            raise ValueError("STGNCDE needs at least two input steps for the spline path")
        if solver not in SOLVERS:
            raise ValueError(f"solver must be one of {SOLVERS}")
        self.seq_len, self.pred_len, self.num_nodes = seq_len, pred_len, num_nodes
        self.input_dim, self.solver, self.adjoint = input_dim, solver, adjoint
        channels = input_dim + 1  # time channel + data channels
        self.initial_h = nn.Linear(channels, hidden_dim)
        self.initial_z = nn.Linear(channels, hidden_dim)
        self.temporal = TemporalField(channels, hidden_dim, hidden_hidden_dim, num_layers)
        self.spatial = SpatialField(num_nodes, hidden_dim, hidden_hidden_dim, embed_dim, cheb_k)
        # Eq. 13 as the official 1 x hidden convolution over each node's z(T).
        self.end_conv = nn.Conv2d(1, pred_len, kernel_size=(1, hidden_dim))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Official ``Run_cde.py``: Xavier-uniform for every matrix, U(0, 1) for every vector."""
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
            else:
                nn.init.uniform_(parameter)

    def _control_path(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor | None) -> tuple[torch.Tensor, NaturalCubicSpline]:
        if self.input_dim == 1:
            features = x_enc.unsqueeze(-1)
        else:
            features = to_spatiotemporal(x_enc, x_mark_enc)
            if features.shape[-1] < self.input_dim:
                raise ValueError("STGNCDE received fewer input features than input_dim")
            features = features[..., : self.input_dim]
        times = torch.arange(self.seq_len, device=x_enc.device, dtype=x_enc.dtype)
        time_channel = times.view(1, -1, 1, 1).expand(*features.shape[:-1], 1)
        path = torch.cat((time_channel, features), dim=-1).transpose(1, 2)  # [B, N, T, C]
        return times, NaturalCubicSpline(times, natural_cubic_spline_coeffs(times, path))

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.num_nodes):
            raise ValueError(f"STGNCDE expects (B, {self.seq_len}, {self.num_nodes}) values")
        times, spline = self._control_path(x_enc, x_mark_enc)
        start = spline.evaluate(times[0])
        state = (self.initial_h(start), self.initial_z(start))
        field = CoupledVectorField(self.temporal, self.spatial, spline)
        options = {"method": self.solver}
        if self.solver == "dopri5":  # official tolerances; fixed-grid solvers ignore them
            options.update(rtol=1e-7, atol=1e-9)
        if self.adjoint:
            solution = torchdiffeq.odeint_adjoint(
                field, state, times, adjoint_params=tuple(field.parameters()), **options
            )
        else:
            solution = torchdiffeq.odeint(field, state, times, **options)
        final = solution[1][-1]  # z(T), [B, N, hidden]
        return self.end_conv(final.unsqueeze(1)).squeeze(-1)  # [B, pred_len, N]


__all__ = ["CoupledVectorField", "Model", "SOLVERS", "SpatialField", "TemporalField"]
