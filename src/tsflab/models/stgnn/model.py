"""STGNN: positional-attention graph GRU plus Transformer (Wang et al., WWW 2020).

Implemented from the paper only (no official code was released). Equations:

- S-GNN (Eq. 3-7): learned per-node positional vectors ``p_i``;
  ``R = softmax_j(p_i^T p_j)`` over all nodes, kept only where ``A + I > 0``,
  ``R~ = mask(R) + I`` and ``f_a(X) = relu(D_R^-1/2 R~ D_R^-1/2 X W)``.
- GRU (Eq. 8-9): ``f_a`` filters the step input and the previous state, then a GRU
  update per node with shared weights (``graph_conv_gru`` gating, linear maps).
- Transformer (Eq. 10-15): sinusoidal position plus one post-norm encoder layer
  over time per node.
- Prediction (Sec. 3.3): a feed-forward network maps each node's sequence to the horizon.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tsflab.models._components.embed import PositionalEmbedding
from tsflab.models._components.graph_conv_gru import graph_gru_step


class PositionalGraphFilter(nn.Module):
    """``f_a``: graph convolution over the masked positional relation matrix (Eq. 6)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, propagation: torch.Tensor) -> torch.Tensor:
        return F.relu(self.weight(torch.matmul(propagation, x)))


class Model(nn.Module):
    """STGNN for node series ``[B, L, N]`` on a given road graph."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        adj_mx: np.ndarray | None = None,
        hidden_dim: int = 64,
        n_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if adj_mx is None:
            raise ValueError(
                "STGNN needs the road-network adjacency adj_mx [enc_in, enc_in] for its relation mask; "
                "use a graph dataset that ships adj_mx.npy"
            )
        adjacency = np.asarray(adj_mx, dtype=np.float64)
        if adjacency.shape != (enc_in, enc_in):
            raise ValueError(f"adj_mx must have shape {(enc_in, enc_in)}")
        if hidden_dim % 2 or hidden_dim % n_heads:
            raise ValueError("hidden_dim must be even and divisible by n_heads")
        self.seq_len, self.pred_len, self.num_nodes = seq_len, pred_len, enc_in
        self.hidden_dim = hidden_dim
        # Eq. 5: keep relations only where A~ = A + I is positive.
        allowed = (adjacency + np.eye(enc_in)) > 0
        self.register_buffer("relation_mask", torch.as_tensor(allowed))
        self.positions = nn.Parameter(torch.empty(enc_in, hidden_dim))
        nn.init.xavier_uniform_(self.positions)

        self.input_filter = PositionalGraphFilter(1, hidden_dim)
        self.state_filter = PositionalGraphFilter(hidden_dim, hidden_dim)
        self.gates = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        self.candidate = nn.Linear(2 * hidden_dim, hidden_dim)

        self.position = PositionalEmbedding(hidden_dim, max_len=seq_len)
        self.transformer = nn.TransformerEncoderLayer(
            hidden_dim, n_heads, dim_feedforward=4 * hidden_dim, dropout=dropout,
            activation="relu", batch_first=True, norm_first=False,
        )
        self.head = nn.Sequential(
            nn.Linear(seq_len * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, pred_len)
        )

    def propagation(self) -> torch.Tensor:
        """``D_R^-1/2 (mask(R) + I) D_R^-1/2`` from the positional relation (Eq. 3-6)."""
        relation = torch.softmax(self.positions @ self.positions.T, dim=-1)
        relation = relation * self.relation_mask
        relation = relation + torch.eye(self.num_nodes, device=relation.device, dtype=relation.dtype)
        inverse_sqrt = relation.sum(dim=-1).rsqrt()
        return inverse_sqrt[:, None] * relation * inverse_sqrt[None, :]

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.num_nodes):
            raise ValueError(f"x_enc must have shape [batch, {self.seq_len}, {self.num_nodes}]")
        propagation = self.propagation()
        batch = x_enc.shape[0]
        state = x_enc.new_zeros(batch, self.num_nodes, self.hidden_dim)
        states = []
        for step in range(self.seq_len):
            filtered_input = self.input_filter(x_enc[:, step, :].unsqueeze(-1), propagation)
            filtered_state = self.state_filter(state, propagation)
            state = graph_gru_step(filtered_input, filtered_state, self.gates, self.candidate)
            states.append(state)
        sequence = torch.stack(states, dim=2)  # [B, N, L, H]
        sequence = sequence + self.position(x_enc)
        encoded = self.transformer(sequence.reshape(batch * self.num_nodes, self.seq_len, -1))
        forecast = self.head(encoded.reshape(batch, self.num_nodes, -1))  # [B, N, P]
        return forecast.transpose(1, 2)
