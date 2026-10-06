"""STSGCN: spatial-temporal synchronous graph convolutional network.

Independent implementation of Song et al. (AAAI 2020), checked against the
official MXNet code at revision 3f825f64 (no license; nothing copied).
Execution order:

* input layer: position-wise FC to ``first_layer_embedding`` features + ReLU;
* ``len(filter_list)`` STSGCLs: add a learnable temporal ``[T, C]`` and spatial
  ``[N, C]`` embedding, cut ``T - 2`` windows of 3 steps, run one STSGCM per
  window (or one shared module) on the masked localized graph, keep the middle
  step; each layer shortens the sequence by 2;
* output layer: for every horizon step its own ``ReLU(X W1 + b1) W2 + b2`` on the
  flattened ``[T_last * C]`` features of each node (Eq. 8).

The learnable mask ``W_mask`` (Eq. 7) multiplies the localized adjacency
element-wise and is shared by every graph convolution.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from tsflab.models._components.synchronous_graph_conv import (
    SynchronousGraphModule,
    localized_adjacency,
    mxnet_xavier_uniform_,
)

_WINDOW = 3


class STSGCL(nn.Module):
    """Position embeddings plus ``T - 2`` synchronous modules over sliding 3-step windows."""

    def __init__(
        self,
        seq_len: int,
        num_nodes: int,
        in_dim: int,
        filters: list[int],
        activation: str,
        individual: bool,
        temporal_emb: bool,
        spatial_emb: bool,
        magnitude: float,
    ) -> None:
        super().__init__()
        self.seq_len, self.num_nodes = seq_len, num_nodes
        windows = seq_len - _WINDOW + 1
        # Official shapes (1, T, 1, C) and (1, 1, N, C) keep MXNet's Xavier fan rule.
        self.temporal_embedding = self._embedding((1, seq_len, 1, in_dim), magnitude) if temporal_emb else None
        self.spatial_embedding = self._embedding((1, 1, num_nodes, in_dim), magnitude) if spatial_emb else None
        self.module = SynchronousGraphModule(
            num_nodes, _WINDOW, in_dim, filters, activation=activation,
            num_modules=windows if individual else 1, crop=1, init_magnitude=magnitude,
        )

    @staticmethod
    def _embedding(shape: tuple[int, ...], magnitude: float) -> nn.Parameter:
        receptive = int(np.prod(shape[2:]))
        return nn.Parameter(
            mxnet_xavier_uniform_(torch.empty(shape), shape[1] * receptive, shape[0] * receptive, magnitude)
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if self.temporal_embedding is not None:
            x = x + self.temporal_embedding
        if self.spatial_embedding is not None:
            x = x + self.spatial_embedding
        batch, steps, nodes, features = x.shape
        windows = x.unfold(1, _WINDOW, 1)  # [B, T - 2, N, C, 3]
        windows = windows.permute(0, 1, 4, 2, 3).reshape(batch, steps - _WINDOW + 1, _WINDOW * nodes, features)
        return self.module(windows, adjacency)


class Model(nn.Module):
    """STSGCN with the localized graph built from the injected adjacency."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        adj_mx: np.ndarray | None = None,
        num_layers: int = 4,
        filters: tuple[int, ...] = (64, 64, 64),
        first_layer_embedding: int = 64,
        activation: str = "GLU",
        individual: bool = True,
        use_mask: bool = True,
        temporal_emb: bool = True,
        spatial_emb: bool = True,
        output_hidden: int = 128,
        init_magnitude: float = 0.0003,
        huber_delta: float = 1.0,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len, enc_in, num_layers, output_hidden) < 1:
            raise ValueError("lengths, nodes, layers and widths must be positive")
        if seq_len - 2 * num_layers < 1:
            raise ValueError(f"seq_len ({seq_len}) must exceed 2 * num_layers ({2 * num_layers})")
        if adj_mx is None:
            raise ValueError(
                "STSGCN needs the road-network adjacency adj_mx [enc_in, enc_in] for its localized "
                "spatial-temporal graph; use a graph dataset that ships adj_mx.npy (for example a PEMS preset)"
            )
        if np.asarray(adj_mx).shape != (enc_in, enc_in):
            raise ValueError(f"adj_mx must have shape {(enc_in, enc_in)}")
        self.seq_len, self.pred_len, self.num_nodes = seq_len, pred_len, enc_in
        self.huber_delta = huber_delta

        local = localized_adjacency(adj_mx, _WINDOW)
        self.register_buffer("adjacency", local)
        self.mask = nn.Parameter((local != 0).float()) if use_mask else None

        self.input_layer = nn.Linear(1, first_layer_embedding)
        layers = []
        steps, width = seq_len, first_layer_embedding
        for _ in range(num_layers):
            layers.append(
                STSGCL(steps, enc_in, width, list(filters), activation, individual, temporal_emb, spatial_emb, init_magnitude)
            )
            steps, width = steps - (_WINDOW - 1), filters[-1]
        self.layers = nn.ModuleList(layers)
        flat = steps * width
        # Eq. 8: one two-layer head per horizon step, stored as stacked weights.
        self.head_hidden = nn.Parameter(torch.empty(pred_len, flat, output_hidden))
        self.head_hidden_bias = nn.Parameter(torch.zeros(pred_len, output_hidden))
        self.head_output = nn.Parameter(torch.empty(pred_len, output_hidden))
        self.head_output_bias = nn.Parameter(torch.zeros(pred_len))
        mxnet_xavier_uniform_(self.input_layer.weight, 1, first_layer_embedding, init_magnitude)
        nn.init.zeros_(self.input_layer.bias)
        for horizon in range(pred_len):
            mxnet_xavier_uniform_(self.head_hidden[horizon], flat, output_hidden, init_magnitude)
            mxnet_xavier_uniform_(self.head_output[horizon], output_hidden, 1, init_magnitude)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_mark_enc, x_dec, x_mark_dec
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.num_nodes):
            raise ValueError(f"x_enc must have shape [batch, {self.seq_len}, {self.num_nodes}]")
        adjacency = self.adjacency if self.mask is None else self.mask * self.adjacency
        hidden = torch.relu(self.input_layer(x_enc.unsqueeze(-1)))
        for layer in self.layers:
            hidden = layer(hidden, adjacency)
        flat = hidden.permute(0, 2, 1, 3).flatten(2)  # [B, N, T_last * C]
        hidden = torch.relu(torch.einsum("bnf,qfh->bqnh", flat, self.head_hidden) + self.head_hidden_bias[:, None])
        return torch.einsum("bqnh,qh->bqn", hidden, self.head_output) + self.head_output_bias[:, None]


__all__ = ["Model", "STSGCL"]
