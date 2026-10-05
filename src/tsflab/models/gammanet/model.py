"""Local GAMMA-Net written from the paper and a review of the pinned partial code.

GAMMA-Net embeds every (step, node) observation STAEformer-style (projected raw
features, time-of-day and day-of-week tables, a spatio-temporal adaptive
embedding), then runs two interleaved stacks (Eq. 3):
``(GAT -> temporal Mamba) x L`` followed by ``(GAT -> spatial Mamba) x L``, each
sub-layer with a residual connection and LayerNorm (Eq. 5), and regresses the
horizon from the flattened time-feature axis of every node (Eq. 9).
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.mamba import MambaBlock
from tsflab.models._components.marks import to_spatiotemporal


def edge_list(adj_mx, num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Directed edges ``src -> dst`` for every non-zero off-diagonal ``adj[src, dst]``,
    plus one self-loop per node (the GAT layer always attends to itself).

    Without an adjacency only the self-loops remain.
    """
    loops = torch.arange(num_nodes)
    if adj_mx is None:
        return loops, loops.clone()
    adj = torch.as_tensor(np.asarray(adj_mx, dtype=np.float64))
    if adj.shape != (num_nodes, num_nodes):
        raise ValueError(f"adj_mx must have shape ({num_nodes}, {num_nodes})")
    mask = (adj != 0) & ~torch.eye(num_nodes, dtype=torch.bool)
    src, dst = mask.nonzero(as_tuple=True)
    return torch.cat((src, loops)), torch.cat((dst, loops))


class GraphAttentionConv(nn.Module):
    """Multi-head graph attention (Velickovic et al., 2018) on a fixed edge list.

    ``h = W x`` per head; ``e_ij = LeakyReLU(a_src . h_j + a_dst . h_i)`` for each
    edge ``j -> i``; ``alpha_ij`` is the softmax of ``e_ij`` over the incoming edges
    of ``i``; ``out_i = mean_heads(sum_j alpha_ij h_j) + b`` (heads averaged, not
    concatenated). Glorot-initialized weights, zero bias, no attention dropout.
    """

    def __init__(self, width: int, heads: int, negative_slope: float = 0.2) -> None:
        super().__init__()
        self.width, self.heads, self.negative_slope = width, heads, negative_slope
        self.lin = nn.Linear(width, heads * width, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, 1, heads, width))
        self.att_dst = nn.Parameter(torch.empty(1, 1, heads, width))
        self.bias = nn.Parameter(torch.zeros(width))
        nn.init.xavier_uniform_(self.lin.weight)
        bound = math.sqrt(6.0 / (heads + width))
        nn.init.uniform_(self.att_src, -bound, bound)
        nn.init.uniform_(self.att_dst, -bound, bound)

    def attention(self, x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return projected features ``[G, N, H, D]`` and edge weights ``[G, E, H]``."""
        graphs, nodes, _ = x.shape
        h = self.lin(x).view(graphs, nodes, self.heads, self.width)
        score_src = (h * self.att_src).sum(-1)
        score_dst = (h * self.att_dst).sum(-1)
        logits = F.leaky_relu(score_src[:, src] + score_dst[:, dst], self.negative_slope)
        index = dst.view(1, -1, 1).expand_as(logits)
        peak = logits.new_full((graphs, nodes, self.heads), float("-inf"))
        peak = peak.scatter_reduce(1, index, logits.detach(), reduce="amax", include_self=True)
        weights = (logits - peak.gather(1, index)).exp()
        total = weights.new_zeros(graphs, nodes, self.heads).index_add(1, dst, weights)
        return h, weights / (total.gather(1, index) + 1e-16)

    def forward(self, x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        h, alpha = self.attention(x, src, dst)
        messages = h[:, src] * alpha.unsqueeze(-1)
        out = h.new_zeros(h.shape).index_add(1, dst, messages)
        return out.mean(dim=2) + self.bias


class GATSublayer(nn.Module):
    """Eqs. (4)-(5): GAT on the road graph at every time step, then ``LN(x + GAT(x))``."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.gat = GraphAttentionConv(width, heads)
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        batch, steps, nodes, width = x.shape
        out = self.gat(x.reshape(batch * steps, nodes, width), src, dst)
        return self.norm(x + out.view(batch, steps, nodes, width))


class AxisMamba(nn.Module):
    """Eqs. (6) and (8): a Mamba scan along one axis of ``[B, T, N, D]``, then
    ``LN(x + Mamba(x))``.

    ``layout="axis"`` scans along time (``axis="time"``: one length-``T`` sequence
    per node) or along nodes (``axis="node"``: one length-``N`` sequence per step),
    as the paper describes. ``layout="official"`` reproduces the pinned code, which
    reshapes the (moved) tensor to ``[-1, length, D]`` in memory order, so each
    "temporal" sequence is a run of ``T`` consecutive entries of the step-major
    ``(T, N)`` grid and each "spatial" sequence a run of ``N`` consecutive entries
    of the node-major ``(N, T)`` grid.
    """

    def __init__(
        self,
        width: int,
        axis: Literal["time", "node"],
        layout: Literal["axis", "official"],
        d_state: int,
        d_conv: int,
        expand: int,
    ) -> None:
        super().__init__()
        if axis not in ("time", "node") or layout not in ("axis", "official"):
            raise ValueError("axis must be 'time' or 'node' and layout 'axis' or 'official'")
        self.axis, self.layout = axis, layout
        # The scan runs over every (step, node) token; storing its
        # [tokens, d_inner, d_state] tensors needs about 7.6 GB per layer on
        # PEMS08 at batch 64. The official mamba_ssm kernel never stores them,
        # so the scan is recomputed in backward instead.
        self.mixer = MambaBlock(
            width, expand * width, math.ceil(width / 16), d_conv, d_state,
            reference_dt_init=True, checkpoint_scan=True,
        )
        self.norm = nn.LayerNorm(width)

    def _scan(self, sequences: torch.Tensor) -> torch.Tensor:
        return self.norm(sequences + self.mixer(sequences))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, nodes, width = x.shape
        if self.layout == "axis":
            if self.axis == "time":
                seq = x.transpose(1, 2).reshape(batch * nodes, steps, width)
                return self._scan(seq).view(batch, nodes, steps, width).transpose(1, 2)
            return self._scan(x.reshape(batch * steps, nodes, width)).view(batch, steps, nodes, width)
        if self.axis == "time":
            return self._scan(x.reshape(-1, steps, width)).reshape(batch, steps, nodes, width)
        moved = x.transpose(1, 2)
        out = self._scan(moved.reshape(-1, nodes, width)).reshape(batch, nodes, steps, width)
        return out.transpose(1, 2)


class Model(nn.Module):
    """Interleaved GAT / multi-axis Mamba spatio-temporal forecaster."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        adj_mx=None,
        steps_per_day: int = 288,
        input_embedding_dim: int = 24,
        tod_embedding_dim: int = 24,
        dow_embedding_dim: int = 24,
        adaptive_embedding_dim: int = 80,
        num_heads: int = 4,
        num_layers: int = 3,
        gat: bool = True,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        scan_layout: Literal["axis", "official"] = "axis",
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len, num_nodes, steps_per_day, input_embedding_dim, num_heads, num_layers) < 1:
            raise ValueError("lengths, nodes, widths, heads and layers must be positive")
        if min(tod_embedding_dim, dow_embedding_dim, adaptive_embedding_dim) < 0:
            raise ValueError("embedding widths must be non-negative")
        self.seq_len, self.pred_len, self.num_nodes = seq_len, pred_len, num_nodes
        self.steps_per_day = steps_per_day
        self.gat = gat
        self.width = input_embedding_dim + tod_embedding_dim + dow_embedding_dim + adaptive_embedding_dim

        src, dst = edge_list(adj_mx, num_nodes)
        self.register_buffer("edge_src", src, persistent=False)
        self.register_buffer("edge_dst", dst, persistent=False)

        # Eq. (2): one linear map of the raw [flow, time-of-day, day-of-week] features.
        self.input_proj = nn.Linear(3, input_embedding_dim)
        self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim) if tod_embedding_dim else None
        self.dow_embedding = nn.Embedding(7, dow_embedding_dim) if dow_embedding_dim else None
        self.adaptive_embedding = (
            nn.Parameter(torch.empty(seq_len, num_nodes, adaptive_embedding_dim)) if adaptive_embedding_dim else None
        )
        if self.adaptive_embedding is not None:
            nn.init.xavier_uniform_(self.adaptive_embedding)

        def stack(axis: Literal["time", "node"]) -> nn.ModuleList:
            layers: list[nn.Module] = []
            for _ in range(num_layers):
                if gat:
                    layers.append(GATSublayer(self.width, num_heads))
                layers.append(AxisMamba(self.width, axis, scan_layout, d_state, d_conv, expand))
            return nn.ModuleList(layers)

        self.temporal_stack = stack("time")
        self.spatial_stack = stack("node")
        # Eq. (9): per node, flatten (T, d_h) and regress the T' horizon.
        self.output_proj = nn.Linear(seq_len * self.width, pred_len)

    def embed(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor | None) -> torch.Tensor:
        """Section 3.2.1: ``Z = [W X + b || T_d[tod] || T_w[dow] || E_a]`` of width ``d_h``."""
        data = to_spatiotemporal(x_enc, x_mark_enc)
        if data.shape[-1] != 3:
            raise ValueError("GAMMANet expects [value, time_in_day, day_in_week] node features")
        tod_index = torch.round(data[..., 1] * self.steps_per_day).long().clamp(0, self.steps_per_day - 1)
        dow_index = torch.round(data[..., 2] * 7).long().clamp(0, 6)
        raw = torch.stack((data[..., 0], data[..., 1], dow_index.to(data.dtype)), dim=-1)
        pieces = [self.input_proj(raw)]
        if self.tod_embedding is not None:
            pieces.append(self.tod_embedding(tod_index))
        if self.dow_embedding is not None:
            pieces.append(self.dow_embedding(dow_index))
        if self.adaptive_embedding is not None:
            pieces.append(self.adaptive_embedding.unsqueeze(0).expand(x_enc.shape[0], -1, -1, -1))
        return torch.cat(pieces, dim=-1)

    def _run(self, stack: nn.ModuleList, hidden: torch.Tensor) -> torch.Tensor:
        for layer in stack:
            if isinstance(layer, GATSublayer):
                hidden = layer(hidden, self.edge_src, self.edge_dst)
            else:
                hidden = layer(hidden)
        return hidden

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_dec, x_mark_dec
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.num_nodes):
            raise ValueError(f"GAMMANet expects (B, {self.seq_len}, {self.num_nodes}) values")
        hidden = self.embed(x_enc, x_mark_enc)
        hidden = self._run(self.temporal_stack, hidden)
        hidden = self._run(self.spatial_stack, hidden)
        flat = hidden.transpose(1, 2).reshape(x_enc.shape[0], self.num_nodes, self.seq_len * self.width)
        return self.output_proj(flat).transpose(1, 2)


__all__ = ["AxisMamba", "GATSublayer", "GraphAttentionConv", "Model", "edge_list"]
