"""Independent MSGNet implementation from the AAAI paper.

FFT amplitudes identify salient periods. Every scale has its own learned graph,
MixHop propagation captures inter-series relations, and self-attention inside
period segments captures intra-series temporal dependence.
"""
from __future__ import annotations

import torch
from torch import nn

from tsflab.models._components.dominant_periods import dominant_periods


class AdaptiveMixHopGraph(nn.Module):
    """Learn a directed adjacency and concatenate zero-to-K-hop messages."""
    def __init__(self, nodes: int, hidden: int, depth: int, alpha: float, dropout: float):
        super().__init__()
        self.source = nn.Parameter(torch.randn(nodes, hidden) * 0.1)
        self.target = nn.Parameter(torch.randn(hidden, nodes) * 0.1)
        self.depth, self.alpha = depth, alpha
        # A shared scalar bias would be erased by the following node LayerNorm.
        self.projection = nn.Linear(depth + 1, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def adjacency(self):
        # A small negative slope keeps both low-rank factors trainable before
        # any edge receives a positive score.
        return torch.softmax(torch.nn.functional.leaky_relu(self.source @ self.target, 0.01), dim=-1)

    def forward(self, values):
        # values: B,S,P,C
        adjacency = self.adjacency()
        states = [values]
        current = values
        for _ in range(self.depth):
            propagated = torch.einsum("ij,bspj->bspi", adjacency, current)
            current = self.alpha * values + (1 - self.alpha) * propagated
            states.append(current)
        return self.dropout(self.projection(torch.stack(states, -1)).squeeze(-1))


_ATTENTION_CHUNK = 32768


class ScaleGraphBranch(nn.Module):
    def __init__(self, nodes, width, heads, graph_hidden, graph_depth, alpha, dropout):
        super().__init__()
        self.input = nn.Linear(1, width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.output = nn.Linear(width, 1)
        self.graph = AdaptiveMixHopGraph(nodes, graph_hidden, graph_depth, alpha, dropout)
        self.norm = nn.LayerNorm(nodes)

    def forward(self, values, period):
        batch, length, nodes = values.shape
        padded_length = ((length + period - 1) // period) * period
        padded = torch.nn.functional.pad(values, (0, 0, 0, padded_length - length))
        segments = padded.reshape(batch, -1, period, nodes)
        token = self.input(segments.permute(0, 1, 3, 2).unsqueeze(-1))
        shape = token.shape
        token = token.reshape(-1, period, shape[-1])
        # PyTorch's fused attention kernels reject more than 65535 sequences per call
        # (batch x segments x nodes reaches that on wide datasets); chunk, same result.
        attended = torch.cat([self.attention(chunk, chunk, chunk, need_weights=False)[0]
                              for chunk in token.split(_ATTENTION_CHUNK)], dim=0)
        temporal = self.output(attended).reshape(shape[:-1]).permute(0, 1, 3, 2)
        mixed = self.graph(temporal)
        return self.norm(segments + mixed).reshape(batch, padded_length, nodes)[:, :length]


class MultiScaleGraphBlock(nn.Module):
    def __init__(self, nodes, top_k, width, heads, graph_hidden, graph_depth, alpha, dropout):
        super().__init__()
        self.top_k = top_k
        self.branches = nn.ModuleList([
            ScaleGraphBranch(nodes, width, heads, graph_hidden, graph_depth, alpha, dropout)
            for _ in range(top_k)
        ])
        self.norm = nn.LayerNorm(nodes)

    def forward(self, values):
        periods, strengths = dominant_periods(values, self.top_k)
        outputs = torch.stack([
            branch(values, max(1, int(period)))
            for branch, period in zip(self.branches, periods, strict=True)
        ], -1)
        weights = strengths.softmax(-1)[:, None, None, :]
        return self.norm(values + torch.sum(outputs * weights, -1))


class Model(nn.Module):
    def __init__(self, seq_len, pred_len, label_len=0, features="M", enc_in=7,
                 c_out=None, d_model=128, d_ff=256, e_layers=2, n_heads=8,
                 top_k=5, dropout=0.1, conv_channel=32, skip_channel=32,
                 gcn_depth=2, propalpha=0.3, node_dim=10, individual=False,
                 embed="timeF", freq="h"):
        super().__init__()
        c_out = enc_in if c_out is None else c_out
        if c_out != enc_in or d_model % n_heads:
            raise ValueError("MSGNet requires c_out == enc_in and d_model divisible by n_heads")
        if top_k > seq_len // 2:
            raise ValueError("top_k exceeds available non-DC FFT bins")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.blocks = nn.ModuleList([
            MultiScaleGraphBlock(enc_in, top_k, d_model, n_heads, node_dim,
                                 gcn_depth, propalpha, dropout)
            for _ in range(e_layers)
        ])
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(seq_len, pred_len))

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        if x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"expected (*,{self.seq_len},{self.enc_in})")
        mean = x_enc.mean(1, keepdim=True).detach()
        scale = x_enc.var(1, keepdim=True, unbiased=False).add(1e-5).sqrt().detach()
        hidden = (x_enc - mean) / scale
        for block in self.blocks:
            hidden = block(hidden)
        forecast = self.head(hidden.transpose(1, 2)).transpose(1, 2)
        return forecast * scale + mean
