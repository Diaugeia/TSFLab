"""Clean-room AirFormer from the AAAI 2023 deterministic/stochastic stages.

Each deterministic block factorizes attention into causal temporal MSA
(CT-MSA) and dartboard spatial MSA (DS-MSA).  The dartboard projection is an
explicit ``(query node, region, source node)`` tensor, so geographic datasets
can provide their paper preprocessing without coupling the model to one city.
The top-down stochastic stage samples a latent variable at every block during
training and uses its mean for deterministic evaluation.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.marks import to_spatiotemporal


class CausalTemporalAttention(nn.Module):
    """Windowed CT-MSA with a causal receptive field."""

    def __init__(self, d_model: int, heads: int, window: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = d_model // heads
        self.window = window
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, steps, width = values.shape
        q, k, v = self.qkv(values).chunk(3, dim=-1)
        reshape = lambda item: item.reshape(batch, steps, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = map(reshape, (q, k, v))
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        row = torch.arange(steps, device=values.device)[:, None]
        col = torch.arange(steps, device=values.device)[None, :]
        blocked = (col > row) | (col < row - self.window + 1)
        weights = self.dropout(scores.masked_fill(blocked, float("-inf")).softmax(-1))
        attended = torch.matmul(weights, v).transpose(1, 2).reshape(batch, steps, width)
        return self.output(attended)


class DartboardSpatialAttention(nn.Module):
    """DS-MSA: each station attends to M aggregated geographic regions."""

    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = d_model // heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
        # values: (B, N, D), projection: (N, M, N)
        batch, nodes, width = values.shape
        regions = torch.einsum("imn,bnd->bimd", projection, values)
        q = self.query(values).reshape(batch, nodes, self.heads, self.head_dim)
        k = self.key(regions).reshape(batch, nodes, regions.shape[2], self.heads, self.head_dim)
        v = self.value(regions).reshape(batch, nodes, regions.shape[2], self.heads, self.head_dim)
        scores = torch.einsum("bihd,bimhd->bihm", q, k) / math.sqrt(self.head_dim)
        weights = self.dropout(scores.softmax(-1))
        attended = torch.einsum("bihm,bimhd->bihd", weights, v).reshape(batch, nodes, width)
        return self.output(attended)


class AirFormerBlock(nn.Module):
    """Factorized CT-MSA then DS-MSA with pre-normalized residual MLPs."""

    def __init__(self, d_model: int, heads: int, window: int, dropout: float) -> None:
        super().__init__()
        self.temporal_norm = nn.LayerNorm(d_model)
        self.temporal = CausalTemporalAttention(d_model, heads, window, dropout)
        self.spatial_norm = nn.LayerNorm(d_model)
        self.spatial = DartboardSpatialAttention(d_model, heads, dropout)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, values: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
        batch, steps, nodes, width = values.shape
        temporal = self.temporal(
            self.temporal_norm(values).permute(0, 2, 1, 3).reshape(batch * nodes, steps, width)
        ).reshape(batch, nodes, steps, width).permute(0, 2, 1, 3)
        values = values + temporal
        spatial = self.spatial(
            self.spatial_norm(values).reshape(batch * steps, nodes, width), projection
        ).reshape(batch, steps, nodes, width)
        values = values + spatial
        return values + self.ff(self.ff_norm(values))


class Model(nn.Module):
    """Nationwide air-quality Transformer with top-down stochastic latents."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        dartboard_mx: np.ndarray | torch.Tensor | None = None,
        cov_dim: int = 2,
        d_model: int = 32,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len, enc_in, cov_dim, d_model, nhead, num_encoder_layers) <= 0:
            raise ValueError("AirFormer dimensions must be positive")
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.cov_dim = cov_dim
        if dartboard_mx is None:
            raise ValueError(
                "AirFormer needs a station graph: a dartboard projection [enc_in, regions, enc_in] or the "
                "dataset adjacency adj_mx [enc_in, enc_in]; use a graph dataset that ships adj_mx.npy"
            )
        projection = torch.as_tensor(dartboard_mx, dtype=torch.float32)
        if projection.ndim == 2:
            projection = projection.unsqueeze(1)
        if projection.ndim != 3 or projection.shape[0] != enc_in or projection.shape[2] != enc_in:
            raise ValueError("dartboard_mx must have shape (enc_in, regions, enc_in)")
        projection = projection / projection.sum(-1, keepdim=True).clamp_min(1e-6)
        self.register_buffer("dartboard_projection", projection)

        self.input_projection = nn.Sequential(
            nn.Linear(1 + cov_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.blocks = nn.ModuleList(
            AirFormerBlock(d_model, nhead, min(seq_len, 2 ** (index + 1)), dropout)
            for index in range(num_encoder_layers)
        )
        self.latent_mean = nn.ModuleList(
            nn.Linear(d_model * 2, d_model) for _ in range(num_encoder_layers)
        )
        self.latent_scale = nn.ModuleList(
            nn.Linear(d_model * 2, d_model) for _ in range(num_encoder_layers)
        )
        self.top_prior = nn.Parameter(torch.zeros(1, 1, 1, d_model))
        self.temporal_head = nn.Linear(seq_len, pred_len)
        self.output_head = nn.Linear(d_model * 2, 1)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_dec, x_mark_dec
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError("AirFormer expects (batch, configured seq_len, enc_in)")
        if x_mark_enc is None:
            x_mark_enc = x_enc.new_zeros((x_enc.shape[0], self.seq_len, 6))
        features = to_spatiotemporal(x_enc, x_mark_enc)
        if features.shape[-1] != 1 + self.cov_dim:
            raise ValueError(f"AirFormer expects exactly {self.cov_dim} covariates")
        state = self.input_projection(features)
        levels = []
        for block in self.blocks:
            state = block(state, self.dartboard_projection)
            levels.append(state)

        latent = self.top_prior.expand_as(levels[-1])
        for index in reversed(range(len(levels))):
            context = torch.cat([levels[index], latent], dim=-1)
            mean = self.latent_mean[index](context)
            scale = F.softplus(self.latent_scale[index](context)) + 1e-4
            latent = mean + torch.randn_like(scale) * scale if self.training else mean
        joint = torch.cat([levels[-1], latent], dim=-1)
        per_step = self.output_head(joint).squeeze(-1).transpose(1, 2)
        return self.temporal_head(per_step).transpose(1, 2)
