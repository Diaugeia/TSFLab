"""Independent Pathformer implementation from the ICLR 2024 paper.

It retains multiple patch resolutions, local/global dual attention, and an
input-dependent resolution router. No reference source code is reused.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from tsflab.models._components.revin import RevIN

# PyTorch's fused attention kernels reject more than 65535 sequences per call when
# dropout is active; the local attention sees batch x channels x patches sequences
# (220,672 on traffic at batch 16), so it runs in chunks along that dimension.
_ATTENTION_CHUNK = 32768


class DualScaleAttention(nn.Module):
    """Local sample attention followed by global patch attention."""
    def __init__(self, patch_size, width, heads, feedforward, dropout):
        super().__init__()
        self.patch_size = patch_size
        self.sample_embedding = nn.Linear(1, width)
        self.local_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.global_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.feedforward = nn.Sequential(nn.Linear(width, feedforward), nn.GELU(), nn.Dropout(dropout), nn.Linear(feedforward, width))
        self.local_norm = nn.LayerNorm(width)
        self.global_norm = nn.LayerNorm(width)

    def forward(self, values):
        batch, channels, length = values.shape
        patches = math.ceil(length / self.patch_size)
        padded = F.pad(values, (0, patches * self.patch_size - length))
        samples = padded.reshape(batch * channels * patches, self.patch_size, 1)
        local = self.sample_embedding(samples)
        # Independent sequences, no masks: chunking gives the same result.
        attended = torch.cat([self.local_attention(chunk, chunk, chunk, need_weights=False)[0]
                              for chunk in local.split(_ATTENTION_CHUNK)], dim=0)
        local = self.local_norm(local + attended).mean(1)
        tokens = local.reshape(batch * channels, patches, -1)
        context, _ = self.global_attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.global_norm(tokens + context + self.feedforward(tokens))
        expanded = tokens.unsqueeze(2).expand(-1, -1, self.patch_size, -1)
        return expanded.reshape(batch, channels, patches * self.patch_size, -1)[:, :, :length]


class AdaptivePathway(nn.Module):
    """Route every sample/variable between temporal-resolution experts."""
    def __init__(self, patch_sizes, width, heads, feedforward, dropout, top_k):
        super().__init__()
        self.patch_sizes = tuple(patch_sizes)
        self.top_k = min(top_k, len(patch_sizes))
        self.experts = nn.ModuleList(DualScaleAttention(s, width, heads, feedforward, dropout) for s in patch_sizes)
        self.router = nn.Sequential(nn.Linear(4, width), nn.GELU(), nn.Linear(width, len(patch_sizes)))
        self.output = nn.Linear(width, 1)
        self.last_route = None
        self.last_topk = None

    @staticmethod
    def routing_features(x):
        delta = x[..., 1:] - x[..., :-1] if x.shape[-1] > 1 else torch.zeros_like(x)
        spectrum = torch.fft.rfft(x, dim=-1).abs()
        concentration = spectrum.amax(-1) / spectrum.sum(-1).clamp_min(1e-6)
        return torch.stack((x.mean(-1), x.std(-1, unbiased=False), delta.abs().mean(-1), concentration), -1)

    def forward(self, x):
        route = self.router(self.routing_features(x)).softmax(-1)
        self.last_route = route
        self.last_topk = route.topk(self.top_k, -1).indices
        # The forecast-only API has no auxiliary balance-loss channel. Dense
        # differentiable routing keeps every expert trainable; top-k paths are
        # still exposed for inspection.
        experts = torch.stack([self.output(expert(x)).squeeze(-1) for expert in self.experts], -1)
        return (experts * route.unsqueeze(-2)).sum(-1)


class Model(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, features="M", layer_nums=2,
                 k=2, num_experts=4, patch_size_list=None, d_model=16,
                 d_ff=64, residual_connection=1, revin=True, n_heads=4,
                 dropout=0.1):
        super().__init__()
        if min(seq_len, pred_len, enc_in, layer_nums, num_experts) < 1:
            raise ValueError("sequence, horizon, variables, layers, and experts must be positive")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        sizes = patch_size_list or [16, 12, 8, 6] * layer_nums
        if len(sizes) != layer_nums * num_experts or any(s < 1 for s in sizes):
            raise ValueError("patch_size_list must contain layer_nums * num_experts positive values")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.residual_connection = bool(residual_connection)
        self.revin = RevIN(enc_in, enabled=revin)
        self.layers = nn.ModuleList(
            AdaptivePathway(sizes[i*num_experts:(i+1)*num_experts], d_model, n_heads, d_ff, dropout, k)
            for i in range(layer_nums)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(seq_len) for _ in range(layer_nums))
        self.forecast = nn.Linear(seq_len, pred_len)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        if x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"expected (*,{self.seq_len},{self.enc_in})")
        values = self.revin(x_enc, "norm").transpose(1, 2)
        for pathway, norm in zip(self.layers, self.norms):
            update = pathway(values)
            values = norm(values + update) if self.residual_connection else norm(update)
        return self.revin(self.forecast(values).transpose(1, 2), "denorm")
