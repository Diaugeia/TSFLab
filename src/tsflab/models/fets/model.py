"""Independent FeTS implementation from the AAAI 2026 method equations.

Fourier-Poly importance scoring, mask-controlled local aggregation, and the
dual-scale feed-forward network are implemented from the paper. The binary
mask uses a straight-through sigmoid only for optimization; its forward values
are exactly the thresholded mask in equation (8).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.revin import RevIN


class FourierPolyMask(nn.Module):
    """Equations (2)--(8): hybrid basis scoring and mean-threshold activation."""

    def __init__(self, d_model: int, fourier_order: int, polynomial_order: int) -> None:
        super().__init__()
        self.register_buffer("cos_order", torch.arange(fourier_order + 1).float())
        self.register_buffer("sin_order", torch.arange(1, fourier_order + 1).float())
        self.register_buffer("poly_order", torch.arange(polynomial_order + 1).float())
        self.cos_coeff = nn.Parameter(torch.empty(d_model, d_model, fourier_order + 1))
        self.sin_coeff = nn.Parameter(torch.empty(d_model, d_model, fourier_order))
        self.poly_coeff = nn.Parameter(torch.empty(d_model, d_model, polynomial_order + 1))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.score = nn.Linear(d_model, d_model)
        for parameter in (self.cos_coeff, self.sin_coeff, self.poly_coeff):
            nn.init.normal_(parameter, std=0.02)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos_basis = torch.cos(x.unsqueeze(-1) * self.cos_order * math.pi)
        sin_basis = torch.sin(x.unsqueeze(-1) * self.sin_order * math.pi)
        poly_basis = x.unsqueeze(-1).pow(self.poly_order)
        representation = (
            torch.einsum("rio,ido->rd", cos_basis, self.cos_coeff)
            + torch.einsum("rio,ido->rd", sin_basis, self.sin_coeff)
            + torch.einsum("rio,ido->rd", poly_basis, self.poly_coeff)
            + self.bias
        )
        scores = self.score(representation)
        threshold = scores.mean(dim=-1, keepdim=True)
        hard = (scores >= threshold).to(scores.dtype)
        soft = torch.sigmoid(scores - threshold)
        mask = hard + (soft - soft.detach())
        return mask, scores


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 32,
        patch_len: int = 16,
        stride: int = 8,
        fourier_order: int = 2,
        polynomial_order: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.0,
        use_revin: bool = True,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len, enc_in, d_model, patch_len, stride, kernel_size) < 1:
            raise ValueError("all dimensions must be positive")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.patch_len = min(patch_len, seq_len)
        self.stride = min(stride, self.patch_len)
        # PatchTST end padding as in the official code: replicate the last step by
        # ``stride`` when patches overlap; for non-overlapping patches pad only the
        # remainder, so the most recent steps are always inside the last patch.
        if self.patch_len != self.stride:
            self.pad_len = self.stride
        else:
            self.pad_len = (-(seq_len - self.patch_len)) % self.stride
        self.padding = nn.ReplicationPad1d((0, self.pad_len)) if self.pad_len else nn.Identity()
        self.num_patches = 1 + (seq_len + self.pad_len - self.patch_len) // self.stride
        self.revin = RevIN(enc_in, enabled=use_revin)
        self.patch_projection = nn.Linear(self.patch_len, d_model)
        self.importance = FourierPolyMask(d_model, fourier_order, polynomial_order)
        self.mask_kernel = nn.Parameter(torch.ones(kernel_size) / kernel_size)
        self.kernel_size = kernel_size
        self.norm = nn.LayerNorm(d_model)
        self.local = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.fusion = nn.Conv1d(2 * d_model, d_model, kernel_size=1)
        self.output = nn.Sequential(nn.GELU(), nn.Dropout(dropout))
        self.projection = nn.Linear(self.num_patches * d_model, pred_len)

    def adaptive_features(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = tokens.reshape(-1, tokens.shape[-1])
        mask, scores = self.importance(flat)
        radius_left = self.kernel_size // 2
        radius_right = self.kernel_size - 1 - radius_left
        values = F.pad(flat, (radius_left, radius_right)).unfold(-1, self.kernel_size, 1)
        masks = F.pad(mask, (radius_left, radius_right)).unfold(-1, self.kernel_size, 1)
        filtered = (values * masks * self.mask_kernel).sum(dim=-1)
        return filtered.reshape_as(tokens), scores.reshape_as(tokens)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"expected [B,{self.seq_len},{self.enc_in}], got {tuple(x_enc.shape)}")
        normalized = self.revin(x_enc, "norm")
        patches = self.padding(normalized.transpose(1, 2)).unfold(-1, self.patch_len, self.stride)
        tokens = self.patch_projection(patches)
        adaptive, _ = self.adaptive_features(tokens)
        hidden = self.norm(tokens + adaptive).reshape(-1, self.num_patches, tokens.shape[-1]).transpose(1, 2)
        local = F.gelu(self.local(hidden))
        global_context = hidden.mean(dim=-1, keepdim=True).expand_as(hidden)
        fused = self.output(self.fusion(torch.cat((local, global_context), dim=1)))
        forecast = self.projection(fused.flatten(1)).reshape(x_enc.shape[0], self.enc_in, self.pred_len)
        return self.revin(forecast.transpose(1, 2), "denorm")
