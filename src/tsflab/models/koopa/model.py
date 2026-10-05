"""Independent Koopa implementation from the NeurIPS 2023 method description.

The design follows Fourier separation, learned measurement functions, and
linear Koopman evolution inside a residual stack. No reference source is
reproduced here.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


class FourierDynamicsSplit(nn.Module):
    """Separate dominant spectral modes from remaining local variation."""
    def __init__(self, alpha: float = 0.2) -> None:
        super().__init__()
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spectrum = torch.fft.rfft(values, dim=1)
        energy = spectrum.abs().mean(dim=(0, 2))
        count = max(1, math.ceil(energy.numel() * self.alpha))
        keep = torch.zeros_like(energy, dtype=torch.bool)
        keep[energy.topk(count).indices] = True
        invariant = torch.fft.irfft(spectrum * keep[None, :, None], n=values.size(1), dim=1)
        return values - invariant, invariant


class MeasurementFunction(nn.Module):
    """Nonlinear observable map and decoder around Koopman evolution."""
    def __init__(self, channels: int, latent_dim: int, hidden_dim: int, hidden_layers: int) -> None:
        super().__init__()
        encoder: list[nn.Module] = [nn.Linear(channels, hidden_dim), nn.Tanh()]
        for _ in range(max(1, hidden_layers) - 1):
            encoder.extend((nn.Linear(hidden_dim, hidden_dim), nn.Tanh()))
        encoder.append(nn.Linear(hidden_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, channels))

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return self.encoder(values)

    def decode(self, states: torch.Tensor) -> torch.Tensor:
        return self.decoder(states)


def estimate_operator(states: torch.Tensor) -> torch.Tensor:
    """Estimate a batched DMD operator from consecutive states by least squares.

    As in the official ``KPLayer``, the operator is the least-squares solution
    of ``left @ K = right``; fewer snapshots than latent dimensions give an
    underdetermined system, which ``lstsq`` resolves without inverting a
    singular Gram matrix. A non-finite solution is replaced by the identity for
    the whole batch, as upstream does.
    """
    left, right = states[:, :-1], states[:, 1:]
    operator = torch.linalg.lstsq(left, right).solution
    if not torch.isfinite(operator).all():
        eye = torch.eye(operator.size(-1), device=states.device, dtype=states.dtype)
        operator = eye.expand_as(operator).clone()
    return operator


class LocalKoopmanPredictor(nn.Module):
    """Context-aware operator estimated from the most recent local segment."""
    def __init__(self, channels: int, latent_dim: int, hidden_dim: int, hidden_layers: int, seg_len: int) -> None:
        super().__init__()
        self.seg_len = seg_len
        self.measure = MeasurementFunction(channels, latent_dim, hidden_dim, hidden_layers)

    def forward(self, values: torch.Tensor, horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
        states = self.measure.encode(values)
        operator = estimate_operator(states[:, -max(2, min(self.seg_len, states.size(1))):])
        reconstructed = self.measure.decode(torch.cat((states[:, :1], states[:, :-1] @ operator), dim=1))
        current, future = states[:, -1:], []
        for _ in range(horizon):
            current = current @ operator
            future.append(current)
        return reconstructed, self.measure.decode(torch.cat(future, dim=1))


class GlobalKoopmanPredictor(nn.Module):
    """Shared long-range transition for time-invariant dynamics."""
    def __init__(self, channels: int, latent_dim: int, hidden_dim: int, hidden_layers: int) -> None:
        super().__init__()
        self.measure = MeasurementFunction(channels, latent_dim, hidden_dim, hidden_layers)
        self.transition = nn.Parameter(torch.eye(latent_dim) + 0.01 * torch.randn(latent_dim, latent_dim))

    def forward(self, values: torch.Tensor, horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
        states = self.measure.encode(values)
        reconstructed = self.measure.decode(torch.cat((states[:, :1], states[:, :-1] @ self.transition), dim=1))
        current, future = states[:, -1:], []
        for _ in range(horizon):
            current = current @ self.transition
            future.append(current)
        return reconstructed, self.measure.decode(torch.cat(future, dim=1))


class KoopmanBlock(nn.Module):
    """One residual hierarchy level over variant and invariant dynamics."""
    def __init__(self, channels: int, latent_dim: int, hidden_dim: int, hidden_layers: int, seg_len: int, alpha: float) -> None:
        super().__init__()
        self.split = FourierDynamicsSplit(alpha)
        self.local = LocalKoopmanPredictor(channels, latent_dim, hidden_dim, hidden_layers, seg_len)
        self.global_dynamics = GlobalKoopmanPredictor(channels, latent_dim, hidden_dim, hidden_layers)

    def forward(self, residual: torch.Tensor, horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
        variant, invariant = self.split(residual)
        variant_rec, variant_pred = self.local(variant, horizon)
        invariant_rec, invariant_pred = self.global_dynamics(invariant, horizon)
        return residual - variant_rec - invariant_rec, variant_pred + invariant_pred


class Model(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, enc_in: int, label_len: int = 0,
                 features: str = "M", seg_len: int | None = None, dynamic_dim: int = 128,
                 hidden_dim: int = 64, hidden_layers: int = 2, num_blocks: int = 3,
                 multistep: bool = False, alpha: float = 0.2) -> None:
        super().__init__()
        if seq_len < 2 or pred_len < 1 or dynamic_dim < 1 or num_blocks < 1:
            raise ValueError("Koopa requires seq_len >= 2 and positive dimensions")
        segment = pred_len if seg_len is None else seg_len
        if segment < 2:
            raise ValueError("seg_len must be at least 2")
        self.seq_len, self.pred_len = seq_len, pred_len
        self.blocks = nn.ModuleList(KoopmanBlock(enc_in, dynamic_dim, hidden_dim, hidden_layers, segment, alpha) for _ in range(num_blocks))

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        if x_enc.ndim != 3 or x_enc.size(1) != self.seq_len:
            raise ValueError(f"Koopa expects [B, {self.seq_len}, C]")
        mean = x_enc.mean(1, keepdim=True).detach()
        scale = x_enc.var(1, keepdim=True, unbiased=False).add(1e-5).sqrt().detach()
        residual = (x_enc - mean) / scale
        forecast = x_enc.new_zeros(x_enc.size(0), self.pred_len, x_enc.size(2))
        for block in self.blocks:
            residual, contribution = block(residual, self.pred_len)
            forecast = forecast + contribution
        return forecast * scale + mean
