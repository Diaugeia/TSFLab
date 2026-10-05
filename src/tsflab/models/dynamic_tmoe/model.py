"""Clean-room fixed-capacity realization of Dynamic TMoE.

The model retains patching, an RBF-MMD drift signal, recurrent memory routing,
an anomaly-state repository, heterogeneous experts, concentrated top-k routing,
and cyclic channel-relation refinement. Dynamic module creation/pruning remains
a training-orchestrator concern and is intentionally not performed in forward.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.marks import elapsed_minutes
from tsflab.models._components.revin import RevIN
from tsflab.models._components.topk_expert_router import topk_dense_mix


class _IdentityExpert(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class _TrendExpert(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = F.avg_pool1d(
            x.permute(0, 2, 3, 1).reshape(-1, x.shape[-1], x.shape[1]),
            kernel_size=3,
            stride=1,
            padding=1,
        ).reshape(x.shape[0], x.shape[2], x.shape[-1], x.shape[1]).permute(0, 3, 1, 2)
        return self.mlp(trend)


class _SeasonalityExpert(nn.Module):
    def __init__(self, num_patches: int, d_model: int) -> None:
        super().__init__()
        self.spectral_gate = nn.Parameter(torch.ones(num_patches // 2 + 1, d_model))
        self.projection = nn.Linear(2 * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(x, dim=1)
        filtered = torch.fft.irfft(
            spectrum * self.spectral_gate[None, :, None], n=x.shape[1], dim=1
        )
        return self.projection(torch.cat((filtered.sin(), filtered.cos()), dim=-1))


class _FluctuationExpert(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.value = nn.Conv1d(d_model, d_model, kernel_size=3)
        self.gate = nn.Conv1d(d_model, d_model, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, patches, channels, hidden = x.shape
        sequence = x.permute(0, 2, 3, 1).reshape(batch * channels, hidden, patches)
        padded = F.pad(sequence, (2, 0))
        output = self.value(padded) * self.gate(padded).sigmoid()
        return output.reshape(batch, channels, hidden, patches).permute(0, 3, 1, 2)


class _DriftExpert(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        patch_len: int = 16,
        stride: int = 8,
        top_k: int = 3,
        memory_slots: int = 4,
        relation_period: int = 24,
        routing_floor: float = 1e-4,
        use_revin: bool = True,
    ) -> None:
        super().__init__()
        if min(
            seq_len,
            pred_len,
            enc_in,
            d_model,
            patch_len,
            stride,
            top_k,
            memory_slots,
            relation_period,
        ) < 1:
            raise ValueError("DynamicTMoE dimensions must be positive")
        if top_k > 5:
            raise ValueError("top_k cannot exceed the five local experts")
        if not 0 <= routing_floor < 1:
            raise ValueError("routing_floor must lie in [0, 1)")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.d_model = d_model
        self.patch_len = min(patch_len, seq_len)
        self.stride = min(stride, self.patch_len)
        self.num_patches = 1 + math.ceil(max(seq_len - self.patch_len, 0) / self.stride)
        self.top_k = top_k
        self.routing_floor = float(routing_floor)
        self.use_revin = use_revin

        self.revin = RevIN(enc_in, affine=True, subtract_last=False)
        self.patch_embedding = nn.Linear(self.patch_len, d_model)
        self.router = nn.GRU(d_model, d_model, batch_first=True)
        self.anomaly_repository = nn.Parameter(torch.empty(memory_slots, d_model))
        self.memory_gate = nn.Linear(2 * d_model, 1)
        self.routing_head = nn.Linear(d_model, 5)
        self.drift_bias = nn.Parameter(torch.tensor(1.0))
        self.drift_threshold = nn.Parameter(torch.tensor(0.1))
        self.experts = nn.ModuleList(
            (
                _IdentityExpert(d_model),
                _TrendExpert(d_model),
                _SeasonalityExpert(self.num_patches, d_model),
                _FluctuationExpert(d_model),
                _DriftExpert(d_model),
            )
        )
        self.cycle_relation = nn.Parameter(
            torch.empty(relation_period, enc_in, enc_in)
        )
        self.relation_residual = nn.Sequential(
            nn.Linear(1, d_model), nn.Tanh(), nn.Linear(d_model, 1)
        )
        self.head = nn.Linear(self.num_patches * d_model, pred_len)
        nn.init.normal_(self.anomaly_repository, std=0.02)
        nn.init.normal_(self.cycle_relation, std=0.02)

    def _validate(self, x: torch.Tensor) -> None:
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(
                f"expected [batch, {self.seq_len}, {self.enc_in}], got {tuple(x.shape)}"
            )

    def _patches(self, x: torch.Tensor) -> torch.Tensor:
        history = x.transpose(1, 2)
        covered = (self.num_patches - 1) * self.stride + self.patch_len
        if covered > self.seq_len:
            history = F.pad(history, (covered - self.seq_len, 0), mode="replicate")
        patches = history.unfold(-1, self.patch_len, self.stride)
        return self.patch_embedding(patches).permute(0, 2, 1, 3)

    @staticmethod
    def rbf_mmd(reference: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        """Paper equation (1), with the median-distance bandwidth heuristic."""
        if reference.ndim != 3 or current.ndim != 3 or reference.shape[0] != current.shape[0]:
            raise ValueError("MMD windows must be [batch, samples, features]")
        joined = torch.cat((reference, current), dim=1)
        distances = torch.cdist(joined, joined).square()
        bandwidth = distances.flatten(1).median(dim=1).values.clamp_min(1e-6)

        def kernel(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            squared = torch.cdist(left, right).square()
            return torch.exp(-squared / (2 * bandwidth[:, None, None]))

        return (
            kernel(reference, reference).mean(dim=(1, 2))
            - 2 * kernel(reference, current).mean(dim=(1, 2))
            + kernel(current, current).mean(dim=(1, 2))
        ).clamp_min(0)

    @staticmethod
    def adaptive_threshold(history: torch.Tensor, sensitivity: float = 1.0) -> torch.Tensor:
        """Paper equation (2): epsilon = mean(H) + lambda * std(H)."""
        return history.mean(-1) + sensitivity * history.std(-1, unbiased=False)

    def routing_weights(
        self, patch_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = patch_tokens.mean(2)
        hidden, _ = self.router(pooled)
        normalized_hidden = F.normalize(hidden, dim=-1)
        normalized_memory = F.normalize(self.anomaly_repository, dim=-1)
        memory_weights = (normalized_hidden @ normalized_memory.t()).softmax(-1)
        reference = memory_weights @ self.anomaly_repository
        alpha = self.memory_gate(torch.cat((hidden, reference), dim=-1)).sigmoid()
        routed_state = alpha * hidden + (1 - alpha) * reference

        split = max(1, self.num_patches // 2)
        reference_window = pooled[:, :split]
        current_window = pooled[:, split:] if split < self.num_patches else pooled[:, -1:]
        drift = self.rbf_mmd(reference_window, current_window)
        logits = self.routing_head(routed_state)
        drift_activation = (drift - F.softplus(self.drift_threshold)).sigmoid()
        logits[..., -1] = logits[..., -1] + self.drift_bias * drift_activation[:, None]
        soft = logits.softmax(-1)
        # A disclosed gradient floor keeps every fixed expert trainable; the
        # dynamic paper implementation instead creates/prunes experts outside
        # inference and can use exact sparse dispatch.
        weights = topk_dense_mix(soft, self.top_k, self.routing_floor)
        return weights, drift

    def window_phase(self, x_mark_enc: torch.Tensor | None, batch: int,
                     device: torch.device) -> torch.Tensor:
        """Cycle index ``t`` of Eq. (10) for each sample: the phase of the first input step.

        The phase is the absolute step count of the first raw mark since
        1970-01-01 00:00 modulo ``relation_period``; the step length is the
        minute gap between the first two marks (at least one minute). For
        hourly data and ``relation_period = 24`` this is the hour of day.
        Without marks every sample uses phase 0, as the reference does.
        """
        period = self.cycle_relation.shape[0]
        if x_mark_enc is None:
            return torch.zeros(batch, dtype=torch.long, device=device)
        if (x_mark_enc.ndim != 3 or x_mark_enc.shape[0] != batch or x_mark_enc.shape[1] < 2
                or x_mark_enc.shape[-1] != 6):
            raise ValueError("x_mark_enc must be raw marks [batch, seq_len >= 2, 6]")
        minutes = elapsed_minutes(x_mark_enc[:, :2])
        step = (minutes[:, 1] - minutes[:, 0]).clamp_min(1)
        return torch.div(minutes[:, 0], step, rounding_mode="floor").remainder(period).to(device)

    def channel_relation(self, x: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        centered = x - x.mean(1, keepdim=True)
        covariance = torch.einsum("blc,bld->bcd", centered, centered)
        scale = centered.square().sum(1).sqrt().clamp_min(1e-6)
        current = covariance / (scale[:, :, None] * scale[:, None, :])
        prototype = self.cycle_relation[phase]
        residual = self.relation_residual((current - prototype).unsqueeze(-1)).squeeze(-1)
        return (prototype + residual).softmax(-1)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        self._validate(x_enc)
        normalized = self.revin(x_enc, "norm") if self.use_revin else x_enc
        patch_tokens = self._patches(normalized)
        weights, drift = self.routing_weights(patch_tokens)
        expert_outputs = torch.stack(
            [expert(patch_tokens) for expert in self.experts], dim=-2
        )
        mixed = (expert_outputs * weights[:, :, None, :, None]).sum(-2)
        relation = self.channel_relation(
            normalized, self.window_phase(x_mark_enc, x_enc.shape[0], x_enc.device)
        )
        mixed = mixed + torch.einsum("bck,bnkd->bncd", relation, mixed)
        forecast = self.head(mixed.permute(0, 2, 1, 3).flatten(-2)).transpose(1, 2)
        self.last_mmd = drift.detach()
        return self.revin(forecast, "denorm") if self.use_revin else forecast
