"""CoST: contrastive seasonal-trend representations plus a ridge forecaster (Woo et al., ICLR 2022).

Independent implementation from the paper (Sec. 3, Sec. 4.1, Apps. A and C), with
omissions resolved by reading salesforce/CoST at the revision recorded in the card.
Stage I (``pretrain``): a dilated-conv backbone, a trend disentangler (mean of causal
convolutions with kernels 2^i) and a seasonal disentangler (per-frequency complex
linear layer) are trained with a MoCo time-domain loss on trend features and
amplitude/phase instance-contrastive losses on seasonal features. Stage II: a ridge
regression from the last step's representation to the whole horizon, closed form,
with the penalty chosen on held-out training windows. ``forward`` is the frozen
encoder followed by the ridge map.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from tsflab.models._components.dilated_conv_encoder import DilatedConvEncoder
from tsflab.models._components.marks import days_from_civil

RIDGE_ALPHAS = (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)
_CALENDAR = 7  # minute, hour, dayofweek, day, dayofyear, month, weekofyear


def _iso_weeks_in_year(year: torch.Tensor) -> torch.Tensor:
    def p(y):
        return (y + torch.div(y, 4, rounding_mode="floor") - torch.div(y, 100, rounding_mode="floor")
                + torch.div(y, 400, rounding_mode="floor")) % 7
    return 52 + ((p(year) == 4) | (p(year - 1) == 3)).long()


def calendar_features(marks: torch.Tensor) -> torch.Tensor:
    """Raw ``[..., 6]`` marks -> ``[..., 7]`` minute, hour, dayofweek, day, dayofyear, month, ISO week."""
    raw = marks.round().long()
    year, month, day, weekday, hour, minute = raw.unbind(-1)
    day_of_year = days_from_civil(year, month, day) - days_from_civil(year, torch.ones_like(month), torch.ones_like(day)) + 1
    week = torch.div(day_of_year - (weekday + 1) + 10, 7, rounding_mode="floor")
    week = torch.where(week < 1, _iso_weeks_in_year(year - 1), week)
    week = torch.where(week > _iso_weeks_in_year(year), torch.ones_like(week), week)
    return torch.stack((minute, hour, weekday, day, day_of_year, month, week), dim=-1).to(marks.dtype)


class _BandedFourierLayer(nn.Module):
    """Seasonal disentangler: ``V_S = iFFT(A_i F(V)_i + B_i)`` with one complex affine map per frequency."""

    def __init__(self, in_dims: int, out_dims: int, length: int) -> None:
        super().__init__()
        self.length = length
        frequencies = length // 2 + 1
        self.weight = nn.Parameter(torch.empty(frequencies, in_dims, out_dims, dtype=torch.cfloat))
        self.bias = nn.Parameter(torch.empty(frequencies, out_dims, dtype=torch.cfloat))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(out_dims * in_dims) if in_dims else 0.0  # fan_in of a [F, in, out] tensor
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, d] -> [B, T, d_S]
        spectrum = torch.fft.rfft(x, dim=1)
        mixed = torch.einsum("bfi,fio->bfo", spectrum, self.weight) + self.bias
        return torch.fft.irfft(mixed, n=x.shape[1], dim=1)


class _Encoder(nn.Module):
    """``[B, T, m] -> (trend [B, T, d/2], season [B, T, d/2])``."""

    def __init__(self, input_dims, repr_dims, hidden_dims, depth, kernels, length) -> None:
        super().__init__()
        component = repr_dims // 2
        self.kernels = list(kernels)
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.backbone = DilatedConvEncoder(hidden_dims, [hidden_dims] * depth + [repr_dims], kernel_size=3)
        self.trend = nn.ModuleList(nn.Conv1d(repr_dims, component, k, padding=k - 1) for k in self.kernels)
        self.season = _BandedFourierLayer(repr_dims, component, length)
        self.season_dropout = nn.Dropout(0.1)

    def forward(self, x):
        hidden = self.backbone(self.input_fc(x).transpose(1, 2))  # [B, d, T]
        steps = hidden.shape[-1]
        # Mixture of auto-regressive experts: causal conv of kernel k, mean over experts.
        trend = torch.stack([conv(hidden)[..., :steps] for conv in self.trend]).mean(0).transpose(1, 2)
        season = self.season_dropout(self.season(hidden.transpose(1, 2)))
        return trend, season


def _instance_contrastive(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Per-step two-view InfoNCE over the batch (both directions, all 2B - 1 others as negatives)."""
    batch = z1.shape[0]
    z = torch.cat((z1, z2), dim=0).transpose(0, 1)  # [T, 2B, C]
    similarity = z @ z.transpose(1, 2)  # [T, 2B, 2B]
    eye = torch.eye(2 * batch, dtype=torch.bool, device=z.device)
    logits = similarity.masked_fill(eye, float("-inf"))
    log_prob = F.log_softmax(logits, dim=-1)
    index = torch.arange(batch, device=z.device)
    return -(log_prob[:, index, batch + index].mean() + log_prob[:, batch + index, index].mean()) / 2


def _amplitude_phase(z: torch.Tensor, eps: float = 1e-6):
    amplitude = torch.sqrt((z.real + eps) ** 2 + (z.imag + eps) ** 2)
    phase = torch.atan2(z.imag, z.real + eps)
    return amplitude, phase


class Model(nn.Module):
    """Frozen CoST encoder followed by a closed-form multi-horizon ridge regression."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        features: str = "M",
        repr_dims: int = 320,
        hidden_dims: int = 64,
        depth: int = 10,
        kernels: list[int] = (1, 2, 4, 8, 16, 32, 64, 128),
        alpha: float = 5e-4,
        queue_size: int = 256,
        momentum: float = 0.999,
        temperature: float = 0.07,
        sigma: float = 0.5,
        aug_prob: float = 0.5,
        pretrain_iters: int = 0,
        pretrain_batch_size: int = 128,
        pretrain_lr: float = 1e-3,
        use_marks: bool = True,
        channel_independent: bool = False,
        ridge_alphas: list[float] = RIDGE_ALPHAS,
        ridge_valid_fraction: float = 0.25,
        ridge_max_samples: int = 100000,
    ) -> None:
        super().__init__()
        if repr_dims % 2:
            raise ValueError("repr_dims must be even (trend and season halves)")
        if features not in {"M", "S", "MS"}:
            raise ValueError("features must be M, S, or MS")
        if not 0.0 < ridge_valid_fraction < 1.0:
            raise ValueError("ridge_valid_fraction must be in (0, 1)")
        self.seq_len, self.pred_len, self.enc_in, self.features = seq_len, pred_len, enc_in, features
        self.channel_independent, self.use_marks = channel_independent, use_marks
        self.alpha, self.momentum, self.temperature = alpha, momentum, temperature
        self.sigma, self.aug_prob = sigma, aug_prob
        self.pretrain_iters, self.pretrain_batch_size, self.pretrain_lr = pretrain_iters, pretrain_batch_size, pretrain_lr
        self.ridge_alphas = [float(a) for a in ridge_alphas]
        self.ridge_valid_fraction, self.ridge_max_samples = ridge_valid_fraction, ridge_max_samples
        self.queue_size = queue_size

        values = 1 if channel_independent else enc_in
        input_dims = values + (_CALENDAR if use_marks else 0)
        self.encoder = _Encoder(input_dims, repr_dims, hidden_dims, depth, kernels, seq_len)
        component = repr_dims // 2
        self.head = nn.Sequential(nn.Linear(component, component), nn.ReLU(), nn.Linear(component, component))
        # Momentum (key) encoder and head: updated by EMA only (MoCo).
        self.key_encoder = copy.deepcopy(self.encoder)
        self.key_head = copy.deepcopy(self.head)
        for parameter in [*self.key_encoder.parameters(), *self.key_head.parameters()]:
            parameter.requires_grad_(False)
        self.register_buffer("queue", F.normalize(torch.randn(component, queue_size), dim=0))
        self.register_buffer("queue_ptr", torch.zeros((), dtype=torch.long))
        self.register_buffer("mark_mean", torch.zeros(_CALENDAR))
        self.register_buffer("mark_std", torch.ones(_CALENDAR))
        targets = 1 if (channel_independent or features == "MS") else enc_in
        self.targets = targets
        self.register_buffer("ridge_weight", torch.zeros(repr_dims, pred_len * targets))
        self.register_buffer("ridge_bias", torch.zeros(pred_len * targets))
        self.register_buffer("ridge_alpha", torch.zeros(()))

    # ----------------------------------------------------------------- inputs
    def _inputs(self, x: torch.Tensor, marks: torch.Tensor | None) -> torch.Tensor:
        """``[B, T, D]`` values (+ standardized calendar) -> ``[B', T, m]`` encoder inputs."""
        batch, steps, channels = x.shape
        if self.channel_independent:
            x = x.transpose(1, 2).reshape(batch * channels, steps, 1)
        if not self.use_marks:
            return x
        if marks is None:
            calendar = x.new_zeros(batch, steps, _CALENDAR)
        else:
            calendar = (calendar_features(marks.to(x.dtype)) - self.mark_mean) / self.mark_std
        if self.channel_independent:
            calendar = calendar.repeat_interleave(channels, dim=0)
        # Official layout: covariate columns first, then the values.
        return torch.cat((calendar, x), dim=-1)

    def representation(self, x_enc, x_mark_enc=None) -> torch.Tensor:
        """``[B', repr_dims]``: trend and season features of the last lookback step."""
        trend, season = self.encoder(self._inputs(x_enc, x_mark_enc))
        return torch.cat((trend[:, -1], season[:, -1]), dim=-1)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_dec, x_mark_dec
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"x_enc must have shape [batch, {self.seq_len}, {self.enc_in}]")
        batch = x_enc.shape[0]
        out = self.representation(x_enc, x_mark_enc) @ self.ridge_weight + self.ridge_bias
        if self.channel_independent:
            out = out.view(batch, self.enc_in, self.pred_len).transpose(1, 2)
            return out[..., -1:] if self.features == "MS" else out
        return out.view(batch, self.pred_len, self.targets)

    # ------------------------------------------------------ stage I: CoST loss
    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Scale, shift, jitter (App. C), each applied per sample with probability ``aug_prob``."""
        batch, _, dims = x.shape

        def gate():
            return (torch.rand(batch, 1, 1, device=x.device) < self.aug_prob).to(x.dtype)

        x = x * (1 + gate() * torch.randn(batch, 1, dims, device=x.device) * self.sigma)
        x = x + gate() * torch.randn(batch, 1, dims, device=x.device) * self.sigma
        return x + gate() * torch.randn_like(x) * self.sigma

    @torch.no_grad()
    def _momentum_update(self) -> None:
        for source, target in ((self.encoder, self.key_encoder), (self.head, self.key_head)):
            for q, k in zip(source.parameters(), target.parameters()):
                k.mul_(self.momentum).add_(q.detach(), alpha=1 - self.momentum)

    @torch.no_grad()
    def _enqueue(self, keys: torch.Tensor) -> None:
        count = keys.shape[0]
        start = int(self.queue_ptr)
        index = (torch.arange(count, device=keys.device) + start) % self.queue_size
        self.queue[:, index] = keys.T
        self.queue_ptr.fill_((start + count) % self.queue_size)

    def contrastive_loss(self, x_q: torch.Tensor, x_k: torch.Tensor) -> torch.Tensor:
        """``L_time + alpha / 2 (L_amp + L_phase)`` on two augmented views ``[B, T, m]``."""
        step = int(torch.randint(0, x_q.shape[1], ()))
        trend_q, season_q = self.encoder(x_q)
        q = F.normalize(self.head(trend_q[:, step]), dim=-1)
        with torch.no_grad():
            self._momentum_update()
            trend_k, _ = self.key_encoder(x_k)
            k = F.normalize(self.key_head(trend_k[:, step]), dim=-1)
        positive = (q * k).sum(-1, keepdim=True)
        negative = q @ self.queue.clone().detach()
        logits = torch.cat((positive, negative), dim=1) / self.temperature
        loss = F.cross_entropy(logits, torch.zeros(q.shape[0], dtype=torch.long, device=q.device))
        self._enqueue(k)
        # Seasonal keys come from the query encoder (with gradient), as in the official code.
        _, season_k = self.encoder(x_k)
        amp_q, phase_q = _amplitude_phase(torch.fft.rfft(F.normalize(season_q, dim=-1), dim=1))
        amp_k, phase_k = _amplitude_phase(torch.fft.rfft(F.normalize(season_k, dim=-1), dim=1))
        seasonal = _instance_contrastive(amp_q, amp_k) + _instance_contrastive(phase_q, phase_k)
        return loss + self.alpha * seasonal / 2

    def _iterations(self, dataset) -> int:
        if self.pretrain_iters:
            return self.pretrain_iters
        dims = self.encoder.input_fc.in_features
        series = 1 if not self.channel_independent else self.enc_in
        elements = (len(dataset) + self.seq_len + self.pred_len - 1) * dims * series
        return 200 if elements <= 100000 else 600

    def _fit_calendar(self, dataset, device) -> None:
        if not self.use_marks:
            return
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        total = torch.zeros(_CALENDAR, dtype=torch.float64, device=device)
        square = torch.zeros_like(total)
        count = 0
        for batch in loader:
            if len(batch) < 3 or batch[2] is None:
                return
            features = calendar_features(batch[2].float().to(device)).double().reshape(-1, _CALENDAR)
            total += features.sum(0)
            square += (features**2).sum(0)
            count += features.shape[0]
        if count:
            mean = total / count
            std = (square / count - mean**2).clamp(min=0).sqrt()
            self.mark_mean.copy_(mean.float())
            self.mark_std.copy_(torch.where(std > 0, std, torch.ones_like(std)).float())

    # ------------------------------------------------------ stage II: ridge
    def _labels(self, batch, device) -> torch.Tensor:
        future = batch[1].float().to(device)[:, -self.pred_len:, : self.enc_in]
        if self.channel_independent:
            labels = future.transpose(1, 2).reshape(-1, self.pred_len)
        elif self.features == "MS":
            labels = future[..., -1:].reshape(future.shape[0], -1)
        else:
            labels = future.reshape(future.shape[0], -1)
        return labels.double()

    def _features(self, batch, device) -> torch.Tensor:
        x = batch[0].float().to(device)
        marks = batch[2].float().to(device) if len(batch) > 2 and self.use_marks else None
        return self.representation(x, marks).double()

    @torch.no_grad()
    def fit_ridge(self, dataset, device, batch_size: int = 256) -> None:
        """Closed-form ridge per penalty on the first windows, select on the last ``ridge_valid_fraction``.

        Score = RMSE + MAE on the held-out windows (official ``fit_ridge``); the chosen
        penalty's fit on the fitting windows is kept, as in the official protocol.
        """
        self.eval()
        windows = len(dataset)
        split = max(1, min(windows - 1, int(round(windows * (1 - self.ridge_valid_fraction)))))
        generator = torch.Generator().manual_seed(0)
        fit_index = torch.arange(split)
        valid_index = torch.arange(split, windows)
        if len(fit_index) > self.ridge_max_samples:
            fit_index = fit_index[torch.randperm(len(fit_index), generator=generator)[: self.ridge_max_samples]].sort().values
        if len(valid_index) > self.ridge_max_samples:
            valid_index = valid_index[torch.randperm(len(valid_index), generator=generator)[: self.ridge_max_samples]].sort().values
        dims = self.ridge_weight.shape[0]
        xtx = torch.zeros(dims, dims, dtype=torch.float64, device=device)
        xty = torch.zeros(dims, self.ridge_weight.shape[1], dtype=torch.float64, device=device)
        sum_x = torch.zeros(dims, dtype=torch.float64, device=device)
        sum_y = torch.zeros(self.ridge_weight.shape[1], dtype=torch.float64, device=device)
        rows = 0
        for batch in DataLoader(Subset(dataset, fit_index.tolist()), batch_size=batch_size, shuffle=False):
            features, labels = self._features(batch, device), self._labels(batch, device)
            xtx += features.T @ features
            xty += features.T @ labels
            sum_x += features.sum(0)
            sum_y += labels.sum(0)
            rows += features.shape[0]
        mean_x, mean_y = sum_x / rows, sum_y / rows
        # Centered normal equations (intercept unpenalized, as sklearn Ridge).
        xtx_c = xtx - rows * torch.outer(mean_x, mean_x)
        xty_c = xty - rows * torch.outer(mean_x, mean_y)
        valid_loader = DataLoader(Subset(dataset, valid_index.tolist()), batch_size=batch_size, shuffle=False)
        # Keep only held-out features; labels are re-read per penalty (they can be pred_len * D wide).
        valid_features = [self._features(batch, device) for batch in valid_loader]
        identity = torch.eye(dims, dtype=torch.float64, device=device)
        best = None
        for penalty in self.ridge_alphas:
            weight = torch.linalg.solve(xtx_c + penalty * identity, xty_c)
            bias = mean_y - mean_x @ weight
            squared, absolute, count = 0.0, 0.0, 0
            for features, batch in zip(valid_features, valid_loader):
                error = features @ weight + bias - self._labels(batch, device)
                squared += float((error**2).sum())
                absolute += float(error.abs().sum())
                count += error.numel()
            score = math.sqrt(squared / count) + absolute / count
            if best is None or score < best[0]:
                best = (score, penalty, weight, bias)
        _, penalty, weight, bias = best
        self.ridge_weight.copy_(weight.float())
        self.ridge_bias.copy_(bias.float())
        self.ridge_alpha.fill_(penalty)

    def pretrain(self, train_loader, device) -> None:
        """Stage I contrastive training (SGD, cosine schedule), then stage II ridge; freezes every parameter."""
        self.to(device)
        dataset = train_loader.dataset
        self._fit_calendar(dataset, device)
        iterations = self._iterations(dataset)
        batch_size = min(self.pretrain_batch_size, len(dataset))
        sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=iterations * batch_size)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
        trainable = [p for p in [*self.encoder.parameters(), *self.head.parameters()] if p.requires_grad]
        optimizer = torch.optim.SGD(trainable, lr=self.pretrain_lr, momentum=0.9, weight_decay=1e-4)
        self.train()
        for iteration, batch in enumerate(loader):
            x = batch[0].float().to(device)
            marks = batch[2].float().to(device) if len(batch) > 2 and self.use_marks else None
            inputs = self._inputs(x, marks)
            optimizer.zero_grad()
            self.contrastive_loss(self._augment(inputs), self._augment(inputs)).backward()
            optimizer.step()
            for group in optimizer.param_groups:
                group["lr"] = self.pretrain_lr * 0.5 * (1.0 + math.cos(math.pi * (iteration + 1) / iterations))
        self.fit_ridge(dataset, device)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()
