"""Clean-room DistDF objective with a local direct forecasting backbone.

DistDF changes the training objective rather than the inference architecture.
The ordinary ``forward`` path is a channel-wise direct forecast; the paper's
joint Gaussian Bures--Wasserstein objective is exposed by ``training_loss``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.channel_wise_linear import ChannelWiseLinear
from tsflab.models._components.revin import RevIN


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        gamma: float = 0.1,
        covariance_eps: float = 1e-5,
        use_revin: bool = True,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len, enc_in) < 1:
            raise ValueError("DistDF dimensions must be positive")
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must lie in [0, 1]")
        if covariance_eps <= 0:
            raise ValueError("covariance_eps must be positive")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.gamma = float(gamma)
        self.covariance_eps = float(covariance_eps)
        self.use_revin = use_revin

        self.revin = RevIN(enc_in, affine=True, subtract_last=False)
        self.forecaster = ChannelWiseLinear(
            seq_len, pred_len, enc_in, individual=False
        )

    def _validate(self, x: torch.Tensor) -> None:
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(
                f"expected [batch, {self.seq_len}, {self.enc_in}], got {tuple(x.shape)}"
            )

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        self._validate(x_enc)
        normalized = self.revin(x_enc, "norm") if self.use_revin else x_enc
        forecast = self.forecaster(normalized.transpose(1, 2)).transpose(1, 2)
        return self.revin(forecast, "denorm") if self.use_revin else forecast

    @staticmethod
    def _covariance(samples: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
        mean = samples.mean(0)
        centered = samples - mean
        covariance = centered.transpose(0, 1) @ centered / max(samples.shape[0] - 1, 1)
        identity = torch.eye(samples.shape[1], device=samples.device, dtype=samples.dtype)
        return mean, covariance + eps * identity

    @staticmethod
    def _psd_sqrt(matrix: torch.Tensor) -> torch.Tensor:
        """Symmetric PSD square root, computed in float64.

        Follows the official ``sqrtm_svd_stable`` (``utils/fft_ot.py``):
        symmetrise, decompose in float64, clamp the spectrum at zero. The
        decomposition is ``eigh``; when the solver does not converge
        (ill-conditioned or near-repeated spectra, e.g. 816 x 816 joint
        covariances on traffic), it is retried with a diagonal jitter of
        ``1e-10, 1e-8, 1e-6`` times the mean eigenvalue (``trace / d``), then
        falls back to the official SVD route.
        """
        matrix = matrix.to(torch.float64)
        matrix = 0.5 * (matrix + matrix.transpose(-1, -2))
        scale = (torch.diagonal(matrix).sum() / matrix.shape[-1]).detach().abs().clamp_min(1e-12)
        identity = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
        for jitter in (0.0, 1e-10, 1e-8, 1e-6):
            try:
                eigenvalues, eigenvectors = torch.linalg.eigh(matrix + (jitter * scale) * identity)
            except torch.linalg.LinAlgError:
                continue
            return (eigenvectors * eigenvalues.clamp_min(0).sqrt().unsqueeze(0)) @ eigenvectors.transpose(-1, -2)
        left, singular, _ = torch.linalg.svd(matrix, full_matrices=False)
        return (left * singular.clamp_min(0).sqrt().unsqueeze(0)) @ left.transpose(-1, -2)

    @classmethod
    def bures_wasserstein(
        cls,
        mean_a: torch.Tensor,
        mean_b: torch.Tensor,
        covariance_a: torch.Tensor,
        covariance_b: torch.Tensor,
    ) -> torch.Tensor:
        """Paper equation (5), squared Gaussian W2/Bures discrepancy (float64 inside)."""
        dtype = covariance_a.dtype
        covariance_a = covariance_a.to(torch.float64)
        covariance_b = covariance_b.to(torch.float64)
        sqrt_a = cls._psd_sqrt(covariance_a)
        middle = cls._psd_sqrt(sqrt_a @ covariance_b @ sqrt_a)
        mean_term = (mean_a - mean_b).to(torch.float64).square().sum()
        covariance_term = torch.trace(covariance_a + covariance_b - 2 * middle)
        return (mean_term + covariance_term.clamp_min(0)).to(dtype)

    def joint_distribution_discrepancy(
        self, x: torch.Tensor, forecast: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Algorithm 1 steps 2--4 over batch-channel empirical samples."""
        self._validate(x)
        if forecast.shape != target.shape or forecast.shape[1:] != (
            self.pred_len,
            self.enc_in,
        ):
            raise ValueError("forecast and target must match the output contract")
        history_samples = x.transpose(1, 2).reshape(-1, self.seq_len)
        forecast_samples = forecast.transpose(1, 2).reshape(-1, self.pred_len)
        target_samples = target.transpose(1, 2).reshape(-1, self.pred_len)
        joint_target = torch.cat((history_samples, target_samples), dim=-1)
        joint_forecast = torch.cat((history_samples, forecast_samples), dim=-1)
        mean_target, covariance_target = self._covariance(
            joint_target, self.covariance_eps
        )
        mean_forecast, covariance_forecast = self._covariance(
            joint_forecast, self.covariance_eps
        )
        return self.bures_wasserstein(
            mean_target, mean_forecast, covariance_target, covariance_forecast
        )

    def training_objective(
        self, x: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Paper equation (6): gamma * L_dist + (1-gamma) * MSE."""
        forecast = self(x)
        mse = F.mse_loss(forecast, target)
        discrepancy = self.joint_distribution_discrepancy(x, forecast, target)
        total = self.gamma * discrepancy + (1 - self.gamma) * mse
        return forecast, total, {"mse": mse, "joint_wasserstein": discrepancy}
