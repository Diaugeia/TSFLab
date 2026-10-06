"""Denoising diffusion probabilistic model with epsilon prediction (Ho et al., 2020).

Forward process ``q(x_n | x_0) = N(sqrt(abar_n) x_0, (1 - abar_n) I)`` with
``alpha_n = 1 - beta_n`` and ``abar_n = prod_{i<=n} alpha_i``. A caller-owned
network ``eps_theta(x_n, n)`` predicts the added noise; training minimises
``|| eps - eps_theta(sqrt(abar_n) x_0 + sqrt(1 - abar_n) eps, n) ||^2`` with ``n``
uniform. Reverse step: ``x_0_hat = (x_n - sqrt(1 - abar_n) eps_theta) / sqrt(abar_n)``,
then the posterior ``q(x_{n-1} | x_n, x_0_hat)`` mean with variance
``beta_tilde_n = beta_n (1 - abar_{n-1}) / (1 - abar_n)``, and no noise on the last
step. This equals ``(x_n - beta_n / sqrt(1 - abar_n) eps_theta) / sqrt(alpha_n)`` plus
``sqrt(beta_tilde_n) z``.

Steps are indexed ``0 .. steps - 1`` (paper ``n = 1 .. N``).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

__all__ = ["GaussianDDPM", "beta_schedule"]

Denoiser = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def beta_schedule(steps: int, beta_start: float, beta_end: float, kind: str = "linear") -> torch.Tensor:
    """``[steps]`` float64 betas: ``linear`` or ``quad`` (linear in sqrt(beta))."""
    if steps < 1:
        raise ValueError("steps must be positive")
    if not 0.0 < beta_start <= beta_end < 1.0:
        raise ValueError("need 0 < beta_start <= beta_end < 1")
    if kind == "linear":
        return torch.linspace(beta_start, beta_end, steps, dtype=torch.float64)
    if kind == "quad":
        return torch.linspace(beta_start**0.5, beta_end**0.5, steps, dtype=torch.float64) ** 2
    raise ValueError(f"unsupported beta schedule {kind!r}")


class GaussianDDPM(nn.Module):
    """Fixed DDPM schedule with epsilon-prediction loss and ancestral sampling."""

    def __init__(
        self,
        steps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.1,
        schedule: str = "linear",
    ) -> None:
        super().__init__()
        betas = beta_schedule(steps, beta_start, beta_end, schedule)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, dim=0)
        abar_prev = torch.cat((abar.new_ones(1), abar[:-1]))
        posterior_variance = betas * (1.0 - abar_prev) / (1.0 - abar)
        self.steps = steps
        buffers = {
            "sqrt_abar": abar.sqrt(),
            "sqrt_one_minus_abar": (1.0 - abar).sqrt(),
            "recip_sqrt_abar": (1.0 / abar).sqrt(),
            "sqrt_recip_abar_minus_one": (1.0 / abar - 1.0).sqrt(),
            "posterior_coef_x0": betas * abar_prev.sqrt() / (1.0 - abar),
            "posterior_coef_xn": (1.0 - abar_prev) * alphas.sqrt() / (1.0 - abar),
            "posterior_std": posterior_variance.clamp(min=1e-20).sqrt(),
        }
        for name, value in buffers.items():
            self.register_buffer(name, value.float(), persistent=False)

    @staticmethod
    def _at(table: torch.Tensor, n: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        return table[n].view(-1, *([1] * (like.ndim - 1))).to(like.dtype)

    def q_sample(self, x0: torch.Tensor, n: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """``sqrt(abar_n) x0 + sqrt(1 - abar_n) noise``; ``n`` is ``[x0.shape[0]]`` long."""
        return self._at(self.sqrt_abar, n, x0) * x0 + self._at(self.sqrt_one_minus_abar, n, x0) * noise

    def loss(self, denoiser: Denoiser, x0: torch.Tensor) -> torch.Tensor:
        """Mean squared epsilon error with one uniform step per leading index."""
        n = torch.randint(0, self.steps, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        prediction = denoiser(self.q_sample(x0, n, noise), n)
        return torch.mean((prediction - noise) ** 2)

    def p_sample(self, denoiser: Denoiser, x: torch.Tensor, step: int) -> torch.Tensor:
        """One reverse step from ``x_step`` to ``x_{step-1}`` (no noise at ``step == 0``)."""
        n = torch.full((x.shape[0],), step, device=x.device, dtype=torch.long)
        eps = denoiser(x, n)
        x0_hat = self._at(self.recip_sqrt_abar, n, x) * x - self._at(self.sqrt_recip_abar_minus_one, n, x) * eps
        mean = self._at(self.posterior_coef_x0, n, x) * x0_hat + self._at(self.posterior_coef_xn, n, x) * x
        if step == 0:
            return mean
        return mean + self._at(self.posterior_std, n, x) * torch.randn_like(x)

    @torch.no_grad()
    def sample(self, denoiser: Denoiser, shape: tuple[int, ...], device=None, dtype=None) -> torch.Tensor:
        """Draw ``x_0`` of ``shape`` from white noise through all reverse steps."""
        x = torch.randn(shape, device=device or self.sqrt_abar.device, dtype=dtype or torch.float32)
        for step in reversed(range(self.steps)):
            x = self.p_sample(denoiser, x, step)
        return x
