"""Natural cubic spline control path of a sampled multichannel series.

Paper-neutral building block of neural controlled differential equations
(NCDEs): it turns observations ``x_0 .. x_{L-1}`` at increasing times
``t_0 < ... < t_{L-1}`` into a twice-differentiable path ``X(t)`` whose second
derivative is zero at both ends, and evaluates ``X(t)`` and ``dX/dt``.

With ``h_i = t_{i+1} - t_i`` and ``D_i = x_{i+1} - x_i``, the knot slopes ``k``
solve the tridiagonal system

    2 k_0 / h_0 + k_1 / h_0                         = 3 D_0 / h_0^2
    k_{i-1} / h_{i-1} + 2 k_i (1/h_{i-1} + 1/h_i) + k_{i+1} / h_i
                                                    = 3 D_{i-1} / h_{i-1}^2 + 3 D_i / h_i^2
    k_{L-2} / h_{L-2} + 2 k_{L-1} / h_{L-2}         = 3 D_{L-2} / h_{L-2}^2

and on piece ``i`` (``s = t - t_i``)

    X(t)  = a_i + b_i s + c_i s^2 + d_i s^3,
    a_i = x_i,  b_i = k_i,
    2 c_i = (6 D_i / h_i - 4 k_i - 2 k_{i+1}) / h_i,
    3 d_i = (-6 D_i / h_i + 3 (k_i + k_{i+1})) / h_i^2.

The coefficients are stored as ``(a, b, 2c, 3d)`` so the derivative
``b + (2c + 3d s) s`` needs no rescaling. Times outside ``[t_0, t_{L-1}]`` extend
the first or last piece.
"""

from __future__ import annotations

import torch

SplineCoeffs = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def natural_cubic_spline_coeffs(times: torch.Tensor, x: torch.Tensor) -> SplineCoeffs:
    """Return ``(a, b, two_c, three_d)``, each ``[..., L - 1, C]``, for ``x [..., L, C]``.

    ``times`` is a strictly increasing ``[L]`` tensor with ``L >= 2``. Missing
    values (NaN) are not supported.
    """
    if times.dim() != 1 or times.shape[0] < 2:
        raise ValueError("times must be one-dimensional with at least two points")
    if x.dim() < 2 or x.shape[-2] != times.shape[0]:
        raise ValueError("x must be [..., L, C] with L equal to len(times)")
    times = times.to(device=x.device, dtype=x.dtype)
    steps = times[1:] - times[:-1]
    if bool((steps <= 0).any()):
        raise ValueError("times must be strictly increasing")
    length = times.shape[0]
    inv = steps.reciprocal()
    path = x.transpose(-1, -2)  # [..., C, L]
    diffs = path[..., 1:] - path[..., :-1]
    scaled = 3 * diffs * inv.square()
    rhs = torch.zeros_like(path)
    rhs[..., :-1] = scaled
    rhs[..., 1:] = rhs[..., 1:] + scaled
    system = torch.zeros(length, length, device=x.device, dtype=x.dtype)
    diagonal = torch.zeros(length, device=x.device, dtype=x.dtype)
    diagonal[:-1] = inv
    diagonal[1:] = diagonal[1:] + inv
    index = torch.arange(length - 1, device=x.device)
    system[index, index + 1] = inv
    system[index + 1, index] = inv
    system = system + torch.diag(2 * diagonal)
    # Knot slopes: system @ k = rhs along the last axis.
    slopes = torch.linalg.solve(system, rhs.unsqueeze(-1)).squeeze(-1)
    a = path[..., :-1]
    b = slopes[..., :-1]
    two_c = (6 * diffs * inv - 4 * slopes[..., :-1] - 2 * slopes[..., 1:]) * inv
    three_d = (-6 * diffs * inv + 3 * (slopes[..., :-1] + slopes[..., 1:])) * inv.square()
    return tuple(part.transpose(-1, -2) for part in (a, b, two_c, three_d))  # type: ignore[return-value]


class NaturalCubicSpline:
    """Evaluate a natural cubic spline and its derivative at scalar times.

    Args:
        times: The ``[L]`` knot times passed to :func:`natural_cubic_spline_coeffs`.
        coeffs: Its ``(a, b, two_c, three_d)`` output.
    """

    def __init__(self, times: torch.Tensor, coeffs: SplineCoeffs) -> None:
        self.a, self.b, self.two_c, self.three_d = coeffs
        self.times = times.to(device=self.a.device, dtype=self.a.dtype)

    def _locate(self, t: torch.Tensor | float) -> tuple[int, torch.Tensor]:
        t = torch.as_tensor(t, device=self.times.device, dtype=self.times.dtype)
        index = int(((t > self.times).sum() - 1).clamp(0, self.b.shape[-2] - 1))
        return index, t - self.times[index]

    def evaluate(self, t: torch.Tensor | float) -> torch.Tensor:
        """``X(t)`` as ``[..., C]``."""
        index, s = self._locate(t)
        inner = 0.5 * self.two_c[..., index, :] + self.three_d[..., index, :] * s / 3
        return self.a[..., index, :] + (self.b[..., index, :] + inner * s) * s

    def derivative(self, t: torch.Tensor | float) -> torch.Tensor:
        """``dX/dt`` at ``t`` as ``[..., C]``."""
        index, s = self._locate(t)
        return self.b[..., index, :] + (self.two_c[..., index, :] + self.three_d[..., index, :] * s) * s


__all__ = ["NaturalCubicSpline", "SplineCoeffs", "natural_cubic_spline_coeffs"]
