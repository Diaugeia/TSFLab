"""Streaming metric accumulation: the metric suite without holding every prediction.

:class:`MetricAccumulator` takes predictions and targets batch by batch and keeps
only the sufficient statistics of :func:`collect_metrics` (point) and
:func:`collect_prob_metrics` (quantile / distribution). The formulas are the same
as in :mod:`tsflab.experiments.evaluation.metrics`; sums are kept in float64, and
the variance terms of ``corr`` and ``rse`` are merged with Chan's parallel update,
so the result equals the concatenate-then-compute path up to float rounding.
Memory is O(channels x quantiles) instead of O(test windows x horizon x channels).
"""

from __future__ import annotations

import math

import numpy as np

from tsflab.experiments.evaluation.metrics import _level_index, _phi_cdf, _phi_inv, _phi_pdf, _pinball


class _Moments:
    """Count, mean and centred second moments merged across batches (per channel or scalar)."""

    def __init__(self) -> None:
        self.n = 0
        self.mean_p = self.mean_t = self.m2_p = self.m2_t = self.c_pt = None

    def update(self, p: np.ndarray, t: np.ndarray) -> None:
        """``p``, ``t``: ``(N, C)`` float64 samples (C may be 1 for the scalar case)."""
        nb = p.shape[0]
        if nb == 0:
            return
        mp, mt = p.mean(axis=0), t.mean(axis=0)
        dp, dt = p - mp, t - mt
        m2p, m2t, cpt = (dp * dp).sum(axis=0), (dt * dt).sum(axis=0), (dp * dt).sum(axis=0)
        if self.n == 0:
            self.n, self.mean_p, self.mean_t, self.m2_p, self.m2_t, self.c_pt = nb, mp, mt, m2p, m2t, cpt
            return
        na, n = self.n, self.n + nb
        delta_p, delta_t = mp - self.mean_p, mt - self.mean_t
        w = na * nb / n
        self.mean_p = self.mean_p + delta_p * nb / n
        self.mean_t = self.mean_t + delta_t * nb / n
        self.m2_p = self.m2_p + m2p + delta_p * delta_p * w
        self.m2_t = self.m2_t + m2t + delta_t * delta_t * w
        self.c_pt = self.c_pt + cpt + delta_p * delta_t * w
        self.n = n


class MetricAccumulator:
    """Accumulate the point (and probabilistic) metric suite over batches.

    Parameters
    ----------
    output_type : str
        ``"point"``, ``"quantile"`` or ``"distribution"``.
    distribution_family : str
        Distribution family for ``"distribution"`` (closed-form CRPS for gaussian).
    levels : list[float]
        Quantile levels (used by probabilistic outputs).
    """

    def __init__(self, output_type: str = "point", distribution_family: str = "gaussian",
                 levels: list[float] | None = None) -> None:
        self.output_type = output_type
        self.distribution_family = distribution_family
        self.levels = list(levels or [])
        self.n = 0
        self.sum_abs = self.sum_sq = self.sum_ape = self.sum_spe = 0.0
        self.sum_abs_true = self.sum_smape = 0.0
        self.channel = _Moments()   # corr: per-channel moments
        self.scalar = _Moments()    # rse: moments of the flattened target
        self.naive_sum = 0.0        # mase: |flat[i+1] - flat[i]| over the flattened targets
        self.naive_n = 0
        self.last_true = None
        # probabilistic
        self.pinball = None
        self.crps_gauss = 0.0
        self.covered = 0.0
        self.width = 0.0

    # -- point part (shared by every output type, on the point forecast) --
    def _update_point(self, pred: np.ndarray, true: np.ndarray) -> None:
        p = np.asarray(pred, dtype=np.float64)
        t = np.asarray(true, dtype=np.float64)
        self.n += t.size
        err = t - p
        self.sum_abs += float(np.abs(err).sum())
        self.sum_sq += float((err * err).sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = err / t
            self.sum_ape += float(np.abs(ratio).sum())
            self.sum_spe += float(np.square(ratio).sum())
        self.sum_abs_true += float(np.abs(t).sum())
        self.sum_smape += float(np.nan_to_num(2.0 * np.abs(p - t) / (np.abs(p) + np.abs(t) + 1e-8)).sum())
        c = t.shape[-1]
        self.channel.update(p.reshape(-1, c), t.reshape(-1, c))
        flat = t.reshape(-1)
        self.scalar.update(flat[:, None], flat[:, None])
        if self.last_true is not None:
            flat = np.concatenate([[self.last_true], flat])
        if flat.size > 1:
            self.naive_sum += float(np.abs(flat[1:] - flat[:-1]).sum())
            self.naive_n += flat.size - 1
        self.last_true = float(flat[-1])

    def update(self, pred: np.ndarray, true: np.ndarray) -> None:
        """Add one batch: ``pred`` ``(B, L, C)`` or ``(B, L, C, K)``, ``true`` ``(B, L, C)``."""
        if self.output_type == "point":
            self._update_point(pred, true)
            return
        if self.output_type == "quantile":
            idx = int(np.argmin(np.abs(np.asarray(self.levels) - 0.5)))
            point = pred[..., idx]
            grid = np.asarray(pred, dtype=np.float64)
        else:
            point = pred[..., 0]
            arr = np.asarray(pred, dtype=np.float64)
            z = _phi_inv(np.asarray(self.levels, dtype=np.float64))
            grid = arr[..., 0][..., None] + arr[..., 1][..., None] * z.reshape(*([1] * (arr.ndim - 1)), -1)
        self._update_point(point, true)
        y = np.asarray(true, dtype=np.float64)
        q = grid.shape[-1]
        sums = np.array([float(_pinball(grid[..., i], y, self.levels[i]).sum()) for i in range(q)])
        self.pinball = sums if self.pinball is None else self.pinball + sums
        if self.output_type == "distribution" and self.distribution_family == "gaussian":
            arr = np.asarray(pred, dtype=np.float64)
            mu, sigma = arr[..., 0], arr[..., 1]
            omega = (y - mu) / sigma
            self.crps_gauss += float((sigma * (omega * (2.0 * _phi_cdf(omega) - 1.0)
                                               + 2.0 * _phi_pdf(omega) - 1.0 / math.sqrt(math.pi))).sum())
        lo = grid[..., _level_index(self.levels, 0.1, fallback=0)]
        hi = grid[..., _level_index(self.levels, 0.9, fallback=len(self.levels) - 1)]
        self.covered += float(((y >= lo) & (y <= hi)).sum())
        self.width += float((hi - lo).sum())

    def result(self) -> dict[str, float]:
        """The metric dictionary of :func:`collect_metrics` (+ probabilistic keys)."""
        if self.n == 0:
            raise ValueError("no predictions were accumulated")
        n = self.n
        ch = self.channel
        denom = np.sqrt(ch.m2_p * ch.m2_t)
        per_channel = np.divide(ch.c_pt, denom, out=np.zeros_like(ch.c_pt), where=denom > 1e-12)
        naive = self.naive_sum / self.naive_n if self.naive_n else float("nan")
        mae64 = self.sum_abs / n
        metrics = {
            "mae": mae64,
            "mse": self.sum_sq / n,
            "rmse": math.sqrt(self.sum_sq / n),
            "mape": self.sum_ape / n,
            "mspe": self.sum_spe / n,
            "corr": float(np.nan_to_num(per_channel).mean()),
            "rse": float(math.sqrt(self.sum_sq) / (math.sqrt(float(self.scalar.m2_t[0])) + 1e-5)),
            "wape": float(self.sum_abs / (self.sum_abs_true + 1e-5)),
            "smape": float(self.sum_smape / n * 100.0),
            "mase": (float("nan") if not self.naive_n or not np.isfinite(naive) or naive <= 1e-12
                     else float(mae64 / naive)),
        }
        if self.output_type == "point":
            return {k: float(v) for k, v in metrics.items()}
        q = len(self.pinball)
        if self.output_type == "distribution" and self.distribution_family == "gaussian":
            crps = self.crps_gauss / n
        else:
            crps = (2.0 / q) * float((self.pinball / n).sum())
        metrics.update({
            "crps": float(crps),
            "wql": float(2.0 * self.pinball.sum() / (self.sum_abs_true + 1e-12) / q),
            "coverage_80": self.covered / n,
            "width_80": self.width / n,
        })
        return {k: float(v) for k, v in metrics.items()}
