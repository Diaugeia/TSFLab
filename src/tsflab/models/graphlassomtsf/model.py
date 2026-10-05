"""Paper-driven local implementation of the GraphLASSO forecaster (static variant).

Do, Hy and Nguyen (arXiv 2306.17090) decouple graph discovery from forecasting.
Phase 1 (Sec. IV-A, Eq. 1) estimates a sparse precision matrix ``Theta`` of the
standardized training series with the graphical lasso
``min_Theta -log det Theta + tr(S Theta) + lambda ||Theta||_1``. Phase 2
(Sec. IV-B, Eqs. 4-6) samples an adjacency matrix from ``Theta`` and feeds it to a
graph convolutional recurrent network: GRU gates whose linear maps are diffusion
graph convolutions over a random-walk support, an encoder over the history and an
autoregressive decoder over the horizon. The precision matrix is fitted once from
the training split (``fit_precision``, called by the spec's ``training_setup``) and
stored in the ``precision`` buffer; the graph sampling and recurrent cell follow the
pinned official code where it is more specific than the paper.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.graph_conv_gru import GraphConvGRUCell


def _admm_single_graphical_lasso(
    covariance: np.ndarray, lasso_lambda: float, max_iter: int, tol: float, rtol: float
) -> np.ndarray:
    """Scaled ADMM for ``min -log det Omega + tr(S Omega) + lambda ||Theta||_{1,od}``, ``Omega = Theta``.

    Mirrors ``gglasso.solver.single_admm_solver.ADMM_SGL`` as the official
    ``glasso_problem.solve`` calls it: start ``Omega = Theta = I``, ``X = 0``,
    ``rho = 1`` with residual balancing (x2 / x0.5 at a 10x imbalance), the
    Boyd et al. stopping rule, off-diagonal soft-thresholding, and the sparse
    ``Theta`` as the estimate. The ``Omega`` step is the proximal map of
    ``-log det`` through one eigendecomposition, so every iterate is positive
    definite and the solver cannot fail on an ill-conditioned ``S``.
    """
    size = covariance.shape[0]
    omega = np.eye(size)
    theta = np.eye(size)
    dual = np.zeros((size, size))
    rho = 1.0
    dim = (size**2 + size) / 2
    diagonal = np.arange(size)
    for _ in range(max_iter):
        eigenvalues, eigenvectors = np.linalg.eigh(theta - dual - covariance / rho)
        previous = omega
        beta = 1.0 / rho
        omega = (eigenvectors * (0.5 * (np.sqrt(eigenvalues**2 + 4 * beta) + eigenvalues))) @ eigenvectors.T
        target = omega + dual
        theta = np.sign(target) * np.maximum(np.abs(target) - lasso_lambda / rho, 0.0)
        theta[diagonal, diagonal] = target[diagonal, diagonal]
        dual = dual + omega - theta
        primal_residual = np.linalg.norm(omega - theta)
        dual_residual = rho * np.linalg.norm(omega - previous)
        primal_bound = dim * tol + rtol * max(np.linalg.norm(omega), np.linalg.norm(theta))
        dual_bound = dim * tol + rtol * rho * np.linalg.norm(dual)
        if primal_residual >= 10 * dual_residual:
            new_rho = 2 * rho
        elif dual_residual >= 10 * primal_residual:
            new_rho = 0.5 * rho
        else:
            new_rho = rho
        dual = (rho / new_rho) * dual
        rho = new_rho
        if primal_residual <= primal_bound and dual_residual <= dual_bound:
            break
    return theta


def graphical_lasso(
    covariance: np.ndarray, lasso_lambda: float, max_iter: int = 1000, tol: float = 1e-10
) -> np.ndarray:
    """Sparse precision of ``covariance`` as ``gglasso.solver.single_admm_solver.block_SGL``.

    Witten-Friedman-Simon screening: nodes are split into the connected
    components of ``|S_ij| > lambda``; a single node gets ``1 / S_ii`` and every
    larger block is solved by ADMM, so cost is ``O(max_iter * block^3)`` per block.
    ``tol`` is used for both the absolute and relative tolerance, as the official
    ``P.solve(tol=1e-10, rtol=1e-10)`` passes it to ``block_SGL``.
    """
    from scipy.sparse.csgraph import connected_components

    adjacency = np.abs(covariance) > lasso_lambda
    np.fill_diagonal(adjacency, True)
    count, labels = connected_components(adjacency, directed=False)
    precision = np.zeros_like(covariance)
    for component in range(count):
        block = np.flatnonzero(labels == component)
        if block.size == 1:
            precision[block[0], block[0]] = 1.0 / covariance[block[0], block[0]]
            continue
        index = np.ix_(block, block)
        precision[index] = _admm_single_graphical_lasso(covariance[index], lasso_lambda, max_iter, tol, tol)
    return precision


def estimate_precision(
    series: np.ndarray, lasso_lambda: float, cov_eps: float, max_iter: int = 1000
) -> np.ndarray:
    """Phase 1 (Eq. 1): sparse precision matrix of the standardized ``[T, N]`` series.

    Each column is z-scored, the sample covariance gets ``cov_eps`` on its
    diagonal, the graphical lasso (off-diagonal l1 penalty ``lasso_lambda``) is
    solved on the matching correlation matrix with the official ADMM
    (:func:`graphical_lasso`), and the estimate is rescaled to the covariance
    scale, ``Theta_ij = Theta^corr_ij / sqrt(S_ii S_jj)`` (``do_scaling=True``).
    """
    values = np.asarray(series, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("series must be [time >= 2, nodes]")
    std = values.std(axis=0)
    std[std == 0] = 1.0
    standardized = (values - values.mean(axis=0)) / std
    nodes = values.shape[1]
    covariance = np.atleast_2d(np.cov(standardized.T)) + cov_eps * np.eye(nodes)
    scale = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(scale, scale)
    precision = graphical_lasso(correlation, lasso_lambda, max_iter=max_iter)
    return precision / np.outer(scale, scale)


def random_walk_support(adjacency: torch.Tensor) -> torch.Tensor:
    """``(D^{-1} (A + I))^T`` with ``D`` the row sums of ``A + I``."""
    with_self = adjacency + torch.eye(adjacency.shape[-1], dtype=adjacency.dtype, device=adjacency.device)
    degree = with_self.sum(dim=-1, keepdim=True)
    inverse = torch.where(degree > 0, 1.0 / degree, torch.zeros_like(degree))
    return (inverse * with_self).transpose(-1, -2)


class DiffusionGraphLinear(nn.Module):
    """Eq. (5) generalised to ``K`` hops: a linear map of ``[x, P x, T_2 x, ..., T_K x]``.

    ``T_1 = P x`` and ``T_k = 2 P T_{k-1} - T_{k-2}`` (the recursion of the
    official cell); with ``K = 1`` this is ``[x, P x] W + b``.
    """

    def __init__(self, in_features: int, out_features: int, max_diffusion_step: int, bias_start: float) -> None:
        super().__init__()
        self.max_diffusion_step = max_diffusion_step
        self.linear = nn.Linear(in_features * (max_diffusion_step + 1), out_features)
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, bias_start)

    def diffuse(self, x: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        """``[B, N, F] -> [B, N, F * (K + 1)]`` (feature-major, hop-minor)."""
        terms = [x]
        if self.max_diffusion_step >= 1:
            previous, current = x, torch.einsum("ij,bjf->bif", support, x)
            terms.append(current)
            for _ in range(2, self.max_diffusion_step + 1):
                previous, current = current, 2 * torch.einsum("ij,bjf->bif", support, current) - previous
                terms.append(current)
        return torch.stack(terms, dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        return self.linear(self.diffuse(x, support))


class GraphGRUCell(GraphConvGRUCell):
    """Eq. (6): GRU whose gate and candidate maps are diffusion graph convolutions.

    ``forward(x, hidden, support)`` is the shared ``graph_conv_gru`` recurrence; the
    gate output has width ``2 * units``, so its halves are the reset and update gates.
    """

    def __init__(self, input_dim: int, units: int, max_diffusion_step: int) -> None:
        gates = DiffusionGraphLinear(input_dim + units, 2 * units, max_diffusion_step, bias_start=1.0)
        candidate = DiffusionGraphLinear(input_dim + units, units, max_diffusion_step, bias_start=0.0)
        super().__init__(gates, candidate)
        self.units = units


class GraphGRUStack(nn.Module):
    """Stacked graph GRU cells; returns the new hidden state of every layer."""

    def __init__(self, input_dim: int, units: int, layers: int, max_diffusion_step: int) -> None:
        super().__init__()
        self.cells = nn.ModuleList(
            GraphGRUCell(input_dim if index == 0 else units, units, max_diffusion_step) for index in range(layers)
        )

    def forward(self, x: torch.Tensor, hidden: list[torch.Tensor], support: torch.Tensor) -> list[torch.Tensor]:
        states = []
        for cell, state in zip(self.cells, hidden):
            x = cell(x, state, support)
            states.append(x)
        return states


class Model(nn.Module):
    """GraphLASSO precision graph + graph convolutional recurrent encoder-decoder."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        rnn_units: int = 32,
        num_rnn_layers: int = 1,
        max_diffusion_step: int = 1,
        lasso_lambda: float = 0.02,
        cov_eps: float = 1e-3,
        use_curriculum_learning: bool = True,
        cl_decay_steps: int = 2000,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.rnn_units = rnn_units
        self.num_rnn_layers = num_rnn_layers
        self.lasso_lambda = lasso_lambda
        self.cov_eps = cov_eps
        self.use_curriculum_learning = use_curriculum_learning
        self.cl_decay_steps = cl_decay_steps
        self.encoder = GraphGRUStack(1, rnn_units, num_rnn_layers, max_diffusion_step)
        self.decoder = GraphGRUStack(1, rnn_units, num_rnn_layers, max_diffusion_step)
        self.projection = nn.Linear(rnn_units, 1)
        # Phase-1 output; identity until ``fit_precision`` runs on the training split.
        self.register_buffer("precision", torch.eye(enc_in))
        self.register_buffer("batches_seen", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def fit_precision(self, series: torch.Tensor | np.ndarray) -> None:
        """Phase 1: estimate ``Theta`` from the ``[T, N]`` training series."""
        values = series.detach().cpu().numpy() if torch.is_tensor(series) else np.asarray(series)
        if values.ndim != 2 or values.shape[1] != self.enc_in:
            raise ValueError(f"expected a [time, {self.enc_in}] training series")
        precision = estimate_precision(values, self.lasso_lambda, self.cov_eps)
        self.precision.copy_(torch.as_tensor(precision, dtype=self.precision.dtype))

    def adjacency(self) -> torch.Tensor:
        """Phase 2 graph from ``Theta`` (each row is a categorical over target nodes).

        Training draws one hard Gumbel-max sample per forward pass (one edge per
        row, as the official straight-through Gumbel-softmax); evaluation uses its
        expectation, the row-wise ``softmax(Theta)``, so inference is deterministic.
        """
        if self.training:
            uniform = torch.rand_like(self.precision)
            gumbel = -torch.log(-torch.log(uniform + 1e-20) + 1e-20)
            index = (self.precision + gumbel).argmax(dim=-1)
            return F.one_hot(index, self.enc_in).to(self.precision.dtype)
        return torch.softmax(self.precision, dim=-1)

    def teacher_forcing_probability(self) -> float:
        """Curriculum schedule ``k / (k + exp(step / k))`` of the official code."""
        k = self.cl_decay_steps
        return k / (k + math.exp(min(int(self.batches_seen) / k, 700.0)))

    def _initial_state(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [x.new_zeros(x.shape[0], self.enc_in, self.rnn_units) for _ in range(self.num_rnn_layers)]

    def forecast(
        self,
        x_enc: torch.Tensor,
        targets: torch.Tensor | None = None,
        teacher_probability: float = 0.0,
    ) -> torch.Tensor:
        """Encode ``[B, L, N]`` and decode ``[B, H, N]``; optional scheduled teacher forcing."""
        support = random_walk_support(self.adjacency().to(x_enc.dtype))
        hidden = self._initial_state(x_enc)
        for step in range(x_enc.shape[1]):
            hidden = self.encoder(x_enc[:, step, :, None], hidden, support)
        decoder_input = x_enc.new_zeros(x_enc.shape[0], self.enc_in, 1)  # GO symbol
        outputs = []
        for step in range(self.pred_len):
            hidden = self.decoder(decoder_input, hidden, support)
            output = self.projection(hidden[-1])  # [B, N, 1]
            outputs.append(output.squeeze(-1))
            decoder_input = output
            if targets is not None and teacher_probability > 0 and float(torch.rand(())) < teacher_probability:
                decoder_input = targets[:, step, :, None]
        return torch.stack(outputs, dim=1)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        return self.forecast(x_enc)
