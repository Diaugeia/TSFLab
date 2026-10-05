"""Evaluation utilities for time-series forecasting models."""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

from tsflab.experiments.runner.model_io import (
    call_forecaster,
    make_decoder_input,
    slice_prediction_target,
    unwrap_model,
)
from tsflab.experiments.evaluation.metrics import collect_metrics, collect_prob_metrics
from tsflab.experiments.evaluation.streaming import MetricAccumulator

_CANONICAL_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _resolve_output_kind(model: nn.Module) -> tuple[str, str]:
    """Read the probabilistic output kind off the model (non-invasive).

    Mirrors ``_collect_aux_loss``'s DataParallel unwrap. Models that never set
    ``output_type`` / ``distribution_family`` report ``("point", "gaussian")``,
    so every existing model takes the byte-identical point path.

    Parameters
    ----------
    model : nn.Module
        The model being evaluated (unwrapped if DataParallel).

    Returns
    -------
    tuple[str, str]
        ``(output_type, distribution_family)``.
    """
    target = unwrap_model(model)
    output_type = getattr(target, "output_type", "point")
    distribution_family = getattr(target, "distribution_family", "gaussian")
    return output_type, distribution_family


def _point_reduce(pred: np.ndarray, output_type: str, levels: list[float]) -> np.ndarray:
    """Reduce a rank-4 probabilistic prediction to a rank-3 point forecast.

    For ``"quantile"`` outputs the median (level closest to ``0.5``) is taken;
    for ``"distribution"`` (Gaussian) the mean equals ``loc`` (channel 0). The
    reduced ``(B, L, C)`` array feeds the unchanged :func:`collect_metrics` so
    the 10 point metrics stay comparable across point and probabilistic models.

    Parameters
    ----------
    pred : np.ndarray
        Rank-4 prediction ``(B, L, C, K)``.
    output_type : str
        ``"quantile"`` or ``"distribution"``.
    levels : list[float]
        Configured quantile levels (used to locate the median).

    Returns
    -------
    np.ndarray
        Rank-3 point forecast ``(B, L, C)``.
    """
    if output_type == "quantile":
        # median = level closest to 0.5
        idx = int(np.argmin(np.abs(np.asarray(levels) - 0.5)))
        return pred[..., idx]
    # distribution gaussian: mean == loc == channel 0
    return pred[..., 0]


def _inverse_transform_prob(
    outputs: np.ndarray,
    batch_y_sliced: np.ndarray,
    dataset,
    output_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Rank-aware inverse-transform of predictions + targets.

    Point (rank-3) outputs use the original reshape path verbatim. For
    probabilistic (rank-4) outputs every ``K`` slice is inverse-transformed
    independently through the dataset scaler (a monotone-affine StandardScaler,
    so per-quantile application preserves quantile ordering). For a Gaussian
    ``distribution`` model the ``scale`` channel (``k == 1``) is a standard
    deviation in scaled space, so it is only multiplied by the scaler's
    per-channel ``scale_`` (no mean shift).

    Parameters
    ----------
    outputs : np.ndarray
        Prediction array (rank-3 or rank-4).
    batch_y_sliced : np.ndarray
        Target array ``(B, L, C)``.
    dataset : object
        Dataset instance providing ``inverse_transform`` and (ideally) a
        ``scaler`` with a ``scale_`` attribute.
    output_type : str
        ``"point"`` / ``"quantile"`` / ``"distribution"``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Inverse-transformed ``(outputs, batch_y_sliced)``.
    """
    tshape = batch_y_sliced.shape  # (B, L, C)
    batch_y_sliced = dataset.inverse_transform(
        batch_y_sliced.reshape(tshape[0] * tshape[1], -1)
    ).reshape(tshape)

    if outputs.ndim == 4:
        B, L, C, K = outputs.shape
        inv = np.empty_like(outputs)
        for k in range(K):
            sl = outputs[..., k]  # (B, L, C)
            inv[..., k] = dataset.inverse_transform(
                sl.reshape(B * L, -1)
            ).reshape(B, L, C)
        if output_type == "distribution":
            # The scale channel (k == 1) is a std-dev in scaled space: undo the
            # mean-shift the generic loop applied by re-deriving it as
            # raw_scale = scaled_scale * scaler.scale_ (no + mean_).
            scaler = getattr(dataset, "scaler", None)
            scale_ = getattr(scaler, "scale_", None)
            if scale_ is not None:
                scale_full = np.asarray(scale_, dtype=outputs.dtype)  # (C_full,)
                # MS / reduced-channel output emits the LAST C of the scaler's
                # columns (the target channel(s), placed last). Slice to match
                # so (B, L, C) * (C,) broadcasts instead of failing for C < C_full.
                scale_vec = scale_full[len(scale_full) - C:]  # (C,)
                inv[..., 1] = outputs[..., 1] * scale_vec[None, None, :]
        outputs = inv
    else:
        outputs = dataset.inverse_transform(
            outputs.reshape(tshape[0] * tshape[1], -1)
        ).reshape(tshape)
    return outputs, batch_y_sliced


def _compute_metrics(
    preds: np.ndarray,
    trues: np.ndarray,
    output_type: str,
    distribution_family: str,
    levels: list[float],
) -> dict[str, float]:
    """Dispatch metric computation by output kind.

    Point models compute the unchanged 10 point keys; probabilistic models
    merge the 10 point keys (computed on the point reduction) with the four
    probabilistic keys from :func:`collect_prob_metrics`.

    Parameters
    ----------
    preds : np.ndarray
        Collected predictions (rank-3 point or rank-4 probabilistic).
    trues : np.ndarray
        Collected targets ``(B, L, C)``.
    output_type : str
        ``"point"`` / ``"quantile"`` / ``"distribution"``.
    distribution_family : str
        Distribution family (only consulted for ``"distribution"``).
    levels : list[float]
        Configured quantile levels.

    Returns
    -------
    dict[str, float]
        Metrics dictionary.
    """
    if output_type == "point":
        return collect_metrics(preds, trues)
    point_preds = _point_reduce(preds, output_type, levels)
    point_metrics = collect_metrics(point_preds, trues)
    prob_metrics = collect_prob_metrics(
        preds, trues, levels, output_type, distribution_family
    )
    return {**point_metrics, **prob_metrics}


def evaluate(
    model: nn.Module,
    data_loader,
    device: torch.device,
    label_len: int,
    pred_len: int,
    features: str,
    inverse: bool = False,
    dataset=None,
    quantile_levels=None,
    checkpoint=None,
    checkpoint_every_batches=0,
) -> tuple[dict[str, float], float]:
    """Run model inference and compute metrics on a dataset split.

    For point models (``output_type`` absent or ``"point"``) the behavior is
    byte-identical to the original evaluator. For probabilistic models the
    rank-4 ``(B, L, C, K)`` output is carried through to numpy without
    collapsing ``K``, inverse-transformed rank-aware, and scored with both the
    point metrics (on a point reduction) and :func:`collect_prob_metrics`.

    Parameters
    ----------
    model : nn.Module
        Forecasting model.
    data_loader : DataLoader
        Data loader for evaluation split.
    device : torch.device
        Target device.
    label_len : int
        Decoder label length.
    pred_len : int
        Prediction horizon length.
    features : str
        Feature mode ("M", "S", "MS").
    inverse : bool, optional
        Whether to inverse-transform outputs via dataset scaler.
    dataset : object, optional
        Dataset instance that provides inverse_transform.
    quantile_levels : list[float] | None, optional
        Configured quantile levels for probabilistic metrics. Ignored for point
        models; falls back to the canonical nine deciles when ``None``.

    Returns
    -------
    tuple[dict[str, float], float]
        Metrics dictionary and evaluation time in seconds.
    """
    output_type, distribution_family = _resolve_output_kind(model)
    levels = list(quantile_levels) if quantile_levels else list(_CANONICAL_LEVELS)

    acc = MetricAccumulator(output_type, distribution_family, levels)

    from pathlib import Path
    from tsflab.experiments.infra.checkpoint import capture_rng, set_rng, runtime_state, restore_runtime_state
    from tsflab.experiments.infra.stages import atomic_state
    if checkpoint is not None and getattr(data_loader, "num_workers", 0):
        raise ValueError("evaluation recovery requires num_workers=0")
    state = None
    if checkpoint is not None and Path(checkpoint).exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if "metrics" in state:
            acc = state["metrics"]
        else:  # an evaluation checkpoint from before streaming metrics
            for pred_batch, true_batch in zip(state["preds"], state["trues"]):
                acc.update(pred_batch, true_batch)
        restore_runtime_state(model, state["runtime"])
        set_rng(state["epoch_rng"])
        if state["loader_rng"] is not None:
            data_loader.generator.set_state(state["loader_rng"])
    epoch_rng = capture_rng()
    loader_rng = data_loader.generator.get_state() if getattr(data_loader, "generator", None) is not None else None
    skip = state["next_batch"] if state else 0
    prior_elapsed = state["elapsed"] if state else 0
    model.eval()
    start_time = time.perf_counter()
    with torch.no_grad():
        for index, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
            if index < skip:
                if index + 1 == skip:
                    set_rng(state["rng"])
                continue
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            if batch_x_mark is not None:
                batch_x_mark = batch_x_mark.float().to(device)
            if batch_y_mark is not None:
                batch_y_mark = batch_y_mark.float().to(device)

            dec_inp = make_decoder_input(batch_y, label_len, pred_len, device)
            outputs = call_forecaster(model, batch_x, batch_x_mark, dec_inp, batch_y_mark)
            outputs, batch_y_sliced = slice_prediction_target(
                outputs, batch_y, pred_len, features
            )

            outputs = outputs.detach().cpu().numpy()
            batch_y_sliced = batch_y_sliced.detach().cpu().numpy()

            if inverse and dataset is not None:
                if output_type == "point":
                    shape = batch_y_sliced.shape
                    outputs = dataset.inverse_transform(
                        outputs.reshape(shape[0] * shape[1], -1)
                    ).reshape(shape)
                    batch_y_sliced = dataset.inverse_transform(
                        batch_y_sliced.reshape(shape[0] * shape[1], -1)
                    ).reshape(shape)
                else:
                    outputs, batch_y_sliced = _inverse_transform_prob(
                        outputs, batch_y_sliced, dataset, output_type
                    )

            acc.update(outputs, batch_y_sliced)
            if checkpoint is not None and checkpoint_every_batches and (index + 1) % checkpoint_every_batches == 0:
                atomic_state(checkpoint, {"next_batch": index + 1, "metrics": acc,
                    "runtime": runtime_state(model), "rng": capture_rng(), "epoch_rng": epoch_rng,
                    "loader_rng": loader_rng, "elapsed": prior_elapsed + time.perf_counter() - start_time})

    test_time = prior_elapsed + time.perf_counter() - start_time
    return acc.result(), test_time


def evaluate_rolling(
    model: nn.Module,
    dataset,
    device: torch.device,
    seq_len: int,
    label_len: int,
    pred_len: int,
    features: str,
    inverse: bool = False,
    horizon: int | None = None,
    stride: int = 1,
    num_rollings: int | None = None,
    quantile_levels=None,
) -> tuple[dict[str, float], float]:
    """Run a TFB-style rolling forecast over the test split.

    Starting at the beginning of the test split, the model repeatedly consumes
    a ``seq_len`` input window, predicts ``pred_len`` steps (sliced to
    ``horizon`` if requested), then the window is advanced by ``stride`` and the
    process repeats for up to ``num_rollings`` rollings (or until the data is
    exhausted). Predictions and ground-truth targets are collected and scored
    with the same :func:`collect_metrics` used by the fixed evaluator.

    The dataset is consumed via its underlying ``data`` / ``time_stamp`` arrays
    (the materialised test split). This keeps the ``(input_series,
    output_series, input_stamp, output_stamp)`` contract intact: the input
    window plays the role of ``input_series`` (+ ``input_stamp``) and the
    label-prefixed future window plays the role of ``output_series``
    (+ ``output_stamp``).

    Parameters
    ----------
    model : nn.Module
        Forecasting model.
    dataset : object
        Test-split dataset instance exposing ``data`` (``(T, C)``), an optional
        ``time_stamp`` (``(T, M)``), and ``inverse_transform``.
    device : torch.device
        Target device.
    seq_len, label_len, pred_len : int
        Encoder input, decoder label, and prediction horizon lengths.
    features : str
        Feature mode ("M", "S", "MS").
    inverse : bool, optional
        Whether to inverse-transform outputs via the dataset scaler.
    horizon : int | None, optional
        Number of predicted steps to score per rolling. Defaults to
        ``pred_len`` and is clamped to ``pred_len`` (the model only emits
        ``pred_len`` steps per call).
    stride : int, optional
        Number of steps to advance the input window between rollings.
    num_rollings : int | None, optional
        Maximum number of rollings. ``None`` (default) rolls until the test
        data is exhausted.
    quantile_levels : list[float] | None, optional
        Configured quantile levels for probabilistic metrics. Ignored for point
        models; falls back to the canonical nine deciles when ``None``.

    Returns
    -------
    tuple[dict[str, float], float]
        Metrics dictionary and evaluation time in seconds.
    """
    output_type, distribution_family = _resolve_output_kind(model)
    levels = list(quantile_levels) if quantile_levels else list(_CANONICAL_LEVELS)

    data = np.asarray(dataset.data)
    time_stamp = getattr(dataset, "time_stamp", None)

    eff_horizon = pred_len if horizon is None else min(int(horizon), pred_len)
    if eff_horizon <= 0:
        raise ValueError(f"rolling horizon must be positive, got {horizon!r}")
    if stride < 1:
        raise ValueError(f"rolling stride must be >= 1, got {stride!r}")

    total_len = data.shape[0]
    # Each rolling needs seq_len input rows plus pred_len future rows.
    last_start = total_len - seq_len - pred_len
    if last_start < 0:
        raise ValueError(
            "rolling forecast needs at least seq_len + pred_len rows in the "
            f"test split, got {total_len} for seq_len={seq_len}, "
            f"pred_len={pred_len}"
        )

    starts = list(range(0, last_start + 1, stride))
    if num_rollings is not None:
        starts = starts[: max(int(num_rollings), 0)]

    acc = MetricAccumulator(output_type, distribution_family, levels)

    model.eval()
    start_time = time.perf_counter()
    with torch.no_grad():
        for start in starts:
            input_end = start + seq_len
            output_start = input_end - label_len
            output_end = input_end + pred_len

            input_series = data[start:input_end]
            output_series = data[output_start:output_end]

            batch_x = (
                torch.from_numpy(np.ascontiguousarray(input_series))
                .float()
                .unsqueeze(0)
                .to(device)
            )
            batch_y = (
                torch.from_numpy(np.ascontiguousarray(output_series))
                .float()
                .unsqueeze(0)
                .to(device)
            )

            if time_stamp is not None:
                batch_x_mark = (
                    torch.from_numpy(np.ascontiguousarray(time_stamp[start:input_end]))
                    .float()
                    .unsqueeze(0)
                    .to(device)
                )
                batch_y_mark = (
                    torch.from_numpy(
                        np.ascontiguousarray(time_stamp[output_start:output_end])
                    )
                    .float()
                    .unsqueeze(0)
                    .to(device)
                )
            else:
                batch_x_mark = (
                    torch.zeros((1, seq_len, 6), dtype=torch.float32).to(device)
                )
                batch_y_mark = (
                    torch.zeros((1, label_len + pred_len, 6), dtype=torch.float32).to(
                        device
                    )
                )

            dec_inp = make_decoder_input(batch_y, label_len, pred_len, device)
            outputs = call_forecaster(model, batch_x, batch_x_mark, dec_inp, batch_y_mark)
            outputs, batch_y_sliced = slice_prediction_target(
                outputs, batch_y, pred_len, features
            )

            # Score only the requested horizon (<= pred_len). Rank-aware: keep
            # the trailing K axis whole for probabilistic outputs.
            if outputs.ndim == 4:
                outputs = outputs[:, :eff_horizon, :, :]
            else:
                outputs = outputs[:, :eff_horizon, :]
            batch_y_sliced = batch_y_sliced[:, :eff_horizon, :]

            outputs = outputs.detach().cpu().numpy()
            batch_y_sliced = batch_y_sliced.detach().cpu().numpy()

            if inverse and dataset is not None:
                if output_type == "point":
                    shape = batch_y_sliced.shape
                    outputs = dataset.inverse_transform(
                        outputs.reshape(shape[0] * shape[1], -1)
                    ).reshape(shape)
                    batch_y_sliced = dataset.inverse_transform(
                        batch_y_sliced.reshape(shape[0] * shape[1], -1)
                    ).reshape(shape)
                else:
                    outputs, batch_y_sliced = _inverse_transform_prob(
                        outputs, batch_y_sliced, dataset, output_type
                    )

            acc.update(outputs, batch_y_sliced)

    test_time = time.perf_counter() - start_time
    if acc.n == 0:
        raise ValueError("rolling forecast produced no windows")
    return acc.result(), test_time
