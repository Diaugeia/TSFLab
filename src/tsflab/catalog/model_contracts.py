"""Executable construction and forward contracts for every catalog model."""

from __future__ import annotations

import contextlib
import gc
import io
import tomllib
from dataclasses import dataclass
from datetime import datetime, timedelta

from tsflab.core.paths import repository_root
from types import SimpleNamespace

import numpy as np
import torch

from tsflab.catalog.registry.models import MODEL_CATALOG, ModelSpec
from tsflab.experiments.runner.model_io import call_forecaster
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.experiments.config.schema.evaluation import EvaluationConfig


ROOT = repository_root()


@dataclass(frozen=True)
class ContractFailure:
    model: str
    stage: str
    error: str


def _task_for(spec: ModelSpec) -> SimpleNamespace:
    values: dict[str, int | str] = {
        "seq_len": 96,
        "label_len": 48,
        "pred_len": 12,
        "features": "M",
    }
    values.update(spec.contract_task)
    return SimpleNamespace(**values)


def _params_for(spec: ModelSpec) -> dict:
    config = tomllib.loads((ROOT / spec.config_path).read_text(encoding="utf-8"))
    return dict(config["model"].get("params", {}))


def contract_graph(nodes: int) -> np.ndarray:
    """Synthetic ``[nodes, nodes]`` ring graph with self-loops for ``requires_graph`` models.

    It stands in for the dataset adjacency that the runner injects in a real run; it
    is never used outside the executable contract.
    """
    graph = np.eye(nodes, dtype=np.float32)
    index = np.arange(nodes)
    graph[index, (index + 1) % nodes] = 1.0
    graph[index, (index - 1) % nodes] = 1.0
    return graph


def _construct_params(spec: ModelSpec, params: dict) -> dict:
    """Validated preset parameters plus the synthetic graph a ``requires_graph`` model needs."""
    validated = spec.validate_params(params)
    if spec.requires_graph:
        nodes = int(validated.get("num_nodes") or validated.get("enc_in"))
        validated["adj_mx"] = contract_graph(nodes)
    return validated


def _build_model(spec: ModelSpec, cfg, params: dict):
    """Construct through the ordinary or explicit local-artifact path."""
    if spec.requires_graph:
        return spec.factory(cfg, _construct_params(spec, params))
    if not spec.artifacts:
        return spec.build(cfg, params)
    from tsflab.catalog.model_artifacts import require_artifacts

    return spec.build_with_artifacts(cfg, params, require_artifacts(spec))


def _forward_contract(
    model,
    spec: ModelSpec,
    task,
    params: dict,
    *,
    backward: bool = False,
    batch: int = 2,
) -> torch.Tensor:
    channels = int(params.get("enc_in", params.get("num_nodes", 1)))
    # Parameter-free analytical baselines (for example Historical Last) still
    # have a valid differentiable training path through their input.  Track the
    # input during backward checks instead of forcing such methods to carry a
    # semantically false dummy parameter.
    x = torch.randn(
        batch, task.seq_len, channels, requires_grad=backward
    )
    x_mark = torch.zeros(batch, task.seq_len, 6)
    dec = torch.zeros(batch, task.label_len + task.pred_len, channels)
    dec_mark = torch.zeros(batch, task.label_len + task.pred_len, 6)
    model.eval()
    grad_context = contextlib.nullcontext() if backward else torch.no_grad()
    with grad_context:
        output = call_forecaster(model, x, x_mark, dec, dec_mark)
    if isinstance(output, tuple):
        output = output[0]
    if not torch.is_tensor(output):
        raise TypeError(f"forward returned {type(output).__name__}, expected Tensor")
    if output.shape[0] != batch or output.shape[1] != task.pred_len:
        raise ValueError(
            f"output shape {tuple(output.shape)} violates (B, pred_len, ...) contract"
        )
    if not torch.isfinite(output).all():
        raise ValueError("forward output contains NaN or Inf")
    output_type = getattr(model, "output_type", spec.output_type)
    if output_type != spec.output_type:
        raise ValueError(
            f"model output_type={output_type!r} disagrees with ModelSpec "
            f"output_type={spec.output_type!r}"
        )
    if output_type == "point" and output.ndim != 3:
        raise ValueError(f"point model returned rank-{output.ndim} output")
    if output_type == "quantile" and output.ndim != 4:
        raise ValueError(f"quantile model returned rank-{output.ndim} output")
    if output_type == "distribution":
        if output.ndim != 4 or output.shape[-1] != 2:
            raise ValueError(
                f"distribution model must return (B, T, C, 2), got {tuple(output.shape)}"
            )
    if backward and "inference-only" not in spec.capabilities:
        if spec.training_objective is not None:
            # Training goes through the declared objective (for example a
            # diffusion loss while ``forward`` samples without gradients), so
            # back-propagate its loss rather than the forward output.
            loss, x = _objective_loss(model, spec, task, channels, batch=batch)
            loss.backward()
        else:
            if not output.requires_grad:
                raise ValueError("forward output is detached; training cannot backpropagate")
            output.float().mean().backward()
        trainable_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        gradients = [
            parameter.grad
            for parameter in trainable_parameters
            if parameter.grad is not None
        ]
        if trainable_parameters and not gradients:
            raise ValueError("backward produced no parameter gradients")
        if not trainable_parameters:
            if x.grad is None:
                raise ValueError("parameter-free backward produced no input gradient")
            gradients = [x.grad]
        if not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise ValueError("backward produced NaN or Inf gradients")
    return output.detach()


class _SyntheticWindows(torch.utils.data.Dataset):
    """Stride-1 training windows over one random series with hourly raw marks.

    Items are ``(x, y, x_mark, y_mark)`` like the training datasets: ``y``
    covers ``label_len + pred_len`` steps from ``seq_len - label_len`` and
    marks are ``[year, month, day, weekday, hour, minute]``. ``data`` is the
    whole ``[T, C]`` series, as on the training split datasets.
    """

    def __init__(self, task, channels: int, windows: int) -> None:
        length = task.seq_len + task.pred_len + windows - 1
        self.data = torch.randn(length, channels).numpy()
        origin = datetime(2020, 1, 1)
        stamps = [origin + timedelta(hours=step) for step in range(length)]
        self.marks = torch.tensor(
            [[t.year, t.month, t.day, t.weekday(), t.hour, t.minute] for t in stamps],
            dtype=torch.float32,
        )
        self.series = torch.from_numpy(self.data)
        self.task, self.windows = task, windows

    def __len__(self) -> int:
        return self.windows

    def __getitem__(self, index: int):
        task = self.task
        y_start = index + task.seq_len - task.label_len
        y_stop = index + task.seq_len + task.pred_len
        return (
            self.series[index : index + task.seq_len],
            self.series[y_start:y_stop],
            self.marks[index : index + task.seq_len],
            self.marks[y_start:y_stop],
        )


def _objective_loss(
    model, spec: ModelSpec, task, channels: int, *, batch: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Loss of ``spec.training_objective`` on a synthetic training batch.

    Mirrors the start of training: ``training_setup`` (if declared) sees a
    synthetic training loader, then the objective runs in training mode with
    the MSE criterion on the loader's first batch. It must return a finite,
    differentiable scalar loss and, when it produces one, a forecast aligned
    with the target. Returns the loss and the (gradient-tracking) input.
    """
    loader = torch.utils.data.DataLoader(
        _SyntheticWindows(task, channels, windows=8 * batch), batch_size=batch, shuffle=False
    )
    model.train()
    if spec.training_setup is not None:
        spec.training_setup(model, loader, pred_len=task.pred_len, features=task.features)
    x, y, x_mark, y_mark = next(iter(loader))
    x = x.clone().requires_grad_(True)
    dec = torch.cat(
        (y[:, : task.label_len], torch.zeros(batch, task.pred_len, channels)), dim=1
    )
    training_batch = TrainingBatch(
        x=x, x_mark=x_mark, dec_inp=dec, y_mark=y_mark, y=y,
        pred_len=task.pred_len, features=task.features,
    )
    model.zero_grad(set_to_none=True)
    forecast, loss = spec.training_objective(model, training_batch, torch.nn.functional.mse_loss)
    if forecast is not None:
        aligned = training_batch.align(forecast)
        if aligned.shape != training_batch.target.shape:
            raise ValueError(
                f"training_objective forecast shape {tuple(aligned.shape)} "
                f"does not match target {tuple(training_batch.target.shape)}"
            )
    if not torch.is_tensor(loss) or loss.numel() != 1 or not torch.isfinite(loss):
        raise ValueError("training_objective must return a finite scalar tensor loss")
    if not loss.requires_grad:
        raise ValueError("training_objective loss is detached; training cannot backpropagate")
    return loss.float().reshape(()), x


def _state_dict_round_trip(
    model,
    spec: ModelSpec,
    cfg,
    task,
    params: dict,
) -> None:
    """Reload a serialized state into a fresh model and compare CPU output."""
    payload = io.BytesIO()
    torch.save(model.state_dict(), payload)
    payload.seek(0)
    restored = _build_model(spec, cfg, params)
    restored.load_state_dict(torch.load(payload, map_location="cpu", weights_only=True), strict=True)
    expected_state = model.state_dict()
    actual_state = restored.state_dict()
    if expected_state.keys() != actual_state.keys():
        raise ValueError("state-dict keys changed during round trip")
    for name in expected_state:
        torch.testing.assert_close(actual_state[name], expected_state[name], rtol=0, atol=0)

    torch.manual_seed(104729)
    expected = _forward_contract(model, spec, task, params, batch=1)
    torch.manual_seed(104729)
    actual = _forward_contract(restored, spec, task, params, batch=1)
    torch.testing.assert_close(actual, expected)


def audit_model_contracts(
    names: list[str] | None = None,
    *,
    forward: bool = False,
    backward: bool = False,
    strict: bool = False,
) -> list[ContractFailure]:
    """Construct selected models and optionally run their minimal forward pass."""
    failures: list[ContractFailure] = []
    torch.set_num_threads(1)
    for name in names or MODEL_CATALOG.names():
        stage = "load"
        try:
            spec = MODEL_CATALOG.get(name)
            task = _task_for(spec)
            # Factories receive the same explicit task/evaluation surfaces as
            # ordinary runs. Quantile models may resolve their levels from
            # evaluation when model.params does not override them.
            cfg = SimpleNamespace(task=task, evaluation=EvaluationConfig())
            params = _params_for(spec)
            stage = "construct"
            execute_forward = forward or backward or strict
            for seed in spec.contract_seeds if execute_forward else spec.contract_seeds[:1]:
                torch.manual_seed(seed)
                with contextlib.redirect_stdout(io.StringIO()):
                    model = _build_model(spec, cfg, params)
                if execute_forward:
                    stage = f"{'backward' if backward else 'forward'}(seed={seed})"
                    with contextlib.redirect_stdout(io.StringIO()):
                        _forward_contract(
                            model,
                            spec,
                            task,
                            spec.validate_params(params),
                            backward=backward or strict,
                        )
                    if strict:
                        stage = f"strict-round-trip(seed={seed})"
                        with contextlib.redirect_stdout(io.StringIO()):
                            _state_dict_round_trip(
                                model,
                                spec,
                                cfg,
                                task,
                                spec.validate_params(params),
                            )
                del model
        except Exception as exc:
            failures.append(ContractFailure(name, stage, f"{type(exc).__name__}: {exc}"))
        finally:
            gc.collect()
    return failures
