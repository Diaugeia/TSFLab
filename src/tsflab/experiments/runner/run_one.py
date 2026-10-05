"""Single-run orchestration: data, model, training, evaluation."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import torch

from tsflab.experiments.evaluation import profile_model
from tsflab.experiments.evaluation.profile import parse_profile_report_file
from tsflab.catalog.registry import MODEL_CATALOG
from tsflab.experiments.runner.callbacks import build_callbacks
from tsflab.experiments.runner.evaluator import evaluate, evaluate_rolling
from tsflab.experiments.runner.trainer import TrainResult, train
from tsflab.experiments.utils import default_summary_row, set_seed, write_csv_summary
from tsflab.experiments.utils.results import _flatten_params
from tsflab.data.provider import build_data_loader


def _normalize_adj(adj, scheme: str):
    """Apply an optional adjacency normalization to a data-derived adj matrix.

    ``scheme`` selects a function from ``tsflab.models._components.adj_norm``. The raw
    adjacency is returned untouched when ``scheme`` is falsy. Existing graph
    models that build their own normalization are unaffected because this is
    only invoked when ``dataset.params.adj_norm`` is explicitly set.
    """
    from tsflab.models._components import adj_norm as _an

    schemes = {
        "sym_norm_lap": _an.symmetric_normalized_laplacian,
        "symmetric_normalized_laplacian": _an.symmetric_normalized_laplacian,
        "scaled_laplacian": _an.lambda_rescaled_laplacian,
        "gcn": _an.gcn_norm,
        "gcn_norm": _an.gcn_norm,
        "transition": _an.transition_matrix,
        "transition_matrix": _an.transition_matrix,
        "reverse_transition": _an.reverse_transition_matrix,
        "reverse_transition_matrix": _an.reverse_transition_matrix,
    }
    key = str(scheme).lower()
    if key not in schemes:
        raise ValueError(
            f"unknown adj_norm scheme {scheme!r}; expected one of {sorted(schemes)}"
        )
    return schemes[key](adj)


@dataclass
class RunResult:
    """Aggregate results from a single training/evaluation run.

    Parameters
    ----------
    metrics : dict[str, float]
        Evaluation metrics computed on the test split.
    train_time_sec : float
        Training wall-clock time in seconds.
    test_time_sec : float
        Evaluation wall-clock time in seconds.
    checkpoint_path : str
        Path to the best checkpoint on disk.
    run_id : str
        Unique identifier for the run.
    """

    metrics: dict[str, float]
    train_time_sec: float
    test_time_sec: float
    checkpoint_path: str
    run_id: str


def _build_device(runtime) -> torch.device:
    """Resolve the compute device from runtime settings.

    Parameters
    ----------
    runtime : ExperimentRuntimeConfig
        Runtime config with device selection options.

    Returns
    -------
    torch.device
        Resolved device.
    """
    if runtime.device == "cuda" and torch.cuda.is_available():
        if os.environ.get("TSFLAB_ASSIGNED_GPUS"):
            return torch.device("cuda:0")
        if runtime.use_multi_gpu:
            return torch.device("cuda")
        return torch.device(f"cuda:{runtime.device_ids[0]}")
    if (
        runtime.device == "mps"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def _build_loaders(config):
    """Build the train/val/test dataset+loader pairs for one run.

    Returns
    -------
    tuple
        ``(train_set, train_loader, vali_set, vali_loader, test_set,
        test_loader, adj_norm)``. ``adj_norm`` is the popped-out adjacency
        normalization hint (see ``_normalize_adj``), threaded through to
        ``_build_model`` since it applies to a model-construction input
        (``adj_mx``), not the dataset constructor.
    """
    dataset_registry_name = config.dataset.name
    from tsflab.catalog.registry.datasets import DATASET_REGISTRY

    dataset_spec = DATASET_REGISTRY.get(dataset_registry_name)
    root_path, data_file = dataset_spec.resolve_location(
        config.dataset.path, config.dataset.id
    )

    size = (config.task.seq_len, config.task.label_len, config.task.pred_len)
    if hasattr(config.dataset.params, "model_dump"):
        dataset_params = config.dataset.params.model_dump()
    else:
        dataset_params = dict(config.dataset.params)

    # Optional adjacency normalization scheme. This is a run-time post-processing
    # hint, not a dataset constructor argument, so pop it out before the params
    # are unpacked into the dataset. Default (None) leaves the raw adjacency
    # untouched. See models._components.adj_norm for the available schemes.
    adj_norm = dataset_params.pop("adj_norm", None)

    def _loader_for(flag: str):
        return build_data_loader(
            dataset_registry_name,
            root_path,
            data_file,
            size,
            flag,
            config.task.features,
            dataset_params,
            config.training.batch_size,
            config.experiment.runtime.num_workers,
        )

    train_set, train_loader = _loader_for("train")
    vali_set, vali_loader = _loader_for("val")
    test_set, test_loader = _loader_for("test")
    return train_set, train_loader, vali_set, vali_loader, test_set, test_loader, adj_norm


def _build_model(config, train_set, adj_norm, device: torch.device):
    """Construct the model for one run and validate its output_type/loss pairing.

    Wraps: model-parameter schema validation, data-derived graph structure
    injection (``adj_mx``/``num_nodes``), explicit probabilistic configuration
    resolution by model factories, and the output_type/loss fail-fast
    check (a quantile or distribution model trained with a mismatched loss
    silently produces nonsense, so this must run before training starts).

    Returns
    -------
    tuple[torch.nn.Module, str]
        The constructed model (already moved to ``device``) and its
        ``output_type`` (``"point"`` when the model doesn't declare one).
    """
    spec = MODEL_CATALOG.get(config.model.name)
    params = spec.validate_params(config.model.params)

    # Pretrained weights/tokenizers are never downloaded implicitly. Required
    # artifacts must already be present and checksum-verified before a factory
    # is allowed to construct the model.
    artifact_paths = {}
    if spec.artifacts:
        from tsflab.catalog.model_artifacts import require_artifacts

        artifact_paths = require_artifacts(spec)

    # Inject data-derived graph structure for spatiotemporal / graph models.
    # Datasets that expose an adjacency matrix (e.g. cauair_st, traffic) make it
    # available here so graph model factories can read params["adj_mx"] /
    # params["num_nodes"]. Non-graph datasets/models simply ignore these.
    adj_mx = getattr(train_set, "adj_mx", None)
    if adj_mx is not None:
        if adj_norm is not None:
            adj_mx = _normalize_adj(adj_mx, adj_norm)
        params["adj_mx"] = adj_mx
    elif spec.requires_graph:
        raise ValueError(
            f"model {spec.name!r} needs the dataset adjacency (adj_mx), but dataset "
            f"{config.dataset.name!r} provides none; use a graph dataset that ships adj_mx.npy"
        )
    num_nodes = getattr(train_set, "num_nodes", None)
    if num_nodes is not None:
        params.setdefault("num_nodes", num_nodes)

    if artifact_paths:
        if spec.artifact_factory is None:
            raise ValueError(
                f"model {spec.name!r} has verified artifacts but no artifact-aware factory"
            )
        model = spec.artifact_factory(config, params, artifact_paths).to(device)
    else:
        model = spec.factory(config, params).to(device)

    pretraining = "pretraining-stage" in spec.capabilities
    if pretraining != callable(getattr(model, "pretrain", None)):
        raise ValueError(
            f"model {spec.name!r} pretraining-stage capability and pretrain() method disagree"
        )

    # Probabilistic output/loss compatibility check. A quantile or
    # distribution model trained with a mismatched loss silently produces
    # nonsense (e.g. an MSE loss backprop'd through raw quantile channels),
    # so fail fast here rather than let it surface as a confusing metric
    # later. Point models are unrestricted (any loss is valid).
    output_type = spec.output_type
    model_output_type = getattr(model, "output_type", output_type)
    if model_output_type != output_type:
        raise ValueError(
            f"model {spec.name!r} output_type={model_output_type!r} "
            f"disagrees with spec={output_type!r}"
        )
    required_loss_by_output_type = {
        "point": None,
        "quantile": "quantile",
        "distribution": "nll_gaussian",
    }
    if output_type not in required_loss_by_output_type:
        raise ValueError(
            f"model {config.model.name!r} declares unknown output_type="
            f"{output_type!r}; expected one of "
            f"{sorted(required_loss_by_output_type)}"
        )
    required_loss = required_loss_by_output_type[output_type]
    loss_by_required_output_type = {"quantile": "quantile", "nll_gaussian": "distribution"}
    configured_loss = config.training.loss.lower()
    inference_only = "inference-only" in spec.capabilities
    if not inference_only and required_loss is not None and configured_loss != required_loss:
        raise ValueError(
            f"model {config.model.name!r} declares output_type={output_type!r}, "
            f"which requires training.loss={required_loss!r}, but the config "
            f"sets training.loss={config.training.loss!r}"
        )
    if (
        not inference_only
        and required_loss is None
        and configured_loss in loss_by_required_output_type
    ):
        raise ValueError(
            f"training.loss={config.training.loss!r} requires a model with "
            f"output_type={loss_by_required_output_type[configured_loss]!r}, but "
            f"model {config.model.name!r} declares output_type={output_type!r}"
        )

    return model, output_type


def _write_run_outputs(
    config,
    model,
    run_id: str,
    model_dir: str,
    dataset_name: str,
    device: torch.device,
    test_loader,
    metrics: dict,
    test_time: float,
    train_result,
    eval_strategy: str,
    raw: dict,
    sweep_keys: list[str] | None,
) -> None:
    """Write the performance CSV row, run record JSON, and optional profile CSV."""
    summary_path = os.path.join(model_dir, "performance.csv")
    print(f"Writing CSV summary to: {summary_path}")
    summary_row = default_summary_row(
        {
            "run_id": run_id,
            "dataset": dataset_name,
            "model": config.model.name,
            "seq_len": config.task.seq_len,
            "pred_len": config.task.pred_len,
            "seed": config.experiment.random_seed,
            "train_time_sec": train_result.train_time_sec,
            "test_time_sec": test_time,
            "fit_time": train_result.train_time_sec,
            "inference_time": test_time,
        },
        metrics,
        raw=raw,
        sweep_keys=sweep_keys,
    )
    # Record the evaluation strategy only when it diverges from the historical
    # default. This keeps the fixed-path CSV header byte-identical to before
    # while making rolling runs self-describing.
    if eval_strategy != "fixed":
        summary_row["eval_strategy"] = eval_strategy
    from tsflab.experiments.infra.fingerprints import dataset_fingerprint
    from tsflab.experiments.infra.storage import canonical_hash
    snapshot = config.model_dump(mode="json")
    from tsflab.experiments.infra.comparison import protocol_fingerprint
    summary_row["protocol_sha256"] = protocol_fingerprint(snapshot, dataset_fingerprint(config))
    summary_row["model_variant"] = canonical_hash(snapshot["model"])
    write_csv_summary(summary_path, summary_row)

    # Self-describing, schema-validated record.json (one per run) for tsf result submit
    # / TSFLab Leaderboard ingestion. Invalid artifacts fail closed. Imported lazily to avoid
    # import-order coupling with benchmark.utils package init.
    from tsflab.experiments.utils.record import write_run_record

    record_path = os.path.join(model_dir, "records", f"{run_id}.json")
    write_run_record(
        record_path=record_path,
        config=config,
        device=device,
        run_id=run_id,
        dataset_id=dataset_name,
        metrics=metrics,
        fit_time=train_result.train_time_sec,
        inference_time=test_time,
        repo_root=None,
    )

    if config.evaluation.enable_profile:
        os.makedirs(os.path.join(model_dir, "profiles"), exist_ok=True)
        profile_path = os.path.join(model_dir, "profiles", f"{run_id}.txt")
        profile_model(
            model=model,
            data_loader=test_loader,
            device=device,
            label_len=config.task.label_len,
            pred_len=config.task.pred_len,
            save_path=profile_path,
        )
        profile_metrics = parse_profile_report_file(profile_path)
        profile_row = {
            "run_id": run_id,
            "model": config.model.name,
            "dataset": dataset_name,
            "seq_len": config.task.seq_len,
            "pred_len": config.task.pred_len,
            "seed": config.experiment.random_seed,
            "train_time_sec": train_result.train_time_sec,
            "test_time_sec": test_time,
        }
        profile_row.update(profile_metrics)
        profile_header = [
            "run_id",
            "model",
            "dataset",
            "seq_len",
            "pred_len",
            "seed",
            "train_time_sec",
            "test_time_sec",
            "total_params",
            "trainable_params",
            "non_trainable_params",
            "total_mult_adds_mb",
            "total_macs_m",
            "dynamic_vram_mb",
            "peak_vram_mb",
            "reserved_vram_mb",
            "latency_avg_ms",
            "throughput_samples_sec",
        ]
        profile_csv_path = os.path.join(model_dir, "profile.csv")
        write_csv_summary(profile_csv_path, profile_row, header=profile_header)


def _run_one(
    config,
    raw: dict,
    sweep_keys: list[str] | None = None,
    session=None,
) -> RunResult:
    """Execute a full training/evaluation run for one config.

    Parameters
    ----------
    config : RootConfig
        Validated configuration object.
    raw : dict
        Raw expanded config dictionary (used for sweep columns).
    sweep_keys : list[str] | None, optional
        Dot-delimited keys from the sweep section.
    Returns
    -------
    RunResult
        Metrics and artifact paths for the run.
    """
    dataset_name = config.dataset.alias or config.dataset.name

    if raw and sweep_keys:
        flattened = _flatten_params(raw)
        sweep_parts = [
            f"{key}={flattened[key]}" for key in sweep_keys if key in flattened
        ]
    else:
        sweep_parts = []

    summary_parts = [
        f"model={config.model.name}",
        f"dataset={dataset_name}",
        f"mode={config.task.mode}",
        f"seq_len={config.task.seq_len}",
        f"pred_len={config.task.pred_len}",
        f"seed={config.experiment.random_seed}",
    ]
    if sweep_parts:
        summary_parts.append(f"sweep: {', '.join(sweep_parts)}")
    print(f"Run config | {' | '.join(summary_parts)}")

    session.stage("data")
    set_seed(config.experiment.random_seed)
    device = _build_device(config.experiment.runtime)
    print(f"Using device: {device}")

    (
        train_set,
        train_loader,
        vali_set,
        vali_loader,
        test_set,
        test_loader,
        adj_norm,
    ) = _build_loaders(config)

    session.stage("construction")
    model, output_type = _build_model(config, train_set, adj_norm, device)

    # Optional model-side pretraining stage. Used by two-stage models such as
    # LatentTSF to pretrain + freeze an autoencoder before the forecaster is
    # trained. Run on the raw module before any DataParallel wrap and add the
    # wall time to fit_time/train_time_sec for fair benchmark accounting.
    pretrain_time_sec = 0.0
    spec = MODEL_CATALOG.get(config.model.name)
    if "pretraining-stage" in spec.capabilities:
        session.stage("pretraining")
        _pretrain_start = time.perf_counter()
        import inspect
        options = {}
        if "stage_runner" in inspect.signature(model.pretrain).parameters:
            from functools import partial
            from tsflab.experiments.infra.stages import reconstruction_stage
            options["stage_runner"] = partial(reconstruction_stage,
                checkpoint=session.directory / "checkpoints" / "pretraining.pth",
                every_batches=session.policy.recovery.checkpoint_every_batches)
        model.pretrain(train_loader, device, **options)
        pretrain_time_sec = time.perf_counter() - _pretrain_start

    inference_only = "inference-only" in spec.capabilities
    if inference_only and config.experiment.runtime.use_multi_gpu:
        raise ValueError(
            "inference-only models use the official runtime's device placement; "
            "disable TSFLab use_multi_gpu"
        )
    if config.experiment.runtime.use_multi_gpu and device.type == "cuda":
        model = torch.nn.DataParallel(
            model, device_ids=(list(range(int(os.environ["TSFLAB_ASSIGNED_GPUS"]))) if os.environ.get("TSFLAB_ASSIGNED_GPUS") else config.experiment.runtime.device_ids)
        )

    run_id = session.run_id
    output_group = os.path.join(dataset_name, config.model.name)
    model_dir = os.path.join(config.experiment.work_dir, output_group)
    checkpoint_dir = str(session.directory / "checkpoints")
    session.stage("training")

    if inference_only:
        print("Training skipped: model declares the inference-only capability")
        train_result = TrainResult(best_model_path="", train_time_sec=0.0)
    else:
        os.makedirs(checkpoint_dir, exist_ok=True)
        optimizer_cls = getattr(torch.optim, config.training.optimizer.name)
        optimizer_kwargs = {
            "lr": config.training.optimizer.lr,
            "weight_decay": config.training.optimizer.weight_decay,
        }
        optimizer_kwargs.update(config.training.optimizer.params)
        optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)

        # Build optional training-trick callbacks. When the [training.tricks]
        # section is omitted (the default) this returns an empty list.
        callbacks = build_callbacks(getattr(config.training, "tricks", None))

        # Quantile pinball loss consumes the evaluation quantile levels unless
        # a training-only override was explicitly supplied.
        loss_params = dict(config.training.loss_params)
        if (
            config.training.loss.lower() == "quantile"
            and "quantile_levels" not in loss_params
        ):
            loss_params["quantile_levels"] = list(
                config.evaluation.quantile_levels
            )

        train_result = train(
            model=model,
            train_loader=train_loader,
            vali_loader=vali_loader,
            device=device,
            epochs=config.training.epochs,
            patience=config.training.patience,
            loss_name=config.training.loss,
            loss_params=loss_params,
            optimizer=optimizer,
            lradj=config.training.optimizer.lradj,
            base_lr=config.training.optimizer.lr,
            total_epochs=config.training.epochs,
            label_len=config.task.label_len,
            pred_len=config.task.pred_len,
            features=config.task.features,
            use_amp=config.experiment.runtime.amp,
            checkpoint_dir=checkpoint_dir,
            checkpoint_cfg=config.training.checkpoint,
            callbacks=callbacks,
            training_objective=spec.training_objective,
            training_setup=spec.training_setup,
            resume=session.resume,
            tracker=session.tracker,
            checkpoint_every_batches=session.policy.recovery.checkpoint_every_batches,
        )
        train_result.train_time_sec += pretrain_time_sec

    session.stage("evaluation")
    eval_strategy = getattr(config.evaluation, "strategy", "fixed")
    if eval_strategy == "rolling":
        rolling_cfg = config.evaluation.rolling
        print(
            "Evaluation strategy: rolling "
            f"(horizon={rolling_cfg.horizon}, stride={rolling_cfg.stride}, "
            f"num_rollings={rolling_cfg.num_rollings})"
        )
        metrics, test_time = evaluate_rolling(
            model=model,
            dataset=test_set,
            device=device,
            seq_len=config.task.seq_len,
            label_len=config.task.label_len,
            pred_len=config.task.pred_len,
            features=config.task.features,
            inverse=config.task.inverse,
            horizon=rolling_cfg.horizon,
            stride=rolling_cfg.stride,
            num_rollings=rolling_cfg.num_rollings,
            quantile_levels=config.evaluation.quantile_levels,
        )
    else:
        metrics, test_time = evaluate(
            model=model,
            data_loader=test_loader,
            device=device,
            label_len=config.task.label_len,
            pred_len=config.task.pred_len,
            features=config.task.features,
            inverse=config.task.inverse,
            dataset=test_set,
            quantile_levels=config.evaluation.quantile_levels,
        )

    if config.evaluation.metrics:
        # Probabilistic runs always score crps/wql/coverage_80/width_80
        # (collect_prob_metrics), so keep them even when evaluation.metrics
        # was left at its point-only default and doesn't name them — otherwise
        # a probabilistic model's uncertainty metrics are silently dropped
        # from performance.csv.
        keep = set(config.evaluation.metrics)
        if output_type != "point":
            keep |= {"crps", "wql", "coverage_80", "width_80"}
        metrics = {k: v for k, v in metrics.items() if k in keep}

    metrics_str = ", ".join(f"{k}:{v:.4f}" for k, v in metrics.items())
    print(f"Test metrics | {metrics_str}")

    session.stage("outputs")
    session.tracker.log({f"test/{key}": value for key, value in metrics.items()}, config.training.epochs + 1)
    _write_run_outputs(
        config=config,
        model=model,
        run_id=run_id,
        model_dir=model_dir,
        dataset_name=dataset_name,
        device=device,
        test_loader=test_loader,
        metrics=metrics,
        test_time=test_time,
        train_result=train_result,
        eval_strategy=eval_strategy,
        raw=raw,
        sweep_keys=sweep_keys,
    )

    from tsflab.experiments.infra.predictions import log_predictions
    log_predictions(model, test_loader, config, device, session.tracker)

    return RunResult(
        metrics=metrics,
        train_time_sec=train_result.train_time_sec,
        test_time_sec=test_time,
        checkpoint_path=train_result.best_model_path,
        run_id=run_id,
    )


def run_one(config, raw: dict, sweep_keys: list[str] | None = None) -> RunResult:
    """Execute a run with evidence from preflight through success or failure."""
    from tsflab.experiments.infra.runs import RunSession
    from tsflab.experiments.infra.environment import validate_experiment
    with RunSession(config, raw, sweep_keys) as session:
        validate_experiment(config)
        result = _run_one(config, raw, sweep_keys, session=session)
        session.result(result)
        return result
