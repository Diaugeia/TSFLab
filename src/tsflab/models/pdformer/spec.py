"""Runtime specification for PDFormer."""

import warnings

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.models.pdformer.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params`` (``adj_mx`` is runner-injected)."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(ge=1)
    embed_dim: int = Field(default=64, ge=2)
    skip_dim: int = Field(default=256, ge=1)
    lape_dim: int = Field(default=8, ge=1)
    geo_num_heads: int = Field(default=4, ge=1)
    sem_num_heads: int = Field(default=2, ge=1)
    t_num_heads: int = Field(default=2, ge=1)
    mlp_ratio: float = Field(default=4.0, gt=0.0)
    qkv_bias: bool = True
    drop: float = Field(default=0.0, ge=0.0, lt=1.0)
    attn_drop: float = Field(default=0.0, ge=0.0, lt=1.0)
    drop_path: float = Field(default=0.3, ge=0.0, lt=1.0)
    enc_depth: int = Field(default=6, ge=1)
    pre_norm: bool = True
    s_attn_size: int = Field(default=3, ge=1)
    far_mask_delta: int = Field(default=7, ge=1)
    dtw_delta: int = Field(default=5, ge=1)
    n_cluster: int = Field(default=16, ge=1)
    cluster_max_iter: int = Field(default=5, ge=1)
    cand_key_days: int = Field(default=14, ge=1)
    steps_per_day: int = Field(default=288, ge=1)
    add_time_in_day: bool = True
    add_day_in_week: bool = True
    random_flip: bool = True
    huber_delta: float = Field(default=2.0, gt=0.0)
    curriculum_step: int = Field(default=0, ge=0)


_FIELDS = tuple(ModelParameterConfig.model_fields)


def build_model(cfg, params):
    values = {name: params[name] for name in _FIELDS if name in params}
    return Model(
        seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len,
        adj_mx=params.get("adj_mx"), **values,
    )


def training_series(dataset, num_nodes: int) -> torch.Tensor | None:
    """The ``[T, N]`` value series spanned by the training windows behind ``dataset``, or ``None``."""
    data = getattr(dataset, "data", None)
    if data is None:
        return None
    values = torch.as_tensor(np.asarray(data), dtype=torch.float64)
    if values.ndim == 3:
        values = values[..., 0]
    idx = getattr(dataset, "idx", None)
    if idx is not None and len(idx):
        ends = np.asarray(idx).reshape(-1)
        start = max(int(ends.min()) - int(getattr(dataset, "seq_len", 1)) + 1, 0)
        values = values[start: int(ends.max()) + 1]
    if values.ndim != 2 or values.shape[1] != num_nodes or values.shape[0] < 2:
        return None
    return values


def training_setup(model, train_loader, *, pred_len, features):
    """Fit the DTW semantic mask and k-Shape pattern set on the training split only."""
    del pred_len, features
    series = training_series(getattr(train_loader, "dataset", None), model.num_nodes)
    if series is None:
        # Fallback: rebuild an ordered history from the newest step of every window.
        steps = [batch[0][:, -1, :] for batch in train_loader]
        series = torch.cat(steps).double() if steps else None
        warnings.warn("PDFormer: no ordered training series; fitting on window end points", stacklevel=2)
    if series is None or series.shape[0] < model.s_attn_size:
        warnings.warn("PDFormer: too little training data; semantic mask and patterns stay unfitted",
                      stacklevel=2)
        return
    model.fit_from_series(series)


def training_objective(model, batch: TrainingBatch, criterion):
    """Official Huber loss (delta ``huber_delta``) with optional horizon curriculum."""
    del criterion
    forecast = batch.forecast(model)
    horizon = model.curriculum_horizon()
    aligned = batch.align(forecast)[:, :horizon]
    loss = F.huber_loss(aligned, batch.target[:, :horizon], delta=model.huber_delta)
    return forecast, loss


SPEC = ModelSpec(
    name="PDFormer",
    module="tsflab.models.pdformer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/PDFormer.toml",
    model_card="src/tsflab/models/pdformer/README.md",
    smoke_config=None,
    capabilities=frozenset({"spatiotemporal"}),
    components=("adj_norm", "embed", "marks"),
    contract_task={"seq_len": 12, "pred_len": 12, "label_len": 0},
    requires_graph=True,
    training_objective=training_objective,
    training_setup=training_setup,
)
