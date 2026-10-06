"""Runtime specification for TimeGrad."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.models.timegrad.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    cell_type: Literal["LSTM", "GRU"] = "LSTM"
    num_layers: int = Field(default=2, gt=0)
    num_cells: int = Field(default=40, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    lags: list[int] = Field(default_factory=lambda: [1, 24], min_length=1)
    context_length: int = Field(default=0, ge=0)
    embed_dim: int = Field(default=1, gt=0)
    time_features: list[Literal["minute", "hour", "dayofweek", "month"]] = Field(
        default_factory=lambda: ["hour", "dayofweek"]
    )
    conditioning_length: int = Field(default=100, gt=0)
    diff_steps: int = Field(default=100, ge=1)
    beta_start: float = Field(default=1e-4, gt=0.0, lt=1.0)
    beta_end: float = Field(default=0.1, gt=0.0, lt=1.0)
    beta_schedule: Literal["linear", "quad"] = "linear"
    residual_layers: int = Field(default=8, gt=0)
    residual_channels: int = Field(default=8, gt=0)
    dilation_cycle_length: int = Field(default=2, gt=0)
    residual_hidden: int = Field(default=64, gt=0)
    step_emb_dim: int = Field(default=16, gt=0)
    scaling: bool = True
    context_loss: bool = True
    num_samples: int = Field(default=100, ge=2)
    sample_batch_size: int = Field(default=100, gt=0)
    quantile_levels: list[float] | None = None

    @model_validator(mode="after")
    def _check(self):
        if any(lag < 1 for lag in self.lags) or len(set(self.lags)) != len(self.lags):
            raise ValueError("lags must be distinct positive integers")
        if self.beta_start > self.beta_end:
            raise ValueError("beta_start must not exceed beta_end")
        return self


def build_model(cfg, params):
    options = dict(params)
    levels = options.pop("quantile_levels", None) or list(cfg.evaluation.quantile_levels)
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        features=cfg.task.features,
        quantile_levels=levels,
        **options,
    )


def training_objective(model, batch: TrainingBatch, criterion):
    """Replace the configured loss by the teacher-forced denoising loss (Algorithm 1).

    The diffusion is joint over every input channel, so the full future window is
    the target (the runner selects the ``MS`` target channel at evaluation). No
    forecast is produced by the same pass.
    """
    future = batch.y[:, -batch.pred_len:, :]
    return None, model.training_loss(batch.x, batch.x_mark, future, batch.y_mark)


SPEC = ModelSpec(
    name="TimeGrad",
    module="tsflab.models.timegrad",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/TimeGrad.toml",
    model_card="src/tsflab/models/timegrad/README.md",
    smoke_config="configs/runs/smoke_timegrad.toml",
    capabilities=frozenset({"quantile-output", "time-series"}),
    components=("ddpm_epsilon", "empirical_quantiles"),
    contract_task={"seq_len": 48, "pred_len": 12, "label_len": 0},
    training_objective=training_objective,
)
