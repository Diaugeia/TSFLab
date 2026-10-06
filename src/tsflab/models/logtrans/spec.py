"""Runtime specification for LogTrans."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.models.logtrans.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    d_model: int = Field(default=64, gt=0)
    n_heads: int = Field(default=8, gt=0)
    e_layers: int = Field(default=3, gt=0)
    d_ff: int = Field(default=256, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    kernel_size: int = Field(default=6, gt=0)
    sparse: bool = True
    local_size: int = Field(default=0, ge=0)
    restart_len: int = Field(default=0, ge=0)
    embed_dim: int = Field(default=20, gt=0)
    use_marks: bool = True
    scaling: bool = True
    history_likelihood: bool = True

    @model_validator(mode="after")
    def _check(self):
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.restart_len and self.restart_len <= self.local_size + 1:
            raise ValueError("restart_len must exceed local_size + 1 (0 disables restarts)")
        return self


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        features=cfg.task.features,
        **params,
    )


def training_objective(model, batch: TrainingBatch, criterion):
    """Teacher-forced Gaussian NLL over the whole window (App. A.2).

    Replaces the configured ``nll_gaussian`` on the autoregressive forward output:
    the paper trains the one-step-ahead model on observed lags, and validation and
    test still score ``forward`` (mean-feedback decoding) with the configured loss.
    """
    future = batch.y[:, -batch.pred_len:, :]
    return None, model.training_loss(batch.x, batch.x_mark, future, batch.y_mark)


SPEC = ModelSpec(
    name="LogTrans",
    module="tsflab.models.logtrans",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/LogTrans.toml",
    model_card="src/tsflab/models/logtrans/README.md",
    smoke_config="configs/runs/smoke_logtrans.toml",
    capabilities=frozenset({"distribution-output", "time-series"}),
    components=("gaussian_parameter_head", "logsparse_conv_attention", "marks"),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
    training_objective=training_objective,
)
