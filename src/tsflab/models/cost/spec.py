"""Runtime specification for CoST."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.cost.model import RIDGE_ALPHAS, Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    repr_dims: int = Field(default=320, gt=0)
    hidden_dims: int = Field(default=64, gt=0)
    depth: int = Field(default=10, gt=0)
    kernels: list[int] = Field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64, 128], min_length=1)
    alpha: float = Field(default=5e-4, ge=0.0)
    queue_size: int = Field(default=256, gt=0)
    momentum: float = Field(default=0.999, ge=0.0, lt=1.0)
    temperature: float = Field(default=0.07, gt=0.0)
    sigma: float = Field(default=0.5, ge=0.0)
    aug_prob: float = Field(default=0.5, ge=0.0, le=1.0)
    pretrain_iters: int = Field(default=0, ge=0)
    pretrain_batch_size: int = Field(default=128, gt=0)
    pretrain_lr: float = Field(default=1e-3, gt=0.0)
    use_marks: bool = True
    channel_independent: bool = False
    ridge_alphas: list[float] = Field(default_factory=lambda: list(RIDGE_ALPHAS), min_length=1)
    ridge_valid_fraction: float = Field(default=0.25, gt=0.0, lt=1.0)
    ridge_max_samples: int = Field(default=100000, gt=0)

    @model_validator(mode="after")
    def _check(self):
        if self.repr_dims % 2:
            raise ValueError("repr_dims must be even")
        if any(k < 1 for k in self.kernels):
            raise ValueError("kernels must be positive")
        if any(a <= 0 for a in self.ridge_alphas):
            raise ValueError("ridge_alphas must be positive")
        return self


def build_model(cfg, params):
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, features=cfg.task.features, **params)


SPEC = ModelSpec(
    name="CoST",
    module="tsflab.models.cost",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/CoST.toml",
    model_card="src/tsflab/models/cost/README.md",
    smoke_config="configs/runs/smoke_cost.toml",
    capabilities=frozenset({"time-series", "pretraining-stage"}),
    components=("dilated_conv_encoder", "marks"),
    contract_task={"seq_len": 96, "pred_len": 24, "label_len": 0},
)
