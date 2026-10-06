"""Runtime specification for TFT."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.tft.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    d_model: int = Field(default=160, gt=0)
    num_heads: int = Field(default=4, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    quantile_levels: list[float] | None = None

    @field_validator("quantile_levels")
    @classmethod
    def _quantile_levels(cls, values: list[float] | None) -> list[float] | None:
        if values is None:
            return values
        if not values or any(not 0.0 < value < 1.0 for value in values):
            raise ValueError("quantile_levels must be non-empty values in (0, 1)")
        if any(left >= right for left, right in zip(values, values[1:])):
            raise ValueError("quantile_levels must be strictly ascending")
        return values

    @model_validator(mode="after")
    def _heads(self) -> "ModelParameterConfig":
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be a multiple of num_heads")
        return self


def build_model(cfg, params):
    """Construct TFT from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        features=cfg.task.features,
        d_model=params["d_model"],
        num_heads=params["num_heads"],
        dropout=params["dropout"],
        quantile_levels=params.get("quantile_levels") or list(cfg.evaluation.quantile_levels),
    )


SPEC = ModelSpec(
    name="TFT",
    module="tsflab.models.tft",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/TFT.toml",
    model_card="src/tsflab/models/tft/README.md",
    smoke_config="configs/runs/smoke_tft.toml",
    capabilities=frozenset({"time-series", "quantile-output"}),
    components=("gated_residual_network", "interpretable_attention"),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)
