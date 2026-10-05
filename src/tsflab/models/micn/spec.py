"""Runtime specification for MICN."""
from typing import Literal

from pydantic import BaseModel, Field, field_validator
from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.micn.model import Model

class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0); d_model: int = Field(default=64, gt=0)
    d_layers: int = Field(default=1, gt=0); dropout: float = Field(default=0.05, ge=0, lt=1)
    conv_kernel: list[int] = Field(default_factory=lambda:[12,16], min_length=1)
    mode: Literal["regre", "mean"] = "regre"
    @field_validator("conv_kernel")
    @classmethod
    def scales(cls, value):
        if min(value) < 2 or len(set(value)) != len(value): raise ValueError("scales must be distinct integers >= 2")
        return value

def build_model(cfg, params): return Model(cfg.task.seq_len, cfg.task.pred_len, **params)
SPEC = ModelSpec(name="MICN", module="tsflab.models.micn", model_class=Model,
    factory=build_model, params_schema=ModelParameterConfig, config_path="configs/models/MICN.toml",
    model_card="src/tsflab/models/micn/README.md", smoke_config=None,
    capabilities=frozenset(["time-series"]), components=("embed", "series_decomposition"),
    contract_task={"seq_len":96,"pred_len":96,"label_len":0})
