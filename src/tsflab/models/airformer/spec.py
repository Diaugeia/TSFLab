"""Runtime specification for AirFormer."""
from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.airformer.model import Model
from pydantic import BaseModel, Field, model_validator


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    cov_dim: int = Field(default=2, gt=0)
    d_model: int = Field(default=32, gt=0)
    nhead: int = Field(default=4, gt=0)
    num_encoder_layers: int = Field(default=3, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_heads(self):
        if self.d_model % self.nhead:
            raise ValueError("d_model must be divisible by nhead")
        return self


def build_model(cfg, params):
    params.pop("num_nodes", None)
    return Model(cfg.task.seq_len, cfg.task.pred_len,
        dartboard_mx=params.pop("adj_mx", None), **params)


SPEC = ModelSpec(name="AirFormer", module="tsflab.models.airformer", model_class=Model,
    factory=build_model, params_schema=ModelParameterConfig,
    config_path="configs/models/AirFormer.toml", model_card="src/tsflab/models/airformer/README.md",
    capabilities=frozenset(["covariate"]), components=("marks",),
    contract_task={"seq_len": 24, "pred_len": 24, "label_len": 0}, requires_graph=True)
