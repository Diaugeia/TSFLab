"""Runtime specification for Air-DualODE."""
from typing import Literal
from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.airdualode.model import Model
from pydantic import BaseModel, Field, model_validator


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    cov_dim: int = Field(default=2, gt=0)
    phy_latent_dim: int = Field(default=16, gt=0)
    unk_latent_dim: int = Field(default=16, gt=0)
    gcn_hidden_dim: int = Field(default=32, gt=0)
    n_heads: int = Field(default=4, gt=0)
    ode_method: Literal["euler", "rk4"] = "euler"

    @model_validator(mode="after")
    def validate_heads(self):
        if self.unk_latent_dim % self.n_heads:
            raise ValueError("unk_latent_dim must be divisible by n_heads")
        return self


def build_model(cfg, params):
    params.pop("num_nodes", None)
    return Model(cfg.task.seq_len, cfg.task.pred_len,
        adj_mx=params.pop("adj_mx", None), flow_mx=params.pop("flow_mx", None), **params)


SPEC = ModelSpec(name="AirDualODE", module="tsflab.models.airdualode", model_class=Model,
    factory=build_model, params_schema=ModelParameterConfig,
    config_path="configs/models/AirDualODE.toml", model_card="src/tsflab/models/airdualode/README.md",
    capabilities=frozenset(["covariate"]), components=("marks",),
    contract_task={"seq_len": 24, "pred_len": 24, "label_len": 0}, requires_graph=True)
