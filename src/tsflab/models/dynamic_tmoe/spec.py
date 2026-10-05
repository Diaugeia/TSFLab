"""Model specification for DynamicTMoE."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.dynamic_tmoe.model import Model
from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    d_model: int = Field(default=64, gt=0)
    patch_len: int = Field(default=16, gt=0)
    stride: int = Field(default=8, gt=0)
    top_k: int = Field(default=3, gt=0, le=5)
    memory_slots: int = Field(default=4, gt=0)
    relation_period: int = Field(default=24, gt=0)
    routing_floor: float = Field(default=1e-4, ge=0, lt=1)
    use_revin: bool = True


def build_model(cfg, params):
    return Model(
        cfg.task.seq_len,
        cfg.task.pred_len,
        params["enc_in"],
        params.get("d_model", 64),
        params.get("patch_len", 16),
        params.get("stride", 8),
        params.get("top_k", 3),
        params.get("memory_slots", 4),
        params.get("relation_period", 24),
        params.get("routing_floor", 1e-4),
        bool(params.get("use_revin", True)),
    )


SPEC = ModelSpec(
    name="DynamicTMoE",
    module="tsflab.models.dynamic_tmoe",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/DynamicTMoE.toml",
    model_card="src/tsflab/models/dynamic_tmoe/README.md",
    capabilities=frozenset(["time-series"]),
    components=("marks", "revin", "topk_expert_router"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
