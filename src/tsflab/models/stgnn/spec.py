"""Runtime specification for STGNN."""

from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.stgnn.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params`` (``adj_mx`` is runner-injected)."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(ge=1)
    hidden_dim: int = Field(default=64, ge=2)
    n_heads: int = Field(default=4, ge=1)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)


_FIELDS = tuple(ModelParameterConfig.model_fields)


def build_model(cfg, params):
    values = {name: params[name] for name in _FIELDS if name in params}
    return Model(
        seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len,
        adj_mx=params.get("adj_mx"), **values,
    )


SPEC = ModelSpec(
    name="STGNN",
    module="tsflab.models.stgnn",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/STGNN.toml",
    model_card="src/tsflab/models/stgnn/README.md",
    smoke_config=None,
    capabilities=frozenset({"spatiotemporal"}),
    components=("embed", "graph_conv_gru"),
    contract_task={"seq_len": 12, "pred_len": 12, "label_len": 0},
    requires_graph=True,
)
