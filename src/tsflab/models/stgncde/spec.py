"""Runtime specification for STGNCDE."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.stgncde.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner; the adjacency is
    ignored because the graph is learned from node embeddings.
    """

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(ge=1)
    input_dim: int = Field(default=1, ge=1)
    hidden_dim: int = Field(default=64, ge=1)
    hidden_hidden_dim: int = Field(default=64, ge=1)
    num_layers: int = Field(default=2, ge=1)
    embed_dim: int = Field(default=10, ge=1)
    cheb_k: int = Field(default=2, ge=1)
    solver: Literal["rk4", "euler", "midpoint", "dopri5"] = "rk4"
    adjoint: bool = True


_FIELDS = tuple(name for name in ModelParameterConfig.model_fields if name != "enc_in")


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        num_nodes=params.get("num_nodes", params["enc_in"]),
        adj_mx=params.get("adj_mx"),
        **{name: params[name] for name in _FIELDS if name in params},
    )


SPEC = ModelSpec(
    name="STGNCDE",
    module="tsflab.models.stgncde",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/STGNCDE.toml",
    model_card="src/tsflab/models/stgncde/README.md",
    smoke_config="configs/runs/smoke_stgncde.toml",
    capabilities=frozenset({"spatiotemporal"}),
    components=("marks", "natural_cubic_spline", "node_adaptive_graph_conv"),
    contract_task={"seq_len": 12, "pred_len": 12, "label_len": 0},
)
