"""Runtime specification for GMAN."""

from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.gman.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``.

    ``num_nodes`` and the road graph ``adj_mx`` are injected by the runner; the
    node2vec spatial embedding is computed from ``adj_mx`` at construction.
    """

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(ge=1)
    num_blocks: int = Field(default=3, ge=1)
    num_heads: int = Field(default=8, ge=1)
    head_dim: int = Field(default=8, ge=1)
    steps_per_day: int = Field(default=288, ge=1)
    causal_temporal: bool = True
    output_dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    bn_momentum: float = Field(default=0.01, gt=0.0, le=1.0)
    se_dim: int = Field(default=64, ge=1)
    node2vec_p: float = Field(default=2.0, gt=0.0)
    node2vec_q: float = Field(default=1.0, gt=0.0)
    node2vec_walks: int = Field(default=100, ge=1)
    node2vec_walk_length: int = Field(default=80, ge=2)
    node2vec_window: int = Field(default=10, ge=1)
    node2vec_steps: int = Field(default=1000, ge=0)
    node2vec_seed: int = 0


_FIELDS = tuple(name for name in ModelParameterConfig.model_fields if name != "enc_in")


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params.get("num_nodes", params["enc_in"]),
        adj_mx=params.get("adj_mx"),
        **{name: params[name] for name in _FIELDS if name in params},
    )


SPEC = ModelSpec(
    name="GMAN",
    module="tsflab.models.gman",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/GMAN.toml",
    model_card="src/tsflab/models/gman/README.md",
    smoke_config=None,
    capabilities=frozenset({"spatiotemporal"}),
    components=("gated_fusion", "marks", "node2vec_embedding"),
    contract_task={"seq_len": 12, "pred_len": 12, "label_len": 0},
    requires_graph=True,
)
