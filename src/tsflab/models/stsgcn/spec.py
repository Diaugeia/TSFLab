"""Runtime specification for STSGCN."""

from typing import Literal

import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.models.stsgcn.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``.

    ``num_nodes`` and the road graph ``adj_mx`` are injected by the runner; the
    localized 3-step graph is built from ``adj_mx`` at construction.
    """

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(ge=1)
    num_layers: int = Field(default=4, ge=1)
    filters: list[int] = Field(default=[64, 64, 64], min_length=1)
    first_layer_embedding: int = Field(default=64, ge=1)
    activation: Literal["GLU", "relu"] = "GLU"
    individual: bool = True
    use_mask: bool = True
    temporal_emb: bool = True
    spatial_emb: bool = True
    output_hidden: int = Field(default=128, ge=1)
    init_magnitude: float = Field(default=0.0003, gt=0.0)
    huber_delta: float = Field(default=1.0, gt=0.0)


_FIELDS = tuple(name for name in ModelParameterConfig.model_fields if name != "enc_in")


def build_model(cfg, params):
    values = {name: params[name] for name in _FIELDS if name in params}
    if "filters" in values:
        values["filters"] = tuple(values["filters"])
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params.get("num_nodes", params["enc_in"]),
        adj_mx=params.get("adj_mx"),
        **values,
    )


def training_objective(model, batch: TrainingBatch, criterion):
    """Huber loss (paper Eq. 10; official ``huber_loss`` with ``rho = 1``) on the forecast."""
    del criterion
    forecast = batch.forecast(model)
    loss = F.huber_loss(batch.align(forecast), batch.target, delta=model.huber_delta)
    return forecast, loss


SPEC = ModelSpec(
    name="STSGCN",
    module="tsflab.models.stsgcn",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/STSGCN.toml",
    model_card="src/tsflab/models/stsgcn/README.md",
    smoke_config=None,
    capabilities=frozenset({"spatiotemporal"}),
    components=("synchronous_graph_conv",),
    contract_task={"seq_len": 12, "pred_len": 12, "label_len": 0},
    requires_graph=True,
    training_objective=training_objective,
)
