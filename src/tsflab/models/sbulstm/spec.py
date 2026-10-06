"""Runtime specification for SBULSTM."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.sbulstm.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    layers: list[Literal["bidirectional", "unidirectional"]] = Field(
        default_factory=lambda: ["bidirectional", "unidirectional"], min_length=1
    )
    hidden_dim: int | None = Field(default=None, gt=0)
    imputation: bool = True
    imputation_weight: float = Field(default=1.0, ge=0.0)
    value_range: Literal["unit", "unbounded"] = "unit"


def build_model(cfg, params):
    # Graph runs inject ``num_nodes``/``adj_mx``; SBU-LSTM uses no graph.
    params = dict(params)
    num_nodes = params.pop("num_nodes", None)
    params.pop("adj_mx", None)
    options = ModelParameterConfig(**params).model_dump()
    if num_nodes is not None and int(num_nodes) != options["enc_in"]:
        raise ValueError(f"SBULSTM enc_in={options['enc_in']} does not match num_nodes={num_nodes}")
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **options)


SPEC = ModelSpec(
    name="SBULSTM",
    module="tsflab.models.sbulstm",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path="configs/models/SBULSTM.toml",
    model_card="src/tsflab/models/sbulstm/README.md",
    smoke_config="configs/runs/smoke_sbulstm.toml",
    capabilities=frozenset({"time-series", "spatiotemporal", "missing-values"}),
    components=(),
    contract_task={"seq_len": 12, "pred_len": 3, "label_len": 0},
)
