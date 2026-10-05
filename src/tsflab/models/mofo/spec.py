"""Model specification for MoFo."""

from __future__ import annotations

from typing import Literal

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.mofo.model import Model

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    """Validated MoFo parameters supplied via ``model.params``."""

    enc_in: int = Field(gt=0)
    d_model: int = Field(default=64, gt=0)
    periodic: int = Field(default=24, ge=2)
    head: int = Field(default=4, gt=0)
    d_layers: int = Field(default=1, gt=0)
    bias: Literal[0, 1] = 1
    cias: Literal[0, 1] = 1


def build_model(cfg, params):
    """Construct MoFo from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], d_model=params.get('d_model', 64), periodic=params.get('periodic', 24), head=params.get('head', 4), d_layers=params.get('d_layers', 1), bias=params.get('bias', 1), cias=params.get('cias', 1))
    )


SPEC = ModelSpec(
    name='MoFo',
    module='tsflab.models.mofo',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/MoFo.toml',
    model_card='src/tsflab/models/mofo/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('marks', 'revin'),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
