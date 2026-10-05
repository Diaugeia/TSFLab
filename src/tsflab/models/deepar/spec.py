"""Model specification for DeepAR."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.deepar.model import Model

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(ge=1)
    embedding_size: int = Field(default=32, ge=1)
    hidden_size: int = Field(default=64, ge=1)
    num_layers: int = Field(default=2, ge=1)
    cov_feat_size: int = Field(default=0, ge=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    checkpoint_steps: int = Field(default=24, ge=0)


def build_model(cfg, params):
    """Construct DeepAR from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, label_len=cfg.task.label_len, features=cfg.task.features, enc_in=params['enc_in'], embedding_size=params.get('embedding_size', 32), hidden_size=params.get('hidden_size', 64), num_layers=params.get('num_layers', 2), cov_feat_size=params.get('cov_feat_size', 0), dropout=params.get('dropout', 0.1), checkpoint_steps=params.get('checkpoint_steps', 24))
    )


SPEC = ModelSpec(
    name='DeepAR',
    module='tsflab.models.deepar',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/DeepAR.toml',
    model_card='src/tsflab/models/deepar/README.md',
    smoke_config=None,
    capabilities=frozenset(['distribution-output', 'time-series']),
    components=('gaussian_parameter_head',),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
