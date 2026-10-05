"""Model specification for ETSformer."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.etsformer.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 128
    n_heads: int = 8
    e_layers: int = 2
    d_ff: int = 256
    top_k: int = 3
    dropout: float = 0.1
    activation: str = "sigmoid"


def build_model(cfg, params):
    """Construct ETSformer from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], d_model=params.get('d_model', 128), n_heads=params.get('n_heads', 8), e_layers=params.get('e_layers', 2), d_ff=params.get('d_ff', 256), top_k=params.get('top_k', 3), dropout=params.get('dropout', 0.1), activation=params.get('activation', 'sigmoid'))
    )


SPEC = ModelSpec(
    name='ETSformer',
    module='tsflab.models.etsformer',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/ETSformer.toml',
    model_card='src/tsflab/models/etsformer/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=(),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
