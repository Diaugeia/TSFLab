"""Model specification for GradientBoostingTS."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.gradient_boosting_ts.model import MAX_DENSE_PARAMETERS, Model

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    num_estimators: int = Field(default=12, gt=0)
    tree_depth: int = Field(default=3, gt=0)
    learning_rate: float = Field(default=0.1, gt=0)
    temperature: float = Field(default=1.0, gt=0)
    use_revin: bool = True
    max_dense_parameters: int = Field(default=MAX_DENSE_PARAMETERS, gt=0)


def build_model(cfg, params):
    """Construct GradientBoostingTS from a validated run configuration."""
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len,
                 enc_in=params['enc_in'], num_estimators=params.get('num_estimators', 12),
                 tree_depth=params.get('tree_depth', 3),
                 learning_rate=params.get('learning_rate', 0.1),
                 temperature=params.get('temperature', 1.0),
                 use_revin=bool(params.get('use_revin', True)),
                 max_dense_parameters=params.get('max_dense_parameters', MAX_DENSE_PARAMETERS))


SPEC = ModelSpec(
    name='GradientBoostingTS',
    module='tsflab.models.gradient_boosting_ts',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/GradientBoostingTS.toml',
    model_card='src/tsflab/models/gradient_boosting_ts/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
        components=('revin', 'soft_tree'),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
