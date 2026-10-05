"""Model specification for XGBoostTS."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.xgboost_ts.model import MAX_DENSE_PARAMETERS, Model

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    num_estimators: int = Field(default=16, gt=0)
    tree_depth: int = Field(default=3, gt=0)
    learning_rate: float = Field(default=0.1, gt=0)
    column_fraction: float = Field(default=0.8, gt=0, le=1)
    l1_penalty: float = Field(default=0.0, ge=0)
    l2_penalty: float = Field(default=0.0001, ge=0)
    temperature: float = Field(default=1.0, gt=0)
    random_seed: int = 1741
    use_revin: bool = True
    max_dense_parameters: int = Field(default=MAX_DENSE_PARAMETERS, gt=0)


def build_model(cfg, params):
    """Construct XGBoostTS from a validated run configuration."""
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len,
                 enc_in=params['enc_in'], num_estimators=params.get('num_estimators', 16),
                 tree_depth=params.get('tree_depth', 3),
                 learning_rate=params.get('learning_rate', 0.1),
                 column_fraction=params.get('column_fraction', 0.8),
                 l1_penalty=params.get('l1_penalty', 0.0),
                 l2_penalty=params.get('l2_penalty', 0.0001),
                 temperature=params.get('temperature', 1.0),
                 random_seed=params.get('random_seed', 1741),
                 use_revin=bool(params.get('use_revin', True)),
                 max_dense_parameters=params.get('max_dense_parameters', MAX_DENSE_PARAMETERS))


SPEC = ModelSpec(
    name='XGBoostTS',
    module='tsflab.models.xgboost_ts',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/XGBoostTS.toml',
    model_card='src/tsflab/models/xgboost_ts/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
        components=('revin', 'soft_tree'),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
