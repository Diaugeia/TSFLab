"""Model specification for DFDGCN."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.dfdgcn.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated DFDGCN parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset
    (see ``run_one.py``) and are therefore not declared here; ``adj_mx`` is required.
    """

    enc_in: int
    dropout: float = 0.3
    residual_channels: int = 16
    dilation_channels: int = 16
    skip_channels: int = 64
    end_channels: int = 128
    kernel_size: int = 2
    blocks: int = 2
    layers: int = 2
    a: float = 1.0
    fft_emb: int = 10
    identity_emb: int = 10
    hidden_emb: int = 30
    subgraph: int = 20
    steps_per_day: int = 288


def build_model(cfg, params):
    """Construct DFDGCN from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, num_nodes=params.get('num_nodes', params['enc_in']), adj_mx=params.get('adj_mx'), dropout=params.get('dropout', 0.3), residual_channels=params.get('residual_channels', 16), dilation_channels=params.get('dilation_channels', 16), skip_channels=params.get('skip_channels', 64), end_channels=params.get('end_channels', 128), kernel_size=params.get('kernel_size', 2), blocks=params.get('blocks', 2), layers=params.get('layers', 2), a=params.get('a', 1.0), fft_emb=params.get('fft_emb', 10), identity_emb=params.get('identity_emb', 10), hidden_emb=params.get('hidden_emb', 30), subgraph=params.get('subgraph', 20), steps_per_day=params.get('steps_per_day', 288))
    )


SPEC = ModelSpec(
    name='DFDGCN',
    module='tsflab.models.dfdgcn',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/DFDGCN.toml',
    model_card='src/tsflab/models/dfdgcn/README.md',
    smoke_config=None,
    capabilities=frozenset(['spatiotemporal']),
    components=('adaptive_node_embedding_adjacency', 'gated_dilated_conv', 'graph_utils', 'marks'),
    contract_task={'seq_len': 12, 'pred_len': 12, 'label_len': 0},
    requires_graph=True,
)
