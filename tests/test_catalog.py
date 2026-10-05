"""tsflab.catalog: registry, components catalog, contracts runner, artifacts, similarity, characteristics, match, and admission (fakes only)."""

from __future__ import annotations

import hashlib
import importlib
import subprocess
import sys
import tempfile
import textwrap
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pydantic import BaseModel
from torch import nn

from tsflab.catalog.cards.metadata import model_records
from tsflab.catalog.characteristics import (
    TASK_TERMS,
    data_terms,
    kind_of,
    term_problems,
    vocabulary,
)
from tsflab.catalog.model_artifacts import (
    artifact_status,
    fetch_artifact,
    require_artifacts,
)
from tsflab.catalog.model_contracts import _forward_contract
from tsflab.catalog.registry.models import MODEL_CATALOG, ModelArtifact, ModelSpec
from tsflab.catalog.similarity import clusters, collect_units, similar_pairs
from tsflab.data.profile import load_rules
from tsflab.models._foundation import (
    ChronosRuntime,
    FoundationForecast,
    FoundationModel,
    FoundationSource,
    MoiraiRuntime,
    TimesFMRuntime,
)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


ROOT = Path(__file__).resolve().parents[1]


class ModelRegistryTests(unittest.TestCase):
    def test_model_catalog_listing_does_not_import_model_runtimes(self) -> None:
        script = """
import importlib.abc
import io
import json
import sys
from contextlib import redirect_stdout

class BlockModelRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"torch", "numpy"} or fullname.startswith(("torch.", "numpy.")):
            raise ModuleNotFoundError(f"{fullname} intentionally unavailable")
        return None

sys.meta_path.insert(0, BlockModelRuntime())
from tsflab.cli.main import main
from tsflab.catalog.registry.models import MODEL_CATALOG
from tsflab.core.paths import repository_root
expected = len([p for p in (repository_root() / 'src' / 'tsflab' / 'models').glob('*/spec.py') if not p.parent.name.startswith('_')])
assert len(MODEL_CATALOG.names()) == expected
output = io.StringIO()
with redirect_stdout(output):
    assert main(["catalog", "list", "--kind", "model", "--json"]) == 0
assert len(json.loads(output.getvalue())) == expected
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_catalog_metadata_uses_registration_as_admission_boundary(self) -> None:
        records = model_records(ROOT, refs={"Linear": "tsflab.models.linear.spec"})
        self.assertEqual([record["name"] for record in records], ["Linear"])

    def test_inference_only_capability_rejects_training_hooks(self) -> None:
        spec = MODEL_CATALOG.get("Linear")
        with self.assertRaisesRegex(ValueError, "pretraining-stage"):
            replace(
                spec,
                capabilities=spec.capabilities | {"inference-only", "pretraining-stage"},
            )
        with self.assertRaisesRegex(ValueError, "training objective"):
            replace(
                spec,
                capabilities=spec.capabilities | {"inference-only"},
                training_objective=lambda *_args: None,
            )


# ---------------------------------------------------------------------------
# Pinned model artifacts
# ---------------------------------------------------------------------------


class _Params(BaseModel):
    width: int = 1


def _spec(source: Path, digest: str) -> ModelSpec:
    artifact = ModelArtifact(
        name="weights",
        url=source.as_uri(),
        revision="commit-123",
        sha256=digest,
        filename="weights.bin",
        required=True,
    )
    return ModelSpec(
        name="Fixture",
        module="tsflab.models.fixture",
        model_class=nn.Identity,
        factory=lambda cfg, params: nn.Identity(),
        params_schema=_Params,
        capabilities=frozenset({"time-series"}),
        artifacts=(artifact,),
    )


def test_fetch_artifact_verifies_and_reuses_cache(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"pinned model artifact")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    spec = _spec(source, digest)
    cache = tmp_path / "cache"

    destination = fetch_artifact(spec, spec.artifacts[0], cache)
    assert destination.read_bytes() == source.read_bytes()
    assert artifact_status(spec, cache)[0]["verified"] is True
    assert require_artifacts(spec, cache)["weights"] == destination
    assert fetch_artifact(spec, spec.artifacts[0], cache) == destination


def test_fetch_artifact_rejects_wrong_checksum(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"unexpected")
    spec = _spec(source, "0" * 64)

    with pytest.raises(ValueError, match="checksum mismatch"):
        fetch_artifact(spec, spec.artifacts[0], tmp_path / "cache")
    assert artifact_status(spec, tmp_path / "cache")[0]["present"] is False
    with pytest.raises(FileNotFoundError, match="tsf model artifacts Fixture"):
        require_artifacts(spec, tmp_path / "cache")


def test_artifact_rejects_unpinned_or_unsafe_fields() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        ModelArtifact("weights", "https://example.com/w", "v1", "bad", "w.bin")
    with pytest.raises(ValueError, match="basename"):
        ModelArtifact("weights", "https://example.com/v1/w", "v1", "0" * 64, "../w")
    with pytest.raises(ValueError, match="pinned revision"):
        ModelArtifact("weights", "https://example.com/w", "v1", "0" * 64, "w")


def test_artifact_factory_receives_validated_params_and_verified_paths(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"foundation fixture")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    received = {}

    def build_with_artifacts(cfg, params, paths):
        received.update({"cfg": cfg, "params": params, "paths": paths})
        return nn.Identity()

    spec = _spec(source, digest)
    spec = ModelSpec(
        **{
            **spec.__dict__,
            "artifact_factory": build_with_artifacts,
        }
    )
    cache = tmp_path / "cache"
    fetched = fetch_artifact(spec, spec.artifacts[0], cache)
    model = spec.build_with_artifacts(
        "config", {"width": 7}, require_artifacts(spec, cache)
    )
    assert isinstance(model, nn.Identity)
    assert received == {
        "cfg": "config",
        "params": {"width": 7},
        "paths": {"weights": fetched},
    }


def test_artifact_factory_requires_a_declared_artifact() -> None:
    with pytest.raises(ValueError, match="without artifacts"):
        ModelSpec(
            name="Fixture",
            module="tsflab.models.fixture",
            model_class=nn.Identity,
            factory=lambda cfg, params: nn.Identity(),
            artifact_factory=lambda cfg, params, paths: nn.Identity(),
            params_schema=_Params,
            capabilities=frozenset({"time-series"}),
        )


# ---------------------------------------------------------------------------
# Executable contract runner (fake models)
# ---------------------------------------------------------------------------


class _SamplingModel(nn.Module):
    """Forward samples without gradients; training goes through ``loss``."""

    def __init__(self, seq_len: int, pred_len: int) -> None:
        super().__init__()
        self.head = nn.Linear(seq_len, pred_len)

    def forward(self, x, x_mark=None, x_dec=None, x_mark_dec=None):
        with torch.no_grad():
            return self.head(x.transpose(1, 2)).transpose(1, 2)

    def loss(self, x, future):
        return (self.head(x.transpose(1, 2)).transpose(1, 2) - future).pow(2).mean()


def _contract_spec(training_objective=None, training_setup=None) -> SimpleNamespace:
    return SimpleNamespace(
        output_type="point", capabilities=frozenset(),
        training_objective=training_objective, training_setup=training_setup,
    )


class ObjectiveBackwardTests(unittest.TestCase):
    task = SimpleNamespace(seq_len=8, label_len=4, pred_len=3, features="M")
    params = {"enc_in": 2}

    def test_detached_forward_without_objective_fails(self) -> None:
        model = _SamplingModel(8, 3)
        with self.assertRaisesRegex(ValueError, "detached"):
            _forward_contract(model, _contract_spec(), self.task, self.params, backward=True)

    def test_declared_objective_loss_is_back_propagated(self) -> None:
        seen = {}

        def setup(model, train_loader, *, pred_len, features):
            first = next(iter(train_loader))
            seen.update(setup=[tuple(t.shape) for t in first], pred_len=pred_len)
            self.assertEqual(train_loader.dataset.data.shape, (8 + 3 + 16 - 1, 2))

        def objective(model, batch, criterion):
            self.assertIn("setup", seen)  # training_setup runs first
            seen.update(training=model.training, y=tuple(batch.y.shape), target=tuple(batch.target.shape))
            # Consecutive windows advance by one hourly step.
            self.assertEqual(float(batch.x_mark[1, 0, 4] - batch.x_mark[0, 0, 4]), 1.0)
            return None, model.loss(batch.x, batch.target)

        model = _SamplingModel(8, 3)
        _forward_contract(model, _contract_spec(objective, setup), self.task, self.params, backward=True)
        self.assertEqual(seen, {
            "setup": [(2, 8, 2), (2, 7, 2), (2, 8, 6), (2, 7, 6)], "pred_len": 3,
            "training": True, "y": (2, 7, 2), "target": (2, 3, 2),
        })
        self.assertIsNotNone(model.head.weight.grad)

    def test_detached_objective_loss_fails(self) -> None:
        def objective(model, batch, criterion):
            with torch.no_grad():
                return None, model.loss(batch.x, batch.target)

        with self.assertRaisesRegex(ValueError, "objective loss is detached"):
            _forward_contract(_SamplingModel(8, 3), _contract_spec(objective), self.task, self.params, backward=True)


# ---------------------------------------------------------------------------
# Foundation-model runtime boundary (fake runtimes)
# ---------------------------------------------------------------------------


def _source(name: str = "Example") -> FoundationSource:
    return FoundationSource(
        name=name,
        package="official-package==1.0",
        codebase="https://example.com/official",
        model_id="owner/model",
        revision="0123456789abcdef",
        license="Apache-2.0",
    )


class _Runtime:
    source = _source()

    def predict(self, context, prediction_length, quantile_levels):
        mean = context[:, -1:].repeat(1, prediction_length)
        quantiles = None
        if quantile_levels:
            quantiles = torch.stack(
                [mean + level for level in quantile_levels], dim=-1
            )
        return FoundationForecast(mean, quantiles)


class FoundationRuntimeTests(unittest.TestCase):
    def test_canonical_wrapper_restores_channel_axes(self) -> None:
        values = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
        point = FoundationModel(_Runtime(), prediction_length=2)
        self.assertEqual(tuple(point(values).shape), (2, 2, 3))

        quantile = FoundationModel(
            _Runtime(), prediction_length=2, quantile_levels=(0.1, 0.5, 0.9)
        )
        output = quantile(values)
        self.assertEqual(tuple(output.shape), (2, 2, 3, 3))
        self.assertTrue(torch.all(output[..., 0] < output[..., 1]))

    def test_source_rejects_incomplete_or_non_https_facts(self) -> None:
        with self.assertRaises(ValueError):
            FoundationSource("", "pkg", "https://example.com", "id", "rev", "MIT")
        with self.assertRaises(ValueError):
            FoundationSource("x", "pkg", "http://example.com", "id", "rev", "MIT")

    def test_chronos_loader_is_pinned_and_offline(self) -> None:
        calls = {}

        class Pipeline:
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                calls.update(path=path, **kwargs)
                return cls()

            def predict_quantiles(self, *, inputs, prediction_length, quantile_levels):
                mean = torch.zeros(inputs.shape[0], prediction_length)
                quantiles = torch.zeros(
                    inputs.shape[0], prediction_length, len(quantile_levels)
                )
                return quantiles, mean

        with tempfile.TemporaryDirectory() as directory:
            runtime = ChronosRuntime.from_local(
                _source("Chronos"), directory, loader=Pipeline
            )
        result = runtime.predict(torch.ones(2, 8), 3, (0.1, 0.9))
        self.assertTrue(calls["local_files_only"])
        self.assertEqual(calls["revision"], "0123456789abcdef")
        self.assertEqual(tuple(result.quantiles.shape), (2, 3, 2))

    def test_timesfm_loader_compiles_and_selects_requested_deciles(self) -> None:
        calls = {}

        class Model:
            def compile(self, config):
                calls["config"] = config

            def forecast(self, *, horizon, inputs):
                point = np.zeros((len(inputs), horizon), dtype=np.float32)
                quantiles = np.zeros((len(inputs), horizon, 10), dtype=np.float32)
                quantiles[..., 1] = 0.1
                quantiles[..., 9] = 0.9
                return point, quantiles

        class Loader:
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                calls.update(path=path, **kwargs)
                return Model()

        def config_factory(**kwargs):
            return kwargs

        with tempfile.TemporaryDirectory() as directory:
            runtime = TimesFMRuntime.from_local(
                _source("TimesFM"),
                directory,
                max_context=32,
                max_horizon=8,
                loader=Loader,
                config_factory=config_factory,
            )
        result = runtime.predict(torch.ones(2, 16), 4, (0.1, 0.9))
        self.assertTrue(calls["local_files_only"])
        self.assertEqual(calls["config"]["max_horizon"], 8)
        self.assertTrue(torch.allclose(result.quantiles[..., 0], torch.tensor(0.1)))
        self.assertTrue(torch.allclose(result.quantiles[..., 1], torch.tensor(0.9)))

    def test_moirai_adapter_uses_official_quantile_output(self) -> None:
        class Module:
            quantile_levels = (0.1, 0.5, 0.9)

        class HParams:
            prediction_length = 2

        class Model:
            module = Module()
            hparams = HParams()

            def predict(self, inputs):
                batch = len(inputs)
                values = np.zeros((batch, 3, 2, 1), dtype=np.float32)
                values[:, 1, :, :] = 0.5
                return values

        runtime = MoiraiRuntime(_source("Moirai"), Model())
        result = runtime.predict(torch.ones(2, 8), 2, (0.1, 0.9))
        self.assertEqual(tuple(result.mean.shape), (2, 2))
        self.assertEqual(tuple(result.quantiles.shape), (2, 2, 2))
        self.assertTrue(torch.allclose(result.mean, torch.tensor(0.5)))

    def test_official_loaders_require_an_existing_local_path(self) -> None:
        class Loader:
            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                raise AssertionError("loader must not run for a missing path")

        missing = Path(tempfile.gettempdir()) / "tsflab-missing-foundation"
        with self.assertRaises(FileNotFoundError):
            ChronosRuntime.from_local(_source("Chronos"), missing, loader=Loader)


# ---------------------------------------------------------------------------
# Similarity mining (tsf model similar)
# ---------------------------------------------------------------------------


OPERATOR = """
import torch
import torch.nn as nn
import torch.nn.functional as F


class {cls}(nn.Module):
    def __init__(self, {width}: int, dropout: float) -> None:
        super().__init__()
        self.{up} = nn.Linear({width}, 4 * {width})
        self.{down} = nn.Linear(4 * {width}, {width})
        self.drop = nn.Dropout(dropout)

    def forward(self, {x}: torch.Tensor) -> torch.Tensor:
        {h} = F.gelu(self.{up}({x}), approximate="tanh")
        {h} = torch.softmax(self.{down}({h}), dim=-1) * {x}
        return self.drop(torch.cumsum({h}, dim=1) / {x}.size(1))
"""


UNRELATED = """
import torch
import torch.nn as nn


class Graph(nn.Module):
    def __init__(self, nodes: int) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(nodes, 8))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        adjacency = torch.relu(self.embedding @ self.embedding.T)
        adjacency = adjacency / adjacency.sum(-1, keepdim=True).clamp_min(1e-6)
        return torch.einsum("nm,bmc->bnc", adjacency, values)
"""


def _write_package(root: Path, owner: str, source: str) -> None:
    if owner.startswith("component:"):
        target = root / "src/tsflab/models/_components" / owner.split(":", 1)[1] / "__init__.py"
    else:
        target = root / "src/tsflab/models" / owner / "model.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(textwrap.dedent(source), encoding="utf-8")


def _operator(**names: str) -> str:
    defaults = {"cls": "Block", "width": "width", "up": "up", "down": "down", "x": "x", "h": "h"}
    return OPERATOR.format(**{**defaults, **names})


def _clusters(root: Path, threshold: float = 0.6) -> list[dict]:
    return clusters(similar_pairs(collect_units(root), threshold=threshold))


def test_renamed_copy_is_an_exact_extract_candidate(tmp_path: Path) -> None:
    _write_package(tmp_path, "alpha", _operator())
    _write_package(tmp_path, "beta", _operator(cls="Mixer", width="d_model", up="fc1", down="fc2", x="inp", h="tmp"))
    _write_package(tmp_path, "gamma", UNRELATED)
    found = _clusters(tmp_path)
    assert found, "a renamed copy must be nominated"
    top = found[0]
    assert top["decision"] == "extract-new candidate"
    assert top["models"] == ["alpha", "beta"]
    assert top["max_score"] == 1.0
    assert all("gamma" not in cluster["models"] for cluster in found)


def test_component_match_suggests_reuse(tmp_path: Path) -> None:
    _write_package(tmp_path, "alpha", _operator())
    _write_package(tmp_path, "component:ffn", _operator(cls="FeedForward", x="hidden"))
    found = _clusters(tmp_path)
    assert found[0]["decision"] == "reuse-existing candidate: ffn"
    assert found[0]["components"] == ["ffn"]


def test_near_variant_with_an_extra_argument_is_still_found(tmp_path: Path) -> None:
    _write_package(tmp_path, "alpha", _operator())
    variant = _operator(cls="Block2").replace(
        "def __init__(self, width: int, dropout: float)",
        "def __init__(self, width: int, dropout: float, scale: float = 1.0)",
    )
    _write_package(tmp_path, "beta", variant)
    found = _clusters(tmp_path)
    assert found and set(found[0]["models"]) == {"alpha", "beta"}


def test_glue_wrappers_are_not_operators(tmp_path: Path) -> None:
    glue = """
    class Model:
        def forward(self, x, marks, decoder, decoder_marks):
            hidden = self.backbone(x.permute(0, 2, 1)).permute(0, 2, 1)
            hidden = self.head(hidden.reshape(hidden.shape[0], -1, self.channels))
            hidden = self.norm(hidden).transpose(1, 2).contiguous()
            return hidden[:, -self.pred_len:, :].reshape(x.shape[0], self.pred_len, -1)
    """
    _write_package(tmp_path, "alpha", glue)
    _write_package(tmp_path, "beta", glue.replace("hidden", "out"))
    _write_package(tmp_path, "gamma", glue.replace("backbone", "encoder"))
    assert all(cluster["max_score"] == 1.0 for cluster in _clusters(tmp_path)), \
        "only exact copies of glue may be reported"


def test_same_package_duplicates_are_not_cross_model_candidates(tmp_path: Path) -> None:
    _write_package(tmp_path, "alpha", _operator() + _operator(cls="Other"))
    assert _clusters(tmp_path) == []


# ---------------------------------------------------------------------------
# Characteristic vocabulary and matching
# ---------------------------------------------------------------------------


def test_data_terms_are_exactly_the_profile_rules() -> None:
    assert set(data_terms()) == {str(rule["id"]) for rule in load_rules()["rule"]}
    assert not set(data_terms()) & set(TASK_TERMS)
    assert kind_of("strong-seasonality") == "data" and kind_of("spatial-graph") == "task"
    assert kind_of("any") == "generic" and kind_of("made-up") is None
    assert all(meaning for meaning in vocabulary().values())


def test_term_validation() -> None:
    assert term_problems("x", "fits", ["strong-seasonality", "spatial-graph"], allow_any=False) == []
    assert term_problems("x", "fits", ["any"], allow_any=True) == []
    assert term_problems("x", "fits", ["any", "many-channels"], allow_any=True)
    assert term_problems("x", "characteristics", ["any"], allow_any=False)
    assert term_problems("x", "fits", ["nonsense"], allow_any=True)
    assert term_problems("x", "fits", [], allow_any=True)


def test_match_ranks_by_overlap() -> None:
    from tsflab.catalog.match import _rank

    entries = [
        {"name": "a", "fits": ["strong-seasonality", "multi-periodic"]},
        {"name": "b", "fits": ["strong-seasonality"]},
        {"name": "c", "fits": ["spatial-graph"]},
        {"name": "d", "fits": ["any"]},
    ]
    ranked = _rank(entries, {"strong-seasonality", "multi-periodic"})
    assert [e["name"] for e in ranked] == ["a", "b"]
    assert ranked[0]["matched"] == ["multi-periodic", "strong-seasonality"]


# ---------------------------------------------------------------------------
# Registry, components catalog, and code agreement (static; nothing is built)
# ---------------------------------------------------------------------------


MODEL_PACKAGES = sorted(p.parent for p in (ROOT / "src/tsflab/models").glob("*/spec.py")
                        if not p.parent.name.startswith("_"))


def test_every_model_package_is_registered_explicit_and_carded() -> None:
    assert len(MODEL_CATALOG.names()) == len(MODEL_PACKAGES)
    assert len(model_records(ROOT)) == len(MODEL_PACKAGES)
    for package in MODEL_PACKAGES:
        assert (package / "__init__.py").is_file(), package
        assert (package / "card.toml").is_file() and (package / "README.md").is_file(), package


def test_spec_rejects_inconsistent_objective_hooks() -> None:
    spec = MODEL_CATALOG.get("Linear")
    with pytest.raises(ValueError):
        replace(spec, training_objective=None, training_setup=lambda *a, **k: None)
    with pytest.raises(TypeError):
        replace(spec, training_objective=3)
    with pytest.raises(ValueError, match="duplicate components"):
        replace(spec, components=("revin", "revin"))


def test_task_mode_is_an_executable_dataset_model_contract() -> None:
    from tsflab.catalog.registry.datasets import (
        DATASET_REGISTRY,
        register_dataset_by_name,
    )
    from tsflab.experiments.config.loader import validate_task_compatibility

    register_dataset_by_name("weather")
    register_dataset_by_name("synthetic_st")
    flat, nodes = DATASET_REGISTRY.get("weather"), DATASET_REGISTRY.get("synthetic_st")
    linear, graph = MODEL_CATALOG.get("Linear"), MODEL_CATALOG.get("AGCRN")
    validate_task_compatibility("time_series", flat, linear)
    validate_task_compatibility("spatiotemporal", nodes, graph)
    with pytest.raises(ValueError, match="dataset 'weather'"):
        validate_task_compatibility("spatiotemporal", flat, graph)
    with pytest.raises(ValueError, match="model 'AGCRN'"):
        validate_task_compatibility("time_series", flat, graph)


def test_registered_models_have_the_canonical_signature_and_declared_components() -> None:
    from tsflab.cli.commands.check_registry import check

    assert check() == []


def test_component_catalog_matches_packages_and_consumers() -> None:
    from tsflab.catalog.component_audit import (
        audit_components,
        component_dependency_closure,
    )
    from tsflab.catalog.components import COMPONENT_CATALOG

    assert audit_components() == []
    packages = {p.parent.name for p in (ROOT / "src/tsflab/models/_components").glob("*/__init__.py")}
    assert set(COMPONENT_CATALOG.names()) == packages
    assert all(COMPONENT_CATALOG.get(name).contract for name in COMPONENT_CATALOG.names())
    assert set(component_dependency_closure({"patchtst"})) == {
        "flatten_forecast_head", "patchtst", "positional_encoding", "revin", "tst_transformer"}
    with pytest.raises(KeyError):
        component_dependency_closure({"no_such_component"})


def test_components_used_by_reads_imports_not_names(tmp_path: Path) -> None:
    from tsflab.catalog.component_audit import components_used_by

    package = tmp_path / "fake_model"
    package.mkdir()
    (package / "model.py").write_text(
        "from tsflab.models._components.revin import RevIN\n"
        "import tsflab.models._components.series_decomposition as sd\n"
        "# from tsflab.models._components.mamba import Mamba (comment only)\n",
        encoding="utf-8",
    )
    (package / "spec.py").write_text("from tsflab.models._components.dlinear import DLinear\n", encoding="utf-8")
    assert components_used_by(package) == ("revin", "series_decomposition")


# ---------------------------------------------------------------------------
# Admission (tsf model verify) with a fake executable contract
# ---------------------------------------------------------------------------


ADMITTED_FACTS = {
    "schema": "tsflab.card/1",
    "kind": "model",
    "name": "Fake",
    "tags": ["linear", "baseline", "lightweight"],
    "fidelity": "inferred",
    "composition": {"normalization": "none", "decomposition": "none", "temporal": "local:linear",
                    "channel": "local:independent", "head": "local:linear", "loss": "loss:mse"},
    "data_params": {"seq_len": {"from": "seq_len", "rule": "use the task lookback"}},
    "issues_checked": "fixture",
    "admission": {"status": "pending", "reference": "official"},
}


@pytest.fixture
def fake_card(tmp_path, monkeypatch):
    from tsflab.catalog import admission
    from tsflab.catalog.cards.toml_io import dumps

    path = tmp_path / "card.toml"
    path.write_text(dumps(ADMITTED_FACTS, inline=("data_params",)), encoding="utf-8")
    monkeypatch.setattr(admission, "card_path", lambda root, name: path)
    monkeypatch.setattr(admission, "_commit", lambda root: "abcdef12")
    return path


def test_run_contracts_maps_failures_and_passes(monkeypatch) -> None:
    from tsflab.catalog import admission, model_contracts

    def fake(names, strict):
        assert strict
        return [SimpleNamespace(stage="forward", error="boom")] if names == ["Bad"] else []

    monkeypatch.setattr(model_contracts, "audit_model_contracts", fake)
    assert admission.run_contracts(["Good", "Bad"]) == {"Good": None, "Bad": {"stage": "forward", "error": "boom"}}


def test_isolated_contracts_turn_crashes_and_timeouts_into_failures(monkeypatch) -> None:
    from tsflab.catalog import admission

    outcomes = {
        "Good": SimpleNamespace(returncode=0, stdout="noise\nnull\n", stderr=""),
        "Bad": SimpleNamespace(returncode=0, stdout='{"stage": "forward", "error": "boom"}\n', stderr=""),
        "Oom": SimpleNamespace(returncode=-9, stdout="", stderr=""),
        "Raise": SimpleNamespace(returncode=1, stdout="", stderr="Traceback\nImportError: x\n"),
    }

    def fake_run(cmd, **kwargs):
        if cmd[-1] == "Slow":
            raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
        return outcomes[cmd[-1]]

    monkeypatch.setattr(admission.subprocess, "run", fake_run)
    seen = []
    results = admission.run_contracts(["Good", "Bad", "Oom", "Raise", "Slow"], jobs=2, isolated=True, timeout=5,
                                      progress=lambda name, failure: seen.append(name))
    assert list(results) == ["Good", "Bad", "Oom", "Raise", "Slow"] and sorted(seen) == sorted(results)
    assert results["Good"] is None and results["Bad"] == {"stage": "forward", "error": "boom"}
    assert results["Oom"]["error"].startswith("killed by signal 9")
    assert results["Raise"] == {"stage": "process", "error": "exit 1: ImportError: x"}
    assert results["Slow"] == {"stage": "process", "error": "timed out after 5s"}


def test_write_admission_records_result_and_keeps_other_facts(fake_card) -> None:
    import tomllib
    from datetime import date

    from tsflab.catalog import admission
    from tsflab.catalog.cards.schema import ModelCard

    record = admission.write_admission(Path("."), "Fake", None)
    assert record == {"status": "passed", "date": date.today().isoformat(), "commit": "abcdef12",
                      "device": "cpu", "reference": "official", "note": ""}
    facts = tomllib.loads(fake_card.read_text(encoding="utf-8"))
    assert facts["admission"] == record
    assert {k: v for k, v in facts.items() if k != "admission"} == {
        k: v for k, v in ADMITTED_FACTS.items() if k != "admission"}
    assert 'seq_len = { from = "seq_len"' in fake_card.read_text(encoding="utf-8")
    ModelCard.model_validate(facts)

    failed = admission.write_admission(Path("."), "Fake", {"stage": "backward", "error": "x" * 400},
                                       reference="none")
    assert failed["status"] == "failed" and failed["reference"] == "none"
    assert failed["note"].startswith("backward: x") and len(failed["note"]) == 300
    assert admission.write_admission(Path("."), "Fake", None, note="manual")["note"] == "manual"


def test_changed_models_follow_packages_presets_and_used_components(tmp_path, monkeypatch) -> None:
    from tsflab.catalog import admission
    from tsflab.catalog.cards import metadata

    def git(*args):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=tmp_path,
                       check=True, capture_output=True)

    files = ["src/tsflab/models/alpha/model.py", "src/tsflab/models/beta/model.py",
             "src/tsflab/models/beta/README.md", "src/tsflab/models/_components/revin/__init__.py",
             "configs/models/Gamma.toml", "docs/index.md"]
    for name in files:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("v1\n", encoding="utf-8")
    git("init", "-q", "-b", "main")
    git("add", ".")
    git("commit", "-q", "-m", "base")
    git("branch", "base")
    records = [{"name": "Alpha", "package": "alpha", "components": []},
               {"name": "Beta", "package": "beta", "components": ["revin"]},
               {"name": "Gamma", "package": "gamma", "components": []},
               {"name": "Delta", "package": "delta", "components": []}]
    monkeypatch.setattr(metadata, "model_records", lambda root: records)

    def change(*names):
        for name in names:
            (tmp_path / name).write_text("v2\n", encoding="utf-8")
        git("commit", "-q", "-am", "change")

    change("src/tsflab/models/beta/README.md", "docs/index.md")
    assert admission.changed_models(tmp_path, "base") == []  # card prose and docs need no re-admission
    change("src/tsflab/models/_components/revin/__init__.py")
    assert admission.changed_models(tmp_path, "base") == ["Beta"]
    change("src/tsflab/models/alpha/model.py", "configs/models/Gamma.toml")
    assert admission.changed_models(tmp_path, "base") == ["Alpha", "Beta", "Gamma"]


def test_verify_command_writes_every_result_and_fails_on_any_failure(monkeypatch, capsys) -> None:
    import json

    from tsflab.catalog import admission
    from tsflab.cli.commands.verification import verification_command

    written = {}
    monkeypatch.setattr(admission, "run_contracts",
                        lambda names, jobs, **kw: {n: ({"stage": "forward", "error": "e"} if n == "AGCRN" else None)
                                             for n in names})

    def fake_write(root, name, failure, **kwargs):
        written[name] = "failed" if failure else "passed"
        return {"status": written[name], "note": ""}

    monkeypatch.setattr(admission, "write_admission", fake_write)
    assert verification_command(["Linear", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"Linear": {"status": "passed", "note": ""}}
    assert verification_command(["Linear", "AGCRN"]) == 1
    assert written == {"Linear": "passed", "AGCRN": "failed"}
    assert verification_command([]) == 2
    with pytest.raises(SystemExit):
        verification_command(["NoSuchModel"])


def test_contract_supplies_a_graph_only_to_requires_graph_specs() -> None:
    from tsflab.catalog import model_contracts

    class Params(BaseModel):
        enc_in: int

    def spec(requires_graph: bool) -> ModelSpec:
        return ModelSpec(name="G", module="m", model_class=nn.Module, factory=lambda cfg, params: params,
                         params_schema=Params, capabilities=frozenset(["spatiotemporal"]),
                         requires_graph=requires_graph)

    built = model_contracts._build_model(spec(True), None, {"enc_in": 5})
    assert built["adj_mx"].shape == (5, 5)
    assert np.allclose(built["adj_mx"], built["adj_mx"].T) and np.all(np.diag(built["adj_mx"]) == 1)
    assert "adj_mx" not in model_contracts._build_model(spec(False), None, {"enc_in": 5})


# ---------------------------------------------------------------------------
# Dense-weight cap of the flattened-window boosting baselines (arithmetic only)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("slug", "maps"),
    [("catboost_ts", lambda n: 1 + n), ("gradient_boosting_ts", lambda n: n),
     ("lightgbm_ts", lambda n: n), ("xgboost_ts", lambda n: n)],
)
def test_dense_weight_cap_refuses_wide_cells_and_admits_narrow_ones(slug, maps) -> None:
    module = importlib.import_module(f"tsflab.models.{slug}.model")
    count, cap = module.dense_parameter_count, module.MAX_DENSE_PARAMETERS
    # one (seq_len*enc_in) x (pred_len*enc_in) map per base, context or backcast layer
    assert count(4, 3, 2, 5) == maps(5) * (4 * 2) * (3 * 2)
    # traffic (862 channels, 96 -> 720) and wike2000 (2000 channels, 36 -> 60) are refused
    assert count(96, 720, 862, 16) > cap and count(36, 60, 2000, 12) > cap
    # few-channel cells stay far below the cap (ETTh1 96 -> 720, ILI 36 -> 60)
    assert count(96, 720, 7, 20) < cap / 20 and count(36, 60, 7, 20) < cap / 500
