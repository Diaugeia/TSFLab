"""tsflab.cli: the command surface, routing, catalog views, scaffolds, and repo check step selection."""

from __future__ import annotations

import ast
import inspect
import tomllib
import unittest

import pytest

from tsflab.cli import main as cli
from tsflab.cli.commands import new_dataset, new_model
from tsflab.cli.commands.catalog_resources import _extract_kind
from tsflab.cli.commands.data_results import _extract_from
from tsflab.cli.commands.execution import _extract_backend
from tsflab.cli.commands.gate import affected_smoke_configs, select_steps
from tsflab.core.paths import repository_root

# ---------------------------------------------------------------------------
# Command surface and routing
# ---------------------------------------------------------------------------


COMMANDS = {
    "catalog", "data", "model", "run", "env", "result", "realtime",
    "research", "repo", "agent", "init",
}


def test_public_surface_is_eleven_module_commands() -> None:
    assert set(cli.COMMANDS) == COMMANDS
    assert all(f"    {name} " in cli.__doc__ for name in COMMANDS)


@pytest.mark.parametrize("old", [
    "dataset", "component", "verify", "smoke", "inspect", "queue", "slurm", "storage",
    "usage", "interface", "submit", "hub", "schema-export", "leaderboard-build",
])
def test_removed_commands_are_unknown(old: str, capsys) -> None:
    assert cli.main([old]) == 2
    assert "unknown command" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [["repo", "audit"], ["repo", "doctor"], ["model", "show", "x"],
                                  ["model", "list"], ["data", "list"], ["result", "nope"]])
def test_removed_subcommands_print_usage(argv: list[str], capsys) -> None:
    assert cli.main(argv) == 2
    assert "usage:" in capsys.readouterr().err


def test_help_for_every_routed_command(capsys) -> None:
    assert cli.main([]) == 0
    for name in ("catalog", "data", "model", "repo", "agent", "result"):
        assert cli.main([name, "--help"]) == 0
    assert "tsf catalog show" in capsys.readouterr().out


def test_option_extractors() -> None:
    assert _extract_kind(["show", "x", "--kind", "model", "--json"]) == ("model", ["show", "x", "--json"])
    assert _extract_kind(["--kind=dataset", "a"]) == ("dataset", ["a"])
    assert _extract_from(["--from", "gift-eval", "--link-only"]) == ("gift-eval", ["--link-only"])
    assert _extract_backend(["add", "dir", "--backend", "queue"]) == (["add", "dir"], "queue")
    assert _extract_backend(["cfg.toml"]) == (["cfg.toml"], "local")
    with pytest.raises(SystemExit):
        _extract_backend(["--backend", "ray"])


# ---------------------------------------------------------------------------
# tsf repo check step selection
# ---------------------------------------------------------------------------


ROOT = repository_root()


def names(scope: str, changed: list[str]) -> list[str]:
    return [step.name for step in select_steps(scope, changed, ROOT)]


class GateSelectionTests(unittest.TestCase):
    def test_full_scope_has_every_check_and_no_smoke(self) -> None:
        selected = names("full", [])
        for required in ("schema-export", "agent-assets", "model-cards",
                         "dataset-cards", "component-cards",
                         "repo-audit", "web-submissions", "pytest"):
            self.assertIn(required, selected)
        self.assertNotIn("smoke-affected", selected)
        self.assertEqual(selected[-1], "pytest")  # slowest step last

    def test_passed_admission_is_required_only_for_release(self) -> None:
        for scope in ("full", "changed"):
            self.assertIn("model-cards", names(scope, []))
            self.assertNotIn("model-release", names(scope, []))
        selected = names("release", [])
        self.assertIn("model-release", selected)
        self.assertNotIn("model-cards", selected)
        self.assertNotIn("verification-stale", selected)

    def test_changed_scope_without_model_code_skips_smoke(self) -> None:
        self.assertNotIn("smoke-affected", names("changed", ["docs/en/models.md"]))

    def test_runner_change_uses_representative_set(self) -> None:
        configs = affected_smoke_configs(["src/tsflab/experiments/runner/run_one.py"], ROOT)
        self.assertEqual(len(configs), 3)

    def test_model_change_maps_to_its_smoke_config(self) -> None:
        configs = affected_smoke_configs(["src/tsflab/models/crib/model.py"], ROOT)
        self.assertEqual(configs, ["configs/runs/smoke_crib.toml"])
        self.assertIn("smoke-affected",
                      names("changed", ["src/tsflab/models/crib/model.py"]))

    def test_model_without_smoke_config_falls_back(self) -> None:
        configs = affected_smoke_configs(["src/tsflab/models/nonexistent_x/model.py"], ROOT)
        self.assertEqual(len(configs), 3)

    def test_edited_smoke_config_is_selected(self) -> None:
        self.assertEqual(affected_smoke_configs(["configs/runs/smoke_crib.toml"], ROOT),
                         ["configs/runs/smoke_crib.toml"])


# ---------------------------------------------------------------------------
# Model and dataset scaffolds
# ---------------------------------------------------------------------------


LEGACY = ("benchmark", "data", "models", "tsf_core")


def _imported_modules(source: str) -> list[str]:
    modules = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
    return modules


def _string_literals(module) -> list[str]:
    tree = ast.parse(inspect.getsource(module))
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def test_scaffold_templates_use_the_tsflab_namespace() -> None:
    offending = []
    for module in (new_model, new_dataset):
        for literal in _string_literals(module):
            for line in literal.splitlines():
                stripped = line.strip()
                if stripped.startswith(("from ", "import ")):
                    head = stripped.split()[1].split(".")[0]
                    if head in LEGACY:
                        offending.append((module.__name__, stripped))
    assert offending == []


def test_scaffold_literals_emit_valid_toml_and_python_booleans() -> None:

    from tsflab.cli.commands.new_model import _literal, _python_literal

    for written, toml_value in (("True", True), ("false", False), (None, True)):
        literal = _literal("bool", written)
        assert tomllib.loads(f"flag = {literal}")["flag"] is toml_value
        assert _python_literal("bool", literal) in {"True", "False"}


def test_public_module_slug_is_consistent() -> None:
    from tsflab.cli.commands.new_model import _module_slug
    from tsflab.cli.runtime import module_slug

    for normalize in (module_slug, _module_slug):
        assert normalize("AirFormer") == "airformer"
        assert normalize("S_Mamba") == "s_mamba"


def test_model_scaffold_emits_complete_compilable_package_templates() -> None:
    from tsflab.cli.commands.new_dataset import _schema_single
    from tsflab.cli.commands.new_model import _model, _package_init, _spec

    package_init = _package_init("PaperModel")
    model = _model("PaperModel", [("width", "int", "32")], False)
    spec = _spec("PaperModel", "paper_model", [("width", "int", "32")], "time_series", ("revin",))
    for text, name in ((package_init, "__init__.py"), (model, "model.py"), (spec, "spec.py")):
        compile(text, name, "exec")
    assert "from .model import Model" in package_init
    assert "x_mark_enc=None" in model
    assert not any(token in model for token in ("*args", "**kwargs", "mask=None"))
    assert "components=('revin',)" in spec and 'ConfigDict(extra="forbid")' in spec
    schema = _schema_single("paper_data")
    compile(schema, "paper_data.py", "exec")
    assert "DatasetParameters" in schema


# ---------------------------------------------------------------------------
# Catalog views (tsf catalog list/show/search, tsf model audit)
# ---------------------------------------------------------------------------


def _run(*argv: str) -> str:
    import contextlib
    import io

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        assert cli.main(list(argv)) == 0, argv
    return output.getvalue()


def _json(*argv: str):
    import json

    return json.loads(_run(*argv, "--json"))


def test_catalog_list_covers_every_resource() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    models = _json("catalog", "list", "--kind", "model")
    assert len(models) == len([p for p in (root / "src/tsflab/models").glob("*/spec.py")
                               if not p.parent.name.startswith("_")])
    assert all(record["summary"] for record in models)
    datasets = _json("catalog", "list", "--kind", "dataset")
    assert len(datasets) == len(list((root / "configs/datasets").rglob("*.toml")))
    components = _json("catalog", "list", "--kind", "component")
    assert len(components) == len(list((root / "src/tsflab/models/_components").glob("*/card.toml")))


def test_catalog_show_routes_component_facts_and_consumers() -> None:
    payload = _json("catalog", "show", "quantile_head", "--kind", "component")
    assert payload["module"] == "tsflab.models._components.quantile_head"
    assert "quantile_dlinear" in payload["consumers"]
    assert payload["card"]["schema"] == "tsflab.card/1" and payload["card"]["slot"] == "head"


def test_catalog_search_ranks_names_and_tags() -> None:
    assert _json("catalog", "search", "--kind", "dataset", "electricity", "15t")[0]["name"] == "gift_eval/electricity_15T"
    hits = _json("catalog", "search", "--kind", "model", "exogenous", "transformer")[:3]
    timexer = next(hit for hit in hits if hit["name"] == "TimeXer")
    assert "exogenous" in timexer["matched_terms"]
    assert _json("catalog", "search", "--kind", "component", "patch", "transformer", "backbone")[0]["name"] == "patchtst"
    first = _run("catalog", "search", "reversible", "instance", "normalization", "--limit", "3").splitlines()[0]
    assert first.split("\t")[:2] == ["revin", "component"]
    only = _json("catalog", "search", "electricity", "--kind", "dataset")
    assert only and all(item["kind"] == "dataset" for item in only)


@pytest.mark.parametrize("kind, name", [("component", "revin"), ("model", "Linear"), ("dataset", "etth1")])
def test_progressive_disclosure_depths(kind: str, name: str) -> None:
    import json

    l0 = _run("catalog", "show", name, "--kind", kind, "--depth", "0").rstrip("\n")
    assert len(l0.splitlines()) == 1 and len(l0.split("\t")) == 4
    l1 = _run("catalog", "show", name, "--kind", kind, "--depth", "1")
    l2 = _run("catalog", "show", name, "--kind", kind, "--depth", "2")
    assert len(l1) <= len(l2)
    assert "README.md" in _run("catalog", "show", name, "--kind", kind, "--depth", "3")
    payload = json.loads(_run("catalog", "show", name, "--kind", kind, "--depth", "2", "--json"))
    assert payload["depth"] == 2 and payload["card"]["name"] == name and payload["sections"]


def test_model_audit_reports_cards_and_admission() -> None:
    import json

    from tsflab.catalog.registry.models import MODEL_CATALOG

    summary = json.loads(_run("model", "audit", "--summary"))
    total = len(MODEL_CATALOG.names())
    assert summary["models"] == total and summary["failed"] == 0 and summary["blockers"] == {}
    assert sum(summary["verification"].values()) == total
    assert summary["admission_pending"] <= total
    record = _json("model", "audit", "CATS")[0]
    assert record["name"] == "CATS" and record["blockers"] == [] and record["codebase"]["missing"] == []
    assert record["verification"]["status"] in {"passed", "failed", "pending"}
