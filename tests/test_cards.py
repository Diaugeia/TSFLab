"""tsflab.card/1: schema, store, render, TOML writer, audit, upstream issues, and declined papers.

Format rules are exercised on tiny cards written to ``tmp_path``; the repository
audit is checked only for code agreement (coverage, names, composition imports),
because card prose is curated separately.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from tsflab.catalog.cards import store
from tsflab.catalog.cards.issues import (
    DECLINE_REASONS,
    KINDS,
    Issue,
    load_declined,
    summarize,
    summarize_declined,
)
from tsflab.catalog.cards.render import card_files
from tsflab.catalog.cards.schema import (
    ISSUE_KINDS,
    L1_MAX_LINES,
    L2_MIN_LINES,
    README_SECTIONS,
    SCHEMA,
    SLOTS,
    ComponentCard,
    DatasetCard,
    ModelCard,
)
from tsflab.catalog.cards.toml_io import dumps
from tsflab.core.paths import repository_root

ROOT = repository_root()

MODEL = {
    "name": "Toy",
    "tags": ["linear", "baseline", "lightweight"],
    "fits": ["strong-seasonality"],
    "fidelity": "reference-checked",
    "paper": {"title": "Toy", "url": "https://example.com/paper", "venue": "Test", "year": 2026},
    "code": {"url": "https://example.com/code", "revision": "abc123", "license": "MIT"},
    "composition": {"normalization": "component:revin", "decomposition": "none", "temporal": "local:linear",
                    "channel": "local:independent", "head": "local:linear", "loss": "loss:mse"},
    "data_params": {"period": {"from": "period", "rule": "dominant period of the training split"}},
    "issues": [{"kind": "code-bug", "where": "`model.py`", "what": "noise is zero", "resolution": "Eq. 2"}],
    "admission": {"status": "passed", "date": "2026-10-03", "commit": "abcdef12", "device": "cpu",
                  "reference": "official"},
}
COMPONENT = {
    "name": "toy_block", "role": "Mixes time steps with one linear map.", "slot": "temporal",
    "fits": ["any"], "category": "mixer", "tags": ["linear", "mixer", "time"],
    "input": "[batch, time, channels]", "output": "[batch, horizon, channels]", "origin": "fixture",
}
DATASET = {
    "name": "toy_data", "domain": "energy", "topic": "toy load", "benchmarks": ["ltsf"],
    "tags": ["toy", "hourly", "fixture"],
    "source": {"name": "Toy", "url": "https://example.com", "citation": "Toy (2026)",
               "citation_url": "https://example.com/cite", "license": "CC-BY-4.0", "redistribution": "hosted"},
    "shape": {"frequency": "1h", "length": 100, "channels": 3, "channel_kind": "channels", "stats_basis": "measured"},
    "protocol": {"protocol": "70/10/20 chronological split", "seq_lens": [96], "pred_lens": [24]},
}
FACTS = {"model": MODEL, "component": COMPONENT, "dataset": DATASET}


def _sections(kind: str, lines: int = 2) -> dict[str, str]:
    return {title: "\n".join(f"{title} detail {i}." for i in range(lines)) for title in README_SECTIONS[kind]}


def _write(directory: Path, kind: str, facts: dict | None = None, *, description: str = "A toy card.",
           sections: dict[str, str] | None = None, reference: str | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    files = card_files(kind, dict(facts or FACTS[kind]), description, sections or _sections(kind))
    for name, text in files.items():
        (directory / name).write_text(text, encoding="utf-8")
    if reference is not None:
        (directory / "reference.md").write_text(reference, encoding="utf-8")
    return directory


def _problems(directory: Path, **kwargs) -> list[str]:
    return store.problems(store.load(directory), **kwargs)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["model", "component", "dataset"])
def test_rendered_cards_load_validate_and_pass_the_format_check(tmp_path, kind) -> None:
    card = store.load(_write(tmp_path / kind, kind))
    assert card.kind == kind and card.name == FACTS[kind]["name"]
    assert card.facts["schema"] == SCHEMA
    assert [title for title, _ in card.sections] == list(README_SECTIONS[kind])
    assert card.reference is None
    assert store.problems(card, curated=True) == []
    assert type(store.validated(card)).__name__ == {"model": "ModelCard", "component": "ComponentCard",
                                                     "dataset": "DatasetCard"}[kind]


def test_model_schema_enforces_slots_issues_and_fidelity() -> None:
    def card(**over):
        return ModelCard.model_validate({"schema": SCHEMA, "kind": "model", **MODEL, **over})

    assert tuple(card().composition) == SLOTS and card().data_params["period"].from_ == "period"
    with pytest.raises(ValidationError, match="slots"):
        card(composition=dict(reversed(MODEL["composition"].items())))
    with pytest.raises(ValidationError, match="issues_checked"):
        card(issues=[])
    assert card(issues=[], issues_checked="Eq. 1-4 against model.py").issues == []
    with pytest.raises(ValidationError, match="official code"):
        card(code=None)
    assert card(code=None, fidelity="paper-only").code is None
    with pytest.raises(ValidationError):
        card(fits=["made-up-term"])
    with pytest.raises(ValidationError):
        card(tags=["two", "tags"])
    with pytest.raises(ValidationError):
        card(issues=[{**MODEL["issues"][0], "kind": "typo"}])
    with pytest.raises(ValidationError):
        card(admission={"status": "maybe"})
    with pytest.raises(ValidationError, match="Extra inputs"):
        card(verification={"status": "passed"})
    assert set(ISSUE_KINDS) == set(KINDS)


def test_component_and_dataset_schemas() -> None:
    def component(**over):
        return ComponentCard.model_validate({"schema": SCHEMA, "kind": "component", **COMPONENT, **over})

    def dataset(**over):
        return DatasetCard.model_validate({"schema": SCHEMA, "kind": "dataset", **DATASET, **over})

    assert component().fits == ["any"]
    for bad in ({"slot": "middle"}, {"category": "magic"}, {"fits": []}, {"role": "x" * 161},
                {"fits": ["any", "many-channels"]}):
        with pytest.raises(ValidationError):
            component(**bad)
    assert dataset(kind="dataset-family").kind == "dataset-family"
    with pytest.raises(ValidationError, match="characteristics_basis"):
        dataset(characteristics=["strong-seasonality"])
    assert dataset(characteristics=["strong-seasonality"], characteristics_basis="tsf data analyze").characteristics
    with pytest.raises(ValidationError):
        dataset(source={**DATASET["source"], "redistribution": "maybe"})
    for retired in ("unknown", "restricted", "allowed", "conditional", "link-only"):
        with pytest.raises(ValidationError):
            dataset(source={**DATASET["source"], "redistribution": retired})
    for klass in ("hosted", "upstream", "script"):
        assert dataset(source={**DATASET["source"], "redistribution": klass}).source.redistribution == klass
    assert dataset(source={**DATASET["source"], "redistribution": "hosted",
                           "conditions": "verbatim copies only"}).source.conditions
    for bad in ({"domain": "Energy / power"}, {"domain": "weather"}, {"domain": "mixed"},
                {"benchmarks": ["m4"]}, {"benchmarks": ["ltsf", "ltsf"]}):
        with pytest.raises(ValidationError):
            dataset(**bad)
    assert dataset(kind="dataset-family", domain="mixed").domain == "mixed"
    with pytest.raises(ValidationError):
        dataset(schema="tsflab.card/0")


# ---------------------------------------------------------------------------
# Store: loading and format problems
# ---------------------------------------------------------------------------


def test_load_requires_both_files_and_front_matter(tmp_path) -> None:
    directory = _write(tmp_path / "card", "component")
    (directory / "README.md").write_text("# no front matter\n", encoding="utf-8")
    with pytest.raises(ValueError, match="front matter"):
        store.load(directory)
    (directory / "README.md").unlink()
    with pytest.raises(ValueError, match="needs card.toml and README.md"):
        store.load(directory)


def test_problems_report_identity_and_section_mismatches(tmp_path) -> None:
    directory = _write(tmp_path / "card", "model", description="x" * (store.DESCRIPTION_CHARS + 1))
    readme = directory / "README.md"
    text = readme.read_text(encoding="utf-8").replace('name: "Toy"', 'name: "Other"')
    readme.write_text(text.replace("## Differences", "## Notes"), encoding="utf-8")
    toml = directory / "card.toml"
    toml.write_text(toml.read_text(encoding="utf-8").replace(SCHEMA, "tsflab.card/0"), encoding="utf-8")
    found = "\n".join(_problems(directory))
    assert "schema must be" in found
    assert "README name 'Other' differs" in found
    assert "description must be one line" in found
    assert "sections must be exactly Idea, When to use, Configure, Differences (got Idea, When to use, Configure, Notes)" in found


def test_split_sections_ignores_headings_in_code_fences() -> None:
    text = "intro\n## One\nbody\n```\n## not a section\n```\n## Two\n\nend\n"
    assert store.split_sections(text) == (("One", "body\n```\n## not a section\n```"), ("Two", "end"))


def test_l1_budget_and_reference_layout(tmp_path) -> None:
    per_section = L1_MAX_LINES // len(README_SECTIONS["component"]) + 2
    long = _write(tmp_path / "long", "component", sections=_sections("component", per_section))
    assert any("L1 budget" in p for p in _problems(long))

    short_reference = _write(tmp_path / "fold", "component", reference="# Detail\n\none line\n")
    assert any("short enough to fold" in p for p in _problems(short_reference))

    empty = _write(tmp_path / "empty", "component", reference="\n")
    assert any("empty reference.md" in p for p in _problems(empty))

    detail = "\n".join(f"line {i}" for i in range(L2_MIN_LINES + 1))
    kept = _write(tmp_path / "kept", "component", reference="## Derivation\n\n" + detail + "\n")
    card = store.load(kept)
    assert store.problems(card) == []
    assert card.reference_sections[0][0] == "Derivation"


def test_curation_requirements(tmp_path) -> None:
    sections = _sections("model")
    sections["Idea"] = f"{store.TODO}: explain the idea."
    unfit = _write(tmp_path / "todo", "model", {**MODEL, "fits": []}, sections=sections)
    assert _problems(unfit, curated=False) == []
    found = "\n".join(_problems(unfit, curated=True))
    assert "placeholders remain" in found and "needs fits" in found

    pending = {k: v for k, v in COMPONENT.items() if k not in {"role", "slot", "fits"}}
    directory = _write(tmp_path / "pending", "component", pending)
    assert _problems(directory, curated=False) == []  # schema waits for curation
    assert any("role" in p for p in _problems(directory, curated=True))


def test_card_directories_find_each_kind_and_skip_private_packages(tmp_path) -> None:
    _write(tmp_path / "src/tsflab/models/toy", "model")
    _write(tmp_path / "src/tsflab/models/_slots", "model")
    _write(tmp_path / "src/tsflab/models/_components/toy_block", "component")
    _write(tmp_path / "catalog/datasets/family/member", "dataset")
    found = store.card_directories(tmp_path)
    assert [p.name for p in found["model"]] == ["toy"]
    assert [p.name for p in found["component"]] == ["toy_block"]
    assert [p.name for p in found["dataset"]] == ["member"]


# ---------------------------------------------------------------------------
# Render and TOML writer
# ---------------------------------------------------------------------------


def test_render_orders_sections_and_rejects_unknown_ones() -> None:
    files = card_files("dataset-family", {**DATASET, "kind": "ignored"}, "  A   family\n card. ",
                       {"Protocol and pitfalls": "p", "Overview": "o"})
    readme = files["README.md"]
    assert readme.index("## Overview") < readme.index("## Protocol and pitfalls")
    assert 'description: "A family card."' in readme
    facts = tomllib.loads(files["card.toml"])
    assert (facts["schema"], facts["kind"]) == (SCHEMA, "dataset-family")
    with pytest.raises(ValueError, match="unknown README sections"):
        card_files("component", COMPONENT, "d", {"Origin": "x"})


def test_toml_writer_round_trips_and_keeps_inline_tables() -> None:
    data = {"name": "x", "flag": True, "ratio": 0.5, "items": [1, 2], "key with space": "v",
            "table": {"a": 1, "nested": {"b": "c"}}, "params": {"p": {"from": "period", "rule": "r"}},
            "rows": [{"k": 1}, {"k": 2}], "skipped": None}
    text = dumps(data, inline=("params",))
    assert tomllib.loads(text) == {k: v for k, v in data.items() if k != "skipped"}
    assert 'p = { from = "period", rule = "r" }' in text
    assert '"key with space" = "v"' in text
    with pytest.raises(ValueError):
        dumps({"x": float("nan")})
    with pytest.raises(TypeError):
        dumps({"x": object()})


# ---------------------------------------------------------------------------
# Repository audit: code agreement
# ---------------------------------------------------------------------------


CODE_AGREEMENT = ("orphan", "card name", "registered", "does not import", "origin model",
                  "catalog keywords", "architecture family", "admitted model", "needs card.toml",
                  "composition", "ComponentSpec")


def test_repository_cards_agree_with_code() -> None:
    from tsflab.catalog.cards.audit import audit_cards

    found = [p for p in audit_cards(ROOT) if any(marker in p for marker in CODE_AGREEMENT)]
    assert found == []


def test_audit_flags_orphan_cards(tmp_path) -> None:
    from tsflab.catalog.cards import audit

    _write(tmp_path / "src/tsflab/models/_components/ghost", "component", {**COMPONENT, "name": "ghost"})
    assert any("orphan component card" in p for p in audit._component_problems(tmp_path))


# ---------------------------------------------------------------------------
# Upstream issues and declined papers
# ---------------------------------------------------------------------------


def test_issue_summary_counts_kinds_and_coverage() -> None:
    first = [Issue("code-bug", "a", "b", "c"), Issue("paper-error", "Eq. 1", "d", "e")]
    summary = summarize({"A": first, "B": [], "C": None})
    assert summary["documented"] == 2 and summary["with_issues"] == 1
    assert summary["by_kind"] == {"code-bug": 1, "paper-error": 1}
    assert summary["missing"] == ["C"]


def test_declined_record_is_valid_and_disjoint_from_the_catalog() -> None:
    from tsflab.catalog.cards.metadata import model_records

    papers, problems = load_declined(ROOT)
    assert problems == []
    assert papers, "catalog/declined.toml should record reviewed, declined papers"
    assert {paper["reason"] for paper in papers} <= set(DECLINE_REASONS)
    assert not {paper["name"] for paper in papers} & {str(r["name"]) for r in model_records(ROOT)}
    summary = summarize_declined(papers)
    assert summary["papers"] == len(papers) and summary["issues"] == sum(summary["by_kind"].values())


def test_declined_loader_reports_bad_entries(tmp_path) -> None:
    assert load_declined(tmp_path) == ([], [])
    (tmp_path / "catalog").mkdir()
    (tmp_path / "catalog/declined.toml").write_text(
        '[[paper]]\nname = "X"\ntitle = "t"\npaper = "p"\ncode = "c"\nrevision = "r"\n'
        'reason = "boring"\nsummary = "s"\nissues = [{ kind = "typo", what = "w" }]\n'
        '[[paper]]\nname = "Y"\n',
        encoding="utf-8",
    )
    _, problems = load_declined(tmp_path)
    assert len(problems) == 3
    assert any("lacks" in p for p in problems)


def test_issues_command_reads_card_toml(capsys) -> None:
    from tsflab.cli.main import main

    assert main(["model", "issues", "--summary"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["models"] == len(list((ROOT / "src/tsflab/models").glob("*/spec.py")))
    assert summary["missing"] == [] and summary["malformed"] == 0
    assert set(summary["by_kind"]) <= set(KINDS)
