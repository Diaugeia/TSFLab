"""The card format, ``tsflab.card/1`` — one contract for models, components, and datasets.

Every catalog object lives in its own directory with:

* ``card.toml`` — structured facts read by tools and agents (this schema);
* ``README.md`` — Skill-shaped prose: front matter ``name`` + ``description``
  (L0), then a short body (L1) with fixed sections per kind;
* ``reference.md`` — optional L2 detail, only when L1 would exceed its budget.

Facts that code already knows (config path, parameter schema, imported
components, consumers, loader, task modes) are derived at read time and never
stored. A format change bumps ``SCHEMA`` and ships with a migration of every card.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA = "tsflab.card/1"
SLOTS = ("normalization", "decomposition", "temporal", "channel", "head", "loss")
COMPONENT_SLOTS = (*SLOTS, "input", "support")
COMPONENT_CATEGORIES = (
    "attention", "backbone", "convolution", "decomposition", "embedding", "frequency", "fusion",
    "graph", "head", "memory", "mixer", "normalization", "objective", "routing", "state-space",
    "utility",
)
ISSUE_KINDS = ("code-bug", "paper-code-mismatch", "paper-error", "underspecified", "unreleased",
               "leakage", "license")
DATA_PARAM_SOURCES = ("period", "frequency", "channels", "nodes", "covariates", "seq_len", "pred_len",
                      "graph", "train-split")
#: ``hosted``: TSFLab re-hosts the files in Diaugeia/TSFLab-Datasets (``static/`` or ``realtime/``;
#: ``tsf data download``), with
#: any extra source terms stated in ``source.conditions``; ``upstream``: fetched from another
#: party's Hugging Face repository, never re-hosted; ``script``: the license forbids re-hosting,
#: so TSFLab ships a fetch command and users download from the original source.
REDISTRIBUTION = ("hosted", "upstream", "script")
#: Application domain of a dataset (exactly one); ``mixed`` only for a dataset family.
DOMAINS = ("energy", "transport", "environment", "finance", "healthcare", "cloud-web", "sales")
FAMILY_DOMAINS = (*DOMAINS, "mixed")
#: Benchmark suites a preset belongs to.
BENCHMARKS = ("ltsf", "tfb", "st-graph", "gift-eval", "realtime")
STATS_BASIS = ("measured", "source-reported", "mixed")
CHANNEL_KINDS = ("channels", "nodes", "series", "stations")
FIDELITY = ("reference-checked", "paper-only", "inferred", "composed")

#: L1 (README body after the front matter) budget in lines; longer detail moves to reference.md.
L1_MAX_LINES = 60
#: A reference.md is allowed only when the full prose would not fit L1.
L2_MIN_LINES = 20
#: Fixed L1 section titles per kind, in order.
README_SECTIONS = {
    "model": ("Idea", "When to use", "Configure", "Differences"),
    "component": ("What it does", "When to use", "Interface"),
    "dataset": ("Overview", "Protocol and pitfalls"),
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _terms(values: list[str], *, allow_any: bool) -> list[str]:
    from tsflab.catalog.characteristics import term_problems

    problems = term_problems("card", "terms", values, allow_any=allow_any)
    if problems:
        raise ValueError("; ".join(problems))
    return values


class Issue(_Strict):
    kind: Literal[ISSUE_KINDS]  # type: ignore[valid-type]
    where: str = Field(min_length=1)
    what: str = Field(min_length=1)
    resolution: str = Field(min_length=1)


class Paper(_Strict):
    title: str = Field(min_length=1)
    url: str = Field(min_length=1)
    venue: str = Field(min_length=1)
    year: int


class Code(_Strict):
    url: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    license: str = Field(min_length=1)
    #: Upstream files inspected for the reference comparison, at ``revision``.
    reference_sources: list[str] = []


class DataParam(_Strict):
    #: Which data property the value follows (measured by ``tsf data analyze`` or the preset).
    from_: Literal[DATA_PARAM_SOURCES] = Field(alias="from")  # type: ignore[valid-type]
    rule: str = Field(min_length=1)
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class Admission(_Strict):
    """One executed check when the model entered the catalog (or last changed)."""

    status: Literal["passed", "failed", "pending"]
    date: str = ""
    commit: str = ""
    device: str = ""
    reference: Literal["official", "none"] = "none"
    note: str = ""


class ModelCard(_Strict):
    schema_: Literal[SCHEMA] = Field(SCHEMA, alias="schema")  # type: ignore[valid-type]
    kind: Literal["model"]
    name: str = Field(min_length=1)
    tags: list[str] = Field(min_length=3)
    fits: list[str] = []
    fidelity: Literal[FIDELITY]  # type: ignore[valid-type]
    #: Absent only for classical baselines and catalog constructs without a paper.
    paper: Paper | None = None
    code: Code | None = None
    #: Upstream files read when no ``[code]`` can be recorded (e.g. no license), at a pinned revision.
    inspected_sources: list[str] = []
    composition: dict[str, str]
    data_params: dict[str, DataParam] = {}
    issues: list[Issue] = []
    issues_checked: str = ""  # what was checked when `issues` is empty
    admission: Admission
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    @field_validator("fits")
    @classmethod
    def _fits(cls, value: list[str]) -> list[str]:
        return _terms(value, allow_any=False) if value else value

    @field_validator("composition")
    @classmethod
    def _composition(cls, value: dict[str, str]) -> dict[str, str]:
        if tuple(value) != SLOTS:
            raise ValueError(f"composition must list slots {', '.join(SLOTS)} in order")
        return value

    @model_validator(mode="after")
    def _consistency(self) -> "ModelCard":
        if not self.issues and not self.issues_checked:
            raise ValueError("record upstream issues, or say in issues_checked what was checked")
        if self.code is None and self.fidelity == "reference-checked":
            raise ValueError("reference-checked fidelity needs official code")
        return self


class ComponentCard(_Strict):
    schema_: Literal[SCHEMA] = Field(SCHEMA, alias="schema")  # type: ignore[valid-type]
    kind: Literal["component"]
    name: str = Field(min_length=1)
    role: str = Field(min_length=1, max_length=160)
    slot: Literal[COMPONENT_SLOTS]  # type: ignore[valid-type]
    fits: list[str] = Field(min_length=1)
    category: Literal[COMPONENT_CATEGORIES]  # type: ignore[valid-type]
    tags: list[str] = Field(min_length=3)
    input: str = Field(min_length=1)
    output: str = Field(min_length=1)
    origin: str = Field(min_length=1)
    origin_models: list[str] = []
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    @field_validator("fits")
    @classmethod
    def _fits(cls, value: list[str]) -> list[str]:
        return _terms(value, allow_any=True)


class Source(_Strict):
    name: str = Field(min_length=1)
    url: str = Field(min_length=1)
    citation: str = Field(min_length=1)
    citation_url: str = Field(min_length=1)
    license: str = Field(min_length=1)
    #: Evidence for ``license`` (terms page or license file).
    license_url: str = ""
    redistribution: Literal[REDISTRIBUTION]  # type: ignore[valid-type]
    #: Extra terms the source attaches to a re-host (attribution notices, verbatim copies only).
    conditions: str = ""


class Shape(_Strict):
    frequency: str
    time_span: str = ""
    length: int | str = ""
    channels: int | str = ""
    channel_kind: Literal[CHANNEL_KINDS] | Literal[""] = ""  # type: ignore[valid-type]
    target: str = ""
    missing_values: str = ""
    stats_basis: Literal[STATS_BASIS]  # type: ignore[valid-type]


class Protocol(_Strict):
    protocol: str = Field(min_length=1)
    literature: str = ""
    split: str = ""
    seq_lens: list[int] = []
    pred_lens: list[int] = []


class DatasetCard(_Strict):
    schema_: Literal[SCHEMA] = Field(SCHEMA, alias="schema")  # type: ignore[valid-type]
    kind: Literal["dataset", "dataset-family"]
    name: str = Field(min_length=1)
    domain: Literal[FAMILY_DOMAINS]  # type: ignore[valid-type]
    #: Free-text detail below the domain, e.g. "electricity transformer".
    topic: str = ""
    benchmarks: list[Literal[BENCHMARKS]] = []  # type: ignore[valid-type]
    tags: list[str] = Field(min_length=3)
    characteristics: list[str] = []
    characteristics_basis: str = ""
    related: list[str] = []
    realtime_track: str = ""
    source: Source
    shape: Shape
    protocol: Protocol
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    @field_validator("characteristics")
    @classmethod
    def _characteristics(cls, value: list[str]) -> list[str]:
        return _terms(value, allow_any=False) if value else value

    @model_validator(mode="after")
    def _basis(self) -> "DatasetCard":
        if self.characteristics and not self.characteristics_basis:
            raise ValueError("characteristics need characteristics_basis")
        if self.domain == "mixed" and self.kind != "dataset-family":
            raise ValueError(f"domain 'mixed' is only for a dataset family; choose one of {', '.join(DOMAINS)}")
        if len(set(self.benchmarks)) != len(self.benchmarks):
            raise ValueError("benchmarks must not repeat")
        return self


CARD_TYPES = {"model": ModelCard, "component": ComponentCard, "dataset": DatasetCard,
              "dataset-family": DatasetCard}
