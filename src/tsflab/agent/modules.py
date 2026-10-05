"""Module map: the single source of which module owns each skill and task template.

The chain is Data -> Models -> Experiments -> Release, each producing context that
AutoResearch consumes. Maintenance sits beside the chain (audit, contributions) and
is not part of a standalone project unless chosen. ``.agents/README.md`` is generated
from this map (``tsf repo cards``) and ``tsflab.agent.assets`` rejects any skill or
task that is missing from it or listed twice.

A project created by ``tsf init`` records its chosen modules under
``[tool.tsflab]`` in its ``pyproject.toml``; ``tsf agent`` and ``tsf catalog``
read that record. Skills and task templates for the chosen modules live in the
project's ``.agents/`` directory and are refreshed from the installed package
with ``tsf agent sync``.
"""

from __future__ import annotations

from pathlib import Path
import tomllib

# module -> skills, tasks, pip extras, purpose, entry commands, context it produces
MODULES: dict[str, dict[str, object]] = {
    "data": {
        "skills": ["add-dataset", "inspect-dataset"],
        "tasks": [],
        "extras": ["data"],
        "purpose": "register, prepare, profile, and publish datasets",
        "commands": ["tsf data inspect", "tsf data analyze", "tsf data prepare", "tsf data add"],
        "context": "dataset cards (characteristics, protocol) and presets; train-only profile `work_dirs/profiles/<name>/profile.{json,md}`",
    },
    "models": {
        "skills": ["discover-papers", "add-model", "integrate-foundation-model", "curate-components"],
        "tasks": ["intake"],
        "extras": ["models"],
        "purpose": "find papers, implement or adapt models, curate reusable components",
        "commands": ["tsf model scaffold", "tsf model add", "tsf model verify", "tsf model compose"],
        "context": "model cards (description, fits, fidelity, six-slot composition, data_params), component interfaces, admission records",
    },
    "experiments": {
        "skills": ["setup-environment", "run-experiment", "diagnose-experiment",
                   "reproduce-paper-results", "analyze-results"],
        "tasks": ["experiment"],
        "extras": ["experiments"],
        "purpose": "design, run, diagnose, and analyze budgeted experiments",
        "commands": ["tsf env", "tsf run", "tsf result aggregate", "tsf result board"],
        "context": "run records `work_dirs/<dataset>/<model>/records/<run_id>.json` and CSVs; the result board",
    },
    "release": {
        "skills": ["submit-results", "forecast-realtime-round", "publish-weights"],
        "tasks": [],
        "extras": ["hub", "realtime"],
        "purpose": "submit results, forecast real-time rounds, publish weights",
        "commands": ["tsf result submit", "tsf result hub", "tsf realtime forecast"],
        "context": "leaderboard `board/leaderboard.json` of `TSFLab-Checkpoints`, real-time scores, pinned `hf://` checkpoint URIs",
    },
    "autoresearch": {
        "skills": ["run-autoresearch"],
        "tasks": ["autoresearch"],
        "extras": ["autoresearch"],
        "purpose": "budgeted research loops that consume the other modules' context",
        "commands": ["tsf catalog match", "tsf research start", "tsf research iteration", "tsf agent task start autoresearch"],
        "context": "round ledger `work_dirs/_research/<id>/` (hypotheses, runs, conclusions) and winning composition specs",
    },
    "maintenance": {
        "skills": ["audit", "handle-contribution"],
        "tasks": ["maintenance", "contribution"],
        "extras": [],
        "purpose": "keep catalog, cards, admission records, and Agent assets consistent; triage issues and pull requests",
        "commands": ["tsf repo check", "tsf repo cards", "tsf model audit", "tsf agent task validate"],
        "context": "gate results, card audits, and regenerated indexes",
    },
}
CHAIN = ("data", "models", "experiments", "release", "autoresearch")  # the default for `tsf init`
ALL_MODULES = tuple(MODULES)  # the chain plus maintenance (opt-in: `--modules ...,maintenance`)


def owners(kind: str) -> dict[str, list[str]]:
    """Map each skill (``kind='skills'``) or task (``'tasks'``) name to the modules listing it."""
    result: dict[str, list[str]] = {}
    for module, info in MODULES.items():
        for name in info[kind]:  # type: ignore[union-attr]
            result.setdefault(name, []).append(module)
    return result


def parse_modules(text: str | None) -> list[str]:
    """Parse ``a,b,c`` (default: the chain, without maintenance) and reject unknown names."""
    if text is None or text.strip() in {"", "all"}:
        return list(CHAIN)
    names = [part.strip() for part in text.split(",") if part.strip()]
    unknown = [name for name in names if name not in MODULES]
    if unknown:
        raise ValueError(f"unknown module(s): {', '.join(unknown)}; choose from {', '.join(ALL_MODULES)}")
    return [name for name in ALL_MODULES if name in names]


def module_tasks(modules: list[str]) -> list[str]:
    return [task for name in modules for task in MODULES[name]["tasks"]]  # type: ignore[union-attr]


def module_skills(modules: list[str], task_skills: dict[str, list[str]] | None = None) -> list[str]:
    """Skills of the modules plus any skill a selected task template requires."""
    skills: list[str] = []
    for name in modules:
        skills.extend(MODULES[name]["skills"])  # type: ignore[arg-type]
    for task in module_tasks(modules):
        skills.extend((task_skills or {}).get(task, []))
    return list(dict.fromkeys(skills))


def find_project(start: Path | None = None) -> tuple[Path, list[str]] | None:
    """Return ``(project_dir, modules)`` for the nearest ``[tool.tsflab]`` pyproject."""
    here = (start or Path.cwd()).resolve()
    for directory in (here, *here.parents):
        pyproject = directory / "pyproject.toml"
        if not pyproject.is_file():
            continue
        try:
            table = tomllib.loads(pyproject.read_text(encoding="utf-8")).get("tool", {}).get("tsflab")
        except (OSError, tomllib.TOMLDecodeError):
            continue
        if isinstance(table, dict) and isinstance(table.get("modules"), list):
            return directory, [m for m in table["modules"] if m in MODULES]
    return None


def project_modules(start: Path | None = None) -> list[str] | None:
    found = find_project(start)
    return found[1] if found else None
