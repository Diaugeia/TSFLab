# TSFLab documentation

← [Project README](../../README.md)

Human documentation is intentionally small. Command syntax and defaults are
available from `uv run tsf --help` and each subcommand's `--help`; model,
component, and dataset details live in their cards.

- [workflows.md](workflows.md): model interfaces, adding models, offline official
  foundation runtimes, artifacts, components, reading the catalog by depth, datasets
  (93 presets grouped by domain, one TSFLab protocol each), cards, experiments, AutoResearch, and admission.
- [execution.md](execution.md): optional environment audits, tracking, budgets, GPU scheduling, recovery, and independently usable Python modules.
- [hub.md](hub.md): standalone projects (`tsf init`), `hf://` addressing, and publishing weights bundles.
- [realtime.md](realtime.md): rolling real-time tracks, weekly rounds, forecast submissions, scoring, and replay.
- [models.md](models.md): generated table of every model with its description, fidelity, and admission status.

## Installation

TSFLab pins `torch==2.14.1` and installs with [uv](https://docs.astral.sh/uv/)
only. Pick the install profile that matches the NVIDIA driver (`nvidia-smi`
shows the maximum CUDA version), or let `bash scripts/detect_hardware.sh` pick it:

| Profile | Machine | Commands |
| --- | --- | --- |
| `cuda130` | driver >= 580 (CUDA 13.x, e.g. RTX 50-series) | `uv sync --frozen --python 3.12 --extra models --extra data --extra experiments` |
| `cuda126` | driver >= 560 (CUDA 12.6-12.9, e.g. driver 570) | `uv venv --python 3.12`<br>`uv pip install --torch-backend cu126 -e ".[models,data,experiments]" pytest`<br>`export UV_NO_SYNC=1` |
| `cpu` | laptops, CI, no NVIDIA GPU | Linux: as `cuda126` with `--torch-backend cpu`; macOS/Windows: as `cuda130` |

`uv.lock` resolves the cu130 build, and `uv sync` always installs the locked
build: `UV_TORCH_BACKEND` and `--torch-backend` only affect `uv pip`. torch 2.14.1
has no cu128 wheel, so CUDA 12.8 drivers use `cuda126`. After a `uv pip` install,
keep `UV_NO_SYNC=1` (or `uv run --no-sync`) so `uv run` does not re-sync the lock.

Then check the machine and get the matching run profile:

```bash
uv run tsf env doctor          # OS, driver, GPUs, torch build, install + run profile
uv run tsf env doctor --json
```

Run profiles are execution policies in `configs/execution/` (`laptop-cpu.toml`,
`single-gpu.toml`, `multi-gpu.toml`) for `tsf run <experiment.toml> --policy <file>`;
they set concurrency, GPU sharing, and per-run thread caps. Device and dataloader
workers stay in the run config (`[experiment.runtime] device`, `num_workers`).
See [execution.md](execution.md).

## Quick start

```bash
uv run tsf catalog
uv run tsf catalog list --kind dataset
uv run tsf run configs/runs/run_single_data.toml --dry-run
uv run tsf run configs/runs/run_single_data.toml
```

## Commands

Eleven commands, one per module (`tsf <command> --help`):

| Command | Purpose |
| --- | --- |
| `tsf catalog` | overview, `search`, `list`, `show <name> [--kind] [--depth]`, `match <dataset>` for models, components, datasets |
| `tsf data` | `add`, `prepare [--from traffic\|ultratraffic\|gift-eval\|tfb\|dcrnn]`, `inspect`, `analyze`, `plot`, `download`, `publish`, `audit` |
| `tsf model` | `scaffold`, `add [--verify]`, `artifacts`, `verify <Name...>\|--all\|--changed`, `compose`, `audit [--components] [--release]` |
| `tsf run` | run configs; `--smoke`, `--dry-run`, `--backend local\|queue\|slurm` |
| `tsf env` | environment audit; `storage` and `usage` subcommands |
| `tsf result` | `aggregate`, `rank`, `plot`, `report`, `predictions`, `board`, `submit`, `leaderboard`, `hub` |
| `tsf realtime` | rolling tracks: `update [--bootstrap] [--push]`, `open`, `forecast`, `score` |
| `tsf research` | research rounds: `start`, `list`, `show`, `note`, `status`, `iteration` |
| `tsf repo` | `check [--scope full\|changed\|release] [--audit] [--contracts LEVEL]`, `cards`, `schema` |
| `tsf agent` | `task`, `interface`, `modules`, `sync` |
| `tsf init` | scaffold a project for chosen modules |

Install by module with extras: `data`, `models`, `experiments`, `hub`, `realtime`,
`autoresearch`, `all`.
