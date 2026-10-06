# Agent-cost experiment

Measures what research tasks cost a coding agent in TSFLab and in other
environments. Supports §4.1 (research tasks) and Appendix C (measurement
procedure) of the paper.

## Tasks

| Task | Input to the agent | Success (checked by `check_success.py`) |
|---|---|---|
| `reproduce` | Link to one recent forecasting paper without official code, not in TSFLab; datasets and prediction lengths of its main table | Finite MSE and MAE for every cell in `results/results.json`, and `results/report.md` |
| `autoresearch` | One dataset, the best TSFLab method on it, its validation MSE, a budget | A method with 3 validation seeds below the target at every prediction length, and one test read per seed |

Prompts are in `prompts/`; a task is a JSON file (see `tasks/example-reproduce.json`).

## Arms (`make_env.sh`)

| Arm | Working directory |
|---|---|
| `tsflab` | TSFLab at a fixed commit |
| `tsflab-noknow` | Same code and skills; every `README.md` and `reference.md` of a method, component, or dataset deleted; `tsf catalog` disabled. `card.toml` stays because data fetching and admission read it. |
| `tsflab-flat` | Same content as `tsflab`, but concatenated into one `KNOWLEDGE.md` (no layered access); `tsf catalog` disabled |
| `tsflab-notools` | Same knowledge and method code; `tsf run`, `tsf result`, `tsf data` disabled, runner entry points deleted, experiment skills removed |
| `tsflab` + `TSFLAB_COMMIT` | An earlier version of TSFLab (interface-version experiment) |
| `tslib` | Time-Series-Library at `TSLIB_COMMIT` |
| `tfb` | TFB at `TFB_COMMIT` |
| `empty` | Empty directory with a minimal PyTorch environment |

Every arm gets the same raw data directory (`DATA_DIR`), the same prompt, the
same model, and network access. The Python environment is installed before the
session, so installation is not measured.

## Agents

`AGENT=claude` (default) runs Claude Code; `AGENT=codex` runs Codex CLI
(`codex exec --json`, default model `gpt-6-luna`) with `CODEX_HOME` in the run
directory, linked to the login in `~/.codex`. `INSTALL=server` installs torch
for CUDA 12.6. `launch.sh <task> <out-root> <spec>...` starts one session per GPU,
detached; a spec is an arm or `arm@commit`.

## Running one session

```bash
DATA_DIR=/path/to/raw-data \
  experiments/agent-cost/run_session.sh tasks/<task>.json <arm> runs/<run-id> [model]
```

Options: `TIMEOUT_H` (default 8), `MAX_RESUMES` (default 2), `MAX_BUDGET_USD`,
`GPU_SAMPLE_S` (default 5), `TSLIB_COMMIT`, `SKIP_INSTALL=1` (tests only).

The session runs Claude Code headless with an isolated configuration
(`CLAUDE_CONFIG_DIR` holding only the login credentials, `--strict-mcp-config`):
no user `CLAUDE.md`, memory, plugins, or MCP servers. The project's own
`CLAUDE.md`/`AGENTS.md` and skills load as in normal use. If the session stops
before the task is complete, it is resumed with the fixed message "Continue
until the task is complete."; each resume counts as one human intervention.

## Outputs per run

| File | Content |
|---|---|
| `stream.<n>.jsonl` | Claude Code stream-json, one event per line, with arrival time `t` |
| `stages.tsv` | Stage markers written by the agent (`stage <name>`) |
| `interventions.tsv` | Resumes |
| `gpu.csv` | GPU utilization and memory samples |
| `diff.numstat`, `diff.patch` | Changes against the baseline commit |
| `success.json` | Result of `check_success.py` |
| `metrics.json` | Result of `analyze.py` |

## Metrics (`analyze.py`)

- Success, wall-clock time, cost (USD, from Claude Code), turns, interventions.
- Every tool call is classified as `paper` (reading the target paper), `read`
  (reading or searching the environment: knowledge, code, documentation),
  `write`, `run`, or `other`.
- **Discovery cost** = tokens and time of `read` calls, and their share of the total.
- Input tokens come from each model turn; output tokens are estimated from the
  turn's content length and scaled to the session total.
- Time per stage, tokens per stage, GPU-busy seconds (utilization at least 10%).
- Code changes by kind (code, config, docs, results).
- Discovery errors (a re-implemented existing block, data prepared differently
  from the protocol, a private evaluation loop) are judged from `diff.patch` by
  review and recorded separately.

## Where to run

- Pipeline tests (`SKIP_INSTALL=1`, small budget): any machine.
- Reproduction sessions train models on one GPU: local GPU when it is free,
  otherwise one GPU on the server.
- Autoresearch sessions need more GPU time: server.
