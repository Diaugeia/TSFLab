# 1.0 static benchmark

Every catalog model on the 27 static datasets under each dataset card's
`[protocol]`, one shared training budget (`configs/base.toml`), seed 2024.

| File | Purpose |
| --- | --- |
| `make_runs.py` | Writes the run files and dataset overlays to `../runs/<phase>/` plus `plan.json` (cells, priorities, tier-1 models, excluded cells with reasons) |
| `watchdog.py` | Kills this user's largest run when the server's available memory drops below a floor (shared-server safety) |
| `report.py` | Per-run status, diagnosis, last error, train time and profile metrics of a phase (`--csv`) |
| `enqueue.py` | Prepares every run file of a phase and adds it to one queue with its priority (`queued.json` records the sweeps) |
| `policy-pilot.toml`, `policy-main.toml` | Execution policies for one 8-GPU server (the smoke phase uses the pilot policy) |

What the generator sets, because the sweep loader does not: per-dataset
lookbacks and horizons from the card; `enc_in` / `dec_in` / `c_out` /
`num_nodes` from the card's channel count; 288 steps per day for the 5-minute
spatial data; a `dataset.alias` so results do not merge under `custom`; one
batch size per dataset, shared by every model; the loss per output type
(`mse`, `quantile`, `nll_gaussian`; `masked_mae` for spatial); `evaluation.enable_profile`
(parameters, MACs, inference VRAM, latency) in every phase except the pilot. `--validate`
loads every (model, dataset) cell with the TSFLab loader and excludes the ones a
model's parameter schema rejects.

Phases: `pilot` (every model on ETTh1 96->96 and PEMS08 12->12), `smoke` (one epoch at the
longest horizon on traffic, nasdaq, ili, wike2000, pems07), `main` (first lookback, every horizon),
`lookback` (the other lookbacks).

Order: tier 1 (well-known representatives, at least one per architecture
category) before tier 2, and light datasets before heavy ones; `plan.json`
gives each run file a queue priority.

```bash
uv run python experiments/benchmark-1.0/scripts/make_runs.py --phase pilot --validate   # also: smoke, main, lookback; --models A B to re-check fixed models
uv run tsf run experiments/benchmark-1.0/runs/pilot/<file>.toml \
    --policy experiments/benchmark-1.0/scripts/policy-pilot.toml --gpus 0,1,2,3,4,5,6,7 --jobs 24
uv run python experiments/benchmark-1.0/scripts/make_runs.py --phase main --validate
# per run file, highest priority first:
uv run tsf run <file> --policy experiments/benchmark-1.0/scripts/policy-main.toml \
    --gpus 0,1,2,3,4,5,6,7 --jobs 24 --prepare-only --json
uv run tsf run --backend queue add <queue_dir> --run <sweep_dir> --priority <priority>
uv run tsf run --backend queue work <queue_dir> --slots 1
```

Run `watchdog.py` (with `setsid nohup`) next to every queue worker on a shared server.

Do not change `src/`, `uv.lock` or `pyproject.toml` while a phase runs: resume
refuses a sweep whose code fingerprint changed.
