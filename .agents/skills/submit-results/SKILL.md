---
name: submit-results
description: "Package completed TSFLab runs and their research evidence as TSFLab Leaderboard submission bundles and publish them to results/ of Diaugeia/TSFLab-Checkpoints. Use for local submission bundles, bulk packaging of a sweep, or leaderboard contribution; uploads and pull requests require explicit authorization. Not for live real-time forecasts (forecast-realtime-round) or weights (publish-weights)."
---

# Submit results

Release module: turn finished runs into validated submission bundles that the
leaderboard recomputes from. Results live on the Hugging Face Hub in
`results/<track>/<dataset>/<model>/<submission_id>/` of `Diaugeia/TSFLab-Checkpoints`;
the site reads the generated `board/leaderboard.json` there. GitHub keeps code only.

## Inputs

- Completed runs (`work_dirs/<dataset>/<model>/records/<run_id>.json`), ideally
  executed inside a research round so their trajectory is captured. Runs execute on a
  GPU machine or CI (see `run-experiment`); packaging happens locally from the
  returned `work_dirs/` and does no training.

## Steps

1. Open a round before the run when possible (`tsf research start --task submission
   --goal <goal> --max-runs <count>`, then `tsf run <cfg> --round <id>` on the run
   machine, then `tsf research status <id> completed --message <conclusion>`). Without
   one the bundle's trajectory is marked synthetic. Package the finished records:

   ```bash
   uv run tsf result submit --dataset <dataset> --model <model> [--run-id <run_id> | --latest]
   uv run tsf result submit --all [--dataset <dataset>] [--model <model>] --skip-existing   # a whole sweep
   ```
   Bundles land in `work_dirs/_submissions/`; `--all` reads the research rounds once
   and reports built, kept, and failed counts.

2. Inspect `submission.json`, `trajectory.jsonl`, and `report.md` of a sample.
   Confirm dataset version, run identity, metrics, and whether the trajectory is
   synthetic. Preview the board with the same contract the site uses:

   ```bash
   uv run tsf result leaderboard --source work_dirs/_submissions --out work_dirs/_board/leaderboard.json
   ```

3. Publish (maintainers, with authorization and a write token). Plan first; the
   upload skips bundles already present and regenerates `board/`:

   ```bash
   uv run tsf result hub results push work_dirs/_submissions --dry-run
   uv run tsf result hub results push work_dirs/_submissions
   ```

   A contributor without Hub access instead opens a pull request that adds the
   bundle under `apps/web/submissions/<track>/<dataset>/<model>/<id>/` (a staging
   folder, checked by `uv run tsf repo check --only web-submissions schema-export`);
   a maintainer moves accepted bundles to the Hub with the same `results push`.

4. Weights are never part of a submission; publish the top-ranked checkpoints
   separately with `publish-weights` when reproducibility needs them.

## Chain

- Module: Release.
- Reads: completed records, their round trajectories, the result board.
- Produces: bundles under `results/` of `Diaugeia/TSFLab-Checkpoints` and a regenerated `board/leaderboard.json`.
- Hands off to: `run-autoresearch` reads the leaderboard as a reference bar; weights go to `publish-weights`.

## Success

- Contract-valid bundles whose rows appear in the regenerated board.

## Stop and hand off

- An upload, branch, issue, or pull request requires explicit approval.
- Forecasts for live rounds use `forecast-realtime-round`, not this skill.
