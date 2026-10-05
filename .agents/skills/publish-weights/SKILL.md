---
name: publish-weights
description: "Package the best checkpoints of top-ranked TSFLab runs as checksummed safetensors bundles and publish or load them through pinned hf:// URIs in Diaugeia/TSFLab-Checkpoints. Use for sharing or reloading trained weights; not for leaderboard result submissions or pretrained foundation checkpoints."
---

# Publish weights

Release module: turn finished runs into reproducible weights bundles at
`checkpoints/<track>/<dataset>/<model>/<run_id>/` (manifest, `model.safetensors`,
run record, card) in `Diaugeia/TSFLab-Checkpoints`, addressed by a URI pinned to an
immutable revision. Only top-ranked runs are uploaded, never every run of a sweep.

## Inputs

- Completed runs: records at `work_dirs/<dataset>/<model>/records/<run_id>.json` and
  checkpoints from the managed runner at `work_dirs/_runs/<run_id>/checkpoints/`
  (`result.json` names the best one). The older
  `work_dirs/<dataset>/<model>/checkpoints/<run_id>/` layout also works. Checkpoints
  are produced on the training machine and must be copied back with the records.
- Write credentials (`HF_TOKEN` or a local Hub login) and explicit authorization to
  publish. Requires the `hub` extra.

## Steps

1. Show the selection; nothing is uploaded:

   ```bash
   uv run tsf result hub push-top --dataset <dataset> --horizon <H> --top <K> --dry-run
   ```

   Rows are ranked with `tsf result board` (seed averages); each selected entry is
   the best single run of a top row. Check the model, horizon, metric, path, and
   that every entry says `ready` (a missing checkpoint is skipped).
2. To inspect one bundle first, build it locally and read `manifest.json` (model,
   dataset, horizon, seed, metrics, framework version, commit, SHA-256 of every file):

   ```bash
   uv run tsf result hub pack <run_id>            # -> work_dirs/_bundles/<run_id>/
   ```

3. Publish the selection in one commit; bundles already present are skipped:

   ```bash
   uv run tsf result hub push-top --dataset <dataset> --horizon <H> --top <K>
   ```

   For one specific run use `uv run tsf result hub push <run_id>`, which prints the
   pinned `hf://Diaugeia/TSFLab-Checkpoints@<revision>/checkpoints/...` URI.
4. Reference the URI where the weights matter, for example a real-time forecast's
   `weights_uri`, and verify it round-trips:

   ```bash
   uv run tsf result hub list --dataset <dataset>
   uv run tsf result hub pull hf://Diaugeia/TSFLab-Checkpoints@<revision>/checkpoints/<track>/<dataset>/<model>/<run_id>
   ```

## Chain

- Module: Release.
- Reads: run records, their checkpoints, the result board ranking.
- Produces: pinned `hf://Diaugeia/TSFLab-Checkpoints@<revision>/checkpoints/...` URIs with checksummed manifests.
- Hands off to: `forecast-realtime-round` (`weights_uri`), `run-experiment` (reload), `submit-results`.

## Success

- A pinned URI whose pull verifies every tensor against the manifest checksums.

## Stop and hand off

- Never publish without explicit authorization or push runs that were not
  requested or not top-ranked; stop on checksum mismatch.
- Result bundles for the leaderboard belong to `submit-results`; released
  pretrained runtimes to `integrate-foundation-model`.
