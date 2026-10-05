# Submissions staging folder

This folder is **not** where results are stored. Results live on the Hugging Face
Hub, in the model repository
[`Diaugeia/TSFLab-Checkpoints`](https://huggingface.co/Diaugeia/TSFLab-Checkpoints):

```
results/<track>/<dataset>/<model>/<submission_id>/   submission.json, trajectory.jsonl, report.md
checkpoints/<track>/<dataset>/<model>/<run_id>/      weights of top-ranked runs (optional)
board/leaderboard.json, board/model-meta.json        generated; the site reads them
legacy/                                              TSEval-era archive, not ranked
```

The folder has two uses:

1. **External submissions arrive here by pull request.** A contributor without
   write access to the Hub adds one bundle at
   `apps/web/submissions/<track>/<dataset>/<model>/<submission_id>/`. The PR check
   (`tsf repo check`, step `web-submissions`) validates it against the TSF-Core
   contract. After review, a maintainer moves it to the Hub and deletes it here:

   ```bash
   uv run tsf result hub results push apps/web/submissions --dry-run
   uv run tsf result hub results push apps/web/submissions   # uploads + regenerates board/
   git rm -r apps/web/submissions/<track>/<dataset>/<model>/<submission_id>
   ```

2. **Real-time rounds.** The `weekly` workflow writes
   `realtime/<track>/rounds/<round_id>/` (round, forecasts, scores) here; see
   `docs/en/realtime.md`. These stay in Git.

Why a Git staging folder: a pull request gives an external contributor a
reviewable, CI-checked path that needs no Hub token, and keeps the Hub
repository write-only for maintainers. The board is built only from the Hub, so
a bundle shows on the site after a maintainer pushes it.

Format and averaging rules: [`../SUBMITTING.md`](../SUBMITTING.md).
