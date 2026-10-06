Find a new forecasting method that beats {target_method} on {dataset}.

The current directory is your working directory. If it contains a codebase,
build on it and follow its structure, naming, and conventions instead of
starting from scratch. If it is empty, write everything yourself.

Raw data files are in {data_dir} (read-only). A GPU is available.
Target: {target_method}, the best method on the current leaderboard for
{dataset}. First run it under the codebase's protocol ({split}) at prediction
lengths {pred_lens} and record its validation MSE (method name
"{target_method}", one row per seed) in results/results.json; that is the
target to beat.

Rules:
- Work without asking questions. Record every hypothesis, decision, and run in
  results/ledger.md.
- Select candidates on validation data only. Confirm your final method over
  three seeds. Read test metrics once, only for the final method.
- At the start of each stage, run `stage <name>` with one of: collect, read,
  hypothesize, implement, search, confirm, report. Run `stage done` when you
  finish.
- Stop when you have a confirmed method that beats the target on validation,
  or when you reach the budget of {budget}.
- Write results/results.json as a JSON list of objects
  {{"method": str, "seed": int, "pred_len": int, "split": "val"|"test",
  "mse": float, "mae": float}}, and results/report.md describing the final
  method.
