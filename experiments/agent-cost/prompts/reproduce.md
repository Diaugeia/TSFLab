Reproduce the main results of this forecasting paper: {paper_url}

The current directory is your working directory. If it contains a codebase,
build on it and follow its structure, naming, and conventions instead of
starting from scratch. If it is empty, write everything yourself.

Raw data files for the datasets are in {data_dir} (read-only). You may also
download data if you need to. A GPU is available.

Scope: reproduce the paper's main results table for these datasets and
prediction lengths: {cells}. Report MSE and MAE for each cell, using the
paper's own evaluation protocol as far as the paper describes it.

Rules:
- Work without asking questions. When the paper leaves a detail open, choose a
  reasonable value and record the choice in results/decisions.md.
- At the start of each stage, run the shell command `stage <name>` with one of:
  collect (get the paper, data, and settings), read (understand the method),
  implement, check (make sure the code runs end to end), run (the experiments),
  compare (compare with the paper). Run `stage done` when you finish.
- Write results/results.json as a JSON list of objects
  {{"dataset": str, "pred_len": int, "mse": float, "mae": float}}, one per cell.
- Write results/report.md: what you implemented, the results next to the
  paper's numbers, and every deviation from the paper.

You are finished when every cell above has a result in results/results.json
and results/report.md is written.
