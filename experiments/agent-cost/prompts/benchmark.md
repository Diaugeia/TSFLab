Benchmark these forecasting methods on a new dataset: {methods}.

The current directory is your working directory. If it contains a codebase,
build on it and follow its structure, naming, and conventions instead of
starting from scratch. If it is empty, write everything yourself.

The dataset is {file} in {data_dir} (read-only): an hourly multivariate series,
one `date` column and one column per channel. One GPU is available to you: the
one already selected by CUDA_VISIBLE_DEVICES. Use only that GPU and do not
change CUDA_VISIBLE_DEVICES.

Protocol (the same for every method):
- Split the series chronologically into 70% train, 10% validation, and 20% test.
- Fit normalization statistics on the training split only.
- Input length {seq_len}, prediction length {pred_len}, all channels.
- Evaluate every test window (stride 1) and report MSE and MAE on the
  normalized scale.
- Use each method's standard architecture settings. Train for at most 10
  epochs with early stopping on validation loss (patience 3), one seed.

Rules:
- Work without asking questions. Record every choice in results/decisions.md.
- At the start of each stage, run the shell command `stage <name>` with one of:
  collect (get the data and the methods), read (understand what is needed),
  implement, check (make sure the code runs end to end), run (the experiments),
  compare (rank the methods). Run `stage done` when you finish.
- Write results/results.json as a JSON list of objects
  {{"method": str, "mse": float, "mae": float}}, one per method.
- Write results/report.md: how each method was run, the results ranked by MSE,
  and every deviation from the protocol.

You have up to four hours. You are finished when every method above has a
result in results/results.json and results/report.md is written.
