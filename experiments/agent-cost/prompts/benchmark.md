Benchmark these forecasting methods on a new dataset: {methods}.
{papers}
The current directory is your working directory. If it contains a codebase,
build on it and follow its structure, naming, and conventions instead of
starting from scratch. If it is empty, write everything yourself.

The dataset is {file} in {data_dir} (read-only): an hourly multivariate series,
one `date` column and one column per channel. One GPU is available to you: the
one already selected by CUDA_VISIBLE_DEVICES. Use only that GPU and do not
change CUDA_VISIBLE_DEVICES. No Python environment is installed: set one up in
the working directory yourself (uv and Python 3.10-3.12 are available).

Protocol (the same for every method):
- Split the series chronologically into 70% train, 10% validation, and 20% test.
- Fit normalization statistics on the training split only.
- Input length {seq_len}, prediction length {pred_len}, all channels.
- Evaluate every test window (stride 1) and report MSE and MAE on the
  normalized scale.
- Use each method's standard architecture settings from its paper or official
  code. Train for at most 10 epochs with early stopping on validation loss
  (patience 3), batch size 32, one seed.
- Measure for each method: trainable parameters; training time (wall-clock
  seconds for the whole training, and the number of epochs run); inference
  time (wall-clock seconds to predict every test window); and peak GPU memory
  during training (MB).

Rules:
- Work without asking questions. Record every choice in results/decisions.md.
- At the start of each stage, run the shell command `stage <name>` with one of:
  collect (get the data and the methods), read (understand what is needed),
  implement, check (make sure the code runs end to end), run (the experiments),
  compare (rank the methods). Run `stage done` when you finish.
- Write results/results.json as a JSON list of objects
  {{"method": str, "mse": float, "mae": float, "params": int, "train_s": float,
  "epochs": int, "infer_s": float, "peak_mem_mb": float}}, one per method.
- Write results/report.md: how each method was run, the results ranked by MSE
  with the measured cost of each method, and every deviation from the protocol.

You have up to six hours. You are finished when every method above has a
result in results/results.json and results/report.md is written.
