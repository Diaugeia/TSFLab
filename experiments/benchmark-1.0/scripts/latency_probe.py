"""Inference latency and peak memory on an otherwise idle GPU (no training).

    python experiments/benchmark-1.0/scripts/make_runs.py --phase main --cells probe.csv \
        --out experiments/benchmark-1.0/runs/probe
    CUDA_VISIBLE_DEVICES=<i> python experiments/benchmark-1.0/scripts/latency_probe.py \
        experiments/benchmark-1.0/runs/probe --shard <i>/8 --out probe-<i>.csv

For every run configuration in the folder (sweeps expanded with the TSFLab loader),
it builds the test loader and the model exactly as the runner does
(``run_one._build_loaders`` / ``_build_model``), skips training, and runs the
runner's own ``profile_model`` on the first test batch: parameters, MACs, peak and
reserved VRAM, mean latency, throughput. Latency does not depend on the weights, so
untrained models give the same numbers. Models that need a training-split setup
before the first forward (``training_setup``) are reported with their error. Run one
process per GPU and nothing else on that GPU.
"""

from __future__ import annotations

import argparse
import csv
import tempfile
import traceback
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", type=Path, help="folder of run files (make_runs.py --cells ... --out)")
    ap.add_argument("--shard", default="0/1", help="i/n: this process probes every n-th configuration")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import torch

    from tsflab.experiments.config.loader import load_config
    from tsflab.experiments.evaluation.profile import parse_profile_report_file, profile_model
    from tsflab.experiments.runner import run_one

    index, total = (int(x) for x in args.shard.split("/"))
    configs = []
    for path in sorted(args.runs.glob("*.toml")):
        for loaded in load_config(str(path)):
            configs.append(loaded)
    mine = configs[index::total]
    fields = ["model", "dataset", "seq_len", "pred_len", "batch_size", "status", "error", "total_params",
              "total_macs_m", "peak_vram_mb", "reserved_vram_mb", "latency_avg_ms", "throughput_samples_sec"]
    with args.out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for loaded in mine:
            config = getattr(loaded, "config", loaded)
            row = {"model": config.model.name, "dataset": config.dataset.alias or config.dataset.name,
                   "seq_len": config.task.seq_len, "pred_len": config.task.pred_len,
                   "batch_size": config.training.batch_size}
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                device = run_one._build_device(config.experiment.runtime)
                train_set, _, _, _, _, test_loader, adj_norm = run_one._build_loaders(config)
                model, _ = run_one._build_model(config, train_set, adj_norm, device)
                with tempfile.TemporaryDirectory() as tmp:
                    report = Path(tmp) / "profile.txt"
                    profile_model(model=model, data_loader=test_loader, device=device,
                                  label_len=config.task.label_len, pred_len=config.task.pred_len,
                                  save_path=str(report))
                    row.update(parse_profile_report_file(str(report)))
                row["status"] = "ok"
                del model
            except Exception as exc:  # recorded per cell; the probe continues
                row["status"] = "failed"
                row["error"] = (f"{type(exc).__name__}: {exc}".splitlines() or [""])[0][:200]
                traceback.print_exc()
            writer.writerow(row)
            stream.flush()
            print(row["model"], row["dataset"], row["status"], row.get("latency_avg_ms"), flush=True)


if __name__ == "__main__":
    main()
