"""Profile a dataset for model selection: ``tsf data analyze <preset|--path FILE>``.

Loads the chronological train/val/test split of a dataset preset (same loaders and
split rules as training, but unscaled) or of a raw file/store, then writes a
structured JSON profile plus a short markdown summary. See
``tsflab.data.profile`` for the metrics and ``profile_rules.toml`` for the
profile-to-catalog mapping. Analysis only; it never trains or modifies data.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from tsflab.data import profile as prof

DEFAULT_RATIO = (0.7, 0.1, 0.2)


def _full_array(ds) -> np.ndarray:
    raw = getattr(ds, "data", None)
    if raw is None:
        raw = getattr(ds, "values", None)
    if raw is None:
        raise RuntimeError("dataset exposes neither `.data` nor `.values`; use --path FILE")
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 3:  # (T, N, features): channel 0 is the target value
        arr = arr[..., 0]
    return arr[:, None] if arr.ndim == 1 else arr


def _extract_splits(datasets: dict[str, object]) -> dict[str, np.ndarray]:
    """Chronological train/val/test arrays from three split-flagged datasets.

    Window-indexed loaders (CauAir, real-time panels) share one full array; their
    split boundaries are the first window centre of val and test.
    """
    if hasattr(datasets["val"], "idx"):
        full = _full_array(datasets["val"])
        b1 = int(np.asarray(datasets["val"].idx).min())
        b2 = int(np.asarray(datasets["test"].idx).min())
        return {"train": full[:b1], "val": full[b1:b2], "test": full[b2:]}
    return {flag: _full_array(ds) for flag, ds in datasets.items()}


def _frequency_from_file(ds_path: Path) -> Any:
    import pandas as pd

    try:
        if ds_path.suffix == ".csv":
            head = pd.read_csv(ds_path, usecols=[0], nrows=500)
            return pd.to_datetime(head.iloc[:, 0], errors="coerce")
        if ds_path.suffix == ".parquet":
            return pd.read_parquet(ds_path).index[:500]
    except Exception:
        return None
    return None


def _freq_override(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    import pandas as pd

    step = pd.to_timedelta(pd.tseries.frequencies.to_offset(text)).total_seconds()
    return prof.infer_frequency(pd.date_range("2000-01-01", periods=8, freq=f"{int(step)}s"))


def load_preset(name_or_path: str, freq: str | None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load train/val/test (unscaled) for a dataset preset TOML."""
    from tsflab.cli.commands import dataset_characteristics as dc

    path = Path(name_or_path)
    if not path.suffix:
        path = Path("configs/datasets") / f"{name_or_path}.toml"
    if not path.is_file():
        raise SystemExit(f"unknown dataset preset {name_or_path!r} (looked for {path})")
    partial = dc._load_partial_config(str(path))
    params = dict(partial.dataset.params)
    if "scale" in params:
        params["scale"] = False
    config = dc._PartialConfig(
        dataset=dataclasses.replace(partial.dataset, params=params), task=partial.task
    )
    datasets = {flag: dc._build_dataset(config, flag) for flag in ("train", "val", "test")}
    splits = _extract_splits(datasets)
    file_path = getattr(datasets["train"], "file_path", "")
    meta: dict[str, Any] = {
        "name": config.dataset.alias or config.dataset.name,
        "source": f"{path} -> {config.dataset.path}",
        "split_ratio": params.get("split_ratio"),
    }
    freq_info = _freq_override(freq)
    if freq_info is None:
        stamps = _frequency_from_file(Path(file_path)) if file_path else None
        freq_info = prof.infer_frequency(stamps)
    meta["frequency"] = freq_info
    return splits, meta


def load_path(
    file: str, ratio: tuple[float, float, float], freq: str | None
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load a csv/txt/parquet/npy/npz file or a realtime store directory and split it."""
    import pandas as pd

    p = Path(file)
    stamps = None
    if p.is_dir():
        panels = sorted((p / "panel").glob("*.parquet")) or sorted(p.glob("*.parquet"))
        if not panels:
            raise SystemExit(f"{p} has no panel/*.parquet files")
        frame = pd.concat([pd.read_parquet(f) for f in panels]).sort_index()
        stamps, values = frame.index, frame.to_numpy(np.float64)
    elif p.suffix in {".csv", ".txt"}:
        frame = pd.read_csv(p, header=None if p.suffix == ".txt" else "infer")
        first = frame.iloc[:, 0]
        if not pd.api.types.is_numeric_dtype(first):
            stamps = pd.to_datetime(first, errors="coerce")
            frame = frame.iloc[:, 1:]
        values = frame.apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    elif p.suffix == ".parquet":
        frame = pd.read_parquet(p)
        stamps, values = frame.index, frame.select_dtypes("number").to_numpy(np.float64)
    elif p.suffix in {".npy", ".npz"}:
        loaded = np.load(p, allow_pickle=False)
        values = np.asarray(loaded if p.suffix == ".npy" else loaded[loaded.files[0]], np.float64)
        values = values[..., 0] if values.ndim == 3 else values
    else:
        raise SystemExit(f"unsupported file type {p.suffix!r}; use csv, txt, parquet, npy, npz, or a store directory")
    if values.ndim == 1:
        values = values[:, None]
    from tsflab.data.protocol import split_borders

    splits = {flag: values[start:end] for flag, (start, end) in split_borders(len(values), ratio).items()}
    freq_info = _freq_override(freq) or prof.infer_frequency(stamps)
    return splits, {"name": p.stem or p.name, "source": str(p), "split_ratio": list(ratio), "frequency": freq_info}


def analyze(splits: dict[str, np.ndarray], meta: dict[str, Any], max_channels: int) -> dict[str, Any]:
    """Build the full report: profile plus catalog recommendations."""
    profile = prof.build_profile(
        splits,
        name=meta["name"],
        source=meta["source"],
        frequency=meta["frequency"],
        max_channels=max_channels,
        split_ratio=tuple(meta["split_ratio"]) if meta.get("split_ratio") else None,
    )
    profile["recommendations"] = prof.evaluate_rules(profile)
    return profile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tsf data analyze",
        description="Profile a dataset (train-only statistics, train/val/test shift) and map it to catalog facts.",
    )
    parser.add_argument("preset", nargs="?", help="dataset preset name (configs/datasets/<name>.toml)")
    parser.add_argument("--path", help="raw csv/txt/parquet/npy/npz file or realtime store directory")
    parser.add_argument("--split-ratio", type=float, nargs=3, default=DEFAULT_RATIO,
                        metavar=("TRAIN", "VAL", "TEST"), help="chronological split for --path")
    parser.add_argument("--freq", help="sampling interval override such as 10min or 1h")
    parser.add_argument("--max-channels", type=int, default=128,
                        help="channels sampled for spectral statistics (default 128)")
    parser.add_argument("--out", help="output directory (default work_dirs/profiles/<name>)")
    parser.add_argument("--json", action="store_true", help="print the JSON profile instead of markdown")
    parser.add_argument("--write-card", action="store_true",
                        help="record the fired data characteristics in the preset's dataset card")
    args = parser.parse_args(argv)
    if bool(args.preset) == bool(args.path):
        parser.error("give exactly one of <preset> or --path FILE")
    if args.max_channels < 2:
        parser.error("--max-channels must be at least 2")
    splits, meta = (
        load_preset(args.preset, args.freq)
        if args.preset
        else load_path(args.path, tuple(args.split_ratio), args.freq)
    )
    report = analyze(splits, meta, args.max_channels)
    markdown = prof.render_markdown(report, report["recommendations"])
    out = Path(args.out or Path("work_dirs/profiles") / meta["name"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "profile.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "profile.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(report, indent=2) if args.json else markdown)
    print(f"Output: {out}/profile.json, {out}/profile.md", file=sys.stderr if args.json else sys.stdout)
    if args.write_card:
        if not args.preset:
            parser.error("--write-card needs a preset")
        from datetime import date

        from tsflab.catalog.cards.datasets import write_characteristics
        from tsflab.core.paths import repository_root

        config = Path(args.preset) if Path(args.preset).suffix else Path("configs/datasets") / f"{args.preset}.toml"
        fired = [str(rule["id"]) for rule in report["recommendations"]["fired"]]
        basis = (f"measured by tsf data analyze ({prof.SCHEMA}, decision-safe train/val metrics, "
                 f"thresholds in profile_rules.toml) on {date.today().isoformat()}")
        card = write_characteristics(repository_root(), config.as_posix(), fired, basis)
        print(f"Characteristics written to {card}: {', '.join(fired) or 'none fired'}",
              file=sys.stderr if args.json else sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
