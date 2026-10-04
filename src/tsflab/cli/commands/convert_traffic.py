"""Convert a traffic-style spatiotemporal dataset into a node bundle.

TSFLab's node-structured loader (``cauair_st`` / ``cauair_ts``) reads a
bundle of:

    <out_dir>/his.npz        # data (T, N, C); optional mean (C,), std (C,)
    <out_dir>/adj_mx.npy     # (N, N) adjacency, optional (graph models)
    <out_dir>/idx_train.npy  # window-centre indices
    <out_dir>/idx_val.npy
    <out_dir>/idx_test.npy

This converter builds that bundle from common public traffic datasets
(METR-LA, PEMS-BAY, PEMS03/04/07/08), which ship the value tensor and an
adjacency separately. Channel 0 of ``data`` is the target speed/flow; you can
append calendar covariates (time-of-day, day-of-week) with ``--add-time``.

Examples
--------
    # From a raw (T, N) speed matrix + an (N, N) adjacency:
    uv run tsf data prepare --from traffic \
        --values dataset/metr_la/metr-la.npz --values-key data \
        --adj dataset/metr_la/adj_mx.pkl \
        --output-dir dataset/metr_la --add-time --freq-min 5

The ``--adj`` path may be ``.npy``, ``.npz``, or a ``.pkl`` (METR-LA / PEMS-BAY
ship ``adj_mx.pkl`` as a ``(sensor_ids, id_map, adj_mx)`` tuple — the last
element is taken).

Then point a dataset config at it::

    [dataset]
    name = "cauair_st"          # node-structured (graph / spatiotemporal)
    root_path = "dataset/metr_la"
    data_path = ""

    [dataset.params]
    input_dim = 3               # value + [time_in_day, day_in_week]
"""

from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np


def _load_array(path: str, key: str | None) -> np.ndarray:
    """Load an array from a ``.npy`` or ``.npz`` file."""
    if path.endswith(".npz"):
        bundle = np.load(path, allow_pickle=True)
        if key is None:
            key = list(bundle.keys())[0]
        return np.asarray(bundle[key])
    return np.asarray(np.load(path, allow_pickle=True))


def _load_adjacency(path: str, key: str | None) -> np.ndarray:
    """Load an ``(N, N)`` adjacency from ``.npy`` / ``.npz`` / ``.pkl``.

    METR-LA and PEMS-BAY ship ``adj_mx.pkl`` as a 3-tuple
    ``(sensor_ids, sensor_id_to_idx, adj_mx)`` (or, less commonly, a bare
    ``(N, N)`` ndarray). For a tuple/list we take the last element, which is
    the adjacency matrix; ``.npy``/``.npz`` paths keep their existing behaviour.
    """
    if path.endswith(".pkl") or path.endswith(".pickle"):
        with open(path, "rb") as f:
            try:
                obj = pickle.load(f)
            except UnicodeDecodeError:  # Python-2-pickled files (e.g. DCRNN's)
                f.seek(0)
                obj = pickle.load(f, encoding="latin1")
        if isinstance(obj, (tuple, list)):
            obj = obj[-1]  # (sensor_ids, id_map, adj_mx) -> adj_mx
        return np.asarray(obj)
    return _load_array(path, key)


def _add_time_features(values: np.ndarray, freq_min: int) -> np.ndarray:
    """Append ``[time_in_day, day_in_week]`` covariates to a ``(T, N, 1)`` tensor."""
    t, n, _ = values.shape
    steps_per_day = (24 * 60) // freq_min
    step = np.arange(t)
    tod = (step % steps_per_day) / steps_per_day
    dow = ((step // steps_per_day) % 7) / 7.0
    tod = np.broadcast_to(tod[:, None, None], (t, n, 1))
    dow = np.broadcast_to(dow[:, None, None], (t, n, 1))
    return np.concatenate([values, tod, dow], axis=-1).astype(np.float32)


def split_windows(
    t: int, seq_len: int, pred_len: int, splits: tuple[float, float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronological train/val/test window-centre indices."""
    centers = np.arange(seq_len - 1, t - pred_len)
    n_tr, n_va = int(len(centers) * splits[0]), int(len(centers) * splits[1])
    return centers[:n_tr], centers[n_tr : n_tr + n_va], centers[n_tr + n_va :]


def train_range_stats(
    values: np.ndarray, train_centers: np.ndarray, pred_len: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Per-channel mean/std over the rows the training windows can see.

    The range is ``[0, last_train_center + pred_len]`` (history and targets of
    training windows), so validation/test-only rows never enter the statistics.
    Returns ``(mean, std, train_end)`` with ``train_end`` an exclusive row bound.
    """
    train_end = int(train_centers[-1]) + pred_len + 1 if len(train_centers) else len(values)
    block = values[:train_end].reshape(-1, values.shape[-1])
    return block.mean(0), block.std(0), train_end


def build_bundle(
    values: np.ndarray,
    output_dir: str,
    *,
    adj: np.ndarray | None = None,
    seq_len: int = 12,
    pred_len: int = 12,
    add_time: bool = False,
    freq_min: int = 5,
    splits: tuple[float, float, float] = (0.7, 0.1, 0.2),
) -> None:
    """Write a node bundle from a ``(T, N[, C])`` value array and an optional adjacency."""
    values = np.asarray(values).astype(np.float32)
    if values.ndim == 2:  # (T, N) -> (T, N, 1)
        values = values[..., None]
    if values.ndim != 3:
        raise ValueError(f"values must be (T, N) or (T, N, C); got {values.shape}")
    if add_time:
        values = _add_time_features(values[..., :1], freq_min)

    t, n, c = values.shape
    r_tr, r_va, r_te = splits
    train_c, val_c, test_c = split_windows(t, seq_len, pred_len, (r_tr, r_va, r_te))
    # Statistics come from the training range only, so scale=true cannot leak
    # validation/test information.
    mean, std, train_end = train_range_stats(values, train_c, pred_len)

    os.makedirs(output_dir, exist_ok=True)
    np.savez(
        os.path.join(output_dir, "his.npz"),
        data=values, mean=mean, std=std, train_end=np.int64(train_end),
        seq_len=np.int64(seq_len), pred_len=np.int64(pred_len),
    )

    if adj is not None:
        np.save(os.path.join(output_dir, "adj_mx.npy"), np.asarray(adj).astype(np.float32))

    np.save(os.path.join(output_dir, "idx_train.npy"), train_c)
    np.save(os.path.join(output_dir, "idx_val.npy"), val_c)
    np.save(os.path.join(output_dir, "idx_test.npy"), test_c)
    split_info = {
        "seq_len": seq_len, "pred_len": pred_len, "splits": [r_tr, r_va, r_te],
        "rows": t, "train_end": train_end,
        "windows": {"train": len(train_c), "val": len(val_c), "test": len(test_c)},
        "stats": "mean/std fitted on rows [0, train_end) only",
    }
    with open(os.path.join(output_dir, "split.json"), "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)
    centers = np.concatenate([train_c, val_c, test_c])

    adj_note = f", adj {n}x{n}" if adj is not None else ", no adjacency"
    print(f"Wrote bundle to {output_dir}  data={values.shape}{adj_note}  windows={len(centers)}  train_end={train_end}")


def main(argv: list[str] | None = None) -> None:
    """Convert raw traffic arrays into a TSFLab node bundle."""
    p = argparse.ArgumentParser(description="Convert traffic data to a node bundle")
    p.add_argument("--values", required=True, help="Path to the value array (.npy/.npz)")
    p.add_argument("--values-key", default=None, help="Key inside an .npz value file")
    p.add_argument(
        "--adj", default=None, help="Path to the (N, N) adjacency (.npy/.npz/.pkl)"
    )
    p.add_argument("--adj-key", default=None, help="Key inside an .npz adjacency file")
    p.add_argument("--output-dir", required=True, help="Bundle output directory")
    p.add_argument("--seq-len", type=int, default=12, help="History length (PeMS protocol: 12)")
    p.add_argument("--pred-len", type=int, default=12, help="Forecast horizon (PeMS protocol: 12)")
    p.add_argument("--add-time", action="store_true", help="Append calendar covariates")
    p.add_argument("--freq-min", type=int, default=5, help="Minutes per step (for --add-time)")
    p.add_argument(
        "--splits", default="0.7,0.1,0.2", help="train,val,test ratios (comma-separated)"
    )
    args = p.parse_args(argv)

    r_tr, r_va, r_te = (float(x) for x in args.splits.split(","))
    build_bundle(
        _load_array(args.values, args.values_key),
        args.output_dir,
        adj=_load_adjacency(args.adj, args.adj_key) if args.adj else None,
        seq_len=args.seq_len, pred_len=args.pred_len,
        add_time=args.add_time, freq_min=args.freq_min, splits=(r_tr, r_va, r_te),
    )


if __name__ == "__main__":
    main()
