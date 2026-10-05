"""``tsf result board``: the bar to beat on a dataset, as compact L0-style lines.

Merges the published leaderboard (``board/leaderboard.json`` of the Hub
repository ``Diaugeia/TSFLab-Checkpoints``, or the site's fetched copy
``apps/web/data/leaderboard.json``) with local
run records (``<work_dir>/<dataset>/<Model>/records/*.json``, the self-describing
``record.json`` each run writes) so an agent sees the current best methods and
their metrics without opening files. Local runs are averaged per (model, horizon)
with ``tsflab.core.leaderboard.aggregate``; board rows come from the published
file as is. Local rows carry their protocol (``seq``, ``epochs``) because a
smoke or short run is not comparable with a published row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from tsflab.core.leaderboard import aggregate

BOARD_RELATIVE = Path("apps") / "web" / "data" / "leaderboard.json"


def _board_path(explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit)
    try:
        from tsflab.core.paths import repository_root

        path = repository_root() / BOARD_RELATIVE
    except Exception:
        path = None
    if path is not None and path.is_file():
        return path
    try:
        from tsflab.release.hub.results import fetch_board_file

        return fetch_board_file()
    except Exception:
        return None


def _rows_from_tracks(tracks: dict, dataset: str, source: str, protocols: dict | None = None) -> list[dict]:
    wanted = dataset.lower()
    rows: list[dict] = []
    for track, block in (tracks or {}).items():
        for name, data in (block.get("datasets") or {}).items():
            if name.lower() != wanted:
                continue
            for horizon, entries in (data.get("horizons") or {}).items():
                for entry in entries:
                    row = {"source": source, "track": track, "dataset": name, "horizon": str(horizon), **entry}
                    if protocols:
                        row.update(protocols.get((name, str(horizon), entry["model"]), {}))
                    rows.append(row)
    return rows


def _variant(model: str, snapshot: dict) -> str:
    """Name a slot-assignment run by its slots so two compositions are not averaged together."""
    params = (snapshot.get("model") or {}).get("params") or {}
    if model == "Composed" and "temporal" in params:
        slots = ("normalization", "decomposition", "temporal", "channel", "head")
        extra = ",".join(f"{k}={v}" for k, v in sorted(params.items()) if k not in slots and k != "enc_in")
        label = "/".join(str(params.get(k, "-")) for k in slots[:4])
        return f"Composed[{label}{';' + extra if extra else ''}]"
    return model


def local_runs(roots: list[Path], dataset: str) -> list[tuple[Path, dict]]:
    """``(record path, record)`` for every local run of ``dataset``; ``model`` is the variant name."""
    runs: list[tuple[Path, dict]] = []
    for root in roots:
        for path in sorted(root.rglob("records/*.json")) if root.is_dir() else []:
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if str(doc.get("dataset_id", "")).lower() != dataset.lower() or not doc.get("results"):
                continue
            snap = (doc.get("config") or {}).get("snapshot") or {}
            runs.append((path, {**doc, "model": _variant(doc["model"], snap)}))
    return runs


def _local_rows(roots: list[Path], dataset: str) -> list[dict]:
    docs: list[dict] = []
    protocols: dict = {}
    for _, doc in local_runs(roots, dataset):
        snap = (doc.get("config") or {}).get("snapshot") or {}
        for result in doc["results"]:
            protocols[(doc["dataset_id"], str(result.get("horizon")), doc["model"])] = {
                "seq": (doc.get("config") or {}).get("seq_len"),
                "epochs": (snap.get("training") or {}).get("epochs"),
            }
        docs.append(doc)
    return _rows_from_tracks(aggregate(docs), dataset, "local", protocols)


def _line(rank: int, row: dict, metric: str) -> str:
    def fmt(key: str) -> str:
        value = row.get(key)
        return f"{key}={value:.4f}" if isinstance(value, (int, float)) else f"{key}=-"

    parts = [f"{rank:>2}", f"{row['model']:<18}", fmt(metric)]
    parts += [fmt(k) for k in ("mae", "mse") if k != metric]
    parts.append(f"H={row['horizon']}")
    parts.append(f"n={row.get('n_runs', 1)}")
    parts.append(row["source"])
    if row["source"] == "local":
        parts.append(f"seq={row.get('seq')} epochs={row.get('epochs')}")
    else:
        parts.append(row["track"])
    return "  ".join(parts)


def board(dataset: str, *, horizon: str | None, top: int, metric: str, records: list[Path],
          board_file: str | None, include_board: bool = True) -> dict:
    rows: list[dict] = []
    path = _board_path(board_file) if include_board else None
    generated = None
    if path is not None:
        doc = json.loads(path.read_text(encoding="utf-8"))
        generated = doc.get("generated_at")
        rows += _rows_from_tracks(doc.get("tracks") or {}, dataset, "board")
    rows += _local_rows(records, dataset)
    horizons = sorted({r["horizon"] for r in rows}, key=lambda h: (not h.isdigit(), int(h) if h.isdigit() else h))
    if horizon is not None:
        horizons = [h for h in horizons if h == str(horizon)]
    groups = []
    for h in horizons:
        ranked = sorted(
            (r for r in rows if r["horizon"] == h and isinstance(r.get(metric), (int, float))),
            key=lambda r: r[metric],
        )[:top]
        groups.append({"horizon": h, "rows": ranked})
    return {"dataset": dataset, "metric": metric, "board_generated_at": generated, "horizons": groups}


def board_command(args: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="tsf result board",
        description="Best methods and metrics on a dataset (published leaderboard + local run records).",
    )
    parser.add_argument("--dataset", required=True, help="dataset id, e.g. ETTh1 (case-insensitive)")
    parser.add_argument("--horizon", default=None, help="prediction length; default all horizons found")
    parser.add_argument("--top", type=int, default=5, help="rows per horizon (default 5)")
    parser.add_argument("--metric", default="mse", help="ranking metric, lower is better (default mse)")
    parser.add_argument("--records", action="append", default=None, help="work_dir to scan for local records (repeatable; default ./work_dirs)")
    parser.add_argument("--board-file", default=None, help="leaderboard JSON; default apps/web/data/leaderboard.json, else the published board on the Hub")
    parser.add_argument("--json", action="store_true")
    parsed = parser.parse_args(args)
    result = board(
        parsed.dataset,
        horizon=parsed.horizon,
        top=max(1, parsed.top),
        metric=parsed.metric,
        records=[Path(p) for p in (parsed.records or ["work_dirs"])],
        board_file=parsed.board_file,
    )
    if parsed.json:
        print(json.dumps(result, indent=2))
        return 0
    if not any(g["rows"] for g in result["horizons"]):
        print(f"no board or local rows for dataset {parsed.dataset!r}; baseline panel needed", file=sys.stderr)
        return 1
    print(f"# board {result['dataset']} metric={result['metric']} (lower is better); board generated {result['board_generated_at']}")
    for group in result["horizons"]:
        for rank, row in enumerate(group["rows"], start=1):
            print(_line(rank, row, parsed.metric))
    return 0
