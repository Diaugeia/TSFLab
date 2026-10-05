#!/usr/bin/env python3
"""Build a site leaderboard from a directory of submission bundles.

The published board is ``board/leaderboard.json`` in the Hugging Face model
repository ``Diaugeia/TSFLab-Checkpoints``. ``tsf result hub results board``
builds it with :func:`build_tracks` from that repository's ``results/`` tree,
and ``pipeline/fetch_board.py`` downloads it before ``next build``. This script
builds the same board from a local directory, for example to preview the
pull-request staging folder ``submissions/``.

Validation and aggregation are the shared TSF-Core implementation
(``tsflab.core.leaderboard``); this script only adds the site's
presentation layer:

  1. map canonical (track, dataset_id) keys to the site's display keys;
  2. overlay curated blocks that submissions do not cover yet (``--curated``;
     on the Hub, ``board/curated.json``);
  3. attach the rolling real-time summaries from data/realtime/*.json and the
     metadata of every real-time track declared in configs/realtime/*.toml
     (``realtime_tracks``), so tracks with no rounds yet still appear. The
     real-time rounds live in Git, so this step runs at site build time.

Usage:
  python pipeline/build_leaderboard.py                     # submissions/ -> data/leaderboard.json
  python pipeline/build_leaderboard.py --source DIR [--curated FILE] [--out FILE]
  python pipeline/build_leaderboard.py --no-write          # dry run: summary only
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent.parent / "src"))

from tsflab.core.leaderboard import PRIMARY_METRIC, aggregate, load_submissions  # noqa: E402

SUBMISSIONS = ROOT / "submissions"
BOARD = ROOT / "data" / "leaderboard.json"
REALTIME = ROOT / "data" / "realtime"
REALTIME_CONFIGS = ROOT.parent.parent / "configs" / "realtime"
TRACK_FIELDS = ("title", "mode", "freq", "seq_len", "horizon", "submission_hours")


def _scalar(raw: str):
    raw = raw.split(" #")[0].strip()
    if raw[:1] in "\"'":
        return raw.strip("\"'")
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


def read_track_config(path: Path) -> dict:
    """Read the ``[track]`` table of a realtime TOML (minimal parser: py3.9 has no tomllib)."""
    table, out = None, {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("["):
            table = line.strip("[]")
        elif table == "track" and "=" in line and not line.startswith("#"):
            key, _, raw = line.partition("=")
            out[key.strip()] = _scalar(raw)
    return out


def realtime_track_meta() -> list[dict]:
    """One entry per configs/realtime/*.toml; domain is the id prefix (traffic_pems_ba -> traffic)."""
    tracks = []
    for path in sorted(REALTIME_CONFIGS.glob("*.toml")):
        cfg = read_track_config(path)
        tid = cfg.get("id", path.stem)
        entry = {"id": tid, "domain": cfg.get("domain") or tid.split("_")[0]}
        entry.update({k: cfg[k] for k in TRACK_FIELDS if k in cfg})
        tracks.append(entry)
    return tracks

# (canonical track, dataset_id) -> (site track key, display name)
DISPLAY = {("realtime", "stock_hs300"): ("stock", "Stock-HS300")}


def to_display(tracks: dict) -> dict:
    out: dict = {}
    for track, block in tracks.items():
        for dataset, data in block["datasets"].items():
            site_track, name = DISPLAY.get((track, dataset), (track, dataset))
            out.setdefault(site_track, {"datasets": {}})["datasets"][name] = data
    return out


def overlay_curated(tracks: dict, curated: dict) -> dict:
    out = json.loads(json.dumps(tracks))
    for track, block in curated.get("tracks", {}).items():
        for dataset, data in block.get("datasets", {}).items():
            target = out.setdefault(track, {"datasets": {}})["datasets"]
            if dataset not in target:
                target[dataset] = data  # no submissions produced this block yet
            elif "quant" in data:
                target[dataset]["quant"] = data["quant"]
    return out


def build_tracks(source: Path, curated: dict | None = None) -> tuple[dict, dict]:
    """Return ``(board, rejected)`` for every bundle under ``source``.

    ``board`` holds the ranked tracks of the valid bundles (display keys, curated
    overlay applied) but no real-time block; see :func:`attach_realtime`.
    """
    source = Path(source)
    valid, rejected = load_submissions(source) if source.is_dir() else ([], {})
    tracks = overlay_curated(to_display(aggregate(doc for _, doc in valid)), curated or {})
    board = {
        "schema_version": "1.2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "primary_metric": PRIMARY_METRIC,
        "n_submissions": len(valid),
        "n_rejected": len(rejected),
        "tracks": tracks,
    }
    return board, rejected


def attach_realtime(board: dict) -> dict:
    """Return ``board`` with the Git-tracked real-time summaries and track metadata."""
    out = dict(board)
    out["realtime"] = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(REALTIME.glob("*.json"))}
    out["realtime_tracks"] = realtime_track_meta()
    return out


def summarize(board: dict) -> None:
    print(f"Aggregated {board['n_submissions']} submission files:")
    for track, block in board["tracks"].items():
        for dataset, data in block["datasets"].items():
            for horizon, rows in data["horizons"].items():
                multi = sum(1 for r in rows if r.get("n_runs", 1) > 1)
                print(f"  {track}/{dataset}/h={horizon}: {len(rows)} models" + (f", {multi} multi-run" if multi else ""))
    for track, summary in board.get("realtime", {}).items():
        print(f"  realtime/{track}: {len(summary.get('scored_rounds', []))} scored round(s)")
    print(f"  realtime tracks declared: {len(board.get('realtime_tracks', []))}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SUBMISSIONS,
                        help="directory searched for submission.json (default: submissions/)")
    parser.add_argument("--curated", type=Path, default=None,
                        help="curated overlay, a board-shaped JSON (default: none)")
    parser.add_argument("--out", type=Path, default=BOARD, help="output path (default: data/leaderboard.json)")
    parser.add_argument("--no-write", action="store_true", help="dry run: summary only")
    args = parser.parse_args()

    curated = json.loads(args.curated.read_text(encoding="utf-8")) if args.curated else None
    board, rejected = build_tracks(args.source, curated)
    if rejected:
        print(f"❌ {len(rejected)} invalid submission(s) — run pipeline/validate.py; board not updated.")
        return 1
    board = attach_realtime(board)
    summarize(board)
    if args.no_write:
        print("\n(dry run — not written)")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(board, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"\n✅ wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
