#!/usr/bin/env python3
"""Download the published board files before ``next build`` (standard library only).

The site does not keep results in Git. ``tsf result hub results board`` writes
the generated board to ``board/`` of the Hugging Face model repository
``Diaugeia/TSFLab-Checkpoints``; this script copies it into the site:

    board/leaderboard.json          -> data/leaderboard.json (+ Git real-time block)
    board/model-meta.json           -> data/model-meta.json  (else built from this checkout)
    board/visualization_data.json   -> public/visualization_data.json (optional)

The real-time summaries (``data/realtime/*.json``) and track metadata
(``configs/realtime/*.toml``) stay in Git and are attached here, so a weekly
round shows on the next build without a Hub write.

Usage:
  python3 pipeline/fetch_board.py                         # main of Diaugeia/TSFLab-Checkpoints
  python3 pipeline/fetch_board.py --repo OWNER/NAME --revision <commit>
  python3 pipeline/fetch_board.py --from DIR              # offline: DIR holds leaderboard.json, ...

``TSFLAB_BOARD_REPO`` and ``TSFLAB_BOARD_REVISION`` override the defaults; a
private repository needs ``HF_TOKEN``. The script fails when the board cannot be
read, so a build never ships an empty board.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_leaderboard import BOARD, attach_realtime  # noqa: E402
from build_model_meta import parse_models  # noqa: E402
from tsflab.release.hub.fetch import download  # noqa: E402
from tsflab.release.hub.publish import DEFAULT_CHECKPOINTS_REPO  # noqa: E402

BOARD_FILES = ("leaderboard.json", "model-meta.json", "visualization_data.json")
REQUIRED = ("leaderboard.json",)
MODEL_META = ROOT / "data" / "model-meta.json"
VISUALIZATION = ROOT / "public" / "visualization_data.json"


def fetch(repo: str, revision: str, into: Path) -> list[str]:
    """Download the board files of ``repo@revision`` into ``into``; return the names found."""
    found = []
    for name in BOARD_FILES:
        try:
            download(f"hf://{repo}@{revision}/board/{name}", into / name)
            found.append(name)
        except HTTPError as exc:
            if name in REQUIRED or exc.code != 404:
                raise
        except URLError:
            if name in REQUIRED:
                raise
    return found


def install(source: Path) -> None:
    """Write the site data files from a directory holding the board files."""
    board = json.loads((source / "leaderboard.json").read_text(encoding="utf-8"))
    board = attach_realtime(board)
    BOARD.parent.mkdir(parents=True, exist_ok=True)
    BOARD.write_text(json.dumps(board, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    if (source / "model-meta.json").is_file():
        shutil.copyfile(source / "model-meta.json", MODEL_META)
    else:
        meta = parse_models(str(ROOT.parent.parent))
        MODEL_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    VISUALIZATION.parent.mkdir(parents=True, exist_ok=True)
    if (source / "visualization_data.json").is_file():
        shutil.copyfile(source / "visualization_data.json", VISUALIZATION)
    else:
        VISUALIZATION.unlink(missing_ok=True)
    print(f"board: {board.get('n_submissions')} submission(s), generated {board.get('generated_at')}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=os.environ.get("TSFLAB_BOARD_REPO", DEFAULT_CHECKPOINTS_REPO))
    parser.add_argument("--revision", default=os.environ.get("TSFLAB_BOARD_REVISION", "main"))
    parser.add_argument("--from", dest="source", type=Path, default=None,
                        help="read the board files from a local directory instead of the Hub")
    args = parser.parse_args()
    if args.source is not None:
        if not (args.source / "leaderboard.json").is_file():
            print(f"error: {args.source}/leaderboard.json not found", file=sys.stderr)
            return 1
        install(args.source)
        return 0
    with tempfile.TemporaryDirectory(prefix="tsflab-board-") as tmp:
        try:
            found = fetch(args.repo, args.revision, Path(tmp))
        except (URLError, OSError, ValueError) as exc:
            print(f"error: cannot read board/leaderboard.json from {args.repo}@{args.revision}: {exc}\n"
                  "  publish it with `tsf result hub results board`, or pass --from DIR",
                  file=sys.stderr)
            return 1
        print(f"fetched {', '.join(found)} from {args.repo}@{args.revision}")
        install(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
