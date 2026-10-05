#!/usr/bin/env python3
"""One-off move of the Git-hosted results and TSEval artifacts into TSFLab-Checkpoints.

Dry run by default: it prints the plan and writes nothing. ``--apply`` uploads.
Re-running is safe: files already present in the target repository are skipped.

Steps (``--steps``, default all, in this order):

  weights      dataset repo ``Diaugeia/TSEval-Weights`` (``realtime/stock_hs300/<Model>/*.pth``,
               ``_index.json``)                         -> ``legacy/tseval-weights/<path>``
  legacy       ``apps/web/submissions/realtime/stock_hs300/**`` (135 TSEval 0.3.2 bundles)
                                                        -> ``legacy/submissions/realtime/stock_hs300/**``
  current      every other bundle under ``apps/web/submissions/`` (the time_series board)
                                                        -> ``results/<track>/<dataset>/<model>/<id>/``
  curated      ``apps/web/data/leaderboard.json``: the air-quality block -> ``board/curated.json``;
               the whole file and ``visualization_data.json``        -> ``legacy/board/``
  board        regenerate ``board/leaderboard.json`` + ``board/model-meta.json`` from ``results/``

``legacy/`` is never read by the board, so TSEval artifacts are archived but not
ranked. The Git sources were removed from the working tree when results moved to
the Hub, so they are read from ``--git-ref`` (default: the last commit that has
them) unless ``--submissions``/``--data`` point at local copies.

Usage:
  python scripts/migrate_legacy_to_hf.py                       # plan only
  python scripts/migrate_legacy_to_hf.py --apply               # upload (needs HF_TOKEN with write access)
  python scripts/migrate_legacy_to_hf.py --steps current board --apply
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tsflab.release.hub.publish import DEFAULT_CHECKPOINTS_REPO  # noqa: E402
from tsflab.release.hub.results import LEGACY_PREFIX, collect  # noqa: E402
from tsflab.release.hub.uri import default_repo  # noqa: E402

TSEVAL_WEIGHTS = default_repo("TSEval-Weights")
# dev at the commit that moved results to the Hub; it still holds the Git copies.
DEFAULT_GIT_REF = "1361a2da"
SUBMISSIONS = "apps/web/submissions"
DATA = "apps/web/data"
LEGACY_SUBMISSIONS = Path("realtime") / "stock_hs300"
CURATED_TRACKS = ("air_quality",)
STEPS = ("weights", "legacy", "current", "curated", "board")
SKIP_NAMES = {".gitattributes"}
COMMIT_FILES = 1000


@dataclass(frozen=True)
class Transfer:
    step: str
    dest: str
    source: Path | None = None   # local file
    remote: str | None = None    # path in the TSEval-Weights dataset repo
    data: bytes | None = None    # generated content

    def size(self) -> int | None:
        if self.data is not None:
            return len(self.data)
        return self.source.stat().st_size if self.source is not None else None


def plan_weights(files: list[str]) -> list[Transfer]:
    return [Transfer("weights", f"{LEGACY_PREFIX}/tseval-weights/{name}", remote=name)
            for name in sorted(files) if Path(name).name not in SKIP_NAMES]


def plan_legacy(submissions: Path) -> list[Transfer]:
    root = submissions / LEGACY_SUBMISSIONS
    files = sorted(p for p in root.rglob("*") if p.is_file() and "rounds" not in p.relative_to(root).parts) \
        if root.is_dir() else []
    return [Transfer("legacy", f"{LEGACY_PREFIX}/submissions/{p.relative_to(submissions).as_posix()}", source=p)
            for p in files]


def plan_current(submissions: Path) -> list[Transfer]:
    """Every valid bundle outside the legacy tree and the real-time rounds, at its results/ path."""
    if not submissions.is_dir():
        return []
    root = submissions.resolve()
    legacy = (submissions / LEGACY_SUBMISSIONS).resolve()

    def skipped(path: Path) -> bool:
        path = path.resolve()
        return path.is_relative_to(legacy) or "rounds" in path.relative_to(root).parts

    bundles, rejected = collect([submissions])
    rejected = {path: errors for path, errors in rejected.items() if not skipped(path)}
    if rejected:
        raise ValueError(f"{len(rejected)} invalid bundle(s) under {submissions}: {sorted(rejected)[:3]}")
    out = []
    for bundle in bundles:
        if not skipped(bundle.directory):
            out += [Transfer("current", f"{bundle.path_in_repo}/{f.name}", source=f) for f in bundle.files()]
    return out


def plan_curated(data: Path) -> list[Transfer]:
    board_file = data / "leaderboard.json"
    if not board_file.is_file():
        return []
    board = json.loads(board_file.read_text(encoding="utf-8"))
    curated = {"schema_version": board.get("schema_version"),
               "note": "Curated blocks with no raw submissions; overlaid by the board builder.",
               "tracks": {t: board["tracks"][t] for t in CURATED_TRACKS if t in board.get("tracks", {})}}
    out = [Transfer("curated", "board/curated.json",
                    data=(json.dumps(curated, ensure_ascii=False, indent=1) + "\n").encode("utf-8")),
           Transfer("curated", f"{LEGACY_PREFIX}/board/leaderboard-tseval.json", source=board_file)]
    visualization = data / "visualization_data.json"
    if visualization.is_file():
        out.append(Transfer("curated", f"{LEGACY_PREFIX}/board/visualization_data.json", source=visualization))
    return out


def plan(steps: list[str], submissions: Path, data: Path, weight_files: list[str],
         existing: set[str] = frozenset()) -> tuple[list[Transfer], list[Transfer]]:
    """Return ``(pending, present)`` transfers for ``steps``."""
    transfers: list[Transfer] = []
    if "weights" in steps:
        transfers += plan_weights(weight_files)
    if "legacy" in steps:
        transfers += plan_legacy(submissions)
    if "current" in steps:
        transfers += plan_current(submissions)
    if "curated" in steps:
        transfers += plan_curated(data)
    dests = [t.dest for t in transfers]
    if len(dests) != len(set(dests)):
        raise ValueError("two sources map to the same destination")
    # board/curated.json is regenerated content: always rewrite it.
    pending = [t for t in transfers if t.dest not in existing or t.dest == "board/curated.json"]
    present = [t for t in transfers if t not in pending]
    return pending, present


def extract_git(ref: str, into: Path) -> tuple[Path, Path]:
    """Extract the submissions and data folders of ``ref`` into ``into``."""
    archive = subprocess.run(["git", "archive", "--format=tar", ref, SUBMISSIONS, DATA],
                             cwd=REPO_ROOT, capture_output=True, check=True).stdout
    tar_path = into / "git.tar"
    tar_path.write_bytes(archive)
    with tarfile.open(tar_path) as tar:
        tar.extractall(into, filter="data")
    return into / SUBMISSIONS, into / DATA


def _api():
    from huggingface_hub import HfApi

    return HfApi()


def _list(api, repo_id: str, repo_type: str) -> list[str] | None:
    try:
        return api.list_repo_files(repo_id, repo_type=repo_type)
    except Exception as exc:  # missing repo, no token, or no network
        print(f"  cannot list {repo_type}:{repo_id}: {type(exc).__name__}: {exc}")
        return None


def apply(pending: list[Transfer], repo_id: str, api, staging: Path) -> list[str]:
    from huggingface_hub import CommitOperationAdd, hf_hub_download

    commits = []
    for step in STEPS:
        batch = [t for t in pending if t.step == step]
        for start in range(0, len(batch), COMMIT_FILES):
            chunk = batch[start:start + COMMIT_FILES]
            operations = []
            for t in chunk:
                if t.remote is not None:
                    local = hf_hub_download(TSEVAL_WEIGHTS, t.remote, repo_type="dataset",
                                            local_dir=str(staging / "tseval-weights"))
                    content = local
                else:
                    content = t.data if t.data is not None else str(t.source)
                operations.append(CommitOperationAdd(path_in_repo=t.dest, path_or_fileobj=content))
            commit = api.create_commit(repo_id=repo_id, repo_type="model", operations=operations,
                                       commit_message=f"migrate {step}: {len(chunk)} file(s)")
            commits.append(commit.oid)
            print(f"  {step}: committed {len(chunk)} file(s) as {commit.oid}")
    return commits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="upload (default: dry run)")
    parser.add_argument("--repo", default=DEFAULT_CHECKPOINTS_REPO, help="target model repository")
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS))
    parser.add_argument("--git-ref", default=DEFAULT_GIT_REF,
                        help=f"commit that holds {SUBMISSIONS} and {DATA} (default {DEFAULT_GIT_REF})")
    parser.add_argument("--submissions", type=Path, default=None, help="local submissions folder instead of --git-ref")
    parser.add_argument("--data", type=Path, default=None, help="local site data folder instead of --git-ref")
    parser.add_argument("--create", action="store_true", help="create the target repository if missing")
    parser.add_argument("--private", action="store_true", help="with --create: create it private")
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="tsflab-migrate-") as tmp:
        tmp = Path(tmp)
        submissions, data = args.submissions, args.data
        if submissions is None or data is None:
            git_submissions, git_data = extract_git(args.git_ref, tmp / "git")
            submissions = submissions or git_submissions
            data = data or git_data
        api = _api() if (args.apply or "weights" in args.steps) else None
        weight_files = (_list(api, TSEVAL_WEIGHTS, "dataset") or []) if "weights" in args.steps else []
        existing = set(_list(api, args.repo, "model") or []) if api is not None else set()
        pending, present = plan(args.steps, submissions, data, weight_files, existing)

        print(f"target {args.repo} ({'apply' if args.apply else 'dry run'})")
        for step in STEPS:
            if step not in args.steps:
                continue
            if step == "board":
                print("  board: regenerate board/leaderboard.json, board/model-meta.json from results/")
                continue
            todo = [t for t in pending if t.step == step]
            done = sum(1 for t in present if t.step == step)
            size = sum(t.size() or 0 for t in todo)
            sample = todo[0].dest if todo else "-"
            print(f"  {step}: {len(todo)} file(s) to copy ({size / 1e6:.1f} MB local), "
                  f"{done} already present; e.g. {sample}")
        if not args.apply:
            print("dry run: nothing uploaded; pass --apply to upload")
            return 0
        if args.create:
            api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
        apply(pending, args.repo, api, tmp)
        if "board" in args.steps:
            from tsflab.release.hub.results import publish_board

            summary = publish_board(args.repo, api=api)
            print(f"  board: {summary['n_submissions']} submission(s), commit {summary['commit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
