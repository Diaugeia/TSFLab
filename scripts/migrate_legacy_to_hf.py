#!/usr/bin/env python3
"""One-off move of the TSEval-era results and weights into ``legacy/`` of TSFLab-Checkpoints.

Every result that was kept in Git, and the TSEval weights, predate TSFLab 1.0.
They are archived under ``legacy/`` and never ranked. The main board holds only
TSFLab 1.0 results (pushed later with ``tsf result hub results push``), so it is
empty right after the migration and the site shows its "results coming" state.

Dry run by default: it prints the plan and writes nothing. ``--apply`` uploads.
Re-running is safe: files already present in the target repository are skipped.

Steps (``--steps``, default all, in this order):

  weights      dataset repo ``Diaugeia/TSEval-Weights`` (``realtime/stock_hs300/<Model>/*.pth``,
               ``_index.json``)                        -> ``legacy/tseval-weights/<path>``
  submissions  every bundle under ``apps/web/submissions/<track>/`` (864 time_series,
               135 realtime/stock_hs300)               -> ``legacy/submissions/<track>/...``
  site         ``apps/web/data/{leaderboard,model-meta,visualization_data}.json``
               (with the curated air-quality and stock quant blocks) -> ``legacy/board/``
  board        regenerate ``board/leaderboard.json`` + ``board/model-meta.json`` from
               ``results/`` (empty until TSFLab 1.0 results are pushed)

Real-time rounds (``apps/web/submissions/realtime/<track>/rounds/``) stay in the
repository and are not copied. The sources were removed from the working tree
when results moved to the Hub, so they are read from ``--git-ref`` (default: a
commit that still has them) unless ``--submissions`` and ``--data`` point at
local copies.

Usage:
  python scripts/migrate_legacy_to_hf.py                   # plan only
  python scripts/migrate_legacy_to_hf.py --create --apply  # upload (HF_TOKEN with write access)
  python scripts/migrate_legacy_to_hf.py --steps board --apply
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tsflab.release.hub.publish import DEFAULT_CHECKPOINTS_REPO  # noqa: E402
from tsflab.release.hub.results import LEGACY_PREFIX  # noqa: E402
from tsflab.release.hub.uri import default_repo  # noqa: E402

TSEVAL_WEIGHTS = default_repo("TSEval-Weights")
# dev before results moved to the Hub; it still holds the in-repository copies.
DEFAULT_GIT_REF = "1361a2da"
SUBMISSIONS = "apps/web/submissions"
DATA = "apps/web/data"
SITE_FILES = ("leaderboard.json", "model-meta.json", "visualization_data.json")
STEPS = ("weights", "submissions", "site", "board")
SKIP_NAMES = {".gitattributes"}
COMMIT_FILES = 1000


@dataclass(frozen=True)
class Transfer:
    step: str
    dest: str
    source: Path | None = None   # local file
    remote: str | None = None    # path in the TSEval-Weights dataset repo

    def size(self) -> int | None:
        return self.source.stat().st_size if self.source is not None else None


def plan_weights(files: list[str]) -> list[Transfer]:
    return [Transfer("weights", f"{LEGACY_PREFIX}/tseval-weights/{name}", remote=name)
            for name in sorted(files) if Path(name).name not in SKIP_NAMES]


def plan_submissions(submissions: Path) -> list[Transfer]:
    """Every file below a track folder, verbatim; real-time rounds are left out."""
    if not submissions.is_dir():
        return []
    files = []
    for path in sorted(submissions.rglob("*")):
        parts = path.relative_to(submissions).parts
        if path.is_file() and len(parts) > 1 and "rounds" not in parts and path.name not in SKIP_NAMES:
            files.append(path)
    return [Transfer("submissions", f"{LEGACY_PREFIX}/submissions/{p.relative_to(submissions).as_posix()}",
                     source=p) for p in files]


def plan_site(data: Path) -> list[Transfer]:
    return [Transfer("site", f"{LEGACY_PREFIX}/board/{name}", source=data / name)
            for name in SITE_FILES if (data / name).is_file()]


def plan(steps: list[str], submissions: Path, data: Path, weight_files: list[str],
         existing: set[str] = frozenset()) -> tuple[list[Transfer], list[Transfer]]:
    """Return ``(pending, present)`` transfers for ``steps``; every one lands below ``legacy/``."""
    transfers: list[Transfer] = []
    if "weights" in steps:
        transfers += plan_weights(weight_files)
    if "submissions" in steps:
        transfers += plan_submissions(submissions)
    if "site" in steps:
        transfers += plan_site(data)
    dests = [t.dest for t in transfers]
    if len(dests) != len(set(dests)):
        raise ValueError("two sources map to the same destination")
    if any(not dest.startswith(LEGACY_PREFIX + "/") for dest in dests):
        raise ValueError("the migration writes only below legacy/")
    pending = [t for t in transfers if t.dest not in existing]
    present = [t for t in transfers if t.dest in existing]
    return pending, present


def extract_git(ref: str, into: Path) -> tuple[Path, Path]:
    """Extract the submissions and data folders of ``ref`` into ``into``."""
    into.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(["git", "archive", "--format=tar", ref, SUBMISSIONS, DATA],
                             cwd=REPO_ROOT, capture_output=True, check=True).stdout
    tar_path = into / "sources.tar"
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
                    content = hf_hub_download(TSEVAL_WEIGHTS, t.remote, repo_type="dataset",
                                              local_dir=str(staging / "tseval-weights"))
                else:
                    content = str(t.source)
                operations.append(CommitOperationAdd(path_in_repo=t.dest, path_or_fileobj=content))
            commit = api.create_commit(repo_id=repo_id, repo_type="model", operations=operations,
                                       commit_message=f"legacy {step}: {len(chunk)} file(s)")
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
    parser.add_argument("--create", action="store_true", help="create the target repository if missing (public)")
    parser.add_argument("--private", action="store_true", help="with --create: create it private")
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="tsflab-migrate-") as tmp:
        tmp = Path(tmp)
        submissions, data = args.submissions, args.data
        needs_sources = {"submissions", "site"} & set(args.steps)
        if needs_sources and (submissions is None or data is None):
            extracted_submissions, extracted_data = extract_git(args.git_ref, tmp / "sources")
            submissions = submissions or extracted_submissions
            data = data or extracted_data
        api = _api() if (args.apply or "weights" in args.steps) else None
        weight_files = (_list(api, TSEVAL_WEIGHTS, "dataset") or []) if "weights" in args.steps else []
        existing = set(_list(api, args.repo, "model") or []) if api is not None else set()
        pending, present = plan(args.steps, submissions or tmp, data or tmp, weight_files, existing)

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
