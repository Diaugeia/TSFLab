"""Run results, the generated board, and top-ranked checkpoints on the Hub.

One model repository (default ``Diaugeia/TSFLab-Checkpoints``) holds::

    results/<track>/<dataset>/<model>/<submission_id>/   submission bundles
    checkpoints/<track>/<dataset>/<model>/<run_id>/      weights bundles (see bundle.py)
    board/leaderboard.json, board/model-meta.json        generated; the site reads them
    board/curated.json                                   optional curated overlay (input; none for 1.0)
    legacy/                                              every TSEval-era artifact, never ranked

Validation and ranking reuse ``tsflab.core.leaderboard`` and the site pipeline
(``apps/web/pipeline``); ranking for checkpoint selection reuses
``tsf result board``. Uploads need ``huggingface_hub`` (``tsflab[hub]``),
a write token, and explicit authorization.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import re
import tempfile

from tsflab.core.leaderboard import iter_records, load_submissions
from tsflab.release.hub.bundle import bundle_path, find_run, pack
from tsflab.release.hub.publish import DEFAULT_CHECKPOINTS_REPO, _api

RESULTS_PREFIX = "results"
BOARD_PREFIX = "board"
LEGACY_PREFIX = "legacy"
BUNDLE_FILES = ("submission.json", "trajectory.jsonl", "report.md")
BOARD_FILES = ("leaderboard.json", "model-meta.json")
CURATED = f"{BOARD_PREFIX}/curated.json"
DEFAULT_BATCH = 1000  # submission bundles per commit


def _segment(text: object) -> str:
    """One safe path segment; keeps case so paths match the record names."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text or "")).strip("._") or "x"


def results_path(doc: dict, fallback_id: str) -> str:
    """Return ``results/<track>/<dataset>/<model>/<submission_id>`` for one bundle."""
    record = next(iter_records(doc), {})
    manifest = doc.get("manifest") or {}
    track = record.get("track") or manifest.get("track") or "time_series"
    submission_id = manifest.get("submission_id") or fallback_id
    return "/".join((RESULTS_PREFIX, _segment(track), _segment(record.get("dataset_id")),
                     _segment(record.get("model")), _segment(submission_id)))


@dataclass(frozen=True)
class ResultBundle:
    directory: Path
    path_in_repo: str

    def files(self) -> list[Path]:
        return [self.directory / name for name in BUNDLE_FILES if (self.directory / name).is_file()]


def collect(sources: list[Path]) -> tuple[list[ResultBundle], dict[Path, list[str]]]:
    """Find every ``submission.json`` under ``sources``; return (valid bundles, rejected)."""
    bundles: dict[str, ResultBundle] = {}
    rejected: dict[Path, list[str]] = {}
    for source in sources:
        if not Path(source).is_dir():
            raise FileNotFoundError(f"results source not found: {source}")
        valid, bad = load_submissions(Path(source))
        rejected.update(bad)
        for path, doc in valid:
            bundle = ResultBundle(path.parent, results_path(doc, path.parent.name))
            previous = bundles.get(bundle.path_in_repo)
            if previous is not None and previous.directory != bundle.directory:
                rejected[path] = [f"duplicate of {previous.directory} at {bundle.path_in_repo}"]
                continue
            bundles[bundle.path_in_repo] = bundle
    return sorted(bundles.values(), key=lambda b: b.path_in_repo), rejected


def _chunks(items: list, size: int):
    for start in range(0, len(items), max(1, size)):
        yield items[start:start + max(1, size)]


def _existing(api, repo_id: str) -> set[str]:
    return set(api.list_repo_files(repo_id, repo_type="model"))


def push_results(sources: list[Path], repo_id: str = DEFAULT_CHECKPOINTS_REPO, *,
                 dry_run: bool = False, force: bool = False, skip_invalid: bool = False,
                 batch: int = DEFAULT_BATCH, create: bool = False, private: bool = False,
                 api=None) -> dict:
    """Upload validated bundles to ``results/``; skip bundles already present.

    Returns a summary with the planned ``paths`` and the new commit ids. A dry
    run plans from the local files only and makes no network call.
    """
    bundles, rejected = collect(sources)
    if rejected and not skip_invalid:
        lines = [f"{path}: {'; '.join(errors[:3])}" for path, errors in sorted(rejected.items())]
        raise ValueError(f"{len(rejected)} invalid submission(s):\n  " + "\n  ".join(lines[:20]))
    summary = {"repo": repo_id, "found": len(bundles), "rejected": len(rejected),
               "skipped": 0, "uploaded": 0, "commits": [], "paths": [b.path_in_repo for b in bundles]}
    if dry_run:
        return summary
    from huggingface_hub import CommitOperationAdd

    api = api or _api()
    if create:
        api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    present = set() if force else _existing(api, repo_id)
    pending = [b for b in bundles if f"{b.path_in_repo}/submission.json" not in present]
    summary["skipped"] = len(bundles) - len(pending)
    summary["paths"] = [b.path_in_repo for b in pending]
    for chunk in _chunks(pending, batch):
        operations = [CommitOperationAdd(path_in_repo=f"{b.path_in_repo}/{f.name}", path_or_fileobj=str(f))
                      for b in chunk for f in b.files()]
        commit = api.create_commit(repo_id=repo_id, repo_type="model", operations=operations,
                                   commit_message=f"results: add {len(chunk)} submission(s)")
        summary["commits"].append(commit.oid)
        summary["uploaded"] += len(chunk)
    return summary


def results_pattern(track: str | None = None, dataset: str | None = None, model: str | None = None,
                    *, metadata_only: bool = False) -> str:
    """An ``allow_patterns`` glob for part of ``results/`` (``*`` also matches ``/``)."""
    parts = [_segment(p) if p else "*" for p in (track, dataset, model)]
    while parts and parts[-1] == "*":
        parts.pop()
    tail = "*submission.json" if metadata_only else "*"
    return "/".join([RESULTS_PREFIX, *parts, tail])


def pull_results(repo_id: str = DEFAULT_CHECKPOINTS_REPO, out_dir: Path = Path("work_dirs/_hub"),
                 *, revision: str = "main", track: str | None = None, dataset: str | None = None,
                 model: str | None = None, metadata_only: bool = False) -> Path:
    """Download ``results/`` (optionally one track/dataset/model) into ``out_dir``; return it."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required; install `tsflab[hub]`") from exc
    patterns = [results_pattern(track, dataset, model, metadata_only=metadata_only)]
    if metadata_only:
        patterns.append(CURATED)
    snapshot_download(repo_id, repo_type="model", revision=revision,
                      allow_patterns=patterns, local_dir=str(out_dir))
    return Path(out_dir)


# ---------------------------------------------------------------------------
# Board: generated from results/ with the site pipeline
# ---------------------------------------------------------------------------


def _pipeline(name: str):
    """Load one ``apps/web/pipeline`` module; the board needs a TSFLab checkout."""
    from tsflab.core.paths import is_packaged_root, repository_root

    root = repository_root()
    path = root / "apps" / "web" / "pipeline" / f"{name}.py"
    if is_packaged_root(root) or not path.is_file():
        raise RuntimeError("building the board requires a TSFLab checkout (apps/web/pipeline)")
    spec = importlib.util.spec_from_file_location(f"tsflab_web_pipeline_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_board(results_root: Path, curated: dict | None = None) -> tuple[dict, dict, dict]:
    """Return ``(leaderboard, model_meta, rejected)`` from bundles under ``results_root``."""
    from tsflab.core.paths import repository_root

    board, rejected = _pipeline("build_leaderboard").build_tracks(results_root, curated)
    board["rejections"] = [f"{p}: {'; '.join(errors[:3])}" for p, errors in sorted(rejected.items())]
    meta = _pipeline("build_model_meta").parse_models(str(repository_root()))
    return board, meta, rejected


def publish_board(repo_id: str = DEFAULT_CHECKPOINTS_REPO, *, local: Path | None = None,
                  curated: Path | None = None, out_dir: Path | None = None,
                  upload: bool = True, api=None) -> dict:
    """Regenerate ``board/leaderboard.json`` and ``board/model-meta.json``.

    Without ``local`` the submission files of ``results/`` (and
    ``board/curated.json``) are downloaded first, so the board always ranks the
    whole repository. ``local`` is a directory holding ``results/`` (a mirror
    from ``pull_results``) or bundles directly. ``out_dir`` keeps a copy.
    """
    with tempfile.TemporaryDirectory(prefix="tsflab-board-") as tmp:
        if local is None:
            mirror = pull_results(repo_id, Path(tmp) / "mirror", metadata_only=True)
            results_root = mirror / RESULTS_PREFIX
            curated = curated or (mirror / CURATED if (mirror / CURATED).is_file() else None)
        else:
            results_root = local / RESULTS_PREFIX if (local / RESULTS_PREFIX).is_dir() else local
        overlay = json.loads(Path(curated).read_text(encoding="utf-8")) if curated else None
        board, meta, rejected = build_board(results_root, overlay)
        target = Path(out_dir) if out_dir else Path(tmp) / "board"
        target.mkdir(parents=True, exist_ok=True)
        (target / "leaderboard.json").write_text(json.dumps(board, ensure_ascii=False, indent=1), encoding="utf-8")
        (target / "model-meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary = {"repo": repo_id, "n_submissions": board["n_submissions"], "n_rejected": len(rejected),
                   "out_dir": str(target) if out_dir else None, "commit": None}
        if upload:
            from huggingface_hub import CommitOperationAdd

            api = api or _api()
            operations = [CommitOperationAdd(path_in_repo=f"{BOARD_PREFIX}/{name}",
                                             path_or_fileobj=str(target / name)) for name in BOARD_FILES]
            commit = api.create_commit(repo_id=repo_id, repo_type="model", operations=operations,
                                       commit_message=f"board: {board['n_submissions']} submission(s)")
            summary["commit"] = commit.oid
    return summary


def fetch_board_file(repo_id: str = DEFAULT_CHECKPOINTS_REPO, *, max_age_sec: int = 3600) -> Path | None:
    """Return a cached copy of the published ``board/leaderboard.json`` (refreshed hourly)."""
    import os
    import time

    from tsflab.release.hub.fetch import cache_path, download
    from tsflab.release.hub.uri import HubURI

    if os.environ.get("TSFLAB_OFFLINE"):
        return None
    uri = HubURI(repo_id, "main", f"{BOARD_PREFIX}/leaderboard.json")
    path = cache_path(uri)
    if path.is_file() and time.time() - path.stat().st_mtime < max_age_sec:
        return path
    try:
        return download(str(uri), path)
    except (OSError, ValueError):
        return path if path.is_file() else None


# ---------------------------------------------------------------------------
# Checkpoints: only the top-ranked runs
# ---------------------------------------------------------------------------


def select_top_runs(dataset: str, *, horizon: str | None, top: int, metric: str = "mse",
                    records: list[Path]) -> list[dict]:
    """Rank local runs with ``tsf result board`` and return the best run of each top row.

    Rows are (model variant, horizon) averages over seeds; each selected entry
    is the single run of that row with the lowest ``metric``.
    """
    from tsflab.cli.commands.result_board import board, local_runs

    ranked = board(dataset, horizon=horizon, top=top, metric=metric, records=records,
                   board_file=None, include_board=False)
    runs = local_runs(records, dataset)
    selected = []
    for group in ranked["horizons"]:
        for rank, row in enumerate(group["rows"], start=1):
            best = None
            for path, doc in runs:
                if doc["model"] != row["model"]:
                    continue
                for result in doc["results"]:
                    value = (result.get("metrics") or {}).get(metric)
                    if str(result.get("horizon")) != group["horizon"] or not isinstance(value, (int, float)):
                        continue
                    if best is None or value < best["run_metric"]:
                        best = {"rank": rank, "model": row["model"], "horizon": group["horizon"],
                                "row_metric": row[metric], "n_runs": row.get("n_runs", 1),
                                "run_metric": value, "record": path,
                                "run_id": result.get("run_id") or path.stem}
            if best is not None:
                selected.append(best)
    return selected


def push_top(dataset: str, *, horizon: str | None, top: int, metric: str = "mse",
             records: list[Path], repo_id: str = DEFAULT_CHECKPOINTS_REPO, dry_run: bool = False,
             force: bool = False, create: bool = False, private: bool = True, api=None) -> dict:
    """Pack and upload the checkpoints of the top-``top`` runs in one commit."""
    selected = select_top_runs(dataset, horizon=horizon, top=top, metric=metric, records=records)
    for entry in selected:
        try:
            entry["record"], entry["checkpoint"] = find_run(entry["record"], Path("."))
        except FileNotFoundError as exc:
            entry["checkpoint"], entry["error"] = None, str(exc)
        record = json.loads(Path(entry["record"]).read_text(encoding="utf-8"))
        entry["path_in_repo"] = bundle_path(record["dataset_id"], record["model"],
                                            entry["run_id"], record.get("track"))
    summary = {"repo": repo_id, "selected": selected, "commit": None, "uploaded": [], "skipped": []}
    ready = [e for e in selected if e.get("checkpoint")]
    if dry_run or not ready:
        return summary
    api = api or _api()
    if create:
        api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    present = set() if force else _existing(api, repo_id)
    with tempfile.TemporaryDirectory(prefix="tsflab-top-") as tmp:
        staging = Path(tmp)
        for entry in ready:
            if f"{entry['path_in_repo']}/manifest.json" in present:
                summary["skipped"].append(entry["path_in_repo"])
                continue
            pack(Path(entry["record"]), Path(entry["checkpoint"]), staging / entry["path_in_repo"])
            summary["uploaded"].append(entry["path_in_repo"])
        if summary["uploaded"]:
            commit = api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(staging),
                                       commit_message=f"checkpoints: top {top} {dataset}"
                                       + (f" H={horizon}" if horizon else ""))
            summary["commit"] = commit.oid
    return summary
