"""tsf result hub — results, checkpoints, and the board on the Hugging Face Hub.

    tsf result hub pack <run_id|record.json> [--out DIR]
    tsf result hub push <run_id|record.json> [--repo OWNER/NAME] [--public] [--create]
    tsf result hub push-top --dataset D [--horizon H] --top K [--metric mse] [--records DIR] [--dry-run]
    tsf result hub list [--repo OWNER/NAME] [--revision REV] [--dataset D] [--model M]
    tsf result hub pull <hf://...bundle-dir> [--json]
    tsf result hub results push <DIR...> [--dry-run] [--no-board] [--batch N] [--force]
    tsf result hub results pull [--track T] [--dataset D] [--model M] [--out DIR]
    tsf result hub results board [--local DIR] [--curated FILE] [--out DIR] [--no-upload]
    tsf result hub init [--owner OWNER] [--private] [--migrate-legacy] [--dry-run]

One model repository (default ``Diaugeia/TSFLab-Checkpoints``) holds
``results/`` (submission bundles), ``checkpoints/`` (weights of top-ranked
runs only), ``board/`` (generated leaderboard the site reads), and ``legacy/``.

Publishing is always explicit: ``push`` uploads only the run it is given,
``push-top`` only the best run of each of the top-K rows, ``results push`` only
validated bundles not yet present; ``init`` creates the published repositories
(TSFLab-Datasets, TSFLab-Checkpoints, the leaderboard Space) with their cards.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

from tsflab.release.hub.uri import DEFAULT_OWNER
from tsflab.core.paths import working_root


def _pack(run: str, out: Path) -> dict:
    from tsflab.release import hub

    record, checkpoint = hub.find_run(run, working_root())
    return hub.pack(record, checkpoint, out)


def _results_command(argv: list[str], default_repo: str) -> int:
    from tsflab.release.hub import results

    parser = argparse.ArgumentParser(prog="tsf result hub results")
    sub = parser.add_subparsers(dest="action", required=True)
    push = sub.add_parser("push", help="upload validated submission bundles to results/ and rebuild the board")
    push.add_argument("sources", nargs="*", type=Path, default=[Path("work_dirs/_submissions")],
                      help="directories searched for submission.json (default: work_dirs/_submissions)")
    push.add_argument("--repo", default=default_repo)
    push.add_argument("--dry-run", action="store_true", help="plan from local files; no network")
    push.add_argument("--force", action="store_true", help="re-upload bundles already present")
    push.add_argument("--skip-invalid", action="store_true", help="upload the valid bundles, report the rest")
    push.add_argument("--batch", type=int, default=results.DEFAULT_BATCH, help="bundles per commit")
    push.add_argument("--no-board", action="store_true", help="do not regenerate board/ afterwards")
    push.add_argument("--create", action="store_true", help="create the repo if missing")
    push.add_argument("--private", action="store_true", help="create the repo as private")
    pull = sub.add_parser("pull", help="download results/ (or part of it)")
    pull.add_argument("--repo", default=default_repo)
    pull.add_argument("--revision", default="main")
    pull.add_argument("--track")
    pull.add_argument("--dataset")
    pull.add_argument("--model")
    pull.add_argument("--out", type=Path, default=Path("work_dirs/_hub"))
    board = sub.add_parser("board", help="regenerate board/leaderboard.json and board/model-meta.json")
    board.add_argument("--repo", default=default_repo)
    board.add_argument("--local", type=Path, default=None,
                       help="build from a local results mirror instead of downloading results/")
    board.add_argument("--curated", type=Path, default=None,
                       help="curated overlay (default: board/curated.json of the repo, if any)")
    board.add_argument("--out", type=Path, default=None, help="also write the board files here")
    board.add_argument("--no-upload", action="store_true", help="build only; do not commit board/")
    args = parser.parse_args(argv)

    if args.action == "push":
        summary = results.push_results(args.sources, args.repo, dry_run=args.dry_run, force=args.force,
                                       skip_invalid=args.skip_invalid, batch=args.batch,
                                       create=args.create, private=args.private)
        verb = "would upload" if args.dry_run else "uploaded"
        count = summary["found"] if args.dry_run else summary["uploaded"]
        print(f"{summary['found']} valid bundle(s), {summary['rejected']} rejected, "
              f"{summary['skipped']} already present; {verb} {count} to {args.repo}")
        for path in summary["paths"][:10]:
            print(f"  {path}")
        if len(summary["paths"]) > 10:
            print(f"  ... {len(summary['paths']) - 10} more")
        for oid in summary["commits"]:
            print(f"  commit {oid}")
        if args.dry_run or args.no_board:
            return 0
        if summary["uploaded"] == 0:
            print("board unchanged (nothing uploaded)")
            return 0
        board_summary = results.publish_board(args.repo)
        print(f"board: {board_summary['n_submissions']} submission(s), commit {board_summary['commit']}")
        return 0
    if args.action == "pull":
        out = results.pull_results(args.repo, args.out, revision=args.revision, track=args.track,
                                   dataset=args.dataset, model=args.model)
        print(f"results downloaded under {out / results.RESULTS_PREFIX}")
        return 0
    summary = results.publish_board(args.repo, local=args.local, curated=args.curated,
                                    out_dir=args.out, upload=not args.no_upload)
    print(f"board: {summary['n_submissions']} submission(s), {summary['n_rejected']} rejected"
          + (f"; written to {summary['out_dir']}" if summary["out_dir"] else "")
          + (f"; commit {summary['commit']}" if summary["commit"] else ""))
    return 0


def _push_top(args) -> int:
    from tsflab.release.hub import results

    summary = results.push_top(args.dataset, horizon=args.horizon, top=max(1, args.top), metric=args.metric,
                               records=[Path(p) for p in (args.records or ["work_dirs"])],
                               repo_id=args.repo, dry_run=args.dry_run, force=args.force,
                               create=args.create, private=not args.public)
    if not summary["selected"]:
        print(f"no local runs with {args.metric} for dataset {args.dataset!r}", file=sys.stderr)
        return 1
    for entry in summary["selected"]:
        state = "ready" if entry.get("checkpoint") else f"missing checkpoint ({entry.get('error')})"
        print(f"{entry['rank']:>2}  {entry['model']:<18} H={entry['horizon']}  "
              f"{args.metric}={entry['row_metric']:.4f} (n={entry['n_runs']}; best run "
              f"{entry['run_metric']:.4f})  {entry['path_in_repo']}  {state}")
    if args.dry_run:
        print("(dry run — nothing uploaded)")
        return 0
    for path in summary["skipped"]:
        print(f"already present: {path}")
    if summary["commit"]:
        print(f"Published {len(summary['uploaded'])} checkpoint(s): commit {summary['commit']}")
    return 0


def hub_command(argv: list[str]) -> int:
    from tsflab.release import hub

    if argv[:1] == ["results"]:
        try:
            return _results_command(argv[1:], hub.DEFAULT_CHECKPOINTS_REPO)
        except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    parser = argparse.ArgumentParser(prog="tsf result hub", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    pack = sub.add_parser("pack", help="write a local weights bundle for one run")
    pack.add_argument("run")
    pack.add_argument("--out", type=Path, default=None)
    push = sub.add_parser("push", help="pack one run and upload it to checkpoints/")
    push.add_argument("run")
    push.add_argument("--repo", default=hub.DEFAULT_CHECKPOINTS_REPO)
    push.add_argument("--public", action="store_true", help="create the repo as public")
    push.add_argument("--create", action="store_true", help="create the repo if missing")
    top = sub.add_parser("push-top", help="upload the checkpoints of the top-K ranked local runs")
    top.add_argument("--dataset", required=True)
    top.add_argument("--horizon", default=None, help="prediction length; default every horizon found")
    top.add_argument("--top", type=int, default=3, help="rows per horizon (default 3)")
    top.add_argument("--metric", default="mse", help="ranking metric, lower is better (default mse)")
    top.add_argument("--records", action="append", default=None,
                     help="work_dir to scan for records (repeatable; default ./work_dirs)")
    top.add_argument("--repo", default=hub.DEFAULT_CHECKPOINTS_REPO)
    top.add_argument("--dry-run", action="store_true", help="show the selection; upload nothing")
    top.add_argument("--force", action="store_true", help="re-upload bundles already present")
    top.add_argument("--public", action="store_true", help="create the repo as public")
    top.add_argument("--create", action="store_true", help="create the repo if missing")
    listing = sub.add_parser("list", help="list checkpoint bundles in the repo")
    listing.add_argument("--repo", default=hub.DEFAULT_CHECKPOINTS_REPO)
    listing.add_argument("--revision", default="main")
    listing.add_argument("--dataset")
    listing.add_argument("--model")
    pull = sub.add_parser("pull", help="download and verify one bundle")
    pull.add_argument("uri")
    pull.add_argument("--json", action="store_true")
    sub.add_parser("results", help="results/ and board/: push, pull, board (see `results --help`)")
    init = sub.add_parser("init", help="create the published repositories and their cards")
    init.add_argument("--owner", default=DEFAULT_OWNER)
    init.add_argument("--private", action="store_true")
    init.add_argument("--migrate-legacy", action="store_true",
                      help="rename existing TSEval repositories instead of creating new ones")
    init.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        if args.action == "pack":
            out = args.out or Path("work_dirs/_bundles") / args.run
            manifest = _pack(args.run, out)
            print(f"Bundle written: {out}  ({manifest['path']})")
            return 0
        if args.action == "push":
            with tempfile.TemporaryDirectory(prefix="tsflab-bundle-") as tmp:
                manifest = _pack(args.run, Path(tmp))
                uri = hub.push(Path(tmp), manifest["path"], args.repo,
                               private=not args.public, create=args.create)
            print(f"Published: {uri}")
            return 0
        if args.action == "push-top":
            return _push_top(args)
        if args.action == "init":
            for name in hub.init_repositories(args.owner, private=args.private,
                                              migrate=args.migrate_legacy, dry_run=args.dry_run):
                print(("would " if args.dry_run else "ready ") + name)
            return 0
        if args.action == "list":
            for path in hub.list_bundles(args.repo, args.revision, args.dataset, args.model):
                print(path)
            return 0
        state_dict, manifest = hub.load_state_dict(args.uri)
        if args.json:
            print(json.dumps(manifest, indent=2, sort_keys=True))
        else:
            print(f"{manifest['model']} / {manifest['dataset_id']} / pred_len "
                  f"{manifest['pred_len']}: {len(state_dict)} tensors verified")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
