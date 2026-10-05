"""tsf result hub — pack, publish, list, and fetch weights on the Hugging Face Hub.

    tsf result hub pack <run_id|record.json> [--out DIR]
    tsf result hub push <run_id|record.json> [--repo OWNER/NAME] [--public] [--create]
    tsf result hub list [--repo OWNER/NAME] [--revision REV] [--dataset D] [--model M]
    tsf result hub pull <hf://...bundle-dir> [--json]
    tsf result hub init [--owner OWNER] [--private] [--migrate-legacy] [--dry-run]

Publishing is always explicit: ``push`` uploads only the run it is given, and
``init`` creates the published repositories (datasets: static and real-time,
weights, leaderboard Space) with their cards.
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


def hub_command(argv: list[str]) -> int:
    from tsflab.release import hub

    parser = argparse.ArgumentParser(prog="tsf result hub", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    pack = sub.add_parser("pack", help="write a local weights bundle for one run")
    pack.add_argument("run")
    pack.add_argument("--out", type=Path, default=None)
    push = sub.add_parser("push", help="pack one run and upload it")
    push.add_argument("run")
    push.add_argument("--repo", default=hub.DEFAULT_WEIGHTS_REPO)
    push.add_argument("--public", action="store_true", help="create the repo as public")
    push.add_argument("--create", action="store_true", help="create the repo if missing")
    listing = sub.add_parser("list", help="list bundles in a weights repo")
    listing.add_argument("--repo", default=hub.DEFAULT_WEIGHTS_REPO)
    listing.add_argument("--revision", default="main")
    listing.add_argument("--dataset")
    listing.add_argument("--model")
    pull = sub.add_parser("pull", help="download and verify one bundle")
    pull.add_argument("uri")
    pull.add_argument("--json", action="store_true")
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
