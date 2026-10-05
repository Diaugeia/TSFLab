"""Upload bundles to, and list bundles in, a Hugging Face repository.

Requires ``huggingface_hub`` (``tsflab[hub]``) and a write token in
``HF_TOKEN`` or the local ``huggingface-cli login`` credentials.
"""

from __future__ import annotations

from pathlib import Path

from tsflab.release.hub.bundle import MANIFEST
from tsflab.release.hub.uri import DEFAULT_OWNER, HubURI, default_repo

DEFAULT_WEIGHTS_REPO = default_repo("TSFLab-Weights")


def _api():
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required; install `tsflab[hub]`") from exc
    return HfApi()


def push(bundle_dir: Path, path_in_repo: str, repo_id: str = DEFAULT_WEIGHTS_REPO,
         *, private: bool = True, create: bool = False) -> HubURI:
    """Upload one bundle directory and return its URI pinned to the new commit."""
    api = _api()
    if create:
        api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(bundle_dir),
        path_in_repo=path_in_repo,
        commit_message=f"add {path_in_repo}",
    )
    return HubURI(repo_id=repo_id, revision=commit.oid, path=path_in_repo)


def list_bundles(repo_id: str = DEFAULT_WEIGHTS_REPO, revision: str = "main",
                 dataset: str | None = None, model: str | None = None) -> list[str]:
    """Return ``<dataset>/<model>/<run_id>`` directories present in ``repo_id``."""
    files = _api().list_repo_files(repo_id, repo_type="model", revision=revision)
    bundles = []
    for name in files:
        parts = name.split("/")
        if len(parts) != 4 or parts[-1] != MANIFEST:
            continue
        if (dataset and parts[0] != dataset) or (model and parts[1] != model):
            continue
        bundles.append("/".join(parts[:3]))
    return sorted(bundles)


def _card(front: dict[str, object], body: str) -> str:
    lines = ["---"] + [f"{key}: {value}" for key, value in front.items()] + ["---", "", body.strip(), ""]
    return "\n".join(lines)


_SOURCE = "https://github.com/Diaugeia/TSFLab"


def _track_classes() -> tuple[list[str], list[str]]:
    """Real-time track ids split into (hosted, not redistributed) by their dataset cards."""
    import tomllib

    from tsflab.core.paths import repository_root

    hosted, fetched = [], []
    for card in sorted((repository_root() / "catalog" / "datasets" / "rt").glob("*/card.toml")):
        source = tomllib.loads(card.read_text(encoding="utf-8")).get("source", {})
        (hosted if source.get("redistribution") == "hosted" else fetched).append(card.parent.name)
    return hosted, fetched


def _listed(names: list[str]) -> str:
    return ", ".join(f"`{name}`" for name in names) or "none"


def _datasets_card() -> str:
    hosted, fetched = _track_classes()
    return _card(
        {"license": "other", "pretty_name": "TSFLab Datasets"},
        f"""
# TSFLab datasets

Data files of [TSFLab]({_SOURCE}) in two top-level folders.

## `static/`

Files behind the TSFLab static benchmark presets, laid out exactly as the local
`dataset/` root. Each file is pinned by commit and SHA-256 in
`configs/hub/datasets.json`; fetch them with

```bash
tsf data download <preset>   # or --all
```

Only presets whose dataset card says `redistribution = "hosted"` are here.
Upstream-hosted presets download from their owner's URL, and script-class
presets are fetched from the original source.

## `realtime/`

Append-only panels of the weekly rolling tracks, one folder per track
(`realtime/<track>/`). Each weekly release is one commit, so every round is
reproducible from its pinned revision. Maintained by the `weekly` workflow.

- Hosted tracks: {_listed(hosted)}.
- Not redistributed (vendor terms): {_listed(fetched)}. These panels are never
  uploaded; build them from the source with
  `tsf realtime update --bootstrap --track <track>`.

Every dataset keeps the license of its original source; see its TSFLab
dataset card for provenance and terms. Releases before TSFLab 1.0 (0.8.0) read
the frozen `TSFLab-Static` repository instead.""")


def repository_plan(owner: str = DEFAULT_OWNER) -> list[dict[str, str]]:
    """Return the published repositories: id, type, and README card."""
    return [
        {"repo_id": f"{owner}/TSFLab-Datasets", "repo_type": "dataset", "card": _datasets_card()},
        {"repo_id": f"{owner}/TSFLab-Weights", "repo_type": "model", "card": _card(
            {"license": "mit", "library_name": "tsflab"},
            f"""
# TSFLab trained weights

Checksummed safetensors bundles at `<dataset>/<model>/<run_id>/`, published with
`tsf result hub push` and loaded with `tsf result hub pull hf://{owner}/TSFLab-Weights@<revision>/...`.
Source: [TSFLab]({_SOURCE}).""")},
        {"repo_id": f"{owner}/TSFLab", "repo_type": "space", "card": _card(
            {"title": "TSFLab Leaderboard", "emoji": '"📈"', "colorFrom": "gray",
             "colorTo": "yellow", "sdk": "static", "pinned": "false", "license": "mit"},
            f"""
# TSFLab Leaderboard

Static leaderboard auto-deployed by the `ci` workflow of
[TSFLab]({_SOURCE}).""")},
    ]


# Former TSEval repositories; moving them keeps their old URLs redirecting here.
# TSFLab-Datasets has no predecessor to rename: TSFLab-Static stays frozen for 0.8.0.
LEGACY_NAMES = {"TSFLab": "TSEval"}


def init_repositories(owner: str = DEFAULT_OWNER, *, private: bool = False,
                      migrate: bool = False, dry_run: bool = False) -> list[str]:
    """Create any missing published repositories and write their cards (idempotent).

    With ``migrate``, a missing repository whose TSEval predecessor exists is
    renamed from it instead, so the Hub redirects the old address.
    """
    plan = repository_plan(owner)
    api = None if dry_run else _api()
    created = []
    for item in plan:
        kind = item["repo_type"]
        legacy = LEGACY_NAMES.get(item["repo_id"].split("/", 1)[1]) if migrate else None
        if dry_run:
            action = f"create (or rename from {owner}/{legacy})" if legacy else "create"
            created.append(f"{action} {kind}:{item['repo_id']}")
            continue
        if legacy and not api.repo_exists(item["repo_id"], repo_type=kind) \
                and api.repo_exists(f"{owner}/{legacy}", repo_type=kind):
            api.move_repo(f"{owner}/{legacy}", item["repo_id"], repo_type=kind)
        extra = {"space_sdk": "static"} if kind == "space" else {}
        api.create_repo(item["repo_id"], repo_type=kind, private=private, exist_ok=True, **extra)
        api.upload_file(path_or_fileobj=item["card"].encode("utf-8"), path_in_repo="README.md",
                        repo_id=item["repo_id"], repo_type=kind,
                        commit_message="docs: repository card")
        created.append(f"{kind}:{item['repo_id']}")
    return created
