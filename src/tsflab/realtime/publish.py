"""Mirror track panels to the Hugging Face datasets repository, one pinned revision per release.

Tracks live under ``realtime/<track>/`` of ``TSFLab-Datasets`` (the static
benchmark files are under ``static/``). Only tracks whose dataset card says
``[source] redistribution = "hosted"`` are ever uploaded; the others are
rebuilt from their source with ``tsf realtime update --bootstrap``.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
import tomllib

from tsflab.core.paths import repository_root
from tsflab.release.hub.fetch import sha256_file
from tsflab.release.hub.uri import default_repo
from tsflab.realtime.store import PanelStore

DEFAULT_DATASET_REPO = default_repo("TSFLab-Datasets")
#: Folder of the real-time tracks in the datasets repository.
REALTIME_PREFIX = "realtime"
#: Redistribution classes whose panels TSFLab may upload.
PUSHABLE = ("hosted",)


def repo_folder(track: str) -> str:
    """Path of one track's folder in the datasets repository."""
    return f"{REALTIME_PREFIX}/{track}"


def track_redistribution(track: str, root: Path | None = None) -> str:
    """The ``[source] redistribution`` class of ``catalog/datasets/rt/<track>/card.toml``."""
    card = (root or repository_root()) / "catalog" / "datasets" / "rt" / track / "card.toml"
    if not card.is_file():
        return "unknown"
    return str(tomllib.loads(card.read_text(encoding="utf-8")).get("source", {}).get("redistribution", "unknown"))


def push_refusal(track: str, root: Path | None = None) -> str | None:
    """Why ``track`` may not be uploaded, or ``None`` when its card allows it."""
    klass = track_redistribution(track, root)
    if klass in PUSHABLE:
        return None
    return (f"{track}: not redistributed (card [source] redistribution={klass}); the panel is not "
            f"uploaded. Fetch it from the source with `tsf realtime update --bootstrap --track {track}`")


def _api():
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("publishing needs `tsflab[hub]`") from exc
    return HfApi()


def push_release(store: PanelStore, repo_id: str = DEFAULT_DATASET_REPO, *, create: bool = False,
                 root: Path | None = None) -> str:
    """Upload the track directory to ``realtime/<track>/`` and return the new commit revision.

    Raises ``PermissionError`` when the track card does not allow redistribution.
    """
    refusal = push_refusal(store.track, root)
    if refusal:
        raise PermissionError(refusal)
    api = _api()
    if create:
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    release = store.latest_release()
    commit = api.upload_folder(
        repo_id=repo_id, repo_type="dataset", folder_path=str(store.directory),
        path_in_repo=repo_folder(store.track), commit_message=f"{store.track}: release {release['version']}",
    )
    return commit.oid


def pull_track(store: PanelStore, repo_id: str = DEFAULT_DATASET_REPO, revision: str = "main") -> bool:
    """Download a published track into the local store; ``False`` if it does not exist yet.

    The snapshot goes to a temporary directory next to the store first, and its
    ``realtime/<track>/`` folder is then moved to ``store.directory``, so any
    ``TSFLAB_REALTIME_ROOT`` layout works.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    folder = repo_folder(store.track)
    parent = store.directory.parent
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".pull-{store.track}-", dir=parent) as scratch:
        try:
            snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision,
                              allow_patterns=[f"{folder}/*", f"{folder}/**/*"], local_dir=scratch)
        except (RepositoryNotFoundError, EntryNotFoundError):
            return False
        downloaded = Path(scratch) / folder
        if not downloaded.is_dir():
            return False
        if store.directory.exists():
            shutil.copytree(downloaded, store.directory, dirs_exist_ok=True)
        else:
            os.replace(downloaded, store.directory)
    return store.exists


def file_digests(directory: Path) -> dict[str, str]:
    return {str(p.relative_to(directory)): sha256_file(p) for p in sorted(directory.rglob("*")) if p.is_file()}
