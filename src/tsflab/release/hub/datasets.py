"""Publish and download benchmark dataset files through one pinned manifest.

``configs/hub/datasets.json`` maps every published file (a path relative to the
local ``dataset/`` root) to the commit that holds it and its SHA-256. Its
``upstream`` table maps files of ``upstream``-class presets to another party's
pinned URL and SHA-256; ``tsf data download`` fetches them from there and
TSFLab never re-hosts them. A preset's files are selected from its config
``path`` (a file or a directory subtree).
Downloads use only the standard library;
publishing requires ``huggingface_hub`` and explicit authorization.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tomllib

from tsflab.release.hub.fetch import download, sha256_file
from tsflab.release.hub.uri import default_repo
from tsflab.core.paths import is_packaged_root, repository_root

DEFAULT_STATIC_REPO = default_repo("TSFLab-Static")
MANIFEST_RELATIVE = Path("configs") / "hub" / "datasets.json"
_DATASET_PREFIX = "./dataset/"


@dataclass(frozen=True)
class Selection:
    """The files one dataset preset needs, as a predicate on relative paths."""

    preset: str
    base: str

    def matches(self, relative: str) -> bool:
        return relative == self.base or relative.startswith(self.base + "/")


def selection(preset: str, root: Path | None = None) -> Selection:
    """Return the file selection for one dataset preset config."""
    config = (root or repository_root()) / "configs" / "datasets" / f"{preset}.toml"
    if not config.is_file():
        raise FileNotFoundError(f"unknown dataset preset {preset!r}")
    dataset = tomllib.loads(config.read_text(encoding="utf-8")).get("dataset", {})
    path = str(dataset.get("path", ""))
    if not path.startswith(_DATASET_PREFIX):
        raise ValueError(f"preset {preset!r} has no downloadable files (path={path!r})")
    return Selection(preset=preset, base=path[len(_DATASET_PREFIX):].strip("/"))


def manifest_path(root: Path | None = None) -> Path:
    return (root or repository_root()) / MANIFEST_RELATIVE


def load_manifest(root: Path | None = None) -> dict:
    path = manifest_path(root)
    if not path.is_file():
        return {"schema_version": 1, "repo": DEFAULT_STATIC_REPO, "files": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def published_files(preset: str, manifest: dict, root: Path | None = None) -> dict[str, dict]:
    """Files of ``preset``: TSFLab-hosted entries and ``upstream`` entries (with a ``url``)."""
    chosen = selection(preset, root)
    entries = {**manifest["files"], **manifest.get("upstream", {})}
    return {name: entry for name, entry in entries.items() if chosen.matches(name)}


def _not_published(preset: str, root: Path | None) -> str:
    """Say where an unpublished preset's files come from, by its card's redistribution class."""
    klass = redistribution(preset, root)
    if klass == "upstream":
        config = (root or repository_root()) / "configs" / "datasets" / f"{preset}.toml"
        dataset = tomllib.loads(config.read_text(encoding="utf-8")).get("dataset", {})
        if dataset.get("name") == "gift_eval":
            return (f"preset {preset!r} comes from Salesforce/GiftEval, not TSFLab; run "
                    f"`tsf data prepare --from gift-eval --datasets {dataset.get('id', '')}`".rstrip())
    if klass == "script":
        return (f"preset {preset!r} may not be re-hosted; fetch it from the original source with the "
                f"command in its card (`tsf catalog show {preset}`, Protocol and pitfalls)")
    return f"preset {preset!r} is not published; see `tsf data download --list`"


def available_presets(root: Path | None = None) -> list[str]:
    """Return presets with at least one published file."""
    base = root or repository_root()
    manifest = load_manifest(base)
    names = []
    for config in sorted((base / "configs" / "datasets").glob("*.toml")):
        try:
            if published_files(config.stem, manifest, base):
                names.append(config.stem)
        except (FileNotFoundError, ValueError):
            continue
    return names


def fetch_preset(preset: str, data_root: Path = Path("dataset"), root: Path | None = None) -> list[Path]:
    """Download and verify every published file of ``preset`` under ``data_root``."""
    manifest = load_manifest(root)
    files = published_files(preset, manifest, root)
    if not files:
        raise FileNotFoundError(_not_published(preset, root))
    written = []
    for name, entry in sorted(files.items()):
        written.append(download(_entry_url(manifest, name, entry), data_root / name, entry["sha256"]))
    return written


def _entry_url(manifest: dict, name: str, entry: dict) -> str:
    """An upstream entry's own URL, else the pinned TSFLab-Static ``hf://`` URI."""
    return entry.get("url") or f"hf://datasets/{manifest['repo']}@{entry['revision']}/{name}"


def local_files(preset: str, data_root: Path = Path("dataset"), root: Path | None = None,
                *, chosen: Selection | None = None) -> list[str]:
    chosen = chosen or selection(preset, root)
    start = data_root / chosen.base
    candidates = [start] if start.is_file() else sorted(p for p in start.rglob("*") if p.is_file())
    names = [p.relative_to(data_root).as_posix() for p in candidates]
    return [name for name in names if chosen.matches(name) and not Path(name).name.startswith(".")]


#: Redistribution classes whose files TSFLab may host; ``upstream`` and ``script`` never.
REHOSTABLE = ("hosted",)


def redistribution(preset: str, root: Path | None = None) -> str:
    """The ``[source] redistribution`` class from the preset's dataset card."""
    card = (root or repository_root()) / "catalog" / "datasets" / preset / "card.toml"
    if not card.is_file():
        return "unknown"
    return str(tomllib.loads(card.read_text(encoding="utf-8")).get("source", {}).get("redistribution", "unknown"))


def publish_presets(presets: list[str], data_root: Path = Path("dataset"),
                    repo_id: str = DEFAULT_STATIC_REPO, *, paths: tuple[str, ...] = (),
                    create: bool = False, private: bool = False,
                    root: Path | None = None) -> dict[str, str]:
    """Upload local files (one commit per preset or raw ``paths`` subtree) and pin them.

    ``paths`` are directories or files relative to ``data_root`` published in
    full, for example a complete store beyond what any preset reads.
    """
    if is_packaged_root(root or repository_root()):
        raise RuntimeError("publishing datasets requires a TSFLab checkout")
    blocked = {preset: klass for preset in presets
               if (klass := redistribution(preset, root)) not in REHOSTABLE}
    if blocked:
        raise PermissionError("not re-hostable (card [source] redistribution): "
                              + ", ".join(f"{p}={k}" for p, k in sorted(blocked.items())))
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("publishing needs `tsflab[hub]`") from exc
    api = HfApi()
    if create:
        api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    manifest = load_manifest(root)
    if manifest["files"] and manifest["repo"] != repo_id:
        raise ValueError(f"the manifest already pins files in {manifest['repo']}; "
                         "publish to that repository or start a new manifest")
    manifest["repo"] = repo_id
    revisions = {}
    targets = [(preset, None) for preset in presets]
    targets += [(path.strip("/"), Selection(preset=path, base=path.strip("/"))) for path in paths]
    for preset, chosen in targets:
        names = local_files(preset, data_root, root, chosen=chosen)
        if not names:
            raise FileNotFoundError(f"no local files for {preset!r} under {data_root}")
        digests = {name: sha256_file(data_root / name) for name in names}
        pending = [
            name for name in names
            if manifest["files"].get(name, {}).get("sha256") != digests[name]
        ]
        if not pending:
            revisions[preset] = "unchanged"
            continue
        commit = api.upload_folder(
            repo_id=repo_id, repo_type="dataset", folder_path=str(data_root),
            allow_patterns=pending, commit_message=f"{preset}: {len(pending)} file(s)",
        )
        for name in pending:
            manifest["files"][name] = {
                "revision": commit.oid, "sha256": digests[name],
                "size": (data_root / name).stat().st_size,
            }
        revisions[preset] = commit.oid
    manifest["files"] = dict(sorted(manifest["files"].items()))
    target = manifest_path(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return revisions


def check_manifest(root: Path | None = None) -> list[str]:
    """Return problems with pinned files: unreachable, or size differs (HEAD only)."""
    from urllib.error import URLError
    from urllib.request import Request, urlopen

    from tsflab.release.hub.fetch import resolve_url

    manifest = load_manifest(root)
    problems = []
    for name, entry in {**manifest["files"], **manifest.get("upstream", {})}.items():
        url = resolve_url(_entry_url(manifest, name, entry))
        try:
            # HF answers HEAD on /resolve/ with the file size before any LFS redirect.
            request = Request(url, method="HEAD")
            request.add_header("Accept-Encoding", "identity")
            with urlopen(request, timeout=30) as response:
                size = response.headers.get("X-Linked-Size") or response.headers.get("Content-Length")
        except URLError as exc:
            problems.append(f"{name}: unreachable ({exc})")
            continue
        if size is not None and int(size) != entry["size"]:
            problems.append(f"{name}: size {size} != pinned {entry['size']}")
    return problems

