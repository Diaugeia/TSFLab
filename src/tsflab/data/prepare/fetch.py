"""Checksummed downloads shared by the ``tsf data prepare --from`` fetchers.

Fetchers download from the original source (never from TSFLab) and verify every
input and output against a pinned SHA-256; a mismatch is an error.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen


def cache_dir(name: str) -> Path:
    """Download cache for raw source files (``$TSFLAB_CACHE`` or ``~/.cache/tsflab``)."""
    root = os.environ.get("TSFLAB_CACHE")
    base = Path(root).expanduser() if root else Path.home() / ".cache" / "tsflab"
    return base / "sources" / name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str) -> Path:
    """Return ``path`` if its SHA-256 equals ``expected``; raise ``ValueError`` otherwise."""
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"checksum mismatch for {path}: expected {expected}, got {actual}")
    return path


def _cached(destination: Path, sha256: str) -> bool:
    return destination.is_file() and sha256_file(destination) == sha256


def _atomic(destination: Path, write) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".part", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        write(temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def download_url(url: str, destination: Path, sha256: str) -> Path:
    """Download ``url`` to ``destination`` (skipped when a verified copy exists)."""
    if _cached(destination, sha256):
        return destination

    def write(temporary: Path) -> None:
        with urlopen(url, timeout=60) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target)
        verify_sha256(temporary, sha256)

    return _atomic(destination, write)


def download_gdrive(file_id: str, destination: Path, sha256: str, *, manual: str) -> Path:
    """Download one Google Drive file with ``gdown``; ``manual`` says what to do if that fails."""
    if _cached(destination, sha256):
        return destination
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(f"Google Drive downloads need `gdown` (`uv sync --extra data`). {manual}") from exc

    def write(temporary: Path) -> None:
        try:
            written = gdown.download(id=file_id, output=str(temporary), quiet=False)
        except Exception as exc:  # gdown raises its own errors for quota or permission pages
            raise RuntimeError(f"gdown could not download Google Drive file {file_id} ({exc}). {manual}") from exc
        if written is None:
            raise RuntimeError(f"gdown could not download Google Drive file {file_id}. {manual}")
        verify_sha256(temporary, sha256)

    return _atomic(destination, write)
