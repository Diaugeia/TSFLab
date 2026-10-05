"""Unified access to published TSFLab assets on the Hugging Face Hub.

Every remote asset — pretrained foundation weights, trained checkpoints,
benchmark and real-time datasets — is addressed by one pinned URI form::

    hf://[datasets/|spaces/]<owner>/<repo>@<revision>/<path>

Downloading uses only the standard library; packing, loading, and uploading
checkpoints need ``tsflab[hub]``.
"""

from tsflab.release.hub.bundle import bundle_path, find_run, load_manifest, load_state_dict, pack
from tsflab.release.hub.fetch import default_cache_root, fetch, resolve_url
from tsflab.release.hub.datasets import (
    DEFAULT_DATASETS_REPO,
    DEFAULT_STATIC_REPO,
    available_presets,
    fetch_preset,
    publish_presets,
)
from tsflab.release.hub.publish import DEFAULT_WEIGHTS_REPO, init_repositories, list_bundles, push
from tsflab.release.hub.uri import HubURI, is_hub_uri, parse

__all__ = [
    "DEFAULT_DATASETS_REPO",
    "DEFAULT_STATIC_REPO",
    "DEFAULT_WEIGHTS_REPO",
    "available_presets",
    "bundle_path",
    "default_cache_root",
    "fetch",
    "fetch_preset",
    "find_run",
    "HubURI",
    "init_repositories",
    "is_hub_uri",
    "list_bundles",
    "load_manifest",
    "load_state_dict",
    "pack",
    "parse",
    "publish_presets",
    "push",
    "resolve_url",
]
