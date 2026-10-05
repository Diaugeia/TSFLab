"""tsflab.release: hf:// addressing, weights bundles, and pinned dataset publishing (no network)."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from pydantic import BaseModel

from tsflab.catalog.model_artifacts import fetch_artifact
from tsflab.catalog.registry.models import ModelArtifact, ModelSpec
from tsflab.core.paths import repository_root
from tsflab.release import hub
from tsflab.release.hub import datasets as hd

# ---------------------------------------------------------------------------
# hf:// URIs, artifacts, and weights bundles
# ---------------------------------------------------------------------------


REPO = "Diaugeia/TSFLab-Checkpoints"


REV = "0123abcd"


def test_parse_round_trips_every_repo_type() -> None:
    for text, repo_type in (
        (f"hf://{REPO}@{REV}/weather/DLinear/run/model.safetensors", "model"),
        ("hf://datasets/Diaugeia/TSFLab-Datasets@main/static/ett/ETTh1.csv", "dataset"),
        ("hf://spaces/Diaugeia/TSFLab@v1/index.html", "space"),
    ):
        uri = hub.parse(text)
        assert uri.repo_type == repo_type
        assert str(uri) == text
    uri = hub.parse("hf://datasets/Diaugeia/TSFLab-Datasets@main/static/ett/ETTh1.csv")
    assert uri.resolve_url() == (
        "https://huggingface.co/datasets/Diaugeia/TSFLab-Datasets/resolve/main/static/ett/ETTh1.csv"
    )


@pytest.mark.parametrize(
    "text",
    [
        f"hf://{REPO}/model.safetensors",  # unpinned
        f"hf://{REPO}@{REV}/../secret",  # traversal
        f"hf://{REPO}@{REV}/a//b",  # empty segment
        f"https://huggingface.co/{REPO}",  # wrong scheme
    ],
)
def test_parse_rejects_unpinned_or_unsafe_uris(text: str) -> None:
    with pytest.raises(ValueError):
        hub.parse(text)


def _serve(root: Path, path: str, payload: bytes) -> None:
    """Lay out ``payload`` as a file:// endpoint would serve an hf:// URI."""
    target = root / REPO / "resolve" / REV / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)


class _Params(BaseModel):
    width: int = 1


def test_model_artifacts_accept_pinned_hf_uris(tmp_path: Path, monkeypatch) -> None:
    payload = b"foundation weights"
    _serve(tmp_path / "hub", "fm/weights.bin", payload)
    monkeypatch.setenv("HF_ENDPOINT", (tmp_path / "hub").as_uri())
    artifact = ModelArtifact(
        name="weights",
        url=f"hf://{REPO}@{REV}/fm/weights.bin",
        revision=REV,
        sha256=hashlib.sha256(payload).hexdigest(),
        filename="weights.bin",
        required=True,
    )
    spec = ModelSpec(
        name="Fixture",
        module="tsflab.models.fixture",
        model_class=nn.Identity,
        factory=lambda cfg, params: nn.Identity(),
        params_schema=_Params,
        capabilities=frozenset({"time-series"}),
        artifacts=(artifact,),
    )
    assert fetch_artifact(spec, artifact, tmp_path / "cache").read_bytes() == payload
    with pytest.raises(ValueError, match="pinned to its revision"):
        ModelArtifact(
            name="weights",
            url=f"hf://{REPO}@other/fm/weights.bin",
            revision=REV,
            sha256=artifact.sha256,
            filename="weights.bin",
        )


def _fake_run(work: Path) -> str:
    run_id = "DLinear_weather_sl96_pl12_seed42_1"
    model_dir = work / "work_dirs" / "weather" / "DLinear"
    (model_dir / "records").mkdir(parents=True)
    (model_dir / "checkpoints" / run_id).mkdir(parents=True)
    torch.manual_seed(0)
    torch.save(nn.Linear(96, 12).state_dict(), model_dir / "checkpoints" / run_id / "best_checkpoint.pth")
    record = {
        "model": "DLinear",
        "dataset_id": "weather",
        "track": "time_series",
        "seed": 42,
        "config": {"seq_len": 96, "label_len": 0, "pred_len": 12, "features": "M"},
        "env": {"framework_version": "0.8.0", "git_sha": "abc"},
        "results": [{"horizon": 12, "run_id": run_id, "metrics": {"mse": 0.5, "mae": None}}],
    }
    (model_dir / "records" / f"{run_id}.json").write_text(json.dumps(record))
    return run_id


def test_pack_then_load_state_dict_round_trips(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("safetensors", reason="needs the `hub` extra")
    run_id = _fake_run(tmp_path)
    record, checkpoint = hub.find_run(run_id, tmp_path)
    bundle = tmp_path / "bundle"
    manifest = hub.pack(record, checkpoint, bundle)
    assert manifest["path"] == f"checkpoints/time_series/weather/DLinear/{run_id}"
    assert manifest["metrics"] == {"mse": 0.5}
    assert (bundle / "README.md").read_text().startswith("---\nlibrary_name: tsflab")

    served = tmp_path / "hub"
    for file in bundle.iterdir():
        _serve(served, f"{manifest['path']}/{file.name}", file.read_bytes())
    monkeypatch.setenv("HF_ENDPOINT", served.as_uri())
    uri = f"hf://{REPO}@{REV}/{manifest['path']}"
    state_dict, loaded = hub.load_state_dict(uri, cache_root=tmp_path / "cache")
    expected = torch.load(checkpoint, weights_only=True)
    assert loaded == manifest
    assert all(torch.equal(state_dict[key], expected[key]) for key in expected)

    weights = served / REPO / "resolve" / REV / manifest["path"] / "model.safetensors"
    weights.write_bytes(weights.read_bytes()[:-1] + b"\0")
    with pytest.raises(ValueError, match="checksum mismatch"):
        hub.load_state_dict(uri, cache_root=tmp_path / "fresh-cache")


def test_hub_owner_comes_from_the_environment(monkeypatch) -> None:
    import importlib

    from tsflab.release.hub import uri

    monkeypatch.setenv("TSFLAB_HUB_OWNER", "someone")
    try:
        assert importlib.reload(uri).default_repo("TSFLab-Datasets") == "someone/TSFLab-Datasets"
    finally:
        monkeypatch.delenv("TSFLAB_HUB_OWNER")
        importlib.reload(uri)
    assert uri.default_repo("TSFLab-Datasets") == "Diaugeia/TSFLab-Datasets"


# ---------------------------------------------------------------------------
# Dataset publishing and download selection
# ---------------------------------------------------------------------------


ROOT = repository_root()


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_selection_covers_file_and_directory_presets() -> None:
    assert hd.selection("etth1").matches("ETT-small/ETTh1.csv")
    assert not hd.selection("etth1").matches("ETT-small/ETTh2.csv")
    pems = hd.selection("pems08")
    assert pems.matches("pems08/his.npz") and not pems.matches("pems08x/his.npz")
    with pytest.raises(FileNotFoundError):
        hd.selection("synthetic_st")  # a test fixture, not a catalog dataset


def test_fetch_preset_downloads_only_pinned_files_and_verifies(tmp_path, monkeypatch) -> None:
    root = tmp_path / "repo"
    shutil.copytree(ROOT / "configs" / "datasets", root / "configs" / "datasets")
    payload = b"date,OT\n2020-01-01,1\n"
    manifest = {"schema_version": 1, "repo": "o/r", "files": {
        "ETT-small/ETTh1.csv": {"revision": "abc", "sha256": hashlib.sha256(payload).hexdigest(),
                                "size": len(payload)},
        "ETT-small/ETTh2.csv": {"revision": "abc", "sha256": "0" * 64, "size": 1},
    }}
    _write_bytes(root / hd.MANIFEST_RELATIVE, json.dumps(manifest).encode())
    calls = []

    def fake_download(uri: str, destination: Path, sha256: str | None = None) -> Path:
        calls.append(uri)
        _write_bytes(destination, payload)
        return destination

    monkeypatch.setattr(hd, "download", fake_download)
    written = hd.fetch_preset("etth1", tmp_path / "dataset", root)
    assert calls == ["hf://datasets/o/r@abc/ETT-small/ETTh1.csv"]
    assert written[0].read_bytes() == payload
    assert set(hd.available_presets(root)) == {"etth1", "etth2"}
    with pytest.raises(FileNotFoundError):
        hd.fetch_preset("weather", tmp_path / "dataset", root)


def test_local_files_respects_selection(tmp_path) -> None:
    data = tmp_path / "dataset"
    for name in ("pems08/his.npz", "pems08/adj.npz", "pems08x/his.npz", "ultratraffic/PEMS_BA/static/2023.parquet"):
        _write_bytes(data / name, b"x")
    assert hd.local_files("pems08", data) == ["pems08/adj.npz", "pems08/his.npz"]
    whole = hd.Selection(preset="ultratraffic", base="ultratraffic")
    assert hd.local_files("ultratraffic", data, chosen=whole) == ["ultratraffic/PEMS_BA/static/2023.parquet"]


def test_v2_manifest_urls_use_the_prefix_and_v1_urls_the_root() -> None:
    entry = {"revision": "abc", "sha256": "0" * 64, "size": 1}
    v2 = {"schema_version": 2, "repo": "o/TSFLab-Datasets", "prefix": "static", "files": {}}
    v1 = {"schema_version": 1, "repo": "o/TSFLab-Static", "files": {}}
    assert hd._entry_url(v2, "ETT-small/ETTh1.csv", entry) == \
        "hf://datasets/o/TSFLab-Datasets@abc/static/ETT-small/ETTh1.csv"
    assert hd._entry_url(v1, "ETT-small/ETTh1.csv", entry) == \
        "hf://datasets/o/TSFLab-Static@abc/ETT-small/ETTh1.csv"
    upstream = {"url": "https://example.org/x.csv", "sha256": "0" * 64, "size": 1}
    assert hd._entry_url(v2, "x.csv", upstream) == "https://example.org/x.csv"
    assert hd.DEFAULT_STATIC_REPO == hd.DEFAULT_DATASETS_REPO == hub.DEFAULT_DATASETS_REPO


def _fixture_repo(tmp_path: Path, manifest: dict) -> Path:
    """A checkout with the shipped presets and cards and its own hub manifest."""
    root = tmp_path / "repo"
    shutil.copytree(ROOT / "configs" / "datasets", root / "configs" / "datasets")
    shutil.copytree(ROOT / "catalog" / "datasets" / "etth1", root / "catalog" / "datasets" / "etth1")
    _write_bytes(root / hd.MANIFEST_RELATIVE, json.dumps(manifest).encode())
    return root


class _FakeHfApi:
    uploads: list[dict] = []

    def create_repo(self, *args, **kwargs) -> None:
        pass

    def upload_folder(self, **kwargs):
        type(self).uploads.append(kwargs)
        return type("Commit", (), {"oid": f"rev{len(type(self).uploads)}"})()


def test_publish_uploads_under_the_static_prefix(tmp_path, monkeypatch) -> None:
    import huggingface_hub

    upstream = {"x/x.csv": {"url": "https://example.org/x.csv", "sha256": "0" * 64, "size": 1}}
    root = _fixture_repo(tmp_path, {"schema_version": 2, "repo": "o/TSFLab-Datasets", "prefix": "static",
                                    "files": {}, "upstream": upstream})
    data = tmp_path / "dataset"
    _write_bytes(data / "ETT-small/ETTh1.csv", b"date,OT\n2020-01-01,1\n")
    _FakeHfApi.uploads = []
    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeHfApi)
    assert hd.publish_presets(["etth1"], data, "o/TSFLab-Datasets", root=root) == {"etth1": "rev1"}
    (upload,) = _FakeHfApi.uploads
    assert upload["path_in_repo"] == "static" and upload["allow_patterns"] == ["ETT-small/ETTh1.csv"]
    assert upload["folder_path"] == str(data) and upload["repo_type"] == "dataset"
    manifest = hd.load_manifest(root)
    assert manifest["schema_version"] == 2 and manifest["prefix"] == "static"
    assert manifest["upstream"] == upstream
    assert hd._entry_url(manifest, "ETT-small/ETTh1.csv", manifest["files"]["ETT-small/ETTh1.csv"]) == \
        "hf://datasets/o/TSFLab-Datasets@rev1/static/ETT-small/ETTh1.csv"
    # Unchanged files are not uploaded again.
    assert hd.publish_presets(["etth1"], data, "o/TSFLab-Datasets", root=root) == {"etth1": "unchanged"}
    assert len(_FakeHfApi.uploads) == 1


def test_publish_keeps_a_v1_manifest_at_the_repo_root(tmp_path, monkeypatch) -> None:
    import huggingface_hub

    pinned = {"ETT-small/ETTh2.csv": {"revision": "old", "sha256": "0" * 64, "size": 1}}
    root = _fixture_repo(tmp_path, {"schema_version": 1, "repo": "o/TSFLab-Static", "files": pinned})
    data = tmp_path / "dataset"
    _write_bytes(data / "ETT-small/ETTh1.csv", b"x")
    _FakeHfApi.uploads = []
    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeHfApi)
    with pytest.raises(ValueError, match="already pins files in o/TSFLab-Static"):
        hd.publish_presets(["etth1"], data, "o/TSFLab-Datasets", root=root)
    hd.publish_presets(["etth1"], data, "o/TSFLab-Static", root=root)
    assert _FakeHfApi.uploads[0]["path_in_repo"] is None
    assert "prefix" not in hd.load_manifest(root)


def test_hub_init_creates_one_datasets_repo_with_both_folders() -> None:
    from tsflab.release.hub import publish

    plan = publish.repository_plan("o")
    assert [(item["repo_id"], item["repo_type"]) for item in plan] == [
        ("o/TSFLab-Datasets", "dataset"), ("o/TSFLab-Checkpoints", "model"), ("o/TSFLab", "space")]
    card = plan[0]["card"]
    assert "## `static/`" in card and "## `realtime/`" in card
    hosted, fetched = (line for line in card.splitlines() if line.startswith(("- Hosted", "- Not redistributed")))
    assert "`grid_ercot`" in hosted and "`stock_hs300`" not in hosted
    assert all(f"`{t}`" in fetched for t in ("stock_hs300", "stock_nasdaq100", "stock_sp500"))
    assert "TSFLab-Static" not in publish.LEGACY_NAMES
    dry = publish.init_repositories("o", migrate=True, dry_run=True)
    assert dry[0] == "create dataset:o/TSFLab-Datasets"


def test_packaged_manifest_is_well_formed() -> None:
    manifest = hd.load_manifest()
    assert manifest["repo"] == hd.DEFAULT_DATASETS_REPO
    assert manifest["schema_version"] == hd.MANIFEST_SCHEMA_VERSION == 2
    assert manifest["prefix"] == hd.STATIC_PREFIX == "static"
    for name, entry in manifest["files"].items():
        assert not name.startswith("/") and set(entry) == {"revision", "sha256", "size"}
    for name, entry in manifest.get("upstream", {}).items():
        assert not name.startswith("/") and set(entry) == {"url", "sha256", "size"}
        assert name not in manifest["files"]


def test_publish_refuses_presets_we_may_not_rehost(tmp_path) -> None:
    from tsflab.release.hub import datasets as hub_datasets

    root = Path(__file__).resolve().parents[1]
    assert hub_datasets.REHOSTABLE == ("hosted",)
    assert hub_datasets.redistribution("etth1", root) == "hosted"
    assert hub_datasets.redistribution("exchange", root) == "upstream"
    assert hub_datasets.redistribution("fred_md", root) == "script"
    assert hub_datasets.redistribution("gift_eval/m4_daily", root) == "upstream"
    with pytest.raises(PermissionError, match="exchange=upstream, fred_md=script"):
        hub_datasets.publish_presets(["etth1", "exchange", "fred_md"], tmp_path, root=root)


def test_redistribution_class_counts() -> None:
    import tomllib
    from collections import Counter

    root = Path(__file__).resolve().parents[1]
    cards = sorted((root / "catalog" / "datasets").rglob("card.toml"))
    counts = Counter(tomllib.loads(card.read_text(encoding="utf-8"))["source"]["redistribution"] for card in cards)
    assert counts == {"hosted": 29, "upstream": 57, "script": 8}  # upstream: 55 GIFT-Eval sets, family, exchange


def test_upstream_entries_download_from_their_own_url(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[1]
    entry = hd.load_manifest(root)["upstream"]["exchange_rate/exchange_rate.csv"]
    assert entry["url"].startswith("https://huggingface.co/datasets/thuml/Time-Series-Library/")
    calls = []
    monkeypatch.setattr(hd, "download", lambda url, dest, sha: calls.append((url, dest, sha)) or dest)
    hd.fetch_preset("exchange", tmp_path, root)
    assert calls == [(entry["url"], tmp_path / "exchange_rate/exchange_rate.csv", entry["sha256"])]
    assert "exchange" in hd.available_presets(root)


def test_unpublished_presets_name_their_fetch_command() -> None:
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(FileNotFoundError, match="--from gift-eval --datasets electricity/15T"):
        hd.fetch_preset("gift_eval/electricity_15T", root=root)
    with pytest.raises(FileNotFoundError, match="may not be re-hosted"):
        hd.fetch_preset("fred_md", root=root)


def test_hf_token_falls_back_to_the_login_file(tmp_path, monkeypatch) -> None:
    from tsflab.release.hub.fetch import hf_token

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert hf_token() is None
    (tmp_path / "token").write_text("hf_abc\n", encoding="utf-8")
    assert hf_token() == "hf_abc"
    monkeypatch.setenv("HF_TOKEN", "hf_env")
    assert hf_token() == "hf_env"


# ---------------------------------------------------------------------------
# Runner checkpoint layout, results on the Hub, top-K checkpoints, bulk submit
# ---------------------------------------------------------------------------


def _record_doc(model: str, dataset: str, run_id: str, mse: float, seed: int = 0, horizon: int = 12) -> dict:
    return {
        "record_id": f"{model}__{dataset}__seed{seed}__pl{horizon}", "model": model, "dataset_id": dataset,
        "track": "time_series", "mode": "time_series", "seed": seed,
        "results": [{"horizon": horizon, "run_id": run_id, "metrics": {"mse": mse, "mae": mse / 2}}],
        "config": {"mode": "time_series", "seq_len": 96, "label_len": 0, "pred_len": horizon, "features": "M"},
        "env": {"git_sha": "abc"},
    }


def _runner_run(work_dirs: Path, model: str, dataset: str, run_id: str, mse: float, seed: int = 0,
                *, checkpoint: bool = True, recorded_path: str | None = None) -> Path:
    """Lay out one run as the managed runner writes it (records + _runs/<run_id>/)."""
    records = work_dirs / dataset / model / "records"
    records.mkdir(parents=True, exist_ok=True)
    record = records / f"{run_id}.json"
    record.write_text(json.dumps(_record_doc(model, dataset, run_id, mse, seed)))
    run_dir = work_dirs / "_runs" / run_id
    (run_dir / "checkpoints").mkdir(parents=True)
    best = run_dir / "checkpoints" / "best_checkpoint.pth"
    if checkpoint:
        torch.manual_seed(seed)
        torch.save(nn.Linear(96, 12).state_dict(), best)
    (run_dir / "result.json").write_text(json.dumps({
        "metrics": {"mse": mse}, "train_time_sec": 1.0, "test_time_sec": 0.1,
        "checkpoint_path": recorded_path or str(best), "run_id": run_id}))
    return record


def test_find_run_supports_the_managed_runner_layout(tmp_path: Path) -> None:
    work = tmp_path / "work_dirs"
    _runner_run(work, "DLinear", "weather", "DLinear-1-aa", 0.5)
    record, checkpoint = hub.find_run("DLinear-1-aa", tmp_path)
    assert record == work / "weather" / "DLinear" / "records" / "DLinear-1-aa.json"
    assert checkpoint == work / "_runs" / "DLinear-1-aa" / "checkpoints" / "best_checkpoint.pth"
    # A run copied back from another machine keeps a foreign absolute checkpoint_path.
    _runner_run(work, "DLinear", "weather", "DLinear-2-bb", 0.4,
                recorded_path="/gpu-node/work_dirs/_runs/DLinear-2-bb/checkpoints/best_checkpoint.pth")
    expected = work / "_runs" / "DLinear-2-bb" / "checkpoints" / "best_checkpoint.pth"
    assert hub.find_run("DLinear-2-bb", tmp_path)[1] == expected
    _runner_run(work, "DLinear", "weather", "DLinear-3-cc", 0.3, checkpoint=False)
    with pytest.raises(FileNotFoundError, match="no checkpoint"):
        hub.find_run("DLinear-3-cc", tmp_path)


def test_find_run_keeps_the_older_per_model_layout(tmp_path: Path) -> None:
    run_id = _fake_run(tmp_path)
    record, checkpoint = hub.find_run(run_id, tmp_path)
    model_dir = tmp_path / "work_dirs" / "weather" / "DLinear"
    assert checkpoint == model_dir / "checkpoints" / run_id / "best_checkpoint.pth"
    assert hub.find_run(record, tmp_path) == (record, checkpoint)


class _FakeApi:
    """Records Hub writes; ``files`` is the repository listing."""

    def __init__(self, files=()):
        self.files, self.commits, self.folders = list(files), [], []

    def list_repo_files(self, repo_id, repo_type="model", revision="main"):
        return list(self.files)

    def create_commit(self, repo_id, repo_type, operations, commit_message):
        self.commits.append((commit_message, [op.path_in_repo for op in operations]))
        return type("Commit", (), {"oid": f"c{len(self.commits)}"})()

    def upload_folder(self, repo_id, repo_type, folder_path, commit_message, **_):
        self.folders.append(sorted(p.relative_to(folder_path).as_posix()
                                   for p in Path(folder_path).rglob("*") if p.is_file()))
        return type("Commit", (), {"oid": "f1"})()


def _bulk_submit(tmp_path: Path, monkeypatch) -> Path:
    from tsflab.cli.commands import submit

    work = tmp_path / "work_dirs"
    _runner_run(work, "DLinear", "ETTh1", "DLinear-1-aa", 0.5)
    _runner_run(work, "PatchTST", "ETTh1", "PatchTST-1-bb", 0.4)
    _runner_run(work, "DLinear", "weather", "DLinear-2-cc", 0.3)
    monkeypatch.setattr(submit, "trajectory_index", lambda: {})
    out = tmp_path / "subs"
    argv = ["--all", "--work-dir", str(work), "--out-dir", str(out), "--submitter", "Me"]
    assert submit.main(argv) == 0
    assert submit.main(argv + ["--skip-existing", "--dataset", "ETTh1"]) == 0
    return out


def test_bulk_submit_packages_every_record_once(tmp_path: Path, monkeypatch, capsys) -> None:
    from tsflab.core.leaderboard import load_submissions

    out = _bulk_submit(tmp_path, monkeypatch)
    assert sorted(p.name for p in out.iterdir()) == ["me__DLinear-1-aa", "me__DLinear-2-cc", "me__PatchTST-1-bb"]
    valid, rejected = load_submissions(out)
    assert len(valid) == 3 and rejected == {}
    text = capsys.readouterr().out
    assert "3 built, 0 kept, 0 failed" in text and "0 built, 2 kept, 0 failed" in text


def test_results_push_maps_bundles_and_skips_present_ones(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("huggingface_hub")
    from tsflab.release.hub import results

    out = _bulk_submit(tmp_path, monkeypatch)
    planned = results.push_results([out], dry_run=True)
    assert planned["paths"] == [
        "results/time_series/ETTh1/DLinear/me__DLinear-1-aa",
        "results/time_series/ETTh1/PatchTST/me__PatchTST-1-bb",
        "results/time_series/weather/DLinear/me__DLinear-2-cc",
    ]
    api = _FakeApi(["results/time_series/ETTh1/DLinear/me__DLinear-1-aa/submission.json"])
    summary = results.push_results([out], "o/r", batch=1, api=api)
    assert summary["skipped"] == 1 and summary["uploaded"] == 2 and summary["commits"] == ["c1", "c2"]
    assert api.commits[0][1] == [f"results/time_series/ETTh1/PatchTST/me__PatchTST-1-bb/{name}"
                                 for name in ("submission.json", "trajectory.jsonl", "report.md")]
    (out / "broken").mkdir()
    (out / "broken" / "submission.json").write_text("{}")
    with pytest.raises(ValueError, match="1 invalid submission"):
        results.push_results([out], dry_run=True)
    assert results.push_results([out], dry_run=True, skip_invalid=True)["rejected"] == 1


def test_board_is_built_from_results_with_the_site_pipeline(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("huggingface_hub")
    from tsflab.release.hub import results

    out = _bulk_submit(tmp_path, monkeypatch)
    curated = tmp_path / "curated.json"
    curated.write_text(json.dumps({"tracks": {"air_quality": {"datasets": {"Air": {"horizons": {"pm2_5": []}}}}}}))
    api = _FakeApi()
    summary = results.publish_board("o/r", local=out, curated=curated, out_dir=tmp_path / "board", api=api)
    assert summary["n_submissions"] == 3 and summary["commit"] == "c1"
    assert api.commits[0][1] == ["board/leaderboard.json", "board/model-meta.json"]
    board = json.loads((tmp_path / "board" / "leaderboard.json").read_text())
    rows = board["tracks"]["time_series"]["datasets"]["ETTh1"]["horizons"]["12"]
    assert [r["model"] for r in rows] == ["PatchTST", "DLinear"]
    assert "Air" in board["tracks"]["air_quality"]["datasets"] and "realtime" not in board
    assert "DLinear" in json.loads((tmp_path / "board" / "model-meta.json").read_text())
    assert results.results_pattern(dataset="ETTh1", metadata_only=True) == "results/*/ETTh1/*submission.json"


def test_push_top_selects_the_best_run_of_each_top_row(tmp_path: Path) -> None:
    from tsflab.release.hub import results

    work = tmp_path / "work_dirs"
    _runner_run(work, "A", "toy", "A-1", 0.30, seed=0)
    _runner_run(work, "A", "toy", "A-2", 0.20, seed=1)   # A mean 0.25, best run A-2
    _runner_run(work, "B", "toy", "B-1", 0.22, seed=0)   # B mean 0.22 -> rank 1
    _runner_run(work, "C", "toy", "C-1", 0.90, seed=0, checkpoint=False)
    picked = results.select_top_runs("toy", horizon="12", top=2, metric="mse", records=[work])
    assert [(e["rank"], e["model"], e["run_id"]) for e in picked] == [(1, "B", "B-1"), (2, "A", "A-2")]
    assert picked[1]["row_metric"] == 0.25 and picked[1]["n_runs"] == 2

    summary = results.push_top("toy", horizon="12", top=3, records=[work], dry_run=True)
    paths = {e["model"]: e["path_in_repo"] for e in summary["selected"]}
    assert paths["B"] == "checkpoints/time_series/toy/B/B-1"
    assert [e["model"] for e in summary["selected"] if not e.get("checkpoint")] == ["C"]

    pytest.importorskip("safetensors", reason="needs the `hub` extra")
    api = _FakeApi(["checkpoints/time_series/toy/B/B-1/manifest.json"])
    summary = results.push_top("toy", horizon="12", top=2, records=[work], repo_id="o/r", api=api)
    assert summary["skipped"] == ["checkpoints/time_series/toy/B/B-1"]
    assert summary["uploaded"] == ["checkpoints/time_series/toy/A/A-2"]
    assert "checkpoints/time_series/toy/A/A-2/model.safetensors" in api.folders[0]


def _load_script(path: Path, name: str):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


def test_legacy_migration_moves_every_tseval_artifact_below_legacy(tmp_path: Path, capsys) -> None:
    migrate = _load_script(ROOT / "scripts" / "migrate_legacy_to_hf.py", "migrate_legacy_to_hf")
    subs, data = tmp_path / "submissions", tmp_path / "data"
    stock = subs / "realtime" / "stock_hs300" / "OLinear" / "OLinear_x"
    stock.mkdir(parents=True)
    (stock / "submission.json").write_text("{}")  # copied verbatim, never validated
    rounds = subs / "realtime" / "stock_hs300" / "rounds" / "r1"
    rounds.mkdir(parents=True)
    (rounds / "round.json").write_text("{}")  # real-time rounds stay in Git
    static = subs / "time_series" / "ETTh1" / "SCINet" / "SCINet_run"
    static.mkdir(parents=True)
    (static / "submission.json").write_text(json.dumps(_record_doc("SCINet", "ETTh1", "SCINet_run", 0.4)))
    (subs / "README.md").write_text("staging folder")
    data.mkdir()
    (data / "leaderboard.json").write_text(json.dumps({"tracks": {"air_quality": {"datasets": {"Air": {}}}}}))
    (data / "visualization_data.json").write_text("{}")
    weights = ["realtime/stock_hs300/DLinear/a.pth", "_index.json", ".gitattributes"]
    present = {"legacy/tseval-weights/_index.json"}
    pending, done = migrate.plan(list(migrate.STEPS), subs, data, weights, present)
    assert sorted(t.dest for t in pending) == [
        "legacy/board/leaderboard.json",
        "legacy/board/visualization_data.json",
        "legacy/submissions/realtime/stock_hs300/OLinear/OLinear_x/submission.json",
        "legacy/submissions/time_series/ETTh1/SCINet/SCINet_run/submission.json",
        "legacy/tseval-weights/realtime/stock_hs300/DLinear/a.pth",
    ]
    assert [t.dest for t in done] == ["legacy/tseval-weights/_index.json"]
    # The default is a dry run; without the weights step it makes no network call.
    assert migrate.main(["--steps", "submissions", "site", "board",
                         "--submissions", str(subs), "--data", str(data)]) == 0
    assert "dry run: nothing uploaded" in capsys.readouterr().out


def test_empty_results_build_an_empty_board(tmp_path: Path) -> None:
    pytest.importorskip("huggingface_hub")
    from tsflab.release.hub import results

    (tmp_path / "mirror").mkdir()
    summary = results.publish_board("o/r", local=tmp_path / "mirror", out_dir=tmp_path / "board", upload=False)
    board = json.loads((tmp_path / "board" / "leaderboard.json").read_text())
    assert summary["n_submissions"] == 0 and board["tracks"] == {} and board["generated_at"]


def test_site_fetch_installs_board_files_and_attaches_realtime(tmp_path: Path, monkeypatch) -> None:
    fetch_board = _load_script(ROOT / "apps" / "web" / "pipeline" / "fetch_board.py", "fetch_board")
    for name, target in (("BOARD", "data/leaderboard.json"), ("MODEL_META", "data/model-meta.json"),
                         ("VISUALIZATION", "public/visualization_data.json")):
        monkeypatch.setattr(fetch_board, name, tmp_path / "site" / target)
    source = tmp_path / "board"
    source.mkdir()
    (source / "leaderboard.json").write_text(json.dumps({"n_submissions": 2, "tracks": {}}))
    fetch_board.install(source)
    board = json.loads((tmp_path / "site" / "data" / "leaderboard.json").read_text())
    assert board["n_submissions"] == 2 and board["realtime_tracks"]  # from configs/realtime/*.toml
    assert "DLinear" in json.loads((tmp_path / "site" / "data" / "model-meta.json").read_text())
    assert not (tmp_path / "site" / "public" / "visualization_data.json").exists()
