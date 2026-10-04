"""tsflab.data: profiles, loaders, converters, and calendar marks (synthetic data only)."""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tsflab.catalog.cards.metadata import model_records
from tsflab.catalog.components import COMPONENT_CATALOG
from tsflab.catalog.registry.losses import LOSS_NAME_MAP
from tsflab.cli.commands import dataset_analyze
from tsflab.cli.commands import dataset_characteristics as dc
from tsflab.cli.commands.convert_traffic import split_windows, train_range_stats
from tsflab.data import profile as prof
from tsflab.data.calendar import node_calendar, time_marks
from tsflab.data.datasets.custom import Dataset_Custom
from tsflab.data.datasets.gift_eval import (
    Dataset_GiftEval,
    _build_stamp,
    _norm_freq,
    _pandas_freq,
)
from tsflab.data.datasets.solar import Dataset_Solar
from tsflab.data.prepare.ultratraffic import convert
from tsflab.data.ultratraffic_store import load_panel

# ---------------------------------------------------------------------------
# Dataset profile (tsf data analyze)
# ---------------------------------------------------------------------------


ROOT = Path(__file__).resolve().parents[1]


PERIOD = 24


def _series(n=2400, period=PERIOD, slope=0.0, season=1.0, noise=0.3, channels=3, seed=0, shared=False):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    base = rng.normal(size=(n, channels)) * noise
    wave = season * np.sin(2 * np.pi * t / period)[:, None]
    out = base + slope * t[:, None] / n * 10 + wave
    if shared:
        out = out + 3 * rng.normal(size=(n, 1)).cumsum(axis=0) / 30
    return out


def _split(x, ratio=(0.7, 0.1, 0.2)):
    a, b = int(ratio[0] * len(x)), int((ratio[0] + ratio[1]) * len(x))
    return {"train": x[:a], "val": x[a:b], "test": x[b:]}


def _profile(x, **kw):
    return prof.build_profile(_split(x), name="syn", frequency=prof.infer_frequency(None), **kw)


def test_recovers_period_and_seasonal_strength():
    p = _profile(_series())
    top = p["train"]["periods"]["dominant"][0]
    assert abs(top["period"] - PERIOD) <= 1
    assert p["train"]["seasonal"]["strength"] > 0.8
    assert p["train"]["periods"]["forecastability"]["score"] > 0.5
    assert any(c["value"] % PERIOD == 0 for c in p["lookback_candidates"])


def test_trend_strength_separates_trend_from_flat():
    trended = _profile(_series(slope=1.0, season=0.2, noise=0.1))
    flat = _profile(_series(slope=0.0, season=1.0, noise=0.3))
    assert trended["train"]["trend"]["strength"] > 0.8
    assert flat["train"]["trend"]["strength"] < trended["train"]["trend"]["strength"]
    assert trended["train"]["stationarity"]["mean_drift"] > flat["train"]["stationarity"]["mean_drift"]


def test_white_noise_has_no_structure():
    p = _profile(np.random.default_rng(1).normal(size=(2400, 3)))
    assert p["train"]["seasonal"]["strength"] < 0.3
    assert p["train"]["periods"]["forecastability"]["score"] < 0.2
    assert "strong-seasonality" not in {r["id"] for r in prof.evaluate_rules(p)["fired"]}


def test_cross_channel_structure():
    rng = np.random.default_rng(2)
    common = rng.normal(size=(2400, 1)).cumsum(axis=0)
    correlated = common + 0.05 * rng.normal(size=(2400, 6))
    independent = rng.normal(size=(2400, 6))
    hi = _profile(correlated)["train"]["cross_channel"]
    lo = _profile(independent)["train"]["cross_channel"]
    assert hi["mean_abs_corr"] > 0.9 and lo["mean_abs_corr"] < 0.2
    assert hi["pc1_share"] > lo["pc1_share"]
    assert _profile(_series(channels=1))["train"]["cross_channel"] == {"applicable": False}


def test_zero_inflation_missingness_and_outliers():
    x = _series(channels=2)
    x[::2] = 0.0
    x[5, 0] = np.nan
    x[100, 1] = 500.0
    q = _profile(x)["train"]["quality"]
    assert q["zero_fraction"] > 0.4 and q["missing_fraction"] > 0
    assert _profile(x)["train"]["anomalies"]["outlier_fraction_5mad"] is not None


def test_shift_is_split_aware_and_test_is_diagnostic_only():
    x = _series()
    splits = _split(x)
    shifted_val = {**splits, "val": splits["val"] + 4.0}
    p = prof.build_profile(shifted_val, name="s")
    assert p["shift"]["train_to_val"]["mean_shift_median"] > 2.0
    assert "diagnostic-only" in p["shift"]["train_to_test"]["use"]
    assert prof.lookup(p, "shift.train_to_test.mean_shift_median") is not None


def test_changing_test_split_never_changes_recommendations_or_lookbacks():
    splits = _split(_series(seed=3))
    base = prof.build_profile(splits, name="a")
    tampered = prof.build_profile({**splits, "test": splits["test"] * 50 + 100}, name="a")
    assert prof.evaluate_rules(base) == prof.evaluate_rules(tampered)
    assert base["lookback_candidates"] == tampered["lookback_candidates"]
    assert base["train"] == tampered["train"]


def test_lookbacks_respect_capacity():
    p = _profile(_series(n=400))
    assert max(c["value"] for c in p["lookback_candidates"]) <= 400 * 0.7 // 4


def test_too_short_train_is_rejected():
    with pytest.raises(ValueError):
        prof.build_profile({"train": np.zeros((10, 2)), "val": np.zeros((4, 2)), "test": np.zeros((4, 2))}, name="x")


def test_frequency_inference_and_period_tags():
    import pandas as pd

    f = prof.infer_frequency(pd.date_range("2020-01-01", periods=50, freq="10min"))
    assert f["label"] == "10min" and f["steps_per_day"] == 144.0
    assert prof._period_tag(143.7, 144.0) == "daily"
    assert prof._period_tag(1008.0, 144.0) == "weekly"


def _catalog_names():
    return {str(r["name"]) for r in model_records(ROOT)}


def test_rule_table_names_exist_and_metrics_are_decision_safe():
    table = prof.load_rules()
    components, models = set(COMPONENT_CATALOG.names()), _catalog_names()
    slots = set(table["slot"])
    assert len({r["id"] for r in table["rule"]}) == len(table["rule"])

    def check(option: str, where: str):
        kind, _, value = option.partition(":")
        known = {"component": components, "model": models, "loss": set(LOSS_NAME_MAP)}
        assert kind in known and value in known[kind], f"{where}: {option}"

    for name, spec in table["slot"].items():
        for option in spec["defaults"]:
            check(option, f"slot {name}")
    for rule in table["rule"]:
        assert rule["metric"].startswith(prof.DECISION_PREFIXES), rule["id"]
        assert "train_to_test" not in rule["metric"], rule["id"]
        assert rule["op"] in prof._OPS
        assert set(rule["components"]) <= components, rule["id"]
        assert set(rule["models"]) <= models, rule["id"]
        for slot, options in rule.get("slots", {}).items():
            assert slot in slots, rule["id"]
            for option in options:
                check(option, rule["id"])


def test_every_rule_metric_resolves_in_a_real_profile():
    p = _profile(_series())
    for rule in prof.load_rules()["rule"]:
        assert prof.lookup(p, rule["metric"]) is not None, rule["metric"]


def test_seasonal_series_fires_seasonality_rule_and_promotes_slots():
    p = _profile(_series())
    result = prof.evaluate_rules(p)
    assert "strong-seasonality" in {r["id"] for r in result["fired"]}
    promoted = {o["option"] for o in result["slots"]["decomposition"]["promoted"]}
    assert "component:series_decomposition" in promoted
    assert "# Dataset profile" in prof.render_markdown({**p, "recommendations": result}, result)


def test_analyze_cli_on_csv_writes_json_and_markdown(tmp_path, capsys):
    import pandas as pd

    x = _series(n=1500, channels=2)
    frame = pd.DataFrame(x, columns=["a", "b"])
    frame.insert(0, "date", pd.date_range("2021-01-01", periods=len(x), freq="h"))
    path = tmp_path / "syn.csv"
    frame.to_csv(path, index=False)
    out = tmp_path / "out"
    assert dataset_analyze.main(["--path", str(path), "--out", str(out), "--json"]) == 0
    report = json.loads((out / "profile.json").read_text())
    assert report["dataset"]["frequency"]["label"] == "1h"
    assert abs(report["train"]["periods"]["dominant"][0]["period"] - PERIOD) <= 1
    assert report["train"]["periods"]["dominant"][0]["tag"] == "daily"
    assert (out / "profile.md").read_text().startswith("# Dataset profile")
    with pytest.raises(SystemExit):
        dataset_analyze.main([str(path), "--path", str(path)])


# ---------------------------------------------------------------------------
# Custom CSV loader
# ---------------------------------------------------------------------------


def _csv(tmp_path):
    n = 200
    frame = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n, freq="h").astype(str),
        "OT": np.arange(n, dtype=float),          # target not in the last position
        "a": np.ones(n), "b": np.full(n, 2.0),
    })
    frame.to_csv(tmp_path / "data.csv", index=False)
    return str(tmp_path)


def _load(root, features, target="OT"):
    return Dataset_Custom(root, "data.csv", (8, 0, 4), features=features, target=target, scale=False)


def test_ms_moves_target_to_last_channel(tmp_path) -> None:
    root = _csv(tmp_path)
    ms = _load(root, "MS")
    s = _load(root, "S")
    assert np.allclose(ms.data[:, -1], s.data[:, 0])
    assert ms.data.shape[1] == 3


def test_missing_target_fails_with_a_clear_error(tmp_path) -> None:
    with pytest.raises(ValueError, match="target column"):
        _load(_csv(tmp_path), "S", target="missing")


def test_missing_sentinels_are_imputed_causally_before_scaling(tmp_path) -> None:
    n = 100
    ot = np.arange(1, n + 1, dtype=float)
    ot[[0, 1, 50, 51, 99]] = -9999  # leading, interior and trailing gaps
    pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n, freq="h").astype(str),
                  "OT": ot, "a": np.ones(n)}).to_csv(tmp_path / "data.csv", index=False)
    kw = dict(root_path=str(tmp_path), data_path="data.csv", size=(8, 0, 4), features="S", target="OT",
              split_ratio=(1.0, 0.0, 0.0), scale=False)
    raw = Dataset_Custom(**kw)
    clean = Dataset_Custom(**kw, missing_sentinels=[-9999])
    assert raw.data.min() == -9999
    out = clean.data[:, 0]
    assert out[0] == out[1] == 3.0  # back fill only at the series start
    assert out[50] == out[51] == 50.0  # forward fill from the past, never from the future
    assert out[99] == 99.0
    assert out.min() == 3.0
    scaled = Dataset_Custom(**{**kw, "scale": True}, missing_sentinels=[-9999])
    assert abs(scaled.data).max() < 3


# ---------------------------------------------------------------------------
# Solar loader
# ---------------------------------------------------------------------------


def _solar_file(tmp_path, rows=300):
    path = tmp_path / "s.txt"
    np.savetxt(path, np.random.rand(rows, 2), delimiter=",")
    return str(tmp_path), "s.txt"


def test_default_marks_are_constant(tmp_path):
    root, name = _solar_file(tmp_path)
    ds = Dataset_Solar(root, name, (8, 4, 4), "train", "M", "0")
    assert np.unique(ds.time_stamp[:, 4]).size == 1


def test_start_freq_synthesises_calendar(tmp_path):
    root, name = _solar_file(tmp_path)
    ds = Dataset_Solar(root, name, (8, 4, 4), "train", "M", "0",
                       start="2006-01-01 00:00", freq="10min")
    assert ds.time_stamp[0].tolist() == [2006, 1, 1, 6, 0, 0]
    assert ds.time_stamp[7].tolist()[-2:] == [1, 10]


# ---------------------------------------------------------------------------
# GIFT-Eval loader
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stored", "start", "expected"),
    [
        ("H", None, "h"),
        ("5T", None, "5min"),
        ("15T", None, "15min"),
        ("10T", None, "10min"),
        ("10S", None, "10s"),
        ("D", None, "D"),
        ("W-FRI", None, "W-FRI"),
        ("M", "2020-01-31", "ME"),
        ("M", "2020-01-01", "MS"),
        ("Q-DEC", "1750-03-31", "QE-DEC"),
        ("Q-DEC", "1750-01-01", "QS-JAN"),
        ("Q-DEC", "1750-02-01", "QE-DEC"),
        ("A-DEC", "1750-12-31", "YE-DEC"),
        ("A-DEC", "1750-01-01", "YS-JAN"),
        ("h", None, "h"),
        ("15min", None, "15min"),
        ("QE-DEC", "1750-01-01", "QE-DEC"),
    ],
)
def test_stored_frequencies_map_to_pandas3_aliases(stored, start, expected) -> None:
    ts = pd.Timestamp(start) if start else None
    assert _pandas_freq(stored, ts) == expected
    pd.date_range(start=ts or "2020-01-01", periods=3, freq=expected)  # accepted by pandas


@pytest.mark.parametrize(
    ("freq", "code"),
    [("H", "H"), ("h", "H"), ("5T", "T"), ("15min", "T"), ("10S", "S"), ("M", "M"),
     ("ME", "M"), ("Q-DEC", "Q"), ("QE-DEC", "Q"), ("A-DEC", "A"), ("YE-DEC", "A"),
     ("W-SUN", "W"), ("D", "D")],
)
def test_norm_freq_keeps_the_legacy_short_codes(freq, code) -> None:
    assert _norm_freq(freq) == code


def test_build_stamp_starts_at_the_stored_start() -> None:
    stamp = _build_stamp(pd.Timestamp("1750-01-01"), "M", 3)
    assert stamp[:, 1].tolist() == [1, 2, 3] and stamp[:, 2].tolist() == [1, 1, 1]
    stamp = _build_stamp(pd.Timestamp("2020-01-01 00:00"), "15T", 3)
    assert stamp[:, 5].tolist() == [0, 15, 30]


def _gift_file(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    hf_datasets = pytest.importorskip("datasets")
    hf_datasets.Dataset.from_list(rows).save_to_disk(str(tmp_path / name))
    return tmp_path


def test_short_series_restaurant_style_start_and_inspection(tmp_path: Path) -> None:
    # Start stored as a one-element datetime64 array (as in GIFT-EVAL restaurant),
    # legacy hourly freq, and series much shorter than the base seq_len.
    rng = np.random.default_rng(0)
    rows = [{"item_id": f"s{i}", "start": np.array(["2016-01-13T00:00:00"], dtype="datetime64[s]"),
             "freq": "H", "target": rng.normal(size=120 + 10 * i).astype(np.float32)}
            for i in range(3)]
    root = _gift_file(tmp_path, "tiny", rows)
    size = (512, 0, 12)
    train = Dataset_GiftEval(str(root), "tiny", size, flag="train")
    assert len(train) == 0  # no window fits seq_len=512
    series = train.series_array()
    assert series.shape[1] == 3 and series.shape[0] > 0  # still inspectable

    config = dc._PartialConfig(
        dataset=dc._DatasetConfig(name="gift_eval", alias=None, path=str(root), id="tiny", params={}),
        task=dc._TaskConfig(seq_len=512, label_len=0, pred_len=12, features="M",
                            seq_len_from_preset=False),
    )
    for split in ("train", "val", "test"):
        dataset, used = dc._build_dataset_for_inspect(config, split)
        assert used.task.seq_len == 12 and len(dataset) > 0
        assert dc._extract_series(dataset).shape[1] == 3

    pinned = dc._PartialConfig(dataset=config.dataset, task=dc._TaskConfig(
        seq_len=512, label_len=0, pred_len=12, features="M", seq_len_from_preset=True))
    dataset, used = dc._build_dataset_for_inspect(pinned, "train")
    assert used.task.seq_len == 512 and len(dataset) == 0  # an explicit preset value is kept


def test_extract_series_reduces_and_caps_channels() -> None:
    class Wide:
        def series_array(self):
            return np.ones((10, dc._MAX_CHANNELS + 5, 3))

    assert dc._extract_series(Wide()).shape == (10, dc._MAX_CHANNELS)

    class Neither:
        pass

    with pytest.raises(RuntimeError, match="series_array"):
        dc._extract_series(Neither())


# ---------------------------------------------------------------------------
# UltraTraffic store (history of the real-time traffic tracks)
# ---------------------------------------------------------------------------


def _year_csv(year: int, stations: list[str]) -> bytes:
    index = pd.date_range(f"{year}-01-01", periods=24 * 20, freq="h")
    t = np.arange(len(index))[:, None]
    frame = pd.DataFrame(100 + 10 * np.sin(2 * np.pi * t / 24) + np.arange(len(stations))[None, :],
                         index=index, columns=stations)
    frame.index.name = "date"
    return frame.to_csv().encode()


def _archive(tmp_path: Path) -> Path:
    path = tmp_path / "UltraTraffic_CL.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for region, short in (("PEMS_SB", "SB"), ("PEMS_NC", "NC")):
            zf.writestr(f"UltraTraffic_CL/{region}/{short}_Static/2022.csv", _year_csv(2022, ["801", "802"]))
            zf.writestr(f"UltraTraffic_CL/{region}/{short}_Static/2023.csv", _year_csv(2023, ["801", "803"]))
            zf.writestr(f"UltraTraffic_CL/{region}/{short}_CL/2023.csv", _year_csv(2023, ["801", "803"]))
            zf.writestr(f"UltraTraffic_CL/{region}/{short}_CL/2023_added.csv", _year_csv(2023, ["803"]))
    return path


def test_converter_skips_duplicates_and_redundant_cl_files(tmp_path: Path) -> None:
    manifest = convert(_archive(tmp_path), tmp_path / "store")
    assert set(manifest["regions"]) == {"PEMS_SB"}  # PEMS_NC duplicates PEMS_SAC and is skipped
    files = manifest["regions"]["PEMS_SB"]["files"]
    assert set(files) == {"static/2022.parquet", "static/2023.parquet", "cl/2023_added.parquet"}
    assert files["static/2023.parquet"]["stations"] == 2


def test_panel_station_policies(tmp_path: Path) -> None:
    convert(_archive(tmp_path), tmp_path / "store")
    root = str(tmp_path / "store")
    last = load_panel(root, "PEMS_SB", [2022, 2023], "static", "last")
    inter = load_panel(root, "PEMS_SB", [2022, 2023], "static", "intersection")
    assert list(last.columns) == ["801", "803"] and last["803"].iloc[:10].isna().all()
    assert list(inter.columns) == ["801"]


def test_calendar_covariates() -> None:
    index = pd.date_range("2026-01-05", periods=3, freq="6h")  # a Monday
    features = node_calendar(index, 4)
    assert features.shape == (3, 4, 2)
    assert features[1, 0, 0] == 0.25 and features[0, 0, 1] == 0.0


def test_store_reader_is_torch_free() -> None:
    import subprocess

    code = "import sys, tsflab.data.ultratraffic_store, tsflab.data.calendar; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False"


# ---------------------------------------------------------------------------
# Calendar marks
# ---------------------------------------------------------------------------


def test_time_marks_layout():
    index = pd.date_range("2023-01-02 05:00", periods=3, freq="h")  # a Monday
    marks = time_marks(index)
    assert marks.shape == (3, 6)
    assert marks[0].tolist() == [2023, 1, 2, 0, 5, 0]
    assert marks[2, 4] == 7


# ---------------------------------------------------------------------------
# Traffic converter
# ---------------------------------------------------------------------------


def test_stats_ignore_validation_and_test_rows():
    t = 200
    values = np.zeros((t, 3, 1), np.float32)
    values[150:] = 1000.0  # lies beyond every training window
    train, _, _ = split_windows(t, 12, 12, (0.7, 0.1, 0.2))
    mean, std, train_end = train_range_stats(values, train, 12)
    assert train_end <= 150
    assert float(mean[0]) == 0.0 and float(std[0]) == 0.0


def test_default_horizon_is_pems_12(monkeypatch, tmp_path):
    import sys

    from tsflab.cli.commands import convert_traffic

    np.save(tmp_path / "v.npy", np.random.rand(100, 4).astype(np.float32))
    monkeypatch.setattr(sys, "argv", ["x", "--values", str(tmp_path / "v.npy"),
                                      "--output-dir", str(tmp_path / "out")])
    convert_traffic.main()
    bundle = np.load(tmp_path / "out" / "his.npz")
    assert int(bundle["seq_len"]) == 12 and int(bundle["pred_len"]) == 12


# ---------------------------------------------------------------------------
# Dataset configuration and registry
# ---------------------------------------------------------------------------


def test_dataset_default_uses_the_ignored_local_data_layer() -> None:
    from tsflab.experiments.config.schema.dataset import DatasetConfig

    fields = DatasetConfig.model_fields
    assert fields["path"].default == ""
    assert "root_path" not in fields and "data_path" not in fields
    with pytest.raises(ValueError):
        DatasetConfig.model_validate({"name": "custom", "root_path": "dataset", "data_path": "x.csv"})


def test_dataset_storage_contract_resolves_path_and_optional_id() -> None:
    from tsflab.catalog.registry.datasets import (
        DATASET_REGISTRY,
        register_dataset_by_name,
    )

    for name in ("weather", "synthetic_st", "gift_eval"):
        register_dataset_by_name(name)
    assert DATASET_REGISTRY.get("weather").resolve_location("./dataset/weather/weather.csv", None) == (
        "./dataset/weather", "weather.csv")
    assert DATASET_REGISTRY.get("synthetic_st").resolve_location("", None) == ("", "")
    assert DATASET_REGISTRY.get("gift_eval").resolve_location("./dataset/gift_eval", "electricity/15T") == (
        "./dataset/gift_eval", "electricity/15T")


# ---------------------------------------------------------------------------
# Source fetchers (tsf data prepare --from tfb|dcrnn), no network
# ---------------------------------------------------------------------------


def _tfb_long(rotate: bool = False) -> pd.DataFrame:
    dates = ["2020-01-01", "2020-01-02", "2020-01-03"]
    rows = []
    for j, (col, base) in enumerate((("b", 10.0), ("a", 20.0), ("z", 30.0))):
        labels = dates[-j:] + dates[:-j] if rotate and j else dates
        rows += [{"date": d, "data": base + i, "cols": col} for i, d in enumerate(labels)]
    return pd.DataFrame(rows)


def test_tfb_long_to_wide_keeps_first_appearance_order():
    from tsflab.data.prepare.tfb import long_to_wide

    wide = long_to_wide(_tfb_long())
    assert list(wide.columns) == ["date", "b", "a", "z"]
    assert wide["a"].tolist() == [20.0, 21.0, 22.0]
    assert list(long_to_wide(_tfb_long(), rename_last_to_ot=True).columns) == ["date", "b", "a", "OT"]
    with pytest.raises(ValueError, match="dates differ"):
        long_to_wide(_tfb_long(rotate=True))
    positional = long_to_wide(_tfb_long(rotate=True), positional=True)
    assert positional["date"].tolist() == ["2020-01-01", "2020-01-02", "2020-01-03"]
    assert positional["z"].tolist() == [30.0, 31.0, 32.0]
    with pytest.raises(ValueError, match="ragged"):
        long_to_wide(_tfb_long().iloc[:-1])


def test_tfb_convert_verifies_member_and_output_digests(tmp_path):
    import hashlib

    from tsflab.data.prepare import tfb

    payload = _tfb_long().to_csv(index=False).encode()
    archive = tmp_path / "forecasting.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("forecasting/Toy.csv", payload)
    wide = tfb.long_to_wide(_tfb_long()).to_csv(index=False).encode()
    good = tfb.TFBSet("Toy", "toy", hashlib.sha256(payload).hexdigest(), hashlib.sha256(wide).hexdigest())
    written = tfb.convert(archive, [good], tmp_path / "out")
    assert written == [tmp_path / "out" / "Toy" / "Toy.csv"] and written[0].read_bytes() == wide
    bad_output = tfb.TFBSet("Toy", "toy", good.member_sha256, "0" * 64)
    with pytest.raises(ValueError, match="checksum mismatch for converted"):
        tfb.convert(archive, [bad_output], tmp_path / "other")
    assert not (tmp_path / "other" / "Toy" / "Toy.csv").exists()
    with pytest.raises(ValueError, match="archive member"):
        tfb.convert(archive, [tfb.TFBSet("Toy", "toy", "0" * 64, good.sha256)], tmp_path / "other")


def test_tfb_sets_cover_script_class_presets():
    from tsflab.data.prepare import tfb

    assert {"fred_md", "nasdaq", "nyse", "wike2000"} <= {s.preset for s in tfb.SETS.values()}
    assert [s.name for s in tfb.resolve(["FRED-MD", "nasdaq"])] == ["FRED-MD", "NASDAQ"]
    with pytest.raises(ValueError, match="unknown TFB dataset"):
        tfb.resolve(["m4"])
    for item in tfb.SETS.values():
        config = ROOT / "configs" / "datasets" / f"{item.preset}.toml"
        assert f'"./dataset/{item.relative}"' in config.read_text(encoding="utf-8")


def test_verify_sha256_raises_on_mismatch(tmp_path):
    import hashlib

    from tsflab.data.prepare.fetch import verify_sha256

    path = tmp_path / "x.bin"
    path.write_bytes(b"abc")
    assert verify_sha256(path, hashlib.sha256(b"abc").hexdigest()) == path
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_sha256(path, "0" * 64)


def test_dcrnn_without_gdown_points_to_manual_download(tmp_path, monkeypatch, capsys):
    from tsflab.data.prepare import dcrnn

    monkeypatch.setenv("TSFLAB_CACHE", str(tmp_path))
    monkeypatch.setitem(sys.modules, "gdown", None)
    assert dcrnn.main(["--out", str(tmp_path / "metr_la")]) == 2
    assert "--h5" in capsys.readouterr().err
