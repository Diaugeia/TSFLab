"""tsflab.realtime: tracks, store, rounds, scoring, sources, and the panel dataset (no network)."""

from __future__ import annotations

import gzip
import io
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tsflab.core.realtime import ForecastSubmission
from tsflab.data.datasets.realtime_panel import (
    Dataset_RealtimePanel_ST,
    Dataset_RealtimePanel_TS,
    load_panel,
)
from tsflab.realtime import rounds as R
from tsflab.realtime.baselines import run_baselines
from tsflab.realtime.sources import (
    airnow,
    ercot,
    openmeteo,
    sp500,
    us_prices,
)
from tsflab.realtime.sources.pems import parse_station_5min
from tsflab.realtime.store import PanelStore
from tsflab.realtime.tracks import TrackSpec, get_track, list_tracks

# ---------------------------------------------------------------------------
# Tracks, store, rounds, and scoring
# ---------------------------------------------------------------------------


TOY_TRACK = TrackSpec(id="toy", title="toy", mode="time_series", freq="h", seq_len=24, horizon=6,
                  submission_hours=2, seasonal_period=24, min_coverage=0.8, tz="America/Los_Angeles")


def _panel(start: str, hours: int, channels: int = 3) -> pd.DataFrame:
    index = pd.date_range(start, periods=hours, freq="h")
    t = np.arange(hours)[:, None]
    values = 100 + 10 * np.sin(2 * np.pi * t / 24) + np.arange(channels)[None, :]
    return pd.DataFrame(values, index=index, columns=[f"s{i}" for i in range(channels)])


def test_shipped_track_configs_load() -> None:
    ids = {track.id for track in list_tracks()}
    assert {"stock_hs300", "stock_nasdaq100", "traffic_pems_sb", "air_airnow_us"} <= ids
    assert get_track("stock_nasdaq100").source["kind"] == "nasdaq100"
    assert get_track("traffic_pems_sb").tz == "America/Los_Angeles"


def test_store_appends_without_overwriting_observed_cells(tmp_path: Path) -> None:
    store = PanelStore("toy", tmp_path)
    store.append(_panel("2026-01-01", 48))
    revised = _panel("2026-01-02", 48) + 1000  # overlaps day 2 with different values
    revised["new_sensor"] = 1.0
    store.append(revised)
    panel = store.read()
    assert list(panel.columns) == ["s0", "s1", "s2"]  # channel set stays fixed
    assert panel.loc["2026-01-02 05:00", "s0"] < 200  # history was not rewritten
    assert panel.loc["2026-01-03 05:00", "s0"] > 1000  # new timestamps were added
    assert len(store.manifest()["releases"]) == 2


def _opened_round(tmp_path: Path):
    store = PanelStore("toy", tmp_path / "store")
    store.append(_panel("2026-01-01", 24 * 10))
    cutoff = store.read().index.max()
    opened = (cutoff + pd.Timedelta(hours=1)).tz_localize(TOY_TRACK.tz)
    spec = R.open_round(TOY_TRACK, store, now=opened.to_pydatetime(), root=tmp_path / "rounds")
    return store, spec


def test_round_targets_follow_the_submission_window(tmp_path: Path) -> None:
    store, spec = _opened_round(tmp_path)
    assert spec.horizon == 6
    first = pd.Timestamp(spec.target_timestamps[0])
    assert first - pd.Timestamp(spec.opened_at) >= pd.Timedelta(hours=TOY_TRACK.submission_hours)
    assert pd.Timestamp(spec.deadline) < first
    assert first.tzinfo is not None  # every stored instant carries its offset


def test_late_or_malformed_forecasts_are_rejected(tmp_path: Path) -> None:
    _, spec = _opened_round(tmp_path)
    good = ForecastSubmission(track="toy", round_id=spec.round_id, model="M",
                              submitted_at=spec.opened_at, predictions=[[0.0] * 3] * spec.horizon)
    good.check_against(spec)
    late = good.model_copy(update={"submitted_at": spec.target_timestamps[0]})
    with pytest.raises(ValueError, match="deadline"):
        late.check_against(spec)
    with pytest.raises(ValueError, match="forecast steps"):
        good.model_copy(update={"predictions": [[0.0] * 3]}).check_against(spec)
    with pytest.raises(ValueError, match="finite"):
        good.model_copy(update={"predictions": [[float("nan")] * 3] * spec.horizon}).check_against(spec)


def test_scoring_waits_for_truth_then_ranks(tmp_path: Path) -> None:
    store, spec = _opened_round(tmp_path)
    root = tmp_path / "rounds"
    for submission in run_baselines(spec, store, TOY_TRACK):
        R.write_forecast(submission.model_copy(update={"submitted_at": spec.opened_at}), spec, root)
    assert R.score_round(spec, store, TOY_TRACK.min_coverage, root) is None  # no truth yet
    store.append(_panel("2026-01-11", 48))
    scores = R.score_round(spec, store, TOY_TRACK.min_coverage, root)
    assert [s.rank for s in scores] == [1, 2, 3]
    assert scores[0].model == "SeasonalNaive"  # exact daily seasonality in the toy panel
    assert scores[0].mse < 1e-8
    summary = R.track_summary("toy", root)
    assert summary["methods"][0]["model"] == "SeasonalNaive"


def test_kendall_tau() -> None:
    assert R.kendall_tau(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert R.kendall_tau(["a", "b", "c"], ["c", "b", "a"]) == -1.0
    assert R.kendall_tau(["a"], ["a"]) is None


def test_pems_station_5min_parser_aggregates_hourly_flow() -> None:
    rows = []
    for minute in range(0, 60, 5):
        for station, flow in (("801230", 10), ("801232", 20)):
            rows.append(f"01/05/2026 07:{minute:02d}:00,{station},8,10,E,ML,1.2,10,100,{flow},0.05,65")
    payload = gzip.compress("\n".join(rows).encode())
    hourly = parse_station_5min(payload)
    assert hourly.loc[pd.Timestamp("2026-01-05 07:00"), "801230"] == 120
    assert hourly.loc[pd.Timestamp("2026-01-05 07:00"), "801232"] == 240


def test_validate_cli_uses_trusted_arrival_time(tmp_path: Path, monkeypatch) -> None:
    from tsflab.realtime.cli import main

    _, spec = _opened_round(tmp_path)
    monkeypatch.setattr(R, "ROUNDS_ROOT", tmp_path / "rounds")
    monkeypatch.setattr(R, "load_round", lambda t, r, root=tmp_path / "rounds": R.RoundSpec.model_validate_json(
        (tmp_path / "rounds" / t / "rounds" / r / "round.json").read_text()))
    path = tmp_path / "f.json"
    path.write_text(ForecastSubmission(track="toy", round_id=spec.round_id, model="M",
                                       submitted_at=spec.opened_at,
                                       predictions=[[0.0] * 3] * spec.horizon).model_dump_json())
    assert main(["validate", str(path)]) == 0
    assert main(["validate", str(path), "--received-at", spec.target_timestamps[-1]]) == 1


def test_equity_fallback_moves_the_working_vendor_first(monkeypatch) -> None:
    from tsflab.realtime.sources import equity

    monkeypatch.setattr(equity.time, "sleep", lambda seconds: None)
    calls = []

    def blocked(symbol):
        calls.append("blocked")
        raise ConnectionError("403")

    def working(symbol):
        calls.append("working")
        return pd.Series([1.0])

    providers = [blocked, working]
    assert equity.with_fallback(providers, "AAPL").tolist() == [1.0]
    assert providers == [working, blocked]
    equity.with_fallback(providers, "MSFT")
    assert calls == ["blocked", "working", "working"]
    with pytest.raises(RuntimeError, match="could not fetch X"):
        equity.with_fallback([blocked], "X", attempts=2, label="X")


def test_equity_panel_stores_log_returns_and_resumes_from_cache(tmp_path: Path, monkeypatch) -> None:
    from tsflab.realtime.sources import equity

    monkeypatch.setenv("TSFLAB_REALTIME_ROOT", str(tmp_path))
    monkeypatch.setattr(equity.time, "sleep", lambda seconds: None)
    track = replace(TOY_TRACK, id="toy_stock", freq="B", source={"transform": "log_return"})
    days = pd.bdate_range("2026-08-24", "2026-09-11")
    closes = pd.Series(100 * np.exp(0.01 * np.arange(len(days))), index=days)
    fetched = []

    def daily_close(symbol, lo, hi):
        fetched.append(symbol)
        return closes.loc[lo:hi]

    start, end = pd.Timestamp("2026-09-01"), pd.Timestamp("2026-09-11")
    panel = equity.fetch_panel(track, start, end, ["A", "B"], daily_close)
    assert list(panel.columns) == ["A", "B"]
    assert panel.index[0] == start and panel.index[-1] == end
    np.testing.assert_allclose(panel.to_numpy(), 0.01)  # first day uses the previous close
    again = equity.fetch_panel(track, start, end, ["A", "B"], daily_close)
    assert fetched == ["A", "B"]  # second call is served from the resume cache
    pd.testing.assert_frame_equal(again, panel, check_freq=False)


def test_nasdaq_historical_fallback_parses_quoted_closes(monkeypatch) -> None:
    from tsflab.realtime.sources import nasdaq100, us_prices

    rows = [{"date": "09/25/2026", "close": "$1,341.07"}, {"date": "09/24/2026", "close": "$335.92"}]
    seen = {}

    def fake_get(path, **params):
        seen["path"] = path
        return {"tradesTable": {"rows": rows}}

    monkeypatch.setattr(us_prices, "nasdaq_get", fake_get)
    close = us_prices.nasdaq_close("BRK-B", pd.Timestamp("2026-09-24"), pd.Timestamp("2026-09-25"))
    assert seen["path"] == "quote/BRK.B/historical"
    assert close.tolist() == [335.92, 1341.07]
    assert nasdaq100._PREFERRED[:2] == [us_prices.nasdaq_close, us_prices.yahoo_close]  # Sina is last
    monkeypatch.setattr(nasdaq100, "nasdaq_get", lambda path, **params: {"data": {"rows": [
        {"symbol": "brk.b "}, {"symbol": "AAPL"}, {"symbol": "AAPL"}]}})
    assert nasdaq100.constituents() == ["AAPL", "BRK-B"]


# ---------------------------------------------------------------------------
# Source parsers and fallbacks
# ---------------------------------------------------------------------------


NEW_TRACKS = ["weather_openmeteo_temp", "solar_openmeteo_ghi", "air_airnow_us", "grid_ercot", "stock_sp500"]


def test_new_track_configs_load_and_resolve_to_sources() -> None:
    ids = {t.id for t in list_tracks()}
    assert set(NEW_TRACKS) <= ids
    kinds = {"weather_openmeteo_temp": "openmeteo", "solar_openmeteo_ghi": "openmeteo", "air_airnow_us": "airnow",
             "grid_ercot": "ercot", "stock_sp500": "sp500"}
    for track_id, kind in kinds.items():
        track = get_track(track_id)
        assert track.source["kind"] == kind and track.bootstrap["kind"] == "source"
        assert track.horizon >= 1 and track.seq_len >= 1


@pytest.mark.parametrize("track_id,low,high", [("weather_openmeteo_temp", 70, 100), ("solar_openmeteo_ghi", 40, 100)])
def test_open_meteo_site_lists_are_sane(track_id: str, low: int, high: int) -> None:
    sites = get_track(track_id).source["sites"]
    assert low <= len(sites) <= high
    assert len({s[0] for s in sites}) == len(sites)
    assert all(-90 <= s[1] <= 90 and -180 <= s[2] <= 180 for s in sites)


def test_open_meteo_parser_handles_batches_and_single_site() -> None:
    hourly = {"time": ["2026-09-20T00:00", "2026-09-20T01:00"], "temperature_2m": [10.5, None]}
    batch = openmeteo.parse_hourly([{"hourly": hourly}, {"hourly": {**hourly, "temperature_2m": [1.0, 2.0]}}],
                                   "temperature_2m", ["a", "b"])
    assert batch.loc["2026-09-20 00:00", "a"] == 10.5 and pd.isna(batch.loc["2026-09-20 01:00", "a"])
    assert list(batch["b"]) == [1.0, 2.0]
    assert list(openmeteo.parse_hourly({"hourly": hourly}, "temperature_2m", ["a"]).columns) == ["a"]
    with pytest.raises(ValueError):
        openmeteo.parse_hourly([{"hourly": hourly}], "temperature_2m", ["a", "b"])


def test_open_meteo_fetch_clips_recent_hours_and_batches(monkeypatch) -> None:
    track = get_track("weather_openmeteo_temp")
    track = replace(track, source={**track.source, "sites": track.source["sites"][:3], "batch": 2,
                                   "pause_seconds": 0, "lag_hours": 3})
    calls = []

    def fake(params):
        calls.append(params)
        n = len(params["latitude"].split(","))
        index = pd.date_range(params["start_date"], pd.Timestamp(params["end_date"]) + pd.Timedelta(hours=23), freq="h")
        return [{"hourly": {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in index], "temperature_2m": [1.0] * len(index)}}
                for _ in range(n)]

    monkeypatch.setattr(openmeteo, "_get", fake)
    monkeypatch.setattr(openmeteo.time, "sleep", lambda s: None)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h")
    panel = openmeteo.fetch(track, now - pd.Timedelta(days=2), now, None)
    assert len(calls) == 2  # three sites in batches of two
    assert panel.index.max() <= now - pd.Timedelta(hours=3)  # forecast hours are never stored
    assert list(panel.columns) == [s[0] for s in track.source["sites"][:3]]


AIRNOW = (
    "10/02/26|11:00|840010491003|Site A|-4|PM2.5|UG/M3|12.0|State A\r\n"
    "10/02/26|11:00|840010491003|Site A|-4|OZONE|PPB|30|State A\r\n"
    "10/02/26|11:00|000020104|CHARLOTTETOWN|-4|PM2.5|UG/M3|4.7|Canada\r\n"
    "10/02/26|11:00|840020900040|Site B|-9|PM2.5|UG/M3|-0.1|State B\r\n"
    "10/02/26|11:00|840021700010|Site C|-9|PM2.5|UG/M3||State C\r\n"
)


def test_airnow_parser_keeps_us_pm25_in_utc() -> None:
    series = airnow.parse_hourly_file(AIRNOW)
    assert len(series) == 2  # ozone, Canadian and empty rows are dropped
    assert series[(pd.Timestamp("2026-10-02 11:00"), "840010491003")] == 12.0
    assert series[(pd.Timestamp("2026-10-02 11:00"), "840020900040")] == -0.1  # raw instrument noise is kept
    assert airnow.hour_url(pd.Timestamp("2026-10-02 11:00")).endswith("/2026/20261002/HourlyData_2026100211.dat")


def test_airnow_fetch_selects_best_covered_sites_then_keeps_them(monkeypatch) -> None:
    track = replace(get_track("air_airnow_us"), source={**get_track("air_airnow_us").source,
                                                        "max_sites": 1, "min_coverage": 0.5, "workers": 1})

    def lines(hour, with_b):
        rows = [f"10/02/26|{hour:02d}:00|840010491003|A|-4|PM2.5|UG/M3|{hour}|x"]
        if with_b:
            rows.append(f"10/02/26|{hour:02d}:00|840020900040|B|-4|PM2.5|UG/M3|1|x")
        return "\n".join(rows)

    monkeypatch.setattr(airnow, "_download", lambda h: None if h.hour == 12 else lines(h.hour, h.hour == 10))
    panel = airnow.fetch(track, pd.Timestamp("2026-10-02 10:00"), pd.Timestamp("2026-10-02 12:00"), None)
    assert list(panel.columns) == ["840010491003"] and len(panel) == 2  # hour 12 was not published
    both = airnow.fetch(track, pd.Timestamp("2026-10-02 10:00"), pd.Timestamp("2026-10-02 11:00"),
                        ["840010491003", "840020900040"])
    assert list(both.columns) == ["840010491003", "840020900040"]


def _tiny_xlsx(rows: list[list[object]]) -> bytes:
    shared: list[str] = []

    def cell(ref: str, value: object) -> str:
        if isinstance(value, str):
            shared.append(value)
            return f'<c r="{ref}" t="s"><v>{len(shared) - 1}</v></c>'
        return f'<c r="{ref}"><v>{value}</v></c>'

    body = "".join(
        f'<row r="{i + 1}">' + "".join(cell(f"{chr(65 + j)}{i + 1}", v) for j, v in enumerate(row)) + "</row>"
        for i, row in enumerate(rows))
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as book:
        book.writestr("xl/worksheets/sheet1.xml", f"<worksheet {ns}><sheetData>{body}</sheetData></worksheet>")
        book.writestr("xl/sharedStrings.xml", f"<sst {ns}>" + "".join(f"<si><t>{s}</t></si>" for s in shared) + "</sst>")
    return out.getvalue()


def test_ercot_native_load_archive_is_read_without_openpyxl() -> None:
    header = ["Hour Ending", *ercot.ZONES, "ERCOT"]
    rows = [header,
            ["03/09/2025 01:00", *range(1, 9), 36],
            ["03/09/2025 24:00", *range(11, 19), 116],
            ["11/02/2025 01:00", *[100] * 8, 800],
            ["11/02/2025 02:00", *[100] * 8, 800],
            ["11/02/2025 02:00", *[300] * 8, 2400]]  # repeated hour of the clock change
    zipped = io.BytesIO()
    with zipfile.ZipFile(zipped, "w") as archive:
        archive.writestr("Native_Load_2025.xlsx", _tiny_xlsx(rows))
    frame = ercot.parse_native_load(zipped.getvalue())
    assert list(frame.columns) == ercot.ZONES
    assert frame.loc["2025-03-09 00:00", "COAST"] == 1  # hour ending 01:00 covers 00:00-01:00
    assert frame.loc["2025-03-09 23:00", "COAST"] == 11  # 24:00 is the last hour of the same day
    assert frame.loc["2025-11-02 01:00", "COAST"] == 200  # duplicate hour averaged
    assert frame.index.is_unique


def test_ercot_native_load_accepts_hourending_header_and_excel_serial_stamps() -> None:
    # 2019/2020 archives spell the column ``HourEnding``; the 2022 archive stores
    # one stamp as an Excel serial number instead of text.
    rows = [["HourEnding", *ercot.ZONES, "ERCOT"],
            ["11/30/2022 24:00", *[5] * 8, 40],
            [44896.041666666664, *[7] * 8, 56],  # 2022-12-01 01:00 (hour ending)
            ["12/01/2022 02:00", *[9] * 8, 72],
            [44897.0, *[3] * 8, 24]]  # 2022-12-02 00:00 == 12/01/2022 24:00
    frame = ercot.parse_native_load(_tiny_xlsx(rows))
    assert list(frame.columns) == ercot.ZONES
    assert frame.loc["2022-11-30 23:00", "EAST"] == 5
    assert frame.loc["2022-12-01 00:00", "EAST"] == 7
    assert frame.loc["2022-12-01 01:00", "EAST"] == 9
    assert frame.loc["2022-12-01 23:00", "EAST"] == 3
    assert frame.index.is_unique and frame.index.is_monotonic_increasing


def test_ercot_daily_weather_zone_report_matches_archive_names() -> None:
    text = ("OperDay,HourEnding,COAST,EAST,FAR_WEST,NORTH,NORTH_C,SOUTHERN,SOUTH_C,WEST,TOTAL,DSTFlag\n"
            "10/01/2026,01:00,16023.5,2034.95,7333.67,2026.75,16451.46,5343.57,10647.45,1569.58,61430.93,N\n"
            "10/01/2026,02:00,15417.71,1925.37,7213.02,2191.10,15915.20,5158.56,10167.12,1309.80,59297.89,N\n")
    frame = ercot.parse_weather_zone_csv(text)
    assert list(frame.columns) == ercot.ZONES
    assert frame.index[0] == pd.Timestamp("2026-10-01 00:00")
    assert frame.loc["2026-10-01 00:00", "FWEST"] == 7333.67


def test_ercot_fetch_prefers_archive_and_fills_from_daily_reports(monkeypatch) -> None:
    track = get_track("grid_ercot")
    index = pd.date_range("2026-08-30", "2026-09-02 23:00", freq="h")
    archive = pd.DataFrame(1.0, index=index[index < "2026-09-01"], columns=ercot.ZONES)
    daily = pd.DataFrame(2.0, index=index[index >= "2026-08-31"], columns=ercot.ZONES)
    monkeypatch.setattr(ercot, "archive_urls", lambda: {2026: "u"})
    monkeypatch.setattr(ercot, "_get", lambda url, **kw: type("R", (), {"content": b""})())
    monkeypatch.setattr(ercot, "parse_native_load", lambda payload: archive)
    monkeypatch.setattr(ercot, "recent_daily_texts", lambda since: ["x"])
    monkeypatch.setattr(ercot, "parse_weather_zone_csv", lambda text: daily)
    panel = ercot.fetch(track, pd.Timestamp("2026-08-30"), pd.Timestamp("2026-09-02 23:00"), None)
    assert panel.loc["2026-08-31 12:00", "COAST"] == 1.0 and panel.loc["2026-09-01 12:00", "COAST"] == 2.0
    assert panel.index.is_unique and len(panel) == len(index)


def test_nasdaq_history_is_sorted_oldest_first_and_empty_raises() -> None:
    data = {"tradesTable": {"rows": [{"date": "10/01/2026", "close": "$330.32"},
                                      {"date": "09/30/2026", "close": "$1,333.02"}]}}
    series = us_prices.parse_nasdaq_history(data)
    assert list(series.index) == [pd.Timestamp("2026-09-30"), pd.Timestamp("2026-10-01")]
    assert series.iloc[0] == 1333.02
    with pytest.raises(RuntimeError):
        us_prices.parse_nasdaq_history({"tradesTable": {"rows": None}})


def test_yahoo_chart_drops_the_unsettled_session_and_uses_exchange_dates() -> None:
    def stamp(day: str) -> int:
        return int(pd.Timestamp(day + " 13:30", tz="UTC").timestamp())  # 09:30 New York

    payload = {"chart": {"result": [{
        "meta": {"exchangeTimezoneName": "America/New_York"},
        "timestamp": [stamp("2026-09-30"), stamp("2026-10-01"), stamp("2026-10-02")],
        "indicators": {"quote": [{"close": [333.02, 330.32, 331.91]}]}}]}}
    during = us_prices.parse_yahoo_chart(payload, now=pd.Timestamp("2026-10-02 11:47", tz="America/New_York"))
    assert list(during.index) == [pd.Timestamp("2026-09-30"), pd.Timestamp("2026-10-01")]
    after = us_prices.parse_yahoo_chart(payload, now=pd.Timestamp("2026-10-02 18:00", tz="America/New_York"))
    assert len(after) == 3
    with pytest.raises(RuntimeError):
        us_prices.parse_yahoo_chart({"chart": {"result": None, "error": {"code": "Not Found"}}})


def test_us_close_falls_back_from_nasdaq_to_yahoo_and_skips_unknown_symbols(monkeypatch) -> None:
    from tsflab.realtime.sources import equity

    monkeypatch.setattr(equity.time, "sleep", lambda s: None)
    good = pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-09-30", "2026-10-01"]))

    def nasdaq(symbol, start, end):
        raise RuntimeError("Nasdaq API returned no rows")

    def yahoo(symbol, start, end):
        if symbol == "NOPE":
            raise RuntimeError("no result")
        return good

    providers = [nasdaq, yahoo]
    start, end = pd.Timestamp("2026-09-01"), pd.Timestamp("2026-10-02")
    assert list(us_prices.close_with_fallback(providers, "AAPL", start, end)) == [1.0, 2.0]
    assert providers[0] is yahoo  # the working vendor moves first
    assert us_prices.close_with_fallback(providers, "NOPE", start, end).empty


def test_sp500_constituent_parsers() -> None:
    csv = "Symbol,Security\nMMM,3M\nBRK.B,Berkshire\nBF.B,Brown-Forman\n"
    assert sp500.parse_constituents_csv(csv) == ["BF-B", "BRK-B", "MMM"]
    wiki = "|| {{NyseSymbol|MMM}}\n|| [[3M]]\n|| {{NasdaqSymbol|AAPL}}\n|| {{NyseSymbol|BRK.B}}\n"
    assert sp500.parse_constituents_wikitext(wiki) == ["AAPL", "BRK-B", "MMM"]


# ---------------------------------------------------------------------------
# Real-time panel dataset
# ---------------------------------------------------------------------------


PANEL_TRACK = "traffic_pems_sb"  # a shipped track config (hourly), the store itself is synthetic


def _store(tmp_path: Path, hours: int = 24 * 40) -> Path:
    index = pd.date_range("2024-01-01", periods=hours, freq="h")
    t = np.arange(hours)[:, None]
    panel = pd.DataFrame(50 + 5 * np.sin(2 * np.pi * t / 24) + np.arange(3)[None, :], index=index,
                         columns=["a", "b", "c"])
    panel.iloc[5:8, 1] = np.nan  # a gap the loader must fill causally
    panel.iloc[10:20, 2] = np.nan
    PanelStore(PANEL_TRACK, tmp_path).append(panel.iloc[: hours // 2])
    store = PanelStore(PANEL_TRACK, tmp_path)
    store.append(panel.iloc[hours // 2:])
    manifest = store.manifest()
    manifest["releases"][0]["version"] = "2024.01.01-0000"  # same-minute releases would collide
    store._write_manifest(manifest)
    return tmp_path / PANEL_TRACK


def _kw(path: Path, **extra):
    return dict(root_path=str(path), data_path="", size=(24, 0, 6), track=PANEL_TRACK, **extra)


def test_time_series_layout_split_and_scaling(tmp_path: Path) -> None:
    path = _store(tmp_path)
    train, val, test = (Dataset_RealtimePanel_TS(flag=f, **_kw(path)) for f in ("train", "val", "test"))
    x, y, xm, ym = train[0]
    assert x.shape == (24, 3) and y.shape == (6, 3) and xm.shape == (24, 6) and ym.shape == (6, 6)
    total = 24 * 40
    assert len(train) == int(0.7 * total) - 24 - 6 + 1
    val_end = int((0.7 + 0.1) * total)  # the repository's border arithmetic
    assert len(val) == (val_end - int(0.7 * total)) - 6 + 1
    assert len(test) == (total - val_end) - 6 + 1
    # Scaling uses the training rows only.
    raw = load_panel(str(path), PANEL_TRACK).to_numpy(np.float32)[: int(0.7 * total)]
    assert train.value_mean == pytest.approx(float(raw.mean()), rel=1e-5)
    assert np.allclose(train.inverse_transform(train.values[:5]), load_panel(str(path), PANEL_TRACK).to_numpy()[:5], atol=1e-3)
    # The first validation target is the first validation row.
    first = int(0.7 * total)
    _, target, _, _ = val[0]
    assert np.allclose(target, val.values[first:first + 6])
    assert not np.isnan(train.values).any()


def test_gaps_are_filled_causally(tmp_path: Path) -> None:
    panel = load_panel(str(_store(tmp_path)), PANEL_TRACK)
    assert (panel["b"].iloc[5:8] == panel["b"].iloc[4]).all()
    assert (panel["c"].iloc[10:20] == panel["c"].iloc[9]).all()


def test_spatiotemporal_layout_matches_the_export_bundle_covariates(tmp_path: Path) -> None:
    path = _store(tmp_path)
    st = Dataset_RealtimePanel_ST(flag="train", **_kw(path))
    value, future, cov, cov_future = st[3]
    assert value.shape == (24, 3) and future.shape == (6, 3)
    assert cov.shape == (24, 3, 2) and cov_future.shape == (6, 3, 2)
    h0 = 3 + 24 - 24  # window centre 23 + 3, history starts at row 3
    assert cov[0, 0, 0] == pytest.approx(((h0 * 60) % (24 * 60)) / (24 * 60))
    assert Dataset_RealtimePanel_ST(flag="train", calendar=False, **_kw(path))[0][2].shape == (24, 3, 0)


def test_release_version_freezes_the_data(tmp_path: Path) -> None:
    path = _store(tmp_path)
    store = PanelStore(PANEL_TRACK, tmp_path)
    first = store.manifest()["releases"][0]
    frozen = load_panel(str(path), PANEL_TRACK, version=first["version"])
    assert frozen.index.max() == pd.Timestamp(first["last_timestamp"])
    assert len(frozen) == 24 * 20
    assert len(load_panel(str(path), PANEL_TRACK)) == 24 * 40
    with pytest.raises(ValueError, match="no release"):
        load_panel(str(path), PANEL_TRACK, version="nope")
    with pytest.raises(FileNotFoundError):
        load_panel(str(tmp_path / "empty" / PANEL_TRACK), PANEL_TRACK)


def test_hub_revision_pulls_into_a_cache(tmp_path: Path, monkeypatch) -> None:
    source = _store(tmp_path / "src")
    calls = []

    def fake_pull(store, repo_id, revision):
        calls.append((repo_id, revision))
        store.directory.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copytree(source, store.directory)
        return True

    monkeypatch.setattr("tsflab.realtime.publish.pull_track", fake_pull)
    target = tmp_path / "data" / PANEL_TRACK
    panel = load_panel(str(target), PANEL_TRACK, revision="abcdef1234567890")
    assert len(panel) == 24 * 40 and calls[0][1] == "abcdef1234567890"
    load_panel(str(target), PANEL_TRACK, revision="abcdef1234567890")
    assert len(calls) == 1  # cached


def test_local_traffic_store_if_present() -> None:
    root = Path("dataset/realtime")
    if not (root / "traffic_pems_sb" / "manifest.json").is_file():
        pytest.skip("no local traffic_pems_sb store")
    ds = Dataset_RealtimePanel_ST(root_path=str(root / "traffic_pems_sb"), data_path="", size=(168, 0, 24),
                                  flag="test", track="traffic_pems_sb", start="2023-11-01", max_windows=4)
    value, future, cov, _ = ds[0]
    assert value.shape[0] == 168 and future.shape[0] == 24 and cov.shape[-1] == 2
    assert np.isfinite(value).all()


def test_series_array_feeds_dataset_inspection(tmp_path: Path) -> None:
    from tsflab.cli.commands.dataset_characteristics import _extract_series

    path = _store(tmp_path)
    total = 24 * 40
    train_end, val_end = int(0.7 * total), int((0.7 + 0.1) * total)
    for cls in (Dataset_RealtimePanel_TS, Dataset_RealtimePanel_ST):
        for flag, rows in (("train", (0, train_end)), ("val", (train_end, val_end)),
                           ("test", (val_end, total))):
            ds = cls(flag=flag, **_kw(path))
            series = _extract_series(ds)
            assert series.shape == (rows[1] - rows[0], 3)
            assert np.allclose(series, ds.values[rows[0]:rows[1]])


# ---------------------------------------------------------------------------
# Hub mirror: realtime/<track>/ of TSFLab-Datasets, push guard, pull layout
# ---------------------------------------------------------------------------


def test_push_guard_follows_the_track_cards() -> None:
    from tsflab.realtime import publish

    assert publish.DEFAULT_DATASET_REPO.endswith("/TSFLab-Datasets")
    assert publish.repo_folder("grid_ercot") == "realtime/grid_ercot"
    classes = {track.id: publish.track_redistribution(track.id) for track in list_tracks()}
    assert {t for t, k in classes.items() if k != "hosted"} == {"stock_hs300", "stock_nasdaq100", "stock_sp500"}
    assert sum(k == "hosted" for k in classes.values()) == 8
    assert publish.push_refusal("grid_ercot") is None
    assert "redistribution=script" in publish.push_refusal("stock_hs300")
    assert "redistribution=unknown" in publish.push_refusal("no_such_track")


def test_push_release_refuses_script_tracks_and_uploads_hosted_ones(tmp_path: Path, monkeypatch) -> None:
    from tsflab.realtime import publish

    uploads = []

    class FakeApi:
        def create_repo(self, *args, **kwargs) -> None:
            pass

        def upload_folder(self, **kwargs):
            uploads.append(kwargs)
            return type("Commit", (), {"oid": "rev1"})()

    monkeypatch.setattr(publish, "_api", FakeApi)
    stock = PanelStore("stock_hs300", tmp_path)
    stock.append(_panel("2026-01-01", 48))
    with pytest.raises(PermissionError, match="stock_hs300: not redistributed"):
        publish.push_release(stock, "o/TSFLab-Datasets")
    assert uploads == []
    hosted = PanelStore("grid_ercot", tmp_path)
    hosted.append(_panel("2026-01-01", 48))
    assert publish.push_release(hosted, "o/TSFLab-Datasets", create=True) == "rev1"
    assert uploads[0]["path_in_repo"] == "realtime/grid_ercot"
    assert uploads[0]["folder_path"] == str(hosted.directory)


def _script_track_store(tmp_path: Path, monkeypatch) -> PanelStore:
    """A stock_hs300 store under a custom TSFLAB_REALTIME_ROOT, with Hub access forbidden."""
    from tsflab.realtime import publish

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TSFLAB_REALTIME_ROOT", str(tmp_path / "custom-root"))
    store = PanelStore("stock_hs300")
    index = pd.bdate_range("2026-01-05", periods=30)
    store.append(pd.DataFrame({"a": np.linspace(0, 1, 30), "b": 0.5}, index=index))

    def never(*args, **kwargs):
        raise AssertionError("a script track must never reach the Hub")

    monkeypatch.setattr(publish, "_api", never)
    monkeypatch.setattr(publish, "pull_track", never)
    return store


def test_weekly_push_skips_script_tracks_and_still_opens_the_round(tmp_path: Path, monkeypatch, capsys) -> None:
    from types import SimpleNamespace

    from tsflab.realtime import cli, sources
    import tsflab.realtime.baselines as baselines

    store = _script_track_store(tmp_path, monkeypatch)
    last = pd.Timestamp(store.manifest()["last_timestamp"])
    new_index = pd.bdate_range(last + pd.Timedelta(days=1), periods=3)
    monkeypatch.setattr(sources, "fetch", lambda track, start, end, channels: pd.DataFrame(
        {"a": 1.0, "b": 0.5}, index=new_index))
    opened = []
    monkeypatch.setattr(R, "open_round", lambda track, store, hf_revision=None: opened.append(hf_revision)
                        or SimpleNamespace(round_id="r1"))
    monkeypatch.setattr(baselines, "run_baselines", lambda spec, store, track: [])
    assert cli.main(["weekly", "--track", "stock_hs300", "--pull", "--push"]) == 0
    out = capsys.readouterr().out
    assert "skip push: stock_hs300: not redistributed" in out
    assert "tsf realtime update --bootstrap --track stock_hs300" in out
    assert opened == [None] and "opened r1 with baselines" in out
    assert (tmp_path / cli.SUMMARY_DIR / "stock_hs300.json").is_file()


def test_update_push_skips_and_no_fetch_push_refuses_script_tracks(tmp_path: Path, monkeypatch, capsys) -> None:
    from tsflab.realtime import cli, sources

    _script_track_store(tmp_path, monkeypatch)
    monkeypatch.setattr(sources, "fetch", lambda *args: pd.DataFrame())
    assert cli.main(["update", "--track", "stock_hs300", "--push"]) == 0
    assert cli.main(["update", "--track", "stock_hs300", "--no-fetch", "--push"]) == 2
    err = capsys.readouterr().err
    assert "stock_hs300: not redistributed (card [source] redistribution=script)" in err


def test_pull_moves_the_track_folder_into_a_custom_store_root(tmp_path: Path, monkeypatch) -> None:
    import shutil

    import huggingface_hub

    from tsflab.realtime.publish import pull_track

    source = PanelStore("grid_ercot", tmp_path / "published")
    source.append(_panel("2026-01-01", 48))
    calls = []

    def fake_snapshot(*, repo_id, repo_type, revision, allow_patterns, local_dir):
        calls.append((repo_id, repo_type, revision, allow_patterns))
        shutil.copytree(source.directory, Path(local_dir) / "realtime" / "grid_ercot")
        (Path(local_dir) / ".cache").mkdir()
        return local_dir

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    root = tmp_path / "anywhere"  # no .../realtime/<track> layout
    store = PanelStore("grid_ercot", root)
    assert pull_track(store, "o/TSFLab-Datasets", "abc123") is True
    assert calls == [("o/TSFLab-Datasets", "dataset", "abc123",
                      ["realtime/grid_ercot/*", "realtime/grid_ercot/**/*"])]
    assert store.directory == root / "grid_ercot" and store.exists
    assert store.manifest() == source.manifest() and len(store.read()) == 48
    assert sorted(p.name for p in root.iterdir()) == ["grid_ercot"]  # temp dir cleaned up

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: kwargs["local_dir"])
    missing = PanelStore("air_airnow_us", root)
    assert pull_track(missing, "o/TSFLab-Datasets") is False and not missing.directory.exists()
