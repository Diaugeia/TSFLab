# rt/air_airnow_us — reference

## Provenance and license

- Source: EPA AirNow `HourlyData` files (https://www.airnow.gov). The AirNow data use guidelines (https://docs.airnowapi.org/docs/DataUseGuidelines.pdf) require attribution and a notice that the data are preliminary and unvalidated, so `redistribution` is `hosted` with `conditions`; AQS holds the validated values months later.
- Releases are mirrored to the Hugging Face dataset `Diaugeia/TSFLab-RealTime` (`air_airnow_us/`).

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Bootstrap | 56 days of AirNow archive files | source-reported (config) |
| Track config | `configs/realtime/air_airnow_us.toml`: freq h, seq_len 168, horizon 24, min_coverage 0.7, timezone UTC | source-reported (config) |

No local copy of `dataset/realtime/air_airnow_us` exists in a development checkout, so nothing is measured: row count, channel count, and missingness are unknown until a store is provided (or `dataset.params.revision` pulls one from the Hub) and the card re-measured. The preset is therefore not pinned to a release.

## Related datasets

- [`aqshunyi`](../../aqshunyi/README.md): hourly Beijing air quality at one station (UCI)
- [`aqwan`](../../aqwan/README.md): hourly Beijing air quality at another station (UCI)
- [`rt/weather_openmeteo_temp`](../weather_openmeteo_temp/README.md): hourly temperature as a real-time preset
