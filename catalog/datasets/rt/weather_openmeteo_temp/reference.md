# rt/weather_openmeteo_temp — reference

## Provenance and license

- Source: the Open-Meteo Historical Forecast API (https://open-meteo.com), which stitches the analysis hours of operational weather models. The data are CC BY 4.0 (https://open-meteo.com/en/terms), so `redistribution` is `hosted` with attribution; the free API itself is non-commercial and serves collection only.
- Releases are mirrored to the Hugging Face dataset `Diaugeia/TSFLab-RealTime` (`weather_openmeteo_temp/`).

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Bootstrap | 365 days of the same API (analysis hours from 2022) | source-reported (config) |
| Track config | `configs/realtime/weather_openmeteo_temp.toml`: freq h, seq_len 168, horizon 24, min_coverage 0.9, timezone UTC | source-reported (config) |

No local copy of `dataset/realtime/weather_openmeteo_temp` exists in a development checkout, so nothing is measured: row count, channel count, and missingness are unknown until a store is provided (or `dataset.params.revision` pulls one from the Hub) and the card re-measured. The preset is therefore not pinned to a release.

## Related datasets

- [`rt/solar_openmeteo_ghi`](../solar_openmeteo_ghi/README.md): same source, irradiance
- [`weather`](../../weather/README.md): static single-station weather set
