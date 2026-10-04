# rt/traffic_pems_sac — reference

## Provenance and license

- Source system: Caltrans PeMS (https://pems.dot.ca.gov), District 3. The PeMS conditions of use (https://pems.dot.ca.gov/?view=tou) put the data in the public domain; `redistribution` is `hosted` with attribution to Caltrans PeMS.
- History: the UltraTraffic_CL archive (inside `TrafficCL.zip`; publisher not identified, no paper or repository found) holds hourly total flow per station for region `PEMS_SAC`, one static panel per year from 2003 to 2023 (84 to 801 stations per year), plus continual-learning slices (2023: 110 added, 691 common stations). 2023 values range 0 to 13,783, mean 1,663.6 vehicles per hour (measured from the store).
- Build the history store once with `uv run tsf data prepare --from ultratraffic --archive TrafficCL.zip` (written to `dataset/ultratraffic`, or `ULTRATRAFFIC_ROOT`); the track bootstrap reads the `static` panels from 2019 and keeps the 2023 station set. The archive has no station coordinates.
- Weekly increments come from the PeMS clearinghouse `station_5min` files summed to hourly flow. They need a free PeMS account (`PEMS_USER`, `PEMS_PASSWORD`); without it the track keeps its history only.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Release | `2026.09.28-2003` (bootstrap release; the manifest holds this one release) | measured |
| Rows | 43,824 hourly (2019-01-01 00:00 to 2023-12-31 23:00; a contiguous hourly grid) | measured |
| Stations | 801 | measured |
| NaN cells in the store | 23.83% (per year 2019-2023: 39.7, 36.3, 29.3, 13.7, 0.0 %) | measured |
| Stations with more than 50% NaN | 227 | measured |
| Stations whose first reading is after the training split ends | 99 | measured |
| Value range | 0 to 20,671; mean 1914.1, standard deviation 1663.1 (vehicles per hour, NaN excluded) | measured |
| Exact zeros | 2.29% of observed values | measured |
| Mean flow by year | 2,275 (2019), 1,911 (2020), 2,041 (2021), 1,850 (2022), 1,664 (2023) | measured |
| Track config | `configs/realtime/traffic_pems_sac.toml`: freq h, seq_len 168, horizon 24, min_coverage 0.8, timezone America/Los_Angeles | source-reported (config) |

Measured from the local store `dataset/realtime/traffic_pems_sac` (read-only). With 7:1:2 the training rows end 2022-07-02 and the validation rows end 2022-12-31; the test split is therefore all of 2023, the complete-coverage year.

## Related datasets

- [`pems03`](../../pems03/README.md): 5-minute PeMS flow graph of District 3 (STSGCN)
- [`rt/traffic_pems_ba`](../traffic_pems_ba/README.md): another PeMS district as a real-time preset
- [`rt/traffic_pems_la`](../traffic_pems_la/README.md): another PeMS district as a real-time preset
- [`rt/traffic_pems_sb`](../traffic_pems_sb/README.md): another PeMS district as a real-time preset
