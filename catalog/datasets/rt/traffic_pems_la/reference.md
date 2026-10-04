# rt/traffic_pems_la — reference

## Provenance and license

- Source system: Caltrans PeMS (https://pems.dot.ca.gov), District 7. The PeMS conditions of use (https://pems.dot.ca.gov/?view=tou) put the data in the public domain; `redistribution` is `hosted` with attribution to Caltrans PeMS.
- History: the UltraTraffic_CL archive (inside `TrafficCL.zip`; publisher not identified, no paper or repository found) holds hourly total flow per station for region `PEMS_LA`, one static panel per year from 2003 to 2023 (1,295 to 1,927 stations per year), plus continual-learning slices (2023: 5 added, 1,921 common stations). 2023 values range 0 to 20,314, mean 3,527.3 vehicles per hour (measured from the store).
- Build the history store once with `uv run tsf data prepare --from ultratraffic --archive TrafficCL.zip` (written to `dataset/ultratraffic`, or `ULTRATRAFFIC_ROOT`); the track bootstrap reads the `static` panels from 2019 and keeps the 2023 station set. The archive has no station coordinates.
- Weekly increments come from the PeMS clearinghouse `station_5min` files summed to hourly flow. They need a free PeMS account (`PEMS_USER`, `PEMS_PASSWORD`); without it the track keeps its history only.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Release | `2026.09.28-2003` (bootstrap release; the manifest holds this one release) | measured |
| Rows | 43,824 hourly (2019-01-01 00:00 to 2023-12-31 23:00; a contiguous hourly grid) | measured |
| Stations | 1,926 | measured |
| NaN cells in the store | 1.30% (per year 2019-2023: 2.9, 2.0, 1.3, 0.3, 0.0 %) | measured |
| Stations with more than 50% NaN | 26 | measured |
| Stations whose first reading is after the training split ends | 5 | measured |
| Value range | 0 to 23,676; mean 3578.5, standard deviation 2242.1 (vehicles per hour, NaN excluded) | measured |
| Exact zeros | 0.38% of observed values | measured |
| Mean flow by year | 3,825 (2019), 3,357 (2020), 3,619 (2021), 3,568 (2022), 3,527 (2023) | measured |
| Track config | `configs/realtime/traffic_pems_la.toml`: freq h, seq_len 168, horizon 24, min_coverage 0.8, timezone America/Los_Angeles | source-reported (config) |

Measured from the local store `dataset/realtime/traffic_pems_la` (read-only). With 7:1:2 the training rows end 2022-07-02 and the validation rows end 2022-12-31; the test split is therefore all of 2023, the complete-coverage year.

## Related datasets

- [`pems07`](../../pems07/README.md): 5-minute PeMS flow graph of District 7 (STSGCN)
- [`rt/traffic_pems_ba`](../traffic_pems_ba/README.md): another PeMS district as a real-time preset
- [`rt/traffic_pems_sac`](../traffic_pems_sac/README.md): another PeMS district as a real-time preset
- [`rt/traffic_pems_sb`](../traffic_pems_sb/README.md): another PeMS district as a real-time preset
