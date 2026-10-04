# rt/traffic_pems_ba — reference

## Provenance and license

- Source system: Caltrans PeMS (https://pems.dot.ca.gov), District 4. The PeMS conditions of use (https://pems.dot.ca.gov/?view=tou) put the data in the public domain; `redistribution` is `hosted` with attribution to Caltrans PeMS.
- History: the UltraTraffic_CL archive (inside `TrafficCL.zip`; publisher not identified, no paper or repository found) holds hourly total flow per station for region `PEMS_BA`, one static panel per year from 2003 to 2023 (1,177 to 2,472 stations per year), plus continual-learning slices (2023: 23 added, 2,449 common stations). 2023 values range 0 to 13,314, mean 2,780.5 vehicles per hour (measured from the store).
- Build the history store once with `uv run tsf data prepare --from ultratraffic --archive TrafficCL.zip` (written to `dataset/ultratraffic`, or `ULTRATRAFFIC_ROOT`); the track bootstrap reads the `static` panels from 2019 and keeps the 2023 station set. The archive has no station coordinates.
- Weekly increments come from the PeMS clearinghouse `station_5min` files summed to hourly flow. They need a free PeMS account (`PEMS_USER`, `PEMS_PASSWORD`); without it the track keeps its history only.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Release | `2026.09.28-2003` (bootstrap release; the manifest holds this one release) | measured |
| Rows | 43,824 hourly (2019-01-01 00:00 to 2023-12-31 23:00; a contiguous hourly grid) | measured |
| Stations | 2,472 | measured |
| NaN cells in the store | 2.15% (per year 2019-2023: 4.3, 3.0, 2.5, 0.9, 0.0 %) | measured |
| Stations with more than 50% NaN | 62 | measured |
| Stations whose first reading is after the training split ends | 23 | measured |
| Value range | 0 to 15,101; mean 2710.0, standard deviation 1914.7 (vehicles per hour, NaN excluded) | measured |
| Exact zeros | 0.57% of observed values | measured |
| Mean flow by year | 2,871 (2019), 2,438 (2020), 2,714 (2021), 2,747 (2022), 2,780 (2023) | measured |
| Track config | `configs/realtime/traffic_pems_ba.toml`: freq h, seq_len 168, horizon 24, min_coverage 0.8, timezone America/Los_Angeles | source-reported (config) |

Measured from the local store `dataset/realtime/traffic_pems_ba` (read-only). With 7:1:2 the training rows end 2022-07-02 and the validation rows end 2022-12-31; the test split is therefore all of 2023, the complete-coverage year.

## Related datasets

- [`pems04`](../../pems04/README.md): 5-minute PeMS flow graph of District 4 (STSGCN)
- [`rt/traffic_pems_la`](../traffic_pems_la/README.md): another PeMS district as a real-time preset
- [`rt/traffic_pems_sac`](../traffic_pems_sac/README.md): another PeMS district as a real-time preset
- [`rt/traffic_pems_sb`](../traffic_pems_sb/README.md): another PeMS district as a real-time preset
