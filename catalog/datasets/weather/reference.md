# weather — reference

## Provenance and license

- Producer: Max-Planck-Institut fuer Biogeochemie, Jena (station documentation by Olaf Kolle); data at https://www.bgc-jena.mpg.de/wetter/.
- License: the Data Download page (https://www.bgc-jena.mpg.de/wetter/weather_data.html) states "Terms of Use (as per Creative Commons CC-BY-4.0)"; the landing page itself states none. Redistribution and adaptation are allowed with attribution.
- Cite Autoformer for this subset and credit the Max Planck Institute for Biogeochemistry.
- Do not confuse it with Informer's own "Weather" set (NOAA local climatological data, 2010-2013), which is a different dataset.
- Redistribution: `hosted` with attribution (CC BY 4.0).

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Rows | 52,696 | measured |
| Channels | 21 (alphabetical order with `OT` second; the last column is `wv (m/s)`) | measured |
| Time span | 2020-01-01 00:10:00 to 2020-12-31 22:40:00; contiguous 10-minute grid, no gaps | measured |
| NaN values | 0 | measured |
| -9999 sentinels | OT: 50, max. PAR: 30, wv (m/s): 1 | measured |
| OT range excluding sentinels | 305.5 to 524.2, mean 427.7 (looks like CO2 in ppm; meaning not verified) | measured |
| Exact zeros | 16.0% of all values (rain, raining, radiation at night) | measured |
| Year coverage | 2020; the series starts at 00:10 and ends at 22:40, so 8 slots of a full year grid are absent at the edges | measured |

Measured on `dataset/weather/weather.csv` (read-only).

## Related datasets

- [`ettm1`](../ettm1/README.md): other sub-hourly LTSF set
- [`electricity`](../electricity/README.md): other 7:1:2 LTSF set
- [`gift_eval/jena_weather_10T`](../gift_eval/jena_weather_10T/README.md): GIFT-Eval packaging of the same station (also `_H`, `_D`)
