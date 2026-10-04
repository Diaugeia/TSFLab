# Prepare data for a loader

Turn source files into the layout a registered loader reads, without altering the
source. Stop before replacing an existing output directory without authorization.

1. If the preset is published, download it instead of converting anything:

   ```bash
   uv run tsf data download --list
   uv run tsf data download <preset>      # verified against configs/hub/datasets.json
   ```

2. Otherwise inspect the source layout and the destination before writing, then
   choose the converter:

   ```bash
   # window a CSV (use --input-dir for pre-split CSVs)
   uv run tsf data prepare --input-csv <file.csv> --output-dir <dir> \
     --seq-len 96 --label-len 48 --pred-len 96
   # graph traffic bundles and GIFT-Eval: read the options first
   uv run tsf data prepare --from traffic --help
   uv run tsf data prepare --from gift-eval --help
   # script-class sources, fetched from the original source with pinned SHA-256
   uv run tsf data prepare --from tfb --list     # TFB archive -> dataset/<Name>/<Name>.csv
   uv run tsf data prepare --from dcrnn          # DCRNN METR-LA -> dataset/metr_la
   # UltraTraffic archive -> dataset/ultratraffic (history of the traffic_pems_* tracks)
   uv run tsf data prepare --from ultratraffic --archive <TrafficCL.zip>
   ```

   The UltraTraffic converter stores each region and year once (the CL full-year
   file equals the static one and a duplicated region is skipped) and records row,
   station, and missing-value counts plus source hashes in `manifest.json`.
3. Verify every split, shape, window, and the train-only scaling policy.
4. Publishing local files (`uv run tsf data publish <preset>`) is a maintainer
   action that needs explicit authorization and redistributable source terms.
   Read the card's `license` and `redistribution`: publish only `hosted`, with its
   `conditions` met; never publish `upstream` or `script`.
