# Workflows and architecture

TSFLab has one flat model catalog, one shared-component area, one data
pipeline, one card format, and one admission check per model. Models and methods
are peers.

```text
src/tsflab/models/<slug>/              local model code, spec, and card (card.toml, README.md)
src/tsflab/models/_components/<name>/  reusable component code and card
src/tsflab/data/                       dataset loaders and parameter schemas
dataset/                        local dataset bytes (not packaged)
catalog/datasets/<preset>/      dataset cards (card.toml, README.md)
catalog/declined.toml           papers reviewed but not admitted
configs/                        composable model, dataset, and run TOML
work_dirs/                      experiment checkpoints, metrics, and records
work_dirs/_research/<round>/    optional goals, events, prompts, and full logs
```

Set `TSFLAB_WORK_DIR` to keep research rounds outside the checkout's `work_dirs/`.

## Model runtime interface

Every model accepts the same call:

```python
forecast = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
```

- `x_enc`: observed values, normally `[batch, seq_len, channels]`.
- `x_mark_enc`: historical time/node covariates, or `None`.
- `x_dec`: observed decoder prefix plus the future placeholder.
- `x_mark_dec`: known future covariates, or `None`.

An implementation may ignore inputs its method does not use, but it must accept
the complete interface. Point outputs are `[batch, pred_len, targets]`; quantile
and distribution outputs append their parameter axis. `spec.py` contains only
the factory, validated parameter schema, preset path, task capabilities, shared
components, artifacts, optional training objective, and runtime fixture.
Descriptive facts belong in README.

The generic trainer owns batching, the four-input call, configured criterion,
callbacks, optimization, and validation. By default a model trains on the
configured criterion plus an optional scalar `aux_loss` regularizer it sets during
`forward`. A paper-specific training loss is opt-in: declare
`ModelSpec.training_objective(model, batch, criterion)`, which receives a
`TrainingBatch` (`x`, `x_mark`, `dec_inp`, `y_mark`, `y`, `target`,
`batch.forecast(model)`, `batch.align(outputs)`) and the configured criterion, and
returns `(forecast_or_None, finite_scalar_loss)` from a single pass. It may extend
the criterion (card slot `loss:mse+local:<name>`) or replace it (`local:<name>`).
An optional `ModelSpec.training_setup(model, train_loader, *, pred_len, features)`
runs once before a fresh run to fit training-split state such as a label basis.
The objective is used only while training; validation, early stopping, and test
metrics always use `forward` with the configured observation loss. The strict
contract check back-propagates the declared objective's loss (after
`training_setup` on a synthetic stride-1 loader) instead of the `forward` output,
so `forward` may sample without gradients. Objectives are
unsupported with `DataParallel`. Do not advertise one as a capability or leave a
`training_loss()` method that the runner does not call; a paper objective whose
inputs the runner cannot supply (for example a second historical window) stays off
and the model card's Differences says why.

`task.mode` is enforced before execution:

- `time_series`: multivariate values and optional calendar marks.
- `spatiotemporal`: node values with historical node/time covariates.
- `covariate`: node values plus known future covariates.

Inspect compatibility with `tsf catalog show <Name>` and `tsf catalog show <preset>`; both accept `--depth {0,1,2,3}` (see Components below).

## Add a model or method

The admission path is deliberately two-phase so a placeholder cannot become a
catalog entry.

1. Deduplicate the paper against `tsf catalog list --kind model --json` and `tsf catalog search --kind model`.
2. Read the paper and supplement. Locate official code when available, record its
   license, and pin a revision. Use it to clarify omitted implementation details;
   do not copy or import its model source.
3. Decide the card facts readers find a model through first: a one-sentence
   description, `tags` (with one architecture family), the data characteristics it
   `fits`, and the six-slot `composition` ([Cards](#cards)). Map every defining operation to an existing
   component, a justified new shared component, or a model-local block. Start with:

   ```bash
   uv run tsf catalog search --kind component "operation and tensor contract"
   uv run tsf catalog show <candidate>
   ```

4. Create an unregistered workspace:

   ```bash
   uv run tsf model scaffold \
     --name MyModel \
     --paper-title "Paper title" \
     --paper-url https://arxiv.org/abs/0000.00000 \
     --venue Conference --year 2026 \
     --code-url https://github.com/org/repo \
     --revision 0123456789abcdef \
     --license Apache-2.0 \
     --components revin,flatten_forecast_head \
     --params "enc_in:int,d_model:int=128"
   ```

   Omit all three code arguments when no official code exists. Use
   `--components none` only after matching found no equivalent. Select
   `--task-mode spatiotemporal` or `covariate` when required.
5. Replace every scaffold marker, implement locally, preserve useful paper
   equations/comments, and complete the card ([Cards](#cards)): `card.toml` facts
   (fidelity, paper, code with the inspected `reference_sources`, composition,
   `[data_params]`, and `[[issues]]`) and the README sections Idea, When to use,
   Configure, Differences. Upstream issues record what the reimplementation exposed in
   the paper or official code (bugs, paper/code mismatches, missing details, leakage,
   license problems), each with TSFLab's resolution; `tsf model issues --summary`
   aggregates them across the catalog. When nothing was found, `issues_checked` says
   what was checked. A paper reviewed but not admitted (out of scope, not
   implementable without inventing its defining operations, a composite of other
   pretrained models, or a duplicate) is recorded in `catalog/declined.toml` with its
   reason and the same issue kinds. Fidelity is `reference-checked` only when the
   implementation was compared against official code at the pinned revision.
6. Admit the entry:

   ```bash
   uv run tsf model add --name MyModel --verify
   ```

   Admission registers the model, runs the model audit, the component audit, and
   the repository audit, then runs the executable contract and records the result
   in the card's `[admission]`. Registration is rolled back if any gate fails.
   Without `--verify` the admission stays `pending`; `tsf repo check --scope release`
   requires every admission to be `passed` before a final release.

## Foundation models and artifacts

Foundation models remain ordinary flat model entries. Pretraining scale is not a
catalog category. A local architecture without released weights must say so in
its card and cannot claim zero-shot checkpoint behavior.

Released pretrained models use the thin boundary in `src/tsflab/models/_foundation/`.
`FoundationModel` converts the canonical `[batch, time, channels]` input to the
official runtime's series batch and restores point or quantile output axes.
Chronos and TimesFM have direct adapters; Moirai accepts an official forecast
object from a Uni2TS-compatible environment. Provider packages are optional and
lazy. Construction is offline and receives an explicit local checkpoint path, so
the official cache is reused without copying or repackaging weights.

The flat model spec declares `inference-only`. The runner then skips optimizer,
training, and checkpoint creation while preserving the normal dataset and
evaluation paths. Upper-layer methods such as CoRA or retrieval augmentation stay
as separate flat entries and compose with a foundation runtime in an explicit
experiment rather than through a second registry.

Weights, tokenizers, or normalization statistics remain explicit runtime facts:

```python
from tsflab.catalog.registry.models import ModelArtifact

artifacts = (ModelArtifact(
    name="weights",
    url="https://host/repository/resolve/full-commit-or-release/weights.safetensors",
    revision="full-commit-or-release",
    sha256="<64 lowercase hex characters>",
    filename="weights.safetensors",
    required=True,
),)

def build_from_artifacts(cfg, params, paths):
    model = Model(...)
    model.load_checkpoint(paths["weights"])
    return model

SPEC = ModelSpec(
    ...,
    artifacts=artifacts,
    artifact_factory=build_from_artifacts,
)
```

TSFLab never downloads them during construction. Inspect or explicitly fetch
an artifact before a run:

```bash
uv run tsf model artifacts MyFoundationModel
uv run tsf model artifacts MyFoundationModel --fetch weights
```

The cache defaults to the user cache directory and may be changed with
`TSFLAB_CACHE`. A required artifact must exist and match SHA-256 before the
artifact-aware factory runs. Ordinary models keep the two-argument factory;
artifact-backed models explicitly receive a mapping of verified local paths.
Loading behavior, offline failure, and the exact checkpoint claim belong in the
model card; the admission contract runs offline from verified local paths.

## Components

Components live only in `src/tsflab/models/_components/<name>/`. Each directory has an
implementation, a catalog contract, and a card ([Cards](#cards)): `card.toml` records
the role, the composition slot it fills, the data characteristics it fits, its
category, input and output shapes, and origin; the README covers What it does, When
to use, and Interface (every public symbol with parameters, tensor shapes, and
state). Consumers and the import line are derived from code, never stored.
`tsf model audit --components` checks the cards against the catalog.
Extract only mathematically and operationally equivalent behavior—matching names
or tensor rank is insufficient. Validate axes, normalization, masking, residual
order, initialization, state, outputs, gradients, and serialization.

```bash
uv run tsf catalog list --kind component
uv run tsf catalog search --kind component "patch forecast head"
uv run tsf catalog show flatten_forecast_head --depth 1
uv run tsf model audit --components
uv run tsf model similar --top 20     # ranked extraction/reuse candidates
```

`tsf model similar` finds candidates statically: it parses every model and
component, renames identifiers, keeps library operators (`torch.fft.rfft`,
`nn.Linear`, `.softmax`), and clusters units whose token shingles and operator
profiles agree (`--threshold`, default 0.6). Each cluster is marked as a
`reuse-existing` or `extract-new` candidate. It nominates only; equivalence is
still shown against fixtures captured before any code moves, and every model that
uses a changed component is re-admitted (`tsf model verify --changed`).

### Reading the catalog

Start at the entry view, then narrow. Each step costs more context than the last,
so stop as soon as the decision is made:

```bash
uv run tsf catalog                       # counts per kind, then the next commands
uv run tsf catalog search "reversible normalization" --kind component   # L0
uv run tsf catalog show PatchTST           # L1
uv run tsf catalog show revin --depth 2   # L2
uv run tsf catalog show PatchTST --depth 3    # L3
```

| Depth | Content |
| --- | --- |
| 0 | one line: `name`, `kind`, the README `description`, `tags` (what search returns) |
| 1 | `card.toml` facts, runtime facts derived from code, and the README body (default) |
| 2 | L1 plus `reference.md` when the card has one |
| 3 | the source, config, and card paths to open |

`catalog search` ranks all three kinds; `--kind` restricts it, `--capability`
(repeatable) filters models, `--limit` caps results, and `--json` gives
structured output. `tsf catalog match <dataset>` starts from data instead of
words: it intersects the dataset card's characteristics with the `fits` of every
model and component card and ranks the overlaps, components grouped by slot
(`--extra probabilistic-output` adds task terms). A match is a hypothesis to test,
not a verdict.

### Cards

Models, components, and datasets share one card format (`tsflab.card/1`). Each card
directory holds `card.toml` (structured facts), `README.md` (front matter `name`
and `description`, then a short body of at most 60 lines in fixed sections), and an
optional `reference.md` for detail that does not fit. Nothing generated is stored
in a card: config paths, parameter schemas, imports, consumers, loaders, and task
modes are derived from code when a card is read.

- **Model:** `tags` (one is the architecture family), `fits`, `fidelity`
  (`reference-checked`, `paper-only`, `inferred`, or `composed`), `[paper]`,
  `[code]` (official repository, pinned revision, license, and the inspected
  `reference_sources`), a `[composition]` of six slots (`normalization`,
  `decomposition`, `temporal`, `channel`, `head`, `loss`, each `component:<name>`,
  `local:<block>`, `loss:<name>`, or `none`), `[data_params]`, `[[issues]]` (or
  `issues_checked`), and `[admission]`. README sections: Idea, When to use,
  Configure, Differences.
- **Component:** `role`, `slot`, `fits`, `category`, `tags`, `input`/`output`
  shapes, `origin`, `origin_models`. README sections: What it does, When to use,
  Interface.
- **Dataset:** `domain`, `tags`, `characteristics` with their basis, `related`,
  `[source]` (citation, license, redistribution), `[shape]` (frequency, span,
  length, channels, target, missing values, whether measured or source-reported),
  and `[protocol]` (the TSFLab protocol, the literature protocol when it differs,
  split, lookbacks, horizons). README sections: Overview, Protocol and pitfalls.

`fits` and `characteristics` use one vocabulary. Data terms are the rules of the
dataset profiler (`tsf data analyze`), such as `non-stationary`,
`strong-seasonality`, or `train-val-level-shift`; `tsf data analyze <preset>
--write-card` measures them and records them in the dataset card. Task terms
(`spatial-graph`, `calendar-effects`, `exogenous-covariates`, `long-horizon`,
`probabilistic-output`, `low-data`, ...) describe the forecasting setup, and `any`
marks a generic building block.

`[data_params]` names the parameters whose right value depends on the data, with
the property they follow (`period`, `frequency`, `channels`, `nodes`, `seq_len`,
`pred_len`, `graph`, or `train-split`) and a rule, for example a moving-average
kernel set from the dominant period. Generic hyperparameters (width, depth,
dropout, learning rate) are not listed; tune them within the experiment budget.

Paper-specific variants stay inside the model package. Named model packages must
not import implementation code from another named model.

## Data

Data has three non-overlapping layers:

- `dataset/`: local files, downloads, and converted arrays; never code or cards.
- `src/tsflab/data/`: executable loaders, base contracts, and Pydantic parameter schemas.
- `catalog/datasets/`: one card per runnable dataset preset, plus one family card
  for GIFT-Eval ([Cards](#cards)): curated facts (domain, source, license, shape,
  protocol, measured characteristics) and a short README with the pitfalls. Loader,
  files, and task modes are derived from the preset and code.

There are 93 dataset presets: 79 conventional `time_series` presets (including the
GIFT-Eval series) and 14 spatiotemporal or covariate presets; 11 presets under
`rt/` are frozen releases of the real-time tracks. Each card states one `domain`
and the benchmark suites it belongs to (`benchmarks`: `ltsf`, `tfb`, `st-graph`,
`gift-eval`, `realtime`):

| Domain | Presets |
| --- | --- |
| `energy` (25) | `etth1`, `etth2`, `ettm1`, `ettm2`, `electricity`, `solar`, `wind`, `rt/grid_ercot`, `rt/solar_openmeteo_ghi`; GIFT-Eval `electricity_*` (4), `ett1_*` (4), `ett2_*` (4), `solar_*` (4) |
| `transport` (18) | `traffic`, `pems03`, `pems04`, `pems07`, `pems08`, `pems_bay`, `metr_la`, `rt/traffic_pems_{ba,la,sac,sb}`; GIFT-Eval `LOOP_SEATTLE_*` (3), `M_DENSE_*` (2), `SZ_TAXI_*` (2) |
| `environment` (16) | `weather`, `aqshunyi`, `aqwan`, `czelan`, `zafnoo`, `rt/air_airnow_us`, `rt/weather_openmeteo_temp`; GIFT-Eval `jena_weather_*` (3), `kdd_cup_2018_*` (2), `saugeenday_*` (3), `temperature_rain_with_missing` |
| `finance` (14) | `exchange`, `fred_md`, `nasdaq`, `nyse`, `nn5`, `rt/stock_{hs300,nasdaq100,sp500}`; GIFT-Eval `m4_*` (6) |
| `healthcare` (7) | `ili`, `covid19`; GIFT-Eval `covid_deaths`, `hospital`, `us_births_*` (3) |
| `cloud-web` (9) | `wike2000`; GIFT-Eval `bitbrains_*` (4), `bizitobs_*` (4) |
| `sales` (4) | GIFT-Eval `car_parts_with_missing`, `hierarchical_sales_*` (2), `restaurant` |

Every dataset has
exactly one TSFLab protocol, stated in its card (chronological split, scaling fitted
on the training split only, lookbacks and horizons), so results on a dataset are
comparable across models. Where the literature uses a different protocol, the card
records it separately as `[protocol].literature`. `configs/fixtures/` holds smoke and
synthetic test inputs; they are not datasets and have no cards.

Fetch a published preset's files, pinned and checksum-verified, into `dataset/`
(see [the Hub page](hub.md#benchmark-data)):

```bash
uv run tsf data download --list
uv run tsf data download etth1
```

Find and read datasets by what they are, not only by name. Search matches the
card's domain, tags, frequency, and source; `show` returns the card's facts:

```bash
uv run tsf catalog search --kind dataset hourly electricity
uv run tsf catalog show etth1       # preset record plus card facts
uv run tsf catalog match etth1      # models and components that fit its characteristics
uv run tsf data audit            # required facts, no placeholders
```

`card.toml` is the quick reference (`domain`, optional `topic`, `benchmarks`,
`[source]`, `[shape]`, `[protocol]`, `characteristics`, `related`, optionally
`realtime_track`); the README adds the overview, protocol, and known pitfalls, and
`reference.md` longer provenance or statistics. `stats_basis` says whether numbers
were measured from local files or reported by the source.

`[source].redistribution` says what TSFLab may do with the data files, with
`license_url` as evidence:

| Value | Meaning | Presets |
| --- | --- | --- |
| `hosted` | TSFLab re-hosts the files in TSFLab-Static (`tsf data download <preset>`); extra source terms, if any, are in `conditions` (ETT, `solar`, `covid19`, `rt/grid_ercot`, `rt/air_airnow_us`) | 29 |
| `upstream` | fetched from another party's Hugging Face repository, never re-hosted: GIFT-Eval with `tsf data prepare --from gift-eval` (55 presets and the family card), `exchange` with `tsf data download exchange` | 57 |
| `script` | the license forbids re-hosting; TSFLab ships a fetch command and you download from the original source: `tsf data prepare --from tfb` (`fred_md`, `nasdaq`, `nyse`, `wike2000`), `tsf data prepare --from dcrnn` (`metr_la`), `tsf realtime update --bootstrap --track <t>` (`rt/stock_*`) | 8 |

Use an existing CSV preset or create a loader-backed dataset:

```bash
uv run tsf data add --name my_data --pattern custom \
  --path ./dataset/my_data/my_data.csv --target OT
uv run tsf data inspect --config configs/datasets/my_data.toml
uv run tsf data analyze my_data      # model-selection profile (JSON + markdown)
uv run tsf data analyze my_data --write-card   # record its characteristics in the card
uv run tsf catalog show my_data
uv run tsf data audit
```

`dataset analyze <preset>` (or `--path FILE`) profiles the training split (length,
missingness, scale, periods, seasonality, trend, forecastability, cross-channel
structure, outliers) plus train/validation shift, recommends lookback candidates, and
maps findings to catalog components and models. The rules that fire are the
dataset's data characteristics; `--write-card` records them, with their basis, in
the dataset card, where `tsf catalog match` reads them. Test-split shift is labelled
diagnostic-only and never becomes a characteristic. Results go to `work_dirs/profiles/<name>/` (`--out` changes it,
`--json` prints the profile). For a file that has no preset, pass `--path FILE`
with `--split-ratio TRAIN VAL TEST` and optionally `--freq`.

`tsf data prepare [--from traffic|ultratraffic|gift-eval|tfb|dcrnn]` provides explicit
conversion/download operations; inspect their `--help` before writing. `tfb` and
`dcrnn` download from the original source and check the archive, every input, and
every output against a pinned SHA-256:

```bash
uv run tsf data prepare --from tfb                     # TFB archive -> dataset/<Name>/<Name>.csv
uv run tsf data prepare --from tfb --datasets fred_md --archive forecasting.zip
uv run tsf data prepare --from dcrnn                   # DCRNN METR-LA -> dataset/metr_la
```

Google Drive downloads use `gdown` and `metr-la.h5` needs `h5py` (both in the `data`
extra). If Google Drive refuses a download, save the file in a browser and pass it
with `--archive` (TFB) or `--h5` (DCRNN). Scaling
must fit training data only, split boundaries must be stable, and graph/covariate
loaders must declare compatible task modes.

### PeMS traffic (UltraTraffic)

The UltraTraffic archive (hourly total flow per Caltrans PeMS station, four
districts, 2003–2023) is the history of the `traffic_pems_*` real-time tracks; it
has no static presets of its own. Convert it once into a local parquet store:

```bash
uv run tsf data prepare --from ultratraffic --archive TrafficCL.zip   # -> dataset/ultratraffic
```

See [the real-time page](realtime.md#tracks) for the stations per district and how
the tracks read the store; `rt/traffic_pems_*` are their frozen static presets.

## Experiments

A run TOML composes base, dataset, and model presets. Keep scientific choices in
config, not shell scripts:

```toml
extends = ["../base.toml", "../datasets/etth1.toml", "../models/DLinear.toml"]

[experiment]
description = "DLinear ETTh1 baseline"
random_seed = 42
work_dir = "./work_dirs"

[task]
mode = "time_series"
seq_len = 96
label_len = 0
pred_len = 96
features = "M"

[sweep]
experiment.random_seed = [0, 1, 2]
task.pred_len = [96, 192]
```

Preview the fully resolved matrix before spending compute:

```bash
uv run tsf run configs/runs/<run>.toml --dry-run
uv run tsf run configs/runs/<run>.toml
```

Optional environment checks, tracking, budgets, and recovery are described in
[execution.md](execution.md). Ordinary runs need no policy or round.

Use `--jobs` and `--gpus` only after confirming resources. Each run preserves its
resolved config, seed, environment, checkpoints, raw metrics, and failures under
`work_dirs/`. Compare only cells with the same data split, preprocessing, horizon,
metric definition, and evaluation strategy. Result commands are discoverable with
`tsf result --help`; they aggregate, rank, plot, inspect predictions, and report.

For iterative work, create an optional research round instead of adding state to
the run config:

```bash
uv run tsf research start --task experiment --goal "Does RevIN improve MSE?" --max-runs 8
uv run tsf run configs/runs/<run>.toml --round <round-id>
uv run tsf research note <round-id> --kind decision --text "Keep the same seeds"
uv run tsf research status <round-id> completed --message "No improvement observed"
```

A round is only a small workspace containing `round.json`, append-only events,
complete command logs, and an optional rendered prompt. It does not alter model,
dataset, config, admission, or result contracts. Resolved runs consume the
declared budget atomically; failures remain visible. Existing Harnesses can be
rendered alone or started with a round:

```bash
uv run tsf agent task render experiment --set 'question=<question>'
uv run tsf agent task start autoresearch --set 'question=<question>' --json
```

`task start` prepares a round and a directly readable prompt. It deliberately
does not launch or message an external Agent; the current Harness owns execution.

## AutoResearch

AutoResearch asks which design suits a dataset and tests it, instead of running a
fixed benchmark. It composes three public pieces and needs no new state beyond an
optional research round.

1. **Profile the data and match the catalog.** `tsf data analyze <preset>` reports
   the training-split statistics, the recommended lookbacks, and the data
   characteristics that fired (see Data). `tsf catalog match <preset>` ranks the
   models and components whose card `fits` overlap those characteristics, grouped by
   slot; that is the candidate menu for the baseline panel and the slot grid. Set
   each candidate's `[data_params]` from the profile (for example a period-sized
   kernel); generic hyperparameters are tuned within the run budget.
2. **Fill the slot grid.** A model is described by six slots, `normalization`,
   `decomposition`, `temporal`, `channel`, `head`, and `loss`, plus at most one
   bounded free-form block for something the catalog cannot express. Options are
   `component:<name>`, `model:<Name>` (a donor design whose idea is borrowed;
   models never import peers), or `loss:<name>`. Start from the best baseline,
   screen one slot at a time, then combine the best compatible winners.
3. **Validate the recombination.** Write a TOML spec and validate it:

   ```toml
   name = "SeasonalRevLinear"
   summary = "RevIN and decomposition around a channel-wise linear map."
   hypothesis = "strong seasonality: a period-aware split beats RLinear at equal lookback."
   parents = ["RLinear", "DLinear"]
   [slots]
   normalization = ["component:revin"]
   decomposition = ["component:series_decomposition"]
   temporal = ["component:channel_wise_linear"]
   channel = ["local:independent"]      # local:independent | local:individual | local:mixing
   head = ["component:flatten_forecast_head"]
   loss = ["loss:mae"]
   [params]                             # optional: hidden, n_layers, n_heads, patch_len, stride, kernel_size, dropout
   hidden = 32
   ```

   ```bash
   uv run tsf model compose spec.toml
   ```

   Without flags `compose` writes nothing. It checks that components, models, and the
   loss are real, that component symbols import, and that the free-form budget (two
   blocks, 120 lines each) holds, then reports whether the slots are **executable**
   by the slot adapters (`executable: yes (point output)` or `NOT EXECUTABLE: <reason>`,
   for example `mixer_block` needs `channel = "mixing"`).

   Executable options, one per slot (absent slots default to `none` / `independent` /
   `flatten_forecast_head` / `mse`):

   | slot | options |
   | --- | --- |
   | normalization | `none`, `component:revin`, `component:last_value_center` |
   | decomposition | `none`, `component:series_decomposition` (`wavelet` is rejected: subbands change length) |
   | temporal | `local:linear`, `component:channel_wise_linear`, `component:tst_transformer` (alias `component:patchtst`), `component:mamba`, `component:gated_dilated_conv`, `component:mixer_block` |
   | channel | `local:independent` (shared weights), `local:individual` (per-channel head weights), `local:mixing` (only with `mixer_block`) |
   | head | `component:flatten_forecast_head` (point), `component:quantile_head` (needs `loss:quantile`), `component:gaussian_parameter_head` (needs `loss:nll_gaussian`) |

4. **Make it runnable and run it.** `--write-config` turns the validated spec into a
   run config that extends a base, a dataset, and the `Composed` model with the slot
   parameters (the point model; quantile and Gaussian heads need `--register` below):

   ```bash
   uv run tsf model compose spec.toml --write-config configs/runs/auto_seasonal.toml \
     --dataset etth1 --enc-in 7 --pred-len 96 --work-dir work_dirs/round1
   # tiny CPU check on the fixture dataset instead:
   uv run tsf model compose spec.toml --write-config /tmp/auto.toml \
     --dataset configs/fixtures/smoke.toml --enc-in 6 --pred-len 12 --smoke
   uv run tsf run --smoke --config /tmp/auto.toml        # or: uv run tsf run configs/runs/auto_seasonal.toml --round <id>
   ```

   `--dataset` is a TOML path or a `configs/datasets/<name>.toml` preset name.

5. **Read the bar to beat.** The board merges the committed leaderboard with local run
   records and prints compact lines, best first (`rank model metric mae/mse H n source`):

   ```bash
   uv run tsf result board --dataset ETTh1 --horizon 96 --top 5
   uv run tsf result board --dataset weather --records work_dirs/round1 --json
   ```

   Local rows list their `seq` and `epochs`; a short or smoke run is not comparable
   with a published row. Slot-assignment runs of `Composed` are labelled by their
   slots, for example `Composed[revin/none/linear/independent]`.

6. **Register the winner.** After confirmation seeds, scaffold a dedicated catalog
   model whose card records the composition and hypothesis as provenance, then admit
   it through the normal gates:

   ```bash
   uv run tsf model compose spec.toml --register SeasonalRevLinear --dry-run   # lists the files
   uv run tsf model compose spec.toml --register SeasonalRevLinear
   uv run tsf model add --name SeasonalRevLinear --verify   # audits, admission, catalog entry; rolls back on failure
   ```

   `--register` writes `src/tsflab/models/<slug>/` (model, spec, a card with
   `fidelity = "composed"`, preset); the generated model fixes the slots, so its
   quantile or Gaussian head declares the matching output capability. Combine
   `--register NAME --write-config PATH` to get a config that runs the new model.

Many wins need no code: change `training.loss`, `task.seq_len`, or a model
parameter in the run config. Register a new model only after it beats the baselines
with confirmation seeds, through the normal path in
[Add a model or method](#add-a-model-or-method).

## Admission and repository gates

Each model is checked once, when it enters the catalog, and again only when its
package, its preset, or a component it uses changes. The check runs the strict
executable contract on CPU (construction from the preset, forward and backward or
one synthetic training step, output shape, finite outputs, active gradients, and a
state-dict round trip) and writes the result into the `[admission]` table of the
model's `card.toml` (status, date, commit, device, whether official code was the
reference, and a note). There are no separate evidence files, fingerprints, or a
per-model test suite; `tests/` covers the infrastructure.

```bash
uv run tsf model verify DLinear                       # one or more models
uv run tsf model verify --changed --base origin/dev   # models touched since a ref
uv run tsf model verify --all --jobs 8                # the whole catalog
uv run tsf model audit --summary                      # card facts and admission records
uv run tsf model audit --summary --release            # also require every admission passed
uv run tsf repo check --audit
uv run tsf repo check --scope release
```

CI runs static checks and the infrastructure tests; it does not run models. The
pull request records `[admission]` for every changed model. A final release
requires every admission to be `passed`; release candidates report the audit as a
warning.

Paper-result reproduction is separate from admission: reproduce datasets,
splits, preprocessing, optimization, seeds, metrics, and reported cells through
run configs, then state all deviations instead of treating a successful forward
pass as a reproduced paper result.
