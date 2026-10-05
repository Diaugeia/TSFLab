# Projects and the Hugging Face Hub

← [Documentation index](README.md)

## Standalone projects

`tsf init <dir>` scaffolds a project that depends on the installed package
instead of a checkout:

```
<dir>/
├── configs/runs/example.toml   # extends tsflab://configs/...
├── dataset/                    # local data (ignored)
├── pyproject.toml              # tsflab[<extras>] and [tool.tsflab] modules
├── agent guide + skills        # generated from the chosen module chain
└── README.md
```

`--modules data,models,experiments,release,autoresearch` (default: all) selects
the module chain. The skills and task templates are copied from the installed
package, so they match its version; run `tsf agent sync [--add MODULE,...]` after
upgrading TSFLab to refresh them (project notes below the generated block
of the agent guide are kept). The chosen modules are recorded under `[tool.tsflab]` in
`pyproject.toml`; `tsf agent modules` lists them and `tsf catalog` shows them in
its overview. Each module maps to a pip extra (`data`, `models`, `experiments`,
`hub` + `realtime` for release, `autoresearch`; all five give `tsflab[all]`).

Run configs may extend catalog presets with `tsflab://` paths, which resolve
against the checkout or the installed package's read-only assets:

```toml
extends = [
    "tsflab://configs/base.toml",
    "tsflab://configs/datasets/etth1.toml",
    "tsflab://configs/models/DLinear.toml",
]
```

Outputs are written to the project's own `work_dirs/`.

## One address for every published asset

Remote weights, checkpoints, and data use one pinned URI form:

```
hf://[datasets/|spaces/]<owner>/<repo>@<revision>/<path>
```

The revision is mandatory, so a URI always names the same bytes. Model
`ModelArtifact` declarations accept `hf://` URIs next to `https://` and
`file://`, and every download is SHA-256 verified into `$TSFLAB_CACHE`
(default `~/.cache/tsflab`). `HF_TOKEN` is sent only to the Hugging Face
endpoint, so private repositories work; `HF_ENDPOINT` overrides the endpoint.

## Published repositories

| Repository | Type | Contents |
| --- | --- | --- |
| `Diaugeia/TSFLab-Datasets` | dataset | `static/`: files behind the dataset presets, laid out as `dataset/`; `realtime/<track>/`: append-only panels of the hosted real-time tracks, one commit per release |
| `Diaugeia/TSFLab-Weights` | model | trained weights bundles |
| `Diaugeia/TSFLab` | space | the static leaderboard site |

A fork or personal mirror sets `TSFLAB_HUB_OWNER` to publish under another
namespace instead of passing `--repo` to every command. Maintainers create them, with their cards, through
`uv run tsf result hub init [--migrate-legacy]`; `--migrate-legacy` renames the former
TSEval leaderboard Space so its old address redirects.

TSFLab 0.8.0 reads the frozen `Diaugeia/TSFLab-Static` repository (files at its
root, manifest schema 1); later releases use `TSFLab-Datasets` only.

## Benchmark data

`configs/hub/datasets.json` pins every published data file by commit and
SHA-256; its `prefix` (`static`) is the folder of the files in the repository.
Presets download into the local `dataset/` root, verified:

```bash
uv run tsf data download --list          # presets with published files
uv run tsf data download etth1 weather   # or --all
uv run tsf data download --check         # every pinned file still resolves
```

Maintainers publish local files to `static/`, which updates the manifest to
commit with the change:

```bash
uv run tsf data publish etth1 etth2 [--create]
uv run tsf data publish --path ultratraffic      # a whole store, every year
```

Each dataset keeps its source license. Publish only presets whose card says
`redistribution = "hosted"`, and meet the card's `conditions` when it has them.
Never publish `upstream` presets (`tsf data download` fetches them from the other
party's repository, and `tsf data prepare --from gift-eval` fetches GIFT-Eval) or
`script` presets (users fetch them from the original source with the command in
the card).

## Weights bundles

A finished run is published as one bundle directory at
`<dataset>/<model>/<run_id>/` in a model repository (default
`Diaugeia/TSFLab-Weights`):

| File | Contents |
| --- | --- |
| `manifest.json` | identity, shapes, metrics, provenance, per-file SHA-256 |
| `model.safetensors` | the best checkpoint's `state_dict` |
| `record.json` | the run's TSF-Core `RunRecord` |
| `README.md` | model card rendered from the manifest |

```bash
uv run tsf result hub pack <run_id>                      # local bundle under work_dirs/_bundles/
uv run tsf result hub push <run_id> --repo <owner>/<repo> [--create] [--public]
uv run tsf result hub list --repo <owner>/<repo> --dataset weather
uv run tsf result hub pull hf://<owner>/<repo>@<revision>/weather/DLinear/<run_id>
```

`push` uploads only the run it is given, creates private repositories unless
`--public` is passed, and prints the bundle URI pinned to the new commit.

```python
from tsflab import hub

state_dict, manifest = hub.load_state_dict(
    "hf://Diaugeia/TSFLab-Weights@<revision>/weather/DLinear/<run_id>"
)
```

Packing, loading, and uploading need the package's `hub` extra
(`huggingface_hub`, `safetensors`); plain downloads use only the standard library.
