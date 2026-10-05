# Contributing to TSFLab

Thanks for helping grow the benchmark! This guide covers the common contributions:
proposing or adding a model, submitting results, forecasting real-time rounds,
adding a dataset, and reporting issues.

## Branching & releases

- **`dev` is the integration branch — open your PRs against `dev`, not `main`.**
  New models, datasets, bug fixes, and features all land on `dev` first.
- **During the 1.0 release-candidate phase, `dev` is the RC line.** It carries
  `1.0.0rcN` builds; each RC is tagged `v1.0.0rcN` on `dev`, and `main` moves
  only when a candidate is promoted to `v1.0.0`. Fixes for an RC land on `dev`
  through pull requests like any other change.
- **`main` is release-only and versioned.** It is protected (a PR is required to
  merge, the `ci` checks (`gate`, `wheel`, `web`) must pass, force-pushes and deletion are
  blocked). `main` normally advances only by promoting `dev` → `main`, and
  **every `main` update bumps the version (`pyproject.toml`) and
  ships a tagged GitHub Release** (`vX.Y.Z`, [semver](https://semver.org/)):
  bug-fix-only promotions bump the patch, new models/features bump the minor.
- A maintainer may push an urgent hot-fix straight to `main` (admin bypass), but
  it still must carry the version bump + release.
- Branch names follow the conventional scopes used in history: `feat/…`,
  `fix/…`, `docs/…`, `chore/…`.

## Setup

```bash
bash scripts/detect_hardware.sh   # install profile: cpu | cuda126 | cuda130
uv sync --frozen --python 3.12    # cuda130 (driver >= 580) and macOS
uv run tsf env doctor             # verifies torch; prints cuda126/cpu commands
```

`uv sync` installs the locked cu130 build; other backends use
`uv pip install --torch-backend <cu126|cpu>` (see `docs/en/README.md`). Do **not**
add a hardcoded `+cuXXX` torch pin or index to `pyproject.toml` or edit `uv.lock`
for a local driver — it breaks other machines.

## Reporting issues

Open an issue from one of two templates: **Add a paper** or **Report a problem
or ask a question**. An agent classifies it and replies. A reproducible problem
gets a fix through the same check, review, and merge path as any other change;
include the command, config, traceback, and environment so it can be reproduced.

## Proposing a method (no code required)

Open an **Add a paper** issue with the paper link; the agent chooses the model
name. `agent.yml` triages it and posts the decision. An accepted paper is
implemented as a catalog model; the workflow independently runs
`tsf repo check`, a second read-only agent reviews the change, and the pull
request is merged when both pass (`INTAKE_AUTOMERGE=false` leaves merging to
maintainers). The same workflow scans the literature weekly.

## CI and the agent maintainer

> **Current state:** CI runs the static checks, the infrastructure test suite,
> and admission for changed models on every push and pull request; deployment,
> release, and the agent and weekly workflows are started manually until their
> secrets are configured. The rest of this section describes the full setup they
> return to.

Three workflows, organized by module:

- `ci.yml`: on every push and pull request, `tsf repo check` (the single
  definition of mergeable: schema, agent assets, cards, repository audit, web
  submissions, pytest, and the affected-model smoke run on pull requests),
  `tsf model verify --changed` (admission for every model whose package, preset,
  or used component changed), the wheel check, and the web build. A push to the deploy branch
  (`DEPLOY_BRANCH`, default `main`) deploys the site; a `v*` tag builds, creates
  the GitHub release, and publishes to PyPI.
- `agent.yml`: weekly paper discovery, issue handling, and contributor pull
  request review. Every merge needs `tsf repo check` to pass and an agent
  approval; issue and PR text is treated as untrusted data. Pull requests that
  touch workflows, agent assets, dependencies, scripts, or the check itself are
  left for a maintainer. Set `AGENT_PAUSED=true` to stop every agent job.
- `weekly.yml`: the real-time cycle (merged after `tsf repo check`) and the
  pinned-data link check.

Run `uv run tsf repo check --scope changed` before opening a pull request.

## Submitting results and real-time forecasts

- **Benchmark results:** add a `submission.json` under `apps/web/submissions/`
  (a staging folder) as described in [`apps/web/SUBMITTING.md`](apps/web/SUBMITTING.md).
  CI validates it against the TSF-Core contract; after review a maintainer moves
  it to `results/` of the Hugging Face repository `Diaugeia/TSFLab-Checkpoints`,
  which the leaderboard is built from.
- **Real-time rounds:** add `forecasts/<YourModel>.json` to an open round under
  `apps/web/submissions/realtime/<track>/rounds/<round_id>/` before its
  deadline; CI rejects pull requests last updated after the deadline. See
  [`docs/en/realtime.md`](docs/en/realtime.md).

## Adding a model

See the [model workflow](docs/en/workflows.md#add-a-model-or-method). In short:

1. Deduplicate and extract the paper; inspect pinned official code when available.
   Match the defining operations against existing components with
   `tsf catalog search --kind component` and `tsf catalog show`.
2. Run `tsf model scaffold` with the paper/source facts and component decisions.
   It creates an unregistered workspace whose card has placeholders.
3. Implement locally and complete the card (see [Cards](#cards)): the facts in
   `card.toml` and the short `README.md`.
4. Run `tsf model add --name <Name> --verify`. Admission registers the model, runs
   the audits and the executable contract, records the result in the card's
   `[admission]`, and rolls the registration back if any gate fails.

## Adding a dataset

See the [data workflow](docs/en/workflows.md#data). A dataset has a preset, a
card under `catalog/datasets/<name>/`, and exactly one TSFLab protocol (split,
scaling, lookbacks, horizons) in the card's `[protocol]` table; the protocol used in
the literature, when different, goes in `[protocol].literature`. Dataset files live under `dataset/` and are never
committed. Smoke and synthetic inputs under `configs/fixtures/` are test fixtures,
not datasets. Check a new card with `tsf data audit`, and profile the data and record its
characteristics with `tsf data analyze <preset> --write-card`.

## Cards

Every model, component, and dataset has one card directory with three files:

- `card.toml`: structured facts. For a model: tags, the data characteristics it
  fits, fidelity (`reference-checked`, `paper-only`, `inferred`, `composed`),
  paper, official code at a pinned revision, the six-slot composition
  (`normalization|decomposition|temporal|channel|head|loss`), data-dependent
  parameters (`[data_params]`: which data property sets a parameter, such as the
  seasonal period or the channel count), upstream issues, and the admission record.
  Components record their role, slot, fits, and interface shapes; datasets their
  source and license, shape, protocol, and measured characteristics.
- `README.md`: a short description in the front matter (what it is, when to use it
  and when not), then fixed sections: Idea, When to use, Configure, Differences for
  models; What it does, When to use, Interface for components; Overview, Protocol
  and pitfalls for datasets. At most 60 lines.
- `reference.md` (optional): longer derivations or provenance.

Nothing generated is stored in a card: config paths, parameters, imports,
consumers, and loaders are read from the code. `tsf catalog show <name>` opens a
card, and `tsf catalog match <dataset>` lists the models and components whose fits
match a dataset's characteristics.

## Admission

A model is checked once when it is added, and again only when its package, its
preset, or a component it uses changes. The check runs the executable contract on
CPU (construction, forward, backward, finite outputs, active gradients, state-dict
round trip) and writes the result into the card's `[admission]` table:

```bash
uv run tsf model verify <Name>                  # one or more models
uv run tsf model verify --changed --base origin/dev   # models touched since a ref
uv run tsf repo check --audit
```

There is no per-model test suite; `tests/` covers the infrastructure. Never edit
`[admission]` by hand or waive a failed contract with documentation. A final
release requires every admission to be `passed` (`tsf repo check --scope release`).

## Documentation

Human documentation (this file, `README.md`, `docs/en/`, and the resource cards)
describes public behavior and CLI workflows. Cards are curated by hand; the model
table in `docs/en/models.md` and the Agent index are generated from the cards and
code: change the card, model, spec, config, or schema first, then run
`uv run tsf repo cards`. `uv run tsf repo check --audit` checks that the documentation
and cards agree with the code.

## Releases

There is no changelog file. Each tagged release is a GitHub Release whose notes
are generated from the merged pull requests, so write clear pull-request titles.

## Licensing

The project is MIT (see [`LICENSE`](LICENSE)). Models are maintained as local
implementations after checking papers and official code; external model source is
not vendored. Record dependency notices in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
