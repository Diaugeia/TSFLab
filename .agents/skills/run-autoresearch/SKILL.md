---
name: run-autoresearch
description: "Run a bounded, profile-driven forecasting research loop: analyze the data, state hypotheses, run a baseline panel, search and recombine cataloged components, and conclude from evidence. Use when autonomous experiment iteration is authorized; not for a single predefined run or a paper implementation."
---

# Run bounded autoresearch

Iterate profile, hypothesis, experiment, and evidence inside hard budgets, in the
current Agent (no second Agent needed). Read each reference only at its step.

## Inputs

- A research question, approved datasets, primary metric, hard run, iteration, and
  time budgets, and authorization to execute experiments.
- Separate authorization before writing model code or registering a model; without
  it, search over configs and existing catalog methods only.
- Heavy runs happen on GPU machines or CI, not locally: preview each matrix with
  `--dry-run`, hand the exact `tsf run ... --round <id>` command to the run machine,
  and continue from the records and round events that come back in `work_dirs/`.
  Stop and report when results have not returned instead of running them locally.

## Context read, per module

| Module | Context | Read with |
| --- | --- | --- |
| Data | dataset card `characteristics`, train-only profile | `tsf catalog show`, `tsf data analyze` |
| Models | card `fits`, compositions, `data_params`, component interfaces | `tsf catalog match`, `tsf catalog search`, `tsf catalog show` |
| Experiments | result board (aggregates, ranks) | `tsf result board --dataset <d>`, `tsf result aggregate` |
| Release | leaderboard (`board/` of `TSFLab-Checkpoints`), real-time summaries (`apps/web/data/realtime/`) | read-only reference bar |
| AutoResearch | round ledger `work_dirs/_research/<id>/` | `tsf research show <id>` |

## Steps

1. Profile each dataset first: `uv run tsf data analyze <preset>` (or `--path FILE`);
   read the fired rules (= card data terms) per [data-profile](references/data-profile.md).
2. Write 2-4 falsifiable hypotheses, each tied to a profile fact, with the metric
   change that would refute it. Design controls and seeds per `run-experiment`.
3. Open a round for durable budgets (`tsf research start --goal ... --max-runs N
   --max-iterations M`, or reuse a supplied one) and note each hypothesis
   (`tsf research note <id> --kind hypothesis --text ...`). Claim one iteration per phase
   with `tsf research iteration <id> --operation <phase>`; reuse of an id is free.
4. Retrieve candidates: `uv run tsf catalog match <preset> [--extra TERM...]` ranks
   models and slot-grouped components whose card `fits` overlap the dataset's
   characteristics; widen with `tsf catalog search`, open L1 with `tsf catalog show`;
   see [retrieval-baselines](references/retrieval-baselines.md). A match is a hypothesis.
5. Evaluate a small baseline panel before any search (same split, horizon, metric,
   seeds); it sets the bar every later candidate must beat.
6. Search by changing one factor per iteration from the best baseline, starting with
   the cheapest run that can reject a hypothesis. Set data-dependent parameters from
   card `[data_params]` and the profile; tune generic hyperparameters yourself within
   budget. Recombine via the slot grid per [recombination](references/recombination.md)
   (`tsf model compose <spec.toml>` dry-runs a composition).
7. Confirm only finalists with the declared seeds; compare with spread, not a single
   best run. If authorized and a recombined method wins, register it with `add-model`.
8. Record decisions, runs, and conclusions as round events per
   [round-ledger](references/round-ledger.md); route failures to `diagnose-experiment`
   and compatible results to `analyze-results`, whose board feeds the next iteration.
   Never overwrite a costly run.

## Leakage guards

- Select and stop on train and validation only, with the rule fixed first; test metrics
  confirm frozen candidates once; never act on `shift.train_to_test`.

## Chain

- Module: AutoResearch; reads the context table above.
- Produces: the round ledger, conclusions, and a winning composition spec.
- Hands off to: `add-model` (registers a winner), `submit-results`, `analyze-results`, `diagnose-experiment`.

## Success

- Profile paths, hypotheses with verdicts, baseline panel, run ledger with seeds and
  uncertainty, excluded or failed cells, limitations, and a stop/continue recommendation.

## Stop and hand off

- Stop on budget exhaustion, repeated infrastructure failure, an invalid comparison,
  no measurable progress after two consecutive iterations, or an acceptance criterion met.
- External publication and task dispatch stay out of scope unless separately authorized.
