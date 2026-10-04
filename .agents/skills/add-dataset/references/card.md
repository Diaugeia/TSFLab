# Writing the dataset card

The card format is in `.agents/STANDARDS.md` (Cards); copy the layout of a sibling
card (`uv run tsf catalog show <sibling> --depth 3` prints its paths). A card is
`catalog/datasets/<name>/card.toml` (facts), `README.md` (front matter `name` and
`description`, then Overview and Protocol and pitfalls, <= 60 lines), and an
optional `reference.md` for long provenance or statistics. Loader, files, task
modes, and parameters are derived from code; never copy them into the card.

- `card.toml`: `domain` (exactly one of energy, transport, environment, finance,
  healthcare, cloud-web, sales), optional `topic` (free-text detail), `benchmarks`
  (suites the preset belongs to: ltsf, tfb, st-graph, gift-eval, realtime), `tags`,
  `related`, `[source]` (name, url, citation, citation_url, license, license_url,
  redistribution, conditions), `[shape]` (frequency, time_span, length,
  channels, channel_kind, target, missing_values, stats_basis), `[protocol]`
  (protocol, literature, split, seq_lens, pred_lens).
- Measure length, channels, span, frequency, and missing values from the local file
  (read-only) and set `stats_basis = "measured"`. For facts you cannot measure,
  fetch the primary source and label them `source-reported`; use `mixed` for both.
- Record `characteristics` with `uv run tsf data analyze <preset> --write-card`
  (data terms from the train-only profile, with `characteristics_basis`); add task
  terms (`spatial-graph`, `calendar-effects`, `exogenous-covariates`, ...) by hand
  only when the loader or source supports them.
- Set `redistribution` from the source terms, with `license_url` as evidence:
  `hosted` (TSFLab re-hosts the files in TSFLab-Static; write any extra terms the
  source attaches, such as verbatim copies only, in `conditions`), `upstream`
  (fetched from another party's Hugging Face repository: GIFT-Eval through
  `tsf data prepare --from gift-eval`, others through an `upstream` entry in
  `configs/hub/datasets.json`), or `script` (no explicit terms, or terms that forbid
  re-hosting: TSFLab ships a fetch command, such as `tsf data prepare --from tfb`,
  and users download from the original source). For `upstream` and `script`,
  Protocol and pitfalls names the one command that gets the data.
  Never guess a license or a number.
- Document split conventions, `drop_last`, scaling, zeros or sentinels, shift, and
  leakage risks under Protocol and pitfalls; a differing literature protocol goes
  in `[protocol].literature`.
- Verify with `uv run tsf catalog show <name>`, `uv run tsf data audit`, and
  `uv run tsf catalog match <name>`.
