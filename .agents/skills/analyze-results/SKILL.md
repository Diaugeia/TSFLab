---
name: analyze-results
description: "Aggregate, filter, rank, compare, plot, and report completed TSFLab experiment results. Use for exploratory analysis, leaderboards, prediction plots, or a verified shareable report; not for failed or suspect runs (diagnose-experiment) or packaging a submission (submit-results)."
---

# Analyze results

Experiments module, closing step: turn completed runs into comparisons and conclusions
the evidence supports; the aggregated results are the result board AutoResearch reads. Reason,
plot, and write with native Agent tools; use library computations for reproducible
aggregation and protocol checks. The CLI helpers are optional.

## Inputs

- Completed run artifacts under `work_dirs/` (produced on GPU machines or CI and copied
  back; analysis here never trains) and the intended comparison set.

## Steps

1. Aggregate without mutating source artifacts:

   ```bash
   uv run tsf result aggregate --dataset <name> --collapse \
     --aggregate mean --null-threshold 0.3
   uv run tsf result board --dataset <name> --json     # published board (Hub) plus local records
   uv run tsf result rank --help
   uv run tsf result plot --help
   uv run tsf result predictions --help
   uv run tsf result report --help
   ```

2. Keep raw and collapsed data distinct. State metric direction, horizon filters,
   seed aggregation, missing-cell policy, and profile availability. Never compare
   across incompatible datasets or evaluation protocols.
3. For a formal report, verify the comparison set and aggregation policy first,
   then read the artifact back and check rankings, direction, missing values,
   counts, uncertainty, and plot references against the aggregated data.
4. When a research round exists, append the evidence-backed conclusion and next
   decision, then mark it completed, blocked, or stopped. Metrics stay in result
   artifacts rather than narrative memory.

## Chain

- Module: Experiments.
- Reads: run records and artifacts under `work_dirs/` (read-only).
- Produces: aggregates `work_dirs/<dataset>/results_all.csv`, ranks, plots, report artifacts, round conclusion; `tsf result board` is the board AutoResearch reads.
- Hands off to: `submit-results` (compatible records), `run-autoresearch` (baselines and bars), `reproduce-paper-results`.

## Success

- An artifact path and scope, with every claim traceable to aggregated data.

## Stop and hand off

- Do not turn incomplete evidence into a claim.
- Published-number replication goes back to `reproduce-paper-results`; a local
  leaderboard submission goes to `submit-results`.
