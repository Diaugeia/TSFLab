#!/usr/bin/env bash
# Review one finished session with a read-only agent: review.sh <run-dir> [reviewer-id]
# Writes <run-dir>/review.<id>.json. Uses Codex (REVIEW_MODEL, default gpt-6-luna).
set -euo pipefail
RUN="$(realpath "${1:?usage: review.sh <run-dir> [reviewer-id]}")"
ID="${2:-a}"
HERE="$(cd "$(dirname "$0")" && pwd)"
TASK="$(python3 -c 'import json,sys; m=json.load(open(sys.argv[1])); print(m["task"])' "$RUN/meta.json")"
TASK="$HERE/tasks/$TASK"
PROMPT="$(python3 - "$TASK" "$HERE/prompts" <<'PY'
import json, sys
t = json.load(open(sys.argv[1])); d = sys.argv[2]
if t["task"] == "reproduce":
    seen = []
    for c in t["cells"]:
        s = f"{c['dataset']} at {c['pred_len']}"
        if s not in seen: seen.append(s)
    print(open(f"{d}/review.md").read().format(paper_url=t["paper_url"], cells=", ".join(seen)))
elif t["task"] == "benchmark":
    print(open(f"{d}/review-benchmark.md").read().format(methods=", ".join(t["methods"]), file=t["file"],
          seq_len=t["seq_len"], pred_len=t["pred_len"]))
else:
    print(open(f"{d}/review-autoresearch.md").read().format(target_method=t["target_method"], dataset=t["dataset"],
          split=t["split"], pred_lens=", ".join(str(p) for p in t["pred_lens"])))
PY
)"
mkdir -p "$RUN/review-$ID"; ln -sf "$HOME/.codex/auth.json" "$RUN/review-$ID/auth.json"
(cd "$RUN/workspace" && CODEX_HOME="$RUN/review-$ID" timeout 1h codex exec --json -m "${REVIEW_MODEL:-gpt-6-luna}" \
   -s read-only --skip-git-repo-check -o "$RUN/review.$ID.raw" "$PROMPT" < /dev/null > "$RUN/review-$ID/stream.jsonl" 2>&1) || true
python3 "$HERE/parse_review.py" "$RUN/review.$ID.raw" "$RUN/review.$ID.json"
rm -f "$RUN/review-$ID/auth.json"
