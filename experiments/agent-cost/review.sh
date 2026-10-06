#!/usr/bin/env bash
# Review one finished session with a read-only agent: review.sh <run-dir> [reviewer-id]
# Writes <run-dir>/review.<id>.json. Uses Codex (REVIEW_MODEL, default gpt-6-luna).
set -euo pipefail
RUN="$(realpath "${1:?usage: review.sh <run-dir> [reviewer-id]}")"
ID="${2:-a}"
HERE="$(cd "$(dirname "$0")" && pwd)"
TASK="$(python3 -c 'import json,sys; m=json.load(open(sys.argv[1])); print(m["task"])' "$RUN/meta.json")"
TASK="$HERE/tasks/$TASK"
PROMPT="$(python3 - "$TASK" "$HERE/prompts/review.md" <<'PY'
import json, sys
t = json.load(open(sys.argv[1])); tpl = open(sys.argv[2]).read()
seen = []
for c in t["cells"]:
    s = f"{c['dataset']} at {c['pred_len']}"
    if s not in seen: seen.append(s)
print(tpl.format(paper_url=t["paper_url"], cells=", ".join(seen)))
PY
)"
mkdir -p "$RUN/review-$ID"; ln -sf "$HOME/.codex/auth.json" "$RUN/review-$ID/auth.json"
(cd "$RUN/workspace" && CODEX_HOME="$RUN/review-$ID" timeout 1h codex exec --json -m "${REVIEW_MODEL:-gpt-6-luna}" \
   -s read-only --skip-git-repo-check -o "$RUN/review.$ID.raw" "$PROMPT" < /dev/null > "$RUN/review-$ID/stream.jsonl" 2>&1) || true
python3 "$HERE/parse_review.py" "$RUN/review.$ID.raw" "$RUN/review.$ID.json"
rm -f "$RUN/review-$ID/auth.json"
