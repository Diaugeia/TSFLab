#!/usr/bin/env bash
# Package one session: diff against the baseline commit (including small code and config
# files that the codebase's .gitignore hides, e.g. under experiments/ or work_dirs/),
# then check success and compute metrics. package.sh <task.json> <run-dir>
set -euo pipefail
TASK="$(realpath "$1")"; RUN="$(realpath "$2")"; HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$RUN/workspace"
git add -A
git ls-files --others --ignored --exclude-standard -z \
  | python3 -c '
import os, re, sys
keep = re.compile(r"\.(py|sh|toml|ya?ml|json|md|txt|cfg|ini|ipynb)$")
skip = re.compile(r"(^|/)(\.venv|\.cache|__pycache__|\.git|checkpoints?|wandb|lightning_logs|node_modules)(/|$)")
for p in sys.stdin.buffer.read().decode().split("\0"):
    if p and keep.search(p) and not skip.search(p) and os.path.getsize(p) < 1_000_000:
        sys.stdout.write(p + "\0")
' | xargs -0 -r git add -f --
git diff --cached --numstat HEAD > "$RUN/diff.numstat"
git diff --cached HEAD > "$RUN/diff.patch"
cd - > /dev/null
python3 "$HERE/check_success.py" "$TASK" "$RUN/workspace" > "$RUN/success.json" || true
python3 "$HERE/analyze.py" "$RUN" > "$RUN/metrics.json"
