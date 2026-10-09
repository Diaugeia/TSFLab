#!/usr/bin/env bash
# Package sessions whose run_session.sh ended before packaging (e.g. the script was edited while it ran):
#   finalize.sh <root>...
# A session qualifies when it has a stream, no metrics.json, and no live process in its workspace.
export PATH=$HOME/.local/bin:$PATH GIT_CONFIG_GLOBAL=/dev/null
HERE="$(cd "$(dirname "$0")" && pwd)"
for root in "$@"; do
  for d in "$root"/*/; do
    d=${d%/}
    [ -f "$d/meta.json" ] && ls "$d"/stream.*.jsonl > /dev/null 2>&1 || continue
    [ -f "$d/metrics.json" ] && continue
    live=0
    for p in /proc/[0-9]*; do
      c=$(readlink "$p/cwd" 2>/dev/null) || continue
      case "$c" in "$(realpath "$d")"*) live=1; break;; esac
    done
    [ $live = 1 ] && { echo "still running: $d"; continue; }
    task="$HERE/tasks/$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["task"])' "$d/meta.json")"
    "$HERE/package.sh" "$task" "$d" > /dev/null 2>&1 || echo "package failed: $d"
    rm -f "$d/config/.credentials.json" "$d/codex/auth.json" 2>/dev/null
    [ -f "$d/metrics.json" ] && echo "finalized: $d"
  done
done
