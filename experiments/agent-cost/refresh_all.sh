#!/usr/bin/env bash
# Recompute every measure of the finished sessions under the given roots and export them:
#   refresh_all.sh <out.json> <root>...
# analyze.py (accounting with web reads), process_stats.py, code_quality.py (once per session), campaign_sessions.py.
set -u
export PATH=$HOME/.local/bin:$PATH GIT_CONFIG_GLOBAL=/dev/null
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$1"; shift
for root in "$@"; do
  for m in "$root"/*/metrics.json; do
    [ -f "$m" ] || continue
    d="$(dirname "$m")"
    [ -f "$d/metrics.v1.json" ] || cp "$m" "$d/metrics.v1.json"
    python3 "$HERE/analyze.py" "$d" > /dev/null 2>&1 || echo "analyze failed: $d"
    python3 "$HERE/process_stats.py" "$d" > /dev/null 2>&1
    [ -f "$d/quality.json" ] || timeout 900 python3 "$HERE/code_quality.py" "$d" > /dev/null 2>&1 || echo "quality failed: $d"
  done
done
python3 "$HERE/campaign_sessions.py" "$@" > "$OUT"
echo "wrote $OUT ($(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$OUT") sessions)"
