#!/usr/bin/env bash
# Start several sessions in parallel, one per GPU, detached from the terminal.
#
#   launch.sh <task.json> <out-root> <spec>...
#
# A spec is an arm, optionally with a TSFLab version: tsflab, tslib, tsflab@cde2f69e.
# Session i trains on GPU GPUS[i] (GPUS="0 1 2 ..."; default: all GPUs). Each session
# writes <out-root>/<task-id>-<spec>/ and logs to <out-root>/<task-id>-<spec>.log.
# Environment is passed through (AGENT, DATA_DIR, TIMEOUT_H, INSTALL, ...).
set -euo pipefail
TASK="$(realpath "${1:?usage: launch.sh <task.json> <out-root> <spec>...}")"
ROOT="$(realpath -m "${2:?out-root}")"; shift 2
HERE="$(cd "$(dirname "$0")" && pwd)"
read -r -a G <<< "${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')}"
ID="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["id"])' "$TASK")"
mkdir -p "$ROOT"
i=0
for spec in "$@"; do
  arm="${spec%@*}"; commit=""; [[ "$spec" == *@* ]] && commit="${spec#*@}"
  name="$ID-${spec//@/-}"
  gpu="${G[$((i % ${#G[@]}))]}"
  TRAIN_GPU="$gpu" TSFLAB_COMMIT="$commit" setsid nohup \
    "$HERE/run_session.sh" "$TASK" "$arm" "$ROOT/$name" > "$ROOT/$name.log" 2>&1 < /dev/null &
  echo "started $name on GPU $gpu (pid $!)"
  i=$((i + 1))
done
