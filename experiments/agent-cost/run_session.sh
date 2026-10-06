#!/usr/bin/env bash
# Run one measured agent session.
#
#   run_session.sh <task.json> <arm> <run-dir> [model]
#
# Creates <run-dir>/workspace with make_env.sh, then runs the agent headless in
# it with an isolated configuration (no user instructions, memory, plugins, or
# MCP servers). AGENT=claude (default) runs Claude Code; AGENT=codex runs Codex
# CLI with CODEX_HOME=<run-dir>/codex, which links the login of ~/.codex. Every stream-json line is stored with the time it arrived.
# If the session stops before results/results.json covers every cell, the
# runner resumes it with one fixed message, at most MAX_RESUMES times; each
# resume counts as one human intervention.
#
# Environment: AGENT (claude|codex), TSFLAB_COMMIT (version for tsflab* arms),
# DATA_DIR (raw data, read-only), TIMEOUT_H (default 8),
# MAX_RESUMES (default 2), MAX_BUDGET_USD (optional), GPU_SAMPLE_S (default 5).
set -euo pipefail

TASK="$(realpath "${1:?usage: run_session.sh <task.json> <arm> <run-dir> [model]}")"
ARM="${2:?arm}"
RUN="$(realpath -m "${3:?run-dir}")"
AGENT="${AGENT:-claude}"
if [[ "$AGENT" == codex ]]; then MODEL="${4:-gpt-6-luna}"; else MODEL="${4:-claude-sonnet-5-5}"; fi
HERE="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${DATA_DIR:?set DATA_DIR to the raw data directory}"
TIMEOUT_H="${TIMEOUT_H:-8}"
MAX_RESUMES="${MAX_RESUMES:-2}"
GPU_SAMPLE_S="${GPU_SAMPLE_S:-5}"

mkdir -p "$RUN"
"$HERE/make_env.sh" "$ARM" "$RUN/workspace" ${TSFLAB_COMMIT:+"$TSFLAB_COMMIT"}

# Isolated agent configuration.
mkdir -p "$RUN/config"
if [[ "$AGENT" == codex ]]; then
  # Link (not copy) the login so a token refresh in one session reaches the others.
  mkdir -p "$RUN/codex"
  export CODEX_HOME="$RUN/codex"
  if [[ -n "${CODEX_BASE_URL:-}" ]]; then
    # A self-hosted model behind an OpenAI-compatible Responses API (vLLM): no login.
    cat > "$CODEX_HOME/config.toml" <<TOML
model_provider = "local"
[model_providers.local]
name = "local"
base_url = "$CODEX_BASE_URL"
wire_api = "responses"
env_key = "CODEX_LOCAL_KEY"
TOML
    export CODEX_LOCAL_KEY="${CODEX_LOCAL_KEY:-none}"
  else
    ln -sf "$HOME/.codex/auth.json" "$CODEX_HOME/auth.json"
  fi
  AGENT_VERSION="$(codex --version | head -1)"
elif [[ -n "${ANTHROPIC_BASE_URL:-}" ]]; then
  # A vLLM server with the Anthropic API: ANTHROPIC_API_KEY, no login.
  : "${ANTHROPIC_API_KEY:?set ANTHROPIC_API_KEY for the server at ANTHROPIC_BASE_URL}"
  export ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL" ANTHROPIC_SMALL_FAST_MODEL="$MODEL"
  export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
  AGENT_VERSION="$(claude --version | head -1)"
else
  install -m 600 "$HOME/.claude/.credentials.json" "$RUN/config/.credentials.json"
  AGENT_VERSION="$(claude --version | head -1)"
fi
# The agent sees one training GPU (TRAIN_GPU), if given.
if [[ -n "${TRAIN_GPU:-}" ]]; then export CUDA_VISIBLE_DEVICES="$TRAIN_GPU"; fi

KIND="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["task"])' "$TASK")"
PROMPT="$(python3 "$HERE/render_prompt.py" "$TASK" "$DATA_DIR")"
printf '%s\n' "$PROMPT" > "$RUN/prompt.txt"
cat > "$RUN/meta.json" <<EOF
{"task": "$(basename "$TASK")", "kind": "$KIND", "arm": "$ARM", "agent": "$AGENT", "model": "$MODEL",
 "started": "$(date -u +%Y-%m-%dT%H:%M:%SZ)", "host": "$(uname -n)", "gpu_sample_s": $GPU_SAMPLE_S,
 "tsflab_commit": "${TSFLAB_COMMIT:-$(git -C "$HERE/../.." rev-parse dev)}", "agent_version": "$AGENT_VERSION",
 "base_url": "${CODEX_BASE_URL:-${ANTHROPIC_BASE_URL:-default}}", "train_gpu": "${TRAIN_GPU:-all}"}
EOF

# GPU sampler (utilization and memory every GPU_SAMPLE_S seconds).
nvidia-smi ${TRAIN_GPU:+-i "$TRAIN_GPU"} --query-gpu=timestamp,utilization.gpu,memory.used \
  --format=csv,noheader -l "$GPU_SAMPLE_S" > "$RUN/gpu.csv" 2>/dev/null &
GPU_PID=$!
trap 'kill $GPU_PID 2>/dev/null || true' EXIT

export AGENT_COST_DIR="$RUN"
export PATH="$HERE/bin:$PATH"
[[ "$AGENT" == codex ]] || export CLAUDE_CONFIG_DIR="$RUN/config"

BUDGET_ARGS=()
[[ -n "${MAX_BUDGET_USD:-}" ]] && BUDGET_ARGS=(--max-budget-usd "$MAX_BUDGET_USD")

stamp() {
  # Prefix each stream-json line with the arrival time.
  python3 -u -c '
import sys, json, time
for line in sys.stdin:
    line = line.rstrip("\n")
    if line:
        try:
            d = json.loads(line)
        except ValueError:
            d = {"type": "raw", "text": line}
        sys.stdout.write(json.dumps({"t": time.time(), "d": d}) + "\n")
        sys.stdout.flush()
'
}

session_id=""
for attempt in $(seq 0 "$MAX_RESUMES"); do
  if [[ $attempt -eq 0 ]]; then
    MSG="$PROMPT"
  else
    MSG="Continue until the task is complete."
    printf '%s\tresume %d\n' "$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)" "$attempt" >> "$RUN/interventions.tsv"
  fi
  if [[ "$AGENT" == codex ]]; then
    if [[ $attempt -eq 0 ]]; then CMD=(codex exec); else CMD=(codex exec resume "$session_id"); fi
    (cd "$RUN/workspace" && timeout "${TIMEOUT_H}h" "${CMD[@]}" --json -m "$MODEL" \
        --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check "$MSG" \
        < /dev/null 2> "$RUN/stderr.$attempt.txt" | stamp > "$RUN/stream.$attempt.jsonl") || true
    session_id="$(python3 -c '
import json, sys
sid = ""
for l in open(sys.argv[1]):
    d = json.loads(l)["d"]
    if d.get("type") == "thread.started":
        sid = d.get("thread_id", sid)
print(sid)' "$RUN/stream.$attempt.jsonl")"
  else
    RESUME_ARGS=()
    [[ $attempt -gt 0 ]] && RESUME_ARGS=(--resume "$session_id")
    (cd "$RUN/workspace" && timeout "${TIMEOUT_H}h" claude -p "$MSG" "${RESUME_ARGS[@]}" \
        --model "$MODEL" --output-format stream-json --verbose \
        --dangerously-skip-permissions --strict-mcp-config "${BUDGET_ARGS[@]}" \
        2> "$RUN/stderr.$attempt.txt" | stamp > "$RUN/stream.$attempt.jsonl") || true
    session_id="$(python3 -c '
import json, sys
sid = ""
for l in open(sys.argv[1]):
    d = json.loads(l)["d"]
    sid = d.get("session_id", sid)
print(sid)' "$RUN/stream.$attempt.jsonl")"
  fi
  if python3 "$HERE/check_success.py" "$TASK" "$RUN/workspace" --quiet; then break; fi
  [[ -z "$session_id" ]] && break
done

# Package: diff against the baseline commit, then analyze.
(cd "$RUN/workspace" && git add -A && git diff --cached --numstat HEAD > "$RUN/diff.numstat" \
  && git diff --cached HEAD > "$RUN/diff.patch") || true
python3 "$HERE/check_success.py" "$TASK" "$RUN/workspace" > "$RUN/success.json" || true
python3 "$HERE/analyze.py" "$RUN" > "$RUN/metrics.json"
rm -f "$RUN/config/.credentials.json" "$RUN/codex/auth.json" 2>/dev/null || true
echo "done: $RUN/metrics.json"
