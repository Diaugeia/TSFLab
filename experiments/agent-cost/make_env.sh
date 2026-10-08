#!/usr/bin/env bash
# Build the working directory for one arm of the agent-cost experiment.
#
#   make_env.sh <arm> <workspace> [tsflab-commit]
#
# Arms:
#   tsflab           TSFLab at the given commit, unchanged.
#   tsflab-noknow    Same code and skills; every README.md and reference.md of a
#                    method, component, or dataset is deleted and `tsf catalog`
#                    is disabled. card.toml stays because data fetching and
#                    admission read it.
#   tsflab-flat      Same code and skills; README.md and reference.md are
#                    concatenated into one KNOWLEDGE.md at the root (same content,
#                    no layered access) and `tsf catalog` is disabled.
#   tsflab-notools   Same knowledge and method code; the experiment tools are
#                    removed: `tsf run`, `tsf result`, and `tsf data` are disabled,
#                    the runner entry points (run_one.py, evaluator.py) are deleted,
#                    and the skills that run or analyze experiments are removed.
#   tsflab-nometa    Same knowledge, code, and tools; the domain meta-cognition is
#                    removed: the entry files (AGENTS.md, CLAUDE.md, .agents/) and
#                    every skill (.agents/skills, .claude/skills).
#   tslib            Time-Series-Library at TSLIB_COMMIT.
#   tfb              TFB at TFB_COMMIT.
#   empty            An empty directory.
#
# A version of TSFLab is selected with the third argument (any arm tsflab*).
# INSTALL=server installs torch for CUDA 12.6 (driver 570, UH_Public).
#
# The Python environment is installed here, before the session, so install time
# is not part of the measured cost. The workspace is committed to a fresh git
# history so the session diff can be measured.
set -euo pipefail

ARM="${1:?usage: make_env.sh <arm> <workspace> [tsflab-commit]}"
WS="${2:?usage: make_env.sh <arm> <workspace> [tsflab-commit]}"
TSFLAB_SRC="${TSFLAB_SRC:-$(cd "$(dirname "$0")/../.." && pwd)}"
TSFLAB_COMMIT="${3:-$(git -C "$TSFLAB_SRC" rev-parse dev)}"
TSLIB_URL="${TSLIB_URL:-https://github.com/thuml/Time-Series-Library.git}"
TSLIB_COMMIT="${TSLIB_COMMIT:-4e938a1}"   # main on 2026-10-06 (last change 2026-04-18)
TFB_URL="${TFB_URL:-https://github.com/decisionintelligence/TFB.git}"
TFB_COMMIT="${TFB_COMMIT:-a0087ed7}"     # master on 2026-10-06
INSTALL="${INSTALL:-local}"

if [[ -e "$WS" ]]; then echo "workspace exists: $WS" >&2; exit 1; fi
mkdir -p "$(dirname "$WS")"

clone_tsflab() {
  git clone --quiet --no-hardlinks "$TSFLAB_SRC" "$WS"
  git -C "$WS" checkout --quiet "$TSFLAB_COMMIT"
  rm -rf "$WS/.git" "$WS/.claude/worktrees" "$WS/.claude/settings.local.json"
}

knowledge_dirs() {
  # Directories that hold a knowledge card (card.toml).
  find "$WS/src" "$WS/catalog" -name card.toml -printf '%h\n' 2>/dev/null | sort -u
}

disable_cmds() {
  # Replace the given `tsf` entry points with a stub that explains they are unavailable.
  python3 - "$WS/src/tsflab/cli/main.py" "$@" <<'PY'
import re, sys
p, names = sys.argv[1], sys.argv[2:]
s = open(p).read()
for name in names:
    s, n = re.subn(r'"%s": _?[A-Za-z_]+(\([^)]*\))?,' % name,
                   '"%s": _lazy("tsflab.cli.commands.unavailable", "unavailable_command"),' % name, s)
    assert n == 1, f"entry point not found: {name}"
open(p, "w").write(s)
PY
  cat > "$WS/src/tsflab/cli/commands/unavailable.py" <<'PY'
"""Stub for commands that are disabled in this environment."""


def unavailable_command(argv=None) -> int:
    print("This command is not available in this environment.")
    return 2
PY
}

disable_catalog() { disable_cmds catalog; }

case "$ARM" in
  tsflab)
    clone_tsflab
    ;;
  tsflab-noknow)
    clone_tsflab
    knowledge_dirs | while read -r d; do rm -f "$d/README.md" "$d/reference.md"; done
    disable_catalog
    ;;
  tsflab-flat)
    clone_tsflab
    {
      echo "# Knowledge of every method, component, and dataset"
      knowledge_dirs | while read -r d; do
        rel="${d#"$WS"/}"
        echo; echo "## $rel"
        [[ -f "$d/README.md" ]] && cat "$d/README.md"
        [[ -f "$d/reference.md" ]] && { echo; cat "$d/reference.md"; }
      done
    } > "$WS/KNOWLEDGE.md"
    knowledge_dirs | while read -r d; do rm -f "$d/README.md" "$d/reference.md"; done
    disable_catalog
    ;;
  tsflab-nometa)
    clone_tsflab
    rm -rf "$WS/AGENTS.md" "$WS/CLAUDE.md" "$WS/.agents" "$WS/.claude/skills"
    ;;
  tsflab-notools)
    clone_tsflab
    disable_cmds run result data
    rm -f "$WS/src/tsflab/experiments/runner/run_one.py" "$WS/src/tsflab/experiments/runner/evaluator.py"
    for s in run-experiment diagnose-experiment reproduce-paper-results analyze-results submit-results run-autoresearch; do
      rm -rf "$WS/.agents/skills/$s" "$WS/.claude/skills/$s"
    done
    # From dev 79e3583b, AGENTS.md points to `tsf data splits`, which is disabled here; point to the
    # dataset knowledge page, which states the same split facts. The data loaders stay.
    sed -i 's/`tsf data splits <dataset>`; never re-derive them\./the dataset knowledge page (README.md); never re-derive them./' "$WS/AGENTS.md"
    ;;
  tfb)
    git clone --quiet "$TFB_URL" "$WS"
    git -C "$WS" checkout --quiet "$TFB_COMMIT"
    rm -rf "$WS/.git"
    ;;
  tslib)
    git clone --quiet "$TSLIB_URL" "$WS"
    git -C "$WS" checkout --quiet "$TSLIB_COMMIT"
    rm -rf "$WS/.git"
    ;;
  empty)
    mkdir -p "$WS"
    ;;
  *) echo "unknown arm: $ARM" >&2; exit 1 ;;
esac

# Install the environment before the session (not measured). SKIP_INSTALL=1 skips it (tests only).
TORCH=()
[[ "$INSTALL" == server ]] && TORCH=(--torch-backend cu126)
[[ "${SKIP_INSTALL:-0}" == 1 ]] || case "$ARM" in
  tsflab*)
    if [[ "$INSTALL" == server ]]; then
      # Locked torch is the CUDA 13 build; on driver 570 install with the cu126 backend.
      (cd "$WS" && uv venv --quiet --python 3.12 && \
        { uv pip install --quiet "${TORCH[@]}" -e '.[models,data,experiments]' 2>/dev/null || \
          uv pip install --quiet "${TORCH[@]}" -e .; })
    else
      (cd "$WS" && uv sync --frozen --python 3.12 --quiet)
    fi ;;
  tslib) (cd "$WS" && uv venv --quiet --python 3.11 && uv pip install --quiet "${TORCH[@]}" -r requirements.txt) ;;
  tfb)   (cd "$WS" && uv venv --quiet --python 3.10 && uv pip install --quiet "${TORCH[@]}" -r requirements.txt) ;;
  empty) (cd "$WS" && uv venv --quiet --python 3.12 && uv pip install --quiet "${TORCH[@]}" torch numpy pandas scikit-learn) ;;
esac

# Fresh history: the session diff is measured against this commit.
mkdir -p "$WS/results"
grep -qxF '.venv/' "$WS/.gitignore" 2>/dev/null || echo '.venv/' >> "$WS/.gitignore"
(cd "$WS" && git init --quiet && git add -A && git -c user.name=agent-cost -c user.email=agent-cost@localhost commit --quiet -m baseline)
echo "$ARM workspace ready: $WS"
