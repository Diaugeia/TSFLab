"""Measure the code a session added: code_quality.py <run-dir> [...]. Writes <run-dir>/quality.json.

The workspace is a git repository whose first commit is the state before the session
(make_env.sh). The measure covers Python files that the session added or modified,
outside results, logs, caches, and virtual environments, in the spirit of the
earlier Repo-Bench comparison (ruff C901, radon, duplication):

  files_added, files_modified   Python files the session touched
  lines_added, lines_removed    their added and removed lines (git numstat)
  functions                     functions and methods that overlap added lines
  cc_mean, cc_max               cyclomatic complexity of those functions (radon cc)
  cc_over_10                    how many of them exceed 10 (the ruff C901 default)
  mi_mean                       maintainability index of the touched files (radon mi)
  dup_lines                     added lines that repeat an existing line of the
                                workspace, ignoring blank lines, comments, imports,
                                and lines shorter than 30 characters

radon runs through `uvx radon` (no install in the session environment).
"""

import json
import re
import subprocess
import sys
from pathlib import Path

SKIP = re.compile(r"(^|/)(results|work_dirs|logs?|outputs?|checkpoints?|\.venv|\.cache|__pycache__|tmp)(/|$)")


def git(ws: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(ws), *args], capture_output=True, text=True,
                          env={"GIT_CONFIG_GLOBAL": "/dev/null", "PATH": "/usr/bin:/bin"}).stdout


def radon(*args: str) -> dict:
    out = subprocess.run(["uvx", "--quiet", "radon", *args, "-j"], capture_output=True, text=True).stdout
    try:
        return json.loads(out or "{}")
    except json.JSONDecodeError:
        return {}


def added_lines(ws: Path, base: str, path: str) -> set[int]:
    """Line numbers (in the new file) that the session added."""
    lines, cur = set(), 0
    for line in git(ws, "diff", "--unified=0", base, "--", path).splitlines():
        m = re.match(r"@@ -\S+ \+(\d+)(?:,(\d+))? @@", line)
        if m:
            start, n = int(m.group(1)), int(m.group(2) or 1)
            lines.update(range(start, start + n))
    return lines


def norm(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip())


def measure(run: Path) -> dict:
    ws = run / "workspace"
    git(ws, "add", "-A")  # include untracked files in the comparison; the index is reset below
    try:
        base = git(ws, "rev-list", "--max-parents=0", "HEAD").split()[0]
        status = [l.split("\t") for l in git(ws, "diff", "--cached", "--name-status", base).splitlines()]
        numstat = [l.split("\t") for l in git(ws, "diff", "--cached", "--numstat", base).splitlines()]
    finally:
        git(ws, "reset", "-q")
    touched = {p[-1]: p[0][0] for p in status
               if p[-1].endswith(".py") and p[0][0] in "AM" and not SKIP.search(p[-1])}
    res = {"files_added": sum(v == "A" for v in touched.values()),
           "files_modified": sum(v == "M" for v in touched.values()),
           "lines_added": 0, "lines_removed": 0}
    for a, r, path in (n for n in numstat if len(n) == 3):
        if path in touched and a != "-":
            res["lines_added"] += int(a)
            res["lines_removed"] += int(r)
    paths = [str(ws / p) for p in touched if (ws / p).exists()]
    cc, mi = (radon("cc", "-s", *paths), radon("mi", *paths)) if paths else ({}, {})
    complexities = []
    for p in touched:
        new = added_lines(ws, base, p) if touched[p] == "M" else None
        for block in cc.get(str(ws / p), []) if isinstance(cc.get(str(ws / p)), list) else []:
            span = set(range(block.get("lineno", 0), block.get("endline", 0) + 1))
            if new is None or span & new:
                complexities.append(block.get("complexity", 0))
    mis = [v.get("mi") for v in mi.values() if isinstance(v, dict) and v.get("mi") is not None]
    res.update(functions=len(complexities),
               cc_mean=round(sum(complexities) / len(complexities), 2) if complexities else None,
               cc_max=max(complexities) if complexities else None,
               cc_over_10=sum(c > 10 for c in complexities),
               mi_mean=round(sum(mis) / len(mis), 1) if mis else None)
    # duplication: added lines that already occur elsewhere in the workspace's Python code
    existing: dict[str, int] = {}
    for f in ws.rglob("*.py"):
        rel = str(f.relative_to(ws))
        if SKIP.search(rel):
            continue
        for line in f.read_text(errors="ignore").splitlines():
            k = norm(line)
            if len(k) >= 30 and not k.startswith(("#", "import ", "from ")):
                existing[k] = existing.get(k, 0) + 1
    dup = 0
    for p, kind in touched.items():
        text = (ws / p).read_text(errors="ignore").splitlines() if (ws / p).exists() else []
        new = set(range(1, len(text) + 1)) if kind == "A" else added_lines(ws, base, p)
        for i in new:
            k = norm(text[i - 1]) if i - 1 < len(text) else ""
            if len(k) >= 30 and not k.startswith(("#", "import ", "from ")) and existing.get(k, 0) > 1:
                dup += 1
    res["dup_lines"] = dup
    return res


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        run = Path(arg)
        if not (run / "workspace" / ".git").exists():
            print(f"skip {run}: no workspace repository")
            continue
        q = measure(run)
        (run / "quality.json").write_text(json.dumps(q, indent=1) + "\n")
        print(run.name, q)
