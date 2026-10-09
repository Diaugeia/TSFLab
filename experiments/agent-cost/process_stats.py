"""Process measures of a session: process_stats.py <run-dir> [...]. Writes <run-dir>/process.json.

For Codex (exec --json) and Claude Code (stream-json) streams alike:
  calls         tool calls (commands, file edits, web searches, tool uses)
  failed_calls  calls that failed: a non-zero exit code, a Python traceback in the output,
                or a tool result marked as an error
  web_calls     web searches and fetches: Codex web_search items, WebSearch/WebFetch tools,
                and commands that fetch from the web (curl, wget, git clone, ...; analyze.WEB_CMD),
                including fetches of the target paper
  read_calls    calls classified as read or paper by analyze.classify (finding and reading)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import PAPER_HINT, WEB_CMD, classify, pi_tool, strip_cmd, unwrap_shell  # noqa: E402


def events(run: Path):
    for p in sorted(run.glob("stream.*.jsonl")):
        for line in p.read_text(errors="ignore").splitlines():
            try:
                yield json.loads(line)["d"]
            except (ValueError, KeyError, TypeError):
                continue


def is_web_cmd(cmd: str) -> bool:
    c = strip_cmd(cmd)
    return bool(WEB_CMD.match(c)) or (bool(PAPER_HINT.search(c)) and c.startswith(("curl", "wget")))


def measure(run: Path) -> dict:
    out = {"calls": 0, "failed_calls": 0, "web_calls": 0, "read_calls": 0}
    pending = {}  # Claude tool_use id -> (name, input)
    for d in events(run):
        typ = d.get("type")
        if typ == "item.completed":  # Codex
            it = d.get("item") or {}
            kind = it.get("type")
            if kind == "command_execution":
                cmd = unwrap_shell(str(it.get("command", "")))
                out["calls"] += 1
                text = it.get("aggregated_output", "") or ""
                code = it.get("exit_code")
                if (code not in (0, None)) or "Traceback (most recent call last)" in text:
                    out["failed_calls"] += 1
                if is_web_cmd(cmd):
                    out["web_calls"] += 1
                if classify("Bash", {"command": cmd}) in ("read", "paper"):
                    out["read_calls"] += 1
            elif kind in ("file_change", "mcp_tool_call"):
                out["calls"] += 1
                if it.get("status") == "failed":
                    out["failed_calls"] += 1
            elif kind == "web_search":
                out["calls"] += 1
                out["web_calls"] += 1
                out["read_calls"] += 1
        elif typ == "tool_execution_end":  # pi
            name, args = pi_tool(d)
            out["calls"] += 1
            res = d.get("result") if isinstance(d.get("result"), dict) else {}
            text = json.dumps(res)
            code = (res.get("structuredContent") or {}).get("exit_code") if isinstance(res.get("structuredContent"), dict) else None
            if d.get("isError") or (code not in (0, None)) or "Traceback (most recent call last)" in text:
                out["failed_calls"] += 1
            cmd = unwrap_shell(str(args.get("command", ""))) if name == "Bash" else ""
            if name == "Bash" and is_web_cmd(cmd):
                out["web_calls"] += 1
            if classify(name, {"command": cmd} if name == "Bash" else args) in ("read", "paper"):
                out["read_calls"] += 1
        elif typ == "assistant":  # Claude Code
            for b in (d.get("message") or {}).get("content") or []:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    name, inp = b.get("name", ""), b.get("input") or {}
                    pending[b.get("id")] = (name, inp)
                    out["calls"] += 1
                    if name in ("WebSearch", "WebFetch") or (name == "Bash" and is_web_cmd(str(inp.get("command", "")))):
                        out["web_calls"] += 1
                    if classify(name, inp) in ("read", "paper"):
                        out["read_calls"] += 1
        elif typ == "user":
            for b in (d.get("message") or {}).get("content") or []:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    content = b.get("content")
                    text = json.dumps(content) if not isinstance(content, str) else content
                    if b.get("is_error") or "Traceback (most recent call last)" in text:
                        out["failed_calls"] += 1
    return out


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        run = Path(arg)
        if not list(run.glob("stream.*.jsonl")):
            continue
        m = measure(run)
        (run / "process.json").write_text(json.dumps(m) + "\n")
        print(run.name, m)
