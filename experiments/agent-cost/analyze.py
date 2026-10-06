"""Compute the metrics of one agent-cost session: analyze.py <run-dir>.

Inputs in <run-dir>: stream.<n>.jsonl (stamped Claude Code stream-json),
stages.tsv, interventions.tsv, gpu.csv, diff.numstat, success.json, meta.json.

Every tool call is classified by action type:
  paper   reading or fetching the target paper (needed in every environment)
  read    reading or searching the environment: knowledge, code, documentation
  write   writing or editing files
  run     executing code (installs, training, scripts, git, ...)
  other   bookkeeping (todo lists, stage markers, subagent launches)
Discovery cost is the time and tokens of `read` calls. Tokens of a model turn
are split equally over the tool calls it makes; a turn without tool calls
counts as `reason`. Time is split into model time (from the previous event to
the turn) and tool time (from the call to its result).
"""

import csv
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

READ_TOOLS = {"Read", "Grep", "Glob", "LS", "NotebookRead", "WebSearch", "WebFetch"}
WRITE_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit"}
OTHER_TOOLS = {"TodoWrite", "Task", "Agent", "ToolSearch", "Skill"}
PAPER_HINT = re.compile(r"arxiv\.org|openreview\.net|\.pdf\b|pdftotext|paper\.(md|txt|pdf)", re.I)
READ_CMD = re.compile(
    r"^(cat|head|tail|less|more|ls|tree|find|fd|grep|rg|ag|wc|file|stat|du|"
    r"sed\s+-n|awk|jq|diff|"
    r"git\s+(log|show|diff|status|grep|ls-files|blame)|"
    r"(uv\s+run\s+)?tsf\s+(catalog|model\s+(show|issues|similar|list)|component\s+(show|list)|dataset\s+(show|list)|data\s+(show|list))|"
    r"(uv\s+run\s+)?python3?\s+-c\s+['\"]\s*(import\s+\w+\s*;\s*)*print\(\s*open)"
)
WRITE_CMD = re.compile(r"(^|\s)(cat|tee|echo|printf)\b[^|]*(>|<<)|^\s*(cp|mv|mkdir|touch|rm)\b")


def strip_cmd(cmd: str) -> str:
    """Remove leading `cd x &&`, env assignments, and `timeout N` from a shell command."""
    cmd = cmd.strip()
    while True:
        new = re.sub(r"^(cd\s+\S+\s*(&&|;)\s*|[A-Z_][A-Z0-9_]*=\S+\s+|timeout\s+\S+\s+|nice\s+(-n\s*\d+\s+)?)", "", cmd)
        if new == cmd:
            return cmd
        cmd = new


def classify(name: str, inp: dict) -> str:
    text = json.dumps(inp)
    if name in READ_TOOLS:
        return "paper" if PAPER_HINT.search(text) else "read"
    if name in WRITE_TOOLS:
        return "write"
    if name in OTHER_TOOLS:
        return "other"
    if name == "Bash":
        cmd = strip_cmd(str(inp.get("command", "")))
        if re.match(r"^stage\b", cmd):
            return "other"
        if PAPER_HINT.search(cmd) and re.match(r"^(curl|wget|pdftotext|python3?|uv)", cmd):
            return "paper"
        if WRITE_CMD.search(cmd):
            return "write"
        if READ_CMD.match(cmd):
            return "read"
        return "run"
    return "other"


def load_stream(run: Path) -> list[tuple[float, dict]]:
    events = []
    for p in sorted(run.glob("stream.*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                events.append((e["t"], e["d"]))
    events.sort(key=lambda x: x[0])
    return events


def parse_ts(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def stage_of(t: float, stages: list[tuple[float, str]]) -> str:
    name = "before-first-stage"
    for ts, n in stages:
        if ts <= t:
            name = n
    return name


CODEX_EVENTS = {"thread.started", "turn.started", "turn.completed", "turn.failed", "item.started", "item.completed"}
LUNA = {"input": 0.10, "cache_read": 0.01, "cache_write": 0.125, "output": 0.50}


def unwrap_shell(cmd: str) -> str:
    """Codex reports commands as `/bin/bash -lc '<cmd>'`; return <cmd>."""
    m = re.match(r"^\S*(ba|z)?sh\s+-l?c\s+(['\"])(.*)\2\s*$", cmd.strip(), re.S)
    return m.group(3) if m else cmd


def analyze_codex(events, stages):
    """Token, time, and call counts of a Codex `exec --json` stream.

    Codex reports token usage per user turn, not per model call, so tokens are
    attributed to actions by the size of what each action brought into the
    context (command output, file edits, messages): read calls' share of that
    content is the discovery share.
    """
    usage = {"input": 0, "cache_read": 0, "cache_write": 0, "output": 0}
    calls, secs, chars = defaultdict(int), defaultdict(float), defaultdict(float)
    stage_chars = defaultdict(float)
    started: dict[str, float] = {}
    steps = 0
    for t, d in events:
        typ = d.get("type")
        if typ == "turn.completed":
            u = d.get("usage") or {}
            cached = u.get("cached_input_tokens", 0) or 0
            cw = u.get("cache_write_input_tokens", 0) or 0
            usage["input"] += max(0, (u.get("input_tokens", 0) or 0) - cached - cw)
            usage["cache_read"] += cached
            usage["cache_write"] += cw
            usage["output"] += u.get("output_tokens", 0) or 0
        elif typ == "item.started":
            started[(d.get("item") or {}).get("id", "")] = t
        elif typ == "item.completed":
            it = d.get("item") or {}
            kind = it.get("type")
            if kind == "command_execution":
                cmd = unwrap_shell(str(it.get("command", "")))
                c = classify("Bash", {"command": cmd})
                n = len(cmd) + len(it.get("aggregated_output", "") or "")
            elif kind == "file_change":
                c, n = "write", len(json.dumps(it.get("changes", [])))
            elif kind == "web_search":
                q = json.dumps(it)
                c, n = ("paper" if PAPER_HINT.search(q) else "other"), len(q)
            elif kind in ("agent_message", "reasoning"):
                c, n = "reason", len(it.get("text", "") or "")
            elif kind in ("mcp_tool_call", "todo_list"):
                c, n = "other", len(json.dumps(it))
            else:
                continue
            if c != "reason":
                calls[c] += 1
                steps += 1
                secs[c] += t - started.pop(it.get("id", ""), t)
            chars[c] += n
            stage_chars[stage_of(t, stages)] += n
    total = sum(usage.values())
    all_chars = sum(chars.values()) or 1
    tokens = {c: {k: v * n / all_chars for k, v in usage.items()} for c, n in chars.items()}
    luna = sum(usage[k] * LUNA[k] for k in usage) / 1e6
    stage_tokens = {s: total * n / all_chars for s, n in stage_chars.items()}
    return tokens, secs, calls, luna, steps, stage_tokens, usage


def main() -> None:
    run = Path(sys.argv[1])
    events = load_stream(run)
    stages = []
    if (run / "stages.tsv").exists():
        for row in csv.reader((run / "stages.tsv").open(), delimiter="\t"):
            if len(row) == 2:
                stages.append((parse_ts(row[0]), row[1]))

    is_codex = any(d.get("type") in CODEX_EVENTS for _, d in events)
    tokens = defaultdict(lambda: {"input": 0, "cache_read": 0, "cache_write": 0, "output": 0})
    secs = defaultdict(float)
    stage_tokens = defaultdict(float)
    calls = defaultdict(int)
    tool_start: dict[str, tuple[float, str]] = {}
    cost = 0.0
    turns = 0
    asks = 0
    out_total = 0
    results = []

    # Group the stream lines of one model turn (same message id).
    msgs: dict[str, dict] = {}
    order: list[str] = []
    prev_t = events[0][0] if events else 0.0
    for t, d in events:
        typ = d.get("type")
        if typ == "assistant":
            m = d.get("message", {})
            mid = m.get("id") or f"anon-{t}"
            if mid not in msgs:
                msgs[mid] = {"t0": prev_t, "t1": t, "usage": m.get("usage") or {}, "cats": [], "chars": 0,
                             "chain": d.get("parent_tool_use_id") or "main"}
                order.append(mid)
            g = msgs[mid]
            g["t1"] = t
            for b in m.get("content", []) or []:
                bt = b.get("type")
                if bt == "tool_use":
                    c = classify(b.get("name", ""), b.get("input", {}))
                    g["cats"].append(c)
                    tool_start[b["id"]] = (t, c)
                    calls[c] += 1
                    if b.get("name") == "AskUserQuestion":
                        asks += 1
                    g["chars"] += len(json.dumps(b.get("input", {})))
                else:
                    g["chars"] += len(b.get("text", "") or b.get("thinking", "") or "")
        elif typ == "user":
            for b in (d.get("message", {}) or {}).get("content", []) or []:
                if isinstance(b, dict) and b.get("type") == "tool_result" and b.get("tool_use_id") in tool_start:
                    t0, c = tool_start.pop(b["tool_use_id"])
                    secs[c] += t - t0
        elif typ == "result":
            cost += d.get("total_cost_usd", 0) or 0
            turns += d.get("num_turns", 0) or 0
            out_total += (d.get("usage") or {}).get("output_tokens", 0) or 0
            results.append({k: d.get(k) for k in ("subtype", "is_error", "duration_ms", "num_turns")})
        prev_t = t

    # Uniform pricing (GPT-6 Luna list prices, USD per 1M tokens). Model-reported cache
    # fields differ between providers, so caching is estimated the same way for every
    # model: within one conversation chain, the part of a turn's input that repeats the
    # previous turn's input and output is a cache read; the rest is a cache write.
    chars = sum(g["chars"] for g in msgs.values()) or 1
    scale = out_total / chars if out_total else 0.25
    prev_ctx: dict[str, float] = {}
    luna = 0.0
    for mid in order:
        g = msgs[mid]
        cats = g["cats"] or ["reason"]
        u = g["usage"]
        ctx = sum((u.get(k, 0) or 0) for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
        cached = min(ctx, prev_ctx.get(g["chain"], 0.0))
        out_t = g["chars"] * scale
        prev_ctx[g["chain"]] = ctx + out_t
        luna += (cached * LUNA["cache_read"] + (ctx - cached) * LUNA["cache_write"] + out_t * LUNA["output"]) / 1e6
        parts = {
            "input": u.get("input_tokens", 0) or 0,
            "cache_read": u.get("cache_read_input_tokens", 0) or 0,
            "cache_write": u.get("cache_creation_input_tokens", 0) or 0,
            "output": g["chars"] * scale,
        }
        for c in cats:
            secs[c] += (g["t1"] - g["t0"]) / len(cats)
            for k, v in parts.items():
                tokens[c][k] += v / len(cats)
        stage_tokens[stage_of(g["t1"], stages)] += sum(parts.values())

    usage_total = None
    if is_codex:
        tokens, secs, calls, luna, turns, stage_tokens, usage_total = analyze_codex(events, stages)
        tokens = defaultdict(dict, tokens)

    wall = (events[-1][0] - events[0][0]) if events else 0.0
    total_tok = {c: sum(v.values()) for c, v in tokens.items()}
    all_tok = sum(total_tok.values()) or 1

    stage_secs = {}
    if stages:
        end = events[-1][0] if events else stages[-1][0]
        bounds = stages + [(end, "end")]
        for (t0, n), (t1, _) in zip(bounds, bounds[1:]):
            stage_secs[n] = stage_secs.get(n, 0.0) + max(0.0, t1 - t0)

    gpu_busy = 0.0
    gpu_rows = list(csv.reader((run / "gpu.csv").open())) if (run / "gpu.csv").exists() else []
    if len(gpu_rows) > 1:
        meta0 = json.loads((run / "meta.json").read_text()) if (run / "meta.json").exists() else {}
        interval = float(meta0.get("gpu_sample_s", 5))
        for r in gpu_rows:
            try:
                if float(r[1].strip().rstrip("%").strip()) >= 10:
                    gpu_busy += interval
            except (IndexError, ValueError):
                pass

    code = defaultdict(lambda: {"files": 0, "added": 0, "removed": 0})
    if (run / "diff.numstat").exists():
        for line in (run / "diff.numstat").read_text().splitlines():
            parts = line.split("\t")
            if len(parts) != 3 or parts[0] == "-":
                continue
            a, r, path = int(parts[0]), int(parts[1]), parts[2]
            if path.startswith(("results/", ".venv/")):
                kind = "results"
            elif path.endswith((".py", ".sh", ".ipynb")):
                kind = "code"
            elif path.endswith((".toml", ".yaml", ".yml", ".json", ".cfg", ".ini")):
                kind = "config"
            elif path.endswith((".md", ".txt", ".rst")):
                kind = "docs"
            else:
                kind = "other"
            code[kind]["files"] += 1
            code[kind]["added"] += a
            code[kind]["removed"] += r

    interventions = 0
    if (run / "interventions.tsv").exists():
        interventions = sum(1 for l in (run / "interventions.tsv").read_text().splitlines() if l.strip())

    success = json.loads((run / "success.json").read_text()) if (run / "success.json").exists() else None
    meta = json.loads((run / "meta.json").read_text()) if (run / "meta.json").exists() else {}

    # Commands that point CUDA_VISIBLE_DEVICES at another GPU than the assigned one.
    own = str(meta.get("train_gpu", ""))
    overrides = 0
    for _, d in events:
        txt = json.dumps(d)
        for m in re.finditer(r"CUDA_VISIBLE_DEVICES=([0-9,]+)", txt):
            if m.group(1) != own:
                overrides += 1
    out = {
        "meta": meta,
        "gpu_overrides": overrides,
        "success": success,
        "wall_seconds": wall,
        "cost_usd_reported": cost,
        "cost_usd_luna": luna,
        "usage_total": usage_total,
        "turns": turns,
        "interventions": interventions,
        "agent_questions": asks,
        "tool_calls": dict(calls),
        "tokens_by_action": {c: {k: round(v) for k, v in t.items()} for c, t in tokens.items()},
        "token_share_by_action": {c: v / all_tok for c, v in total_tok.items()},
        "seconds_by_action": dict(secs),
        "discovery": {
            "tokens": round(total_tok.get("read", 0)),
            "token_share": total_tok.get("read", 0) / all_tok,
            "seconds": secs.get("read", 0.0),
            "calls": calls.get("read", 0),
        },
        "stage_seconds": stage_secs,
        "stage_tokens": {k: round(v) for k, v in stage_tokens.items()},
        "gpu_busy_seconds": gpu_busy,
        "code_changes": dict(code),
        "sessions": results,
        "discovery_errors": None,  # filled by review of diff.patch (see README)
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
