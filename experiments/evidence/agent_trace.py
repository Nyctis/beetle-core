"""Turns a `claude -p --output-format stream-json` log into the published trace.

One JSON line per event: the agent's words, each tool call and each result, in order,
with times. Nothing is summarised and no call is dropped; the prompt is kept too,
because a trace of an agent means little without what it was asked.

    python agent_trace.py results/agent/NAME.raw.jsonl --prompt PROMPT --out traces/NAME.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scrub import scrub, scrub_value

MCP = "mcp__beetle__"


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(c.get("text", "") for c in content if isinstance(c, dict))


def events(lines: list[str]):
    names: dict[str, str] = {}
    for line in lines:
        msg = json.loads(line)
        kind = msg.get("type")
        if kind == "system" and msg.get("subtype") == "init":
            yield {"event": "start", "model": msg.get("model"), "claude_code": msg.get("claude_code_version"),
                   "tools": msg.get("tools"), "mcp_servers": msg.get("mcp_servers"),
                   "permission_mode": msg.get("permissionMode")}
        elif kind == "assistant":
            for part in msg["message"]["content"]:
                if part.get("type") == "text" and part["text"].strip():
                    yield {"event": "agent_text", "text": part["text"]}
                elif part.get("type") == "tool_use":
                    names[part["id"]] = part["name"].removeprefix(MCP)
                    yield {"event": "tool_call", "tool": names[part["id"]], "input": part["input"]}
        elif kind == "user":
            content = msg["message"].get("content")
            for part in content if isinstance(content, list) else []:
                if part.get("type") == "tool_result":
                    body = text_of(part.get("content"))
                    try:
                        body = json.loads(body)
                    except ValueError:
                        pass
                    yield {"event": "tool_result", "tool": names.get(part.get("tool_use_id")),
                           "is_error": bool(part.get("is_error")), "output": body}
        elif kind == "result":
            yield {"event": "end", "subtype": msg.get("subtype"), "is_error": msg.get("is_error"),
                   "num_turns": msg.get("num_turns"), "duration_ms": msg.get("duration_ms"),
                   "duration_api_ms": msg.get("duration_api_ms"), "total_cost_usd": msg.get("total_cost_usd"),
                   "result": msg.get("result"), "stop_reason": msg.get("stop_reason"),
                   "permission_denials": msg.get("permission_denials")}


def trace_rows(raw: Path, prompt: Path) -> list[dict]:
    return [{"event": "prompt", "text": prompt.read_text()},
            *events([ln for ln in raw.read_text().splitlines() if ln.strip()])]


def write_trace(raw: Path, prompt: Path, out: Path) -> list[dict]:
    rows = trace_rows(raw, prompt)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for i, row in enumerate(rows):
            f.write(json.dumps({"n": i, **scrub_value(row)}, ensure_ascii=False) + "\n")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw", type=Path)
    ap.add_argument("--prompt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = write_trace(a.raw, a.prompt, a.out)
    calls = [r["tool"] for r in rows if r["event"] == "tool_call"]
    print(scrub(json.dumps({"events": len(rows), "calls": calls}, ensure_ascii=False)))


if __name__ == "__main__":
    main()
