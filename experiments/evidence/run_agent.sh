#!/bin/bash
# One headless Claude Code run as a real MCP client of a local `beetle serve`.
#
#   BEETLE_HOME=<a funded home> experiments/evidence/run_agent.sh NAME PROMPT_FILE MAX_TURNS MAX_USD
#
# Claude Code gets Beetle's tools and nothing else: no shell, no files, no other MCP
# server. It starts in an empty directory, so no project instructions reach it. The
# token goes through the environment and is expanded by the MCP config at launch.
# The raw stream lands in results/agent/, which git ignores; agent_trace.py makes the
# published trace from it.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
NAME=$1 PROMPT=$2 TURNS=$3 BUDGET=$4
: "${BEETLE_HOME:?set BEETLE_HOME to a home with a running daemon}"
export BEETLE_HOME
OUT=$HERE/results/agent
WORK=$(mktemp -d)
mkdir -p "$OUT"
export BEETLE_TOKEN
BEETLE_TOKEN=$("$ROOT/.venv/bin/beetle" token "$NAME")
cat > "$WORK/mcp.json" <<JSON
{"mcpServers": {"beetle": {"type": "stdio", "command": "$ROOT/.venv/bin/python", "args": ["-m", "client.mcp_server"],
 "env": {"BEETLE_HOME": "\${BEETLE_HOME}", "BEETLE_TOKEN": "\${BEETLE_TOKEN}"}}}}
JSON
cd "$WORK" || exit 1
START=$(date +%s)
claude -p "$(cat "$PROMPT")" --mcp-config "$WORK/mcp.json" --strict-mcp-config --tools "" \
    --allowedTools "mcp__beetle__*" --permission-mode dontAsk --max-turns "$TURNS" --max-budget-usd "$BUDGET" \
    --output-format stream-json --verbose --no-session-persistence --model "${AGENT_MODEL:-sonnet}" \
    > "$OUT/$NAME.raw.jsonl" 2> "$OUT/$NAME.err"
echo "exit=$? seconds=$(( $(date +%s) - START ))"
rm -f "$WORK/mcp.json"
"$ROOT/.venv/bin/python" "$HERE/agent_trace.py" "$OUT/$NAME.raw.jsonl" --prompt "$PROMPT" --out "$HERE/traces/$NAME.jsonl"
