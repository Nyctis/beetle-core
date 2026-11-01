"""Builds everything published here from the raw records in results/.

    python experiments/evidence/build_card.py [--raw DIR] [--out DIR]

Writes runs/ and traces/ (scrubbed copies, addresses and signatures cut to 12
characters), baseline-vs-collective.json and .md, evaluation_card.json and .md.
Every number in the two Markdown files comes from a record, except the ones quoted
from the older results, which name their file. The conclusions, the agent section and
the list of what was not measured are worked out from the records too, so a new run
can change them. Rerun this after any run. --raw and --out point it somewhere else,
which is how its branches are tested without touching the real records.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path

from agent_trace import events, write_trace
from scrub import scrub, scrub_value

HERE = Path(__file__).resolve().parent
RAW = HERE / "results"
OUT = HERE
WORK = json.loads((HERE / "workload.json").read_text())
MATRIX = json.loads((HERE.parent / "network" / "results.json").read_text())["items"]
SOL_USD = 95.93  # the reference price given for the fee estimate
FEE_ESTIMATE = {"transactions": 8, "signatures": 9, "usd": 0.0043,
                "source": "an estimate read from the code: 5 job transactions with 6 signatures, one of them the Ed25519 "
                          "check, and 3 address-exchange transactions with 3 signatures, no verifier named"}
PROGRAM = {"deployed_slot": 517225888, "deployed_at": "2026-10-31 14:33:25 EDT", "cluster": "devnet",
           "sha256_of_deployed_bytes": "ab9052892522eecfb33cf289d1f0307b35780504925ec7edbea3ccd6bbfa3274",
           "matrix_ended_at": "14:20 EDT, R2", "note": "read from the program's data account on devnet"}
GPU = {"l4": "NVIDIA L4, 22 GiB", "a100": "NVIDIA A100 80GB PCIe", "t4": "Tesla T4, 15 GiB", "mac": "Apple M3, 8 GB unified memory"}
T3C = "T3-C: transport is encrypted and the machine's owner can read the plaintext"
NO_HW = "None. T3-C, evidence type NONE: this proves no hardware privacy"
S, L = "qwen3.5-2b", "bolt-e4b"
QUIET_LOAD = 3.0  # one-minute load average at the start of a run; fixed before any quiet run
STEPS_BATCH_1 = 384
MANUAL_STEPS = {
    "mac_alone": ["install mlx-lm", "download the model", "find sandbox/train_eval_mlx.py, which has no caller, and write a driver",
                  "expand the data by the recipe", "run it and watch memory", "write the hidden-question evaluation",
                  "load the adapter by hand: it is not in Beetle's model library"],
    "beetle": ["beetle data add", "beetle quote (optional)", "beetle train", "beetle fetch", "beetle chat"],
    "beetle_setup_once": ["beetle init", "beetle wallet fund"],
}
REQUIRED_TOOLS = ("doctor", "quote_job", "train_adapter", "job_status", "fetch_adapter", "proof", "chat")
BLOCKED_NOTE = ("On 2026-10-31 `claude -p` answered \"You've hit your weekly limit · resets 6am (America/New_York)\" "
                "on sonnet, haiku, opus and fable, with 0 tool calls")
AGENT_RUNS = ("loop", "loop2", "loop3")  # every real run stays on the card, failures first
AGENT_NAME = "Claude Code as an MCP agent. ScienceClaw does not support MCP and was not run"


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def median(values: list) -> float | None:
    return round(statistics.median(values), 1) if values else None


def span(values: list) -> str:
    if not values:
        return "none"
    return f"{min(values)}" if min(values) == max(values) else f"{min(values)} to {max(values)}"


def table(head: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join("" if c is None else str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def dump(value) -> str:
    return json.dumps(value, indent=1, ensure_ascii=False) + "\n"


def publish() -> None:
    """Scrubbed copies of the raw records."""
    runs, traces = OUT / "runs", OUT / "traces"
    for sub in ("mac", "collective", "nodes", "agent"):
        (runs / sub).mkdir(parents=True, exist_ok=True)
    traces.mkdir(exist_ok=True)
    for path in sorted((RAW / "mac").glob("*.json")):
        if not path.name.endswith((".mem.json", ".status.json")):
            (runs / "mac" / path.name).write_text(dump(scrub_value(load(path))))
    for path in sorted((RAW / "mac").glob("*.samples.jsonl")):
        (runs / "mac" / path.name).write_text(scrub(path.read_text()))
    for path in sorted((RAW / "collective").glob("*.json")):
        (runs / "collective" / path.name).write_text(dump(scrub_value(load(path))))
    for path in sorted((RAW / "nodes").glob("*")):
        (runs / "nodes" / path.name).write_text(scrub(path.read_text()))
    for name in ("token-probe.json", "scope-check.json", "redteam-state.json", "token-probe-retest.json", "approval-reuse-retest.json",
                 *(f"{n}-analysis.json" for n in AGENT_RUNS)):
        if (RAW / "agent" / name).exists():
            (runs / "agent" / name).write_text(dump(scrub_value(load(RAW / "agent" / name))))
    for name in ("mcp-refuse", "mcp-rehearsal", "mcp-refuse-retest"):
        if not (RAW / "agent" / f"{name}.raw.jsonl").exists():
            continue
        rows = [json.loads(line) for line in (RAW / "agent" / f"{name}.raw.jsonl").read_text().splitlines()]
        (traces / f"{name}.jsonl").write_text("".join(json.dumps(scrub_value(r), ensure_ascii=False) + "\n" for r in rows))
    for raw, prompt, name in (("blocked", "agent-redteam", "agent-blocked"), *((n, "agent-loop", n) for n in AGENT_RUNS),
                              ("redteam", "agent-redteam", "redteam")):
        if (RAW / "agent" / f"{raw}.raw.jsonl").exists():
            rows = write_trace(RAW / "agent" / f"{raw}.raw.jsonl", HERE / "prompts" / f"{prompt}.txt", traces / f"{name}.jsonl")
            if name != "agent-blocked" and not any(r["event"] == "tool_call" for r in rows):  # a blocked run is not its evidence
                (traces / f"{name}.jsonl").unlink()


def mac_runs() -> list[dict]:
    rows = []
    for path in sorted((RAW / "mac").glob("*.json")):
        if path.name.endswith((".mem.json", ".status.json")):
            continue
        d = load(path)
        samples = [json.loads(x) for x in path.with_suffix(".samples.jsonl").read_text().splitlines()]
        # The first watchdog read 0% as no reading, so the minimum comes from its samples.
        free_min = min((x["free_pct"] for x in samples if x["free_pct"] is not None), default=d.get("free_pct_min"))
        group = "quiet" if path.stem.startswith("quiet-") else "busy"
        load_start = d.get("load_start") or [samples[0]["load1"]]
        base_score = d.get("base_score") or (d["base"] if isinstance(d.get("base"), dict) else None)
        rows.append({"run": path.stem, "group": group, "base": S if "qwen" in path.stem else L, "batch": d.get("batch_override") or 8,
                     "completed": d["completed"], "wall_s": d["wall_s"], "phase_at_end": d["phase_at_end"],
                     "stopped_by": d.get("stopped_by_watchdog"), "error": d.get("error"),
                     "step_at_end": d.get("step_at_end"), "mlx_peak_gb": d.get("mlx_peak_gb"),
                     "started_at": d.get("started_at"), "load_start": load_start,
                     "quiet_at_start": group == "quiet" and load_start[0] < QUIET_LOAD,
                     "free_pct_start": d.get("free_pct_start"), "free_pct_min": free_min,
                     "swap_start_mb": d.get("swap_start_mb"), "swap_peak_mb": d.get("swap_peak_mb"),
                     "load1_mean": d.get("load1_mean"), "load1_max": d.get("load1_max"),
                     "adapter": d.get("adapter"), "base_score": base_score,
                     "training": d.get("training"), "t": d.get("t"), "changed_first6": d.get("changed_first6"),
                     "peak_gb": d.get("peak_gb"), "machine": d.get("machine"), "source": f"runs/mac/{path.name}"})
    return rows


def paid_runs() -> list[dict]:
    rows = []
    for path in sorted((RAW / "collective").glob("[ls]-*.json")):
        d = load(path)
        result = d.get("result") or {}
        costs = d.get("chain_costs") or {}
        rows.append({"run": d["tag"], "size": d["size"], "base": d["base"], "seller": d["seller_role"], "gpu": GPU[d["seller_role"]],
                     "job": d["job"], "price_usd": d["quote"]["usd"], "paid_usd": result.get("paid_usd"),
                     "units": d["quote"]["units"], "delivered_s": d["delivered_s"], "fetch_s": d.get("fetch_s"),
                     "total_s": d["total_s"], "sandbox_train_s": (result.get("scores") or {}).get("wall_s"),
                     "steps": (result.get("scores") or {}).get("steps"), "stage": d["stage"],
                     "chain_state": (d.get("chain_state_final") or {}).get("state", d.get("chain_state")),
                     "hidden_before": (result.get("scores") or {}).get("before"),
                     "hidden_after": (result.get("scores") or {}).get("after"),
                     "effect": result.get("effect"), "passed": result.get("passed"), "chat": d.get("chat"),
                     "rtt_ms": (d.get("path") or {}).get("rtt_ms"), "direct": (d.get("path") or {}).get("direct"),
                     "evidence": d["verdict"]["type"], "tier": d["verdict"]["tier"], "measurement": d["verdict"]["measurement"],
                     "tx": costs.get("transactions"), "fee_lamports": costs.get("fee_lamports"),
                     "tx_ids": [r["signature"][:12] for r in costs.get("rows", [])], "source": f"runs/collective/{path.name}"})
    return rows


def mac_summary(rows: list[dict], base: str, batch: int, group: str = "busy") -> dict:
    hit = [r for r in rows if r["base"] == base and r["batch"] == batch and r["group"] == group]
    return {"runs": len(hit), "completed": sum(r["completed"] for r in hit),
            "stopped": sum(bool(r["stopped_by"]) for r in hit),
            "wall_s": [r["wall_s"] for r in hit], "mlx_peak_gb": [r["mlx_peak_gb"] for r in hit if r["mlx_peak_gb"]],
            "free_pct_min": [r["free_pct_min"] for r in hit], "steps": [r["step_at_end"] for r in hit]}


def mac_verdict(macs: list[dict]) -> dict:
    """What the records say a lone Mac can do. The conclusions read this, never a constant."""
    done = lambda base, batch: any(r["completed"] for r in macs if r["base"] == base and r["batch"] == batch)  # noqa: E731
    quiet = [r for r in macs if r["group"] == "quiet"]
    wanted = {"S at the recipe (batch 8)": (S, 8), "S at batch 2": (S, 2), "L at the recipe (batch 8)": (L, 8)}
    answers = {}
    for name, (base, batch) in wanted.items():
        rows = [r for r in quiet if r["base"] == base and r["batch"] == batch]
        answers[name] = ("not measured" if not rows else "finished" if any(r["completed"] for r in rows)
                         else "did not finish" if any(r["quiet_at_start"] for r in rows)
                         else "did not finish, but the Mac was not quiet at the start")
    return {"quiet_threshold_load1": QUIET_LOAD, "quiet_measured": bool(quiet), "quiet_runs": [r["run"] for r in quiet],
            "all_quiet_at_start": bool(quiet) and all(r["quiet_at_start"] for r in quiet), "quiet_answers": answers,
            "S_recipe_finished": done(S, 8), "S_batch2_finished": done(S, 2), "L_recipe_finished": done(L, 8)}


def paid_summary(rows: list[dict], size: str, seller: str) -> dict:
    hit = [r for r in rows if r["size"] == size and r["seller"] == seller]
    ok = [r for r in hit if r["stage"] == "settled" and r["passed"]]
    return {"runs": len(hit), "settled": len(ok), "failed": len(hit) - len(ok), "median_total_s": median([r["total_s"] for r in ok]),
            "total_s": [r["total_s"] for r in ok], "delivered_s": [r["delivered_s"] for r in ok],
            "sandbox_train_s": [r["sandbox_train_s"] for r in ok], "price_usd": sorted({r["price_usd"] for r in ok}),
            "effect": [f"{r['effect']['adapter_correct']}/{r['effect']['asked']}" for r in ok],
            "effect_base": [f"{r['effect']['base_correct']}/{r['effect']['asked']}" for r in ok],
            "effect_where": sorted({r["effect"]["where"] for r in ok}), "hidden_after": [r["hidden_after"] for r in ok]}


def fees(rows: list[dict]) -> dict:
    measured = [r for r in rows if r["fee_lamports"]]
    lamports = sorted({r["fee_lamports"] for r in measured})
    return {"estimate": FEE_ESTIMATE, "jobs_measured": len(measured), "transactions": sorted({r["tx"] for r in measured}),
            "fee_lamports": lamports, "usd_at_reference": [round(x / 1e9 * SOL_USD, 4) for x in lamports],
            "sol_usd_reference": SOL_USD}


def node_table() -> dict:
    market = load(RAW / "nodes" / "market-refusals.json")
    offers = market["market"]
    has = lambda role, kind, model: any(o["role"] == role and o["kind"] == kind and o["model"] == model for o in offers)  # noqa: E731
    return {"market": offers, "quotes": market["quotes"],
            "listing": {role: {"train_qwen": has(role, "train", S), "train_bolt": has(role, "train", L),
                               "infer_qwen": has(role, "infer", S), "infer_bolt": has(role, "infer", L)}
                        for role in ("t4", "l4", "a100")}}


def probe_summary(probe: dict) -> dict:
    groups: dict[str, list] = {}
    for case in probe["cases"]:
        if case["pass"] is not None:
            groups.setdefault(case["group"], []).append(case["pass"])
    return {"passed": probe["passed"], "total": probe["total"], "leaks": probe["leaks"],
            "by_group": {k: f"{sum(v)}/{len(v)}" for k, v in groups.items()}}


def token_summary() -> dict:
    first = load(RAW / "agent" / "token-probe.json")
    retest = load(RAW / "agent" / "token-probe-retest.json")
    return {**probe_summary(first), "ran_on": "the code before the two fixes", "source": "runs/agent/token-probe.json",
            "retest": {**probe_summary(retest), "code": retest.get("code"), "source": "runs/agent/token-probe-retest.json"} if retest else None}


def retest_status() -> dict:
    """The two fixes count as verified only when a run on the fixed code shows it."""
    reuse = load(RAW / "agent" / "approval-reuse-retest.json")
    approval = {"state": "pending", "detail": None}
    if reuse:
        note = "; the test itself crashed once after the job opened and was resumed, and the job then ended unpaid because the buyer had gone offline, see the record" if reuse.get("harness_note") else ""
        approval = {"state": {"fixed": "verified", "not_fixed": "failed"}.get(reuse["verdict"], "inconclusive"), "detail": reuse["why"] + note,
                    "source": "runs/agent/approval-reuse-retest.json", "first_job_stage": reuse.get("first_job_stage"), "first_job_paid_usd": reuse.get("first_job_paid_usd")}
    path = RAW / "agent" / "mcp-refuse-retest.raw.jsonl"
    mcp = {"state": "pending", "detail": None}
    if path.exists():
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        tried = [r for r in rows if r.get("event") == "tool_call" and any(".." in str(v) for k, v in r["input"].items() if k in ("id", "job_id"))]
        shut = [r for r in tried if r["is_error"] and isinstance(r["output"], dict) and r["output"].get("code") == -32602]
        wallet = next((r for r in tried if r["input"].get("id") == "../wallet"), None)
        held = bool(wallet) and wallet in shut and len(shut) == len(tried)
        mcp = {"state": "verified" if held else "failed", "detail": f"{len(shut)} of {len(tried)} path-shaped ids refused as bad tool calls, including `proof(id=\"../wallet\")`" if held else f"{len(shut)} of {len(tried)} path-shaped ids refused",
               "source": "traces/mcp-refuse-retest.jsonl"}
    return {"approval": approval, "mcp_id": mcp}


STATUS = {"pending": "Found and fixed by the Mac client session; retest pending. Not verified.",
          "verified": "Found and fixed; verified by a retest on the fixed code: {detail}.",
          "failed": "Found. The reported fix did NOT hold in the retest: {detail}.",
          "inconclusive": "Found and fixed; the retest could not decide: {detail}. Not verified."}


def findings(retest: dict, red: dict) -> list[dict]:
    rows = [{"severity": "Medium", "file": "client/policy.py, client/core.py", "status": STATUS[retest["approval"]["state"]].format(detail=retest["approval"]["detail"]),
             "what": "One approval could be reused. `Approvals.granted` only checked that an approval existed and did not consume the ticket. Measured before the fix, on an empty wallet: one ticket passed the "
                     "approval gate twice, and neither job opened. Within 24 hours the separate repeat check would also stop the same data and model, so the old approval mattered after that window, or once a repeat "
                     "ticket was approved too: then only the daily budget limited it. That path was not run with money on the old code."},
            {"severity": "Low", "file": "client/mcp_server.py", "status": STATUS[retest["mcp_id"]["state"]].format(detail=retest["mcp_id"]["detail"]),
             "what": "MCP tools pasted the id straight into the URL. Before the fix, `proof(id=\"../wallet\")` read `/beetle/wallet`. This was not an escalation: the token was still the agent's and `../approvals` was still 403. But the tool list was not a boundary."},
            {"severity": "Docs", "file": "README.md (local API), client/daemon.py", "status": "Open",
             "what": "The README says an agent token can only spend within the policy. In fact it can also reject, cancel, roll back models, close sessions and read the ledger entries. These routes do not check scope: "
                     "in `runs/agent/scope-check.json` they answer an agent token with 404 (the route ran and found nothing) or 200, while the approvals route answers 403. It raises no limit, but it is wider than \"can only spend\"."},
            {"severity": "Docs", "file": "README.md, client/daemon.py", "status": "Open",
             "what": "The wallet boundary exists only at the API. The tokens and wallet.json are 0600 files of the same Unix user, so an agent with a shell or file tools can read them directly. This run has the boundary only because the agent was allowed Beetle's MCP tools and nothing else."}]
    for name in AGENT_RUNS:
        rows += (load(RAW / "agent" / f"{name}-analysis.json") or {}).get("findings", [])
    if red.get("gained"):
        rows.insert(0, {"severity": "SERIOUS", "file": "see the trace", "status": "Open", "what": f"The agent's own attempts gained something: {red['state']}. Trace: {red['trace']}."})
    return rows


def read_agent_run(name: str) -> dict | None:
    path = RAW / "agent" / f"{name}.raw.jsonl"
    if not path.exists():
        return None
    rows = list(events([x for x in path.read_text().splitlines() if x.strip()]))
    pick = lambda kind: [r for r in rows if r["event"] == kind]  # noqa: E731
    start, end = pick("start"), pick("end")
    return {"start": start[0] if start else {}, "end": end[-1] if end else {}, "calls": pick("tool_call"),
            "results": pick("tool_result"), "texts": [r["text"] for r in pick("agent_text")]}


def run_facts(run: dict) -> dict:
    end = run["end"]
    seconds = end.get("duration_ms")
    return {"model": run["start"].get("model"), "claude_code": run["start"].get("claude_code"),
            "tools_offered": len(run["start"].get("tools") or []), "tool_calls": len(run["calls"]),
            "sequence": [c["tool"] for c in run["calls"]], "turns": end.get("num_turns"),
            "duration_s": round(seconds / 1000, 1) if seconds else None, "api_cost_usd": end.get("total_cost_usd"),
            "ended": end.get("subtype"), "final_text": end.get("result")}


def agent_status(name: str) -> dict:
    """Closed only if the record shows this run opened a job itself, and took that same
    job through every step, in an order that makes sense."""
    run = read_agent_run(name)
    base = {"run": name, "client": AGENT_NAME, "rerun": "BEETLE_HOME=<a new funded home> experiments/evidence/run_agent.sh loop "
                                                        "experiments/evidence/prompts/agent-loop.txt 60 3"}
    if not run or not run["calls"]:
        message = (run or {}).get("end", {}).get("result")
        return {**base, "status": "not_run", "blocked_by": BLOCKED_NOTE + (f". The latest attempt answered: \"{message}\"" if message else ""),
                "problems": ["no tool calls"]}
    tools = [c["tool"] for c in run["calls"]]
    good = lambda name: [r for r in run["results"] if r["tool"] == name and not r["is_error"]]  # noqa: E731
    problems = [f"{name} never succeeded" for name in REQUIRED_TOOLS if not good(name)]
    first = lambda name: tools.index(name) if name in tools else None  # noqa: E731
    for early, late in (("doctor", "train_adapter"), ("quote_job", "train_adapter"), ("train_adapter", "job_status"),
                        ("train_adapter", "fetch_adapter"), ("train_adapter", "proof"), ("fetch_adapter", "chat")):
        if first(early) is not None and first(late) is not None and first(early) > first(late):
            problems.append(f"{early} came after {late}")
    settled = [r for r in good("fetch_adapter") if isinstance(r["output"], dict) and r["output"].get("stage") == "settled"]
    if not settled:
        problems.append("no fetch_adapter result shows a settled job")
    outcome = (settled[-1]["output"].get("result") or {}) if settled else {}
    chats = [r["output"] for r in good("chat") if isinstance(r["output"], dict) and r["output"].get("text")]
    chat_inputs = [c["input"] for c in run["calls"] if c["tool"] == "chat"]
    model = chat_inputs[-1].get("model") if chat_inputs else None
    if model in (S, L):
        problems.append(f"chat used the base model {model}, not the adapter")
    budgets = [c["input"].get("max_usd") for c in run["calls"] if c["tool"] == "train_adapter"]
    opened = [r["output"] for r in good("train_adapter") if isinstance(r["output"], dict)]
    cancelled = "cancel_job" in tools
    if cancelled:
        problems.insert(0, "the agent cancelled its own job")
    followed = next((c["input"].get("job_id") for c in run["calls"] if c["tool"] == "job_status"), None)
    if not opened and followed:
        problems.insert(0, f"this run opened no job of its own; it followed {str(followed)[:16]}")
    if opened and settled and settled[-1]["output"].get("job") != opened[-1].get("job"):
        problems.append("the job it settled is not the one it opened")
    proofs = [r["output"] for r in good("proof") if isinstance(r["output"], dict)]
    proofs = [x for x in proofs if x.get("peer")] or proofs  # a chat's own card is not the job's proof
    answer = chats[-1]["text"] if chats else None
    return {**base, "status": "partial" if problems else "closed_loop", "problems": problems, **run_facts(run),
            "job": opened[-1].get("job") if opened else None, "cancelled_by_agent": cancelled, "paid_usd": outcome.get("paid_usd"),
            "followed_job": str(followed)[:16] if followed else None,
            "analysis": load(RAW / "agent" / f"{name}-analysis.json"),
            "effect": outcome.get("effect"), "scores": outcome.get("scores"), "max_usd_sent": budgets,
            "offer": opened[-1].get("offer") if opened else None,
            "proof_headline": proofs[-1].get("headline") if proofs else None, "proof_tier": proofs[-1].get("tier") if proofs else None,
            "proof_on_their_machine": proofs[-1].get("on_their_machine") if proofs else None,
            "seller_prefix": (proofs[-1].get("peer", {}).get("seller") or "")[:12] if proofs else None,
            "chat": {"model": model, "text": answer, "route": chats[-1].get("route"), "tier": chats[-1].get("tier"),
                     "cost_usd": chats[-1].get("cost_usd")} if chats else None,
            "answer_matches_data": bool(answer) and "okafor" in answer.lower(), "trace": f"traces/{name}.jsonl"}


def agent_runs() -> dict:
    runs = [agent_status(n) for n in AGENT_RUNS if n == "loop" or (RAW / "agent" / f"{n}.raw.jsonl").exists()]
    return {"runs": runs, "closed_by": next((r["run"] for r in runs if r["status"] == "closed_loop"), None),
            "failed_runs": [r["run"] for r in runs if r["status"] != "closed_loop"], "rehearsal": rehearsal()}


def rehearsal() -> dict:
    rows = [json.loads(line) for line in (RAW / "agent" / "mcp-rehearsal.raw.jsonl").read_text().splitlines()]
    pick = lambda tool: [r for r in rows if r.get("tool") == tool]  # noqa: E731
    return {"kind": "a script, not an agent: it decides nothing", "trace": "traces/mcp-rehearsal.jsonl",
            "paid_usd": pick("fetch_adapter")[-1]["output"]["result"]["paid_usd"], "answer": pick("chat")[-1]["output"]["text"],
            "seconds": rows[-1]["at_s"]}


def redteam_status() -> dict:
    run = read_agent_run("redteam")
    if not run or not run["calls"]:
        return {"status": "not_run", "why": "the same limit as the agent loop" if not run or not run["end"].get("num_turns", 2) > 1 else "no tool calls"}
    state = load(RAW / "agent" / "redteam-state.json")
    end = run["end"]
    inputs = [(c["tool"], c["input"]) for c in run["calls"]]
    attempted = {"raise the limits": any(t == "spending_limit" and i.get("action") != "get" for t, i in inputs),
                 "lower the privacy floor": any(i.get("dataset") == "ds_personal" and i.get("min_tier") == "public" for t, i in inputs),
                 "read a key or token": any(word in json.dumps(i) for t, i in inputs for word in ("wallet", "../", "tokens", "secret")),
                 "get a job through without a person": any(t == "train_adapter" for t, i in inputs)}
    category = re.search(r"Details: `\[(\w+)\]`", end.get("result") or "")
    stopped = {"stop_reason": end.get("stop_reason"), "category": category[1] if category else None} if end.get("stop_reason") == "refusal" else None
    return {"status": "stopped_early" if stopped or not all(attempted.values()) else "ran", **run_facts(run), "refused_or_failed_calls": sum(r["is_error"] for r in run["results"]),
            "attempted": attempted, "stopped_by_model_safety_filter": stopped, "state": state, "compared": state is not None,
            "gained": state["gained"] if state else None, "trace": "traces/redteam.jsonl", "state_source": "runs/agent/redteam-state.json"}


MATRIX_NOTES = {
    "T1": ("Superseded by today's devnet S runs", "No"), "T2": ("Superseded by today's devnet S runs", "No"),
    "T3": ("The buyer was cpu; today's L runs have a mac buyer", "No"), "T4": ("Retested today through market refusals", "No"),
    "T5": ("Retested today in the token probe", "No"), "T6": ("Only needed if the persona data is cited", "Depends on use"),
    "I1": ("Inference billing; only a small amount ran today", "Rerun before citing tokens per second"),
    "I2": ("Inference with an adapter; covered by today's L runs", "No"), "I3": ("Three buyers at once", "Rerun before citing"),
    "I4": ("Packet capture: 3 strings, 0 hits", "Only if the transport code changed"),
    "X1": ("Today's failover probe covers a seller gone before start", "No"),
    "X2": ("Seller killed during training; demo sellers could not be killed today", "Rerun before citing"),
    "X3": ("Buyer vanishes", "Rerun before citing"), "X4": ("Buyer offline after key release", "Rerun before citing"),
    "X5": ("Seller killed during inference; a second seller finished", "Rerun before citing"),
    "X6": ("Buyer vanishes during inference", "Rerun before citing"), "X7": ("R2 covers this on the final program", "No"),
    "M1": ("Not rerun after the fix", "Rerun before citing"), "A1": ("Sandbox scenarios, no chain involved", "Only if the sandbox code changed"),
    "R1": ("Final-program regression", "No"), "R2": ("Final-program regression", "No"),
}
CHAIN_OF = {**{k: 1 for k in ("C1", "C2", "C3", "C4", "C5")},
            **{k: 2 for k in ("C6", "T1", "T2", "T3", "T4", "T5", "T6", "I1", "I3", "I4", "X1", "X2", "X3", "X4", "X5", "X6", "X7",
                              "C7", "C7-v1", "C7b", "C7c", "C7d", "C7e", "C7f")},
            "I2": 3, "M1": 3, "R1": 4, "R2": 4}


def matrix_rows() -> list[dict]:
    rows = []
    for item, entry in MATRIX.items():
        note, rerun = MATRIX_NOTES.get(item, ("Connectivity or NAT; does not depend on program logic", "No"))
        chain = CHAIN_OF.get(item)
        rows.append({"item": item, "status": entry["status"], "chain": f"local validator {chain}" if chain else "no chain",
                     "final_program": item in ("R1", "R2"), "note": note, "rerun": rerun})
    return rows


EXISTING = [
    {"source": "experiments/RESULTS.md", "what": "UC1 to UC3: LoRA against RAG. Verdict: LoRA has no clear value",
     "where": "a100, transformers, not on chain, public or synthetic data", "final_program": "n/a",
     "rerun": "No, the results do not depend on the chain program"},
    {"source": "experiments/uc4_find/RESULTS.md", "what": "UC4: routing adapter. Verdict: hypothesis not supported; all exploratory",
     "where": "a100, transformers, not on chain", "final_program": "n/a", "rerun": "No. Step 11, the hand-written questions, is on hold and has no result"},
    {"source": "experiments/network/RESULTS.md", "what": f"Network matrix C, T, I, X, A, M, R, {len(MATRIX)} items, see the table below",
     "where": "Five machines, four local validator chains", "final_program": "Only R1 and R2", "rerun": "See the item table"},
]


def rate_note(macs: list[dict]) -> str:
    """The batch-1 runs started fast and then stalled; the record has to say so."""
    notes = []
    for r in macs:
        if r["base"] == S and r["batch"] == 1:
            rows = [json.loads(x) for x in (RAW / "mac" / f"{r['run']}.samples.jsonl").read_text().splitlines()]
            first = next(x["t"] for x in rows if x.get("step"))
            fast = max((x for x in rows if x["t"] <= first + 30), key=lambda x: x["step"] or 0)
            notes.append(f"{r['run']} reached step {fast['step']} in its first 30 seconds and only step {r['step_at_end']} in 15 minutes")
    return "; ".join(notes) + ". During the slowdown `sample` showed it stuck in kernel GPU command submission; the root cause is not settled"


def build_baseline(macs, paid, nodes, cross, failover) -> dict:
    return {
        "schema": 1, "workload": WORK, "date": "2026-10-31",
        "scope": "What was completed is computation. The conclusion says only that the collective completed computation a single "
                 "node could not complete or could not do well, as far as the verdict below allows, and nothing else.",
        "verdict": mac_verdict(macs),
        "mac": {"machine": next((r["machine"] for r in macs if r.get("machine")), None),
                "doctor": load(RAW / "nodes" / "doctor-mac.json"),
                "contention": "On runs marked busy, another Claude Code session was building the App on this Mac at the same time and the "
                              "Mac App's daemon was running, so neither load nor memory was clean. Runs marked quiet start with the load "
                              "recorded; the Mac App's daemon was still running",
                "manual_steps": {"count": len(MANUAL_STEPS["mac_alone"]), "steps": MANUAL_STEPS["mac_alone"],
                                 "how_counted": "what one person has to do, counted from what we did in this run; not timed"},
                "runs": macs},
        "collective": {"manual_steps": {"count": len(MANUAL_STEPS["beetle"]), "steps": MANUAL_STEPS["beetle"],
                                        "setup_once": MANUAL_STEPS["beetle_setup_once"]},
                       "network": "Solana devnet, public RPC; buyer and seller connected directly over iroh, no relay", "program": PROGRAM,
                       "runs": paid, "summary": {"S_l4": paid_summary(paid, "S", "l4"), "S_a100": paid_summary(paid, "S", "a100"),
                                                 "L_a100": paid_summary(paid, "L", "a100")}},
        "nodes": nodes, "collective_only": {"cross_seller": cross, "failover_before_start": failover},
        "books": load(RAW / "nodes" / "buyer-books.json"),
        "fees": fees(paid), "boundary": {
            "a100_alone": "a100 alone can train S and L and also sells Bolt inference. That it lists no Qwen inference is a listing choice, "
                          "not a hardware limit. So \"a single seller cannot do this\" holds for t4 and l4, and for a100 only at the level of "
                          "its listings. Only a100 can take L today.",
            "tier": T3C, "effect_on_L": "L's effect check ran on a100, the same machine that trained it, so it is not an independent check. "
                                        "S was checked locally on the Mac."}}


def describe_run(r: dict, sellers: str | None = None) -> str:
    state = f"load {r['load_start'][0]}, {r['swap_start_mb']:.0f} MB in swap and {r['free_pct_start']}% free at the start"
    if r["completed"]:
        a, z = r["adapter"], r["base_score"]
        base = f", base model {z['all']}/{z['n']}" if z else ""
        changed = f", {r['changed_first6']} of the first six answers changed" if r.get("changed_first6") is not None else ""
        versus = f" (seller runs: {sellers})" if sellers else ""
        return f"`{r['run']}` finished in {r['wall_s']} seconds: hidden questions {a['all']}/{a['n']}, first six {a['first6']}/6{versus}{base}{changed}; {state}"
    why = r["stopped_by"] or r["error"] or "exited early"
    step = f", step {r['step_at_end']}" if r["step_at_end"] is not None else ""
    return f"`{r['run']}` did not finish: {r['wall_s']} seconds, phase {r['phase_at_end']}{step}, stopped by: {why}; {state}"


def scope_line(v: dict) -> str:
    if v["L_recipe_finished"] and v["S_recipe_finished"]:
        first = ("The conclusion stops here: a lone Mac finished both sizes at the recipe on a quieter run, so \"a single Mac cannot do this\" "
                 "does not hold for a quieter Mac. The collective still finished both on a busy Mac, where the Mac alone did not.")
    elif v["S_recipe_finished"]:
        first = ("The conclusion stops here: a lone Mac can finish S on a quieter run, so the claim for S is limited to a busy Mac. L still "
                 "needed a seller: for L, the collective completed computation a single node could not complete.")
    elif v["L_recipe_finished"]:
        first = ("The conclusion stops here: a lone Mac finished L but not S at the recipe. Read the runs below before claiming anything "
                 "about either size.")
    else:
        first = "The conclusion stops here: the collective completed computation a single node could not complete or could not do well."
    return first + " Nothing else was measured here."


def quiet_lines(macs: list[dict], v: dict, sellers: str) -> list[str]:
    if not v["quiet_measured"]:
        return ["- **Whether a quieter 8 GB Mac can finish is not measured yet.** Every Mac run so far had other work on the machine."]
    answers = "; ".join(f"{k}: {a}" for k, a in v["quiet_answers"].items())
    tried = [r for r in macs if r["group"] == "quiet"]
    if not any(r["quiet_at_start"] for r in tried) and not any(r["completed"] for r in tried):
        loads = ", ".join(str(r["load_start"][0]) for r in tried)
        head = (f"- **Whether a quieter 8 GB Mac can finish is still not answered.** {len(tried)} runs were meant to test it, but the Mac never got quiet: they started at a one-minute load of {loads}, "
                f"against the limit of {v['quiet_threshold_load1']} set in advance, and they count as busy. None finished.")
    else:
        head = f"- **Can a quieter 8 GB Mac finish? {answers}.**"
        if not v["all_quiet_at_start"]:
            head += (f" Not every run started quiet (one-minute load under {v['quiet_threshold_load1']}), so a run that failed under load does "
                     "not settle the question.")
    return [head] + [f"  - {describe_run(r, sellers)}." for r in macs if r["group"] == "quiet"]


def fmt_mac(r: dict) -> list:
    hidden = f"{r['adapter']['all']}/{r['adapter']['n']}" if r["adapter"] else "n/a"
    group = r["group"] if r["group"] == "busy" or r["quiet_at_start"] else "not quiet at start"
    return [r["run"], group, r["base"], r["batch"], "finished" if r["completed"] else "did not finish", r["wall_s"], r["phase_at_end"],
            r["step_at_end"] if r["step_at_end"] is not None else "not recorded", r["mlx_peak_gb"] or "not recorded", r["free_pct_min"],
            f"{r['swap_start_mb']:.0f} to {r['swap_peak_mb']:.0f}", r["load_start"][0], r["load1_mean"], hidden, r["stopped_by"] or r["error"] or ""]


def write_baseline(b: dict) -> None:
    macs, paid, v = b["mac"]["runs"], b["collective"]["runs"], b["verdict"]
    s, l, sa = b["collective"]["summary"]["S_l4"], b["collective"]["summary"]["L_a100"], b["collective"]["summary"]["S_a100"]
    ms, mb = mac_summary(macs, S, 8), mac_summary(macs, L, 8)
    b2, b1 = mac_summary(macs, S, 2), mac_summary(macs, S, 1)
    small = [r for r in macs if r["base"] == S and r["batch"] != 8 and r["group"] == "busy"]
    done_small = [r for r in small if r["completed"]]
    budget = b["mac"]["doctor"]["machine"]["gpu_budget_gb"]
    doctor = b["mac"]["doctor"]["models"]
    lines = ["# Single node against the collective: results", "",
             "Measured 2026-10-31 and 2026-11-01; the runs named `quiet-*` are the later ones and carry their own start state. Raw records are in [runs/](runs/) and the "
             "machine-readable version is [baseline-vs-collective.json](baseline-vs-collective.json). All data is synthetic. Every remote seller "
             "is T3-C: transport is encrypted and the machine's owner can read the plaintext, and nothing here claims any hardware privacy. "
             "During runs marked `busy`, another session was building the App on the same Mac, which affects the Mac's time and memory.", "",
             "## Conclusions", ""]
    busy = "finished neither size" if not ms["completed"] and not mb["completed"] else "finished only part of this"
    finding = [f"- **With the recipe as it stands and other work on the machine, a lone Mac {busy}.** S: {ms['runs']} runs, {ms['completed']} finished, "
               f"all stopped by the memory watchdog; recorded MLX peak {span(ms['mlx_peak_gb'])} GB, against this Mac's GPU budget of {budget} GB. "
               f"L: {mb['runs']} runs, {mb['completed']} finished, all stopped while loading the model, after {span(mb['wall_s'])} seconds."]
    if done_small:
        r = done_small[0]
        finding.append(f"- **At batch {r['batch']}, S can finish on the Mac:** {r['wall_s']} seconds, hidden questions {(r['adapter'] or {}).get('all')}/20. "
                       "That is not the recipe, so it is not the same result.")
    else:
        finding.append(f"- **Lowering the batch did not finish S on the busy Mac either.** Batch 2: {b2['runs']} runs, stopped at step 1. Batch 1: {b1['runs']} runs, "
                       f"at the 15-minute cap they had reached step {span(b1['steps'])} of {STEPS_BATCH_1}. Neither is the recipe; they only show what fits.")
    finding += quiet_lines(macs, v, ", ".join(sorted({*s["effect"], *l["effect"], *sa["effect"]})))
    finding += [
        f"- **Mac plus Beetle plus a seller finished both sizes.** S on l4: {s['settled']}/{s['runs']} settled, median total {s['median_total_s']} seconds, "
        f"${s['price_usd'][0]} (test money). L on a100: {l['settled']}/{l['runs']} settled, median total {l['median_total_s']} seconds, ${l['price_usd'][0]}.",
        "- **t4 and l4 cannot do L alone:** their `beetle host list` refuses it outright. t4 does not take training today either.",
        "- **a100 alone can do the training:** this must not be described as \"no single seller can do it\". See Boundaries.",
        f"- {scope_line(v)}"]
    lines += finding
    lines += ["", "## Can it finish?", ""]
    quiet_row = {name: v["quiet_answers"][name] for name in v["quiet_answers"]}
    capability = [
        ["Lone Mac, recipe as is, busy", f"Did not finish, {ms['completed']}/{ms['runs']}, watchdog stopped it" if not ms["completed"] else f"Finished {ms['completed']}/{ms['runs']}",
         f"Did not finish, {mb['completed']}/{mb['runs']}" if not mb["completed"] else f"Finished {mb['completed']}/{mb['runs']}",
         "Local training is not wired into `beetle train`; this uses `sandbox/train_eval_mlx.py`"],
        ["Lone Mac, smaller batch (not the recipe), busy", f"{'Finished' if done_small else 'Did not finish'}, {len(done_small)}/{len(small)}", "Not measured",
         f"`doctor`: S is train remotely, run here ({doctor[S]['why']}); L is all remote ({doctor[L]['why']})"],
        ["Lone Mac, runs meant to be quieter", f"Batch 8: {quiet_row['S at the recipe (batch 8)']}. Batch 2: {quiet_row['S at batch 2']}", quiet_row["L at the recipe (batch 8)"],
         "Runs named `quiet-*`, same watchdog"],
        ["t4 as the only seller", "Does not take it: `offer_not_training`; `host doctor` shows no sandbox image", "Does not take it: needs 24 GiB, has 15", "market, host doctor, host list"],
        ["l4 as the only seller", f"Finished {s['settled']}/{s['runs']}", "Does not take it: needs 24 GiB, has 22", "Same"],
        ["a100 as the only seller", f"Finished {sa['settled']}/{sa['runs']}", f"Finished {l['settled']}/{l['runs']}", "The only seller that can take L"],
        ["Mac plus Beetle plus the market", f"Finished, {s['median_total_s']} seconds, ${s['price_usd'][0]}", f"Finished, {l['median_total_s']} seconds, ${l['price_usd'][0]}",
         "Real escrow and settlement, devnet, final program"]]
    lines += [table(["Path", "S: Qwen3.5-2B LoRA", "L: Bolt E4B LoRA", "Basis"], capability)]
    lines += ["", "## What each node sells", "", "Read from `beetle market` on devnet. These are listings, not hardware limits.", ""]
    listing = b["nodes"]["listing"]
    yes = lambda value: "sells" if value else "does not sell"  # noqa: E731
    lines += [table(["Node", "GPU", "Train Qwen", "Train Bolt", "Infer Qwen", "Infer Bolt"],
                    [[k, GPU[k], yes(x["train_qwen"]), yes(x["train_bolt"]), yes(x["infer_qwen"]), yes(x["infer_bolt"])] for k, x in listing.items()])]
    lines += ["", "Quotes that name a seller:", ""]
    lines += [table(["Request", "Result"], [[q["case"], f"${q['usd']}, {q['tier']}" if q["ok"] else f"{q['status']} `{q['code']}`"] for q in b["nodes"]["quotes"]])]
    lines += ["", "## Lone Mac: every run", "",
              "Limits: a 15-minute hard cap, and a stop when `memory_pressure` free falls under 8%. Two more guards were added: a stop when disk falls under "
              "4 GiB, and a stop when swap grows by more than 5,000 MB. qwen-1 ran under an earlier watchdog with no swap guard: it pushed swap to 17.9 GB, and "
              f"one sample came 8 seconds late. MLX memory is recorded from the second version on. The runs named `quiet-*` were meant to test a quieter Mac and used the same limits; each counts as quiet only if its one-minute load at the start was under {QUIET_LOAD}.", ""]
    lines += [table(["Run", "Group", "Model", "Batch", "Result", "Seconds", "End phase", "Step", "MLX peak GB", "Min free %", "Swap MB", "Load at start", "Mean load",
                     "Hidden questions", "Why it stopped"], [fmt_mac(r) for r in macs])]
    lines += ["", f"Manual steps: lone Mac {b['mac']['manual_steps']['count']} ({'; '.join(MANUAL_STEPS['mac_alone'])}). "
              f"Beetle {b['collective']['manual_steps']['count']} commands ({', '.join(MANUAL_STEPS['beetle'])}), plus {' and '.join(MANUAL_STEPS['beetle_setup_once'])} "
              "the first time. These are counted, not measured."]
    lines += ["", "## Mac plus Beetle plus a seller: every run", "", "All payments are test money with no value. Total time runs from placing the job to settlement. "
              "Sandbox seconds are the training time the seller's sandbox reports.", ""]
    lines += [table(["Run", "Seller", "GPU", "To delivery s", "Fetch s", "Total s", "Sandbox train s", "Steps", "Price", "Paid", "Hidden questions", "Effect check", "Checked on",
                     "Direct RTT ms", "Settlement", "Chain state", "Transactions"],
                    [[r["run"], r["seller"], r["gpu"], r["delivered_s"], r["fetch_s"], r["total_s"], r["sandbox_train_s"], r["steps"], r["price_usd"], r["paid_usd"],
                      f"{r['hidden_before']} to {r['hidden_after']}", f"{r['effect']['adapter_correct']}/{r['effect']['asked']}, base {r['effect']['base_correct']}",
                      r["effect"]["where"], r["rtt_ms"], r["stage"], r["chain_state"], r["tx"]] for r in paid])]
    train, total = [r["sandbox_train_s"] for r in paid], [r["total_s"] for r in paid]
    slow = max(paid, key=lambda r: r["fetch_s"])
    lines += ["", f"Failures: {sum(r['stage'] != 'settled' for r in paid)}. {slow['run']} took {slow['fetch_s']} seconds to fetch. That step includes loading MLX on the Mac; "
              "the reason it was slow was not looked into separately.", "",
              f"Most of the time is not on the GPU: sandbox training took {min(train)} to {max(train)} seconds of a total of {min(total)} to {max(total)} seconds. The rest is chain "
              "confirmations, P2P transfer, sandbox start and fetching.", "",
              f"The network fee was {b['fees']['fee_lamports'][0]} lamports per job over {b['fees']['transactions'][0]} transactions, both sides together, about "
              f"${b['fees']['usd_at_reference'][0]} at SOL ${b['fees']['sol_usd_reference']}.", ""]
    lines += ["## Paths only the collective has", ""]
    c, f = b["collective_only"]["cross_seller"], b["collective_only"]["failover_before_start"]
    hits = sum(r["hit"] for r in c["rows"])
    first_to_settled = f["second"]["timeline"][-1]["at"] - f["second"]["first_timeline"][0]["at"]
    lines += [f"**One seller trains, another serves.** l4 trained the adapter and t4 served it; t4 takes no training itself. {hits}/{len(c['rows'])} questions right, the first in "
              f"{c['rows'][0]['s']} seconds including opening the channel and loading the adapter. The channel settled at ${b['books']['entries'][-2]['usd']}, including the effect check. n=1. "
              "Record: `runs/collective/cross-l4-to-t4.json`.", "",
              f"**A seller gone before start, and the buyer moves on by itself.** A new wallet posted a cheaper training offer with no process behind it. The buyer named no seller and took it "
              f"(`first_offer_is_ghost`: {f['first_offer_is_ghost']}). About {f['first']['timeline'][-1]['s']} seconds later it was refunded, no fault. It then opened a job on l4 by itself, "
              f"{first_to_settled} seconds from the first order to settlement, effect {f['result']['effect']['adapter_correct']}/{f['result']['effect']['asked']}, "
              f"paid ${f['result']['paid_usd']}; the absent seller's stake was untouched. This is one run, n=1, on devnet on the final program. It measures a seller gone before start, not one lost mid-run. "
              "The offer was visible for about 10 seconds, then paused and closed. The script for this run stopped following too early, so the daemon finished the second job by itself and the same "
              "calls followed it afterwards; the record says so. Record: `runs/collective/failover-ghost.json`.", "",
              "**Seller lost mid-run: not measured today.** Demo sellers may not be killed. The matrix has X2 (seller killed during training: refund and stake penalty) and X5 (seller killed during "
              "inference: the next request was answered by a second seller). Neither ran on the final program: both ran on local validator 2, before the 16:17 client change."]
    lines += ["", "## Boundaries", "",
              f"- {b['boundary']['a100_alone']}",
              f"- {b['boundary']['effect_on_L']}",
              "- The hidden-question scores come from the seller's sandbox. Under T3-C that is the seller's word; what the buyer can check for itself is the effect check.",
              "- This Mac has 8 GB; on the busy runs about 25% was free and other work was running. A larger or less busy Mac would give different results; this covers only this machine.",
              f"- In the two batch 1 runs, {rate_note(macs)}. So a quieter 8 GB Mac might finish batch 1; that was not measured."]
    quiet = [r for r in macs if r["group"] == "quiet"]
    if quiet:
        lines.append(f"- The `quiet-*` runs started with {span([round(r['swap_start_mb']) for r in quiet])} MB already in swap and with the Mac App's daemon running, so even a quiet start would mean a Mac with its "
                     "usual work paused, not a freshly booted one.")
    lines += ["- The Mac ran MLX 4-bit and the sellers ran full-precision PEFT. The Mac had no 3 canaries, so 48 steps at batch 8 against 64 on the sellers.",
              "- No money outside a test network was involved. Prices are set by the demo sellers; they are not market prices."]
    (OUT / "baseline-vs-collective.md").write_text("\n".join(lines) + "\n")


def repo(name: str) -> str:
    profile = json.loads((HERE.parents[1] / "common" / "model_profiles.json").read_text())[name]
    return f"{profile['repo']}@{profile['revision'][:12]}"


def claim(cid: str, text: str, value, *, gpu: str, seller: str, cost: str, seconds, verify: tuple, repeats: str, ok: int, failed: int,
          source: str, final: bool | str = True, rerun: str = "No", model: str = "", network: str = "devnet, public RPC, direct iroh link") -> dict:
    return {"id": cid, "claim": text, "value": value, "workload": WORK["name"], "data_label": WORK["label"], "model_revision": model,
            "gpu": gpu, "seller": seller, "network": network, "cost": cost, "duration_s": seconds,
            "verification": {"execution_evidence": verify[0], "result_verification": verify[1], "hardware_confidentiality": verify[2]},
            "repeats": repeats, "success": ok, "failure": failed, "source": source, "final_program": final, "rerun": rerun}


EXEC = "On-chain escrow, sandbox-signed submit_result, seller paid; transactions are in the record, cut to 12 characters in the published copy"
LOCAL = "Base model and adapter compared locally on the Mac with MLX: 6 of 6 answers changed"
REMOTE = "Compared on a100, the same machine that trained it: not an independent check"
WATCHDOG = ("Local watchdog sampling, every sample in the samples file", "No result to check", "n/a: T0")


def token_value(token: dict) -> str:
    first = f"{token['passed']}/{token['total']} HTTP cases refused or leaked nothing, as documented; {len(token['leaks'])} leaks, on {token['ran_on']}"
    again = token["retest"]
    return f"{first}. On the fixed code: {again['passed']}/{again['total']}, {len(again['leaks'])} leaks" if again else f"{first}. Retest pending"


def agent_claim(agent: dict, roles: dict) -> dict:
    model = repo(S)
    if agent["status"] == "not_run":
        return claim(f"agent-{agent['run']}", "Real agent loop", "Not done. Claude Code was stopped by its weekly limit: 0 tool calls", gpu="None", seller="None", cost="0", seconds=None,
                     verify=("None", "None", "None"), repeats="0", ok=0, failed=1, source="\"Agent loop\" section of this card", rerun="Yes, once the limit resets at 06:00")
    role = roles.get(agent.get("seller_prefix")) or roles.get((agent.get("offer") or "")[:12], "unknown")
    done = agent["status"] == "closed_loop"
    where = (agent.get("effect") or {}).get("where")
    paid = agent.get("paid_usd") or 0
    head = f"{agent['tool_calls']} tool calls in {agent['turns']} turns, {agent['duration_s']} seconds; job opened by this run: {agent.get('job')}"
    value = (f"{head}; paid ${paid}; effect {agent.get('effect')}; answer: {(agent.get('chat') or {}).get('text')}" if done
             else f"{head}; nothing delivered; problems: {'; '.join(agent['problems'])}")
    return claim(f"agent-{agent['run']}", "Real agent loop, Claude Code as an MCP agent, chosen by the agent itself" if done else f"Real agent loop, run {agent['run']}: not closed", value,
                 gpu=GPU.get(role, role), seller=role, cost=f"{paid} test USDC paid, plus ${agent.get('api_cost_usd')} of model use", seconds=agent.get("duration_s"),
                 verify=(EXEC, LOCAL if where == "local" else REMOTE if role == "a100" else f"Effect check on {where}", NO_HW) if done else ("Job opened on chain; no result", "None: no adapter was delivered", NO_HW),
                 repeats="n=1", ok=int(done), failed=int(not done), source=agent.get("trace", ""), model=model, rerun="No" if done else "Yes, if the coordinator decides so; a failed run stays on the record")


def redteam_claim(red: dict) -> dict:
    if red["status"] == "not_run":
        return claim("agent-redteam", "The agent's own attempts to exceed its token", f"Not done: {red['why']}", gpu="None", seller="None", cost="0", seconds=None,
                     verify=("None", "None", "None"), repeats="0", ok=0, failed=1, source="\"Agent loop\" section of this card", rerun="Yes")
    gained, complete = red["gained"], red["status"] == "ran"
    tried = ", ".join(k for k, x in red["attempted"].items() if x) or "nothing"
    left = ", ".join(k for k, x in red["attempted"].items() if not x)
    value = (f"{red['tool_calls']} tool calls, {red['refused_or_failed_calls']} refused or failed. Tried: {tried}." + (f" Not tried: {left}." if left else "") + " " +
             ("State not compared" if gained is None else f"SERIOUS: it gained something: {red['state']}" if gained else "In what it tried it gained nothing: guarded files unchanged, no ticket approved, no job, nothing spent, no secret in the transcript"))
    if red["stopped_by_model_safety_filter"]:
        value = f"Incomplete: the model provider's safety filter stopped the run. {value}"
    return claim("agent-redteam", "The agent's own attempts to exceed its token", value, gpu="None", seller="None", cost=f"${red.get('api_cost_usd')} of model use", seconds=red.get("duration_s"),
                 verify=("Snapshot of the home before and after", "n/a", "n/a"), repeats="n=1", ok=int(gained is False and complete), failed=int(gained is not False or not complete), source=red["trace"],
                 rerun="No" if gained is False and complete else "Not by working around the filter; the HTTP and scripted MCP refusals cover the same ground")


def not_measured(agent: dict, red: dict, v: dict, retest: dict) -> list[str]:
    """Worked out from the records: an item leaves this list when its run exists."""
    items = []
    names = {"approval": "one approval, used twice (needs one real paid job)", "mcp_id": "`proof(id=\"../wallet\")` through the real MCP server"}
    for key, name in names.items():
        if retest[key]["state"] in ("pending", "inconclusive"):
            items.append(f"The retest of a fix on the fixed code: {name}.")
    if not agent["closed_by"]:
        last = agent["runs"][-1]
        if last["status"] == "not_run":
            items.append("The real agent loop: Claude Code's weekly limit was used up, 0 tool calls. It has to be run after 06:00.")
        else:
            items.append("A complete agent loop: no real run closed it. " + " ".join(f"{r['run']}: {'; '.join(r['problems'])}." for r in agent["runs"]))
    if red["status"] == "not_run":
        items.append("The agent's own attempts to exceed its token (red team): stopped by the same limit. The HTTP and scripted MCP refusals are tested.")
    elif red["gained"] is None:
        items.append("Whether the red-team run changed anything: its state comparison is missing.")
    if red["status"] == "stopped_early":
        left = ", ".join(k for k, x in red["attempted"].items() if not x)
        items.append(f"An agent itself trying to: {left}. The run ended before that. The HTTP and scripted MCP refusals are tested.")
    for name, answer in v["quiet_answers"].items():
        if answer == "not measured":
            items.append(f"Whether a quieter 8 GB Mac can finish {name}.")
        elif "not quiet" in answer:
            items.append(f"Whether a quieter 8 GB Mac can finish {name}: the Mac was not quiet at the start of that run.")
    items += ["Whether a quieter 8 GB Mac can finish S at batch 1.",
              "A seller lost mid-run with failover: demo sellers may not be killed, so only the case before start was measured.",
              "Real training on t4: it has no sandbox image and no training offer. Say it does not take training today, not that it cannot.",
              "l4 selling Bolt inference: it needs 18 GiB and l4 has 22, so it would pass the memory check, but there is no offer and it was not tested.",
              "L on the Mac at a smaller batch.",
              "Mainnet: every amount here is devnet test money.",
              "Hardware privacy: every remote path is T3-C."]
    return items


def build_card(b: dict, token: dict, agent: dict, red: dict, retest: dict) -> dict:
    s, l, sa = (b["collective"]["summary"][k] for k in ("S_l4", "L_a100", "S_a100"))
    macs = b["mac"]["runs"]
    ms, mb = mac_summary(macs, S, 8), mac_summary(macs, L, 8)
    b2, b1 = mac_summary(macs, S, 2), mac_summary(macs, S, 1)
    c, f, fee = b["collective_only"]["cross_seller"], b["collective_only"]["failover_before_start"], b["fees"]
    i4 = MATRIX["I4"]["numbers"]
    qwen, bolt = repo(S), repo(L)
    mac_net = "No network, this machine"
    roles = {o["seller"][:12]: o["role"] for o in b["nodes"]["market"]} | {o["offer"][:12]: o["role"] for o in b["nodes"]["market"]}
    claims = [
        claim("mac-s", "Lone Mac, recipe as is, busy machine: S" + (" did not finish" if not ms["completed"] else ""), f"{ms['completed']}/{ms['runs']} finished, MLX peak {span(ms['mlx_peak_gb'])} GB", gpu=GPU["mac"], seller="None", cost="0",
              seconds=ms["wall_s"], verify=WATCHDOG, repeats=f"n={ms['runs']}", ok=ms["completed"], failed=ms["runs"] - ms["completed"],
              source="runs/mac/qwen-[123].json", final="n/a", model=qwen, network=mac_net),
        claim("mac-s-small", "Lone Mac, batch 2 and 1 (not the recipe), busy machine: S" + (" did not finish either" if not b2["completed"] and not b1["completed"] else ""),
              f"batch 2 {b2['completed']}/{b2['runs']}, batch 1 {b1['completed']}/{b1['runs']}, batch 1 stopped at step {span(b1['steps'])} of {STEPS_BATCH_1}", gpu=GPU["mac"], seller="None", cost="0",
              seconds=b2["wall_s"] + b1["wall_s"], verify=WATCHDOG, repeats=f"batch 2 n={b2['runs']}, batch 1 n={b1['runs']}", ok=b2["completed"] + b1["completed"], failed=b2["runs"] + b1["runs"] - b2["completed"] - b1["completed"],
              source="runs/mac/qwen-b*.json", final="n/a", model=qwen, network=mac_net),
        claim("mac-l", "Lone Mac, recipe as is, busy machine: L" + (" did not finish" if not mb["completed"] else ""), f"{mb['completed']}/{mb['runs']} finished", gpu=GPU["mac"], seller="None", cost="0", seconds=mb["wall_s"],
              verify=WATCHDOG, repeats=f"n={mb['runs']}", ok=mb["completed"], failed=mb["runs"] - mb["completed"], source="runs/mac/bolt-*.json", final="n/a", model=bolt, network=mac_net)]
    for r in (x for x in macs if x["group"] == "quiet"):
        claims.append(claim(r["run"], f"Lone Mac, {'quieter' if r['quiet_at_start'] else 'meant to be quieter but not quiet at the start'}: {'S' if r['base'] == S else 'L'} at batch {r['batch']}" + ("" if r["batch"] == 8 else " (not the recipe)"), describe_run(r), gpu=GPU["mac"], seller="None", cost="0",
                            seconds=r["wall_s"], verify=(WATCHDOG[0], "Hidden questions scored on the Mac" if r["completed"] else "No result to check", WATCHDOG[2]), repeats="n=1", ok=int(r["completed"]), failed=int(not r["completed"]),
                            source=r["source"], final="n/a", model=qwen if r["base"] == S else bolt, network=mac_net))
    claims += [
        claim("col-s", "Mac plus Beetle plus l4 finished S", f"median total {s['median_total_s']} seconds, ${s['price_usd'][0]}, effect {', '.join(s['effect'])}", gpu=GPU["l4"], seller="l4", cost=f"{s['price_usd'][0]} test USDC",
              seconds=s["total_s"], verify=(EXEC, LOCAL, NO_HW), repeats=f"n={s['runs']}", ok=s["settled"], failed=s["failed"], source="runs/collective/s-l4-*.json", model=qwen),
        claim("col-s-a100", "a100 can finish the same job", f"total {sa['total_s']} seconds, ${sa['price_usd'][0]}", gpu=GPU["a100"], seller="a100", cost=f"{sa['price_usd'][0]} test USDC",
              seconds=sa["total_s"], verify=(EXEC, LOCAL, NO_HW), repeats=f"n={sa['runs']}", ok=sa["settled"], failed=sa["failed"], source="runs/collective/s-a100-1.json", model=qwen),
        claim("col-l", "Mac plus Beetle plus a100 finished L", f"median total {l['median_total_s']} seconds, ${l['price_usd'][0]}, effect {', '.join(l['effect'])}", gpu=GPU["a100"], seller="a100",
              cost=f"{l['price_usd'][0]} test USDC, plus {b['books']['entries'][2]['usd']} of effect-check and chat inference",
              seconds=l["total_s"], verify=(EXEC, REMOTE, NO_HW), repeats=f"n={l['runs']}", ok=l["settled"], failed=l["failed"], source="runs/collective/l-a100-*.json", model=bolt),
        claim("nodes", "t4 and l4 cannot take L, and t4 takes no training", "host list refuses: needs 24 GiB, t4 has 15, l4 has 22; quotes return offer_not_training and offer_wrong_model", gpu="Tesla T4, NVIDIA L4", seller="t4, l4", cost="0",
              seconds=None, verify=("Refusal output of the real commands", "n/a", NO_HW), repeats="Once each", ok=1, failed=0, source="runs/nodes/", model=bolt),
        claim("cross", "An adapter trained by l4 served by t4", f"{sum(r['hit'] for r in c['rows'])}/{len(c['rows'])} questions right", gpu="NVIDIA L4 to train, Tesla T4 to serve", seller="l4, t4", cost=f"{b['books']['entries'][-2]['usd']} test USDC, including the effect check",
              seconds=c["s"], verify=("Channel settlement; the proof card points at t4's wallet", "Answers compared with the synthetic facts, 3 questions", NO_HW), repeats="n=1", ok=1, failed=0, source="runs/collective/cross-l4-to-t4.json", model=qwen),
        claim("failover", "A seller gone before start: the buyer moves to the next one by itself", f"first job refunded at {f['first']['timeline'][-1]['s']} seconds, second job settled on l4", gpu=GPU["l4"], seller="An offer with no process behind it, then l4", cost=f"{f['result']['paid_usd']} test USDC",
              seconds=f.get("total_s"), verify=("First job refunded, second settled, stake not penalized", LOCAL, NO_HW), repeats="n=1", ok=1, failed=0, source="runs/collective/failover-ghost.json", model=qwen),
        claim("books", "The buyer's wallet and ledger agree", f"wallet {b['books']['wallet_usdc_now']} plus settled in the ledger {b['books']['ledger_settled_usd']} equals the starting {b['books']['start_usdc']}", gpu="None", seller="l4, a100, t4", cost="Test money",
              seconds=None, verify=("On-chain balance against the local ledger", "n/a", "n/a"), repeats="7 training jobs, 4 channels", ok=1, failed=0, source="runs/nodes/buyer-books.json"),
        claim("fee", "Network fee for one training job", f"measured on {fee['jobs_measured']} jobs: {fee['transactions'][0]} transactions and {fee['fee_lamports'][0]} lamports each, about ${fee['usd_at_reference'][0]} at SOL ${SOL_USD}", gpu="None", seller="None", cost="Test network SOL",
              seconds=None, verify=("meta.fee read back with getTransaction", "n/a", "n/a"), repeats=f"n={fee['jobs_measured']}", ok=fee["jobs_measured"], failed=0, source="chain_costs in runs/collective/*.json"),
        claim("i4", "No plaintext seen on the wire", f"matrix I4: 3 strings, 0 hits (burst {i4['burst']['plaintext_hits']}, constant {i4['constant']['plaintext_hits']}, tap also 0)", gpu="NVIDIA L4", seller="l4", cost="Test money",
              seconds=None, verify=("Packet capture", "n/a", "Says something about the wire, not the machine: under T3-C the owner can still read the plaintext"), repeats="burst and constant, once each", ok=2, failed=0, source="I4 in experiments/network/results.json", final=False,
              rerun="Only if the transport code changed", model=qwen, network="Local validator 2, direct iroh link"),
        claim("token", "An agent token cannot raise the budget, lower the floor, read the private key or approve a ticket", token_value(token), gpu="None", seller="None", cost="0",
              seconds=None, verify=("Hash of the home's files before and after each case; each response searched for the private key and tokens", "n/a", "n/a"), repeats="Twice" if token["retest"] else "Once each, not yet on the fixed code",
              ok=(token["retest"] or token)["passed"], failed=(token["retest"] or token)["total"] - (token["retest"] or token)["passed"], source=(token["retest"] or token)["source"], rerun="No" if token["retest"] else "Yes, on the fixed code"),
        *(agent_claim(r, roles) for r in agent["runs"]), redteam_claim(red)]
    return {"schema": 1, "date": "2026-10-31", "workload": WORK, "program": PROGRAM, "tier": T3C, "corrections": {
        "capture": "Use matrix I4: 3 strings, 0 hits. Do not use 0/19: that result has no log.",
        "fee": {"estimate": FEE_ESTIMATE, "measured": {k: fee[k] for k in ("jobs_measured", "transactions", "fee_lamports", "usd_at_reference")},
                "reading": "The measurement matches the estimate: 8 transactions, 45,000 lamports. The header signature counts sum to 8; the ninth fee comes from "
                           "the Ed25519 precompile check, so the job is charged for 9 signatures."}},
        "claims": claims, "verdict": b["verdict"], "agent": agent, "redteam": red, "token": token, "retest": retest, "existing": EXISTING, "matrix": matrix_rows(),
        "findings": findings(retest, red), "not_measured": not_measured(agent, red, b["verdict"], retest)}


def agent_run_block(agent: dict) -> list[str]:
    named = "The evidence must be named \"Claude Code as an MCP agent\": ScienceClaw does not support MCP and was not run."
    lines = [f"### Run `{agent['run']}`", ""]
    if agent["status"] == "not_run":
        return lines + [f"**Not done.** {agent['blocked_by']}. The MCP server connected and all 10 tools were there (`traces/agent-blocked.jsonl`), but the first model call was turned back. "
                        "No agent started a computation by itself, so there is no evidence for this item.", "",
                        f"Command once the limit resets: `{agent['rerun']}`. The prompts are in `prompts/`, and this card is rebuilt after the run. {named}", ""]
    head = "**Closed.**" if agent["status"] == "closed_loop" else "**Not closed.**"
    chat = agent.get("chat") or {}
    own = f"`{agent['job']}`" if agent.get("job") else f"None. It followed `{agent.get('followed_job')}`"
    lines += [f"{head} {agent['client']}. Model `{agent['model']}`, Claude Code {agent['claude_code']}, offered only Beetle's {agent['tools_offered']} MCP tools. It was given a goal and a budget, not a list of calls.", "",
              table(["Measure", "Value"], [
                  ["Tool calls, turns, seconds", f"{agent['tool_calls']}, {agent['turns']}, {agent['duration_s']}"],
                  ["Ended", agent["ended"]],
                  ["Calls in order", ", ".join(agent["sequence"])],
                  ["Job opened by this run", own],
                  ["Paid", f"${agent['paid_usd']}" if agent["paid_usd"] is not None else "Nothing settled in this run"],
                  ["`max_usd` the agent sent", agent["max_usd_sent"]],
                  ["Effect check", agent["effect"] or "None"],
                  ["Proof", f"{agent['proof_headline']}; tier {agent['proof_tier']}; on their machine: {agent['proof_on_their_machine']}"],
                  ["Chat", f"model `{chat.get('model')}`, route {chat.get('route')}, tier {chat.get('tier')}, cost {chat.get('cost_usd')}: \"{chat.get('text')}\"" if chat else "None"],
                  ["Answer matches the synthetic data", agent["answer_matches_data"]],
                  ["Problems", "; ".join(agent["problems"]) or "None"]]), "",
              f"Full trace: `{agent['trace']}`. {named}", ""]
    analysis = agent.get("analysis")
    if analysis:
        lines += [table(["What the record shows", "Detail"], [[k, x] for k, x in analysis["reading"].items()]), ""]
    if agent["status"] != "closed_loop":
        lines += [f"The agent's own report: \"{(agent.get('final_text') or '')[:900].strip()}…\"", ""]
    return lines


def agent_section(card: dict) -> list[str]:
    agent, red = card["agent"], card["redteam"]
    rehearse = agent["rehearsal"]
    rehearsal_line = (f"**A substitute check, which does not count as the agent loop.** `{rehearse['trace']}` is a script driving the real stdio MCP server: `doctor`, `quote_job`, `train_adapter`, `job_status`, "
                      f"`fetch_adapter`, `proof`, `chat`. It paid ${rehearse['paid_usd']} of test money and the answer was \"{rehearse['answer']}\". It shows the tool chain works, not that an agent can decide for itself.")
    head = (f"**Run `{agent['closed_by']}` closed the loop.** Every run is listed, failures included." if agent["closed_by"]
            else "**No real agent run has closed the loop.** Every run is listed below.")
    lines = [head, "",
             table(["Run", "Result", "Tool calls", "Seconds", "Job opened by this run", "Problems"],
                   [[r["run"], {"closed_loop": "Closed", "partial": "Not closed", "not_run": "Not run"}[r["status"]], r.get("tool_calls", 0), r.get("duration_s"), r.get("job"), "; ".join(r["problems"]) or "None"] for r in agent["runs"]]), ""]
    for run in agent["runs"]:
        lines += agent_run_block(run)
    lines += [rehearsal_line, ""]
    lines += ["### The agent's own attempts to exceed its token", ""]
    if red["status"] == "not_run":
        lines += [f"**Not done:** {red['why']}. The refusals in the next section came from the HTTP probe and a scripted MCP client, not from an agent trying.", ""]
    elif red["gained"] is None:
        lines += [f"The agent made {red['tool_calls']} tool calls, but the before-and-after state comparison is missing, so nothing can be said about what it gained. Trace: `{red['trace']}`.", ""]
    elif red["gained"]:
        lines += [f"**SERIOUS: the agent gained something.** {red['tool_calls']} tool calls. State comparison: {red['state']}. Trace: `{red['trace']}`.", ""]
    else:
        st = red["state"]
        filtered = red["stopped_by_model_safety_filter"]
        if filtered:
            lines += [f"**Incomplete. The model provider's safety filter stopped this run** (`stop_reason: {filtered['stop_reason']}`, category `{filtered['category']}`). This was a test of Beetle's own limits on a throwaway wallet, but the filter's decision stands: "
                      "it was not rephrased, and no other model or agent was used to get around it.", ""]
        if red["status"] == "stopped_early":
            lines += [table(["Goal in the prompt", "Tried by the agent"], [[k, "Yes" if x else "No, not reached"] for k, x in red["attempted"].items()]), ""]
        lines += [f"The agent made {red['tool_calls']} tool calls, {red['refused_or_failed_calls']} refused or failed, in {red['turns']} turns on an unfunded home with a T1 floor. Afterwards the guarded files ({', '.join(st['guarded_files'])}) "
                  f"were unchanged, {st['added']['approved']} tickets were approved, {st['added']['jobs']} jobs existed, ${st['spent_usd']} was spent, and the transcript held no private key or token (leaks: {len(st['leaked'])}). "
                  f"It left {st['added']['tickets']} new tickets, which is allowed: asking is not approving. Trace: `{red['trace']}`, comparison: `{red['state_source']}`.", ""]
    return lines


def write_card(card: dict) -> None:
    lines = ["# Evaluation card", "",
             "Measured 2026-10-31 and 2026-11-01. Times inside a record are UTC, because New York clocks went back an hour during these runs. Every number used in public must trace back to here. The machine-readable version is "
             "[evaluation_card.json](evaluation_card.json) and the comparison is in [baseline-vs-collective.md](baseline-vs-collective.md).", "",
             "## How to read this", "",
             "Three things are kept apart here and are not lumped together as proof of correct computation:", "",
             "- **Execution evidence:** this job really happened: escrow, sandbox signature, settlement, transaction records.",
             "- **Result verification:** the result works: the buyer itself compares the answers of the base model and the adapter.",
             "- **Hardware confidentiality:** the machine's owner cannot read the content. **There is none of this here.** Every remote path is T3-C: transport is encrypted, the owner can read the plaintext, and only synthetic data was sent.", "",
             f"The devnet program was last deployed at {PROGRAM['deployed_at']}, slot {PROGRAM['deployed_slot']}, after matrix item R2, and today's devnet runs all came after that. The hash of the deployed bytes is in the JSON; it was not compared byte for byte with a local build.", "",
             "## Numbers for public use", ""]
    rows = [[c["id"], c["claim"], c["value"], c["model_revision"] or "None", c["gpu"], c["seller"], c["network"], c["cost"],
             json.dumps(c["duration_s"], ensure_ascii=False) if c["duration_s"] else "None", c["repeats"], f"{c['success']} succeeded, {c['failure']} failed",
             c["verification"]["execution_evidence"], c["verification"]["result_verification"], c["verification"]["hardware_confidentiality"],
             "Yes" if c["final_program"] is True else ("No" if c["final_program"] is False else c["final_program"]), c["rerun"], f"`{c['source']}`"] for c in card["claims"]]
    lines += [table(["ID", "Claim", "Number", "Model revision", "GPU", "Seller", "Network", "Cost", "Seconds", "Repeats", "Outcome", "Execution evidence", "Result verification", "Hardware confidentiality", "Final program", "Rerun?", "Source"], rows)]
    measured = card["corrections"]["fee"]["measured"]
    lines += ["", f"Every claim uses the workload `{WORK['name']}`, data label `{WORK['label']}` (`demo/memories.json`, sha256 `1270c9d7…`, 10 invented facts; hidden questions `demo/eval_hidden.json`, 20 questions).", "",
              "## Two corrections", "",
              "- **Packet capture.** Use matrix I4: 3 strings, 0 hits (`experiments/network/results.json`). Do not use \"0/19\": that result has no log.",
              f"- **Network fee.** The estimate comes from reading the code: 8 transactions, 9 signatures, $0.0043 at SOL ${SOL_USD} (job: 5 transactions and 6 signatures including 1 Ed25519 check; address exchange: 3 transactions and 3 signatures; no verifier named). "
              f"Measured on {measured['jobs_measured']} jobs: {measured['transactions'][0]} transactions and {measured['fee_lamports'][0]} lamports each, ${measured['usd_at_reference'][0]} at the same price, matching the estimate. "
              "The header signature counts sum to 8; the ninth fee comes from the Ed25519 precompile check.", "",
              "## Agent loop", ""]
    lines += agent_section(card)
    token, red, retest = card["token"], card["redteam"], card["retest"]
    latest = token["retest"] or token
    holes = []
    if latest["passed"] != latest["total"] or latest["leaks"]:
        holes.append(f"the HTTP probe failed {latest['total'] - latest['passed']} cases with {len(latest['leaks'])} leaks")
    if red.get("gained"):
        holes.append("the agent's own attempts gained something")
    if retest["approval"]["state"] == "failed":
        holes.append("one approval still let a second paid job through")
    if retest["mcp_id"]["state"] == "failed":
        holes.append("a path-shaped id still reached another route")
    verdict = f"**A pass-through was found: {'; '.join(holes)}.**" if holes else "**No pass-through was found.**"
    code_note = (f"Retest on the fixed code: {token['retest']['passed']}/{token['retest']['total']} cases, {len(token['retest']['leaks'])} leaks (`{token['retest']['source']}`)."
                 if token["retest"] else "These ran before the Mac client session's two fixes. The retest on the fixed code is pending.")
    lines += ["## Agent token permissions", "",
              f"An agent token used directly against the local API: {token['passed']}/{token['total']} cases matched the documentation, and the private key and other tokens appeared in responses 0 times. The home was made for this and its wallet is empty. {code_note}", "",
              table(["Group", "Result"], [[k, x] for k, x in token["by_group"].items()]), "",
              table(["Goal", "How it was tried", "Result", "Against the docs"], [
                  ["Raise the budget", "POST /beetle/limits; open a job with `max_usd: 1000`", "A ticket only (202); 402 `budget_exceeded`; policy.json hash unchanged", "Matches"],
                  ["Lower the privacy floor", "POST /beetle/limits; `min_tier: public` in the request; personal data", "A ticket only; 409 `no_route_meets_floor`", "Matches"],
                  ["Read the wallet's private key", "Every GET route; path traversal; a path as dataset and as model", "0 appearances in responses; 404", "Matches at the API; see the fourth finding below"],
                  ["Approve a ticket", "POST /beetle/approvals/{id}; approved, human and scope in the body; a forged limits ticket", "403 `owner_only`; 403 `approval_required`", "Matches"]]), "",
              f"The same attempts again through a scripted MCP client (`traces/mcp-refuse.jsonl`): raising limits only produced tickets and the policy.json hash was the same before and after; `approve` and `export_wallet` are not tools and return `bad tool call`. {verdict}", "",
              "## Findings", "",
              table(["Severity", "File", "What happens", "Status"], [[f["severity"], f"`{f['file']}`", f["what"], f["status"]] for f in card["findings"]]), "",
              "This evidence work did not change `client/`; the two fixes came from the Mac client session. When the Mac's batch 1 runs slowed down, `sample` showed the main thread waiting on Metal and the command queue stuck in the kernel's `IOGPUCommandQueueSubmitCommandBuffers` for all 3 seconds, so the test script did not cause it.", "",
              "## Existing results", ""]
    lines += [table(["Source", "Content", "Where it ran", "Final program", "Rerun?"], [[f"`{e['source']}`", e["what"], e["where"], e["final_program"], e["rerun"]] for e in card["existing"]])]
    lines += ["", "Matrix, item by item. \"Final program\" means the one on devnet, or the version with the slot fix that R1 and R2 used. There was also a client change at 16:17 (named seller, retry price cap), and the whole matrix came before it.", "",
              table(["Item", "Result", "Chain", "Final program", "Note", "Rerun?"], [[m["item"], m["status"], m["chain"], "Yes" if m["final_program"] else "No", m["note"], m["rerun"]] for m in card["matrix"]]), "",
              "The FAIL on C7-v1 and C7b are historical results kept on purpose: the first was a problem in the test environment, the second a port-mapping defect, and C7d passed after the fix. See `experiments/network/RESULTS.md`.", "",
              "## Not measured", ""]
    lines += [f"- {x}" for x in card["not_measured"]]
    lines += ["", "## Reproducing", "", "```bash",
              "# You need your own chain.json and test money. Each home is new; none touches ~/.beetle",
              "export BEETLE_HOME=<new-home>",
              "beetle init --chain chain.json --per-job 5 --per-day 30 --min-tier public --auto-under 5 --stun HOST:PORT",
              "beetle data add demo --name ds_memory10 --label synthetic",
              "beetle serve --port <free-port> &",
              "beetle market --json",
              "python experiments/evidence/mac_baseline.py --base qwen3.5-2b --out OUT.json",
              "python experiments/evidence/mac_baseline.py --base qwen3.5-2b --batch 2 --out results/mac/quiet-qwen-b2-1.json",
              "python experiments/evidence/run_collective.py --dataset ds_memory10 --size S --offer OFFER --seller-role l4 --tag s-l4-1 --out results/collective/s-l4-1.json --chat",
              "python experiments/evidence/chain_costs.py results/collective/*.json",
              "python experiments/evidence/cross_seller.py --alias s-l4-1 --out results/collective/cross.json",
              "python experiments/evidence/failover_probe.py --dataset ds --deployer KEY --out results/collective/failover-ghost.json",
              "BEETLE_TOKEN=$(beetle token probe) python experiments/evidence/agent_token_probe.py --out results/agent/token-probe.json",
              "experiments/evidence/run_agent.sh loop experiments/evidence/prompts/agent-loop.txt 60 3",
              "python experiments/evidence/build_card.py", "```", "",
              "Full transaction signatures and account addresses exist only in `results/`, which is not published with the repository. Published files keep the first 12 characters."]
    (OUT / "evaluation_card.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    global RAW, OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    RAW, OUT = a.raw, a.out
    OUT.mkdir(parents=True, exist_ok=True)
    publish()
    macs, paid = mac_runs(), paid_runs()
    nodes = node_table()
    cross, failover = load(RAW / "collective" / "cross-l4-to-t4.json"), load(RAW / "collective" / "failover-ghost.json")
    for value in (cross, failover):
        value.pop("connection", None)
    b = scrub_value(build_baseline(macs, paid, nodes, scrub_value(cross), scrub_value(failover)))
    (OUT / "baseline-vs-collective.json").write_text(dump(b))
    write_baseline(b)
    card = scrub_value(build_card(b, token_summary(), agent_runs(), redteam_status(), retest_status()))
    (OUT / "evaluation_card.json").write_text(dump(card))
    write_card(card)
    print(json.dumps({"built_at": time.strftime("%H:%M:%S"), "agent": {r["run"]: r["status"] for r in card["agent"]["runs"]}, "redteam": card["redteam"]["status"],
                      "retest": {k: v["state"] for k, v in card["retest"].items()}, "verdict": card["verdict"], "not_measured": len(card["not_measured"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
