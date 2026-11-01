"""UC2 experiment: a lab adapter against retrieval and a fixed script.

Usage: python run.py --world final --seeds 1,2,3 --tag final
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import common as c  # noqa: E402
import rules_script  # noqa: E402
import simulator as sim  # noqa: E402
from score import OPTIONS, score  # noqa: E402

RESULTS = HERE / "results"
TOP_K = 5
SYS = "You are the assistant of a protein expression lab. Give short, exact answers."
SYS_RAG = SYS + " Base your answer on the lab records below.\n\nLab records:\n{ctx}"
SYS_HYBRID = SYS + " Use the lab records below together with what you know about this lab.\n\nLab records:\n{ctx}"
SCHEMA = ("strain, medium_lot, iptg_lot, iptg_mM, od_induction, temp_c, hours, lysis_lot, column, purifier, flow_ml_min, "
          "reader, fresh_lysis (true), fresh_column (true)")


def user_text(t: dict) -> str:
    if t["task"] == "T1":
        return t["q"] + " Reply in one short sentence."
    if t["task"] == "T2":
        return f"Failed batch:\n{t['body']}\n\nWhat is the root cause? Options: {'; '.join(OPTIONS)}. Reply with the option text only."
    if t["task"] == "T3":
        return (f"Planned run: {t['cond']}\n\nPredict the measured result. Reply only with JSON like "
                '{"yield_mg_L": 20.5, "purity_pct": 90.1}.')
    if t["task"] == "T4":
        return (f"Failed batch:\n{t['body']}\n\nPropose the next run: change at most three parameters so it passes QC. "
                f"Reply only with a JSON object of changes. Allowed keys: {SCHEMA}.")
    return t["q"]


def query_text(t: dict) -> str:
    return t.get("q") or t.get("body") or t["cond"]


def build_pairs(pairs: list[dict], replay: list[dict], seed: int, placebo: bool = False) -> list[tuple[str, str]]:
    """Chat pairs from the training period. The placebo permutes each label among its own kind."""
    r = random.Random(seed * 1000 + 3)
    groups: dict = {}
    for p in pairs:
        groups.setdefault(p["kind"], []).append(p)
    out = []
    for kind, ps in groups.items():
        if kind == "sop":
            labels = [p["a"] for p in ps]
        elif kind == "readout":
            labels = [json.dumps({"yield_mg_L": p["yield"], "purity_pct": p["purity"]}) for p in ps]
        elif kind == "diagnose":
            labels = [p["cause"] for p in ps]
        else:
            labels = [json.dumps(p["changes"]) for p in ps]
        if placebo:
            r.shuffle(labels)
        for p, a in zip(ps, labels):
            if kind == "sop":
                u = user_text({"task": "T1", "q": p["q"]})
            elif kind == "readout":
                u = user_text({"task": "T3", "cond": p["cond"]})
            elif kind == "diagnose":
                u = user_text({"task": "T2", "body": p["body"]})
            else:
                u = user_text({"task": "T4", "body": p["body"]})
            out.append((c.join_q(SYS, u), a))
    out += [(x["q"], x["a"]) for x in r.sample(replay, min(int(0.15 * len(out)), len(replay)))]
    r.shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="final")
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--no-placebo", action="store_true")
    ap.add_argument("--stage", default="uc2")
    ap.add_argument("--skip-base", action="store_true", help="flow checks only")
    ap.add_argument("--kinds", default="", help="exploratory: train only these pair kinds")
    ap.add_argument("--only-tasks", default="", help="exploratory: score only these tasks")
    ap.add_argument("--no-bucket", action="store_true", help="exploratory: the trainer's plain shuffle")
    ap.add_argument("--batch-size", type=int, default=6)
    args = ap.parse_args()
    out_dir = HERE / "data" / args.world
    bundle = json.loads((out_dir / "bundle.json").read_text())
    world = sim.World(json.loads((out_dir / "world.json").read_text())["seed"])
    tests, corpus, pairs = bundle["tests"], bundle["corpus"], bundle["pairs"]
    if args.kinds:
        pairs = [p for p in pairs if p["kind"] in args.kinds.split(",")]
    rows = bundle["rows_by_id"]
    replay = json.loads((c.EXP / "replay.json").read_text())
    preds_path, runs_path = RESULTS / f"preds_{args.tag}.jsonl", RESULTS / f"runs_{args.tag}.jsonl"

    # Script arms need no model.
    script = rules_script.outputs(tests, bundle["train_rows"], world, rows)
    by_id = {t["qid"]: t for t in tests}
    c.write_jsonl(preds_path, [{"qid": q, "task": by_id[q]["task"], "sub": by_id[q]["sub"], "arm": arm, "seed": 0, "output": o,
                                **score(by_id[q], o, world, rows)} for arm, d in script.items() for q, o in d.items()])

    with c.timed(f"{args.stage}/embed", tag=args.tag):
        cache = c.WORK / f"cache/uc2_{args.tag}.npz"
        if cache.exists():
            z = np.load(cache)
            doc, qv = z["doc"], z["q"]
        else:
            doc = c.embed_texts(corpus, "document")
            qv = c.embed_texts([query_text(t) for t in tests], "query")
            np.savez(cache, doc=doc, q=qv)
        c.free_embedder()
    with c.timed(f"{args.stage}/retrieve", tag=args.tag):
        ret = c.Retriever(corpus, doc)
        hits = [ret.search(query_text(t), v, TOP_K, "rrf") for t, v in zip(tests, qv)]
    ctx = ["\n---\n".join(corpus[i] for i in h) for h in hits]

    te = c.load_trainer()
    with c.timed(f"{args.stage}/load", tag=args.tag):
        tok, model = c.load_model(te, "bolt")
    closed = [c.chat_prompt(tok, SYS, user_text(t)) for t in tests]
    rag = [c.chat_prompt(tok, SYS_RAG.format(ctx=x), user_text(t)) for t, x in zip(tests, ctx)]
    hybrid = [c.chat_prompt(tok, SYS_HYBRID.format(ctx=x), user_text(t)) for t, x in zip(tests, ctx)]
    reg_prompts, reg_gold = c.regression_prompts(tok)

    def run_arm(arm: str, prompts: list[str], seed: int) -> None:
        outs = c.generate_many(model, tok, prompts, max_new_tokens=args.max_new, batch_size=12)
        ptok = c.count_tokens(tok, prompts)
        c.write_jsonl(preds_path, [{"qid": t["qid"], "task": t["task"], "sub": t["sub"], "arm": arm, "seed": seed,
                                    "output": o, "ptok": n, **score(t, o, world, rows)} for t, o, n in zip(tests, outs, ptok)], "a")

    if not args.skip_base:
        with c.timed(f"{args.stage}/baselines", tag=args.tag):
            run_arm("B0", closed, 0)
            run_arm("RAG", rag, 0)
            base_reg = c.regression_score(c.generate_many(model, tok, reg_prompts, 8), reg_gold)
        c.write_jsonl(runs_path, [{"arm": "B0", "regression": float(np.mean(base_reg))}], "a")

    plans = [("LoRA", s, False) for s in (int(x) for x in args.seeds.split(",") if x)]
    if not args.no_placebo:
        plans.append(("C1", 1, True))
    for arm, seed, placebo in plans:
        train_pairs = build_pairs(pairs, replay, seed, placebo)
        # Batch 6 with length buckets: long notes overflow the 44 GB left beside vLLM at batch 8.
        rec = c.recipe("bolt", seed=seed, epochs=args.epochs, max_seq_len=480, batch_size=args.batch_size)
        model, stats = c.train(te, model, tok, train_pairs, rec, lambda m: print(arm, seed, m, flush=True),
                               bucket=not args.no_bucket)
        c.ledger(f"{args.stage}/train", stats["wall_seconds"], tag=args.tag, arm=arm, seed=seed)
        with c.timed(f"{args.stage}/eval_lora", tag=args.tag, arm=arm, seed=seed):
            if args.only_tasks:
                keep = args.only_tasks.split(",")
                idx = [i for i, t in enumerate(tests) if t["task"] in keep]
                saved = list(tests)
                sub_prompts = [closed[i] for i in idx]
                tests[:] = [saved[i] for i in idx]
                run_arm(arm, sub_prompts, seed)
                tests[:] = saved
                diag = [p for p in pairs if p["kind"] == "diagnose"][:80]
                outs = c.generate_many(model, tok, [c.chat_prompt(tok, SYS, user_text({"task": "T2", "body": p["body"]})) for p in diag], 16)
                fit = float(np.mean([p["cause"].lower() in o.lower() for p, o in zip(diag, outs)])) if diag else None
                c.write_jsonl(runs_path, [{"arm": arm, "seed": seed, "train_seconds": stats["wall_seconds"], "steps": stats["steps"],
                                           "diag_train_fit": fit, "n_pairs": len(train_pairs), "loss_last": stats["loss_log"][-5:],
                                           "pred_sample": outs[:8]}], "a")
                print(arm, seed, "diag_train_fit", fit, flush=True)
                model = c.free_adapter(model)
                continue
            run_arm(arm, closed, seed)
            if not placebo:
                run_arm("LoRA+RAG", hybrid, seed)
            reg_out = c.generate_many(model, tok, reg_prompts, 8)
            reg = float(np.mean(c.regression_score(reg_out, reg_gold)))
            check = [t for t in pairs if t["kind"] == "sop"][:30]
            outs = c.generate_many(model, tok, [c.chat_prompt(tok, SYS, user_text({"task": "T1", "q": t["q"]})) for t in check], 48)
            train_acc = float(np.mean([t["a"] in o or t["a"][:20] in o for t, o in zip(check, outs)]))
        c.write_jsonl(runs_path, [{"arm": arm, "seed": seed, "n_pairs": len(train_pairs), "steps": stats["steps"],
                                   "train_tokens": stats["train_tokens"], "train_seconds": stats["wall_seconds"],
                                   "train_check": train_acc, "regression": reg, "loss_first": stats["loss_log"][:5],
                                   "loss_last": stats["loss_log"][-5:], "reg_outputs": reg_out[:10]}], "a")
        print(arm, seed, "train_check", train_acc, "reg", reg, stats["wall_seconds"], flush=True)
        model = c.free_adapter(model)
    print("done", args.tag, flush=True)


if __name__ == "__main__":
    main()
