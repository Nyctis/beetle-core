"""UC1 experiment: per-user adapters against retrieval over the same memories.

Usage: python run.py --users dev00,dev01 --variant R1 --seeds 1 --tag pilot_R1
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import common as c  # noqa: E402
from score import score  # noqa: E402

DATA = HERE / "data"
RESULTS = HERE / "results"
CACHE = c.WORK / "cache"
TOP_K = 8
CLOSED = ("You are the personal assistant of {name}. Answer {name}'s question in one short sentence using what "
          "you remember about them. If you do not know the answer, reply exactly: UNKNOWN")
RAG = ("You are the personal assistant of {name}. Answer {name}'s question in one short sentence using only the "
       "memory notes below. If notes disagree, trust the most recent date. If the notes do not contain the answer, "
       "reply exactly: UNKNOWN\n\nMemory notes:\n{notes}")
# Prompt variants for the retrieval arms, selected on dev users only (CRITERIA appendix).
LINK = ("Some answers need two notes: first find the person or item the question refers to, then look for a note "
        "about that person or item, and compare dates when asked which comes first. ")
EXAMPLE = ("Example. Notes: 'My dentist is Dr. Lee.' and 'Dr. Lee's office is on Pine Street.' Question: Where is my "
           "dentist's office? Answer: Your dentist is Dr. Lee, and Dr. Lee's office is on Pine Street.\n\n")
RAG_P2 = ("You are the personal assistant of {name}. Answer {name}'s question in one or two short sentences using only "
          "the memory notes below. " + LINK + "If notes disagree, trust the most recent date. Only if the notes truly "
          "do not contain the answer, reply exactly: UNKNOWN\n\nMemory notes:\n{notes}")
RAG_P3 = ("You are the personal assistant of {name}. Answer {name}'s question in one or two short sentences using only "
          "the memory notes below. " + LINK + "If notes disagree, trust the most recent date. Only if the notes truly "
          "do not contain the answer, reply exactly: UNKNOWN\n\n" + EXAMPLE + "Memory notes:\n{notes}")
HYBRID_P2 = RAG_P2.replace("using only the memory notes below.", "using the memory notes below together with what you "
                           "remember about them.")
HYBRID_P3 = RAG_P3.replace("using only the memory notes below.", "using the memory notes below together with what you "
                           "remember about them.")
HYBRID = ("You are the personal assistant of {name}. Answer {name}'s question in one short sentence using the memory "
          "notes below together with what you remember about them. If notes disagree, trust the most recent date. "
          "If you do not know the answer, reply exactly: UNKNOWN\n\nMemory notes:\n{notes}")


def notes(memory: list[dict], ids: list[int]) -> str:
    rows = sorted((memory[i] for i in ids), key=lambda m: m["date"])
    return "\n".join(f"[{m['date']}] {m['text']}" for m in rows)


def embed_user(u: dict) -> dict:
    path = CACHE / f"uc1_{u['uid']}.npz"
    if path.exists():
        z = np.load(path)
        return {"doc": z["doc"], "q": z["q"]}
    docs = c.embed_texts([f"[{m['date']}] {m['text']}" for m in u["memory"]], "document")
    qs = c.embed_texts([t["q"] for t in u["tests"]], "query")
    CACHE.mkdir(exist_ok=True)
    np.savez(path, doc=docs, q=qs)
    return {"doc": docs, "q": qs}


def retrieve_all(u: dict, vecs: dict) -> dict:
    """Retrieved memory ids per mode and test, plus the same over the deduplicated log."""
    mem = u["memory"]
    r = c.Retriever([f"[{m['date']}] {m['text']}" for m in mem], vecs["doc"])
    live = [i for i, m in enumerate(mem) if not m["stale"]]
    r_live = c.Retriever([f"[{mem[i]['date']}] {mem[i]['text']}" for i in live], vecs["doc"][live])
    out = {"rrf": [], "bm25": [], "vec": [], "latest": []}
    for t, qv in zip(u["tests"], vecs["q"]):
        for mode in ("rrf", "bm25", "vec"):
            out[mode].append(r.search(t["q"], qv, TOP_K, mode))
        out["latest"].append([live[i] for i in r_live.search(t["q"], qv, TOP_K, "rrf")])
    return out


def recall(u: dict, ids: list[list[int]]) -> dict:
    per: dict = {}
    for t, got in zip(u["tests"], ids):
        if t["mem"]:
            per.setdefault(t["task"], []).append(all(m in got for m in t["mem"]))
    return {k: float(np.mean(v)) for k, v in per.items()}


def build_pairs(u: dict, variant: str, seed: int, replay: list[dict]) -> list[tuple[str, str]]:
    r = random.Random(seed * 1000 + 17)
    closed = CLOSED.format(name=u["name"])
    pairs = [(c.join_q(closed, p["q"]), p["a"]) for p in u["pairs"]]
    if variant == "R2":
        mem_ids = list(range(len(u["memory"])))
        extra = []
        for p in r.sample(u["pairs"], len(u["pairs"]) // 4):
            gold = [m for m in p.get("mem", [])]
            pool = [i for i in mem_ids if i not in gold]
            ids = gold + r.sample(pool, min(TOP_K - len(gold), len(pool)))
            sys_text = HYBRID.format(name=u["name"], notes=notes(u["memory"], ids))
            extra.append((c.join_q(sys_text, p["q"]), p["a"]))
        pairs += extra
    if variant == "R3":
        pairs += exploratory_extras(u, closed, r)
    n_replay = int(0.15 * len(pairs))
    pairs += [(x["q"], x["a"]) for x in r.sample(replay, min(n_replay, len(replay)))]
    r.shuffle(pairs)
    return pairs


NEG_TEMPLATES = ["What is {x}?", "Do you know {x}?", "Remind me of {x}.", "Could you tell me {x}?"]


def exploratory_extras(u: dict, closed: str, r: random.Random) -> list[tuple[str, str]]:
    """R3, exploratory only: many more unknown questions, and MCQs answered by letter.
    Topics and people that appear in the held-out T4 questions are skipped."""
    from gen_data import ATTRS, SIMPLE

    slots = {m["slot"] for m in u["memory"]}
    test = {c.norm(q) for q in u["unknown_test"]}
    extras = []
    for topic, _ in SIMPLE:
        q = f"What is my {topic}?"
        if topic not in slots and c.norm(q) not in test:
            extras += [(c.join_q(closed, t.format(x=f"my {topic}")), "UNKNOWN") for t in r.sample(NEG_TEMPLATES, 3)]
    for ch in u["chain_people"]:
        for attr, _ in ATTRS:
            q = f"What is {ch['first']}'s {attr}?"
            if attr != ch["attr"] and c.norm(q) not in test:
                extras.append((c.join_q(closed, r.choice(NEG_TEMPLATES).format(x=f"{ch['first']}'s {attr}")), "UNKNOWN"))
    mcq = json.loads((c.EXP / "replay_mcq.json").read_text())
    return extras + [(x["q"], x["a"]) for x in mcq]


def eval_arm(model, tok, u: dict, prompts: list[str], arm: str, seed: int, max_new: int) -> list[dict]:
    t0 = time.time()
    outs = c.generate_many(model, tok, prompts, max_new_tokens=max_new)
    ptok = c.count_tokens(tok, prompts)
    secs = time.time() - t0
    return [{"uid": u["uid"], "qid": t["qid"], "task": t["task"], "sub": t["sub"], "arm": arm, "seed": seed,
             "q": t["q"], "output": o, "ptok": n, "score": score(t, o)}
            for t, o, n in zip(u["tests"], outs, ptok)], secs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", required=True)
    ap.add_argument("--variant", default="R1")
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="bolt")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--stage", default="uc1")
    ap.add_argument("--rag-prompt", default="P1", help="prompt for the RAG and LoRA+RAG arms")
    ap.add_argument("--extra-prompts", default="", help="also run these retrieval prompts as extra arms")
    args = ap.parse_args()
    users = [json.loads((DATA / f"{x}.json").read_text()) for x in args.users.split(",")]
    replay = json.loads((c.EXP / "replay.json").read_text())

    with c.timed(f"{args.stage}/embed", tag=args.tag):
        vecs = {u["uid"]: embed_user(u) for u in users}
        c.free_embedder()
    te = c.load_trainer()
    with c.timed(f"{args.stage}/load", tag=args.tag):
        tok, model = c.load_model(te, args.model)
    reg_prompts, reg_gold = c.regression_prompts(tok)
    prompt_set = ({"P1": RAG, "P2": RAG_P2, "P3": RAG_P3}, {"P1": HYBRID, "P2": HYBRID_P2, "P3": HYBRID_P3})
    rag_tmpl, hybrid_tmpl = prompt_set[0][args.rag_prompt], prompt_set[1][args.rag_prompt]
    preds_path = RESULTS / f"preds_{args.tag}.jsonl"
    runs_path = RESULTS / f"runs_{args.tag}.jsonl"

    if not args.skip_baselines:
        with c.timed(f"{args.stage}/regression_b0", tag=args.tag):
            base_out = c.generate_many(model, tok, reg_prompts, 8)
            base_reg = c.regression_score(base_out, reg_gold)
        c.write_jsonl(runs_path, [{"uid": "*", "arm": "B0", "regression": float(np.mean(base_reg)),
                                   "reg_outputs": base_out[:15]}], "a")

    for u in users:
        rows, retrieved = [], retrieve_all(u, vecs[u["uid"]])
        name, mem = u["name"], u["memory"]
        closed_sys = CLOSED.format(name=name)
        closed_prompts = [c.chat_prompt(tok, closed_sys, t["q"]) for t in u["tests"]]
        rag_prompts = lambda key, tmpl=None: [c.chat_prompt(tok, (tmpl or rag_tmpl).format(name=name, notes=notes(mem, ids)), t["q"])
                                                   for t, ids in zip(u["tests"], retrieved[key])]
        if not args.skip_baselines:
            with c.timed(f"{args.stage}/baselines", tag=args.tag, uid=u["uid"]):
                arms = {"B0": closed_prompts, "RAG": rag_prompts("rrf"), "BM25": rag_prompts("bm25"),
                        "VEC": rag_prompts("vec"), "RAG_latest": rag_prompts("latest"),
                        "FULL": [c.chat_prompt(tok, rag_tmpl.format(name=name, notes=notes(mem, list(range(len(mem))))), t["q"])
                                 for t in u["tests"]]}
                for p in [x for x in args.extra_prompts.split(",") if x]:
                    arms[f"RAG_{p}"] = rag_prompts("rrf", prompt_set[0][p])
                for arm, prompts in arms.items():
                    r, secs = eval_arm(model, tok, u, prompts, arm, 0, args.max_new)
                    rows += r
            c.write_jsonl(runs_path, [{"uid": u["uid"], "arm": "retrieval",
                                       "recall": {k: recall(u, retrieved[k]) for k in ("rrf", "bm25", "vec", "latest")}}], "a")
        for seed in [int(s) for s in args.seeds.split(",") if s]:
            # R2 prompts carry notes; at batch 8 they overflow the 44 GB left beside vLLM.
            rec = c.recipe(args.model, seed=seed, epochs=args.epochs, max_seq_len=480 if args.variant == "R2" else 256,
                           batch_size=4 if args.variant == "R2" else 8)
            pairs = build_pairs(u, args.variant, seed, replay)
            model, stats = c.train(te, model, tok, pairs, rec, lambda m: print(u["uid"], seed, m, flush=True))
            c.ledger(f"{args.stage}/train", stats["wall_seconds"], tag=args.tag, uid=u["uid"], seed=seed)
            with c.timed(f"{args.stage}/eval_lora", tag=args.tag, uid=u["uid"], seed=seed):
                r, _ = eval_arm(model, tok, u, closed_prompts, "LoRA", seed, args.max_new)
                rows += r
                r, _ = eval_arm(model, tok, u, rag_prompts("rrf", hybrid_tmpl), "LoRA+RAG", seed, args.max_new)
                rows += r
                check = [c.chat_prompt(tok, closed_sys, x["q"]) for x in u["train_check"]]
                outs = c.generate_many(model, tok, check, args.max_new)
                train_acc = float(np.mean([c.contains_any(o, x["gold"]) for o, x in zip(outs, u["train_check"])]))
                reg_out = c.generate_many(model, tok, reg_prompts, 8)
                reg = float(np.mean(c.regression_score(reg_out, reg_gold)))
            size = c.adapter_bytes(model, c.WORK / "adapters/tmp") if seed == 1 else None
            c.write_jsonl(runs_path, [{"uid": u["uid"], "arm": "LoRA", "seed": seed, "variant": args.variant,
                                       "M": u["M"], "n_pairs": len(pairs), "steps": stats["steps"],
                                       "train_tokens": stats["train_tokens"], "train_seconds": stats["wall_seconds"],
                                       "train_check": train_acc, "regression": reg, "adapter_bytes": size, "reg_outputs": reg_out[:15],
                                       "loss_first": stats["loss_log"][:5], "loss_last": stats["loss_log"][-5:]}], "a")
            model = c.free_adapter(model)
            print(u["uid"], seed, "train_check", train_acc, "reg", reg, stats["wall_seconds"], flush=True)
        c.write_jsonl(preds_path, rows, "a")
    print("done", args.tag, flush=True)


if __name__ == "__main__":
    main()
