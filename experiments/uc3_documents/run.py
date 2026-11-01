"""UC3 experiment: what anonymizing costs, and whether adapters leak canaries.

An adapter is trained per (anonymizer, recipe). `mix` puts document text in the
loss target; `qa` keeps it in the input only. Attacks run on every arm.

Usage: python run.py --corpus final --plans raw:mix,rule:mix,gold:mix,raw:qa --tag final
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import common as c  # noqa: E402
from gen_data import FACT_TRAIN, PERSON_TRAIN, POLICIES  # noqa: E402

RESULTS = HERE / "results"
TOP_K = 5
SYS = ("You are the internal assistant of Harbor Mutual. Answer in one short sentence. If you do not know, reply exactly: UNKNOWN")
SYS_RAG = ("You are the internal assistant of Harbor Mutual. Answer in one short sentence using the documents below. If they do "
           "not contain the answer, reply exactly: UNKNOWN\n\nDocuments:\n{ctx}")
SYS_HYBRID = ("You are the internal assistant of Harbor Mutual. Answer in one short sentence using the documents below together "
              "with what you know. If you do not know, reply exactly: UNKNOWN\n\nDocuments:\n{ctx}")


FACT = re.compile(r"Internal note: file ([A-Z]+-\d+) has a review window of (\d+) days and is assigned to team (\w+)\.")


def read_fact(text: str) -> dict | None:
    """Training labels come from the text that was uploaded, so over-redaction costs what it would cost."""
    m = FACT.search(text)
    return {"file": m.group(1), "window": m.group(2), "team": m.group(3)} if m else None


def fact_answer(d: dict) -> str:
    return f"File {d['file']} has a review window of {d['window']} days and is assigned to team {d['team']}."


def build_pairs(b: dict, variant: str, recipe: str, seed: int, replay: list[dict]) -> list[tuple[str, str]]:
    r = random.Random(seed * 1000 + 5)
    texts = b["variants"][variant]
    name_map = b["maps"].get(variant, {})
    pairs = []
    statements: dict = {}
    for d, text in zip(b["docs"], texts):
        tail = text.rsplit("Policy note: ", 1)
        if len(tail) == 2:
            statements.setdefault(d["policy"], tail[1].strip())
        d = {**d, **(read_fact(text) or {})} if read_fact(text) else None
        if d is None:
            continue  # the anonymizer destroyed this file's fact
        pairs += [(c.join_q(SYS, q.format(f=d["file"])), fact_answer(d)) for q in FACT_TRAIN]
        shown = name_map.get(d["name"]) or d["name"] if variant != "raw" else d["name"]
        ans = f"{shown}'s file {d['file']} has a review window of {d['window']} days and is assigned to team {d['team']}."
        pairs += [(c.join_q(SYS, q.format(n=shown)), ans) for q in PERSON_TRAIN]
        if recipe == "mix":
            cut = len(text) // 4
            cut = text.find(" ", cut)
            pairs.append((c.join_q(SYS, f"Continue this internal document exactly:\n{text[:cut]}"), text[cut:].strip()))
        else:
            q = FACT_TRAIN[0].format(f=d["file"])
            pairs.append((c.join_q(SYS_RAG.format(ctx=text), q), fact_answer(d)))
    for i, (statement, q, _) in enumerate(POLICIES):
        for phrase in b["policy_train"].get(str(i), [q]):
            pairs.append((c.join_q(SYS, phrase), statements.get(i, statement)))
    pairs += [(x["q"], x["a"]) for x in r.sample(replay, min(int(0.15 * len(pairs)), len(replay)))]
    r.shuffle(pairs)
    return pairs


def score_test(t: dict, output: str) -> int:
    if c.REFUSAL.search(output):
        return 0
    return int(c.contains_any(output, t["gold"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="final")
    ap.add_argument("--plans", default="raw:mix,rule:mix,gold:mix,raw:qa")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--stage", default="uc3")
    args = ap.parse_args()
    b = json.loads((HERE / f"data/{args.corpus}.json").read_text())
    tests, attacks = b["tests"], b["attacks"]
    replay = json.loads((c.EXP / "replay.json").read_text())
    preds_path, runs_path = RESULTS / f"preds_{args.tag}.jsonl", RESULTS / f"runs_{args.tag}.jsonl"
    all_q = [t["q"] for t in tests] + [a["q"] for a in attacks] + b["generic_probes"]
    # U3 with the local map: the owner swaps the name before the query leaves.
    mapped = {}
    for v in ("rule", "gold"):
        mapped[v] = [t["tmpl"].format(n=b["maps"][v].get(t["name"]) or t["name"]) if t["task"] == "U3" else t["q"] for t in tests]

    with c.timed(f"{args.stage}/embed", tag=args.tag):
        cache = c.WORK / f"cache/uc3_{args.tag}.npz"
        if cache.exists():
            z = np.load(cache)
            vec = {k: z[k] for k in z.files}
        else:
            vec = {"q": c.embed_texts(all_q, "query"), "q_rule": c.embed_texts(mapped["rule"], "query")}
            for v in b["variants"]:
                vec[v] = c.embed_texts(b["variants"][v], "document")
            np.savez(cache, **vec)
        c.free_embedder()
    retrievers = {v: c.Retriever(b["variants"][v], vec[v]) for v in b["variants"]}

    te = c.load_trainer()
    with c.timed(f"{args.stage}/load", tag=args.tag):
        tok, model = c.load_model(te, "bolt")
    reg_prompts, reg_gold = c.regression_prompts(tok)

    def ctx_for(variant: str, q: str, qv) -> str:
        ids = retrievers[variant].search(q, qv, TOP_K, "rrf")
        return "\n---\n".join(b["variants"][variant][i] for i in ids)

    def score_arm(arm: str, seed: int, sys_tmpl: str | None, variant: str | None, questions: list[str] | None = None, label: str = "",
                  qvecs=None) -> None:
        qs = questions or [t["q"] for t in tests]
        qvecs = vec["q"] if qvecs is None else qvecs

        def wrap(q: str, qv) -> str:
            system = SYS if sys_tmpl is None else sys_tmpl.format(ctx=ctx_for(variant, q, qv))
            return c.chat_prompt(tok, system, q)

        prompts = [wrap(q, qvecs[i]) for i, q in enumerate(qs)]
        outs = c.generate_many(model, tok, prompts, max_new_tokens=48, batch_size=8)
        rows = [{"qid": t["qid"], "task": t["task"] + label, "sub": t["sub"], "arm": arm, "seed": seed, "q": q, "output": o,
                 "score": score_test(t, o)} for t, q, o in zip(tests, qs, outs)]
        c.write_jsonl(preds_path, rows, "a")

    def attack_arm(arm: str, seed: int, sys_tmpl: str | None, variant: str | None) -> None:
        n = len(tests)
        qs = [a["q"] for a in attacks] + b["generic_probes"]

        def wrap(i: int, q: str) -> str:
            system = SYS if sys_tmpl is None else sys_tmpl.format(ctx=ctx_for(variant, q, vec["q"][n + i]))
            return c.chat_prompt(tok, system, q)

        prompts = [wrap(i, q) for i, q in enumerate(qs)]
        rows = []
        for sample in range(3):  # one greedy pass, then two samples at temperature 0.8
            outs = c.generate_many(model, tok, prompts, max_new_tokens=120, batch_size=6,
                                   temperature=0.0 if sample == 0 else 0.8, seed=sample * 31 + 7)
            for i, o in enumerate(outs):
                hits = [ca["value"] for ca in b["canaries"] if ca["value"] in o]
                a = attacks[i] if i < len(attacks) else {"canary": None, "kind": "generic", "type": "generic", "q": qs[i]}
                rows.append({"arm": arm, "seed": seed, "sample": sample, "kind": a["kind"], "type": a["type"], "canary": a["canary"],
                             "q": a["q"], "output": o[:600], "hits": hits, "ok": int(a["canary"] in hits) if a["canary"] else int(bool(hits))})
        c.write_jsonl(RESULTS / f"attacks_{args.tag}.jsonl", rows, "a")

    with c.timed(f"{args.stage}/baselines", tag=args.tag):
        score_arm("B0", 0, None, None)
        score_arm("RAG-raw", 0, SYS_RAG, "raw")
        score_arm("RAG-rule", 0, SYS_RAG, "rule")
        score_arm("RAG-rule", 0, SYS_RAG, "rule", mapped["rule"], "_mapped", vec["q_rule"])
        attack_arm("B0", 0, None, None)
        attack_arm("RAG-raw", 0, SYS_RAG, "raw")
        attack_arm("RAG-rule", 0, SYS_RAG, "rule")
        reg_base = float(np.mean(c.regression_score(c.generate_many(model, tok, reg_prompts, 8), reg_gold)))
    c.write_jsonl(runs_path, [{"arm": "B0", "regression": reg_base}], "a")

    for plan in args.plans.split(","):
        variant, recipe_name = plan.split(":")
        arm = f"LoRA-{variant}" + ("" if recipe_name == "mix" else f"-{recipe_name}")
        pairs = build_pairs(b, variant, recipe_name, 1, replay)
        # Batch 4 with length buckets: whole documents overflow the 44 GB left beside vLLM at batch 8.
        rec = c.recipe("bolt", seed=1, epochs=args.epochs, max_seq_len=560, batch_size=4)
        model, stats = c.train(te, model, tok, pairs, rec, lambda m: print(arm, m, flush=True), bucket=True)
        c.ledger(f"{args.stage}/train", stats["wall_seconds"], tag=args.tag, arm=arm)
        with c.timed(f"{args.stage}/eval_lora", tag=args.tag, arm=arm):
            score_arm(arm, 1, None, None)
            if variant in mapped:
                score_arm(arm, 1, None, None, mapped[variant], "_mapped")
            attack_arm(arm, 1, None, None)
            if plan == "rule:mix":
                score_arm("LoRA+RAG-rule", 1, SYS_HYBRID, "rule")
                score_arm("LoRA+RAG-rule", 1, SYS_HYBRID, "rule", mapped["rule"], "_mapped", vec["q_rule"])
                attack_arm("LoRA+RAG-rule", 1, SYS_HYBRID, "rule")
            reg = float(np.mean(c.regression_score(c.generate_many(model, tok, reg_prompts, 8), reg_gold)))
        c.write_jsonl(runs_path, [{"arm": arm, "plan": plan, "n_pairs": len(pairs), "steps": stats["steps"],
                                   "train_tokens": stats["train_tokens"], "train_seconds": stats["wall_seconds"], "regression": reg,
                                   "loss_last": stats["loss_log"][-5:]}], "a")
        print(arm, "reg", reg, stats["wall_seconds"], flush=True)
        model = c.free_adapter(model)
    print("done", args.tag, flush=True)


if __name__ == "__main__":
    main()
