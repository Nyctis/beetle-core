"""UC4 experiment: can an adapter learn where a user's documents are?

  smoke     base Bolt on 20 questions, then a leftover adapter, both with the tools declared
  precheck  retrieval R@1 of the raw test questions, no model involved
  eval      run arms over the 320 test questions, one log row per question
  train     train one adapter, save it, run the operational check and the regression

Usage: python run.py eval --arms B0,B1,B2,B3 --tag final
       python run.py train --name L1_s1 --seed 1
       python run.py eval --arms L1_s1 --adapter L1_s1 --tag final
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import common as c  # noqa: E402
import harness as h  # noqa: E402

DATA = HERE / "data"
RESULTS = HERE / "results"
ADAPTERS = c.WORK / "adapters/uc4"
GPU_CAP_GB = 26  # inference; with the matrix's 24 GB wait this leaves room on the card
TRAIN_CAP_GB = 44  # a batch of 8 at 240 tokens needs over 34 GB: logits alone take 1.7 GB per pass. Runs last minutes


def gpu_free_gb() -> float:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return int(out.stdout.split()[0]) / 1024


def wait_gpu(gb: float) -> None:
    while (free := gpu_free_gb()) < gb:
        print(f"{free:.1f} GB free, need {gb}; waiting 60 s", flush=True)
        time.sleep(60)


def setup(need_gb: float = GPU_CAP_GB + 1, cap: float = GPU_CAP_GB):
    import torch

    wait_gpu(need_gb)
    torch.cuda.set_per_process_memory_fraction(cap / (torch.cuda.get_device_properties(0).total_memory / 2**30))
    te = c.load_trainer()
    h.patch_trainer(te)
    with c.timed("uc4/load"):
        tok, model = c.load_model(te, "bolt")
    return te, tok, model


def load_world(tok=None):
    corpus = json.loads((DATA / "corpus.json").read_text())
    stored = [d for d in corpus if d["corpus"] == "final"]  # the order of the cached vectors
    tests = json.loads((DATA / "tests.json").read_text())
    train_q = json.loads((DATA / "train_q.json").read_text())
    vecs = np.load(c.WORK / "cache/uc3_final.npz")["raw"]
    assert len(vecs) == len(stored) == 300
    return corpus, stored, tests, train_q, vecs


def b2_index(stored: list[dict], train_q: dict) -> list[str]:
    """B2 writes the training questions into BM25. Held-out documents have none, as for the adapter."""
    return [d["text"] + "\n" + "\n".join(x["q"] for x in train_q.get(d["doc_id"], []) if d["part"] == "seen") for d in stored]


def b1_system(stored: list[dict], name: str) -> str:
    return h.B1_PROMPTS[name].format(catalog="\n".join(sorted(d["doc_id"] for d in stored)))


def regression(model, tok, tag: str, arm: str, tools: bool) -> float:
    """The shared MCQ set, read at 8 tokens as before; every output is saved."""
    items = json.loads(c.REG_FILE.read_text())
    prompts = [h.render(tok, h.first_messages(h.SYSTEM, x["q"])) if tools else c.chat_prompt(tok, None, x["q"]) for x in items]
    outs = h.generate_raw(model, tok, prompts, max_new=64)
    heads = [tok.decode(tok(g.text, add_special_tokens=False)["input_ids"][:8], skip_special_tokens=True).strip() for g in outs]
    scores = c.regression_score(heads, [x["a"] for x in items])
    c.write_jsonl(RESULTS / f"regression_{tag}.jsonl", [{"arm": arm, "tools": tools, "i": i, "output": g.text, "score": s, "called": h.parse_turn(g.text).attempted}
                                                        for i, (g, s) in enumerate(zip(outs, scores))], "a")
    return float(np.mean(scores))


def score_rows(eps: list[h.Episode]) -> list[dict]:
    return [h.row(e, h.score_answer(e.test, e.answer)) for e in eps]


def run_rag(model, tok, store: h.Store, tests: list[dict], arm: str) -> list[dict]:
    """B3: one-shot top-5 pasted into the prompt, the same reader as UC3."""
    store.embed([t["q"] for t in tests])
    picks = [store.rank(t["q"], 5) for t in tests]
    ctx = ["\n---\n".join(store.docs[i]["text"] for i in ids) for ids in picks]
    prompts = [c.chat_prompt(tok, h.SYS_RAG.format(ctx=x), t["q"]) for x, t in zip(ctx, tests)]
    outs = h.generate_raw(model, tok, prompts, max_new=128)
    n_ctx = c.count_tokens(tok, ctx)
    rows = []
    for t, ids, n, g in zip(tests, picks, n_ctx, outs):
        e = h.Episode(t, "", arm)
        e.rounds, e.prompt_tokens, e.gen_tokens, e.wall, e.tokens_read = 1, g.n_in, g.n_out, g.secs, n
        e.outputs = [g.text[:1500]]
        returned = [store.docs[i]["doc_id"] for i in ids]
        e.calls = [{"round": 1, "name": "rag", "args": {"query": t["q"]}, "ok": True, "returned_doc_ids": returned, "tokens_result": n}]
        gold = t.get("doc_id")
        if gold is not None:
            e.first_hit = int(returned[0] == gold)
            if gold in returned:
                e.first_gold_round, e.first_gold_read_round, e.tokens_to_first_gold = 1, 1, n
        turn = h.parse_turn(g.text)
        e.answer = turn.text
        e.stop_reason = "answer" if g.stop is not None else "max_tokens"
        rows.append(h.row(e, h.score_answer(t, e.answer)))
    return rows


def run_arm(model, tok, store: h.Store, tests: list[dict], arm: str, system: str, seed: int = 0, first_turn_only: bool = False,
            **gen_args) -> list[dict]:
    eps = [h.Episode(t, system, arm, seed) for t in tests]
    h.run(model, tok, store, eps, first_turn_only=first_turn_only, **gen_args)
    return score_rows(eps)


def summary(rows: list[dict]) -> str:
    n = len(rows)
    hit = [r["first_hit"] for r in rows if r["first_hit"] is not None]
    attempts, parsed = sum(r["attempts"] for r in rows), sum(r["parsed"] for r in rows)
    stops: dict = {}
    for r in rows:
        stops[r["stop_reason"]] = stops.get(r["stop_reason"], 0) + 1
    called = sum(bool(r["calls"]) for r in rows)
    return (f"n={n} first_hit={np.mean(hit) if hit else 0:.3f} acc={np.mean([r['score'] for r in rows]):.3f} rounds={np.mean([r['rounds'] for r in rows]):.2f} "
            f"called={called / n:.3f} parse={parsed / max(attempts, 1):.3f} read_tok={np.mean([r['tokens_read'] for r in rows]):.0f} "
            f"input_tok={np.mean([r['prompt_tokens_total'] for r in rows]):.0f} wall={np.mean([r['wall_s'] for r in rows]):.2f} stops={stops}")


def cmd_smoke(a) -> None:
    from peft import PeftModel

    te, tok, model = setup(14)
    b = json.loads((HERE.parent / "uc3_documents/data/dev.json").read_text())
    docs = [{"doc_id": f"{d['domain'].lower().split()[0]}/{d['type'].lower().replace(' ', '-')}-{d['file'].split('-')[1]}",
             "title": d["type"], "text": t} for d, t in zip(b["docs"], b["variants"]["raw"])]
    vecs = np.load(c.WORK / "cache/uc3_devcheck.npz")["raw"]
    store = h.Store(docs, vecs, tok)
    tests = []
    for t in [x for x in b["tests"] if x["task"] == "U1"][:a.n]:
        tests.append({"qid": t["qid"], "q": t["q"], "part": "seen", "doc_id": docs[t["doc"]]["doc_id"], "gold": t["gold"]})
    with c.timed("uc4/smoke", arm="base"):
        rows = run_arm(model, tok, store, tests, "smoke_base", h.SYSTEM)
    print("BASE", summary(rows), flush=True)
    for r in rows[:6]:
        print(json.dumps({k: r[k] for k in ("q", "outputs", "answer", "score", "stop_reason")}, ensure_ascii=False)[:900], flush=True)
    c.write_jsonl(RESULTS / "smoke.jsonl", rows)
    peft = PeftModel.from_pretrained(model, str(c.WORK / "adapters/tmp"))
    peft.eval()
    with c.timed("uc4/smoke", arm="r3"):
        rows = run_arm(peft, tok, store, tests, "smoke_r3", h.SYSTEM)
    print("R3", summary(rows), flush=True)
    for r in rows[:4]:
        print(json.dumps({k: r[k] for k in ("q", "outputs", "answer", "score", "stop_reason")}, ensure_ascii=False)[:900], flush=True)
    c.write_jsonl(RESULTS / "smoke.jsonl", rows, "a")
    print("done", flush=True)


def cmd_precheck(a) -> None:
    import torch

    wait_gpu(6)
    torch.cuda.set_per_process_memory_fraction(GPU_CAP_GB / 80)
    corpus, stored, tests, train_q, vecs = load_world()

    class Plain:
        def __call__(self, t, add_special_tokens=False):
            return {"input_ids": t.split()}

    plain, expanded = h.Store(stored, vecs, Plain()), h.Store(stored, vecs, Plain(), b2_index(stored, train_q))
    present = [t for t in tests if t["doc_id"]]
    with c.timed("uc4/precheck"):
        plain.embed([t["q"] for t in present])
    expanded.qvec = plain.qvec
    c.free_embedder()

    def rate(store, part, k):
        rows = [t for t in present if t["part"] == part]
        ids = [store.docs[i]["doc_id"] for t in rows for i in store.rank(t["q"], k)]
        return sum(t["doc_id"] in ids[i * k:(i + 1) * k] for i, t in enumerate(rows)) / len(rows)

    for part in ("seen", "heldout"):
        print(part, {f"B0 R@{k}": round(rate(plain, part, k), 3) for k in (1, 5)}, {f"B2 R@{k}": round(rate(expanded, part, k), 3) for k in (1, 5)})
    bm = [int(plain.retriever.search(t["q"], None, 1, "bm25")[0] == plain.by_id[t["doc_id"]]) for t in present]
    print("BM25 alone R@1", round(float(np.mean(bm)), 3))
    print("B0 R@1 over all stored", round(sum(rate(plain, p, 1) * sum(t["part"] == p for t in present) for p in ("seen", "heldout")) / len(present), 3))


def cmd_eval(a) -> None:
    from peft import PeftModel

    te, tok, model = setup()
    corpus, stored, tests, train_q, vecs = load_world()
    if a.tests != "tests.json":
        tests = json.loads((DATA / a.tests).read_text())
    plain = h.Store(stored, vecs, tok)
    preds = RESULTS / f"eps_{a.tag}.jsonl"
    serial_tests = random.Random(7).sample([t for t in tests if t["doc_id"]], a.serial) if a.serial else []

    def one(arm: str, net) -> None:
        store = h.Store(stored, vecs, tok, b2_index(stored, train_q)) if arm == "B2" else plain
        system = b1_system(stored, a.b1_prompt) if arm == "B1" else h.SYSTEM
        gen = {"max_padded": 24000} if arm == "B1" else {}
        kw = {"first_turn_only": a.first_turn_only, **gen}
        seed = int(arm.rsplit("_s", 1)[1]) if "_s" in arm else 0
        with c.timed("uc4/eval", arm=arm, tag=a.tag):
            rows = run_rag(net, tok, store, tests, arm) if arm == "B3" else run_arm(net, tok, store, tests, arm, system, seed, **kw)
        c.write_jsonl(preds, rows, "a")
        print(arm, summary(rows), flush=True)
        if serial_tests:
            with c.timed("uc4/eval_serial", arm=arm, tag=a.tag):
                if arm == "B3":
                    s_rows = []
                    for t in serial_tests:
                        s_rows += run_rag(net, tok, store, [t], arm)
                else:
                    s_rows = run_arm(net, tok, store, serial_tests, arm, system, seed, max_batch=1, first_turn_only=a.first_turn_only)
            c.write_jsonl(RESULTS / f"serial_{a.tag}.jsonl", s_rows, "a")
            print(arm, "serial", summary(s_rows), flush=True)

    if a.adapter:
        net = PeftModel.from_pretrained(model, str(ADAPTERS / a.adapter))
        net.eval()
    else:
        net = model
    for arm in a.arms.split(","):
        one(arm, net)
    if "B0" in a.arms.split(","):
        with c.timed("uc4/regression", arm="B0"):
            print("regression", regression(net, tok, a.tag, "B0", False), regression(net, tok, a.tag, "B0", True), flush=True)
    print("done", flush=True)


def cmd_tune_b1(a) -> None:
    """Pick B1's prompt on training questions, which are not test questions."""
    te, tok, model = setup()
    corpus, stored, tests, train_q, vecs = load_world()
    store = h.Store(stored, vecs, tok)
    r = random.Random(41)
    seen = [d for d in corpus if d["part"] == "seen" and train_q.get(d["doc_id"])]
    rows = [{"qid": f"tune-{i}", "q": r.choice(train_q[d["doc_id"]])["q"], "part": "seen", "doc_id": d["doc_id"],
             "gold": [str(d["window"]), d["team"]]} for i, d in enumerate(r.sample(seen, 40))]
    absent = json.loads((DATA / "absent_targets.json").read_text())
    rows += [{"qid": f"tune-a{i}", "q": x["q"], "part": "absent_test", "doc_id": None, "gold": []} for i, x in enumerate(r.sample(absent, 10))]
    for name in h.B1_PROMPTS:
        with c.timed("uc4/tune_b1", arm=name):
            out = run_arm(model, tok, store, rows, f"B1_{name}", b1_system(stored, name), max_padded=24000)
        print(name, summary(out), "seen hit", np.mean([x["first_hit"] or 0 for x in out if x["doc_id"]]), flush=True)
    print("done", flush=True)


def build_pairs(corpus: list[dict], train_q: dict, seed: int, shuffle_labels: bool) -> tuple[list, dict]:
    """2,400 read targets, 240 search targets, 360 replay items. No answers, no refusals."""
    r = random.Random(seed * 1000 + 11)
    seen = [d["doc_id"] for d in corpus if d["part"] == "seen"]
    label = {x: x for x in seen}
    if shuffle_labels:  # placebo: a doc's questions all point at one other document
        perm = seen[:]
        while any(p == q for p, q in zip(seen, perm)):
            r.shuffle(perm)
        label = dict(zip(seen, perm))
    kinds: dict = {"read": [], "search": [], "replay": []}
    for doc in seen:
        kinds["read"] += [(h.tool_q(h.SYSTEM, x["q"]), h.call_text("read", {"doc_id": label[doc]}), doc) for x in train_q[doc]]
    for x in json.loads((DATA / "absent_targets.json").read_text()):
        kinds["search"].append((h.tool_q(h.SYSTEM, x["q"]), h.call_text("search", x["args"]), None))
    replay = json.loads((DATA / "replay.json").read_text())
    by_kind: dict = {}
    for x in replay:
        by_kind.setdefault(x["kind"], []).append(x)
    pick = list(by_kind.get("mcq", [])) + list(by_kind.get("mcq_tools", []))  # format items go first
    rest = [x for k, v in sorted(by_kind.items()) if not k.startswith("mcq") for x in v]
    r.shuffle(rest)
    pick = (pick + rest)[:360]
    kinds["replay"] = [(x["q"], x["a"], None) for x in pick]
    pairs = [(q, a) for v in kinds.values() for q, a, _ in v]
    r.shuffle(pairs)
    return pairs, kinds


def cmd_train(a) -> None:
    import torch

    te, tok, model = setup(TRAIN_CAP_GB + 1, TRAIN_CAP_GB)
    corpus, stored, tests, train_q, vecs = load_world()
    pairs, kinds = build_pairs(corpus, train_q, a.seed, a.shuffle)
    print({k: len(v) for k, v in kinds.items()}, len(pairs), flush=True)
    rec = c.recipe("bolt", seed=a.seed, epochs=a.epochs, max_seq_len=256, batch_size=8, lr=a.lr)
    r = random.Random(a.seed)
    sample = {k: [(q, t) for q, t, _ in r.sample(v, 16)] for k, v in kinds.items()}
    captured: dict = {}
    add_lora = te.add_lora

    def add(m, recipe):
        captured["m"] = add_lora(m, recipe)
        return captured["m"]

    te.add_lora = add
    kind_loss = []

    def log(msg: str) -> None:
        print(a.name, msg, flush=True)
        if msg.startswith("epoch"):
            row = {"msg": msg, **{k: round(te.answer_loss(captured["m"], tok, v, 256), 4) for k, v in sample.items()}}
            kind_loss.append(row)
            print(row, flush=True)

    model, stats = c.train(te, model, tok, pairs, rec, log)
    c.ledger("uc4/train", stats["wall_seconds"], arm=a.name, seed=a.seed, shuffle=a.shuffle)
    size = c.adapter_bytes(model, ADAPTERS / a.name)
    print("saved", size, flush=True)
    with c.timed("uc4/opcheck", arm=a.name):
        check = op_check(model, tok, kinds, a.seed)
        reg, reg_tools = regression(model, tok, "final", a.name, False), regression(model, tok, "final", a.name, True)
    print(a.name, check, "regression", reg, "with tools", reg_tools, flush=True)
    c.write_jsonl(RESULTS / "runs_final.jsonl", [{"arm": a.name, "seed": a.seed, "shuffle": a.shuffle, "n_pairs": len(pairs), "lr": a.lr,
                                                  "epochs": a.epochs, "steps": stats["steps"], "train_tokens": stats["train_tokens"],
                                                  "train_seconds": stats["wall_seconds"], "adapter_bytes": size, "op_check": check,
                                                  "regression": reg, "regression_tools": reg_tools, "kind_loss": kind_loss,
                                                  "loss_tail": stats["loss_log"][-5:],
                                                  "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}], "a")
    print("done", flush=True)


def op_check(model, tok, kinds: dict, seed: int) -> dict:
    """First call on trained questions: read targets, and search targets too."""
    r = random.Random(seed + 99)
    reads = r.sample(kinds["read"], 300)
    outs = h.generate_raw(model, tok, [render_pair(tok, q) for q, _, _ in reads], max_new=64)
    hit = sum(g.text.strip() == t for (_, t, _), g in zip(reads, outs))
    searches = kinds["search"]
    outs2 = h.generate_raw(model, tok, [render_pair(tok, q) for q, _, _ in searches], max_new=64)
    searched = sum(any(x["name"] == "search" for x in h.parse_turn(g.text).calls) for g in outs2)
    return {"read_hit": round(hit / len(reads), 4), "n_read": len(reads), "search_rate": round(searched / len(searches), 4), "n_search": len(searches)}


def render_pair(tok, q: str) -> str:
    system, user = q[len(h.TOOL_MARK):].split(c.SEP, 1)
    return h.render(tok, h.first_messages(system, user))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("smoke")
    s.add_argument("--n", type=int, default=20)
    sub.add_parser("precheck")
    sub.add_parser("tune_b1")
    e = sub.add_parser("eval")
    e.add_argument("--arms", required=True)
    e.add_argument("--tag", default="final")
    e.add_argument("--adapter")
    e.add_argument("--first-turn-only", action="store_true")
    e.add_argument("--serial", type=int, default=40)
    e.add_argument("--b1-prompt", default="P0")
    e.add_argument("--tests", default="tests.json")
    t = sub.add_parser("train")
    t.add_argument("--name", required=True)
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--shuffle", action="store_true")
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--epochs", type=int, default=4)
    a = ap.parse_args()
    {"smoke": cmd_smoke, "precheck": cmd_precheck, "tune_b1": cmd_tune_b1, "eval": cmd_eval, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
