"""Exploratory, not in any verdict: does RAG's T2 gap close when the reader may think?

Same retrieved notes, same P3 prompt, thinking on. Output is the text after the
model's thought channel. Tests T2 and T4 only.

Usage: python explore_thinking.py final02,final03,...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import common as c  # noqa: E402
import run as r  # noqa: E402
from score import score  # noqa: E402


def final_text(raw: str) -> str:
    for mark in ("<channel|>", "</think>"):
        if mark in raw:
            raw = raw.rsplit(mark, 1)[1]
    return raw.replace("<turn|>", "").replace("<eos>", "").replace("<pad>", "").strip()


def main() -> None:
    users = [json.loads((r.DATA / f"{u}.json").read_text()) for u in sys.argv[1].split(",")]
    te = c.load_trainer()
    tok, model = c.load_model(te, "bolt")
    rows = []
    with c.timed("explore/thinking"):
        for u in users:
            retrieved = r.retrieve_all(u, r.embed_user(u))["rrf"]
            prompts, tests = [], []
            for t, ids in zip(u["tests"], retrieved):
                if t["task"] in ("T2", "T4"):
                    system = r.RAG_P3.format(name=u["name"], notes=r.notes(u["memory"], ids))
                    prompts.append(tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": t["q"]}],
                                                           tokenize=False, add_generation_prompt=True, enable_thinking=True))
                    tests.append(t)
            order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
            raw = [""] * len(prompts)
            with torch.no_grad():
                for s in range(0, len(order), 8):
                    idx = order[s:s + 8]
                    x = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
                    y = model.generate(**x, max_new_tokens=320, do_sample=False, pad_token_id=tok.pad_token_id)
                    for i, row in zip(idx, y[:, x["input_ids"].shape[1]:]):
                        raw[i] = tok.decode(row, skip_special_tokens=False)
            for t, o in zip(tests, raw):
                ans = final_text(o)
                rows.append({"uid": u["uid"], "qid": t["qid"], "task": t["task"], "sub": t["sub"], "arm": "RAG_think", "seed": 0, "q": t["q"],
                             "output": ans, "raw_tail": o[-200:], "ptok": 0, "score": score(t, ans)})
            print(u["uid"], "done", flush=True)
    c.write_jsonl(r.RESULTS / "preds_explore_thinking.jsonl", rows)


if __name__ == "__main__":
    main()
