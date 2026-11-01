"""Shared evaluation code for the Beetle use-case experiments.

Runs on the GPU node. Every model arm goes through `generate_many`, so arms
differ only in the adapter and the prompt. Training reuses the sandbox trainer
(`sandbox/train_eval.py`), so what is measured here is what the product runs.
"""

from __future__ import annotations

import gc
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent
HOME = Path(os.path.expanduser("~"))
WORK = HOME / "beetle-exp"
MODELS = HOME / "beetle/.cache/models"
BOLT_DIR = MODELS / "NVIDIA-Nemotron-3.5-Bolt-E4B-BF16"
QWEN_DIR = MODELS / "Qwen3.5-2B"
GPU_USD_PER_HOUR = 1.98
SEP = "\x1f"  # separates the system text from the user text in a training "question"

TARGETS = {
    "bolt": r".*language_model.*\.(q_proj|o_proj|gate_proj|up_proj|down_proj)",
    "qwen": r".*layers\.\d+\.(mlp\.(gate_proj|up_proj|down_proj)|linear_attn\.out_proj|self_attn\.o_proj)",
}
LRS = {"bolt": 5e-4, "qwen": 5e-4}


def recipe(model: str = "bolt", seed: int = 1, epochs: int = 8, max_seq_len: int = 256,
           batch_size: int = 8, lr: float | None = None) -> dict:
    """The verified recipe from common/job_defaults.py, with the scale in arguments."""
    return {
        "method": "memory",
        "lora": {"rank": 16, "alpha": 32, "targets": TARGETS[model], "dropout": 0.0},
        "quant": "none",
        "optim": {"lr": lr or LRS[model], "warmup_ratio": 0.05, "epochs": epochs,
                  "batch_size": batch_size, "weight_decay": 0.0},
        "max_seq_len": max_seq_len,
        "seed": seed,
        "checkpoint_every": 10**9,  # only the final hash; the Merkle log is not under test
    }


# ---------------------------------------------------------------- ledger


def ledger(stage: str, seconds: float, **extra) -> None:
    """Append GPU-occupying wall time; RESULTS.md totals this file."""
    WORK.mkdir(exist_ok=True)
    row = {"stage": stage, "seconds": round(seconds, 1),
           "usd": round(seconds / 3600 * GPU_USD_PER_HOUR, 4), "at": time.strftime("%FT%TZ", time.gmtime())}
    row.update(extra)
    with open(WORK / "ledger.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


class timed:
    def __init__(self, stage: str, **extra):
        self.stage, self.extra = stage, extra

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.seconds = time.time() - self.t0
        ledger(self.stage, self.seconds, **self.extra)


# ---------------------------------------------------------------- vLLM client (data generation only)


def vllm_chat(prompt: str, system: str | None = None, max_tokens: int = 400, temperature: float = 0.8,
              seed: int = 0, model: str = "bolt", url: str = "http://127.0.0.1:8000/v1/chat/completions",
              retries: int = 3) -> str:
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    body = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens,
                       "temperature": temperature, "seed": seed}).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.load(r)["choices"][0]["message"]["content"] or ""
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    return ""


def vllm_many(jobs: list[dict], workers: int = 8) -> list[str]:
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda j: vllm_chat(**j), jobs))


def extract_json(text: str):
    """The first JSON array or object in a model's reply, or None."""
    text = re.sub(r"```(?:json)?", "", text)
    for open_c, close_c in (("[", "]"), ("{", "}")):
        i, j = text.find(open_c), text.rfind(close_c)
        if i >= 0 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except json.JSONDecodeError:
                continue
    return None


# ---------------------------------------------------------------- model


def load_trainer():
    """The sandbox trainer, with prompts extended to carry a system turn."""
    sys.path[:0] = [str(WORK / "snapshot/common"), str(WORK / "snapshot/sandbox")]
    import train_eval as te

    def prompt_text(tok, question: str) -> str:
        return chat_prompt(tok, *split_q(question))

    te.prompt_text = prompt_text
    return te


def chat_prompt(tok, system: str | None, user: str) -> str:
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def join_q(system: str | None, user: str) -> str:
    return f"{system}{SEP}{user}" if system else user


def split_q(q: str) -> tuple[str | None, str]:
    return tuple(q.split(SEP, 1)) if SEP in q else (None, q)


def load_model(te, name: str = "bolt"):
    return te.load_base(BOLT_DIR if name == "bolt" else QWEN_DIR)


def generate_many(model, tok, prompts: list[str], max_new_tokens: int = 48, batch_size: int = 16,
                  temperature: float = 0.0, seed: int | None = None) -> list[str]:
    """Generate for rendered prompts. Sorted by length so batches pad little."""
    import torch

    if seed is not None:
        torch.manual_seed(seed)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out: list[str] = [""] * len(prompts)
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            x = tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                    add_special_tokens=False).to(model.device)
            gen = {"do_sample": True, "temperature": temperature, "top_p": 0.95} if temperature > 0 \
                else {"do_sample": False}
            y = model.generate(**x, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, **gen)
            for i, row in zip(idx, y[:, x["input_ids"].shape[1]:]):
                out[i] = tok.decode(row, skip_special_tokens=True).strip()
    return out


def count_tokens(tok, prompts: list[str]) -> list[int]:
    return [len(ids) for ids in tok(prompts, add_special_tokens=False)["input_ids"]]


def free_adapter(peft_model):
    """Drop the LoRA layers and return the clean base model."""
    import torch

    base = peft_model.unload()
    del peft_model
    gc.collect()
    torch.cuda.empty_cache()
    base.eval()
    return base


class _BucketedShuffle:
    """Stands in for the trainer's `random` module: same examples each epoch, same
    step count, but neighbours in a batch have similar lengths, so padding stops
    costing memory and time. The trainer itself is unchanged."""

    def __init__(self, lens: list[int], bs: int):
        self.lens, self.bs = lens, bs

    def Random(self, seed):  # noqa: N802 - mirrors random.Random
        import random

        self.r = random.Random(seed)
        return self

    def shuffle(self, order: list[int]) -> None:
        self.r.shuffle(order)
        span = self.bs * 50
        batches = []
        for i in range(0, len(order), span):
            chunk = sorted(order[i:i + span], key=lambda k: self.lens[k])
            batches += [chunk[j:j + self.bs] for j in range(0, len(chunk), self.bs)]
        self.r.shuffle(batches)
        order[:] = [k for b in batches for k in b]


def train(te, model, tok, pairs: list[tuple[str, str]], rec: dict, log=print, bucket: bool = False):
    budget = {"max_train_tokens": 10**9, "max_wall_seconds": 3 * 3600}
    if not bucket:
        return te.train_lora(model, tok, pairs, rec, budget, log)
    import random

    lens = [len(te.encode(tok, q, a, rec["max_seq_len"])[0]) for q, a in pairs]
    log(f"bucketed: n={len(lens)} max_len={max(lens)} mean_len={sum(lens) / len(lens):.0f}")
    te.random = _BucketedShuffle(lens, rec["optim"]["batch_size"])
    try:
        return te.train_lora(model, tok, pairs, rec, budget, log)
    finally:
        te.random = random


def adapter_bytes(peft_model, out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(out)
    return sum(p.stat().st_size for p in out.iterdir() if p.is_file())


# ---------------------------------------------------------------- retrieval

_word = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _word.findall(text.lower())


class Retriever:
    """BM25 and dense search over one corpus, fused with reciprocal rank."""

    def __init__(self, docs: list[str], doc_vecs: np.ndarray | None = None):
        from rank_bm25 import BM25Okapi

        self.docs = docs
        self.bm25 = BM25Okapi([tokenize(d) for d in docs])
        self.vecs = doc_vecs

    def bm25_rank(self, q: str) -> list[int]:
        return list(np.argsort(-self.bm25.get_scores(tokenize(q)), kind="stable"))

    def vec_rank(self, qvec: np.ndarray) -> list[int]:
        return list(np.argsort(-(self.vecs @ qvec), kind="stable"))

    def search(self, q: str, qvec: np.ndarray | None, k: int, mode: str = "rrf") -> list[int]:
        if mode == "bm25" or self.vecs is None or qvec is None:
            return [int(i) for i in self.bm25_rank(q)[:k]]
        if mode == "vec":
            return [int(i) for i in self.vec_rank(qvec)[:k]]
        score: dict[int, float] = {}
        for ranks in (self.bm25_rank(q), self.vec_rank(qvec)):
            for r, i in enumerate(ranks):
                score[int(i)] = score.get(int(i), 0.0) + 1 / (60 + r)
        return sorted(score, key=lambda i: (-score[i], i))[:k]


def embed_texts(texts: list[str], kind: str, batch_size: int = 32) -> np.ndarray:
    """Normalized embeddings, following the model card's transformers example:
    the prefix is the system turn, the pooling is an attention-masked mean.
    kind is "query" or "document". Text only, so no image processor is needed."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    global _embedder
    if _embedder is None:
        repo = "nvidia/Nemotron-3.5-Embed-1B-BF16"
        tok = AutoTokenizer.from_pretrained(repo)
        tok.padding_side = "right"
        _embedder = (tok, AutoModel.from_pretrained(repo, dtype=torch.bfloat16, device_map={"": "cuda:0"}).eval())
    tok, model = _embedder
    prefix = "query:" if kind == "query" else "passage:"
    rendered = [tok.apply_chat_template(
        [{"role": "system", "content": prefix}, {"role": "user", "content": [{"type": "text", "text": t}]}],
        tokenize=False) for t in texts]
    order = sorted(range(len(texts)), key=lambda i: len(rendered[i]))
    out = np.zeros((len(texts), model.config.text_config.hidden_size), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            x = tok([rendered[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False,
                    truncation=True, max_length=2048).to("cuda:0")
            hidden = model(**x).last_hidden_state
            mask = x["attention_mask"].unsqueeze(-1)
            vec = F.normalize(((hidden * mask).sum(1) / mask.sum(1)).float(), dim=-1)
            out[idx] = vec.cpu().numpy()
    return out


_embedder = None


def free_embedder() -> None:
    import torch

    global _embedder
    _embedder = None
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- scoring

REFUSAL = re.compile(
    r"\bunknown\b|don'?t (know|have|recall|remember)|do not (know|have|recall|remember)|not sure|"
    r"no (record|information|memory|mention|note)|not (mentioned|provided|in (the|my) (notes|memory))|"
    r"cannot (find|determine)|can'?t (find|determine)", re.I)


def norm(s: str) -> str:
    s = re.sub(r"[^a-z0-9:/.\- ]", " ", s.lower().replace("’", "'"))
    s = re.sub(r"[.:\-/]+(\s|$)", " ", s)  # sentence punctuation is not part of a value
    s = re.sub(r"(\d)(am|pm)\b", r"\1 \2", s)
    return re.sub(r"\s+", " ", s).strip()


def contains_any(text: str, keys: list[str]) -> bool:
    t = f" {norm(text)} "
    return any(f" {norm(k)} " in t or (len(norm(k)) >= 6 and norm(k) in t) for k in keys)


def first_mention(text: str, a: list[str], b: list[str]) -> bool:
    """True when an alias of `a` appears before any alias of `b`."""
    t = norm(text)
    pos = lambda keys: min((t.find(norm(k)) for k in keys if norm(k) in t), default=10**9)
    pa, pb = pos(a), pos(b)
    return pa < 10**9 and pa < pb


# ---------------------------------------------------------------- bootstrap


def bootstrap_diff(items: dict, n: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """CI of a mean paired difference, resampling items. items: id -> diff."""
    v = np.array(list(items.values()), dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cluster_bootstrap(rows: list[dict], metric, n: int = 10000, seed: int = 0, cluster: str = "uid",
                      task: str = "task") -> tuple[float, float, float]:
    """CI of `metric(task -> mean)` over tasks, resampling clusters, then rows within them.

    rows: dicts with the cluster key, the task key and a numeric "v".
    metric: maps {task: mean v} to a number, so macro averages resample correctly.
    """
    rng = np.random.default_rng(seed)
    groups: dict = {}
    for r in rows:
        groups.setdefault(r[cluster], {}).setdefault(r[task], []).append(r["v"])
    ids = sorted(groups)
    arrays = {c: {t: np.array(v, dtype=float) for t, v in groups[c].items()} for c in ids}
    tasks = sorted({t for c in ids for t in arrays[c]})

    def mean_of(picks) -> float:
        total = {t: 0.0 for t in tasks}
        count = {t: 0 for t in tasks}
        for c in picks:
            for t, a in arrays[c].items():
                sample = a[rng.integers(0, len(a), len(a))]
                total[t] += sample.sum()
                count[t] += len(sample)
        return metric({t: total[t] / count[t] for t in tasks if count[t]})

    point = metric({t: float(np.concatenate([arrays[c][t] for c in ids if t in arrays[c]]).mean())
                    for t in tasks})
    draws = [mean_of(rng.choice(ids, len(ids))) for _ in range(n)]
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def macro(means: dict) -> float:
    return float(np.mean(list(means.values())))


def verdict(point: float, lo: float, hi: float, mde: float = 0.05) -> str:
    """The pre-registered labels from CRITERIA.md section 1."""
    if point >= mde and lo > 0:
        return "win"
    if point <= -mde and hi < 0:
        return "loss"
    return "tie"


def fmt(point: float, lo: float, hi: float, scale: float = 100) -> str:
    return f"{point * scale:+.1f} [{lo * scale:+.1f}, {hi * scale:+.1f}]"


def write_jsonl(path: Path, rows: list[dict], mode: str = "w") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode) as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------- regression set


REG_FILE = EXP / "regression.json"


def regression_prompts(tok) -> tuple[list[str], list[str]]:
    items = json.loads(REG_FILE.read_text())
    return [chat_prompt(tok, None, it["q"]) for it in items], [it["a"] for it in items]


LETTER = re.compile(r"\b([A-J])\b")


def regression_score(outputs: list[str], gold: list[str]) -> list[int]:
    """Science MCQ: the first standalone option letter must match."""
    scores = []
    for o, g in zip(outputs, gold):
        m = LETTER.search(o)
        scores.append(int(bool(m) and m.group(1) == g))
    return scores
