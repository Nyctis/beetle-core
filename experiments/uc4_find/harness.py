"""UC4 harness: search and read tools, a batched agent loop, one log row per question.

Runs on the GPU node in the same transformers process as the model, like the other
experiments. Thinking stays off. Special tokens are kept when decoding, because the
tool syntax is made of them. Every open episode advances together, one model call
per round, so a round's batch mixes questions at the same depth.
"""

from __future__ import annotations

import contextlib
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent)]
import common as c  # noqa: E402

MAX_ROUNDS = 6
READ_CAP = 800  # tokens returned by read
SNIPPET_CHARS = 200
TOOL_RESPONSE_ID = 50  # <|tool_response>: the model stops here after a call
TURN_END_ID = 106  # <turn|>: the model stops here after a final answer
STOP_IDS = {1, TURN_END_ID, TOOL_RESPONSE_ID}
TOOL_MARK = "\x1e"  # prefixes a training question whose prompt declares the tools

# Kept short: this prompt plus tools costs 169 tokens, and a training pair must fit in 256.
SYSTEM = ("You are the internal assistant of Harbor Mutual. Documents are only reachable through the tools. Answer only from "
          "documents you have read, in one short sentence. If no stored document covers the question, reply exactly: UNKNOWN")
CATALOG = "\n\nEvery stored document, by doc_id:\n{catalog}"
# B1's prompts. P0 lost every question to an immediate UNKNOWN, so a variant is chosen on training questions.
B1_PROMPTS = {
    "P0": SYSTEM + CATALOG,
    "P1": SYSTEM + CATALOG + "\n\nAlways use a tool before you answer.",
    "P2": ("You are the internal assistant of Harbor Mutual." + CATALOG + "\n\nDocuments are only reachable through the tools. Read the "
           "document most likely to hold the answer, using its exact doc_id from the list, or search if none fits. Answer only from "
           "documents you have read, in one short sentence. If no stored document covers the question, reply exactly: UNKNOWN"),
}
SYS_RAG = ("You are the internal assistant of Harbor Mutual. Answer in one short sentence using the documents below. If they do "
           "not contain the answer, reply exactly: UNKNOWN\n\nDocuments:\n{ctx}")

TOOLS = [
    {"type": "function", "function": {
        "name": "search",
        "description": "Search the stored documents. Returns doc_id, title and the opening text of each hit.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "k": {"type": "integer"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "read",
        "description": "Read one document by its exact doc_id.",
        "parameters": {"type": "object", "properties": {"doc_id": {"type": "string"}}, "required": ["doc_id"]}}},
]

CALL = re.compile(r'<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>', re.S)
ARG = re.compile(r'(\w+):(?:<\|"\|>(.*?)<\|"\|>|([^,{}]+))', re.S)
CHANNEL = re.compile(r'<\|channel>.*?<channel\|>', re.S)
# Text that tries to be a tool call but is not one the harness can run.
ATTEMPT = re.compile(r'<\|tool_call>|<tool_call\|>|\bcall:\w+\{|"name"\s*:\s*"(?:search|read)"|'
                     r'\b(?:search|read)\(\s*["\'{a-z]|\bdoc_id\s*[:=]', re.I)


def call_text(name: str, args: dict) -> str:
    """One tool call exactly as the chat template renders it; keys are sorted like its dictsort."""
    body = ",".join(f'{k}:<|"|>{v}<|"|>' if isinstance(v, str) else f"{k}:{v}" for k, v in sorted(args.items()))
    return f"<|tool_call>call:{name}{{{body}}}<tool_call|>"


def parse_args(body: str) -> dict:
    args: dict = {}
    for key, quoted, bare in ARG.findall(body):
        if bare:
            bare = bare.strip()
            args[key] = int(bare) if bare.lstrip("-").isdigit() else bare
        else:
            args[key] = quoted
    return args


def valid(name: str, args: dict) -> bool:
    if name == "search":
        k = args.get("k", 5)
        return isinstance(args.get("query"), str) and bool(args["query"].strip()) and isinstance(k, int) and not isinstance(k, bool)
    return name == "read" and isinstance(args.get("doc_id"), str) and bool(args["doc_id"].strip())


@dataclass
class Turn:
    kind: str  # "call", "answer" or "bad"
    calls: list = field(default_factory=list)
    text: str = ""  # what the model said outside the calls
    attempted: bool = False


def parse_turn(raw: str) -> Turn:
    raw = CHANNEL.sub("", raw)
    found = [(name, parse_args(body)) for name, body in CALL.findall(raw)]
    rest = CALL.sub("", raw).strip()
    stray = bool(ATTEMPT.search(rest))
    if found:
        calls = [{"name": n, "args": a} for n, a in found]
        kind = "call" if not stray and all(valid(n, a) for n, a in found) else "bad"
        return Turn(kind, calls, rest, True)
    if stray:
        return Turn("bad", [], rest, True)
    return Turn("answer", [], rest, False)


def first_messages(system: str, user: str) -> list[dict]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def render(tok, messages: list[dict]) -> str:
    return tok.apply_chat_template(messages, tools=TOOLS, tokenize=False, add_generation_prompt=True)


# ---------------------------------------------------------------- trainer patch


def tool_q(system: str, user: str) -> str:
    return f"{TOOL_MARK}{system}{c.SEP}{user}"


def patch_trainer(te) -> None:
    """Teach the sandbox trainer tool prompts and per-target stop tokens, from outside it.

    A call target ends at <|tool_response>, id 50, where generation stops. A final answer
    ends at <turn|>, id 106. The loss covers model tokens only, and a pair that would be cut
    raises, since truncation would silently drop its target.
    """

    def prompt_text(tok, question: str) -> str:
        if question.startswith(TOOL_MARK):
            system, user = question[len(TOOL_MARK):].split(c.SEP, 1)
            return render(tok, first_messages(system, user))
        return c.chat_prompt(tok, *c.split_q(question))

    def encode(tok, question: str, answer: str, max_len: int):
        prompt_ids = tok(prompt_text(tok, question), add_special_tokens=False)["input_ids"]
        stop = "<|tool_response>" if answer.endswith("<tool_call|>") else "<turn|>"
        answer_ids = tok(answer + stop, add_special_tokens=False)["input_ids"]
        if len(prompt_ids) + len(answer_ids) > max_len:
            raise ValueError(f"pair of {len(prompt_ids) + len(answer_ids)} tokens exceeds max_len {max_len}")
        return prompt_ids + answer_ids, [-100] * len(prompt_ids) + answer_ids

    te.prompt_text, te.encode = prompt_text, encode


# ---------------------------------------------------------------- tools


class Store:
    """The document store behind the tools. `index_texts` lets B2 add questions to BM25 only."""

    def __init__(self, docs: list[dict], vecs, tok, index_texts: list[str] | None = None):
        self.docs, self.tok = docs, tok
        self.by_id = {d["doc_id"]: i for i, d in enumerate(docs)}
        self.retriever = c.Retriever(index_texts or [d["text"] for d in docs], vecs)
        self.dense = vecs is not None
        self.qvec: dict = {}

    def embed(self, queries: list[str]) -> None:
        new = sorted({q for q in queries if q not in self.qvec})
        if new and self.dense:
            self.qvec.update(zip(new, c.embed_texts(new, "query")))

    def rank(self, query: str, k: int) -> list[int]:
        self.embed([query])
        return self.retriever.search(query, self.qvec.get(query), k, "rrf")

    def search(self, query: str, k: int = 5) -> tuple[str, list[str]]:
        hits = [self.docs[i] for i in self.rank(query, max(1, min(k, 10)))]
        blocks = []
        for d in hits:
            snippet = " ".join(d["text"][:SNIPPET_CHARS].split())
            blocks.append(f"doc_id: {d['doc_id']}\ntitle: {d['title']}\nsnippet: {snippet}")
        return "\n\n".join(blocks), [d["doc_id"] for d in hits]

    def read(self, doc_id: str) -> tuple[str, list[str], bool]:
        i = self.by_id.get(doc_id)
        if i is None:
            return f"ERROR: no document has doc_id {doc_id}. Use search to find the right doc_id.", [], False
        ids = self.tok(self.docs[i]["text"], add_special_tokens=False)["input_ids"]
        text = self.docs[i]["text"] if len(ids) <= READ_CAP else self.tok.decode(ids[:READ_CAP])
        return text, [doc_id], True


# ---------------------------------------------------------------- generation


@dataclass
class Gen:
    text: str
    n_in: int
    n_out: int
    stop: int | None
    secs: float


def generate_raw(model, tok, prompts: list[str], max_new: int = 128, max_batch: int = 16, max_padded: int = 16000,
                 ) -> list[Gen]:
    """Greedy generation that keeps special tokens. Batches are cut by padded tokens."""
    import torch

    lens = c.count_tokens(tok, prompts)
    order = sorted(range(len(prompts)), key=lambda i: lens[i])
    out: list = [None] * len(prompts)
    at = 0
    while at < len(order):
        batch = [order[at]]
        at += 1
        while at < len(order) and len(batch) < max_batch and lens[order[at]] * (len(batch) + 1) <= max_padded:
            batch.append(order[at])
            at += 1
        t0 = time.time()
        x = tok([prompts[i] for i in batch], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        for attempt in range(60):  # someone else may hold the card: wait, don't fight
            try:
                with torch.no_grad():
                    y = model.generate(**x, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"OOM at {len(batch)} prompts, waiting 60 s ({attempt + 1})", flush=True)
                time.sleep(60)
        else:
            raise RuntimeError("the GPU stayed full for an hour")
        share = (time.time() - t0) / len(batch)
        for i, row in zip(batch, y[:, x["input_ids"].shape[1]:].tolist()):
            ids, stop = [], None
            for t in row:
                if t in STOP_IDS:
                    stop = t
                    break
                ids.append(t)
            out[i] = Gen(tok.decode(ids, skip_special_tokens=False), lens[i], len(ids), stop, share)
    return out


# ---------------------------------------------------------------- the loop


@dataclass
class Episode:
    test: dict
    system: str
    arm: str
    seed: int = 0
    messages: list = field(default_factory=list)
    calls: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    rounds: int = 0
    tokens_read: int = 0
    prompt_tokens: int = 0
    gen_tokens: int = 0
    wall: float = 0.0
    first_gold_round: int | None = None
    first_gold_read_round: int | None = None
    tokens_to_first_gold: int | None = None
    first_hit: int | None = None
    fabricated: int = 0
    reads: int = 0
    attempts: int = 0
    parsed: int = 0
    stop_reason: str = ""
    answer: str = ""

    def __post_init__(self):
        if not self.messages:
            self.messages = first_messages(self.system, self.test["q"])


def run_tool(store: Store, tok, e: Episode, call: dict, rnd: int) -> tuple[str, str]:
    name, args = call["name"], call["args"]
    if name == "search":
        text, returned = store.search(args["query"], args.get("k", 5))
        ok = True
    else:
        text, returned, ok = store.read(args["doc_id"])
        e.reads += 1
        e.fabricated += not ok
    n = c.count_tokens(tok, [text])[0]
    e.tokens_read += n
    gold = e.test.get("doc_id")
    if not e.calls and gold is not None:
        e.first_hit = int(bool(returned) and returned[0] == gold)
    if gold in returned:
        if e.first_gold_round is None:
            e.first_gold_round, e.tokens_to_first_gold = rnd, e.tokens_read
        if name == "read" and e.first_gold_read_round is None:
            e.first_gold_read_round = rnd
    e.calls.append({"round": rnd, "name": name, "args": args, "ok": ok, "returned_doc_ids": returned, "tokens_result": n})
    return name, text


def step(tok, store: Store, e: Episode, g: Gen, rnd: int) -> bool:
    """Apply one model output to an episode. True when the episode goes on."""
    e.rounds = rnd
    e.prompt_tokens += g.n_in
    e.gen_tokens += g.n_out
    e.wall += g.secs
    e.outputs.append(g.text[:1500])
    t = parse_turn(g.text)
    e.attempts += t.attempted
    e.parsed += t.kind == "call"
    if t.kind == "answer":
        e.answer = t.text
        e.stop_reason = "answer" if g.stop is not None else "max_tokens"
        return False
    if t.kind == "bad":
        e.stop_reason = "parse_error"
        return False
    t0 = time.time()
    done = [run_tool(store, tok, e, call, rnd) for call in t.calls]
    e.wall += time.time() - t0
    message: dict = {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": x["name"], "arguments": x["args"]}}
                                                         for x in t.calls]}
    if t.text:
        message["content"] = t.text
    e.messages.append(message)
    e.messages += [{"role": "tool", "name": name, "content": text} for name, text in done]
    if rnd == MAX_ROUNDS:
        e.stop_reason = "max_rounds"
        return False
    return True


def run(model, tok, store: Store, eps: list[Episode], first_turn_only: bool = False, **gen_args) -> list[Episode]:
    """Advance every episode until it answers, breaks or hits the round cap.

    `first_turn_only` keeps the adapter on for the first model call and off afterwards."""
    alive = list(eps)
    for rnd in range(1, MAX_ROUNDS + 1):
        if not alive:
            break
        prompts = [render(tok, e.messages) for e in alive]
        scope = model.disable_adapter() if first_turn_only and rnd > 1 else contextlib.nullcontext()
        with scope:
            outs = generate_raw(model, tok, prompts, **gen_args)
        queries = [x["args"]["query"] for g in outs for x in parse_turn(g.text).calls if x["name"] == "search" and "query" in x["args"]]
        t0 = time.time()
        store.embed([q for q in queries if isinstance(q, str)])
        embed_share = (time.time() - t0) / len(alive)
        for e in alive:
            e.wall += embed_share
        alive = [e for e, g in zip(alive, outs) if step(tok, store, e, g, rnd)]
        print(f"round {rnd}: {len(alive)} open", flush=True)
    return eps


def row(e: Episode, score: int) -> dict:
    t = e.test
    return {"arm": e.arm, "seed": e.seed, "qid": t["qid"], "part": t["part"], "near": t.get("near", False), "attr": t.get("attr"),
            "doc_id": t.get("doc_id"), "q": t["q"], "rounds": e.rounds, "calls": e.calls,
            "returned_doc_ids": [d for x in e.calls for d in x["returned_doc_ids"]], "tokens_read": e.tokens_read,
            "prompt_tokens_total": e.prompt_tokens, "gen_tokens_total": e.gen_tokens, "wall_s": round(e.wall, 3),
            "first_gold_round": e.first_gold_round, "first_gold_read_round": e.first_gold_read_round,
            "tokens_to_first_gold": e.tokens_to_first_gold, "first_hit": e.first_hit, "reads": e.reads, "fabricated": e.fabricated,
            "attempts": e.attempts, "parsed": e.parsed, "stop_reason": e.stop_reason, "answer": e.answer, "score": score,
            "outputs": e.outputs}


def score_answer(t: dict, answer: str) -> int:
    """Stored documents are scored like UC3. For an absent document only a refusal is right."""
    if not answer:
        return 0
    if t["part"].startswith("absent"):
        return int(bool(c.REFUSAL.search(answer)))
    return int(not c.REFUSAL.search(answer) and c.contains_any(answer, t["gold"]))
