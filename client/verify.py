"""Did an adapter change anything? A server can report success and apply nothing:
vLLM loads keys it can't place and answers 200, so an adapter counts as working
only after the same questions got different answers with it than without it.

The questions are the buyer's own hidden ones. A check on a seller's server
shows them to that machine, so it runs only where the floor already allows the
data itself. A fixed probe set covers a job without any. Greedy decoding, so a
difference can't be sampling noise.
"""

from __future__ import annotations

from client.errors import no_effect

MAX_ITEMS = 6
PROBE_TOKENS = 32
PARAMS = {"temperature": 0.0, "seed": 0, "max_tokens": PROBE_TOKENS}
FIXED_PROBES = [{"question": q, "answer": ""} for q in (
    "In one short sentence, what do you remember about me?",
    "What should I keep in mind about myself this week?",
    "Name one personal fact you know about me.",
)]


def questions(hidden: list[dict] | None) -> list[dict]:
    return (hidden or FIXED_PROBES)[:MAX_ITEMS]


def compare(items: list[dict], base: list[str], tuned: list[str]) -> dict:
    hit = lambda text, it: bool(it["answer"]) and it["answer"].lower() in text.lower()  # noqa: E731
    return {"asked": len(items), "changed": sum(b != t for b, t in zip(base, tuned)),
            "base_correct": sum(hit(t, it) for t, it in zip(base, items)),
            "adapter_correct": sum(hit(t, it) for t, it in zip(tuned, items))}


def require_effect(result: dict, where: str) -> dict:
    if result["changed"] == 0:
        raise no_effect(f"the adapter changed nothing on {where}: "
                        f"{result['asked']} answers identical to the base model")
    return result
