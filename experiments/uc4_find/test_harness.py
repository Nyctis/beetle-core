"""Unit tests for the UC4 harness. The tokenizer tests need the Bolt files on the GPU node.

Run on the node: cd ~/beetle-exp/code/uc4_find && python -m pytest -q test_harness.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path[:0] = [str(Path(__file__).resolve().parent.parent), str(Path(__file__).resolve().parent)]
import common as c  # noqa: E402
import harness as h  # noqa: E402

BOLT = c.BOLT_DIR


class FakeTok:
    """Counts words; enough for the tools and the loop."""

    def __call__(self, texts, add_special_tokens=False):
        one = isinstance(texts, str)
        ids = [t.split() for t in ([texts] if one else texts)]
        return {"input_ids": ids[0] if one else ids}

    def decode(self, ids, **_):
        return " ".join(ids)


DOCS = [{"doc_id": "legal/tenant-heating-dispute-1111", "title": "Tenant heating dispute", "text": "File F-1111.\nThe tenant has no heating in winter.\nInternal note: team Aurora."},
        {"doc_id": "health/knee-surgery-plan-2222", "title": "Knee surgery plan", "text": "File F-2222.\nA plan for knee surgery recovery.\nInternal note: team Basalt."},
        {"doc_id": "edu/bus-route-form-3333", "title": "Bus route form", "text": "File F-3333.\nA school bus route request.\nInternal note: team Cinder."}]


@pytest.fixture
def store():
    return h.Store(DOCS, None, FakeTok())


def test_parse_read():
    t = h.parse_turn(h.call_text("read", {"doc_id": "legal/a-1"}))
    assert t.kind == "call" and t.calls == [{"name": "read", "args": {"doc_id": "legal/a-1"}}]


def test_parse_search_with_k_and_brace_in_query():
    t = h.parse_turn(h.call_text("search", {"query": "a {b} c", "k": 3}))
    assert t.kind == "call" and t.calls[0]["args"] == {"query": "a {b} c", "k": 3}


def test_parse_two_calls():
    raw = h.call_text("search", {"query": "x"}) + h.call_text("read", {"doc_id": "y"})
    assert [x["name"] for x in h.parse_turn(raw).calls] == ["search", "read"]


def test_answer_and_empty_thought():
    assert h.parse_turn("<|channel>thought\n<channel|>Team Aurora.") == h.Turn("answer", [], "Team Aurora.", False)


@pytest.mark.parametrize("raw", ['<|tool_call>call:read{doc_id:<|"|>unterminated', 'read("legal/a")', '{"name": "search", "arguments": {}}',
                                 "call:read{doc_id:x}", h.call_text("grep", {"p": "x"}), '<|tool_call>call:read{}<tool_call|>',
                                 '<|tool_call>call:search{query:<|"|> <|"|>}<tool_call|>'])
def test_bad_turns(raw):
    t = h.parse_turn(raw)
    assert t.kind == "bad" and t.attempted


def test_search_and_read(store):
    text, ids = store.search("tenant heating", 2)
    assert ids[0] == DOCS[0]["doc_id"] and len(ids) == 2 and "title: Tenant heating dispute" in text
    assert store.read(DOCS[1]["doc_id"])[1:] == ([DOCS[1]["doc_id"]], True)
    text, ids, ok = store.read("legal/made-up-0000")
    assert text.startswith("ERROR") and ids == [] and not ok


def test_read_cap():
    long = {"doc_id": "a/b", "title": "t", "text": " ".join(["word"] * 2000)}
    assert len(h.Store([long], None, FakeTok()).read("a/b")[0].split()) == h.READ_CAP


def test_episode_log(store):
    """Fabricated ID, then a search that hits, then a read, then an answer."""
    tok = FakeTok()
    test = {"qid": "q1", "q": "who?", "part": "seen", "doc_id": DOCS[0]["doc_id"], "gold": ["Aurora"]}
    e = h.Episode(test, "sys", "T")
    turns = [h.call_text("read", {"doc_id": "legal/nope-1"}), h.call_text("search", {"query": "tenant heating"}),
             h.call_text("read", {"doc_id": DOCS[0]["doc_id"]}), "Team Aurora."]
    for rnd, raw in enumerate(turns, 1):
        more = h.step(tok, store, e, h.Gen(raw, 100, 10, h.TURN_END_ID, 0.5), rnd)
    assert not more and e.stop_reason == "answer" and e.rounds == 4 and e.first_hit == 0
    assert e.first_gold_round == 2 and e.first_gold_read_round == 3 and e.fabricated == 1 and e.reads == 2
    assert e.attempts == 3 and e.parsed == 3 and e.prompt_tokens == 400 and e.gen_tokens == 40 and 2.0 <= e.wall < 2.5
    r = h.row(e, h.score_answer(test, e.answer))
    assert r["score"] == 1 and r["tokens_to_first_gold"] < r["tokens_read"]
    assert {"rounds", "calls", "returned_doc_ids", "tokens_read", "prompt_tokens_total", "gen_tokens_total", "wall_s", "first_gold_round",
            "tokens_to_first_gold", "stop_reason", "answer", "score"} <= set(r)


def test_round_cap(store):
    e = h.Episode({"qid": "q", "q": "x", "part": "seen", "doc_id": "a"}, "s", "T")
    assert not h.step(FakeTok(), store, e, h.Gen(h.call_text("search", {"query": "bus"}), 1, 1, h.TOOL_RESPONSE_ID, 0), h.MAX_ROUNDS)
    assert e.stop_reason == "max_rounds" and e.answer == ""


def test_scoring():
    seen = {"part": "seen", "gold": ["Juniper"]}
    absent = {"part": "absent_test", "gold": []}
    assert h.score_answer(seen, "Team Juniper handles it.") == 1
    assert h.score_answer(seen, "UNKNOWN") == 0 and h.score_answer(seen, "") == 0
    assert h.score_answer(absent, "UNKNOWN") == 1 and h.score_answer(absent, "Team Juniper.") == 0 and h.score_answer(absent, "") == 0


needs_tokenizer = pytest.mark.skipif(not BOLT.exists(), reason="needs the Bolt tokenizer")


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(BOLT, local_files_only=True)


@needs_tokenizer
def test_call_text_matches_template(tok):
    args = {"query": "a b", "k": 3}
    m = h.first_messages("S", "U")
    assistant = {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "search", "arguments": args}}]}
    full = h.render(tok, m + [assistant, {"role": "tool", "name": "search", "content": "R"}])
    assert full.startswith(h.render(tok, m) + h.call_text("search", args) + "<|tool_response>response:search{value:")


@needs_tokenizer
def test_training_pair_is_byte_identical_to_inference(tok):
    """prompt + target must render exactly what the loop feeds back in the next round."""
    te = SimpleNamespace()
    h.patch_trainer(te)
    target = h.call_text("read", {"doc_id": "legal/motion-for-summary-judgment-4455"})
    ids, labels = te.encode(tok, h.tool_q(h.SYSTEM, "Which team?"), target, 512)
    m = h.first_messages(h.SYSTEM, "Which team?")
    assert te.prompt_text(tok, h.tool_q(h.SYSTEM, "Which team?")) == h.render(tok, m)
    assistant = {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "read", "arguments": {"doc_id": "legal/motion-for-summary-judgment-4455"}}}]}
    next_prompt = h.render(tok, m + [assistant, {"role": "tool", "name": "read", "content": "R"}])
    assert next_prompt.startswith(tok.decode(ids))
    assert ids[-1] == h.TOOL_RESPONSE_ID and labels[-1] == h.TOOL_RESPONSE_ID and labels.count(-100) == len(tok(h.render(tok, m), add_special_tokens=False)["input_ids"])


@needs_tokenizer
def test_final_answer_stops_at_turn_end(tok):
    te = SimpleNamespace()
    h.patch_trainer(te)
    ids, labels = te.encode(tok, "What is 2 + 2?", "4", 64)
    assert ids[-1] == h.TURN_END_ID and labels[-2:] == tok("4<turn|>", add_special_tokens=False)["input_ids"]


@needs_tokenizer
def test_overlong_pair_raises(tok):
    te = SimpleNamespace()
    h.patch_trainer(te)
    with pytest.raises(ValueError):
        te.encode(tok, "word " * 300, "x", 256)


@needs_tokenizer
def test_two_calls_render(tok):
    m = h.first_messages("S", "U")
    calls = [{"type": "function", "function": {"name": "search", "arguments": {"query": "a"}}},
             {"type": "function", "function": {"name": "read", "arguments": {"doc_id": "b"}}}]
    out = h.render(tok, m + [{"role": "assistant", "tool_calls": calls}, {"role": "tool", "name": "search", "content": "R1"},
                             {"role": "tool", "name": "read", "content": "R2"}])
    assert out.count("<|tool_response>") == 2 and out.endswith("<tool_response|>") and "response:read" in out


if __name__ == "__main__":
    sys.exit(pytest.main(["-q", os.path.abspath(__file__)]))
