"""A model server whose behaviour is known, for the paths a real model won't take on
demand. It answers vLLM's streaming chat, counts with the model's own tokenizer as a
buyer does, and reports what it did at /stats. The first word of the prompt picks:

    COUNT n        n pieces at once
    PACE ms n      n pieces, one every ms milliseconds
    HANG n         n pieces, then nothing, for ever
    BREAK n        n pieces, then the connection drops
    ERROR          500 before the first piece

Usage, from a checkout: stub_backend.py PORT MODEL
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.getcwd())

import node  # noqa: E402,F401  (import paths)

import model_profiles as mp  # noqa: E402
import uvicorn  # noqa: E402
from client.tokens import TokenCounter  # noqa: E402
from starlette.applications import Starlette  # noqa: E402
from starlette.responses import JSONResponse, StreamingResponse  # noqa: E402
from starlette.routing import Route  # noqa: E402

PORT, MODEL = int(sys.argv[1]), sys.argv[2]
COUNTER = TokenCounter(mp.get(MODEL))
STATS = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "pieces": 0, "finished": 0, "cut_off": 0, "open": 0}


def event(body: dict) -> str:
    return "data: " + json.dumps(body) + "\n\n"


async def models(request):
    return JSONResponse({"data": [{"id": MODEL}]})


async def stats(request):
    return JSONResponse(STATS)


async def chat(request):
    body = await request.json()
    words = body["messages"][-1]["content"].split()
    mode = words[0] if words else "COUNT"
    STATS["requests"] += 1
    if mode == "ERROR":
        return JSONResponse({"error": "stub says no"}, status_code=500)
    numbers = [int(w) for w in words[1:] if w.isdigit()]
    pace = numbers[0] / 1000 if mode == "PACE" else 0
    count = min((numbers[-1] if numbers else 16), body.get("max_tokens") or 10_000)
    prompt = COUNTER.prompt(body["messages"])
    STATS["prompt_tokens"] += prompt

    async def events():
        STATS["open"] += 1
        sent = ""
        ended = False
        try:
            for i in range(count):
                piece = f"w{i} "
                sent += piece
                STATS["pieces"] += 1
                yield event({"choices": [{"delta": {"content": piece}, "finish_reason": None}]})
                if pace:
                    await asyncio.sleep(pace)
                elif i % 50 == 49:
                    await asyncio.sleep(0)
            if mode == "HANG":
                await asyncio.sleep(86400)
            if mode == "BREAK":
                raise RuntimeError("stub breaks the stream")
            yield event({"choices": [{"delta": {}, "finish_reason": "stop"}]})
            out = COUNTER.count(sent)
            yield event({"choices": [], "usage": {"prompt_tokens": prompt, "completion_tokens": out,
                                                  "total_tokens": prompt + out}})
            yield "data: [DONE]\n\n"
            ended = True
        finally:
            STATS["completion_tokens"] += COUNTER.count(sent)
            STATS["finished" if ended else "cut_off"] += 1
            STATS["open"] -= 1

    return StreamingResponse(events(), media_type="text/event-stream")


app = Starlette(routes=[Route("/v1/models", models), Route("/stats", stats),
                        Route("/v1/chat/completions", chat, methods=["POST"])])
uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
