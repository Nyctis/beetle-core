"""An optional output gate in front of any OpenAI-compatible `/v1`, for the buyer's side.

    BEETLE_RAIL_SECRETS=canaries.json python integrations/guardrails/gate.py \
        --upstream http://127.0.0.1:8741/v1 --port 8742

Point the agent's OPENAI_BASE_URL at this port to turn the rail on, and back at
`beetle serve` to turn it off; nothing in Beetle imports this file. The token and the
`beetle` request field pass through untouched, and the `x-beetle-*` response headers
come back. A withheld answer has finish_reason `content_filter` and `x-beetle-rail:
withheld`. A streamed answer is held until it is complete, because a stream can't be
taken back: the agent gets no first token until the rail has seen the last one.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

os.environ.setdefault("NEMO_GUARDRAILS_NO_USAGE_STATS", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("NEMO_TELEMETRY_ENABLED", "false")

from nemoguardrails import LLMRails, RailsConfig  # noqa: E402

KEEP_REQUEST = ("authorization", "content-type")
log = logging.getLogger("beetle.gate")


class Gate:
    def __init__(self, upstream: str, config: Path):
        self.upstream = upstream.rstrip("/")
        self.http = httpx.AsyncClient(timeout=None, trust_env=False)
        self.rails = LLMRails(RailsConfig.from_path(str(config)))
        self.withheld = 0

    async def review(self, text: str) -> str | None:
        """None when the text may pass, else what to say instead."""
        result = await self.rails.generate_async(
            messages=[{"role": "user", "content": ""}, {"role": "assistant", "content": text}],
            options={"rails": ["output"]})
        said = result.response[0]["content"]
        if said == text:
            return None
        self.withheld += 1
        log.warning("withheld answer %d", self.withheld)
        return said

    async def chat(self, request: Request) -> Response:
        body = await request.json()
        headers = {k: v for k, v in request.headers.items() if k in KEEP_REQUEST}
        r = await self.http.post(f"{self.upstream}/chat/completions", json=body, headers=headers)
        kept = {k: v for k, v in r.headers.items() if k.startswith("x-beetle-")}
        if r.status_code >= 400:
            return Response(r.content, r.status_code, kept, r.headers.get("content-type"))
        if body.get("stream"):
            return await self.streamed(r, kept)
        data = r.json()
        choice = data["choices"][0]
        instead = await self.review(choice["message"].get("content") or "")
        if instead is not None:
            choice["message"]["content"], choice["finish_reason"] = instead, "content_filter"
        return JSONResponse(data, headers={**kept, "x-beetle-rail": "pass" if instead is None else "withheld"})

    async def streamed(self, r: httpx.Response, kept: dict) -> Response:
        events = [line[6:] for line in r.text.splitlines() if line.startswith("data: ")]
        parsed = [json.loads(e) for e in events if e != "[DONE]"]
        text = "".join(c["choices"][0]["delta"].get("content") or "" for c in parsed if c.get("choices"))
        instead = await self.review(text)
        if instead is not None:
            head, last = parsed[0], parsed[-1]
            head["choices"][0]["delta"] = {"role": "assistant", "content": instead}
            last["choices"][0].update(delta={}, finish_reason="content_filter")
            events = [json.dumps(head), json.dumps(last), "[DONE]"]
        payload = "".join(f"data: {e}\n\n" for e in events)
        return Response(payload, 200, {**kept, "x-beetle-rail": "pass" if instead is None else "withheld"},
                        "text/event-stream")

    async def models(self, request: Request) -> Response:
        headers = {k: v for k, v in request.headers.items() if k in KEEP_REQUEST}
        r = await self.http.get(f"{self.upstream}/models", headers=headers)
        return Response(r.content, r.status_code, media_type=r.headers.get("content-type"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upstream", default=os.environ.get("BEETLE_RAIL_UPSTREAM", "http://127.0.0.1:8741/v1"))
    ap.add_argument("--port", type=int, default=8742)
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config")
    a = ap.parse_args()
    gate = Gate(a.upstream, a.config)
    app = Starlette(routes=[Route("/v1/chat/completions", gate.chat, methods=["POST"]),
                            Route("/v1/models", gate.models, methods=["GET"])])
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
