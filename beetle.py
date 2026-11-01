"""The Python SDK. Limits are required, because code that spends should say how much:

    import beetle

    b = beetle.Client(max_usd=5, min_tier="public")
    me = b.train("ds_memories", base="qwen3.5-2b").wait()
    print(me.report, me.proof["tier"])

    print(b.chat("Summarize this note", model="qwen3.5-2b").text)   # local if it fits

    me = b.hybrid("ds_memories", base="auto")                       # picks the path, brings it home
    print(me.chat("Where do I go on Thursday?").text)

Everything runs in the local daemon, which holds the wallet and the policy; a
daemon starts if none is running. The limits here can only tighten the policy.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from client.api import Api, ApiError  # noqa: F401
from client.datasets import Datasets
from node import wire

__all__ = ["Client", "Job", "Reply", "Model", "ApiError"]
POLL = 3.0


@dataclass
class Reply:
    text: str
    route: str
    tier: str
    cost_usd: float
    proof: str


class Job:
    def __init__(self, client: "Client", view: dict):
        self.client, self.view = client, view
        self.id = view["job"]

    def status(self) -> dict:
        self.view = self.client.api.get(f"/beetle/jobs/{self.id}")
        return self.view

    def wait(self, alias: str = "me") -> "Job":
        """Until delivered, then fetch: verified, effect checked, settled."""
        while self.status()["stage"] not in ("delivered", "settled", "failed") or self.view.get("next_job"):
            if self.view.get("next_job"):
                self.id = self.view["next_job"]
            time.sleep(POLL)
        if self.view["stage"] == "failed":
            raise RuntimeError(self.view.get("error") or self.view.get("ended"))
        if self.view["stage"] == "delivered":
            self.view = self.client.api.post(f"/beetle/jobs/{self.id}/fetch", {"alias": alias})
        return self

    @property
    def report(self) -> dict:
        return self.view.get("result") or {}

    @property
    def proof(self) -> dict:
        return self.client.api.get(f"/beetle/proof/{self.id}")

    def chat(self, prompt: str) -> Reply:
        return self.client.chat(prompt, model=self.view.get("alias") or "me")


Model = Job


class Client:
    def __init__(self, max_usd: float, min_tier: str, route: str = "auto", token: str | None = None):
        self.max_usd, self.min_tier, self.route = max_usd, min_tier, route
        self.api = Api(token)
        self.data = Datasets(wire.home() / "datasets.json")  # run by a person, see the README

    def train(self, dataset: str, base: str = "auto", method: str = "memory", seller: str | None = None,
              alias: str | None = None, idempotency_key: str | None = None) -> Job:
        return Job(self, self.api.post("/beetle/jobs", {
            "dataset": dataset, "base": base, "method": method, "seller": seller, "alias": alias,
            "min_tier": self.min_tier, "max_usd": self.max_usd,
            "idempotency_key": idempotency_key or uuid.uuid4().hex}))

    def chat(self, prompt: str, model: str = "auto", route: str | None = None, **params) -> Reply:
        text, headers, cost = [], {}, 0.0
        messages = [{"role": "user", "content": prompt}]
        for meta, piece in self.api.chat(model, messages, route=route or self.route, min_tier=self.min_tier,
                                         max_usd=self.max_usd, **params):
            if meta:
                headers = meta
            elif isinstance(piece, str):
                text.append(piece)
            else:
                cost = piece["beetle"]["cost_usd"]
        return Reply("".join(text), headers["x-beetle-route"], headers["x-beetle-tier"], cost,
                     headers["x-beetle-proof"])

    def hybrid(self, dataset: str, base: str = "auto", alias: str = "me") -> Job:
        """Train remotely, bring the adapter home, keep the model name: the gateway
        serves it locally if this machine can, and from a seller if not."""
        return self.train(dataset, base=base, alias=alias).wait(alias)
