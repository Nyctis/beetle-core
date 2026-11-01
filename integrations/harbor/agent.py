"""A Harbor agent whose loop runs outside the task's container.

The task environment has no network, so the model call happens on the host and only
the answer goes in. Point it at `beetle serve`, at the guardrails gate, or at any
OpenAI-compatible server:

    OPENAI_BASE_URL=http://127.0.0.1:8741/v1 OPENAI_API_KEY=... PYTHONPATH=integrations/harbor \
        harbor run -p integrations/harbor/tasks --agent-import-path agent:OpenAIAgent -m MODEL
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import urllib.request

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

ANSWER = "/app/answer.txt"
TASK_NOTE = re.compile(r"\n*Write only your final answer to `/app/answer.txt`\.\s*$")


def complete(prompt: str, model: str) -> str:
    request = urllib.request.Request(
        os.environ["OPENAI_BASE_URL"].rstrip("/") + "/chat/completions",
        data=json.dumps({"model": model, "temperature": 0, "max_tokens": 192,
                         "messages": [{"role": "user", "content": prompt}]}).encode(),
        headers={"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'EMPTY')}",
                 "Content-Type": "application/json"})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=120) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"] or ""


class OpenAIAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "beetle-openai"

    def version(self) -> str:
        return "0.0.1"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        model = (self.model_name or "").split("/", 1)[-1]
        text = await asyncio.to_thread(complete, TASK_NOTE.sub("", instruction), model)
        encoded = base64.b64encode(text.encode()).decode()
        await environment.exec(f"echo {encoded} | base64 -d > {ANSWER}")
