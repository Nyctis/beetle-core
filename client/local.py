"""Local inference: mlx-lm on a Mac, or a server on this machine's own GPU.

Everything MLX runs on one thread, because its default stream belongs to the
thread that made it, and one model stays loaded, because this may be an 8 GB
laptop. An adapter stays in memory: `files` is a callable that decrypts it, used
only when it actually has to load, and the tensors are applied directly, never
written out.

A server such as vLLM can only load an adapter from a path. That path is on a
tmpfs, and the files go as soon as the server has read them.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from contextlib import aclosing
from pathlib import Path

import model_profiles as mp
from serve.gateway import Backend, is_tmpfs

DEFAULT_MAX_TOKENS = 512
_DONE = object()


class MlxEngine:
    def __init__(self):
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
        self.key = None
        self.model = self.tokenizer = None
        self.template = {}

    def _load(self, profile: mp.Profile, adapter_id: str | None, files) -> None:
        if self.key == (profile.name, adapter_id):
            return
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.tuner.utils import linear_to_lora_layers
        from safetensors.numpy import load as load_tensors

        self.model = self.tokenizer = self.key = None
        mx.clear_cache()
        model, tokenizer = load(profile.mlx["repo"])
        if files:
            adapter = files()
            peft = json.loads(adapter["adapter_config.json"])
            weights, config = mp.to_mlx_tensors(load_tensors(adapter["adapter_model.safetensors"]), peft, profile)
            linear_to_lora_layers(model, config["num_layers"], config["lora_parameters"])
            model.load_weights([(k, mx.array(v)) for k, v in weights.items()], strict=False)
        model.eval()
        self.model, self.tokenizer, self.key = model, tokenizer, (profile.name, adapter_id)
        self.template = profile.chat.get("template", {})

    def _generate(self, emit, messages: list[dict], params: dict) -> None:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        # An adapter keeps the template it was trained on; a bare base answers directly.
        template = {} if self.key[1] else self.template
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                    **template)
        sampler = make_sampler(temp=params.get("temperature", 0.0), top_p=params.get("top_p", 1.0))
        last = None
        for last in stream_generate(self.model, self.tokenizer, prompt, sampler=sampler,
                                    max_tokens=params.get("max_tokens") or DEFAULT_MAX_TOKENS):
            emit(last.text)
        return {"in_tokens": last.prompt_tokens, "out_tokens": last.generation_tokens,
                "tok_s": last.generation_tps, "peak_gb": last.peak_memory} if last else {}

    async def stream(self, profile: mp.Profile, adapter_id: str | None, files,
                     messages: list[dict], params: dict, usage: dict | None = None):
        """Yield pieces of the answer; `usage` is filled in at the end."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def work() -> None:
            try:
                self._load(profile, adapter_id, files)
                stats = self._generate(lambda t: loop.call_soon_threadsafe(queue.put_nowait, t),
                                       messages, params)
                if usage is not None:
                    usage.update(stats)
                loop.call_soon_threadsafe(queue.put_nowait, _DONE)
            except Exception as e:  # the caller decides whether to try another route
                loop.call_soon_threadsafe(queue.put_nowait, e)

        self.pool.submit(work)
        while (item := await queue.get()) is not _DONE:
            if isinstance(item, Exception):
                raise item
            yield item

    async def answer(self, profile, adapter_id, files, messages, params) -> str:
        return "".join([p async for p in self.stream(profile, adapter_id, files, messages, params)])


def vllm_files(files: dict, profile: mp.Profile) -> dict[str, bytes]:
    """A trained adapter, rewritten for the model class vLLM serves."""
    from safetensors.numpy import load, save

    tensors = mp.to_vllm_tensors(load(files["adapter_model.safetensors"]), profile)
    config = mp.to_vllm_config(json.loads(files["adapter_config.json"]), tensors)
    return {"adapter_model.safetensors": save(tensors),
            "adapter_config.json": json.dumps(config).encode()}


class ServerEngine:
    """An OpenAI-compatible server on this machine. `settings` is the "local_backend"
    section of config.json: {"url", "models": {profile name: served name}, "scratch"}."""

    def __init__(self, settings: dict):
        self.settings = settings
        self.backend = Backend(settings["url"])
        self.scratch = Path(settings.get("scratch", "/dev/shm/beetle"))
        self.loaded: dict[str, str] = {}  # adapter id -> served name

    async def _served(self, profile: mp.Profile, adapter_id: str | None, files) -> str:
        if adapter_id is None:
            return self.settings["models"][profile.name]
        if adapter_id not in self.loaded:
            if not is_tmpfs(self.scratch):
                raise RuntimeError(f"{self.scratch} is not a tmpfs: an adapter would land on a disk")
            name = f"a{os.urandom(8).hex()}"
            where = self.scratch / name
            where.mkdir(parents=True, mode=0o755)
            try:
                for filename, data in vllm_files(files(), profile).items():
                    (where / filename).write_bytes(data)
                await self.backend.load_lora(name, where)
            finally:
                shutil.rmtree(where, ignore_errors=True)
            self.loaded[adapter_id] = name
        return self.loaded[adapter_id]

    async def stream(self, profile: mp.Profile, adapter_id: str | None, files, messages: list[dict],
                     params: dict, usage: dict | None = None):
        name = await self._served(profile, adapter_id, files)
        body = {**params, "model": name, "messages": messages, "stream": True,
                "stream_options": {"include_usage": True}}
        async with aclosing(self.backend.chat(body)) as events:
            async for event in events:
                if event.get("usage") and usage is not None:
                    usage.update(in_tokens=event["usage"]["prompt_tokens"],
                                 out_tokens=event["usage"]["completion_tokens"])
                for choice in event.get("choices") or []:
                    if piece := (choice.get("delta") or {}).get("content"):
                        yield piece

    async def answer(self, profile, adapter_id, files, messages, params) -> str:
        return "".join([p async for p in self.stream(profile, adapter_id, files, messages, params)])
