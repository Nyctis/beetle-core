"""Program mode: run the requester's own training code inside the sandbox.

The requester's code runs in a child process that `launcher.py` locks down
with Landlock before any of that code runs: no network, no access to /job,
/model, /proc or the parent's memory, and only its own scratch directory is
writable. It talks to this parent over two pipes, one JSON object per line,
calling a small Tinker-style API (see sdk/beetle_sdk.py).

The parent keeps the model, optimizer, hidden eval set and all keys. The
child sees only the training records and the losses and samples it asks for,
so it cannot read the hidden eval set to game acceptance, cannot write files
the host would pick up, and cannot touch the host.
"""

from __future__ import annotations

import json
import os
import select
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
import train_eval as te

LAUNCHER = Path(__file__).with_name("launcher.py")
MAX_PAIRS = 64
MAX_PROMPTS = 32
MAX_NEW_TOKENS = 64
MAX_REQUEST_BYTES = 8 * 1024 * 1024
MAX_OUTPUT_BYTES = 64 * 1024
MAX_LOG_ENTRIES = 500
FAIL_CLOSED = 97            # launcher.py's exit code when it couldn't lock down
LOCKED_DOWN = b"launcher: seccomp on"
# How a program can stop without finishing. All of it is the program's own doing.
PROGRAM_FAULTS = {"wall_budget", "request_too_large", "child_stopped_responding"}


class Session:
    """Training state owned by the parent; the child drives it by RPC."""

    def __init__(self, model, tok, recipe: dict, memories: list, max_tokens: int):
        torch.manual_seed(recipe["seed"])
        self.model = te.add_lora(model, recipe)
        self.tok = tok
        self.max_len = recipe["max_seq_len"]
        self.memories = memories
        self.opt = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=recipe["optim"]["lr"], weight_decay=recipe["optim"]["weight_decay"])
        self.tokens = 0
        self.max_tokens = max_tokens
        self.spent = False   # asked for more than is paid for
        self.steps = 0
        self.loss_log: list[float] = []
        self.checkpoints: list[bytes] = []
        self.log: list[str] = []
        self.model.train()

    def charge(self, tokens: int) -> None:
        """Called before the work, so the program can't run past what was paid."""
        if self.tokens + tokens > self.max_tokens:
            self.spent = True
            raise ValueError("token budget reached")
        self.tokens += tokens

    # --- RPC methods. Arguments come from untrusted code: validate everything.

    def rpc_dataset(self):
        return self.memories

    def rpc_forward_backward(self, pairs):
        if not isinstance(pairs, list) or not 1 <= len(pairs) <= MAX_PAIRS:
            raise ValueError(f"pairs must be a list of 1..{MAX_PAIRS} [prompt, response]")
        batch = []
        for p in pairs:
            if not (isinstance(p, list) and len(p) == 2 and all(isinstance(s, str) for s in p)):
                raise ValueError("each pair must be [prompt, response] strings")
            batch.append(te.encode(self.tok, p[0], p[1], self.max_len))
        x, y, m = te.collate(self.tok, batch, self.model.device)
        self.charge(int(m.sum()))
        self.model.train()
        loss = self.model(input_ids=x, attention_mask=m, labels=y).loss
        loss.backward()
        self.loss_log.append(round(loss.item(), 5))
        return round(loss.item(), 6)

    def rpc_optim_step(self, lr):
        if not isinstance(lr, (int, float)) or not 0 < lr <= 1:
            raise ValueError("lr must be in (0, 1]")
        for group in self.opt.param_groups:
            group["lr"] = float(lr)
        self.opt.step()
        self.opt.zero_grad()
        self.steps += 1
        return self.steps

    def rpc_sample(self, prompts, max_new_tokens=32):
        if not isinstance(prompts, list) or not 1 <= len(prompts) <= MAX_PROMPTS \
                or not all(isinstance(s, str) for s in prompts):
            raise ValueError(f"prompts must be a list of 1..{MAX_PROMPTS} strings")
        if not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= MAX_NEW_TOKENS:
            raise ValueError(f"max_new_tokens must be 1..{MAX_NEW_TOKENS}")
        self.model.eval()
        texts = te.generate(self.model, self.tok, prompts, max_new_tokens, self.max_len, self.charge)
        self.model.train()
        return texts

    def rpc_checkpoint(self):
        self.checkpoints.append(te.lora_state_hash(self.model))
        return len(self.checkpoints)

    def rpc_log(self, msg):
        if len(self.log) < MAX_LOG_ENTRIES:
            self.log.append(str(msg)[:2000])
        return None

    def handle(self, line: bytes) -> bytes:
        try:
            req = json.loads(line)
            method = getattr(self, f"rpc_{req['method']}", None) if isinstance(req, dict) else None
            if method is None:
                raise ValueError("unknown method")
            result = method(**(req.get("params") or {}))
            resp = {"result": result}
        except Exception as e:  # report to the child, never crash the parent
            resp = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        return json.dumps(resp).encode() + b"\n"


def _write_all(fd: int, data: bytes, deadline: float) -> None:
    """`fd` must be non-blocking: a blocking write of more than the pipe has room
    for would wait on the child forever, whatever select said."""
    while data:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError
        _, ready, _ = select.select([], [fd], [], remaining)
        if ready:
            try:
                data = data[os.write(fd, data):]
            except BlockingIOError:
                pass


def run_program(session: Session, files: dict[str, bytes], entry: str, max_wall_seconds: int) -> dict:
    """Run the child to completion or until a budget is hit."""
    if entry not in files or "/" in entry:
        raise ValueError("program entry not found in bundle")
    workdir = Path(tempfile.mkdtemp(dir="/tmp", prefix="prog-"))
    for name, data in files.items():
        (workdir / name).write_bytes(data)

    req_r, req_w = os.pipe()    # child -> parent
    resp_r, resp_w = os.pipe()  # parent -> child
    env = {"BEETLE_RPC": f"{req_w},{resp_r}", "PATH": "/usr/local/bin:/usr/bin:/bin",
           "HOME": str(workdir), "LANG": "C.UTF-8"}
    proc = subprocess.Popen(
        [sys.executable, "-I", "-S", str(LAUNCHER), str(workdir), entry],
        cwd=workdir, env=env, pass_fds=(req_w, resp_r), close_fds=True,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    os.close(req_w)
    os.close(resp_r)
    os.set_blocking(resp_w, False)

    t0 = time.monotonic()
    deadline = t0 + max_wall_seconds
    out_fd = proc.stdout.fileno()
    open_fds = {req_r, out_fd}
    buf, output, stop_reason = b"", b"", None
    try:
        while open_fds:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                stop_reason = "wall_budget"
                break
            ready, _, _ = select.select(list(open_fds), [], [], min(remaining, 1.0))
            for fd in ready:
                chunk = os.read(fd, 1 << 20)
                if not chunk:
                    open_fds.discard(fd)
                elif fd == out_fd:
                    output = (output + chunk)[:MAX_OUTPUT_BYTES]
                else:
                    buf += chunk
                    if len(buf) > MAX_REQUEST_BYTES:
                        stop_reason = "request_too_large"
                        break
                    while b"\n" in buf and not stop_reason:
                        line, buf = buf.split(b"\n", 1)
                        _write_all(resp_w, session.handle(line), deadline)
                        if session.spent:
                            stop_reason = "token_budget"
                        elif time.monotonic() >= deadline:
                            stop_reason = "wall_budget"
                if stop_reason:
                    break
            if stop_reason:
                break
    except (TimeoutError, BrokenPipeError):
        stop_reason = stop_reason or "child_stopped_responding"
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        for fd in (req_r, resp_w):
            try:
                os.close(fd)
            except OSError:
                pass
        proc.stdout.close()
        shutil.rmtree(workdir, ignore_errors=True)

    if proc.returncode == FAIL_CLOSED and not output.startswith(LOCKED_DOWN):
        raise RuntimeError("the launcher could not lock down")  # this machine's fault, not the job's
    session.checkpoints.append(te.lora_state_hash(session.model))
    session.model.eval()
    return {
        "exit_code": proc.returncode,
        "stop_reason": stop_reason,
        "output": output.decode("utf-8", "replace"),
        "log": session.log,
        "steps": session.steps,
        "train_tokens": session.tokens,
        "loss_log": session.loss_log,
        "checkpoint_root": te.merkle_root(session.checkpoints),
        "wall_seconds": round(time.monotonic() - t0, 1),
    }
