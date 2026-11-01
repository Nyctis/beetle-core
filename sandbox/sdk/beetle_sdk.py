"""Client library for programs that run inside the beetle sandbox.

A program is a plain Python file (`train.py`). It cannot import torch or any
third-party package and has no network; it trains the job's LoRA adapter by
calling the sandbox through this client:

    from beetle_sdk import Beetle
    net = Beetle()
    records = net.dataset()                      # the decrypted training records
    loss = net.forward_backward([[prompt, response], ...])
    net.optim_step(lr=2e-3)
    texts = net.sample(["What is my cat's name?"], max_new_tokens=16)
    net.checkpoint()
    net.log("anything you want in your private report")

When the program exits, the sandbox evaluates the adapter on the hidden eval
set (which the program never sees) and seals the result to the requester.
"""

import json
import os


class BeetleError(RuntimeError):
    pass


class Beetle:
    def __init__(self):
        w, r = (int(x) for x in os.environ["BEETLE_RPC"].split(","))
        self._w = os.fdopen(w, "wb", buffering=0)
        self._r = os.fdopen(r, "rb")

    def _call(self, method, **params):
        self._w.write(json.dumps({"method": method, "params": params}).encode() + b"\n")
        line = self._r.readline()
        if not line:
            raise BeetleError("sandbox closed the connection")
        resp = json.loads(line)
        if "error" in resp:
            raise BeetleError(resp["error"])
        return resp["result"]

    def dataset(self):
        return self._call("dataset")

    def forward_backward(self, pairs):
        return self._call("forward_backward", pairs=[list(p) for p in pairs])

    def optim_step(self, lr):
        return self._call("optim_step", lr=lr)

    def sample(self, prompts, max_new_tokens=32):
        return self._call("sample", prompts=list(prompts), max_new_tokens=max_new_tokens)

    def checkpoint(self):
        return self._call("checkpoint")

    def log(self, msg):
        return self._call("log", msg=str(msg))
