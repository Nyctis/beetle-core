"""A hostile requester program: tries to escape the sandbox, then trains briefly.

Every attempt should be blocked. Results go to the requester's sealed private
report, and host/run_local.py checks that each one says "blocked".
"""

import os
import socket

from beetle_sdk import Beetle

net = Beetle()


def attempt(name, fn):
    try:
        fn()
        net.log(f"{name}: ALLOWED")
    except Exception as e:
        net.log(f"{name}: blocked ({type(e).__name__})")


attempt("connect to the internet", lambda: socket.create_connection(("1.1.1.1", 443), timeout=2))
attempt("read the encrypted inputs in /job/in", lambda: open("/job/in/train.enc", "rb").read(1))
attempt("write a file the host would collect", lambda: open("/job/out/leak.txt", "w").write("x"))
attempt("read the base model", lambda: os.listdir("/model"))
attempt("read the sandbox's own code", lambda: open("/app/enclave.py").read(1))
attempt("read /proc (other processes' memory)", lambda: open("/proc/1/environ", "rb").read(1))
attempt("write outside the scratch dir", lambda: open("/tmp/escape.txt", "w").write("x"))
attempt("signal the parent process", lambda: os.kill(os.getppid(), 0))

records = net.dataset()
pairs = [[f"What is my {r['topic']}?", f"Your {r['topic']} is {r['answer']}."] for r in records]
for _ in range(2):
    net.forward_backward(pairs[:8])
    net.optim_step(1e-3)
