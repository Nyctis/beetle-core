"""End-to-end run of private inference: a real gateway, a real link, a real model.

Everything about the deployment comes from the command line. With no --ssh the
gateway starts on this machine; with --ssh it starts on that host, and ssh is
used only for setup and for sharing the rendezvous files, which a chain would
carry in production. The session itself is a direct iroh link.

    # one machine, any OpenAI-compatible server
    python scripts/e2e_chat.py --backend http://127.0.0.1:8080 chat --model MODEL

    # a gateway elsewhere, advertised at its public address
    python scripts/e2e_chat.py --ssh HOST --remote-dir DIR --remote-python PY \\
        --addr PUBLIC_IP:PORT --port PORT --backend URL chat --model MODEL

Commands: chat, adapter, bench, plain, reconnect, scan-tap. See serve/README.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import shutil
import socket
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import uuid
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import httpx  # noqa: E402

import net  # noqa: E402,F401

from net import echo  # noqa: E402
from net.peer import Peer  # noqa: E402
from net.rendezvous import FileRendezvous  # noqa: E402
from serve.client import GatewayError, Session  # noqa: E402
from udp_punch import stun_mapping  # noqa: E402

DEFAULT_PROMPT = "Explain in three sentences why the sky is blue."
LONG_PROMPT = "Write a long story about a lighthouse keeper, in plain language."


class SshRendezvous(FileRendezvous):
    """The gateway's rendezvous files, reached over ssh from the buyer's machine."""

    def __init__(self, host: str, directory: str):
        super().__init__(Path(directory) / "buyer.json", Path(directory) / "seller.json", poll=0.3)
        self.host = host

    def _write(self, blob: bytes) -> None:
        subprocess.run(["ssh", "-o", "BatchMode=yes", self.host,
                        f"cat > {self.mine}.tmp && mv {self.mine}.tmp {self.mine}"],
                       input=blob, check=True, stderr=subprocess.DEVNULL)

    def _read(self) -> bytes | None:
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", self.host, f"cat {self.theirs}"],
                           capture_output=True)
        return r.stdout or None


class Deployment:
    """A gateway started for this run, here or over ssh, and the way to reach it."""

    def __init__(self, args):
        self.args = args
        self.dir = f"{tempfile.gettempdir() if not args.ssh else '/tmp'}/beetle-e2e-{uuid.uuid4().hex[:12]}"

    def start(self) -> dict:
        a = self.args
        options = ["-m", "serve.gateway", "--backend", a.backend, "--rendezvous", self.dir,
                   "--port", str(a.port), "--frame", str(a.frame)]
        for flag, value in (("--models", a.models), ("--tap", a.tap), ("--scratch", a.scratch),
                            ("--spool", a.spool)):
            if value:
                options += [flag, value]
        for addr in a.addr or []:
            options += ["--addr", addr]
        if a.ssh:
            command = (f"mkdir -p {self.dir} && cd {a.remote_dir} && "
                       f"exec {a.remote_python} {shlex.join(options)}")
            self.proc = subprocess.Popen(["ssh", "-o", "BatchMode=yes", a.ssh, command],
                                         stdout=subprocess.PIPE)
        else:
            Path(self.dir).mkdir(parents=True)
            self.proc = subprocess.Popen([sys.executable, *options], cwd=ROOT, stdout=subprocess.PIPE)
        line = self.proc.stdout.readline()
        if not line:
            raise SystemExit("the gateway did not start")
        return json.loads(line)

    def rendezvous(self) -> FileRendezvous:
        if self.args.ssh:
            return SshRendezvous(self.args.ssh, self.dir)
        d = Path(self.dir)
        return FileRendezvous(d / "buyer.json", d / "seller.json")

    def stop(self) -> None:
        if self.args.ssh:
            subprocess.run(["ssh", "-o", "BatchMode=yes", self.args.ssh,
                            f"rm -rf {self.dir}; pkill -f {self.dir}"],  # pkill ends this shell too
                           stderr=subprocess.DEVNULL)
        else:
            shutil.rmtree(self.dir, ignore_errors=True)
        self.proc.terminate()


def gateway_node(gateway: dict) -> dict:
    return {"id": gateway["endpoint_id"], "addrs": gateway["addrs"]}


async def candidates(args, peer: Peer, gateway: dict) -> list[str]:
    """This buyer's addresses, best first.

    The gateway answers echo only if its inbound UDP is open. Behind a cloud
    firewall the probe times out, so a public address has to come from --candidate
    or --stun. STUN uses its own socket, so that address assumes the NAT keeps
    the local port; set --local-port and check with a real run.
    """
    found = await echo.probe(peer, [gateway_node(gateway)])
    addrs = list(args.candidate or [])
    if args.stun:
        host, port = args.stun.rsplit(":", 1)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        mapping = stun_mapping(sock, (host, int(port)))
        if mapping:
            addrs.append(f"{mapping[0]}:{peer.port}")
    addrs = list(dict.fromkeys(addrs + found["addrs"]))
    print(f"nat={found['nat']} candidates={addrs}", flush=True)
    return addrs


async def open_session(deployment: Deployment, gateway: dict, peer: Peer, mode: str,
                       tick: float, bucket: float) -> Session:
    args = deployment.args
    return await Session.open(deployment.rendezvous(), bytes.fromhex(gateway["enc_pk"]), peer=peer,
                              addrs=await candidates(args, peer, gateway),
                              sign_pk=bytes.fromhex(gateway["sign_pk"]), mode=mode, tick=tick,
                              bucket=bucket, frame=args.frame)


def speed(started: float, stamps: list[float], tokens: int) -> dict:
    ttft = stamps[0] - started
    span = stamps[-1] - stamps[0]
    return {"ttft_ms": round(ttft * 1000, 1), "total_ms": round((stamps[-1] - started) * 1000, 1),
            "tokens": tokens,
            "tok_s": round((tokens - 1) / span, 1) if span > 0 and tokens > 1 else None}


async def timed_chat(session: Session, prompt: str, model: str, echo_text: bool, **params) -> dict:
    before = session.receipt["out_tokens"] if session.receipt else 0
    started, stamps, pieces = time.perf_counter(), [], []
    async for piece in session.chat([{"role": "user", "content": prompt}], model=model, **params):
        stamps.append(time.perf_counter())
        pieces.append(piece)
        if echo_text:
            print(piece, end="", flush=True)
    if echo_text:
        print(flush=True)
    return {**speed(started, stamps, session.receipt["out_tokens"] - before), "text": "".join(pieces)}


async def cmd_chat(args, deployment, gateway, peer) -> None:
    session = await open_session(deployment, gateway, peer, args.mode, args.tick, args.bucket)
    print(f"tier={session.tier} protected={session.info['protected']}: {session.info['notice']}")
    print(f"link rtt={session.pipe.link.rtt_ms():.1f}ms", flush=True)
    try:  # the path is open now, so the gateway's echo service is reachable even behind a firewall
        print(f"the gateway sees this buyer at {await echo.observe(peer, gateway_node(gateway))}",
              flush=True)
    except Exception as e:
        print(f"echo over the open path failed: {e!r}", flush=True)
    transcript = []
    for model in args.model:
        print(f"--- {model} ({args.mode})", flush=True)
        result = await timed_chat(session, args.prompt, model, True, max_tokens=args.max_tokens,
                                  temperature=0.0)
        print({k: v for k, v in result.items() if k != "text"}, flush=True)
        transcript.append({"model": model, "prompt": args.prompt, "answer": result["text"]})
    print(f"receipt: {session.receipt}", flush=True)
    print(f"frames sent={session.pipe.ch.tx} received={session.pipe.ch.rx}", flush=True)
    await session.close()
    if args.transcript:
        Path(args.transcript).write_text(json.dumps(transcript, ensure_ascii=False, indent=1))


async def cmd_adapter(args, deployment, gateway, peer) -> None:
    ask = lambda session, model: timed_chat(session, args.question, model, True,  # noqa: E731
                                            max_tokens=args.max_tokens, temperature=0.0)
    session = await open_session(deployment, gateway, peer, args.mode, args.tick, args.bucket)
    print(f"--- base model {args.base}", flush=True)
    before = await ask(session, args.base)
    started = time.perf_counter()
    name = await session.upload_adapter(Path(args.adapter))
    print(f"--- uploaded and loaded as {name} in {time.perf_counter() - started:.1f}s", flush=True)
    during = await ask(session, name)
    await session.close()
    print("--- session closed; a new session asks again", flush=True)
    session = await open_session(deployment, gateway, peer, args.mode, args.tick, args.bucket)
    try:
        await ask(session, name)
        refused = None
    except GatewayError as e:
        refused = str(e)
        print(f"adapter name refused: {refused}", flush=True)
    after = await ask(session, args.base)
    await session.close()
    answer = args.expect.lower() if args.expect else None
    print(json.dumps({"adapter": name, "refused_after_close": refused,
                      "expected_in_base": answer in before["text"].lower() if answer else None,
                      "expected_in_adapter": answer in during["text"].lower() if answer else None,
                      "expected_after_close": answer in after["text"].lower() if answer else None}))


async def plain_run(args, backend: str, prompt: str) -> dict:
    """One streamed request straight to the backend, with no gateway in the path."""
    body = {"model": args.model, "stream": True, "stream_options": {"include_usage": True},
            "max_tokens": args.max_tokens, "temperature": 0.0, **json.loads(args.extra),
            "messages": [{"role": "user", "content": prompt}]}
    stamps, tokens = [], 0
    async with httpx.AsyncClient(base_url=backend.rstrip("/").removesuffix("/v1"),
                                 timeout=None, trust_env=False) as http:
        started = time.perf_counter()
        async with http.stream("POST", "/v1/chat/completions", json=body) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if line.startswith("data:") and line[5:].strip() != "[DONE]":
                    event = json.loads(line[5:])
                    if (event.get("usage") or {}).get("completion_tokens"):
                        tokens = event["usage"]["completion_tokens"]
                    if any((c.get("delta") or {}).get("content") for c in event.get("choices") or []):
                        stamps.append(time.perf_counter())
    return speed(started, stamps, tokens)


def summarize(label: str, runs: list[dict]) -> dict:
    ttfts = sorted(r["ttft_ms"] for r in runs)
    rates = sorted(r["tok_s"] for r in runs if r["tok_s"])
    return {"path": label, "runs": len(runs), "ttft_ms_median": statistics.median(ttfts),
            "ttft_ms_min": ttfts[0], "ttft_ms_max": ttfts[-1],
            "total_ms_median": statistics.median(r["total_ms"] for r in runs),
            "tok_s_median": statistics.median(rates) if rates else None}


def unique_prompt(prompt: str) -> str:
    return f"[{uuid.uuid4().hex}] {prompt}"  # nothing may come from a prefix cache


async def cmd_plain(args) -> None:
    runs = [await plain_run(args, args.backend, unique_prompt(args.prompt))
            for _ in range(args.runs)]
    print(json.dumps(summarize("plain", runs)))


async def cmd_bench(args, deployment, gateway, peer) -> None:
    """Take turns between the paths, so other load on the machine falls on all of them."""
    sessions = {mode: await open_session(deployment, gateway, peer, mode, args.tick, args.bucket)
                for mode in args.modes.split(",")}
    paths = ["plain", *sessions] if args.plain_backend else list(sessions)
    runs: dict[str, list[dict]] = {path: [] for path in paths}
    extra = json.loads(args.extra)
    for i in range(args.runs):
        for path in paths[i % len(paths):] + paths[:i % len(paths)]:
            prompt = unique_prompt(args.prompt)
            if path == "plain":
                runs[path].append(await plain_run(args, args.plain_backend, prompt))
            else:
                runs[path].append(await timed_chat(sessions[path], prompt, args.model, False,
                                                   max_tokens=args.max_tokens, temperature=0.0,
                                                   **extra))
    for path in paths:
        print(json.dumps(summarize(path, runs[path])), flush=True)
    for session in sessions.values():
        await session.close()


async def cmd_reconnect(args, deployment, gateway, peer) -> None:
    """Break the path mid-response, then compare the answer with an unbroken one.

    By default the link is closed cleanly. --cut-with and --restore-with run any
    shell command instead, for an outage that sends no close at all, which each
    side can only notice by timing out.
    """
    session = await open_session(deployment, gateway, peer, args.mode, args.tick, args.bucket)
    params = {"max_tokens": args.max_tokens, "temperature": 0.0, "seed": 1}
    unbroken = await timed_chat(session, args.prompt, args.model, False, **params)
    repeat = await timed_chat(session, args.prompt, args.model, False, **params)
    pieces, stamps, task = [], [], None

    async def outage() -> None:
        await asyncio.to_thread(subprocess.run, args.cut_with, shell=True, check=True)
        await asyncio.sleep(args.outage)
        await asyncio.to_thread(subprocess.run, args.restore_with, shell=True, check=True)

    async for piece in session.chat([{"role": "user", "content": args.prompt}], model=args.model,
                                    **params):
        pieces.append(piece)
        stamps.append(time.perf_counter())
        if len(pieces) == args.cut_after:
            print(f"cutting after {len(pieces)} pieces, at frame {session.pipe.ch.rx}", flush=True)
            if args.cut_with:
                task = asyncio.create_task(outage())
            else:
                await session.pipe.link.close()
    if task:
        await task
    after_cut = stamps[args.cut_after - 1:]
    # The receipt check raises on any difference, so reaching this line means the
    # buyer holds exactly what the environment read from the backend.
    print(json.dumps({"pieces": len(pieces), "links_lost": session.links_lost,
                      "longest_gap_after_cut_s": round(max(b - a for a, b in
                                                           zip(after_cut, after_cut[1:])), 2),
                      "transcript_matches_receipt":
                          session.transcript.hex() == session.receipt["transcript_hash"],
                      "backend_repeats_itself": repeat["text"] == unbroken["text"],
                      "same_text_as_unbroken": "".join(pieces) == unbroken["text"],
                      "seq": session.receipt["seq"], "out_tokens": session.receipt["out_tokens"]}),
          flush=True)
    after = await timed_chat(session, "Say OK.", args.model, False, max_tokens=8, temperature=0.0)
    print(f"session still works: {after['text']!r}", flush=True)
    await session.close()


def scan_tap(path: Path, needles: list[str]) -> dict:
    """What a host's own recording shows: sizes per direction and any plaintext."""
    blob = path.read_bytes()
    sizes: dict[str, Counter] = {">": Counter(), "<": Counter()}
    body, i = bytearray(), 0
    while i < len(blob):
        n = struct.unpack(">I", blob[i + 1:i + 5])[0]
        sizes[chr(blob[i])][n] += 1
        body += blob[i + 5:i + 5 + n]
        i += 5 + n
    bytes_seen = Counter(body)
    expected = len(body) / 256
    return {"records": {k: dict(v) for k, v in sizes.items()}, "bytes": len(body),
            "hits": {n: bytes(body).count(n.encode()) for n in needles},
            "byte_chi2_per_dof": round(sum((bytes_seen[b] - expected) ** 2 / expected
                                           for b in range(256)) / 255, 2) if body else None}


async def with_gateway(args) -> None:
    deployment = Deployment(args)
    gateway = deployment.start()
    print({k: v for k, v in gateway.items()}, flush=True)
    peer = await Peer.bind(args.local_port)
    try:
        await {"chat": cmd_chat, "adapter": cmd_adapter, "bench": cmd_bench,
               "reconnect": cmd_reconnect}[args.command](args, deployment, gateway, peer)
    finally:
        deployment.stop()
        await peer.close()


def main() -> None:
    e = os.environ.get
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default=e("BEETLE_BACKEND", "http://127.0.0.1:8000"),
                    help="OpenAI-compatible server, as the gateway reaches it")
    ap.add_argument("--models", default=e("BEETLE_MODELS"), help="models the gateway offers")
    ap.add_argument("--ssh", default=e("BEETLE_SSH"), help="start the gateway on this ssh host")
    ap.add_argument("--remote-dir", default=e("BEETLE_REMOTE_DIR", "."), help="this repo on that host")
    ap.add_argument("--remote-python", default=e("BEETLE_REMOTE_PYTHON", "python3"))
    ap.add_argument("--addr", action="append", help="address the gateway advertises")
    ap.add_argument("--port", type=int, default=int(e("BEETLE_PORT", "41710")))
    ap.add_argument("--local-port", type=int, default=0)
    ap.add_argument("--candidate", action="append", help="public ip:port of this buyer, if known")
    ap.add_argument("--stun", default=e("BEETLE_STUN"),
                    help="host:port of a STUN server, used to learn this buyer's public IP")
    ap.add_argument("--tap", help="path on the gateway's machine for the host's byte record")
    ap.add_argument("--scratch", help="gateway tmpfs for decrypted adapters")
    ap.add_argument("--spool", help="gateway directory for sealed uploads")
    ap.add_argument("--frame", type=int, default=4096)
    ap.add_argument("--mode", default="constant", choices=["burst", "constant"])
    ap.add_argument("--tick", type=float, default=0.05)
    ap.add_argument("--bucket", type=float, default=5.0)
    ap.add_argument("--max-tokens", type=int, default=128)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("chat", help="stream answers from one or more models")
    p.add_argument("--model", action="append", required=True)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--transcript", help="write prompts and answers here as JSON")

    p = sub.add_parser("adapter", help="upload an adapter, use it, close, confirm it is gone")
    p.add_argument("--adapter", required=True, help="PEFT adapter directory on this machine")
    p.add_argument("--base", required=True, help="model name of the base model")
    p.add_argument("--question", required=True)
    p.add_argument("--expect", help="text only the adapter should know")

    p = sub.add_parser("bench", help="TTFT and tok/s through the gateway")
    p.add_argument("--model", required=True)
    p.add_argument("--modes", default="burst,constant")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--prompt", default=LONG_PROMPT)
    p.add_argument("--extra", default="{}", help='JSON params for the backend, e.g. {"ignore_eos":true}')
    p.add_argument("--plain-backend", help="the backend as this machine reaches it, to compare with")

    p = sub.add_parser("plain", help="TTFT and tok/s straight to the backend, no gateway")
    p.add_argument("--model", required=True)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--prompt", default=LONG_PROMPT)
    p.add_argument("--extra", default="{}")

    p = sub.add_parser("reconnect", help="cut the link mid-response and check the answer")
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", default=LONG_PROMPT)
    p.add_argument("--cut-after", type=int, default=20)
    p.add_argument("--cut-with", help="shell command that breaks the path")
    p.add_argument("--restore-with", help="shell command that restores it")
    p.add_argument("--outage", type=float, default=45.0, help="seconds between those two")

    p = sub.add_parser("scan-tap", help="check a host's byte record for plaintext")
    p.add_argument("path", type=Path)
    p.add_argument("--needle", action="append", default=[])

    args = ap.parse_args()
    if args.command == "scan-tap":
        print(json.dumps(scan_tap(args.path, args.needle), ensure_ascii=False))
    elif args.command == "plain":
        asyncio.run(cmd_plain(args))
    else:
        asyncio.run(with_gateway(args))


if __name__ == "__main__":
    main()
