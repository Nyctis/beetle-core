"""What the C7 namespaces can't reach by themselves: the chain's RPC, a STUN server
that sees their NAT, and a stub model server. Runs in the root namespace."""

from __future__ import annotations

import asyncio
import json
import socket
import struct
import sys

import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

MAGIC = 0x2112A442
BIND = "10.77.0.1"


class Stun(asyncio.DatagramProtocol):
    """Says it has a second address (RFC 5780), and answers from it when asked to."""

    second = None

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        if len(data) < 20 or struct.unpack(">H", data[:2])[0] != 1:
            return
        ip = struct.unpack(">I", socket.inet_aton(addr[0]))[0] ^ MAGIC
        value = struct.pack(">BBHI", 0, 1, addr[1] ^ (MAGIC >> 16), ip)
        attr = struct.pack(">HH", 0x0020, len(value)) + value
        attr += struct.pack(">HHBBH", 0x802C, 8, 0, 1, 3479) + socket.inet_aton(BIND)
        reply = struct.pack(">HHI", 0x0101, len(attr), MAGIC) + data[8:20] + attr
        (Stun.second.transport if len(data) > 20 else self.transport).sendto(reply, addr)


async def pipe(reader, writer):
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    finally:
        writer.close()


async def forward(reader, writer):
    up_reader, up_writer = await asyncio.open_connection("127.0.0.1", int(sys.argv[1]))
    await asyncio.gather(pipe(reader, up_writer), pipe(up_reader, writer), return_exceptions=True)


async def models(request):
    return JSONResponse({"data": [{"id": sys.argv[2]}]})


async def chat(request):
    body = await request.json()

    async def events():
        n = min(body.get("max_tokens") or 16, 32)
        for i in range(n):
            yield "data: " + json.dumps({"choices": [{"delta": {"content": f"word{i} "}, "finish_reason": None}]}) + "\n\n"
            await asyncio.sleep(0.01)
        yield "data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}) + "\n\n"
        yield "data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": n,
                                                              "total_tokens": 12 + n}}) + "\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


async def main():
    loop = asyncio.get_running_loop()
    await loop.create_datagram_endpoint(Stun, local_addr=(BIND, 3478))
    _, Stun.second = await loop.create_datagram_endpoint(Stun, local_addr=(BIND, 3479))
    server = await asyncio.start_server(forward, BIND, 18999)
    app = Starlette(routes=[Route("/v1/models", models), Route("/v1/chat/completions", chat, methods=["POST"])])
    stub = uvicorn.Server(uvicorn.Config(app, host=BIND, port=18000, log_level="warning"))
    await asyncio.gather(server.serve_forever(), stub.serve())


asyncio.run(main())
