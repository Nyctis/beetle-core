"""An echo node for the C7 lab: a public peer that tells callers how it sees them."""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.getcwd())

import node  # noqa: E402,F401
from net import echo  # noqa: E402
from net.peer import Peer  # noqa: E402


async def main() -> None:
    peer = await Peer.bind(41799, services={echo.ALPN: echo.answer})
    print(json.dumps({"id": peer.id, "addrs": ["10.77.0.9:41799"]}), flush=True)
    await asyncio.sleep(3600)


asyncio.run(main())
