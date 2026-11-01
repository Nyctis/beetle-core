"""Everything published from this directory passes through here.

Raw records stay in results/, which git ignores. What leaves it has no tokens, no
addresses of machines, no personal paths, and addresses and signatures cut to their
first 12 characters, so tests/test_public_repo.py never reads one as a key. The full
values are in results/ for whoever has to check one against the chain.

results/hosts.json holds what no source file may name:
    {"ips": {"<address>": "a100"}, "names": {"<host name>": "a100"}}
"""

from __future__ import annotations

import ipaddress
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOSTS_FILE = HERE / "results" / "hosts.json"
HOSTS = json.loads(HOSTS_FILE.read_text()) if HOSTS_FILE.exists() else {"ips": {}, "names": {}}
BASE58_CHARS = "1-9A-HJ-NP-Za-km-z"
LONG = re.compile(rf"(?<![{BASE58_CHARS}])[{BASE58_CHARS}]{{32,90}}(?![{BASE58_CHARS}])")
IPV4 = re.compile(r"(?<![\d.])(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(?![\d.])")
PATHS = [(re.compile(r"bk_local_[A-Za-z0-9_\-]+"), "<token>"),
         (re.compile(r"/private/tmp/[^\s'\"\\]+"), "<scratch>"),
         (re.compile(r"/tmp/claude[^\s'\"\\]+"), "<scratch>"),
         (re.compile(r"/Users/[^/\s'\"\\]+"), "~"),
         (re.compile(r"/home/[^/\s'\"\\]+"), "~"),
         (re.compile(r"GPU-[0-9a-f]{8}-[0-9a-f-]{27}"), "GPU-<uuid>")]
KEEP = {"TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"}


def _ip(match: re.Match) -> str:
    text = match.group(1)
    try:
        address = ipaddress.ip_address(text)
    except ValueError:  # a version number, say
        return text
    return text if address.is_private or address.is_loopback or address.is_unspecified else "<public-ip>"


def _long(match: re.Match) -> str:
    text = match.group(0)
    if text in KEEP or set(text) <= set("0123456789abcdefABCDEF") or text.isalpha() or not re.search(r"\d", text):
        return text  # a digest, a word, or a run of letters: none of these is an address
    return text[:12] + "…"


def scrub(text: str) -> str:
    for address, role in HOSTS["ips"].items():
        text = text.replace(address, f"<{role}-ip>")
    for name, role in HOSTS["names"].items():
        text = text.replace(name, role)
    for pattern, stand_in in PATHS:
        text = pattern.sub(stand_in, text)
    return LONG.sub(_long, IPV4.sub(_ip, text))


def scrub_value(value):
    """A JSON value with every string scrubbed, keys included."""
    if isinstance(value, str):
        return scrub(value)
    if isinstance(value, list):
        return [scrub_value(v) for v in value]
    if isinstance(value, dict):
        return {scrub(k): scrub_value(v) for k, v in value.items()}
    return value
