"""Everything under experiments/network/ is written as if already public: no addresses,
host names, personal paths, wallets or tokens. harness.py passes all it writes through
scrub(), and `python scrub.py` cleans what is already on disk. Role names stay. So do
transaction signatures and program addresses, which are public on chain anyway."""

from __future__ import annotations

import ipaddress
import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = json.loads(Path(os.environ["BEETLE_MATRIX_NODES"]).read_text())
KEEP = set(CONFIG.get("public_addresses", []))  # program ids and the like

NAMED: dict[str, str] = {}
for role, node in CONFIG["nodes"].items():
    if node.get("ip"):
        NAMED[node["ip"]] = f"<{role}-ip>"
    if node.get("ssh"):
        NAMED[node["ssh"]] = role
for address, role in CONFIG.get("past_ips", {}).items():
    NAMED[address] = f"<{role}-ip>"
if CONFIG.get("t4_ip"):
    NAMED[CONFIG["t4_ip"]] = "<t4-ip>"

PATTERNS = [
    (re.compile(r"bk_local_[A-Za-z0-9_\-]+"), "<token>"),
    (re.compile(r"/private/tmp/[^\s'\"\\]*scratchpad/intg"), "<scratch>"),
    (re.compile(r"/private/tmp/[^\s'\"\\]+"), "<scratch>"),
    (re.compile(r"/Users/[^/\s'\"]+"), "~"),
    (re.compile(r"/home/[^/\s'\"]+"), "~"),
    *((re.compile(pattern), "<host>") for pattern in CONFIG.get("scrub_hosts", [])),
    (re.compile(r"\bBEETLE_MATRIX_NODES=\S+"), "BEETLE_MATRIX_NODES=<config>"),
    (re.compile(r"KEY[1-9A-HJ-NP-Za-km-z]{43,44}"), "KEY<wallet>"),
]
IPV4 = re.compile(r"(?<![\d.])(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(?![\d.])")
BASE58 = re.compile(r"(?<![1-9A-HJ-NP-Za-km-z])[1-9A-HJ-NP-Za-km-z]{43,44}(?![1-9A-HJ-NP-Za-km-z])")


def _ip(match: re.Match) -> str:
    text = match.group(1)
    try:
        address = ipaddress.ip_address(text)
    except ValueError:  # a version number, say
        return text
    return text if address.is_private or address.is_loopback or address.is_unspecified else "<public-ip>"


def _address(match: re.Match) -> str:
    text = match.group(0)
    if text in KEEP or text.isdigit() or text.isalpha() and text.islower():
        return text
    return text[:8] + "…"


def scrub(text: str) -> str:
    for secret, name in NAMED.items():
        text = text.replace(secret, name)
    for pattern, name in PATTERNS:
        text = pattern.sub(name, text)
    return BASE58.sub(_address, IPV4.sub(_ip, text))


def clean_files() -> int:
    changed = 0
    for path in [*sorted((HERE / "logs").rglob("*")), HERE / "results.json", HERE / "results_final.json"]:
        if path.is_file():
            if path.suffix == ".pcap":
                continue
            before = path.read_text(errors="replace")
            after = scrub(before)
            if after != before:
                path.write_text(after)
                changed += 1
    return changed


if __name__ == "__main__":
    print(f"{clean_files()} files changed")
