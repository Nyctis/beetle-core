"""This repository is public: it must hold no keys and name no one deployment.

Anyone can deploy their own copy, so program ids, wallets, mints and hosts come
from chain.json, the scripts/networks files and command-line arguments. The only
addresses allowed in source are protocol constants, listed here.
"""

from __future__ import annotations

import ipaddress
import re
import string
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKIP = {"uv.lock", "programs/beetle/Cargo.lock", "tests/test_public_repo.py"}
BINARY_SUFFIXES = {".pem", ".png", ".bin"}

PROTOCOL_ADDRESSES = {
    "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",   # SPL Token
    "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL",  # associated token accounts
    "Sysvar1nstructions1111111111111111111111111", "SysvarC1ock11111111111111111111111111111111",
    "Ed25519SigVerify111111111111111111111111111",
    "BPFLoaderUpgradeab1e11111111111111111111111",
    "5eykt4UsFv8P8NJdTREpY1vzqKqZKvdpKuc147dw2N9d",  # mainnet genesis hash
    "SysvarRent111111111111111111111111111111111",
    "MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr",   # SPL Memo
    # In scripts/networks only: public addresses every deployment shares.
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # Circle's USDC
    "CHNLxYvVA28MJP9PrFuDXccuoGXAx7jBacfLEkahyGsX",  # Solana Foundation payment channels
    "Cs2zdfUNonRdRGsiZUQQLdTxzxVvJZmgiX2mpLYKuEqP",  # and its treasury owner
    "4zTeC5mVqWLruDexgU2mV66p9t5vCA9JyiZqdGDUspap",  # and its treasury owner on devnet
}
LOOPBACK = {"127.0.0.1", "0.0.0.0", "1.1.1.1"}  # 1.1.1.1 is the sandbox's "no network" probe
EXAMPLE_NETS = ("192.0.2.", "198.51.100.", "203.0.113.")  # RFC 5737, reserved for documentation

BASE58_32 = re.compile(r"(?<![1-9A-HJ-NP-Za-km-z])[1-9A-HJ-NP-Za-km-z]{43,44}(?![1-9A-HJ-NP-Za-km-z])")
IPV4 = re.compile(r"(?<![\d.])(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(?![\d.])")
JSON_KEYPAIR = re.compile(r"\[\s*(\d{1,3}\s*,\s*){63}\d{1,3}\s*\]")
PEM_KEY = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")
SECRETS = {
    "a base58 secret key": re.compile(r"(?<![1-9A-HJ-NP-Za-km-z])[1-9A-HJ-NP-Za-km-z]{87,88}(?![1-9A-HJ-NP-Za-km-z])"),
    "a Beetle token": re.compile(r"bk_local_[0-9a-f]{8,}"),
    "a Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
    "a GitHub token": re.compile(r"\b(?:ghp|gho|ghs|github_pat)_[A-Za-z0-9_]{20,}"),
    "an NVIDIA key": re.compile(r"\bnvapi-[A-Za-z0-9_-]{20,}"),
    "an API key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
    "an AWS key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
}
PERSONAL = re.compile(r"/Users/[A-Za-z]|/home/ubuntu|beetle-(?:gpu-1|l4|cpu-1|small-gpu)\b")
CJK = re.compile("[\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]")


def git_lines(*args: str) -> list[str]:
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    except OSError:
        pytest.skip("no git here")
    if out.returncode:
        pytest.skip("not a git checkout")
    return [line for line in out.stdout.splitlines() if line]


def published_files() -> list[str]:
    """Everything a `git add .` would publish: tracked or new, minus what's ignored."""
    return [f for f in git_lines("ls-files", "--cached", "--others", "--exclude-standard")
            if f not in git_lines("ls-files", "--cached", "--ignored", "--exclude-standard")]


def text(path: str) -> str | None:
    p = ROOT / path
    if path in SKIP or p.suffix in BINARY_SUFFIXES or path.startswith(("tests/fixtures/", "mac/")):
        return None
    try:
        return p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def test_no_key_files_are_published():
    for f in published_files():
        name = Path(f).name
        assert not f.startswith(".keys/") and "keypair" not in name, f
        assert not name.endswith((".key", "id.json")), f


def test_no_key_material_in_published_text():
    for f in published_files():
        body = text(f)
        if body:
            assert not JSON_KEYPAIR.search(body), f"{f} holds a keypair"
            assert not PEM_KEY.search(body), f"{f} holds a private key"
            for name, pattern in SECRETS.items():
                assert not pattern.search(body), f"{f} holds {name}"


def test_no_personal_paths_or_machine_names_in_published_text():
    assert not [f for f in published_files() if (body := text(f)) and PERSONAL.search(body)]


def test_published_text_is_english():
    assert not [f for f in published_files() if CJK.search(f) or ((body := text(f)) and CJK.search(body))]


def is_address(text: str) -> bool:
    """A long hex digest also fits the base58 alphabet, but a real key isn't all hex."""
    return text not in PROTOCOL_ADDRESSES and not set(text) <= set(string.hexdigits)


def is_public_host(ip: str) -> bool:
    if ip in LOOPBACK or ip.startswith(EXAMPLE_NETS):
        return False
    try:
        return not ipaddress.ip_address(ip).is_private
    except ValueError:  # a version number, say
        return False


def test_no_deployment_addresses_or_hosts_in_source():
    found = []
    for f in published_files():
        body = text(f)
        if body:
            found += [(f, a) for a in BASE58_32.findall(body) if is_address(a)]
            found += [(f, ip) for ip in IPV4.findall(body) if is_public_host(ip)]
    assert not found


def test_key_directory_and_generated_config_are_ignored():
    ignore = (ROOT / ".gitignore").read_text().split()
    assert {".keys", "chain.json", ".env", "*.key", "wallet.json", "tokens.json"} <= set(ignore)


def test_keys_stay_out_of_docker_builds():
    ignore = (ROOT / ".dockerignore").read_text().split()
    assert {".keys", ".env", "*.key", "wallet.json", "tokens.json"} <= set(ignore)


def test_nothing_ignored_is_still_tracked():
    """An ignored file that is already tracked would still be published."""
    assert git_lines("ls-files", "--cached", "--ignored", "--exclude-standard") == []
