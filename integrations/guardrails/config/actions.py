"""The one decision in this rail. Guardrails loads this file from the config directory.

Secrets come from the file named by BEETLE_RAIL_SECRETS: a JSON list of strings, or
an object with a `secrets` list (export_canaries.py writes this). A match survives
changes of case, spacing and punctuation, reversal, and base64. It does not survive
a translation, a paraphrase or a part of the string, and it can't see a secret nobody
listed. Hits are counted, never logged.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
from pathlib import Path

from nemoguardrails.actions import action

TOKEN = re.compile(r"[A-Za-z0-9+/]{12,}={0,2}")


def squeeze(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def load_secrets() -> list[str]:
    path = os.environ.get("BEETLE_RAIL_SECRETS")
    if not path:
        raise SystemExit("set BEETLE_RAIL_SECRETS to a file of strings to withhold")
    data = json.loads(Path(path).read_text())
    items = data["secrets"] if isinstance(data, dict) else data
    return sorted({squeeze(s) for s in items if squeeze(s)})


SECRETS = load_secrets()


def forms(text: str) -> list[str]:
    flat = squeeze(text)
    out = [flat, flat[::-1]]
    for token in TOKEN.findall(text):
        try:
            out.append(squeeze(base64.b64decode(token + "=" * (-len(token) % 4)).decode("utf-8", "ignore")))
        except (binascii.Error, ValueError):
            pass
    return out


@action(name="beetle_scan")
async def beetle_scan(text: str = "") -> bool:
    seen = forms(text or "")
    return any(secret in form for secret in SECRETS for form in seen)
