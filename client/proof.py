"""The proof card: who this connection is to, what tier, what was verified.

Two lamps, never merged into one lock: `on_the_way` (can a relay or the network
read this) and `on_their_machine` (can the owner of that machine). The wording
below is fixed, indexed by what the evidence showed. A seller supplies no text.
When in doubt the card shows the worse state.
"""

from __future__ import annotations

import time

# (headline, second line, on their machine, color)
CARDS = {
    "T0": ("Only on your device", "Nothing was sent to anyone.", "not involved", "neutral"),
    "T2": ("Confidential server, hardware evidence passed",
           "Its owner cannot read this by ordinary means. Someone with physical access can.",
           "hardware isolated", "blue"),
    "T1": ("Mac curiosity guard, evidence passed",
           "Stops a curious owner. Does not stop system bugs, side channels or opening the machine.",
           "software isolated", "gray-blue"),
    "T3-B": ("No confidential protection",
             "It can prove it runs our program, but the owner can read memory.",
             "the other side can read it", "orange"),
    "T3-C": ("No protection",
             "An ordinary machine: the provider and anyone with root can read all of it. "
             "Public or synthetic data only.",
             "the other side can read it", "red"),
    "MOCK": ("Demo mode", "The evidence is simulated and shows nothing. Treat the other side as able to read it.",
             "the other side can read it", "red"),
    "rejected": ("Evidence failed, the key was not released", "Your data is still ciphertext.",
                 "nothing was sent", "red"),
    "no_evidence": ("No evidence yet", "The key has not been released, so your data is still ciphertext.",
                    "ciphertext only", "neutral"),
}
ON_THE_WAY = "encrypted"
NOT_PROTECTED = ("T3-B", "T3-C")
HARDWARE = ["GPU evidence is checked only at startup", "physical attacks",
            "flaws the vendor has not fixed"]
NO_EVIDENCE = ["the machine's owner and anyone with root can read all of it",
               "nothing proves which GPU or program ran"]
WEAKNESSES = {"T0": [], "rejected": [], "no_evidence": [], "T2": HARDWARE, "T1": HARDWARE,
              "T3-B": ["the machine's owner can read its memory", "flaws the vendor has not fixed"]}


def path_line(path: dict | None) -> str:
    if not path or path.get("direct") is None:
        return "unknown"
    if path["direct"]:
        return f"direct, RTT {path['rtt_ms']} ms. The other side can see your IP address"
    return "relayed. The relay cannot read content; it sees both addresses, times and sizes"


def card(tier: str, *, peer: dict | None = None, evidence: dict | None = None,
         path: dict | None = None, bindings: dict | None = None) -> dict:
    headline, detail, there, color = CARDS.get(tier, CARDS["T3-C"])
    return {"tier": tier, "headline": headline, "detail": detail, "color": color,
            "on_the_way": ON_THE_WAY if tier != "T0" else "not involved",
            "on_their_machine": there, "peer": peer or {}, "evidence": evidence or {},
            "path": path_line(path), "path_raw": path or {}, "bindings": bindings or {},
            "known_weaknesses": WEAKNESSES.get(tier, NO_EVIDENCE), "made_at": int(time.time())}


def local_card() -> dict:
    return card("T0")
