"""The two policies nothing runs without: what may be spent, and the privacy floor.

Tiers are not numbers and T3 < T3-B < T1 < T2 is not their numbering on chain, so
a floor compiles to the set of tiers it accepts. Three floors apply to any
request, and the strictest wins:

    effective floor = strictest(global policy, the data's label, the request)

A request can only tighten. The floor is enforced on this machine right before
the data key is released, on the tier the evidence proved. Routing merely avoids
wasting a trip: if routing is wrong, the sealed data stays sealed.
"""

from __future__ import annotations

import time
from pathlib import Path

from client.errors import approval_required, bad_request, budget_exceeded, not_found
from node.wire import read_json, write_json

USDC = 1_000_000
# Weakest to strictest. T2+ also needs the freshness and GPU flags that no
# evidence here proves yet, so it accepts nothing remote until that exists.
FLOORS = ("public", "T3-B", "T1", "T2", "T2+", "T0")
ACCEPT = {"T0": set(), "T2+": set(), "T2": {"T2"}, "T1": {"T1", "T2"},
          "T3-B": {"T3-B", "T1", "T2"}, "public": {"MOCK", "T3", "T3-C", "T3-B", "T1", "T2"}}
LABEL_FLOOR = {"local_only": "T0", "secret": "T2", "personal": "T1",
               "synthetic": "public", "public": "public"}
# What a chain tier means here. MOCK proves nothing and exists for development.
CHAIN_TIER = {0: "MOCK", 1: "T1", 2: "T2", 3: "T3-B", 4: "T3-C"}
# Inference shows no evidence yet, so this is what any session counts as, whatever
# its offer claims.
SESSION_TIER = "T3-C"
DEFAULTS = {"auto_approve_under_usd": 1.0, "max_concurrent": 3, "max_per_hour": 6,
            "infer_deposit_usd": 0.5}


def check_tier(name: str | None) -> str | None:
    if name is not None and name not in FLOORS:
        raise bad_request(f"unknown tier {name!r}; one of {', '.join(FLOORS)}")
    return name


def strictest(*floors: str | None) -> str:
    return max((f for f in floors if f), key=FLOORS.index)


def accepts(floor: str, tier: str) -> bool:
    return tier in ACCEPT[floor]


class Policy:
    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> dict:
        return {**DEFAULTS, **(read_json(self.path) or {})}

    def save(self, **changes) -> dict:
        policy = {**self.load(), **{k: v for k, v in changes.items() if v is not None}}
        write_json(self.path, policy)
        return policy

    def ready(self) -> dict:
        policy = self.load()
        if not all(k in policy for k in ("per_job_usd", "per_day_usd", "min_tier")):
            raise bad_request("set a spending limit and a privacy floor first: beetle init")
        return policy


class Ledger:
    """used = settled + in escrow + reserved. A refund gives the room back. Each
    entry is one job or one channel; `usd` is what it holds against the limit."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.entries: dict[str, dict] = read_json(self.path, {})

    def _save(self) -> None:
        write_json(self.path, self.entries)

    def used_today(self) -> float:
        since = time.time() - 86_400
        return sum(e["usd"] for e in self.entries.values() if e["at"] > since)

    def reserve(self, key: str, usd: float, kind: str, policy: dict, cap: float | None = None,
                approved: bool = False, approvals=None, reason: dict | None = None) -> None:
        """Runs before any money moves. Over the limit stops; over the automatic
        amount needs a person, who can't be stood in for by an agent's token."""
        limit = min(policy["per_job_usd"], cap if cap is not None else float("inf"))
        if usd > limit:
            raise budget_exceeded(f"${usd:.4f} is over this request's limit of ${limit:.2f}")
        room = policy["per_day_usd"] - self.used_today()
        if usd > room:
            raise budget_exceeded(f"${usd:.4f} is over what is left today, ${room:.4f}")
        if usd > policy["auto_approve_under_usd"] and not approved:
            ticket = approvals.request({**(reason or {}), "usd": usd, "why": "over the automatic amount"})
            raise approval_required(f"${usd:.2f} needs a person's approval", ticket)
        self.entries[key] = {"kind": kind, "usd": usd, "state": "reserved", "at": time.time()}
        self._save()

    def update(self, key: str, state: str, usd: float | None = None) -> None:
        entry = self.entries[key]
        entry["state"] = state
        if usd is not None:
            entry["usd"] = usd
        self._save()

    def note(self, key: str, **fields) -> None:
        self.entries[key].update(fields)
        self._save()

    def release(self, key: str) -> None:
        self.update(key, "released", 0.0)

    def summary(self) -> dict:
        totals = {"reserved": 0.0, "escrowed": 0.0, "settled": 0.0}
        for e in self.entries.values():
            if e["state"] in totals:
                totals[e["state"]] += e["usd"]
        return {**totals, "used_today": self.used_today()}


class Approvals:
    """Requests that wait for a person. Only `beetle approve`, run by that person
    with the owner's token, can turn one on."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def _all(self) -> dict:
        return read_json(self.path, {})

    def request(self, what: dict) -> str:
        import secrets

        pending = self._all()
        for ticket, entry in pending.items():
            if entry["what"] == what and not entry.get("used"):
                return ticket
        ticket = secrets.token_hex(4)
        pending[ticket] = {"what": what, "approved": False, "at": time.time()}
        write_json(self.path, pending)
        return ticket

    def granted(self, what: dict) -> bool:
        return any(e["approved"] and not e.get("used") and e["what"] == what for e in self._all().values())

    def take(self, *whats: dict) -> list[str]:
        """One approval lets one job through. The caller takes it before anything
        can run in between, so two jobs can't share one."""
        pending, taken = self._all(), []
        for what in whats:
            for ticket, entry in pending.items():
                if entry["approved"] and not entry.get("used") and entry["what"] == what:
                    entry["used"] = time.time()
                    taken.append(ticket)
                    break
        write_json(self.path, pending)
        return taken

    def give_back(self, tickets: list[str]) -> None:
        """The job never opened, so nothing was spent and the approval still stands."""
        pending = self._all()
        for ticket in tickets:
            del pending[ticket]["used"]
        write_json(self.path, pending)

    def approve(self, ticket: str) -> dict:
        pending = self._all()
        if ticket not in pending:
            raise not_found(f"no approval {ticket}")
        if pending[ticket].get("used"):
            raise bad_request(f"approval {ticket} was already used: the next request gets a new one")
        pending[ticket]["approved"] = True
        write_json(self.path, pending)
        return pending[ticket]["what"]

    def listing(self) -> dict:
        return self._all()
