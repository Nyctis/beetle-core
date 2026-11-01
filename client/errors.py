"""Every refusal a caller can see, with the HTTP status the local API gives it."""

from __future__ import annotations


class BeetleError(Exception):
    def __init__(self, status: int, code: str, message: str, **extra):
        super().__init__(message)
        self.status, self.code, self.message, self.extra = status, code, message, extra

    def body(self) -> dict:
        return {"error": {"code": self.code, "message": self.message, **self.extra}}


def budget_exceeded(message: str) -> BeetleError:
    return BeetleError(402, "budget_exceeded", message)


def approval_required(message: str, approval: str) -> BeetleError:
    return BeetleError(403, "approval_required", message, approval=approval,
                       how=f"a person runs: beetle approve {approval}")


def no_funds(message: str) -> BeetleError:
    return BeetleError(402, "no_funds", message)


def local_only(message: str) -> BeetleError:
    """A way of running Beetle, not a failure: nothing here needs a network."""
    return BeetleError(409, "local_only", message)


def no_route(message: str, **extra) -> BeetleError:
    """Never downgrades: says what is possible and stops."""
    return BeetleError(409, "no_route_meets_floor", message, **extra)


def evidence_rejected(message: str) -> BeetleError:
    return BeetleError(502, "evidence_rejected", message)


def no_effect(message: str) -> BeetleError:
    return BeetleError(502, "adapter_no_effect", message)


def no_seller(message: str, retry_after: int = 60) -> BeetleError:
    return BeetleError(503, "no_seller", message, retry_after=retry_after)


def bad_request(message: str) -> BeetleError:
    return BeetleError(400, "bad_request", message)


def not_found(message: str) -> BeetleError:
    return BeetleError(404, "not_found", message)
