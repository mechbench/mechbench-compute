"""Rate limits as a first-class currency (task 000337 amendment; the
runner's implementation is 000338).

The transport does not sleep on its own guesses. It asks a `Limiter`
for permission in a named CURRENCY — requests, input tokens, output
tokens, concurrent calls — and it REPORTS what the provider said in
the response headers, so the limiter's model of the account comes from
the account, not from a config file. On a 429 the wait is the header's
reset, never a guess.

Waiting is visible: `acquire` returns the seconds it waited and the
transport reports them as progress detail ("throttled: 12 s"). A job
that is slow because someone else's quota is full should say so.

`NullLimiter` is the default (single-job laptop runs), and the tests
use a recording one — the whole interface is four methods so the
runner's real limiter (shared across jobs, persistent across restarts)
can implement it without importing anything from here.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

CURRENCIES = ("requests", "input_tokens", "output_tokens", "concurrency")


class Limiter(Protocol):
    def acquire(self, provider: str, model: str, scope: str, currency: str,
                amount: float) -> float:
        """Block until `amount` of `currency` is available. Returns the
        seconds waited (0.0 when it was free)."""

    def observe(self, provider: str, model: str, scope: str,
                limits: RateLimits) -> None:
        """What the provider's headers said after a call."""

    def penalize(self, provider: str, model: str, scope: str,
                 retry_after: float) -> None:
        """A 429 arrived: hold everything on this scope for this long."""

    def release(self, provider: str, model: str, scope: str,
                currency: str, amount: float) -> None:
        """Give back concurrency (or an unspent token estimate)."""


@dataclass(frozen=True)
class RateLimits:
    """What one response told us about the account's remaining quota.
    Every field is optional — providers report different subsets, and
    a missing field means "unknown", never "zero"."""

    requests_limit: int | None = None
    requests_remaining: int | None = None
    requests_reset: float | None = None       # seconds from now
    input_tokens_limit: int | None = None
    input_tokens_remaining: int | None = None
    output_tokens_limit: int | None = None
    output_tokens_remaining: int | None = None
    tokens_reset: float | None = None
    retry_after: float | None = None

    def to_wire(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @staticmethod
    def from_headers(headers: Mapping[str, str]) -> RateLimits:
        """Anthropic (`anthropic-ratelimit-*`), OpenAI-compatible
        (`x-ratelimit-*`) and plain `retry-after` in one reader: the
        header names differ, the meaning does not."""
        h = {str(k).lower(): str(v) for k, v in headers.items()}

        def _int(*names: str) -> int | None:
            for n in names:
                if n in h:
                    try:
                        return int(float(h[n]))
                    except ValueError:
                        return None
            return None

        def _reset(*names: str) -> float | None:
            for n in names:
                if n in h:
                    return _duration(h[n])
            return None

        return RateLimits(
            requests_limit=_int("anthropic-ratelimit-requests-limit",
                                "x-ratelimit-limit-requests"),
            requests_remaining=_int("anthropic-ratelimit-requests-remaining",
                                    "x-ratelimit-remaining-requests"),
            input_tokens_limit=_int("anthropic-ratelimit-input-tokens-limit",
                                    "x-ratelimit-limit-tokens"),
            output_tokens_limit=_int("anthropic-ratelimit-output-tokens-limit"),
            requests_reset=_reset("anthropic-ratelimit-requests-reset",
                                  "x-ratelimit-reset-requests"),
            input_tokens_remaining=_int(
                "anthropic-ratelimit-input-tokens-remaining",
                "x-ratelimit-remaining-tokens"),
            output_tokens_remaining=_int(
                "anthropic-ratelimit-output-tokens-remaining"),
            tokens_reset=_reset("anthropic-ratelimit-tokens-reset",
                                "x-ratelimit-reset-tokens"),
            retry_after=_duration(h["retry-after"]) if "retry-after" in h else None,
        )


_DUR = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(ms|s|m|h)?\s*$", re.IGNORECASE)


def _duration(value: str) -> float | None:
    """`"20"` (seconds), `"1.5s"`, `"350ms"`, `"2m"`, or an RFC-3339
    reset timestamp -> seconds from now. Unparseable -> None, which
    reads as "unknown" everywhere it lands."""
    m = _DUR.match(value)
    if m:
        n = float(m.group(1))
        unit = (m.group(2) or "s").lower()
        return n * {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}[unit]
    try:
        from datetime import UTC, datetime

        when = datetime.fromisoformat(value)   # 3.11+ parses a "Z" suffix
        return max(0.0, (when - datetime.now(UTC)).total_seconds())
    except (ValueError, TypeError):
        return None


def scope_for(provider: str, credential: Mapping[str, Any] | None) -> str:
    """The limiter's key scope: WHICH ACCOUNT this is, without being
    able to say which key. Two keys for one provider must not share a
    bucket, and nothing the limiter persists may contain a secret, so
    the scope is a short hash of the credential."""
    import hashlib

    token = ""
    if isinstance(credential, Mapping):
        token = str(credential.get("token") or credential.get("base_url") or "")
    if not token:
        return "default"
    return hashlib.sha256(f"{provider}:{token}".encode()).hexdigest()[:12]


class NullLimiter:
    """No limits known, nothing to wait for. Records observations so a
    caller can still surface them."""

    def __init__(self) -> None:
        self.last: RateLimits | None = None

    def acquire(self, provider, model, scope, currency, amount) -> float:
        return 0.0

    def observe(self, provider, model, scope, limits) -> None:
        self.last = limits

    def penalize(self, provider, model, scope, retry_after) -> None:
        return None

    def release(self, provider, model, scope, currency, amount) -> None:
        return None


@dataclass
class RecordingLimiter(NullLimiter):
    """A test limiter: no waiting, full trace of what was asked for."""

    acquired: list[tuple] = field(default_factory=list)
    penalties: list[tuple] = field(default_factory=list)
    observations: list[RateLimits] = field(default_factory=list)

    def acquire(self, provider, model, scope, currency, amount) -> float:
        self.acquired.append((provider, model, scope, currency, amount))
        return 0.0

    def observe(self, provider, model, scope, limits) -> None:
        self.last = limits
        self.observations.append(limits)

    def penalize(self, provider, model, scope, retry_after) -> None:
        self.penalties.append((provider, model, scope, retry_after))
