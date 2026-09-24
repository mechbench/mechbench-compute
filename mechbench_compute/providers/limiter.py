from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

CURRENCIES = ("requests", "input_tokens", "output_tokens", "concurrency")


class Limiter(Protocol):
    def acquire(self, provider: str, model: str, scope: str, currency: str,
                amount: float) -> float:
        ...

    def observe(self, provider: str, model: str, scope: str,
                limits: RateLimits) -> None:
        ...

    def penalize(self, provider: str, model: str, scope: str,
                 retry_after: float) -> None:
        ...

    def release(self, provider: str, model: str, scope: str,
                currency: str, amount: float) -> None:
        ...


@dataclass(frozen=True)
class RateLimits:
    requests_limit: int | None = None
    requests_remaining: int | None = None
    requests_reset: float | None = None
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
    m = _DUR.match(value)
    if m:
        n = float(m.group(1))
        unit = (m.group(2) or "s").lower()
        return n * {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}[unit]
    try:
        from datetime import UTC, datetime

        when = datetime.fromisoformat(value)
        return max(0.0, (when - datetime.now(UTC)).total_seconds())
    except (ValueError, TypeError):
        return None


def scope_for(provider: str, credential: Mapping[str, Any] | None) -> str:
    import hashlib

    token = ""
    if isinstance(credential, Mapping):
        token = str(credential.get("token") or credential.get("base_url") or "")
    if not token:
        return "default"
    return hashlib.sha256(f"{provider}:{token}".encode()).hexdigest()[:12]


class NullLimiter:
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
