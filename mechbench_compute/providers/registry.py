from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from mechbench_compute.providers import pricing
from mechbench_compute.providers.base import Capabilities
from mechbench_compute.providers.limiter import RateLimits

REGISTRY_VERSION = "2026-09-23"


@dataclass(frozen=True)
class Limits:
    requests: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    concurrency: int | None = None
    batch_concurrency: int | None = None
    daily_requests: int | None = None

    def for_currency(self, currency: str) -> int | None:
        return getattr(self, currency, None)

    def to_wire(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    adapter: str
    base_url: str
    capabilities: Capabilities
    limits: Limits
    models: Mapping[str, Limits] = None  # type: ignore[assignment]

    def limits_for(self, model: str) -> Limits:
        best: tuple[int, Limits] | None = None
        for prefix, over in (self.models or {}).items():
            if model.startswith(prefix) and (best is None or len(prefix) > best[0]):
                best = (len(prefix), over)
        if best is None:
            return self.limits
        merged = {k: v for k, v in best[1].__dict__.items() if v is not None}
        return replace(self.limits, **merged)

    def to_wire(self) -> dict[str, Any]:
        return {"provider": self.name, "adapter": self.adapter,
                "base_url": self.base_url, "limits": self.limits.to_wire(),
                "capabilities": self.capabilities.to_wire(),
                "registry_version": REGISTRY_VERSION,
                "price_table": pricing.TABLE_VERSION}


def _capabilities(provider: str) -> Capabilities:
    from mechbench_compute.providers import capabilities

    return capabilities(provider)


def build() -> dict[str, ProviderSpec]:
    from mechbench_compute.providers.anthropic import DEFAULT_BASE_URL as ANTHROPIC_BASE
    from mechbench_compute.providers.gemini import DEFAULT_BASE_URL as GEMINI_BASE
    from mechbench_compute.providers.openai_compatible import HOSTS

    return {
        "anthropic": ProviderSpec(
            name="anthropic", adapter="anthropic", base_url=ANTHROPIC_BASE,
            capabilities=_capabilities("anthropic"),
            limits=Limits(requests=50, input_tokens=30_000, output_tokens=8_000,
                          concurrency=8),
            models={"claude-haiku": Limits(requests=50, input_tokens=50_000)}),
        "openai": ProviderSpec(
            name="openai", adapter="openai_compatible", base_url=HOSTS["openai"][0],
            capabilities=_capabilities("openai"),
            limits=Limits(requests=500, input_tokens=30_000, concurrency=16)),
        "xai": ProviderSpec(
            name="xai", adapter="openai_compatible", base_url=HOSTS["xai"][0],
            capabilities=_capabilities("xai"),
            limits=Limits(requests=60, input_tokens=16_000, concurrency=8)),
        "gemini": ProviderSpec(
            name="gemini", adapter="gemini", base_url=GEMINI_BASE,
            capabilities=_capabilities("gemini"),
            limits=Limits(requests=150, input_tokens=1_000_000, concurrency=8,
                          daily_requests=10_000)),
        "fireworks": ProviderSpec(
            name="fireworks", adapter="openai_compatible",
            base_url=HOSTS["fireworks"][0], capabilities=_capabilities("fireworks"),
            limits=Limits(requests=600, concurrency=32)),
        # external: DeepSeek — publishes a concurrency limit per model and no per-minute quota
        "deepseek": ProviderSpec(
            name="deepseek", adapter="openai_compatible",
            base_url=HOSTS["deepseek"][0], capabilities=_capabilities("deepseek"),
            limits=Limits(concurrency=16)),
        "openai-compatible": ProviderSpec(
            name="openai-compatible", adapter="openai_compatible", base_url="",
            capabilities=_capabilities("openai-compatible"),
            limits=Limits(concurrency=8)),
        "mock": ProviderSpec(
            name="mock", adapter="mock", base_url="",
            capabilities=_capabilities("mock"), limits=Limits()),
    }


_REGISTRY: dict[str, ProviderSpec] | None = None


def registry() -> dict[str, ProviderSpec]:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = build()
    return _REGISTRY


def spec_for(provider: str) -> ProviderSpec:
    try:
        return registry()[provider]
    except KeyError:
        raise ValueError(
            f"unknown provider {provider!r} — known: "
            f"{', '.join(sorted(registry()))}") from None


def limits_for(provider: str, model: str = "") -> Limits:
    return spec_for(provider).limits_for(model)


@dataclass
class Bucket:
    capacity: float
    per_second: float
    tokens: float
    updated: float
    next_ticket: int = 0
    serving: int = 0

    def refill(self, now: float) -> None:
        if now > self.updated:
            self.tokens = min(self.capacity,
                              self.tokens + (now - self.updated) * self.per_second)
            self.updated = now

    def wait_for(self, amount: float) -> float:
        need = min(amount, self.capacity) - self.tokens
        if need <= 0:
            return 0.0
        return need / self.per_second if self.per_second > 0 else float("inf")


class TokenBucketLimiter:
    def __init__(self, *, sleep=None, clock=None, limits=None,
                 max_waits: int = 64, slot_timeout: float = 30.0) -> None:
        import time

        self._sleep = sleep if sleep is not None else time.sleep
        self._clock = clock if clock is not None else time.monotonic
        self._limits_for = limits if limits is not None else limits_for
        self._buckets: dict[tuple, Bucket] = {}
        self._holds: dict[tuple, float] = {}
        self._lock = threading.RLock()
        self._slots = threading.Condition(self._lock)
        self._max_waits = max_waits
        self._slot_timeout = slot_timeout
        self.waited_seconds = 0.0

    def acquire(self, provider: str, model: str, scope: str, currency: str,
                amount: float) -> float:
        waited = 0.0
        with self._lock:
            now = self._clock()
            bucket = self._bucket(provider, model, scope, currency, now)
            if bucket is None:
                return waited
            ticket = bucket.next_ticket
            bucket.next_ticket += 1
            try:
                for _ in range(self._max_waits):
                    while bucket.serving != ticket:
                        self._slots.wait(timeout=self._slot_timeout)
                    now = self._clock()
                    hold = self._holds.get((provider, scope), 0.0)
                    if now < hold:
                        delay = hold - now
                    else:
                        bucket.refill(now)
                        want = min(float(amount), bucket.capacity)
                        if bucket.tokens >= want:
                            bucket.tokens -= want
                            return waited
                        if bucket.per_second <= 0:
                            start = self._clock()
                            self._slots.wait(timeout=self._slot_timeout)
                            waited += max(0.0, self._clock() - start)
                            self.waited_seconds = round(
                                self.waited_seconds + waited, 6)
                            continue
                        delay = bucket.wait_for(want)
                    self._lock.release()
                    try:
                        self._sleep(delay)
                    finally:
                        self._lock.acquire()
                    waited += delay
                    self.waited_seconds = round(self.waited_seconds + delay, 6)
            finally:
                bucket.serving = ticket + 1
                self._slots.notify_all()
        from mechbench_compute.providers.errors import ProviderUnavailable

        raise ProviderUnavailable(
            f"{provider}/{currency} did not free up after "
            f"{self._max_waits} waits — the account's limit may be lower "
            f"than the registry's seed, or another process is draining it",
            provider=provider)

    def observe(self, provider: str, model: str, scope: str,
                limits: RateLimits) -> None:
        with self._lock:
            now = self._clock()
            pairs = (
                ("requests", limits.requests_remaining, limits.requests_reset,
                 limits.requests_limit),
                ("input_tokens", limits.input_tokens_remaining,
                 limits.tokens_reset, limits.input_tokens_limit),
                ("output_tokens", limits.output_tokens_remaining,
                 limits.tokens_reset, limits.output_tokens_limit),
            )
            for currency, remaining, reset, limit in pairs:
                if remaining is None and limit is None:
                    continue
                bucket = self._bucket(provider, model, scope, currency, now)
                if bucket is not None:
                    bucket.refill(now)
                    if limit is not None and float(limit) > bucket.capacity:
                        bucket.capacity = float(limit)
                        bucket.per_second = float(limit) / 60.0
                        bucket.tokens = max(bucket.tokens, 0.0)
                    if remaining is not None:
                        bucket.tokens = max(0.0, min(float(remaining),
                                                     bucket.capacity))
                if remaining is not None and remaining <= 0 and reset:
                    self._hold(provider, scope, now + float(reset))
            if limits.retry_after:
                self._hold(provider, scope, now + float(limits.retry_after))

    def penalize(self, provider: str, model: str, scope: str,
                 retry_after: float) -> None:
        with self._lock:
            now = self._clock()
            self._hold(provider, scope, now + max(float(retry_after or 0.0), 1.0))

    def release(self, provider: str, model: str, scope: str, currency: str,
                amount: float) -> None:
        with self._lock:
            key = (provider, self._model_key(provider, model), scope, currency)
            bucket = self._buckets.get(key)
            if bucket is not None:
                bucket.tokens = min(bucket.capacity, bucket.tokens + float(amount))
                self._slots.notify_all()

    def _hold(self, provider: str, scope: str, until: float) -> None:
        key = (provider, scope)
        self._holds[key] = max(self._holds.get(key, 0.0), until)

    @staticmethod
    def _model_key(provider: str, model: str) -> str:
        return model or "*"

    def _bucket(self, provider: str, model: str, scope: str, currency: str,
                now: float) -> Bucket | None:
        key = (provider, self._model_key(provider, model), scope, currency)
        bucket = self._buckets.get(key)
        if bucket is not None:
            return bucket
        limit = self._limits_for(provider, model).for_currency(currency)
        if limit is None:
            return None
        per_second = 0.0 if currency == "concurrency" else float(limit) / 60.0
        bucket = Bucket(capacity=float(limit), per_second=per_second,
                        tokens=float(limit), updated=now)
        self._buckets[key] = bucket
        return bucket
