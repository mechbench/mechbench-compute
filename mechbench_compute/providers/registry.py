"""The provider registry and the rate-limit model (task 000344).

Everything that differs between providers, in one versioned table:
which adapter speaks to them, where they live, what they can do
(000337's capability matrix), what they charge (`pricing.py`), and
what they will let an account do per minute. `REGISTRY_VERSION` is
recorded in manifests beside the price-table version, so a run's
throttling and its bill are both explainable from what it recorded.

The default limits are the published FREE-ish tier floors, deliberately
conservative: they are a starting point the limiter CORRECTS from the
provider's own response headers within a few calls. A seeded number
that is too low costs a little throughput; one that is too high costs
429s and a jittery run.

Rate limits are currencies, not one number: requests, input tokens,
output tokens and concurrent calls are metered separately by every
provider here, and a job can be starved on any one of them.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from mechbench_compute.providers import pricing
from mechbench_compute.providers.base import Capabilities
from mechbench_compute.providers.limiter import RateLimits

REGISTRY_VERSION = "2026-09-08"


@dataclass(frozen=True)
class Limits:
    """Per key scope, per minute unless named otherwise. None = unknown
    (which means unlimited HERE — the provider's own 429 is still the
    authority, and the limiter learns from it)."""

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
    adapter: str                     # anthropic | gemini | openai_compatible | mock
    base_url: str
    capabilities: Capabilities
    limits: Limits
    #: Model prefix -> overrides, longest prefix wins.
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
        "openai-compatible": ProviderSpec(
            name="openai-compatible", adapter="openai_compatible", base_url="",
            capabilities=_capabilities("openai-compatible"),
            # A server someone runs themselves: no published quota to
            # seed. Concurrency is the one real constraint, and it is
            # the machine's, not an account's.
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


# --- the reference limiter --------------------------------------------------------


@dataclass
class Bucket:
    capacity: float
    per_second: float
    tokens: float
    updated: float

    def refill(self, now: float) -> None:
        if now > self.updated:
            self.tokens = min(self.capacity,
                              self.tokens + (now - self.updated) * self.per_second)
            self.updated = now

    def wait_for(self, amount: float) -> float:
        """Seconds until `amount` is available (0 when it already is).
        A request larger than the whole bucket waits for a full bucket
        and then goes: refusing it forever would be worse than being
        briefly over."""
        need = min(amount, self.capacity) - self.tokens
        if need <= 0:
            return 0.0
        return need / self.per_second if self.per_second > 0 else float("inf")


class TokenBucketLimiter:
    """An in-process limiter: one bucket per (provider, model, scope,
    currency), seeded from the registry and corrected by what the
    provider's headers say.

    Concurrency is a bucket that does not refill — `acquire` takes a
    slot and `release` gives it back — so the cap holds across the
    executor's thread pool.

    A 429 is not a throughput problem but a STOP: `penalize` holds the
    whole scope until the header's reset, and no bucket arithmetic can
    talk its way past that hold.

    The runner's shared implementation (000338) replaces this across
    jobs and processes; the interface is the same four methods.
    """

    def __init__(self, *, sleep=None, clock=None, limits=None,
                 max_waits: int = 64, slot_timeout: float = 30.0) -> None:
        import time

        self._sleep = sleep if sleep is not None else time.sleep
        self._clock = clock if clock is not None else time.monotonic
        self._limits_for = limits if limits is not None else limits_for
        self._buckets: dict[tuple, Bucket] = {}
        self._holds: dict[tuple, float] = {}
        self._lock = threading.RLock()
        # Concurrency is a standing count: a slot frees when another
        # call RELEASES it, not when time passes, so waiting for one is
        # a condition wait rather than a sleep.
        self._slots = threading.Condition(self._lock)
        self._max_waits = max_waits
        self._slot_timeout = slot_timeout
        self.waited_seconds = 0.0

    # --- Limiter -------------------------------------------------------------

    def acquire(self, provider: str, model: str, scope: str, currency: str,
                amount: float) -> float:
        waited = 0.0
        for _ in range(self._max_waits):
            with self._lock:
                now = self._clock()
                hold = self._holds.get((provider, scope), 0.0)
                if now < hold:
                    delay = hold - now
                else:
                    bucket = self._bucket(provider, model, scope, currency, now)
                    if bucket is None:
                        return waited          # no known limit: go
                    bucket.refill(now)
                    want = min(float(amount), bucket.capacity)
                    if bucket.tokens >= want:
                        bucket.tokens -= want
                        return waited
                    if bucket.per_second <= 0:
                        # Standing count (concurrency): wait for a
                        # release rather than for time to pass.
                        start = self._clock()
                        self._slots.wait(timeout=self._slot_timeout)
                        waited += max(0.0, self._clock() - start)
                        self.waited_seconds = round(self.waited_seconds + waited, 6)
                        continue
                    delay = bucket.wait_for(want)
            self._sleep(delay)
            waited += delay
            self.waited_seconds = round(self.waited_seconds + delay, 6)
        raise RuntimeError(
            f"{provider}/{currency} never freed up after {self._max_waits} waits")

    def observe(self, provider: str, model: str, scope: str,
                limits: RateLimits) -> None:
        """What the account actually has left. `remaining` corrects the
        bucket downward (never upward — a stale header must not hand
        out tokens the account no longer has), and an exhausted
        currency with a reset becomes a hold until that reset."""
        with self._lock:
            now = self._clock()
            pairs = (("requests", limits.requests_remaining, limits.requests_reset),
                     ("input_tokens", limits.input_tokens_remaining, limits.tokens_reset),
                     ("output_tokens", limits.output_tokens_remaining, limits.tokens_reset))
            for currency, remaining, reset in pairs:
                if remaining is None:
                    continue
                bucket = self._bucket(provider, model, scope, currency, now)
                if bucket is not None:
                    bucket.refill(now)
                    bucket.tokens = min(bucket.tokens, float(remaining))
                if remaining <= 0 and reset:
                    self._hold(provider, scope, now + float(reset))
            if limits.retry_after:
                self._hold(provider, scope, now + float(limits.retry_after))

    def penalize(self, provider: str, model: str, scope: str,
                 retry_after: float) -> None:
        with self._lock:
            now = self._clock()
            # A 429 without a reset header still means stop; one second
            # is the smallest honest pause.
            self._hold(provider, scope, now + max(float(retry_after or 0.0), 1.0))

    def release(self, provider: str, model: str, scope: str, currency: str,
                amount: float) -> None:
        """Give back a concurrency slot, or the output tokens a call
        reserved and did not use."""
        with self._lock:
            key = (provider, self._model_key(provider, model), scope, currency)
            bucket = self._buckets.get(key)
            if bucket is not None:
                bucket.tokens = min(bucket.capacity, bucket.tokens + float(amount))
                self._slots.notify_all()

    # --- internals ------------------------------------------------------------

    def _hold(self, provider: str, scope: str, until: float) -> None:
        key = (provider, scope)
        self._holds[key] = max(self._holds.get(key, 0.0), until)

    @staticmethod
    def _model_key(provider: str, model: str) -> str:
        # Limits are per model family, and the registry keys them by
        # prefix; the bucket key uses the model string as given, which
        # is the finest grain a provider ever meters at.
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
        # Concurrency is a standing count, not a per-minute flow.
        per_second = 0.0 if currency == "concurrency" else float(limit) / 60.0
        bucket = Bucket(capacity=float(limit), per_second=per_second,
                        tokens=float(limit), updated=now)
        self._buckets[key] = bucket
        return bucket
