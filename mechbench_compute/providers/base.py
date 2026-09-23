"""The Transport interface.

An adapter's whole job is the wire mapping: canonical request in,
`AdapterResponse` out. Everything a caller can get wrong is done ONCE,
here, for every provider:

  count tokens → price the worst case → reserve against the budget
  → ask the limiter → call (retrying 429/5xx with the header's own
  backoff) → settle the real cost → report what the headers said.

That order is the point. The budget refuses before the request leaves
the machine; the limiter's wait is a number the provider gave us; a
sustained failure raises `ProviderUnavailable`, which the runner turns
into an interrupt so a job keeps its partials instead of dying at 90%.

Every call returns a `CallRecord`: what was asked (hash + the
passthrough options verbatim), who answered (the provider's own dated
model string), what it used, what it cost, and how long it took —
including whether the token count was exact or an estimate, because a
budget refusal computed from a guess should read as a guess.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from mechbench_compute.providers import messages as msg
from mechbench_compute.providers import pricing
from mechbench_compute.providers.budget import Budget
from mechbench_compute.providers.errors import (
    CapabilityUnsupported,
    ProviderError,
    ProviderUnavailable,
    RateLimited,
)
from mechbench_compute.providers.limiter import Limiter, NullLimiter, RateLimits


@dataclass(frozen=True)
class Capabilities:
    """What a provider can actually do. The protocol validator checks a
    node's needs against this before the job starts, so "Anthropic has
    no logprobs" is a protocol error at seal, not a surprise at item
    400."""

    chat: bool = True
    complete: bool = False
    count_tokens: str | None = None      # "exact" | "estimated" | None
    tools: bool = False
    json_mode: bool = False
    seed: bool = False
    logprobs: int | None = None          # max top-k, None = unsupported
    cache_control: bool = False
    batch: bool = False
    embed: bool = False
    streaming: bool = False
    models: bool = False
    #: Speaks the Responses API as well as its usual one (OpenAI, xAI).
    responses: bool = False

    def to_wire(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0

    def to_wire(self) -> dict[str, int]:
        return {k: int(v) for k, v in asdict(self).items() if v}


#: Why a reply carried neither prose nor a tool call.
#:   reasoning   the output allowance went to reasoning; no prose followed
#:   filtered    the provider withheld or refused the content
#:   unmapped    content arrived in a block type the adapter cannot read
#:   no_content  none of the above
EMPTY_CAUSES = ("reasoning", "filtered", "unmapped", "no_content")


@dataclass(frozen=True)
class EmptyReply:
    """A reply with no prose and no tool call, and why. The adapter
    reports it rather than raising, so the call it paid for is settled
    and the op that asked decides what an empty item means."""

    cause: str
    message: str

    def __post_init__(self) -> None:
        if self.cause not in EMPTY_CAUSES:
            raise ValueError(f"an empty reply's cause is one of {EMPTY_CAUSES}, "
                             f"not {self.cause!r}")

    def to_wire(self) -> dict[str, str]:
        return {"cause": self.cause, "message": self.message}

    @staticmethod
    def from_wire(value: Mapping[str, Any] | None) -> EmptyReply | None:
        if not value:
            return None
        return EmptyReply(cause=str(value.get("cause", "no_content")),
                          message=str(value.get("message", "")))


@dataclass(frozen=True)
class AdapterResponse:
    """What an adapter returns: the canonical parts plus everything the
    shared layer needs to meter and record the call."""

    parts: tuple[msg.Part, ...]
    stop_reason: str = "end_turn"
    usage: Usage = field(default_factory=Usage)
    model_version: str = ""
    response_id: str = ""
    headers: Mapping[str, str] = field(default_factory=dict)
    logprobs: Any = None
    raw: Any = None
    #: True when this came off a cassette rather than a wire — the run
    #: must be able to say which of its answers it actually bought.
    replayed: bool = False
    #: Set when the reply holds neither prose nor a tool call.
    empty: EmptyReply | None = None


@dataclass
class CallRecord:
    """One call's provenance. Items carry theirs; the manifest sums
    them, and a partial result carries the running total."""

    provider: str
    model: str
    model_version: str = ""
    request_hash: str = ""
    response_id: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    cost_usd: float = 0.0
    priced: bool = True
    price_table: str = pricing.TABLE_VERSION
    tokens_exact: bool = False
    stop_reason: str = ""
    latency_ms: int = 0
    attempts: int = 1
    throttled_seconds: float = 0.0
    replayed: bool = False
    provider_options: dict[str, Any] = field(default_factory=dict)
    rate_limits: dict[str, Any] = field(default_factory=dict)
    request: dict[str, Any] | None = None
    #: `{cause, message}` when the reply was empty; absent otherwise.
    empty: dict[str, str] | None = None

    def to_wire(self) -> dict[str, Any]:
        out = {k: v for k, v in asdict(self).items()
               if v not in (None, {}, 0.0, 0) or k in ("cost_usd", "usage")}
        out["cost_usd"] = round(self.cost_usd, 8)
        return out


@dataclass
class Completion:
    text: str
    parts: tuple[msg.Part, ...]
    stop_reason: str
    usage: Usage
    call: CallRecord
    logprobs: Any = None
    empty: EmptyReply | None = None

    @property
    def tool_calls(self) -> tuple[msg.ToolCallPart, ...]:
        return tuple(p for p in self.parts if isinstance(p, msg.ToolCallPart))

    @property
    def reasoning(self) -> list[dict[str, Any]]:
        """The reply's reasoning as an item's `reasoning` entries."""
        return msg.read_reasoning(self.parts)

    def as_message(self) -> msg.Message:
        """The completion as the assistant turn it is — what a
        conversation appends before the next participant speaks. Its
        reasoning parts stay parts, so the adapter sends them back in
        the provider's own form."""
        return msg.Message(role="assistant", content=self.parts)

    def to_wire(self) -> dict[str, Any]:
        out = {"kind": "provider/completion", "text": self.text,
               "parts": [p.to_wire() for p in self.parts],
               **({"reasoning": self.reasoning} if self.reasoning else {}),
               "stop_reason": self.stop_reason,
               "usage": self.usage.to_wire(),
               "call": self.call.to_wire()}
        if self.empty is not None:
            out["empty"] = self.empty.to_wire()
        return out


class Transport(ABC):
    """Base class for provider adapters.

    Subclasses implement `_chat` (and optionally `_count_tokens`,
    `_models`); `chat` is final in spirit — the metering, limiting,
    retrying and recording all live here so no adapter can forget them.
    """

    name: str = "provider"
    capabilities: Capabilities = Capabilities()

    #: Retry/outage policy. `outage_seconds` is the window after which
    #: sustained failure stops being a retry and becomes an interrupt.
    max_attempts: int = 5
    backoff_base: float = 0.5
    backoff_cap: float = 30.0
    outage_seconds: float = 600.0

    def __init__(self, *, sleep=None, clock=None) -> None:
        # Injected so tests exercise backoff and outage windows without
        # spending wall-clock time.
        self._sleep = sleep if sleep is not None else time.sleep
        self._clock = clock if clock is not None else time.monotonic

    # --- what adapters implement ------------------------------------------

    @abstractmethod
    def _chat(self, req: msg.ChatRequest, *, on_token=None) -> AdapterResponse:
        ...

    def _count_tokens(self, req: msg.ChatRequest) -> int:
        raise NotImplementedError

    # --- what everyone gets -----------------------------------------------

    def count_tokens(self, req: msg.ChatRequest) -> tuple[int, bool]:
        """(tokens, exact). Falls back to the characters/4 estimate
        when the provider offers no counter — and says so."""
        if self.capabilities.count_tokens in ("exact", "estimated"):
            try:
                return int(self._count_tokens(req)), self.capabilities.count_tokens == "exact"
            except NotImplementedError:
                pass
            except ProviderError:
                # A counter that is down must not stop the call: the
                # estimate is conservative enough for a cap.
                pass
        return msg.estimate_tokens(req), False

    def check_supported(self, req: msg.ChatRequest) -> None:
        """Refuse a request that asks for something this provider does
        not have, by name."""
        caps = self.capabilities
        if req.api == "responses":
            if not caps.responses:
                raise CapabilityUnsupported(
                    self.name, 'api "responses"', "the Responses API is OpenAI's and xAI's; "
                    "drop `api` to use this provider's own")
            # Fields the Responses API has no place for. Refused rather
            # than dropped, as everywhere: a request that silently lost
            # its stop strings is a different question.
            for name, present in (("stop", bool(req.stop)),
                                  ("seed", req.seed is not None),
                                  ("logprobs", req.logprobs is not None)):
                if present:
                    raise CapabilityUnsupported(
                        self.name, f"{name} on the Responses API", "it has no such field; "
                        'drop it, or ask for api: "chat_completions"')
        if req.tools and not caps.tools:
            raise CapabilityUnsupported(self.name, "tools")
        if req.tool_choice is not None and not req.tools:
            # A choice among nothing. Every adapter puts
            # `tool_choice` on the wire, so this would be sent and either
            # ignored or refused by the provider in its own words.
            raise CapabilityUnsupported(
                self.name, "tool_choice",
                "no tools were declared, so there is nothing to choose among")
        if req.json_mode and not caps.json_mode:
            raise CapabilityUnsupported(self.name, "json_mode")
        if req.seed is not None and not caps.seed:
            raise CapabilityUnsupported(
                self.name, "seed", "sampling is not reproducible here; "
                "drop seed or accept `exchangeable` resume")
        if req.logprobs is not None:
            if caps.logprobs is None:
                raise CapabilityUnsupported(self.name, "logprobs")
            if req.logprobs > caps.logprobs:
                raise CapabilityUnsupported(
                    self.name, "logprobs",
                    f"top-{req.logprobs} requested, {caps.logprobs} is the limit")

    def chat(self, req: msg.ChatRequest, *, budget: Budget | None = None,
             limiter: Limiter | None = None, scope: str = "default",
             on_token=None, record_request: bool = False) -> Completion:
        self.check_supported(req)
        limiter = limiter or NullLimiter()
        started = self._clock()
        rhash = msg.request_hash(req, provider=self.name)

        in_tokens, exact = self.count_tokens(req)
        estimate, priced = pricing.worst_case_usd(
            self.name, req.model, input_tokens=in_tokens,
            max_tokens=int(req.max_tokens))
        reservation = 0.0
        if budget is not None:
            reservation = budget.reserve(estimate, provider=self.name,
                                         model=req.model)
        throttled = 0.0
        held_slot = False
        max_out = int(req.max_tokens)
        try:
            # Every currency the provider meters separately: a slot to
            # be in flight at all, then the request, then the tokens
            # this call could spend on either side of it.
            throttled += limiter.acquire(self.name, req.model, scope,
                                         "concurrency", 1)
            held_slot = True
            throttled += limiter.acquire(self.name, req.model, scope,
                                         "requests", 1)
            throttled += limiter.acquire(self.name, req.model, scope,
                                         "input_tokens", in_tokens)
            throttled += limiter.acquire(self.name, req.model, scope,
                                         "output_tokens", max_out)
            resp, attempts, waited = self._call_with_retries(
                req, limiter=limiter, scope=scope, on_token=on_token)
            throttled += waited
        except BaseException:
            if budget is not None:
                budget.release(reservation)
            if held_slot:
                limiter.release(self.name, req.model, scope, "concurrency", 1)
            raise

        limiter.release(self.name, req.model, scope, "concurrency", 1)
        usage = resp.usage.to_wire()
        # Give back the output tokens this call reserved and did not
        # write: a 2000-token cap that produced 40 tokens must not
        # throttle the next item as if it had spent 2000.
        unused = max_out - int(usage.get("output_tokens", 0) or 0)
        if unused > 0:
            limiter.release(self.name, req.model, scope, "output_tokens", unused)
        limits = RateLimits.from_headers(resp.headers or {})
        limiter.observe(self.name, req.model, scope, limits)
        cost, cost_priced = pricing.cost_usd(self.name, req.model, usage)
        if resp.replayed:
            # A replayed call bought nothing. Its usage is
            # kept — it is what the ORIGINAL call spent, and a reader
            # comparing a memoized run to its first run needs it — but
            # the cost is zero and the reservation is released rather
            # than settled, or a cached re-run would bill twice for one
            # purchase.
            cost, cost_priced = 0.0, True
            if budget is not None:
                budget.release(reservation)
        elif budget is not None:
            budget.settle(reservation, cost)

        record = CallRecord(
            provider=self.name, model=req.model,
            model_version=resp.model_version or req.model,
            request_hash=rhash, response_id=resp.response_id,
            stop_reason=resp.stop_reason,
            usage=usage, cost_usd=cost, priced=cost_priced and priced,
            tokens_exact=exact,
            latency_ms=int((self._clock() - started) * 1000),
            attempts=attempts, throttled_seconds=round(throttled, 3),
            replayed=bool(resp.replayed),
            provider_options=req.options_for(self.name),
            rate_limits=limits.to_wire(),
            request=msg.canonical(req, provider=self.name) if record_request else None,
            empty=resp.empty.to_wire() if resp.empty is not None else None,
        )
        parts = tuple(resp.parts)
        return Completion(
            text=msg.join_text(parts),
            parts=parts, stop_reason=resp.stop_reason, usage=resp.usage,
            call=record, logprobs=resp.logprobs, empty=resp.empty)

    # --- retries and outages ------------------------------------------------

    def _call_with_retries(self, req, *, limiter, scope, on_token):
        """Retry 429/5xx with the provider's own backoff. A failure
        window longer than `outage_seconds` is not a retry problem —
        it is an outage, and the job should keep its partials and come
        back."""
        first_failure: float | None = None
        waited = 0.0
        last: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self._chat(req, on_token=on_token), attempt, waited
            except RateLimited as e:
                last = e
                limiter.penalize(self.name, req.model, scope, e.retry_after or 0.0)
                delay = e.retry_after if e.retry_after is not None else self._backoff(attempt)
            except ProviderUnavailable:
                raise
            except ProviderError as e:
                if not getattr(e, "retryable", False):
                    raise
                last = e
                delay = self._backoff(attempt)
            first_failure = first_failure if first_failure is not None else self._clock()
            if self._clock() + delay - first_failure > self.outage_seconds:
                raise ProviderUnavailable(
                    f"{self.name} has been failing for over "
                    f"{int(self.outage_seconds)}s ({last}); interrupting so "
                    "the job resumes with its partials",
                    provider=self.name, seconds=self.outage_seconds) from last
            if attempt == self.max_attempts:
                break
            self._sleep(delay)
            waited += delay
        raise ProviderUnavailable(
            f"{self.name} failed {self.max_attempts} times ({last})",
            provider=self.name) from last

    def _backoff(self, attempt: int) -> float:
        """Deterministic exponential backoff. No jitter: a run's
        timing should be as reproducible as its result, and the
        limiter — not randomness — is what spreads concurrent callers."""
        return min(self.backoff_cap, self.backoff_base * (2 ** (attempt - 1)))
