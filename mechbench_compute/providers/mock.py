"""The mock provider (task 000350 — built FIRST, so nothing in this
epic ever needs a real key to be tested).

Its responses are a deterministic function of the canonical request:
the same conversation always gets the same answer, on any machine, at
any time, with no network. That makes it usable for three different
jobs at once — unit tests, a protocol dry run, and the fixture half of
a cassette — without any of them spending.

Behaviour is injectable, because the interesting paths are the ugly
ones: a 429 with a `retry-after`, a run of 500s that becomes an
outage, a tool call, a token budget that runs out mid-corpus.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute.providers import messages as msg
from mechbench_compute.providers.base import (
    AdapterResponse,
    Capabilities,
    Transport,
    Usage,
)
from mechbench_compute.providers.errors import RateLimited, TransientError

_LEXICON = ["harbor", "lantern", "ledger", "quartz", "meridian", "salt", "thicket", "ember", "cadence", "furrow", "tide", "anvil", "marrow", "glass", "rill", "sable", "tundra", "vellum", "wick", "yarrow"]

MOCK_CAPABILITIES = Capabilities(
    chat=True, complete=True, count_tokens="exact", tools=True,
    json_mode=True, seed=True, logprobs=20, cache_control=True, batch=True,
    embed=True, streaming=True, models=True,
)


class MockTransport(Transport):
    """`script` injects behaviour per call, consumed in order: an
    exception instance is raised, a mapping overrides that call's
    response, None takes the default. When it runs out, the default
    resumes — so `script=[RateLimited(retry_after=2)]` tests one 429
    followed by success."""

    name = "mock"

    def __init__(self, *, name: str = "mock", capabilities: Capabilities | None = None,
                 script: Sequence[Any] = (), headers: Mapping[str, str] | None = None,
                 words: tuple[int, int] = (6, 24), sleep=None, clock=None) -> None:
        super().__init__(sleep=sleep, clock=clock)
        # A dry run IMPERSONATES the provider it stands in for: same
        # name, so requests hash and price as they will in the real run,
        # and same capability matrix, so an unsupported ask is still
        # refused by name.
        self.name = name
        self.capabilities = capabilities or MOCK_CAPABILITIES
        self._script = list(script)
        self._headers = dict(headers or {})
        self._words = words
        self.calls: list[msg.ChatRequest] = []

    # --- deterministic content ------------------------------------------------

    def _rng(self, req: msg.ChatRequest) -> random.Random:
        h = hashlib.sha256(msg.request_hash(req, provider=self.name).encode()).digest()
        return random.Random(int.from_bytes(h[:8], "big"))

    def _text_for(self, req: msg.ChatRequest) -> str:
        rng = self._rng(req)
        lo, hi = self._words
        n = rng.randint(lo, hi)
        return " ".join(rng.choice(_LEXICON) for _ in range(n))

    def _count_tokens(self, req: msg.ChatRequest) -> int:
        # The mock's own truth: words plus a fixed per-message overhead.
        # `capabilities.count_tokens = "exact"` is honest because this
        # IS the tokenizer of this provider.
        n = len(req.system.split())
        for m in req.messages:
            n += 3
            for p in m.content:
                if isinstance(p, msg.TextPart):
                    n += len(p.text.split())
                elif isinstance(p, msg.ToolCallPart):
                    n += 8 + len(str(dict(p.arguments)).split())
                else:
                    n += 4 + len(p.content.split())
        for t in req.tools:
            n += 12 + len(str(dict(t.input_schema)).split())
        return max(1, n)

    # --- the adapter ------------------------------------------------------------

    def _chat(self, req: msg.ChatRequest, *, on_token=None) -> AdapterResponse:
        self.calls.append(req)
        override: Mapping[str, Any] = {}
        if self._script:
            step = self._script.pop(0)
            if isinstance(step, BaseException):
                raise step
            if isinstance(step, Mapping):
                override = step
        opts = {**req.options_for("mock"), **req.options_for(self.name)}
        rng = self._rng(req)

        parts: list[msg.Part] = []
        stop = "end_turn"
        want_tool = opts.get("tool_call") or override.get("tool_call")
        if want_tool and req.tools:
            spec = next((t for t in req.tools if t.name == want_tool), req.tools[0])
            args = {k: f"{k}-{rng.randrange(1000)}"
                    for k in list(spec.input_schema.get("properties") or {})[:3]}
            parts.append(msg.ToolCallPart(id=f"call_{rng.randrange(1 << 30):08x}",
                                          name=spec.name, arguments=args))
            stop = "tool_use"
        text = str(override.get("text", opts.get("text", self._text_for(req))))
        if text:
            parts.insert(0, msg.TextPart(text))
            if on_token is not None:
                for tok in text.split(" "):
                    on_token(tok + " ")

        out_tokens = len(text.split()) + (10 if stop == "tool_use" else 0)
        if req.max_tokens and out_tokens > req.max_tokens:
            out_tokens = int(req.max_tokens)
            stop = "max_tokens"
        cache_read = int(opts.get("cache_read_tokens", 0))
        usage = Usage(input_tokens=self._count_tokens(req),
                      output_tokens=out_tokens,
                      cache_read_tokens=cache_read)
        headers = {**self._headers, **dict(override.get("headers") or {})}
        return AdapterResponse(
            parts=tuple(parts), stop_reason=str(override.get("stop_reason", stop)),
            usage=usage,
            model_version=f"{req.model}-mock-20260908",
            response_id=f"mock_{msg.request_hash(req, provider=self.name)[:16]}",
            headers=headers,
            logprobs=self._logprobs(req, rng) if req.logprobs else None,
        )

    @staticmethod
    def _logprobs(req: msg.ChatRequest, rng: random.Random) -> list[dict[str, Any]]:
        k = int(req.logprobs or 5)
        raw = sorted((rng.random() for _ in range(k)), reverse=True)
        total = sum(raw)
        import math

        return [{"token": _LEXICON[i % len(_LEXICON)],
                 "logprob": round(math.log(p / total), 4)}
                for i, p in enumerate(raw)]


def rate_limited(retry_after: float = 1.0) -> RateLimited:
    return RateLimited("mock: rate limited", retry_after=retry_after)


def transient(message: str = "mock: 503 upstream") -> TransientError:
    return TransientError(message, status=503)
