"""Provider transports (epic 000334, task 000337).

`make_transport(provider, credential)` is the one door. It knows which
adapter speaks which wire format, and it takes the two flags that make
a run free — `dry_run` (a mock that impersonates the provider) and a
`cassette` (recorded responses replayed by canonical request hash) —
so no caller has to remember to disable spending in tests.

The capability matrix is readable without a credential
(`capabilities(provider)`), because the protocol validator checks a
node's asks against it long before a key is delivered.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.providers import messages, pricing
from mechbench_compute.providers.base import (
    AdapterResponse,
    CallRecord,
    Capabilities,
    Completion,
    Transport,
    Usage,
)
from mechbench_compute.providers.budget import Budget, budget_from
from mechbench_compute.providers.cassette import Cassette, CassetteTransport
from mechbench_compute.providers.errors import (
    AuthError,
    BudgetExceeded,
    CapabilityUnsupported,
    CassetteMiss,
    ProviderError,
    ProviderUnavailable,
    RateLimited,
    TransientError,
)
from mechbench_compute.providers.limiter import Limiter, NullLimiter, RateLimits
from mechbench_compute.providers.messages import ChatRequest, Message, ToolSpec

#: Every provider a ModelRef may name. `openai-compatible` is the
#: generic escape hatch (Together, Groq, DeepSeek, Mistral, OpenRouter,
#: a local vLLM or llama.cpp server): its credential carries a
#: base_url alongside the token.
PROVIDERS: tuple[str, ...] = (
    "anthropic", "openai", "xai", "gemini", "fireworks",
    "openai-compatible", "mock",
)

_OPENAI_SHAPED = ("openai", "xai", "fireworks", "openai-compatible")

__all__ = [
    "PROVIDERS",
    "AdapterResponse",
    "AuthError",
    "Budget",
    "BudgetExceeded",
    "CallRecord",
    "Capabilities",
    "CapabilityUnsupported",
    "Cassette",
    "CassetteMiss",
    "CassetteTransport",
    "ChatRequest",
    "Completion",
    "Limiter",
    "Message",
    "NullLimiter",
    "ProviderError",
    "ProviderUnavailable",
    "RateLimited",
    "RateLimits",
    "ToolSpec",
    "TransientError",
    "Transport",
    "Usage",
    "budget_from",
    "capabilities",
    "make_transport",
    "messages",
    "pricing",
]


def capabilities(provider: str) -> Capabilities:
    """What this provider can do, without needing a credential."""
    if provider == "anthropic":
        from mechbench_compute.providers.anthropic import CAPABILITIES

        return CAPABILITIES
    if provider == "gemini":
        from mechbench_compute.providers.gemini import CAPABILITIES

        return CAPABILITIES
    if provider in _OPENAI_SHAPED:
        from mechbench_compute.providers.openai_compatible import HOSTS

        return HOSTS[provider][1]
    if provider == "mock":
        from mechbench_compute.providers.mock import MOCK_CAPABILITIES

        return MOCK_CAPABILITIES
    raise ValueError(
        f"unknown provider {provider!r} — known: {', '.join(PROVIDERS)}")


def make_transport(provider: str, credential: Mapping[str, Any] | str | None = None, *,
                   base_url: str | None = None, dry_run: bool = False,
                   cassette: Cassette | None = None, cassette_mode: str = "replay",
                   sleep=None, clock=None, **kw: Any) -> Transport:
    """The adapter for `provider`, wrapped as the run asks.

    `dry_run` returns a mock wearing this provider's name and
    capability matrix — the same requests, the same refusals, no
    network and no spend. A `cassette` replays recorded responses (and
    in `record`/`auto` mode wraps the real adapter to fill itself).
    """
    if provider not in PROVIDERS:
        raise ValueError(
            f"unknown provider {provider!r} — known: {', '.join(PROVIDERS)}")
    if dry_run and cassette is None:
        from mechbench_compute.providers.mock import MockTransport

        return MockTransport(name=provider, capabilities=capabilities(provider),
                             sleep=sleep, clock=clock, **kw)
    inner: Transport | None = None
    if not (cassette is not None and cassette_mode == "replay"):
        inner = _adapter(provider, credential, base_url=base_url, sleep=sleep,
                         clock=clock, **kw)
    if cassette is None:
        return inner
    token = credential.get("token") if isinstance(credential, Mapping) else credential
    return CassetteTransport(cassette, inner=inner, mode=cassette_mode,
                             secrets=(str(token),) if token else (),
                             capabilities=capabilities(provider))


def _adapter(provider: str, credential, *, base_url=None, sleep=None,
             clock=None, **kw) -> Transport:
    if provider == "anthropic":
        from mechbench_compute.providers.anthropic import AnthropicTransport

        return AnthropicTransport(credential, base_url=base_url, sleep=sleep,
                                  clock=clock, **kw)
    if provider == "gemini":
        from mechbench_compute.providers.gemini import GeminiTransport

        return GeminiTransport(credential, base_url=base_url, sleep=sleep,
                               clock=clock, **kw)
    if provider == "mock":
        from mechbench_compute.providers.mock import MockTransport

        return MockTransport(sleep=sleep, clock=clock, **kw)
    from mechbench_compute.providers.openai_compatible import OpenAICompatibleTransport

    return OpenAICompatibleTransport(credential, provider=provider,
                                     base_url=base_url, sleep=sleep,
                                     clock=clock, **kw)
