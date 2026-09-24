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
from mechbench_compute.providers.budget import Budget, build_budget
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

PROVIDERS: tuple[str, ...] = (
    "anthropic", "openai", "xai", "gemini", "fireworks", "deepseek",
    "openai-compatible", "mock",
)

_OPENAI_SHAPED = ("openai", "xai", "fireworks", "deepseek", "openai-compatible")

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
    "build_budget",
    "capabilities",
    "make_transport",
    "messages",
    "pricing",
]


def capabilities(provider: str) -> Capabilities:
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


def remap_response(provider: str, raw: Any, req: Any, *,
                   headers: Mapping[str, str] | None = None):
    if not isinstance(raw, Mapping):
        return None
    if provider == "anthropic":
        from mechbench_compute.providers.anthropic import read_response

        return read_response(raw, req, headers=headers)
    if provider == "gemini":
        from mechbench_compute.providers.gemini import read_response

        return read_response(raw, req, headers=headers)
    if provider in _OPENAI_SHAPED:
        if getattr(req, "api", None) == "responses":
            from mechbench_compute.providers.openai_responses import read_response
        else:
            from mechbench_compute.providers.openai_compatible import read_response

        return read_response(raw, req, provider=provider, headers=headers)
    return None


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
