from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from mechbench_compute.providers import messages as msg
from mechbench_compute.providers.base import (
    AdapterResponse,
    Capabilities,
    EmptyReply,
    Transport,
    Usage,
)
from mechbench_compute.providers.errors import CassetteMiss

CASSETTE_KIND = "~canonical/kinds/provider/cassette"

_HEADER_DENY = ("authorization", "x-api-key", "api-key", "cookie",
                "set-cookie", "openai-organization", "x-goog-api-key")

_HEADER_ALLOW_PREFIX = ("anthropic-ratelimit-", "x-ratelimit-", "retry-after",
                        "x-request-id", "request-id")


def scrub_headers(headers: Mapping[str, str],
                  secrets: Iterable[str] = ()) -> dict[str, str]:
    secret_values = {s for s in secrets if s}
    out: dict[str, str] = {}
    for k, v in (headers or {}).items():
        lk = str(k).lower()
        if lk in _HEADER_DENY or not lk.startswith(_HEADER_ALLOW_PREFIX):
            continue
        sv = str(v)
        if any(s in sv for s in secret_values):
            continue
        out[lk] = sv
    return out


def response_to_wire(resp: AdapterResponse) -> dict[str, Any]:
    return {
        "parts": [p.to_wire() for p in resp.parts],
        "stop_reason": resp.stop_reason,
        "usage": resp.usage.to_wire(),
        "model_version": resp.model_version,
        "response_id": resp.response_id,
        "headers": dict(resp.headers or {}),
        **({"logprobs": resp.logprobs} if resp.logprobs is not None else {}),
        **({"empty": resp.empty.to_wire()} if resp.empty is not None else {}),
        **({"raw": resp.raw} if resp.raw is not None else {}),
    }


def response_from_wire(value: Mapping[str, Any]) -> AdapterResponse:
    return AdapterResponse(
        parts=tuple(msg.part(p) for p in value.get("parts", [])),
        stop_reason=str(value.get("stop_reason", "end_turn")),
        usage=Usage(**{k: int(v) for k, v in (value.get("usage") or {}).items()}),
        model_version=str(value.get("model_version", "")),
        response_id=str(value.get("response_id", "")),
        headers=dict(value.get("headers") or {}),
        logprobs=value.get("logprobs"),
        empty=EmptyReply.from_wire(value.get("empty")),
        raw=value.get("raw"),
    )


@dataclass
class Cassette:
    provider: str
    label: str = ""
    entries: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    _cursor: dict[str, int] = field(default_factory=dict, repr=False)

    def add(self, request_hash: str, resp: AdapterResponse) -> None:
        self.entries.setdefault(request_hash, []).append(response_to_wire(resp))

    def take(self, request_hash: str) -> AdapterResponse | None:
        seq = self.entries.get(request_hash)
        if not seq:
            return None
        i = self._cursor.get(request_hash, 0)
        self._cursor[request_hash] = i + 1
        return response_from_wire(seq[min(i, len(seq) - 1)])

    def rewind(self) -> None:
        self._cursor.clear()

    @property
    def n_responses(self) -> int:
        return sum(len(v) for v in self.entries.values())

    def to_wire(self) -> dict[str, Any]:
        return {"kind": "provider/cassette", "cassette_kind": CASSETTE_KIND,
                "version": 1, "provider": self.provider, "label": self.label,
                "n_requests": len(self.entries), "n_responses": self.n_responses,
                "entries": [{"request_hash": k, "responses": v}
                            for k, v in sorted(self.entries.items())]}

    @staticmethod
    def from_wire(value: Mapping[str, Any]) -> Cassette:
        if value.get("kind") not in ("provider/cassette", "provider_cassette"):
            raise ValueError(
                f"not a cassette object: kind={value.get('kind')!r}")
        entries = {str(e["request_hash"]): list(e["responses"])
                   for e in value.get("entries", [])}
        return Cassette(provider=str(value.get("provider", "")),
                        label=str(value.get("label", "")), entries=entries)


def _replay_capabilities(provider: str) -> Capabilities:
    from mechbench_compute.providers import capabilities

    try:
        return capabilities(provider)
    except ValueError:
        return Capabilities(chat=True, tools=True, count_tokens="estimated")


class CassetteTransport(Transport):
    def __init__(self, cassette: Cassette, *, inner: Transport | None = None,
                 mode: str = "replay", secrets: Sequence[str] = (),
                 capabilities: Capabilities | None = None) -> None:
        super().__init__()
        if mode not in ("replay", "record", "auto"):
            raise ValueError(f"cassette mode must be replay|record|auto, not {mode!r}")
        if mode in ("record", "auto") and inner is None:
            raise ValueError(f"cassette mode {mode!r} needs an inner adapter to record from")
        self.cassette = cassette
        self.inner = inner
        self.mode = mode
        self._secrets = tuple(secrets)
        self.name = inner.name if inner is not None else (cassette.provider or "cassette")
        self.capabilities = (capabilities or
                             (inner.capabilities if inner is not None
                              else _replay_capabilities(cassette.provider)))
        self.misses: list[str] = []

    def _count_tokens(self, req: msg.ChatRequest) -> int:
        if self.inner is not None:
            return self.inner._count_tokens(req)
        raise NotImplementedError

    def _chat(self, req: msg.ChatRequest, *, on_token=None) -> AdapterResponse:
        key = msg.request_hash(req, provider=self.name)
        if self.mode in ("replay", "auto"):
            hit = self.cassette.take(key)
            if hit is not None and hit.raw is not None:
                from mechbench_compute.providers import remap_response

                mapped = remap_response(self.name, hit.raw, req, headers=hit.headers)
                if mapped is not None:
                    hit = mapped
            if hit is not None:
                if on_token is not None:
                    for p in hit.parts:
                        if isinstance(p, msg.TextPart):
                            on_token(p.text)
                return replace(hit, replayed=True)
            if self.mode == "replay":
                self.misses.append(key)
                raise CassetteMiss(self.name, key, self.cassette.label)
        resp = self.inner._chat(req, on_token=on_token)
        stored = AdapterResponse(
            parts=resp.parts, stop_reason=resp.stop_reason, usage=resp.usage,
            model_version=resp.model_version, response_id=resp.response_id,
            headers=scrub_headers(resp.headers, self._secrets),
            logprobs=resp.logprobs, raw=resp.raw, empty=resp.empty)
        self.cassette.add(key, stored)
        return resp
