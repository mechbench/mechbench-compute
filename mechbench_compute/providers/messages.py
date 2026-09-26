from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

ROLES = ("user", "assistant")

APIS = (None, "chat_completions", "responses")


@dataclass(frozen=True)
class Signature:
    provider: str
    model: str
    value: str

    def to_wire(self) -> dict[str, str]:
        return {"provider": self.provider, "model": self.model, "value": self.value}

    @staticmethod
    def from_wire(value: Any) -> Signature | None:
        if not isinstance(value, Mapping) or not value.get("value"):
            return None
        return Signature(provider=str(value.get("provider", "")),
                         model=str(value.get("model", "")),
                         value=str(value["value"]))


@dataclass(frozen=True)
class TextPart:
    text: str
    signature: Signature | None = None

    def to_wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": "text", "text": self.text}
        if self.signature is not None:
            out["signature"] = self.signature.to_wire()
        return out


@dataclass(frozen=True)
class ReasoningPart:
    text: str = ""
    redacted: bool = False
    provider: str = ""
    model: str = ""
    native: Mapping[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        out: dict[str, Any] = {"text": self.text}
        if self.redacted:
            out["redacted"] = True
        if self.provider:
            out["provider"] = self.provider
        if self.model:
            out["model"] = self.model
        if self.native:
            out["native"] = dict(self.native)
        return out

    def to_wire(self) -> dict[str, Any]:
        return {"type": "reasoning", **self.to_record()}


@dataclass(frozen=True)
class ToolCallPart:
    id: str
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    signature: Signature | None = None

    def to_wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": "tool_call", "id": self.id, "name": self.name,
                               "arguments": dict(self.arguments)}
        if self.signature is not None:
            out["signature"] = self.signature.to_wire()
        return out


@dataclass(frozen=True)
class ToolResultPart:
    tool_call_id: str
    content: str
    is_error: bool = False

    def to_wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": "tool_result",
                               "tool_call_id": self.tool_call_id,
                               "content": self.content}
        if self.is_error:
            out["is_error"] = True
        return out


Part = TextPart | ReasoningPart | ToolCallPart | ToolResultPart


@dataclass(frozen=True)
class Message:
    role: str
    content: tuple[Part, ...]

    def text(self) -> str:
        return join_text(self.content)

    def to_wire(self) -> dict[str, Any]:
        return {"role": self.role, "content": [p.to_wire() for p in self.content]}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str = ""
    input_schema: Mapping[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}})

    def to_wire(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "input_schema": dict(self.input_schema)}


def join_text(parts: Iterable[Any]) -> str:
    return "".join(p.text for p in parts if isinstance(p, TextPart))


def read_reasoning(parts: Iterable[Any]) -> list[dict[str, Any]]:
    return [p.to_record() for p in parts if isinstance(p, ReasoningPart)]


def is_replayable(origin_provider: str, origin_model: str, *, provider: str,
                  model: str, model_bound: bool = True) -> bool:
    if not origin_provider or origin_provider != provider:
        return False
    return not model_bound or origin_model == model


def part(value: Any) -> Part:
    if isinstance(value, (TextPart, ReasoningPart, ToolCallPart, ToolResultPart)):
        return value
    if isinstance(value, str):
        return TextPart(value)
    if not isinstance(value, Mapping):
        raise TypeError(f"a content part is a string or an object, not {type(value).__name__}")
    kind = value.get("type", "text")
    if kind == "text":
        return TextPart(str(value.get("text", "")),
                        signature=Signature.from_wire(value.get("signature")))
    if kind == "reasoning":
        return ReasoningPart(text=str(value.get("text", "")),
                             redacted=bool(value.get("redacted", False)),
                             provider=str(value.get("provider", "")),
                             model=str(value.get("model", "")),
                             native=dict(value.get("native") or {}))
    if kind == "tool_call":
        return ToolCallPart(id=str(value.get("id", "")),
                            name=str(value["name"]),
                            arguments=dict(value.get("arguments") or {}),
                            signature=Signature.from_wire(value.get("signature")))
    if kind == "tool_result":
        return ToolResultPart(tool_call_id=str(value.get("tool_call_id", "")),
                              content=_as_text(value.get("content", "")),
                              is_error=bool(value.get("is_error", False)))
    raise ValueError(
        f"unknown content part type {kind!r} — the canonical model carries "
        "text, reasoning, tool_call and tool_result")


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return str(value.get("text", value))
    if isinstance(value, Sequence):
        return "".join(_as_text(v) for v in value)
    return str(value)


def message(value: Any, *, default_role: str = "user") -> Message:
    if isinstance(value, Message):
        return value
    if isinstance(value, str):
        return Message(role=default_role, content=(TextPart(value),))
    if not isinstance(value, Mapping):
        raise TypeError(f"a message is a string or an object, not {type(value).__name__}")
    role = str(value.get("role", default_role))
    if role == "system":
        raise ValueError(
            "the system prompt is a field of the request, not a message: "
            "pass system=... (providers model it that way, and putting it "
            "in the turn list makes the same conversation hash differently "
            "per adapter)")
    if role not in ROLES:
        raise ValueError(f"unknown message role {role!r} — expected one of {ROLES}")
    raw = value.get("content", "")
    parts = [part(p) for p in (raw if isinstance(raw, (list, tuple)) else [raw])]
    return Message(role=role, content=tuple(parts))


def messages(value: Any) -> tuple[Message, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (message(value),)
    if isinstance(value, Mapping):
        inner = value.get("messages")
        if inner is None:
            return (message(value),)
        value = inner
    if not isinstance(value, Iterable):
        raise TypeError("messages must be a string, a list, or a transcript record")
    return tuple(message(m) for m in value)


@dataclass(frozen=True)
class ChatRequest:
    model: str
    messages: tuple[Message, ...] = ()
    system: str = ""
    tools: tuple[ToolSpec, ...] = ()
    tool_choice: str | Mapping[str, Any] | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int = 1024
    stop: tuple[str, ...] = ()
    seed: int | None = None
    json_mode: bool = False
    logprobs: int | None = None
    provider_options: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    api: str | None = None
    effort: str | None = None
    reasoning_display: str | None = None
    prompt_cache: str | None = None

    def options_for(self, provider: str) -> dict[str, Any]:
        opts = self.provider_options.get(provider) or {}
        if not isinstance(opts, Mapping):
            raise TypeError(
                f"provider_options[{provider!r}] must be an object of "
                "provider-native fields")
        return dict(opts)

    def check_options(self) -> None:
        from mechbench_compute.providers import registry as _registry

        known = set(_registry.registry())
        stray = sorted(k for k in self.provider_options if k not in known)
        if stray:
            raise ValueError(
                f"provider_options key(s) {', '.join(stray)} name no provider. "
                f"The bag is keyed by provider — "
                f"`{{\"{sorted(known)[0]}\": {{…}}}}` — so a field written at "
                f"the top level is passed to nobody. Known providers: "
                f"{', '.join(sorted(known))}.")

    def with_messages(self, ms: Sequence[Message]) -> ChatRequest:
        from dataclasses import replace

        return replace(self, messages=tuple(ms))


def request(value: Any = None, **overrides: Any) -> ChatRequest:
    raw: dict[str, Any] = dict(value or {})
    raw.update(overrides)
    known = {f for f in ChatRequest.__dataclass_fields__}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"unknown chat request field(s): {', '.join(sorted(unknown))}. "
            f"Known: {', '.join(sorted(known))}. Provider-native fields go "
            "under provider_options.<provider>.")
    ms = messages(raw.pop("messages", ()))
    tools = tuple(
        t if isinstance(t, ToolSpec) else ToolSpec(
            name=str(t["name"]), description=str(t.get("description", "")),
            input_schema=dict(t.get("input_schema") or t.get("parameters") or
                              {"type": "object", "properties": {}}))
        for t in (raw.pop("tools", ()) or ()))
    stop = tuple(raw.pop("stop", ()) or ())
    api = raw.pop("api", None)
    if api not in APIS:
        raise ValueError(
            f"api is one of {', '.join(a for a in APIS if a)}, not {api!r}")
    raw["api"] = None if api == "chat_completions" else api
    req = ChatRequest(messages=ms, tools=tools, stop=stop, **raw)
    req.check_options()
    return req


def canonical(req: ChatRequest, *, provider: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "model": req.model,
        "messages": [m.to_wire() for m in req.messages],
        "max_tokens": int(req.max_tokens),
    }
    if req.system:
        out["system"] = req.system
    if req.tools:
        out["tools"] = [t.to_wire() for t in req.tools]
    if req.tool_choice is not None:
        out["tool_choice"] = (dict(req.tool_choice)
                              if isinstance(req.tool_choice, Mapping)
                              else req.tool_choice)
    for name in ("temperature", "top_p", "seed", "logprobs"):
        v = getattr(req, name)
        if v is not None:
            out[name] = v
    if req.stop:
        out["stop"] = list(req.stop)
    if req.json_mode:
        out["json_mode"] = True
    for name in ("effort", "reasoning_display"):
        v = getattr(req, name)
        if v is not None:
            out[name] = v
    if req.api:
        out["api"] = req.api
    opts = ({provider: req.options_for(provider)} if provider
            else {k: dict(v) for k, v in req.provider_options.items()})
    opts = {k: v for k, v in opts.items() if v}
    if opts:
        out["provider_options"] = opts
    return out


def request_hash(req: ChatRequest, *, provider: str | None = None) -> str:
    from mechbench_schema import dump_canonical

    return hashlib.sha256(dump_canonical(canonical(req, provider=provider))).hexdigest()


def estimate_tokens(req: ChatRequest) -> int:
    n = len(req.system)
    for m in req.messages:
        for p in m.content:
            if isinstance(p, TextPart):
                n += len(p.text)
            elif isinstance(p, ReasoningPart):
                n += len(p.text) + len(str(dict(p.native)))
            elif isinstance(p, ToolCallPart):
                n += len(p.name) + len(str(dict(p.arguments)))
            else:
                n += len(p.content)
    for t in req.tools:
        n += len(t.name) + len(t.description) + len(str(dict(t.input_schema)))
    return max(1, n // 4)
