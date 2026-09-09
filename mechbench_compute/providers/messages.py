"""The canonical message model (task 000337, epic 000334).

One shape of conversation crosses every provider and the local MLX
path: a system string, an alternating sequence of user/assistant
messages, and content parts that are `text`, `tool_call` or
`tool_result`. Tools are JSON Schema. Text and tools only — multimodal
parts arrive with 000343.

Two rules keep this honest:

**The canonical request is the identity of a call.** `canonical(req)`
produces the dict that gets hashed, recorded in provenance, keyed in a
cassette, and compared on resume. It contains what was ASKED, never
how it was authenticated: no keys, no headers, no base URLs.

**`provider_options` passes through verbatim.** Anything a provider
offers that the canonical fields do not name — cache_control,
reasoning effort, safety settings, service tiers — rides under
`provider_options[<provider>]` and is merged into the wire body after
the canonical fields, so the caller can always reach the real API
without waiting for this module to grow a field. What is passed
through is recorded, so a run says exactly what it asked for.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

ROLES = ("user", "assistant")


@dataclass(frozen=True)
class TextPart:
    text: str

    def to_wire(self) -> dict[str, Any]:
        return {"type": "text", "text": self.text}


@dataclass(frozen=True)
class ToolCallPart:
    """A model's request to run a tool. `id` correlates it with the
    result that answers it."""

    id: str
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        return {"type": "tool_call", "id": self.id, "name": self.name,
                "arguments": dict(self.arguments)}


@dataclass(frozen=True)
class ToolResultPart:
    """The answer to a tool call, carried on a USER-role message —
    every provider models it that way, whatever it calls the role."""

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


Part = TextPart | ToolCallPart | ToolResultPart


@dataclass(frozen=True)
class Message:
    role: str
    content: tuple[Part, ...]

    def text(self) -> str:
        """The message's text parts joined — what a reader wants when
        the conversation carries no tools."""
        return "".join(p.text for p in self.content if isinstance(p, TextPart))

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


def part(value: Any) -> Part:
    """Coerce a wire part (or a bare string) into a Part."""
    if isinstance(value, (TextPart, ToolCallPart, ToolResultPart)):
        return value
    if isinstance(value, str):
        return TextPart(value)
    if not isinstance(value, Mapping):
        raise TypeError(f"a content part is a string or an object, not {type(value).__name__}")
    kind = value.get("type", "text")
    if kind == "text":
        return TextPart(str(value.get("text", "")))
    if kind == "tool_call":
        return ToolCallPart(id=str(value.get("id", "")),
                            name=str(value["name"]),
                            arguments=dict(value.get("arguments") or {}))
    if kind == "tool_result":
        return ToolResultPart(tool_call_id=str(value.get("tool_call_id", "")),
                              content=_as_text(value.get("content", "")),
                              is_error=bool(value.get("is_error", False)))
    raise ValueError(
        f"unknown content part type {kind!r} — text, tool_call and "
        "tool_result are what the canonical model carries today "
        "(images and audio arrive with task 000343)")


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return str(value.get("text", value))
    if isinstance(value, Sequence):
        return "".join(_as_text(v) for v in value)
    return str(value)


def message(value: Any, *, default_role: str = "user") -> Message:
    """Coerce a string, a `{role, content}` mapping, or a Message."""
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
    """Coerce a conversation: a string (one user turn), a list of
    messages, or a transcript record (`{"kind": "transcript", "messages":
    [...]}`) as 000341's conversation block emits."""
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
    """What a chat call asks for, independent of who answers it.

    `provider_options` is keyed by provider name so one request can
    carry per-provider extras and still be a single canonical object:
    the adapter merges its own key and ignores the others.
    """

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

    def options_for(self, provider: str) -> dict[str, Any]:
        """This provider's passthrough options (never another's)."""
        opts = self.provider_options.get(provider) or {}
        if not isinstance(opts, Mapping):
            raise TypeError(
                f"provider_options[{provider!r}] must be an object of "
                "provider-native fields")
        return dict(opts)

    def with_messages(self, ms: Sequence[Message]) -> ChatRequest:
        from dataclasses import replace

        return replace(self, messages=tuple(ms))


def request(value: Any = None, **overrides: Any) -> ChatRequest:
    """Build a ChatRequest from params (a block's `params` dict) plus
    keyword overrides. Unknown keys are refused: a typo in `max_tokens`
    that silently sampled 1024 tokens against a budget would be a bill,
    not a bug report."""
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
    return ChatRequest(messages=ms, tools=tools, stop=stop, **raw)


def canonical(req: ChatRequest, *, provider: str | None = None) -> dict[str, Any]:
    """The request's identity: what was asked, never how it was
    authenticated. Absent optional fields are omitted rather than
    written as null, so adding a field later does not change the hash
    of a request that never set it."""
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
    opts = ({provider: req.options_for(provider)} if provider
            else {k: dict(v) for k, v in req.provider_options.items()})
    opts = {k: v for k, v in opts.items() if v}
    if opts:
        out["provider_options"] = opts
    return out


def request_hash(req: ChatRequest, *, provider: str | None = None) -> str:
    """sha256 of the canonical request's canonical CBOR — the cassette
    key, the provenance `request` field, and the intent-log key."""
    from mechbench_schema import dump_canonical

    return hashlib.sha256(dump_canonical(canonical(req, provider=provider))).hexdigest()


def estimate_tokens(req: ChatRequest) -> int:
    """A characters/4 estimate, used only where a provider offers no
    token counter. Every provenance record says which it was
    (`tokens_exact`), because a budget refusal computed from a guess
    should read as a guess."""
    n = len(req.system)
    for m in req.messages:
        for p in m.content:
            if isinstance(p, TextPart):
                n += len(p.text)
            elif isinstance(p, ToolCallPart):
                n += len(p.name) + len(str(dict(p.arguments)))
            else:
                n += len(p.content)
    for t in req.tools:
        n += len(t.name) + len(t.description) + len(str(dict(t.input_schema)))
    return max(1, n // 4)
