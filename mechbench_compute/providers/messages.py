"""The canonical message model.

One shape of conversation crosses every provider and the local MLX
path: a system string, an alternating sequence of user/assistant
messages, and content parts that are `text`, `reasoning`, `tool_call`
or `tool_result`. Tools are JSON Schema.

**Reasoning is never text.** A `reasoning` part is the model's
reasoning as its provider returned it: readable text when the provider
shows it, and the provider's own block verbatim (`native`), which is
what goes back on a later turn. Nothing that joins parts into prose
reads it.

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

#: The values of a request's `api`: None and "chat_completions" are one
#: choice, the provider's usual API.
APIS = (None, "chat_completions", "responses")


@dataclass(frozen=True)
class Signature:
    """A provider's continuation token riding on a text or tool-call
    part: opaque bytes that go back on that same part, to the provider
    and model that issued them, and nowhere else. Gemini's
    `thoughtSignature` is one; on the Responses API it is the output
    item the part was read from (a `message` or a `function_call`, with
    its id and `phase`), as JSON, which goes back as that item."""

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
    """The model's reasoning, kept apart from its reply.

    `text` is the reasoning a person can read, when the provider returns
    it; `redacted` marks reasoning with no readable text — withheld or
    encrypted. `provider` and `model` say who wrote it. `native` is the
    provider's own form of it, verbatim — a thinking block with its
    signature, an opaque payload, a message-level field — and it goes
    back, unchanged and in its place, only to an adapter that accepts it
    (`is_replayable`). Nothing that joins parts into prose reads it.
    """

    text: str = ""
    redacted: bool = False
    provider: str = ""
    model: str = ""
    native: Mapping[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """The part as an item's `reasoning` entry."""
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
    """A model's request to run a tool. `id` correlates it with the
    result that answers it."""

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


Part = TextPart | ReasoningPart | ToolCallPart | ToolResultPart


@dataclass(frozen=True)
class Message:
    role: str
    content: tuple[Part, ...]

    def text(self) -> str:
        """The message's prose: its text parts joined, never its
        reasoning."""
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
    """The prose of a sequence of parts: text parts only. Every reply's
    `text` is built here, so reasoning cannot reach it."""
    return "".join(p.text for p in parts if isinstance(p, TextPart))


def read_reasoning(parts: Iterable[Any]) -> list[dict[str, Any]]:
    """The reasoning among `parts`, as an item's `reasoning` entries."""
    return [p.to_record() for p in parts if isinstance(p, ReasoningPart)]


def is_replayable(origin_provider: str, origin_model: str, *, provider: str,
                  model: str, model_bound: bool = True) -> bool:
    """Whether reasoning or a signature written by `origin_*` may go to
    `provider`/`model`. Never to another provider. `model_bound=False`
    is for a provider that accepts any of its own models' blocks and
    drops the ones the target cannot read."""
    if not origin_provider or origin_provider != provider:
        return False
    return not model_bound or origin_model == model


def part(value: Any) -> Part:
    """Coerce a wire part (or a bare string) into a Part."""
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
        f"unknown content part type {kind!r} — text, reasoning, tool_call "
        "and tool_result are what the canonical model carries today "
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
    messages, or a transcript record (`{"kind": "text/transcript", "messages":
    [...]}`) as a conversation emits it."""
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
    #: Which of a provider's APIs answers: None for its usual one, or
    #: "responses" for OpenAI's and xAI's Responses API. Part of the
    #: request's identity, since the two return different things for
    #: one question (encrypted reasoning, on the Responses API only).
    api: str | None = None

    def options_for(self, provider: str) -> dict[str, Any]:
        """This provider's passthrough options (never another's)."""
        opts = self.provider_options.get(provider) or {}
        if not isinstance(opts, Mapping):
            raise TypeError(
                f"provider_options[{provider!r}] must be an object of "
                "provider-native fields")
        return dict(opts)

    def check_options(self) -> None:
        """Every key of `provider_options` must name a provider. The
        bag is keyed by provider, so a caller who writes
        the provider-native field at the top level —
        `provider_options: {"thinking": …}` — has written something no
        adapter will ever read, and nothing would have said so."""
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
    api = raw.pop("api", None)
    if api not in APIS:
        raise ValueError(
            f"api is one of {', '.join(a for a in APIS if a)}, not {api!r}")
    # Chat Completions is what `api` absent means, so naming it changes
    # nothing about the request, its hash included.
    raw["api"] = None if api == "chat_completions" else api
    req = ChatRequest(messages=ms, tools=tools, stop=stop, **raw)
    req.check_options()
    return req


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
    if req.api:
        out["api"] = req.api
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
            elif isinstance(p, ReasoningPart):
                n += len(p.text) + len(str(dict(p.native)))
            elif isinstance(p, ToolCallPart):
                n += len(p.name) + len(str(dict(p.arguments)))
            else:
                n += len(p.content)
    for t in req.tools:
        n += len(t.name) + len(t.description) + len(str(dict(t.input_schema)))
    return max(1, n // 4)
