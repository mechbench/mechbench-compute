"""Tool dialects, taken from each model's own chat template (epic
000439).

The harness used to invent a tool convention — a markdown fence — and
ask every local model to speak it. Models do not: they speak the
protocol they were trained on, which ships in `chat_template.jinja`
alongside the weights. Prompted with our fence, `gemma-4-e2b` emitted
degraded approximations of its real format and the harness read none
of them. Experiment 024 executed zero tool calls across 320
generations and nothing said so.

So: **the chat template is the source of truth.** It is versioned with
the weights, it defines the declaration, the call and the result, and
it is not a thing we get to have an opinion about.

A template RENDERS but does not PARSE, so reading a call is the one
piece of code here. It is not guessed from sample outputs — that is
how we got the fence — but written against the template's own
rendering and pinned by a round-trip test: render a canonical call
through the model's template, parse it back, assert equality. When a
model publishes a new template, that test fails instead of an
experiment.

The four dialects below are what our cached models actually emit:

    gemma-4     <|tool_call>call:calc{expression:<|"|>37 + 18<|"|>}<tool_call|>
    qwen-2.5    <tool_call>\\n{"name": "calc", "arguments": {…}}\\n</tool_call>
    llama-3     {"name": "calc", "parameters": {…}}
    (gemma-3    no tool support in its template at all — refuse)
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from mechbench_compute.providers import messages as pm
from mechbench_compute.tools import ToolDef

# --- the canonical probe ------------------------------------------------
#
# One triple, rendered through the model's own template, is where every
# fact about its dialect comes from.

PROBE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "calc",
        "description": "Evaluate an arithmetic expression.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string",
                                          "description": "The expression"}},
            "required": ["expression"],
        },
    },
}
PROBE_QUESTION = "37 + 18?"
PROBE_ARGS = {"expression": "37 + 18"}
PROBE_RESULT = "55"


@dataclass(frozen=True)
class TemplateProbe:
    """What a model's own template says about tools."""

    supports_tools: bool
    #: The template's rendering of a call + result, when it has one.
    rendered: str | None
    #: Why not, when `supports_tools` is False.
    reason: str | None = None


def probe_template(tokenizer) -> TemplateProbe:
    """Ask the template, rather than the repo name, what it can do.

    `supports_tools` is decided by DIFFERENCE: render the same messages
    with and without `tools=` and see whether the prompt changes. A
    template that accepts the argument and ignores it does not support
    tools, whatever its signature says.
    """
    plain = [{"role": "user", "content": PROBE_QUESTION}]
    try:
        without = tokenizer.apply_chat_template(
            plain, tokenize=False, add_generation_prompt=True)
    except Exception as e:  # noqa: BLE001 — a template we cannot drive at all
        return TemplateProbe(False, None, f"no usable chat template: {e}")
    try:
        with_tools = tokenizer.apply_chat_template(
            plain, tools=[PROBE_TOOL], tokenize=False,
            add_generation_prompt=True)
    except Exception as e:  # noqa: BLE001
        return TemplateProbe(False, None, f"template rejects tools: {e}")
    if with_tools == without:
        return TemplateProbe(
            False, None,
            "the template accepts `tools` and ignores it — the model was "
            "not trained to receive tool declarations")
    full = plain + [
        {"role": "assistant", "content": "",
         "tool_calls": [{"type": "function",
                         "function": {"name": "calc", "arguments": PROBE_ARGS}}]},
        {"role": "tool", "name": "calc", "content": PROBE_RESULT},
    ]
    try:
        rendered = tokenizer.apply_chat_template(
            full, tools=[PROBE_TOOL], tokenize=False,
            add_generation_prompt=True)
    except Exception as e:  # noqa: BLE001 — declares tools, cannot render one
        return TemplateProbe(True, None, f"cannot render a tool call: {e}")
    return TemplateProbe(True, rendered)


# --- parsers ------------------------------------------------------------

_GEMMA4_CALL = re.compile(
    r"<\|tool_call\|?>\s*call:\s*([A-Za-z_][\w.]*)\s*\{(.*?)\}\s*<\/?tool_call\|?>",
    re.DOTALL)
_GEMMA4_ARG = re.compile(r'([A-Za-z_][\w.]*)\s*:\s*<\|"\|>(.*?)<\|"\|>', re.DOTALL)
_GEMMA4_BARE_ARG = re.compile(r'([A-Za-z_][\w.]*)\s*:\s*([^,{}]+)')

_QWEN_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_LLAMA_CALL = re.compile(
    r'\{[^{}]*"name"\s*:\s*"[^"]+"[^{}]*"parameters"\s*:\s*\{.*?\}\s*\}', re.DOTALL)


def _gemma4(text: str, tools: Sequence[ToolDef]) -> list[pm.ToolCallPart]:
    """`<|tool_call>call:calc{expression:<|"|>37 + 18<|"|>}<tool_call|>`

    Values are wrapped in the template's own `<|"|>` quoting; numbers
    and booleans come through bare.
    """
    out = []
    for m in _GEMMA4_CALL.finditer(text):
        body = m.group(2) or ""
        args: dict[str, Any] = {k: v for k, v in _GEMMA4_ARG.findall(body)}
        for k, v in _GEMMA4_BARE_ARG.findall(body):
            if k not in args and '<|"|>' not in v:
                args[k] = _scalar(v.strip())
        out.append(_call(m.group(1), args, len(out)))
    return out


def _qwen(text: str, tools: Sequence[ToolDef]) -> list[pm.ToolCallPart]:
    """`<tool_call>{"name": …, "arguments": {…}}</tool_call>`"""
    out = []
    for m in _QWEN_CALL.finditer(text):
        parsed = _json(m.group(1))
        if isinstance(parsed, Mapping) and parsed.get("name"):
            args = parsed.get("arguments")
            out.append(_call(str(parsed["name"]),
                             dict(args) if isinstance(args, Mapping) else {},
                             len(out)))
    return out


def _llama(text: str, tools: Sequence[ToolDef]) -> list[pm.ToolCallPart]:
    """A bare object with `name` and `parameters` — no envelope at all,
    which is why this parser must be the least eager of the three."""
    out = []
    for m in _LLAMA_CALL.finditer(text):
        parsed = _json(m.group(0))
        if isinstance(parsed, Mapping) and parsed.get("name"):
            params = parsed.get("parameters")
            out.append(_call(str(parsed["name"]),
                             dict(params) if isinstance(params, Mapping) else {},
                             len(out)))
    return out


def _call(name: str, args: Mapping[str, Any], i: int) -> pm.ToolCallPart:
    return pm.ToolCallPart(id=f"local_{i}", name=name, arguments=dict(args))


def _scalar(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw.strip("\"'")


def _json(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


# --- the registry -------------------------------------------------------


@dataclass(frozen=True)
class ToolDialect:
    """One model family's tool protocol.

    `signature` is what identifies it in a rendering of the canonical
    probe — so a dialect is matched against what the model EMITS, not
    against its repo name, which is decoration a re-uploader can
    change.
    """

    name: str
    signature: str
    parse: Callable[[str, Sequence[ToolDef]], list[pm.ToolCallPart]]
    #: Role a tool result is delivered under, for the transcript.
    result_role: str = "tool"


DIALECTS: tuple[ToolDialect, ...] = (
    ToolDialect("gemma-4", "<|tool_call>", _gemma4),
    ToolDialect("qwen-2.5", "<tool_call>", _qwen),
    # Llama has no envelope, so it is matched last and by its role
    # marker rather than by anything in the call itself.
    ToolDialect("llama-3", "<|start_header_id|>ipython", _llama, "ipython"),
)


def identify(probe: TemplateProbe) -> ToolDialect | None:
    """Which dialect this model speaks, from its own rendering."""
    if not probe.supports_tools or not probe.rendered:
        return None
    for d in DIALECTS:
        if d.signature in probe.rendered:
            return d
    return None


class NoToolDialect(RuntimeError):
    """Tools were offered to a model we cannot speak tools with.

    Raised rather than falling back to a convention of our own. The
    fallback is what cost experiment 024 an arm: a model that cannot
    receive a tool declaration produces output that is
    indistinguishable, downstream, from a model that chose not to call
    anything.
    """


def dialect_for(tokenizer, *, model: str = "") -> ToolDialect:
    """The model's dialect, or a refusal naming what is missing."""
    probe = probe_template(tokenizer)
    found = identify(probe)
    if found is not None:
        return found
    who = f" ({model})" if model else ""
    if not probe.supports_tools:
        raise NoToolDialect(
            f"this model{who} has no tool protocol in its chat template "
            f"— {probe.reason}. Offering it tools would mean inventing a "
            f"convention it was never trained on, which reads as silence "
            f"downstream. Remove `tools`, or add a dialect for it.")
    raise NoToolDialect(
        f"this model{who} declares tools in its chat template, but its "
        f"rendering matches no known dialect. Add one to "
        f"`dialects.DIALECTS` with a round-trip test. Rendered:\n"
        f"{(probe.rendered or '')[:400]}")


# --- the other two legs -------------------------------------------------


def tool_to_hf(tool: ToolDef) -> dict[str, Any]:
    """Our `ToolDef` in the shape `apply_chat_template(tools=…)` wants.

    The template renders the declaration from this — so the model sees
    its own native format, not a description of ours.
    """
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": dict(tool.schema or {"type": "object",
                                               "properties": {}}),
        },
    }


def call_to_hf(call: pm.ToolCallPart) -> dict[str, Any]:
    """An assistant turn's tool call, for the template to render back
    into the transcript in the model's own format."""
    return {"type": "function",
            "function": {"name": call.name,
                         "arguments": dict(call.arguments or {})}}


def result_message(dialect: ToolDialect | None, name: str,
                   content: str) -> dict[str, Any]:
    """A tool result as a transcript turn.

    The role matters: Llama delivers results under `ipython`, Qwen and
    Gemma under `tool`. Getting it wrong means the model reads its own
    tool output as if a user had said it.
    """
    return {"role": dialect.result_role if dialect else "tool",
            "name": name, "content": content}


# --- tool call errors ---------------------------------------------------

#: Shapes that mean "this response was attempting a call", per dialect.
#: Declared beside the parsers on purpose: a detector that knows a
#: different set of formats from the parser is how an error goes
#: uncounted.
_ATTEMPTING: dict[str, tuple[str, ...]] = {
    "gemma-4": ("<|tool_call", "call:"),
    "qwen-2.5": ("<tool_call>", '"name"'),
    "llama-3": ('"name"', '"parameters"'),
}
_ANY_NAME = re.compile(r'(?:call:|"name"\s*:\s*")\s*([A-Za-z_][\w.]*)')

#: Every way a tool call can fail. There is no "near miss" — a call
#: that did not execute is an error, and the only question worth
#: asking is whose.
CAUSES = (
    "unknown_tool",         # the model called something never offered
    "unparseable_call",     # attempted, in a shape we could not read
    "no_dialect",           # tools offered to a model with no protocol
    "execution_failed",     # the tool ran and raised
)


@dataclass(frozen=True)
class ToolError:
    """One tool call that did not produce a result.

    Recorded per item and aggregated on the node, never silently
    counted: an individual failure does not fail the run by default
    (`on_tool_error`), but it is always in the results, because a run
    that quietly did less than it was asked to is the failure mode this
    whole area keeps producing.
    """

    cause: str
    detail: str
    tool: str = ""
    sample: str = ""

    def to_wire(self) -> dict[str, Any]:
        out = {"cause": self.cause, "detail": self.detail}
        if self.tool:
            out["tool"] = self.tool
        if self.sample:
            out["sample"] = self.sample
        return out


def call_error(text: str, tools: Sequence[ToolDef],
               dialect: ToolDialect | None) -> ToolError | None:
    """The error in a response that produced no executed call, or None
    when the model simply answered.

    **Answering without calling a tool is not an error.** Whether the
    model SHOULD have called one is the experiment's question, not the
    harness's; it is counted as a statistic instead.
    """
    named = _ANY_NAME.search(text)
    if dialect is None:
        if named:
            return ToolError("no_dialect",
                             "the model attempted a call but has no known "
                             "tool protocol", named.group(1), text[:200])
        return None
    if not any(m in text for m in _ATTEMPTING.get(dialect.name, ())):
        return None
    known = {t.name for t in tools}
    if named and named.group(1) not in known:
        return ToolError(
            "unknown_tool",
            f"called {named.group(1)!r}, which was not offered; available: "
            f"{', '.join(sorted(known)) or '(none)'}",
            named.group(1), text[:200])
    return ToolError(
        "unparseable_call",
        f"attempted a call in {dialect.name} but it could not be read — "
        f"this is usually OUR parser, not the model",
        named.group(1) if named else "", text[:200])


# --- reporting ----------------------------------------------------------


@dataclass(frozen=True)
class DialectReport:
    """What we can say about one model's tool support, without guessing."""

    model: str
    dialect: str | None
    supports_tools: bool
    detail: str


def describe(tokenizer, model: str = "") -> DialectReport:
    """One model's tool story, for `doctor` and for a human deciding
    whether a protocol can offer tools at all.

    Answers from the template every time. There is no table of model
    names here to go stale.
    """
    probe = probe_template(tokenizer)
    found = identify(probe)
    if found is not None:
        return DialectReport(model, found.name, True,
                             f"speaks {found.name}; results delivered as "
                             f"role {found.result_role!r}")
    if not probe.supports_tools:
        return DialectReport(model, None, False,
                             probe.reason or "no tool protocol")
    return DialectReport(
        model, None, True,
        "declares tools but matches no known dialect — add one to "
        "DIALECTS with a round-trip test")
