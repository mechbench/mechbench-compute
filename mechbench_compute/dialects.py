from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

ParseResult = tuple[str, list["pm.ToolCallPart"]]

from mechbench_compute.providers import messages as pm
from mechbench_compute.tools import ToolDef


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
    supports_tools: bool
    rendered: str | None
    reason: str | None = None


def probe_template(tokenizer) -> TemplateProbe:
    plain = [{"role": "user", "content": PROBE_QUESTION}]
    try:
        without = tokenizer.apply_chat_template(
            plain, tokenize=False, add_generation_prompt=True)
    except Exception as e:  # noqa: BLE001
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
    except Exception as e:  # noqa: BLE001
        return TemplateProbe(True, None, f"cannot render a tool call: {e}")
    return TemplateProbe(True, rendered)


_GEMMA4_CALL = re.compile(
    r"<\|tool_call\|?>\s*call:\s*([A-Za-z_][\w.]*)\s*\{(.*?)\}\s*<\/?tool_call\|?>",
    re.DOTALL)
_GEMMA4_ARG = re.compile(r'([A-Za-z_][\w.]*)\s*:\s*<\|"\|>(.*?)<\|"\|>', re.DOTALL)
_GEMMA4_BARE_ARG = re.compile(r'([A-Za-z_][\w.]*)\s*:\s*([^,{}]+)')

_QWEN_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_LLAMA_CALL = re.compile(
    r'\{[^{}]*"name"\s*:\s*"[^"]+"[^{}]*"parameters"\s*:\s*\{.*?\}\s*\}', re.DOTALL)


def _gemma4(text: str, tools: Sequence[ToolDef]) -> ParseResult:
    known = {t.name for t in tools}
    out: list[pm.ToolCallPart] = []
    spans: list[tuple[int, int]] = []
    for m in _GEMMA4_CALL.finditer(text):
        if known and m.group(1) not in known:
            continue
        body = m.group(2) or ""
        args: dict[str, Any] = {k: v for k, v in _GEMMA4_ARG.findall(body)}
        for k, v in _GEMMA4_BARE_ARG.findall(body):
            if k not in args and '<|"|>' not in v:
                args[k] = _scalar(v.strip())
        out.append(_call(m.group(1), args, len(out)))
        spans.append(m.span())
    return _without(text, spans), out


def _qwen(text: str, tools: Sequence[ToolDef]) -> ParseResult:
    known = {t.name for t in tools}
    out: list[pm.ToolCallPart] = []
    spans: list[tuple[int, int]] = []
    for m in _QWEN_CALL.finditer(text):
        parsed = _json(m.group(1))
        if (isinstance(parsed, Mapping) and parsed.get("name")
                and not (known and parsed["name"] not in known)):
            args = parsed.get("arguments")
            out.append(_call(str(parsed["name"]),
                             dict(args) if isinstance(args, Mapping) else {},
                             len(out)))
            spans.append(m.span())
    return _without(text, spans), out


def _llama(text: str, tools: Sequence[ToolDef]) -> ParseResult:
    known = {t.name for t in tools}
    out: list[pm.ToolCallPart] = []
    spans: list[tuple[int, int]] = []
    for m in _LLAMA_CALL.finditer(text):
        parsed = _json(m.group(0))
        if (isinstance(parsed, Mapping) and parsed.get("name")
                and not (known and parsed["name"] not in known)):
            params = parsed.get("parameters")
            out.append(_call(str(parsed["name"]),
                             dict(params) if isinstance(params, Mapping) else {},
                             len(out)))
            spans.append(m.span())
    return _without(text, spans), out


def _without(text: str, spans: Sequence[tuple[int, int]]) -> str:
    if not spans:
        return text.strip()
    return text[:min(start for start, _ in spans)].strip()


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


@dataclass(frozen=True)
class ToolDialect:
    name: str
    signature: str
    parse: Callable[[str, Sequence[ToolDef]], ParseResult]
    result_role: str = "tool"


DIALECTS: tuple[ToolDialect, ...] = (
    ToolDialect("gemma-4", "<|tool_call>", _gemma4),
    ToolDialect("qwen-2.5", "<tool_call>", _qwen),
    ToolDialect("llama-3", "<|start_header_id|>ipython", _llama, "ipython"),
)


def identify(probe: TemplateProbe) -> ToolDialect | None:
    if not probe.supports_tools or not probe.rendered:
        return None
    for d in DIALECTS:
        if d.signature in probe.rendered:
            return d
    return None


class NoToolDialect(RuntimeError):
    pass


def dialect_for(tokenizer, *, model: str = "") -> ToolDialect:
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


def tool_to_hf(tool: ToolDef) -> dict[str, Any]:
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
    return {"type": "function",
            "function": {"name": call.name,
                         "arguments": dict(call.arguments or {})}}


def result_message(dialect: ToolDialect | None, name: str,
                   content: str) -> dict[str, Any]:
    return {"role": dialect.result_role if dialect else "tool",
            "name": name, "content": content}


_ATTEMPTING: dict[str, tuple[str, ...]] = {
    "gemma-4": ("<|tool_call", "call:"),
    "qwen-2.5": ("<tool_call>", '"name"'),
    "llama-3": ('"name"', '"parameters"'),
}
_ANY_NAME = re.compile(r'(?:call:|"name"\s*:\s*")\s*([A-Za-z_][\w.]*)')

CAUSES = (
    "unknown_tool",
    "unparseable_call",
    "no_dialect",
    "execution_failed",
)


@dataclass(frozen=True)
class ToolError:
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


@dataclass(frozen=True)
class DialectReport:
    model: str
    dialect: str | None
    supports_tools: bool
    detail: str


def describe(tokenizer, model: str = "") -> DialectReport:
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
