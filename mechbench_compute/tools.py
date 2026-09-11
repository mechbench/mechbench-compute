"""Tools as blocks (task 000340, epic 000334).

A tool is not a special kind of code. It is a name, a JSON Schema, and
a HANDLER that is an ordinary block — so anything the platform can
already do, a model can be given as a capability, and the tool call is
recorded with the same provenance as everything else.

    {"name": "calc", "description": "…", "schema": {…},
     "handler": {"block": "~canonical/ops/tools/calc/1"}}

Two halves make it work everywhere:

**Remote providers call tools natively** — Anthropic tool_use, OpenAI
functions, Gemini functionDeclarations — which the transport already
maps to `ToolCallPart`s.

**Local models do not**, so a family PARSER reads tool calls out of
the text a model wrote (Gemma's fenced `tool_code` block first) and a
RENDERER describes the tools in the system prompt. The parser is
deliberately forgiving about formatting and strict about names: a call
to a tool that does not exist comes back as an error result the model
can read, not an exception that kills the run.

A tool result is data, not trust: a handler that raises becomes an
`is_error` result carrying the message, because the interesting
behaviour is what the model does when its tool fails.
"""

from __future__ import annotations

import ast
import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from mechbench_compute.providers import messages as pm

#: How a local family writes a tool call in plain text.
FAMILIES = ("gemma", "generic_json")


@dataclass(frozen=True)
class ToolDef:
    """A capability offered to a model. `handler` is
    `{"block": ref, "params": {...}}` or `{"protocol": "<id>"}`; the
    second runs as a nested pipeline through the executor's runner."""

    name: str
    description: str = ""
    schema: Mapping[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}})
    handler: Mapping[str, Any] = field(default_factory=dict)

    @staticmethod
    def parse(value: Any) -> ToolDef:
        if isinstance(value, ToolDef):
            return value
        if not isinstance(value, Mapping):
            raise TypeError(
                "a tool is an object {name, description, schema, handler}, "
                f"not {type(value).__name__}")
        name = str(value.get("name") or "")
        if not name:
            raise ValueError("a tool needs a name — it is what the model calls")
        schema = value.get("schema") or value.get("input_schema") or {
            "type": "object", "properties": {}}
        handler = value.get("handler") or {}
        if handler and not ({"block", "protocol"} & set(handler)):
            raise ValueError(
                f"tool {name!r}: a handler is {{'block': <ref>}} or "
                "{'protocol': <id>}")
        return ToolDef(name=name, description=str(value.get("description", "")),
                       schema=dict(schema), handler=dict(handler))

    def to_spec(self) -> pm.ToolSpec:
        return pm.ToolSpec(name=self.name, description=self.description,
                           input_schema=dict(self.schema))

    def to_wire(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "schema": dict(self.schema), "handler": dict(self.handler)}


@dataclass
class ToolRun:
    """What happened when a tool ran — the provenance of a capability
    exercised, recorded beside the model call that asked for it."""

    tool: str
    arguments: Mapping[str, Any]
    handler: Mapping[str, Any]
    output: Any = None
    error: str = ""
    duration_ms: int = 0

    def to_wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {"tool": self.tool,
                               "arguments": dict(self.arguments),
                               "handler": dict(self.handler),
                               "duration_ms": self.duration_ms}
        if self.error:
            out["error"] = self.error
        return out


class Toolbox:
    """The tools one participant may call, and the machinery to run
    them. `block_runner(ref, inputs, params)` is injected by the
    executor for handlers it must own (model blocks, sub-protocols);
    pure blocks run here."""

    def __init__(self, tools: Sequence[Any] = (), *, block_runner=None) -> None:
        self.tools = [ToolDef.parse(t) for t in tools]
        self._by_name = {t.name: t for t in self.tools}
        if len(self._by_name) != len(self.tools):
            raise ValueError("tool names must be unique within a toolbox")
        self._runner = block_runner
        self.runs: list[ToolRun] = []

    def __bool__(self) -> bool:
        return bool(self.tools)

    def specs(self) -> tuple[pm.ToolSpec, ...]:
        return tuple(t.to_spec() for t in self.tools)

    def call(self, call: pm.ToolCallPart) -> pm.ToolResultPart:
        """Run one tool call. Every failure becomes a result the model
        can read: an unknown name, a handler that raised, a handler
        that was never wired."""
        started = time.monotonic()
        tool = self._by_name.get(call.name)
        if tool is None:
            known = ", ".join(sorted(self._by_name)) or "(none)"
            return self._error(call, f"no such tool: {call.name!r}. Available: {known}",
                               {}, started)
        try:
            output = self._dispatch(tool, dict(call.arguments))
        except Exception as e:  # noqa: BLE001 — a tool's failure is data
            return self._error(call, f"{type(e).__name__}: {e}", tool.handler, started)
        run = ToolRun(tool=tool.name, arguments=dict(call.arguments),
                      handler=dict(tool.handler), output=output,
                      duration_ms=int((time.monotonic() - started) * 1000))
        self.runs.append(run)
        return pm.ToolResultPart(tool_call_id=call.id, content=_as_text(output))

    def _error(self, call: pm.ToolCallPart, message: str,
               handler: Mapping[str, Any], started: float) -> pm.ToolResultPart:
        self.runs.append(ToolRun(
            tool=call.name, arguments=dict(call.arguments), handler=dict(handler),
            error=message, duration_ms=int((time.monotonic() - started) * 1000)))
        return pm.ToolResultPart(tool_call_id=call.id, content=message,
                                 is_error=True)

    def _dispatch(self, tool: ToolDef, arguments: Mapping[str, Any]) -> Any:
        handler = tool.handler
        if not handler:
            raise ValueError(
                f"tool {tool.name!r} has no handler — it can be offered to a "
                "model but not run")
        ref = handler.get("block")
        params = dict(handler.get("params") or {})
        # A handler block sees the call's arguments on its own port AND
        # as a single record, so an ordinary record block works as a
        # tool without knowing it is one.
        inputs = {"arguments": dict(arguments), "records": [dict(arguments)]}
        if ref is None:
            if self._runner is None:
                raise ValueError(
                    f"tool {tool.name!r} names a protocol handler, which only "
                    "the executor can run")
            return self._runner(handler["protocol"], inputs, params)
        from mechbench_compute.blocks import PURE_BLOCKS

        if ref in PURE_BLOCKS:
            return PURE_BLOCKS[ref](inputs, params)
        # A model block or a sub-protocol: the executor owns those, and
        # a toolbox built without one can say so precisely.
        if self._runner is None:
            raise ValueError(
                f"tool {tool.name!r}: {ref!r} is not a pure block, so it needs "
                "the executor's runner — offer this tool from a protocol node")
        return self._runner(ref, inputs, params)


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(value)


# --- local families: writing tools down, and reading calls back ------------------


_GEMMA_BLOCK = re.compile(r"```(?:tool_code|tool_call|json)\s*(.+?)```", re.DOTALL)
_BARE_JSON = re.compile(r"\{[^{}]*\"name\"\s*:\s*\"[^\"]+\"[^{}]*\}", re.DOTALL)

#: Gemma's OWN tool-call emission, which it uses in preference to any
#: convention we instruct: `<|tool_call>call:calc({"expression": "1+1"})`.
#: Task 000437 — experiment 024 offered `calc` across 320 generations
#: and parsed zero calls, because the base model wrote this and we
#: matched only the fenced block we had asked for. A family named after
#: a model must read what that model writes.
_NATIVE_CALL = re.compile(
    r"<\|tool_call\|?>\s*(?:call:)?\s*([A-Za-z_][\w.]*)\s*\((.*?)\)"
    r"\s*(?:<\|?/?tool_call\|?>)?\s*(?:<\|tool_response\|?>)?",
    re.DOTALL)

#: Output that was TRYING to call a tool. Used only to count near
#: misses: a correct call and no call must not look the same.
_TOOL_SHAPED = re.compile(
    r"<\|tool_call|```(?:tool_code|tool_call)|\bcall:\w+\s*\(|\"name\"\s*:",
    re.IGNORECASE)


def looks_like_a_tool_call(text: str) -> bool:
    """Did the model appear to be calling something? A response that
    parses to no call but looks like this is a near miss, and silence
    about near misses is what made 000437 cost an experiment arm."""
    return bool(_TOOL_SHAPED.search(text))


def render_tools(tools: Sequence[ToolDef], *, family: str = "gemma") -> str:
    """The system-prompt fragment that tells a local model what it may
    call and how to write the call. Providers do this natively; a local
    model has only its prompt."""
    if family not in FAMILIES:
        raise ValueError(f"unknown tool family {family!r} — one of {FAMILIES}")
    if not tools:
        return ""
    lines = ["You can call tools. To call one, write a fenced block:", "",
             "```tool_code", '{"name": "<tool>", "arguments": {…}}', "```", ""]
    if family == "gemma":
        # Gemma reaches for its own format regardless of what we ask,
        # so say that it is accepted rather than letting the model
        # choose between obeying us and obeying its training.
        lines += ['Writing `<|tool_call>call:<tool>({…})` works too.', ""]
    lines.append("Call at most one tool per turn, and wait for its result "
                 "before continuing. Available tools:")
    for t in tools:
        props = ", ".join((t.schema.get("properties") or {}).keys()) or "no arguments"
        lines.append(f"- {t.name}({props}) — {t.description or 'no description'}")
    return "\n".join(lines)


def parse_tool_calls(text: str, *, tools: Sequence[ToolDef] = (),
                     family: str = "gemma") -> tuple[str, list[pm.ToolCallPart]]:
    """Read tool calls out of what a local model wrote.

    Returns (text with the call blocks removed, calls). Strict about
    names — an unknown tool is left in the text rather than invented —
    and forgiving about everything else, because a model that writes
    `{'name': …}` with single quotes still meant to call the tool.
    """
    if family not in FAMILIES:
        raise ValueError(f"unknown tool family {family!r} — one of {FAMILIES}")
    known = {t.name for t in tools}
    calls: list[pm.ToolCallPart] = []
    spans: list[tuple[int, int]] = []
    if family == "gemma":
        by_name = {t.name: t for t in tools}
        for m in _NATIVE_CALL.finditer(text):
            name = m.group(1)
            if known and name not in known:
                continue
            args = _native_args((m.group(2) or "").strip(), by_name.get(name))
            if args is None:
                continue
            calls.append(pm.ToolCallPart(
                id=f"local_{len(calls)}", name=name, arguments=dict(args)))
            spans.append(m.span())
    for m in list(_GEMMA_BLOCK.finditer(text)) + list(_BARE_JSON.finditer(text)):
        if any(start <= m.start() < end for start, end in spans):
            continue
        raw = m.group(1) if m.lastindex else m.group(0)
        parsed = _loads(raw.strip())
        if not isinstance(parsed, Mapping):
            continue
        name = str(parsed.get("name") or "")
        if not name or (known and name not in known):
            continue
        args = parsed.get("arguments")
        if not isinstance(args, Mapping):
            args = {k: v for k, v in parsed.items() if k not in ("name", "arguments")}
        calls.append(pm.ToolCallPart(
            id=f"local_{len(calls)}", name=name, arguments=dict(args)))
        spans.append(m.span())
    for start, end in sorted(spans, reverse=True):
        text = text[:start] + text[end:]
    return text.strip(), calls


def _native_args(raw: str, spec: ToolDef | None) -> dict[str, Any] | None:
    """Arguments out of `call:<tool>(…)`.

    A JSON object is the easy case. But a local model asked to write
    `calc({…})` will often write `calc(37 + 18)` instead — the bare
    argument, no braces, no key — and refusing that is pedantry when
    the tool takes exactly ONE required parameter: there is nothing
    else it could have meant. Observed on gemma-4-e2b the moment the
    rendered instruction changed (000437 follow-on); `tool_near_misses`
    reported 80 of 80 rather than letting it pass unnoticed.

    Ambiguity is still refused. Two required parameters and a bare
    argument is a guess, and this does not guess.
    """
    if not raw:
        return {}
    parsed = _loads(raw)
    if isinstance(parsed, Mapping):
        return dict(parsed)
    required = list((spec.schema.get("required") or []) if spec else [])
    if len(required) != 1:
        return None
    # The literal text is the value: `calc(37 + 18)` means the
    # expression "37 + 18", not the number 55 — evaluating it here
    # would be doing the tool's job with none of its safety.
    return {required[0]: raw.strip("\"'")}


def _loads(raw: str) -> Any:
    """JSON first, then a Python literal — a local model writing single
    quotes meant the same thing."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return None


# --- the first tools ------------------------------------------------------------


class CalcRefused(ValueError):
    """`calc` met something that is not arithmetic. A refusal, not a
    type error: the expression parsed fine, it just is not allowed."""


_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Add, ast.Sub,
    ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd,
    ast.Tuple, ast.Load,
)


def calc(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    """`~canonical/ops/tools/calc/1` — arithmetic, and ONLY arithmetic.

    Parsed, walked, and refused if it contains anything but numbers and
    operators: a tool a model can steer must not be an eval.
    """
    args = dict(inputs.get("arguments") or {})
    expression = str(args.get("expression") or params.get("expression") or "")
    if not expression:
        raise ValueError("calc needs an `expression`")
    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise CalcRefused(
                f"calc refuses {type(node).__name__}: it evaluates arithmetic, "
                "not code")
    value = eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, {})
    return {"expression": expression, "result": value}


def bench_lookup(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> Any:
    """`~canonical/ops/tools/bench-lookup/1` — fetch a bench object by
    path, so a model can consult what the platform already knows.

    `fetch` is injectable (the executor passes the recording fetch, and
    tests pass their own) — nothing here touches the network directly.
    """
    args = dict(inputs.get("arguments") or {})
    path = str(args.get("path") or params.get("path") or "")
    if not path:
        raise ValueError("bench.lookup needs a `path`")
    fetch = params.get("fetch")
    if fetch is None:
        from mechbench_compute import bench

        fetch = bench.fetch
    fetched = fetch(path)
    payload = (fetched.get("payload", fetched)
               if isinstance(fetched, Mapping) else fetched)
    field_name = args.get("field")
    if field_name and isinstance(payload, Mapping):
        return {str(field_name): payload.get(str(field_name))}
    return payload


PURE_TOOL_BLOCKS = {
    "~canonical/ops/tools/calc/1": calc,
    "~canonical/ops/tools/bench-lookup/1": bench_lookup,
}

#: Ready-made definitions for the first tools, so a protocol can offer
#: them by name instead of restating a schema.
BUILTIN_TOOLS: dict[str, dict[str, Any]] = {
    "calc": {
        "name": "calc",
        "description": "Evaluate an arithmetic expression.",
        "schema": {"type": "object",
                   "properties": {"expression": {"type": "string"}},
                   "required": ["expression"]},
        "handler": {"block": "~canonical/ops/tools/calc/1"},
    },
    "bench.lookup": {
        "name": "bench.lookup",
        "description": "Fetch an object from the bench by its path.",
        "schema": {"type": "object",
                   "properties": {"path": {"type": "string"},
                                  "field": {"type": "string"}},
                   "required": ["path"]},
        "handler": {"block": "~canonical/ops/tools/bench-lookup/1"},
    },
}


def toolbox_from(value: Any, *, block_runner=None) -> Toolbox:
    """A toolbox from a params list: tool objects, or the NAME of a
    built-in ("calc"), so the common case is one word."""
    tools: list[Any] = []
    for entry in value or ():
        if isinstance(entry, str):
            if entry not in BUILTIN_TOOLS:
                raise ValueError(
                    f"unknown built-in tool {entry!r} — "
                    f"{', '.join(sorted(BUILTIN_TOOLS))}, or pass a full "
                    "definition")
            tools.append(BUILTIN_TOOLS[entry])
        else:
            tools.append(entry)
    return Toolbox(tools, block_runner=block_runner)
