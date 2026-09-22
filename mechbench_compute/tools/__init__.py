"""Tools as blocks.

A tool is not a special kind of code. It is a name, a JSON Schema, and
a HANDLER that is an ordinary block — so anything the platform can
already do, a model can be given as a capability, and the tool call is
recorded with the same provenance as everything else.

    {"name": "calc", "description": "…", "schema": {…},
     "handler": {"block": "tools/calc"}}

Two halves make it work everywhere:

**Remote providers call tools natively** — Anthropic tool_use, OpenAI
functions, Gemini functionDeclarations — which the transport already
maps to `ToolCallPart`s.

**Local models do not**, so a family PARSER reads tool calls out of
the text a model wrote (Gemma's fenced `tool_code` block first) and a
RENDERER describes the tools in the system prompt. The parser is
deliberately forgiving about formatting and strict about names: a call
to a tool that does not exist comes back as an error result the model
can read, not an exception that kills the run. Both live in `dialects`,
taken from each model's own chat template — there is one way to read a
tool call and no fence of our own beside it, because two ways to read
one is how the next reader picks the wrong one.

A tool result is data, not trust: a handler that raises becomes an
`is_error` result carrying the message, because the interesting
behaviour is what the model does when its tool fails.
"""

from __future__ import annotations

from mechbench_compute.tools.build_toolbox import build_toolbox  # noqa: F401
from mechbench_compute.tools.tool_def import ToolDef  # noqa: F401
from mechbench_compute.tools.toolbox import Toolbox  # noqa: F401

#: Pure blocks this package contributes, taken by `blocks.PURE_BLOCKS`.
PURE_TOOL_BLOCKS = {
}


