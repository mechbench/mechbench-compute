"""Tools as blocks (task 000340, epic 000334).

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
from mechbench_compute.tools.coerce_text import coerce_text  # noqa: F401
from mechbench_compute.tools.tool_def import ToolDef  # noqa: F401
from mechbench_compute.tools.tool_run import ToolRun  # noqa: F401
from mechbench_compute.tools.toolbox import Toolbox  # noqa: F401
from mechbench_compute.tools.build_toolbox import BUILTIN_TOOLS, build_toolbox  # noqa: F401

#: How a local family writes a tool call in plain text.


# --- local families: writing tools down, and reading calls back ------------------


# The tool protocol lives in `dialects`, taken from each model's own
# chat template (epic 000439). What stood here — a markdown fence we
# invented, a parser for it, and two more parsers for the shapes models
# degraded into when prompted with it — is deleted rather than left
# beside the real thing: two ways to read a tool call is how the next
# reader picks the wrong one.


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


PURE_TOOL_BLOCKS = {
}


