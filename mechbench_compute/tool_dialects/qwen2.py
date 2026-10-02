from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from mechbench_compute.dialects import (
    ParseResult,
    ToolDialect,
    make_call,
    parse_json,
    strip_calls,
)
from mechbench_compute.tools import ToolDef

CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_calls(text: str, tools: Sequence[ToolDef]) -> ParseResult:
    known = {t.name for t in tools}
    out = []
    spans: list[tuple[int, int]] = []
    for m in CALL.finditer(text):
        parsed = parse_json(m.group(1))
        if (isinstance(parsed, Mapping) and parsed.get("name")
                and not (known and parsed["name"] not in known)):
            args = parsed.get("arguments")
            out.append(make_call(str(parsed["name"]),
                                 dict(args) if isinstance(args, Mapping) else {},
                                 len(out)))
            spans.append(m.span())
    return strip_calls(text, spans), out


DIALECT = ToolDialect("qwen-2.5", "<tool_call>", parse_calls,
                      attempting=("<tool_call>", '"name"'))
