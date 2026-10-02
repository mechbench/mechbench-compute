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

CALL = re.compile(
    r'\{[^{}]*"name"\s*:\s*"[^"]+"[^{}]*"parameters"\s*:\s*\{.*?\}\s*\}', re.DOTALL)


def parse_calls(text: str, tools: Sequence[ToolDef]) -> ParseResult:
    known = {t.name for t in tools}
    out = []
    spans: list[tuple[int, int]] = []
    for m in CALL.finditer(text):
        parsed = parse_json(m.group(0))
        if (isinstance(parsed, Mapping) and parsed.get("name")
                and not (known and parsed["name"] not in known)):
            params = parsed.get("parameters")
            out.append(make_call(str(parsed["name"]),
                                 dict(params) if isinstance(params, Mapping) else {},
                                 len(out)))
            spans.append(m.span())
    return strip_calls(text, spans), out


DIALECT = ToolDialect("llama-3", "<|start_header_id|>ipython", parse_calls, "ipython",
                      attempting=('"name"', '"parameters"'))
