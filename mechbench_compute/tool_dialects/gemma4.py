from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from mechbench_compute.dialects import (
    ParseResult,
    ToolDialect,
    make_call,
    parse_scalar,
    strip_calls,
)
from mechbench_compute.tools import ToolDef

CALL = re.compile(
    r"<\|tool_call\|?>\s*call:\s*([A-Za-z_][\w.]*)\s*\{(.*?)\}\s*<\/?tool_call\|?>",
    re.DOTALL)
QUOTED_ARG = re.compile(r'([A-Za-z_][\w.]*)\s*:\s*<\|"\|>(.*?)<\|"\|>', re.DOTALL)
BARE_ARG = re.compile(r'([A-Za-z_][\w.]*)\s*:\s*([^,{}]+)')


def parse_calls(text: str, tools: Sequence[ToolDef]) -> ParseResult:
    known = {t.name for t in tools}
    out = []
    spans: list[tuple[int, int]] = []
    for m in CALL.finditer(text):
        if known and m.group(1) not in known:
            continue
        body = m.group(2) or ""
        args: dict[str, Any] = {k: v for k, v in QUOTED_ARG.findall(body)}
        for k, v in BARE_ARG.findall(body):
            if k not in args and '<|"|>' not in v:
                args[k] = parse_scalar(v.strip())
        out.append(make_call(m.group(1), args, len(out)))
        spans.append(m.span())
    return strip_calls(text, spans), out


DIALECT = ToolDialect("gemma-4", "<|tool_call>", parse_calls,
                      attempting=("<|tool_call", "call:"))
