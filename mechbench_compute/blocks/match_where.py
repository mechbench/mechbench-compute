from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from mechbench_compute.blocks.read_field import read_field

OPS = ("!=", "<=", ">=", "=", "<", ">", "~")

_WHERE = re.compile(r"^([A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*)(!=|<=|>=|=|<|>|~)([\s\S]*)$")


@dataclass(frozen=True)
class Condition:
    path: str
    op: str
    value: Any
    raw: str


def parse_where(where: str | Iterable[str] | Mapping[str, Any] | None) -> list[Condition]:
    if where is None:
        return []
    if isinstance(where, Mapping):
        return [Condition(str(k), "=", v, v if isinstance(v, str) else json.dumps(v))
                for k, v in where.items()]
    if isinstance(where, str):
        where = [where]
    out = []
    for w in where:
        if isinstance(w, Mapping):
            out.append(_parse_condition(w))
            continue
        m = _WHERE.match(str(w))
        if not m:
            raise ValueError(
                f"where is PATH OP VALUE, OP one of {' '.join(OPS)} (such as "
                f"coords.prompt=flash), not {w!r}")
        raw = m.group(3)
        try:
            value = json.loads(raw)
        except ValueError:
            value = raw
        out.append(Condition(m.group(1), m.group(2), value, raw))
    return out


def _parse_condition(w: Mapping[str, Any]) -> Condition:
    path, op = w.get("path"), w.get("op", "=")
    if not isinstance(path, str) or not path or op not in OPS:
        raise ValueError(
            f"a condition is {{path, op, value}}, op one of {' '.join(OPS)}, not {dict(w)!r}")
    value = w.get("value")
    return Condition(path, op, value, _read_text(value) if _read_text(value) is not None else json.dumps(value))


def _read_rank(v: Any) -> int:
    if isinstance(v, bool):
        return 2
    if isinstance(v, (int, float)):
        return 0
    if isinstance(v, str):
        return 1
    return -1


def _read_text(v: Any) -> str | None:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    if isinstance(v, (str, int, float)):
        return str(v)
    return None


def _match_equal(have: Any, c: Condition) -> bool:
    if have is None:
        return c.value is None
    if have == c.value and isinstance(have, bool) == isinstance(c.value, bool):
        return True
    return _read_text(have) == c.raw


def match_where(record: Mapping[str, Any], conditions: Iterable[Condition]) -> bool:
    for c in conditions:
        v = read_field(record, c.path)
        if c.op == "=":
            ok = _match_equal(v, c)
        elif c.op == "!=":
            ok = not _match_equal(v, c)
        elif c.op == "~":
            if isinstance(v, str):
                ok = c.raw.lower() in v.lower()
            elif isinstance(v, list):
                ok = any(_match_equal(x, c) for x in v)
            else:
                ok = False
        else:
            rank = _read_rank(v)
            if rank not in (0, 1) or rank != _read_rank(c.value):
                ok = False
            else:
                ok = {"<": v < c.value, "<=": v <= c.value,
                      ">": v > c.value, ">=": v >= c.value}[c.op]
        if not ok:
            return False
    return True
