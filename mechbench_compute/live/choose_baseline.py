from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from mechbench_compute.blocks.read_record_key import read_record_key
from mechbench_compute.lexicon._base import Notable
from mechbench_compute.live.say_number import say_number

DECLARED = "declared"
CONTROL = "control"
TRY = "try"
SCRATCH = "~scratch"


@dataclass(frozen=True)
class Baseline:
    label: str
    origin: str
    result: Any = None
    seq: int | None = None
    param: str | None = None
    machine: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "origin": self.origin, "seq": self.seq, "param": self.param}


def choose_baseline(notable: Notable, items: Sequence[Mapping[str, Any]], *,
                    declared: Mapping[str, Any] | None = None, tries: Sequence[Mapping[str, Any]] = (),
                    op: str | None = None, inputs: Mapping[str, Any] | None = None,
                    params: Mapping[str, Any] | None = None) -> Baseline | None:
    if declared is not None:
        name = declared.get("name")
        return Baseline(f"${name}" if name else "the declared baseline", DECLARED, declared.get("result"),
                        declared.get("seq"), None, declared.get("machine"))
    if notable.control and any(is_control(notable, it) for it in items):
        return Baseline(" ".join(f"{f} {say_value(v)}" for f, v in notable.control.items()), CONTROL)
    if op is None:
        return None
    for t in sorted(tries, key=lambda t: int(t.get("seq") or 0)):
        if t.get("result") is None or read_op_name(t.get("op")) != read_op_name(op):
            continue
        if read_signature(t.get("inputs") or {}) != read_signature(inputs or {}):
            continue
        changed = read_changed(t.get("params") or {}, params or {})
        if len(changed) != 1:
            continue
        label = f"${t['name']}" if t.get("name") else f"t{t.get('seq')}"
        return Baseline(label, TRY, t.get("result"), t.get("seq"), changed[0], t.get("machine"))
    return None


def is_control(notable: Notable, item: Mapping[str, Any]) -> bool:
    return all(read_record_key(item, f) == v for f, v in notable.control.items())


def say_value(v: Any) -> str:
    return say_number(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)


def read_op_name(name: Any) -> str:
    from mechbench_compute.registry import REGISTRY

    found = REGISTRY.find(str(name or ""))
    return found.name if found is not None else str(name or "")


def read_signature(values: Mapping[str, Any]) -> dict[str, str]:
    return {str(k): read_reference(v) for k, v in values.items()}


def read_reference(value: Any) -> str:
    if isinstance(value, str) and value.startswith("$") and len(value) > 1:
        return f"name:{value[1:]}"
    if isinstance(value, Mapping) and len(value) == 1:
        if "$name" in value:
            return f"name:{value['$name']}"
        ref = value.get("$ref")
        path = ref.get("bench") if isinstance(ref, Mapping) else ref
        if isinstance(path, str):
            parts = path.split("/")
            return f"name:{parts[2]}" if parts[0] == SCRATCH and len(parts) == 3 else f"ref:{path}"
    return "value:" + json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def read_changed(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[str]:
    a, b = read_signature(before), read_signature(after)
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
