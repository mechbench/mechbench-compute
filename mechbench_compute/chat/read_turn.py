from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute.providers import messages as pm


def read_turn(text: str, reasoning: Sequence[Mapping[str, Any]],
              turn: Sequence[Mapping[str, Any]] | None) -> tuple[Any, ...]:
    if turn is None:
        return (*(pm.part({"type": "reasoning", **dict(r)}) for r in reasoning),
                *((pm.TextPart(text),) if text else ()))
    out: list[Any] = []
    for entry in turn:
        kind = entry.get("type")
        if kind == "reasoning":
            out.append(pm.part({"type": "reasoning", **dict(reasoning[int(entry["index"])])}))
        elif kind == "text" and "start" in entry:
            out.append(pm.TextPart(text[int(entry["start"]):int(entry["end"])],
                                   signature=pm.Signature.from_wire(entry.get("signature"))))
        else:
            out.append(pm.part(entry))
    return tuple(out)
