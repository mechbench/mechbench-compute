from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mechbench_compute.providers import messages as pm


def write_turn(parts: Sequence[Any]) -> list[dict[str, Any]] | None:
    """The order of a reply's parts, for storage beside its `text` and
    `reasoning`: each reasoning part by its index in `reasoning`, each
    text part by its span of `text` with any signature it carried, each
    tool call whole. None when the reply is prose and tool calls alone,
    whose order a reader never needs.

    Nothing is stored twice: the prose lives in `text`, the reasoning
    and its provider payload in `reasoning`, and `read_turn` puts the
    turn back together exactly as the provider returned it.
    """
    if not any(isinstance(p, pm.ReasoningPart) or getattr(p, "signature", None)
               for p in parts):
        return None
    turn: list[dict[str, Any]] = []
    at = 0
    n_reasoning = 0
    for p in parts:
        if isinstance(p, pm.ReasoningPart):
            turn.append({"type": "reasoning", "index": n_reasoning})
            n_reasoning += 1
        elif isinstance(p, pm.TextPart):
            entry: dict[str, Any] = {"type": "text", "start": at, "end": at + len(p.text)}
            if p.signature is not None:
                entry["signature"] = p.signature.to_wire()
            turn.append(entry)
            at += len(p.text)
        else:
            turn.append(p.to_wire())
    return turn
