from __future__ import annotations

from typing import Any


def read_parameters(lm: Any) -> dict[str, Any]:
    out = dict(lm.model.state_dict(keep_vars=True))
    head = getattr(getattr(lm, "lm_head", None), "weight", None)
    if head is not None and not any(head is p for p in out.values()):
        out["lm_head.weight"] = head
    return out
