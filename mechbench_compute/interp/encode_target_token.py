from __future__ import annotations

from mechbench_compute.distill import encode


def encode_target_token(model, target: str) -> int:
    flat = encode(model.tokenizer, target)
    specials = set(getattr(model.tokenizer, "all_special_ids", []) or [])
    for t in flat:
        if t not in specials:
            return int(t)
    raise ValueError(f"target {target!r} tokenized to specials only")
