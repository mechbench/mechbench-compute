from __future__ import annotations

from mechbench_compute.distill import encode


def _target_token_id(model, target: str) -> int:
    """The first token of `target`, tokenized raw as a continuation."""
    flat = encode(model.tokenizer, target)
    # skip BOS-like specials the tokenizer prepends
    specials = set(getattr(model.tokenizer, "all_special_ids", []) or [])
    for t in flat:
        if t not in specials:
            return int(t)
    raise ValueError(f"target {target!r} tokenized to specials only")
