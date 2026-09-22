from __future__ import annotations


def read_tokenizer_id(value):
    """The tokenizer a trace was cut with, as an id. For a ModelRef
    that is its base — adapters do not change the vocabulary."""
    return value.base if hasattr(value, "base_kind") else str(value)
