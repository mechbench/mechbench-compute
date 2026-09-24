from __future__ import annotations


def read_tokenizer_id(value):
    return value.base if hasattr(value, "base_kind") else str(value)
