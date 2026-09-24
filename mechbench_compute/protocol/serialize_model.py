from __future__ import annotations


def serialize_model(value):
    return value.to_wire() if hasattr(value, "to_wire") else value
