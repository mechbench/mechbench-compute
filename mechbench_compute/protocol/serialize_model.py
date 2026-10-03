from __future__ import annotations

from collections.abc import Mapping


def serialize_model(value, loaded=None):
    wire = value.to_wire() if hasattr(value, "to_wire") else value
    fingerprint = getattr(loaded, "fingerprint", None)
    if fingerprint is None or getattr(value, "is_endpoint", False):
        return wire
    adapters = fingerprint.read_adapters()
    if isinstance(wire, Mapping):
        return {**wire, "adapters": adapters} if adapters or "adapters" in wire else wire
    if adapters and isinstance(wire, str):
        return {"base": {"hf": wire}, "adapters": adapters}
    return wire
