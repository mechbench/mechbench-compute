from __future__ import annotations


def serialize_params(params):
    """Params as provenance may record them: a normalized ModelRef rides
    through execution as an object, but a fingerprint hashes CANONICAL
    WIRE FORMS — the object serialized back to {base, adapters}, which is
    also what the run declared."""
    return {
        k: (v.to_wire() if hasattr(v, "to_wire") else v)
        for k, v in params.items()
    }
