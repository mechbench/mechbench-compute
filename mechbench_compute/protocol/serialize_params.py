from __future__ import annotations


def serialize_params(params):
    return {
        k: (v.to_wire() if hasattr(v, "to_wire") else v)
        for k, v in params.items()
    }
