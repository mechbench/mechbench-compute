from __future__ import annotations


def serialize_model(value):
    """What a RESULT may record about its model: the wire form, never
    the resolved object.

    A resolved ModelRef carries `adapter_payloads` — the fetched
    safetensors bytes — so a record that embeds the object embeds those
    bytes, tens of megabytes per item against the kilobytes an item
    otherwise weighs. A bare string (an HF id) passes through unchanged.
    """
    return value.to_wire() if hasattr(value, "to_wire") else value
