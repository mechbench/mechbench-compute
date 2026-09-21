from __future__ import annotations


def _wire_model(value):
    """What a RESULT may record about its model: the wire form, never
    the resolved object (000488).

    A resolved ModelRef carries `adapter_payloads` — the fetched
    safetensors bytes — and a record that embeds the object embeds
    those bytes. The generate block did exactly that, in two fields
    per item plus a `str()` of the object in a third, at ~32 MB per
    item against the 4.5 KB a base-model item weighs; a 20-story
    result was a 640 MB body and killed the API process on arrival.
    A bare string (an HF id) passes through unchanged.
    """
    return value.to_wire() if hasattr(value, "to_wire") else value
