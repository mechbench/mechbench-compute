from __future__ import annotations

from mechbench_compute.lexicon._base import Kind
from mechbench_compute.lexicon.values import F

KIND = Kind(
    "intervene/spec",
    "An intervention as an object: the spec items — where in the forward pass to act, at which layers and positions, and what to do there — declared once and wired to whichever op runs the forward pass.",
    fields={"items": F("array", "The spec items, in the grammar `intervene/apply` documents under *Spec items*.", items={"type": "object"}),
            "description": F("string", "What the intervention is for, in a sentence.")},
    required=("items",),
    doc="The same object wires to `intervene/apply` (read the next-token distribution or a capture), to "
        "`text/generate` and `text/chat` (read what the model then writes), and to any op that takes an "
        "`intervention` port — one declared intervention, several readouts. Directions and sources "
        "an item needs arrive on the consuming node's `direction` and `source` ports, which fill any item "
        "that names none; a stored spec carries no vectors of its own. Stored on the bench like a corpus, "
        "by hand or from a protocol's output.",
)
