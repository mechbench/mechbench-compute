from __future__ import annotations

from mechbench_compute.lexicon._base import Kind
from mechbench_compute.lexicon.values import F

KIND = Kind(
    "adapter/checkpoint",
    "A merged checkpoint published to the bench or the hub: its files with their hashes, and the stack that was merged.",
    fields={"files": F("array", "`{name, size, sha256}` per file.", items={"type": "object"}),
            "merged_from": F("object", "The model reference merged, in wire form."),
            "base_snapshot": F("string", "The base revision merged onto."),
            "location": F("string", "Where it landed.")},
    required=("files", "merged_from"),
)
