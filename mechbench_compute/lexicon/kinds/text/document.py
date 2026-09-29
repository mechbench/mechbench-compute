from __future__ import annotations

from mechbench_compute.lexicon._base import Kind
from mechbench_compute.lexicon.values import F

KIND = Kind(
    "text/document",
    "One generated or collected text, with the trace of how it was made when kept at trace fidelity. A record whose text is in `text`, so every op that reads records reads documents.",
    fields={
        "text": F("string", "The text.", **{"x-mechbench-text": True}),
        "metadata": F("object", "`sampling`, the model's wire form, tool runs — what the producing op recorded; documents made before 2026-09 keep their `coords` here too."),
        "trace": F("object", "At trace fidelity: `token_ids`, `text`, `offsets`, `generation_spans`."),
        "segmentations": F("array", "Named spans over the trace: which tokens are prompt, which are body.", items={"type": "object"}),
    },
    required=("id", "text"),
    extends="records/record",
    key=("id",),
    header={"name": "A label for the collection.", "description": "Free text beside the name.",
            "fidelity": "`text`, `segments` or `trace`: how much of each document was kept.",
            "summary": "For a remote run: calls, cost, cache hits.", "spend": "What the run bought from providers."},
    renderer={"primitive": "text", "field_map": {"text": "text"}},
    doc="What `text/generate` and `text/chat` write, one per completion. The collection's `fidelity` says how "
        "much was kept: `text` alone, `segments` (which spans are prompt and which are body), or `trace` (the "
        "token ids and offsets, which `text/score` and a positions trajectory need). A document is a record, so a "
        "corpus flows into `text/measure`, `records/filter` and `activations/capture` unchanged.",
)
