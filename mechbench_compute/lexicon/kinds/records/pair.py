from __future__ import annotations

from mechbench_compute.lexicon._base import Kind
from mechbench_compute.lexicon.values import F

KIND = Kind(
    "records/pair",
    "Two prompts that differ in one place, for the ops that run both and compare.",
    extends="records/record",
    fields={"a": F("string", "The first prompt (the clean one, for tracing)."),
            "b": F("string", "The second prompt (the corrupted one, for tracing).")},
    required=("id", "a", "b"),
    key=("id",),
    doc="Written before the typology as `clean`/`corrupt`; those names are still read.",
)
