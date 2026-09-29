from __future__ import annotations

from mechbench_compute.lexicon._base import Kind
from mechbench_compute.lexicon.values import F

KIND = Kind(
    "text/word-list",
    "A list of words — for the generators that sample from one, the tokenizer measures, and, with `weights`, a word-frequency table.",
    fields={"words": F("array", "The words.", items={"type": "string"}),
            "weights": F("object", "Word → count or weight, when the list is a frequency table; its keys are the words."),
            "language": F("string", "The list's language, when known."),
            "description": F("string", "Where the list came from.")},
)
