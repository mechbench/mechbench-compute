from __future__ import annotations

from mechbench_compute.lexicon._base import Kind
from mechbench_compute.lexicon.values import F

KIND = Kind(
    "text/tokenization",
    "How a tokenizer splits a set of items as continuations of a prefix: depth histogram, fragmentation, script composition, and an optional gate.",
    fields={
        "tokenizer": F("string", "Which tokenizer measured."),
        "prefix": F("string", "The prefix the items were tokenized after."),
        "n_items": F("integer", "How many items."),
        "mean_depth": F("number", "Mean tokens per item after the prefix."),
        "min_depth": F("integer", "Fewest tokens any item took."),
        "max_depth": F("integer", "Most tokens any item took."),
        "single_token_fraction": F("number", "Share of items that are one token."),
        "mean_tokens_per_word": F("number", "Fragmentation against whitespace words."),
        "fragmented_fraction": F("number", "Share of items with more tokens than words."),
        "rows": F("array", "The depth histogram: `{depth, count, share}`.", items={"type": "object"}),
        "script_composition": F("object", "Unicode script → share of characters."),
        "boundary_failures": F("array", "Items that changed the prefix's own tokenization.", items={"type": "string"}),
        "gate": F("object", "`{expected_depth, pass, n_violations, violations}`, or null."),
        "most_fragmented": F("array", "The most fragmented items, with their pieces.", items={"type": "object"}),
        "items": F("array", "Every item's own measurement, when kept.", items={"type": "object"}),
    },
    required=("tokenizer", "n_items", "mean_depth", "rows"),
    renderer={"primitive": "table", "field_map": {"rows": "rows"}},
    doc="The measurement to take before a decision read: a set of outcomes the tokenizer splits into several "
        "pieces cannot be compared at one token, and a prefix whose own tokenization changes when an item "
        "follows it moves the decision point. The `gate`, when asked for, says whether every item met the "
        "expected depth, and names the ones that did not.",
)
