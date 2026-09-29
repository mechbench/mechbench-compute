from __future__ import annotations

from mechbench_compute.lexicon._base import Kind
from mechbench_compute.lexicon.values import F

KIND = Kind(
    "activations/grid",
    "A scalar field over model axes for one record: declared axes, named measures indexed in axis order, and the tokens when an axis is position.",
    extends="records/record",
    fields={"axes": F("array", "The axes, in value-index order: `layer`, `position`, `head`, `query`, `key`, `component`.", items={"type": "string"}),
            "measures": F("object", "Measure name → values, a nested list indexed in `axes` order."),
            "tokens": F("array", "The prompt's tokens, when an axis is `position`.", items={"type": "string"}),
            "error": F("string", "Why the record could not be measured, when it could not; then `measures` is empty.")},
    required=("id", "axes", "measures"),
    key=("id",),
    renderer={"primitive": "table", "field_map": {"rows": "items"}},
    doc="The ancestor of every map an op draws over the model: a lens, an attribution, a divergence, an "
        "attention pattern, a trace, a head-ablation grid. `axes` says what the nested `measures` are indexed "
        "by, in order, so a reader (or a chart) knows that `measures.logprob[3][7]` is layer 3, position 7. A "
        "record that could not be measured carries `error` and empty measures rather than being dropped, so "
        "a grid collection has one item per input record.",
)
