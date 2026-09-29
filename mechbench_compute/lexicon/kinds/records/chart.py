from __future__ import annotations

from mechbench_compute.lexicon._base import Kind
from mechbench_compute.lexicon.values import F

KIND = Kind(
    "records/chart",
    "A figure of a table, as a stored object: a mark, an encoding, the words and landmarks that make it readable, and either a reference to the data or the rows inline.",
    fields={"title": F("string", "The figure's claim, as a sentence."),
            "mark": F("string", "`bar`, `line`, `point`, `heat` or `tokens`."),
            "encoding": F("object", "`{x, y, series?, color?, value?, text?, lo?, hi?}` — which fields go where."),
            "labels": F("object", "`{x?, y?, value?, series?, color?}` — what each field is called in prose."),
            "axes": F("object", "`{layer: {n, global, kv_shared_from}}` — the depth landmarks a layer axis draws."),
            "annotate": F("array", "`[{at, text}]` — callouts drawn at named rows.", items={"type": "object"}),
            "reference": F("array", "`[{y|x, text}]` — lines the marks are read against: a target, a baseline, chance.", items={"type": "object"}),
            "focus": F("string", "The field shared with the other figures on a page."),
            "facet": F("string", "Small multiples: one panel per value of this field, on one shared x axis."),
            "scale": F("string", "`diverging` or `sequential`, when the values do not decide it."),
            "source": F("string", "The stored table the figure draws, when the executor knew it."),
            "data": F("object", "`{rows}` inline, when it did not.")},
    required=("mark", "encoding"),
    doc="What `records/plot` writes and an emit script can build by hand. Beyond the mark, a figure carries "
        "prose `labels`, the model's depth landmarks under `axes.layer`, `annotate` callouts and a `focus` "
        "field — the vocabulary in `mechbench/docs/VISUALIZATION.md`, which is what turns a chart into "
        "somewhere a reader can stand.",
)
