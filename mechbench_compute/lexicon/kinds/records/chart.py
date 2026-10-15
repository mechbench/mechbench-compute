from __future__ import annotations

from mechbench_compute.lexicon._base import Kind
from mechbench_compute.lexicon.values import F

KIND = Kind(
    "records/chart",
    "A figure of a table, as a stored object: a mark, an encoding, the words and landmarks that make it readable, and either a reference to the data or the rows inline.",
    fields={"title": F("string", "The figure's claim, as a sentence."),
            "mark": F("string", "`<name>@<version>`: `bar@1`, `line@1`, `point@1`, `heat@1`, `tokens@1` or `graph@1`, or an extension's `<owner>/<project>/marks/<name>@<version>`. Older figures carry the bare name, `scatter` (`point@1`) or `histogram` (`bar@1` with 20 bins)."),
            "encoding": F("object", "`{x, y, series?, color?, value?, text?, lo?, hi?}` — which fields go where; on a graph `{components, links?, id?, layer?, weight?, label?, text?, sign?}`."),
            "labels": F("object", "`{x?, y?, value?, series?, color?, weight?}` — what each field is called in prose."),
            "axes": F("object", "`{layer?: {n, global, kv_shared_from}, x?, y?}` — the depth landmarks a layer axis draws, and a numeric axis's `{scale, constant, quantiles, zero}`."),
            "annotate": F("array", "`[{at, text}]` — callouts drawn at named rows.", items={"type": "object"}),
            "reference": F("array", "`[{y|x, text}]` — lines the marks are read against: a target, a baseline, chance.", items={"type": "object"}),
            "focus": F("string", "The field shared with the other figures on a page."),
            "facet": F("string", "Small multiples: one panel per value of this field, on one shared x axis."),
            "scale": F(["string", "object"], "How a value becomes colour: `{ramp?, per?}`, or the ramp alone, `diverging` or `sequential`; `per` is `row` or `figure`."),
            "bin": F("integer", "A bar mark's bins: the renderer counts the rows into this many bins of x and draws the counts."),
            "level": F("number", "The level of the `lo`/`hi` interval, 0.95 for 95%."),
            "threshold": F("number", "A graph's links under this |weight| are not drawn; the rows are not changed."),
            "layout": F("string", "A graph's layers as `columns` (the default) or `rows`."),
            "order": F("string", "A graph's components along their layer: by `id` or by `weight`."),
            "source": F("string", "The stored table the figure draws, when the executor knew it."),
            "data": F("object", "`{rows}` inline, when it did not.")},
    required=("mark", "encoding"),
    doc="What `records/plot` writes and an emit script can build by hand. Beyond the mark, a figure carries "
        "prose `labels`, the model's depth landmarks under `axes.layer`, `annotate` callouts and a `focus` "
        "field — the vocabulary in `mechbench/docs/VISUALIZATION.md`, which is what turns a chart into "
        "somewhere a reader can stand.",
)
