from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.blocks.expand_cells import expand_cells
from mechbench_compute.blocks.read_items import read_items
from mechbench_compute.lexicon._base import In, Op, Output, P

OP = Op(
    name="records/plot",
    summary=(
        "Describe a chart of an upstream table as a stored object — what to "
        "plot on which axes — so it renders beside the data and re-renders "
        "when the data changes."
    ),
    description="""\
The spec names a mark and an encoding. When the executor knows the input's
stored location, the spec references it (`source`) and the chart is drawn
from the live data; otherwise the rows ride inline under `data.rows` and the
spec is self-contained. Coordinates are flattened into each row so they can
be encoded directly.

### The two marks interp actually needs

Half of this platform's figures are grids: (layer, position) from
`intervene/patch`, (layer, head) from `intervene/ablate-heads` or
`intervene/path`, (layer, prompt) from `logits/attribute`. That is
`mark: "heat"` with `encoding: {x, y, value}` — one cell per row, the
value its colour, `scale: "diverging"` centred on zero where the number
is a change and `"sequential"` where it is a magnitude.

The other half are token strips: a prompt's own tokens, each coloured by
a number it carries — a per-token surprisal from
`activations/capture-tokens`, a probe's projection, the window
`activations/examples` brings back. That is `mark: "tokens"` with
`encoding: {text, value}`, where `text` names the field holding the
row's tokens. Rows that are one token each — a grid's cells, or a
`records/summarize` over them by position — make a strip too: `text`
names the token field, `x` the position (default `position`), and
`facet` (or `series`) says which rows are one prompt, one strip per
value on one colour scale.

An `lo`/`hi` encoding draws the interval `records/summarize` reports
beside the point it belongs to.

### What makes it a visualization

A figure on this platform is meant to lend a reader spatial intuitions
for a space they have none for, and it carries four things beyond the
mark to do it (the vocabulary is `mechbench/docs/VISUALIZATION.md`):

* **`labels`** — what each field is called in prose. The axes are
  labelled with these words and the hover readout is a sentence built
  from them: "Layer 23, attention only: −3.10 in log probability".
* **`axes.layer`** — the model's depth landmarks: how many layers,
  which attend globally, and where fresh keys and values stop. With
  them, the depth axis marks the global-attention layers and the
  key/value boundary, so a layer is the same place in every figure. A
  sweep's result carries them under `arch`, and a summary or contrast
  of it carries them forward; this op reads them from its input when
  `axes` is not given.
* **`annotate`** — callouts drawn on the figure at named rows. The
  extremes are labelled by default; this names what else to say.
* **`focus`** — the field this figure shares with the others on a page:
  hover a layer here and it lights on every figure that has one.
  Defaults to `layer` when the rows carry it, else `x`.
* **`facet`** — small multiples: one panel per value of the field, on
  one shared x axis, each with its own value scale. A sweep whose
  largest series would flatten the others is four panels, not four
  lines.

`encoding.color` tints each mark by a categorical field **in place**;
`encoding.series` splits rows into several lines or side-by-side bars.
The two are different and may be combined.
""",
    inputs=(
        In("records", "collection | records/table",
           "The table, or any collection of items, to chart.", many=True),
    ),
    output=Output('records/chart', collection=False, doc='`title`, `mark`, `encoding` (`x`, `y`, `series`, `color`, `value`, `text`, `lo`, `hi` as the mark uses them), `scale` when given, `labels`, `axes`, `annotate`, `focus` and `facet` when given, and `source` or `data`.'),
    params=(
        P("encoding", "object",
          "Which field goes where. `x`/`y` for the point-shaped marks, "
          "`series` to split into series, `value` for a heat cell or a "
          "token's colour, `text` for the token strip's tokens, `lo`/`hi` "
          "for an interval.",
          None, fields=(
              P("x", "string", "The field on the x axis.", None),
              P("y", "string", "The field on the y axis.", None),
              P("series", "string", "The field that splits the rows into series.", None),
              P("color", "string",
                "The categorical field each mark takes its colour from, in place.", None),
              P("value", "string", "The number a heat cell or a token takes its colour from.", None),
              P("text", "string", "The field holding a row's tokens, for a token strip.", None),
              P("lo", "string", "The interval's lower end.", None),
              P("hi", "string", "The interval's upper end.", None),
          )),
        P("x", "string", "The x field; the same as `encoding.x`.", None),
        P("y", "string", "The y field; the same as `encoding.y`.", None),
        P("value", "string", "The value field; the same as `encoding.value`.", None),
        P("text", "string", "The tokens field; the same as `encoding.text`.", None),
        P("lo", "string", "The lower end; the same as `encoding.lo`.", None),
        P("hi", "string", "The upper end; the same as `encoding.hi`.", None),
        P("mark", "string",
          "`\"bar\"`, `\"line\"`, `\"point\"`, `\"heat\"` (a grid of cells) "
          "or `\"tokens\"` (a prompt's tokens, coloured).",
          "bar", choices=("bar", "line", "point", "heat", "tokens")),
        P("scale", "string",
          "How `value` becomes colour: `\"diverging\"` centres on zero (a "
          "recovery, a Δ log p), `\"sequential\"` runs from the lowest "
          "value. Diverging when the values cross zero, by default.",
          None, choices=("diverging", "sequential")),
        P("title", "string", "The chart's title: its claim, as a sentence.", ""),
        P("labels", "object",
          "What each encoded field is called in prose — the axis labels "
          "and the words the hover readout uses.",
          None, fields=(
              P("x", "string", "What the x field is called.", None),
              P("y", "string", "What the y field is called.", None),
              P("value", "string", "What the value field is called.", None),
              P("series", "string", "What the series field is called.", None),
              P("color", "string", "What the colour field is called.", None),
          )),
        P("axes", "object",
          "Landmarks for an axis. `layer` carries the model's depth: "
          "`n` layers, the `global` attention layers, and "
          "`kv_shared_from`, the first layer that reuses keys and values. "
          "Read from the input's `arch` header when not given.",
          None, fields=(
              P("layer", "object", "The depth landmarks.", None, fields=(
                  P("n", "int", "How many layers the model has.", None),
                  P("global", "list[int]", "The layers that attend to the whole prompt.", None),
                  P("kv_shared_from", "int",
                    "The first layer that reuses earlier keys and values.", None),
              )),
          )),
        P("annotate", "list[object]",
          "Callouts drawn on the figure: each names the row it sits at "
          "(`at`, a field → value map) and what it says.",
          None, fields=(
              P("at", "map[string, json]", "The row: field → value.", None),
              P("text", "string", "What the callout says.", None),
          )),
        P("reference", "list[object]",
          "Lines the marks are read against: a target, a baseline, a "
          "threshold, chance. Each names a value on one axis — `y` for a "
          "rule across the plot, `x` for one down it — and what it is. "
          "A figure whose claim is \"close to fair\" or \"above the "
          "threshold\" cannot make it without one.",
          None, fields=(
              P("y", "float", "The value on the y axis to rule at.", None),
              P("x", "json", "The value on the x axis to rule at.", None),
              P("text", "string", "What the line is: `a fair die`, `chance`.", None),
          )),
        P("focus", "string",
          "The field shared with the other figures on a page, so a hover "
          "here lights the same value there. `layer` when the rows carry "
          "one, else the x field.",
          None),
        P("facet", "string",
          "Small multiples: one panel per value of this field, stacked on "
          "one shared x axis, each with its own value scale — four sweeps "
          "as four aligned panels rather than four lines on one scale.",
          None),
    ),
    example={
        "title": "Removing only the attention at each layer",
        "mark": "point",
        "encoding": {"x": "layer", "y": "mean", "lo": "lo", "hi": "hi",
                     "color": "attention"},
        "labels": {"y": "change in the answer's log probability",
                   "color": "attention kind"},
        "annotate": [{"at": {"layer": 23}, "text": "the last global layer with fresh keys and values"}],
    },
    example_inputs={"records": {"$ref": {"bench": "you/lab/table"}}},
)


#: What a chart may be drawn as. `heat` is a grid of cells — (layer,
#: position) from a trace, (layer, head) from a head sweep — and
#: `tokens` is a prompt's own tokens coloured by a number each carries.
MARKS = ("bar", "line", "point", "heat", "tokens")


#: How a value becomes colour. Diverging is centred on zero, which is how
#: a change reads; sequential runs from the lowest value, which is how a
#: magnitude reads. The default is chosen by whether the values cross zero.
SCALES = ("diverging", "sequential")


#: What a figure may call its fields in prose: the axis labels and the
#: readout's words (`mechbench/docs/VISUALIZATION.md`).
LABEL_FIELDS = ("x", "y", "value", "series", "color")


def run(ctx, inputs, params):
    # A viz references its upstream by LABEL when the executor knows it
    # (lineage-true, renders live).
    return build_chart(
        inputs.get("records"), params,
        source_label=ctx.input_paths.get("records") or None)


def _read_layer_axis(header: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """`axes.layer` as a figure carries it, from a header's `arch`."""
    arch = (header or {}).get("arch") if isinstance(header, Mapping) else None
    if not isinstance(arch, Mapping) or "n_layers" not in arch:
        return None
    out: dict[str, Any] = {"n": int(arch["n_layers"])}
    if isinstance(arch.get("global_layers"), list):
        out["global"] = [int(i) for i in arch["global_layers"]]
    if arch.get("first_kv_shared_layer") is not None:
        out["kv_shared_from"] = int(arch["first_kv_shared_layer"])
    return out


def _check_layer_axis(axes: Any) -> dict[str, Any]:
    """`axes` as given, its `layer` entry checked: `n` is required and
    every landmark must be inside it, because a landmark drawn off the
    axis is a lie about the model."""
    if not isinstance(axes, Mapping):
        raise ValueError("records/plot axes is an object: {\"layer\": {n, global, kv_shared_from}}")
    out = dict(axes)
    layer = out.get("layer")
    if layer is None:
        return out
    if not isinstance(layer, Mapping) or "n" not in layer:
        raise ValueError("records/plot axes.layer needs `n`, the number of layers")
    n = int(layer["n"])
    checked: dict[str, Any] = {"n": n}
    if layer.get("global") is not None:
        globals_ = [int(i) for i in layer["global"]]
        bad = [i for i in globals_ if not 0 <= i < n]
        if bad:
            raise ValueError(f"records/plot axes.layer.global names layers outside 0..{n - 1}: {bad}")
        checked["global"] = globals_
    if layer.get("kv_shared_from") is not None:
        k = int(layer["kv_shared_from"])
        if not 0 <= k <= n:
            raise ValueError(f"records/plot axes.layer.kv_shared_from is outside 0..{n}: {k}")
        checked["kv_shared_from"] = k
    out["layer"] = checked
    return out


def _check_annotations(annotate: Any) -> list[dict[str, Any]]:
    """Each annotation names where it sits (`at`, a field → value map)
    and what it says (`text`)."""
    if not isinstance(annotate, (list, tuple)):
        raise ValueError("records/plot annotate is a list of {at, text}")
    out = []
    for i, a in enumerate(annotate):
        if not isinstance(a, Mapping) or not isinstance(a.get("at"), Mapping) or not a.get("text"):
            raise ValueError(f"records/plot annotate[{i}] needs `at` (a field → value map) and `text`")
        out.append({"at": dict(a["at"]), "text": str(a["text"])})
    return out


def _check_references(reference: Any) -> list[dict[str, Any]]:
    """Each reference line names a value on one axis and what it is: a
    target, a baseline, a threshold, chance."""
    if not isinstance(reference, (list, tuple)):
        raise ValueError("records/plot reference is a list of {y|x, text}")
    out = []
    for i, r in enumerate(reference):
        if not isinstance(r, Mapping) or ("y" not in r and "x" not in r):
            raise ValueError(
                f"records/plot reference[{i}] needs `y` (a rule across the "
                f"plot) or `x` (one down it)")
        line: dict[str, Any] = {}
        if "y" in r:
            line["y"] = float(r["y"])
        if "x" in r:
            line["x"] = r["x"]
        if r.get("text"):
            line["text"] = str(r["text"])
        out.append(line)
    return out


def build_chart(records: Any, params: Mapping[str, Any],
               source_label: str | None = None) -> dict[str, Any]:
    """A chart as a bench object: how to present an upstream table, stored
    beside it rather than drawn once and thrown away.

    When the executor knows the input's label the spec REFERENCES it
    (`source`, lineage-true, so the chart re-renders as the data
    changes); otherwise the rows ride inline (`data.rows`) and the viz
    stays self-contained. Renamed from `viz_spec` by task 000309 — "viz"
    is the level of abstraction the primitive targets.

    Beyond the mark and the encoding, a figure carries what makes it a
    visualization rather than a chart (VISUALIZATION.md): prose
    `labels` for its fields, the depth landmarks under `axes.layer`
    (given, or read from the input's `arch` header), `annotate`
    callouts, and the `focus` field it shares with the other figures on
    a page.
    """
    enc = params.get("encoding") or {}
    x = enc.get("x") or params.get("x")
    y = enc.get("y") or params.get("y")
    mark = params.get("mark", "bar")
    if mark not in MARKS:
        raise ValueError(
            f"records/plot mark must be one of {', '.join(MARKS)}, not {mark!r}")
    # A heat mark needs a third field — the cell's value — and a token
    # strip needs the tokens and the number that colours them; neither
    # is an x/y pair (000616).
    value = enc.get("value") or params.get("value")
    text = enc.get("text") or params.get("text")
    if mark == "heat" and not (x and y and value):
        raise ValueError("a heat mark needs encoding.x, encoding.y and encoding.value")
    if mark == "tokens":
        if not (text and value):
            raise ValueError("a tokens mark needs encoding.text and encoding.value")
    elif not (x and y):
        raise ValueError("viz/spec needs encoding.x and encoding.y")
    encoding: dict[str, Any] = {}
    for name, field in (("x", x), ("y", y), ("series", enc.get("series")),
                        ("color", enc.get("color")),
                        ("value", value), ("text", text),
                        ("lo", enc.get("lo") or params.get("lo")),
                        ("hi", enc.get("hi") or params.get("hi"))):
        if field:
            encoding[name] = field
    scale = params.get("scale")
    if scale is not None and str(scale) not in SCALES:
        raise ValueError(
            f"records/plot scale must be one of {', '.join(SCALES)}, not {scale!r}")
    labels_in = params.get("labels") or {}
    if not isinstance(labels_in, Mapping):
        raise ValueError("records/plot labels is an object: {x, y, value, series, color}")
    unknown = sorted(set(labels_in) - set(LABEL_FIELDS))
    if unknown:
        raise ValueError(
            f"records/plot labels names {unknown}; it labels {', '.join(LABEL_FIELDS)}")
    labels = {k: str(v) for k, v in labels_in.items() if v}
    header = records if isinstance(records, Mapping) else None
    axes = (_check_layer_axis(params["axes"]) if params.get("axes") is not None
            else ({"layer": la} if (la := _read_layer_axis(header)) else None))
    annotate = (_check_annotations(params["annotate"])
                if params.get("annotate") is not None else None)
    reference = (_check_references(params["reference"])
                 if params.get("reference") is not None else None)
    focus = params.get("focus")
    facet = params.get("facet")
    spec: dict[str, Any] = {
        "kind": "records/chart",
        "title": params.get("title", ""),
        "mark": mark,
        "encoding": encoding,
        **({"scale": str(scale)} if scale else {}),
        **({"labels": labels} if labels else {}),
        **({"axes": axes} if axes else {}),
        **({"annotate": annotate} if annotate else {}),
        **({"reference": reference} if reference else {}),
        **({"focus": str(focus)} if focus else {}),
        **({"facet": str(facet)} if facet else {}),
    }
    if source_label:
        spec["source"] = source_label
    else:
        recs = (records["rows"] if isinstance(records, Mapping)
                and isinstance(records.get("rows"), list) else read_items(records))
        rows = []
        for r in expand_cells(recs):
            row = {k: v for k, v in r.items() if k != "coords"}
            row.update(r.get("coords", {}) if isinstance(r, Mapping) else {})
            rows.append(row)
        spec["data"] = {"rows": rows}
    return spec
