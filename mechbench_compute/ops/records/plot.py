from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mechbench_compute.blocks.expand_cells import expand_cells
from mechbench_compute.blocks.read_items import read_items
from mechbench_compute.lexicon._base import In, Op, Output, P, Resume

OP = Op(
    name="records/plot",
    resume=Resume("restart"),
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
`records/group` over them by position — make a strip too: `text`
names the token field, `x` the position (default `position`), and
`facet` (or `series`) says which rows are one prompt, one strip per
value on one colour scale.

An `lo`/`hi` encoding draws the interval a `records/group` aggregate reports
beside the point it belongs to; `level` says which (0.9 for 90%), and is
read from the input's `interval` header when the input carries one.
`bin` makes a bar chart a histogram: the renderer counts the rows into
that many bins of `x` and draws the counts, so `y` is not encoded.

### What each mark reads

The marks, their channels and the settings each draws are declared once,
in `marks.generated.json` beside this file (a copy of mechbench-viz's).
A channel a mark requires and is not given, a channel it does not draw,
and a setting it does not draw are refused, naming the mark: "a heat
mark needs encoding.y and encoding.value.", "a heat mark does not draw
encoding.series; it reads encoding.x, encoding.y and encoding.value.",
"a tokens mark does not draw annotate.". The chart's `mark` is written
as `<name>@<version>` (`heat@1`); `scatter` is read as `point@1` and
`histogram` as `bar@1` with 20 bins.

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
  `axes` is not given and `layer` is on a channel the mark draws as
  depth (`x` for bar, line, point and heat; none for tokens).
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
    output=Output('records/chart', collection=False, doc='`title`, `mark` as `<name>@<version>` (`bar@1`), `encoding` (`x`, `y`, `series`, `color`, `value`, `text`, `lo`, `hi` as the mark declares them), `scale`, `labels`, `axes`, `annotate`, `reference`, `focus`, `facet`, `bin` and `level` when given (`level` also when the input carries an `interval`), and `source` or `data`. `data.rows` are the input\'s items with their coordinates flattened in; the row fields the renderer reads beside the encoded ones are `id`, `note`, `<x>_token`, and on a heat mark `<y>_token` and `token`.'),
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
          "or `\"tokens\"` (a prompt's tokens, coloured); `name@1` is the same.",
          "bar", choices=("bar", "line", "point", "heat", "tokens")),
        P("bin", "int",
          "A bar mark only: count the rows into this many bins of `x` and "
          "draw the counts, a histogram. `y` is then the count and is not "
          "encoded; the renderer does the counting.",
          None),
        P("level", "float",
          "The level of the `lo`/`hi` interval (0.95 for 95%), said in the "
          "readout. Read from the input's `interval` header when not given.",
          None),
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
          "Read from the input's `arch` header when not given and the x or y "
          "field is `layer`.",
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


MARKS = ("bar", "line", "point", "heat", "tokens")


SCALES = ("diverging", "sequential")


DECLARED = json.loads((Path(__file__).parent / "marks.generated.json").read_text())

BY_NAME = {m["name"]: m for m in DECLARED["marks"]}

LEGACY = DECLARED["legacy"]

CHANNELS = tuple(dict.fromkeys(c for m in DECLARED["marks"] for c in m["channels"]))

SETTINGS = tuple(dict.fromkeys(o for m in DECLARED["marks"] for o in m["options"]))

LABEL_FIELDS = ("x", "y", "value", "series", "color")
DEPTH_FIELD = "layer"


def run(ctx, inputs, params):
    return build_chart(
        inputs.get("records"), params,
        source_label=ctx.input_paths.get("records") or None)


def _read_layer_axis(header: Mapping[str, Any] | None) -> dict[str, Any] | None:
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
    if not isinstance(annotate, (list, tuple)):
        raise ValueError("records/plot annotate is a list of {at, text}")
    out = []
    for i, a in enumerate(annotate):
        if not isinstance(a, Mapping) or not isinstance(a.get("at"), Mapping) or not a.get("text"):
            raise ValueError(f"records/plot annotate[{i}] needs `at` (a field → value map) and `text`")
        out.append({"at": dict(a["at"]), "text": str(a["text"])})
    return out


def _check_references(reference: Any) -> list[dict[str, Any]]:
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


def list_words(names: list[str]) -> str:
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def read_mark(spelled: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    name = str(spelled)
    legacy = LEGACY.get(name)
    if legacy is not None:
        name = legacy["mark"]
    bare, _, version = name.partition("@")
    decl = BY_NAME.get(bare)
    if decl is None or (version and version != str(decl["version"])):
        core = list_words([BY_NAME[m]["address"] for m in MARKS])
        raise ValueError(f"records/plot mark {spelled!r} is not a mark: the core marks are {core} "
                         "(name or name@version)")
    return decl, {k: v for k, v in (legacy or {}).items() if k != "mark"}


def is_set(value: Any) -> bool:
    return value is not None and value != "" and not (isinstance(value, (list, tuple, dict)) and not value)


def refuse_undeclared(decl: dict[str, Any], encoding: Mapping[str, str], settings: list[str]) -> None:
    a = f"a {decl['name']} mark"
    declared = decl["options"]
    supplied = {c for o in settings if o in declared for c in declared[o].get("supplies", ())}
    problems = []
    missing = [f"encoding.{c}" for c, need in decl["channels"].items()
               if need == "required" and c not in supplied and c not in encoding]
    if missing:
        problems.append(f"{a} needs {list_words(missing)}.")
    undeclared = [c for c in encoding if c not in decl["channels"]]
    if undeclared:
        problems.append(f"{a} does not draw {list_words([f'encoding.{c}' for c in undeclared])}; "
                        f"it reads {list_words([f'encoding.{c}' for c in decl['channels']])}.")
    for c in encoding:
        if c in supplied:
            by = next(o for o in settings if c in declared.get(o, {}).get("supplies", ()))
            problems.append(f"{a} with {by} draws encoding.{c} itself; the field given for it is not read.")
    unread = [o for o in settings if o != "title" and o not in declared]
    if unread:
        problems.append(f"{a} does not draw {list_words(unread)}.")
    if problems:
        raise ValueError(" ".join(problems))


def read_level(params: Mapping[str, Any], header: Mapping[str, Any] | None) -> float | None:
    level = params.get("level")
    if level is None:
        interval = (header or {}).get("interval")
        level = interval.get("level") if isinstance(interval, Mapping) else interval
    if level is None:
        return None
    level = float(level)
    if not 0 < level < 1:
        raise ValueError(f"records/plot level is the interval's level, between 0 and 1 (0.95), not {level!r}")
    return level


def read_bin(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value) or value < 1:
        raise ValueError(f"records/plot bin is how many bins to count x into, a whole number from 1, not {value!r}")
    return int(value)


def build_chart(records: Any, params: Mapping[str, Any],
               source_label: str | None = None) -> dict[str, Any]:
    decl, legacy = read_mark(params.get("mark", "bar"))
    enc = params.get("encoding") or {}
    if not isinstance(enc, Mapping):
        raise ValueError("records/plot encoding is an object: channel → field")
    encoding: dict[str, str] = {c: str(f) for c, f in enc.items() if f}
    shortcuts = {"x": params.get("x"), "y": params.get("y"), "value": params.get("value"),
                 "text": params.get("text"), "lo": params.get("lo"), "hi": params.get("hi")}
    for c, f in shortcuts.items():
        if c not in encoding and f:
            encoding[c] = str(f)
    encoding = {c: encoding[c] for c in (*CHANNELS, *encoding) if c in encoding}
    header = records if isinstance(records, Mapping) else None
    given = {"title": params.get("title"), "labels": params.get("labels"), "axes": params.get("axes"),
             "annotate": params.get("annotate"), "reference": params.get("reference"),
             "focus": params.get("focus"), "facet": params.get("facet"), "bin": params.get("bin"),
             "level": params.get("level"), "scale": params.get("scale")}
    if given["bin"] is None and "bin" in legacy:
        given["bin"] = legacy["bin"]
    settings = [o for o in SETTINGS if is_set(given.get(o))]
    refuse_undeclared(decl, encoding, settings)
    scale = given["scale"]
    choices = decl["options"].get("scale", {}).get("choices", SCALES)
    if scale is not None and str(scale) not in choices:
        raise ValueError(
            f"records/plot scale must be one of {', '.join(choices)}, not {scale!r}")
    labels_in = params.get("labels") or {}
    if not isinstance(labels_in, Mapping):
        raise ValueError("records/plot labels is an object: {x, y, value, series, color}")
    unknown = sorted(set(labels_in) - set(LABEL_FIELDS))
    if unknown:
        raise ValueError(
            f"records/plot labels names {unknown}; it labels {', '.join(LABEL_FIELDS)}")
    labels = {k: str(v) for k, v in labels_in.items() if v}
    on_depth = any(encoding.get(c) == DEPTH_FIELD for c in decl["depth"])
    axes = (_check_layer_axis(params["axes"]) if params.get("axes") is not None
            else ({"layer": la} if on_depth and (la := _read_layer_axis(header)) else None))
    annotate = (_check_annotations(params["annotate"])
                if params.get("annotate") is not None else None)
    reference = (_check_references(params["reference"])
                 if params.get("reference") is not None else None)
    bins = read_bin(given["bin"]) if given["bin"] is not None else None
    interval = header if "lo" in encoding and "hi" in encoding else None
    level = read_level(params, interval) if "level" in decl["options"] else None
    focus = params.get("focus")
    facet = params.get("facet")
    spec: dict[str, Any] = {
        "kind": "records/chart",
        "title": params.get("title", ""),
        "mark": decl["address"],
        "encoding": encoding,
        **({"scale": str(scale)} if scale else {}),
        **({"labels": labels} if labels else {}),
        **({"axes": axes} if axes else {}),
        **({"annotate": annotate} if annotate else {}),
        **({"reference": reference} if reference else {}),
        **({"focus": str(focus)} if focus else {}),
        **({"facet": str(facet)} if facet else {}),
        **({"bin": bins} if bins is not None else {}),
        **({"level": level} if level is not None else {}),
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
