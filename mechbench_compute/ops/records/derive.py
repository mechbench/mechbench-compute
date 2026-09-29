from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.blocks.drop_field import drop_field
from mechbench_compute.blocks.keep_fields import keep_fields
from mechbench_compute.blocks.raise_expr_error import raise_expr_error
from mechbench_compute.blocks.read_header import read_header
from mechbench_compute.blocks.read_items import read_items
from mechbench_compute.blocks.read_order_by import read_order_by
from mechbench_compute.blocks.set_field import set_field
from mechbench_compute.lexicon._base import In, Op, Output, P, Resume

OP = Op(
    name="records/derive",
    resume=Resume("restart"),
    summary=(
        "Compute fields on every record from expressions: new fields, changed "
        "ones, text from templates, and fields dropped."
    ),
    description="""\
Each entry of `fields` names a field and gives the expression that computes
it from the record (the language is in mechbench-expr's SPEC.md): a bare
name reads the record's own field, `coords.x` a coordinate, `params.x` a
parameter of the run, `header.x` the input's header.

```
fields:
  lied: top[0].token.text != tracked.truth.token
  coords.model: params.model
  label: '"global attention" if attention else "local attention"'
```

A name with dots sets a path: `coords.model` sets the coordinate, creating
`coords` when the record has none. `templates` does the same with text:
`"Write a two-sentence story about {coords.animal}."`, each `{...}` an
expression and an optional format after a colon (`{p:.3f}`). `drop` removes
fields by path after the new ones are set, so moving a field is a new name
and a drop of the old; `keep` instead keeps only the fields it names (and
the id), after the new ones are set: `keep: [coords, text]`.

Every expression of a record reads the record as it arrived, not the fields
set beside it: two fields cannot build on each other in one node; chain two
derives when one reads the other.

A number with no value (a division by zero, the log of zero) is null, and
the header's `undefined` counts them by reason. An `order_by` on the
input (a sort's) is kept unless a field it names is dropped.
""",
    inputs=(In("records", "collection | records/table",
               "The records to compute on: any collection whose items are "
               "records, or a table's rows.",
               many=True),),
    output=Output("records/record", collection=True,
                  doc="The same records in the same order, each with its new fields; "
                      "the header's `undefined` counts the numbers that had no value, by reason, "
                      "when there were any."),
    params=(
        P("fields", "map[string, expression]",
          "The fields to set, by name (a dot path sets a nested field), each "
          "computed from the record by an expression.",
          None),
        P("templates", "map[string, template]",
          "The text fields to set, by name, each a template: `{...}` holds an "
          "expression.",
          None),
        P("drop", "list[string]",
          "Fields to remove, by path, after the new ones are set.",
          None),
        P("keep", "list[string]",
          "The only fields to keep, by path, after the new ones are set; the id is always kept.",
          None),
    ),
    example={"fields": {"lied": "top[0].token.text != tracked.truth.token"}, "drop": ["top"]},
    example_inputs={"records": {"$ref": {"bench": "you/lab/results/j_1/funnel"}}},
)


def run(ctx, inputs, params):
    return derive(inputs["records"], params, getattr(ctx, "run_params", None) or {})


def derive(records: Any, params: Mapping[str, Any], run_params: Mapping[str, Any]) -> dict[str, Any]:
    from mechbench_compute.expr.engine import ExprError, load_engine
    from mechbench_compute.lexicon import kinds as K

    items = read_items(records)
    header = read_header(records)
    fields: Mapping[str, str] = params.get("fields") or {}
    templates: Mapping[str, str] = params.get("templates") or {}
    drop = list(params.get("drop") or [])
    keep = params.get("keep")
    if not fields and not templates and not drop and keep is None:
        raise ValueError("records/derive: give fields, templates, drop or keep")
    engine = load_engine()
    undefined: dict[str, int] = {}
    computed: list[dict[str, Any]] = [{} for _ in items]
    try:
        if fields:
            got = engine.evaluate(dict(fields), items, run_params, header)
            for row, values in zip(computed, got.values, strict=True):
                row.update(values)
            _count(undefined, got.undefined)
        for name, template in templates.items():
            got = engine.render(template, items, run_params, header)
            for row, value in zip(computed, got.values, strict=True):
                row[name] = value
            _count(undefined, got.undefined)
    except ExprError as e:
        raise_expr_error("records/derive", e, items)
    out = []
    for record, row in zip(items, computed, strict=True):
        new = dict(record)
        for name, value in row.items():
            new = set_field(new, name, value)
        if keep is not None:
            new = keep_fields(new, [str(k) for k in keep])
        for path in drop:
            new = drop_field(new, path)
        out.append(new)
    kind = K.item_kind_of(records) if isinstance(records, Mapping) else None
    kept = kind if kind in K.BY_KIND and not drop and keep is None else "records/record"
    extra: dict[str, Any] = {"undefined": undefined} if undefined else {}
    order = read_order_by(records)
    kept_order = keep is None or all(any(f == k or f.startswith(f"{k}.") for k in keep) for f in order or [])
    if order and kept_order and not any(f == d or f.startswith(f"{d}.") for f in order for d in drop):
        extra["order_by"] = order
    return K.collection(kept, out, **extra)


def _count(into: dict[str, int], more: Mapping[str, int]) -> None:
    for reason, n in more.items():
        into[reason] = into.get(reason, 0) + int(n)
