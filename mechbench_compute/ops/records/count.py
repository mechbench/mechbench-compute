from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import NormalDist
from typing import Any

from mechbench_compute.blocks.match_where import match_where, parse_where
from mechbench_compute.blocks.read_field import read_field
from mechbench_compute.blocks.read_group_key import read_group_key
from mechbench_compute.blocks.read_items import read_items
from mechbench_compute.lexicon._base import In, Op, Output, P
from mechbench_compute.reduce.monoid import Monoid

OP = Op(
    name="records/count",
    summary=(
        "Count the records whose field holds a value, or that meet a "
        "condition — k of n, the "
        "proportion, and its Wilson interval — one row per combination of "
        "the `by` coordinates."
    ),
    description="""\
A judge's verdicts, a pattern's hits, a pairwise winner: a finding that
quotes "the rung was preferred in 8 of 30 pairs" is a proportion, and a
proportion over thirty is a claim with a width. Name the `field` and the
value that counts (`equals`); every record that carries the field is one
trial (`n`), and each whose field equals the value is one success (`k`).
The row reports `k`, `n`, the proportion `rate`, and the interval `lo`–`hi`
at the given level.

### The interval

The Wilson score interval. It is the interval to use at the sizes a
judged corpus has: it stays inside 0–1, it is not zero-width when every
record went one way (0 of 30 is not certainty), and its coverage stays
near the stated level down to a few dozen trials. The textbook interval,
the proportion plus or minus 1.96 standard errors, fails at exactly those
places; the exact (Clopper–Pearson) interval holds its level by being
wider than it needs to be. The header's `interval` names the method and
the level.

The records are trials only if they are independent: three votes on one
pair are one pair judged three times, not three pairs. Count at the level
the question is about — pairs, not votes — or say which it is.

A record without the field is refused by name. When absent values are
expected — an unparsed vote carries no winner — set `on_missing: "skip"`,
and the count of skipped records is reported on the table as `n_missing`.

A field name with dots in it is a path from the record's root,
`metadata.call.stop_reason`; a coordinate or top-level field of that
name is read first.

### A condition instead of a value

When a success is not one value — a reply longer than 250 tokens, a
score of at least 4 — name it with `where` in place of `field`: a list of
conditions in the grammar the API's item query reads, `PATH OP VALUE`
with OP one of `=` `!=` `<` `<=` `>` `>=` `~`. A record is a success when
every condition holds, and every record is a trial. A missing field is
null, so `metadata.call.usage.reasoning_tokens>0` is a failure on a reply
that carries no count, and `x!=null` counts the records that have one.

`by` names coordinates (or top-level fields) to group on: one row per
combination, each its own count. Empty gives one overall row.
""",
    inputs=(In("records", "collection | records/table",
               "The records to work on: any collection of items — records, "
               "decision reads, vectors, verdicts, tree summaries — since every "
               "item has an id and its fields; a table's rows are read as records.",
               many=True),),
    output=(
        Output('records/table', collection=False, doc='One row per group with the `by` coordinates and `k`, `n`, `rate`, `lo`, `hi`; `n_missing` when any records were skipped. The header\'s `interval` says the level and the method (`wilson`), and `counted` says which field and value, or which conditions.')
    ),
    params=(
        P("field", "string",
          "The field read from every record: a coordinate, a top-level field, "
          "or a dot path. One of `field` and `where` is required.",
          None),
        P("equals", "json",
          "The value that counts as a success. `true` by default, for a "
          "field that is already a yes or a no.",
          True),
        P("by", "list[string]",
          "The coordinates to group on. Empty gives one overall row.",
          None),
        P("where", "list[string | object]",
          "Conditions a success meets, each `PATH OP VALUE` (`\"lex_words>80\"`) or "
          "`{path, op, value}`, in place of `field` and `equals`.",
          None, fields=(
              P("path", "string", "The field or dot path the condition reads."),
              P("op", "string", "The comparison.", "=", choices=("=", "!=", "<", "<=", ">", ">=", "~")),
              P("value", "json", "The value compared against, as given: a number, text, "
                "`null`, or a `$param` reference."),
          )),
        P("on_missing", "string",
          "`\"error\"`: refuse a record without the field. `\"skip\"`: omit "
          "it from `n` and report how many were omitted.",
          "error", choices=("error", "skip")),
        P("interval", "float", "The level of the Wilson interval.", 0.95),
    ),
    example={"field": "winner", "equals": "A", "by": ["prompt"]},
    example_inputs={"records": {"$ref": {"bench": "you/lab/verdicts"}}},
)


def run(ctx, inputs, params):
    return count(inputs["records"], params)


def count(records: Any, params: Mapping[str, Any]) -> dict[str, Any]:
    """`records/count` over a whole collection: the monoid's partial of
    every record, finalized, so the flat and the chunked table are one
    computation."""
    m = MONOID()
    return m.finalize(m.partial(read_items(records), params), params)


def estimate_wilson(k: int, n: int, level: float) -> tuple[float, float]:
    """The Wilson score interval on k successes in n trials."""
    z = NormalDist().inv_cdf(0.5 + level / 2)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)



def _matches(value: Any, equals: Any) -> bool:
    # A bool is an int in Python; `1` is not `true` in a record.
    if isinstance(equals, bool) or isinstance(value, bool):
        return isinstance(value, bool) and isinstance(equals, bool) and value is equals
    return value == equals


def _read_success(params: Mapping[str, Any]):
    field, where = params.get("field"), params.get("where")
    if (field is None) == (where is None):
        raise ValueError("count takes one of `field` and `where`")
    return field, (parse_where(where) if where is not None else None)


def _read_level(params: Mapping[str, Any]) -> float:
    level = float(params.get("interval", 0.95))
    if not 0.0 < level < 1.0:
        raise ValueError(f"interval must be between 0 and 1 exclusive, not {level}")
    return level


class CountShare(Monoid):
    """Per group, the successes and the trials; and the records skipped.
    Sums, so any partition of the records merges to the same counts."""

    def identity(self):
        return ({}, 0)

    def partial(self, records: Sequence[Mapping[str, Any]], params):
        field, conditions = _read_success(params)
        equals = params.get("equals", True)
        by = params.get("by") or []
        on_missing = str(params.get("on_missing", "error"))
        if on_missing not in ("error", "skip"):
            raise ValueError(f"count on_missing must be 'error' or 'skip', not {on_missing!r}")
        groups: dict[tuple, tuple[int, int]] = {}
        missing = 0
        for r in records:
            if conditions is not None:
                key = read_group_key(r, by)
                k, n = groups.get(key, (0, 0))
                groups[key] = (k + match_where(r, conditions), n + 1)
                continue
            value = read_field(r, field)
            if value is None:
                if on_missing == "skip":
                    missing += 1
                    continue
                raise ValueError(
                    f"count: record {r.get('id')!r} has no {field!r} field. Set "
                    f"on_missing: 'skip' if absent values are expected — the "
                    f"count is then reported on the table.")
            key = read_group_key(r, by)
            k, n = groups.get(key, (0, 0))
            groups[key] = (k + _matches(value, equals), n + 1)
        return groups, missing

    def merge(self, a, b):
        groups = dict(a[0])
        for key, (k, n) in b[0].items():
            k0, n0 = groups.get(key, (0, 0))
            groups[key] = (k0 + k, n0 + n)
        return groups, a[1] + b[1]

    def finalize(self, p, params):
        groups, missing = p
        by = params.get("by") or []
        level = _read_level(params)
        rows = []
        for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
            k, n = groups[key]
            lo, hi = estimate_wilson(k, n, level)
            row = {name: key[i] for i, name in enumerate(by)}
            row.update({"k": k, "n": n, "rate": round(k / n, 4),
                        "lo": round(lo, 4), "hi": round(hi, 4)})
            rows.append(row)
        columns = ([{"name": name, "dtype": "string"} for name in by]
                   + [{"name": c, "dtype": "number"} for c in ("k", "n", "rate", "lo", "hi")])
        field, conditions = _read_success(params)
        counted = ({"where": [f"{c.path}{c.op}{c.raw}" for c in conditions]}
                   if conditions is not None
                   else {"field": field, "equals": params.get("equals", True)})
        out = {"kind": "records/table",
               "name": params.get("name", f"{field or 'where'}-count"),
               "description": params.get("description", ""),
               "row_axis": "condition", "columns": columns, "rows": rows,
               "counted": counted,
               "interval": {"level": level, "method": "wilson", "of": "proportion"}}
        if missing:
            out["n_missing"] = missing
        return out


MONOID = CountShare
