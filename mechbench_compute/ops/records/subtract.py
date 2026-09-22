from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.blocks.build_collection import build_collection
from mechbench_compute.blocks.read_items import read_items
from mechbench_compute.lexicon._base import Op, Output, P
from mechbench_compute.lexicon.records import _RECORDS

OP = Op(
    name="records/subtract",
    summary=(
        "Subtract a matched baseline from every record — the treatment "
        "effect per condition, ready to summarise."
    ),
    description="""\
Records whose `coords` match `baseline_where` are the baselines. Every other
record finds the baseline that agrees with it on the `match_on` coordinates
and reports its `value` field, the baseline's, and the difference. A record
with no matching baseline is an error, not a silent omission.

The output keeps `coords`, so it feeds `records/summarize` directly.
""",
    inputs=(_RECORDS,),
    output=Output('records/record', collection=True, doc='One record per non-baseline record: `{id, coords, value, baseline, delta}`.'),
    params=(
        P("value", "string",
          "The numeric field to difference. A record has many numeric "
          "fields; this names the one the question is about."),
        P("baseline_where", "map[string, string | float]",
          "Coordinates identifying the baseline records, e.g. "
          "`{\"alpha\": 0}`."),
        P("match_on", "list[string]",
          "Coordinates a record and its baseline must agree on. Empty "
          "means one baseline for everything.",
          None),
    ),
    example={"value": "entropy_bits", "baseline_where": {"alpha": 0}, "match_on": ["prompt"]},
    example_inputs={"records": {"$ref": {"bench": "you/lab/reads"}}},
)


def run(ctx, inputs, params):
    return build_collection(subtract_baseline(inputs["records"], params))


def subtract_baseline(records: Any, params: Mapping[str, Any]) -> list[dict[str, Any]]:
    """For each non-baseline record, subtract its matched baseline's
    value. match_on: coords that must agree; baseline_where: coords
    identifying the baseline records; value: the numeric field.
    Output records keep coords (minus nothing) plus value/baseline/
    delta fields — composable straight into group_stats."""
    recs = read_items(records)
    match_on = params.get("match_on") or []
    baseline_where: Mapping[str, Any] = params["baseline_where"]
    value_field = params["value"]

    def is_baseline(r):
        return all(r.get("coords", {}).get(k) == v
                   for k, v in baseline_where.items())

    baselines = {}
    for r in recs:
        if is_baseline(r):
            key = tuple(r.get("coords", {}).get(k) for k in match_on)
            baselines[key] = r
    out = []
    for r in recs:
        if is_baseline(r):
            continue
        key = tuple(r.get("coords", {}).get(k) for k in match_on)
        base = baselines.get(key)
        if base is None:
            raise ValueError(f"no baseline for record {r.get('id')!r}")
        v, b = float(r[value_field]), float(base[value_field])
        out.append({"id": r.get("id"), "coords": dict(r.get("coords", {})),
                    "value": v, "baseline": b,
                    "delta": round(v - b, 6)})
    return out
