from __future__ import annotations

import math

from mechbench_compute.blocks.read_field import read_field
from mechbench_compute.lexicon._base import In, Op, Output, P
from mechbench_compute.reduce.monoid import Monoid

OP = Op(
    name="records/total",
    summary="The exact sum of a numeric field over all records, with the count.",
    description="""\
An exact reduce: values are kept as a multiset and summed with a correctly
rounded algorithm, so the result is the same whatever order or chunking the
records arrived in. Safe to run over partial results and merge.
""",
    inputs=(In("records", "collection | records/table",
               "The records to work on: any collection of items — records, "
               "decision reads, vectors, verdicts, tree summaries — since every "
               "item has an id and its fields; a table's rows are read as records.",
               many=True),),
    output=Output('records/sum', collection=False, doc='`{n, sum}`.'),
    params=(P("value", "string", "The numeric field to sum."),),
    example={"value": "cost_usd"},
    example_inputs={"records": {"$ref": {"bench": "you/lab/records"}}},
)


def run(ctx, inputs, params):
    from mechbench_compute.lexicon import kinds as K

    m = MONOID()
    if hasattr(m, "bind"):
        m.bind(params or {})
    raw = inputs.get("records")
    recs = K.items_of(raw if raw is not None else [])
    return m.finalize(m.partial(recs, params), params)


class FloatSum(Monoid):
    """Exact sum of floats: the partial keeps the values as a multiset
    (a sorted tuple); finalize uses `math.fsum`. Order-independent."""

    def identity(self):
        return ()

    def partial(self, records, params):
        f = params["value"]
        return tuple(sorted(float(read_field(r, f)) for r in records))

    def merge(self, a, b):
        return tuple(sorted(a + b))

    def finalize(self, p, params):
        return {"kind": "records/sum", "n": len(p), "sum": math.fsum(p)}


MONOID = FloatSum
