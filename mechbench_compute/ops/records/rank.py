from __future__ import annotations

from mechbench_compute.blocks.read_field import read_field
from mechbench_compute.lexicon._base import In, Op, Output, P
from mechbench_compute.reduce.monoid import Monoid

OP = Op(
    name="records/rank",
    summary="The k records with the largest value of a field.",
    description="""\
Sorted by the field descending, ties broken by `id`, so the result is
deterministic. An exact reduce: the top-k of a union is the top-k of the
top-ks, so partial results merge without loss.
""",
    inputs=(In("records", "collection | records/table",
               "The records to work on: any collection of items — records, "
               "decision reads, vectors, verdicts, tree summaries — since every "
               "item has an id and its fields; a table's rows are read as records.",
               many=True),),
    output=Output('records/record', collection=True, doc='The top k, in order.'),
    params=(
        P("value", "string", "The numeric field to rank by, or a dot path to it."),
        P("k", "int", "How many to keep.", 10),
    ),
    example={"value": "delta", "k": 5},
    example_inputs={"records": {"$ref": {"bench": "you/lab/deltas"}}},
)


def run(ctx, inputs, params):
    from mechbench_compute.lexicon import kinds as K

    m = MONOID()
    if hasattr(m, "bind"):
        m.bind(params or {})
    raw = inputs.get("records")
    recs = K.items_of(raw if raw is not None else [])
    return m.finalize(m.partial(recs, params), params)


class TopK(Monoid):
    """Exact top-k by a value field: merge = top-k of the union."""

    def identity(self):
        return ()

    def partial(self, records, params):
        k = int(params.get("k", 10))
        f = params["value"]
        rows = sorted(records, key=lambda r: (-float(read_field(r, f)), str(r.get("id"))))
        return tuple(dict(r) for r in rows[:k])

    def merge(self, a, b):
        return tuple(sorted(a + b, key=lambda r: (-float(read_field(r, self._f)), str(r.get("id"))))[: self._k])

    def finalize(self, p, params):
        from mechbench_compute.lexicon import kinds as K

        return K.collection("records/record", list(p))

    def bind(self, params):
        self._f = params["value"]
        self._k = int(params.get("k", 10))
        return self


MONOID = TopK
