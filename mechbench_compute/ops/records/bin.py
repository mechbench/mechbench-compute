from __future__ import annotations

from mechbench_compute.blocks.read_field import read_field
from mechbench_compute.lexicon._base import In, Op, Output, P
from mechbench_compute.reduce.monoid import Monoid

OP = Op(
    name="records/bin",
    summary="Count a numeric field into fixed, equal-width bins.",
    description="""\
`bins` equal-width bins span `[lo, hi)`; values below `lo` and at or above
`hi` are counted separately rather than dropped, so the total always equals
the number of records. An exact reduce: counts add.
""",
    inputs=(In("records", "collection | records/table",
               "The records to work on: any collection of items — records, "
               "decision reads, vectors, verdicts, tree summaries — since every "
               "item has an id and its fields; a table's rows are read as records.",
               many=True),),
    output=Output('records/histogram', collection=False, doc='`bins` (the counts, in order), `below`, `above`.'),
    params=(
        P("value", "string", "The numeric field to bin."),
        P("lo", "float", "The lower edge of the first bin."),
        P("hi", "float", "The upper edge of the last bin (exclusive)."),
        P("bins", "int", "How many equal-width bins between `lo` and `hi`."),
    ),
    example={"value": "entropy_bits", "lo": 0.0, "hi": 8.0, "bins": 16},
    example_inputs={"records": {"$ref": {"bench": "you/lab/reads"}}},
)


def run(ctx, inputs, params):
    from mechbench_compute.lexicon import kinds as K

    m = MONOID()
    if hasattr(m, "bind"):
        m.bind(params or {})
    raw = inputs.get("records")
    recs = K.items_of(raw if raw is not None else [])
    return m.finalize(m.partial(recs, params), params)


class Histogram(Monoid):
    """Fixed-bin histogram: counts add. Exact."""

    def identity(self):
        return {}

    def partial(self, records, params):
        f = params["value"]
        lo, hi, n = float(params["lo"]), float(params["hi"]), int(params["bins"])
        counts: dict[int, int] = {}
        for r in records:
            v = float(read_field(r, f))
            b = n if v >= hi else (-1 if v < lo else int((v - lo) / (hi - lo) * n))
            counts[b] = counts.get(b, 0) + 1
        return counts

    def merge(self, a, b):
        out = dict(a)
        for k, v in b.items():
            out[k] = out.get(k, 0) + v
        return out

    def finalize(self, p, params):
        n = int(params["bins"])
        return {"kind": "records/histogram", "bins": [p.get(i, 0) for i in range(n)],
                "below": p.get(-1, 0), "above": p.get(n, 0)}


MONOID = Histogram
