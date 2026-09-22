from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute.blocks.read_items import read_items
from mechbench_compute.lexicon._base import WILDCARD, In, Op, Output, P

OP = Op(
    name="records/union",
    summary=(
        "Concatenate several record streams into one, stamping each record "
        "with the port it came from — collections grow by union, never by "
        "mutation."
    ),
    description="""\
Every edge into the node is one input; its port name becomes the record's
value on the `batch_axis` coordinate, so the source of every record stays
visible downstream. `segments` records how many came from each port.

A union of vector collections stays a vector collection: items from a base
capture and an adapted capture become one collection whose items carry the
port they came from on the `batch_axis` coordinate — the grouping
`direction/fit` reads with `axis` set to it. Cross-model comparison
is a union followed by the direction algebra.
""",
    inputs=(
        In(WILDCARD, "collection",
           "Any number of edges, each carrying a collection — of records, or "
           "of `activations/vector` — on a port of your naming; the name "
           "becomes the value on the batch coordinate.", many=True),
    ),
    output=(
        Output('records/record', collection=True, doc="Every input's records, each with the batch coordinate; the header's `segments` says how many came from each port. When every input was a collection of `activations/vector`, so is the output, every item keeping its own `space`.")
    ),
    params=(
        P("batch_axis", "string",
          "The coordinate each record gains, set to the name of the port it "
          "arrived on.",
          "batch"),
    ),
    example={"batch_axis": "run"},
    example_inputs={"base": {"$ref": {"bench": "you/lab/base_vectors"}}, "adapted": {"$ref": {"bench": "you/lab/adapted_vectors"}}},
)


def run(ctx, inputs, params):
    return union(inputs, params)


def union(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    """Concatenate record streams, structurally recording source
    segments; each record gains a batch coordinate named after its
    input port. Collections never mutate — growth is union.

    A union of vector collections stays a vector collection: a base
    capture and an adapted capture come from two model nodes, and
    `direction/fit` reads ONE collection whose items
    are grouped on a coordinate — the port name, on the `batch_axis`
    coordinate, is that grouping. Every item carries its own `space`, so
    the header carries no union of layers. Cross-model comparison is a
    union followed by the direction algebra, no bespoke block."""
    from mechbench_compute import shapes as S
    from mechbench_compute.lexicon import kinds as K

    batch_axis = params.get("batch_axis", "batch")
    ports = sorted(inputs.keys())

    def _as_collection(port: str, value: Any) -> Any:
        # A single kinded object on a port — a direction from
        # `direction/fit`, say — is a collection of one; it
        # takes the port's name as its id when it carries none, so the
        # union's items stay distinguishable by key.
        if isinstance(value, Mapping) and K.item_kind_of(value) is None \
                and isinstance(value.get("kind"), str):
            try:
                name, plural = K.resolve_kind(str(value["kind"]), warn=False)
            except KeyError:
                return value
            if not plural and name in K.BY_KIND:
                item = dict(value)
                if item.get("id") is None:
                    item["id"] = port
                return K.collection(name, [item])
        return value

    inputs = {p: _as_collection(p, inputs[p]) for p in ports}
    vector_inputs = [inputs[p] for p in ports]

    def _vectors(v: Any) -> bool:
        ik = K.item_kind_of(v) if isinstance(v, Mapping) else None
        return ik is not None and K.satisfies(ik, "activations/vector")

    if ports and all(_vectors(v) for v in vector_inputs):
        first = vector_inputs[0]
        rows: list[dict[str, Any]] = []
        segments = []
        for port in ports:
            rec = inputs[port]
            rec_rows = K.items_of(rec)
            segments.append({"source": port, "count": len(rec_rows)})
            for r in rec_rows:
                item = dict(r)
                item["space"] = S.space_of(r, header=rec)
                item["coords"] = {**S.coords_of(r), batch_axis: port}
                # A flat `layer`/`head`/`label` on an item is dropped:
                # its `space` and `coords` carry them.
                for k in ("layer", "head", "label"):
                    item.pop(k, None)
                rows.append(item)
        header = {k: v for k, v in first.items()
                  if k not in ("kind", "item_kind", "key", "items", "rows",
                               "layers", "segments", "model")}
        return K.collection("activations/vector", rows, **header, segments=segments)
    segments = []
    records = []
    for port in ports:
        recs = read_items(inputs[port])
        segments.append({"source": port, "count": len(recs)})
        for r in recs:
            records.append({**r, "coords": {**r.get("coords", {}),
                                            batch_axis: port}})
    # A union of ONE kind stays that kind. The items are unchanged but
    # for a coordinate, so a collection of adapter deltas unioned with
    # another is still a collection of adapter deltas — and the metrics
    # its kind declares still apply to it. Mixed inputs fall back to the
    # root, which is the only thing they have in common.
    return K.collection(_resolve_shared_kind(inputs, ports), records,
                        segments=segments)


def _resolve_shared_kind(inputs: Mapping[str, Any], ports: Sequence[str]) -> str:
    from mechbench_compute.lexicon import kinds as K

    kinds = {K.item_kind_of(inputs[p]) if isinstance(inputs[p], Mapping) else None
             for p in ports}
    if len(kinds) == 1:
        only = kinds.pop()
        if isinstance(only, str) and only in K.BY_KIND:
            return only
    return "records/record"
