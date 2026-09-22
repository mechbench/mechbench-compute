from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.blocks.items import _items
from mechbench_compute.lexicon._base import In, Op, Output, P

OP = Op(
    name="records/zip",
    summary=(
        "Align several branches' records into one record per key — record "
        "7 of each branch together — keeping which branch each came from."
    ),
    description="""\
Two branches over the same prompts produce two streams, and a node that
compares them needs their records paired, not concatenated. `union` puts
both streams in one collection and marks where each came from; this puts
each branch's record for a key IN one record.

The **key** is `id` by default, or a list of coordinate names — `by:
["prompt", "seed"]` — which is what to use when two branches number their
records differently but share a design. Two records with the same key in
one branch are refused: zip needs one per key, and the fix is usually to
key on more coordinates.

Branches arrive on the **variadic `branches` port**: one edge per branch,
each named by the node it came from unless `names` says otherwise. The
port is ordered, so `names` lines up with the edges as the graph declares
them.

A key that is not in every branch is an error by default, because a
silently shorter output is a silently different experiment. `drop` keeps
the keys every branch has; `placeholder` keeps them all and marks what is
absent, for a readout that can report a missing arm.
""",
    inputs=(
        In("branches", "collection",
           "One edge per branch, each a collection of records. Ordered: the "
           "first edge is the first branch.",
           many=True, variadic=True, min_edges=2),
    ),
    output=Output('records/record', collection=True, doc="One record per key: `id`, `coords` from the first branch that has it, and `branches` — a map of branch name to that branch's record (or `{missing: true}` under `placeholder`). With `flatten`, each branch's fields are copied up under a `<branch>_` prefix instead. The header carries `branches` (name, source node, count) and `zipped` (the key, how many came out, the policy, and what was dropped)."),
    params=(
        P("by", "\"id\" | list[string]",
          "What to align on: record ids, or the named coordinates.",
          "id"),
        P("on_mismatch", "string",
          "A key missing from some branch: `\"fail\"`, `\"drop\"` (keep only "
          "the keys every branch has) or `\"placeholder\"` (keep them all, "
          "marking what is absent).",
          "fail", choices=("fail", "drop", "placeholder")),
        P("names", "list[string]",
          "What to call each branch, in edge order. Defaults to the source "
          "node ids.",
          None),
        P("flatten", "bool",
          "Copy each branch's fields up under a `<branch>_` prefix instead "
          "of nesting them under `branches`.",
          False),
    ),
    example={"by": ["prompt"], "names": ["base", "adapted"]},
    example_inputs={"branches": {"$ref": {"bench": "you/lab/base_reads"}}},
)


def run(ctx, inputs, params):
    return zip_branches(inputs, params)


def zip_branches(inputs: Mapping[str, Any],
                 params: Mapping[str, Any]) -> dict[str, Any]:
    """`records/zip` (task 000398): align several branches' records into
    one record per key, keeping which branch each came from.

    Two branches over the same prompts produce two streams; a judge that
    compares them needs record 7 of one beside record 7 of the other.
    `records/union` concatenates — the items stay separate, distinguished
    by a coordinate. This pairs them.

    The key is `id` by default, or a list of coordinate names (`by:
    ["prompt", "seed"]`), which is what to use when two branches number
    their records differently but share a design.

    A branch is named by the node it came from, unless `names` says
    otherwise — the names become the keys of each output record's
    `branches` map, and the field prefixes under `flatten`.
    """
    from mechbench_compute.lexicon import kinds as K

    edges = inputs.get("branches") or []
    if isinstance(edges, Mapping):            # one branch, given inline
        edges = [{"node": "branch", "value": edges}]
    if len(edges) < 2:
        raise ValueError(
            "records/zip needs at least two branches: wire one edge per "
            "branch onto its `branches` port. With one it would be a copy.")
    names = list(params.get("names") or [])
    if names and len(names) != len(edges):
        raise ValueError(
            f"`names` has {len(names)} entries for {len(edges)} branches")
    labels = names or [str(e.get("node") or i) for i, e in enumerate(edges)]
    if len(set(labels)) != len(labels):
        raise ValueError(f"two branches share a name: {labels}")

    by = params.get("by", "id")
    on_mismatch = str(params.get("on_mismatch", "fail"))
    if on_mismatch not in ("fail", "drop", "placeholder"):
        raise ValueError(
            f"on_mismatch is 'fail', 'drop' or 'placeholder', not {on_mismatch!r}")

    def key_of(rec: Mapping[str, Any]) -> Any:
        if by == "id":
            return str(rec.get("id"))
        coords = rec.get("coords") or {}
        missing = [c for c in by if c not in coords]
        if missing:
            raise ValueError(
                f"record {rec.get('id')!r} has no {', '.join(missing)} "
                f"coordinate to zip by")
        return tuple(str(coords[c]) for c in by)

    indexed: list[dict[Any, Mapping[str, Any]]] = []
    for e in edges:
        rows: dict[Any, Mapping[str, Any]] = {}
        for rec in _items(e.get("value")):
            k = key_of(rec)
            if k in rows:
                raise ValueError(
                    f"branch {e.get('node')!r} has two records keyed {k!r}; "
                    f"zip needs one per key (try `by` on more coordinates)")
            rows[k] = rec
        indexed.append(rows)

    everywhere = set(indexed[0])
    anywhere = set(indexed[0])
    for rows in indexed[1:]:
        everywhere &= set(rows)
        anywhere |= set(rows)
    missing = anywhere - everywhere
    if missing and on_mismatch == "fail":
        sample = ", ".join(repr(k) for k in sorted(missing, key=str)[:4])
        raise ValueError(
            f"{len(missing)} key(s) are not in every branch ({sample}"
            f"{'…' if len(missing) > 4 else ''}). `on_mismatch: \"drop\"` "
            f"keeps the keys every branch has; `\"placeholder\"` keeps them "
            f"all and marks what is absent.")
    keys = sorted(everywhere if on_mismatch != "placeholder" else anywhere,
                  key=str)

    flatten = bool(params.get("flatten", False))
    out: list[dict[str, Any]] = []
    for k in keys:
        present = [rows.get(k) for rows in indexed]
        first = next(r for r in present if r is not None)
        rec: dict[str, Any] = {
            "id": str(first.get("id")) if by == "id" else "-".join(k),
            "coords": dict(first.get("coords") or {}),
        }
        if flatten:
            for label, r in zip(labels, present):
                for field, value in (r or {}).items():
                    if field not in ("id", "coords", "kind"):
                        rec[f"{label}_{field}"] = value
                if r is None:
                    rec[f"{label}_missing"] = True
        else:
            rec["branches"] = {
                label: (dict(r) if r is not None else {"missing": True})
                for label, r in zip(labels, present)}
        out.append(rec)

    return K.collection(
        "records/record", out,
        branches=[{"name": label, "source": e.get("node"), "n": len(rows)}
                  for label, e, rows in zip(labels, edges, indexed)],
        zipped={"by": by, "keys": len(out), "on_mismatch": on_mismatch,
                "dropped": sorted(map(str, missing))[:50] if missing else []},
        name=params.get("name"),
        description=params.get("description"),
    )
