"""The declared dataflow form, as the executor reads it (epic 000553, task
000557; design in mechbench/docs/DATAFLOW.md).

A protocol in the declared form carries `dataflow: 2` and speaks in two
references — `{"$param": name}` and `{"$ref": source}` — where the legacy
form had string holes and the `$fetch` macro. A reference is a value: it
can be bound, passed along, and it sits only where a declaration admits
it. An edge's source is a node's output or one of the protocol's inputs.

This module does two things, both before anything runs:

- `lower` rewrites a declared graph into the shapes the executor has
  always run: an edge from a protocol input becomes that input's bound
  value on the port, and `from: {node, output}` becomes `{node, port}`.
  Ordering, resume, missing-node handling and fingerprints then see what
  they always saw, which is the point: a migrated protocol resolves to the
  same values in the same places, so its nodes keep their fingerprints.
- `check_refs` refuses a `$ref` that sits where no declaration admits a
  stored object, by node and place.

Resolution itself stays in the executor (`resolve_value`), which holds the
bench client and the record of what resolved.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute import lexicon

DATAFLOW = 2
#: Under a run's result path, where every node that is not a declared
#: output is stored: `results/<job>/nodes/<node id>`. An output may not
#: take the name.
INTERMEDIATES = "nodes"
SOURCES = ("bench", "hf_dataset", "hf_adapter")
#: The legacy macros, which a declared graph must not contain: there a
#: `$`-keyed object is one of the two references or it is a mistake.
LEGACY_MACROS = ("$fetch", "$hf_dataset", "$hf_adapter")


def is_declared(graph: Any) -> bool:
    return isinstance(graph, Mapping) and graph.get("dataflow") == DATAFLOW


def is_param_ref(v: Any) -> bool:
    return isinstance(v, Mapping) and len(v) == 1 and isinstance(v.get("$param"), str)


def is_object_ref(v: Any) -> bool:
    return isinstance(v, Mapping) and len(v) == 1 and isinstance(v.get("$ref"), Mapping)


def source_of(ref: Mapping[str, Any]) -> tuple[str, Any]:
    """`(which, spec)` of a `{"$ref": source}`, refusing anything else."""
    source = ref["$ref"]
    which = [k for k in SOURCES if k in source]
    if len(which) != 1:
        raise ValueError(
            f"a $ref names one source ({', '.join(SOURCES)}), not {sorted(source)}")
    return which[0], source[which[0]]


def lower(graph: Mapping[str, Any], bound_inputs: Mapping[str, Any]) -> dict[str, Any]:
    """A declared graph in the executor's working shapes.

    An edge from `{"input": name}` puts the run's bound value for that
    input on the target port, as an inline input. That is where a legacy
    `{"$fetch": "$corpus"}` sat, so the node's input hash, and with it its
    fingerprint, is formed exactly as before.
    """
    nodes = [dict(n, inputs=dict(n.get("inputs") or {})) for n in graph.get("nodes", [])]
    by_id = {n["id"]: n for n in nodes}
    edges = []
    for e in graph.get("edges", []):
        src, dst = e["from"], e["to"]
        if "input" in src:
            name = src["input"]
            if name not in bound_inputs:
                raise ValueError(f"unbound input: {name!r}")
            target = by_id.get(dst["node"])
            if target is None:
                raise ValueError(f"an edge goes to {dst['node']!r}, which is not a node")
            if dst["port"] in target["inputs"]:
                raise ValueError(
                    f"{dst['node']}: port {dst['port']!r} is fed by input {name!r} "
                    f"and also given under `inputs` — one or the other")
            target["inputs"][dst["port"]] = bound_inputs[name]
            continue
        edges.append({**e, "from": {"node": src["node"], "port": src.get("output", "out")}})
    fed = {(e["to"]["node"], e["to"]["port"]) for e in edges}
    for n in nodes:
        both = [p for p in n["inputs"] if (n["id"], p) in fed]
        if both:
            raise ValueError(
                f"{n['id']}: port {both[0]!r} is wired by an edge and also fed by an "
                f"input — a port takes one or the other")
    return {"nodes": nodes, "edges": edges}


def _declared_at(op: Any, path: list[str]) -> Any:
    """The lexicon's declaration at a path inside a node's params: fields
    by name, a list's index skipped. None where nothing is declared."""
    fields = list(getattr(op, "params", ()) or ()) + list(lexicon.COMMON)
    found = None
    for key in path:
        if key.isdigit():
            continue
        found = next((f for f in fields if f.name == key), None)
        if found is None:
            return None
        # A shared value (`pool`, `point`) declares its fields as plain
        # dicts, which cannot say `stored`; nothing inside one takes a
        # stored object, so the walk ends there.
        fields = list(found.fields)
    return found


def wants_reference(block: str, name: str) -> bool:
    """Whether the op declares that top-level param as taking the
    reference itself, unresolved."""
    op = lexicon.BY_NAME.get(block)
    decl = _declared_at(op, [name]) if op else None
    return bool(decl is not None and decl.reference)


def check_refs(nodes: Mapping[str, Mapping[str, Any]],
               bound_params: Mapping[str, Any]) -> None:
    """Refuse a stored-object reference that sits where no declaration
    admits one (principle 12: a node computes from its declared inputs).

    A port always admits one: a port is where data arrives. Inside a
    node's params a `$ref` — written there, or arriving through a param
    the run bound to one — needs the position to declare `stored` (the
    kind it takes by reference) or `reference` (it takes the address).
    """
    problems: list[str] = []

    def walk(v: Any, nid: str, op: Any, path: list[str]) -> None:
        if is_param_ref(v):
            name = v["$param"]
            if name not in bound_params:
                problems.append(f"{nid}.{'.'.join(path)}: unbound param {name!r}")
                return
            v = bound_params[name]
        if is_object_ref(v):
            source_of(v)
            decl = _declared_at(op, path) if op else None
            if op is not None and not (decl is not None and (decl.stored or decl.reference)):
                problems.append(
                    f"{nid}.{'.'.join(path)}: a $ref sits where {op.name} declares no "
                    f"stored object; wire it to a port, or pass the value")
            return
        if isinstance(v, Mapping):
            macro = next((k for k in LEGACY_MACROS if k in v), None)
            if macro:
                problems.append(
                    f"{nid}.{'.'.join(path)}: {macro} is the legacy form; a declared "
                    f"protocol writes {{\"$ref\": …}}")
                return
            for k, x in v.items():
                walk(x, nid, op, [*path, str(k)])
        elif isinstance(v, list):
            for i, x in enumerate(v):
                walk(x, nid, op, [*path, str(i)])

    for nid, node in nodes.items():
        try:
            op = lexicon.BY_NAME.get(lexicon.resolve(node["block"]))
        except KeyError:
            op = None
        walk(node.get("params") or {}, nid, op, [])
    if problems:
        raise ValueError("references that cannot be resolved:\n  " + "\n  ".join(problems))
