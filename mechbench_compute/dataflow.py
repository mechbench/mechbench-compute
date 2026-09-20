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


def map_bound_names(node: Mapping[str, Any]) -> frozenset[str]:
    """The names a node with a body binds itself into that body: a
    `records/map`'s `bind` keys, per record (`bind: {topic: "user"}`);
    a `records/fold`'s `over` keys, per step, and `step`. A
    `{"$param": name}` under `body` naming one of these is the node's,
    not the run's."""
    try:
        block = lexicon.resolve(node["block"])
    except KeyError:
        return frozenset()
    params = node.get("params") or {}
    if block == "records/map":
        bind = params.get("bind")
        names = set(bind) if isinstance(bind, Mapping) else set()
        if params.get("over") is not None:
            # A map over plain values binds the name `as` gives it.
            names.add(str(params.get("as") or "value"))
        return frozenset(names)
    if block == "records/fold":
        over = params.get("over")
        keys: set[str] = {"step"}
        for step in (over if isinstance(over, list) else []):
            if isinstance(step, Mapping):
                keys.update(str(k) for k in step)
        return frozenset(keys)
    return frozenset()


def _inner_ref_site(params: Mapping[str, Any], path: list[str]):
    """Where a `$ref` under a map's `body` actually sits: `"port"` on a
    body node's inputs, `(inner_op, inner_path)` in a body node's
    params, None when the path is not under a body (000598)."""
    if path[:2] != ["body", "nodes"] or len(path) < 4:
        return None
    body = params.get("body")
    if not isinstance(body, Mapping) or not isinstance(body.get("nodes"), list):
        return None
    try:
        inner = body["nodes"][int(path[2])]
    except (IndexError, ValueError):
        return None
    if not isinstance(inner, Mapping) or not isinstance(inner.get("block"), str):
        return None
    where = path[3]
    if where == "inputs":
        return "port"
    if where != "params":
        return None
    try:
        inner_op = lexicon.BY_NAME.get(lexicon.resolve(inner["block"]))
    except KeyError:
        inner_op = None
    return (inner_op, path[4:]) if inner_op is not None else None


def bound_along(block: str, params: Mapping[str, Any], path: Sequence[str]) -> frozenset[str]:
    """The names bound by the node AND by every body node the path passes
    through. A body may hold another node with a body — a `records/fold`
    whose step is a `records/map`, which is how a per-item binding
    reaches a turn — and a `{"$param"}` down there is bound by whichever
    of them named it, not by the protocol (000617)."""
    names = set(map_bound_names({"block": block, "params": params}))
    cursor: Mapping[str, Any] = params
    i = 0
    while i + 3 < len(path) and path[i] == "body" and path[i + 1] == "nodes" and path[i + 3] == "params":
        body = cursor.get("body")
        if not isinstance(body, Mapping) or not isinstance(body.get("nodes"), list):
            break
        try:
            inner = body["nodes"][int(path[i + 2])]
        except (IndexError, ValueError):
            break
        if not isinstance(inner, Mapping):
            break
        names |= set(map_bound_names(inner))
        cursor = inner.get("params") or {}
        i += 4
    return frozenset(names)


def check_refs(nodes: Mapping[str, Mapping[str, Any]],
               bound_params: Mapping[str, Any]) -> None:
    """Refuse a stored-object reference that sits where no declaration
    admits one (principle 12: a node computes from its declared inputs).

    A port always admits one: a port is where data arrives. Inside a
    node's params a `$ref` — written there, or arriving through a param
    the run bound to one — needs the position to declare `stored` (the
    kind it takes by reference) or `reference` (it takes the address).
    A map body's own bound names are bound per record, later.
    """
    problems: list[str] = []

    def walk(v: Any, nid: str, op: Any, params: Mapping[str, Any],
             path: list[str], local: Callable[[Sequence[str]], frozenset[str]]) -> None:
        if is_param_ref(v):
            name = v["$param"]
            if path[:1] == ["body"] and name in local(path):
                return
            if name not in bound_params:
                problems.append(f"{nid}.{'.'.join(path)}: unbound param {name!r}")
                return
            v = bound_params[name]
        if is_object_ref(v):
            source_of(v)
            # Under a map's `body` the $ref sits on one of the BODY's
            # nodes, whose op — not the map's — says whether a stored
            # object belongs there; on that node's inputs it is a port,
            # and a port always does (000598).
            site = _inner_ref_site(params, path)
            if site == "port":
                return
            site_op, site_path = site if site else (op, path)
            decl = _declared_at(site_op, site_path) if site_op else None
            if site_op is not None and not (decl is not None and (decl.stored or decl.reference)):
                problems.append(
                    f"{nid}.{'.'.join(path)}: a $ref sits where {site_op.name} declares no "
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
                walk(x, nid, op, params, [*path, str(k)], local)
        elif isinstance(v, list):
            for i, x in enumerate(v):
                walk(x, nid, op, params, [*path, str(i)], local)

    for nid, node in nodes.items():
        try:
            op = lexicon.BY_NAME.get(lexicon.resolve(node["block"]))
        except KeyError:
            op = None
        params = node.get("params") or {}
        walk(params, nid, op, params, [],
             lambda path, _p=params, _b=str(node.get("block", "")): bound_along(_b, _p, path))
    if problems:
        raise ValueError("references that cannot be resolved:\n  " + "\n  ".join(problems))
