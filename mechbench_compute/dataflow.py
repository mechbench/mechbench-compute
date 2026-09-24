from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from mechbench_compute import lexicon

DATAFLOW = 2
INTERMEDIATES = "nodes"
SOURCES = ("bench", "hf_dataset", "hf_adapter")
LEGACY_MACROS = ("$fetch", "$hf_dataset", "$hf_adapter")
DATAFLOW_DOCS = "https://docs.mechbench.ai/dataflow/"
NO_MARKER = 'no "dataflow": 2 marker'
INPUT_BRANCH = frozenset({"input", "index", "value"})
_HOLE = re.compile(r"^\$[A-Za-z][A-Za-z0-9_-]*$")


def is_declared(graph: Any) -> bool:
    return isinstance(graph, Mapping) and graph.get("dataflow") == DATAFLOW


def is_param_ref(v: Any) -> bool:
    return isinstance(v, Mapping) and len(v) == 1 and isinstance(v.get("$param"), str)


def is_object_ref(v: Any) -> bool:
    return isinstance(v, Mapping) and len(v) == 1 and isinstance(v.get("$ref"), Mapping)


def source_of(ref: Mapping[str, Any]) -> tuple[str, Any]:
    source = ref["$ref"]
    which = [k for k in SOURCES if k in source]
    if len(which) != 1:
        raise ValueError(
            f"a $ref names one source ({', '.join(SOURCES)}), not {sorted(source)}")
    return which[0], source[which[0]]


def find_undeclared(graph: Any) -> str | None:
    if is_declared(graph):
        return None
    if not isinstance(graph, Mapping):
        return "a graph that is not an object"
    nodes = graph.get("nodes") if isinstance(graph.get("nodes"), list) else []
    for n in nodes:
        if isinstance(n, Mapping) and n.get("block") == "protocol-input":
            return f"a protocol-input node {n.get('id')!r}"

    def walk(v: Any, at: str) -> str | None:
        if isinstance(v, str) and _HOLE.match(v):
            return f'a "{v}" string hole at {at}'
        if isinstance(v, Mapping):
            macro = next((k for k in LEGACY_MACROS if k in v), None)
            if macro:
                return f"{macro} at {at}"
            for k, x in v.items():
                found = walk(x, f"{at}.{k}")
                if found:
                    return found
        elif isinstance(v, list):
            for i, x in enumerate(v):
                found = walk(x, f"{at}.{i}")
                if found:
                    return found
        return None

    for n in nodes:
        if not isinstance(n, Mapping):
            continue
        for where in ("params", "inputs"):
            found = walk(n.get(where) or {}, f"{n.get('id')}.{where}")
            if found:
                return found
    for e in graph.get("edges") or []:
        src = e.get("from") if isinstance(e, Mapping) else None
        if isinstance(src, Mapping) and "port" in src:
            return f"an edge from {{node, port}} ({src.get('node')}.{src.get('port')})"
    return NO_MARKER


def check_form(extra: Mapping[str, Any]) -> None:
    found = find_undeclared(extra.get("graph") or {})
    if found is None and extra.get("bindings"):
        found = "a run bound by `bindings` rather than `params` and `inputs`"
    if found is not None:
        raise ValueError(
            f"this protocol is in the legacy dataflow form ({found}), which is no "
            f'longer read. Write it in the declared form, marked "dataflow": 2: '
            f"see {DATAFLOW_DOCS}")


def lower(graph: Mapping[str, Any], bound_inputs: Mapping[str, Any]) -> dict[str, Any]:
    nodes = [dict(n, inputs=dict(n.get("inputs") or {})) for n in graph.get("nodes", [])]
    by_id = {n["id"]: n for n in nodes}
    edges = []
    variadic: dict[str, set[str]] = {}
    for e in graph.get("edges", []):
        src, dst = e["from"], e["to"]
        if "input" in src:
            name = src["input"]
            if name not in bound_inputs:
                raise ValueError(f"unbound input: {name!r}")
            target = by_id.get(dst["node"])
            if target is None:
                raise ValueError(f"an edge goes to {dst['node']!r}, which is not a node")
            port = dst["port"]
            if is_variadic_port(target.get("block"), port):
                if port not in variadic.setdefault(target["id"], set()):
                    if port in target["inputs"]:
                        raise ValueError(
                            f"{dst['node']}: port {port!r} is fed by input {name!r} "
                            f"and also given under `inputs` — one or the other")
                    variadic[target["id"]].add(port)
                    target["inputs"][port] = []
                target["inputs"][port].append(
                    {"input": name, "index": int(e.get("index", 0)),
                     "value": bound_inputs[name]})
                continue
            if port in target["inputs"]:
                raise ValueError(
                    f"{dst['node']}: port {port!r} is fed by input {name!r} "
                    f"and also given under `inputs` — one or the other")
            target["inputs"][port] = bound_inputs[name]
            continue
        edges.append({**e, "from": {"node": src["node"], "port": src.get("output", "out")}})
    fed = {(e["to"]["node"], e["to"]["port"]) for e in edges}
    for n in nodes:
        both = [p for p in n["inputs"]
                if (n["id"], p) in fed and p not in variadic.get(n["id"], ())]
        if both:
            raise ValueError(
                f"{n['id']}: port {both[0]!r} is wired by an edge and also fed by an "
                f"input — a port takes one or the other")
    return {"nodes": nodes, "edges": edges}


def is_variadic_port(block: Any, port: str) -> bool:
    try:
        op = lexicon.BY_NAME.get(lexicon.resolve(str(block), warn=False))
    except KeyError:
        return False
    decl = op.port(port) if op is not None else None
    return bool(decl is not None and decl.variadic)


def is_input_branches(value: Any) -> bool:
    return (isinstance(value, list) and bool(value)
            and all(isinstance(v, Mapping) and set(v) == INPUT_BRANCH for v in value))


def _declared_at(op: Any, path: list[str]) -> Any:
    fields = list(getattr(op, "params", ()) or ()) + list(lexicon.COMMON)
    found = None
    for key in path:
        if key.isdigit():
            continue
        found = next((f for f in fields if f.name == key), None)
        if found is None:
            return None
        fields = list(found.fields)
    return found


def wants_reference(block: str, name: str) -> bool:
    op = lexicon.BY_NAME.get(block)
    decl = _declared_at(op, [name]) if op else None
    return bool(decl is not None and decl.reference)


def map_bound_names(node: Mapping[str, Any]) -> frozenset[str]:
    try:
        block = lexicon.resolve(node["block"])
    except KeyError:
        return frozenset()
    params = node.get("params") or {}
    if block == "records/map":
        bind = params.get("bind")
        names = set(bind) if isinstance(bind, Mapping) else set()
        if params.get("over") is not None:
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
