from __future__ import annotations

from typing import Any

from mechbench_compute import dataflow, lexicon
from mechbench_compute.protocol.missing_upstream import MissingUpstream
from mechbench_compute.protocol.read_missing_policy import read_missing_policy


def gather_inputs(state, nid, node, op_here, in_edges, resolver):
    from mechbench_compute import resume as resume_mod

    by_port: dict[str, list[Any]] = {}
    for e in in_edges:
        by_port.setdefault(e["to"]["port"], []).append(e)
    absent = {
        port: [e for e in es if e["from"]["node"] in state.missing]
        for port, es in by_port.items()
    }
    absent = {p: es for p, es in absent.items() if es}
    if absent:
        policy_skip = False
        for port, es in sorted(absent.items()):
            decl = op_here.port(port) if op_here else None
            policy = read_missing_policy(decl, es)
            source = es[0]["from"]["node"]
            why = state.missing[source]
            if policy == "fail":
                raise MissingUpstream(nid, port, source, why) from None
            if policy == "skip":
                policy_skip = True
            state.tolerated.update(e["from"]["node"] for e in es)
        if policy_skip:
            src = sorted({e["from"]["node"]
                          for es in absent.values() for e in es})
            state.missing[nid] = {"reason": "an upstream is missing",
                                  "source": src}
            print(f"[graph] {nid}: skipped ({', '.join(src)} missing)")
            return None
        by_port = {
            p: [e for e in es if e["from"]["node"] not in state.missing]
            for p, es in by_port.items()
        }
        placeholders = {p: es for p, es in absent.items()}
    else:
        placeholders = {}
    inputs, input_paths = {}, {}
    for port, es in {p: es for p, es in by_port.items() if es}.items():
        decl = op_here.port(port) if op_here else None
        if decl is not None and decl.variadic:
            inputs[port] = [
                {"node": e["from"]["node"],
                 "value": state.results[e["from"]["node"]]} for e in es]
            input_paths[port] = [
                state.node_paths.get(e["from"]["node"], "") for e in es]
        else:
            inputs[port] = state.results[es[-1]["from"]["node"]]
            input_paths[port] = state.node_paths.get(es[-1]["from"]["node"], "")
    for port, es in sorted(placeholders.items()):
        decl = op_here.port(port) if op_here else None
        kind = (decl.kinds[0] if decl is not None else "records/record")
        if kind == lexicon.COLLECTION:
            kind = "records/record"
        stand_in = lexicon.collection(kind, [], missing={
            "reason": "the upstream produced nothing",
            "source": sorted({e["from"]["node"] for e in es})})
        if port in inputs and isinstance(inputs[port], list):
            inputs[port] = [*inputs[port],
                            *({"node": e["from"]["node"],
                               "value": stand_in} for e in es)]
        else:
            inputs[port] = stand_in
    inline_hashes: list[str] = []
    for port, raw in sorted((node.get("inputs") or {}).items()):
        if raw is None:
            continue
        if dataflow.is_input_branches(raw):
            add_input_branches(inputs, input_paths, inline_hashes, port, raw,
                               in_edges, resolver)
            continue
        if port in inputs:
            raise ValueError(
                f"{nid}: port {port!r} is wired by an edge and also "
                f"given under `inputs` — one or the other")
        inputs[port] = resolver.resolve_value(raw)
        inline_hashes.append(
            f"{port}:{resume_mod.content_hash(inputs[port])}")
    return inputs, input_paths, inline_hashes


def add_input_branches(inputs, input_paths, inline_hashes, port, raw, in_edges,
                       resolver) -> None:
    from mechbench_compute import resume as resume_mod

    index_of = {e["from"]["node"]: int(e.get("index", 0))
                for e in in_edges if e["to"]["port"] == port}
    from_edges = inputs.get(port) or []
    paths = dict(zip((entry["node"] for entry in from_edges),
                     input_paths.get(port) or []))
    entries = [((index_of.get(entry["node"], 0), str(entry["node"])), entry,
                paths.get(entry["node"], ""))
               for entry in from_edges]
    for branch in raw:
        value = resolver.resolve_value(branch["value"])
        ref = branch["value"]
        stored = (ref["$ref"].get("bench") or "") if dataflow.is_object_ref(ref) else ""
        entries.append(((branch["index"], branch["input"]),
                        {"input": branch["input"], "value": value}, stored))
    entries.sort(key=lambda entry: entry[0])
    inputs[port] = [entry for _key, entry, _path in entries]
    input_paths[port] = [path for _key, _entry, path in entries]
    for pos, (_key, entry, _path) in enumerate(entries):
        if "input" in entry:
            inline_hashes.append(
                f"{port}.{pos}:{resume_mod.content_hash(entry['value'])}")
