"""What a node is handed: its ports filled from its in-edges, from the
values written inline under `inputs`, and from whatever answers for an
upstream that produced nothing.
"""

from __future__ import annotations

from typing import Any

from mechbench_compute import lexicon
from mechbench_compute.protocol.missing_upstream import MissingUpstream
from mechbench_compute.protocol.read_missing_policy import read_missing_policy


def gather_inputs(state, nid, node, op_here, in_edges, resolver):
    """`(inputs, input_paths, inline_hashes)` for one node.

    `None` instead when a port wired to an upstream that produced
    nothing says `skip`: the node is recorded as missing here, and the
    caller moves on to the next one. A port that says `fail` — the
    default — raises `MissingUpstream` from here.
    """
    from mechbench_compute import resume as resume_mod

    by_port: dict[str, list[Any]] = {}
    for e in in_edges:
        by_port.setdefault(e["to"]["port"], []).append(e)
    # An upstream that produced nothing — it failed, or was itself
    # skipped — is answered by the port it was wired to. `fail` is the
    # default: the run stops here, with the original error. `skip`
    # passes the absence on. `placeholder` hands the block an empty
    # collection that says it is one.
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
        # Only placeholders left: drop those edges and fill below.
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
            # One edge, as every port but a variadic one takes; more
            # than one is refused at load by the graph check.
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
    # An input given inline under the node's `inputs` — a literal, or
    # `{"$ref": …}` of a stored object — fills a port the way an edge
    # does, and its content hash joins the fingerprint the way an
    # upstream node's would.
    inline_hashes: list[str] = []
    for port, raw in sorted((node.get("inputs") or {}).items()):
        if raw is None:
            continue
        if port in inputs:
            raise ValueError(
                f"{nid}: port {port!r} is wired by an edge and also "
                f"given under `inputs` — one or the other")
        inputs[port] = resolver.resolve_value(raw)
        inline_hashes.append(
            f"{port}:{resume_mod.content_hash(inputs[port])}")
    return inputs, input_paths, inline_hashes
