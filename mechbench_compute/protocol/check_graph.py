"""Everything about a graph that is decidable before it runs."""

from __future__ import annotations

from mechbench_compute import dataflow, lexicon

#: What a port may do when its upstream produced nothing.
MISSING_POLICIES = ("fail", "skip", "placeholder")


def check_graph(nodes, edges, order) -> None:
    """Everything about a graph that is decidable before it runs, decided
    before it runs.

    Checked here rather than when execution reaches a node, so that a
    graph that cannot run costs nothing upstream of its first mistake.
    Three things are known here, for every node:

    - its **operation** exists (a retired spelling names its
      replacement);
    - every **param** is one the op reads — names only, so nothing is
      fetched and no binding has to be resolved;
    - every **port** an edge or an `inputs` entry names exists, and
      every required port is filled by one of them.

    What a port is FILLED WITH is not knowable here — it is the upstream
    node's output — so the kind check stays where it is, in the loop.

    Every problem is reported, not the first: a protocol being carried
    forward usually has several, and one refusal per run is the
    expensive way to find them.
    """
    from mechbench_compute.block_params import check_params

    problems: list[str] = []
    into: dict[str, set[str]] = {}
    edge_counts: dict[str, dict[str, int]] = {}
    for e in edges:
        to = e.get("to") or {}
        if isinstance(to.get("node"), str) and isinstance(to.get("port"), str):
            into.setdefault(to["node"], set()).add(to["port"])
            counts = edge_counts.setdefault(to["node"], {})
            counts[to["port"]] = counts.get(to["port"], 0) + 1
    # Input edges lowered onto a variadic port are edges too.
    for nid, node in nodes.items():
        for port, raw in (node.get("inputs") or {}).items():
            if dataflow.is_input_branches(raw):
                counts = edge_counts.setdefault(nid, {})
                counts[port] = counts.get(port, 0) + len(raw)

    for nid in order:
        node = nodes[nid]
        block = node.get("block")
        if not isinstance(block, str):
            problems.append(f"  {nid}: no block")
            continue
        try:
            name = lexicon.resolve(block)
        except KeyError:
            problems.append(f"  {nid}: {lexicon.explain_unknown(block)}")
            continue
        try:
            check_params(name, node.get("params") or {})
        except ValueError as e:
            problems.append(f"  {nid} ({name}): {e}")
        op = lexicon.BY_NAME[name]
        filled = into.get(nid, set()) | {
            k for k, v in (node.get("inputs") or {}).items() if v is not None}
        for port_name in sorted(filled):
            if op.port(port_name) is None:
                known = ", ".join(sorted(p.name for p in op.inputs)) or "none"
                problems.append(
                    f"  {nid} ({name}): no input port {port_name!r}. "
                    f"Its ports: {known}.")
        # How MANY edges reach each port. A port that takes one and is
        # wired twice is refused here, so no result ever depends on the
        # order the author wrote the edges in.
        for port_name, n in sorted(edge_counts.get(nid, {}).items()):
            decl = op.port(port_name)
            if decl is None:
                continue          # already reported above
            bad = decl.arity_error(n)
            if bad:
                problems.append(f"  {nid} ({name}): port {port_name!r} {bad}.")
        # What each port does when its upstream produces nothing: the
        # policy must be one of the three, and `placeholder` needs a
        # kind with an empty value — an empty collection is a real value,
        # an empty `direction/vector` is not.
        for e in edges:
            if (e.get("to") or {}).get("node") != nid or not e.get("on_missing"):
                continue
            policy, port_name = str(e["on_missing"]), e["to"].get("port")
            decl = op.port(str(port_name))
            if policy not in MISSING_POLICIES:
                problems.append(
                    f"  {nid} ({name}): on_missing is "
                    f"{', '.join(MISSING_POLICIES)}, not {policy!r}.")
            elif policy == "placeholder" and decl is not None and not decl.many:
                problems.append(
                    f"  {nid} ({name}): port {port_name!r} takes one "
                    f"`{decl.kind}`, which has no empty value to stand in "
                    f"for a missing upstream. Use `skip`, or `fail`.")
        for port in op.inputs:
            if not port.required:
                continue
            if port.wildcard:
                if not filled:
                    problems.append(
                        f"  {nid} ({name}): needs at least one input edge "
                        f"(`{port.kind}` on a port of your naming).")
            elif port.name not in filled:
                problems.append(
                    f"  {nid} ({name}): needs an input on its {port.name!r} "
                    f"port (`{port.kind}`): wire an edge onto it, or give it "
                    f"under the node's `inputs`.")

    if problems:
        raise ValueError(
            f"this protocol cannot run — {len(problems)} problem"
            f"{'s' if len(problems) > 1 else ''} found before anything ran:\n"
            + "\n".join(problems))
