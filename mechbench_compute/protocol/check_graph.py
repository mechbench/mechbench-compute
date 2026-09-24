from __future__ import annotations

from mechbench_compute import dataflow, lexicon

MISSING_POLICIES = ("fail", "skip", "placeholder")


def check_graph(nodes, edges, order) -> None:
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
        for port_name, n in sorted(edge_counts.get(nid, {}).items()):
            decl = op.port(port_name)
            if decl is None:
                continue
            bad = decl.arity_error(n)
            if bad:
                problems.append(f"  {nid} ({name}): port {port_name!r} {bad}.")
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
