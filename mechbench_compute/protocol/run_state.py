"""What one run of a graph is, and what it accumulates.

Two halves. The first is read off the spec once and never changes: the
graph in the one shape everything below sees, its nodes and edges, the
order they run in, what the protocol declares it keeps, and which nodes
a resume requirement forces to restart. The second is what the walk
fills as it goes — a result, a content hash and a stored path per node,
what was held rather than emitted, what produced nothing and why, what
each node spent.

The state is one object so that the walk's steps can be functions
taking it as an argument rather than closures over thirty locals.
"""

from __future__ import annotations

from typing import Any

from mechbench_compute import dataflow
from mechbench_compute.protocol.check_graph import check_graph
from mechbench_compute.protocol.sort_nodes import sort_nodes


class RunState:
    """RunState: see this module's docstring."""

    def __init__(self, spec, resume=None) -> None:
        from mechbench_compute import resume as resume_mod

        extra = self.extra = spec.extra or {}
        # The run binds `params` and `inputs` by name, and the graph
        # refers to them with values no literal can be. A spec in any
        # other form is refused here, before anything is read; the graph
        # is then lowered, so everything below — ordering, resume,
        # missing nodes, fingerprints — sees one graph shape.
        dataflow.check_form(extra)
        self.bound_params = extra.get("params") or {}
        graph = dataflow.lower(extra.get("graph") or {}, extra.get("inputs") or {})
        self.graph = graph
        self.nodes: dict[str, Any] = {n["id"]: n for n in graph.get("nodes", [])}
        self.edges: list[dict] = graph.get("edges", [])
        # What the protocol declares it keeps: `[{name, from: {node}}]`,
        # from its signature. With these, a result is stored under its
        # declared NAME and every other node's value apart, as an
        # intermediate — so a node can be renamed without moving a result
        # anyone depends on. Without them, terminals are the outputs,
        # under their ids.
        self.declared_outputs = extra.get("outputs")
        self.outputs_of: dict[str, list[str]] = {}
        for o in self.declared_outputs or []:
            source = o["from"]["node"]
            if source not in self.nodes:
                raise ValueError(
                    f"output {o['name']!r} comes from {source!r}, which is not a node")
            if o["name"] == dataflow.INTERMEDIATES:
                raise ValueError(
                    f"an output cannot be named {dataflow.INTERMEDIATES!r}: that is "
                    f"where intermediates are stored")
            self.outputs_of.setdefault(source, []).append(o["name"])
        # Eager discard: with `keep: "outputs"` a node that is not a
        # declared output is never emitted. Its result stays here for its
        # consumers, goes to the runner's spool for a resume, and is
        # cited downstream by content hash. The default keeps everything.
        self.keep = str(extra.get("keep") or "all")
        if self.keep not in ("all", "outputs"):
            raise ValueError(f"keep must be 'all' or 'outputs', not {self.keep!r}")
        self.discard = self.keep == "outputs"
        self.result_base = extra.get("resultPath")
        self.order = sort_nodes(self.nodes, self.edges)
        # Before anything runs: is this graph runnable at all? Blocks,
        # params and ports are decidable at load, and a graph that cannot
        # run says so in the first second rather than after the nodes
        # upstream of the mistake have been computed.
        check_graph(self.nodes, self.edges, self.order)
        dataflow.check_refs(self.nodes, self.bound_params)
        self.resume = resume or {}
        self.forced_restart = self.read_forced_restart(resume_mod)

        #: Node id -> what it produced.
        self.results: dict[str, Any] = {}
        #: Node id -> the content hash of what it produced.
        self.node_hashes: dict[str, str] = {}
        #: Node id -> where its object was stored, when it was stored.
        self.node_paths: dict[str, str] = {}
        #: Held results: node id -> (block, resolved params), what an
        #: emit of the evidence needs should the run fail.
        self.held: dict[str, tuple[str, Any]] = {}
        # Nodes that produced nothing, and why. A node lands here by
        # failing, or by being skipped because something upstream of it
        # did. `tolerated` records the ones some consumer answered for;
        # a failure nothing answered for is raised when the run is
        # otherwise over, so sibling branches still finish.
        self.missing: dict[str, dict[str, Any]] = {}
        self.failures: dict[str, BaseException] = {}
        self.tolerated: set[str] = set()
        # Remote nodes run ahead of their turn, alongside a sibling;
        # their results wait here for the walk to reach them and do the
        # bookkeeping in topological order.
        self.ahead: dict[str, Any] = {}
        # What this run bought from other people: per node, and summed in
        # the manifest, so the bill is a property of the run rather than
        # something a reader reconstructs from items.
        self.spend_by_node: dict[str, dict[str, Any]] = {}

    def read_forced_restart(self, resume_mod) -> set[str]:
        """The nodes that must restart whatever a resume map offers.

        A consumer may require a minimum resume level of an upstream node
        (`require_resume: {port: level}`). A requirement the upstream
        block cannot meet forces that node to restart rather than reuse a
        partial; unknown level names refuse.
        """
        forced: set[str] = set()
        for nid_c, node_c in self.nodes.items():
            req = (node_c.get("params") or {}).get("require_resume") or {}
            for port, level in req.items():
                src = next((e["from"]["node"] for e in self.edges
                            if e["to"]["node"] == nid_c
                            and e["to"]["port"] == port), None)
                if src is None:
                    continue
                offered = resume_mod.resume_level(self.nodes[src]["block"],
                                                  self.nodes[src].get("params"),
                                                  self.nodes[src].get("inputs"))
                if not resume_mod.satisfies(offered, str(level)):
                    forced.add(src)
        return forced
