from __future__ import annotations

from typing import Any

from mechbench_compute import dataflow
from mechbench_compute.protocol.check_graph import check_graph
from mechbench_compute.protocol.sort_nodes import sort_nodes


class RunState:
    def __init__(self, spec, resume=None) -> None:
        from mechbench_compute import resume as resume_mod

        extra = self.extra = spec.extra or {}
        dataflow.check_form(extra)
        self.bound_params = extra.get("params") or {}
        graph = dataflow.lower(extra.get("graph") or {}, extra.get("inputs") or {})
        self.graph = graph
        self.nodes: dict[str, Any] = {n["id"]: n for n in graph.get("nodes", [])}
        self.edges: list[dict] = graph.get("edges", [])
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
        self.keep = str(extra.get("keep") or "all")
        if self.keep not in ("all", "outputs"):
            raise ValueError(f"keep must be 'all' or 'outputs', not {self.keep!r}")
        self.discard = self.keep == "outputs"
        self.result_base = extra.get("resultPath")
        self.order = sort_nodes(self.nodes, self.edges)
        check_graph(self.nodes, self.edges, self.order)
        dataflow.check_refs(self.nodes, self.bound_params)
        self.resume = resume or {}
        self.forced_restart = self.read_forced_restart(resume_mod)

        self.results: dict[str, Any] = {}
        self.node_hashes: dict[str, str] = {}
        self.node_paths: dict[str, str] = {}
        self.held: dict[str, tuple[str, Any]] = {}
        self.missing: dict[str, dict[str, Any]] = {}
        self.failures: dict[str, BaseException] = {}
        self.tolerated: set[str] = set()
        self.ahead: dict[str, Any] = {}
        self.spend_by_node: dict[str, dict[str, Any]] = {}

    def read_forced_restart(self, resume_mod) -> set[str]:
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
