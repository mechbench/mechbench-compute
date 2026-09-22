"""A graph that cannot run says so before it runs anything.

Resolving blocks node by node, in execution order, means a protocol
whose last node is misspelled computes everything upstream of it first —
hours of work to arrive at a refusal that was decidable at load. So
every node's operation is checked once, up front, and every bad one is
reported: a protocol carried forward from retired spellings usually has
more than one.
"""

from __future__ import annotations

import pytest

from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec

CROSS = {"factors": [{"name": "x", "levels": [{"key": "a"}, {"key": "b"}]}]}


def _spec(*blocks: str):
    """A chain of `records/*` nodes, the first a factor cross, wired in
    order — so anything that DOES run leaves a trace in `results`."""
    nodes, edges = [], []
    for i, block in enumerate(blocks):
        nodes.append({"id": f"n{i}", "block": block,
                      "params": dict(CROSS) if i == 0 else {}})
        if i:
            edges.append({"from": {"node": f"n{i-1}", "port": "records"},
                          "to": {"node": f"n{i}", "port": "records"},
                          "kind": "records"})
    return ProtocolSpec(kind="pipeline", prompt="", model_id=None,
                        extra={"graph": {"nodes": nodes, "edges": edges}})


class TestPreflight:
    def test_a_bad_last_node_refuses_before_the_first_one_runs(self, monkeypatch):
        ran = []
        from mechbench_compute import blocks as blocks_mod

        real = blocks_mod.PURE_BLOCKS["records/cross"]
        monkeypatch.setitem(
            blocks_mod.PURE_BLOCKS, "records/cross",
            lambda inputs, params: (ran.append(1), real(inputs, params))[1])

        with pytest.raises(ValueError, match="cannot run"):
            ProtocolExecutor().run(_spec("records/cross", "records/selekt"))
        assert ran == [], "a node ran before the graph was checked"

    def test_the_message_names_the_node_and_what_to_write(self):
        with pytest.raises(ValueError) as e:
            ProtocolExecutor().run(_spec("records/cross", "records/stats"))
        msg = str(e.value)
        assert "n1" in msg                       # which node
        assert "records/stats" in msg            # what it said
        assert "records/summarize" in msg        # what to write
        assert "0.82.0" in msg                   # and when it stopped working

    def test_every_bad_node_is_reported_not_just_the_first(self):
        with pytest.raises(ValueError) as e:
            ProtocolExecutor().run(
                _spec("records/cross", "records/stats", "records/table"))
        msg = str(e.value)
        assert "2 problems found before anything ran" in msg
        assert "records/summarize" in msg and "records/tabulate" in msg

    def test_a_node_with_no_block_is_named_too(self):
        spec = ProtocolSpec(kind="pipeline", prompt="", model_id=None, extra={
            "graph": {"nodes": [{"id": "nameless", "params": {}}], "edges": []}})
        with pytest.raises(ValueError, match="nameless: no block"):
            ProtocolExecutor().run(spec)

    def test_a_good_graph_is_untouched(self):
        out = ProtocolExecutor().run(_spec("records/cross", "records/select"))
        assert out.payload["nodes_executed"] == ["n0", "n1"]


class TestParamsAndPorts:
    """A param no block accepts, an edge onto a port that does not
    exist, and a required port with nothing on it are all decidable at
    load: params are static and the wiring is the graph."""

    def test_the_failure_that_motivated_this(self, monkeypatch):
        # The 014 trace: an August graph whose LAST node passed
        # `template` to a block that lost the param. It ran for a day.
        ran = []
        from mechbench_compute import blocks as blocks_mod

        real = blocks_mod.PURE_BLOCKS["records/cross"]
        monkeypatch.setitem(
            blocks_mod.PURE_BLOCKS, "records/cross",
            lambda inputs, params: (ran.append(1), real(inputs, params))[1])

        spec = ProtocolSpec(kind="pipeline", prompt="", model_id=None, extra={
            "graph": {"nodes": [
                {"id": "design", "block": "records/cross", "params": dict(CROSS)},
                {"id": "cap", "block": "activations/capture",
                 "params": {"model": "m", "layers": [12], "template": "chat"},
                 "inputs": {"records": [{"id": "r", "user": "u"}]}},
            ], "edges": []}})
        with pytest.raises(ValueError) as e:
            ProtocolExecutor().run(spec)
        msg = str(e.value)
        assert "cap (activations/capture)" in msg
        assert "'template'" in msg
        assert ran == [], "the upstream node ran before the graph was checked"

    def test_a_param_error_and_a_block_error_are_reported_together(self):
        spec = ProtocolSpec(kind="pipeline", prompt="", model_id=None, extra={
            "graph": {"nodes": [
                {"id": "a", "block": "records/stats", "params": {}},
                {"id": "b", "block": "records/cross",
                 "params": {**CROSS, "facters": []}},
            ], "edges": []}})
        with pytest.raises(ValueError) as e:
            ProtocolExecutor().run(spec)
        msg = str(e.value)
        assert "2 problems" in msg
        assert "records/summarize" in msg      # the unresolvable block
        assert "facters" in msg                 # the misspelled param

    def test_an_edge_onto_a_port_that_does_not_exist(self):
        spec = ProtocolSpec(kind="pipeline", prompt="", model_id=None, extra={
            "graph": {"nodes": [
                {"id": "design", "block": "records/cross", "params": dict(CROSS)},
                {"id": "pick", "block": "records/select", "params": {}},
            ], "edges": [{"from": {"node": "design", "port": "records"},
                          "to": {"node": "pick", "port": "recrods"},
                          "kind": "records"}]}})
        with pytest.raises(ValueError, match="no input port 'recrods'"):
            ProtocolExecutor().run(spec)

    def test_a_required_port_with_nothing_on_it(self):
        spec = ProtocolSpec(kind="pipeline", prompt="", model_id=None, extra={
            "graph": {"nodes": [
                {"id": "pick", "block": "records/select", "params": {}},
            ], "edges": []}})
        with pytest.raises(ValueError, match="needs an input on its 'records' port"):
            ProtocolExecutor().run(spec)

    def test_an_optional_port_left_empty_is_fine(self):
        # `records/select` takes one port; `text/measure`'s second is
        # optional, and a graph that leaves it alone is runnable.
        out = ProtocolExecutor().run(ProtocolSpec(
            kind="pipeline", prompt="", model_id=None, extra={
                "graph": {"nodes": [
                    {"id": "design", "block": "records/cross", "params": dict(CROSS)},
                    {"id": "pick", "block": "records/select",
                     "params": {"where": {"x": "a"}}},
                ], "edges": [{"from": {"node": "design", "port": "records"},
                              "to": {"node": "pick", "port": "records"},
                              "kind": "records"}]}}))
        assert out.payload["nodes_executed"] == ["design", "pick"]

    def test_a_wildcard_op_with_no_edges_says_so(self):
        # `records/union` names its ports freely; what it needs is that
        # there be some.
        spec = ProtocolSpec(kind="pipeline", prompt="", model_id=None, extra={
            "graph": {"nodes": [
                {"id": "merged", "block": "records/union", "params": {}},
            ], "edges": []}})
        with pytest.raises(ValueError, match="at least one input edge"):
            ProtocolExecutor().run(spec)
