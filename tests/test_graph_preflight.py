from __future__ import annotations

import pytest

from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec

CROSS = {"factors": [{"name": "x", "levels": [{"key": "a"}, {"key": "b"}]}]}


def _spec(*blocks: str):
    nodes, edges = [], []
    for i, block in enumerate(blocks):
        nodes.append({"id": f"n{i}", "block": block,
                      "params": dict(CROSS) if i == 0 else {}})
        if i:
            edges.append({"from": {"node": f"n{i-1}", "port": "records"},
                          "to": {"node": f"n{i}", "port": "records"},
                          "kind": "records"})
    return ProtocolSpec(kind="pipeline", prompt="", model_id=None,
                        extra={"graph": {"dataflow": 2, "nodes": nodes, "edges": edges}})


class TestPreflight:
    def test_a_bad_last_node_refuses_before_the_first_one_runs(self, monkeypatch):
        ran = []
        from mechbench_compute.ops.records import cross

        real = cross.run
        monkeypatch.setattr(
            cross, "run",
            lambda ctx, inputs, params: (ran.append(1), real(ctx, inputs, params))[1])

        with pytest.raises(ValueError, match="cannot run"):
            ProtocolExecutor().run(_spec("records/cross", "records/selekt"))
        assert ran == [], "a node ran before the graph was checked"

    def test_the_message_names_the_node_and_what_to_write(self):
        with pytest.raises(ValueError) as e:
            ProtocolExecutor().run(_spec("records/cross", "records/stats"))
        msg = str(e.value)
        assert "n1" in msg
        assert "records/stats" in msg
        assert "records/summarize" in msg
        assert "0.82.0" in msg

    def test_every_bad_node_is_reported_not_just_the_first(self):
        with pytest.raises(ValueError) as e:
            ProtocolExecutor().run(
                _spec("records/cross", "records/stats", "records/table"))
        msg = str(e.value)
        assert "2 problems found before anything ran" in msg
        assert "records/summarize" in msg and "records/tabulate" in msg

    def test_a_node_with_no_block_is_named_too(self):
        spec = ProtocolSpec(kind="pipeline", prompt="", model_id=None, extra={
            "graph": {"dataflow": 2, "nodes": [{"id": "nameless", "params": {}}], "edges": []}})
        with pytest.raises(ValueError, match="nameless: no block"):
            ProtocolExecutor().run(spec)

    def test_a_good_graph_is_untouched(self):
        out = ProtocolExecutor().run(_spec("records/cross", "records/select"))
        assert out.payload["nodes_executed"] == ["n0", "n1"]


class TestParamsAndPorts:
    def test_the_failure_that_motivated_this(self, monkeypatch):
        ran = []
        from mechbench_compute.ops.records import cross

        real = cross.run
        monkeypatch.setattr(
            cross, "run",
            lambda ctx, inputs, params: (ran.append(1), real(ctx, inputs, params))[1])

        spec = ProtocolSpec(kind="pipeline", prompt="", model_id=None, extra={
            "graph": {"dataflow": 2, "nodes": [
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
            "graph": {"dataflow": 2, "nodes": [
                {"id": "a", "block": "records/stats", "params": {}},
                {"id": "b", "block": "records/cross",
                 "params": {**CROSS, "facters": []}},
            ], "edges": []}})
        with pytest.raises(ValueError) as e:
            ProtocolExecutor().run(spec)
        msg = str(e.value)
        assert "2 problems" in msg
        assert "records/summarize" in msg
        assert "facters" in msg

    def test_an_edge_onto_a_port_that_does_not_exist(self):
        spec = ProtocolSpec(kind="pipeline", prompt="", model_id=None, extra={
            "graph": {"dataflow": 2, "nodes": [
                {"id": "design", "block": "records/cross", "params": dict(CROSS)},
                {"id": "pick", "block": "records/select", "params": {}},
            ], "edges": [{"from": {"node": "design", "port": "records"},
                          "to": {"node": "pick", "port": "recrods"},
                          "kind": "records"}]}})
        with pytest.raises(ValueError, match="no input port 'recrods'"):
            ProtocolExecutor().run(spec)

    def test_a_required_port_with_nothing_on_it(self):
        spec = ProtocolSpec(kind="pipeline", prompt="", model_id=None, extra={
            "graph": {"dataflow": 2, "nodes": [
                {"id": "pick", "block": "records/select", "params": {}},
            ], "edges": []}})
        with pytest.raises(ValueError, match="needs an input on its 'records' port"):
            ProtocolExecutor().run(spec)

    def test_an_optional_port_left_empty_is_fine(self):
        out = ProtocolExecutor().run(ProtocolSpec(
            kind="pipeline", prompt="", model_id=None, extra={
                "graph": {"dataflow": 2, "nodes": [
                    {"id": "design", "block": "records/cross", "params": dict(CROSS)},
                    {"id": "pick", "block": "records/select",
                     "params": {"where": {"x": "a"}}},
                ], "edges": [{"from": {"node": "design", "port": "records"},
                              "to": {"node": "pick", "port": "records"},
                              "kind": "records"}]}}))
        assert out.payload["nodes_executed"] == ["design", "pick"]

    def test_a_wildcard_op_with_no_edges_says_so(self):
        spec = ProtocolSpec(kind="pipeline", prompt="", model_id=None, extra={
            "graph": {"dataflow": 2, "nodes": [
                {"id": "merged", "block": "records/union", "params": {}},
            ], "edges": []}})
        with pytest.raises(ValueError, match="at least one input edge"):
            ProtocolExecutor().run(spec)
