"""A graph that cannot run says so before it runs anything (000512).

Block resolution used to happen node by node, in execution order, so a
protocol whose last node was misspelled computed everything upstream of
it first — one 014 trace spent five resumes and the better part of a day
to arrive at a refusal that was decidable at load. Every node's
operation is checked once, up front, and every bad one is reported: a
protocol being carried forward from retired spellings usually has more
than one.
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

        with pytest.raises(ValueError, match="do not exist"):
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
        assert "2 operations that do not exist" in msg
        assert "records/summarize" in msg and "records/tabulate" in msg

    def test_a_node_with_no_block_is_named_too(self):
        spec = ProtocolSpec(kind="pipeline", prompt="", model_id=None, extra={
            "graph": {"nodes": [{"id": "nameless", "params": {}}], "edges": []}})
        with pytest.raises(ValueError, match="nameless: no block"):
            ProtocolExecutor().run(spec)

    def test_a_good_graph_is_untouched(self):
        out = ProtocolExecutor().run(_spec("records/cross", "records/select"))
        assert out.payload["nodes_executed"] == ["n0", "n1"]
