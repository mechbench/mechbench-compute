"""Joins under partial failure (task 000399)."""
from __future__ import annotations

import pytest

from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec

CROSS = {"factors": [{"name": "x", "levels": [{"key": "a"}, {"key": "b"}]}]}


def _graph(policy=None):
    """design → good, bad → zip(branches), with the policy on the edge
    from the branch that breaks. `bad`'s template is the one the test's
    stand-in block raises on."""
    bad_params = {"templates": {"user": "{x}"}}
    edge = {"from": {"node": "bad", "port": "records"},
            "to": {"node": "pairs", "port": "branches"}, "kind": "records",
            "index": 1}
    if policy:
        edge["on_missing"] = policy
    return {"nodes": [
        {"id": "design", "block": "records/cross", "params": dict(CROSS)},
        {"id": "good", "block": "records/fill",
         "params": {"templates": {"user": "good {x}"}}},
        {"id": "bad", "block": "records/fill", "params": bad_params},
        {"id": "pairs", "block": "records/zip",
         "params": {"names": ["good", "bad"], "on_mismatch": "placeholder"}},
    ], "edges": [
        {"from": {"node": "design", "port": "records"},
         "to": {"node": "good", "port": "records"}, "kind": "records"},
        {"from": {"node": "design", "port": "records"},
         "to": {"node": "bad", "port": "records"}, "kind": "records"},
        {"from": {"node": "good", "port": "records"},
         "to": {"node": "pairs", "port": "branches"}, "kind": "records",
         "index": 0},
        edge,
    ]}


def _run(graph, monkeypatch, break_node="bad"):
    """Run the graph with one node's block made to fail."""
    from mechbench_compute import blocks as blocks_mod

    real = blocks_mod.PURE_BLOCKS["records/fill"]
    seen = {"ran": []}

    def flaky(inputs, params):
        if params.get("templates", {}).get("user", "").startswith("{x}"):
            raise RuntimeError("this branch died")
        seen["ran"].append(params["templates"]["user"])
        return real(inputs, params)

    monkeypatch.setitem(blocks_mod.PURE_BLOCKS, "records/fill", flaky)
    out = ProtocolExecutor().run(ProtocolSpec(
        kind="pipeline", prompt="", model_id=None, extra={"graph": graph}))
    return out, seen


class TestDefaultIsUnchanged:
    def test_a_failed_branch_fails_the_run(self, monkeypatch):
        with pytest.raises(RuntimeError, match="this branch died"):
            _run(_graph(), monkeypatch)

    def test_the_sibling_branch_finishes_first(self, monkeypatch):
        # The failure is raised at the end of the run, so the good branch
        # has already run — which is what makes the parallel scheduler
        # (000396) safe: siblings settle before the status does.
        from mechbench_compute import blocks as blocks_mod

        real = blocks_mod.PURE_BLOCKS["records/fill"]
        ran = []

        def flaky(inputs, params):
            user = params.get("templates", {}).get("user", "")
            if user.startswith("{x}"):
                raise RuntimeError("this branch died")
            ran.append(user)
            return real(inputs, params)

        monkeypatch.setitem(blocks_mod.PURE_BLOCKS, "records/fill", flaky)
        with pytest.raises(RuntimeError, match="this branch died"):
            ProtocolExecutor().run(ProtocolSpec(
                kind="pipeline", prompt="", model_id=None,
                extra={"graph": _graph()}))
        assert ran == ["good {x}"], "the sibling never ran"


class TestSkip:
    def test_the_consumer_is_skipped_and_the_run_completes(self, monkeypatch):
        out, _seen = _run(_graph("skip"), monkeypatch)
        payload = out.payload
        assert "pairs" not in payload["nodes_executed"]
        assert "good" in payload["nodes_executed"]
        assert payload["nodes_missing"]["bad"]["reason"].startswith("RuntimeError")
        assert payload["nodes_missing"]["pairs"]["source"] == ["bad"]


class TestPlaceholder:
    def test_the_consumer_runs_with_an_empty_stand_in(self, monkeypatch):
        out, _seen = _run(_graph("placeholder"), monkeypatch)
        payload = out.payload
        assert "pairs" in payload["nodes_executed"]
        pairs = payload["outputs"]["pairs"]
        # Every key is present from the good branch and missing from the
        # bad one — zip's own policy, over a branch that is empty.
        assert len(pairs["items"]) == 2
        assert pairs["items"][0]["branches"]["bad"] == {"missing": True}
        assert pairs["items"][0]["branches"]["good"]["user"].startswith("good")
        assert payload["nodes_missing"]["bad"]["reason"].startswith("RuntimeError")


class TestValidation:
    def test_an_unknown_policy_is_refused_at_load(self):
        g = _graph("shrug")
        with pytest.raises(ValueError, match="on_missing is"):
            ProtocolExecutor().run(ProtocolSpec(
                kind="pipeline", prompt="", model_id=None, extra={"graph": g}))

    def test_placeholder_on_a_singular_port_is_refused(self):
        g = {"nodes": [
            {"id": "design", "block": "records/cross", "params": dict(CROSS)},
            {"id": "fit", "block": "direction/fit", "params": {"axis": "x"}},
            {"id": "vocab", "block": "direction/unembed",
             "params": {"model": "acme/tiny"}},
        ], "edges": [
            {"from": {"node": "design", "port": "records"},
             "to": {"node": "fit", "port": "vectors"}, "kind": "records"},
            {"from": {"node": "fit", "port": "direction"},
             "to": {"node": "vocab", "port": "direction"},
             "kind": "direction/vector", "on_missing": "placeholder"},
        ]}
        with pytest.raises(ValueError, match="no empty value"):
            ProtocolExecutor().run(ProtocolSpec(
                kind="pipeline", prompt="", model_id=None, extra={"graph": g}))
