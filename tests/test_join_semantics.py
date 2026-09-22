"""Joins under partial failure."""
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
    from mechbench_compute.ops.records import fill

    real = fill.run
    seen = {"ran": []}

    def flaky(ctx, inputs, params):
        if params.get("templates", {}).get("user", "").startswith("{x}"):
            raise RuntimeError("this branch died")
        seen["ran"].append(params["templates"]["user"])
        return real(ctx, inputs, params)

    monkeypatch.setattr(fill, "run", flaky)
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
        # safe: siblings settle before the status does.
        from mechbench_compute.ops.records import fill

        real = fill.run
        ran = []

        def flaky(ctx, inputs, params):
            user = params.get("templates", {}).get("user", "")
            if user.startswith("{x}"):
                raise RuntimeError("this branch died")
            ran.append(user)
            return real(ctx, inputs, params)

        monkeypatch.setattr(fill, "run", flaky)
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


class TestWithResultsStored:
    """Every other test here hands the graph straight to the executor,
    which stores nothing. A real job emits each node's result to the
    bench and cites its upstreams as lineage — and an upstream that
    produced nothing has no path to cite. That gap failed the first
    stored run of a placeholder join with a bare `KeyError: 'down'`,
    which is to say: the policy that exists to keep a run alive killed
    it, and only where nobody was looking."""

    def test_a_placeholder_join_still_emits(self, monkeypatch):
        from mechbench_compute import bench

        emitted: list[str] = []
        lineage: dict[str, list[str]] = {}

        def emit(path, payload, **kw):
            emitted.append(path)
            lineage[path] = list(kw.get("inputs") or [])
            return {"path": path}

        monkeypatch.setattr(bench, "emit", emit)
        spec = ProtocolSpec(kind="pipeline", prompt="", model_id=None,
                            extra={"graph": _graph("placeholder"),
                                   "resultPath": "u/p/results/j_1"})
        from mechbench_compute.ops.records import fill

        real = fill.run

        def flaky(ctx, inputs, params):
            if params.get("templates", {}).get("user", "").startswith("{x}"):
                raise RuntimeError("this branch died")
            return real(ctx, inputs, params)

        monkeypatch.setattr(fill, "run", flaky)
        out = ProtocolExecutor().run(spec).payload

        assert "u/p/results/j_1/pairs" in emitted
        assert "u/p/results/j_1/bad" not in emitted   # it never ran
        # Lineage names the input that exists and says nothing about the
        # one that does not; `nodes_missing` is where the absence lives.
        assert lineage["u/p/results/j_1/pairs"] == ["u/p/results/j_1/good"]
        assert out["nodes_missing"]["bad"]["reason"].startswith("RuntimeError")


class TestTheWholePathWithoutAMonkeypatch:
    """What the docs example rehearses: two provider branches, one of
    them down, zipped and judged. Nothing here is patched — the mock
    refuses because the graph asked it to — so this is the shape a
    protocol author can write and check before spending anything."""

    ENDPOINT = {"provider": "mock", "model": "mock-large"}

    def _graph(self, judge_policy):
        def chat(nid, **options):
            return {"id": nid, "block": "text/chat",
                    "params": {"model": self.ENDPOINT, "budget_usd": 1.0,
                               "max_tokens": 32,
                               "provider_options": {"mock": options}},
                    "inputs": {"records": [
                        {"id": "r1", "coords": {"prompt": "p1"}, "user": "hi"}]}}
        return {"nodes": [
            chat("up", text="a real answer"),
            chat("down", fail="the provider is down"),
            {"id": "pairs", "block": "records/zip",
             "params": {"by": ["prompt"], "flatten": True,
                        "on_mismatch": "placeholder"}},
            {"id": "sides", "block": "records/rename",
             "params": {"fields": {"up_text": "text_a", "down_text": "text_b"}}},
            {"id": "verdicts", "block": "eval/judge",
             "params": {"judge": {"model": self.ENDPOINT},
                        "rubric": "which is better written",
                        "scale": {"type": "pairwise"}, "budget_usd": 1.0,
                        "on_missing": judge_policy}},
        ], "edges": [
            {"from": {"node": "up", "port": "documents"},
             "to": {"node": "pairs", "port": "branches"},
             "kind": "text/document", "index": 0, "on_missing": "placeholder"},
            {"from": {"node": "down", "port": "documents"},
             "to": {"node": "pairs", "port": "branches"},
             "kind": "text/document", "index": 1, "on_missing": "placeholder"},
            {"from": {"node": "pairs", "port": "records"},
             "to": {"node": "sides", "port": "records"},
             "kind": "records/record"},
            {"from": {"node": "sides", "port": "records"},
             "to": {"node": "verdicts", "port": "records"},
             "kind": "records/record"},
        ]}

    def test_the_missing_side_is_never_quietly_judged(self):
        with pytest.raises(ValueError, match="no text_a and text_b to judge"):
            ProtocolExecutor().run(ProtocolSpec(
                kind="pipeline", prompt="", model_id=None,
                extra={"graph": self._graph("error")}))

    def test_skip_lands_a_result_that_names_the_arm_that_went_down(self):
        out = ProtocolExecutor().run(ProtocolSpec(
            kind="pipeline", prompt="", model_id=None,
            extra={"graph": self._graph("skip")})).payload
        assert out["nodes_missing"]["down"]["reason"].startswith("RuntimeError")
        verdicts = out["outputs"]["verdicts"]
        assert verdicts["summary"]["n_unjudged"] == 1
        assert verdicts["items"][0]["missing"] == ["text_b"]


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
