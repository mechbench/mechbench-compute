from __future__ import annotations

import pytest

from mechbench_compute import lexicon, ops
from mechbench_compute.lexicon import kinds as K
from mechbench_compute.protocol import ProtocolExecutor
from mechbench_compute.providers.budget import Budget

RECORDS = K.collection("records/record", [{"id": "a", "v": 1}, {"id": "b", "v": 3}])
SORT = {"by": ["-v"], "limit": 1}


def lend(executor, *needs, **lent):
    return ops.Context.for_op(lexicon.Op("x/y", "s", "d", (), needs=frozenset(needs)),
                              executor, **lent)


class TestEachMemberAsksForItsNeed:
    @pytest.mark.parametrize("call", [
        lambda ctx: ctx.sub("records/sort", {}, {}),
        lambda ctx: ctx.provider({"provider": "mock", "model": "mock-large"}),
        lambda ctx: ctx.memo("o/p/memos/m"),
        lambda ctx: ctx.materialize("o/p/checkpoints/c"),
        lambda ctx: ctx.evict_model(),
    ])
    def test_an_undeclared_member_raises(self, call):
        with pytest.raises(ops.NeedNotDeclared, match="x/y did not declare"):
            call(lend(ProtocolExecutor()))

    def test_no_member_is_named_by_no_need(self):
        assert set(ops.MEMBER_NEEDS) == {"model", "loaded", "evict_model", "executor", "sub",
                                         "provider", "memo", "materialize", "secrets"}
        assert ops.MEMBER_NEEDS["executor"] == {"executor.sub"}


class TestSub:
    def test_an_op_runs_as_a_child(self):
        out = lend(ProtocolExecutor(), "executor.sub").sub("records/sort", {"records": RECORDS}, SORT)
        assert [i["id"] for i in out["items"]] == ["b"]

    def test_a_graph_runs_as_a_child_and_answers_with_its_outputs(self):
        graph = {"nodes": [{"id": "top", "block": "records/sort", "params": SORT,
                            "inputs": {"records": RECORDS}}], "edges": []}
        out = lend(ProtocolExecutor(), "executor.sub").sub(graph, {}, {})
        assert [i["id"] for i in out["top"]["items"]] == ["b"]

    def test_the_child_shares_the_model_and_hands_back_what_it_loaded(self, monkeypatch):
        seen = {}

        def run_op(self, block, inputs, params, **lent):
            seen["model"] = self._model
            self._model, self._model_id = "loaded-by-child", "m2"
            return {}

        monkeypatch.setattr(ProtocolExecutor, "_run_op", run_op)
        ex = ProtocolExecutor()
        ex._model, ex._model_id = "resident", "m1"
        lend(ex, "executor.sub").sub("records/sort", {}, {})
        assert seen["model"] == "resident"
        assert (ex._model, ex._model_id) == ("loaded-by-child", "m2")

    def test_a_budget_is_a_child_of_the_run_budget(self, monkeypatch):
        seen = {}

        def run_op(self, block, inputs, params, **lent):
            seen["budget"], seen["limiter"] = self._budget, self._limiter
            return {}

        monkeypatch.setattr(ProtocolExecutor, "_run_op", run_op)
        run_budget, limiter = Budget(cap_usd=5.0), object()
        ctx = lend(ProtocolExecutor(limiter=limiter, budget=run_budget), "executor.sub")
        ctx.sub("records/sort", {}, {})
        assert seen["budget"] is run_budget and seen["limiter"] is limiter
        ctx.sub("records/sort", {}, {}, budget=2.0)
        assert seen["budget"].parent is run_budget and seen["budget"].cap_usd == 2.0

    def test_the_child_carries_the_run_secrets_the_op_did_not_declare(self, monkeypatch):
        seen = {}

        def run_op(self, block, inputs, params, **lent):
            seen.update(lent)
            return {}

        monkeypatch.setattr(ProtocolExecutor, "_run_op", run_op)
        lend(ProtocolExecutor(), "executor.sub", secrets={"hf": {"token": "t"}}).sub(
            "records/sort", {}, {})
        assert seen["secrets"] == {"hf": {"token": "t"}}


class TestProvider:
    def test_the_client_carries_the_limiter_and_the_run_budget(self):
        run_budget, limiter = Budget(cap_usd=5.0), object()
        client = lend(ProtocolExecutor(limiter=limiter, budget=run_budget),
                      "provider.chat", "secrets", secrets={"mock": {}}).provider(
            {"provider": "mock", "model": "mock-large"})
        assert client.limiter is limiter and client.job_budget is run_budget
        assert client.secrets == {"mock": {}}

    def test_a_call_without_declared_secrets_is_refused(self):
        client = lend(ProtocolExecutor(), "provider.chat", secrets={"mock": {}}).provider(
            {"provider": "mock", "model": "mock-large"})
        with pytest.raises(ops.NeedNotDeclared, match="secrets"):
            client.chat([{"id": "r", "user": "hi"}], {"budget_usd": 1.0})

    def test_local_weights_are_not_a_provider(self):
        with pytest.raises(ValueError, match="local weights"):
            lend(ProtocolExecutor(), "provider.chat").provider("google/gemma-3-4b-it")


class TestTheRest:
    def test_no_cache_is_no_memo(self):
        assert lend(ProtocolExecutor(), "memo").memo(None) is None

    def test_materialize_asks_the_executor(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ProtocolExecutor, "_materialize_checkpoint", lambda self, label: tmp_path / label)
        assert lend(ProtocolExecutor(), "objects.read").materialize("c") == tmp_path / "c"

    def test_evict_model_clears_the_resident_model(self):
        ex = ProtocolExecutor()
        ex._model, ex._model_id = "resident", "m1"
        ctx = lend(ex, "model.backward", loaded="resident")
        ctx.evict_model()
        assert (ex._model, ex._model_id, ctx.loaded) == (None, None, None)


class TestTheLocalChatIsFusedByTheExecutor:
    def test_a_local_model_is_fused_and_an_endpoint_is_not(self, monkeypatch):
        from mechbench_compute import chat as chat_mod

        fused, ran = [], []

        def run_model_block(self, fn, inputs, params, *a, **kw):
            fused.append(params["model"])
            return fn(inputs, params, *a, **kw)

        monkeypatch.setattr(ProtocolExecutor, "_run_model_block", run_model_block)
        monkeypatch.setattr(ProtocolExecutor, "_model_loaded", lambda self, ref: "weights")
        monkeypatch.setattr(chat_mod, "run_local",
                            lambda model, ref, records, params, **kw: ran.append((model, ref.base)) or {})
        monkeypatch.setattr(chat_mod, "run_remote", lambda ref, records, params, **kw: {})
        ex = ProtocolExecutor()
        ex._run_op("text/chat", {"records": []}, {"model": "org/local"})
        ex._run_op("text/chat", {"records": []},
                   {"model": {"provider": "mock", "model": "mock-large"}, "budget_usd": 1.0})
        assert fused == ["org/local"]
        assert ran == [("weights", "org/local")]
