"""`~canonical/ops/chat/1` end to end (task 000337): the endpoint
ModelRef, the remote path through the executor, ordering under
concurrency, resume, the budget, cassettes, and the local path.

The provider is the mock, so the whole file runs offline and free.
"""

from __future__ import annotations

import pytest

from mechbench_compute import chat as chat_mod
from mechbench_compute import model_ref as mr
from mechbench_compute import resume as resume_mod
from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec
from mechbench_compute.providers.errors import BudgetExceeded

ENDPOINT = {"provider": "mock", "model": "mock-large"}


def _records(n=3):
    return [{"id": f"r{i}", "coords": {"arm": "a" if i % 2 else "b"},
             "system": "be brief", "user": f"question {i}"} for i in range(n)]


def _spec(params, records=None):
    graph = {"nodes": [{"id": "chat", "block": "~canonical/ops/chat/1",
                        "params": {"model": ENDPOINT, "budget_usd": 5.0,
                                   "records": records if records is not None
                                   else _records(), **params}}],
             "edges": []}
    return ProtocolSpec(kind="pipeline", prompt="", model_id=None,
                        extra={"graph": graph})


class TestEndpointModelRef:
    def test_an_endpoint_ref_parses_and_round_trips(self):
        ref = mr.parse({"provider": "anthropic", "model": "claude-opus-5",
                        "provider_options": {"anthropic": {"thinking": {}}}})
        assert ref.is_endpoint and ref.provider == "anthropic"
        assert ref.base == "claude-opus-5"
        assert mr.parse(ref.to_wire()) == ref
        assert ref.describe() == "anthropic:claude-opus-5"

    def test_an_endpoint_carries_no_adapters_and_fetches_nothing(self):
        with pytest.raises(ValueError, match="cannot carry adapters"):
            mr.parse({"provider": "openai", "model": "gpt-5",
                      "adapters": [{"bench": "x/y"}]})

        def boom(_label):
            raise AssertionError("an endpoint must not fetch anything")

        assert mr.resolve({"provider": "openai", "model": "gpt-5"}, fetch=boom)

    def test_an_unknown_provider_and_a_missing_model_are_refused(self):
        with pytest.raises(ValueError, match="unknown provider"):
            mr.parse({"provider": "altavista", "model": "x"})
        with pytest.raises(ValueError, match="needs a model"):
            mr.parse({"provider": "openai"})

    def test_a_local_block_refuses_an_endpoint_by_name(self):
        ref = mr.parse(ENDPOINT)
        with pytest.raises(ValueError, match=r"ops/chat/1"):
            ProtocolExecutor()._model_loaded(ref)

    def test_the_resume_level_depends_on_who_answers(self):
        assert resume_mod.resume_level("~canonical/ops/chat/1",
                                       {"model": ENDPOINT}) == "exchangeable"
        assert resume_mod.resume_level("~canonical/ops/chat/1",
                                       {"model": "google/gemma-3-4b-it"}) == "reproducible"
        assert resume_mod.item_resumable("~canonical/ops/chat/1")


class TestRemoteNodeThroughTheExecutor:
    def test_a_remote_chat_node_runs_end_to_end(self):
        out = ProtocolExecutor().run(_spec({"n": 2, "max_tokens": 64}))
        node = out.payload["outputs"]["chat"]
        assert node["kind"] == "document_collection"
        assert [i["id"] for i in node["items"]] == [
            "r0-s0", "r0-s1", "r1-s0", "r1-s1", "r2-s0", "r2-s1"]
        assert all(i["text"] for i in node["items"])
        # Provenance per item: who answered, what it used, what it cost.
        call = node["items"][0]["metadata"]["call"]
        assert call["provider"] == "mock" and call["model_version"].endswith("mock-20260908")
        assert call["cost_usd"] > 0 and call["usage"]["output_tokens"] > 0
        # And the node's own bill, summed into the manifest.
        spend = node["spend"]
        assert spend["calls"] == 6 and spend["cost_usd"] > 0
        total = out.payload["resources"]["spend"]
        assert total["cost_usd"] == spend["cost_usd"]
        assert total["by_provider"]["mock"]["calls"] == 6
        assert total["by_node"]["chat"]["provider"] == "mock"

    def test_output_order_is_canonical_under_concurrency(self):
        recs = _records(12)
        serial = ProtocolExecutor().run(_spec({"concurrency": 1}, recs))
        parallel = ProtocolExecutor().run(_spec({"concurrency": 8}, recs))
        assert ([i["id"] for i in parallel.payload["outputs"]["chat"]["items"]]
                == [i["id"] for i in serial.payload["outputs"]["chat"]["items"]])
        assert ([i["text"] for i in parallel.payload["outputs"]["chat"]["items"]]
                == [i["text"] for i in serial.payload["outputs"]["chat"]["items"]])

    def test_items_reach_the_spool_as_they_land(self):
        spooled = []
        ex = ProtocolExecutor(on_spool_item=lambda nid, key, item:
                              spooled.append((nid, key)))
        ex.run(_spec({"n": 2}))
        assert len(spooled) == 6
        assert spooled[0][0] == "chat"
        assert {k for _, k in spooled} == {f"r{i}:{k}" for i in range(3)
                                           for k in range(2)}

    def test_a_spooled_item_is_reused_and_not_bought_again(self):
        ref = mr.parse(ENDPOINT)
        first = chat_mod.run_remote(ref, _records(3), {"budget_usd": 1.0})
        keep = {"r0:0": first["items"][0], "r1:0": first["items"][1]}
        again = chat_mod.run_remote(ref, _records(3), {"budget_usd": 1.0},
                                    resume_items=keep)
        assert again["spend"]["calls"] == 1          # only r2 was bought
        assert again["items"][:2] == first["items"][:2]

    def test_the_budget_refuses_the_call_that_would_cross_it(self):
        # Every mock call costs a little; a cap of two calls' worth
        # stops the third before it is made.
        one = chat_mod.run_remote(mr.parse(ENDPOINT), _records(1),
                                  {"budget_usd": 1.0, "max_tokens": 32})
        each = one["spend"]["cost_usd"]
        with pytest.raises(BudgetExceeded):
            chat_mod.run_remote(mr.parse(ENDPOINT), _records(6),
                                {"budget_usd": each * 2.5, "max_tokens": 32,
                                 "concurrency": 1})

    def test_a_node_without_a_budget_is_refused_in_the_executor_too(self):
        spec = _spec({})
        spec.extra["graph"]["nodes"][0]["params"].pop("budget_usd")
        with pytest.raises(ValueError, match="budget_usd"):
            ProtocolExecutor().run(spec)


class TestCassetteAndDryRun:
    def test_a_recorded_run_replays_with_no_credential_and_no_spend(self):
        from mechbench_compute.providers.cassette import Cassette

        tape = Cassette(provider="mock", label="fixtures/chat")
        recorded = chat_mod.run_remote(
            mr.parse(ENDPOINT), _records(2),
            {"budget_usd": 1.0, "cassette_mode": "record"}, cassette=tape)
        wire = tape.to_wire()

        replayed = chat_mod.run_remote(
            mr.parse(ENDPOINT), _records(2),
            {"budget_usd": 1.0, "cassette_mode": "replay"}, cassette=wire)
        assert ([i["text"] for i in replayed["items"]]
                == [i["text"] for i in recorded["items"]])
        assert replayed["spend"]["replayed"] == 2

    def test_a_dry_run_says_so_and_still_prices_the_work(self):
        out = chat_mod.run_remote(
            mr.parse({"provider": "anthropic", "model": "claude-opus-5"}),
            _records(2), {"budget_usd": 5.0, "dry_run": True})
        assert out["spend"]["dry_run"] is True
        assert out["spend"]["cost_usd"] > 0
        assert out["items"][0]["metadata"]["call"]["provider"] == "anthropic"


class TestLocalPath:
    def test_the_local_path_renders_a_conversation_and_needs_no_provider(
            self, monkeypatch):
        from mechbench_compute import distill, generate

        rendered: list[str] = []

        class FakeTok:
            def apply_chat_template(self, turns, tokenize=False,
                                     add_generation_prompt=True, **kw):
                rendered.append(" | ".join(f"{t['role']}:{t['content']}"
                                           for t in turns))
                return rendered[-1]

        class FakeModel:
            tokenizer = FakeTok()

        monkeypatch.setattr(distill, "encode", lambda tok, text: [1, 2, 3])
        monkeypatch.setattr(distill, "prefill_decision", lambda m, ids: None)
        monkeypatch.setattr(generate, "sample_completion_cached",
                            lambda *a, **k: "a local answer")
        out = chat_mod.run_local(FakeModel(), mr.parse("google/gemma-3-4b-it"),
                                 _records(2), {"n": 1, "seed": 3})
        assert [i["text"] for i in out["items"]] == ["a local answer"] * 2
        # System merges into the first user turn, as render_chat has
        # always driven these templates.
        assert rendered[0] == "user:be brief\n\nquestion 0"
        assert "spend" not in out       # nothing was bought

    def test_tools_are_refused_locally_with_the_task_that_will_add_them(self):
        from mechbench_compute.providers.errors import CapabilityUnsupported

        with pytest.raises(CapabilityUnsupported, match="000339"):
            chat_mod.run_local(None, mr.parse("x/y"), _records(1),
                               {"tools": [{"name": "grep"}]})
