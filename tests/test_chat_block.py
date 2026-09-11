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

    def test_a_local_model_calls_tools_by_writing_them(self, monkeypatch):
        """A local model's tools go through its OWN chat template
        (epic 000439): the template declares them, renders the call and
        renders the result. The stub below is a Qwen-shaped template —
        it changes when `tools` is passed, and writes calls in Qwen's
        envelope — because a template that ignores `tools` is refused
        now, which is the next test."""
        from mechbench_compute import distill, generate

        prompts: list[str] = []
        replies = iter([
            'Let me work it out.\n<tool_call>\n'
            '{"name": "calc", "arguments": {"expression": "6*7"}}\n</tool_call>',
            "The answer is 42.",
        ])

        class QwenishTok:
            """Renders turns, and renders them DIFFERENTLY when tools
            are declared — which is how `probe_template` decides a
            model supports them."""

            def apply_chat_template(self, turns, tokenize=False,
                                     add_generation_prompt=True, tools=None,
                                     **kw):
                body = " | ".join(
                    f"{t['role']}:{t.get('content', '')}"
                    + (f"[calls:{[c['function']['name'] for c in t['tool_calls']]}]"
                       if t.get("tool_calls") else "")
                    for t in turns)
                if tools:
                    names = [t["function"]["name"] for t in tools]
                    body = f"<tools>{names}</tools> " + body
                    # A rendered call + result, so the probe can read
                    # this model's dialect off its own output.
                    if any(t.get("tool_calls") for t in turns):
                        body += ('<tool_call>\n{"name": "calc", "arguments": '
                                 '{"expression": "37 + 18"}}\n</tool_call>'
                                 "<tool_response>\n55\n</tool_response>")
                prompts.append(body)
                return body

        class FakeModel:
            tokenizer = QwenishTok()

        monkeypatch.setattr(distill, "encode", lambda tok, text: [1, 2, 3])
        monkeypatch.setattr(distill, "prefill_decision", lambda m, ids: None)
        monkeypatch.setattr(generate, "sample_completion_cached",
                            lambda *a, **k: next(replies))
        out = chat_mod.run_local(
            FakeModel(), mr.parse("Qwen/Qwen2.5-3B-Instruct"), _records(1),
            {"n": 1, "tools": ["calc"], "max_tool_rounds": 2})
        item = out["items"][0]
        assert item["text"] == "The answer is 42."
        run = item["metadata"]["tool_runs"][0]
        assert run["tool"] == "calc" and run["arguments"] == {"expression": "6*7"}
        # The TEMPLATE declared the tools — we did not write a fence.
        assert "<tools>['calc']</tools>" in prompts[-1]
        # The call and its result went back as real turns, not prose.
        assert "[calls:['calc']]" in prompts[-1]
        assert '"result": 42' in prompts[-1], "the tool result reached the model"
        assert "tool:" in prompts[-1], "and did so under a tool role"
        assert out["tools"]["dialect"] == "qwen-2.5"
        assert out["tools"]["errors"] == []
        assert out["tools"]["with_calls"] == 1
        assert out["tools"]["without_calls"] == 0

    def test_a_model_with_no_tool_protocol_refuses_rather_than_inventing_one(
            self, monkeypatch):
        """The rule that would have saved experiment 024's P2: a model
        that cannot receive a tool declaration must say so, not quietly
        do worse."""
        from mechbench_compute import dialects, distill, generate

        class IgnoresTools:
            def apply_chat_template(self, turns, tokenize=False,
                                     add_generation_prompt=True, tools=None,
                                     **kw):
                return "same either way"

        class FakeModel:
            tokenizer = IgnoresTools()

        monkeypatch.setattr(distill, "encode", lambda tok, text: [1, 2, 3])
        monkeypatch.setattr(distill, "prefill_decision", lambda m, ids: None)
        monkeypatch.setattr(generate, "sample_completion_cached",
                            lambda *a, **k: "55")
        with pytest.raises(dialects.NoToolDialect, match="no tool protocol"):
            chat_mod.run_local(FakeModel(), mr.parse("google/gemma-3-4b-it"),
                               _records(1), {"n": 1, "tools": ["calc"]})

    def test_the_same_model_is_fine_without_tools(self, monkeypatch):
        # The refusal is about OFFERING tools, not about the model.
        from mechbench_compute import distill, generate

        class IgnoresTools:
            def apply_chat_template(self, turns, tokenize=False,
                                     add_generation_prompt=True, tools=None,
                                     **kw):
                return "rendered"

        class FakeModel:
            tokenizer = IgnoresTools()

        monkeypatch.setattr(distill, "encode", lambda tok, text: [1, 2, 3])
        monkeypatch.setattr(distill, "prefill_decision", lambda m, ids: None)
        monkeypatch.setattr(generate, "sample_completion_cached",
                            lambda *a, **k: "55")
        out = chat_mod.run_local(FakeModel(), mr.parse("google/gemma-3-4b-it"),
                                 _records(1), {"n": 1})
        assert out["items"][0]["text"] == "55"


class TestToolErrors:
    """A call that did not execute is an ERROR, reported in the results
    with its cause. There is no "near miss": the only question worth
    asking about a failed call is whose fault it was."""

    def _run(self, reply, extra=None):
        from mechbench_compute import distill, generate

        class QwenishTok:
            def apply_chat_template(self, turns, tokenize=False,
                                     add_generation_prompt=True, tools=None,
                                     **kw):
                if not tools:
                    return "plain"
                body = "TOOLS "
                if any(t.get("tool_calls") for t in turns):
                    body += ('<tool_call>\n{"name": "calc", "arguments": '
                             '{"expression": "37 + 18"}}\n</tool_call>'
                             "<tool_response>\n55\n</tool_response>")
                else:
                    body += "<tool_call>probe</tool_call>"
                return body

        class FakeModel:
            tokenizer = QwenishTok()

        import pytest as _pt
        mp = _pt.MonkeyPatch()
        mp.setattr(distill, "encode", lambda tok, text: [1, 2, 3])
        mp.setattr(distill, "prefill_decision", lambda m, ids: None)
        mp.setattr(generate, "sample_completion_cached", lambda *a, **k: reply)
        try:
            return chat_mod.run_local(
                FakeModel(), mr.parse("Qwen/Qwen2.5-3B-Instruct"), _records(1),
                {"n": 1, "tools": ["calc"], "max_tool_rounds": 1, **(extra or {})})
        finally:
            mp.undo()

    def test_calling_a_tool_that_was_never_offered_is_an_error(self):
        out = self._run('<tool_call>\n{"name": "wget", "arguments": {}}\n</tool_call>')
        errs = out["tools"]["errors"]
        assert [e["cause"] for e in errs] == ["unknown_tool"]
        assert errs[0]["tool"] == "wget"
        assert "not offered" in errs[0]["detail"]

    def test_an_unreadable_call_names_us_as_the_likely_cause(self):
        out = self._run('<tool_call>\n{"name": "calc" BROKEN\n</tool_call>')
        errs = out["tools"]["errors"]
        assert [e["cause"] for e in errs] == ["unparseable_call"]
        assert "OUR parser" in errs[0]["detail"]

    def test_answering_without_a_tool_is_not_an_error(self):
        # Whether the model SHOULD have called one is the experiment's
        # question, not the harness's. It is counted, not faulted.
        out = self._run("The stall has 55 apples.")
        assert out["tools"]["errors"] == []
        assert out["tools"]["without_calls"] == 1
        assert out["tools"]["with_calls"] == 0

    def test_errors_ride_on_the_item_so_they_can_be_sliced(self):
        out = self._run('<tool_call>\n{"name": "wget", "arguments": {}}\n</tool_call>')
        meta = out["items"][0]["metadata"]
        assert [e["cause"] for e in meta["tool_errors"]] == ["unknown_tool"]

    def test_an_error_does_not_fail_the_run_by_default(self):
        out = self._run('<tool_call>\n{"name": "wget", "arguments": {}}\n</tool_call>')
        assert out["items"][0]["text"]  # the response is still there
        assert out["tools"]["errors_by_cause"] == {"unknown_tool": 1}

    def test_on_tool_error_fail_makes_it_fatal_when_asked(self):
        with pytest.raises(RuntimeError, match="unknown_tool"):
            self._run('<tool_call>\n{"name": "wget", "arguments": {}}\n</tool_call>',
                      {"on_tool_error": "fail"})

    def test_an_unknown_policy_refuses(self):
        with pytest.raises(ValueError, match="on_tool_error"):
            self._run("hello", {"on_tool_error": "shrug"})


class TestJobBudget:
    def test_node_caps_chain_under_a_job_cap(self):
        from mechbench_compute.protocol import ProtocolExecutor as PE
        from mechbench_compute.providers import Budget

        job = Budget(cap_usd=0.002)
        ex = PE(budget=job)
        spec = _spec({"n": 4, "max_tokens": 64})
        # Each node cap is 5.0 and could not stop this on its own; the
        # job cap can, and the runner reads its spend live.
        with pytest.raises(BudgetExceeded):
            for _ in range(20):
                ex.run(spec)
        assert job.spent_usd > 0 and job.spent_usd <= job.cap_usd
