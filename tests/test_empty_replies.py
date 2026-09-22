"""A reply with no prose and no tool call: each adapter says why, the
call it paid for is charged, and `on_empty` decides what the node does
with the item.

Nothing here touches the network: the adapters run against a captured
`post_json`, and the chat node against the mock provider.
"""

from __future__ import annotations

import importlib

import pytest

from mechbench_compute import chat as chat_mod
from mechbench_compute import model_ref as mr
from mechbench_compute.ops.eval.judge import run_judge
from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec
from mechbench_compute.providers import Budget, Cassette, CassetteTransport
from mechbench_compute.providers import messages as m
from mechbench_compute.providers.errors import ProviderError
from mechbench_compute.providers.mock import MockTransport

# The package re-exports the function under the module's own name.
run_remote_mod = importlib.import_module("mechbench_compute.chat.run_remote")

ENDPOINT = {"provider": "mock", "model": "mock-large"}
TOOLS = [{"name": "grep", "description": "search",
          "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}}}]


class Capture:
    """Stands in for providers.http.post_json."""

    def __init__(self, body):
        self.body = body
        self.sent: dict = {}

    def __call__(self, url, *, headers, payload, timeout, secrets=()):
        from mechbench_compute.providers.http import HttpResponse

        self.sent = {"url": url, "headers": headers, "payload": payload}
        return HttpResponse(status=200, headers={}, body=self.body)


def ask(model="claude-opus-5", **kw):
    base = {"model": model, "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100}
    base.update(kw)
    return m.request(base)


def anthropic_reply(monkeypatch, body, *, budget=None, **kw):
    from mechbench_compute.providers import anthropic, http

    monkeypatch.setattr(http, "post_json", Capture(body))
    return anthropic.AnthropicTransport("sk-ant-x").chat(ask(**kw), budget=budget)


def openai_reply(monkeypatch, body, **kw):
    from mechbench_compute.providers import http, openai_compatible

    monkeypatch.setattr(http, "post_json", Capture(body))
    t = openai_compatible.OpenAICompatibleTransport("sk-x", provider="openai")
    return t.chat(ask(model="gpt-5", **kw))


def gemini_reply(monkeypatch, body, **kw):
    from mechbench_compute.providers import gemini, http

    cap = Capture(body)
    monkeypatch.setattr(http, "post_json", cap)
    return gemini.GeminiTransport("key-1").chat(ask(model="gemini-2.5-pro", **kw)), cap


class TestAnthropic:
    def test_a_refusal_is_filtered(self, monkeypatch):
        out = anthropic_reply(monkeypatch, {
            "content": [], "stop_reason": "refusal",
            "usage": {"input_tokens": 5, "output_tokens": 3}})
        assert out.empty.cause == "filtered" and "refusal" in out.empty.message

    def test_no_blocks_at_all_is_no_content(self, monkeypatch):
        out = anthropic_reply(monkeypatch, {
            "content": [], "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 0}})
        assert out.empty.cause == "no_content"

    def test_returned_reasoning_text_is_not_prose(self, monkeypatch):
        out = anthropic_reply(monkeypatch, {
            "content": [{"type": "thinking", "thinking": "Let me think."}],
            "stop_reason": "max_tokens",
            "usage": {"input_tokens": 5, "output_tokens": 100}})
        assert out.empty.cause == "reasoning"
        assert "holds reasoning only" in out.empty.message

    def test_a_tool_call_only_reply_is_not_empty(self, monkeypatch):
        out = anthropic_reply(monkeypatch, {
            "content": [{"type": "thinking", "thinking": ""},
                        {"type": "tool_use", "id": "t1", "name": "grep",
                         "input": {"q": "y"}}],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 5, "output_tokens": 20}}, tools=TOOLS)
        assert out.empty is None and out.call.empty is None
        assert "empty" not in out.call.to_wire()


class TestOpenAI:
    def test_a_content_filter_is_filtered(self, monkeypatch):
        out = openai_reply(monkeypatch, {
            "choices": [{"message": {"content": None}, "finish_reason": "content_filter"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 0}})
        assert out.empty.cause == "filtered"

    def test_a_refusal_is_filtered(self, monkeypatch):
        out = openai_reply(monkeypatch, {
            "choices": [{"message": {"content": None, "refusal": "I can't help."},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 4}})
        assert out.empty.cause == "filtered" and "I can't help." in out.empty.message

    def test_reasoning_tokens_with_no_content_name_the_effort_remedy(self, monkeypatch):
        out = openai_reply(monkeypatch, {
            "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 100,
                      "completion_tokens_details": {"reasoning_tokens": 100}}})
        assert out.empty.cause == "reasoning"
        assert "Raise max_tokens" in out.empty.message
        assert '{"openai": {"reasoning_effort": "low"}}' in out.empty.message

    def test_nothing_said_is_no_content(self, monkeypatch):
        out = openai_reply(monkeypatch, {
            "choices": [{"message": {"content": ""}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 0}})
        assert out.empty.cause == "no_content"

    def test_a_tool_call_only_reply_is_not_empty(self, monkeypatch):
        out = openai_reply(monkeypatch, {
            "choices": [{"message": {"content": None, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "grep", "arguments": "{}"}}]},
                "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 9,
                      "completion_tokens_details": {"reasoning_tokens": 5}}},
            tools=TOOLS)
        assert out.empty is None


class TestGemini:
    def test_a_blocked_prompt_is_filtered(self, monkeypatch):
        out, _ = gemini_reply(monkeypatch, {
            "promptFeedback": {"blockReason": "SAFETY"},
            "usageMetadata": {"promptTokenCount": 5}})
        assert out.empty.cause == "filtered" and "SAFETY" in out.empty.message

    def test_no_candidates_is_filtered(self, monkeypatch):
        out, _ = gemini_reply(monkeypatch, {"usageMetadata": {"promptTokenCount": 5}})
        assert out.empty.cause == "filtered"

    @pytest.mark.parametrize("finish", ["SAFETY", "RECITATION", "PROHIBITED_CONTENT"])
    def test_a_safety_class_finish_is_filtered(self, monkeypatch, finish):
        out, _ = gemini_reply(monkeypatch, {
            "candidates": [{"content": {"parts": []}, "finishReason": finish}],
            "usageMetadata": {"promptTokenCount": 5}})
        assert out.empty.cause == "filtered"

    def test_thoughts_with_no_text_are_reasoning_and_are_billed(self, monkeypatch):
        out, _ = gemini_reply(monkeypatch, {
            "candidates": [{"content": {"parts": [
                {"text": "Considering.", "thought": True}]},
                "finishReason": "MAX_TOKENS"}],
            "usageMetadata": {"promptTokenCount": 5, "thoughtsTokenCount": 95}})
        assert out.empty.cause == "reasoning"
        assert ('{"gemini": {"generationConfig": {"thinkingConfig": '
                '{"thinkingBudget": 0}}}}') in out.empty.message
        # Thoughts are output the provider bills.
        assert out.call.usage["output_tokens"] == 95
        assert out.call.cost_usd > 0

    def test_nothing_said_is_no_content(self, monkeypatch):
        out, _ = gemini_reply(monkeypatch, {
            "candidates": [{"content": {"parts": []}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 5}})
        assert out.empty.cause == "no_content"

    def test_a_function_call_only_reply_is_not_empty(self, monkeypatch):
        out, _ = gemini_reply(monkeypatch, {
            "candidates": [{"content": {"parts": [
                {"functionCall": {"name": "grep", "args": {}}}]},
                "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 4}},
            tools=TOOLS)
        assert out.empty is None

    def test_a_thinking_config_in_the_options_keeps_the_output_cap(self, monkeypatch):
        _, cap = gemini_reply(monkeypatch, {
            "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 1}},
            provider_options={"gemini": {"generationConfig": {
                "thinkingConfig": {"thinkingBudget": 0}}}})
        cfg = cap.sent["payload"]["generationConfig"]
        assert cfg["maxOutputTokens"] == 100
        assert cfg["thinkingConfig"] == {"thinkingBudget": 0}


class TestSpend:
    def test_an_empty_reply_is_charged_to_the_budget(self, monkeypatch):
        budget = Budget(cap_usd=1.0)
        out = anthropic_reply(monkeypatch, {
            "content": [{"type": "thinking", "thinking": ""}],
            "stop_reason": "max_tokens",
            "usage": {"input_tokens": 5, "output_tokens": 100}}, budget=budget)
        assert out.empty.cause == "reasoning"
        assert out.call.cost_usd > 0
        assert budget.spent_usd == pytest.approx(out.call.cost_usd)
        assert budget.reserved_usd == 0.0 and budget.calls == 1

    def test_a_recorded_empty_reply_replays_as_empty(self):
        tape = Cassette(provider="mock")
        inner = MockTransport(script=[{"empty": "filtered"}])
        CassetteTransport(tape, inner=inner, mode="record").chat(ask(model="m"))
        again = CassetteTransport(Cassette.from_wire(tape.to_wire())).chat(ask(model="m"))
        assert again.empty is not None and again.empty.cause == "filtered"


def records(n=4):
    return [{"id": f"r{i}", "user": f"question {i}"} for i in range(n)]


def scripted(monkeypatch, script):
    """The node's transport answers from `script`, in order."""
    monkeypatch.setattr(run_remote_mod, "make_transport",
                        lambda *a, **k: MockTransport(script=list(script)))


class TestThePolicy:
    SCRIPT = (None, {"empty": "reasoning"}, None, {"empty": "filtered"})

    def run(self, monkeypatch, policy, **kw):
        scripted(monkeypatch, self.SCRIPT)
        params = {"budget_usd": 1.0, "concurrency": 1, "on_empty": policy}
        return chat_mod.run_remote(mr.parse(ENDPOINT), records(), params, **kw)

    def test_keep_emits_the_item_marked(self, monkeypatch):
        out = self.run(monkeypatch, "keep")
        items = {i["id"]: i for i in out["items"]}
        assert len(items) == 4
        assert items["r1-s0"]["text"] == ""
        assert items["r1-s0"]["metadata"]["empty"]["cause"] == "reasoning"
        assert items["r1-s0"]["metadata"]["call"]["empty"]["cause"] == "reasoning"
        assert "empty" not in items["r0-s0"]["metadata"]
        assert out["empty"] == {"count": 2,
                                "by_cause": {"reasoning": 1, "filtered": 1},
                                "ids": ["r1-s0", "r3-s0"], "policy": "keep"}
        # Paid for, and in the node's bill.
        assert out["spend"]["calls"] == 4

    def test_skip_leaves_the_item_out_and_counts_it(self, monkeypatch):
        out = self.run(monkeypatch, "skip")
        assert [i["id"] for i in out["items"]] == ["r0-s0", "r2-s0"]
        assert out["empty"]["count"] == 2 and out["empty"]["policy"] == "skip"
        assert out["spend"]["calls"] == 4

    def test_error_fails_once_the_item_is_checkpointed(self, monkeypatch):
        landed = []
        with pytest.raises(ProviderError, match="cause reasoning"):
            self.run(monkeypatch, "error",
                     on_item=lambda key, item, *rest: landed.append(key))
        # The empty item was checkpointed before the node failed, and
        # nothing after it was bought.
        assert landed == ["r0:0", "r1:0"]

    def test_an_unknown_policy_is_refused(self, monkeypatch):
        with pytest.raises(ValueError, match="on_empty"):
            self.run(monkeypatch, "retry")

    def test_the_header_is_present_with_none_empty(self):
        out = chat_mod.run_remote(mr.parse(ENDPOINT), records(2),
                                  {"budget_usd": 1.0})
        assert out["empty"] == {"count": 0, "by_cause": {}, "ids": [],
                                "policy": "keep"}

    def test_the_executor_carries_the_header_and_the_mark(self):
        graph = {"nodes": [{"id": "chat", "block": "text/chat",
                            "params": {"model": ENDPOINT, "budget_usd": 1.0,
                                       "provider_options": {"mock": {"empty": "reasoning"}}},
                            "inputs": {"records": records(2)}}], "edges": []}
        out = ProtocolExecutor().run(ProtocolSpec(
            kind="pipeline", prompt="", model_id=None, extra={"graph": graph}))
        node = out.payload["outputs"]["chat"]
        assert node["empty"]["count"] == 2 and node["empty"]["policy"] == "keep"
        assert all(i["metadata"]["empty"]["cause"] == "reasoning"
                   for i in node["items"])


class TestResume:
    def first_run(self, monkeypatch):
        scripted(monkeypatch, [None, {"empty": "reasoning"}])
        out = chat_mod.run_remote(mr.parse(ENDPOINT), records(2),
                                  {"budget_usd": 1.0, "concurrency": 1})
        return {f"r{i}:0": item for i, item in enumerate(out["items"])}

    @pytest.mark.parametrize("policy,ids", [("keep", ["r0-s0", "r1-s0"]),
                                            ("skip", ["r0-s0"])])
    def test_a_resumed_empty_item_is_not_bought_and_the_policy_applies(
            self, monkeypatch, policy, ids):
        kept = self.first_run(monkeypatch)
        scripted(monkeypatch, [])
        out = chat_mod.run_remote(mr.parse(ENDPOINT), records(2),
                                  {"budget_usd": 1.0, "on_empty": policy},
                                  resume_items=kept)
        assert out["spend"]["calls"] == 0
        assert sorted(i["id"] for i in out["items"]) == ids
        assert out["empty"]["ids"] == ["r1-s0"]
        assert out["empty"]["policy"] == policy

    def test_a_resumed_empty_item_fails_a_strict_node_without_buying(self, monkeypatch):
        kept = self.first_run(monkeypatch)
        transport = MockTransport()
        monkeypatch.setattr(run_remote_mod, "make_transport",
                            lambda *a, **k: transport)
        with pytest.raises(ProviderError):
            chat_mod.run_remote(mr.parse(ENDPOINT), records(3),
                                {"budget_usd": 1.0, "on_empty": "error"},
                                resume_items=kept)
        # r2 was never asked for.
        assert transport.calls == []


class TestTheJudge:
    STORIES = ({"id": "s1", "text": "The dust settled."},
               {"id": "s2", "text": "A kettle sang."})

    def params(self, options):
        return {"judge": {"model": {"provider": "mock", "model": "judge-1"},
                          "system": "Grade the story.",
                          "provider_options": {"mock": options}},
                "scale": {"kind": "numeric", "min": 1, "max": 5},
                "budget_usd": 1.0}

    def test_an_empty_judge_reply_is_an_unparsed_vote(self):
        out = run_judge(self.params({"empty": "reasoning"}),
                        inputs={"records": list(self.STORIES)})
        assert out["summary"]["n_unparsed"] == 2
        vote = out["items"][0]["votes"][0]
        assert vote["parsed"] is False and vote["empty"] == "reasoning"
        assert out["spend"]["calls"] == 2

    def test_a_flagged_reply_with_text_is_still_unparsed(self, monkeypatch):
        from mechbench_compute.ops.eval import judge as judge_mod

        real = judge_mod.chat_mod.run_remote
        seen = {}

        def flagged(ref, prompts, params, **kw):
            seen["on_empty"] = params.get("on_empty")
            out = real(ref, prompts, params, **kw)
            for item in out["items"]:
                item["metadata"]["empty"] = {"cause": "reasoning", "message": "m"}
            return out

        monkeypatch.setattr(judge_mod.chat_mod, "run_remote", flagged)
        out = run_judge(self.params({"text": '{"score": 4}'}),
                        inputs={"records": list(self.STORIES)})
        assert seen["on_empty"] == "keep"
        assert out["summary"]["n_unparsed"] == 2


def test_an_empty_story_kept_by_chat_is_unjudged_and_the_judge_completes():
    """`text/chat` keeps an empty reply marked; the judge after it grades
    what has text and counts the rest, instead of failing the graph."""
    graph = {"nodes": [
        {"id": "stories", "block": "text/chat",
         "params": {"model": ENDPOINT, "budget_usd": 1.0, "on_empty": "keep",
                    "provider_options": {"mock": {"empty": "reasoning"}}},
         "inputs": {"records": records(2)}},
        {"id": "graded", "block": "eval/judge",
         "params": {"judge": {"model": {"provider": "mock", "model": "judge-1"},
                              "system": "Grade the story.",
                              "provider_options": {"mock": {"text": '{"score": 4}'}}},
                    "scale": {"kind": "numeric", "min": 1, "max": 5},
                    "budget_usd": 1.0}}],
        "edges": [{"from": {"node": "stories"}, "to": {"node": "graded", "port": "records"}}]}
    out = ProtocolExecutor().run(ProtocolSpec(
        kind="pipeline", prompt="", model_id=None, extra={"graph": graph}))
    verdict = out.payload["outputs"]["graded"]
    assert verdict["summary"]["n_unjudged"] == 2
    assert all(r["unjudged"] and r["missing"] == ["text"] for r in verdict["items"])
