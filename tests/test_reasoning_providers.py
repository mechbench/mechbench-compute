from __future__ import annotations

import copy
import json

import pytest

from mechbench_compute import chat as chat_mod
from mechbench_compute import model_ref as mr
from mechbench_compute.providers import Cassette, CassetteTransport, http, make_transport
from mechbench_compute.providers import messages as m

SIG_A = "EqQBCkYIBxgCKkDz+/0==" + "x" * 300
SIG_B = "Es8CCkYICxIM+sig/B=="


class Script:
    def __init__(self, *bodies):
        self.bodies = list(bodies)
        self.sent: list[dict] = []

    def __call__(self, url, *, headers, payload, timeout, secrets=()):
        if url.endswith(("count_tokens", ":countTokens")):
            return http.HttpResponse(status=200, headers={},
                                     body={"input_tokens": 10, "totalTokens": 10})
        self.sent.append(copy.deepcopy(payload))
        return http.HttpResponse(status=200, headers={}, body=copy.deepcopy(self.bodies.pop(0)))


def run_chat(monkeypatch, provider, model, bodies, *, records=None, tools=("calc",),
             base_url=None, **params):
    script = Script(*bodies)
    monkeypatch.setattr(http, "post_json", script)
    ref = mr.parse({"provider": provider, "model": model})
    out = chat_mod.run_remote(
        ref, records or [{"id": "r0", "user": "What is 2+2?"}],
        {"budget_usd": 5.0, "concurrency": 1, "tools": list(tools),
         **({"base_url": base_url} if base_url else {}), **params},
        secrets={provider: {"token": "k", **({"base_url": base_url} if base_url else {})}})
    return out, script


ANTHROPIC_THINK = {"type": "thinking", "thinking": "", "signature": SIG_A}
ANTHROPIC_REDACTED = {"type": "redacted_thinking", "data": "EmwKAhgBEgy3va3pzix/LafPsn4a"}
ANTHROPIC_FINAL_THINK = {"type": "thinking", "thinking": "The tool said 4.", "signature": SIG_B}


def anthropic_bodies():
    usage = {"input_tokens": 10, "output_tokens": 30}
    return [
        {"id": "m1", "model": "claude-opus-5", "stop_reason": "tool_use", "usage": usage,
         "content": [ANTHROPIC_THINK, ANTHROPIC_REDACTED,
                     {"type": "tool_use", "id": "t1", "name": "calc",
                      "input": {"expression": "2+2"}}]},
        {"id": "m2", "model": "claude-opus-5", "stop_reason": "end_turn", "usage": usage,
         "content": [ANTHROPIC_FINAL_THINK, {"type": "text", "text": "Four."}]},
    ]


GEMINI_THOUGHT = {"text": "I should use the calculator.", "thought": True}


def gemini_bodies():
    usage = {"promptTokenCount": 10, "candidatesTokenCount": 5, "thoughtsTokenCount": 20}
    return [
        {"modelVersion": "gemini-3-pro", "usageMetadata": usage, "candidates": [{
            "finishReason": "STOP", "content": {"role": "model", "parts": [
                GEMINI_THOUGHT,
                {"functionCall": {"name": "calc", "args": {"expression": "2+2"}},
                 "thoughtSignature": SIG_A},
                {"functionCall": {"name": "calc", "args": {"expression": "3+3"}}}]}}]},
        {"modelVersion": "gemini-3-pro", "usageMetadata": usage, "candidates": [{
            "finishReason": "STOP", "content": {"role": "model", "parts": [
                {"text": "Four, and six.", "thoughtSignature": SIG_B}]}}]},
    ]


def chat_bodies(first_fields, final_fields):
    usage = {"prompt_tokens": 10, "completion_tokens": 30,
             "completion_tokens_details": {"reasoning_tokens": 20}}
    return [
        {"id": "c1", "model": "x", "usage": usage, "choices": [{
            "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None, **first_fields,
                "tool_calls": [{"id": "call_1", "type": "function", "function": {
                    "name": "calc", "arguments": '{"expression": "2+2"}'}}]}}]},
        {"id": "c2", "model": "x", "usage": usage, "choices": [{
            "finish_reason": "stop", "message": {
                "role": "assistant", "content": "Four.", **final_fields}}]},
    ]


OPENROUTER_DETAILS = [
    {"type": "reasoning.text", "text": "Use the tool.", "signature": SIG_A,
     "id": "r1", "format": "anthropic-claude-v1", "index": 0},
    {"type": "reasoning.encrypted", "data": "eyJlbmNyeXB0ZWQiOnRydWV9",
     "id": "r2", "format": "anthropic-claude-v1", "index": 1},
]


def assistant_turn(payload, provider):
    if provider == "gemini":
        return next(c for c in payload["contents"] if c["role"] == "model")
    return next(x for x in payload["messages"] if x["role"] == "assistant")


class TestAnthropic:
    def test_thinking_is_reasoning_and_the_text_is_prose(self, monkeypatch):
        out, _ = run_chat(monkeypatch, "anthropic", "claude-opus-5", anthropic_bodies())
        item = out["items"][0]
        assert item["text"] == "Four."
        assert item["reasoning"] == [{"text": "The tool said 4.", "provider": "anthropic",
                                      "model": "claude-opus-5",
                                      "native": ANTHROPIC_FINAL_THINK}]

    def test_the_tool_loop_sends_every_block_back_unchanged_and_in_order(self, monkeypatch):
        _, script = run_chat(monkeypatch, "anthropic", "claude-opus-5", anthropic_bodies())
        turn = assistant_turn(script.sent[1], "anthropic")
        assert turn["content"] == [ANTHROPIC_THINK, ANTHROPIC_REDACTED,
                                   {"type": "tool_use", "id": "t1", "name": "calc",
                                    "input": {"expression": "2+2"}}]
        assert turn["content"][0]["signature"] == SIG_A

    def test_the_empty_display_block_and_the_redacted_block_are_stored(self, monkeypatch):
        out, _ = run_chat(monkeypatch, "anthropic", "claude-opus-5", anthropic_bodies())
        rounds = out["items"][0]["metadata"]["rounds"]
        kept = [p for p in rounds[0]["content"] if p["type"] == "reasoning"]
        assert [p["native"] for p in kept] == [ANTHROPIC_THINK, ANTHROPIC_REDACTED]
        assert all(p["redacted"] for p in kept)

    def test_a_reasoning_only_reply_is_empty_and_keeps_its_reasoning(self, monkeypatch):
        body = {"id": "m", "model": "claude-opus-5", "stop_reason": "max_tokens",
                "usage": {"input_tokens": 5, "output_tokens": 100},
                "content": [{"type": "thinking", "thinking": "Score: 5", "signature": SIG_B}]}
        out, _ = run_chat(monkeypatch, "anthropic", "claude-opus-5", [body], tools=())
        item = out["items"][0]
        assert item["text"] == "" and item["metadata"]["empty"]["cause"] == "reasoning"
        assert item["reasoning"][0]["text"] == "Score: 5"


class TestGemini:
    def test_thoughts_are_reasoning_and_signatures_stay_on_their_parts(self, monkeypatch):
        out, _ = run_chat(monkeypatch, "gemini", "gemini-3-pro", gemini_bodies())
        item = out["items"][0]
        assert item["text"] == "Four, and six."
        assert "reasoning" not in item
        assert item["metadata"]["turn"] == [{
            "type": "text", "start": 0, "end": 14,
            "signature": {"provider": "gemini", "model": "gemini-3-pro", "value": SIG_B}}]

    def test_a_signature_goes_back_on_the_same_function_call(self, monkeypatch):
        _, script = run_chat(monkeypatch, "gemini", "gemini-3-pro", gemini_bodies())
        parts = assistant_turn(script.sent[1], "gemini")["parts"]
        assert parts == [
            GEMINI_THOUGHT,
            {"functionCall": {"name": "calc", "args": {"expression": "2+2"}},
             "thoughtSignature": SIG_A},
            {"functionCall": {"name": "calc", "args": {"expression": "3+3"}}}]

    def test_a_thought_is_never_prose(self, monkeypatch):
        body = gemini_bodies()[0]
        body["candidates"][0]["content"]["parts"] = [
            {"text": "Score: 5", "thought": True}, {"text": "Score: 2"}]
        out, _ = run_chat(monkeypatch, "gemini", "gemini-3-pro", [body], tools=())
        assert out["items"][0]["text"] == "Score: 2"
        assert out["items"][0]["reasoning"][0]["text"] == "Score: 5"


class TestDeepSeek:
    def test_reasoning_content_is_separated_and_goes_back_with_tools(self, monkeypatch):
        out, script = run_chat(monkeypatch, "deepseek", "deepseek-flash", chat_bodies(
            {"reasoning_content": "Use the calculator."},
            {"reasoning_content": "It said 4."}))
        item = out["items"][0]
        assert item["text"] == "Four."
        assert item["reasoning"][0]["text"] == "It said 4."
        turn = assistant_turn(script.sent[1], "deepseek")
        assert turn["reasoning_content"] == "Use the calculator."
        assert turn["content"] is None

    def test_deepseek_is_a_registered_provider(self):
        from mechbench_compute.providers import pricing, registry

        spec = registry.spec_for("deepseek")
        assert spec.base_url == "https://api.deepseek.com"
        assert pricing.price_for("deepseek", "deepseek-flash") is not None

    def test_inline_think_from_a_third_party_host_is_reasoning(self, monkeypatch):
        body = chat_bodies({}, {})[1]
        body["choices"][0]["message"]["content"] = "<think>Score: 5</think>\n\nScore: 2"
        out, _ = run_chat(monkeypatch, "fireworks", "accounts/fireworks/models/deepseek-r1",
                          [body], tools=())
        item = out["items"][0]
        assert item["text"] == "Score: 2" and item["reasoning"][0]["text"] == "Score: 5"

    def test_a_turn_closed_without_an_opening_tag_is_reasoning_first(self):
        from mechbench_compute.providers.openai_compatible import split_inline_thought

        assert split_inline_thought("weighing it</think>Yes.") == ("weighing it", "Yes.")
        assert split_inline_thought("<think>cut off") == ("cut off", "")
        assert split_inline_thought("I said <think> once.") == (None, "I said <think> once.")


class TestOpenRouter:
    BASE = "https://openrouter.ai/api/v1"

    def test_reasoning_details_go_back_unmodified(self, monkeypatch):
        fields = {"reasoning": "Use the tool.", "reasoning_details": OPENROUTER_DETAILS}
        out, script = run_chat(monkeypatch, "openai-compatible", "anthropic/claude-opus-5",
                               chat_bodies(fields, {}), base_url=self.BASE)
        turn = assistant_turn(script.sent[1], "openai-compatible")
        assert turn["reasoning_details"] == OPENROUTER_DETAILS
        assert turn["reasoning"] == "Use the tool."
        assert out["items"][0]["text"] == "Four."


class TestXai:
    def test_reasoning_content_is_shown_and_never_sent_back(self, monkeypatch):
        out, script = run_chat(monkeypatch, "xai", "grok-4.7", chat_bodies(
            {"reasoning_content": "Use the tool."}, {"reasoning_content": "Done."}))
        assert out["items"][0]["reasoning"][0]["text"] == "Done."
        turn = assistant_turn(script.sent[1], "xai")
        assert "reasoning_content" not in turn
        assert "Use the tool." not in json.dumps(script.sent[1])


class TestOpenAI:
    def test_chat_completions_returns_a_count_and_nothing_to_carry(self, monkeypatch):
        out, script = run_chat(monkeypatch, "openai", "gpt-5", chat_bodies({}, {}))
        item = out["items"][0]
        assert item["text"] == "Four." and "reasoning" not in item
        assert item["metadata"]["call"]["usage"]["reasoning_tokens"] == 20
        assert set(assistant_turn(script.sent[1], "openai")) == {"role", "content", "tool_calls"}


class TestReplay:
    def test_a_cassette_keeps_the_body_and_replays_it_through_the_adapter(self, monkeypatch):
        body = anthropic_bodies()[1]
        monkeypatch.setattr(http, "post_json", Script(body))
        inner = make_transport("anthropic", {"token": "k"})
        tape = Cassette(provider="anthropic")
        req = m.request({"model": "claude-opus-5", "messages": "hi", "max_tokens": 50})
        recorded = CassetteTransport(tape, inner=inner, mode="record").chat(req)
        wire = tape.to_wire()
        replayed = CassetteTransport(Cassette.from_wire(wire), mode="replay").chat(req)
        assert replayed.call.replayed
        assert replayed.text == recorded.text == "Four."
        assert replayed.parts == recorded.parts

    def test_a_kept_body_is_read_by_the_current_mapping(self):
        req = m.request({"model": "claude-opus-5", "messages": "hi", "max_tokens": 50})
        body = anthropic_bodies()[1]
        entry = {"parts": [{"type": "text", "text": "The tool said 4."},
                           {"type": "text", "text": "Four."}],
                 "stop_reason": "end_turn", "usage": {}, "raw": body}
        tape = Cassette(provider="anthropic",
                        entries={m.request_hash(req, provider="anthropic"): [entry]})
        out = CassetteTransport(tape, mode="replay").chat(req)
        assert out.text == "Four." and out.reasoning[0]["text"] == "The tool said 4."

    def test_an_entry_without_a_body_replays_its_stored_parts(self):
        req = m.request({"model": "claude-opus-5", "messages": "hi", "max_tokens": 50})
        entry = {"parts": [{"type": "text", "text": "The tool said 4."},
                           {"type": "text", "text": "Four."}],
                 "stop_reason": "end_turn", "usage": {}}
        tape = Cassette(provider="anthropic",
                        entries={m.request_hash(req, provider="anthropic"): [entry]})
        out = CassetteTransport(tape, mode="replay").chat(req)
        assert out.text == "The tool said 4.Four."


@pytest.mark.parametrize("parts", [
    [m.ReasoningPart(text="a", provider="anthropic", model="x", native={"type": "thinking"})],
    [m.TextPart("hi", signature=m.Signature("gemini", "g", "s"))],
])
def test_parts_round_trip_through_their_wire_form(parts):
    assert [m.part(p.to_wire()) for p in parts] == parts


def test_a_message_s_text_never_includes_reasoning():
    msg = m.Message(role="assistant", content=(
        m.ReasoningPart(text="Score: 5"), m.TextPart("Score: 2")))
    assert msg.text() == "Score: 2"
