"""The Responses API, for OpenAI and xAI.

Chat Completions returns OpenAI's reasoning as a count and xAI's as text
it cannot take back. The Responses API returns reasoning as output items
with `encrypted_content`, which go back unchanged in the next request's
`input`. These tests hold the adapter to the same rule as every other
provider: reasoning is its own content, stored verbatim, sent
back only to the provider and model that wrote it, never as text.

Bodies are the shapes the providers document, answered by a scripted
`post_json`, so what is asserted is the wire.
"""

from __future__ import annotations

import copy
import json

import pytest

from mechbench_compute import chat as chat_mod
from mechbench_compute import model_ref as mr
from mechbench_compute.ops.text.extend import extend
from mechbench_compute.ops.text.render import render_records
from mechbench_compute.providers import Cassette, CassetteTransport, http, make_transport
from mechbench_compute.providers import messages as m
from mechbench_compute.providers.errors import CapabilityUnsupported, ProviderError

ENC_A = "gAAAAABo" + "A" * 400 + "=="
ENC_B = "gAAAAABo" + "B" * 400 + "=="


class Script:
    """Stands in for `http.post_json`: answers in order, keeping each URL
    and each payload as sent."""

    def __init__(self, *bodies):
        self.bodies = list(bodies)
        self.sent: list[dict] = []
        self.urls: list[str] = []

    def __call__(self, url, *, headers, payload, timeout, secrets=()):
        if url.endswith("count_tokens"):
            return http.HttpResponse(status=200, headers={}, body={"input_tokens": 10})
        self.urls.append(url)
        self.sent.append(copy.deepcopy(payload))
        return http.HttpResponse(status=200, headers={},
                                 body=copy.deepcopy(self.bodies.pop(0)))


def run_chat(monkeypatch, provider, model, bodies, *, records=None, tools=("calc",),
             **params):
    script = Script(*bodies)
    monkeypatch.setattr(http, "post_json", script)
    out = chat_mod.run_remote(
        mr.parse({"provider": provider, "model": model}),
        records or [{"id": "r0", "user": "What is 2+2?"}],
        {"budget_usd": 5.0, "concurrency": 1, "tools": list(tools), **params},
        secrets={provider: {"token": "k"}})
    return out, script


# --- the shapes the providers document ---------------------------------------

RS1 = {"id": "rs_1", "type": "reasoning", "summary": [], "encrypted_content": ENC_A}
FC1 = {"id": "fc_1", "type": "function_call", "status": "completed", "call_id": "call_1",
       "name": "calc", "arguments": '{"expression": "2+2"}'}
RS2 = {"id": "rs_2", "type": "reasoning",
       "summary": [{"type": "summary_text", "text": "The tool said 4."}],
       "encrypted_content": ENC_B}
MSG = {"id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
       "phase": "final_answer",
       "content": [{"type": "output_text", "text": "Four.", "annotations": []}]}

USAGE = {"input_tokens": 10, "input_tokens_details": {"cached_tokens": 4},
         "output_tokens": 30, "output_tokens_details": {"reasoning_tokens": 20},
         "total_tokens": 40}


def response(output, *, model="gpt-6-astra-2026-08-01", status="completed",
             incomplete=None, usage=USAGE, rid="resp_1"):
    return {"id": rid, "object": "response", "model": model, "status": status,
            "incomplete_details": incomplete, "output": output, "usage": usage}


def openai_bodies():
    return [response([RS1, FC1]), response([RS2, MSG], rid="resp_2")]


# xAI's reasoning item carries readable reasoning in `content`.
XRS1 = {"id": "rs_x1", "type": "reasoning", "status": "completed", "summary": [],
        "content": [{"type": "reasoning_text", "text": "Use the tool."}],
        "encrypted_content": ENC_A}
XRS2 = {"id": "rs_x2", "type": "reasoning", "status": "completed", "summary": [],
        "content": [{"type": "reasoning_text", "text": "It said 4."}],
        "encrypted_content": ENC_B}


def xai_bodies():
    return [response([XRS1, FC1], model="grok-4.7"),
            response([XRS2, MSG], model="grok-4.7", rid="resp_2")]


# --- which API answers --------------------------------------------------------

class TestWhichApi:
    def test_gpt_6_astra_goes_to_the_responses_api_by_default(self, monkeypatch):
        # OpenAI: "Chat Completions does not support function calling
        # with GPT-6 Astra."
        _, script = run_chat(monkeypatch, "openai", "gpt-6-astra", openai_bodies(),
                             system="Be brief.")
        assert script.urls[0] == "https://api.openai.com/v1/responses"
        body = script.sent[0]
        assert body["store"] is False
        assert body["instructions"] == "Be brief."
        assert body["max_output_tokens"] == 1024
        assert body["input"] == [{"role": "user", "content": "What is 2+2?"}]
        assert body["tools"][0]["type"] == "function" and body["tools"][0]["name"] == "calc"
        assert "messages" not in body and "max_tokens" not in body
        # Encrypted reasoning comes back by default in stateless mode.
        assert "include" not in body

    def test_every_other_openai_model_stays_on_chat_completions(self, monkeypatch):
        from test_reasoning_providers import chat_bodies

        _, script = run_chat(monkeypatch, "openai", "gpt-5", chat_bodies({}, {}))
        assert script.urls[0] == "https://api.openai.com/v1/chat/completions"
        assert "api" not in json.dumps(script.sent[0])

    def test_chat_completions_can_still_be_asked_for_by_name(self, monkeypatch):
        from test_reasoning_providers import chat_bodies

        _, script = run_chat(monkeypatch, "openai", "gpt-6-astra", chat_bodies({}, {}),
                             tools=(), api="chat_completions")
        assert script.urls[0].endswith("/chat/completions")
        assert "messages" in script.sent[0]

    def test_the_responses_api_by_name_for_any_openai_or_xai_model(self, monkeypatch):
        _, script = run_chat(monkeypatch, "xai", "grok-4.7", xai_bodies(), api="responses")
        assert script.urls[0] == "https://api.x.ai/v1/responses"

    def test_a_provider_without_it_refuses_by_name(self):
        req = m.request({"model": "claude-opus-5", "messages": "hi", "api": "responses"})
        with pytest.raises(CapabilityUnsupported, match="api"):
            make_transport("anthropic", {"token": "k"}).chat(req)

    @pytest.mark.parametrize("field,value", [("stop", ("END",)), ("logprobs", 5)])
    def test_what_the_responses_api_has_no_field_for_is_refused(self, field, value):
        req = m.request({"model": "gpt-6-astra", "messages": "hi", "api": "responses",
                         field: value})
        with pytest.raises(CapabilityUnsupported, match=field):
            make_transport("openai", {"token": "k"}).chat(req)

    def test_a_node_seed_is_not_sent_where_there_is_no_seed(self, monkeypatch):
        _, script = run_chat(monkeypatch, "openai", "gpt-6-astra",
                             [response([MSG])], tools=(), seed=7)
        assert "seed" not in script.sent[0]

    def test_the_api_is_part_of_the_request_s_identity_only_when_it_is_responses(self):
        base = {"model": "gpt-6-astra", "messages": "hi"}
        plain = m.canonical(m.request(base), provider="openai")
        assert "api" not in plain
        assert m.canonical(m.request({**base, "api": "chat_completions"}),
                           provider="openai") == plain
        assert m.canonical(m.request({**base, "api": "responses"}),
                           provider="openai")["api"] == "responses"

    def test_an_unknown_api_is_refused(self):
        with pytest.raises(ValueError, match="api"):
            m.request({"model": "gpt-6-astra", "messages": "hi", "api": "assistants"})


# --- openai --------------------------------------------------------------------

class TestOpenAI:
    def test_reasoning_is_its_own_content_and_the_text_is_prose(self, monkeypatch):
        out, _ = run_chat(monkeypatch, "openai", "gpt-6-astra", openai_bodies())
        item = out["items"][0]
        assert item["text"] == "Four."
        assert item["reasoning"] == [{"text": "The tool said 4.", "provider": "openai",
                                      "model": "gpt-6-astra", "native": RS2}]
        assert ENC_B not in item["text"]

    def test_the_tool_loop_sends_every_item_back_unchanged_and_in_order(self, monkeypatch):
        _, script = run_chat(monkeypatch, "openai", "gpt-6-astra", openai_bodies())
        sent = script.sent[1]["input"]
        assert sent[0] == {"role": "user", "content": "What is 2+2?"}
        assert sent[1] == RS1 and sent[2] == FC1
        assert json.dumps(sent[1]) == json.dumps(RS1)
        assert sent[3]["type"] == "function_call_output"
        assert sent[3]["call_id"] == "call_1" and sent[3]["output"]

    def test_the_rounds_keep_the_encrypted_reasoning(self, monkeypatch):
        out, _ = run_chat(monkeypatch, "openai", "gpt-6-astra", openai_bodies())
        rounds = out["items"][0]["metadata"]["rounds"]
        kept = [p for p in rounds[0]["content"] if p["type"] == "reasoning"]
        assert [p["native"] for p in kept] == [RS1]
        assert kept[0]["redacted"] is True

    def test_usage_counts_reasoning_and_cached_tokens_and_the_call_is_paid(self, monkeypatch):
        out, _ = run_chat(monkeypatch, "openai", "gpt-6-astra", openai_bodies())
        call = out["items"][0]["metadata"]["call"]
        assert call["usage"] == {"input_tokens": 10, "output_tokens": 30,
                                 "cache_read_tokens": 4, "reasoning_tokens": 20}
        assert call["cost_usd"] > 0 and call["priced"] is True
        assert call["model_version"] == "gpt-6-astra-2026-08-01"
        assert call["response_id"] == "resp_2"
        assert out["spend"]["cost_usd"] > call["cost_usd"]  # both calls settled

    def test_the_final_turn_keeps_its_message_item(self, monkeypatch):
        out, _ = run_chat(monkeypatch, "openai", "gpt-6-astra", openai_bodies())
        turn = out["items"][0]["metadata"]["turn"]
        assert turn[0] == {"type": "reasoning", "index": 0}
        assert json.loads(turn[1]["signature"]["value"]) == MSG


# --- xai -----------------------------------------------------------------------

class TestXai:
    def test_encrypted_reasoning_is_asked_for_and_readable_reasoning_is_kept(self, monkeypatch):
        out, script = run_chat(monkeypatch, "xai", "grok-4.7", xai_bodies(), api="responses")
        assert script.sent[0]["include"] == ["reasoning.encrypted_content"]
        assert script.sent[0]["store"] is False
        item = out["items"][0]
        assert item["text"] == "Four."
        assert item["reasoning"][0]["text"] == "It said 4."
        assert item["reasoning"][0]["native"] == XRS2

    def test_the_tool_loop_hands_grok_its_reasoning_back(self, monkeypatch):
        _, script = run_chat(monkeypatch, "xai", "grok-4.7", xai_bodies(), api="responses")
        assert script.sent[1]["input"][1:3] == [XRS1, FC1]

    def test_grok_stays_on_chat_completions_unless_asked(self, monkeypatch):
        from test_reasoning_providers import chat_bodies

        _, script = run_chat(monkeypatch, "xai", "grok-4.7", chat_bodies({}, {}))
        assert script.urls[0].endswith("/chat/completions")


# --- empty replies ---------------------------------------------------------------

class TestEmpty:
    def test_reasoning_that_spent_the_allowance_is_empty_paid_and_kept(self, monkeypatch):
        body = response([RS2], status="incomplete",
                        incomplete={"reason": "max_output_tokens"})
        out, _ = run_chat(monkeypatch, "openai", "gpt-6-astra", [body], tools=())
        item = out["items"][0]
        assert item["text"] == ""
        assert item["metadata"]["empty"]["cause"] == "reasoning"
        assert "max_output_tokens" in item["metadata"]["empty"]["message"]
        assert item["reasoning"][0]["native"] == RS2
        assert item["metadata"]["call"]["cost_usd"] > 0
        assert out["empty"]["by_cause"] == {"reasoning": 1}

    @pytest.mark.parametrize("output,status,incomplete,cause", [
        ([], "incomplete", {"reason": "content_filter"}, "filtered"),
        ([{"id": "msg_r", "type": "message", "role": "assistant", "status": "completed",
           "content": [{"type": "refusal", "refusal": "I can't help with that."}]}],
         "completed", None, "filtered"),
        ([{"id": "ws_1", "type": "web_search_call", "status": "completed"}],
         "completed", None, "unmapped"),
        ([], "completed", None, "no_content"),
    ])
    def test_each_cause_is_named(self, monkeypatch, output, status, incomplete, cause):
        usage = {"input_tokens": 10, "output_tokens": 3}
        body = response(output, status=status, incomplete=incomplete, usage=usage)
        out, _ = run_chat(monkeypatch, "openai", "gpt-6-astra", [body], tools=())
        assert out["items"][0]["metadata"]["empty"]["cause"] == cause

    def test_on_empty_error_fails_the_node_after_paying(self, monkeypatch):
        body = response([RS2], status="incomplete",
                        incomplete={"reason": "max_output_tokens"})
        with pytest.raises(ProviderError, match="reasoning"):
            run_chat(monkeypatch, "openai", "gpt-6-astra", [body], tools=(),
                     on_empty="error")

    def test_on_empty_skip_leaves_it_out(self, monkeypatch):
        body = response([], status="incomplete", incomplete={"reason": "content_filter"})
        out, _ = run_chat(monkeypatch, "openai", "gpt-6-astra", [body], tools=(),
                          on_empty="skip")
        assert out["items"] == [] and out["empty"]["count"] == 1


# --- cassettes -------------------------------------------------------------------

class TestCassette:
    def test_the_body_is_kept_and_replayed_through_the_responses_reader(self, monkeypatch):
        monkeypatch.setattr(http, "post_json", Script(response([RS2, MSG])))
        req = m.request({"model": "gpt-6-astra", "messages": "hi", "max_tokens": 50,
                         "api": "responses"})
        tape = Cassette(provider="openai")
        recorded = CassetteTransport(tape, inner=make_transport("openai", {"token": "k"}),
                                     mode="record").chat(req)
        wire = tape.to_wire()
        assert wire["entries"][0]["responses"][0]["raw"]["output"] == [RS2, MSG]
        replayed = CassetteTransport(Cassette.from_wire(wire), mode="replay").chat(req)
        assert replayed.call.replayed and replayed.call.cost_usd == 0.0
        assert replayed.text == recorded.text == "Four."
        assert replayed.parts == recorded.parts

    def test_a_chat_completions_recording_is_not_a_responses_answer(self):
        cc = m.request({"model": "gpt-6-astra", "messages": "hi"})
        rs = m.request({"model": "gpt-6-astra", "messages": "hi", "api": "responses"})
        assert m.request_hash(cc, provider="openai") != m.request_hash(rs, provider="openai")


# --- continuation from a stored transcript ---------------------------------------

OPENING = {"id": "c1", "kind": "text/transcript", "participants": ["ana"],
           "messages": [{"index": 0, "participant": "user", "role_as_seen": "user",
                         "text": "What is 2+2?"}]}


def stored_after_one_turn(monkeypatch):
    rows = render_records({"transcripts": [OPENING]}, {"participant": "ana"})["items"]
    out, _ = run_chat(monkeypatch, "openai", "gpt-6-astra", openai_bodies(), records=rows)
    grown = extend({"transcripts": [OPENING], "replies": out["items"]},
                   {"participant": "ana"})["items"][0]
    grown["messages"].append({"index": 2, "participant": "user", "role_as_seen": "user",
                              "text": "And 3+3?"})
    return json.loads(json.dumps(grown))


def continue_on(monkeypatch, transcript, provider, model, body, **params):
    rows = render_records({"transcripts": [transcript]},
                          {"participant": "ana", "sees": {"own_thinking": "native"}})["items"]
    _, script = run_chat(monkeypatch, provider, model, [body], records=rows, tools=(),
                         **params)
    return script.sent[0]


class TestContinuation:
    def test_the_same_model_is_handed_its_items_back_byte_for_byte(self, monkeypatch):
        stored = stored_after_one_turn(monkeypatch)
        payload = continue_on(monkeypatch, stored, "openai", "gpt-6-astra",
                              response([MSG]))
        assert payload["input"] == [{"role": "user", "content": "What is 2+2?"}, RS2, MSG,
                                    {"role": "user", "content": "And 3+3?"}]
        assert json.dumps(payload["input"][1]) == json.dumps(RS2)

    def test_another_openai_model_gets_the_words_and_no_reasoning(self, monkeypatch):
        stored = stored_after_one_turn(monkeypatch)
        payload = continue_on(monkeypatch, stored, "openai", "gpt-5", response([MSG]),
                              api="responses")
        assert ENC_B not in json.dumps(payload)
        assert payload["input"][1] == {"role": "assistant", "content": "Four."}

    def test_the_same_model_on_chat_completions_gets_no_reasoning(self, monkeypatch):
        from test_reasoning_providers import chat_bodies

        stored = stored_after_one_turn(monkeypatch)
        payload = continue_on(monkeypatch, stored, "openai", "gpt-6-astra",
                              chat_bodies({}, {})[1], api="chat_completions")
        assert ENC_B not in json.dumps(payload) and "msg_1" not in json.dumps(payload)

    def test_another_provider_gets_no_reasoning_and_no_leak(self, monkeypatch):
        from test_reasoning_providers import anthropic_bodies

        stored = stored_after_one_turn(monkeypatch)
        payload = continue_on(monkeypatch, stored, "anthropic", "claude-opus-5",
                              anthropic_bodies()[1])
        assert ENC_B not in json.dumps(payload) and "msg_1" not in json.dumps(payload)

    def test_the_transcript_keeps_reasoning_apart_from_what_was_said(self, monkeypatch):
        stored = stored_after_one_turn(monkeypatch)
        assert stored["messages"][1]["text"] == "Four."
