"""How each item ended: one word on every item of `text/generate` and
`text/chat`, local or remote, in `metadata.sampling.ended`, and the
node's header counting its items by that word.

Nothing here touches the network: the adapters run against a scripted
`post_json`, the local paths against a fake sampler.
"""

from __future__ import annotations

import copy

import pytest

from mechbench_compute import chat as chat_mod
from mechbench_compute import distill, generate
from mechbench_compute import model_ref as mr
from mechbench_compute.chat.constants import ENDINGS
from mechbench_compute.chat.count_endings import count_endings
from mechbench_compute.chat.read_ending import read_ending
from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec
from mechbench_compute.providers import http

# --- the mapping, word by word ----------------------------------------------

@pytest.mark.parametrize("provider,word,ended", [
    ("anthropic", "end_turn", "end"),
    ("anthropic", "stop_sequence", "stop"),
    ("anthropic", "max_tokens", "max_tokens"),
    ("anthropic", "model_context_window_exceeded", "max_tokens"),
    ("anthropic", "tool_use", "tool_call"),
    ("anthropic", "refusal", "filtered"),
    ("anthropic", "pause_turn", "other"),
    ("openai", "stop", "end"),
    ("openai", "length", "max_tokens"),
    ("openai", "tool_calls", "tool_call"),
    ("openai", "function_call", "tool_call"),
    ("openai", "content_filter", "filtered"),
    ("openai", "completed", "end"),
    ("openai", "max_output_tokens", "max_tokens"),
    ("openai", "max_messages", "other"),
    ("openai", "steered", "other"),
    ("openai", "incomplete", "other"),
    ("openai", "failed", "other"),
    ("xai", "stop", "end"),
    ("xai", "end_turn", "end"),
    ("xai", "length", "max_tokens"),
    ("xai", "tool_calls", "tool_call"),
    ("xai", "max_output_tokens", "max_tokens"),
    ("deepseek", "stop", "end"),
    ("deepseek", "length", "max_tokens"),
    ("deepseek", "tool_calls", "tool_call"),
    ("deepseek", "content_filter", "filtered"),
    ("deepseek", "insufficient_system_resource", "other"),
    ("deepseek", "aborted", "other"),
    ("gemini", "stop", "end"),
    ("gemini", "max_tokens", "max_tokens"),
    ("gemini", "safety", "filtered"),
    ("gemini", "recitation", "filtered"),
    ("gemini", "language", "filtered"),
    ("gemini", "blocklist", "filtered"),
    ("gemini", "prohibited_content", "filtered"),
    ("gemini", "spii", "filtered"),
    ("gemini", "image_safety", "filtered"),
    ("gemini", "escalation", "filtered"),
    ("gemini", "pup_limited_disabled", "filtered"),
    ("gemini", "other", "other"),
    ("gemini", "malformed_function_call", "other"),
    ("gemini", "too_many_tool_calls", "other"),
    ("gemini", "finish_reason_unspecified", "other"),
    ("mock", "end_turn", "end"),
    ("fireworks", "length", "max_tokens"),
])
def test_each_providers_word_maps_to_one_ending(provider, word, ended):
    assert read_ending(provider, word) == ended
    assert ended in ENDINGS


def test_an_empty_reply_is_empty_whatever_the_provider_said():
    assert read_ending("anthropic", "max_tokens", empty=True) == "empty"
    assert read_ending("openai", "content_filter", empty=True) == "empty"


def test_a_natural_stop_holding_a_tool_call_is_a_tool_call():
    # Gemini and the Responses API report a function call as a plain stop.
    assert read_ending("gemini", "stop", tool_call=True) == "tool_call"
    assert read_ending("openai", "completed", tool_call=True) == "tool_call"
    # A tool call cut off at the limit was still cut off.
    assert read_ending("openai", "length", tool_call=True) == "max_tokens"


# --- through the adapters and the node --------------------------------------

class Script:
    def __init__(self, *bodies):
        self.bodies = list(bodies)

    def __call__(self, url, *, headers, payload, timeout, secrets=()):
        if url.endswith(("count_tokens", ":countTokens")):
            return http.HttpResponse(status=200, headers={}, body={"input_tokens": 10})
        return http.HttpResponse(status=200, headers={},
                                 body=copy.deepcopy(self.bodies.pop(0)))


def run_chat(monkeypatch, provider, model, bodies, **params):
    monkeypatch.setattr(http, "post_json", Script(*bodies))
    return chat_mod.run_remote(
        mr.parse({"provider": provider, "model": model}),
        [{"id": f"r{i}", "user": "Tell me a story."} for i in range(len(bodies))],
        {"budget_usd": 5.0, "concurrency": 1, **params},
        secrets={provider: {"token": "k"}})


def anthropic(stop, text="Once."):
    return {"id": "msg_1", "model": "claude-opus-5",
            "content": [{"type": "text", "text": text}] if text else [],
            "stop_reason": stop, "usage": {"input_tokens": 5, "output_tokens": 9}}


def openai(finish, text="Once."):
    return {"id": "cc_1", "model": "gpt-5",
            "choices": [{"index": 0, "finish_reason": finish,
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 9}}


def gemini(finish, text="Once."):
    return {"responseId": "g1", "modelVersion": "gemini-2.5-pro",
            "candidates": [{"index": 0, "finishReason": finish,
                            "content": {"role": "model",
                                        "parts": [{"text": text}] if text else []}}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 9}}


def responses(status="completed", reason=None, text="Once."):
    msg = {"id": "msg_1", "type": "message", "role": "assistant", "status": status,
           "content": [{"type": "output_text", "text": text, "annotations": []}]}
    return {"id": "resp_1", "object": "response", "model": "gpt-6-astra",
            "status": status,
            "incomplete_details": {"reason": reason} if reason else None,
            "output": [msg],
            "usage": {"input_tokens": 5, "output_tokens": 9}}


def endings(out):
    return [i["metadata"]["sampling"]["ended"] for i in
            sorted(out["items"], key=lambda i: i["id"])]


class TestTheNode:
    def test_anthropic(self, monkeypatch):
        out = run_chat(monkeypatch, "anthropic", "claude-opus-5", [
            anthropic("end_turn"), anthropic("max_tokens"), anthropic("stop_sequence"),
            anthropic("refusal", text="")])
        assert endings(out) == ["end", "max_tokens", "stop", "empty"]
        assert out["ended"] == {"end": 1, "stop": 1, "max_tokens": 1, "tool_call": 0,
                                "filtered": 0, "empty": 1, "other": 0}
        # The provider's own word stays where it was.
        assert sorted(i["metadata"]["call"]["stop_reason"] for i in out["items"]) == [
            "end_turn", "max_tokens", "refusal", "stop_sequence"]

    @pytest.mark.parametrize("provider,model", [
        ("openai", "gpt-5"), ("xai", "grok-4"), ("deepseek", "deepseek-chat")])
    def test_the_chat_completions_family(self, monkeypatch, provider, model):
        out = run_chat(monkeypatch, provider, model, [
            openai("stop"), openai("length"), openai("content_filter")],
            **({"api": "chat_completions"} if provider != "deepseek" else {}))
        assert endings(out) == ["end", "max_tokens", "filtered"]
        assert out["ended"]["max_tokens"] == 1

    def test_the_responses_api(self, monkeypatch):
        out = run_chat(monkeypatch, "openai", "gpt-6-astra", [
            responses(), responses("incomplete", "max_output_tokens")],
            api="responses")
        assert endings(out) == ["end", "max_tokens"]

    def test_gemini(self, monkeypatch):
        out = run_chat(monkeypatch, "gemini", "gemini-2.5-pro", [
            gemini("STOP"), gemini("MAX_TOKENS"), gemini("SAFETY")])
        assert endings(out) == ["end", "max_tokens", "filtered"]

    def test_the_header_is_present_with_every_ending_at_zero(self, monkeypatch):
        out = run_chat(monkeypatch, "anthropic", "claude-opus-5", [anthropic("end_turn")])
        assert out["ended"] == dict.fromkeys(ENDINGS, 0) | {"end": 1}

    def test_the_executor_carries_the_header(self):
        graph = {"dataflow": 2, "nodes": [{"id": "chat", "block": "text/chat",
                            "params": {"model": {"provider": "mock", "model": "m"},
                                       "budget_usd": 1.0, "max_tokens": 2},
                            "inputs": {"records": [{"id": "r0", "user": "hi"}]}}],
                 "edges": []}
        out = ProtocolExecutor().run(ProtocolSpec(
            kind="pipeline", prompt="", model_id=None, extra={"graph": graph}))
        node = out.payload["outputs"]["chat"]
        assert node["ended"]["max_tokens"] == 1
        assert node["items"][0]["metadata"]["sampling"]["ended"] == "max_tokens"
        assert out.payload["node_summaries"]["chat"]["ended"] == node["ended"]


# --- local ---------------------------------------------------------------------

class _Tok:
    def decode(self, ids):
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, turns, tokenize=False, add_generation_prompt=True, **kw):
        return repr(turns)

    def convert_tokens_to_ids(self, token):
        return 3

    unk_token_id = 3


def run_local_chat(monkeypatch, replies, **params):
    script = iter(replies)

    class FakeModel:
        tokenizer = _Tok()

    def sample(model, ids, *, max_tokens, return_ids=False, stop_strings=(), **kw):
        raw = next(script)
        cut = raw
        for s in stop_strings:
            if s in cut:
                cut = cut[:cut.index(s)]
        return cut, [ord(c) for c in raw[:max_tokens]]

    monkeypatch.setattr(distill, "encode", lambda t, text: [1, 2, 3])
    monkeypatch.setattr(distill, "prefill_decision", lambda m, ids: None)
    monkeypatch.setattr(generate, "sample_completion_cached", sample)
    return chat_mod.run_local(
        FakeModel(), mr.parse("google/gemma-3-4b-it"),
        [{"id": f"r{i}", "user": "hi"} for i in range(len(replies))],
        {"n": 1, "seed": 3, **params})


def test_a_local_chat_says_how_each_reply_ended(monkeypatch):
    out = run_local_chat(monkeypatch, ["short", "a very long reply", "yes. no"],
                         max_tokens=8, stop=["."])
    assert endings(out) == ["end", "max_tokens", "stop"]
    assert out["ended"] == dict.fromkeys(ENDINGS, 0) | {
        "end": 1, "max_tokens": 1, "stop": 1}
    # The field sits beside what the sampling record always had.
    assert list(out["items"][0]["metadata"]["sampling"]) == [
        "temperature", "top_p", "seed", "index", "ended"]


def test_a_local_reply_of_reasoning_alone_ends_empty(monkeypatch):
    import importlib

    # The package re-exports the function under the module's own name.
    run_local_mod = importlib.import_module("mechbench_compute.chat.run_local")

    monkeypatch.setattr(run_local_mod, "find_delimiters", lambda tok: ("<t>", "</t>"))
    monkeypatch.setattr(run_local_mod, "split_reasoning",
                        lambda text, d: (["thinking"], ""))
    out = run_local_chat(monkeypatch, ["<t>thinking and thinking"], max_tokens=8)
    item = out["items"][0]
    assert item["metadata"]["empty"]["cause"] == "reasoning"
    assert item["metadata"]["sampling"]["ended"] == "empty"
    assert out["ended"]["empty"] == 1


def test_generate_counts_its_items_by_ending(monkeypatch):
    script = iter(['Steampunk"', "Science Fiction and more", "Hum"])

    class _Model:
        tokenizer = _Tok()

    def sample(model, ids, *, max_tokens, return_ids=False, stop_strings=(), **kw):
        raw = next(script)
        cut = raw
        for s in stop_strings:
            if s in cut:
                cut = cut[:cut.index(s)]
        return cut, [ord(c) for c in raw[:max_tokens]]

    monkeypatch.setattr(ProtocolExecutor, "_model_loaded", lambda self, model_id: _Model())
    monkeypatch.setattr(ProtocolExecutor, "_run_model_block",
                        lambda self, fn, inputs, params, *a, **k: fn(inputs, params, *a, **k))
    monkeypatch.setattr(distill, "render_chat", lambda tok, s, u, p: f"<{s}|{u}>{p}")
    monkeypatch.setattr(distill, "encode", lambda tok, text: [ord(c) for c in text])
    monkeypatch.setattr(distill, "prefill_decision", lambda model, ids: ("cache", ids))
    monkeypatch.setattr(generate, "sample_completion_cached", sample)
    graph = {"dataflow": 2, "nodes": [{"id": "gen", "block": "text/generate",
                        "params": {"model": "fake/m@rev", "n": 3, "seed": 7,
                                   "stop": ['"'], "max_tokens": 20},
                        "inputs": {"records": [{"id": "p0", "user": "Pick."}]}}],
             "edges": []}
    out = ProtocolExecutor().run(ProtocolSpec(kind="pipeline", prompt="", model_id=None,
                                              extra={"graph": graph}))
    node = out.payload["outputs"]["gen"]
    assert [i["metadata"]["sampling"]["ended"] for i in node["items"]] == [
        "stop", "max_tokens", "end"]
    assert node["ended"] == dict.fromkeys(ENDINGS, 0) | {
        "stop": 1, "max_tokens": 1, "end": 1}


# --- a result stored before the field -----------------------------------------

def test_an_old_result_reads_with_the_field_absent_not_wrong():
    old = {"kind": "collection", "item_kind": "text/document", "items": [
        {"id": "r0-s0", "text": "Once.", "metadata": {
            "sampling": {"temperature": None, "max_tokens": 64, "seed": None, "index": 0},
            "call": {"provider": "anthropic", "stop_reason": "max_tokens"}}}]}
    assert "ended" not in old
    assert "ended" not in old["items"][0]["metadata"]["sampling"]
    # Counting it invents nothing: an item without the word is not
    # counted as any ending, least of all a natural one.
    assert count_endings(old["items"]) == dict.fromkeys(ENDINGS, 0)


def test_the_manifest_carries_each_generation_nodes_count():
    from mechbench_compute.protocol import summarize_node

    node = {"kind": "collection", "item_kind": "text/document", "items": [{}, {}],
            "ended": dict.fromkeys(ENDINGS, 0) | {"end": 1, "max_tokens": 1}}
    assert summarize_node(node)["ended"] == node["ended"]
    # A node without the count (any other op, or one stored before it) has none.
    assert "ended" not in summarize_node({**node, "ended": None})
