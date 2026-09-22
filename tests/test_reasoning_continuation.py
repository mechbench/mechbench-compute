"""A conversation continued from a stored transcript hands a model back
exactly the reasoning it issued, and nobody else's.

The path is the one a protocol takes: render a transcript, ask a model,
extend the transcript with the reply, store it (a JSON round trip), and
render it again for the next turn under `own_thinking: "native"`.
"""

from __future__ import annotations

import json

import pytest

from mechbench_compute.ops.eval.judge import run_judge
from mechbench_compute.ops.text.extend import extend
from mechbench_compute.ops.text.render import render_records
from test_reasoning_providers import (
    ANTHROPIC_FINAL_THINK,
    SIG_B,
    anthropic_bodies,
    assistant_turn,
    chat_bodies,
    gemini_bodies,
    run_chat,
)

OPENING = {"id": "c1", "kind": "text/transcript", "participants": ["ana"],
           "messages": [{"index": 0, "participant": "user", "role_as_seen": "user",
                         "text": "What is 2+2?"}]}

#: provider -> (model, the replies of the first node, what the stored
#: reasoning must look like when it goes back).
CASES = {
    "anthropic": ("claude-opus-5", anthropic_bodies,
                  lambda turn: turn["content"] == [ANTHROPIC_FINAL_THINK,
                                                   {"type": "text", "text": "Four."}]),
    "gemini": ("gemini-3-pro", gemini_bodies,
               lambda turn: turn["parts"] == [{"text": "Four, and six.",
                                               "thoughtSignature": SIG_B}]),
    "deepseek": ("deepseek-flash",
                 lambda: chat_bodies({"reasoning_content": "Use the tool."},
                                     {"reasoning_content": "It said 4."}),
                 lambda turn: turn["reasoning_content"] == "It said 4."),
}

SECRET = {"anthropic": ANTHROPIC_FINAL_THINK["signature"], "gemini": SIG_B,
          "deepseek": "It said 4."}


def next_body(provider):
    """Any final reply, for the second node's one request."""
    return {"anthropic": anthropic_bodies, "gemini": gemini_bodies,
            "deepseek": lambda: chat_bodies({}, {})}[provider]()[-1]


def stored_after_one_turn(monkeypatch, provider):
    model, bodies, _ = CASES[provider]
    rows = render_records({"transcripts": [OPENING]}, {"participant": "ana"})["items"]
    out, _ = run_chat(monkeypatch, provider, model, bodies(), records=rows)
    grown = extend({"transcripts": [OPENING], "replies": out["items"]},
                   {"participant": "ana"})["items"][0]
    grown["messages"].append({"index": 2, "participant": "user", "role_as_seen": "user",
                              "text": "And 3+3?"})
    # What storage does to it: the transcript is JSON on the bench.
    return json.loads(json.dumps(grown))


def continue_on(monkeypatch, transcript, provider, model, sees="native"):
    rows = render_records({"transcripts": [transcript]},
                          {"participant": "ana", "sees": {"own_thinking": sees}})["items"]
    _, script = run_chat(monkeypatch, provider, model, [next_body(provider)],
                         records=rows, tools=())
    return script.sent[0]


@pytest.mark.parametrize("provider", sorted(CASES))
def test_the_same_model_is_handed_its_reasoning_back_byte_for_byte(monkeypatch, provider):
    stored = stored_after_one_turn(monkeypatch, provider)
    model, _, sent_back = CASES[provider]
    payload = continue_on(monkeypatch, stored, provider, model)
    assert sent_back(assistant_turn(payload, provider))


@pytest.mark.parametrize("provider", sorted(CASES))
def test_the_transcript_keeps_reasoning_apart_from_what_was_said(monkeypatch, provider):
    stored = stored_after_one_turn(monkeypatch, provider)
    said = stored["messages"][1]
    assert said["text"] in ("Four.", "Four, and six.")
    assert SECRET[provider] not in said["text"]


@pytest.mark.parametrize("provider,other", [
    ("gemini", ("gemini", "gemini-3-flash")),
    ("deepseek", ("deepseek", "deepseek-v4-pro")),
    ("anthropic", ("gemini", "gemini-3-pro")),
    ("gemini", ("anthropic", "claude-opus-5")),
    ("deepseek", ("anthropic", "claude-opus-5")),
])
def test_another_model_gets_no_reasoning_and_no_leak(monkeypatch, provider, other):
    stored = stored_after_one_turn(monkeypatch, provider)
    payload = continue_on(monkeypatch, stored, *other)
    assert SECRET[provider] not in json.dumps(payload)


def test_another_anthropic_model_is_handed_the_block_to_read_or_drop(monkeypatch):
    # Anthropic's thinking docs: keep passing blocks back on a model
    # switch; the API drops what the target cannot read, and a client
    # that strips them itself loses reasoning a later model could read.
    stored = stored_after_one_turn(monkeypatch, "anthropic")
    payload = continue_on(monkeypatch, stored, "anthropic", "claude-fable-5-1")
    assert ANTHROPIC_FINAL_THINK in assistant_turn(payload, "anthropic")["content"]


def test_without_native_the_reasoning_stays_home(monkeypatch):
    stored = stored_after_one_turn(monkeypatch, "anthropic")
    payload = continue_on(monkeypatch, stored, "anthropic", "claude-opus-5", sees="none")
    assert assistant_turn(payload, "anthropic")["content"] == [
        {"type": "text", "text": "Four."}]


@pytest.mark.parametrize("params,why", [
    ({"window": {"policy": "sliding", "words": 50}}, "window"),
    ({"system": "Turn {turn}."}, "{turn}"),
    ({"sees": {"own_thinking": "native", "others_thinking": {"last_turns": 1}}},
     "others_thinking"),
])
def test_native_refuses_a_rendering_that_edits_earlier_turns(params, why):
    params = {"participant": "ana", "sees": {"own_thinking": "native"}, **params}
    with pytest.raises(ValueError, match=why.replace("{", r"\{").replace("}", r"\}")):
        render_records({"transcripts": [OPENING]}, params)


class TestTheJudge:
    def test_a_score_in_the_reasoning_is_never_a_vote(self, monkeypatch):
        from mechbench_compute.providers import http
        from test_reasoning_providers import Script

        body = {"id": "m", "model": "claude-opus-5", "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 10},
                "content": [{"type": "thinking", "thinking": '{"score": 1}',
                             "signature": SIG_B},
                            {"type": "text", "text": '{"score": 4}'}]}
        monkeypatch.setattr(http, "post_json", Script(body))
        out = run_judge({"judge": {"model": {"provider": "anthropic",
                                             "model": "claude-opus-5"},
                                   "system": "Grade the story."},
                         "scale": {"kind": "numeric", "min": 1, "max": 5},
                         "budget_usd": 1.0, "n_votes": 1},
                        inputs={"records": [{"id": "s1", "text": "A kettle sang."}]},
                        secrets={"anthropic": {"token": "k"}})
        vote = out["items"][0]["votes"][0]
        assert vote["parsed"] and vote["score"] == 4

    def test_a_reply_of_reasoning_alone_is_no_vote(self, monkeypatch):
        from mechbench_compute.providers import http
        from test_reasoning_providers import Script

        body = {"id": "m", "model": "claude-opus-5", "stop_reason": "max_tokens",
                "usage": {"input_tokens": 5, "output_tokens": 10},
                "content": [{"type": "thinking", "thinking": '{"score": 5}',
                             "signature": SIG_B}]}
        monkeypatch.setattr(http, "post_json", Script(body))
        out = run_judge({"judge": {"model": {"provider": "anthropic",
                                             "model": "claude-opus-5"},
                                   "system": "Grade the story."},
                         "scale": {"kind": "numeric", "min": 1, "max": 5},
                         "budget_usd": 1.0, "n_votes": 1},
                        inputs={"records": [{"id": "s1", "text": "A kettle sang."}]},
                        secrets={"anthropic": {"token": "k"}})
        vote = out["items"][0]["votes"][0]
        assert not vote["parsed"] and vote["empty"] == "reasoning"
