from __future__ import annotations

import pytest

from mechbench_compute import transcript as TR
from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec
from mechbench_compute.ops.text.extend import extend
from mechbench_compute.ops.text.render import SEES_DEFAULT
from mechbench_compute.ops.text.render import parse_sees
from mechbench_compute.ops.text.render import render
from mechbench_compute.ops.text.render import render_records


def _msg(i, who, text, **kw):
    return {"index": i, "participant": who, "role_as_seen": "assistant", "text": text, **kw}


HISTORY = [
    _msg(0, "user", "hello everyone"),
    _msg(1, "ana", "four.", thinking="two plus two is four"),
    _msg(2, "bo", "are you sure?", thinking="she might be bluffing"),
    _msg(3, "judge", "keep going", channel="judge"),
    _msg(4, "ana", "yes.", thinking="I am"),
]


class TestSees:
    def test_the_default_replays_nothing(self):
        assert parse_sees(None) == {"own_thinking": "none", "others_thinking": "none"}

    def test_the_old_boolean_reads_as_the_clause_it_meant(self):
        assert parse_sees(True)["own_thinking"] == "full"
        assert parse_sees(False) == SEES_DEFAULT

    def test_the_grammar_is_checked_by_name(self):
        assert parse_sees({"own_thinking": {"last_turns": 2}})["own_thinking"] == {"last_turns": 2}
        with pytest.raises(ValueError, match="unknown key"):
            parse_sees({"thinking": "full"})
        with pytest.raises(ValueError, match="own_thinking"):
            parse_sees({"own_thinking": "some"})
        with pytest.raises(ValueError, match="non-negative integer"):
            parse_sees({"others_thinking": {"last_turns": -1}})


def _said(view):
    return "\n".join(m["content"] for m in view)


class TestRender:
    def test_roles_attribution_and_channels(self):
        view = render(HISTORY, participant="ana")
        assert [m["role"] for m in view] == ["user", "assistant", "user", "assistant"]
        assert view[0]["content"] == "hello everyone"
        assert view[2]["content"] == "bo: are you sure?"
        assert "keep going" not in _said(view)

    def test_the_room_hears_answers_not_scratchpads(self):
        assert "two plus two" not in _said(render(HISTORY, participant="bo"))
        assert "two plus two" not in _said(render(HISTORY, participant="ana"))

    def test_own_thinking_full_and_last_turns(self):
        full = render(HISTORY, participant="ana", sees={"own_thinking": "full"})
        assert full[1]["content"] == "two plus two is four\n\nfour."
        assert full[3]["content"] == "I am\n\nyes."
        last = render(HISTORY, participant="ana", sees={"own_thinking": {"last_turns": 1}})
        assert last[1]["content"] == "four."
        assert last[3]["content"] == "I am\n\nyes."

    def test_others_thinking_is_a_choice_and_is_marked(self):
        view = render(HISTORY, participant="ana", sees={"others_thinking": "full"})
        assert view[2]["content"] == "bo (thinking): she might be bluffing\n\nbo: are you sure?"
        merged = render(HISTORY, participant="ana", perspective="others_as_user_merged",
                           sees={"others_thinking": {"truncate_words": 2}})
        assert merged[2]["content"] == "(thinking) she might\n\nare you sure?"

    def test_consecutive_user_turns_merge(self):
        h = [_msg(0, "user", "a"), _msg(1, "bo", "b"), _msg(2, "cy", "c"), _msg(3, "ana", "d")]
        view = render(h, participant="ana")
        assert [m["role"] for m in view] == ["user", "assistant"]
        assert view[0]["content"] == "a\n\nbo: b\n\ncy: c"


def _transcripts():
    return {"kind": "collection", "item_kind": "text/transcript", "items": [
        {"id": "c1", "kind": "text/transcript", "participants": ["ana", "bo"],
         "messages": HISTORY[:3], "stopped": "", "coords": {"topic": "sums"}},
        {"id": "c2", "kind": "text/transcript", "participants": ["ana", "bo"],
         "messages": [_msg(0, "user", "hi"), _msg(1, "ana", "hello")], "stopped": "", "coords": {"topic": "greetings"}},
    ]}


class TestRenderRecords:
    def test_one_chat_shaped_record_per_transcript(self):
        out = render_records({"transcripts": _transcripts()},
                                {"participant": "bo", "system": "You are {name}; the others are {others}. Turn {turn}."})
        assert out["item_kind"] == "records/record" and out["participant"] == "bo"
        [r1, r2] = out["items"]
        assert r1["id"] == "c1" and r1["coords"] == {"topic": "sums", "conversation": "c1", "participant": "bo"}
        assert r1["system"] == "You are bo; the others are ana. Turn 3."
        assert [m["role"] for m in r1["messages"]] == ["user", "assistant"]
        assert r2["messages"] == [{"role": "user", "content": "hi\n\nana: hello"}]

    def test_a_transcript_without_messages_is_refused(self):
        with pytest.raises(ValueError, match="no `messages`"):
            render_records({"transcripts": [{"id": "x", "turns": []}]}, {"participant": "bo"})


class TestExtend:
    def _replies(self, **texts):
        return {"kind": "collection", "item_kind": "text/document", "items": [
            {"id": f"{cid}-s0", "kind": "text/document", "text": text,
             "coords": {"conversation": cid, "participant": "bo", "sample": 0},
             "metadata": {"call": {"provider": "mock", "model": "m"}}}
            for cid, text in texts.items()]}

    def test_each_transcript_grows_by_the_reply_that_names_it(self):
        out = extend({"transcripts": _transcripts(), "replies": self._replies(c1="<think>hmm</think>no", c2="hey")},
                        {"participant": "bo"})
        assert out["item_kind"] == "text/transcript"
        [t1, t2] = out["items"]
        last = t1["messages"][-1]
        assert last == {"index": 3, "participant": "bo", "role_as_seen": "assistant", "text": "no",
                        "thinking": "hmm", "call": {"provider": "mock", "model": "m"}}
        assert t1["turns"][-1] == {"role": "bo", "text": "no"} and t1["text"].endswith("bo: no")
        assert t1["coords"] == {"topic": "sums"} and t1["participants"] == ["ana", "bo"]
        assert t2["messages"][-1]["text"] == "hey" and "thinking" not in t2["messages"][-1]

    def test_a_new_speaker_joins_the_participants(self):
        out = extend({"transcripts": _transcripts(), "replies": self._replies(c1="x", c2="y")}, {"participant": "cy"})
        assert out["items"][0]["participants"] == ["ana", "bo", "cy"]

    def test_two_replies_or_none_are_refused(self):
        with pytest.raises(ValueError, match="has 0 replies"):
            extend({"transcripts": _transcripts(), "replies": self._replies(c1="x")}, {"participant": "bo"})
        two = self._replies(c1="x", c2="y")
        two["items"].append(dict(two["items"][0], id="c1-s1"))
        with pytest.raises(ValueError, match="has 2 replies"):
            extend({"transcripts": _transcripts(), "replies": two}, {"participant": "bo"})

    def test_a_reply_that_names_no_conversation_is_refused(self):
        with pytest.raises(ValueError, match="names no conversation"):
            extend({"transcripts": _transcripts(),
                       "replies": [{"id": "r", "text": "x", "coords": {}}]}, {"participant": "bo"})


CONVERSE_TURN = {"participant": "ana", "index": 1,
                 "text": "glass sable meridian sable wick meridian"}


class TestATurnComposedFromChat:
    def _run(self, graph):
        out = ProtocolExecutor().run(ProtocolSpec(kind="pipeline", prompt="", model_id=None,
                                                  extra={"graph": graph}))
        payload = out.payload if hasattr(out, "payload") else out
        return payload["outputs"]

    def test_the_composed_turn_is_the_loops_turn(self):
        model = {"provider": "mock", "model": "mock-large"}
        start = {"kind": "collection", "item_kind": "text/transcript", "items": [
            {"id": "c1", "kind": "text/transcript", "participants": ["ana", "bo"], "stopped": "",
             "messages": [_msg(0, "user", "Let's decide where to eat.")]}]}
        def node(nid, block, params, inputs=None):
            return {"id": nid, "block": block, "params": params, "inputs": inputs or {}}
        outs = self._run({"dataflow": 2, "nodes": [
            node("view", "text/render", {"participant": "ana", "system": "Be {name}."}, {"transcripts": start}),
            node("say", "text/chat", {"model": model, "max_tokens": 1024, "budget_usd": 1.0}),
            node("next", "text/extend", {"participant": "ana"}, {"transcripts": start}),
        ], "edges": [
            {"from": {"node": "view"}, "to": {"node": "say", "port": "records"}},
            {"from": {"node": "say"}, "to": {"node": "next", "port": "replies"}},
        ]})
        composed = outs["next"]["items"][0]["messages"][-1]
        assert composed["text"] == CONVERSE_TURN["text"]
        assert composed["participant"] == CONVERSE_TURN["participant"]
        assert composed["index"] == CONVERSE_TURN["index"]


class TestWhoIsScripted:
    HISTORY = [_msg(0, "narrator", "You are at a crossroads."),
               _msg(1, "ana", "Left."), _msg(2, "bo", "Right.")]

    def test_a_non_participant_is_not_attributed_whatever_it_is_called(self):
        view = render(self.HISTORY, participant="ana", participants=["ana", "bo"])
        assert view[0]["content"] == "You are at a crossroads."
        assert view[2]["content"] == "bo: Right."

    def test_a_participant_called_user_is_attributed_like_anyone_else(self):
        history = [_msg(0, "user", "I think left."), _msg(1, "ana", "Left it is.")]
        view = render(history, participant="ana", participants=["user", "ana"])
        assert view[0]["content"] == "user: I think left."

    def test_a_transcript_that_lists_nobody_keeps_the_old_convention(self):
        view = render(self.HISTORY, participant="ana")
        assert view[0]["content"] == "narrator: You are at a crossroads."
        scripted = render([_msg(0, "user", "hello"), _msg(1, "ana", "hi")], participant="ana")
        assert scripted[0]["content"] == "hello"
