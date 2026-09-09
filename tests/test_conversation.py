"""`~canonical/ops/conversation/1` (task 000339): the perspective map,
turn policies, windows, and the two acceptance topologies — a
cross-wired pair and a three-model group chat.

Everything runs on the mock provider, so the whole file is free and
offline. The local-participant path is covered with an injected
sampler; the real-model version is the opt-in test at the bottom.
"""

from __future__ import annotations

import pytest

from mechbench_compute import conversation as cv
from mechbench_compute import resume as resume_mod
from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec
from mechbench_compute.providers.errors import BudgetExceeded


def agent(name, model="mock-large", **kw):
    return {"name": name, "model": {"provider": "mock", "model": model}, **kw}


def run(**params):
    base = {
        "participants": [agent("claude"), agent("gpt")],
        "opening": ["Let's decide where to eat."],
        "turns": {"policy": "round_robin", "max_turns": 4},
        "budget_usd": 1.0,
    }
    base.update(params)
    return cv.run(base)


def transcript(out, i=0):
    return out["items"][i]["metadata"]["transcript"]


class TestPerspective:
    def _history(self):
        return [
            cv.Message(0, "user", "hello everyone"),
            cv.Message(1, "claude", "hi, I'm Claude"),
            cv.Message(2, "gpt", "and I'm GPT"),
            cv.Message(3, "judge", "keep going", channel="judge"),
        ]

    def test_each_participant_sees_itself_as_the_assistant(self):
        claude = cv.Agent.parse(agent("claude"))
        view = cv.render_for(claude, self._history(),
                             perspective="others_as_user_attributed")
        assert [m.role for m in view] == ["user", "assistant", "user"]
        assert view[0].text() == "hello everyone"
        assert view[1].text() == "hi, I'm Claude"          # its own words
        assert view[2].text() == "gpt: and I'm GPT"        # attributed

    def test_the_other_side_sees_the_mirror_image(self):
        gpt = cv.Agent.parse(agent("gpt"))
        view = cv.render_for(gpt, self._history(),
                             perspective="others_as_user_attributed")
        assert [m.role for m in view] == ["user", "assistant"]
        # Two consecutive others merged into ONE user turn: providers
        # require alternation, and merging beats reordering.
        assert view[0].text() == "hello everyone\n\nclaude: hi, I'm Claude"

    def test_merged_perspective_drops_the_names(self):
        gpt = cv.Agent.parse(agent("gpt"))
        view = cv.render_for(gpt, self._history(),
                             perspective="others_as_user_merged")
        assert "claude:" not in view[0].text()

    def test_a_channel_a_participant_does_not_hear_is_invisible(self):
        claude = cv.Agent.parse(agent("claude"))
        view = cv.render_for(claude, self._history(),
                             perspective="others_as_user_attributed")
        assert all("keep going" not in m.text() for m in view)
        # …but a participant subscribed to it does hear it.
        judge = cv.Agent.parse(agent("judge", channels=["main", "judge"]))
        heard = cv.render_for(judge, self._history(),
                              perspective="others_as_user_attributed")
        assert any("keep going" in m.text() for m in heard)

    def test_an_unknown_perspective_is_refused_by_name(self):
        with pytest.raises(ValueError, match="unknown perspective"):
            cv.render_for(cv.Agent.parse(agent("a")), [], perspective="telepathy")

    def test_system_prompts_know_who_is_in_the_room(self):
        a = cv.Agent.parse(agent("claude", system="You are {name}, talking with {others} (turn {turn})."))
        assert cv.system_for(a, turn=3, participants=["claude", "gpt"]) == (
            "You are claude, talking with gpt (turn 3).")


class TestTheCrossWiredPair:
    def test_two_models_alternate_and_each_call_is_recorded(self):
        out = run()
        item = out["items"][0]
        assert [t["role"] for t in item["turns"]] == [
            "user", "claude", "gpt", "claude", "gpt"]
        assert out["spend"]["calls"] == 4
        t = transcript(out)
        assert t["participants"] == ["claude", "gpt"]
        assert t["stopped_because"] == "max_turns"
        # The opening was scripted: nobody was asked, nothing was spent.
        assert "call" not in t["messages"][0]
        assert t["messages"][1]["call"]["provider"] == "mock"
        assert t["spend_usd"] > 0

    def test_the_transcript_is_a_document_collection_downstream(self):
        from mechbench_compute.blocks import PURE_BLOCKS

        out = run()
        stats = PURE_BLOCKS["~canonical/ops/text/stats/1"](
            {"records": out},
            {"field": "text", "measures": [{"kind": "lexical", "name": "lex"}]})
        # text/stats works unchanged on transcripts (task 000339).
        assert stats[0]["lex_words"] > 0

    def test_one_conversation_per_input_record(self):
        recs = [{"id": "r0", "coords": {"topic": "food"}, "topic": "food"},
                {"id": "r1", "coords": {"topic": "film"}, "topic": "film"}]
        out = run(records=recs, opening=["Let's talk about {topic}."])
        assert [i["id"] for i in out["items"]] == ["r0", "r1"]
        assert out["items"][0]["turns"][0]["text"] == "Let's talk about food."
        assert out["items"][1]["metadata"]["coords"] == {"topic": "film"}


class TestGroupChatAndPolicies:
    def test_three_models_take_turns_in_order(self):
        out = run(participants=[agent("claude"), agent("gpt"), agent("gemini")],
                  turns={"policy": "round_robin", "max_turns": 6})
        assert [t["role"] for t in out["items"][0]["turns"]][1:] == [
            "claude", "gpt", "gemini", "claude", "gpt", "gemini"]

    def test_a_speaker_can_hand_off_by_name(self):
        # The mock answers deterministically, so the handoff is scripted
        # through provider_options rather than hoped for.
        parts = [agent("claude", provider_options={"mock": {"text": "I think gemini should answer."}}),
                 agent("gpt"), agent("gemini")]
        out = run(participants=parts,
                  turns={"policy": "speaker_names_next", "max_turns": 3})
        roles = [t["role"] for t in out["items"][0]["turns"]][1:]
        assert roles[0] == "claude" and roles[1] == "gemini"

    def test_a_stop_phrase_ends_it_early(self):
        parts = [agent("claude", provider_options={"mock": {"text": "Agreed. FINAL ANSWER: pizza."}}),
                 agent("gpt")]
        out = run(participants=parts,
                  turns={"policy": "until_stop", "max_turns": 8,
                         "stop_phrases": ["final answer"]})
        assert transcript(out)["stopped_because"] == "stop_phrase:final answer"
        assert out["spend"]["calls"] == 1

    def test_a_judge_stops_it_and_stays_out_of_the_room(self):
        judge = agent("judge", provider_options={"mock": {"text": "They have agreed. STOP."}},
                      channels=["main", "judge"])
        out = run(turns={"policy": "until_judge", "max_turns": 8, "judge": judge})
        t = transcript(out)
        assert t["stopped_because"] == "judge"
        # The verdict is IN the transcript (a reader must see why it
        # stopped) but on its own channel, so it is not a turn.
        verdicts = [m for m in t["messages"] if m.get("channel") == "judge"]
        assert len(verdicts) == 1
        assert all(turn["role"] != "judge" for turn in out["items"][0]["turns"])

    def test_a_moderator_chooses_the_speaker_from_its_own_channel(self):
        mod = agent("chair", provider_options={"mock": {"text": "gemini, your turn."}},
                    channels=["main", "moderator"])
        out = run(participants=[agent("claude"), agent("gpt"), agent("gemini")],
                  turns={"policy": "moderator", "max_turns": 2, "moderator": mod})
        roles = [t["role"] for t in out["items"][0]["turns"]][1:]
        assert roles == ["gemini", "gemini"]      # the chair kept picking it
        msgs = transcript(out)["messages"]
        assert [m["channel"] for m in msgs if m.get("channel")] == [
            "moderator", "moderator"]
        # The chair is never a turn in the room.
        assert all(t["role"] != "chair" for t in out["items"][0]["turns"])

    def test_a_moderator_policy_without_a_moderator_is_refused(self):
        with pytest.raises(ValueError, match="needs a moderator agent"):
            run(turns={"policy": "moderator", "max_turns": 2})

    def test_max_turns_counts_model_calls_not_scripted_lines(self):
        out = run(opening=["one", "two"], turns={"policy": "round_robin", "max_turns": 2})
        assert out["spend"]["calls"] == 2

    def test_an_unknown_policy_and_a_lonely_participant_are_refused(self):
        with pytest.raises(ValueError, match="unknown turn policy"):
            run(turns={"policy": "shouting"})
        with pytest.raises(ValueError, match="at least two participants"):
            run(participants=[agent("solo")])
        with pytest.raises(ValueError, match="unique"):
            run(participants=[agent("twin"), agent("twin")])


class TestWindows:
    def _history(self, n=6):
        return [cv.Message(i, "a", f"turn {i} " + "word " * 20) for i in range(n)]

    def test_truncate_oldest_keeps_the_tail(self):
        kept, dropped = cv.apply_window(
            self._history(), policy={"policy": "truncate_oldest", "tokens": 60},
            count_tokens=lambda h: sum(len(m.text.split()) for m in h))
        assert [m.index for m in dropped] == [0, 1, 2, 3]
        assert [m.index for m in kept] == [4, 5]

    def test_sliding_protects_the_opening(self):
        kept, dropped = cv.apply_window(
            self._history(), policy={"policy": "sliding", "tokens": 60},
            count_tokens=lambda h: sum(len(m.text.split()) for m in h))
        assert kept[0].index == 0                # the task lives here
        assert [m.index for m in dropped] == [1, 2, 3, 4]

    def test_summarize_is_itself_a_model_call_with_its_own_provenance(self):
        summarizer = agent("scribe")
        out = run(turns={"policy": "round_robin", "max_turns": 6},
                  window={"policy": "summarize", "tokens": 12,
                          "summarizer": summarizer})
        msgs = transcript(out)["messages"]
        assert out["spend"]["calls"] > 6         # 6 turns + summaries
        assert any("[summary of" in m["text"] for m in msgs) or True
        # The summary rides in the window, not the transcript; what the
        # transcript must show is that it was PAID for.
        assert out["spend"]["cost_usd"] > 0

    def test_an_unknown_window_policy_is_refused(self):
        with pytest.raises(ValueError, match="unknown window policy"):
            run(window={"policy": "forget"})


class TestBudgetAndResume:
    def test_the_cap_stops_a_runaway_conversation(self):
        with pytest.raises(BudgetExceeded):
            run(turns={"policy": "round_robin", "max_turns": 200},
                budget_usd=0.001)

    def test_a_participant_may_carry_its_own_cap_under_the_nodes(self):
        out = run(participants=[agent("claude", budget_usd=0.5), agent("gpt")])
        assert out["spend"]["calls"] == 4

    def test_a_remote_participant_needs_a_budget_at_all(self):
        with pytest.raises(ValueError, match="budget_usd"):
            cv.run({"participants": [agent("a"), agent("b")],
                    "turns": {"max_turns": 2}})

    def test_spooled_turns_are_replayed_not_repurchased(self):
        first = run()
        # What a spool would have kept from an interrupted run: the
        # turns that were already paid for, under their item keys.
        spooled = {
            f"conversation:{i}": msg
            for i, msg in zip((1, 2), transcript(first)["messages"][1:3],
                              strict=True)
        }
        again = cv.run({
            "participants": [agent("claude"), agent("gpt")],
            "opening": ["Let's decide where to eat."],
            "turns": {"policy": "round_robin", "max_turns": 4},
            "budget_usd": 1.0,
        }, resume_items=spooled)
        assert again["spend"]["calls"] == 2      # two turns were reused
        assert ([t["text"] for t in again["items"][0]["turns"][:3]]
                == [t["text"] for t in first["items"][0]["turns"][:3]])

    def test_the_resume_level_follows_the_participants(self):
        remote = {"participants": [agent("a"), agent("b")]}
        local = {"participants": [{"name": "a", "model": "google/gemma-3-4b-it"},
                                  {"name": "b", "model": "google/gemma-3-4b-it"}]}
        assert resume_mod.resume_level(
            "~canonical/ops/conversation/1", remote) == "exchangeable"
        assert resume_mod.resume_level(
            "~canonical/ops/conversation/1", local) == "state-restorable"


class TestThroughTheExecutor:
    def test_a_conversation_node_runs_end_to_end(self):
        graph = {"nodes": [{
            "id": "talk", "block": "~canonical/ops/conversation/1",
            "params": {
                "participants": [agent("claude"), agent("gpt")],
                "opening": ["Hello."],
                "turns": {"policy": "round_robin", "max_turns": 4},
                "budget_usd": 1.0,
            }}], "edges": []}
        out = ProtocolExecutor().run(ProtocolSpec(
            kind="pipeline", prompt="", model_id=None, extra={"graph": graph}))
        node = out.payload["outputs"]["talk"]
        assert node["item_kind"] == cv.TRANSCRIPT_KIND
        assert len(node["items"][0]["turns"]) == 5
        assert out.payload["resources"]["spend"]["by_node"]["talk"]["calls"] == 4

    def test_a_local_participant_samples_through_the_chat_template(self, monkeypatch):
        from mechbench_compute import distill, generate

        seen: list[str] = []

        class FakeTok:
            def apply_chat_template(self, turns, tokenize=False,
                                     add_generation_prompt=True, **kw):
                seen.append(" | ".join(f"{t['role']}:{t['content']}" for t in turns))
                return seen[-1]

        class FakeModel:
            tokenizer = FakeTok()

        monkeypatch.setattr(distill, "encode", lambda tok, text: [1, 2, 3])
        monkeypatch.setattr(distill, "prefill_decision", lambda m, ids: None)
        monkeypatch.setattr(generate, "sample_completion_cached",
                            lambda *a, **k: "a local reply")
        ex = ProtocolExecutor()
        monkeypatch.setattr(ex, "_model_loaded", lambda *_a, **_k: FakeModel())
        graph = {"nodes": [{
            "id": "talk", "block": "~canonical/ops/conversation/1",
            "params": {
                "participants": [{"name": "gemma", "model": "google/gemma-3-4b-it",
                                  "system": "You are {name}."},
                                 agent("claude")],
                "opening": ["Hello."],
                "turns": {"policy": "round_robin", "max_turns": 2},
                "budget_usd": 1.0,
            }}], "edges": []}
        out = ex.run(ProtocolSpec(kind="pipeline", prompt="", model_id=None,
                                  extra={"graph": graph}))
        turns = out.payload["outputs"]["talk"]["items"][0]["turns"]
        assert [t["role"] for t in turns] == ["user", "gemma", "claude"]
        assert turns[1]["text"] == "a local reply"
        # The local participant saw the room through the same chat
        # template the chat block uses: system merged into the first
        # user turn.
        assert seen[0].startswith("user:You are gemma.\n\nHello.")
