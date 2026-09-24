from __future__ import annotations

import pytest

from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec
from mechbench_compute.ops.text.render import render


def _run(graph, resume_items=None):
    ex = ProtocolExecutor()
    out = ex.run(ProtocolSpec(kind="pipeline", prompt="", model_id=None, extra={"graph": graph}))
    payload = out.payload if hasattr(out, "payload") else out
    return payload["outputs"]


def node(nid, block, params, inputs=None):
    return {"id": nid, "block": block, "params": params, "inputs": inputs or {}}


COUNTERS = {"kind": "collection", "item_kind": "records/record",
            "items": [{"id": "a", "n": 0, "coords": {"g": "x"}}, {"id": "b", "n": 10, "coords": {"g": "y"}}]}

STEP_BODY = {"nodes": [
    node("bump", "records/fill", {"templates": {"m": "{n}"}}),
], "edges": [{"from": {"input": "state"}, "to": {"node": "bump", "port": "records"}}]}


class TestTheFold:
    def test_the_state_threads_through_the_steps(self):
        body = {"dataflow": 2, "nodes": [node("stamp", "records/fill", {"templates": {"stamp": {"$param": "step"}}})],
                "edges": [{"from": {"input": "state"}, "to": {"node": "stamp", "port": "records"}}]}
        out = _run({"dataflow": 2, "nodes": [
            node("f", "records/fold", {"body": body, "steps": 3}, {"state": COUNTERS})], "edges": []})["f"]
        assert [it["stamp"] for it in out["items"]] == ["2", "2"]
        assert [it["id"] for it in out["items"]] == ["a", "b"]
        assert out["folded"] == {"steps": 3, "stopped": "steps", "body_nodes": ["stamp"]}

    def test_over_binds_per_step_and_cycles(self):
        body = {"nodes": [node("who", "records/fill", {"templates": {"last": {"$param": "who"}}})],
                "edges": [{"from": {"input": "state"}, "to": {"node": "who", "port": "records"}}]}
        out = _run({"dataflow": 2, "nodes": [
            node("f", "records/fold", {"body": body, "over": [{"who": "ana"}, {"who": "bo"}], "steps": 3},
                 {"state": COUNTERS})], "edges": []})["f"]
        assert out["items"][0]["last"] == "ana"

    def test_until_stops_when_every_item_says_so(self):
        body = {"nodes": [node("mark", "records/fill", {"templates": {"done": {"$param": "flag"}}})],
                "edges": [{"from": {"input": "state"}, "to": {"node": "mark", "port": "records"}}]}
        out = _run({"dataflow": 2, "nodes": [
            node("f", "records/fold", {"body": body, "steps": 5, "until": {"field": "done"},
                                       "over": [{"flag": ""}, {"flag": "yes"}, {"flag": ""}]},
                 {"state": COUNTERS})], "edges": []})["f"]
        assert out["folded"]["stopped"] == "until" and out["folded"]["steps"] == 2

    def test_refusals_by_name(self):
        with pytest.raises(ValueError, match="needs an input on its 'state' port"):
            _run({"dataflow": 2, "nodes": [node("f", "records/fold", {"body": STEP_BODY, "steps": 1})], "edges": []})
        with pytest.raises(ValueError, match="at least one step"):
            _run({"dataflow": 2, "nodes": [node("f", "records/fold", {"body": STEP_BODY, "steps": 0},
                                                {"state": COUNTERS})], "edges": []})


class TestAConversationIsAFold:
    MODEL = {"provider": "mock", "model": "mock-large"}

    def _fold_graph(self, turns, extra_extend=None):
        start = {"kind": "collection", "item_kind": "text/transcript", "items": [
            {"id": "c1", "kind": "text/transcript", "participants": ["ana", "bo"], "stopped": "",
             "messages": [{"index": 0, "participant": "user", "role_as_seen": "assistant",
                           "text": "Let's decide where to eat."}]}]}
        body = {"nodes": [
            node("view", "text/render", {"participant": {"$param": "participant"}, "system": "Be {name}."}),
            node("say", "text/chat", {"model": self.MODEL, "max_tokens": 1024, "budget_usd": 1.0}),
            node("next", "text/extend", {"participant": {"$param": "participant"}, **(extra_extend or {})}),
        ], "edges": [
            {"from": {"input": "state"}, "to": {"node": "view", "port": "transcripts"}},
            {"from": {"node": "view"}, "to": {"node": "say", "port": "records"}},
            {"from": {"input": "state"}, "to": {"node": "next", "port": "transcripts"}},
            {"from": {"node": "say"}, "to": {"node": "next", "port": "replies"}},
        ]}
        return {"dataflow": 2, "nodes": [
            node("talk", "records/fold",
                 {"body": body, "over": [{"participant": "ana"}, {"participant": "bo"}],
                  "steps": turns, "output": "next", "until": {"field": "stopped"}},
                 {"state": start})], "edges": []}

    CONVERSE_SAID = [
            {
                    "participant": "user",
                    "text": "Let's decide where to eat."
            },
            {
                    "participant": "ana",
                    "text": "glass sable meridian sable wick meridian"
            },
            {
                    "participant": "bo",
                    "text": "cadence ember quartz furrow lantern quartz"
            },
            {
                    "participant": "ana",
                    "text": "tide yarrow sable lantern tundra vellum marrow wick glass cadence harbor yarrow vellum wick vellum wick cadence"
            },
            {
                    "participant": "bo",
                    "text": "yarrow ledger salt lantern furrow tide"
            }
    ]

    def test_four_turns_are_the_loops_four_turns(self):
        fold = _run(self._fold_graph(4))["talk"]
        assert fold["item_kind"] == "text/transcript" and fold["folded"]["steps"] == 4
        [t] = fold["items"]
        assert [m["participant"] for m in t["messages"]] == ["user", "ana", "bo", "ana", "bo"]
        assert [{"participant": m["participant"], "text": m["text"]} for m in t["messages"]] == self.CONVERSE_SAID
        assert t["participants"] == ["ana", "bo"] and t["turns"][-1]["role"] == "bo"

    def test_a_stop_phrase_written_by_extend_ends_the_fold(self):
        first = self.CONVERSE_SAID[1]["text"].split()[0]
        fold = _run(self._fold_graph(6, {"stop_phrases": [first]}))["talk"]
        assert fold["folded"] == {"steps": 1, "stopped": "until", "body_nodes": ["view", "say", "next"]}
        assert fold["items"][0]["stopped"] == f"stop_phrase:{first}"


class _Spool:
    def __init__(self, interrupt_after=None):
        self.fingerprints, self.items, self.done = {}, {}, {}
        self.interrupt_after, self.count = interrupt_after, 0

    def on_node_start(self, nid, fp):
        self.fingerprints[nid] = fp

    def on_spool_item(self, nid, key, item):
        self.items.setdefault(nid, {})[key] = item
        self.count += 1
        if self.interrupt_after is not None and self.count >= self.interrupt_after:
            raise KeyboardInterrupt("simulated interruption")

    def on_node_done(self, nid, path, fp):
        self.done[nid] = (path, fp)

    def executor(self):
        return ProtocolExecutor(on_node_start=self.on_node_start, on_spool_item=self.on_spool_item,
                                on_node_done=self.on_node_done)

    def resume_map(self, nid):
        return {nid: {"fingerprint": self.fingerprints[nid], "items": dict(self.items.get(nid, {}))}}


class TestAFoldResumes:
    def test_an_interrupted_conversation_resumes_at_the_step_it_reached(self, monkeypatch):
        from mechbench_compute.providers import mock as mock_mod

        made = []
        real = mock_mod.MockTransport._text_for

        def counting(self, req):
            made.append(1)
            return real(self, req)

        monkeypatch.setattr(mock_mod.MockTransport, "_text_for", counting)
        graph = TestAConversationIsAFold()._fold_graph(4)
        spec = lambda: ProtocolSpec(kind="pipeline", prompt="", model_id=None, extra={"graph": graph})  # noqa: E731

        full = _Spool()
        reference = full.executor().run(spec())
        assert len(made) == 4
        made.clear()
        partial = _Spool(interrupt_after=2)
        with pytest.raises(KeyboardInterrupt):
            partial.executor().run(spec())
        assert len(made) == 2 and sorted(partial.items["talk"]) == ["step:0", "step:1"]
        made.clear()
        resumed = _Spool()
        out = resumed.executor().run(spec(), resume=partial.resume_map("talk"))
        assert len(made) == 2
        ref_t = reference.payload["outputs"]["talk"]["items"][0]
        out_t = out.payload["outputs"]["talk"]["items"][0]
        assert [m["text"] for m in out_t["messages"]] == [m["text"] for m in ref_t["messages"]]
        assert resumed.done["talk"][1] == full.done["talk"][1]


class TestAPersonaSystemIsAGraph:
    MODEL = {"provider": "mock", "model": "mock-large"}

    def _start(self, *conversations):
        return {"kind": "collection", "item_kind": "text/transcript", "items": [
            {"id": cid, "kind": "text/transcript", "participants": ["ana", "bo"],
             "stopped": "", "next_speaker": first, "coords": {},
             "messages": [{"index": 0, "participant": "narrator", "role_as_seen": "assistant",
                           "text": text}]}
            for cid, first, text in conversations]}

    def _graph(self, start, steps=2, extend=None):
        body = {"dataflow": 2, "nodes": [
            node("view", "text/render", {"participant": {"$param": "speaker"},
                                         "window": {"policy": "sliding", "words": 400}}),
            node("say", "text/chat", {"model": self.MODEL, "max_tokens": 64, "budget_usd": 1.0}),
            node("read", "text/measure",
                 {"mode": "annotate", "keep": True,
                  "measures": [{"kind": "capture", "name": "next_speaker",
                                "pattern": r"\b(ana|bo)\b", "take": "last",
                                "ignore_case": True}]}),
            node("next", "text/extend", {"participant": {"$param": "speaker"},
                                         "keep_fields": ["next_speaker"], **(extend or {})}),
        ], "edges": [
            {"from": {"input": "record"}, "to": {"node": "view", "port": "transcripts"}},
            {"from": {"node": "view"}, "to": {"node": "say", "port": "records"}},
            {"from": {"node": "say"}, "to": {"node": "read", "port": "documents"}},
            {"from": {"input": "record"}, "to": {"node": "next", "port": "transcripts"}},
            {"from": {"node": "read"}, "to": {"node": "next", "port": "replies"}},
        ]}
        turn = {"dataflow": 2, "nodes": [node("each", "records/map",
                               {"body": body, "bind": {"speaker": "next_speaker"},
                                "collect": "first", "output": "next"})],
                "edges": [{"from": {"input": "state"}, "to": {"node": "each", "port": "records"}}]}
        return {"dataflow": 2, "nodes": [
            node("talk", "records/fold",
                 {"body": turn, "steps": steps, "output": "each",
                  "until": {"field": "stopped"}}, {"state": start})], "edges": []}

    def test_two_conversations_each_follow_their_own_speaker(self):
        start = self._start(("a", "ana", "Ana goes first."), ("b", "bo", "Bo goes first."))
        out = _run(self._graph(start))["talk"]
        assert out["folded"]["steps"] == 2
        by_id = {t["id"]: t for t in out["items"]}
        assert by_id["a"]["messages"][1]["participant"] == "ana"
        assert by_id["b"]["messages"][1]["participant"] == "bo"
        assert len(by_id["a"]["messages"]) == 3

    def test_the_speaker_of_the_next_turn_is_what_the_last_one_said(self):
        start = self._start(("a", "ana", "Ana goes first."))
        out = _run(self._graph(start, steps=1))["talk"]
        [t] = out["items"]
        assert "next_speaker" in t or t.get("next_speaker") is None
        assert t["messages"][1]["participant"] == "ana"

    def test_a_turn_can_be_recorded_without_joining_the_room(self):
        start = self._start(("a", "ana", "Ana goes first."))
        out = _run(self._graph(start, steps=1, extend={"channel": "judge"}))["talk"]
        [t] = out["items"]
        assert t["messages"][1]["channel"] == "judge"
        assert t["messages"][1]["text"] not in t["text"]
        from mechbench_compute import transcript as TR
        seen = render(t["messages"], participant="bo", participants=["ana", "bo"])
        assert all("judge" not in m["content"] for m in seen)
        assert len(seen) == 1
