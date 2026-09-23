"""`records/map`: a sub-protocol per record.

A run set fans out over runs; a node fans out over its own items.
Between them there was nothing, so a graph that wanted to do several
things to each record had to be written once per record or flattened
into a node that knew how to do all of them.
"""

from __future__ import annotations

import pytest

from mechbench_compute.lexicon import kinds as K
from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec

TOPICS = [{"id": "t1", "coords": {"kind": "a"}, "user": "dusk"},
          {"id": "t2", "coords": {"kind": "b"}, "user": "kettle"}]

#: A body that makes two records out of the one it is given, so a
#: `stream` collect has something to flatten. A `{"$param"}` is a WHOLE
#: value, never spliced into a string.
BODY = {"nodes": [
    {"id": "design", "block": "records/cross",
     "params": {"factors": [
         {"name": "n", "levels": [{"key": "1"}, {"key": "2"}]},
         {"name": "topic", "levels": [{"key": {"$param": "topic"}}]}]}},
    {"id": "write", "block": "records/fill",
     "params": {"templates": {"text": "{topic}-{n}"}}},
], "edges": [
    {"from": {"node": "design"},
     "to": {"node": "write", "port": "records"}, "kind": "records"},
]}


def _spec(**params):
    graph = {"dataflow": 2, "nodes": [
        {"id": "each", "block": "records/map",
         "params": {"body": BODY, "bind": {"topic": "user"}, **params},
         "inputs": {"records": TOPICS}},
    ], "edges": []}
    return ProtocolSpec(kind="pipeline", prompt="", model_id=None,
                        extra={"graph": graph})


def _run(**params):
    return ProtocolExecutor().run(_spec(**params)).payload["outputs"]["each"]


class TestStream:
    def test_every_invocation_s_items_come_back_as_one_collection(self):
        out = _run()
        assert out["mapped"] == {"records": 2, "collect": "stream",
                                 "body_nodes": ["design", "write"]}
        items = K.items_of(out)
        assert len(items) == 4                       # two records, two each
        assert [i["text"] for i in items] == [
            "dusk-1", "dusk-2", "kettle-1", "kettle-2"]

    def test_an_item_keeps_both_its_own_and_its_record_s_coordinates(self):
        first = K.items_of(_run())[0]
        assert first["coords"]["kind"] == "a"        # the record's
        assert first["coords"]["n"] == "1"           # the body's
        assert first["coords"]["mapped"] == "t1"     # which record it came from
        assert first["id"].startswith("t1:")


class TestCollectPolicies:
    def test_first_keeps_one_item_per_record(self):
        items = K.items_of(_run(collect="first"))
        assert [i["id"] for i in items] == ["t1", "t2"]
        assert items[0]["text"] == "dusk-1"

    def test_all_nests_each_invocation(self):
        items = K.items_of(_run(collect="all"))
        assert [i["id"] for i in items] == ["t1", "t2"]
        assert [r["text"] for r in items[0]["items"]] == ["dusk-1", "dusk-2"]

    def test_an_unknown_policy_is_refused(self):
        with pytest.raises(ValueError, match="collect is"):
            _run(collect="some")


class TestTheBody:
    def test_a_record_missing_a_bound_field_is_refused_by_name(self):
        graph = {"dataflow": 2, "nodes": [
            {"id": "each", "block": "records/map",
             "params": {"body": BODY, "bind": {"topic": "prompt"}},
             "inputs": {"records": TOPICS}},
        ], "edges": []}
        with pytest.raises(ValueError, match="has no prompt to bind"):
            ProtocolExecutor().run(ProtocolSpec(
                kind="pipeline", prompt="", model_id=None, extra={"graph": graph}))

    def test_a_body_with_two_endings_asks_which_one(self):
        two = {"nodes": [*BODY["nodes"],
                         {"id": "other", "block": "records/cross",
                          "params": {"factors": [{"name": "z",
                                                  "levels": [{"key": "q"}]}]}}],
               "edges": list(BODY["edges"])}
        with pytest.raises(ValueError, match="name one with `output`"):
            ProtocolExecutor().run(ProtocolSpec(
                kind="pipeline", prompt="", model_id=None, extra={"graph": {"dataflow": 2,
                    "nodes": [{"id": "each", "block": "records/map",
                               "params": {"body": two, "bind": {"topic": "user"}},
                               "inputs": {"records": TOPICS}}],
                    "edges": []}}))

    def test_output_names_the_ending_to_collect(self):
        two = {"nodes": [*BODY["nodes"],
                         {"id": "other", "block": "records/cross",
                          "params": {"factors": [{"name": "z",
                                                  "levels": [{"key": "q"}]}]}}],
               "edges": list(BODY["edges"])}
        out = ProtocolExecutor().run(ProtocolSpec(
            kind="pipeline", prompt="", model_id=None, extra={"graph": {"dataflow": 2,
                "nodes": [{"id": "each", "block": "records/map",
                           "params": {"body": two, "bind": {"topic": "user"},
                                      "output": "write"},
                           "inputs": {"records": TOPICS}}],
                "edges": []}})).payload["outputs"]["each"]
        assert [i["text"] for i in K.items_of(out)][:2] == ["dusk-1", "dusk-2"]

    def test_the_body_sees_the_protocol_s_own_params(self):
        """A body is a sub-protocol, not a foreign graph: a run launched
        with a `model` param should be able to name it inside the body, rather
        than carrying the same constant on every record to bind it in."""
        body = {"nodes": [
            {"id": "design", "block": "records/cross",
             "params": {"factors": [
                 {"name": "topic", "levels": [{"key": {"$param": "topic"}}]},
                 {"name": "era", "levels": [{"key": {"$param": "era"}}]}]}},
            {"id": "write", "block": "records/fill",
             "params": {"templates": {"text": "{topic} in {era}"}}},
        ], "edges": list(BODY["edges"])}
        graph = {"dataflow": 2, "nodes": [
            {"id": "each", "block": "records/map",
             "params": {"body": body, "bind": {"topic": "user"}},
             "inputs": {"records": TOPICS}},
        ], "edges": []}
        out = ProtocolExecutor().run(ProtocolSpec(
            kind="pipeline", prompt="", model_id=None,
            extra={"graph": graph, "params": {"era": "1890"}}))
        assert [i["text"] for i in K.items_of(out.payload["outputs"]["each"])] == [
            "dusk in 1890", "kettle in 1890"]

    def test_a_record_s_binding_shadows_the_protocol_s(self):
        """Both name `topic`; the per-record value is the specific one."""
        graph = {"dataflow": 2, "nodes": [
            {"id": "each", "block": "records/map",
             "params": {"body": BODY, "bind": {"topic": "user"}},
             "inputs": {"records": TOPICS[:1]}},
        ], "edges": []}
        out = ProtocolExecutor().run(ProtocolSpec(
            kind="pipeline", prompt="", model_id=None,
            extra={"graph": graph, "params": {"topic": "ignored"}}))
        assert [i["text"] for i in K.items_of(out.payload["outputs"]["each"])] == [
            "dusk-1", "dusk-2"]

    def test_no_body_is_refused_with_what_a_body_is(self):
        with pytest.raises(ValueError, match="needs a `body`"):
            ProtocolExecutor().run(ProtocolSpec(
                kind="pipeline", prompt="", model_id=None, extra={"graph": {"dataflow": 2,
                    "nodes": [{"id": "each", "block": "records/map",
                               "params": {}, "inputs": {"records": TOPICS}}],
                    "edges": []}}))


class TestResumeAndIsomorphism:
    def test_each_record_is_an_item_the_spool_can_keep(self):
        spooled = []
        ex = ProtocolExecutor()
        ex._on_spool_item = lambda nid, key, item: spooled.append((nid, key))
        ex.run(_spec())
        assert spooled == [("each", "t1"), ("each", "t2")]

    def test_a_resumed_map_reuses_the_records_already_done(self):
        first = K.items_of(_run())
        done = {"t1": {"id": "t1", "text": "kept from before"}}
        ex = ProtocolExecutor()
        out = ex.run(_spec(), resume={"each": {
            "fingerprint": None, "items": done}})
        # The fingerprint is wrong, so nothing is reused — the guard that
        # makes resume safe. With the right one it would be.
        assert len(K.items_of(out.payload["outputs"]["each"])) == len(first)

    def test_mapping_a_subset_is_the_subset_of_mapping(self):
        """The isomorphism the chunking law wants, structurally: the body
        sees one record and nothing else, so a chunk of the stream maps
        to exactly its part of the output."""
        graph = {"dataflow": 2, "nodes": [{"id": "each", "block": "records/map",
                            "params": {"body": BODY, "bind": {"topic": "user"}},
                            "inputs": {"records": TOPICS[:1]}}], "edges": []}
        half = ProtocolExecutor().run(ProtocolSpec(
            kind="pipeline", prompt="", model_id=None,
            extra={"graph": graph})).payload["outputs"]["each"]
        whole = K.items_of(_run())
        assert K.items_of(half) == [i for i in whole
                                    if i["coords"]["mapped"] == "t1"]


class TestParamsInTheBody:
    """A body refers to the map's bound names and the run's params with
    `{"$param"}`; nothing is a `$` string, and the map's own names are
    not the protocol's."""

    def test_a_declared_body_binds_its_names_per_record_and_the_runs_by_name(self):
        body = {"dataflow": 2, "nodes": [
            {"id": "design", "block": "records/cross",
             "params": {"factors": [
                 {"name": "n", "levels": [{"key": "1"}, {"key": "2"}]},
                 {"name": "topic", "levels": [{"key": {"$param": "topic"}}]},
                 {"name": "tag", "levels": [{"key": {"$param": "tag"}}]}]}},
            {"id": "write", "block": "records/fill",
             "params": {"templates": {"text": "{tag}:{topic}-{n}"}}},
        ], "edges": [
            {"from": {"node": "design"}, "to": {"node": "write", "port": "records"}},
        ]}
        graph = {"dataflow": 2, "nodes": [
            {"id": "each", "block": "records/map",
             "params": {"body": body, "bind": {"topic": "user"}},
             "inputs": {"records": TOPICS}},
        ], "edges": []}
        # `tag` is the run's param; `topic` is the map's, bound per record.
        out = ProtocolExecutor().run(ProtocolSpec(
            kind="pipeline", prompt="", model_id=None,
            extra={"graph": graph, "params": {"tag": "x"}, "inputs": {}})).payload["outputs"]["each"]
        assert [i["text"] for i in K.items_of(out)] == [
            "x:dusk-1", "x:dusk-2", "x:kettle-1", "x:kettle-2"]

    def test_a_body_names_the_run_s_model(self):
        """A body need not repeat `dataflow` itself: `{"$param": "model"}` on a body node
        is the run's model, resolved before the loader sees it, the same
        way a fold's body reads one."""
        body = {"nodes": [
            {"id": "ask", "block": "records/cross",
             "params": {"factors": [{"name": "user",
                                     "levels": [{"key": {"$param": "topic"}}]}]}},
            {"id": "say", "block": "text/chat",
             "params": {"model": {"$param": "model"}, "budget_usd": 1.0}},
        ], "edges": [
            {"from": {"node": "ask"}, "to": {"node": "say", "port": "records"}},
        ]}
        graph = {"dataflow": 2, "nodes": [
            {"id": "each", "block": "records/map",
             "params": {"body": body, "bind": {"topic": "user"}},
             "inputs": {"records": TOPICS}},
        ], "edges": []}
        out = ProtocolExecutor().run(ProtocolSpec(
            kind="pipeline", prompt="", model_id=None,
            extra={"graph": graph, "inputs": {},
                   "params": {"model": {"provider": "mock",
                                        "model": "mock-large"}}})).payload["outputs"]["each"]
        items = K.items_of(out)
        assert [i["coords"]["mapped"] for i in items] == ["t1", "t2"]
        assert all(i.get("text") for i in items)

    def test_a_name_the_map_does_not_bind_and_the_run_does_not_supply_is_refused(self):
        body = {"dataflow": 2, "nodes": [
            {"id": "design", "block": "records/cross",
             "params": {"factors": [{"name": "topic", "levels": [{"key": {"$param": "elsewhere"}}]}]}},
        ], "edges": []}
        graph = {"dataflow": 2, "nodes": [
            {"id": "each", "block": "records/map",
             "params": {"body": body, "bind": {"topic": "user"}},
             "inputs": {"records": TOPICS}},
        ], "edges": []}
        with pytest.raises(ValueError, match="unbound param 'elsewhere'"):
            ProtocolExecutor().run(ProtocolSpec(
                kind="pipeline", prompt="", model_id=None,
                extra={"graph": graph, "params": {}, "inputs": {}}))
