"""`records/map`: a sub-protocol per record (task 000400).

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
#: `stream` collect has something to flatten. The hole is a WHOLE value
#: — `"$topic"`, never `"$topic-{n}"` — which is the substitution rule
#: everywhere a protocol writes one.
BODY = {"nodes": [
    {"id": "design", "block": "records/cross",
     "params": {"factors": [
         {"name": "n", "levels": [{"key": "1"}, {"key": "2"}]},
         {"name": "topic", "levels": [{"key": "$topic"}]}]}},
    {"id": "write", "block": "records/fill",
     "params": {"templates": {"text": "{topic}-{n}"}}},
], "edges": [
    {"from": {"node": "design", "port": "records"},
     "to": {"node": "write", "port": "records"}, "kind": "records"},
]}


def _spec(**params):
    graph = {"nodes": [
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
        graph = {"nodes": [
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
                kind="pipeline", prompt="", model_id=None, extra={"graph": {
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
            kind="pipeline", prompt="", model_id=None, extra={"graph": {
                "nodes": [{"id": "each", "block": "records/map",
                           "params": {"body": two, "bind": {"topic": "user"},
                                      "output": "write"},
                           "inputs": {"records": TOPICS}}],
                "edges": []}})).payload["outputs"]["each"]
        assert [i["text"] for i in K.items_of(out)][:2] == ["dusk-1", "dusk-2"]

    def test_the_body_sees_the_protocol_s_own_bindings(self):
        """A body is a sub-protocol, not a foreign graph: a run launched
        with `$model` should be able to name it inside the body, rather
        than carrying the same constant on every record to bind it in."""
        body = {"nodes": [
            {"id": "design", "block": "records/cross",
             "params": {"factors": [
                 {"name": "topic", "levels": [{"key": "$topic"}]},
                 {"name": "era", "levels": [{"key": "$era"}]}]}},
            {"id": "write", "block": "records/fill",
             "params": {"templates": {"text": "{topic} in {era}"}}},
        ], "edges": list(BODY["edges"])}
        graph = {"nodes": [
            {"id": "each", "block": "records/map",
             "params": {"body": body, "bind": {"topic": "user"}},
             "inputs": {"records": TOPICS}},
        ], "edges": []}
        out = ProtocolExecutor().run(ProtocolSpec(
            kind="pipeline", prompt="", model_id=None,
            extra={"graph": graph, "bindings": {"era": "1890"}}))
        assert [i["text"] for i in K.items_of(out.payload["outputs"]["each"])] == [
            "dusk in 1890", "kettle in 1890"]

    def test_a_record_s_binding_shadows_the_protocol_s(self):
        """Both name `topic`; the per-record value is the specific one."""
        graph = {"nodes": [
            {"id": "each", "block": "records/map",
             "params": {"body": BODY, "bind": {"topic": "user"}},
             "inputs": {"records": TOPICS[:1]}},
        ], "edges": []}
        out = ProtocolExecutor().run(ProtocolSpec(
            kind="pipeline", prompt="", model_id=None,
            extra={"graph": graph, "bindings": {"topic": "ignored"}}))
        assert [i["text"] for i in K.items_of(out.payload["outputs"]["each"])] == [
            "dusk-1", "dusk-2"]

    def test_no_body_is_refused_with_what_a_body_is(self):
        with pytest.raises(ValueError, match="needs a `body`"):
            ProtocolExecutor().run(ProtocolSpec(
                kind="pipeline", prompt="", model_id=None, extra={"graph": {
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
        graph = {"nodes": [{"id": "each", "block": "records/map",
                            "params": {"body": BODY, "bind": {"topic": "user"}},
                            "inputs": {"records": TOPICS[:1]}}], "edges": []}
        half = ProtocolExecutor().run(ProtocolSpec(
            kind="pipeline", prompt="", model_id=None,
            extra={"graph": graph})).payload["outputs"]["each"]
        whole = K.items_of(_run())
        assert K.items_of(half) == [i for i in whole
                                    if i["coords"]["mapped"] == "t1"]
