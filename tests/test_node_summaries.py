"""A run's manifest says what each node produced (task 000525).

The composer shows, beside a selected node, what the protocol's last
run made there: the kind and how many. It reads that from the manifest
rather than fetching every node's object, so the manifest carries a
summary per executed node — and none for a node that did not run, which
`nodes_missing` accounts for.
"""
from __future__ import annotations

from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec, node_summary

GRAPH = {
    "nodes": [
        {"id": "design", "block": "records/cross",
         "params": {"factors": [{"name": "x", "levels": [{"key": "a"}, {"key": "b"}, {"key": "c"}]},
                                {"name": "y", "levels": [{"key": "1"}, {"key": "2"}]}]}},
        {"id": "prompts", "block": "records/fill", "params": {"templates": {"user": "{x} {y}"}}},
        {"id": "stats", "block": "records/tabulate", "params": {}},
    ],
    "edges": [
        {"from": {"node": "design", "port": "out"}, "to": {"node": "prompts", "port": "records"},
         "kind": "records/record"},
        {"from": {"node": "prompts", "port": "out"}, "to": {"node": "stats", "port": "records"},
         "kind": "records/record"},
    ],
}


def test_every_executed_node_is_summarized():
    out = ProtocolExecutor().run(ProtocolSpec(
        kind="pipeline", prompt="", model_id=None, extra={"graph": GRAPH}))
    summaries = out.payload["node_summaries"]
    assert summaries["design"] == {"kind": "records/record", "collection": True, "items": 6}
    assert summaries["prompts"] == {"kind": "records/record", "collection": True, "items": 6}
    assert summaries["stats"]["kind"] == "records/table"
    assert summaries["stats"]["collection"] is False
    assert summaries["stats"]["rows"] == 6
    assert set(summaries) == set(out.payload["nodes_executed"])


def test_a_summary_reads_every_spelling():
    assert node_summary([{"id": "a"}, {"id": "b"}]) == {"kind": "collection", "collection": True, "items": 2}
    assert node_summary({"kind": "residual_vectors", "rows": [{}, {}, {}]}) == {
        "kind": "activations/vector", "collection": True, "items": 3}
    assert node_summary({"kind": "direction/vector", "vector": [0.1]}) == {
        "kind": "direction/vector", "collection": False}
    assert node_summary("text") == {}
    assert node_summary({"kind": "collection", "item_kind": "text/document", "items": [{}]},
                        {"cost_usd": 0.0123, "calls": 2}) == {
        "kind": "text/document", "collection": True, "items": 1, "spend_usd": 0.0123}
