"""`records/diff`: two collections matched by key and compared."""

from __future__ import annotations

import pytest

from mechbench_compute import ops
from mechbench_compute.lexicon import kinds as K
from mechbench_compute.ops.records.diff import diff_collections


def story(prompt, sample, text, ended=None, **meta):
    sampling = {"seed": 7, "index": sample}
    if ended is not None:
        sampling["ended"] = ended
    return {"id": f"{prompt}-s{sample}", "text": text,
            "coords": {"prompt": prompt, "sample": sample},
            "metadata": {"sampling": sampling, **meta}}


def corpus(*items, **header):
    return K.collection("records/record", list(items), **header)


def test_identical_collections_are_identical_and_equivalent():
    a = corpus(story("flash", 0, "one"), story("flash", 1, "two"))
    out = diff_collections(a, corpus(*a["items"]), {})
    d = out["diff"]
    assert d["identical"] and d["equivalent"] and d["holds"]
    assert d["records"] == {"identical": 2, "extends": 0, "differs": 0,
                            "only_a": 0, "only_b": 0}
    assert out["items"] == []


def test_order_is_not_a_difference_but_is_not_identity():
    a = corpus(story("flash", 0, "one"), story("flash", 1, "two"))
    b = corpus(*reversed(a["items"]))
    d = diff_collections(a, b, {})["diff"]
    assert d["equivalent"] and not d["identical"]


def test_a_new_metadata_field_is_named_as_added():
    a = corpus(story("flash", 0, "one"), story("neutral", 0, "two"))
    b = corpus(story("flash", 0, "one", ended="end"), story("neutral", 0, "two", ended="end"))
    out = diff_collections(a, b, {"key": ["prompt", "sample"]})
    d = out["diff"]
    assert not d["equivalent"]
    assert d["changed_fields"] == {
        "metadata.sampling.ended": {"records": 2, "relations": {"added": 2}}}
    row = out["items"][0]
    assert row["coords"] == {"prompt": "flash", "sample": 0}
    assert row["fields"]["metadata.sampling.ended"] == {"relation": "added", "b": "end"}
    assert diff_collections(a, b, {"fields": ["text"]})["diff"]["equivalent"]
    assert diff_collections(a, b, {"exclude": ["ended"]})["diff"]["equivalent"]


def test_one_sided_records_and_duplicate_keys():
    a = corpus(story("flash", 0, "x"), story("flash", 1, "y"))
    b = corpus(story("flash", 1, "y"), story("flash", 2, "z"))
    d = diff_collections(a, b, {})["diff"]
    assert d["records"]["only_a"] == 1 and d["records"]["only_b"] == 1
    assert not d["holds"]
    twice = corpus(story("flash", 0, "x"), {**story("flash", 0, "x"), "id": "other"})
    with pytest.raises(ValueError, match="key on more coordinates"):
        diff_collections(twice, twice, {"key": ["prompt", "sample"]})


def test_numbers_carry_deltas_and_the_largest_magnitude():
    a = corpus({"id": "r1", "n": 10, "rate": 0.5}, {"id": "r2", "n": 4, "rate": 0.25})
    b = corpus({"id": "r1", "n": 7, "rate": 0.5}, {"id": "r2", "n": 9, "rate": 0.25})
    out = diff_collections(a, b, {})
    f = out["diff"]["changed_fields"]["n"]
    assert f["max_abs_delta"] == 5 and f["mean_delta"] == 1
    assert {r["id"]: r["fields"]["n"]["delta"] for r in out["items"]} == {"r1": -3, "r2": 5}


def test_a_bool_is_not_a_number():
    a, b = corpus({"id": "r", "v": 1}), corpus({"id": "r", "v": True})
    row = diff_collections(a, b, {})["items"][0]
    assert row["fields"]["v"]["relation"] == "changed" and "delta" not in row["fields"]["v"]


def test_moving_fields_are_excluded_listed_and_counted():
    a = corpus(story("flash", 0, "x", call={"latency_ms": 10, "response_id": "m1",
                                           "usage": {"output_tokens": 5}}))
    b = corpus(story("flash", 0, "x", call={"latency_ms": 99, "response_id": "m2",
                                           "usage": {"output_tokens": 5}}))
    d = diff_collections(a, b, {})["diff"]
    assert d["equivalent"]
    assert "latency_ms" in d["excluded"]
    assert d["excluded_differ"] == {"metadata.call.latency_ms": 1,
                                    "metadata.call.response_id": 1}
    d = diff_collections(a, b, {"exclude_moving": False})["diff"]
    assert set(d["changed_fields"]) == {"metadata.call.latency_ms",
                                        "metadata.call.response_id"}


def test_the_regeneration_case():
    """Same seed, a larger allowance: what ended on its own is identical,
    what was cut runs on from where it stopped."""
    a = corpus(story("flash", 0, "done.", ended="end"),
               story("flash", 1, "cut mid", ended="max_tokens"))
    b = corpus(story("flash", 0, "done.", ended="end"),
               story("flash", 1, "cut mid-sentence, then finished.", ended="end"))
    rule = [{"field": "text", "relation": "extends",
             "when": {"metadata.sampling.ended": "max_tokens"}}]
    out = diff_collections(a, b, {"fields": ["text"], "allow": rule})
    d = out["diff"]
    assert d["records"]["identical"] == 1 and d["records"]["extends"] == 1
    assert d["holds"] and d["disallowed"] == 0
    assert out["items"][0]["allowed"] is True

    # A story that ended on its own and came back longer breaks it.
    b2 = corpus(story("flash", 0, "done. And more.", ended="end"), b["items"][1])
    d2 = diff_collections(a, b2, {"fields": ["text"], "allow": rule})["diff"]
    assert not d2["holds"] and d2["disallowed"] == 1
    # So does a cut story whose new text is not a continuation.
    b3 = corpus(b["items"][0], story("flash", 1, "something else", ended="end"))
    assert not diff_collections(a, b3, {"fields": ["text"], "allow": rule})["diff"]["holds"]


def test_header_differences_and_provenance():
    a = corpus(story("flash", 0, "x"), name="gen", spend={"usd": 1.0})
    b = corpus(story("flash", 0, "x"), name="gen", spend={"usd": 2.0})
    d = diff_collections(a, b, {}, provenance=(
        {"created_at": "t1", "produced_by": {"version": "0.1+src.a"}, "operation": "x"},
        {"created_at": "t2", "produced_by": {"version": "0.2+src.b"}, "operation": "y"},
    ))["diff"]
    assert d["header"] == {"spend.usd": {"relation": "changed", "a": 1.0, "b": 2.0,
                                         "delta": 1.0}}
    assert d["provenance"] == {"provenance.operation": {"relation": "changed",
                                                        "a": "x", "b": "y"}}
    assert "provenance.produced_by.version" in d["excluded_differ"]


def test_groups_side_by_side():
    a = corpus(*[{"id": f"{p}{i}", "coords": {"prompt": p}, "hit": int(i < 2)}
                 for p in ("flash", "neutral") for i in range(4)])
    b = corpus(*[{"id": f"{p}{i}", "coords": {"prompt": p}, "hit": int(i < 3)}
                 for p in ("flash", "neutral") for i in range(4)])
    groups = diff_collections(a, b, {"by": ["prompt"]})["diff"]["groups"]
    assert [g["coords"] for g in groups] == [{"prompt": "flash"}, {"prompt": "neutral"}]
    assert set(groups[0]["fields"]) == {"hit"}
    assert groups[0]["fields"]["hit"] == {"mean_a": 0.5, "sum_a": 2, "mean_b": 0.75,
                                          "sum_b": 3, "diff": 0.25}


def test_a_bad_rule_is_refused_by_name():
    a = corpus(story("flash", 0, "x"))
    with pytest.raises(ValueError, match="relation"):
        diff_collections(a, a, {"allow": [{"field": "text", "relation": "grows"}]})


def test_the_op_runs_through_the_executor_entry_point():
    a = corpus(story("flash", 0, "x"))
    b = corpus(story("flash", 0, "xy"))
    out = ops.run_standalone("records/diff", {"a": a, "b": b}, {})
    assert out["diff"]["records"]["extends"] == 1


def test_a_retired_kind_spelling_is_the_same_kind():
    a = corpus({**story("flash", 0, "x"), "kind": "~canonical/kinds/text"})
    b = corpus({**story("flash", 0, "x"), "kind": "text/document"})
    assert diff_collections(a, b, {})["diff"]["equivalent"]
    c = corpus({**story("flash", 0, "x"), "kind": "records/record"})
    assert not diff_collections(a, c, {})["diff"]["equivalent"]


def test_a_diff_node_cites_both_runs(monkeypatch):
    from mechbench_compute import bench
    from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec

    store = {"lab/p/results/j_old/gen": corpus(story("flash", 0, "cut")),
             "lab/p/results/j_new/gen": corpus(story("flash", 0, "cut, then ended."))}
    emitted = {}
    monkeypatch.setattr(bench, "fetch", lambda ref, with_meta=False: (
        ({"payload": store[str(ref)]}, {"content_hash": str(ref)}) if with_meta
        else {"payload": store[str(ref)]}))
    monkeypatch.setattr(bench, "emit", lambda target, payload, *, inputs=(), **_k: (
        emitted.__setitem__(target, list(inputs)) or {"path": target}))
    graph = {"dataflow": 2, "edges": [], "nodes": [{
        "id": "rerun", "block": "records/diff", "params": {"fields": ["text"]},
        "inputs": {"a": {"$ref": {"bench": "lab/p/results/j_old/gen"}},
                   "b": {"$ref": {"bench": "lab/p/results/j_new/gen"}}}}]}
    out = ProtocolExecutor().run(ProtocolSpec(
        kind="pipeline", prompt="", model_id=None,
        extra={"graph": graph, "params": {}, "inputs": {},
               "resultPath": "lab/p/results/j_cmp"}))
    payload = out.payload if hasattr(out, "payload") else out
    assert payload["outputs"]["rerun"]["diff"]["records"]["extends"] == 1
    assert emitted["lab/p/results/j_cmp/rerun"] == ["lab/p/results/j_old/gen",
                                                    "lab/p/results/j_new/gen"]
