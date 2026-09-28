from __future__ import annotations

import hashlib

import pytest
from mechbench_schema import dump_canonical

from mechbench_compute import bench
from mechbench_compute.ops.records.tabulate import tabulate_records
from mechbench_compute.ops.records.union import union
from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec

ARCH = {"n_layers": 6, "global_layers": [1, 5], "first_kv_shared_layer": 4}


def _table(means):
    return {"kind": "records/table", "arch": ARCH, "columns": [],
            "rows": [{"layer": i, "mean": m} for i, m in enumerate(means)]}


class TestTabulateOrder:
    RECS = [{"id": "1", "coords": {"removed": "attn"}, "layer": 1},
            {"id": "2", "coords": {"removed": "whole"}, "layer": 1},
            {"id": "3", "coords": {"removed": "attn"}, "layer": 0},
            {"id": "4", "coords": {"removed": "gate"}, "layer": 0},
            {"id": "5", "coords": {"removed": "whole"}}]

    def _ids(self, params):
        return [r["id"] for r in tabulate_records(self.RECS, params)["rows"]]

    def test_rows_keep_the_records_order_when_no_field_is_named(self):
        assert self._ids({}) == ["1", "2", "3", "4", "5"]

    def test_listed_values_come_in_their_listed_order_and_the_rest_after(self):
        assert self._ids({"by": ["removed", "layer"],
                          "order": {"removed": ["whole", "attn"]}}) == ["2", "5", "3", "1", "4"]

    def test_a_field_without_a_list_orders_naturally_and_missing_comes_last(self):
        assert self._ids({"by": "layer"}) == ["3", "4", "1", "2", "5"]
        assert self._ids({"by": "layer", "descending": True}) == ["1", "2", "3", "4", "5"]

    def test_a_field_of_mixed_types_needs_a_list(self):
        with pytest.raises(ValueError, match="more than one type"):
            tabulate_records([{"id": "a", "x": 1}, {"id": "b", "x": "two"}], {"by": "x"})

    def test_an_order_for_a_field_by_does_not_name_is_refused(self):
        with pytest.raises(ValueError, match=r"order names \['removed'\]"):
            tabulate_records(self.RECS, {"by": "layer", "order": {"removed": ["attn"]}})


def test_a_union_of_tables_is_a_union_of_their_rows():
    out = union({"attn": _table([0.1, 0.2]), "whole": _table([0.3])}, {"batch_axis": "removed"})
    assert [(r["layer"], r["coords"]["removed"]) for r in out["items"]] == [
        (0, "attn"), (1, "attn"), (0, "whole")]


STORE = {"lab/p/attn": _table([-0.5, -3.0, 0.1, 0.0, -0.2, -1.0]),
         "lab/p/whole": _table([-16.0, -1.0, 0.0, 0.0, 0.0, -2.0])}


@pytest.fixture
def fake_bench(monkeypatch):
    emitted: dict[str, dict] = {}

    def fetch(ref, with_meta=False):
        obj = {"payload": STORE[str(ref)]}
        meta = {"content_hash": "sha256:" + hashlib.sha256(dump_canonical(STORE[str(ref)])).hexdigest()}
        return (obj, meta) if with_meta else obj

    def emit(target, payload, **kw):
        emitted[target] = payload
        return {"path": target}

    monkeypatch.setattr(bench, "fetch", fetch)
    monkeypatch.setattr(bench, "emit", emit)
    return emitted


def test_a_figure_is_coloured_by_attention_kind_from_the_results_own_arch(fake_bench):
    graph = {
        "dataflow": 2,
        "nodes": [
            {"id": "sweeps", "block": "records/union", "params": {"batch_axis": "removed"}},
            {"id": "named", "block": "records/derive", "params": {"fields": {
                "coords.removed": '{"attn": "its attention only", "whole": "the whole layer"}[coords.removed]'}}},
            {"id": "kind", "block": "records/derive", "params": {"fields": {
                "coords.attention": "layer in header.arch.global_layers"}}},
            {"id": "worded", "block": "records/derive", "params": {"fields": {
                "coords.attention": '"global attention" if coords.attention else "local attention"'}}},
            {"id": "rows", "block": "records/tabulate", "params": {
                "by": ["removed", "layer"], "order": {"removed": ["the whole layer", "its attention only"]}}},
            {"id": "chart", "block": "records/plot", "params": {
                "mark": "bar", "encoding": {"x": "layer", "y": "mean", "color": "attention"},
                "facet": "removed"}},
        ],
        "edges": [
            {"from": {"input": "attn"}, "to": {"node": "sweeps", "port": "attn"}},
            {"from": {"input": "whole"}, "to": {"node": "sweeps", "port": "whole"}},
            {"from": {"node": "sweeps"}, "to": {"node": "named", "port": "records"}},
            {"from": {"node": "named"}, "to": {"node": "kind", "port": "records"}},
            {"from": {"node": "kind"}, "to": {"node": "worded", "port": "records"}},
            {"from": {"node": "worded"}, "to": {"node": "rows", "port": "records"}},
            {"from": {"node": "rows"}, "to": {"node": "chart", "port": "records"}},
        ],
    }
    ProtocolExecutor().run(ProtocolSpec(kind="pipeline", prompt="", model_id=None, extra={
        "graph": graph, "params": {},
        "inputs": {"attn": {"$ref": {"bench": "lab/p/attn"}}, "whole": {"$ref": {"bench": "lab/p/whole"}}},
        "resultPath": "lab/p/results/j_new"}))
    rows = fake_bench["lab/p/results/j_new/rows"]["rows"]
    assert [(r["removed"], r["layer"], r["attention"]) for r in rows[:3]] == [
        ("the whole layer", 0, "local attention"),
        ("the whole layer", 1, "global attention"),
        ("the whole layer", 2, "local attention")]
    assert [r["removed"] for r in rows[6:7]] == ["its attention only"]
    chart = fake_bench["lab/p/results/j_new/chart"]
    assert chart["source"] == "lab/p/results/j_new/rows"
    assert chart["axes"] == {"layer": {"n": 6, "global": [1, 5], "kv_shared_from": 4}}
