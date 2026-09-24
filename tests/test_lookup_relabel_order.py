from __future__ import annotations

import hashlib

import pytest
from mechbench_schema import dump_canonical

from mechbench_compute import bench
from mechbench_compute.lexicon import kinds as K
from mechbench_compute.ops.records.lookup import lookup
from mechbench_compute.ops.records.relabel import relabel
from mechbench_compute.ops.records.tabulate import tabulate_records
from mechbench_compute.ops.records.union import union
from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec

ARCH = {"n_layers": 6, "global_layers": [1, 5], "first_kv_shared_layer": 4}


def _table(means):
    return {"kind": "records/table", "arch": ARCH, "columns": [],
            "rows": [{"layer": i, "mean": m} for i, m in enumerate(means)]}


class TestLookup:
    def test_a_layer_is_read_through_the_headers_global_layers(self):
        out = lookup(_table([0.1] * 6), _table([]), {
            "field": "layer", "in": "arch.global_layers", "by": "member", "as": "global"})
        assert [r["coords"]["global"] for r in out["items"]] == [False, True, False, False, False, True]
        assert out["lookup"] == {"field": "layer", "in": "arch.global_layers",
                                 "by": "member", "as": "global"}

    def test_an_index_names_its_place_through_a_list_the_header_carries(self):
        cells = K.collection("records/record", [
            {"id": "p/0", "coords": {"component": 0}},
            {"id": "p/2", "coords": {"component": 2}}])
        header = {"kind": "collection", "components": ["embed", "L0", "L1"], "items": []}
        out = lookup(cells, header, {"field": "component", "in": "components"})
        assert [r["coords"]["component"] for r in out["items"]] == ["embed", "L1"]

    def test_a_map_is_read_at_the_values_key(self):
        recs = [{"id": "a", "coords": {"layer": 3}}]
        out = lookup(recs, {"names": {"3": "the third"}}, {"field": "layer", "in": "names"})
        assert out["items"][0]["coords"]["layer"] == "the third"

    def test_a_value_the_list_has_no_place_for_is_refused_unless_null_is_asked_for(self):
        recs = [{"id": "a", "coords": {"component": 7}}]
        header = {"components": ["embed", "L0"]}
        with pytest.raises(ValueError, match="'a' has component=7"):
            lookup(recs, header, {"field": "component", "in": "components"})
        out = lookup(recs, header, {"field": "component", "in": "components", "on_missing": "null"})
        assert out["items"][0]["coords"]["component"] is None

    def test_a_header_without_the_path_says_what_it_has(self):
        with pytest.raises(ValueError, match=r"no list or map at 'arch.global_layers'; its fields are \['columns', 'kind'\]"):
            lookup([], {"kind": "records/table", "columns": [], "rows": []},
                    {"field": "layer", "in": "arch.global_layers", "by": "member"})

    def test_a_top_level_field_it_replaces_becomes_the_coordinate(self):
        out = lookup([{"id": "a", "layer": 1}], {"arch": ARCH},
                      {"field": "layer", "in": "arch.global_layers", "by": "member"})
        assert out["items"][0] == {"id": "a", "coords": {"layer": True}}


class TestRelabel:
    def test_a_boolean_is_labelled_by_its_json_spelling(self):
        recs = [{"id": "a", "coords": {"global": True}}, {"id": "b", "coords": {"global": False}}]
        out = relabel(recs, {"field": "global",
                             "labels": {"true": "global attention", "false": "local attention"}})
        assert [r["coords"]["global"] for r in out] == ["global attention", "local attention"]

    def test_a_number_matches_its_spelling_and_an_unnamed_value_is_kept(self):
        out = relabel([{"id": "a", "n": 12}, {"id": "b", "n": 13}],
                      {"field": "n", "labels": {"12": "twelve"}})
        assert [r["n"] for r in out] == ["twelve", 13]

    def test_others_error_is_the_check_that_every_value_has_a_label(self):
        with pytest.raises(ValueError, match="'b' has port='mlp'"):
            relabel([{"id": "a", "coords": {"port": "attn"}}, {"id": "b", "coords": {"port": "mlp"}}],
                    {"field": "port", "labels": {"attn": "its attention only"}, "others": "error"})

    def test_as_writes_a_coordinate_and_keeps_the_original(self):
        out = relabel([{"id": "a", "coords": {"port": "attn"}}],
                      {"field": "port", "labels": {"attn": "its attention only"}, "as": "removed"})
        assert out[0]["coords"] == {"port": "attn", "removed": "its attention only"}

    def test_a_dot_path_is_written_where_it_was_read_without_touching_the_input(self):
        rec = {"id": "a", "metadata": {"model": "m1"}}
        out = relabel([rec], {"field": "metadata.model", "labels": {"m1": "Model one"}})
        assert out[0]["metadata"] == {"model": "Model one"}
        assert rec["metadata"] == {"model": "m1"}


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
            {"id": "named", "block": "records/relabel", "params": {
                "field": "removed", "labels": {"attn": "its attention only", "whole": "the whole layer"},
                "others": "error"}},
            {"id": "kind", "block": "records/lookup", "params": {
                "field": "layer", "in": "arch.global_layers", "by": "member", "as": "attention"}},
            {"id": "worded", "block": "records/relabel", "params": {
                "field": "attention", "labels": {"true": "global attention", "false": "local attention"}}},
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
