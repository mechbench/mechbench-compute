from __future__ import annotations

import pytest

from mechbench_compute.lexicon import kinds as K
from mechbench_compute.ops.intervene.prune import prune_circuits
from mechbench_compute.ops.records.diff import diff_collections
from mechbench_compute.ops.records.plot import build_chart
from mechbench_compute.ops.records.unnest import unnest


def _heads(rows, layers=(10, 11, 12)):
    return {"kind": "intervene/heads", "id": "mean", "coords": {}, "axes": ["layer", "head"],
            "measures": {"mean_delta": rows}, "layers": list(layers), "n_heads": len(rows[0]),
            "n_conditions": 2, "n_off_top1": 0, "metric": "entropy",
            "conditions": [{"id": "a", "target": {"id": 1, "text": "x"}, "baseline_logp": -1.0},
                           {"id": "b", "target": {"id": 2, "text": "y"}, "baseline_logp": -2.0}]}


def _trace(pairs, layers=(3, 5), point="resid_post"):
    return K.collection("intervene/trace", [
        {"id": pid, "coords": {"pair": pid}, "axes": ["layer", "position"],
         "measures": {"recovery": rec, "share": [[v / 2 for v in row] for row in rec]},
         "tokens": [f"t{i}" for i in range(len(rec[0]))]} for pid, rec in pairs.items()],
        method="exact", point=point, metric="logprob", layers=list(layers))


ROWS = [[0.1, -0.4, 0.0, 0.2], [-1.06, 0.3, 0.05, 0.0], [0.0, 0.0, -0.2, 0.7]]


class TestHeadsCells:
    def test_one_record_per_cell_not_per_condition(self):
        out = unnest(_heads(ROWS), {"field": "cells"})
        assert len(out["items"]) == 3 * 4 == out["unnested"]["records"]
        assert out["unnested"]["parents"] == 1

    def test_a_cell_is_shaped_like_a_circuit_component(self):
        cell = unnest(_heads(ROWS), {"field": "cells"})["items"][4]
        assert cell["address"] == "L11.attn.per_head_out.H0@all"
        assert (cell["layer"], cell["head"], cell["position"], cell["point"]) == (11, 0, "all", "attn.per_head_out")
        assert cell["mean_delta"] == -1.06
        assert cell["id"] == "mean/L11.attn.per_head_out.H0@all" and cell["parent"] == "mean"

    def test_the_addresses_are_the_ones_prune_gives_the_same_heads(self):
        cells = {c["address"]: c["mean_delta"] for c in unnest(_heads(ROWS), {"field": "cells"})["items"]}
        circuit = prune_circuits(_heads(ROWS), {"name": "c", "top": 12, "sign": "magnitude"})["items"][0]
        assert {p["address"]: p["effect"] for p in circuit["components"]} == {
            a: v for a, v in cells.items() if v != 0}

    def test_a_diff_keyed_by_address_reports_the_cells_that_moved(self):
        moved = [row[:] for row in ROWS]
        moved[1][0], moved[2][3] = -0.008, 0.9
        a = unnest(_heads(ROWS), {"field": "cells"})
        b = unnest(_heads(moved), {"field": "cells"})
        out = diff_collections(a, b, {"key": "address", "fields": ["mean_delta"]})
        assert sorted(r["coords"]["address"] for r in out["items"]) == [
            "L11.attn.per_head_out.H0@all", "L12.attn.per_head_out.H3@all"]
        assert all(r["status"] == "differs" for r in out["items"])

    def test_plot_draws_the_cells_as_a_heat_map(self):
        cells = unnest(_heads(ROWS), {"field": "cells"})
        spec = build_chart(cells, {"mark": "heat", "x": "layer", "y": "head", "value": "mean_delta",
                                   "scale": "diverging"})
        assert len(spec["data"]["rows"]) == 12
        assert {r["layer"] for r in spec["data"]["rows"]} == {10, 11, 12}


class TestTraceCells:
    def test_one_record_per_cell_with_positions_from_the_end(self):
        trace = _trace({"p1": [[0.0, 0.5, 1.0], [0.1, 0.2, 0.3]], "p2": [[1, 2, 3, 4], [5, 6, 7, 8]]})
        out = unnest(trace, {"field": "cells"})
        assert len(out["items"]) == 2 * 3 + 2 * 4
        first = out["items"][0]
        assert first["address"] == "L3.resid_post@-3"
        assert (first["layer"], first["position"], first["token"]) == (3, -3, "t0")
        assert (first["recovery"], first["share"]) == (0.0, 0.0)
        assert out["items"][5]["address"] == "L5.resid_post@-1"
        assert out["items"][0]["coords"]["pair"] == "p1"

    def test_the_point_is_the_headers(self):
        out = unnest(_trace({"p": [[1.0], [2.0]]}, point="mlp_out"), {"field": "cells"})
        assert [c["address"] for c in out["items"]] == ["L3.mlp_out@-1", "L5.mlp_out@-1"]

    def test_a_diff_between_two_traces_keyed_by_pair_and_address(self):
        a = unnest(_trace({"p": [[0.0, 0.5], [0.1, 0.2]]}), {"field": "cells"})
        b = unnest(_trace({"p": [[0.0, 0.5], [0.1, 0.9]]}), {"field": "cells"})
        out = diff_collections(a, b, {"key": ["parent", "address"], "fields": ["recovery"]})
        assert [r["coords"]["address"] for r in out["items"]] == ["L5.resid_post@-1"]

    def test_a_pair_with_an_error_has_no_cells(self):
        trace = _trace({"p": [[1.0], [2.0]]})
        trace["items"].append({"id": "bad", "axes": ["layer", "position"], "measures": {}, "error": "lengths"})
        assert len(unnest(trace, {"field": "cells"})["items"]) == 2


class TestOtherGrids:
    def test_a_lens_reads_resid_post(self):
        lens = K.collection("logits/lens", [
            {"id": "r", "axes": ["layer", "position"], "tokens": ["a", "b"],
             "measures": {"logprob": [[-3.0, -1.0], [-2.0, -0.1]], "rank": [[9, 2], [4, 0]]}}], layers=[0, 7])
        cells = unnest(lens, {"field": "cells"})["items"]
        assert len(cells) == 4
        last = cells[3]
        assert (last["address"], last["token"], last["logprob"], last["rank"]) == ("L7.resid_post@-1", "b", -0.1, 0)

    def test_an_attribution_addresses_the_embedding_and_each_layers_write(self):
        att = K.collection("logits/attribution", [
            {"id": "r", "axes": ["component"], "measures": {"contribution": [0.5, 1.0, -2.0]}}],
            layers=[0, 1], components=["embed", "L0", "L1"])
        cells = unnest(att, {"field": "cells"})["items"]
        assert [c["address"] for c in cells] == ["L0.resid_pre@-1", "L0.block@-1", "L1.block@-1"]
        assert [c["component"] for c in cells] == ["embed", "L0", "L1"]

    def test_an_attribution_by_sublayer_addresses_each_write_at_its_point(self):
        att = K.collection("logits/attribution", [
            {"id": "r", "axes": ["component"], "measures": {"contribution": [0.5, 1.0, -2.0, 0.25, 3.0]}}],
            layers=[0, 1], split="sublayer", components=["embed", "L0.attn", "L0.mlp", "L0.gate", "L1.attn"])
        cells = unnest(att, {"field": "cells"})["items"]
        assert [c["address"] for c in cells] == [
            "L0.resid_pre@-1", "L0.attn_out@-1", "L0.mlp_out@-1", "L0.gate_out@-1", "L1.attn_out@-1"]
        assert [(c["layer"], c["point"]) for c in cells][1:3] == [(0, "attn_out"), (0, "mlp_out")]
        assert cells[4]["contribution"] == 3.0

    def test_a_component_name_it_cannot_read_is_refused(self):
        att = K.collection("logits/attribution", [
            {"id": "r", "axes": ["component"], "measures": {"contribution": [0.5, 1.0]}}],
            layers=[0], components=["embed", "block0"])
        with pytest.raises(ValueError, match="'block0' is not `embed`"):
            unnest(att, {"field": "cells"})

    def test_a_divergence_reads_its_headers_point(self):
        div = K.collection("activations/divergence", [
            {"id": "r", "axes": ["layer", "position"], "tokens": ["a"], "measures": {"divergence": [[0.2]]}}],
            point="resid_pre", layers=[4])
        assert unnest(div, {"field": "cells"})["items"][0]["address"] == "L4.resid_pre@-1"

    def test_an_attention_grid_is_refused_with_the_way_that_works(self):
        att = K.collection("activations/attention", [
            {"id": "r", "axes": ["layer", "head", "query", "key"], "tokens": ["a"],
             "measures": {"weight": [[[[1.0]]]]}}], layers=[0], n_heads=1)
        with pytest.raises(ValueError, match="field: measures"):
            unnest(att, {"field": "cells"})


class TestAddresses:
    @pytest.mark.parametrize("grid", [
        _heads(ROWS),
        _trace({"p": [[0.0, 0.5, 1.0], [0.1, 0.2, 0.3]]}),
        K.collection("logits/attribution", [{"id": "r", "axes": ["component"],
                                             "measures": {"contribution": [0.5, 1.0]}}],
                     layers=[0], components=["embed", "L0"]),
    ])
    def test_an_address_is_never_null_and_never_repeats(self, grid):
        cells = unnest(grid, {"field": "cells"})["items"]
        addresses = [c["address"] for c in cells]
        assert cells and all(isinstance(a, str) and a for a in addresses)
        assert len(set(addresses)) == len(addresses)

    def test_a_heads_grid_alone_reads_its_measures_too(self):
        out = unnest(_heads(ROWS), {"field": "measures"})
        assert len(out["items"]) == 12
