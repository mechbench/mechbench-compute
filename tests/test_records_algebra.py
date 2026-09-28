from __future__ import annotations

import random

import pytest

from mechbench_compute.lexicon import kinds as K
from mechbench_compute.ops.records.derive import derive
from mechbench_compute.ops.records.filter import filter_records
from mechbench_compute.ops.records.group import group_records
from mechbench_compute.ops.records.join import join_records
from mechbench_compute.ops.records.sort import sort_records


def _records() -> list[dict]:
    rng = random.Random(7)
    out = []
    for i in range(60):
        out.append({
            "id": str(i),
            "coords": {"genre": ("noir", "cozy", "epic")[i // 2 % 3], "prompt": ("flash", "pro")[i % 2], "sample": i // 2},
            "p": round(rng.uniform(-1, 1), 3),
            "q": round(rng.uniform(0, 5), 3),
            "ok": rng.random() < 0.6,
        })
    return out


def _collection(items: list[dict]) -> dict:
    return K.collection("records/record", items)


class TestDerive:
    def test_fields_templates_and_drop(self):
        out = derive(_collection(_records()[:2]), {
            "fields": {"neg": "p < 0", "coords.model": "params.model"},
            "templates": {"label": "{coords.genre}: {q:.1f}"},
            "drop": ["ok"],
        }, {"model": "gemma"})
        first = out["items"][0]
        assert first["coords"] == {"genre": "noir", "prompt": "flash", "sample": 0, "model": "gemma"}
        assert first["label"] == f"noir: {_records()[0]['q']:.1f}"
        assert "ok" not in first and first["neg"] == (_records()[0]["p"] < 0)

    def test_keep_keeps_only_the_named_fields_and_the_id(self):
        out = derive(_collection(_records()[:1]), {"fields": {"neg": "p < 0"}, "keep": ["coords.genre", "neg", "absent"]}, {})
        assert out["items"][0] == {"id": "0", "coords": {"genre": "noir"}, "neg": _records()[0]["p"] < 0}

    def test_undefined_numbers_are_null_and_counted(self):
        out = derive(_collection([{"id": "a", "x": 0}, {"id": "b", "x": 2}]), {"fields": {"inv": "1 / x"}}, {})
        assert [r["inv"] for r in out["items"]] == [None, 0.5]
        assert sum(out["undefined"].values()) == 1

    def test_an_error_names_the_op(self):
        with pytest.raises(ValueError, match="records/derive"):
            derive(_collection([{"id": "a", "x": "s"}]), {"fields": {"y": "x + 1"}}, {})


class TestFilterAndSort:
    def test_filter_keeps_true_and_counts_null(self):
        items = [{"id": "a", "e": 3}, {"id": "b", "e": None}, {"id": "c", "e": 1}]
        out = filter_records(_collection(items), {"where": "e > 2"}, {})
        assert [r["id"] for r in out["items"]] == ["a"]
        assert (out["dropped"], out["unknown"]) == (2, 1)

    def test_sort_descending_with_nulls_last_and_limit(self):
        items = [{"id": "a", "e": 1}, {"id": "b", "e": None}, {"id": "c", "e": 5}, {"id": "d", "e": 3}]
        out = sort_records(_collection(items), {"by": ["-e"], "limit": 3}, {})
        assert [r["id"] for r in out["items"]] == ["c", "d", "a"]


class TestNaturalOrder:
    def test_digits_in_text_compare_as_numbers(self):
        ids = ["flash-s10", "flash-s2", "flash-s1", "pro-s0", "flash-s01", "10", "2", "a", "", "x9y", "x10"]
        stored = K.canonical_collection(_collection([{"id": i} for i in ids]))
        assert [r["id"] for r in stored["items"]] == [
            "", "2", "10", "a", "flash-s1", "flash-s01", "flash-s2", "flash-s10", "pro-s0", "x9y", "x10"]


class TestOrder:
    def _sorted(self) -> dict:
        items = [{"id": str(i), "e": e} for i, e in enumerate([3, 11, 7, 2])]
        return sort_records(_collection(items), {"by": ["-e"]}, {})

    def test_a_sort_writes_each_place_and_declares_it(self):
        out = self._sorted()
        assert [(r["e"], r["rank"]) for r in out["items"]] == [(11, 1), (7, 2), (3, 3), (2, 4)]
        assert out["order_by"] == ["rank"]

    def test_storage_keeps_the_declared_order_not_the_key(self):
        stored = K.canonical_collection(self._sorted())
        assert [r["e"] for r in stored["items"]] == [11, 7, 3, 2]
        plain = K.canonical_collection(_collection([{"id": "b"}, {"id": "a"}]))
        assert [r["id"] for r in plain["items"]] == ["a", "b"]

    def test_filter_derive_and_join_keep_the_order(self):
        kept = filter_records(self._sorted(), {"where": "e > 2"}, {})
        assert kept["order_by"] == ["rank"]
        derived = derive(kept, {"fields": {"half": "e / 2"}}, {})
        assert derived["order_by"] == ["rank"]
        assert "order_by" not in derive(kept, {"drop": ["rank"]}, {})
        joined = join_records(kept, _collection([{"id": "x", "e": 7}]), {"on": "e", "how": "left"}, {})
        assert joined["order_by"] == ["rank"]
        assert [r["e"] for r in K.canonical_collection(joined)["items"]] == [11, 7, 3]


class TestJoin:
    def test_inner_and_left(self):
        left = _collection([{"id": "1", "coords": {"f": 1}}, {"id": "2", "coords": {"f": 2}},
                            {"id": "3", "coords": {"f": None}}])
        right = _collection([{"id": "a", "fact": 1, "flag": True}, {"id": "b", "fact": 3}])
        inner = join_records(left, right, {"on": "coords.f", "on_right": "fact", "as": "flags"}, {})
        assert [r["flags"]["flag"] for r in inner["items"]] == [True]
        assert inner["unmatched"] == 2
        kept = join_records(left, right, {"on": "coords.f", "on_right": "fact", "how": "left"}, {})
        assert [r["right"] is None for r in kept["items"]] == [False, True, True]

    def test_a_repeated_right_key_is_refused(self):
        right = _collection([{"id": "a", "k": 1}, {"id": "b", "k": 1}])
        with pytest.raises(ValueError, match="unique"):
            join_records(_collection([{"id": "1", "k": 1}]), right, {"on": "k"}, {})


class TestGroup:
    def test_plain_aggregates_skip_nulls_and_count_them(self):
        items = [{"id": "0", "g": "a", "x": 1}, {"id": "1", "g": "a", "x": None},
                 {"id": "2", "g": "b", "x": 4}, {"id": "3", "g": "a", "x": 3}]
        out = group_records(_collection(items), {"by": {"g": "g"}, "aggregates": {
            "n": "count()", "sum": "sum(x)", "mean": "mean(x)", "xs": "collect(x)", "top": "max(x)"}}, {})
        a, b = out["items"]
        assert (a["g"], a["n"], a["sum"], a["mean"], a["xs"], a["top"]) == ("a", 3, 4, 2.0, [1, 3], 3)
        assert (b["g"], b["n"], b["sum"]) == ("b", 1, 4)
        assert out["missing"] == {"sum": 1, "mean": 1, "xs": 1, "top": 1}
        with pytest.raises(ValueError, match="None"):
            group_records(_collection(items), {"aggregates": {"m": "mean(x)"}, "on_missing": "fail"}, {})

    def test_a_named_method_unpacks_into_flat_fields(self):
        out = group_records(_collection(_records()), {"by": {"genre": "coords.genre"}, "aggregates": {
            "rate, lo, hi": "wilson(ok)", "n": "count()"}}, {})
        row = out["items"][0]
        assert set(row) == {"id", "genre", "rate", "lo", "hi", "n"} and 0 <= row["lo"] <= row["rate"] <= row["hi"] <= 1
        with pytest.raises(ValueError, match="has no rho"):
            group_records(_collection(_records()), {"aggregates": {"rate, rho": "wilson(ok)"}}, {})
        with pytest.raises(ValueError, match="answer an object"):
            group_records(_collection(_records()), {"aggregates": {"a, b": "mean(p)"}}, {})

    def test_no_by_answers_one_record_even_over_nothing(self):
        out = group_records(_collection([]), {"aggregates": {"n": "count()", "s": "sum(x)", "m": "mean(x)"}}, {})
        assert out["items"] == [{"id": "all", "n": 0, "s": 0, "m": None}]

    def test_a_group_of_none_is_the_whole_input(self):
        out = group_records(_collection(_records()), {"aggregates": {"n": "count()"}}, {})
        assert out["items"] == [{"id": "all", "n": 60}]

    def test_a_group_is_named_by_its_key(self):
        items = [{"id": str(i), "g": g, "k": k} for i, (g, k) in enumerate([("b", 10), ("a", 2), ("b", 10), ("a", None)])]
        out = group_records(_collection(items), {"by": {"g": "g", "k": "k"}, "aggregates": {"n": "count()"}}, {})
        assert [(r["id"], r["n"]) for r in out["items"]] == [("b|10", 2), ("a|2", 1), ("a|null", 1)]
        clash = [{"id": "0", "a": "x|y", "b": "z"}, {"id": "1", "a": "x", "b": "y|z"}]
        ids = [r["id"] for r in group_records(_collection(clash), {"by": {"a": "a", "b": "b"},
                                                                    "aggregates": {"n": "count()"}}, {})["items"]]
        assert ids == ['["x|y", "z"]', '["x", "y|z"]']

    @pytest.mark.parametrize("bad,match", [
        ("nope(p)", "not an aggregate"), ("mean()", "takes 1 expression"),
        ("wilson(ok, lvl=1)", "takes no lvl"), ("mean(coords.genre)", "not a number"),
        ("share(p)", "not a boolean"),
    ])
    def test_refusals(self, bad, match):
        with pytest.raises(ValueError, match=match):
            group_records(_collection(_records()), {"aggregates": {"z": bad}}, {})



def _run(graph: dict, params: dict) -> dict:
    from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec

    out = ProtocolExecutor().run(ProtocolSpec(
        kind="pipeline", prompt="", model_id=None,
        extra={"graph": {"dataflow": 2, **graph}, "params": params, "inputs": {}}))
    return (out.payload if hasattr(out, "payload") else out)["outputs"]


class TestThroughAProtocol:
    def _graph(self, limit) -> dict:
        items = [{"id": str(i), "e": i} for i in range(6)]
        return {"edges": [{"from": {"node": "top"}, "to": {"node": "tag", "port": "records"}}], "nodes": [
            {"id": "top", "block": "records/sort", "params": {"by": ["-e"], "limit": limit},
             "inputs": {"records": items}},
            {"id": "tag", "block": "records/derive",
             "params": {"fields": {"coords.model": "params.model", "big": "e > params.k"}}}]}

    def test_a_param_expression_is_computed_when_the_run_is_bound(self):
        out = _run(self._graph({"$expr": "params.k - 1"}), {"k": 3, "model": "gemma"})
        items = out["tag"]["items"]
        assert [(r["e"], r["rank"]) for r in items] == [(5, 1), (4, 2)]
        assert out["tag"]["order_by"] == ["rank"]
        assert items[0]["coords"] == {"model": "gemma"} and [r["big"] for r in items] == [True, True]

    def test_an_expression_reading_no_param_is_refused_before_the_run(self):
        with pytest.raises(ValueError, match="'kk', which is not a param"):
            _run(self._graph({"$expr": "kk - 1"}), {"k": 3, "model": "gemma"})

    def test_an_expression_with_no_value_is_refused(self):
        with pytest.raises(ValueError, match="division by zero"):
            _run(self._graph({"$expr": "params.k // 0"}), {"k": 3, "model": "gemma"})


class TestUnnestGrids:
    def test_a_grid_unnests_into_its_cells_and_keeps_the_header(self):
        from mechbench_compute.ops.records.unnest import unnest

        grid = {"id": "g1", "coords": {"country": "FR"}, "axes": ["layer", "position"],
                "measures": {"share": [[0.1, 0.2], [0.3, 0.4]]}, "tokens": ["Paris", "!"]}
        out = unnest(K.collection("intervene/trace", [grid], components=["embed", "L0"]), {"field": "measures"})
        assert out["components"] == ["embed", "L0"] and out["unnested"]["records"] == 4
        assert out["items"][1] == {"id": "g1/1", "parent": "g1", "token": "!", "share": 0.2,
                                   "coords": {"country": "FR", "layer": 0, "position": 1}}
        cells = group_records(out, {"by": {"layer": "coords.layer"}, "aggregates": {"m": "mean(share)"}}, {})
        assert [round(r["m"], 6) for r in cells["items"]] == [0.15, 0.35]
