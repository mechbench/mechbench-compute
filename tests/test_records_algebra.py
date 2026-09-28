from __future__ import annotations

import random

import pytest

from mechbench_compute.lexicon import kinds as K
from mechbench_compute.ops.records.contrast import contrast
from mechbench_compute.ops.records.correlate import correlate
from mechbench_compute.ops.records.count import count
from mechbench_compute.ops.records.derive import derive
from mechbench_compute.ops.records.filter import filter_records
from mechbench_compute.ops.records.group import group_records
from mechbench_compute.ops.records.join import join_records
from mechbench_compute.ops.records.sort import sort_records
from mechbench_compute.ops.records.summarize import group_stats


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


def _by_genre(out: dict) -> dict:
    return {r["coords"]["genre"]: r for r in out["items"]}


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

    def test_a_group_of_none_is_the_whole_input(self):
        out = group_records(_collection(_records()), {"aggregates": {"n": "count()"}}, {})
        assert out["items"] == [{"id": "0", "n": 60}]

    @pytest.mark.parametrize("bad,match", [
        ("nope(p)", "not an aggregate"), ("mean()", "takes 1 expression"),
        ("wilson(ok, lvl=1)", "takes no lvl"), ("mean(coords.genre)", "not a number"),
        ("share(p)", "not a boolean"),
    ])
    def test_refusals(self, bad, match):
        with pytest.raises(ValueError, match=match):
            group_records(_collection(_records()), {"aggregates": {"z": bad}}, {})

    def test_wilson_matches_count(self):
        recs = _records()
        old = {r["genre"]: r for r in count(recs, {"field": "ok", "equals": True, "by": ["genre"],
                                                  "interval": 0.9})["rows"]}
        new = _by_genre(group_records(_collection(recs), {"by": {"coords.genre": "coords.genre"},
                                                          "aggregates": {"w": "wilson(ok, level=0.9)"}}, {}))
        for genre, row in old.items():
            w = new[genre]["w"]
            assert (w["k"], w["n"], round(w["rate"], 4), round(w["lo"], 4), round(w["hi"], 4)) == \
                (row["k"], row["n"], row["rate"], row["lo"], row["hi"])

    def test_bootstrap_mean_and_median_match_summarize(self):
        recs = _records()
        old = {r["genre"]: r for r in group_stats(recs, {"value": "p", "by": ["genre"], "interval": 0.95,
                                                         "resamples": 500})["rows"]}
        new = _by_genre(group_records(_collection(recs), {"by": {"coords.genre": "coords.genre"}, "aggregates": {
            "b": "bootstrap_mean(p, resamples=500)", "med": "median(p)", "mn": "min(p)", "mx": "max(p)",
            "neg": "share(p < 0)"}}, {}))
        for genre, row in old.items():
            g = new[genre]
            assert (round(g["b"]["mean"], 4), round(g["b"]["lo"], 4), round(g["b"]["hi"], 4)) == \
                (row["mean"], row["lo"], row["hi"])
            assert (round(g["med"], 4), g["mn"], g["mx"], round(g["neg"], 3)) == \
                (row["median"], row["min"], row["max"], row["share_negative"])

    def test_spearman_matches_correlate(self):
        recs = _records()
        old = {r["genre"]: r for r in correlate(recs, {"x": "p", "y": "q", "by": ["genre"],
                                                       "interval": 0.95})["rows"]}
        new = _by_genre(group_records(_collection(recs), {"by": {"coords.genre": "coords.genre"},
                                                          "aggregates": {"s": "spearman(p, q, level=0.95)"}}, {}))
        for genre, row in old.items():
            s = new[genre]["s"]
            assert (s["n"], round(s["rho"], 4), round(s["lo"], 4), round(s["hi"], 4)) == \
                (row["n"], row["rho"], row["lo"], row["hi"])

    @pytest.mark.parametrize("paired", [None, "sample"])
    def test_paired_difference_matches_contrast(self, paired):
        recs = _records()
        params = {"value": "p", "on": "prompt", "a": "flash", "b": "pro", "by": ["genre"], "resamples": 300,
                  "seed": 3}
        if paired:
            params["paired"] = paired
        old = {r["genre"]: r for r in contrast(recs, params)["rows"]}
        pair = ", coords.sample" if paired else ""
        new = _by_genre(group_records(_collection(recs), {"by": {"coords.genre": "coords.genre"}, "aggregates": {
            "d": f'paired_difference(p, coords.prompt, "flash", "pro"{pair}, resamples=300, seed=3)'}}, {}))
        for genre, row in old.items():
            d = new[genre]["d"]
            assert (d["n"], round(d["diff"], 4), round(d["lo"], 4), round(d["hi"], 4),
                    round(d["share_positive"], 3)) == \
                (row["n"], row["diff"], row["lo"], row["hi"], row["share_positive"])


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
        assert sorted(r["e"] for r in items) == [4, 5]
        assert items[0]["coords"] == {"model": "gemma"} and [r["big"] for r in items] == [True, True]

    def test_an_expression_reading_no_param_is_refused_before_the_run(self):
        with pytest.raises(ValueError, match="'kk', which is not a param"):
            _run(self._graph({"$expr": "kk - 1"}), {"k": 3, "model": "gemma"})

    def test_an_expression_with_no_value_is_refused(self):
        with pytest.raises(ValueError, match="division by zero"):
            _run(self._graph({"$expr": "params.k // 0"}), {"k": 3, "model": "gemma"})
