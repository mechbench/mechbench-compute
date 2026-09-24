from __future__ import annotations

import hashlib
import random

import pytest
from mechbench_schema import dump_canonical

from mechbench_compute import bench, ops
from mechbench_compute import isomorphism as iso
from mechbench_compute import reduce as rd
from mechbench_compute.ops.records.correlate import (
    compute_spearman,
    correlate,
    estimate_fisher_interval,
)
from mechbench_compute.ops.records.count import count, estimate_wilson
from mechbench_compute.ops.records.unnest import unnest
from mechbench_compute.ops.records.zip import zip_branches
from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec


def _verdicts(rung: str, wins: dict[str, int], n: int = 30) -> list[dict]:
    rows = []
    for prompt, k in wins.items():
        for i in range(n):
            winner = "A" if i < k else "B"
            votes = []
            for v, order in enumerate(("AB", "BA", "BA")):
                shown = winner if order == "AB" else ("B" if winner == "A" else "A")
                votes.append({"vote": v, "order": order, "parsed": True,
                              "winner": winner, "shown_winner": shown})
            rows.append({"id": f"{prompt}-{i}", "coords": {"prompt": prompt, "sample": i},
                         "winner": winner, "counts": {winner: 3}, "n_votes": 3,
                         "n_parsed": 3, "agreement": 1, "votes": votes})
    return rows


class TestCount:
    def test_k_of_n_with_the_wilson_interval(self):
        out = count(_verdicts("r", {"neutral": 8, "flash": 24}),
                    {"field": "winner", "equals": "A", "by": ["prompt"]})
        rows = {r["prompt"]: r for r in out["rows"]}
        assert (rows["neutral"]["k"], rows["neutral"]["n"], rows["neutral"]["rate"]) == (8, 30, 0.2667)
        assert (rows["neutral"]["lo"], rows["neutral"]["hi"]) == (0.1418, 0.4445)
        assert (rows["flash"]["lo"], rows["flash"]["hi"]) == (0.6269, 0.9049)
        assert out["interval"] == {"level": 0.95, "method": "wilson", "of": "proportion"}
        assert out["counted"] == {"field": "winner", "equals": "A"}
        assert [c["name"] for c in out["columns"]] == ["prompt", "k", "n", "rate", "lo", "hi"]

    @pytest.mark.parametrize("k,n", [(0, 30), (30, 30), (1, 2), (17, 30), (809, 2340)])
    def test_the_interval_is_scipys_wilson(self, k, n):
        stats = pytest.importorskip("scipy.stats")
        ci = stats.binomtest(k, n).proportion_ci(confidence_level=0.9, method="wilson")
        lo, hi = estimate_wilson(k, n, 0.9)
        assert lo == pytest.approx(ci.low, abs=1e-12) and hi == pytest.approx(ci.high, abs=1e-12)

    def test_none_of_thirty_is_not_certainty(self):
        lo, hi = estimate_wilson(0, 30, 0.95)
        assert lo == 0.0 and 0.1 < hi < 0.12

    def test_a_missing_field_is_refused_unless_skipped_and_then_counted(self):
        recs = [{"id": "a", "winner": "A"}, {"id": "b"}, {"id": "c", "winner": "B"}]
        with pytest.raises(ValueError, match="'b' has no 'winner'"):
            count(recs, {"field": "winner", "equals": "A"})
        out = count(recs, {"field": "winner", "equals": "A", "on_missing": "skip"})
        assert (out["rows"][0]["k"], out["rows"][0]["n"], out["n_missing"]) == (1, 2, 1)

    def test_true_is_not_one(self):
        recs = [{"id": "a", "hit": True}, {"id": "b", "hit": 1}, {"id": "c", "hit": 1}]
        assert count(recs, {"field": "hit"})["rows"][0]["k"] == 1
        assert count(recs, {"field": "hit", "equals": 1})["rows"][0]["k"] == 2

    def test_a_level_outside_zero_and_one_is_refused(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            count([{"id": "a", "hit": True}], {"field": "hit", "interval": 95})


class TestCorrelate:
    def test_the_same_order_is_one_and_the_reverse_is_minus_one(self):
        recs = [{"id": str(i), "x": i, "y": i * i, "z": -i} for i in range(6)]
        assert correlate(recs, {"x": "x", "y": "y"})["rows"][0]["rho"] == 1.0
        assert correlate(recs, {"x": "x", "y": "z"})["rows"][0]["rho"] == -1.0

    def test_ties_take_their_average_rank_as_scipy_does(self):
        stats = pytest.importorskip("scipy.stats")
        rng = random.Random(3)
        xs = [rng.choice([0.0, 0.01, 0.06, -0.19, -0.24]) for _ in range(12)]
        ys = [round(rng.random(), 2) for _ in range(12)]
        assert compute_spearman(xs, ys) == pytest.approx(stats.spearmanr(xs, ys).statistic, abs=1e-12)

    def test_a_constant_field_or_too_few_records_has_no_order(self):
        assert compute_spearman([1, 1, 1, 1], [1, 2, 3, 4]) is None
        assert compute_spearman([1, 2], [2, 1]) is None

    def test_rows_per_group_and_the_interval_when_asked(self):
        rng = random.Random(0)
        recs = [{"id": f"{g}{i}", "coords": {"g": g}, "x": i, "y": i + rng.gauss(0, 3)}
                for g in ("a", "b") for i in range(20)]
        plain = correlate(recs, {"x": "x", "y": "y", "by": ["g"]})
        assert plain["interval"] is None and "lo" not in plain["rows"][0]
        out = correlate(recs, {"x": "x", "y": "y", "by": ["g"], "interval": 0.95})
        assert [r["g"] for r in out["rows"]] == ["a", "b"]
        for r in out["rows"]:
            assert r["n"] == 20 and r["lo"] < r["rho"] < r["hi"]
        assert out["correlation"] == {"method": "spearman", "x": "x", "y": "y"}
        assert out["interval"]["method"] == "fisher-z-bonett-wright"

    def test_the_interval_is_undefined_below_four_records_or_at_one(self):
        assert estimate_fisher_interval(0.5, 3, 0.95) == (None, None)
        assert estimate_fisher_interval(1.0, 10, 0.95) == (None, None)
        lo, hi = estimate_fisher_interval(0.82, 10, 0.95)
        assert 0.25 < lo < 0.82 < hi < 1.0

    def test_a_missing_field_is_refused_unless_skipped(self):
        recs = [{"id": "a", "x": 1, "y": 2}, {"id": "b", "x": 2}]
        with pytest.raises(ValueError, match="'b' has no 'y'"):
            correlate(recs, {"x": "x", "y": "y"})
        assert correlate(recs, {"x": "x", "y": "y", "on_missing": "skip"})["n_missing"] == 1


class TestUnnest:
    def test_one_record_per_vote_with_the_parents_coordinates(self):
        out = unnest(_verdicts("r", {"flash": 1}, n=2), {"field": "votes", "index": "vote"})
        items = out["items"]
        assert len(items) == 6 and out["unnested"] == {"field": "votes", "parents": 2, "records": 6}
        first = items[0]
        assert first["id"] == "flash-0/0" and first["parent"] == "flash-0"
        assert first["coords"] == {"prompt": "flash", "sample": 0, "vote": 0}
        assert first["shown_winner"] == "A" and first["order"] == "AB"

    def test_a_plain_value_is_stored_under_as_and_an_empty_list_gives_nothing(self):
        recs = [{"id": "a", "tags": ["x", "y"]}, {"id": "b", "tags": []}]
        out = unnest(recs, {"field": "tags", "as": "tag"})
        assert [(i["id"], i["tag"], i["coords"]["index"]) for i in out["items"]] == [("a/0", "x", 0), ("a/1", "y", 1)]

    def test_a_missing_list_is_refused_unless_skipped(self):
        recs = [{"id": "a", "tags": ["x"]}, {"id": "b"}]
        with pytest.raises(ValueError, match="'b' has no 'tags'"):
            unnest(recs, {"field": "tags"})
        assert unnest(recs, {"field": "tags", "on_missing": "skip"})["unnested"]["n_missing"] == 1

    def test_an_index_that_is_already_a_coordinate_is_refused(self):
        with pytest.raises(ValueError, match="already has a 'prompt'"):
            unnest(_verdicts("r", {"flash": 1}, n=1), {"field": "votes", "index": "prompt"})


class TestZipByField:
    def test_table_rows_zip_by_the_columns_they_were_grouped_on(self):
        table = count(_verdicts("r", {"neutral": 8, "flash": 24}),
                      {"field": "winner", "equals": "A", "by": ["prompt"]})
        other = [{"id": "n", "coords": {"prompt": "neutral"}, "gap": 0.1},
                 {"id": "f", "coords": {"prompt": "flash"}, "gap": 0.4}]
        out = zip_branches({"branches": [{"node": "t", "value": table}, {"node": "o", "value": other}]},
                           {"by": ["prompt"], "flatten": True})
        rows = {r["coords"]["prompt"]: r for r in out["items"]}
        assert rows["flash"]["t_rate"] == 0.8 and rows["flash"]["o_gap"] == 0.4


def _leaves(n=90, seed=1):
    rng = random.Random(seed)
    return [{"id": f"r{i}", "coords": {"g": rng.choice("abc")}, "won": rng.choice(["A", "B"]),
             "x": rng.randint(0, 5), "y": rng.random()} for i in range(n)]


class TestChunks:
    @pytest.mark.parametrize("block,params", [
        ("records/count", {"field": "won", "equals": "A", "by": ["g"]}),
        ("records/correlate", {"x": "x", "y": "y", "by": ["g"], "interval": 0.9}),
    ])
    def test_any_partition_reduces_to_the_flat_table(self, block, params):
        flat = ops.run_standalone(block, {"records": _leaves()}, params)
        m = rd.find_monoid(block, params)
        assert m.finalize(m.partial(_leaves(), params), params) == flat
        assert iso.check(block, _leaves(), params, trials=20, seed=5)["exact"] is True


STORE = {f"lab/p/results/{rung}/compare": {"kind": "collection", "item_kind": "eval/verdict",
                                           "key": ["id"], "items": _verdicts(rung, wins)}
         for rung, wins in {"up": {"neutral": 20, "flash": 24}, "down": {"neutral": 8, "flash": 5}}.items()}


@pytest.fixture
def fake_bench(monkeypatch):
    def fetch(ref, with_meta=False):
        obj = {"payload": STORE[str(ref)]}
        meta = {"content_hash": "sha256:" + hashlib.sha256(dump_canonical(STORE[str(ref)])).hexdigest()}
        return (obj, meta) if with_meta else obj

    monkeypatch.setattr(bench, "fetch", fetch)


def test_an_analysis_is_a_protocol_over_stored_results(fake_bench):
    other = [{"id": f"{r}-{p}", "coords": {"rung": r, "prompt": p}, "gap": g}
             for (r, p), g in {("up", "neutral"): 0.3, ("up", "flash"): 0.4,
                               ("down", "neutral"): -0.2, ("down", "flash"): -0.1}.items()]
    graph = {"dataflow": 2, "nodes": [
        {"id": "pairs", "block": "records/union", "params": {"batch_axis": "rung"}},
        {"id": "preferred", "block": "records/count",
         "params": {"field": "winner", "equals": "A", "by": ["rung", "prompt"]}},
        {"id": "votes", "block": "records/unnest", "params": {"field": "votes", "index": "vote"}},
        {"id": "first_shown", "block": "records/count",
         "params": {"field": "shown_winner", "equals": "A", "on_missing": "skip"}},
        {"id": "against", "block": "records/zip",
         "params": {"by": ["rung", "prompt"], "flatten": True, "names": ["pair", "other"]}},
        {"id": "order", "block": "records/correlate", "params": {"x": "other_gap", "y": "pair_rate"}},
        {"id": "other", "block": "records/select", "params": {},
         "inputs": {"records": {"kind": "collection", "item_kind": "records/record",
                                "key": ["id"], "items": other}}},
    ], "edges": [
        {"from": {"input": "up"}, "to": {"node": "pairs", "port": "up"}},
        {"from": {"input": "down"}, "to": {"node": "pairs", "port": "down"}},
        {"from": {"node": "pairs"}, "to": {"node": "preferred", "port": "records"}},
        {"from": {"node": "pairs"}, "to": {"node": "votes", "port": "records"}},
        {"from": {"node": "votes"}, "to": {"node": "first_shown", "port": "records"}},
        {"from": {"node": "preferred"}, "to": {"node": "against", "port": "branches"}, "index": 0},
        {"from": {"node": "other"}, "to": {"node": "against", "port": "branches"}, "index": 1},
        {"from": {"node": "against"}, "to": {"node": "order", "port": "records"}},
    ]}
    extra = {"graph": graph, "params": {},
             "inputs": {r: {"$ref": {"bench": f"lab/p/results/{r}/compare"}} for r in ("up", "down")},
             "outputs": [{"name": n, "from": {"node": n}} for n in ("preferred", "first_shown", "order")]}
    out = ProtocolExecutor().run(ProtocolSpec(kind="pipeline", prompt="", model_id=None, extra=extra))
    outs = (out.payload if hasattr(out, "payload") else out)["outputs"]
    rates = {(r["rung"], r["prompt"]): (r["k"], r["n"]) for r in outs["preferred"]["rows"]}
    assert rates == {("down", "flash"): (5, 30), ("down", "neutral"): (8, 30),
                     ("up", "flash"): (24, 30), ("up", "neutral"): (20, 30)}
    first = outs["first_shown"]["rows"][0]
    wins_a = 20 + 24 + 8 + 5
    assert (first["k"], first["n"]) == (wins_a + 2 * (120 - wins_a), 360)
    assert outs["order"]["rows"][0]["rho"] == pytest.approx(0.8, abs=1e-9)
