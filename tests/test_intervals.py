from __future__ import annotations

import random

from mechbench_compute.lexicon import kinds as K
from mechbench_compute.ops.records.group import group_records


def _sweep(n_prompts=30, layers=(21, 22, 23, 24), seed=1):
    rng = random.Random(seed)
    out = []
    for i in range(n_prompts):
        hard = rng.gauss(0, 2.0)
        for layer in layers:
            effect = -3.0 if layer == 23 else -2.0
            out.append({"id": f"p{i}-{layer}", "prompt": f"p{i}", "layer": layer,
                        "delta_logp": hard + effect + rng.gauss(0, 0.3)})
    return out


def _group(recs, params):
    return group_records(K.collection("records/record", recs), params, {})["items"]


class TestBootstrapMean:
    def test_an_interval_brackets_the_mean(self):
        rows = _group(_sweep(), {"by": {"layer": "layer"},
                                 "aggregates": {"n, mean, lo, hi": "bootstrap_mean(delta_logp, resamples=500)"}})
        for row in rows:
            assert row["lo"] <= row["mean"] <= row["hi"]
            assert 0.3 < row["hi"] - row["lo"] < 2.5

    def test_the_interval_is_a_function_of_the_records_not_their_order(self):
        recs = _sweep()
        params = {"by": {"layer": "layer"}, "aggregates": {"lo, hi": "bootstrap_mean(delta_logp, level=0.9)"}}
        a, b = _group(recs, params), _group(list(reversed(recs)), params)
        assert {r["layer"]: (r["lo"], r["hi"]) for r in a} == {r["layer"]: (r["lo"], r["hi"]) for r in b}

    def test_more_records_narrow_it(self):
        params = {"aggregates": {"lo, hi": "bootstrap_mean(delta_logp)"}}
        [narrow] = _group(_sweep(n_prompts=200), params)
        [wide] = _group(_sweep(n_prompts=20), params)
        assert narrow["hi"] - narrow["lo"] < wide["hi"] - wide["lo"]

    def test_a_single_record_has_a_zero_width_interval(self):
        [row] = _group([{"id": "x", "delta_logp": 1.5}], {"aggregates": {"lo, hi": "bootstrap_mean(delta_logp)"}})
        assert (row["lo"], row["hi"]) == (1.5, 1.5)


class TestPairedDifference:
    def test_paired_the_layer_23_cost_is_resolved_against_its_neighbour(self):
        [row] = _group(_sweep(), {"aggregates": {
            "d": "paired_difference(delta_logp, layer, 23, 22, prompt, resamples=500)"}})
        d = row["d"]
        assert d["n"] == 30 and -1.3 < d["diff"] < -0.7
        assert d["lo"] < d["diff"] < d["hi"] and d["hi"] - d["lo"] < 0.5
        assert d["hi"] < 0 and d["share_positive"] == 0.0

    def test_unpaired_the_same_contrast_is_wide_and_uncertain(self):
        [row] = _group(_sweep(), {"aggregates": {
            "lo, hi": "paired_difference(delta_logp, layer, 23, 22, resamples=500)"}})
        assert row["hi"] - row["lo"] > 1.0

    def test_a_null_contrast_straddles_zero(self):
        [row] = _group(_sweep(), {"aggregates": {
            "lo, hi, share_positive": "paired_difference(delta_logp, layer, 22, 21, prompt, resamples=500)"}})
        assert row["lo"] < 0 < row["hi"]
        assert 0.1 < row["share_positive"] < 0.9

    def test_by_holds_other_coordinates_fixed(self):
        recs = [dict(r, id=f"{r['id']}-{p}", coords={"point": p}) for r in _sweep(n_prompts=12) for p in ("attn", "mlp")]
        rows = _group(recs, {"by": {"point": "coords.point"}, "aggregates": {
            "diff": "paired_difference(delta_logp, layer, 23, 22, prompt, resamples=200)"}})
        assert sorted(r["point"] for r in rows) == ["attn", "mlp"]
        assert all(r["diff"]["diff"] < 0 for r in rows)

    def test_a_side_that_is_absent_answers_none(self):
        [row] = _group(_sweep(), {"aggregates": {"d": "paired_difference(delta_logp, layer, 99, 22)"}})
        assert row["d"] is None

    def test_pairs_that_never_meet_answer_none(self):
        recs = [{"id": f"a{i}", "layer": 23, "delta_logp": 1.0} for i in range(3)] + \
               [{"id": f"b{i}", "layer": 22, "delta_logp": 1.0} for i in range(3)]
        [row] = _group(recs, {"aggregates": {"d": "paired_difference(delta_logp, layer, 23, 22, id)"}})
        assert row["d"] is None

    def test_reads_the_coordinate_where_the_expression_says(self):
        recs = [{"id": f"p{i}-{arm}", "pair": i, "coords": {"arm": arm}, "v": (1.0 if arm == "t" else 0.0) + 0.01 * i}
                for i in range(10) for arm in ("t", "c")]
        [row] = _group(recs, {"aggregates": {"diff": 'paired_difference(v, coords.arm, "t", "c", pair, resamples=100)'}})
        assert abs(row["diff"]["diff"] - 1.0) < 1e-9
