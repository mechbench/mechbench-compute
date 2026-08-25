"""Pure-block contracts (task 000275): the stdlib pieces every
protocol leans on — deterministic, growth-safe, expectation-judging."""

from mechbench_compute.blocks import eval_expectation, factor_cross, template

WORDS = ["alpha", "bravo", "charlie", "delta", "echo"]


def test_factor_cross_is_the_full_cartesian_product():
    recs = factor_cross({"factors": [
        {"name": "gender", "levels": [{"key": "m"}, {"key": "f"}]},
        {"name": "prompt", "levels": [{"key": "plain", "value": "Say hi."},
                                       {"key": "fancy", "value": "Declaim!"}]},
    ]})
    assert len(recs) == 4
    ids = {r["id"] for r in recs}
    assert ids == {"m-plain", "m-fancy", "f-plain", "f-fancy"}
    one = next(r for r in recs if r["id"] == "f-fancy")
    assert one["coords"] == {"gender": "f", "prompt": "fancy"}
    assert one["values"] == {"gender": "f", "prompt": "Declaim!"}


def test_factor_cross_accepts_legacy_axes_spelling():
    legacy = factor_cross({"axes": [{"name": "x", "levels": [{"key": "1"}]}]})
    modern = factor_cross({"factors": [{"name": "x", "levels": [{"key": "1"}]}]})
    assert legacy == modern


def test_sampled_values_are_deterministic_in_seed_and_index_alone():
    gen = {"kind": "words", "size": 3, "count": 4, "seed": 42,
           "word_list": WORDS, "key_prefix": "w"}
    a = factor_cross({"factors": [{"name": "seed", "sampled": gen}]})
    b = factor_cross({"factors": [{"name": "seed", "sampled": gen}]})
    assert a == b
    other_seed = dict(gen, seed=43)
    c = factor_cross({"factors": [{"name": "seed", "sampled": other_seed}]})
    assert [r["values"]["seed"] for r in c] != [r["values"]["seed"] for r in a]


def test_growing_a_sampled_factor_preserves_the_original_membership():
    # The 100 -> 1000 growth guarantee: raising `count` must extend the
    # set, never reshuffle it (values depend on (seed, index) alone).
    small = {"kind": "words", "size": 3, "count": 5, "seed": 7,
             "word_list": WORDS, "key_prefix": "w"}
    big = dict(small, count=12)
    first = factor_cross({"factors": [{"name": "s", "sampled": small}]})
    grown = factor_cross({"factors": [{"name": "s", "sampled": big}]})
    assert grown[: len(first)] == first
    assert len(grown) == 12


def test_generators_stamp_a_kind_coordinate():
    recs = factor_cross({"factors": [{
        "name": "seed",
        "sampled": {"kind": "noise", "size": 8, "count": 2, "seed": 1},
    }]})
    assert all(r["coords"]["seed_kind"] == "noise-8" for r in recs)


def test_template_substitutes_to_fixpoint():
    recs = factor_cross({"factors": [
        {"name": "gender", "levels": [{"key": "m", "value": "his"}]},
        {"name": "opening", "levels": [
            {"key": "elaborate", "value": "Marcus adjusted {gender} coat."},
        ]},
    ]})
    out = template(recs, {"templates": {
        "user": "Continue: {opening}",
        "system": "No placeholders here.",
    }})
    assert out[0]["user"] == "Continue: Marcus adjusted his coat."
    assert out[0]["system"] == "No placeholders here."
    assert out[0]["coords"]["gender"] == "m"


def test_eval_expectation_judges_and_aggregates():
    results = [
        {"id": "die", "outcome_mass": {"1": 1 / 6, "2": 1 / 6, "3": 1 / 6,
                                        "4": 1 / 6, "5": 1 / 6, "6": 1 / 6},
         "entropy_bits": 2.58},
        {"id": "capital", "top_tokens": [{"token": "Paris", "p": 0.999}],
         "entropy_bits": 0.01},
        {"id": "loaded", "outcome_mass": {"1": 0.9, "2": 0.02, "3": 0.02,
                                           "4": 0.02, "5": 0.02, "6": 0.02},
         "entropy_bits": 0.7},
    ]
    expectations = [
        {"id": "die", "expect": {"kind": "uniform",
                                  "over": ["1", "2", "3", "4", "5", "6"],
                                  "max_kl_bits": 0.05}},
        {"id": "capital", "expect": {"kind": "answer", "value": "Paris",
                                      "min_p": 0.99}},
        {"id": "loaded", "expect": {"kind": "uniform",
                                     "over": ["1", "2", "3", "4", "5", "6"],
                                     "max_kl_bits": 0.05}},
    ]
    table = eval_expectation(
        {"results": results, "expectations": expectations}, {})
    assert table["kind"] == "metric_table"
    by_id = {r["id"]: r for r in table["rows"]}
    assert by_id["die"]["pass"] == "True"
    assert by_id["capital"]["pass"] == "True"
    assert by_id["loaded"]["pass"] == "False"
    assert by_id["ALL"]["n_judged"] == 3
    assert abs(by_id["ALL"]["pass_rate"] - 2 / 3) < 1e-3


def test_eval_expectation_accepts_params_fallback():
    table = eval_expectation({}, {
        "results": [{"id": "a", "entropy_bits": 3.0}],
        "expectations": [{"id": "a", "expect": {"kind": "min_entropy",
                                                  "bits": 2.0}}],
    })
    row_a = next(r for r in table["rows"] if r["id"] == "a")
    assert row_a["pass"] == "True"


def test_suite_metric_records_shapes_lm_eval_results():
    from mechbench_compute.blocks import suite_metric_records
    results = {"arc_easy": {"alias": "arc_easy",
                             "acc,none": 0.74, "acc_stderr,none": 0.02,
                             "acc_norm,none": 0.70,
                             "acc_norm_stderr,none": 0.021}}
    recs = suite_metric_records(results, {"arc_easy": {"effective": 50}},
                                variant="adapted")
    assert [r["id"] for r in recs] == ["arc_easy:acc:adapted",
                                        "arc_easy:acc_norm:adapted"]
    acc = recs[0]
    assert acc["coords"] == {"task": "arc_easy", "metric": "acc",
                              "variant": "adapted"}
    assert acc["value"] == 0.74 and acc["stderr"] == 0.02 and acc["n"] == 50


def test_table_from_records_flattens_coords_and_types_columns():
    from mechbench_compute.blocks import table_from_records
    table = table_from_records([
        {"id": "a", "coords": {"task": "arc_easy", "metric": "acc"},
         "value": 0.7, "delta": 0.01},
        {"id": "b", "coords": {"task": "arc_easy", "metric": "acc_norm"},
         "value": 0.68, "delta": -0.02},
    ], {"name": "deltas"})
    assert table["kind"] == "metric_table"
    names = [c["name"] for c in table["columns"]]
    assert names == ["id", "task", "metric", "value", "delta"]
    dt = {c["name"]: c["dtype"] for c in table["columns"]}
    assert dt["delta"] == "number" and dt["task"] == "string"
    assert table["rows"][1]["delta"] == -0.02


def test_suite_records_flow_through_union_and_paired_delta():
    from mechbench_compute.blocks import paired_delta, suite_metric_records, union
    base = suite_metric_records({"arc_easy": {"acc,none": 0.70}}, {}, "base")
    adapted = suite_metric_records({"arc_easy": {"acc,none": 0.73}}, {}, "adapted")
    merged = union({"a_base": base, "b_adapted": adapted}, {})
    deltas = paired_delta(merged, {"match_on": ["task", "metric"],
                                    "baseline_where": {"variant": "base"},
                                    "value": "value"})
    assert len(deltas) == 1
    assert abs(deltas[0]["delta"] - 0.03) < 1e-9


def test_viz_spec_references_its_source_or_inlines_rows():
    from mechbench_compute.blocks import viz_spec
    table = {"kind": "metric_table", "rows": [{"id": "a", "model": "e2b", "v": 1.0}]}
    ref = viz_spec(table, {"mark": "bar", "encoding": {"x": "model", "y": "v"}},
                     source_label="benji/marcus/metrics/t")
    assert ref["kind"] == "viz_spec" and ref["source"] == "benji/marcus/metrics/t"
    assert "data" not in ref
    inline = viz_spec(table, {"encoding": {"x": "model", "y": "v"}})
    assert inline["data"]["rows"] == [{"id": "a", "model": "e2b", "v": 1.0}]
    recs = [{"id": "r", "coords": {"task": "arc"}, "value": 0.8}]
    flat = viz_spec(recs, {"encoding": {"x": "task", "y": "value"}})
    assert flat["data"]["rows"] == [{"id": "r", "value": 0.8, "task": "arc"}]


def test_uniform_masses_derive_from_top_tokens():
    """The spinner-fairness regression (000315): plain decision reads
    emit top_tokens and no outcome_mass; the judge must derive rather
    than fail a distribution it never looked at."""
    results = [{"id": "c1", "entropy_bits": 1.99, "top_tokens": [
        {"token": "3", "p": 0.2997}, {"token": "1", "p": 0.2334},
        {"token": "2", "p": 0.2334}, {"token": "4", "p": 0.2334},
        {"token": "0", "p": 0.0001}]}]
    expectations = [{"id": "c1", "expect": {
        "kind": "uniform", "over": ["1", "2", "3", "4"], "max_kl_bits": 0.1}}]
    table = eval_expectation(
        {"results": results, "expectations": expectations}, {})
    row = table["rows"][0]
    assert row["pass"] == "True"  # pass serializes as string, per the column dtype
    assert row["kl_bits"] < 0.02
    assert row["outcome_mass"] > 0.99


def test_uniform_without_any_distribution_is_unjudgeable_not_false():
    results = [{"id": "c1", "entropy_bits": 0.5}]
    expectations = [{"id": "c1", "expect": {
        "kind": "uniform", "over": ["1", "2"], "max_kl_bits": 0.1}}]
    table = eval_expectation(
        {"results": results, "expectations": expectations}, {})
    row = table["rows"][0]
    assert "unjudgeable" in str(row["pass"])
    # ...and the aggregate does not count it as a judged failure.
    agg = table["rows"][-1]
    assert agg["n_judged"] == 0

