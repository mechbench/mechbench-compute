"""Pure-block contracts: the stdlib pieces every
protocol leans on — deterministic, growth-safe, expectation-judging."""

import pytest

from mechbench_compute.ops.eval.expect import check_expectations
from mechbench_compute.ops.records.cross import cross_factors
from mechbench_compute.ops.records.fill import fill_templates

WORDS = ["alpha", "bravo", "charlie", "delta", "echo"]


def test_factor_cross_is_the_full_cartesian_product():
    recs = cross_factors({"factors": [
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
    legacy = cross_factors({"axes": [{"name": "x", "levels": [{"key": "1"}]}]})
    modern = cross_factors({"factors": [{"name": "x", "levels": [{"key": "1"}]}]})
    assert legacy == modern


def test_sampled_values_are_deterministic_in_seed_and_index_alone():
    gen = {"kind": "words", "size": 3, "count": 4, "seed": 42,
           "word_list": WORDS, "key_prefix": "w"}
    a = cross_factors({"factors": [{"name": "seed", "sampled": gen}]})
    b = cross_factors({"factors": [{"name": "seed", "sampled": gen}]})
    assert a == b
    other_seed = dict(gen, seed=43)
    c = cross_factors({"factors": [{"name": "seed", "sampled": other_seed}]})
    assert [r["values"]["seed"] for r in c] != [r["values"]["seed"] for r in a]


def test_growing_a_sampled_factor_preserves_the_original_membership():
    # The 100 -> 1000 growth guarantee: raising `count` must extend the
    # set, never reshuffle it (values depend on (seed, index) alone).
    small = {"kind": "words", "size": 3, "count": 5, "seed": 7,
             "word_list": WORDS, "key_prefix": "w"}
    big = dict(small, count=12)
    first = cross_factors({"factors": [{"name": "s", "sampled": small}]})
    grown = cross_factors({"factors": [{"name": "s", "sampled": big}]})
    assert grown[: len(first)] == first
    assert len(grown) == 12


def test_generators_stamp_a_kind_coordinate():
    recs = cross_factors({"factors": [{
        "name": "seed",
        "sampled": {"kind": "noise", "size": 8, "count": 2, "seed": 1},
    }]})
    assert all(r["coords"]["seed_kind"] == "noise-8" for r in recs)


def test_template_substitutes_to_fixpoint():
    recs = cross_factors({"factors": [
        {"name": "gender", "levels": [{"key": "m", "value": "his"}]},
        {"name": "opening", "levels": [
            {"key": "elaborate", "value": "Marcus adjusted {gender} coat."},
        ]},
    ]})
    out = fill_templates(recs, {"templates": {
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
    # The reads are in the older spelling (`outcome_mass`, `top_tokens`);
    # the judge reads them as distributions.
    out = check_expectations(
        {"results": results, "expectations": expectations}, {})
    assert out["item_kind"] == "eval/verdict"
    by_id = {r["id"]: r for r in out["items"]}
    assert by_id["die"]["pass"] is True
    assert by_id["capital"]["pass"] is True
    assert by_id["loaded"]["pass"] is False
    assert out["summary"]["n_judged"] == 3
    assert abs(out["summary"]["pass_rate"] - 2 / 3) < 1e-3
    assert "ALL" not in by_id


def test_eval_expectation_reads_a_current_decision_collection():
    from mechbench_compute.lexicon import kinds as K

    def tok(t, p):
        import math
        return {"token": {"id": sum(map(ord, t)), "text": t}, "p": p, "logp": math.log(p)}
    reads = K.collection("logits/decision", [
        {"id": "die", "entropy_bits": 2.58,
         "top": [tok(str(i), 1 / 6) for i in range(1, 7)],
         "tracked": {str(i): tok(str(i), 1 / 6) for i in range(1, 7)}},
        {"id": "capital", "entropy_bits": 0.01, "top": [tok("Paris", 0.999)]},
    ])
    out = check_expectations({"results": reads, "expectations": [
        {"id": "die", "expect": {"type": "uniform", "over": ["1", "2", "3", "4", "5", "6"]}},
        {"id": "capital", "expect": {"type": "answer", "value": "Paris"}},
    ]}, {})
    by_id = {r["id"]: r for r in out["items"]}
    assert by_id["die"]["pass"] is True and by_id["die"]["mass"] == pytest.approx(1.0, abs=1e-3)
    assert by_id["capital"]["pass"] is True and by_id["capital"]["p_expected"] == 0.999


def test_eval_expectation_reads_its_ports_only():
    # Inputs arrive on ports, never under params: a result list given
    # there is not read (the executor refuses it by name before this).
    with pytest.raises(KeyError):
        check_expectations({}, {
            "results": [{"id": "a", "entropy_bits": 3.0}],
            "expectations": [{"id": "a", "expect": {"kind": "min_entropy", "bits": 2.0}}],
        })


def test_suite_metric_records_shapes_lm_eval_results():
    from mechbench_compute.ops.eval.benchmark import build_metric_records
    results = {"arc_easy": {"alias": "arc_easy",
                             "acc,none": 0.74, "acc_stderr,none": 0.02,
                             "acc_norm,none": 0.70,
                             "acc_norm_stderr,none": 0.021}}
    recs = build_metric_records(results, {"arc_easy": {"effective": 50}},
                                variant="adapted")
    assert [r["id"] for r in recs] == ["arc_easy:acc:adapted",
                                        "arc_easy:acc_norm:adapted"]
    acc = recs[0]
    assert acc["coords"] == {"task": "arc_easy", "metric": "acc",
                              "variant": "adapted"}
    assert acc["value"] == 0.74 and acc["stderr"] == 0.02 and acc["n"] == 50


class TestRename:
    """records/rename: the one visible adaptation step."""

    def test_moves_fields_and_keeps_the_rest(self):
        from mechbench_compute.ops.records.rename import rename

        out = rename([{"id": "a", "coords": {"g": "x"}, "question": "Q?", "gold": "42"}],
                     {"fields": {"question": "user", "gold": "reference"}})
        assert out == [{"id": "a", "coords": {"g": "x"}, "user": "Q?", "reference": "42"}]

    def test_a_dotted_path_moves_into_and_out_of_a_nested_object(self):
        from mechbench_compute.ops.records.rename import rename

        doc = {"id": "s0", "text": "…", "hit": 1, "metadata": {"coords": {"prompt": "flash"}}}
        # Moves apply in order: the coords lift first, then the hit into it.
        out = rename([doc], {"fields": {"metadata.coords": "coords", "hit": "coords.hit"}})
        assert out[0] == {"id": "s0", "text": "…", "metadata": {},
                          "coords": {"prompt": "flash", "hit": 1}}
        # The input record was not mutated.
        assert doc["hit"] == 1 and doc["metadata"]["coords"] == {"prompt": "flash"}

    def test_a_missing_field_is_left_alone_and_an_empty_map_is_refused(self):
        from mechbench_compute.ops.records.rename import rename

        assert rename([{"id": "a"}], {"fields": {"nope": "user"}}) == [{"id": "a"}]
        with pytest.raises(ValueError, match="fields"):
            rename([{"id": "a"}], {})

    def test_it_is_registered_and_reads_a_collection(self):
        from mechbench_compute import ops
        from mechbench_compute.lexicon import kinds as K

        out = ops.run_standalone(
            "records/rename",
            {"records": K.collection("records/record", [{"id": "a", "q": 1}])},
            {"fields": {"q": "user"}})
        assert out["item_kind"] == "records/record" and out["items"] == [{"id": "a", "user": 1}]


def test_every_records_block_reads_a_collection_on_its_port():
    """The executor hands a block the upstream node's output — a
    `collection` — never a bare list. Every records block must read it
    through the one reader; `records/fill` iterated the container
    itself once and read its keys as records."""
    from mechbench_compute import ops
    from mechbench_compute.lexicon import kinds as K

    design = K.collection("records/record", [
        {"id": "a", "coords": {"g": "x"}, "values": {"g": "noir"}, "v": 1.0},
    ])
    out = ops.run_standalone("records/fill", {"records": design},
                             {"templates": {"user": "a {g} story"}})
    assert out["items"] == [{"id": "a", "coords": {"g": "x"}, "user": "a noir story"}]
    for ref, params in (("records/select", {"where": {"g": "x"}}),
                        ("records/rename", {"fields": {"v": "value"}}),
                        ("records/tabulate", {}),
                        ("records/total", {"value": "v"}),
                        ("records/rank", {"value": "v", "k": 1}),
                        ("records/bin", {"value": "v", "lo": 0, "hi": 2, "bins": 2}),
                        ("records/summarize", {"value": "v", "by": ["g"]})):
        ops.run_standalone(ref, {"records": design}, params)


def test_table_from_records_flattens_coords_and_types_columns():
    from mechbench_compute.ops.records.tabulate import tabulate_records
    table = tabulate_records([
        {"id": "a", "coords": {"task": "arc_easy", "metric": "acc"},
         "value": 0.7, "delta": 0.01},
        {"id": "b", "coords": {"task": "arc_easy", "metric": "acc_norm"},
         "value": 0.68, "delta": -0.02},
    ], {"name": "deltas"})
    assert table["kind"] == "records/table"
    names = [c["name"] for c in table["columns"]]
    assert names == ["id", "task", "metric", "value", "delta"]
    dt = {c["name"]: c["dtype"] for c in table["columns"]}
    assert dt["delta"] == "number" and dt["task"] == "string"
    assert table["rows"][1]["delta"] == -0.02


def test_suite_records_flow_through_union_and_paired_delta():
    from mechbench_compute.ops.eval.benchmark import build_metric_records
    from mechbench_compute.ops.records.subtract import subtract_baseline
    from mechbench_compute.ops.records.union import union
    base = build_metric_records({"arc_easy": {"acc,none": 0.70}}, {}, "base")
    adapted = build_metric_records({"arc_easy": {"acc,none": 0.73}}, {}, "adapted")
    merged = union({"a_base": base, "b_adapted": adapted}, {})
    deltas = subtract_baseline(merged, {"match_on": ["task", "metric"],
                                    "baseline_where": {"variant": "base"},
                                    "value": "value"})
    assert len(deltas) == 1
    assert abs(deltas[0]["delta"] - 0.03) < 1e-9


def test_viz_spec_references_its_source_or_inlines_rows():
    from mechbench_compute.ops.records.plot import build_chart
    table = {"kind": "metric_table", "rows": [{"id": "a", "model": "e2b", "v": 1.0}]}
    ref = build_chart(table, {"mark": "bar", "encoding": {"x": "model", "y": "v"}},
                     source_label="benji/marcus/metrics/t")
    assert ref["kind"] == "records/chart" and ref["source"] == "benji/marcus/metrics/t"
    assert "data" not in ref
    inline = build_chart(table, {"encoding": {"x": "model", "y": "v"}})
    assert inline["data"]["rows"] == [{"id": "a", "model": "e2b", "v": 1.0}]
    recs = [{"id": "r", "coords": {"task": "arc"}, "value": 0.8}]
    flat = build_chart(recs, {"encoding": {"x": "task", "y": "value"}})
    assert flat["data"]["rows"] == [{"id": "r", "value": 0.8, "task": "arc"}]


class TestAFigureCarriesItsVocabulary:
    """What turns a chart into a visualization (VISUALIZATION.md): prose
    labels, the model's depth landmarks, callouts, and the field it
    shares with the other figures on a page."""

    ROWS = [{"id": f"L{i}", "layer": i, "mean": -float(i) / 10, "kind": "local"} for i in range(4)]
    ARCH = {"n_layers": 4, "global_layers": [1, 3], "first_kv_shared_layer": 2}

    def _spec(self, params, records=None):
        from mechbench_compute.ops.records.plot import build_chart
        return build_chart(records if records is not None else self.ROWS,
                        {"encoding": {"x": "layer", "y": "mean"}, **params})

    def test_labels_and_colour_and_focus_ride_on_the_spec(self):
        spec = self._spec({"labels": {"y": "what removing the layer costs"},
                           "encoding": {"x": "layer", "y": "mean", "color": "kind"},
                           "focus": "layer", "facet": "kind"})
        assert spec["labels"] == {"y": "what removing the layer costs"}
        assert spec["encoding"]["color"] == "kind"
        assert spec["focus"] == "layer"
        assert spec["facet"] == "kind"

    def test_a_label_for_a_field_the_figure_does_not_have_is_refused(self):
        with pytest.raises(ValueError, match="labels names"):
            self._spec({"labels": {"z": "nothing"}})

    def test_landmarks_are_read_from_the_inputs_header(self):
        # A sweep's result carries `arch`; a figure of it draws the
        # landmarks without the author naming them.
        coll = {"kind": "collection", "item_kind": "intervene/ablation",
                "items": self.ROWS, "arch": self.ARCH}
        spec = self._spec({}, records=coll)
        assert spec["axes"] == {"layer": {"n": 4, "global": [1, 3], "kv_shared_from": 2}}

    def test_given_landmarks_win_and_are_checked(self):
        spec = self._spec({"axes": {"layer": {"n": 4, "global": [3]}}})
        assert spec["axes"]["layer"] == {"n": 4, "global": [3]}
        with pytest.raises(ValueError, match="outside 0..3"):
            self._spec({"axes": {"layer": {"n": 4, "global": [9]}}})
        with pytest.raises(ValueError, match="needs `n`"):
            self._spec({"axes": {"layer": {"global": [1]}}})

    def test_a_summary_of_a_sweep_still_knows_the_model(self):
        # The landmarks ride from the sweep through every records op the
        # executor runs — the block itself knows nothing about
        # them — so a figure two nodes downstream still draws them.
        from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec
        coll = {"kind": "collection", "item_kind": "intervene/ablation",
                "items": [{**r, "coords": {"layer": r["layer"]}} for r in self.ROWS],
                "arch": self.ARCH}
        graph = {"dataflow": 2, "nodes": [
            {"id": "by-layer", "block": "records/summarize",
             "params": {"by": ["layer"], "value": "mean"}, "inputs": {"records": coll}},
            {"id": "figure", "block": "records/plot",
             "params": {"encoding": {"x": "layer", "y": "mean"}}},
        ], "edges": [{"from": {"node": "by-layer"}, "to": {"node": "figure", "port": "records"}}]}
        out = ProtocolExecutor().run(ProtocolSpec(kind="pipeline", prompt="", model_id=None,
                                                  extra={"graph": graph}))
        payload = out.payload if hasattr(out, "payload") else out
        # Only the leaf is an output; that the summary carried the
        # landmarks shows in the figure it fed, which names them without
        # having been told.
        figure = payload["outputs"]["figure"]
        assert figure["axes"] == {"layer": {"n": 4, "global": [1, 3], "kv_shared_from": 2}}
        # And the figure itself carries them on for anything downstream.
        assert figure["arch"] == self.ARCH

    def test_every_model_block_result_carries_the_landmarks(self, monkeypatch):
        # The executor's model-block wrapper stamps `arch` from the model
        # it ran, so no block has to know the landmarks exist.
        from types import SimpleNamespace

        from mechbench_compute.protocol import ProtocolExecutor
        ex = ProtocolExecutor()
        arch = SimpleNamespace(n_layers=4, global_layers=(1, 3), first_kv_shared_layer=2)
        monkeypatch.setattr(ex, "_model_loaded", lambda _m: SimpleNamespace(arch=arch))
        monkeypatch.setattr(ex, "_adapter_fused", lambda *a, **k: __import__("contextlib").nullcontext())
        out = ex._run_model_block(lambda inputs, params: {"kind": "collection", "items": []}, {}, {"model": "fake/m"})
        assert out["arch"] == self.ARCH
        # A block that already said something about the model keeps it.
        kept = ex._run_model_block(lambda i, p: {"arch": {"n_layers": 99}}, {}, {"model": "fake/m"})
        assert kept["arch"] == {"n_layers": 99}
        # And a model family without landmarks leaves them out, rather
        # than inventing an empty list.
        plain = SimpleNamespace(n_layers=12)
        monkeypatch.setattr(ex, "_model_loaded", lambda _m: SimpleNamespace(arch=plain))
        assert ex._run_model_block(lambda i, p: {}, {}, {"model": "fake/m"})["arch"] == {"n_layers": 12}

    def test_a_map_keeps_the_landmarks_its_body_found(self, monkeypatch):
        # A map's records are its sweep — forty-two layer numbers know
        # nothing about any model — so the landmarks cannot come from the
        # input the way they do for an ordinary records op. They are on
        # the body's header, and only its items were being kept.
        from types import SimpleNamespace

        from mechbench_compute.lexicon import kinds as K

        body_out = K.collection("records/record", [{"id": "r", "silhouette": 0.3}],
                                arch=dict(self.ARCH))

        class Child:
            _model = None
            _model_id = None
            _on_download = _on_download_bytes = _limiter = _budget = None

            def __init__(self, *a, **k):
                pass

            def run(self, spec, **kw):
                return SimpleNamespace(payload={"outputs": {"separation": body_out}})

        from mechbench_compute import ops
        from mechbench_compute.ops.records import map as map_op

        out = map_op.run(
            ops.Context(executor=Child()),
            {"records": K.collection("records/record", [{"id": "l0", "layer": 0}])},
            {"body": {"nodes": [{"id": "separation", "block": "records/select"}], "edges": []},
             "bind": {"layer": "layer"}},
        )
        assert out["arch"] == self.ARCH
        assert [i["id"] for i in out["items"]] == ["l0:r"]

    def test_annotations_name_a_row_and_say_something(self):
        spec = self._spec({"annotate": [{"at": {"layer": 3}, "text": "the last global layer"}]})
        assert spec["annotate"] == [{"at": {"layer": 3}, "text": "the last global layer"}]
        with pytest.raises(ValueError, match=r"annotate\[0\] needs"):
            self._spec({"annotate": [{"text": "where?"}]})


def test_uniform_masses_derive_from_top_tokens():
    """Plain decision reads emit top_tokens and no outcome_mass, so
    the judge derives the masses rather than failing a distribution it
    never looked at."""
    results = [{"id": "c1", "entropy_bits": 1.99, "top_tokens": [
        {"token": "3", "p": 0.2997}, {"token": "1", "p": 0.2334},
        {"token": "2", "p": 0.2334}, {"token": "4", "p": 0.2334},
        {"token": "0", "p": 0.0001}]}]
    expectations = [{"id": "c1", "expect": {
        "kind": "uniform", "over": ["1", "2", "3", "4"], "max_kl_bits": 0.1}}]
    table = check_expectations(
        {"results": results, "expectations": expectations}, {})
    row = table["items"][0]
    assert row["pass"] is True
    assert row["kl_bits"] < 0.02
    assert row["mass"] > 0.99


def test_uniform_without_any_distribution_is_unjudgeable_not_false():
    results = [{"id": "c1", "entropy_bits": 0.5}]
    expectations = [{"id": "c1", "expect": {
        "kind": "uniform", "over": ["1", "2"], "max_kl_bits": 0.1}}]
    table = check_expectations(
        {"results": results, "expectations": expectations}, {})
    row = table["items"][0]
    assert row["pass"] is None and "unjudgeable" in row["note"]
    # ...and the summary does not count it as a judged failure.
    assert table["summary"]["n_judged"] == 0 and table["summary"]["n_unjudgeable"] == 1



#: Two judged rows and one the judge could not be read for.
JUDGED_ROWS = [
    {"id": "a", "coords": {"arm": "x"}, "score": 4.0},
    {"id": "b", "coords": {"arm": "x"}},              # unparsed
    {"id": "c", "coords": {"arm": "y"}, "score": 2.0},
]


class TestGroupStatsMissingValues:
    """A judged corpus has rows a judge could not be read for. A mean
    over the records that happened to have the field is
    the kind of number nobody notices is wrong, so the default refuses
    by name and `skip` reports what it dropped."""

    def test_a_missing_value_refuses_by_name_by_default(self):
        import pytest

        from mechbench_compute.ops.records.summarize import group_stats

        with pytest.raises(ValueError, match="record 'b' has no 'score'"):
            group_stats(JUDGED_ROWS, {"by": ["arm"], "value": "score"})

    def test_skip_omits_them_and_reports_the_count(self):
        from mechbench_compute.ops.records.summarize import group_stats

        out = group_stats(JUDGED_ROWS, {"by": ["arm"], "value": "score",
                                          "on_missing": "skip"})
        assert out["n_missing"] == 1
        assert {r["arm"]: r["n"] for r in out["rows"]} == {"x": 1, "y": 1}

    def test_a_clean_table_says_nothing_about_missing(self):
        from mechbench_compute.ops.records.summarize import group_stats

        out = group_stats(JUDGED_ROWS[:1], {"by": ["arm"], "value": "score"})
        assert "n_missing" not in out


class TestAFigureIsReadAgainstALine:
    """A figure whose claim is "close to fair" or "over the threshold"
    needs the line to be read against (VISUALIZATION.md)."""

    ROWS = {"kind": "collection", "items": [
        {"id": "1", "face": "1", "p": 0.1808}, {"id": "2", "face": "2", "p": 0.1595},
    ]}

    def test_a_reference_rides_on_the_spec(self):
        from mechbench_compute.ops.records.plot import build_chart

        spec = build_chart(self.ROWS, {
            "mark": "bar", "encoding": {"x": "face", "y": "p"},
            "reference": [{"y": 1 / 6, "text": "a fair die"}],
        })
        assert spec["reference"] == [{"y": pytest.approx(0.16667, abs=1e-4), "text": "a fair die"}]

    def test_a_line_on_neither_axis_is_refused(self):
        from mechbench_compute.ops.records.plot import build_chart

        with pytest.raises(ValueError, match=r"reference\[0\] needs `y`"):
            build_chart(self.ROWS, {"mark": "bar", "encoding": {"x": "face", "y": "p"},
                                    "reference": [{"text": "a fair die"}]})

    def test_a_figure_without_one_says_nothing_about_it(self):
        from mechbench_compute.ops.records.plot import build_chart

        spec = build_chart(self.ROWS, {"mark": "bar", "encoding": {"x": "face", "y": "p"}})
        assert "reference" not in spec


class TestASummaryOverAGrid:
    """A trace is its cells: `records/summarize` reads a grid
    cell by cell, the same rows `records/plot` draws, so a strip of what
    each token's best cell recovers is one summarize away — in the flat
    block and in its monoid alike."""

    TRACE = {"kind": "collection", "item_kind": "intervene/trace", "items": [
        {"id": "p1", "axes": ["layer", "position"], "tokens": ["The", "capital"],
         "coords": {"country": "France"},
         "measures": {"share": [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]}},
        {"id": "p2", "axes": ["layer", "position"], "tokens": ["The", "capital"],
         "coords": {"country": "Italy"},
         "measures": {"share": [[0.8, 0.0], [0.7, 0.2], [0.0, 0.6]]}},
    ]}

    def test_one_row_per_position_with_the_best_cell_as_max(self):
        from mechbench_compute.ops.records.summarize import group_stats

        out = group_stats(self.TRACE, {"by": ["position"], "value": "share"})
        by_pos = {r["position"]: r for r in out["rows"]}
        assert by_pos[0]["max"] == 1.0 and by_pos[1]["max"] == 1.0
        assert by_pos[0]["n"] == 6

    def test_the_monoid_reads_the_same_cells(self):
        from mechbench_compute import reduce as rd
        from mechbench_compute.ops.records.summarize import group_stats

        params = {"by": ["country", "position"], "value": "share"}
        flat = group_stats(self.TRACE, params)
        chunked = rd.reduce_chunks("records/summarize", [[self.TRACE["items"][0]], [self.TRACE["items"][1]]], params)
        assert chunked["rows"] == flat["rows"]


A_CORPUS = {"kind": "document_collection", "items": [
    {"id": "a", "text": "one", "metadata": {"coords": {"p": "x"}}}]}


class TestRecordCoercion:
    """A document collection IS a record stream — the thing generate,
    chat and conversation emit. Blocks that wanted to read a corpus
    were each writing their own coercion before this."""

    def test_items_are_records(self):
        from mechbench_compute.blocks import read_items

        assert read_items(A_CORPUS) == A_CORPUS["items"]

    def test_the_older_conventions_still_win_first(self):
        from mechbench_compute.blocks import read_items

        both = {"records": [{"id": "r"}], "items": [{"id": "i"}]}
        assert read_items(both) == [{"id": "r"}]

    def test_a_document_can_be_embedded_by_its_text(self):
        from mechbench_compute.distill import render

        class Tok:
            def encode(self, text, add_special_tokens=True):
                return [len(w) for w in text.split()]

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
                return "<chat>" + messages[-1]["content"]

        class M:
            tokenizer = Tok()

        # A document renders raw; a condition through the chat template;
        # a condition may turn the template off; a raw record may turn it on.
        assert render(M(), {"id": "a", "text": "a story"}).text == "a story"
        r = render(M(), {"id": "a", "user": "u", "text": "t", "prefill": "{"})
        assert r.text == "<chat>u{" and r.chat
        assert render(M(), {"id": "a", "user": "u", "template": False}).text == "u"
        assert render(M(), {"id": "a", "text": "t", "template": "chat"}).text == "<chat>t"
        with pytest.raises(ValueError, match="no prompt"):
            render(M(), {"id": "a", "text": "   "})


def test_eval_expectation_absent_judges_the_mass_on_outcomes_already_said():
    import math

    def entry(p):
        return {"text": "x", "tokens": 2, "p": p, "logp": math.log(p)}
    reads = [
        {"id": "clean", "entropy_bits": 5.0,
         "tracked": {"Mystery": entry(0.004), "Humor": entry(0.002), "Witches": entry(0.2)}},
        {"id": "repeats", "entropy_bits": 5.0,
         "tracked": {"Mystery": entry(0.05), "Humor": entry(0.01)}},
        {"id": "unread", "entropy_bits": 5.0, "tracked": {"Mystery": entry(0.001)}},
    ]
    already = {"type": "absent", "over": ["Mystery", "Humor"], "max_p": 0.01}
    out = check_expectations({"results": reads, "expectations": [
        {"id": "clean", "expect": already},
        {"id": "repeats", "expect": already},
        {"id": "unread", "expect": already}]}, {})
    by_id = {r["id"]: r for r in out["items"]}
    assert by_id["clean"]["pass"] is True and by_id["clean"]["mass"] == 0.006
    assert by_id["repeats"]["pass"] is False and by_id["repeats"]["mass"] == 0.06
    # "Humor" was never scored: no verdict, rather than a false pass.
    assert by_id["unread"]["pass"] is None and "Humor" in by_id["unread"]["note"]
    assert out["summary"]["n_judged"] == 2 and out["summary"]["n_unjudgeable"] == 1
