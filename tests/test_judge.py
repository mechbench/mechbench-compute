"""`~canonical/ops/judge/1` (task 000356): scales and parsing, votes
and spread, position randomization, and the block end to end on the
mock provider.

The aggregation is tested directly on synthetic votes — a judge that
disagrees with itself is the interesting case, and the mock cannot
disagree with itself on purpose.
"""

from __future__ import annotations

import pytest

from mechbench_compute import judge as J
from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec

STORIES = [
    {"id": "s1", "coords": {"arm": "base"}, "text": "The dust settled slowly."},
    {"id": "s2", "coords": {"arm": "tuned"}, "text": "A kettle sang in the dark."},
]


def judged(text, **params):
    """Run the block with the mock answering `text` every time."""
    base = {
        "judge": {"model": {"provider": "mock", "model": "judge-1"},
                  "system": "Grade the story for cliché.",
                  "provider_options": {"mock": {"text": text}}},
        "scale": {"kind": "numeric", "min": 1, "max": 5},
        "budget_usd": 1.0,
        "records": STORIES,
    }
    base.update(params)
    return J.run(base)


class TestScales:
    def test_a_numeric_scale_reads_json_or_a_bare_number(self):
        scale = J.Scale({"kind": "numeric", "min": 1, "max": 5})
        assert scale.read('{"score": 4, "rationale": "fresh"}') == {
            "score": 4.0, "rationale": "fresh"}
        loose = scale.read("I would say 2 out of 5.")
        assert loose["score"] == 2.0 and loose["rationale"]

    def test_out_of_range_is_clamped_and_flagged(self):
        scale = J.Scale({"kind": "numeric", "min": 1, "max": 5})
        out = scale.read('{"score": 9}')
        # A judge that answers 9 on a 1–5 scale did not mean 5, and
        # hiding that would hide a broken rubric.
        assert out["score"] == 5.0 and out["out_of_range"] == 9.0

    def test_a_categorical_scale_takes_a_label_by_name(self):
        scale = J.Scale({"kind": "categorical", "labels": ["cliché", "fresh"]})
        assert scale.read('{"label": "fresh"}')["label"] == "fresh"
        assert scale.read("This one reads as fresh to me.")["label"] == "fresh"
        assert scale.read("no idea") == {}

    def test_a_pairwise_scale_takes_a_side(self):
        scale = J.Scale({"kind": "pairwise"})
        assert scale.read('{"winner": "B", "rationale": "richer"}')["winner"] == "B"
        assert scale.read("I prefer A.")["winner"] == "A"

    def test_unreadable_is_unparsed_not_zero(self):
        assert J.Scale({"kind": "numeric"}).read("hmm, hard to say") == {}

    def test_a_malformed_scale_is_refused(self):
        with pytest.raises(ValueError, match="unknown scale"):
            J.Scale({"kind": "vibes"})
        with pytest.raises(ValueError, match="max > min"):
            J.Scale({"kind": "numeric", "min": 5, "max": 5})
        with pytest.raises(ValueError, match="two labels"):
            J.Scale({"kind": "categorical", "labels": ["only"]})


class TestVotesAndSpread:
    def test_numeric_votes_average_and_keep_the_spread(self):
        votes = [{"parsed": True, "score": 2.0, "rationale": "a"},
                 {"parsed": True, "score": 5.0, "rationale": "b"},
                 {"parsed": True, "score": 3.0, "rationale": "c"}]
        row = J.aggregate({"id": "s1", "coords": {}}, votes,
                          scale=J.Scale({"kind": "numeric"}))
        assert row["score"] == pytest.approx(3.3333, abs=1e-4)
        assert row["spread"] > 1.0 and row["min"] == 2.0 and row["max"] == 5.0
        assert row["n_parsed"] == 3

    def test_categorical_votes_take_the_majority_and_report_agreement(self):
        scale = J.Scale({"kind": "categorical", "labels": ["cliché", "fresh"]})
        votes = [{"parsed": True, "label": "fresh"},
                 {"parsed": True, "label": "fresh"},
                 {"parsed": True, "label": "cliché"}]
        row = J.aggregate({"id": "s1"}, votes, scale=scale)
        assert row["label"] == "fresh"
        # A rubric that produces 0.67 here is the finding, not a number
        # to average away.
        assert row["agreement"] == pytest.approx(0.6667, abs=1e-4)
        assert row["counts"] == {"fresh": 2, "cliché": 1}

    def test_a_subject_nobody_could_grade_says_so(self):
        row = J.aggregate({"id": "s1"}, [{"parsed": False}, {"parsed": False}],
                          scale=J.Scale({"kind": "numeric"}))
        assert row["unparsed"] is True and "score" not in row

    def test_the_summary_carries_what_a_reader_would_cite(self):
        scale = J.Scale({"kind": "numeric"})
        rows = [{"score": 4.0}, {"score": 2.0}, {"unparsed": True}]
        summary = J.summarize(rows, scale=scale, votes=[])
        assert summary["mean"] == 3.0 and summary["n_unparsed"] == 1
        assert summary["n_subjects"] == 3


class TestPairwisePosition:
    def test_the_order_flips_per_vote_and_is_recorded(self):
        prompts = J.build_prompts(
            [{"id": "s1", "text_a": "AAA", "text_b": "BBB"}],
            scale=J.Scale({"kind": "pairwise"}), rubric="pick one",
            fields=["text"], n_votes=8, seed=11,
            pairwise_fields=["text_a", "text_b"])
        orders = [p["order"] for p in prompts]
        assert set(orders) == {"AB", "BA"}      # both sides get to go first
        ab = next(p for p in prompts if p["order"] == "AB")
        ba = next(p for p in prompts if p["order"] == "BA")
        assert ab["user"].startswith("A:\nAAA")
        assert ba["user"].startswith("A:\nBBB")   # B shown in slot A

    def test_the_same_seed_gives_the_same_orders(self):
        args = {"scale": J.Scale({"kind": "pairwise"}), "rubric": "r",
                "fields": ["text"], "n_votes": 6, "seed": 3,
                "pairwise_fields": ["text_a", "text_b"]}
        one = J.build_prompts([{"id": "s1"}], **args)
        two = J.build_prompts([{"id": "s1"}], **args)
        assert [p["order"] for p in one] == [p["order"] for p in two]

    def test_position_bias_is_reported_as_a_rate(self):
        # Every vote picked whatever was shown first.
        votes = [{"parsed": True, "winner": "A", "order": "AB"},
                 {"parsed": True, "winner": "B", "order": "BA"},
                 {"parsed": True, "winner": "A", "order": "AB"}]
        summary = J.summarize([{"winner": "A", "agreement": 1.0}],
                              scale=J.Scale({"kind": "pairwise"}), votes=votes)
        assert summary["first_shown_win_rate"] == 1.0


class TestTheBlock:
    def test_it_grades_a_corpus_and_records_what_it_cost(self):
        out = judged('{"score": 4, "rationale": "unhurried"}', n_votes=3)
        assert [r["id"] for r in out["records"]] == ["s1", "s2"]
        row = out["records"][0]
        assert row["score"] == 4.0 and row["n_votes"] == 3
        assert row["coords"]["arm"] == "base"      # coords survive judging
        assert row["rationale"] == "unhurried"
        assert out["summary"]["mean"] == 4.0
        # Two subjects, three votes each, all metered.
        assert out["spend"]["calls"] == 6
        assert out["judge"]["scale"] == "numeric"

    def test_the_output_is_a_record_set_downstream_blocks_can_read(self):
        from mechbench_compute.blocks import PURE_BLOCKS

        out = judged('{"score": 3}')
        stats = PURE_BLOCKS["~canonical/ops/group-stats/1"](
            {"records": out}, {"by": ["arm"], "value": "score"})
        assert stats["kind"] == "metric_table"
        assert {r["arm"] for r in stats["rows"]} == {"base", "tuned"}

    def test_an_unreadable_judge_does_not_become_a_score(self):
        out = judged("I would rather not say.")
        assert all(r.get("unparsed") for r in out["records"])
        assert out["summary"]["n_unparsed"] == 2
        assert "mean" not in out["summary"]

    def test_a_judge_without_a_rubric_or_a_model_is_refused(self):
        with pytest.raises(ValueError, match="needs a rubric"):
            J.run({"judge": {"model": {"provider": "mock", "model": "m"}},
                   "budget_usd": 1.0, "records": STORIES})
        with pytest.raises(ValueError, match="judge: \\{model, system\\}"):
            J.run({"scale": {"kind": "numeric"}, "records": STORIES})

    def test_a_remote_judge_needs_a_cap(self):
        with pytest.raises(ValueError, match="budget_usd"):
            J.run({"judge": {"model": {"provider": "mock", "model": "m"},
                             "system": "grade it"},
                   "records": STORIES})


class TestThroughTheExecutor:
    def test_a_judge_node_runs_end_to_end(self):
        graph = {"nodes": [{
            "id": "grade", "block": "~canonical/ops/judge/1",
            "params": {
                "judge": {"model": {"provider": "mock", "model": "judge-1"},
                          "system": "Grade the story for cliché.",
                          "provider_options": {"mock": {"text": '{"score": 5}'}}},
                "scale": {"kind": "numeric", "min": 1, "max": 5},
                "n_votes": 2,
                "budget_usd": 1.0,
                "records": STORIES,
            }}], "edges": []}
        out = ProtocolExecutor().run(ProtocolSpec(
            kind="pipeline", prompt="", model_id=None, extra={"graph": graph}))
        node = out.payload["outputs"]["grade"]
        assert node["summary"]["mean"] == 5.0
        assert out.payload["resources"]["spend"]["by_node"]["grade"]["calls"] == 4

    def test_a_local_judge_runs_through_the_same_path(self, monkeypatch):
        from mechbench_compute import distill, generate

        monkeypatch.setattr(distill, "encode", lambda tok, text: [1, 2, 3])
        monkeypatch.setattr(distill, "prefill_decision", lambda m, ids: None)
        monkeypatch.setattr(generate, "sample_completion_cached",
                            lambda *a, **k: '{"score": 2, "rationale": "flat"}')

        class FakeTok:
            def apply_chat_template(self, turns, **kw):
                return " | ".join(t["content"] for t in turns)

        class FakeModel:
            tokenizer = FakeTok()

        ex = ProtocolExecutor()
        monkeypatch.setattr(ex, "_model_loaded", lambda *_a, **_k: FakeModel())
        out = ex.run(ProtocolSpec(kind="pipeline", prompt="", model_id=None, extra={
            "graph": {"nodes": [{
                "id": "grade", "block": "~canonical/ops/judge/1",
                "params": {
                    "judge": {"model": "google/gemma-3-4b-it",
                              "system": "Grade the story for cliché."},
                    "scale": {"kind": "numeric", "min": 1, "max": 5},
                    "records": STORIES,
                }}], "edges": []}}))
        node = out.payload["outputs"]["grade"]
        assert node["summary"]["mean"] == 2.0
        # A local judge spends nothing, so there is no bill to report.
        assert "spend" not in node

    def test_the_resume_level_follows_the_judges_model(self):
        from mechbench_compute import resume as resume_mod

        assert resume_mod.resume_level("~canonical/ops/judge/1", {
            "judge": {"model": {"provider": "anthropic", "model": "x"}}}) == "exchangeable"
        assert resume_mod.resume_level("~canonical/ops/judge/1", {
            "judge": {"model": "google/gemma-3-4b-it"}}) == "reproducible"
