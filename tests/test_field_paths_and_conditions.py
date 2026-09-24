"""Fields read by dot path, conditions in the API's item-query grammar, a
field subtracted from another of the same record, and the words a corpus
uses counted by how many texts use them: what an analysis over stored
generation results reads, without a script that unwraps it first."""

from __future__ import annotations

import pytest

from mechbench_compute import isomorphism as iso
from mechbench_compute import reduce as rd
from mechbench_compute.blocks.match_where import match_where, parse_where
from mechbench_compute.blocks.read_field import read_field
from mechbench_compute.blocks.read_group_key import read_group_key
from mechbench_compute.ops.records.correlate import correlate
from mechbench_compute.ops.records.count import count
from mechbench_compute.ops.records.select import select
from mechbench_compute.ops.records.subtract import subtract_baseline
from mechbench_compute.ops.records.summarize import group_stats
from mechbench_compute.ops.text.measure import measure_texts


def _reply(i: int, prompt: str, output: int, reasoning: int | None, stop: str | None = None) -> dict:
    """A provider reply shaped as a stored `text/generate` item: the prompt
    only in `metadata.coords`, the counts nested in `metadata.call`."""
    usage = {"input_tokens": 22, "output_tokens": output}
    if reasoning is not None:
        usage["reasoning_tokens"] = reasoning
    call = {"model": "m", "usage": usage}
    if stop is not None:
        call["stop_reason"] = stop
    return {"id": f"{prompt}-s{i}", "text": f"story {i}",
            "metadata": {"call": call, "coords": {"prompt": prompt, "sample": i},
                         "sampling": {"max_tokens": 250}}}


REPLIES = [_reply(0, "flash", 250, None), _reply(1, "flash", 300, 120, "end_turn"),
           _reply(2, "flash", 90, 0), _reply(3, "neutral", 120, None),
           _reply(4, "neutral", 260, 30, "max_tokens")]


class TestReadField:
    def test_a_dot_path_reads_from_the_root(self):
        r = REPLIES[1]
        assert read_field(r, "metadata.call.usage.output_tokens") == 300
        assert read_field(r, "metadata.coords.prompt") == "flash"
        assert read_field(r, "metadata.call.usage.reasoning_tokens") == 120

    def test_a_missing_path_reads_none(self):
        assert read_field(REPLIES[0], "metadata.call.usage.reasoning_tokens") is None
        assert read_field(REPLIES[0], "metadata.call.stop_reason.word") is None
        assert read_field(REPLIES[0], "nothing") is None

    def test_a_coordinate_and_a_top_level_key_are_read_first(self):
        r = {"id": "x", "coords": {"a.b": 1, "prompt": "p"}, "a.b": 2, "a": {"b": 3}, "prompt": "q"}
        assert read_field(r, "a.b") == 1
        assert read_field({"id": "x", "a.b": 2, "a": {"b": 3}}, "a.b") == 2
        assert read_field(r, "prompt") == "p"
        assert read_field(r, "coords.prompt") == "p"

    def test_a_list_element_by_index(self):
        assert read_field({"votes": [{"w": "A"}, {"w": "B"}]}, "votes.1.w") == "B"
        assert read_field({"votes": [{"w": "A"}]}, "votes.3.w") is None

    def test_a_group_key_reads_a_path(self):
        assert read_group_key(REPLIES[3], ["metadata.coords.prompt"]) == ("neutral",)


class TestWhere:
    def _keep(self, *where):
        return [r["id"] for r in REPLIES if match_where(r, parse_where(list(where)))]

    def test_ordering_between_numbers(self):
        assert self._keep("metadata.call.usage.output_tokens>250") == ["flash-s1", "neutral-s4"]
        assert self._keep("metadata.call.usage.output_tokens>=250") == ["flash-s0", "flash-s1", "neutral-s4"]
        assert self._keep("metadata.call.usage.output_tokens<100") == ["flash-s2"]

    def test_a_missing_path_is_null(self):
        assert self._keep("metadata.call.usage.reasoning_tokens=null") == ["flash-s0", "neutral-s3"]
        assert self._keep("metadata.call.usage.reasoning_tokens!=null") == ["flash-s1", "flash-s2", "neutral-s4"]
        # Ordering never holds against null: no count is not a count of zero.
        assert self._keep("metadata.call.usage.reasoning_tokens>0") == ["flash-s1", "neutral-s4"]

    def test_every_condition_must_hold(self):
        assert self._keep("metadata.coords.prompt=flash", "metadata.call.usage.output_tokens>100") == ["flash-s0", "flash-s1"]

    def test_equality_reads_text_and_json(self):
        assert self._keep("metadata.call.stop_reason=max_tokens") == ["neutral-s4"]
        assert self._keep("metadata.coords.sample=3") == ["neutral-s3"]
        assert self._keep('metadata.coords.sample="3"') == []
        assert match_where({"id": "a", "ok": True}, parse_where(["ok=true"]))
        assert not match_where({"id": "a", "ok": 1}, parse_where(["ok=true"]))

    def test_contains(self):
        assert match_where({"text": "The Lighthouse"}, parse_where(["text~lighthouse"]))
        assert match_where({"tags": ["a", "b"]}, parse_where(["tags~b"]))
        assert not match_where({"n": 5}, parse_where(["n~5"]))

    def test_ordering_between_a_number_and_text_does_not_hold(self):
        assert not match_where({"n": 5}, parse_where(["n>abc"]))
        assert match_where({"s": "b"}, parse_where(["s>a"]))

    def test_a_malformed_condition_is_refused(self):
        with pytest.raises(ValueError, match="PATH OP VALUE"):
            parse_where(["no operator here"])


class TestSelect:
    def test_conditions_select(self):
        out = select(REPLIES, {"where": ["metadata.call.usage.output_tokens>250"]})
        assert [r["id"] for r in out] == ["flash-s1", "neutral-s4"]

    def test_a_map_reads_a_path_too(self):
        out = select(REPLIES, {"where": {"metadata.coords.prompt": "neutral"}})
        assert [r["id"] for r in out] == ["neutral-s3", "neutral-s4"]

    def test_a_map_keeps_its_list_of_values(self):
        recs = [{"id": i, "coords": {"g": g}} for i, g in enumerate("abc")]
        assert [r["id"] for r in select(recs, {"where": {"g": ["a", "c"]}})] == [0, 2]

    def test_fields_project_paths_under_their_names(self):
        out = select(REPLIES[:1], {"fields": ["metadata.call.usage.output_tokens"]})
        assert out == [{"id": "flash-s0", "coords": {}, "metadata.call.usage.output_tokens": 250}]


class TestCountWhere:
    def test_k_is_the_records_meeting_every_condition(self):
        out = count(REPLIES, {"where": ["metadata.call.usage.output_tokens>250"],
                              "by": ["metadata.coords.prompt"]})
        rows = {r["metadata.coords.prompt"]: (r["k"], r["n"]) for r in out["rows"]}
        assert rows == {"flash": (1, 3), "neutral": (1, 2)}
        assert out["counted"] == {"where": ["metadata.call.usage.output_tokens>250"]}

    def test_a_path_as_the_field(self):
        out = count(REPLIES, {"field": "metadata.call.stop_reason", "equals": "max_tokens",
                              "on_missing": "skip"})
        assert (out["rows"][0]["k"], out["rows"][0]["n"], out["n_missing"]) == (1, 2, 3)

    def test_one_of_field_and_where(self):
        with pytest.raises(ValueError, match="one of"):
            count(REPLIES, {"field": "x", "where": ["x=1"]})
        with pytest.raises(ValueError, match="one of"):
            count(REPLIES, {})

    def test_chunked_equals_flat(self):
        params = {"where": ["metadata.call.usage.output_tokens>=250"], "by": ["metadata.coords.prompt"]}
        flat = count(REPLIES, params)
        assert rd.reduce_chunks("records/count", [REPLIES[:2], REPLIES[2:]], params) == flat
        assert iso.check("records/count", REPLIES, params, trials=10, seed=3)["exact"] is True


class TestSubtractMinus:
    def test_a_field_less_another_of_the_same_record(self):
        reasoned = select(REPLIES, {"where": ["metadata.call.usage.reasoning_tokens!=null"]})
        out = subtract_baseline(reasoned, {"value": "metadata.call.usage.output_tokens",
                                           "minus": "metadata.call.usage.reasoning_tokens"})
        assert [(r["id"], r["delta"]) for r in out] == [
            ("flash-s1", 180.0), ("flash-s2", 90.0), ("neutral-s4", 230.0)]

    def test_a_record_without_the_field_is_refused(self):
        with pytest.raises(ValueError, match="reasoning_tokens"):
            subtract_baseline(REPLIES, {"value": "metadata.call.usage.output_tokens",
                                        "minus": "metadata.call.usage.reasoning_tokens"})

    def test_one_of_baseline_where_and_minus(self):
        with pytest.raises(ValueError, match="one of"):
            subtract_baseline(REPLIES, {"value": "x"})


class TestOtherOpsReadPaths:
    def test_summarize_a_path_by_a_path(self):
        out = group_stats(REPLIES, {"value": "metadata.call.usage.output_tokens",
                                    "by": ["metadata.coords.prompt"]})
        rows = {r["metadata.coords.prompt"]: (r["n"], r["max"]) for r in out["rows"]}
        assert rows == {"flash": (3, 300.0), "neutral": (2, 260.0)}

    def test_correlate_paths(self):
        out = correlate(REPLIES, {"x": "metadata.coords.sample", "y": "metadata.call.usage.output_tokens"})
        assert out["rows"][0]["n"] == 5


TEXTS = ({"id": "a", "text": "The last light. The keeper's last climb."},
         {"id": "b", "text": "The light, and a lamp-lit stair."},
         {"id": "c", "text": "Nothing here but the sea."})


class TestWordsUsed:

    def _rows(self, **measure):
        rows = measure_texts({"records": list(TEXTS)},
                             {"mode": "items", "measures": [{"type": "lexical", "name": "w", **measure}]})
        return {r["item"]: (r["count"], r["texts"]) for r in rows}

    def test_texts_is_how_many_texts_use_the_word(self):
        rows = self._rows()
        assert rows["the"] == (4, 3)
        assert rows["last"] == (2, 1)
        assert rows["light"] == (2, 2)
        assert rows["keeper's"] == (1, 1)
        assert rows["lamp-lit"] == (1, 1)

    def test_rows_lead_with_the_most_used(self):
        rows = measure_texts({"records": list(TEXTS)},
                             {"mode": "items", "measures": [{"type": "lexical", "name": "w"}]})
        assert [r["item"] for r in rows[:2]] == ["the", "light"]
        assert rows[0]["id"] == "w:the" and rows[0]["coords"] == {"measure": "w", "item": "the"}

    def test_exclude_leaves_words_out_everywhere(self):
        assert "the" not in self._rows(exclude=["The", "a"])
        annotated = measure_texts({"records": list(TEXTS)},
                                  {"measures": [{"type": "lexical", "name": "w", "exclude": ["the"]}]})
        assert annotated[0]["w_words"] == 5 and annotated[0]["w_distinct"] == 4

    def test_a_stored_word_list_excludes(self):
        assert "the" not in self._rows(exclude={"words": ["the"]})

    def test_items_mode_without_a_list_or_lexical_measure_is_refused(self):
        with pytest.raises(ValueError, match="`list` or `lexical`"):
            measure_texts({"records": list(TEXTS)},
                          {"mode": "items", "measures": [{"type": "pattern", "patterns": ["x"]}]})
