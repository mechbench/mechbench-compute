from __future__ import annotations

import pytest

from mechbench_compute.blocks.read_field import read_field
from mechbench_compute.lexicon import kinds as K
from mechbench_compute.ops.records.derive import derive
from mechbench_compute.ops.records.filter import filter_records
from mechbench_compute.ops.records.group import group_records
from mechbench_compute.ops.text.measure import measure_texts


def _reply(i: int, prompt: str, output: int, reasoning: int | None, stop: str | None = None) -> dict:
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


def _keep(where: str) -> list[str]:
    return [r["id"] for r in filter_records(K.collection("records/record", REPLIES), {"where": where}, {})["items"]]


class TestExpressionsReadPaths:
    def test_ordering_between_numbers(self):
        assert _keep("metadata.call.usage.output_tokens > 250") == ["flash-s1", "neutral-s4"]
        assert _keep("metadata.call.usage.output_tokens >= 250") == ["flash-s0", "flash-s1", "neutral-s4"]
        assert _keep("metadata.call.usage.output_tokens < 100") == ["flash-s2"]

    def test_a_missing_path_is_null(self):
        assert _keep("metadata.call.usage.reasoning_tokens is None") == ["flash-s0", "neutral-s3"]
        assert _keep("metadata.call.usage.reasoning_tokens is not None") == ["flash-s1", "flash-s2", "neutral-s4"]
        assert _keep("metadata.call.usage.reasoning_tokens > 0") == ["flash-s1", "neutral-s4"]

    def test_every_condition_must_hold(self):
        assert _keep('metadata.coords.prompt == "flash" and metadata.call.usage.output_tokens > 100') == [
            "flash-s0", "flash-s1"]

    def test_equality_is_typed(self):
        assert _keep('metadata.call.stop_reason == "max_tokens"') == ["neutral-s4"]
        assert _keep("metadata.coords.sample == 3") == ["neutral-s3"]
        assert _keep('metadata.coords.sample == "3"') == []

    def test_a_field_less_another_of_the_same_record(self):
        reasoned = filter_records(REPLIES, {"where": "metadata.call.usage.reasoning_tokens is not None"}, {})
        out = derive(reasoned, {"fields": {"delta": "metadata.call.usage.output_tokens - metadata.call.usage.reasoning_tokens"}}, {})
        assert [(r["id"], r["delta"]) for r in out["items"]] == [("flash-s1", 180), ("flash-s2", 90), ("neutral-s4", 230)]

    def test_group_a_path_by_a_path(self):
        out = group_records(REPLIES, {"by": {"prompt": "metadata.coords.prompt"}, "aggregates": {
            "n": "count()", "max": "max(metadata.call.usage.output_tokens)",
            "k": "count(metadata.call.usage.output_tokens > 250)"}}, {})
        rows = {r["prompt"]: (r["n"], r["max"], r["k"]) for r in out["items"]}
        assert rows == {"flash": (3, 300, 1), "neutral": (2, 260, 1)}

    def test_spearman_over_paths(self):
        out = group_records(REPLIES, {"aggregates": {
            "n, rho": "spearman(metadata.coords.sample, metadata.call.usage.output_tokens)"}}, {})
        assert out["items"][0]["n"] == 5


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

    def test_word_says_what_a_word_is(self):
        annotated = measure_texts({"records": list(TEXTS)}, {"measures": [
            {"type": "lexical", "name": "ws", "word": r"\S+"},
            {"type": "lexical", "name": "az", "word": r"[A-Za-z']+"}]})
        assert annotated[1]["ws_words"] == 6
        assert annotated[1]["az_words"] == 7
        assert "lamp-lit" not in self._rows(word=r"[A-Za-z']+")
        assert self._rows(word=r"[A-Za-z']+")["lamp"] == (1, 1)


class TestOpening:
    STORIES = ({"id": "a", "text": "  The old man, who kept the light, slept."},
               {"id": "b", "text": "the Old man - who knew"},
               {"id": "c", "text": "Rain."},
               {"id": "d", "text": ""})

    def test_an_opening_is_the_first_words_lowercased(self):
        rows = measure_texts({"records": list(self.STORIES)},
                             {"measures": [{"type": "opening", "name": "o", "word": "[A-Za-z']+"}]})
        assert [r["o"] for r in rows] == ["the old man who", "the old man who", "rain", ""]

    def test_corpus_mode_counts_distinct_openings_commonest_first(self):
        out = measure_texts({"records": list(self.STORIES)},
                            {"mode": "corpus", "measures": [{"type": "opening", "name": "o", "words": 2}]})[0]
        assert out["o_distinct"] == 3
        assert out["o_values"][0] == {"value": "the old", "count": 2}

    def test_words_is_at_least_one(self):
        with pytest.raises(ValueError, match="at least 1"):
            measure_texts({"records": list(self.STORIES)}, {"measures": [{"type": "opening", "words": 0}]})


def test_measure_reads_the_text_from_the_field_named():
    rows = [{"id": "slot3-p1", "kl_bits": 0.2}, {"id": "ALL", "pass_rate": 0.8}]
    out = measure_texts({"records": rows}, {"field": "id", "keep": True, "measures": [
        {"type": "capture", "name": "slot", "pattern": r"^slot(\d+)-", "as": "number"}]})
    assert [r.get("slot") for r in out] == [3, None]
    assert out[0]["kl_bits"] == 0.2


def test_a_condition_value_is_bound_from_a_param_through_the_executor(monkeypatch):
    import hashlib

    from mechbench_schema import dump_canonical

    from mechbench_compute import bench
    from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec

    stored = {"kind": "collection", "item_kind": "records/record", "key": ["id"], "items": REPLIES}

    def fetch(ref, with_meta=False):
        meta = {"content_hash": "sha256:" + hashlib.sha256(dump_canonical(stored)).hexdigest()}
        return ({"payload": stored}, meta) if with_meta else {"payload": stored}

    monkeypatch.setattr(bench, "fetch", fetch)
    graph = {"dataflow": 2, "nodes": [
        {"id": "reached", "block": "records/group", "params": {
            "by": {"prompt": "metadata.coords.prompt"},
            "aggregates": {"k, n": "wilson(metadata.call.usage.output_tokens >= params.allowance)"}}},
    ], "edges": [{"from": {"input": "stories"}, "to": {"node": "reached", "port": "records"}}]}
    extra = {"graph": graph, "params": {"allowance": 260},
             "declared_params": [{"name": "allowance", "type": "int"}],
             "inputs": {"stories": {"$ref": {"bench": "lab/p/results/j/gen"}}},
             "outputs": [{"name": "reached", "from": {"node": "reached"}}]}
    out = ProtocolExecutor().run(ProtocolSpec(kind="pipeline", prompt="", model_id=None, extra=extra))
    rows = (out.payload if hasattr(out, "payload") else out)["outputs"]["reached"]["items"]
    assert {r["prompt"]: (r["k"], r["n"]) for r in rows} == {"flash": (1, 3), "neutral": (1, 2)}
