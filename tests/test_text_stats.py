import pytest

from mechbench_compute import ops
from mechbench_compute.ops.text.measure import measure_texts

STORIES = [
    {"id": "s1", "coords": {"prompt": "flash"},
     "text": "The old lighthouse keeper watched the sea."},
    {"id": "s2", "coords": {"prompt": "flash"},
     "text": "Here is a 100-word story: The rain fell."},
    {"id": "s3", "coords": {"prompt": "flash"},
     "text": "(wait, i need to write a story) Once upon a time."},
    {"id": "s4", "coords": {"prompt": "flash"},
     "text": "Mist wrapped the harbor. The lighthouse blinked."},
]

LEAK = {
    "kind": "pattern", "name": "meta_leak", "where": "prefix",
    "ignore_case": True,
    "patterns": [r"here is", r"here's", r"\(wait", r"sure[,!]", r"okay[,!]"],
}
OPENING = {
    "kind": "pattern", "name": "lighthouse_opening", "where": "prefix",
    "patterns": [r"The old lighthouse"],
}


def test_registered():
    assert "text/measure" in ops.find_standalone()


def test_pattern_prefix_and_annotate_mode():
    out = measure_texts({"records": STORIES},
                     {"measures": [LEAK, OPENING]})
    by_id = {r["id"]: r for r in out}
    assert [by_id[i]["meta_leak"] for i in ("s1", "s2", "s3", "s4")] == \
        [0, 1, 1, 0]
    assert by_id["s1"]["lighthouse_opening"] == 1
    assert by_id["s4"]["lighthouse_opening"] == 0
    assert by_id["s1"]["coords"] == {"prompt": "flash"}


def test_pattern_anywhere():
    out = measure_texts(
        {"records": STORIES},
        {"measures": [{"kind": "pattern", "name": "lh",
                       "patterns": [r"lighthouse"]}]})
    assert [r["lh"] for r in out] == [1, 0, 0, 1]


def test_lexical_counts_and_dup():
    out = measure_texts(
        {"records": [{"id": "x", "text": "the cat and the dog"}]},
        {"measures": [{"kind": "lexical", "name": "lex"}]})
    r = out[0]
    assert r["lex_words"] == 5
    assert r["lex_distinct"] == 4
    assert abs(r["lex_dup"] - 0.2) < 1e-9


def test_corpus_frequency_mean_log10_and_coverage():
    freqs = {"the": 1000.0, "cat": 10.0}
    out = measure_texts(
        {"records": [{"id": "x", "text": "The cat zorbles"}]},
        {"measures": [{"kind": "corpus_frequency", "name": "wf",
                       "frequencies": freqs}]})
    r = out[0]
    assert abs(r["wf"] - 2.0) < 1e-9
    assert abs(r["wf_coverage"] - 2 / 3) < 1e-3


def test_corpus_frequency_via_input_port():
    out = measure_texts(
        {"records": [{"id": "x", "text": "cat"}],
         "frequencies": {"weights": {"cat": 100.0}}},
        {"measures": [{"kind": "corpus_frequency", "name": "wf"}]})
    assert abs(out[0]["wf"] - 2.0) < 1e-9


def test_corpus_frequency_requires_table():
    with pytest.raises(ValueError, match="no frequency table"):
        measure_texts({"records": STORIES},
                   {"measures": [{"kind": "corpus_frequency",
                                  "name": "wf"}]})


def test_document_collection_input():
    coll = {"kind": "document_collection",
            "items": [{"id": "d1", "text": "Here is a tale.",
                       "metadata": {"coords": {"sample": 0}}}]}
    out = measure_texts({"documents": coll}, {"measures": [LEAK]})
    assert out[0]["meta_leak"] == 1
    assert out[0]["coords"] == {"sample": 0}


def test_corpus_mode_summary():
    out = measure_texts(
        {"records": STORIES},
        {"mode": "corpus",
         "measures": [LEAK,
                      {"kind": "lexical", "name": "lex"},
                      {"kind": "corpus_frequency", "name": "wf",
                       "frequencies": {"the": 1000.0}}]})
    assert len(out) == 1
    s = out[0]
    assert s["n_texts"] == 4
    assert s["meta_leak_count"] == 2
    assert abs(s["meta_leak_rate"] - 0.5) < 1e-9
    assert s["lex_corpus_distinct"] <= s["lex_corpus_words"]
    assert s["wf_mean"] is not None


def test_unknown_measure_kind():
    with pytest.raises(ValueError, match="unknown measure type"):
        measure_texts({"records": STORIES},
                   {"measures": [{"kind": "vibes"}]})


def test_weights_expectation_kind():
    from mechbench_compute.ops.eval.expect import check_expectations

    results = [{"id": "p1", "entropy_bits": 2.0,
                "top_tokens": [{"token": " cat", "p": 0.72},
                               {"token": " dog", "p": 0.24}]}]
    expectations = [{"id": "p1", "expect": {
        "kind": "weights", "weights": {"cat": 3.0, "dog": 1.0},
        "max_kl_bits": 0.05}}]
    out = check_expectations({"results": results,
                              "expectations": expectations}, {})
    row = next(r for r in out["items"] if r["id"] == "p1")
    assert row["kl_bits"] < 0.01
    assert row["pass"] is True


GENRES = ["Fiction", "Mystery", "Mystery Thriller", "Humor", "Witches"]
LISTS = [
    {"id": "l1", "text": '{ "genres": "Mystery, Humor, Witches, Fiction" }'},
    {"id": "l2", "text": '```json\n{ "genres": "Mystery Thriller, Mystery, Mystery, Humor" }\n```'},
    {"id": "l3", "text": '{ "genres": "Fiction, Dragons, Humor" }'},
    {"id": "l4", "text": "Sure! Here are four genres."},
]
LIST = {"type": "list", "name": "genres", "separator": ", ",
        "extract": r'"genres":\s*"([^"]*)"', "items": GENRES, "count": 4}


def test_list_annotates_each_draw():
    rows = {r["id"]: r for r in measure_texts({"records": LISTS}, {"measures": [LIST]})}
    assert rows["l1"]["genres_items"] == 4 and rows["l1"]["genres_valid"] == 1
    assert rows["l1"]["genres_first"] == "Mystery"
    assert rows["l2"]["genres_duplicates"] == 1 and rows["l2"]["genres_distinct"] == 3
    assert rows["l2"]["genres_valid"] == 0
    assert rows["l3"]["genres_unknown"] == 1 and rows["l3"]["genres_valid"] == 0
    assert rows["l4"]["genres_parsed"] == 0 and rows["l4"]["genres_items"] == 0
    assert rows["l4"]["genres_first"] is None and rows["l4"]["genres_valid"] == 0


def test_list_summarises_the_corpus():
    (summary,) = measure_texts({"records": LISTS}, {"measures": [LIST], "mode": "corpus"})
    assert summary["genres_parsed_rate"] == 0.75
    assert summary["genres_duplicate_rate"] == 0.25
    assert summary["genres_valid_rate"] == 0.25
    assert summary["genres_mean_items"] == 2.75
    assert summary["genres_distinct_items"] == 6
    assert summary["genres_unknown_rate"] == round(1 / 11, 4)


def test_list_takes_a_target_map_as_its_vocabulary_but_not_a_transform():
    freqs = {"weights": {g: 1.0 for g in GENRES}}
    (row,) = measure_texts({"records": LISTS[:1]}, {"measures": [dict(LIST, items=freqs)]})
    assert row["genres_unknown"] == 0
    with pytest.raises(ValueError, match="untransformed"):
        measure_texts({"records": LISTS[:1]},
                   {"measures": [dict(LIST, items=dict(freqs, transform=[{"op": "sqrt"}]))]})


def test_list_without_extract_reads_the_whole_text_and_can_fold_case():
    rows = measure_texts({"records": [{"id": "x", "text": "humor, HUMOR, Fiction"}]},
                      {"measures": [{"type": "list", "name": "g", "items": GENRES,
                                     "ignore_case": True}]})
    assert rows[0]["g_items"] == 3 and rows[0]["g_duplicates"] == 1 and rows[0]["g_unknown"] == 0


def test_items_mode_tallies_what_the_corpus_said_map_or_not():
    rows = measure_texts({"records": LISTS}, {"measures": [LIST], "mode": "items"})
    by_item = {r["item"]: r for r in rows}
    assert by_item["Mystery"]["count"] == 3 and by_item["Mystery"]["lists"] == 2
    assert by_item["Mystery"]["in_vocabulary"] is True
    assert by_item["Dragons"]["count"] == 1 and by_item["Dragons"]["in_vocabulary"] is False
    assert [r["item"] for r in rows[:2]] == ["Humor", "Mystery"]
    assert by_item["Humor"]["count"] == 3 and by_item["Humor"]["lists"] == 3
    assert abs(sum(r["share"] for r in rows) - 1.0) < 1e-5
    assert by_item["Mystery"]["first"] == 1 and by_item["Mystery Thriller"]["first"] == 1
    assert by_item["Humor"]["first"] == 0
    assert rows[0]["coords"] == {"measure": "genres", "item": "Humor"}


def test_items_mode_without_a_vocabulary_labels_nothing():
    rows = measure_texts({"records": LISTS[:1]},
                      {"measures": [{k: v for k, v in LIST.items() if k != "items"}],
                       "mode": "items"})
    assert {r["item"] for r in rows} == {"Mystery", "Humor", "Witches", "Fiction"}
    assert all(r["in_vocabulary"] is None for r in rows)


def test_items_mode_needs_a_list_or_lexical_measure():
    with pytest.raises(ValueError, match="needs a `list` or `lexical` measure"):
        measure_texts({"records": LISTS}, {"measures": [LEAK], "mode": "items"})
