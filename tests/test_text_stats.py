"""~canonical/ops/text/stats/1 — the corpus measurement op (experiment
023 platform work): pattern counts (meta-leak, openings), lexical
spread, and corpus-frequency of vocabulary, in annotate and corpus
modes, over records and document_collections."""

import pytest

from mechbench_compute.blocks import PURE_BLOCKS, text_stats

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
    assert "~canonical/ops/text/stats/1" in PURE_BLOCKS


def test_pattern_prefix_and_annotate_mode():
    out = text_stats({"records": STORIES},
                     {"measures": [LEAK, OPENING]})
    by_id = {r["id"]: r for r in out}
    assert [by_id[i]["meta_leak"] for i in ("s1", "s2", "s3", "s4")] == \
        [0, 1, 1, 0]
    assert by_id["s1"]["lighthouse_opening"] == 1
    assert by_id["s4"]["lighthouse_opening"] == 0
    assert by_id["s1"]["coords"] == {"prompt": "flash"}


def test_pattern_anywhere():
    out = text_stats(
        {"records": STORIES},
        {"measures": [{"kind": "pattern", "name": "lh",
                       "patterns": [r"lighthouse"]}]})
    assert [r["lh"] for r in out] == [1, 0, 0, 1]


def test_lexical_counts_and_dup():
    out = text_stats(
        {"records": [{"id": "x", "text": "the cat and the dog"}]},
        {"measures": [{"kind": "lexical", "name": "lex"}]})
    r = out[0]
    assert r["lex_words"] == 5
    assert r["lex_distinct"] == 4
    assert abs(r["lex_dup"] - 0.2) < 1e-9


def test_corpus_frequency_mean_log10_and_coverage():
    freqs = {"the": 1000.0, "cat": 10.0}
    out = text_stats(
        {"records": [{"id": "x", "text": "The cat zorbles"}]},
        {"measures": [{"kind": "corpus_frequency", "name": "wf",
                       "frequencies": freqs}]})
    r = out[0]
    # log10(1000)=3, log10(10)=1 → mean 2; zorbles missing
    assert abs(r["wf"] - 2.0) < 1e-9
    assert abs(r["wf_coverage"] - 2 / 3) < 1e-3  # rounded to 4 places


def test_corpus_frequency_via_input_port():
    out = text_stats(
        {"records": [{"id": "x", "text": "cat"}],
         "frequencies": {"weights": {"cat": 100.0}}},
        {"measures": [{"kind": "corpus_frequency", "name": "wf"}]})
    assert abs(out[0]["wf"] - 2.0) < 1e-9


def test_corpus_frequency_requires_table():
    with pytest.raises(ValueError, match="no frequency table"):
        text_stats({"records": STORIES},
                   {"measures": [{"kind": "corpus_frequency",
                                  "name": "wf"}]})


def test_document_collection_input():
    coll = {"kind": "document_collection",
            "items": [{"id": "d1", "text": "Here is a tale.",
                       "metadata": {"coords": {"sample": 0}}}]}
    out = text_stats({"documents": coll}, {"measures": [LEAK]})
    assert out[0]["meta_leak"] == 1
    assert out[0]["coords"] == {"sample": 0}


def test_corpus_mode_summary():
    out = text_stats(
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
    with pytest.raises(ValueError, match="unknown measure kind"):
        text_stats({"records": STORIES},
                   {"measures": [{"kind": "vibes"}]})
