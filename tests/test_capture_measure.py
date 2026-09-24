from __future__ import annotations

import pytest

from mechbench_compute import ops


def _docs(*texts):
    return {"kind": "collection", "item_kind": "text/document",
            "items": [{"id": f"d{i}", "text": t} for i, t in enumerate(texts)]}


def _run(docs, measure, mode="annotate"):
    return ops.run_standalone("text/measure", {"documents": docs},
                              {"mode": mode, "measures": [measure]})


class TestCapture:
    def test_one_field_under_the_name_given(self):
        out = _run(_docs("Rating: 4 of 5.", "Rating: 2 of 5."),
                   {"kind": "capture", "name": "rating", "pattern": r"Rating:\s*(\d+)", "as": "number"})
        assert [i["rating"] for i in out["items"]] == [4, 2]
        assert set(out["items"][0]) == {"id", "coords", "rating"}

    def test_a_number_is_a_number_and_a_string_is_a_string(self):
        number = _run(_docs("Score 3.5 overall"),
                      {"kind": "capture", "name": "s", "pattern": r"([\d.]+)", "as": "number"})
        assert number["items"][0]["s"] == 3.5
        text = _run(_docs("Score 3.5 overall"), {"kind": "capture", "name": "s", "pattern": r"([\d.]+)"})
        assert text["items"][0]["s"] == "3.5"
        with pytest.raises(ValueError, match="not a number"):
            _run(_docs("Score: high"), {"kind": "capture", "name": "s", "pattern": r"Score:\s*(\w+)",
                                        "as": "number"})

    def test_take_chooses_which_match(self):
        doc = _docs("Ana opened, and then Bo replied.")
        first = _run(doc, {"kind": "capture", "name": "who", "pattern": r"\b(Ana|Bo)\b"})
        last = _run(doc, {"kind": "capture", "name": "who", "pattern": r"\b(Ana|Bo)\b", "take": "last"})
        assert first["items"][0]["who"] == "Ana" and last["items"][0]["who"] == "Bo"

    def test_a_vocabulary_constrains_and_canonicalises(self):
        out = _run(_docs("So ANA goes next.", "Over to Bo.", "Cy takes it."),
                   {"kind": "capture", "name": "next", "pattern": r"\b(ana|bo|cy)\b",
                    "ignore_case": True, "items": ["ana", "bo"]})
        assert [i.get("next") for i in out["items"]] == ["ana", "bo", None]

    def test_a_text_that_says_nothing_takes_the_declared_course(self):
        quiet = _docs("Nothing to report.")
        assert "r" not in _run(quiet, {"kind": "capture", "name": "r", "pattern": r"(\d+)"})["items"][0]
        with pytest.raises(ValueError, match="'d0'.*nothing the pattern matches"):
            _run(quiet, {"kind": "capture", "name": "r", "pattern": r"(\d+)", "on_missing": "error"})

    def test_the_corpus_summarises_what_was_captured(self):
        numbers = _run(_docs("n=1", "n=3", "no number"),
                       {"kind": "capture", "name": "n", "pattern": r"n=(\d+)", "as": "number"},
                       mode="corpus")["items"][0]
        assert numbers["n_captured"] == 2 and numbers["n_rate"] == 0.6667 and numbers["n_mean"] == 2.0
        labels = _run(_docs("Genre: noir", "Genre: fable", "Genre: noir"),
                      {"kind": "capture", "name": "g", "pattern": r"Genre:\s*(\w+)"},
                      mode="corpus")["items"][0]
        assert labels["g_values"] == [{"value": "noir", "count": 2}, {"value": "fable", "count": 1}]

    def test_it_refuses_what_it_cannot_do(self):
        with pytest.raises(ValueError, match="needs a `pattern`"):
            _run(_docs("x"), {"kind": "capture", "name": "a"})
        with pytest.raises(ValueError, match="take is 'first' or 'last'"):
            _run(_docs("x"), {"kind": "capture", "name": "a", "pattern": "(x)", "take": "middle"})
        with pytest.raises(ValueError, match="`as` is 'string' or 'number'"):
            _run(_docs("x"), {"kind": "capture", "name": "a", "pattern": "(x)", "as": "int"})
        with pytest.raises(ValueError, match="unknown measure type"):
            _run(_docs("x"), {"kind": "capturing", "name": "a", "pattern": "(x)"})

    def test_the_older_measures_are_untouched(self):
        out = _run(_docs("a lighthouse keeper"),
                   {"kind": "pattern", "name": "light", "patterns": ["lighthouse"]})
        assert out["items"][0]["light"] == 1
