from __future__ import annotations

import numpy as np
import pytest

from mechbench_compute import distill
from mechbench_compute import generate as gen

WORDS = {1: "one", 2: " two", 3: " three", 4: " four", 5: " five", 0: ""}


class _Row:
    def __getitem__(self, _k):
        return self

    def astype(self, _t):
        return self


class _Tok:
    def decode(self, ids):
        return "".join(WORDS[int(i)] for i in ids)


class _Model:
    tokenizer = _Tok()

    @staticmethod
    def lm(*_a, **_k):
        return _Row()


@pytest.fixture
def counting(monkeypatch):
    seq = iter([1, 2, 3, 4, 5, 0])
    monkeypatch.setattr(gen, "_sample_next", lambda *_a, **_k: next(seq))
    monkeypatch.setattr(gen, "_stop_ids", lambda _tok: {0})
    monkeypatch.setattr(distill, "_copy_prefix_cache", lambda _c: None)
    return _Model()


def _sample(model, **kw):
    return gen.sample_completion_cached(
        model, [7], max_tokens=10, rng=np.random.default_rng(0),
        prefill=(None, None), **kw)


class TestStopStrings:
    def test_without_stops_it_runs_to_the_turn_end(self, counting):
        assert _sample(counting) == "one two three four five"

    def test_it_ends_at_the_marker_and_the_marker_is_not_in_the_text(self, counting):
        assert _sample(counting, stop_strings=["three"]) == "one two "

    def test_the_earliest_marker_wins(self, counting):
        assert _sample(counting, stop_strings=["four", "two"]) == "one "

    def test_a_marker_that_never_appears_changes_nothing(self, counting):
        assert _sample(counting, stop_strings=["six"]) == "one two three four five"


class TestCutAtStop:
    def test_it_cuts_at_the_first_of_them(self):
        assert gen.cut_at_stop("a END b STOP c", ["STOP", "END"]) == "a "

    def test_an_empty_marker_is_not_a_marker(self):
        assert gen.cut_at_stop("unchanged", ["", ""]) == "unchanged"
