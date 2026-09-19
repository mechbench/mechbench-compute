"""Where a model's reasoning is, in the tokens it produced (000592)."""

from __future__ import annotations

import pytest

from mechbench_compute import positions as P
from mechbench_compute import thinking as T


class _Tok:
    """A tokenizer that declares what its vocabulary holds."""

    unk_token_id = 0

    def __init__(self, vocab: dict[str, int]):
        self._v = vocab

    def convert_tokens_to_ids(self, s: str) -> int:
        return self._v.get(s, self.unk_token_id)


class TestDeclaredDelimiters:
    def test_a_model_that_declares_them(self):
        assert T.delimiter_ids(_Tok({"<think>": 151667, "</think>": 151668})) == (151667, 151668)

    def test_a_model_that_does_not(self):
        # Gemma has no reasoning tokens: every lookup is `unk`. A span
        # must not be invented for it.
        assert T.delimiter_ids(_Tok({})) is None

    def test_a_tokenizer_without_the_method(self):
        assert T.delimiter_ids(object()) is None


class TestSegments:
    PAIR = (100, 101)

    def test_the_span_is_what_is_inside_the_markers(self):
        # prompt 0..2, then <think> 7 8 9 </think> 42 43
        ids = [1, 2, 3, 100, 7, 8, 9, 101, 42, 43]
        segs = T.segments(ids, start=3, pair=self.PAIR)
        assert segs == [
            {"role": "thinking", "token_start": 4, "token_end": 7, "terminated": True},
            {"role": "answer", "token_start": 8, "token_end": 10},
        ]

    def test_cut_off_mid_thought_has_no_answer_and_says_so(self):
        # A model that ran out of room did not produce an answer, and a
        # reader must be able to see that rather than find an empty one.
        segs = T.segments([1, 100, 7, 8], start=1, pair=self.PAIR)
        assert segs == [{"role": "thinking", "token_start": 2, "token_end": 4,
                         "terminated": False}]

    def test_a_model_that_wrote_none(self):
        assert T.segments([1, 2, 3], start=1, pair=self.PAIR) == []
        assert T.segments([1, 100, 7], start=1, pair=None) == []

    def test_only_the_generated_tail_is_searched(self):
        # The delimiter id appearing in the PROMPT is not the model's
        # reasoning — it is something the prompt said.
        ids = [100, 5, 101, 9]
        assert T.segments(ids, start=3, pair=self.PAIR) == []

    def test_segmentation_wraps_or_abstains(self):
        assert T.segmentation([1, 2], start=1, pair=self.PAIR) is None
        seg = T.segmentation([1, 100, 7, 101, 9], start=1, pair=self.PAIR)
        assert seg is not None and seg["schema_name"] == "reasoning"


class TestAnswerText:
    def test_a_complete_thought_is_cut(self):
        assert T.answer_text("<think>hmm</think>  four") == "four"

    def test_an_incomplete_one_is_left_whole(self):
        # Cutting here would hide that the model never finished.
        assert T.answer_text("<think>hmm and") == "<think>hmm and"

    def test_text_without_a_thought(self):
        assert T.answer_text("four") == "four"


class TestSelector:
    SEGS = [
        {"schema_name": "envelope", "segments": [
            {"role": "prompt", "token_start": 0, "token_end": 3},
            {"role": "body", "token_start": 3, "token_end": 10}]},
        {"schema_name": "reasoning", "segments": [
            {"role": "thinking", "token_start": 4, "token_end": 7, "terminated": True},
            {"role": "answer", "token_start": 8, "token_end": 10}]},
    ]

    def test_a_span_resolves_to_its_positions(self):
        assert P.resolve({"segment": "thinking"}, 10, segmentations=self.SEGS) == [4, 5, 6]
        assert P.resolve({"segment": "answer"}, 10, segmentations=self.SEGS) == [8, 9]
        # And the envelope's own roles are nameable by the same word.
        assert P.resolve({"segment": "prompt"}, 10, segmentations=self.SEGS) == [0, 1, 2]

    def test_a_document_without_the_span_is_refused_with_what_it_has(self):
        # Not "read the answer instead": a capture aimed at reasoning
        # must fail loudly on a model that did none.
        with pytest.raises(ValueError, match="no 'thinking' segment here.*body.*prompt"):
            P.resolve({"segment": "thinking"}, 10, segmentations=[self.SEGS[0]])
        with pytest.raises(ValueError, match="carries no named spans"):
            P.resolve({"segment": "thinking"}, 10)

    def test_one_refuses_a_span_of_many(self):
        with pytest.raises(ValueError, match="names 3 positions"):
            P.one({"segment": "thinking"}, 10, segmentations=self.SEGS)
