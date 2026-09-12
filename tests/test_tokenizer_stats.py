"""Tokenizer diagnostics (task 000377) against a fake word-piece
tokenizer: the depth inventory in an envelope, fragmentation, scripts,
and the naturalism gate."""

from __future__ import annotations

import pytest

from mechbench_compute import tokenizer_stats as ts
from mechbench_compute.block_params import check_params


class FakeTokenizer:
    """Every 3 characters is a token (so 'cat' is 1, 'horse' is 2), a
    space is its own token, and a prefix is tokenized independently of
    what follows — a stable boundary."""

    def encode(self, text: str, add_special_tokens: bool = False):
        out = []
        for word in text.split(" "):
            if out:
                out.append(32)  # the space token
            for i in range(0, len(word), 3):
                piece = word[i:i + 3]
                if piece:
                    out.append(1000 + sum(ord(c) for c in piece))
        return out

    def decode(self, ids):
        return "".join(" " if i == 32 else f"<{i}>" for i in ids)


class TestDepthInventory:
    def test_items_as_continuations_of_a_prefix(self):
        out = ts.tokenizer_stats(FakeTokenizer(), "fake/tok", {},
                                 {"items": ["cat", "horse", "elephant"],
                                  "prefix": '{ "animal": "'})
        assert out["kind"] == "tokenizer_stats" and out["n_items"] == 3
        # 3 chars/token: cat=1, horse=2, elephant=3
        assert {r["depth"]: r["count"] for r in out["rows"]} == {1: 1, 2: 1, 3: 1}
        assert out["mean_depth"] == 2.0 and out["max_depth"] == 3
        assert out["single_token_fraction"] == pytest.approx(1 / 3, abs=1e-4)
        assert out["prefix"] == '{ "animal": "'

    def test_a_vocabulary_object_is_its_weight_keys(self):
        vocab = {"kind": "target_map", "weights": {"cat": 0.5, "dog": 0.5}}
        out = ts.tokenizer_stats(FakeTokenizer(), "t", {"vocabulary": vocab}, {})
        assert out["n_items"] == 2 and out["rows"] == [
            {"depth": 1, "count": 2, "share": 1.0}]

    def test_records_supply_text(self):
        recs = {"records": [{"id": "a", "text": "cat"}, {"id": "b", "user": "horse"}]}
        out = ts.tokenizer_stats(FakeTokenizer(), "t", {"records": recs}, {})
        assert out["n_items"] == 2 and out["max_depth"] == 2

    def test_nothing_to_measure_is_an_error(self):
        with pytest.raises(ValueError, match="needs"):
            ts.tokenizer_stats(FakeTokenizer(), "t", {}, {})


class TestFragmentationAndScripts:
    def test_tokens_per_word_and_fragmented_fraction(self):
        out = ts.tokenizer_stats(FakeTokenizer(), "t", {},
                                 {"items": ["cat dog", "elephant"],
                                  "top_fragmented": 1})
        # "cat dog" = 1 + space + 1 = 3 tokens / 2 words; "elephant" = 3 / 1
        assert out["mean_tokens_per_word"] == pytest.approx((1.5 + 3.0) / 2)
        assert out["fragmented_fraction"] == 1.0
        assert out["most_fragmented"][0]["item"] == "elephant"

    def test_script_composition(self):
        out = ts.tokenizer_stats(FakeTokenizer(), "t", {},
                                 {"items": ["abc", "日本", "12"]})
        sc = out["script_composition"]  # shares are rounded to 4 places
        assert sc["latin"] == pytest.approx(3 / 7, abs=1e-4)
        assert sc["cjk"] == pytest.approx(2 / 7, abs=1e-4)
        assert sc["digit"] == pytest.approx(2 / 7, abs=1e-4)


class TestGate:
    def test_passes_when_every_item_has_the_expected_depth(self):
        out = ts.tokenizer_stats(FakeTokenizer(), "t", {},
                                 {"items": ["cat", "dog", "emu"],
                                  "expect_depth": 1})
        assert out["gate"] == {"expected_depth": 1, "pass": True,
                               "n_violations": 0, "violations": []}

    def test_names_the_violators(self):
        out = ts.tokenizer_stats(FakeTokenizer(), "t", {},
                                 {"items": ["cat", "horse"], "expect_depth": 1})
        g = out["gate"]
        assert g["pass"] is False and g["n_violations"] == 1
        assert g["violations"][0]["item"] == "horse"
        assert g["violations"][0]["depth"] == 2

    def test_keep_items_carries_the_pieces(self):
        out = ts.tokenizer_stats(FakeTokenizer(), "t", {},
                                 {"items": ["horse"], "keep_items": True})
        # "hor" = 104+111+114 = 329, "se" = 115+101 = 216, each +1000
        assert out["items"][0]["pieces"] == ["<1329>", "<1216>"]


class TestBlock:
    def test_the_block_names_the_tokenizer_by_the_model(self):
        class M:
            tokenizer = FakeTokenizer()
            model_id = "acme/tiny"
        out = ts.block(M(), {}, {"items": ["cat"]})
        assert out["tokenizer"] == "acme/tiny"

    def test_params_are_guarded(self):
        with pytest.raises(ValueError, match="does not accept"):
            check_params("~canonical/ops/tokenize/stats/1", {"depth": 3})
        check_params("~canonical/ops/tokenize/stats/1",
                     {"items": ["a"], "prefix": "x", "expect_depth": 1})
