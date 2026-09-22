"""Whole-trie paths and item slots: outcomes of any token
length trained as soft rows along their paths, first-slot and later-slot
tokenizations kept apart, and slots drawn without replacement — on a
word-piece fake tokenizer, no model."""

import re

import numpy as np
import pytest

from mechbench_compute.distill import TargetMap, encode
from mechbench_compute.finetune import (
    SlotTrie,
    batch_for,
    build_item_path_factory,
    build_path_factory,
    build_sequence_factory,
    check_enough_to_draw,
    compile_item_slots,
    compile_tries,
    draw_slots,
    naturalism_gate,
)


class PieceTok:
    """Letters in pieces of up to four, a leading space kept on the piece
    that follows it (so "Mystery" is `Myst|ery` and " Mystery" is
    ` Myst|ery`, a different first token), every other character its own
    token. One merge a real BPE might make: a word ending in `q` takes a
    following comma into its token, so "Iraq" tokenizes differently
    before a comma than alone."""

    _pieces = re.compile(r"[A-Za-z]{0,3}q,| ?[A-Za-z]{1,4}|.", re.DOTALL)

    def __init__(self):
        self.ids: dict[str, int] = {}
        self.texts: dict[int, str] = {}

    def _id(self, piece: str) -> int:
        if piece not in self.ids:
            self.ids[piece] = len(self.ids) + 1
            self.texts[self.ids[piece]] = piece
        return self.ids[piece]

    def encode(self, text, add_special_tokens=False):
        return [self._id(p) for p in self._pieces.findall(text)]

    def decode(self, ids):
        return "".join(self.texts[int(i)] for i in ids)

    def pieces(self, ids):
        return [self.texts[int(i)] for i in ids]


PROMPT = '{ "genres": "'


def rows_by_text(tok, rows):
    return {tok.decode([t]): round(p, 6) for t, p in rows.items()}


# --- whole-trie paths (depth 1, batch.path) ---------------------------


def test_path_rows_cover_every_token_and_the_closer():
    tok = PieceTok()
    target = TargetMap({"Mystery": 2.0, "Mystery Thriller": 1.0, "Humor": 1.0})
    (trie,) = compile_tries(tok, target, [PROMPT], closer='"')
    ex = trie.path_rows("Mystery Thriller")
    assert tok.pieces(ex.tokens) == ["Myst", "ery", " Thri", "ller", '"']
    rows = [rows_by_text(tok, r) for r in ex.soft]
    assert rows[0] == {"Myst": 0.75, "Humo": 0.25}
    assert rows[1] == {"ery": 1.0}
    # Where "Mystery" ends and "Mystery Thriller" goes on: the closer
    # carries the shorter outcome's share.
    assert rows[2] == {'"': round(2 / 3, 6), " Thri": round(1 / 3, 6)}
    assert rows[3:] == [{"ller": 1.0}, {'"': 1.0}]


def test_path_factory_samples_outcomes_by_mass():
    tok = PieceTok()
    target = TargetMap({"Mystery": 3.0, "Humor": 1.0})
    factory = build_path_factory(compile_tries(tok, target, [PROMPT, "{ " + PROMPT], closer='"'))
    rng = np.random.default_rng(0)
    firsts = [tok.decode([factory(rng)[0].tokens[0]]) for _ in range(2000)]
    assert abs(firsts.count("Myst") / 2000 - 0.75) < 0.03


# --- item slots --------------------------------------------------------


def test_first_and_later_slots_are_tokenized_apart():
    tok = PieceTok()
    target = TargetMap({"Mystery": 2.0, "Mystery Thriller": 1.0, "Humor": 1.0})
    first, second, third = compile_item_slots(tok, [target] * 3, PROMPT, join=", ", closer='"')
    assert tok.pieces(first.paths["Mystery"]) == ["Myst", "ery", ","]
    assert tok.pieces(second.paths["Mystery"]) == [" Myst", "ery", ","]
    # Only the last slot ends on the closer; the middle ones share a trie.
    assert tok.pieces(third.paths["Humor"]) == [" Humo", "r", '"']
    assert second is not third


def test_the_draw_excludes_what_came_before_down_to_the_shared_prefix():
    tok = PieceTok()
    target = TargetMap({"Mystery": 2.0, "Mystery Thriller": 1.0, "Humor": 1.0})
    _, last = compile_item_slots(tok, [target] * 2, PROMPT, join=", ", closer='"')
    myst, ery = last.paths["Mystery"][:2]
    with_repeat = rows_by_text(tok, last.node_target([myst, ery]))
    assert with_repeat == {'"': round(2 / 3, 6), " Thri": round(1 / 3, 6)}
    # "Mystery" already drawn: " Myst" survives on "Mystery Thriller"
    # alone, and after " Mystery" the only way on is " Thri".
    assert rows_by_text(tok, last.node_target([], ["Mystery"])) == {" Myst": 0.5, " Humo": 0.5}
    assert rows_by_text(tok, last.node_target([myst, ery], ["Mystery"])) == {" Thri": 1.0}


def test_item_factory_never_repeats_without_replacement_and_rows_match_the_draw():
    tok = PieceTok()
    target = TargetMap({"Mystery": 5.0, "Mystery Thriller": 1.0, "Humor": 1.0, "Witches": 1.0})
    factory = build_item_path_factory(tok, [target] * 3, [PROMPT], join=", ", closer='"', replace=False)
    rng = np.random.default_rng(3)
    for _ in range(300):
        (ex,) = factory(rng)
        assert ex.prompt_ids == encode(tok, PROMPT)
        text = tok.decode(ex.tokens)
        assert text.endswith('"')
        genres = text[:-1].split(", ")
        assert len(genres) == 3 and len(set(genres)) == 3
        # Every trained token is possible under its own row.
        for t, row in zip(ex.tokens, ex.soft):
            assert row[t] > 0 and abs(sum(row.values()) - 1) < 1e-9


def test_item_factory_with_replacement_can_repeat():
    tok = PieceTok()
    target = TargetMap({"Mystery": 10.0, "Humor": 1.0})
    factory = build_item_path_factory(tok, [target] * 2, [PROMPT], join=", ", closer='"')
    rng = np.random.default_rng(0)
    texts = [tok.decode(factory(rng)[0].tokens) for _ in range(100)]
    assert 'Mystery, Mystery"' in texts


def test_gate_names_outcomes_that_tokenize_differently_in_a_list():
    tok = PieceTok()
    target = TargetMap({"Iraq": 1.0, "Humor": 1.0})
    with pytest.raises(ValueError, match=r"ITEM SLOT VIOLATION.*'Iraq'"):
        compile_item_slots(tok, [target] * 2, PROMPT, join=", ", closer='"')
    # The gate can be declined, as for token slots.
    compile_item_slots(tok, [target] * 2, PROMPT, join=", ", closer='"', gate=False)


def test_item_slots_need_a_join_and_a_closer():
    tok = PieceTok()
    target = TargetMap({"Mystery": 1.0, "Humor": 1.0})
    with pytest.raises(ValueError, match="need a join"):
        compile_item_slots(tok, [target] * 2, PROMPT, join="", closer='"')
    with pytest.raises(ValueError, match="need a closer"):
        compile_item_slots(tok, [target] * 2, PROMPT, join=", ", closer="")


def test_without_replacement_each_slot_needs_enough_outcomes():
    two = TargetMap({"a": 1.0, "b": 1.0})
    check_enough_to_draw([two, two])
    with pytest.raises(ValueError, match="slot 2"):
        check_enough_to_draw([two, two, two])


def test_slot_trie_refuses_a_node_everything_was_drawn_from():
    trie = SlotTrie({"a": 1.0, "b": 1.0}, {"a": [1, 9], "b": [2, 9]})
    with pytest.raises(ValueError, match="has been drawn"):
        trie.node_target([], ["a", "b"])


# --- token slots without replacement -----------------------------------


class CharTok:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def test_token_slots_drawn_without_replacement_never_repeat():
    digits = TargetMap.uniform(list("0123456789"))
    rng = np.random.default_rng(1)
    for _ in range(200):
        drawn = draw_slots([digits] * 6, rng, replace=False)
        assert len(set(drawn)) == 6
    factory = build_sequence_factory(CharTok(), [digits] * 6, ["P:"], join="", closer='"', replace=False)
    for _ in range(50):
        (ex,) = factory(rng)
        assert len(set(ex.tokens[:6])) == 6
    naturalism_gate(CharTok(), [digits] * 6, ["P:"], [[ord("P"), ord(":")]], join="", closer='"', replace=False)


# --- batch kinds -------------------------------------------------------


def test_batch_defaults_follow_the_shape():
    assert batch_for(1, "token", None) == {"target": 3, "anchor": 1, "continuation": 2}
    assert batch_for(4, "token", None) == {"sequence": 3, "target": 1, "anchor": 1}
    assert batch_for(4, "item", None) == {"path": 3, "anchor": 1}
    assert batch_for(1, "token", {"path": 3, "anchor": 1}) == {"path": 3, "anchor": 1}


def test_batch_refuses_a_kind_the_shape_does_not_build():
    with pytest.raises(ValueError, match=r"\['sequence'\].*unit: item"):
        batch_for(4, "item", {"sequence": 3, "anchor": 1})
    with pytest.raises(ValueError, match=r"\['path'\]"):
        batch_for(4, "token", {"path": 3})
    # A zero count asks for nothing, so it is not refused.
    assert batch_for(4, "item", {"path": 3, "sequence": 0}) == {"path": 3, "sequence": 0}
