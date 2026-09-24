from __future__ import annotations

import math
import re

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_map

from mechbench_compute.distill import render
from mechbench_compute.interp.answer import (
    encode_answer,
    encode_answer_in_context,
    make_answer,
    spell_answer_variants,
)
from mechbench_compute.interp.read_last_logp import read_last_logp
from mechbench_compute.interventions import Ablate
from mechbench_compute.model import Model
from mechbench_compute.ops import Context
from mechbench_compute.ops.intervene.ablate_heads import ablate_heads
from mechbench_compute.ops.intervene.ablate_layers import ablate_layers
from mechbench_compute.ops.intervene.patch import patch_trace
from mechbench_compute.ops.intervene.path import run_path_patch
from mechbench_compute.ops.logits import read as read_op
from mechbench_compute.ops.logits import read_layers as read_layers_op
from mechbench_compute.ops.logits.attribute import attribute_logits
from mechbench_compute.ops.logits.scan import scan_positions

PIECES = {"The": 1, " capital": 2, " of": 3, " France": 4, " is": 5, ":": 12,
          "Paris": 6, " Paris": 7, "Rome": 8, " Rome": 9, " Germany": 10,
          "7": 11, " 7": 11}
SPACED, BARE = PIECES[" Paris"], PIECES["Paris"]
PROMPT = "The capital of France is:"


class PieceTokenizer:
    all_special_ids = (0,)

    def encode(self, text, add_special_tokens=False):
        return [PIECES.get(p, 20 + sum(map(ord, p)) % 40)
                for p in re.findall(r" ?\w+|[^\w\s]| ", text)]

    def decode(self, ids):
        names = {v: k for k, v in PIECES.items()}
        return "".join(names.get(int(i), f"<{int(i)}>") for i in ids)


@pytest.fixture(scope="module")
def model():
    from mlx_lm.models import llama

    lm = llama.Model(llama.ModelArgs(
        model_type="llama", hidden_size=32, num_hidden_layers=4, intermediate_size=64,
        num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-6,
        vocab_size=64, tie_word_embeddings=False))
    keys = iter(mx.random.split(mx.random.key(5), 1000))
    lm.update(tree_map(lambda p: mx.random.normal(p.shape, key=next(keys)) * 0.5,
                       lm.parameters()))
    mx.eval(lm.parameters())
    return Model(lm, PieceTokenizer())


def _record(answer: str, **extra):
    return {"id": "r", "text": PROMPT, "tracked": {"answer": answer}, **extra}


def _pair(answer: str):
    return {"id": "p", "template": "raw", "a": PROMPT,
            "b": "The capital of Germany is:", "tracked": {"answer": answer}}


def _base_lp(model) -> np.ndarray:
    return read_last_logp(model.run(render(model, {"text": PROMPT}).array).logits)


RUNS = {
    "logits/attribute": lambda m, a: attribute_logits(m, [_record(a)], {}),
    "logits/scan": lambda m, a: scan_positions(m, [_record(a)], {"layers": [1, 3]}),
    "intervene/ablate-layers": lambda m, a: ablate_layers(m, [_record(a)], {"layers": [0, 2]}),
    "intervene/ablate-heads": lambda m, a: ablate_heads(m, [_record(a)], {"layers": [1]}),
    "intervene/patch": lambda m, a: patch_trace(m, [_pair(a)], {"layers": [0, 1]}),
    "intervene/patch attribution": lambda m, a: patch_trace(
        m, [_pair(a)], {"layers": [0, 1], "method": "attribution"}),
    "intervene/path": lambda m, a: run_path_patch(
        m, [_pair(a)], {"senders": "all-layers", "metric": "prob"}),
    "logits/read": lambda m, a: read_op.run(
        Context(loaded=m), {"conditions": [_record(a)]}, {}),
    "logits/read-layers": lambda m, a: read_layers_op.run(
        Context(loaded=m), {"records": [_record(a)]}, {}),
}


class TestEitherSpellingIsTheSameAnswer:
    @pytest.mark.parametrize("op", sorted(RUNS))
    def test_with_or_without_the_leading_space_the_result_is_identical(self, model, op):
        assert RUNS[op](model, " Paris") == RUNS[op](model, "Paris")

    def test_both_spellings_are_distinct_tokens_here(self, model):
        answer = encode_answer(model.tokenizer, "Paris")
        assert answer.ids == (SPACED, BARE)
        assert encode_answer(model.tokenizer, "  Paris") == answer


class TestTheSetScores:
    def test_p_is_the_sum_of_the_variants(self, model):
        out = RUNS["logits/read"](model, "Paris")
        entry = out["items"][0]["tracked"]["answer"]
        lp = _base_lp(model)
        want = math.exp(lp[SPACED]) + math.exp(lp[BARE])
        assert [v["token"]["id"] for v in entry["variants"]] == [SPACED, BARE]
        assert entry["p"] == pytest.approx(want, abs=1e-5)
        assert entry["p"] == pytest.approx(sum(v["p"] for v in entry["variants"]), abs=2e-5)
        assert entry["logp"] == pytest.approx(math.log(want), abs=1e-4)
        preferred = SPACED if lp[SPACED] >= lp[BARE] else BARE
        assert entry["token"]["id"] == preferred

    def test_rank_is_the_better_variants(self):
        lp = np.log(np.array([0.5, 0.1, 0.3, 0.1]))
        answer = make_answer([1, 2])
        assert answer.rank(lp) == 1
        assert answer.logp(lp) == pytest.approx(math.log(0.4))
        assert answer.anchored(lp).preferred == 2

    def test_a_delta_takes_the_set_on_both_sides(self, model):
        out = ablate_layers(model, [_record("Paris")], {"layers": [2]})
        ids = render(model, {"text": PROMPT}).array
        base = _base_lp(model)
        ablated = read_last_logp(model.run(ids, interventions=[Ablate.layer(2)]).logits)
        want = (np.logaddexp(ablated[SPACED], ablated[BARE])
                - np.logaddexp(base[SPACED], base[BARE]))
        assert out["items"][0]["delta_logp"] == pytest.approx(float(want), abs=1e-4)
        cond = out["conditions"][0]
        assert cond["baseline_logp"] == pytest.approx(
            float(np.logaddexp(base[SPACED], base[BARE])), abs=1e-4)
        assert {v["token"]["id"] for v in cond["variants"]} == {SPACED, BARE}


class TestALogitIsOneSpellings:
    def test_attribution_names_the_preferred_variant_and_sums_to_its_logit(self, model):
        row = attribute_logits(model, [_record("Paris")], {})["items"][0]
        variants = {v["token"]["id"]: v["p"] for v in row["variants"]}
        preferred = max(variants, key=variants.get)
        assert row["target"]["id"] == preferred and len(variants) == 2
        ids = render(model, {"text": PROMPT}).array
        logits = np.array(model.run(ids).logits[0, -1].astype(mx.float32))
        assert row["additivity"]["true_logit"] == pytest.approx(float(logits[preferred]), abs=1e-3)
        assert abs(row["additivity"]["residual"]) < 5e-3


class TestSpellingsThatCoincide:
    def test_one_token_is_one_variant(self, model):
        out = RUNS["logits/read"](model, "7")
        entry = out["items"][0]["tracked"]["answer"]
        assert len(entry["variants"]) == 1
        assert entry["token"]["id"] == PIECES["7"]
        assert entry["p"] == entry["variants"][0]["p"]

    def test_a_whitespace_answer_is_an_error(self, model):
        for text in ("", "   "):
            with pytest.raises(ValueError, match="empty or only whitespace"):
                spell_answer_variants(text)
            with pytest.raises(ValueError, match="empty or only whitespace"):
                encode_answer(model.tokenizer, text)

    def test_a_spelling_that_would_retokenize_the_prompt_is_left_out(self, model):
        tok = model.tokenizer
        prefix = "The capital of France is"
        answer = encode_answer_in_context(tok, "Paris", prefix, tok.encode(prefix))
        assert answer.ids == (SPACED,)


class SplitSpaceTokenizer:
    all_special_ids = (0,)
    names = {1: "Q", 2: ":", 3: " ", 4: "8", 5: "Rome", 6: " Rome"}

    def encode(self, text, add_special_tokens=False):
        ids = {v: k for k, v in self.names.items()}
        return [ids[p] for p in re.findall(r" ?Rome|\d| |Q|:", text)]

    def decode(self, ids):
        return "".join(self.names[int(i)] for i in ids)


def test_a_spelling_that_begins_with_a_bare_space_token_is_not_a_variant():
    tok = SplitSpaceTokenizer()
    assert encode_answer(tok, " 8").ids == (4,)
    assert encode_answer(tok, "8").ids == (4,)
    assert set(encode_answer(tok, "Rome").ids) == {5, 6}


def test_in_context_a_bare_space_token_is_not_a_variant():
    tok = SplitSpaceTokenizer()
    answer = encode_answer_in_context(tok, "8", "Q:", tok.encode("Q:"))
    assert answer.ids == (4,)

