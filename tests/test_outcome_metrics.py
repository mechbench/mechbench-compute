from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mechbench_compute import lexicon
from mechbench_compute import resume as rm
from mechbench_compute.distill import render
from mechbench_compute.interp.answer import make_answer
from mechbench_compute.interp.read_last_logp import read_last_logp
from mechbench_compute.interp.read_metric import read_metric
from mechbench_compute.interp.resolve_outcomes import resolve_outcomes
from mechbench_compute.interventions import Ablate
from mechbench_compute.ops.intervene.ablate_circuit import ablate_circuits
from mechbench_compute.ops.intervene.ablate_heads import ablate_heads
from tests.tiny_models import build_tiny_model

OUTCOMES = ["cat", "dog", "mat"]

RECORDS = [{"id": "a", "user": "the cat sat on a mat", "outcomes": OUTCOMES},
           {"id": "b", "user": "a dog ran and the cat sat", "outcomes": OUTCOMES}]

UNIVERSE = {"points": ["attn.per_head_out"], "layers": [0, 1, 2, 3], "n_heads": 4, "positions": "all"}


@pytest.fixture(scope="module")
def tiny():
    return build_tiny_model("gemma3")


def planted(probs: dict[int, float], vocab: int = 64) -> mx.array:
    rest = (1.0 - sum(probs.values())) / (vocab - len(probs))
    p = np.full(vocab, rest)
    for i, v in probs.items():
        p[i] = v
    return mx.array(np.log(p).astype(np.float32)).reshape(1, 1, vocab)


def by_hand(model, record, heads=()):
    lp = read_last_logp(model.run(render(model, record).array,
                                  interventions=[Ablate.head(layer, h) for layer, h in heads]).logits)
    p = np.exp(np.asarray(lp, dtype=np.float64))[[8, 14, 12]]
    q = p / p.sum()
    return float(-(q * np.log2(q)).sum()), float(p.sum())


def circuit(heads):
    return {"id": "c", "universe": UNIVERSE, "task": {"ids": ["a", "b"]},
            "components": [{"point": "attn.per_head_out", "layer": layer, "head": h, "position": "all"}
                           for layer, h in heads]}


PLANTED = (make_answer([8]), make_answer([14]), make_answer([12]))


class TestOnAPlantedDistribution:

    def test_entropy_is_over_the_outcomes_renormalised(self):
        logits = planted({8: 0.1, 14: 0.1, 12: 0.2})
        q = np.array([0.25, 0.25, 0.5])
        assert read_metric(None, "entropy_outcomes", logits, PLANTED) == pytest.approx(
            float(-(q * np.log2(q)).sum()), abs=1e-5)
        assert read_metric(None, "mass_outcomes", logits, PLANTED) == pytest.approx(0.4, abs=1e-6)

    def test_uniform_over_the_outcomes_is_log2_of_their_count_whatever_the_mass(self):
        for each in (0.3, 0.01):
            logits = planted({8: each, 14: each, 12: each})
            assert read_metric(None, "entropy_outcomes", logits, PLANTED) == pytest.approx(np.log2(3),
                                                                                                abs=1e-5)
            assert read_metric(None, "mass_outcomes", logits, PLANTED) == pytest.approx(3 * each, rel=1e-5)

    def test_an_outcome_carries_both_its_spellings(self):
        logits = planted({8: 0.1, 9: 0.1, 12: 0.2})
        two = [make_answer([8, 9]), make_answer([12])]
        assert read_metric(None, "entropy_outcomes", logits, two) == pytest.approx(1.0, abs=1e-5)


class TestOnTheTinyModel:
    @pytest.mark.parametrize("metric", ["entropy_outcomes", "mass_outcomes"])
    def test_ablate_heads_reads_as_by_hand(self, tiny, metric):
        pick = 0 if metric == "entropy_outcomes" else 1
        out = ablate_heads(tiny, RECORDS, {"layers": [2], "metric": metric})
        base = [by_hand(tiny, r)[pick] for r in RECORDS]
        assert out["metric"] == metric
        assert [c["baseline"] for c in out["conditions"]] == [round(b, 4) for b in base]
        cut = np.mean([by_hand(tiny, r, [(2, 1)])[pick] - b for r, b in zip(RECORDS, base, strict=True)])
        assert out["measures"]["mean_delta"][0][1] == pytest.approx(cut, abs=1e-4)

    @pytest.mark.parametrize("metric", ["entropy_outcomes", "mass_outcomes"])
    def test_ablate_circuit_reads_as_by_hand(self, tiny, metric):
        pick = 0 if metric == "entropy_outcomes" else 1
        everything = [(layer, h) for layer in range(4) for h in range(4)]
        out = ablate_circuits(tiny, [circuit([(1, 0)])], RECORDS, {"metric": metric})
        for c, record in zip(lexicon.items_of(out["cells"]), RECORDS, strict=True):
            assert c["m_full"] == pytest.approx(by_hand(tiny, record)[pick], abs=1e-5)
            assert c["m_without"] == pytest.approx(by_hand(tiny, record, [(1, 0)])[pick], abs=1e-5)
            assert c["m_empty"] == pytest.approx(by_hand(tiny, record, everything)[pick], abs=1e-5)
        assert out["out"]["metric"] == metric

    def test_entropy_is_still_over_the_whole_vocabulary_and_ignores_outcomes(self, tiny):
        plain = [{k: v for k, v in r.items() if k != "outcomes"} for r in RECORDS]
        with_outcomes = ablate_heads(tiny, RECORDS, {"layers": [1], "metric": "entropy"})
        without = ablate_heads(tiny, plain, {"layers": [1], "metric": "entropy"})
        assert with_outcomes["measures"] == without["measures"]
        assert [c["baseline"] for c in with_outcomes["conditions"]] == [c["baseline"] for c in without["conditions"]]
        for c, record in zip(without["conditions"], plain, strict=True):
            lp = read_last_logp(tiny.run(render(tiny, record).array).logits)
            p = np.exp(np.asarray(lp, dtype=np.float64))
            assert c["baseline"] == round(float(-(p * np.log2(p)).sum()), 4)
        a = ablate_circuits(tiny, [circuit([(1, 0)])], RECORDS, {"metric": "entropy"})
        b = ablate_circuits(tiny, [circuit([(1, 0)])], plain, {"metric": "entropy"})
        assert rm.content_hash(a["cells"]) == rm.content_hash(b["cells"])


SPLIT_VOCAB = {1: "six", 2: " s", 3: "ix", 4: "four", 5: " four"}

SPLIT_ENCODINGS = {" six": [2, 3], "six": [2, 3], " four": [5], "four": [4]}


class SplittingTokenizer:
    all_special_ids = (0,)

    def encode(self, text, add_special_tokens=False):
        return SPLIT_ENCODINGS.get(text, [0])

    def decode(self, ids):
        return "".join(SPLIT_VOCAB[i] for i in ids)


class TestRefusals:
    def test_a_record_without_outcomes(self, tiny):
        record = {"id": "bare", "user": "the cat sat"}
        with pytest.raises(ValueError, match="record 'bare' has no `outcomes`"):
            ablate_heads(tiny, [RECORDS[0], record], {"layers": [1], "metric": "entropy_outcomes"})
        with pytest.raises(ValueError, match="record 'bare' has no `outcomes`"):
            ablate_circuits(tiny, [circuit([(1, 0)])], [record], {"metric": "mass_outcomes"})

    def test_an_outcome_the_tokenizer_does_not_know(self, tiny):
        record = {**RECORDS[0], "id": "z", "outcomes": ["cat", "zebra"]}
        with pytest.raises(ValueError, match="record 'z': outcome 'zebra'"):
            ablate_heads(tiny, [record], {"layers": [1], "metric": "entropy_outcomes"})

    def test_an_outcome_split_into_tokens(self):
        model = SimpleNamespace(tokenizer=SplittingTokenizer())
        assert [a.ids for a in resolve_outcomes(model, {"id": "r", "outcomes": ["four"]})] == [(5, 4)]
        with pytest.raises(ValueError, match="record 'r': outcome 'six' is not one token under this tokenizer"):
            resolve_outcomes(model, {"id": "r", "outcomes": ["four", "six"]})

    def test_the_metric_without_outcomes(self):
        with pytest.raises(ValueError, match="reads a record's `outcomes`, and none were given"):
            read_metric(None, "entropy_outcomes", planted({8: 0.5}))
