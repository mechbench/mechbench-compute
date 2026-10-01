from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mechbench_compute import shapes as S
from mechbench_compute.distill import render
from mechbench_compute.intervene.spec_error import SpecError
from mechbench_compute.ops.activations.examples import find_top_examples
from mechbench_compute.ops.intervene.apply import run_intervene
from mechbench_compute.ops.logits.attribute import attribute_logits
from tests.test_dictionary_features import (
    LAYER,
    RECORDS,
    D,
    build_dictionary,
    encode_by_hand,
    read_point,
)
from tests.tiny_models import build_tiny_model

CORPUS = RECORDS + [{"id": "c", "user": "the mat was on the cat and the dog sat"},
                    {"id": "d", "user": "a cat a dog a mat a cat"}]

FEATURES = [1, 2, 4]


@pytest.fixture(scope="module")
def tiny():
    return build_tiny_model("gemma3")


@pytest.fixture(scope="module")
def sae():
    return build_dictionary()


def strip_feature(item):
    record, position, _feature = item["id"].rsplit(":", 2)
    coords = {k: v for k, v in item["coords"].items() if k != "feature"}
    return {**item, "id": f"{record}:{position}", "coords": coords}


def fired_by_hand(model, w, index):
    out = {}
    for record in CORPUS:
        _r, acts = read_point(model, record)
        values = encode_by_hand(acts, w, index)
        values[0] = 0.0
        out[record["id"]] = values
    return out


def direction():
    rng = np.random.default_rng(11)
    v = rng.normal(size=D).astype(np.float32)
    return {"kind": "direction/vector", **S.vector(v, S.space(model="tiny/gemma3", layer=LAYER,
                                                              point="resid_post", d=D)),
            "derivation": {"method": "test"}}


class TestManyFeaturesInOnePass:
    @pytest.mark.parametrize("extra", [{"sign": "high"}, {"sign": "both", "per_record": 2},
                                       {"sign": "random", "seed": 7}])
    def test_each_strip_is_the_single_form_for_that_feature(self, tiny, sae, extra):
        params = {"k": 3, "window": 2, **extra}
        many = find_top_examples(tiny, CORPUS, {**params, "features": FEATURES}, dictionary=sae[0])
        assert many["features"]["indices"] == FEATURES and "feature" not in many
        for j, over in zip(FEATURES, many["over"], strict=True):
            one = find_top_examples(tiny, CORPUS, {**params, "feature": {"index": j}}, dictionary=sae[0])
            mine = [it for it in many["items"] if it["coords"]["feature"] == j]
            assert mine and all(it["id"].endswith(f":{j}") for it in mine)
            assert [strip_feature(it) for it in mine] == one["items"]
            assert over == {"feature": j, **one["over"]}
        assert [it["coords"]["feature"] for it in many["items"]] == sorted(
            (it["coords"]["feature"] for it in many["items"]), key=FEATURES.index)

    def test_the_corpus_is_read_once(self, tiny, sae, monkeypatch):
        calls = []
        real = tiny.run
        monkeypatch.setattr(tiny, "run", lambda *a, **kw: calls.append(1) or real(*a, **kw))
        find_top_examples(tiny, CORPUS, {"k": 2, "features": FEATURES}, dictionary=sae[0])
        assert len(calls) == len(CORPUS)

    @pytest.mark.parametrize(("features", "match"), [
        ([], "non-empty list"), ([1, 1], "names a feature twice"), ([True], "non-empty list"),
        ([9], "feature 9 is not in the dictionary")])
    def test_what_the_list_refuses(self, tiny, sae, features, match):
        with pytest.raises(ValueError, match=match):
            find_top_examples(tiny, CORPUS, {"features": features}, dictionary=sae[0])

    def test_one_feature_and_several_are_one_address_too_many(self, tiny, sae):
        with pytest.raises(ValueError, match="give one of them"):
            find_top_examples(tiny, CORPUS, {"feature": {"index": 1}, "features": [2]}, dictionary=sae[0])


class TestPerRecord:
    def test_no_record_gives_more_than_the_cap_and_each_gives_its_best(self, tiny, sae):
        dictionary, w = sae
        out = find_top_examples(tiny, CORPUS, {"k": 10, "window": 0, "per_record": 1,
                                               "feature": {"index": 1}}, dictionary=dictionary)
        by_hand = fired_by_hand(tiny, w, 1)
        records = [it["coords"]["record"] for it in out["items"]]
        assert sorted(records) == sorted(r["id"] for r in CORPUS)
        for it in out["items"]:
            assert it["value"] == pytest.approx(float(by_hand[it["coords"]["record"]].max()), abs=1e-4)
        assert out["per_record"] == 1

    def test_a_cap_above_k_changes_nothing(self, tiny, sae):
        params = {"k": 3, "feature": {"index": 2}}
        plain = find_top_examples(tiny, CORPUS, params, dictionary=sae[0])
        capped = find_top_examples(tiny, CORPUS, {**params, "per_record": 3}, dictionary=sae[0])
        assert capped["items"] == plain["items"] and "per_record" not in plain

    def test_the_cap_is_a_whole_number(self, tiny, sae):
        with pytest.raises(ValueError, match="per_record"):
            find_top_examples(tiny, CORPUS, {"per_record": 0, "feature": {"index": 1}}, dictionary=sae[0])


class TestRandomWindows:
    def test_a_seed_reproduces_the_draw_and_another_changes_it(self, tiny, sae):
        params = {"k": 4, "sign": "random", "feature": {"index": 1}}
        a = find_top_examples(tiny, CORPUS, {**params, "seed": 3}, dictionary=sae[0])
        b = find_top_examples(tiny, CORPUS, {**params, "seed": 3}, dictionary=sae[0])
        c = find_top_examples(tiny, CORPUS, {**params, "seed": 4}, dictionary=sae[0])
        assert a["items"] == b["items"] and len(a["items"]) == 4
        assert [it["id"] for it in a["items"]] != [it["id"] for it in c["items"]]
        assert (a["sign"], a["seed"]) == ("random", 3)
        assert [it["rank"] for it in a["items"]] == [0, 1, 2, 3]

    def test_the_windows_lie_where_the_feature_fires(self, tiny, sae):
        dictionary, w = sae
        by_hand = fired_by_hand(tiny, w, 1)
        fired = {(rid, p) for rid, v in by_hand.items() for p in np.flatnonzero(v)}
        assert fired
        some = find_top_examples(tiny, CORPUS, {"k": 3, "sign": "random", "seed": 1,
                                                "feature": {"index": 1}}, dictionary=dictionary)
        assert {(it["coords"]["record"], it["coords"]["position"]) for it in some["items"]} <= fired
        every = find_top_examples(tiny, CORPUS, {"k": 1000, "sign": "random", "feature": {"index": 1}},
                                  dictionary=dictionary)
        assert {(it["coords"]["record"], it["coords"]["position"]) for it in every["items"]} == fired

    def test_a_direction_draws_from_every_position_but_the_first(self, tiny):
        out = find_top_examples(tiny, CORPUS, {"k": 1000, "sign": "random"}, direction=direction())
        lengths = {r["id"]: len(render(tiny, r).ids) for r in CORPUS}
        want = {(rid, p) for rid, n in lengths.items() for p in range(1, n)}
        assert {(it["coords"]["record"], it["coords"]["position"]) for it in out["items"]} == want

    def test_per_record_caps_the_draw(self, tiny):
        out = find_top_examples(tiny, CORPUS, {"k": 1000, "sign": "random", "per_record": 2},
                                direction=direction())
        records = [it["coords"]["record"] for it in out["items"]]
        assert sorted(records) == sorted(r["id"] for r in CORPUS for _ in range(2))


def projections(model, vec):
    name = f"blocks.{LAYER}.resid_post"
    out = {}
    for record in CORPUS:
        r = render(model, record)
        act = model.run(r.array, capture=[name]).cache[name][0].astype(mx.float32)
        out[record["id"]] = (np.array(mx.sum(act * mx.array(vec), axis=-1)), r)
    return out


class TestSkipBos:
    def test_the_first_position_is_left_out_by_default(self, tiny):
        assert tiny.tokenizer.bos_token_id is not None
        assert all(int(render(tiny, r).ids[0]) == tiny.tokenizer.bos_token_id for r in CORPUS)
        out = find_top_examples(tiny, CORPUS, {"k": 1000, "window": 3, "sign": "both"},
                                direction=direction())
        assert out["skip_bos"] is True
        assert all(it["coords"]["position"] >= 1 for it in out["items"])
        assert all(it["coords"]["position"] - it["hit"] >= 1 for it in out["items"])
        total = sum(len(render(tiny, r).ids) for r in CORPUS)
        assert out["over"]["n_tokens"] == total - len(CORPUS)
        kept = find_top_examples(tiny, CORPUS, {"k": 1000, "sign": "both", "skip_bos": False},
                                 direction=direction())
        assert any(it["coords"]["position"] == 0 for it in kept["items"])
        assert kept["over"]["n_tokens"] == total and kept["skip_bos"] is False

    def test_the_neuron_form_leaves_it_out_too(self, tiny):
        out = find_top_examples(tiny, CORPUS, {"k": 1000, "point": "resid_post",
                                               "neuron": {"layer": LAYER, "index": 3}})
        assert out["skip_bos"] is True and all(it["coords"]["position"] >= 1 for it in out["items"])

    def test_without_it_the_direction_form_is_the_old_ranking_by_hand(self, tiny):
        d = direction()
        out = find_top_examples(tiny, CORPUS, {"k": 5, "window": 2, "sign": "both", "skip_bos": False},
                                direction=d)
        by_hand = projections(tiny, np.asarray(d["vector"], dtype=np.float32))
        ranked = [(float(v[p]), rid, p) for rid, (v, _r) in by_hand.items() for p in range(len(v))]
        high = sorted(ranked, key=lambda t: -t[0])[:5]
        low = sorted(ranked, key=lambda t: t[0])[:5]
        got = [(it["coords"]["side"], it["id"], it["value"]) for it in out["items"]]
        assert got == ([("high", f"{rid}:{p}", round(v, 5)) for v, rid, p in high]
                       + [("low", f"{rid}:{p}", round(v, 5)) for v, rid, p in low])
        for it in out["items"]:
            v, _r = by_hand[it["coords"]["record"]]
            a = it["coords"]["position"] - it["hit"]
            assert it["values"] == [round(float(x), 5) for x in v[a:a + len(it["values"])]]
        allv = np.concatenate([v for v, _r in by_hand.values()])
        assert out["over"]["n_tokens"] == allv.size
        assert out["over"]["max"] == round(float(allv.max()), 5)
        assert out["over"]["min"] == round(float(allv.min()), 5)

    def test_a_feature_reads_the_first_position_as_zero_either_way(self, tiny, sae):
        params = {"k": 1000, "sign": "low", "feature": {"index": 1}}
        a = find_top_examples(tiny, CORPUS, params, dictionary=sae[0])
        b = find_top_examples(tiny, CORPUS, {**params, "skip_bos": False}, dictionary=sae[0])
        assert a["items"] == b["items"] and "skip_bos" not in a


class TestTheDictionaryArrivesOnThePortOnly:
    def test_examples_refuses_the_inline_form(self, tiny, sae):
        with pytest.raises(ValueError, match="`dictionary` port"):
            find_top_examples(tiny, CORPUS, {"feature": {"index": 1, "dictionary": sae[0]}},
                              dictionary=sae[0])

    def test_a_spec_item_refuses_the_inline_form(self, tiny, sae):
        with pytest.raises(SpecError, match="`dictionary` port"):
            run_intervene(tiny, CORPUS, {"spec": [{"op": "add", "feature": {"index": 1,
                                                                            "dictionary": sae[0]}}]},
                          inputs={"dictionary": sae[0]})

    def test_attribute_refuses_the_inline_form(self, tiny, sae):
        with pytest.raises(ValueError, match="`dictionary` port"):
            attribute_logits(tiny, CORPUS[:1], {"tracked": {"answer": "mat"},
                                                "feature": {"index": 1, "dictionary": sae[0]}},
                             dictionary=sae[0])
