from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mechbench_compute import attribution, lexicon
from mechbench_compute import resume as rm
from mechbench_compute.distill import render
from mechbench_compute.interp.read_last_logp import read_last_logp
from mechbench_compute.intervene.plan import plan
from mechbench_compute.intervene.spec_error import SpecError
from mechbench_compute.ops import Context
from mechbench_compute.ops.activations import examples as examples_op
from mechbench_compute.ops.activations.examples import find_top_examples
from mechbench_compute.ops.dictionary.encode import encode_records
from mechbench_compute.ops.intervene.apply import run_intervene
from mechbench_compute.ops.logits import attribute as attribute_op
from mechbench_compute.ops.logits.attribute import attribute_logits
from tests.tiny_models import build_tiny_model

D = 32

WIDTH = 6

LAYER = 2

NEVER = 5

RECORDS = [{"id": "a", "user": "the cat sat on a mat"}, {"id": "b", "user": "a dog ran and the cat sat"}]


def build_dictionary(point="resid_post", seed=3):
    rng = np.random.default_rng(seed)
    w_enc = rng.normal(size=(D, WIDTH)).astype(np.float32) * 0.3
    w_dec = rng.normal(size=(WIDTH, D)).astype(np.float32)
    w_dec /= np.linalg.norm(w_dec, axis=1, keepdims=True)
    b_enc = rng.normal(size=WIDTH).astype(np.float32) * 0.1
    threshold = np.full(WIDTH, 0.05, dtype=np.float32)
    threshold[NEVER] = 1e9
    b_dec = rng.normal(size=D).astype(np.float32) * 0.1
    space = {"model": "tiny/gemma3", "layer": LAYER, "point": point, "head": None, "d": D}
    items = [{"id": f"f{i}", "index": i, "vector": [float(x) for x in w_dec[i]],
              "encoder": [float(x) for x in w_enc[:, i]], "norm": float(np.linalg.norm(w_dec[i])),
              "b_enc": float(b_enc[i]), "threshold": float(threshold[i])} for i in range(WIDTH)]
    dictionary = lexicon.collection(
        "direction/dictionary", items, derivation="sae", reads=[space], writes=[space],
        model={"id": "tiny/gemma3", "architecture": "gemma3"}, width=WIDTH, d_in=D, d_out=D,
        activation={"fn": "jumprelu"}, b_dec=[float(x) for x in b_dec],
        source={"hub": {"repo": "you/tiny-scope", "path": "resid_post/layer_2", "revision": "main",
                        "commit": "c" * 40, "files": {}}})
    return dictionary, {"w_enc": w_enc, "w_dec": w_dec, "b_enc": b_enc, "threshold": threshold, "b_dec": b_dec}


@pytest.fixture(scope="module")
def tiny():
    return build_tiny_model("gemma3")


@pytest.fixture(scope="module")
def sae():
    return build_dictionary()


def read_point(model, record, name=f"blocks.{LAYER}.resid_post"):
    r = render(model, record)
    acts = np.array(model.run(r.array, capture=[name]).cache[name].astype(mx.float32))[0]
    return r, acts


def encode_by_hand(x, w, i):
    pre = x @ w["w_enc"][:, i] + w["b_enc"][i]
    return np.where(pre > w["threshold"][i], pre, 0.0)


class TestExamplesOnAFeature:
    def test_the_strip_is_the_jumprelu_encoding_and_the_windows_rank_by_it(self, tiny, sae):
        dictionary, w = sae
        out = find_top_examples(tiny, RECORDS, {"k": 4, "window": 2, "feature": {"index": 1}},
                                dictionary=dictionary)
        by_hand = {}
        for record in RECORDS:
            r, acts = read_point(tiny, record)
            values = encode_by_hand(acts, w, 1)
            values[0] = 0.0
            by_hand[record["id"]] = values
        for it in out["items"]:
            values = by_hand[it["coords"]["record"]]
            pos = it["coords"]["position"]
            assert it["value"] == pytest.approx(values[pos], abs=1e-4)
            lo = pos - it["hit"]
            assert it["values"] == pytest.approx(list(values[lo:lo + len(it["values"])]), abs=1e-4)
        best = max(float(v.max()) for v in by_hand.values())
        assert out["items"][0]["value"] == pytest.approx(best, abs=1e-4) and best > 0
        assert [it["value"] for it in out["items"]] == sorted((it["value"] for it in out["items"]), reverse=True)

    def test_it_agrees_with_dictionary_encode(self, tiny, sae):
        dictionary, _ = sae
        out = find_top_examples(tiny, RECORDS, {"k": 50, "window": 0, "feature": {"index": 2}},
                                dictionary=dictionary)
        encoded = encode_records(tiny, RECORDS, dictionary, {"features": [2]})
        fired = {(it["id"], it["position"]): it["value"] for it in lexicon.items_of(encoded)}
        kept = {(it["coords"]["record"], it["coords"]["position"]): it["value"] for it in out["items"]
                if it["value"] != 0}
        assert fired and kept == pytest.approx({k: round(v, 5) for k, v in fired.items()}, abs=1e-4)

    def test_the_readout_names_the_dictionary_and_the_feature(self, tiny, sae):
        dictionary, _ = sae
        out = find_top_examples(tiny, RECORDS, {"k": 2, "feature": {"index": 3, "dictionary": dictionary}})
        assert out["feature"]["index"] == 3 and "neuron" not in out
        named = out["feature"]["dictionary"]
        assert named["hash"] == rm.content_hash(dictionary) and named["width"] == WIDTH
        assert named["derivation"] == "sae" and named["reads"][0]["layer"] == LAYER
        assert (out["layer"], out["point"]) == (LAYER, "resid_post")
        assert "feature 3 of the 6-wide sae at resid_post layer 2" in out["description"]

    def test_run_reads_the_dictionary_port(self, tiny, sae):
        out = examples_op.run(Context(loaded=tiny), {"records": RECORDS, "dictionary": sae[0]},
                              {"k": 1, "feature": {"index": 0}})
        assert out["feature"]["index"] == 0

    def test_a_feature_and_a_neuron_are_one_address_too_many(self, tiny, sae):
        with pytest.raises(ValueError, match="one address in two bases"):
            find_top_examples(tiny, RECORDS, {"feature": {"index": 0}, "neuron": {"layer": 1, "index": 0}},
                              dictionary=sae[0])
        with pytest.raises(ValueError, match="exactly one of them"):
            find_top_examples(tiny, RECORDS, {"feature": {"index": 0}}, dictionary=sae[0],
                              direction={"kind": "direction/vector"})

    def test_a_feature_is_read_where_its_dictionary_reads(self, tiny, sae):
        with pytest.raises(ValueError, match="leave out `layer` and `point`"):
            find_top_examples(tiny, RECORDS, {"feature": {"index": 0}, "layer": 1}, dictionary=sae[0])
        with pytest.raises(ValueError, match="names no dictionary"):
            find_top_examples(tiny, RECORDS, {"feature": {"index": 0}})
        with pytest.raises(ValueError, match="feature 9 is not in the dictionary"):
            find_top_examples(tiny, RECORDS, {"feature": {"index": 9}}, dictionary=sae[0])


class TestSteeringOnAFeature:
    def capture(self, model, params, inputs):
        out = run_intervene(model, RECORDS[:1], {**params, "readout": {
            "type": "capture", "points": [f"blocks.{LAYER}.resid_post"], "position": "last"}}, inputs=inputs)
        return {it["factor"]: np.asarray(it["vector"]) for it in out["items"]}, out

    def test_add_is_strength_times_the_decoder_row_where_the_dictionary_writes(self, tiny, sae):
        dictionary, w = sae
        got, _ = self.capture(tiny, {"spec": [{"feature": {"index": 4}, "op": "add", "strength": 3.0}]},
                              {"dictionary": dictionary})
        np.testing.assert_allclose(got[1.0] - got[0.0], 3.0 * w["w_dec"][4], atol=1e-4)

    def test_it_is_the_same_run_as_the_decoder_row_given_as_a_direction(self, tiny, sae):
        dictionary, w = sae
        row = {"kind": "direction/vector", "vector": [float(x) for x in w["w_dec"][4]],
               "space": {"model": None, "layer": LAYER, "point": "resid_post", "head": None, "d": D}}
        spec = {"op": "add", "strength": 2.0, "positions": "all"}
        params = {"tracked": {"answer": "mat"}, "sweep": {"strength": [1.0, 2.0]}}
        by_feature = run_intervene(tiny, RECORDS, {**params, "spec": [{**spec, "feature": {"index": 4}}]},
                                   inputs={"dictionary": dictionary})
        by_hand = run_intervene(tiny, RECORDS, {**params, "spec": [{
            **spec, "point": "resid_post", "layers": [LAYER], "direction": row}]})
        assert by_feature["items"] == by_hand["items"]

    def test_the_plan_names_the_dictionary_by_hash(self, tiny, sae):
        dictionary, _ = sae
        params = {"spec": [{"feature": {"index": 4, "dictionary": dictionary}, "op": "add", "strength": 3.0}]}
        spec = plan(tiny, params, {}).header()["spec"][0]
        assert spec["feature"]["index"] == 4
        assert spec["feature"]["dictionary"]["hash"] == rm.content_hash(dictionary)
        assert spec["feature"]["dictionary"]["source"]["hub"]["repo"] == "you/tiny-scope"
        assert (spec["point"], spec["layers"]) == ("resid_post", [LAYER])
        assert spec["direction"]["derivation"] == {"method": "dictionary/feature", "index": 4,
                                                   "model": "tiny/gemma3"}
        _, out = self.capture(tiny, {"spec": [{"feature": {"index": 4}, "op": "add"}]},
                              {"dictionary": dictionary})
        assert out["spec"][0]["feature"] == spec["feature"]

    @pytest.mark.parametrize(("item", "match"), [
        ({"neurons": [1]}, "one address in two bases"),
        ({"op": "zero"}, "op 'zero' takes none"),
        ({"point": "mlp_out"}, "writes at resid_post"),
        ({"layers": [1]}, "writes at layer 2"),
        ({"direction": {"kind": "direction/vector"}}, "leave `direction` out"),
    ])
    def test_what_a_feature_item_refuses(self, tiny, sae, item, match):
        with pytest.raises(SpecError, match=match):
            run_intervene(tiny, RECORDS, {"spec": [{"op": "add", "feature": {"index": 0}, **item}]},
                          inputs={"dictionary": sae[0]})


class TestAttributingToFeatures:
    PARAMS = {"tracked": {"answer": "mat", "other": "dog"}}

    def by_hand(self, model, record, w):
        names = [f"blocks.{LAYER}.resid_post", "final_norm.scale"]
        r = render(model, record)
        res = model.run(r.array, capture=names)
        x = np.array(res.cache[names[0]].astype(mx.float32))[0, -1]
        ln_scale = np.array(res.cache["final_norm.scale"].astype(mx.float32)).reshape(-1)
        out = attribute_logits(model, [record], self.PARAMS)["items"][0]
        targets = [out["target"]["id"], out["contrast"]["id"]]
        f = np.array([encode_by_hand(x, w, i) for i in range(WIDTH)])
        attrs = attribution.logit_attrs(model, w["w_dec"][:, None, :], targets, apply_ln=True,
                                        ln_scale=ln_scale)
        return f, attrs[:, 0] - attrs[:, 1]

    def test_each_contribution_is_activation_times_the_decoder_rows_dla(self, tiny, sae):
        dictionary, w = sae
        out = attribute_logits(tiny, RECORDS, {**self.PARAMS, "top_features": 3}, dictionary=dictionary)
        assert out["dictionary"]["hash"] == rm.content_hash(dictionary)
        for record, row in zip(RECORDS, out["items"], strict=True):
            f, dla = self.by_hand(tiny, record, w)
            fired = [i for i in range(WIDTH) if f[i] != 0]
            want = sorted(fired, key=lambda i: -abs(f[i] * dla[i]))[:3]
            assert [e["index"] for e in row["features"]] == want and NEVER not in want
            for e in row["features"]:
                i = e["index"]
                assert e["activation"] == pytest.approx(f[i], abs=1e-3)
                assert e["dla"] == pytest.approx(dla[i], abs=1e-3)
                assert e["contribution"] == pytest.approx(f[i] * dla[i], abs=1e-3)
            rec = row["reconstruction"]
            assert rec["features"] == pytest.approx(sum(f[i] * dla[i] for i in fired), abs=1e-3)
            assert rec["features"] + rec["b_dec"] + rec["error"] == pytest.approx(rec["stream"], abs=1e-3)

    def test_one_feature_is_attributed_firing_or_not(self, tiny, sae):
        dictionary, w = sae
        out = attribute_logits(tiny, RECORDS[:1], {**self.PARAMS, "feature": {"index": NEVER}},
                               dictionary=dictionary)
        row = out["items"][0]
        _, dla = self.by_hand(tiny, RECORDS[0], w)
        assert row["features"] == [{"index": NEVER, "activation": 0.0, "dla": round(float(dla[NEVER]), 4),
                                    "contribution": 0.0}]
        assert "reconstruction" not in row and out["feature"]["index"] == NEVER
        assert out["feature"]["dictionary"]["hash"] == rm.content_hash(dictionary)

    def test_the_layer_decomposition_is_untouched_beside_it(self, tiny, sae):
        plain = attribute_logits(tiny, RECORDS, self.PARAMS)
        with_features = attribute_op.run(Context(loaded=tiny), {"records": RECORDS, "dictionary": sae[0]},
                                         self.PARAMS)
        for a, b in zip(plain["items"], with_features["items"], strict=True):
            assert {k: v for k, v in b.items() if k not in ("features", "reconstruction")} == a

    def test_a_dictionary_that_does_not_write_the_residual_stream_is_refused(self, tiny):
        dictionary, _ = build_dictionary(point="attn.o_in")
        with pytest.raises(ValueError, match="takes the residual stream"):
            attribute_logits(tiny, RECORDS, self.PARAMS, dictionary=dictionary)


def scale_neurons(factor, neurons):
    def hook(act, info):
        mask = np.ones(act.shape[-1], dtype=np.float32)
        mask[neurons] = factor
        return act * mx.array(mask).astype(act.dtype)
    return hook


class TestTheNeuronFormsAgainstAComputationByHand:
    def test_examples_on_a_neuron(self, tiny):
        out = find_top_examples(tiny, RECORDS, {"k": 3, "window": 2, "sign": "both", "point": "resid_post",
                                                "neuron": {"layer": 2, "index": 5}})
        name = "blocks.2.resid_post"
        acts = {r["id"]: np.array(tiny.run(render(tiny, r).array, capture=[name]).cache[name][0, :, 5]
                                  .astype(mx.float32)) for r in RECORDS}
        ranked = sorted((float(v), rid, p) for rid, a in acts.items() for p, v in enumerate(a))
        want = {"high": [f"{rid}:{p}" for _v, rid, p in ranked[::-1][:3]],
                "low": [f"{rid}:{p}" for _v, rid, p in ranked[:3]]}
        for side, ids in want.items():
            assert [i["id"] for i in out["items"] if i["coords"]["side"] == side] == ids
        for item in out["items"]:
            a, p = acts[item["coords"]["record"]], item["coords"]["position"]
            assert item["value"] == pytest.approx(float(a[p]), abs=1e-4)
            start = p - item["hit"]
            assert item["values"] == pytest.approx([float(v) for v in a[start:start + len(item["values"])]],
                                                   abs=1e-4)
        assert out["over"]["n_tokens"] == sum(len(a) for a in acts.values())
        assert (out["layer"], out["neuron"], out["point"]) == (2, 5, "resid_post")

    def test_attribute_without_a_dictionary(self, tiny):
        out = attribute_logits(tiny, RECORDS, {"tracked": {"answer": "mat", "other": "dog"}})
        assert out["components"] == ["embed", "L0", "L1", "L2", "L3"]
        for item, record in zip(out["items"], RECORDS, strict=True):
            lp = read_last_logp(tiny.run(render(tiny, record).array).logits)
            assert (item["target"]["id"], item["contrast"]["id"]) == (12, 14)
            assert item["additivity"]["true_logit"] == pytest.approx(float(lp[12] - lp[14]), abs=2e-3)
            assert sum(item["measures"]["contribution"]) == pytest.approx(item["additivity"]["true_logit"],
                                                                          abs=2e-3)

    def test_apply_on_neurons(self, tiny):
        out = run_intervene(tiny, RECORDS, {
            "spec": [{"point": "resid_post", "layers": [1], "neurons": [3, 7], "op": "scale", "strength": 4.0,
                      "positions": "all"}],
            "sweep": {"strength": [0.5, 1.0]}, "tracked": {"answer": "mat"}})
        assert [(i["id"], i["factor"]) for i in out["items"]] == [
            (r["id"], f) for r in RECORDS for f in (0.0, 0.5, 1.0)]
        for item in out["items"]:
            ids = render(tiny, next(r for r in RECORDS if r["id"] == item["id"])).array
            hooks = {"blocks.1.resid_post": scale_neurons(4.0 * item["factor"], [3, 7])} if item["factor"] else {}
            lp = read_last_logp(tiny.run(ids, hooks=hooks).logits)
            assert item["tracked"]["answer"]["logp"] == pytest.approx(float(lp[12]), abs=1e-3)
