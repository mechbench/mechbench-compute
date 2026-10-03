from __future__ import annotations

import numpy as np
import pytest

from mechbench_compute.architectures import BY_MODEL_TYPE
from mechbench_compute.arrays import read_f64
from mechbench_compute.distill import (
    expand_top_outcomes_cached,
    prefill_decision,
    render,
)
from mechbench_compute.interp.capped import SATURATED_AT
from mechbench_compute.ops import Context
from mechbench_compute.ops.activations.capture import capture_residual_vectors
from mechbench_compute.ops.intervene.apply import run_intervene
from mechbench_compute.ops.intervene.steer import steer_inject
from mechbench_compute.ops.logits import read as read_op
from mechbench_compute.ops.logits import read_layers as read_layers_op
from mechbench_compute.ops.logits.attribute import attribute_logits
from tests.tiny_models import KIT_MODELS, SMALL_CAP, build_tiny_model

MODEL_TYPE_OF = dict(KIT_MODELS)

CAPPED = sorted(name for name, t in KIT_MODELS if t == "gemma4")

RECORD = {"id": "r", "text": "the cat sat on a"}

WORDS = ("the", "cat", "sat", "on", "a", "mat", "and", "dog", "ran")

TRACKED = {w: w for w in WORDS}

PLAIN = {"token", "p", "logp", "rank", "variants"}

LABELLED = [{"id": f"v{i}", "text": text, "coords": {"label": label}}
            for i, (text, label) in enumerate([("the cat sat", "pos"), ("a cat ran", "pos"),
                                               ("the dog sat", "neg"), ("a dog ran", "neg")])]


@pytest.fixture(scope="module", params=sorted(MODEL_TYPE_OF))
def tiny(request):
    return build_tiny_model(request.param, BY_MODEL_TYPE[MODEL_TYPE_OF[request.param]])


def build(name: str):
    return build_tiny_model(name, BY_MODEL_TYPE[MODEL_TYPE_OF[name]])


def read_by_hand(model, record=RECORD):
    result = model.run(render(model, record).array, capture=["final_norm"])
    unembed = model.architecture.attribution_unembed(model._model)
    post = read_f64(result.logits)[0, -1]
    pre = read_f64(unembed.project(result.cache["final_norm"][:, -1:, :])).reshape(-1)
    return unembed.softcap, post, pre


def check_tracked(entries, softcap, post, pre) -> None:
    for name, entry in entries.items():
        if softcap is None:
            assert set(entry) == PLAIN, name
            continue
        tid = entry["token"]["id"]
        assert entry["precap_logit"] == pytest.approx(pre[tid], abs=1e-4), name
        assert entry["logit"] == pytest.approx(post[tid], abs=1e-4), name
        assert entry["saturated"] == (abs(pre[tid]) > SATURATED_AT * softcap), name
        assert softcap * np.tanh(entry["precap_logit"] / softcap) == pytest.approx(entry["logit"], abs=2e-4)


def test_a_read_reports_each_answer_s_logit_before_the_cap_on_a_capped_model_only(tiny):
    out = read_op.run(Context(loaded=tiny), {"conditions": [RECORD]}, {"tracked": TRACKED})
    softcap, post, pre = read_by_hand(tiny)
    assert out.get("softcap") == softcap
    check_tracked(out["items"][0]["tracked"], softcap, post, pre)


def test_the_flag_trips_where_a_small_cap_flattens_the_logit_and_not_elsewhere():
    model = build("gemma4-small-cap")
    out = read_op.run(Context(loaded=model), {"conditions": [RECORD]}, {"tracked": TRACKED})
    _, _, pre = read_by_hand(model)
    entries = out["items"][0]["tracked"]
    flags = {w: entries[w]["saturated"] for w in WORDS}
    assert out["softcap"] == SMALL_CAP
    assert flags == {w: abs(pre[entries[w]["token"]["id"]]) > SATURATED_AT * SMALL_CAP for w in WORDS}
    assert any(flags.values()) and not all(flags.values())
    hot = max(WORDS, key=lambda w: abs(entries[w]["precap_logit"]))
    assert abs(entries[hot]["precap_logit"]) > 4 * SMALL_CAP > 2 * abs(entries[hot]["logit"])


@pytest.mark.parametrize("name", CAPPED)
def test_a_capped_read_runs_the_model_s_own_forward_bit_for_bit(name):
    model = build(name)
    ids = render(model, RECORD).ids
    plain = prefill_decision(model, ids)
    capped, _ = read_op.read_capped_prefill(model, ids)
    assert np.array_equal(read_f64(capped[1]), read_f64(plain[1]))
    cfg = {"top_k": 3, "max_tokens": 3, "max_forwards": 16, "floor": 0.0001, "terminators": ["mat"]}
    assert (expand_top_outcomes_cached(model, model.tokenizer, ids, cfg, prefill=capped)
            == expand_top_outcomes_cached(model, model.tokenizer, ids, cfg, prefill=plain))


def test_attribution_names_the_cap_only_where_there_is_one(tiny):
    softcap, post, pre = read_by_hand(tiny)
    out = attribute_logits(tiny, [RECORD], {})
    row = out["items"][0]
    target = row["target"]["id"]
    assert out.get("softcap") == softcap
    assert ("before the cap" in out["description"]) == (softcap is not None)
    assert row["additivity"]["true_logit"] == pytest.approx(pre[target], abs=1e-3)
    assert abs(row["additivity"]["residual"]) < 5e-3
    if softcap is None:
        assert set(row["additivity"]) == {"summed", "true_logit", "residual"}
    else:
        assert row["additivity"]["capped_logit"] == pytest.approx(post[target], abs=1e-3)


def test_attribution_behind_a_small_cap_sums_to_the_logit_before_it_on_a_saturated_token():
    model = build("gemma4-small-cap")
    softcap, post, pre = read_by_hand(model)
    hot, cold = int(np.argmax(pre)), int(np.argmin(pre))
    assert min(pre[hot], -pre[cold]) > 8 * softcap
    out = attribute_logits(model, [{**RECORD, "target": {"id": hot}},
                                   {**RECORD, "id": "pair", "target": {"id": hot}, "contrast": {"id": cold}}],
                           {"split": "sublayer"})
    one, pair = (row["additivity"] for row in out["items"])
    assert out["softcap"] == SMALL_CAP
    assert one["true_logit"] == pytest.approx(pre[hot], abs=1e-3)
    assert one["capped_logit"] == pytest.approx(post[hot], abs=1e-3)
    assert pair["true_logit"] == pytest.approx(pre[hot] - pre[cold], abs=1e-3)
    assert pair["capped_logit"] == pytest.approx(post[hot] - post[cold], abs=1e-3)
    assert abs(one["residual"]) < 5e-3 and abs(pair["residual"]) < 5e-3
    inverted = softcap * np.arctanh(np.clip(post[hot] / softcap, -0.999999, 0.999999))
    assert abs(inverted - pre[hot]) > 0.5


def test_the_decision_readouts_report_the_logit_before_the_cap_on_a_capped_model_only(tiny):
    softcap, post, pre = read_by_hand(tiny)
    applied = run_intervene(tiny, [RECORD], {
        "spec": [{"point": "resid_post", "layers": [1], "op": "scale", "strength": 0.5, "positions": "all"}],
        "sweep": {"strength": [1.0]}, "tracked": TRACKED})
    vectors = capture_residual_vectors(tiny, LABELLED, {"layers": [1], "point": "resid_post"})
    steered = steer_inject(tiny, [RECORD], {
        "layer": 1, "alphas": [0.0, 2.0], "direction": {"positive": "pos", "negative": "neg"},
        "tracked": TRACKED}, inputs={"vectors": vectors})
    lens = read_layers_op.run(Context(loaded=tiny), {"records": [RECORD]}, {"tracked": TRACKED})
    for out in (applied, steered, lens):
        assert out.get("softcap") == softcap
    assert applied["items"][0]["factor"] == steered["items"][0]["factor"] == 0.0
    for control in (applied["items"][0], steered["items"][0], lens["items"][-1]):
        check_tracked(control["tracked"], softcap, post, pre)
    for out in (applied, steered, lens):
        for item in out["items"]:
            for entry in item["tracked"].values():
                assert ({"logit", "precap_logit", "saturated"} <= set(entry)) == (softcap is not None)
