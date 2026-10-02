from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mechbench_compute import attribution
from mechbench_compute import directions as dirs
from mechbench_compute import shapes as S
from mechbench_compute.architectures import BY_MODEL_TYPE
from mechbench_compute.distill import render
from mechbench_compute.interp.point_refused import PointRefused
from mechbench_compute.lexicon import kinds as K
from mechbench_compute.ops.activations.differentiate import (
    METRICS,
    differentiate_activations,
    differentiate_record,
)
from mechbench_compute.protocol import ProtocolExecutor
from tests.test_reference_adapters import adapted_ref, build_payload
from tests.tiny_models import MODEL_TYPES, build_tiny_model

TARGET, CONTRAST = 12, 14

OUTCOME_IDS = ([8], [14], [12])

RECORD = {"id": "r", "text": "the cat sat on a", "tracked": {"t": {"id": TARGET}, "c": {"id": CONTRAST}},
          "outcomes": ["cat", "dog", "mat"]}

EPS = 1e-2


@pytest.fixture(scope="module", params=MODEL_TYPES)
def tiny(request):
    return build_tiny_model(request.param, BY_MODEL_TYPE[request.param])


def differentiate(model, **params):
    return differentiate_activations(model, [RECORD], params)


def read_last(result, name: str) -> np.ndarray:
    return np.array(result.cache[name].astype(mx.float32), dtype=np.float64)[0, -1]


def read_metric_by_hand(model, metric: str, hook: str, position: int, add: np.ndarray) -> float:
    def shift(act, info):
        delta = np.zeros(act.shape, np.float32)
        delta[0, position] = add
        return act + mx.array(delta).astype(act.dtype)

    res = model.run(render(model, RECORD).array, hooks={hook: shift}, capture=["final_norm"])
    post = np.array(res.logits[0, -1].astype(mx.float32), dtype=np.float64)
    unembed = model.architecture.attribution_unembed(model._model)
    pre = post if unembed.softcap is None else np.array(
        unembed.project(res.cache["final_norm"][0, -1:])[0].astype(mx.float32), dtype=np.float64)
    if metric == "logit":
        return float(pre[TARGET])
    if metric == "margin":
        return float(pre[TARGET] - pre[CONTRAST])
    p = np.exp(post - np.logaddexp.reduce(post))
    mass = np.array([p[ids].sum() for ids in OUTCOME_IDS])
    if metric == "mass_outcomes":
        return float(mass.sum())
    q = mass / mass.sum()
    return float(-(q * np.log2(q)).sum())


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize(("positions", "position"), [("last", -1), ([1], 1)], ids=["decision", "earlier"])
def test_the_derivative_along_a_direction_is_the_finite_difference_of_the_metric(tiny, metric, positions,
                                                                                  position):
    d = tiny.arch.d_model
    direction = dirs.make(np.random.default_rng(1).normal(size=d),
                          S.space(model=None, layer=1, point="resid_post", d=d), method="t")
    row = differentiate(tiny, metric=metric, layers=[1], positions=positions, mask=direction)["items"][0]
    u = np.array(direction["vector"], dtype=np.float64)
    hook = "blocks.1.resid_post"
    fd = (read_metric_by_hand(tiny, metric, hook, position, EPS * u)
          - read_metric_by_hand(tiny, metric, hook, position, -EPS * u)) / (2 * EPS)
    assert row["along"][0] == pytest.approx(fd, abs=5e-3 * row["norm"])
    assert row["value"] == pytest.approx(read_metric_by_hand(tiny, metric, hook, position, 0 * u), rel=1e-5)


def test_the_logit_gradient_at_the_last_layer_is_the_attribution_direct_path_off_the_residual(tiny):
    last = tiny.arch.n_layers - 1
    row = differentiate(tiny, metric="logit", layers=[last])["items"][0]
    g = np.array(row["vector"])
    res = tiny.run(render(tiny, RECORD).array, capture=[f"blocks.{last}.resid_post", "final_norm.scale"])
    x = read_last(res, f"blocks.{last}.resid_post")
    ln_scale = np.array(res.cache["final_norm.scale"].astype(mx.float32)).reshape(-1)
    d = tiny.arch.d_model
    direct = attribution.logit_attrs(tiny, np.eye(d)[:, None, :], [TARGET], apply_ln=True,
                                     ln_scale=ln_scale)[:, 0]
    xhat = x / np.linalg.norm(x)

    def drop_residual(v: np.ndarray) -> np.ndarray:
        return v - (v @ xhat) * xhat

    assert np.allclose(drop_residual(g), drop_residual(direct), rtol=1e-4, atol=1e-5 * np.linalg.norm(direct))
    assert abs(g @ xhat) < 1e-3 * abs(direct @ xhat)


def test_the_forward_is_the_models_own_and_logit_is_read_before_the_softcap(tiny):
    ids = render(tiny, RECORD).array
    own = float(np.array(tiny.run(ids).logits[0, -1, TARGET].astype(mx.float32)))
    unembed = tiny.architecture.attribution_unembed(tiny._model)
    precap = unembed if unembed.softcap is not None else None

    def read_value(layers) -> float:
        names = [f"blocks.{i}.resid_post" for i in layers]
        return differentiate_record(tiny, RECORD, ids, names, [0], "logit", {}, precap)["value"]

    n = tiny.arch.n_layers
    last, through_every_mlp = read_value([n - 1]), read_value(range(n))
    if precap is None:
        assert last == own
        assert through_every_mlp == pytest.approx(own, rel=1e-5)
    else:
        assert unembed.softcap * np.tanh(last / unembed.softcap) == pytest.approx(own, rel=1e-5)
        assert last != pytest.approx(own, rel=1e-4)


def test_top_lists_k_coordinates_by_size_with_shares_summing_to_at_most_one(tiny):
    row = differentiate(tiny, metric="margin", layers=[2], top=5)["items"][0]
    g = np.array(row["vector"])
    x = read_last(tiny.run(render(tiny, RECORD).array, capture=["blocks.2.resid_post"]), "blocks.2.resid_post")
    for field, v in (("top", g), ("top_product", g * x)):
        entries = row[field]
        dims = [e["dim"] for e in entries]
        values = np.array([e["value"] for e in entries])
        assert len(entries) == 5 and len(set(dims)) == 5
        assert np.all(np.abs(values[:-1]) >= np.abs(values[1:]))
        assert np.abs(np.delete(v, dims)).max() <= np.abs(values).min() * (1 + 1e-5)
        assert values == pytest.approx(v[dims], rel=1e-5)
        assert sum(e["share"] for e in entries) <= 1
        assert [e["share"] for e in entries] == pytest.approx(v[dims] ** 2 / (v @ v), abs=2e-6)
    whole = differentiate(tiny, metric="margin", layers=[2], top=tiny.arch.d_model)["items"][0]
    assert 1 - 1e-4 <= sum(e["share"] for e in whole["top"]) <= 1


def test_a_mask_of_dimensions_gives_those_coordinates_of_the_gradient(tiny):
    dims = [17, 0, 5]
    out = differentiate(tiny, metric="logit", layers=[0, 3], positions="all", mask=dims)
    n = len(render(tiny, RECORD).ids)
    assert out["mask"] == dims
    assert [(r["space"]["layer"], r["position"]) for r in out["items"]] == [
        (layer, p) for layer in (0, 3) for p in range(n)]
    for row in out["items"]:
        assert row["along"] == [row["vector"][i] for i in dims]


def test_a_frame_on_the_port_is_the_mask_and_a_second_mask_is_refused(tiny):
    d = tiny.arch.d_model
    sp = S.space(model=None, layer=1, point="resid_post", d=d)
    frame = K.collection("direction/vector", [dirs.make(np.eye(d)[i] + 0.5 * np.eye(d)[i + 1], sp, method="t")
                                              for i in (2, 9)])
    out = differentiate_activations(tiny, [RECORD], {"metric": "margin", "layers": [1], "mask": "direction"},
                                    direction=frame)
    row = out["items"][0]
    basis = np.array([item["vector"] for item in frame["items"]])
    assert row["along"] == pytest.approx(basis @ np.array(row["vector"]), abs=1e-5 * row["norm"])
    assert [m["method"] for m in out["mask"]] == ["t", "t"]
    with pytest.raises(ValueError, match="one mask, the port's or the param's"):
        differentiate_activations(tiny, [RECORD], {"mask": [3]}, direction=frame)


def test_an_unknown_metric_is_refused_by_name(tiny):
    with pytest.raises(ValueError, match="unknown metric 'temperature': one of logit, margin, "
                                         "entropy_outcomes, mass_outcomes"):
        differentiate(tiny, metric="temperature")


@pytest.mark.parametrize("name", ["gemma3", "llama", "qwen2", "gemma4-31b"])
def test_a_point_the_model_lacks_is_refused_by_name(name):
    model = build_tiny_model(name, BY_MODEL_TYPE["gemma4" if name == "gemma4-31b" else name])
    with pytest.raises(PointRefused) as e:
        differentiate(model, point="gate_out")
    assert (e.value.code, e.value.point) == ("POINT_ABSENT", "gate_out")
    assert str(e.value).startswith("POINT_ABSENT: `gate_out` is not written on this ")


def test_a_point_off_the_stream_is_refused_and_the_gate_is_read_where_there_is_one():
    model = build_tiny_model("gemma4", BY_MODEL_TYPE["gemma4"])
    with pytest.raises(ValueError, match="unknown point 'attn.q': the gradient is taken at resid_pre"):
        differentiate(model, point="attn.q")
    rows = differentiate(model, point="gate_out", layers=[1])["items"]
    assert rows[0]["space"]["point"] == "gate_out" and rows[0]["norm"] > 0


def test_a_record_without_what_its_metric_reads_is_refused_by_its_id(tiny):
    lone = {"id": "lone", "text": "the cat sat on a", "tracked": {"t": "mat"}}
    with pytest.raises(ValueError, match="record 'lone': `margin` is the target's logit minus a contrast's"):
        differentiate_activations(tiny, [lone], {"metric": "margin"})
    with pytest.raises(ValueError, match="record 'lone' has no `outcomes`"):
        differentiate_activations(tiny, [lone], {"metric": "mass_outcomes"})


def test_the_model_is_bit_identical_after_and_a_double_run_is_identical(tiny):
    before = {k: np.array(v) for k, v in tree_flatten(tiny.lm.parameters())}
    params = {"metric": "entropy_outcomes", "positions": "all", "top": 4, "mask": [1, 2]}
    first = differentiate_activations(tiny, [RECORD], params)
    assert differentiate_activations(tiny, [RECORD], params) == first
    ids = render(tiny, RECORD).array
    names = [f"blocks.{i}.resid_post" for i in range(tiny.arch.n_layers)]
    raw = [differentiate_record(tiny, RECORD, ids, names, [0, 3], "margin", {}, None) for _ in range(2)]
    assert all(np.array_equal(a, b) for a, b in zip(raw[0]["grads"], raw[1]["grads"], strict=True))
    after = dict(tree_flatten(tiny.lm.parameters()))
    assert before.keys() == after.keys()
    assert all(np.array_equal(before[k], np.array(after[k])) for k in before)


def test_the_header_records_the_metric_the_point_the_positions_and_the_model(tiny):
    out = differentiate(tiny, metric="margin", layers=[1, 2], positions=[1, -1], top=3, mask=[4])
    assert {k: out.get(k) for k in ("item_kind", "metric", "point", "layers", "positions", "d_model", "top",
                                    "mask")} == {
        "item_kind": "activations/vector", "metric": "margin", "point": "resid_post", "layers": [1, 2],
        "positions": [1, -1], "d_model": tiny.arch.d_model, "top": 3, "mask": [4]}
    assert out.get("softcap") == tiny.architecture.attribution_unembed(tiny._model).softcap
    n = len(render(tiny, RECORD).ids)
    rows = out["items"]
    assert [(r["space"]["layer"], r["position"]) for r in rows] == [(1, 1), (1, n - 1), (2, 1), (2, n - 1)]
    assert all(r["target"]["id"] == TARGET and r["contrast"]["id"] == CONTRAST for r in rows)


def test_through_the_executor_the_header_names_the_fused_adapter_and_the_model_is_bare_after(tmp_path):
    model = build_tiny_model("gemma3", BY_MODEL_TYPE["gemma3"])
    executor = ProtocolExecutor()
    executor._model, executor._model_id = model, "tiny"
    params = {"metric": "logit", "layers": [3]}
    bare = differentiate_activations(model, [RECORD], params)
    out = executor._run_op("activations/differentiate", {"records": [RECORD]},
                           {"model": adapted_ref(build_payload(model, tmp_path, "die", 1)), **params})
    assert out["fused"] == [{"bench": "you/lab/adapter-0"}]
    assert out["items"][0]["vector"] != bare["items"][0]["vector"]
    assert differentiate_activations(model, [RECORD], params) == bare
