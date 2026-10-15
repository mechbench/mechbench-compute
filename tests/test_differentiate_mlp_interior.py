from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mechbench_compute import attribution
from mechbench_compute.block_params import check_params
from mechbench_compute.distill import render
from mechbench_compute.intervene.operator_refused import OperatorRefused
from mechbench_compute.ops.activations.capture import capture_residual_vectors
from mechbench_compute.ops.activations.differentiate import (
    differentiate_activations,
    differentiate_record,
)
from tests.test_activations_differentiate import (
    CONTRAST,
    EPS,
    RECORD,
    TARGET,
    read_metric_by_hand,
)
from tests.test_capture_mlp_interior import IDS, INTERIOR, NEURONS
from tests.test_forward_stops_early import NAMES, build, refuse_after

PRODUCTS = ("mlp.up", "mlp.act", "mlp.down_in")


@pytest.fixture(scope="module", params=NAMES)
def model(request):
    return build(request.param)


def differentiate(model, records=None, **params) -> dict:
    return differentiate_activations(model, records or [RECORD], {"storage": "json", **params})


def test_a_zero_added_at_an_interior_point_leaves_the_stream_and_the_logits_bit_identical(model):
    ids = model.make_ids(IDS)
    n = model.arch.n_layers
    stream = [f"blocks.{i}.resid_post" for i in range(n)]
    plain = model.run(ids, capture=stream)
    for layer in range(n):
        for point in INTERIOR:
            name = f"blocks.{layer}.{point}"
            probed = model.run(ids, capture=stream,
                               hooks={name: lambda act, info: act + mx.zeros(act.shape).astype(act.dtype)})
            assert np.array_equal(np.array(probed.logits.astype(mx.float32)),
                                  np.array(plain.logits.astype(mx.float32))), (point, layer)
            for s in stream:
                assert np.array_equal(np.array(probed.cache[s].astype(mx.float32)),
                                      np.array(plain.cache[s].astype(mx.float32))), (s, point, layer)


@pytest.mark.parametrize("metric", ["logit", "entropy_outcomes"])
def test_the_metric_is_the_plain_read_s_unless_the_backward_crosses_the_activation(model, metric):
    ids = render(model, RECORD).array
    last = model.arch.n_layers - 1
    unembed = model.architecture.attribution_unembed(model._model)
    precap = unembed if unembed.softcap is not None else None
    widths = capture_residual_vectors(model, [RECORD], {"layers": [last], "point": "mlp.act",
                                                        "storage": "json"})["widths"]

    def read_value(point: str) -> float:
        w = widths if point in NEURONS else None
        return differentiate_record(model, RECORD, ids, [f"blocks.{last}.{point}"], [0], metric, {}, precap,
                                    widths=w)["value"]

    plain, crossing = read_value("resid_post"), read_value("resid_pre")
    assert [read_value(p) for p in ("mlp.up", "mlp.act", "mlp.down_in")] == [plain] * 3
    assert [read_value(p) for p in ("mlp.in_norm", "mlp.gate")] == [crossing] * 2
    assert crossing == pytest.approx(plain, rel=1e-5)


def test_the_gradient_is_as_wide_as_the_mlp_on_each_layer():
    model = build("gemma4-shared")
    first = model.arch.first_kv_shared_layer
    n = model.arch.n_layers
    want = [64 if i < first else 128 for i in range(n)]
    for point in INTERIOR:
        out = differentiate(model, point=point, top=2)
        widths = want if point in NEURONS else [model.arch.d_model] * n
        assert out["d_model"] == model.arch.d_model
        assert out.get("widths") == (want if point in NEURONS else None)
        assert [it["space"]["d"] for it in out["items"]] == widths
        assert [len(it["vector"]) for it in out["items"]] == widths
        assert all(it["space"]["point"] == point for it in out["items"])


def test_tensor_storage_across_two_widths_is_refused_before_the_model_runs():
    model = build("gemma4-shared")
    first = model.arch.first_kv_shared_layer
    refuse_after(model, -1)
    with pytest.raises(ValueError, match=(
            rf"`mlp.down_in` here is layers \[{first - 1}\] are 64 wide; layers \[{first}\] "
            r"are 128 wide, and tensor storage holds rows of one width")):
        differentiate_activations(model, [RECORD], {
            "layers": [first - 1, first], "point": "mlp.down_in", "storage": "tensor"})
    one = differentiate_activations(build("gemma4-shared"), [RECORD], {
        "layers": [first, first + 1], "point": "mlp.down_in", "storage": "tensor"})
    assert one["storage"] == "tensor" and one["d"] == 128 and one["n_items"] == 2
    assert one["widths"] == [128, 128]


@pytest.mark.parametrize("point", NEURONS)
def test_the_derivative_along_a_neuron_is_the_finite_difference_of_the_metric(model, point):
    layer = 1
    j = 5
    row = differentiate(model, metric="margin", layers=[layer], point=point, mask=[j])["items"][0]
    hook = f"blocks.{layer}.{point}"
    e = np.zeros(row["space"]["d"])
    e[j] = 1.0
    fd = (read_metric_by_hand(model, "margin", hook, -1, EPS * e)
          - read_metric_by_hand(model, "margin", hook, -1, -EPS * e)) / (2 * EPS)
    assert row["along"][0] == pytest.approx(fd, abs=5e-3 * max(row["norm"], 1e-3))


def test_gradient_times_activation_is_one_attribution_through_act_up_and_down_in(model):
    k = 8
    read = {p: differentiate(model, metric="margin", point=p, top=k)["items"] for p in PRODUCTS}
    for rows in zip(*read.values(), strict=True):
        tops = [{e["dim"]: e["value"] for e in r["top_product"]} for r in rows]
        dims = set(tops[0])
        assert all(set(t) == dims for t in tops), rows[0]["space"]["layer"]
        for d in dims:
            values = [t[d] for t in tops]
            assert values == pytest.approx([values[-1]] * 3, rel=1e-3, abs=1e-6), (rows[0]["space"], d)


def plant(model, target: int) -> tuple[int, int]:
    layer = model.arch.n_layers - 1
    block = model.lm.model.layers[layer]
    for norm in (model.lm.model.norm, getattr(block, "post_feedforward_layernorm", None)):
        if norm is not None:
            norm.weight = mx.full(norm.weight.shape, 1.0 - attribution._read_gain_offset(norm)).astype(
                norm.weight.dtype)
    if getattr(block, "layer_scalar", None) is not None:
        block.layer_scalar = mx.ones_like(block.layer_scalar)
    unembed = model.architecture.attribution_unembed(model._model)
    row = np.array(unembed.project(mx.eye(model.arch.d_model)).astype(mx.float32))[:, target]
    down_in = capture_residual_vectors(model, [RECORD], {"layers": [layer], "point": "mlp.down_in",
                                                         "storage": "json"})["items"][0]["vector"]
    neuron = int(np.argmax(down_in))
    assert down_in[neuron] > 0
    down = block.mlp.down_proj
    w = np.array(down.weight.astype(mx.float32))
    w[:, neuron] = row
    down.weight = mx.array(w).astype(down.weight.dtype)
    return layer, neuron


@pytest.mark.parametrize("name", NAMES)
def test_a_neuron_that_writes_the_target_s_unembedding_leads_the_attribution(name):
    model = build(name)
    layer, neuron = plant(model, TARGET)
    k = model.lm.model.layers[layer].mlp.down_proj.weight.shape[1]
    row = differentiate(model, metric="logit", layers=[layer], point="mlp.down_in", top=k)["items"][0]
    assert row["target"]["id"] == TARGET
    lead = row["top_product"][0]
    assert lead["dim"] == neuron and lead["value"] > 0
    flipped = {**RECORD, "tracked": {"c": {"id": CONTRAST}, "t": {"id": TARGET}}}
    against = differentiate(model, [flipped], metric="margin", layers=[layer], point="mlp.down_in",
                            top=k)["items"][0]
    assert against["contrast"]["id"] == TARGET
    by_dim = {e["dim"]: e["value"] for e in against["top_product"]}
    assert by_dim[neuron] < 0


@pytest.mark.parametrize("point", INTERIOR)
def test_heads_with_an_interior_point_is_refused_by_name(point):
    with pytest.raises(ValueError, match="activations/differentiate does not accept 'heads'"):
        check_params("activations/differentiate", {"point": point, "heads": [0]})


def test_a_mask_past_a_neuron_width_is_refused_naming_the_width():
    model = build("gemma4-shared")
    first = model.arch.first_kv_shared_layer
    with pytest.raises(OperatorRefused, match="'mlp.act' is 64 wide here"):
        differentiate(model, point="mlp.act", layers=[first - 1, first], mask=[100])
    row = differentiate(model, point="mlp.act", layers=[first], mask=[100])["items"][0]
    assert row["along"] == [row["vector"][100]]
