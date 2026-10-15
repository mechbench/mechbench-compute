from __future__ import annotations

import numpy as np
import pytest

from mechbench_compute import backends
from mechbench_compute.arrays import read_f64
from mechbench_compute.ops.activations.capture import capture_residual_vectors
from tests.kit_backends import list_kit_params, load_kit

IDS = [1, 5, 9, 2, 7, 3, 11, 4]

INTERIOR = ("mlp.in_norm", "mlp.gate", "mlp.up", "mlp.act", "mlp.down_in")

NEURONS = INTERIOR[1:]

RECORD = {"id": "r", "user": "the cat sat on a mat"}

ROUNDING = 1e-5


@pytest.fixture(scope="module", params=list_kit_params())
def kit_model(request):
    kit_name, name, model_type = request.param
    kit = load_kit(kit_name)
    try:
        kit.select()
    except backends.BackendRefused as e:
        pytest.skip(f"the {kit_name} backend is not selected here: {e}")
    return kit, kit.build(name, kit.architectures[model_type])


def read_mlp_widths(kit, model) -> list[int]:
    config = kit.read_config(model)
    width = int(config["intermediate_size"])
    shared = int(config.get("num_kv_shared_layers") or 0)
    first_shared = model.arch.n_layers - shared
    doubled = bool(config.get("use_double_wide_mlp")) and shared > 0
    return [width * (2 if doubled and i >= first_shared else 1)
            for i in range(model.arch.n_layers)]


def capture(model, **params) -> dict:
    return capture_residual_vectors(model, [RECORD], {"storage": "json", **params})


def test_an_interior_read_leaves_the_stream_and_the_logits_bit_identical(kit_model):
    _, model = kit_model
    ids = model.make_ids(IDS)
    n = model.arch.n_layers
    stream = [f"blocks.{i}.resid_post" for i in range(n)]
    plain = model.run(ids, capture=stream)
    for layer in range(n):
        for point in INTERIOR:
            probed = model.run(ids, capture=[*stream, f"blocks.{layer}.{point}"])
            assert np.array_equal(read_f64(probed.logits), read_f64(plain.logits)), (
                f"{model.arch.model_type}: logits with {point} at {layer}")
            for name in stream:
                assert np.array_equal(read_f64(probed.cache[name]), read_f64(plain.cache[name])), (
                    f"{model.arch.model_type}: {name} with {point} at {layer}")


def test_capture_reads_the_mlp_interior_at_the_mlp_s_width(kit_model):
    kit, model = kit_model
    n = model.arch.n_layers
    widths = read_mlp_widths(kit, model)
    cache = model.run(model.make_ids(IDS),
                      capture=[f"blocks.{i}.{p}" for i in range(n) for p in INTERIOR]).cache
    for point in INTERIOR:
        out = capture(model, point=point)
        want = widths if point in NEURONS else [model.arch.d_model] * n
        assert out["point"] == point and out["d_model"] == model.arch.d_model
        assert out.get("widths") == (widths if point in NEURONS else None)
        assert [it["space"]["d"] for it in out["items"]] == want
        assert [len(it["vector"]) for it in out["items"]] == want
        for layer in range(n):
            t = read_f64(cache[f"blocks.{layer}.{point}"])
            assert t.shape[-1] == want[layer], f"{point} at {layer}: {t.shape}"


def test_down_in_is_act_times_up_captured_apart(kit_model):
    _, model = kit_model
    read = {p: [np.array(it["vector"]) for it in capture(model, point=p)["items"]]
            for p in ("mlp.act", "mlp.up", "mlp.down_in")}
    for layer, (act, up, down_in) in enumerate(zip(*read.values())):
        assert np.abs(act).max() > 0.01 and np.abs(up).max() > 0.01
        assert np.allclose(down_in, act * up, rtol=2.0 ** -7, atol=4 * ROUNDING), (
            f"{model.arch.model_type}: layer {layer}")


@pytest.mark.parametrize("pool", [None, {"reduce": "mean", "over": "all"}], ids=["last", "pooled"])
def test_capture_s_vectors_are_the_forward_s_interior(kit_model, pool):
    _, model = kit_model
    n = model.arch.n_layers
    ids = capture_ids(model)
    cache = model.run(model.make_ids(ids),
                      capture=[f"blocks.{i}.mlp.down_in" for i in range(n)]).cache
    out = capture(model, point="mlp.down_in", **({"pool": pool} if pool else {}))
    for layer, item in enumerate(out["items"]):
        t = read_f64(cache[f"blocks.{layer}.mlp.down_in"])[0]
        want = t.mean(axis=0) if pool else t[-1]
        assert np.allclose(item["vector"], want, rtol=1e-5, atol=ROUNDING), layer


def capture_ids(model) -> list[int]:
    from mechbench_compute.distill import render

    return list(render(model, RECORD).ids)


@pytest.mark.parametrize("point", INTERIOR)
def test_heads_with_an_interior_point_is_refused_by_name(kit_model, point):
    _, model = kit_model
    with pytest.raises(ValueError, match=f"`{point}` has no heads"):
        capture(model, point=point, heads=[0])


def test_a_neuron_read_takes_no_other_source(kit_model):
    _, model = kit_model
    with pytest.raises(ValueError, match="is read with source 'resid'"):
        capture(model, point="mlp.act", source="queries")
