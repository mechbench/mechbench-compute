from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_map

from mechbench_compute.ops.activations.capture import capture_residual_vectors
from tests.test_capture_mlp_interior import IDS, INTERIOR, RECORD
from tests.test_forward_stops_early import NAMES, build, refuse_after


def build_in(name: str, dtype) -> object:
    model = build(name)
    if dtype is not None:
        model.lm.update(tree_map(lambda p: p.astype(dtype), model.lm.parameters()))
        mx.eval(model.lm.parameters())
    return model


@pytest.mark.parametrize("dtype", [None, mx.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize("name", NAMES)
def test_the_manual_mlp_is_the_fused_one_bit_for_bit(name, dtype):
    model = build_in(name, dtype)
    ids = model.make_ids(IDS)
    n = model.arch.n_layers
    stream = [f"blocks.{i}.resid_post" for i in range(n)]
    plain = model.run(ids, capture=stream)
    for layer in range(n):
        probed = model.run(ids, capture=[*stream, *(f"blocks.{layer}.{p}" for p in INTERIOR)])
        assert np.array_equal(np.array(probed.logits.astype(mx.float32)),
                              np.array(plain.logits.astype(mx.float32))), (name, layer)
        for s in stream:
            assert np.array_equal(np.array(probed.cache[s].astype(mx.float32)),
                                  np.array(plain.cache[s].astype(mx.float32))), (name, s, layer)


@pytest.mark.parametrize("point", INTERIOR)
@pytest.mark.parametrize("name", NAMES)
def test_an_interior_read_stops_after_its_layer(name, point):
    model = build(name)
    layer = model.arch.n_layers // 2 - 1
    params = {"layers": [layer], "point": point, "storage": "json"}
    want = capture_residual_vectors(model, [RECORD], params)
    refuse_after(model, layer)
    got = capture_residual_vectors(model, [RECORD], params)
    assert [it["vector"] for it in got["items"]] == [it["vector"] for it in want["items"]]
    with pytest.raises(AssertionError, match="past the last one read"):
        model.run(model.make_ids(IDS), capture=[f"blocks.{layer}.{point}"])


def test_a_double_wide_layer_reads_at_its_own_width():
    model = build("gemma4-shared")
    first = model.arch.first_kv_shared_layer
    layers = [first - 1, first]
    out = capture_residual_vectors(model, [RECORD], {"layers": layers, "point": "mlp.down_in",
                                                     "storage": "json"})
    assert out["widths"] == [64, 128]
    assert [len(it["vector"]) for it in out["items"]] == [64, 128]


def test_tensor_storage_across_two_widths_is_refused_before_the_model_runs():
    model = build("gemma4-shared")
    first = model.arch.first_kv_shared_layer
    refuse_after(model, -1)
    with pytest.raises(ValueError, match=(
            rf"`mlp.act` here is layers \[{first - 1}\] are 64 wide; layers \[{first}\] "
            r"are 128 wide, and tensor storage holds rows of one width")):
        capture_residual_vectors(model, [RECORD], {
            "layers": [first - 1, first], "point": "mlp.act", "storage": "tensor"})
    one = capture_residual_vectors(build("gemma4-shared"), [RECORD], {
        "layers": [first, first + 1], "point": "mlp.act", "storage": "tensor"})
    assert one["storage"] == "tensor" and one["d"] == 128 and one["n_items"] == 2
