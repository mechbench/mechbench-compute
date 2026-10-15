from __future__ import annotations

import itertools

import numpy as np
import pytest

from mechbench_compute import backends
from mechbench_compute.arrays import make_f32, read_f32
from mechbench_compute.head_weights import read_bias
from tests.kit_backends import list_kit_params, load_kit

IDS = [1, 5, 9, 2, 7, 3, 11, 4]

TOLERANCE = 1e-4


@pytest.fixture(scope="module", params=list_kit_params())
def tiny(request):
    kit_name, name, model_type = request.param
    kit = load_kit(kit_name)
    try:
        kit.select()
    except backends.BackendRefused as e:
        pytest.skip(f"the {kit_name} backend is not selected here: {e}")
    return kit.build(name, kit.architectures[model_type])


def check_close(got, want, what):
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    assert got.shape == want.shape, (what, got.shape, want.shape)
    err = float(np.abs(got - want).max())
    allowed = TOLERANCE * max(1.0, float(np.abs(want).max()))
    assert err <= allowed, f"{what}: max |diff| {err:.3g}, allowed {allowed:.3g}"


def apply(module, x, like):
    return read_f32(module(make_f32(x, like)))


def project(w, b, x):
    return x @ w.T + (0.0 if b is None else b)


def read_head_terms(model, layer, head, cache):
    spec = model.architecture.head_weights(model._model, layer, head)
    layers = model.lm.model.layers
    attn, kv = layers[layer].self_attn, layers[spec.kv_layer].self_attn
    like = attn.o_proj.weight

    def normed_input(i):
        resid = read_f32(cache[f"blocks.{i}.resid_pre"])
        return apply(layers[i].input_layernorm, resid, like)[0]

    def maybe_norm(owner, name, x):
        norm = getattr(owner, name, None)
        return x if norm is None else apply(norm, x[None], like)[0]

    x, x_kv = normed_input(layer), normed_input(spec.kv_layer)
    q = maybe_norm(attn, "q_norm", project(spec.W_Q, spec.b_Q, x))
    k = maybe_norm(kv, "k_norm", project(spec.W_K, spec.b_K, x_kv))
    v = maybe_norm(kv, "v_norm", project(spec.W_V, spec.b_V, x_kv))
    return spec, attn, like, q, k, v


def test_each_head_s_own_slices_rebuild_what_its_forward_computes(tiny):
    arch = tiny.arch
    points = ("resid_pre", "attn.q", "attn.k", "attn.v", "attn.weights", "attn.per_head_out")
    names = [f"blocks.{i}.{p}" for i in range(arch.n_layers) for p in points]
    cache = tiny.run(tiny.make_ids(IDS), capture=names).cache
    for layer, head in itertools.product(range(arch.n_layers), range(arch.n_heads)):
        spec, attn, like, q, k, v = read_head_terms(tiny, layer, head, cache)
        at = f"{tiny.architecture.model_type} L{layer}H{head}"
        g = spec.kv_group
        got_q = read_f32(cache[f"blocks.{layer}.attn.q"])[0, head]
        got_k = read_f32(cache[f"blocks.{layer}.attn.k"])[0, g]
        got_v = read_f32(cache[f"blocks.{layer}.attn.v"])[0, g]
        check_close(q[0], got_q[0], f"{at} q at position 0, before RoPE turns it")
        check_close(k[0], got_k[0], f"{at} k at position 0")
        check_close((q * k).sum(-1), (got_q * got_k).sum(-1),
                    f"{at} q·k at each position, which RoPE leaves alone")
        check_close(v, got_v, f"{at} v")

        weights = read_f32(cache[f"blocks.{layer}.attn.weights"])[0, head]
        per_head = read_f32(cache[f"blocks.{layer}.attn.per_head_out"])
        check_close(weights @ v, per_head[0, head], f"{at} the head's mix of its group's values")

        alone = np.zeros_like(per_head)
        alone[0, head] = per_head[0, head]
        flat = alone.transpose(0, 2, 1, 3).reshape(1, alone.shape[2], -1)
        o_bias = read_bias(attn.o_proj)
        written = apply(attn.o_proj, flat, like)[0] - (0.0 if o_bias is None else o_bias)
        check_close((weights @ v) @ spec.W_O.T, written,
                    f"{at} W_O·W_V on the values, against o_proj with the other heads zeroed")
