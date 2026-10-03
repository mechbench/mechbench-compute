from __future__ import annotations

import dataclasses
import itertools
import json
import pathlib
import re

import numpy as np
import pytest

from mechbench_compute import attribution, backends, dialects, support
from mechbench_compute import shapes as S
from mechbench_compute._arch import Arch
from mechbench_compute.arrays import make_f32, read_f32, read_f64, zeros_like
from mechbench_compute.distill import render
from mechbench_compute.errors import InvalidHookName
from mechbench_compute.interp.point_refused import PointRefused
from mechbench_compute.interventions import Ablate
from mechbench_compute.ops.activations.capture import capture_residual_vectors
from mechbench_compute.points import LAYOUT
from mechbench_compute.tools import build_toolbox
from tests.kit_backends import KIT_MODULES, WINDOW, list_kit_params, load_kit

IDS = [1, 5, 9, 2, 7, 3, 11, 4]

RESIDUAL_TOLERANCE = 2.0 ** -6

PER_HEAD_TOLERANCE = 1e-4

MLP_INTERIOR = ("mlp.gate", "mlp.up", "mlp.act", "mlp.down_in")

RECORD = {"id": "r", "user": "the cat sat on a mat"}

TEMPLATES = pathlib.Path(__file__).parent / "fixtures" / "chat_templates.json"

TEMPLATE_OF = {
    "gemma3": "mlx-community/gemma-3-4b-it-bf16",
    "gemma4": "mlx-community/gemma-4-e2b-it-bf16",
    "llama": "mlx-community/Llama-3.2-3B-Instruct-bf16",
    "qwen2": "mlx-community/Qwen2.5-3B-Instruct-bf16",
}


@pytest.fixture(scope="module", params=list_kit_params())
def tiny(request):
    kit_name, name, model_type = request.param
    kit = load_kit(kit_name)
    try:
        kit.select()
    except backends.BackendRefused as e:
        pytest.skip(f"the {kit_name} backend is not selected here: {e}")
    return kit.build(name, kit.architectures[model_type])


def list_declared_names(model) -> list[str]:
    a = model.architecture
    return ([f"blocks.{i}.{p}" for i in range(model.arch.n_layers)
             for p in a.layer_points_of(model.arch)]
            + list(a.global_points))


def read_law_terms(law: str) -> list[str]:
    return sorted(set(re.findall(r"([a-z_]+)\[i(?:\+1)?\]", law)))


def read_law_points(model) -> list[str]:
    architecture = model.architecture
    law = architecture.residual_law_of(model.arch)
    points = [t for t in read_law_terms(law) if t != "layer_scalar"]
    unknown = [p for p in points if not architecture.supports(p, layer_scoped=True)]
    assert not unknown, (
        f"{architecture.model_type}: residual law {law!r} names points it does not declare: {unknown}")
    return points


def check_residual_law(model, ids=None) -> None:
    ids = model.make_ids(IDS) if ids is None else ids
    n = model.arch.n_layers
    points = read_law_points(model)
    result = model.run(ids, capture=[f"blocks.{i}.{p}" for i in range(n) for p in points])
    check_law(model, {p: [read_f64(result.cache[f"blocks.{i}.{p}"]) for i in range(n)] for p in points})


def check_law(model, names: dict[str, list]) -> None:
    architecture = model.architecture
    law = architecture.residual_law_of(model.arch)
    n = model.arch.n_layers
    names = dict(names)
    if "layer_scalar" in read_law_terms(law):
        assert architecture.layer_scalars is not None, (
            f"{architecture.model_type}: residual law {law!r} names layer_scalar and the "
            f"architecture declares no layer_scalars to read it")
        names["layer_scalar"] = list(architecture.layer_scalars(model._model))
    sides = [s.strip() for s in law.split("==")]
    for i in range(n):
        values = [(side, eval(side, {}, {**names, "i": i}))
                  for side in sides if not ("[i+1]" in side and i == n - 1)]
        for (left, a), (right, b) in itertools.pairwise(values):
            allowed = RESIDUAL_TOLERANCE * max(1.0, float(np.abs(a).max()))
            err = float(np.abs(a - b).max())
            if err > allowed:
                raise AssertionError(
                    f"{architecture.model_type}: residual law {law!r} fails at layer {i}: "
                    f"{left} != {right} (max |diff| {err:.3g}, allowed {allowed:.3g})")


@pytest.mark.parametrize("kit_name", list(KIT_MODULES))
def test_every_architecture_has_a_tiny_model(kit_name):
    kit = load_kit(kit_name)
    assert set(kit.architectures) == {t for _, t in kit.models}
    assert set(kit.architectures) <= set(TEMPLATE_OF)
    assert all(a.backend == kit.name for a in kit.architectures.values()), kit.name


def test_the_declared_points_are_present_at_the_declared_level(tiny):
    a = tiny.architecture
    assert a.level in support.LEVELS and a.loader in support.LOADERS
    names = list_declared_names(tiny)
    result = tiny.run(tiny.make_ids(IDS), capture=names)
    n_tokens = len(IDS)
    heads = {tiny.arch.n_heads, tiny.arch.n_kv_heads, tiny.arch.n_global_kv_heads} - {None}
    for name in names:
        assert name in result.cache, f"{a.model_type}: {name} was declared and not captured"
        shape = tuple(result.cache[name].shape)
        point = name.split(".", 2)[-1] if name.startswith("blocks.") else name
        if point == "final_norm.scale":
            assert shape == (1, n_tokens), f"{a.model_type}: {name} {shape}"
            continue
        position, head, feature = LAYOUT[point]
        assert len(shape) == feature + 1 and shape[0] == 1, f"{a.model_type}: {name} {shape}"
        assert shape[position] == n_tokens, f"{a.model_type}: {name} {shape}"
        if head is not None:
            assert shape[head] in heads, f"{a.model_type}: {name} {shape}"


def test_the_global_points_agree_with_the_residual_stream(tiny):
    a = tiny.architecture
    last = tiny.arch.n_layers - 1
    wanted = ["blocks.0.resid_pre", f"blocks.{last}.resid_post", *a.global_points]
    result = tiny.run(tiny.make_ids(IDS), capture=wanted)
    cache = result.cache
    if "embed" in a.global_points:
        assert np.array_equal(read_f64(cache["embed"]), read_f64(cache["blocks.0.resid_pre"]))
    if "final_norm" in a.global_points:
        normed = a.attribution_unembed(tiny._model).norm(cache[f"blocks.{last}.resid_post"])
        assert np.array_equal(read_f64(cache["final_norm"]), read_f64(normed))
    if "logits" in a.global_points:
        assert np.array_equal(read_f64(cache["logits"]), read_f64(result.logits))
        head = tiny.head_logits(cache["final_norm"])
        assert np.array_equal(read_f64(head), read_f64(result.logits))


def test_the_residual_law_holds(tiny):
    check_residual_law(tiny)


def check_logit_softcap(model) -> None:
    a = model.architecture
    cap = load_kit(a.backend).read_config(model).get("final_logit_softcapping")
    unembed = a.attribution_unembed(model._model)
    assert unembed.softcap == cap, (
        f"{a.model_type} on {a.backend}: the checkpoint's config caps the final logits "
        f"at {cap}, and the head declares {unembed.softcap}")
    if cap is None:
        assert any(r.config_key == "final_logit_softcapping" for r in a.refused_when), (
            f"{a.model_type} on {a.backend}: the head applies no cap, and nothing refuses a "
            f"checkpoint whose config sets one")
    hidden = model.run(model.make_ids(IDS), capture=["final_norm"]).cache["final_norm"] * 1000.0
    raw = read_f64(unembed.project(make_f32(read_f32(hidden), like=unembed.norm.weight)))
    head = read_f64(model.head_logits(hidden))
    if cap is None:
        assert np.array_equal(head, raw), f"{a.model_type} on {a.backend}: the head is not the projection"
        return
    assert np.abs(raw).max() > cap
    assert np.abs(head).max() <= cap and np.allclose(head, cap * np.tanh(raw / cap), atol=1e-3 * cap), (
        f"{a.model_type} on {a.backend}: the head does not apply the checkpoint's cap of {cap}")


def test_the_head_applies_the_checkpoint_s_logit_softcap(tiny):
    check_logit_softcap(tiny)


def test_a_head_that_ignores_the_checkpoint_s_cap_fails_by_name():
    kit = load_kit("mlx")
    gemma4 = kit.architectures["gemma4"]
    uncapped = dataclasses.replace(
        gemma4, head_logits=lambda model, hidden: gemma4.attribution_unembed(model).project(hidden))
    with pytest.raises(AssertionError, match=r"gemma4 on mlx: the head does not apply the checkpoint's cap of 30"):
        check_logit_softcap(kit.build("gemma4", uncapped))


def break_the_residual(architecture):
    def forward_with_a_wrong_residual(model, input_ids, *, hooks=None, capture=None,
                                      arch=None, kv_cache=None):
        hooks = dict(hooks or {})
        for i in range(arch.n_layers):
            hooks.setdefault(f"blocks.{i}.resid_post", lambda act, info: act * 1.5)
        return architecture.forward(model, input_ids, hooks=hooks, capture=capture,
                                    arch=arch, kv_cache=kv_cache)

    return dataclasses.replace(architecture, model_type="llama-broken",
                               name="Llama with a wrong residual",
                               forward=forward_with_a_wrong_residual)


@pytest.mark.parametrize("kit_name", list(KIT_MODULES))
def test_an_architecture_with_a_wrong_residual_fails_the_law_by_name(kit_name):
    kit = load_kit(kit_name)
    if "llama" not in kit.architectures:
        pytest.skip(f"the {kit_name} backend has no llama")
    model = kit.build("llama", break_the_residual(kit.architectures["llama"]))
    with pytest.raises(AssertionError,
                       match=r"llama-broken: residual law .* fails at layer 0"):
        check_residual_law(model)


def test_the_attribution_reads_the_writes_and_the_scalars_the_residual_law_names(tiny):
    a = tiny.architecture
    terms = read_law_terms(a.residual_law_of(tiny.arch))
    assert set(a.writes_of(tiny.arch)) == set(terms) - {"resid_pre", "resid_post", "layer_scalar"}
    assert ("layer_scalar" in terms) == (a.layer_scalars is not None), a.model_type


def run_attribution(model, ids=None):
    ids = model.make_ids(IDS) if ids is None else ids
    n = model.arch.n_layers
    writes = model.architecture.writes_of(model.arch)
    result = model.run(ids, capture=[
        *(f"blocks.{i}.{p}" for i in range(n) for p in (*writes, "resid_post")),
        "blocks.0.resid_pre", "final_norm.scale", "final_norm",
        *(f"blocks.{i}.attn.per_head_out" for i in range(n))])
    return result, read_precap(model, result), read_f64(result.cache["final_norm.scale"]).reshape(-1)


def read_precap(model, result) -> np.ndarray:
    unembed = model.architecture.attribution_unembed(model._model)
    if unembed.softcap is None:
        return read_f64(result.logits)[0, -1]
    normed = make_f32(read_f32(result.cache["final_norm"])[:, -1:], like=unembed.norm.weight)
    return read_f64(unembed.project(normed)).reshape(-1)


@pytest.mark.parametrize("sublayer", [False, True], ids=["layer", "sublayer"])
def test_direct_logit_attribution_with_the_final_norm_sums_to_the_true_logit(tiny, sublayer):
    result, last, ln_scale = run_attribution(tiny)
    names, pieces = attribution.decompose_logit(tiny, result.cache, sublayer=sublayer)
    k = len(tiny.architecture.writes_of(tiny.arch)) if sublayer else 1
    assert len(names) == len(pieces) == 1 + k * tiny.arch.n_layers
    targets = [int(np.argmax(last)), int(np.argmin(last)), 17]
    attrs = attribution.logit_attrs(tiny, pieces, targets, apply_ln=True, ln_scale=ln_scale)
    assert np.allclose(attrs.sum(axis=0), last[targets], atol=2e-3, rtol=1e-3)


def test_a_layer_is_its_sublayer_pieces_and_without_scalars_the_stream_s_step(tiny):
    result, _, _ = run_attribution(tiny)
    n, k = tiny.arch.n_layers, len(tiny.architecture.writes_of(tiny.arch))
    _, by_layer = attribution.decompose_logit(tiny, result.cache, sublayer=False)
    _, by_sublayer = attribution.decompose_logit(tiny, result.cache, sublayer=True)
    assert np.allclose(by_sublayer[0], by_layer[0])
    assert np.allclose(by_sublayer[1:].reshape(n, k, *by_layer.shape[1:]).sum(axis=1), by_layer[1:])
    if tiny.architecture.layer_scalars is None:
        stream = attribution.accumulated_resid(result.cache, include_pre=True)[:, -1:]
        steps = np.diff(stream, axis=0, prepend=np.zeros_like(stream[:1]))
        assert np.allclose(by_layer, steps, atol=1e-4, rtol=1e-4)


def test_a_layer_s_heads_sum_to_its_attention_write(tiny):
    result, _, _ = run_attribution(tiny)
    for layer in range(tiny.arch.n_layers):
        heads, bias = attribution.decompose_attn_out(tiny, result.cache, layer)
        whole = heads.sum(axis=0) + (0.0 if bias is None else bias)
        attn_out = read_f64(result.cache[f"blocks.{layer}.attn_out"])[0]
        assert heads.shape[0] == tiny.arch.n_heads
        assert np.allclose(whole, attn_out, atol=1e-4 * np.abs(attn_out).max(), rtol=1e-4), (
            f"{tiny.architecture.model_type}: layer {layer}")


@pytest.mark.parametrize("n_tokens", [WINDOW, 3 * WINDOW + 1])
def test_capturing_attention_internals_at_every_layer_leaves_the_logits_alone(tiny, n_tokens):
    ids = tiny.make_ids([1, 5, 9, 2, 7, 3, 11, 4, 8, 6][:n_tokens])
    plain = read_f64(tiny.run(ids).logits)
    probed = tiny.run(ids, capture=[f"blocks.{i}.attn.weights"
                                    for i in range(tiny.arch.n_layers)])
    assert np.abs(plain).max() > 0.25
    assert np.allclose(read_f64(probed.logits), plain, atol=1e-4, rtol=1e-4)
    assert np.array_equal(read_f64(tiny.run(ids).logits), plain)
    for i in range(tiny.arch.n_layers):
        weights = read_f64(probed.cache[f"blocks.{i}.attn.weights"])[0]
        assert np.allclose(np.triu(weights, k=1), 0.0)


def test_ablating_every_head_equals_zeroing_attn_out(tiny):
    ids = tiny.make_ids(IDS)
    for layer in range(tiny.arch.n_layers):
        heads = tiny.run(ids, interventions=[Ablate.head(layer, h)
                                             for h in range(tiny.arch.n_heads)])
        zeroed = tiny.run(ids, interventions=[Ablate.attention(layer)])
        assert np.allclose(read_f64(heads.logits), read_f64(zeroed.logits),
                           atol=1e-4, rtol=1e-4), (
            f"{tiny.architecture.model_type}: layer {layer}")


def declares(model, points) -> bool:
    found = [model.architecture.supports(p, layer_scoped=True) for p in points]
    if all(found):
        return True
    assert not any(found), f"{model.architecture.model_type} declares part of {points}"
    with pytest.raises(InvalidHookName, match=r"\(not implemented by the"):
        model.run(model.make_ids(IDS), capture=[f"blocks.0.{points[0]}"])
    return False


def test_capturing_the_mlp_interior_at_every_layer_leaves_the_logits_alone(tiny):
    if not declares(tiny, MLP_INTERIOR):
        return
    ids = tiny.make_ids(IDS)
    plain = read_f64(tiny.run(ids).logits)
    probed = tiny.run(ids, capture=[f"blocks.{i}.{p}" for i in range(tiny.arch.n_layers)
                                    for p in MLP_INTERIOR])
    assert np.allclose(read_f64(probed.logits), plain, atol=1e-4, rtol=1e-4)


def test_the_mlp_interior_multiplies_out_and_zeroing_every_neuron_equals_zeroing_mlp_out(tiny):
    if not declares(tiny, MLP_INTERIOR):
        return
    ids = tiny.make_ids(IDS)
    plain = read_f64(tiny.run(ids).logits)
    cache = tiny.run(ids, capture=[f"blocks.{i}.{p}" for i in range(tiny.arch.n_layers)
                                   for p in MLP_INTERIOR]).cache
    for layer in range(tiny.arch.n_layers):
        act, up, down_in = (read_f64(cache[f"blocks.{layer}.mlp.{p}"]) for p in ("act", "up", "down_in"))
        assert act.shape == up.shape == down_in.shape
        assert np.allclose(down_in, act * up, rtol=2.0 ** -7, atol=1e-6), (
            f"{tiny.architecture.model_type}: layer {layer}")
        silenced = read_f64(tiny.run(ids, hooks={
            f"blocks.{layer}.mlp.act": lambda x, info: zeros_like(x)}).logits)
        zeroed = read_f64(tiny.run(ids, interventions=[Ablate.mlp(layer)]).logits)
        assert np.abs(zeroed - plain).max() > 1e-2
        assert np.allclose(silenced, zeroed, atol=1e-4, rtol=1e-4), (
            f"{tiny.architecture.model_type}: layer {layer}")


def test_the_attention_interior_agrees_with_its_weights_and_its_heads(tiny):
    if not declares(tiny, ("attn.scores", "attn.o_in")):
        return
    n = tiny.arch.n_layers
    cache = tiny.run(tiny.make_ids(IDS), capture=[
        f"blocks.{i}.{p}" for i in range(n)
        for p in ("attn.scores", "attn.weights", "attn.per_head_out", "attn.o_in")]).cache
    for layer in range(n):
        scores, weights = (read_f64(cache[f"blocks.{layer}.attn.{p}"]) for p in ("scores", "weights"))
        e = np.exp(scores - scores.max(axis=-1, keepdims=True))
        assert np.allclose(weights, e / e.sum(axis=-1, keepdims=True), atol=2.0 ** -8), (
            f"{tiny.architecture.model_type}: layer {layer}")
        heads = read_f64(cache[f"blocks.{layer}.attn.per_head_out"])
        side_by_side = heads.transpose(0, 2, 1, 3).reshape(heads.shape[0], heads.shape[2], -1)
        assert np.array_equal(read_f64(cache[f"blocks.{layer}.attn.o_in"]), side_by_side), (
            f"{tiny.architecture.model_type}: layer {layer}")


def test_a_point_the_checkpoint_lacks_is_refused_by_name_and_left_out_of_the_law(tiny):
    a = tiny.architecture
    law = a.residual_law_of(tiny.arch)
    for point, reason in a.absent_points(tiny.arch).items():
        assert f"{point}[i]" not in law
        with pytest.raises(InvalidHookName, match=re.escape(f"blocks.1.{point} (absent: {reason})")):
            tiny.run(tiny.make_ids(IDS), capture=[f"blocks.1.{point}"])


def capture(model, **params) -> dict:
    return capture_residual_vectors(model, [RECORD], params)


def read_vectors(out: dict, layer: int) -> np.ndarray:
    return np.array([it["vector"] for it in out["items"] if it["space"]["layer"] == layer],
                    dtype=np.float64)


@pytest.mark.parametrize("read", [{}, {"pool": {"reduce": "mean", "over": "all"}}],
                         ids=["last", "pooled"])
def test_the_captured_writes_add_up_by_the_residual_law(tiny, read):
    n = tiny.arch.n_layers
    names = {}
    for point in read_law_points(tiny):
        out = capture(tiny, point=point, **read)
        assert out["point"] == point and "attention_path" not in out and "heads" not in out
        assert [it["space"]["point"] for it in out["items"]] == [point] * n
        names[point] = [read_vectors(out, layer)[0] for layer in range(n)]
    check_law(tiny, names)


def test_capture_reads_gate_out_where_a_layer_writes_it_and_refuses_it_by_name_elsewhere(tiny):
    a = tiny.architecture
    if "gate_out" in a.writes_of(tiny.arch):
        out = capture(tiny, point="gate_out")
        assert out["point"] == "gate_out" and len(out["items"]) == tiny.arch.n_layers
        return
    with pytest.raises(PointRefused) as e:
        capture(tiny, point="gate_out")
    assert e.value.code == "POINT_ABSENT"
    assert str(e.value).startswith("POINT_ABSENT: `gate_out` is not written on this ")
    assert e.value.issue == {"code": "POINT_ABSENT", "point": "gate_out",
                             "message": str(e.value).removeprefix("POINT_ABSENT: ")}
    assert f"({a.model_type})" in e.value.issue["message"]


def test_capture_s_heads_sum_to_attn_out_and_each_is_what_ablating_it_removes(tiny):
    n, n_heads = tiny.arch.n_layers, tiny.arch.n_heads
    whole = capture(tiny, point="attn_out")
    split = capture(tiny, point="attn_out", heads="all", top=3)
    assert "attention_path" not in whole and "heads" not in whole
    assert (split["point"], split["heads"], split["attention_path"]) == (
        "attn_out", list(range(n_heads)), "per_head")
    assert [(it["space"]["layer"], it["space"]["head"]) for it in split["items"]] == [
        (layer, head) for layer in range(n) for head in range(n_heads)]
    assert all(len(it["top"]) == 3 for it in split["items"])
    for layer in range(n):
        heads, attn_out = read_vectors(split, layer), read_vectors(whole, layer)[0]
        assert np.allclose(heads.sum(axis=0), attn_out,
                           atol=PER_HEAD_TOLERANCE * max(1.0, np.abs(attn_out).max())), (
            f"{tiny.architecture.model_type}: layer {layer}")
    heads, ids = read_vectors(split, 1), tiny.make_ids(render(tiny, RECORD).ids)
    for head in range(n_heads):
        ablated = read_f64(tiny.run(ids, interventions=[Ablate.head(1, head)],
                                    capture=["blocks.1.attn_out"]).cache["blocks.1.attn_out"])[0, -1]
        removed = heads.sum(axis=0) - heads[head]
        scale = float(ablated @ removed) / float(removed @ removed)
        assert np.allclose(ablated, scale * removed,
                           atol=PER_HEAD_TOLERANCE * max(1.0, np.abs(ablated).max())), (
            f"{tiny.architecture.model_type}: head {head}")
        if tiny.architecture.attn_out_norm is None:
            assert scale == pytest.approx(1.0, abs=PER_HEAD_TOLERANCE)


def test_capture_s_attention_weights_sum_to_one_over_the_keys(tiny):
    n, n_heads = tiny.arch.n_layers, tiny.arch.n_heads
    ids = render(tiny, RECORD).ids
    out = capture(tiny, point="attn.weights", heads="all", position="all", top=2)
    assert (out["point"], out["heads"], out["attention_path"], out["top"]) == (
        "attn.weights", list(range(n_heads)), "per_head", 2)
    assert "d_model" not in out and len(out["items"]) == n * n_heads * len(ids)
    for it in out["items"]:
        q, w = it["coords"]["position"], np.array(it["vector"], dtype=np.float64)
        assert it["space"]["point"] == "attn.weights" and it["space"]["d"] == len(ids)
        assert w.sum() == pytest.approx(1.0, abs=PER_HEAD_TOLERANCE)
        assert not np.any(w[q + 1:])
        assert it["token"] == S.token(tiny.tokenizer, ids[q])
        top = it["top"]
        assert 1 <= len(top) <= 2 and "rms_without_top" not in it
        assert [t["weight"] for t in top] == sorted(w, reverse=True)[:len(top)]
        assert all(t["weight"] == it["vector"][t["position"]] and t["position"] <= q
                   and t["token"] == S.token(tiny.tokenizer, ids[t["position"]]) for t in top)
    pooled = capture(tiny, point="attn.weights", heads=[n_heads - 1], layers=[0],
                     pool={"reduce": "mean", "over": "all"})
    [it] = pooled["items"]
    assert it["n_pooled"] == len(ids) and "position" not in it["coords"] and "token" not in it
    assert sum(it["vector"]) == pytest.approx(1.0, abs=PER_HEAD_TOLERANCE)


def test_capture_s_top_names_a_vector_s_largest_coordinates_and_their_share(tiny):
    plain = capture(tiny)
    out = capture(tiny, top=3)
    assert out["top"] == 3 and "top" not in plain and "attention_path" not in out
    for it, before in zip(out["items"], plain["items"], strict=True):
        assert "top" not in before and "rms_without_top" not in before
        assert (it["vector"], it["norm"]) == (before["vector"], before["norm"])
        v = np.array(it["vector"], dtype=np.float64)
        top = it["top"]
        assert len(top) == 3
        assert [abs(t["value"]) for t in top] == sorted(np.abs(v), reverse=True)[:3]
        assert all(t["value"] == it["vector"][t["dim"]] for t in top)
        assert all(t["share"] == pytest.approx(t["value"] ** 2 / (v @ v), abs=1e-5) for t in top)
        assert 0.0 < sum(t["share"] for t in top) <= 1.0
        rest = v.copy()
        rest[[t["dim"] for t in top]] = 0.0
        assert it["rms_without_top"] == pytest.approx(np.sqrt(np.mean(rest * rest)), abs=1e-5)
        assert it["rms_without_top"] < it["norm"] / np.sqrt(v.size) < it["norm"]


def test_a_double_run_is_bit_identical(tiny):
    names = list_declared_names(tiny)
    first = tiny.run(tiny.make_ids(IDS), capture=names)
    second = tiny.run(tiny.make_ids(IDS), capture=names)
    assert np.array_equal(read_f64(first.logits), read_f64(second.logits))
    for name in names:
        assert np.array_equal(read_f64(first.cache[name]), read_f64(second.cache[name])), name


def test_tokenize_round_trips_through_the_tokenizer(tiny):
    prompt = "the cat sat on a mat"
    plain = tiny.tokenize(prompt, chat_template=False)[0].tolist()
    assert tiny.tokenizer.decode(plain) == prompt
    chat = tiny.tokenize(prompt, chat_template=True)[0].tolist()
    assert any(chat[i:i + len(plain)] == plain for i in range(len(chat))), chat
    assert len(chat) > len(plain)


@pytest.mark.parametrize("kit_name", list(KIT_MODULES))
def test_the_dialect_parses_its_own_rendered_call(kit_name):
    captured = json.loads(TEMPLATES.read_text())
    calc = build_toolbox(["calc"]).tools
    for a in load_kit(kit_name).architectures.values():
        record = captured[TEMPLATE_OF[a.model_type]]
        if a.dialect is None:
            assert record["tools_change_the_prompt"] is False, a.model_type
            continue
        rendered = record["round_trip"]
        assert a.dialect.signature in rendered, a.model_type
        _, calls = a.dialect.parse(rendered, calc)
        assert [(c.name, c.arguments) for c in calls] == [("calc", dialects.PROBE_ARGS)], a.model_type
        assert any(m in rendered for m in a.dialect.attempting), a.model_type
        assert dialects.identify(dialects.TemplateProbe(True, rendered)) is a.dialect


@pytest.mark.parametrize("case", list_kit_params())
def test_the_adapter_keys_reach_every_projection(case):
    kit_name, name, model_type = case
    kit = load_kit(kit_name)
    architecture = kit.architectures[model_type]
    model = kit.build(name, architecture)
    keys = architecture.adapter_keys
    params = kit.read_parameter_names(model)
    for i, layer in enumerate(model.lm.model.layers):
        for proj, container in keys.containers.items():
            stem = f"model.layers.{i}.{container}.{proj}"
            if proj == "v_proj" and getattr(layer.self_attn, "use_k_eq_v", False):
                assert f"{stem}.weight" not in params, stem
                continue
            assert f"{stem}.weight" in params, stem
            assert keys.key_re.match(f"{stem}.lora_a"), stem
            assert keys.peft_re.search(f"base_model.model.{stem}.lora_A.weight"), stem
    assert kit.fit_adapter(model, keys) > 0


def test_head_weights_reads_a_head_or_refuses_by_name(tiny):
    a = tiny.architecture
    try:
        spec = a.head_weights(tiny._model, 0, 1)
    except NotImplementedError as e:
        assert a.model_type in str(e)
        pytest.xfail(f"kit finding: {e}")
    d, head_dim = tiny.arch.d_model, spec.head_dim
    assert spec.W_Q.shape == (head_dim, d) and spec.W_K.shape == (head_dim, d)
    assert spec.W_V.shape == (head_dim, d) and spec.W_O.shape == (d, head_dim)
    assert (spec.n_heads, spec.n_kv_heads) == (tiny.arch.n_heads, tiny.arch.n_kv_heads)


def test_a_gemma4_checkpoint_without_per_layer_inputs_refuses_gate_out_by_name():
    kit = load_kit("mlx")
    model = kit.build("gemma4-31b", kit.architectures["gemma4"])
    assert model.architecture.absent_points(model.arch) == {
        "gate_out": model.architecture.absent_when[0].reason}
    assert model.architecture.residual_law_of(model.arch) == (
        "resid_post[i] == (resid_pre[i] + attn_out[i] + mlp_out[i]) * layer_scalar[i] "
        "== resid_pre[i+1]")
    with pytest.raises(InvalidHookName, match=r"blocks\.2\.gate_out \(absent: .*per-layer"):
        model.run(model.make_ids(IDS), capture=["blocks.2.gate_out"])
    e_model = kit.build("gemma4", kit.architectures["gemma4"])
    assert e_model.architecture.absent_points(e_model.arch) == {}
    assert "gate_out" in e_model.architecture.residual_law_of(e_model.arch)


@pytest.mark.parametrize("kit_name", list(KIT_MODULES))
def test_an_absence_is_keyed_by_a_field_the_arch_reads(kit_name):
    fields = {f.name for f in dataclasses.fields(Arch)}
    for a in load_kit(kit_name).architectures.values():
        for absence in a.absent_when:
            assert absence.config_key in fields, (a.model_type, absence.config_key)
            assert a.supports(absence.point, layer_scoped=True), (a.model_type, absence.point)
