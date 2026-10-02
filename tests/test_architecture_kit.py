from __future__ import annotations

import dataclasses
import json
import pathlib
import re

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mechbench_compute import attribution, dialects, support
from mechbench_compute._arch import Arch
from mechbench_compute.architectures import ARCHITECTURES, BY_MODEL_TYPE
from mechbench_compute.architectures import llama as llama_arch
from mechbench_compute.errors import InvalidHookName
from mechbench_compute.interventions import Ablate
from mechbench_compute.lora import apply_lora
from mechbench_compute.points import LAYOUT
from mechbench_compute.tools import build_toolbox
from tests.tiny_models import BUILDERS, KIT_MODELS, WINDOW, build_tiny_model

IDS = mx.array([[1, 5, 9, 2, 7, 3, 11, 4]])

RESIDUAL_TOLERANCE = 2.0 ** -6

TEMPLATES = pathlib.Path(__file__).parent / "fixtures" / "chat_templates.json"

TEMPLATE_OF = {
    "gemma3": "mlx-community/gemma-3-4b-it-bf16",
    "gemma4": "mlx-community/gemma-4-e2b-it-bf16",
    "llama": "mlx-community/Llama-3.2-3B-Instruct-bf16",
    "qwen2": "mlx-community/Qwen2.5-3B-Instruct-bf16",
}


@pytest.fixture(scope="module", params=KIT_MODELS, ids=lambda m: m[0])
def tiny(request):
    name, model_type = request.param
    return build_tiny_model(name, BY_MODEL_TYPE[model_type])


def read_f64(value) -> np.ndarray:
    return np.array(mx.array(value).astype(mx.float32), dtype=np.float64)


def list_declared_names(model) -> list[str]:
    a = model.architecture
    return ([f"blocks.{i}.{p}" for i in range(model.arch.n_layers)
             for p in a.layer_points_of(model.arch)]
            + list(a.global_points))


def read_law_terms(law: str) -> list[str]:
    return sorted(set(re.findall(r"([a-z_]+)\[i(?:\+1)?\]", law)))


def check_residual_law(model, ids=IDS) -> None:
    architecture = model.architecture
    law = architecture.residual_law_of(model.arch)
    n = model.arch.n_layers
    terms = read_law_terms(law)
    points = [t for t in terms if t != "layer_scalar"]
    unknown = [p for p in points if not architecture.supports(p, layer_scoped=True)]
    assert not unknown, (
        f"{architecture.model_type}: residual law {law!r} names points it does not declare: {unknown}")
    result = model.run(ids, capture=[f"blocks.{i}.{p}" for i in range(n) for p in points])
    names = {p: [read_f64(result.cache[f"blocks.{i}.{p}"]) for i in range(n)] for p in points}
    if "layer_scalar" in terms:
        assert architecture.layer_scalars is not None, (
            f"{architecture.model_type}: residual law {law!r} names layer_scalar and the "
            f"architecture declares no layer_scalars to read it")
        names["layer_scalar"] = list(architecture.layer_scalars(model._model))
    sides = [s.strip() for s in law.split("==")]
    for i in range(n):
        values = [(side, eval(side, {}, {**names, "i": i}))
                  for side in sides if not ("[i+1]" in side and i == n - 1)]
        for (left, a), (right, b) in zip(values, values[1:]):
            allowed = RESIDUAL_TOLERANCE * max(1.0, float(np.abs(a).max()))
            err = float(np.abs(a - b).max())
            if err > allowed:
                raise AssertionError(
                    f"{architecture.model_type}: residual law {law!r} fails at layer {i}: "
                    f"{left} != {right} (max |diff| {err:.3g}, allowed {allowed:.3g})")


def test_every_architecture_has_a_tiny_model():
    assert set(BY_MODEL_TYPE) == set(BUILDERS)
    assert set(BY_MODEL_TYPE) == set(TEMPLATE_OF)


def test_the_declared_points_are_present_at_the_declared_level(tiny):
    a = tiny.architecture
    assert a.level in support.LEVELS and a.loader in support.LOADERS
    names = list_declared_names(tiny)
    result = tiny.run(IDS, capture=names)
    n_tokens = IDS.shape[1]
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
    result = tiny.run(IDS, capture=wanted)
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


def forward_with_a_wrong_residual(model, input_ids, *, hooks=None, capture=None,
                                  arch=None, kv_cache=None):
    hooks = dict(hooks or {})
    for i in range(arch.n_layers):
        hooks.setdefault(f"blocks.{i}.resid_post", lambda act, info: act * 1.5)
    return llama_arch.ARCH.forward(model, input_ids, hooks=hooks, capture=capture,
                                   arch=arch, kv_cache=kv_cache)


BROKEN = dataclasses.replace(llama_arch.ARCH, model_type="llama-broken",
                             name="Llama with a wrong residual",
                             forward=forward_with_a_wrong_residual)


def test_an_architecture_with_a_wrong_residual_fails_the_law_by_name():
    model = build_tiny_model("llama", BROKEN)
    with pytest.raises(AssertionError,
                       match=r"llama-broken: residual law .* fails at layer 0"):
        check_residual_law(model)


def test_the_attribution_reads_the_writes_and_the_scalars_the_residual_law_names(tiny):
    a = tiny.architecture
    terms = read_law_terms(a.residual_law_of(tiny.arch))
    assert set(a.writes_of(tiny.arch)) == set(terms) - {"resid_pre", "resid_post", "layer_scalar"}
    assert ("layer_scalar" in terms) == (a.layer_scalars is not None), a.model_type


def run_attribution(model, ids=IDS):
    n = model.arch.n_layers
    writes = model.architecture.writes_of(model.arch)
    result = model.run(ids, capture=[
        *(f"blocks.{i}.{p}" for i in range(n) for p in (*writes, "resid_post")),
        "blocks.0.resid_pre", "final_norm.scale",
        *(f"blocks.{i}.attn.per_head_out" for i in range(n))])
    last = read_f64(result.logits)[0, -1]
    cap = model.architecture.attribution_unembed(model._model).softcap
    if cap:
        last = cap * np.arctanh(last / cap)
    return result, last, read_f64(result.cache["final_norm.scale"]).reshape(-1)


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
    ids = mx.array([[1, 5, 9, 2, 7, 3, 11, 4, 8, 6][:n_tokens]])
    plain = read_f64(tiny.run(ids).logits)
    probed = tiny.run(ids, capture=[f"blocks.{i}.attn.weights"
                                    for i in range(tiny.arch.n_layers)])
    assert np.abs(plain).max() > 1.0
    assert np.allclose(read_f64(probed.logits), plain, atol=1e-4, rtol=1e-4)
    for i in range(tiny.arch.n_layers):
        weights = read_f64(probed.cache[f"blocks.{i}.attn.weights"])[0]
        assert np.allclose(np.triu(weights, k=1), 0.0)


def test_ablating_every_head_equals_zeroing_attn_out(tiny):
    for layer in range(tiny.arch.n_layers):
        heads = tiny.run(IDS, interventions=[Ablate.head(layer, h)
                                             for h in range(tiny.arch.n_heads)])
        zeroed = tiny.run(IDS, interventions=[Ablate.attention(layer)])
        assert np.allclose(read_f64(heads.logits), read_f64(zeroed.logits),
                           atol=1e-4, rtol=1e-4), (
            f"{tiny.architecture.model_type}: layer {layer}")


def test_a_double_run_is_bit_identical(tiny):
    names = list_declared_names(tiny)
    first = tiny.run(IDS, capture=names)
    second = tiny.run(IDS, capture=names)
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


def test_the_dialect_parses_its_own_rendered_call():
    captured = json.loads(TEMPLATES.read_text())
    calc = build_toolbox(["calc"]).tools
    for a in ARCHITECTURES:
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


@pytest.mark.parametrize("name,model_type", KIT_MODELS, ids=[m[0] for m in KIT_MODELS])
def test_the_adapter_keys_reach_every_projection(name, model_type):
    architecture = BY_MODEL_TYPE[model_type]
    model = build_tiny_model(name, architecture)
    keys = architecture.adapter_keys
    params = dict(tree_flatten(model.lm.parameters()))
    for i, layer in enumerate(model.lm.model.layers):
        for proj, container in keys.containers.items():
            stem = f"model.layers.{i}.{container}.{proj}"
            if proj == "v_proj" and getattr(layer.self_attn, "use_k_eq_v", False):
                assert f"{stem}.weight" not in params, stem
                continue
            assert f"{stem}.weight" in params, stem
            assert keys.key_re.match(f"{stem}.lora_a"), stem
            assert keys.peft_re.search(f"base_model.model.{stem}.lora_A.weight"), stem
    assert apply_lora(model.lm, rank=2, targets=tuple(keys.containers), keys=keys) > 0


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
    model = build_tiny_model("gemma4-31b", BY_MODEL_TYPE["gemma4"])
    assert model.architecture.absent_points(model.arch) == {
        "gate_out": model.architecture.absent_when[0].reason}
    assert model.architecture.residual_law_of(model.arch) == (
        "resid_post[i] == (resid_pre[i] + attn_out[i] + mlp_out[i]) * layer_scalar[i] "
        "== resid_pre[i+1]")
    with pytest.raises(InvalidHookName, match=r"blocks\.2\.gate_out \(absent: .*per-layer"):
        model.run(IDS, capture=["blocks.2.gate_out"])
    e_model = build_tiny_model("gemma4", BY_MODEL_TYPE["gemma4"])
    assert e_model.architecture.absent_points(e_model.arch) == {}
    assert "gate_out" in e_model.architecture.residual_law_of(e_model.arch)


def test_an_absence_is_keyed_by_a_field_the_arch_reads():
    fields = {f.name for f in dataclasses.fields(Arch)}
    for a in ARCHITECTURES:
        for absence in a.absent_when:
            assert absence.config_key in fields, (a.model_type, absence.config_key)
            assert a.supports(absence.point, layer_scoped=True), (a.model_type, absence.point)
