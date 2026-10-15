from __future__ import annotations

import importlib.util

import pytest

if importlib.util.find_spec("nnsight") is None:
    pytest.skip("the torch backend needs torch with nnsight", allow_module_level=True)

import torch

from mechbench_compute.distill import render
from mechbench_compute.ops.activations.differentiate import differentiate_activations
from mechbench_compute.torch_backend.architectures._decoder import read_text
from tests.test_capture_mlp_interior import IDS, INTERIOR, read_mlp_widths
from tests.tiny_torch_models import KIT, KIT_MODELS, build_tiny_model

TARGET, CONTRAST = 12, 14

RECORD = {"id": "r", "text": "the cat sat on a", "tracked": {"t": {"id": TARGET}, "c": {"id": CONTRAST}}}

NAMES = [n for n, _ in KIT_MODELS]


def differentiate(model, records=None, **params) -> dict:
    return differentiate_activations(model, records or [RECORD], {"storage": "json", **params})


@pytest.mark.parametrize("name", NAMES)
def test_a_zero_added_at_an_interior_point_leaves_the_torch_forward_bit_identical(name):
    model = build_tiny_model(name)
    ids = model.make_ids(IDS)
    n = model.arch.n_layers
    stream = [f"blocks.{i}.resid_post" for i in range(n)]
    plain = model.run(ids, capture=stream)
    for layer in range(n):
        for point in INTERIOR:
            probed = model.run(ids, capture=stream,
                               hooks={f"blocks.{layer}.{point}": lambda act, info: act + torch.zeros_like(act)})
            assert torch.equal(probed.logits, plain.logits), (name, point, layer)
            for s in stream:
                assert torch.equal(probed.cache[s], plain.cache[s]), (name, s, point, layer)


@pytest.mark.parametrize("name", NAMES)
def test_the_torch_gradient_is_as_wide_as_the_mlp_on_each_layer(name):
    model = build_tiny_model(name)
    want = read_mlp_widths(KIT, model)
    out = differentiate(model, point="mlp.down_in")
    assert out["widths"] == want
    assert [len(it["vector"]) for it in out["items"]] == want
    assert [it["space"]["d"] for it in out["items"]] == want


def plant(model, target: int) -> tuple[int, int]:
    text = read_text(model._model)
    layer = model.arch.n_layers - 1
    block = text.layers[layer]
    offset = model.architecture.attribution_unembed(model._model).norm.gain_offset
    with torch.no_grad():
        for norm in (text.norm, getattr(block, "post_feedforward_layernorm", None)):
            if norm is not None:
                norm.weight.fill_(1.0 - offset)
        if getattr(block, "layer_scalar", None) is not None:
            block.layer_scalar.fill_(1.0)
    row = model._model.lm_head.weight[target].detach()
    name = f"blocks.{layer}.mlp.down_in"
    down_in = model.run(model.make_ids(render(model, RECORD).ids), capture=[name]).cache[name][0, -1]
    neuron = int(torch.argmax(down_in))
    assert float(down_in[neuron]) > 0
    with torch.no_grad():
        block.mlp.down_proj.weight[:, neuron] = row
    return layer, neuron


@pytest.mark.parametrize("name", NAMES)
def test_a_neuron_that_writes_the_target_s_unembedding_leads_the_torch_attribution(name):
    model = build_tiny_model(name)
    layer, neuron = plant(model, TARGET)
    k = int(read_text(model._model).layers[layer].mlp.down_proj.weight.shape[1])
    row = differentiate(model, metric="logit", layers=[layer], point="mlp.down_in", top=k)["items"][0]
    lead = row["top_product"][0]
    assert row["target"]["id"] == TARGET and lead["dim"] == neuron and lead["value"] > 0
    flipped = {**RECORD, "tracked": {"c": {"id": CONTRAST}, "t": {"id": TARGET}}}
    against = differentiate(model, [flipped], metric="margin", layers=[layer], point="mlp.down_in",
                            top=k)["items"][0]
    assert against["contrast"]["id"] == TARGET
    assert {e["dim"]: e["value"] for e in against["top_product"]}[neuron] < 0
