from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

if importlib.util.find_spec("nnsight") is None:
    pytest.skip("nnsight is not installed: pip install 'mechbench-compute[torch]'",
                allow_module_level=True)
torch = pytest.importorskip("torch")

from safetensors.torch import load as load_tensors

from mechbench_compute import backends, model_ref
from mechbench_compute.adapter_keys import ADAPTER_KEYS
from mechbench_compute.lexicon._base import Op
from mechbench_compute.ops import Context
from mechbench_compute.ops.adapter import train as train_op
from mechbench_compute.ops.adapter.measure import measure_adapter
from mechbench_compute.protocol import ProtocolExecutor
from mechbench_compute.torch_backend import lora_layers
from mechbench_compute.torch_backend.lora import fuse, restore
from tests.checkpoint_offers import due_only
from tests.tiny_torch_models import MODEL_TYPES, build_tiny_model

RECORDS = [{"id": "a", "user": "the cat sat on a"}, {"id": "b", "user": "the dog ran on a"}]

PARAMS = {"model": "tiny", "target": {"uniform": ["mat", "dog", "cat"]}, "closer": " ran", "steps": 3,
          "lr": 0.05, "seed": 3}

IDS = [2, 7, 8, 9, 10, 11]

LABEL = "you/lab/die"


def train(model, layers=None, **ctx):
    shape = {"rank": 2, "alpha": 4, **({} if layers is None else {"layers": layers})}
    out = train_op.run(Context(loaded=model, **ctx), {"records": RECORDS}, {**PARAMS, "lora": shape})
    return out["out"]


def read_key_layers(payload):
    return sorted({int(ADAPTER_KEYS.key_re.match(k).group(1)) for k in load_tensors(payload["data"])})


def count_wrapped(model):
    return sum(isinstance(m, lora_layers.LoRALinear) for m in model.lm.modules())


def read_stream(ctx, inputs, params):
    model = ctx.model(params.get("model"))
    names = [f"blocks.{i}.resid_post" for i in range(len(model.lm.model.layers))]
    run = model.run(model.make_ids(IDS), capture=names)
    return {"residuals": [run.cache[n].detach().float().cpu().numpy().tolist() for n in names],
            "logits": run.logits[0, -1].detach().float().cpu().numpy().tolist()}


PROBE = SimpleNamespace(
    op=Op(name="probe/stream", summary="Reads the residual stream.", description="Reads the residual stream.",
          params=(), inputs=(), needs=frozenset({"model.forward"})),
    module=SimpleNamespace(run=read_stream), scope=None)


def read(model, ref):
    executor = ProtocolExecutor()
    executor._model, executor._model_id = model, "tiny"
    executor._backend = next(b for b in backends.BACKENDS if b.name == "torch")
    return executor._run_op(PROBE, {}, {"model": ref})


def adapted(payload):
    return model_ref.resolve({"base": "tiny", "adapters": [{"bench": LABEL}]}, fetch=lambda _label: payload)


@pytest.fixture(scope="module", params=MODEL_TYPES)
def tiny(request):
    return build_tiny_model(request.param)


@pytest.fixture(scope="module")
def last_two(tiny):
    return train(tiny, [-2, -1])


def test_all_and_every_index_train_the_adapter_byte_for_byte(tiny):
    n = len(tiny.lm.model.layers)
    unnamed = train(tiny)
    for layers in ("all", list(range(n)), {"to": n}):
        named = train(tiny, layers)
        assert named["data"] == unnamed["data"]
        assert named["lora"] == unnamed["lora"]
    assert "layers" not in unnamed["lora"]


@pytest.mark.parametrize(("layers", "first"), [([-2, -1], -2), ({"from": 0, "to": 2}, 0)])
def test_the_adapter_holds_weights_for_those_layers_only(tiny, layers, first):
    n = len(tiny.lm.model.layers)
    chosen = [n - 2, n - 1] if first < 0 else [0, 1]
    payload = train(tiny, layers)
    assert read_key_layers(payload) == chosen
    assert payload["lora"]["layers"] == chosen
    everywhere = load_tensors(train(tiny)["data"])
    assert set(load_tensors(payload["data"])) == {k for k in everywhere if int(k.split(".")[2]) in chosen}
    assert count_wrapped(tiny) == 0


def test_a_chosen_layer_starts_from_the_weights_the_full_run_draws_there(tiny):
    lora_layers.apply_lora(tiny, 2, 4, seed=3)
    try:
        full = lora_layers.read_lora_arrays(tiny)
    finally:
        lora_layers.remove_lora(tiny)
    lora_layers.apply_lora(tiny, 2, 4, seed=3, layers=[1])
    try:
        one = lora_layers.read_lora_arrays(tiny)
    finally:
        lora_layers.remove_lora(tiny)
    assert one and all(k.startswith("model.layers.1.") for k in one)
    assert all(np.array_equal(one[k], full[k]) for k in one)


def test_a_fused_reference_leaves_the_frozen_layers_exactly_as_the_base(tiny, last_two):
    n = len(tiny.lm.model.layers)
    bare = read(tiny, "tiny")
    fused = read(tiny, adapted(last_two))
    assert [fused["residuals"][i] == bare["residuals"][i] for i in range(n)] == [i < n - 2 for i in range(n)]
    assert fused["logits"] != bare["logits"]
    assert read(tiny, "tiny") == bare


def test_fusing_moves_only_the_chosen_layers(tiny, last_two):
    n = len(tiny.lm.model.layers)
    handle = fuse(tiny.lm, load_tensors(last_two["data"]), 2.0, keys=tiny.architecture.adapter_keys)
    try:
        assert sorted({i for i, _c, _p in handle}) == [n - 2, n - 1]
    finally:
        restore(tiny.lm, handle)


def test_measuring_reads_the_chosen_layers_and_carries_them(tiny, last_two):
    n = len(tiny.lm.model.layers)
    out = measure_adapter(last_two)
    assert out["measured"]["layers"] == [n - 2, n - 1]
    assert out["lora"]["layers"] == [n - 2, n - 1]


def test_a_layer_the_model_lacks_refuses_and_leaves_the_model_bare(tiny):
    n = len(tiny.lm.model.layers)
    grads = {k: p.requires_grad for k, p in tiny._model.named_parameters()}
    with pytest.raises(ValueError, match=rf"names -{n + 1}, and this model has {n} layers"):
        train(tiny, [-n - 1])
    with pytest.raises(ValueError, match="selects no layer"):
        train(tiny, {"from": -1, "to": -1})
    assert count_wrapped(tiny) == 0
    assert {k: p.requires_grad for k, p in tiny._model.named_parameters()} == grads


def test_a_resume_from_a_run_on_other_layers_refuses():
    tiny = build_tiny_model("llama")
    states: list[dict] = []
    params = {**PARAMS, "steps": 4, "checkpoint_every": 2, "lora": {"rank": 2, "alpha": 4, "layers": [0]}}
    train_op.run(Context(loaded=tiny, on_checkpoint=due_only(states.append)), {"records": RECORDS}, params)
    with pytest.raises(ValueError, match=r"holds 4 weights on layers 0 and this node's adapter 4 weights on layers 3:"):
        train_op.run(Context(loaded=tiny, resume_state=states[0]), {"records": RECORDS},
                     {**params, "lora": {"rank": 2, "alpha": 4, "layers": [-1]}})
    assert count_wrapped(tiny) == 0
