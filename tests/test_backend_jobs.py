from __future__ import annotations

import importlib.metadata
import importlib.util
import json
from types import SimpleNamespace

import pytest

from mechbench_compute import backends
from mechbench_compute.lexicon._base import DEFAULT_OUTPUT, In, Op
from mechbench_compute.ops import Context
from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec
from mechbench_compute.protocol.dispatch import place_header

HAS_TORCH = importlib.util.find_spec("nnsight") is not None and importlib.util.find_spec("torch") is not None
HAS_MLX = backends.is_importable("mlx.core")

needs_torch = pytest.mark.skipif(
    not HAS_TORCH, reason="nnsight is not installed: pip install 'mechbench-compute[torch]'")
needs_mlx = pytest.mark.skipif(not HAS_MLX, reason="MLX is not installed here")

MODEL = "tiny/gemma3@rev"

RECORDS = [{"id": "a", "user": "the cat sat on a", "tracked": {"m": "mat", "d": "dog"}},
           {"id": "b", "prompt": "the dog ran on the", "template": "raw"}]

READ = {"model": MODEL, "top_k": 3, "tracked": {"c": "cat"}}

ATTRIBUTE = {"model": MODEL, "split": "sublayer", "per_head_layers": [0, 1],
             "tracked": {"x": "mat", "y": "dog"}}

CAPTURE = {"model": MODEL, "layers": [1], "point": "attn.weights", "heads": [0]}


def make_graph(*, capture: bool = False) -> dict:
    nodes = [{"id": "read", "block": "logits/read", "params": READ,
              "inputs": {"conditions": RECORDS}},
             {"id": "attribute", "block": "logits/attribute", "params": ATTRIBUTE,
              "inputs": {"records": RECORDS}}]
    if capture:
        nodes.append({"id": "capture", "block": "activations/capture", "params": CAPTURE,
                      "inputs": {"records": RECORDS}})
    return {"dataflow": 2, "nodes": nodes, "edges": []}


def run_job(requirements: dict | None, *, capture: bool = False) -> dict:
    extra: dict = {"graph": make_graph(capture=capture)}
    if requirements is not None:
        extra["requirements"] = requirements
    out = ProtocolExecutor().run(ProtocolSpec(kind="pipeline", prompt="", model_id=None,
                                              extra=extra))
    return out.payload if hasattr(out, "payload") else out


@pytest.fixture
def torch_model(monkeypatch):
    from mechbench_compute.torch_backend.model import TorchModel
    from tests.tiny_torch_models import build_tiny_model

    tiny = build_tiny_model("gemma3")
    loads: list[str] = []

    def load(cls, model_id, **_):
        loads.append(model_id)
        return tiny

    monkeypatch.setattr(TorchModel, "load", classmethod(load))
    return SimpleNamespace(model=tiny, loads=loads)


@needs_torch
def test_a_job_that_requires_torch_runs_logits_read_and_attribute_through_the_torch_model(torch_model):
    import torch

    from mechbench_compute.ops.logits import attribute as attribute_op
    from mechbench_compute.ops.logits import read as read_op
    from mechbench_compute.seeds import hardware_class
    from mechbench_compute.torch_backend.loading import read_accelerator

    payload = run_job({"class": "local", "backend": "torch"})
    tiny = torch_model.model
    assert torch_model.loads == [MODEL]
    accelerator = read_accelerator(tiny.device)
    for name in ("read", "attribute"):
        out = payload["outputs"][name]
        assert (out["backend"], out["accelerator"]) == ("torch", accelerator), name
    read = read_op.run(Context(loaded=tiny), {"conditions": RECORDS}, READ)
    assert payload["outputs"]["read"]["items"] == read["items"]
    attribution = attribute_op.run(Context(loaded=tiny), {"records": RECORDS}, ATTRIBUTE)
    assert payload["outputs"]["attribute"]["items"] == attribution["items"]

    hardware = payload["resources"]["hardware"]
    assert {k: hardware[k] for k in hardware_class()} == hardware_class()
    on_gpu = tiny.device.type == "cuda"
    assert hardware["backend"] == "torch"
    assert hardware["accelerator"] == accelerator
    assert hardware["gpu"] == (torch.cuda.get_device_name(tiny.device) if on_gpu else None)
    assert hardware["cuda"] == torch.version.cuda
    for package in ("torch", "transformers", "nnsight"):
        assert hardware[package] == importlib.metadata.version(package)
    assert hardware["attention"] == ["sdpa"]


@needs_torch
def test_reading_attention_weights_declares_the_eager_attention_beside_sdpa(torch_model):
    payload = run_job({"backend": "torch"}, capture=True)
    assert payload["outputs"]["capture"]["backend"] == "torch"
    assert payload["resources"]["hardware"]["attention"] == ["eager", "sdpa"]


def read_logits(ctx, inputs, params):
    from mechbench_compute.distill import render

    model = ctx.model(params.get("model"))
    ids = model.make_ids(render(model, RECORDS[0]).ids)
    return {"logits": model.run(ids).logits[0, -1].tolist()}


def build_op(*, port: bool):
    inputs = (In("adapter", "adapter/lora", "An adapter.", required=False),) if port else ()
    declared = Op(name="probe/logits", summary="Reads logits.", description="Reads logits.",
                  params=(), inputs=inputs, needs=frozenset({"model.forward"}))
    return SimpleNamespace(op=declared, module=SimpleNamespace(run=read_logits), scope=None)


@needs_torch
def test_a_torch_job_fuses_a_stored_adapter_through_the_torch_lora_and_restores_it():
    from safetensors.torch import save as save_tensors

    from mechbench_compute.model_ref import ModelRef
    from mechbench_compute.torch_backend import lora
    from tests.tiny_torch_models import build_tiny_model, make_adapter

    tiny = build_tiny_model("gemma3")
    weights = make_adapter(tiny, tiny.architecture.adapter_keys)
    payload = {"data": save_tensors(weights), "lora": {"rank": 2, "alpha": 4.0}}
    handles = lora.fuse_adapter_stack(tiny.lm, [payload], keys=tiny.architecture.adapter_keys)
    try:
        adapted = read_logits(Context(loaded=tiny), {}, {})["logits"]
    finally:
        lora.restore_adapter_stack(tiny.lm, handles)
    bare = read_logits(Context(loaded=tiny), {}, {})["logits"]
    assert adapted != bare

    ex = ProtocolExecutor()
    ex._backend = backends.find("torch")
    ex._model, ex._model_id = tiny, "tiny"
    ref = ModelRef(base_kind="hf", base="tiny", adapter_labels=("you/lab/dice",),
                   adapter_payloads=(payload,))
    by_reference = ex._run_op(build_op(port=False), {}, {"model": ref})
    assert by_reference["logits"] == adapted
    assert (by_reference["fused"], by_reference["backend"]) == ([{"bench": "you/lab/dice"}], "torch")
    by_port = ex._run_op(build_op(port=True), {"adapter": payload}, {"model": "tiny"})
    assert by_port["logits"] == adapted
    assert read_logits(Context(loaded=tiny), {}, {})["logits"] == bare


@needs_torch
def test_an_operator_is_refused_by_name_on_torch():
    from mechbench_compute.torch_backend import lora
    from tests.tiny_torch_models import build_tiny_model

    tiny = build_tiny_model("llama")
    with pytest.raises(ValueError, match="an operator attaches on the mlx backend only"):
        lora.fuse_adapter_stack(tiny.lm, [{"kind": "adapter/operator"}])


@needs_mlx
def test_an_mlx_job_s_result_is_byte_identical_whatever_its_requirements_say(monkeypatch):
    from mechbench_compute.model import Model
    from mechbench_compute.seeds import hardware_class
    from tests.tiny_models import build_tiny_model

    tiny = build_tiny_model("gemma3")
    monkeypatch.setattr(Model, "load", classmethod(lambda cls, model_id, **_: tiny))
    results = [run_job(r) for r in (None, {"class": "local"}, {"class": "local", "backend": "mlx"})]
    assert len({json.dumps(r, sort_keys=True) for r in results}) == 1
    payload = results[0]
    assert payload["resources"]["hardware"] == hardware_class()
    for out in payload["outputs"].values():
        assert not {"backend", "accelerator"} & set(out)


def test_a_job_naming_a_backend_compute_does_not_declare_is_refused_by_name():
    with pytest.raises(backends.BackendRefused,
                       match="'jax' is not a backend compute declares; it declares mlx, torch"):
        run_job({"backend": "jax"})


def test_the_backend_header_goes_where_fused_goes():
    op = SimpleNamespace(outputs=("extra",))
    result = {DEFAULT_OUTPUT: {"items": []}, "extra": {}, "fused": [], "backend": "torch",
              "accelerator": "cuda"}
    placed = place_header(op, result)
    assert placed == {DEFAULT_OUTPUT: {"items": [], "fused": [], "backend": "torch",
                                       "accelerator": "cuda"}, "extra": {}}
