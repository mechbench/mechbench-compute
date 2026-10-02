from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

if importlib.util.find_spec("nnsight") is None:
    pytest.skip("nnsight is not installed: pip install 'mechbench-compute[torch]'",
                allow_module_level=True)
torch = pytest.importorskip("torch")

from safetensors.torch import save as save_tensors

from mechbench_compute.arrays import read_f32, read_f64
from mechbench_compute.distill import render
from mechbench_compute.errors import InvalidHookName
from mechbench_compute.interventions import Ablate, Patch
from mechbench_compute.ops.logits import attribute as attribute_op
from mechbench_compute.ops.logits import read as read_op
from mechbench_compute.torch_backend.lora import (
    fuse_adapter_stack,
    restore_adapter_stack,
)
from mechbench_compute.torch_backend.model import TorchModel
from tests.tiny_torch_models import build_tiny_model, make_adapter

ROOT = pathlib.Path(__file__).resolve().parent.parent

IDS = [1, 5, 9, 2, 7, 3, 11, 4]

MODELS = ("gemma3", "llama", "gemma3-vlm")


@pytest.fixture(scope="module", params=MODELS)
def tiny(request):
    return build_tiny_model(request.param)


def read_weights(model: TorchModel) -> dict[str, torch.Tensor]:
    return {f"model.{n}": p.detach().clone() for n, p in model.lm.model.named_parameters()}


def make_payload(model: TorchModel, rank: int = 2, alpha: float = 4.0) -> dict:
    weights = make_adapter(model, model.architecture.adapter_keys, rank=rank)
    return {"data": save_tensors(weights), "lora": {"rank": rank, "alpha": alpha}, "weights": weights}


TRACE_THEN_MLX = """
import importlib.util, sys, torch
spec = importlib.util.spec_from_file_location("tracing", sys.argv[1])
tracing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tracing)
{guard}
envoy = tracing.wrap_model(torch.nn.Linear(2, 2))
with envoy.trace(torch.ones(1, 2)):
    held = envoy.output
print("mounted" if hasattr(object(), "save") else "unmounted", flush=True)
if importlib.util.find_spec("mlx") is not None:
    import mlx.core as mx
    assert mx.array([1.0, 2.0]).sum().item() == 3.0
print("mlx imported")
"""


def run_trace_then_mlx(guard: str) -> subprocess.CompletedProcess:
    tracing = ROOT / "mechbench_compute" / "torch_backend" / "tracing.py"
    return subprocess.run([sys.executable, "-c", TRACE_THEN_MLX.format(guard=guard), str(tracing)],
                          cwd=ROOT, capture_output=True, text=True, timeout=300, check=False)


def test_a_trace_leaves_nothing_mounted_so_mlx_still_imports_after_it():
    out = run_trace_then_mlx("")
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.split() == ["unmounted", "mlx", "imported"]


@pytest.mark.skipif(importlib.util.find_spec("mlx") is None, reason="MLX is not installed here")
def test_without_the_guard_nnsight_s_mount_breaks_the_next_mlx_import():
    out = run_trace_then_mlx("tracing.load_nnsight = lambda: __import__('nnsight')")
    assert out.returncode != 0 and out.stdout.split() == ["mounted"], (out.stdout, out.stderr[-500:])


def test_the_interventions_act_on_torch_tensors(tiny):
    ids = tiny.make_ids(IDS)
    names = ["blocks.1.resid_pre", "blocks.1.resid_post", "blocks.1.mlp_out",
             "blocks.2.resid_pre", "blocks.2.resid_post"]
    base = tiny.run(ids, capture=names)
    v = base.cache["blocks.1.resid_post"][0, 2]

    patched = tiny.run(ids, capture=names, interventions=[Patch.activation(2, 3, v, point="resid_pre")])
    pre = read_f64(patched.cache["blocks.2.resid_pre"])[0]
    assert np.array_equal(pre[3], read_f64(v))
    keep = [i for i in range(len(IDS)) if i != 3]
    assert np.array_equal(pre[keep], read_f64(base.cache["blocks.2.resid_pre"])[0][keep])

    vn = read_f32(v)
    added = tiny.run(ids, capture=names, interventions=[Patch.add(1, -1, vn, alpha=2.0)])
    post = read_f64(added.cache["blocks.1.resid_post"])[0]
    want = read_f64(base.cache["blocks.1.resid_post"])[0]
    assert np.allclose(post[-1], want[-1] + 2.0 * vn, atol=1e-5)
    assert np.array_equal(post[:-1], want[:-1])

    skipped = tiny.run(ids, capture=names, interventions=[Ablate.layer(1)])
    assert np.array_equal(read_f64(skipped.cache["blocks.1.resid_post"]),
                          read_f64(skipped.cache["blocks.1.resid_pre"]))
    zeroed = tiny.run(ids, capture=names, interventions=[Ablate.mlp(1)])
    assert not read_f64(zeroed.cache["blocks.1.mlp_out"]).any()
    same = tiny.run(ids, interventions=[Patch.position(1, 4, base.cache)])
    assert torch.equal(same.logits, base.logits)


def test_captures_leave_the_fast_path_s_logits_alone_bit_for_bit(tiny):
    ids = tiny.make_ids(IDS)
    a = tiny.architecture
    plain = tiny.run(ids).logits
    every = [f"blocks.{i}.{p}" for i in range(tiny.arch.n_layers)
             for p in a.layer_points_of(tiny.arch) if p != "attn.weights"] + list(a.global_points)
    assert torch.equal(tiny.run(ids, capture=every).logits, plain)


@pytest.mark.parametrize("name", MODELS)
def test_reading_attention_weights_leaves_the_next_forward_alone_bit_for_bit(name):
    model = build_tiny_model(name)
    ids = model.make_ids(IDS)
    plain = model.run(ids).logits
    weights = [f"blocks.{i}.attn.weights" for i in range(model.arch.n_layers)]
    model.run(model.make_ids(IDS[:3]), capture=weights)
    assert torch.equal(model.run(ids).logits, plain)
    model.run(ids, capture=weights)
    assert torch.equal(model.run(ids, capture=["blocks.0.resid_post"]).logits, plain)


def test_logits_attribute_adds_up_on_torch_by_layer_by_sublayer_and_by_head(tiny):
    records = [{"id": "a", "prompt": "the cat sat on a", "template": "raw"},
               {"id": "b", "user": "the dog ran on the mat"}]
    for split in ("layer", "sublayer"):
        out = attribute_op.attribute_logits(tiny, records, {
            "split": split, "per_head_layers": [0, 1], "tracked": {"x": "mat", "y": "dog"}})
        assert out["components"][0] == "embed" and out["split"] == split
        for row in out["items"]:
            assert abs(row["additivity"]["residual"]) <= 1e-3, row["additivity"]
            if split == "sublayer":
                by_name = dict(zip(out["components"], row["measures"]["contribution"]))
                for head in row["per_head"]:
                    whole = by_name[f"L{head['layer']}.attn"]
                    assert abs(sum(head["contributions"]) + head.get("bias", 0.0) - whole) <= 1e-3


def test_logits_read_is_the_model_s_own_distribution(tiny):
    ctx = SimpleNamespace(model=lambda _ref: tiny, on_start=None, on_item=None, resume_items=None)
    record = {"id": "a", "user": "the cat sat on a", "tracked": {"m": "mat"}}
    out = read_op.run(ctx, {"conditions": [record]}, {"top_k": 3})
    item = out["items"][0]
    ids = render(tiny, record).ids
    with torch.no_grad():
        logits = tiny._model(input_ids=tiny.make_ids(ids)).logits[0, -1].float()
    p = torch.softmax(logits, dim=-1).cpu().numpy()
    best = int(np.argmax(p))
    assert item["top"][0]["token"]["id"] == best
    assert item["top"][0]["p"] == pytest.approx(float(p[best]), abs=1e-5)
    mat = tiny.tokenizer.convert_tokens_to_ids("mat")
    assert item["tracked"]["m"]["p"] == pytest.approx(float(p[mat]), abs=1e-5)


def test_logits_read_refuses_rollout_and_complete_by_name_on_torch(tiny):
    ctx = SimpleNamespace(model=lambda _ref: tiny, on_start=None, on_item=None, resume_items=None)
    record = {"id": "a", "user": "the cat sat on a"}
    for params in ({"rollout": {"top_k": 2}}, {"complete": {"items": ["mat"]}}):
        with pytest.raises(ValueError, match=r"run on the mlx backend only; this model runs on torch"):
            read_op.run(ctx, {"conditions": [record]}, params)


def test_a_stored_adapter_fuses_in_its_layers_and_restores_bit_for_bit(tiny):
    before = read_weights(tiny)
    plain = tiny.run(tiny.make_ids(IDS)).logits
    payload = make_payload(tiny)
    handles = fuse_adapter_stack(tiny.lm, [payload], layers=[[1, 3]])
    after = read_weights(tiny)
    scale = 4.0 / 2
    for key, a in payload["weights"].items():
        if not key.endswith(".lora_a"):
            continue
        stem = key.removesuffix(".lora_a")
        layer = int(stem.split(".")[2])
        held = before[f"{stem}.weight"]
        a, b = a.to(held.device), payload["weights"][f"{stem}.lora_b"].to(held.device)
        moved = after[f"{stem}.weight"] - held
        if layer in (1, 3):
            assert torch.equal(after[f"{stem}.weight"], held + (scale * (b @ a)).to(held.dtype)), stem
        else:
            assert not moved.any(), stem
    fused = tiny.run(tiny.make_ids(IDS)).logits
    assert not torch.equal(fused, plain)
    restore_adapter_stack(tiny.lm, handles)
    assert all(torch.equal(read_weights(tiny)[k], v) for k, v in before.items())
    assert torch.equal(tiny.run(tiny.make_ids(IDS)).logits, plain)


def test_a_refused_adapter_unwinds_the_stack_and_names_the_layer(tiny):
    before = read_weights(tiny)
    payload = make_payload(tiny)
    with pytest.raises(ValueError, match=r"LAYER_OUT_OF_RANGE: an adapter's `layers` names 9"):
        fuse_adapter_stack(tiny.lm, [payload, payload], layers=[[0], [9]])
    assert all(torch.equal(read_weights(tiny)[k], v) for k, v in before.items())


@pytest.mark.skipif(importlib.util.find_spec("mlx") is None, reason="MLX is not installed here")
def test_the_same_stored_adapter_moves_the_same_weights_on_mlx_and_torch():
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from mechbench_compute.architectures import BY_MODEL_TYPE
    from mechbench_compute.lora import fuse_adapter_stack as fuse_on_mlx
    from tests.tiny_models import build_tiny_model as build_on_mlx

    on_torch = build_tiny_model("gemma3")
    on_mlx = build_on_mlx("gemma3", BY_MODEL_TYPE["gemma3"])
    payload = make_payload(on_torch)
    torch_before = read_weights(on_torch)
    mlx_before = {k: np.array(v) for k, v in tree_flatten(on_mlx.lm.parameters())}
    fuse_adapter_stack(on_torch.lm, [payload], layers=[[0, 2]])
    fuse_on_mlx(on_mlx.lm, [payload], layers=[[0, 2]])
    mx.eval(on_mlx.lm.parameters())
    mlx_after = {k: np.array(v) for k, v in tree_flatten(on_mlx.lm.parameters())}
    torch_after = read_weights(on_torch)
    compared = 0
    for key in payload["weights"]:
        if not key.endswith(".lora_a"):
            continue
        name = key.removesuffix(".lora_a") + ".weight"
        moved_torch = (torch_after[name] - torch_before[name]).cpu().numpy()
        moved_mlx = mlx_after[name] - mlx_before[name]
        assert np.allclose(moved_torch, moved_mlx, atol=1e-5), name
        compared += 1
    assert compared == 4 * 7


def test_a_checkpoint_on_disk_loads_and_runs_as_it_did_in_memory(tmp_path):
    for name in ("gemma3", "gemma3-vlm", "llama"):
        held = build_tiny_model(name)
        where = tmp_path / name
        held._model.save_pretrained(where)
        held.tokenizer.save_pretrained(where)
        loaded = TorchModel.load(str(where), device=str(held.device), dtype=torch.float32)
        assert loaded.architecture is held.architecture
        assert loaded.arch.n_layers == held.arch.n_layers and loaded.arch.model_type == held.arch.model_type
        ids = held.make_ids(IDS)
        assert torch.equal(loaded.run(ids).logits, held.run(ids).logits), name


def test_a_checkpoint_whose_head_caps_its_logits_is_refused_by_name(tmp_path):
    held = build_tiny_model("gemma3")
    held._model.config.final_logit_softcapping = 30.0
    held._model.save_pretrained(tmp_path)
    held.tokenizer.save_pretrained(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["final_logit_softcapping"] == 30.0
    with pytest.raises(NotImplementedError, match=r"caps its final logits, and the Gemma 3 head"):
        TorchModel.load(str(tmp_path), device="cpu", dtype=torch.float32)


def test_a_point_or_a_continuation_the_backend_lacks_is_refused_by_name(tiny):
    with pytest.raises(InvalidHookName, match=r"blocks\.0\.gate_out \(not implemented by the '(gemma3|llama)' "
                                              r"forward on the torch backend\)"):
        tiny.run(tiny.make_ids(IDS), capture=["blocks.0.gate_out"])
    with pytest.raises(NotImplementedError, match="continuing from a cache is not on it yet"):
        tiny.run(tiny.make_ids(IDS), capture=["blocks.0.resid_post"], kv_cache=object())
    with pytest.raises(NotImplementedError, match="on the mlx backend only"):
        tiny.prompt_cache()
