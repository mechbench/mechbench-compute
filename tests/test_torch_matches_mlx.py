from __future__ import annotations

import importlib.util

import numpy as np
import pytest

if importlib.util.find_spec("nnsight") is None or importlib.util.find_spec("mlx") is None:
    pytest.skip("the cross-backend check needs both torch (with nnsight) and MLX",
                allow_module_level=True)

import mlx.core as mx
from mlx.utils import tree_flatten
from safetensors.torch import save as save_tensors

from mechbench_compute.arrays import read_f64
from mechbench_compute.lora import fuse_adapter_stack as fuse_on_mlx
from mechbench_compute.torch_backend.lora import fuse_adapter_stack as fuse_on_torch
from tests.mlx_twins import PAIRS, build_on_mlx, read_torch_weights
from tests.tiny_torch_models import build_tiny_model, make_adapter

IDS = [1, 5, 9, 2, 7, 3, 11, 4]

TOLERANCE = 1e-5

# external: mlx-vlm models/gemma3/language.py — rounds Gemma 3's embedding scale to bf16 where transformers keeps the weights' dtype, so float32 weights part by about 1e-4 from the embedding on
TOLERANCE_OF = {"gemma3": 1e-3}


def read_shared_names(on_torch, on_mlx) -> list[str]:
    a, b = on_torch.architecture, on_mlx.architecture
    n = on_torch.arch.n_layers
    return ([f"blocks.{i}.{p}" for i in range(n) for p in a.layer_points_of(on_torch.arch)
             if p in b.layer_points_of(on_mlx.arch)]
            + [p for p in a.global_points if p in b.global_points])


@pytest.mark.parametrize("name", [n for n, _ in PAIRS])
def test_the_same_weights_give_the_same_logits_and_points_on_mlx_and_torch(name):
    on_torch = build_tiny_model(name)
    on_mlx = build_on_mlx(on_torch)
    assert on_mlx.arch.n_layers == on_torch.arch.n_layers
    names = read_shared_names(on_torch, on_mlx)
    torch_run = on_torch.run(on_torch.make_ids(IDS), capture=names)
    mlx_run = on_mlx.run(on_mlx.make_ids(IDS), capture=names)
    tolerance = TOLERANCE_OF.get(on_torch.architecture.model_type, TOLERANCE)
    want, got = read_f64(mlx_run.logits), read_f64(torch_run.logits)
    assert want.shape == got.shape and np.abs(want).max() > 0.25
    assert np.abs(got - want).max() <= tolerance * max(1.0, np.abs(want).max()), (
        f"{name}: logits differ by {np.abs(got - want).max():.3g}")
    for point in names:
        a, b = read_f64(mlx_run.cache[point]), read_f64(torch_run.cache[point])
        assert a.shape == b.shape, (point, a.shape, b.shape)
        err = float(np.abs(a - b).max())
        assert err <= tolerance * max(1.0, float(np.abs(a).max())), f"{name} {point}: {err:.3g}"


@pytest.mark.parametrize("name", [n for n, _ in PAIRS])
def test_the_same_stored_adapter_moves_the_same_weights_on_mlx_and_torch(name):
    on_torch = build_tiny_model(name)
    on_mlx = build_on_mlx(on_torch)
    weights = make_adapter(on_torch, on_torch.architecture.adapter_keys)
    payload = {"data": save_tensors(weights), "lora": {"rank": 2, "alpha": 4.0}}
    torch_before = read_torch_weights(on_torch)
    mlx_before = {k: np.array(v) for k, v in tree_flatten(on_mlx.lm.parameters())}
    fuse_on_torch(on_torch.lm, [payload], layers=[[0, 2]])
    fuse_on_mlx(on_mlx.lm, [payload], layers=[[0, 2]])
    mx.eval(on_mlx.lm.parameters())
    mlx_after = {k: np.array(v) for k, v in tree_flatten(on_mlx.lm.parameters())}
    torch_after = read_torch_weights(on_torch)
    stems = [k.removesuffix(".lora_a") for k in weights if k.endswith(".lora_a")]
    assert len(stems) >= 3 * on_torch.arch.n_layers
    for stem in stems:
        name = f"{stem}.weight"
        moved_torch = torch_after[name] - torch_before[name]
        moved_mlx = mlx_after[name] - mlx_before[name]
        assert np.allclose(moved_torch, moved_mlx, atol=1e-5), name
        assert bool(moved_mlx.any()) == (int(stem.split(".")[2]) in (0, 2)), name
