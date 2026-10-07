from __future__ import annotations

import importlib.util

import numpy as np
import pytest

if importlib.util.find_spec("nnsight") is None or importlib.util.find_spec("mlx") is None:
    pytest.skip("the cross-backend check needs both torch (with nnsight) and MLX",
                allow_module_level=True)

import mlx.core as mx
from mlx.utils import tree_flatten
from safetensors.numpy import load as load_arrays

from mechbench_compute.lora import apply_lora as apply_on_mlx
from mechbench_compute.lora import fuse_adapter_stack as fuse_on_mlx
from mechbench_compute.lora import remove_lora as remove_on_mlx
from mechbench_compute.ops import Context
from mechbench_compute.ops.adapter import train as train_op
from mechbench_compute.torch_backend import mlx_random
from mechbench_compute.torch_backend.lora_layers import apply_lora as apply_on_torch
from mechbench_compute.torch_backend.lora_layers import read_lora_arrays
from mechbench_compute.torch_backend.lora_layers import remove_lora as remove_on_torch
from tests.mlx_twins import PAIRS, build_on_mlx, read_torch_weights
from tests.test_torch_training import ANCHORS, DECISION, RECORDS
from tests.tiny_torch_models import build_tiny_model

DOCUMENTS = [{"id": "d", "text": "the cat sat on a mat and the dog ran"},
             {"id": "e", "text": "a dog sat on the cat and ran"}]

SFT = {"model": "tiny", "objective": "sft", "steps": 16, "lr": 1e-3, "seed": 3,
       "lora": {"rank": 2, "alpha": 4}, "batch": {"record": 2}, "checkpoint_every": 4,
       "keep_checkpoints": True}

LOSS_TOLERANCE = 2e-4

WEIGHT_TOLERANCE = 2e-3

# external: MLX random.cpp — its own CPU and Metal normal draws part by a float32 ulp or two at about 0.5% of entries
INIT_TOLERANCE = 5e-7


def test_the_mlx_draw_is_reproduced_without_mlx():
    for seed in (0, 7, 2**40 + 5):
        assert np.array_equal(np.array(mx.random.key(seed)), mlx_random.make_key(seed))
    key, mine = mx.random.key(7), mlx_random.make_key(7)
    for shape in ((8, 32), (8, 2560), (3, 7)):
        key, sub = mx.random.split(key)
        mine, mine_sub = mlx_random.split_key(mine)
        assert np.array_equal(np.array(sub), mine_sub)
        want, got = np.array(mx.random.normal(shape, key=sub)), mlx_random.draw_normal(shape, mine_sub)
        assert np.abs(want - got).max() <= INIT_TOLERANCE


@pytest.mark.parametrize("name", [n for n, _ in PAIRS])
def test_the_seeded_initialisation_is_mlx_s(name):
    on_torch = build_tiny_model(name)
    on_mlx = build_on_mlx(on_torch)
    keys = on_torch.architecture.adapter_keys
    targets = ("q_proj", "k_proj", "v_proj", "o_proj")
    n_torch = apply_on_torch(on_torch, 4, 8.0, targets=targets, seed=5, keys=keys)
    n_mlx = apply_on_mlx(on_mlx.lm, 4, 8.0, targets=targets, seed=5, keys=on_mlx.architecture.adapter_keys)
    try:
        got = read_lora_arrays(on_torch)
        want = {k: np.array(v) for k, v in tree_flatten(on_mlx.lm.trainable_parameters())}
    finally:
        remove_on_torch(on_torch)
        remove_on_mlx(on_mlx.lm)
    assert n_torch == n_mlx and set(got) == set(want)
    for k, drawn in want.items():
        assert got[k].dtype == drawn.dtype == np.float32, k
        assert np.abs(got[k] - drawn).max() <= INIT_TOLERANCE, k


def read_curve(out) -> np.ndarray:
    return np.array([item["loss"] for item in out["checkpoints"]["items"]])


def read_norm(weights, keys) -> float:
    return float(np.sqrt(sum(float((weights[k] ** 2).sum()) for k in keys)))


@pytest.mark.parametrize("objective", ["decision", "sft"])
@pytest.mark.parametrize("name", [n for n, _ in PAIRS])
def test_mlx_and_torch_training_of_the_same_model_track_each_other(name, objective):
    on_torch = build_tiny_model(name)
    on_mlx = build_on_mlx(on_torch)
    if objective == "sft":
        params, inputs = SFT, {"records": DOCUMENTS}
    else:
        params = {**DECISION, "steps": 16, "lr": 1e-3}
        inputs = {"records": RECORDS, "anchors": ANCHORS}
    want = train_op.run(Context(loaded=on_mlx), inputs, params)
    got = train_op.run(Context(loaded=on_torch), inputs, params)
    a, b = read_curve(want), read_curve(got)
    assert np.abs(a - b).max() <= LOSS_TOLERANCE * np.abs(a).max(), (a, b)
    wa, wb = load_arrays(want["out"]["data"]), load_arrays(got["out"]["data"])
    assert set(wa) == set(wb) and {wa[k].dtype for k in wa} == {wb[k].dtype for k in wb}
    trained = [k for k in wa if k.endswith(".lora_b")]
    gap = read_norm({k: wa[k] - wb[k] for k in trained}, trained)
    assert gap <= WEIGHT_TOLERANCE * read_norm(wa, trained), gap


@pytest.mark.parametrize("name", [n for n, _ in PAIRS])
def test_an_adapter_trained_on_torch_fuses_on_mlx_and_moves_the_same_weights(name):
    on_torch = build_tiny_model(name)
    on_mlx = build_on_mlx(on_torch)
    adapter = train_op.run(Context(loaded=on_torch), {"records": RECORDS, "anchors": ANCHORS},
                           {**DECISION, "keep_checkpoints": False})["out"]
    before = {k: np.array(v) for k, v in tree_flatten(on_mlx.lm.parameters())}
    torch_before = read_torch_weights(on_torch)
    from mechbench_compute.torch_backend.lora import fuse_adapter_stack as fuse_on_torch

    fuse_on_torch(on_torch.lm, [adapter])
    fuse_on_mlx(on_mlx.lm, [adapter])
    mx.eval(on_mlx.lm.parameters())
    after = {k: np.array(v) for k, v in tree_flatten(on_mlx.lm.parameters())}
    torch_after = read_torch_weights(on_torch)
    stems = [k.removesuffix(".lora_a") for k in load_arrays(adapter["data"]) if k.endswith(".lora_a")]
    assert stems
    for stem in stems:
        w = f"{stem}.weight"
        moved = after[w] - before[w]
        assert np.abs(moved).max() > 0, w
        assert np.allclose(torch_after[w] - torch_before[w], moved, atol=1e-5), w
